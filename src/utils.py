from matplotlib.widgets import Slider
import itertools
import os

import matplotlib
import numpy as np
import pydicom
import highdicom as hd
from highdicom.seg import Segmentation
from matplotlib import pyplot as plt
from pydicom.datadict import mask_match
from scipy import ndimage
from . import metrics
import SimpleITK as sitk


def load_ct(ct_info):
    """
    Reb el diccionari d'un ct i retorna una llista (dcm_slices) amb tots els arxius dcm (ordenats) i ct: volum en unitats Hounsfield.
    """
    dicomdir_path = ct_info['path']
    acquisition_number = ct_info['acquisition_number']


    dcm_slices = [
        pydicom.dcmread(slice_path, force=True)
        for slice_path in dicomdir_path.glob("*.dcm")
    ]
    # comprovam que tinguin slicelocation i seleccionam els de la metixa adquisicio
    dcm_slices =[slice for slice in dcm_slices if _slice_z(slice) is not None]

    if acquisition_number is not None:
        filtrats = [s for s in dcm_slices
                    if getattr(s, 'AcquisitionNumber', None) == acquisition_number]
    else:
        filtrats = []

    if not filtrats:
        from collections import Counter
        comptes = Counter(getattr(s, 'AcquisitionNumber', None) for s in dcm_slices)
        if comptes:
            acq_major, _ = comptes.most_common(1)[0]
            filtrats = [s for s in dcm_slices
                        if getattr(s, 'AcquisitionNumber', None) == acq_major]

    dcm_slices = filtrats

    if not dcm_slices:
        raise ValueError(f"No s'han trobat slices valides a: {dicomdir_path}")

    # Ordenem per Z de major a menor
    dcm_slices.sort(key=lambda s: -_slice_z(s))

    ct = np.stack([to_hu(s) for s in dcm_slices], axis=0)

    return dcm_slices,ct

def _slice_z(dcm):
    """
    Retorna la coordenada Z d'un tall.
    ImagePositionPatient[2] i, si no hi es, SliceLocation.
    Retorna None si no es pot determinar.
    """
    ipp = getattr(dcm, 'ImagePositionPatient', None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return float(ipp[2])
        except (TypeError, ValueError):
            pass
    sl = getattr(dcm, 'SliceLocation', None)
    if sl is not None:
        try:
            return float(sl)
        except (TypeError, ValueError):
            pass
    return None

def to_hu(dcm):
    """
    Funció per obtenir valors en HU d'un arixu dcm
    """
    slope = float(getattr(dcm, 'RescaleSlope', 1.0))
    intercept = float(getattr(dcm, 'RescaleIntercept', 0.0))
    return dcm.pixel_array.astype(np.float32) * slope + intercept

def align_mask_to_ct(mask, seg, dcm_slices): #crec que es podria borrar
    """
    Alinea la màscara SEG (n_seg_slices, rows, cols) al CT (n_ct_slices, rows, cols)
    seleccionant les slices de la màscara que corresponen a les posicions Z del CT.

    Paràmetres:
    mask     : np.ndarray (n_seg_slices, rows, cols) retornada per load_segmentation
    seg      : highdicom Segmentation object retornat per load_segmentation
    dcm_slices: llista de dcm llegits per load_ct, ordenats de major a menor Z

    Retorna: mask_aligned : np.ndarray (n_ct_slices, rows, cols)
    """
    # Coordenades Z del CT (una per slice, ordre igual que dcm_slices)
    ct_z = [float(s.ImagePositionPatient[2]) for s in dcm_slices]

    # Coordenades Z de la màscara SEG (reconstruïdes igual que a load_segmentation)
    pfgs = seg.PerFrameFunctionalGroupsSequence
    seg_z_set = []
    for frame_info in pfgs:
        z = float(frame_info.PlanePositionSequence[0].ImagePositionPatient[2])
        if z not in seg_z_set:
            seg_z_set.append(z)
    seg_z_sorted = sorted(set(seg_z_set), reverse=True)  # mateix ordre que mask
    z_to_seg_idx = {z: i for i, z in enumerate(seg_z_sorted)}

    rows, cols = mask.shape[1], mask.shape[2]
    mask_aligned = np.zeros((len(ct_z), rows, cols), dtype=mask.dtype)

    for ct_idx, z in enumerate(ct_z):
        # Busquem la slice SEG més propera a la Z del CT (tolerància 2 mm)
        closest_z = min(seg_z_sorted, key=lambda sz: abs(sz - z))
        if abs(closest_z - z) < 2.0:
            seg_idx = z_to_seg_idx[closest_z]
            mask_aligned[ct_idx] = mask[seg_idx]

    return mask_aligned


def load_segmentation(dicom_path):
    """
    Carrega un fitxer DICOM-SEG i retorna la màscara combinada de Liver + Mass.
    Utilitza pixel_array directament per compatibilitat màxima amb qualsevol SEG.
    """
    seg = hd.seg.segread(dicom_path)

    # Nombre de frames i dimensions de cada frame
    n_frames = seg.NumberOfFrames
    rows = seg.Rows
    cols = seg.Columns

    # Segments d'interès (per número, usant label com a guia)
    liver_nums = set(seg.get_segment_numbers(segment_label='Liver') or [1])
    mass_nums  = set(seg.get_segment_numbers(segment_label='Mass')  or [2])

    # pixel_array té shape (n_frames, rows, cols) — un frame per segment per slice
    pix = seg.pixel_array  # (n_frames, rows, cols) si és multi-frame
    if pix.ndim == 2:
        pix = pix[np.newaxis]  # per si és mono-frame

    # Recollim coordenada Z de cada frame per saber a quina llesca pertany
    # i el número de segment per saber a quin segment pertany
    pfgs = seg.PerFrameFunctionalGroupsSequence

    # Determinem el rang de coordenades Z úniques → nombre de slices
    z_coords = []
    for frame_info in pfgs:
        pos = frame_info.PlanePositionSequence[0].ImagePositionPatient
        z_coords.append(float(pos[2]))

    unique_z = sorted(set(z_coords), reverse=True)  # ordre superior a inferior
    n_slices = len(unique_z)
    z_to_idx = {z: i for i, z in enumerate(unique_z)}

    # Inicialitzem la màscara multi-etiqueta 3D (slices, rows, cols)
    # Valors: 0=fons, 1=Liver, 2=Mass
    mask_labeled = np.zeros((n_slices, rows, cols), dtype=np.int8)

    for frame_idx, frame_info in enumerate(pfgs):
        # Número de segment d'aquest frame
        seg_num = frame_info.SegmentIdentificationSequence[0].ReferencedSegmentNumber

        # Coordenada Z d'aquest frame → índex de slice
        z = float(frame_info.PlanePositionSequence[0].ImagePositionPatient[2])
        slice_idx = z_to_idx[z]

        frame_data = pix[frame_idx].astype(bool)

        if seg_num in liver_nums:
            # Liver=1 (no sobreescrivim si ja hi ha Mass=2)
            mask_labeled[slice_idx][frame_data & (mask_labeled[slice_idx] == 0)] = 1
        elif seg_num in mass_nums:
            # Mass=2 (sobreescriu Liver si se solapen)
            mask_labeled[slice_idx][frame_data] = 2

    return seg, mask_labeled


def visualize(img, mask, mask_color_rgb=(0, 1, 0), slice_idx=(None, None, None), alpha=0.5,
              spatial_resolution=(1, 1, 1), path_to_save=None, show = True):
    """
    Vizualitza 'slice' indicada amb la segmentació i la guarda al path_to_save.
    """
    # Rang HU adaptatiu: percentil 1-99 del CT per no perdre contrast ---
    vmin = float(np.percentile(img, 1))
    vmax = float(np.percentile(img, 99))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Detectem si la màscara és multi-etiqueta (1=Liver, 2=Mass) ---
    unique_vals = np.unique(mask[mask > 0]) if np.any(mask > 0) else []
    is_multilabel = len(unique_vals) > 1 or (len(unique_vals) == 1 and unique_vals[0] == 2)

    #Índex de slice: al centre de la màscara (max coverage) ---
    resolved_slice_idx = list(slice_idx)
    nonzero = np.argwhere(mask > 0)
    for np_ax in range(3):
        if resolved_slice_idx[np_ax] is None:
            if len(nonzero) > 0:
                resolved_slice_idx[np_ax] = int(np.median(nonzero[:, np_ax]))
            else:
                resolved_slice_idx[np_ax] = img.shape[np_ax] // 2

    #Mida de figura adaptativa basada en les dimensions físiques ---
    dz = spatial_resolution[0] if spatial_resolution[0] else 1.0
    dy = spatial_resolution[1] if spatial_resolution[1] else 1.0
    dx = spatial_resolution[2] if spatial_resolution[2] else dy
    phys_ax = np.array([
        img.shape[0] * dz,
        img.shape[1] * dy,
        img.shape[2] * dx,
    ])
    scale = 6.0 / max(phys_ax[1], phys_ax[2])
    fig_w = max(phys_ax[2] * scale * 2, 6.0)
    fig_h = max(phys_ax[1] * scale + phys_ax[0] * scale, 4.0)

    fig, fig_axes = plt.subplots(
        2, 2,
        layout='compressed',
        figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': [max(phys_ax[0], 1), max(phys_ax[1], 1)]}
    )
    fig.delaxes(fig_axes[0, 1])

    for i, j in itertools.product(range(2), range(2)):
        fig_axes[i, j].get_xaxis().set_visible(False)
        fig_axes[i, j].get_yaxis().set_visible(False)

    ax_np2fig = {0: (0, 0), 1: (1, 0), 2: (1, 1)}

    for np_ax in range(3):
        slice_index = resolved_slice_idx[np_ax]
        img_slice = img.take(slice_index, axis=np_ax)
        img_rgb = matplotlib.colormaps['bone'](norm(img_slice))[..., :3].copy()

        mask_slice = mask.take(slice_index, axis=np_ax)

        if is_multilabel:
            # Liver (etiqueta 1) → verd
            liver_px = (mask_slice == 1)
            img_rgb[liver_px] = (
                img_rgb[liver_px] * (1 - alpha) +
                np.array([0.0, 0.9, 0.2]) * alpha
            )
            # Mass (etiqueta 2) → vermell
            mass_px = (mask_slice == 2)
            img_rgb[mass_px] = (
                img_rgb[mass_px] * (1 - alpha) +
                np.array([1.0, 0.1, 0.1]) * alpha
            )
        else:
            mask_nonzero = np.tile((mask_slice > 0)[..., np.newaxis], [1, 1, 3])
            mask_rgb = (mask_slice > 0)[..., np.newaxis] * np.array(mask_color_rgb).reshape(1, 1, 3)
            img_rgb = img_rgb * (1 - alpha * mask_nonzero) + mask_rgb * alpha * mask_nonzero

        px_resolution = np.delete(np.array([dz, dy, dx]), np_ax)
        fig_axes[ax_np2fig[np_ax]].imshow(
            img_rgb,
            aspect=float(px_resolution[0]) / float(px_resolution[1])
        )

    if path_to_save is not None:
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        plt.savefig(path_to_save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()



def visualize_interactive(ct, mask, spatial_resolution=(1, 1, 1)):
    """
    Visualitzador interactiu amb slider per recorrer les slices axials.
 
    Paràmetres:
    ct               : np.ndarray (n_slices, rows, cols) en unitats Hounsfield (HU),
                       tal com el retorna load_ct. Substitueix l'anterior 'slices'
                       (llista pydicom) per garantir consistencia amb visualize().
    mask             : np.ndarray (n_slices, rows, cols) retornat per load_segmentation
                       i alineat al CT. Valors: 0=fons, 1=Liver, 2=Mass.
    spatial_resolution: tupla (dz, dy, dx) en mm, obtinguda amb get_spatial_resolution().
                       S'usa per calcular l'aspect ratio correcte de cada pixel i
                       evitar que la imatge aparegui distorsionada.
    """
 
    # 1. Comprovem que les dimensions coincideixin
    if ct.shape != mask.shape:
        print(f"Atencio! Les dimensions del CT {ct.shape} no coincideixen amb la mascara {mask.shape}.")
        return
 
    # 2. Contrast adaptatiu: percentil 1-99 de la imatge CT en HU,
    #    igual que fa la funcio visualize() per garantir consistencia visual.
    vmin = float(np.percentile(ct, 1))
    vmax = float(np.percentile(ct, 99))
 
    # 3. Aspect ratio: cada pixel axial te dimensions (dy x dx) en mm.
    #    Si dy != dx (que es habitual en CT), sense correccio la imatge apareix estirada.
    #    spatial_resolution = (dz, dy, dx)
    dz = spatial_resolution[0] if spatial_resolution[0] else 1.0
    dy = spatial_resolution[1] if spatial_resolution[1] else 1.0
    dx = spatial_resolution[2] if spatial_resolution[2] else dy
    # En una vista axial (files=Y, columnes=X): aspect = dy/dx
    aspect_axial = dy / dx
 
    # 4. Detectem si la mascara es multi-etiqueta (Liver + Mass) o binaria
    unique_vals = np.unique(mask[mask > 0]) if np.any(mask > 0) else []
    is_multilabel = len(unique_vals) > 1 or (len(unique_vals) == 1 and unique_vals[0] == 2)
 
    # 5. Configurem la figura de Matplotlib
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
 
    current_slice = ct.shape[0] // 2
 
    # Mostrem la primera slice amb contrast HU i aspect ratio correctes
    im = ax.imshow(ct[current_slice], cmap="bone", vmin=vmin, vmax=vmax, aspect=aspect_axial)
 
    # 6. Superposem la mascara amb colors per etiqueta
    #    Usem RGBA (4 canals) per poder pintar Liver=verd i Mass=vermell
    #    amb transparencia sobre el CT de fons.
    def build_overlay(slice_idx):
        """Construeix un array RGBA per superposar la mascara sobre el CT."""
        mask_slice = mask[slice_idx]
        # Inicialitzem tota la imatge transparent
        rgba = np.zeros((*mask_slice.shape, 4), dtype=np.float32)
        if is_multilabel:
            # Liver (etiqueta 1) -> verd semitransparent
            liver_px = (mask_slice == 1)
            rgba[liver_px] = [0.0, 0.9, 0.2, 0.45]
            # Mass (etiqueta 2) -> vermell semitransparent
            mass_px = (mask_slice == 2)
            rgba[mass_px] = [1.0, 0.1, 0.1, 0.45]
        else:
            # Mascara binaria -> verd semitransparent
            binary_px = (mask_slice > 0)
            rgba[binary_px] = [0.0, 1.0, 0.0, 0.45]
        return rgba
 
    overlay = ax.imshow(build_overlay(current_slice), aspect=aspect_axial)
 
    ax.set_title(f"Slice {current_slice}/{ct.shape[0] - 1}")
    ax.axis("off")
 
    # 7. Configurem el Slider per recorrer les slices
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, ct.shape[0] - 1, valinit=current_slice, valstep=1)
 
    # 8. Funcio d'actualitzacio cridada cada vegada que es mou el slider
    def update(val):
        slice_idx = int(slider.val)
        im.set_data(ct[slice_idx])
        overlay.set_data(build_overlay(slice_idx))
        ax.set_title(f"Slice {slice_idx}/{ct.shape[0] - 1}")
        fig.canvas.draw_idle()
 
    slider.on_changed(update)
    plt.show()

def get_spatial_resolution(dcm_object):
    res_slice = None
    for field_name in ['SpacingBetweenSlices', 'SliceThickness']:
        if field_name in dcm_object:
            res_slice = dcm_object[field_name].value.real
            break
    return (
        res_slice,
        dcm_object.PixelSpacing[0].real,
        dcm_object.PixelSpacing[1].real,
    )

# Funció per agafar l'acquisition_number del primer arxiu dcm que trobi
def get_acquisition_number(path_serie):
    for filename in os.listdir(path_serie):
        if filename.endswith(".dcm"):
            dcm_path = os.path.join(path_serie, filename)
            try:
                # Llegim només capçaleres per ser més ràpids
                dicom = pydicom.dcmread(dcm_path, force=True, stop_before_pixels=True)
                if hasattr(dicom, 'AcquisitionNumber'):
                    return int(dicom.AcquisitionNumber) if dicom.AcquisitionNumber is not None else 0
            except:
                pass
            break  # Parem al primer .dcm per no obrir-los tots
    return None

def get_orientacio_segmentacio(seg):
    orientacio = seg.SharedFunctionalGroupsSequence[0].PlaneOrientationSequence[0].ImageOrientationPatient
    return np.array(orientacio, dtype=float)

def get_orientation_de_imatge(img):
    orientacio = img.ImageOrientationPatient #vector de dimensio 6
    return np.array(orientacio, dtype=float)

def get_canvi_orientacio(orientacio_obj):
    #assumeixo que nomes canvien d'orientació axial
    orientacio_estandard = np.array([1, 0, 0, 0, 1, 0]).reshape(2, 3)
    obj_orient = np.array(orientacio_obj, dtype=float).reshape(2, 3)

    # Comprovar si les orientacions ja coincideixen
    if np.allclose(obj_orient, orientacio_estandard, atol=1e-3):
        return None  # No cal cap transformació

    # Vectors directors de cada orientació
    estandard_vector_director_files = orientacio_estandard[0]  # Vector de files
    estandard_vector_director_columnes = orientacio_estandard[1]  # Vector de columnes
    obj_vector_director_files = obj_orient[0]
    obj_vector_director_columnes = obj_orient[1]

    # Detectar si es necessita invertir les files o les columnes
    invertir_files = np.dot(estandard_vector_director_files, obj_vector_director_files) < 0 #producte escalar
    invertir_columnes = np.dot(estandard_vector_director_columnes, obj_vector_director_columnes) < 0

    # Detectar si es necessita una rotació 90° o 270°
    files_columnes_intercanviades = abs(np.dot(estandard_vector_director_files, obj_vector_director_columnes)) > 0.9

    # Detectar angle de rotació
    rotacio= 0  # Nombre de rotacions de 90°

    if files_columnes_intercanviades:
        # Els eixos estan intercanviats -> rotació de 90° o 270°
        if np.dot(estandard_vector_director_files, obj_vector_director_columnes) > 0:
            rotacio = 1  # 90°
        else:
            rotacio = 3  # 270°

    if invertir_files and invertir_columnes: #si s'han d'invertir les dues es una rotacio
            rotacio = 2  # 180°
            invertir_files = False
            invertir_columnes= False
    #si hi ha rotacio, que retorni invertir_files false (cas en que rotacio == 2, ens surt invertir files i columnes)
    return {
        'invertir_files': invertir_files,
        'invertir_columnes': invertir_columnes,
        'rotacio': rotacio
    }


def corregir_orientacio(objecte, orientacio_obj):
    """

    :param objecte: imatge 3D  o mascara
    :param orientacio_obj: vector [ , , , , , ] donat per get_orientacio...
    :param verbose: si volem els prints o no
    :return: copia del objecte amb la orientacio estadard
    """
    # Calcular transformació necessària
    transform = get_canvi_orientacio(orientacio_obj)

    # Si no cal cap transformació
    if transform is None:

        return objecte


    # Aplicar transformacions
    obj_corregit = objecte.copy()

    # 1. Aplicar rotació 90 graus si cal
    if transform['rotacio'] != 0:
        obj_corregit = np.rot90(obj_corregit, k=transform['rotacio'], axes=(1, 2))

    # 2. Aplicar flips si cal
    if transform['invertir_files']:
        obj_corregit = np.flip(obj_corregit, axis=2)

    if transform['invertir_columnes']:
        obj_corregit = np.flip(obj_corregit, axis=1)

    return obj_corregit



def build_sitk_image(vol, dcm_slices, is_mask=False):
    """
    Construeix un objecte SimpleITK.Image amb la geometria DICOM correcta
    (origin, spacing, direction) a partir d'un volum numpy i les seves slices.
    """
    img = sitk.GetImageFromArray(vol.astype(np.float32 if not is_mask else np.uint8))

    # Spacing: (dx, dy, dz) en mm
    dy, dx = float(dcm_slices[0].PixelSpacing[0]), float(dcm_slices[0].PixelSpacing[1])
    # dz: distancia absoluta entre slices consecutives
    if len(dcm_slices) > 1:
        z0 = float(dcm_slices[0].ImagePositionPatient[2])
        z1 = float(dcm_slices[1].ImagePositionPatient[2])
        dz = abs(z1 - z0)
    else:
        dz = float(getattr(dcm_slices[0], 'SliceThickness', 1.0))
    img.SetSpacing((dx, dy, dz))

    # Origin: posicio del primer voxel del primer slice
    # load_ct ordena Z descendent, per tant el primer slice te la Z mes alta
    origin = dcm_slices[0].ImagePositionPatient
    img.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))

    # Direction: per simplicitat assumim axial standard. Si tinguessis CTs
    # amb orientacions estranyes caldria llegir ImageOrientationPatient.
    # Posem Z invertit perque load_ct ordena descendent.
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, -1))

    return img

def build_sitk_image_from_seg(mask, seg):
    """
    Construeix un sitk.Image amb la geometria fisica de la SEG, llegida directament de l'objecte Segmentation de
    highdicom (no de dcm_slices).

    Parametres:
    mask: np.ndarray (n_slices, rows, cols), la mascara ja reconstruida per load_segmentation (Z descendent).
    seg : objecte Segmentation de highdicom retornat per load_segmentation.

    Retorna: sitk.Image (uint8) amb origin/spacing/direction de la SEG.
    """
    shared = seg.SharedFunctionalGroupsSequence[0]

    # PixelSpacing (dy entre files, dx entre columnes)
    pixel_measures = shared.PixelMeasuresSequence[0]
    dy, dx = float(pixel_measures.PixelSpacing[0]), float(pixel_measures.PixelSpacing[1])

    #Orientacio (ImageOrientationPatient: 6 valors)
    orient = shared.PlaneOrientationSequence[0].ImageOrientationPatient
    row_cos = np.array([float(v) for v in orient[0:3]])  # direccio de les columnes (X)
    col_cos = np.array([float(v) for v in orient[3:6]])  # direccio de les files (Y)

    #Z de cada frame, per calcular dz i trobar el frame de Z mes alta ---
    pfgs = seg.PerFrameFunctionalGroupsSequence
    z_coords = []
    positions = {}  # z -> ImagePositionPatient completa
    for frame_info in pfgs:
        ipp = frame_info.PlanePositionSequence[0].ImagePositionPatient
        z = float(ipp[2])
        z_coords.append(z)
        positions[z] = [float(ipp[0]), float(ipp[1]), float(ipp[2])]

    unique_z = sorted(set(z_coords), reverse=True)  # MATEIX ordre que load_segmentation

    # dz: distancia fisica entre talls consecutius
    if len(unique_z) > 1:
        difs = np.abs(np.diff(unique_z))
        dz = float(np.mean(difs))
        if not np.allclose(difs, difs[0], atol=1e-2):
            print(f"[AVIS] SEG: espaiat Z no uniforme "
                  f"({difs.min():.3f}-{difs.max():.3f} mm); s'usa la mitjana {dz:.3f}")
    else:
        dz = float(getattr(pixel_measures, 'SpacingBetweenSlices', 1.0))

    # Construim la imatge
    img = sitk.GetImageFromArray(mask.astype(np.uint8))
    img.SetSpacing((dx, dy, dz))

    # Origin: posició del tall amb Z més alta (= tall 0 del volum numpy)
    origin = positions[unique_z[0]]
    img.SetOrigin((origin[0], origin[1], origin[2]))

    # Direction: dos primers vectors de la orientacio DICOM; el tercer (Z)
    # el posem negatiu (-row x col) per coherencia amb la Z descendent de la  mascara,
    # exactament com build_sitk_image fa amb direction Z = -1.
    z_cos = -np.cross(row_cos, col_cos)
    direction = (
        row_cos[0], col_cos[0], z_cos[0],
        row_cos[1], col_cos[1], z_cos[1],
        row_cos[2], col_cos[2], z_cos[2],
    )
    img.SetDirection(tuple(float(v) for v in direction))

    return img

def resample_mask_to_ct(mask, seg, dcm_slices_ct):
    """
    Resampleja la mascara a la "graella" del CT d'aquesta fase, usant la geometria fisica DICOM.

    Cada voxel de la graella CT pren el valor de la màscara a la seva posició fisica real.
    Això evita talls buits enmig quan les Z de la SEG i el CT no coincideixen.

    Parametres:
    mask: np.ndarray (n_seg_slices, rows, cols), binaria o 0/1/2
    seg: llista dcm que defineix la geometria de la SEG
    dcm_slices_ct: llista dcm del CT d'aquesta fase (referencia)

    Retorna: mask_resampled: np.ndarray amb la MATEIXA shape que el CT
    """
    # Construim la mascara com a imatge SimpleITK amb la seva geometria
    mask_img = build_sitk_image_from_seg(mask, seg)
    # Construim la imatge CT NOMES per usar-la com a referencia geometrica
    #  (no ens cal el contingut, nomes origin/spacing/direction/size)
    ct_ref = build_sitk_image(
        np.zeros((len(dcm_slices_ct),
                  int(dcm_slices_ct[0].Rows),
                  int(dcm_slices_ct[0].Columns))),
        dcm_slices_ct, is_mask=False
    )

    #Resamplegem la mascara a la graella del CT (nearest neighbor)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ct_ref)          # copia origin/spacing/dir/size del CT
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(sitk.Transform())     # identitat: només resamplejem

    mask_resampled_img = resampler.Execute(mask_img)

    # Tornem a numpy i binaritzem
    mask_resampled = sitk.GetArrayFromImage(mask_resampled_img)
    # DIAGNOSTIC TEMPORAL: comparar spacing Z de SEG i CT
    print(f"    [resample] SEG spacing={mask_img.GetSpacing()}  CT spacing={ct_ref.GetSpacing()}")
    print(f"    [resample] SEG size={mask_img.GetSize()}  CT size={ct_ref.GetSize()}")

    return (mask_resampled > 0).astype(np.uint8)

def resample_to_common_grid(vol1, dcm_slices_1, vol2, dcm_slices_2,
                            is_mask=False, target_spacing=None,
                            return_sitk=False):
    """
    Resamplejam dos volums a una graella fisica comuna isotropica en Z.

    Parametres:
        vol1, vol2: arrays 3D
        dcm_slices_1, dcm_slices_2: llistes d'objectes DICOM ordenats igual que els volums
        is_mask: True per a mascares (interpolacio nearest neighbor),
                 False per a CTs (interpolacio linear)
        target_spacing: si None, agafa l'espaiat mes fi dels dos volums
        return_sitk: si True, retorna tambe els sitk.Image resamplejats
                     (necessari per a metriques de superficie en mm)

    Retorna:
        vol1_resampled, vol2_resampled: numpy arrays amb la mateixa shape
        (si return_sitk=True, retorna a mes img1_resampled, img2_resampled)
    """
    #Construim objectes SimpleITK
    img1 = build_sitk_image(vol1, dcm_slices_1, is_mask=is_mask)
    img2 = build_sitk_image(vol2, dcm_slices_2, is_mask=is_mask)

    #Decidim l'espaiat objectiu (mes fi dels dos)
    spacing1 = np.array(img1.GetSpacing())
    spacing2 = np.array(img2.GetSpacing())
    if target_spacing is None:
        target_spacing = tuple(np.minimum(spacing1, spacing2).tolist())

    #Determinem el bounding box fisic comu (unio dels dos volums)
    def physical_bounds(img):
        size = np.array(img.GetSize())
        origin = np.array(img.GetOrigin())
        spacing = np.array(img.GetSpacing())
        direction = np.array(img.GetDirection()).reshape(3, 3)
        corner_low = origin
        corner_high = origin + direction @ (spacing * (size - 1))
        return np.minimum(corner_low, corner_high), np.maximum(corner_low, corner_high)

    low1, high1 = physical_bounds(img1)
    low2, high2 = physical_bounds(img2)
    common_low = np.minimum(low1, low2)
    common_high = np.maximum(high1, high2)

    #Calculem la mida (en voxels) de la graella comuna
    target_spacing_arr = np.array(target_spacing)
    common_size = np.ceil((common_high - common_low) / target_spacing_arr).astype(int) + 1

    print(f"Espaiats originals: {spacing1} vs {spacing2}")
    print(f"Espaiat objectiu: {target_spacing_arr}")
    print(f"Mida de la graella comuna: {tuple(common_size.tolist())} voxels")
    print(f"Bounding box fisic: low={common_low}, high={common_high}")

    # 5. Configurem el resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(target_spacing_arr.tolist()))
    resampler.SetSize([int(s) for s in common_size.tolist()])
    resampler.SetOutputOrigin(tuple(common_low.tolist()))
    resampler.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    resampler.SetTransform(sitk.Transform())

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(-1000.0)

    #Apliquem el resampling als dos volums
    img1_resampled = resampler.Execute(img1)
    img2_resampled = resampler.Execute(img2)

    # Tornem a numpy
    vol1_np = sitk.GetArrayFromImage(img1_resampled)
    vol2_np = sitk.GetArrayFromImage(img2_resampled)

    if is_mask:
        vol1_np = (vol1_np > 0).astype(np.uint8)
        vol2_np = (vol2_np > 0).astype(np.uint8)

    if return_sitk:
        return vol1_np, vol2_np, img1_resampled, img2_resampled
    return vol1_np, vol2_np

def save_mask_npy(mask, output_path):
    """
    Desa una mascara binaria 3D com a .npy.
    No deses la geometria fisica: aquesta es recupera des de les dcm_slices
    del CT corresponent quan calgui (p. ex. al resampling).
    """
    np.save(str(output_path), mask.astype(np.uint8))


def load_mask_npy(input_path):
    """
    Carrega una mascara desada amb save_mask_npy i la torna a binaritzar
    com a salvaguarda.
    """
    mask = np.load(str(input_path)).astype(np.uint8)
    return (mask > 0).astype(np.uint8)



def get_largest_connected_component(mask):
    """
    Retorna nomes el component connex mes gran d'una mascara binaria 3D.

    MedSAM2 pot deixar 'illots' fora del fetge (melsa, ronyo, columna) durant
    la propagacio. Com que el fetge es un unic volum connex, ens quedem nomes
    amb la component mes gran i descartem la resta.

    Parametres:
        mask: np.ndarray 3D, binaria (0/1) o amb valors > 0 com a fetge.

    Retorna: np.ndarray uint8 (0/1) amb nomes la component mes gran.
    """
    binary = (mask > 0)
    if not binary.any():
        return np.zeros_like(mask, dtype=np.uint8)

    # Etiqueta cada grup de voxels connexos (connectivitat per cares: 6-veinatge)
    labeled, n_components = ndimage.label(binary)
    if n_components <= 1:
        return binary.astype(np.uint8)

    # Mida (en voxels) de cada component; ignorem el fons (label 0)
    sizes = ndimage.sum(binary, labeled, range(1, n_components + 1))
    largest_label = int(np.argmax(sizes)) + 1  # +1 perque sizes[0] es el label 1

    return (labeled == largest_label).astype(np.uint8)


def carrega_fases_pacient(sample,SEGMENTATIONS_PATH):
    """
    Retorna un dict {fase: {'mask': np.ndarray, 'dcm': dcm_slices, 'ct': ct}}
    nomes per a les fases que tenen .npy.
    """
    name = sample["name"]
    patient_dir = SEGMENTATIONS_PATH / name
    fases = {}
    for ct_info in sample["list_ct_paths"]:
        fase = ct_info["phase"]
        npy_path = patient_dir / f"{fase}_segmentation.npy"
        if not npy_path.exists():
            continue
        mask = load_mask_npy(npy_path)
        dcm_slices, ct = load_ct(ct_info)

        # La màscara guardada i el CT han de tenir el mateix nombre de talls.
        if mask.shape != ct.shape:
            print(f"  [avis] {name}/{fase}: shape mask {mask.shape} != CT {ct.shape}; salto fase")
            continue
        fases[fase] = {"mask": mask, "dcm": dcm_slices, "ct": ct}
    return fases


def compara_parella(dataA, dataB):
    """
    Remostreja les dues mascares a graella comuna i retorna (rvd, dsc).
    RVD calculat de A respecte B (B = referencia).
    """
    mA, mB = resample_to_common_grid(
        dataA["mask"], dataA["dcm"],
        dataB["mask"], dataB["dcm"],
        is_mask=True,
    )

    return metrics.metric_to_compare_two_regions_rvd(mA, mB), metrics.metric_to_compare_two_regions_dsc(mA, mB)

# Visualització (3 vistes per fase, reutilitzant utils.visualize)


def visualitza_fases(name, fases, output_dir):
    """Un PNG de 3 vistes per fase: CT + mascara MedSAM2 superposada."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fase, data in fases.items():
        # spatial_resolution (dz, dy, dx) a partir del primer dcm.
        dcm0 = data["dcm"][0]
        try:
            dy, dx = [float(v) for v in dcm0.PixelSpacing]
        except Exception:
            dy = dx = 1.0
        try:
            dz = abs(float(dcm0.SliceThickness))
        except Exception:
            dz = 1.0
        out_path = output_dir / f"{name}_{fase}_seg.png"
        visualize(
            data["ct"], data["mask"],
            mask_color_rgb=(0, 1, 0),
            spatial_resolution=(dz, dy, dx),
            path_to_save=str(out_path),
            show=False,
        )
        print(f" Visualització dels tres talls desada: {out_path.name}")


