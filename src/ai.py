
import torch
import numpy as np
from pathlib import Path
from sam2.build_sam import build_sam2_video_predictor_npz
from PIL import Image as PILImage
from scipy.ndimage import binary_fill_holes
from . import utils

CHECKPOINT = Path("MedSAM2/checkpoints/MedSAM2_latest.pt")
CONFIG = "configs/sam2.1_hiera_t512.yaml"

def preproces(ct,level, width):#MedSAM2 requereix passar les unitats de HU a unit8
    #ct ha de ser n arxiu amb modality == CT

    lower = level - width / 2
    upper = level + width / 2
    clipped = np.clip(ct, lower,upper)
    normalitzats = (clipped-lower)/(upper-lower) #valors entr
    ct_uint8 = (normalitzats*255.0).astype(np.uint8) #queda una mascara amb el fetge "resaltat"
    return ct_uint8

def mask_2d_to_bbox(m2d: np.ndarray) -> np.ndarray:
    """
    Donada una mascara 2D binaria (H, W), retorna la bounding box que la
    conte en format [x_min, y_min, x_max, y_max] (coordenades de pixel).

    Convencio d'eixos: en un array numpy (H, W), el primer eix son les FILES
    (direccio y) i el segon les COLUMNES (direccio x). SAM2 espera la box en
    ordre [x_min, y_min, x_max, y_max], per tant hem de mapar:
        x  <-> columnes (axis=1)
        y  <-> files    (axis=0)
    Aquest ordre es la font d'error tipica; el respectem explicitament.
    """
    files = np.any(m2d > 0, axis=1)   # quines files contenen fetge
    columnes = np.any(m2d > 0, axis=0)  # quines columnes contenen fetge

    y_min, y_max = np.where(files)[0][[0, -1]]
    x_min, x_max = np.where(columnes)[0][[0, -1]]

    # +1 a max perque la box sigui inclusiva del darrer pixel amb fetge
    return np.array([x_min, y_min, x_max + 1, y_max + 1], dtype=np.float32)


def extract_prompt_boxes(mask_loaded, n_prompts_target = 8, every_n=None):
    """
    Retorna una llista de (slice_idx, bbox) per usar com a prompts de BOX.
    Parametres:
        mask_loaded: np.ndarray (n_slices, H, W), binaria (0/1), alineada al CT
                     de la fase de referencia (PV) i projectada a la graella comuna.
        every_n: cada quants talls amb fetge posem una box.

    Retorna: llista [(slice_idx, bbox), ...] amb bbox = [x_min, y_min, x_max, y_max].
    """
    pixels_per_slice = (mask_loaded > 0).sum(axis=(1, 2))

    if pixels_per_slice.sum() == 0:
        raise ValueError(
            "Mascara buida: cap voxel de fetge, no es pot extreure prompt "
            "(probable desalineament Z entre la SEG i el CT d'aquesta fase)."
        )

    slices_amb_fetge = np.where(pixels_per_slice > 0)[0]
    key_slice_idx = int(np.argmax(pixels_per_slice))
    z0, z1 = int(slices_amb_fetge[0]), int(slices_amb_fetge[-1])

    n_amb_fetge = len(slices_amb_fetge)

    #pas adaptatiu: s'adapta al nombre de talls de cada ct
    if every_n is None:
        every_n = max(1,n_amb_fetge//max(1,n_prompts_target))

    seleccionats = set(range(z0, z1 + 1, every_n))
    seleccionats.add(z0)
    seleccionats.add(z1)
    seleccionats.add(key_slice_idx)

    prompts = []
    for idx in sorted(seleccionats):
        m2d = (mask_loaded[idx] > 0).astype(np.uint8)
        if m2d.sum() == 0:
            continue  # saltem talls sense fetge dins el rang
        bbox = mask_2d_to_bbox(m2d)
        prompts.append((idx, bbox))

    return prompts

def extract_prompt_mask(mask_loaded, n_prompts_target=8, every_n=None):

    pixels_per_slice = (mask_loaded > 0).sum(axis=(1, 2))
    if pixels_per_slice.sum() == 0:
        raise ValueError("Mascara buida: cap voxel de fetge ...")

    slices_amb_fetge = np.where(pixels_per_slice > 0)[0]
    key_slice_idx = int(np.argmax(pixels_per_slice))
    z0, z1 = int(slices_amb_fetge[0]), int(slices_amb_fetge[-1])
    n_amb_fetge = len(slices_amb_fetge)

    # Pas adaptatiu: aproximadament n_prompts_target ancoratges.
    if every_n is None:
        every_n = max(1, n_amb_fetge // max(1, n_prompts_target))

    seleccionats = set(range(z0, z1 + 1, every_n))
    seleccionats.add(z0)
    seleccionats.add(z1)
    seleccionats.add(key_slice_idx)

    prompts = []
    for idx in sorted(seleccionats):
        m2d = (mask_loaded[idx] > 0).astype(np.uint8)
        if m2d.sum() == 0:
            continue
        prompts.append((idx, m2d))
    return prompts


def redim_vol_a_rgb(img_uint8: np.ndarray, target_size: int= 512)-> np.ndarray:
    n_slices = img_uint8.shape[0]
    out = np.zeros((n_slices, 3, target_size, target_size), dtype=np.float32)
    for i in range(n_slices):
        # Convertim cada 'slice' a imatge PIL i redimensionem
        pil_img = PILImage.fromarray(img_uint8[i]).resize((target_size, target_size), PILImage.BILINEAR)
        arr = np.array(pil_img, dtype=np.float32) / 255.0  #passem la imatge PIL a numpy i obtenim un range de [0, 1]
        # Repliquem el canal gris a RGB (volem la imatge a color pel model SAM2)
        out[i] = np.stack([arr, arr, arr], axis=0)
    return out



def liver_segmentation(ct, mask_loaded, n_propmts_target = 12):
    """
    Paràmetres:
        ct: volum CT (slices, H, W) en HU.
        mask_loaded: màscara ground truth alineada (slices, H, W), valors 0/1/2.
        every_n: separació (en talls) entre prompts de màscara.

    Retorna la màscara 3D de la segmentació (slices, H, W) de MedSAM2.
    """
    mask_loaded = (mask_loaded > 0).astype(np.uint8)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Llista de (slice_idx, mask_2d) per prompts de mascara
    prompts = extract_prompt_boxes(mask_loaded, n_prompts_target=n_propmts_target)
    print(f"  [prompt] {len(prompts)} talls ancorats: {[idx for idx, _ in prompts]}")

    predictor = build_sam2_video_predictor_npz(CONFIG, CHECKPOINT, device=device)

    # Preprocessat i tensor (igual que abans)
    ct_uint8 = preproces(ct, level=60, width=150)
    ct_rgb = redim_vol_a_rgb(ct_uint8)
    tensor = torch.from_numpy(ct_rgb).to(device=device, dtype=torch.float32)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    tensor = (tensor - mean) / std

    height, width = ct.shape[1], ct.shape[2]

    segs_3D = np.zeros(ct_uint8.shape, dtype=np.uint8)

    inference_state = predictor.init_state(tensor, height, width)
    for idx, bbox in prompts:
        predictor.add_new_points_or_box(inference_state, idx, obj_id=1, box=bbox)

    # Frames anotats, ordenats. Endavant arrenquem del mes baix; enrere, del mes alt.
    prompt_indices = sorted(idx for idx, _ in prompts)
    first_prompt = prompt_indices[0]
    last_prompt = prompt_indices[-1]

    # ENDAVANT: des del primer prompt fins al final del volum.
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=first_prompt, reverse=False):
        segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

    # ENRERE: des de l'ULTIM prompt cap a 0, SENSE reset ni reinicialitzar l'estat.
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state, start_frame_idx=last_prompt, reverse=True):
        segs_3D[out_frame_idx, (out_mask_logits[0] > 0.0).cpu().numpy()[0]] = 1

    predictor.reset_state(inference_state)

    # Ens quedem nomes amb la component connexa mes gran: el fetge es un unic
    # volum, i MedSAM2 pot haver deixat illots a organs veins durant la propagacio.
    segs_3D = utils.get_largest_connected_component(segs_3D)
    #segs_3D = binary_fill_holes(segs_3D).astype(np.uint8)

    return segs_3D