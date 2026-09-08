import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter


def loss_function_for_optimizer_dice(mask1, mask2):
    """
    Aquesta funció de pèrdua es basa en la mètrica DSC, i com hem de minimitzar definirem 1 - DSC
    """
    m1 = mask1.astype(bool) #passem les mascares a booleanes
    m2 = mask2.astype(bool)

    #interseccio dels dos volums
    interseccio = np.count_nonzero(m1 & m2)

    #suma volum total
    mask1_volum = np.count_nonzero(m1 == 1)
    mask2_volum = np.count_nonzero(m2 == 1)
    suma_voxels = mask1_volum + mask2_volum

    # Si les dues mascares son buides, considerem que el solapament es perfecte (perdua = 0)
    if suma_voxels == 0:
        return 0.0

    dice = (2.0 * interseccio) / suma_voxels

    # Retornem 1 - DSC
    return 1.0 - dice

def loss_function_for_optimizer_soft_dice(mask1,mask2,sigma=2):
    """Soft-dice: soft_dice = 2 * sum(p*q) / (sum(p) + sum(q))"""
    m1 = mask1.astype(float)
    m2 = mask2.astype(float)

    p = gaussian_filter(m1,sigma)
    q = gaussian_filter(m2,sigma)
    num = 2.0 * np.sum(p * q)
    den = np.sum(p) + np.sum(q)
    if den == 0:
        return 1.0  # dues mascares buides -> solapament perfecte per conveni

    return 1.0 - num / den


def metric_to_compare_two_regions_dsc(mask1, mask2): #posare la mateixa demoment. La meva intencia és posar-ne més d'una
    """
       Aquesta funció de pèrdua es basa en la mètrica DSC, i com hem de minimitzar definirem 1 - DSC
       """
    m1 = mask1.astype(bool)  # ens asseguram que les màscares siguin binàries
    m2 = mask2.astype(bool)

    # intersecció dels dos volums
    interseccio = np.count_nonzero(m1 & m2)

    # suma volum total
    mask1_area = np.count_nonzero(m1 == 1)
    mask2_area = np.count_nonzero(m2 == 1)
    suma_voxels = mask1_area + mask2_area

    # Si les dues mascares son buides, considerem que el solapament es perfecte (perdua = 0)
    if suma_voxels == 0:
        return 0.0

    dice = (2.0 * interseccio) / suma_voxels

    # Retornem  DSC
    return dice
def metric_to_compare_two_regions_rvd(mask1, mask2):
    """
    Relative Volume Difference de mask1 respecte mask2 (referència).
    RVD = (|mask1| - |mask2|) / |mask2|

    Retorna NaN si la referencia es buida.
    """
    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    v1 = np.count_nonzero(m1)
    v2 = np.count_nonzero(m2)
    if v2 == 0:
        return float("nan")
    return float((v1 - v2) / v2)

def distancia_superficies(mask1,mask2):
    """
    Paràmetres: màscares binàries en format stik.Image
    Calcula per a cada punt de la superfície d'una màscara la distància en mm al punt més proper de la superfície
    de l'altra, i ho fa en ambdós sentits.
     """
    #Prenem la frontera
    frontera1 = sitk.LabelContour(mask1, fullyConnected = True)
    frontera2 = sitk.LabelContour(mask2, fullyConnected = True)

    #distància de cada voxel (de tot l'espai) a cada punt de la frontera
    distanciafrontera1 = sitk.Abs(sitk.SignedMaurerDistanceMap(
        frontera1, squaredDistance=False, useImageSpacing=True))
    distanciafrontera2 = sitk.Abs(sitk.SignedMaurerDistanceMap(
        frontera2, squaredDistance=False, useImageSpacing=True))

    #convertim les imatges simpleITK a numpy
    arr_frontera1 = sitk.GetArrayViewFromImage(frontera1).astype(bool)
    arr_frontera2 = sitk.GetArrayViewFromImage(frontera2).astype(bool)
    arr_distfront1 = sitk.GetArrayViewFromImage(distanciafrontera1)
    arr_distfront2 = sitk.GetArrayViewFromImage(distanciafrontera2)


    d_1to2 = arr_distfront2[arr_frontera1] #vector de distancies d(x, ∂B) per cada x de A
    d_2to1 = arr_distfront1[arr_frontera2] #simetric de B a A

    return d_1to2, d_2to1

def _is_empty(img):
    """True si la màscara no té cap vòxel positiu"""
    return int(sitk.GetArrayViewFromImage(img).sum()) == 0

#distància Average Symmetric Surface Distance
def distancia_superficies_roi(mask_fixa_arr, mask_mobil_arr, spacing, pad=3):
    """
    Igual que distancia_superficies però retallant a la bounding box conjunta.
    Resultat numèricament idèntic al càlcul sobre tota la graella (cap superfície
    truncada), però molt més ràpid.

    mask_*_arr : arrays 3D numpy (z, y, x), mateixa forma.
    spacing    : (sx, sy, sz) en mm (ordre sitk x, y, z), com img.GetSpacing().
    pad        : marge en vòxels perquè la frontera no toqui la vora del retall.
    """
    union = (mask_fixa_arr > 0) | (mask_mobil_arr > 0)
    if not union.any():
        return np.array([]), np.array([])

    zs, ys, xs = np.where(union)
    Z, Y, X = union.shape
    z0, z1 = max(zs.min() - pad, 0), min(zs.max() + pad + 1, Z)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, Y)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, X)

    f_crop = mask_fixa_arr[z0:z1, y0:y1, x0:x1]
    m_crop = mask_mobil_arr[z0:z1, y0:y1, x0:x1]

    img_f = sitk.GetImageFromArray(f_crop.astype(np.uint8)); img_f.SetSpacing(spacing)
    img_m = sitk.GetImageFromArray(m_crop.astype(np.uint8)); img_m.SetSpacing(spacing)

    return distancia_superficies(img_f, img_m)  # reutilitza la teva funció existent

def metric_to_compare_two_regions_assd(d_1to2, d_2to1):
    """
    Atributs: distancies entre superficies de les dues mascaresja calculades
    Retorna la distància: ASSD = ( Σ_{p∈∂A} d(p,∂B) + Σ_{q∈∂B} d(q,∂A) ) / (|∂A| + |∂B|)
    """

    n1, n2 = len(d_1to2), len(d_2to1)
    if n1 + n2 == 0:
        return float("nan")

    return float((d_1to2.sum() + d_2to1.sum()) / (n1 + n2))

def metric_to_compare_two_regions_hd(d_1to2,d_2to1):
    """

    Distancia Hausdorff
    HD(A, B) = max( max_{x in ∂A} d(x, ∂B),
                    max_{y in ∂ B} d(y, ∂A) )
    """
    if len(d_1to2) == 0 or len(d_2to1) == 0:
        return float("nan")
    return float(max(d_1to2.max(), d_2to1.max()))

def metric_to_compare_two_regions_hd95(d_1to2, d_2to1):
    """
    Atributs: vector de distancies entre superficies en les dues direccions
    Retorna HD95 = max( P95({d(p,∂B) : p∈∂A}), P95({d(q,∂A) : q∈∂B}) )
    (en mm)
    """
    if len(d_1to2) == 0 or len(d_2to1) == 0:
        return float("nan")

    return float(max(np.percentile(d_1to2, 95), np.percentile(d_2to1, 95)))

def metric_to_compare_two_regions_nsd(d_1to2, d_2to1, tau):
    """
    Surface Dice at tolerance (NSD)

    """
    if tau < 0:
        raise ValueError("tau ha de ser >= 0")

    n1, n2 = len(d_1to2), len(d_2to1)
    if n1 + n2 == 0:
        return float("nan")

    within = np.count_nonzero(d_1to2 <= tau) + np.count_nonzero(d_2to1 <= tau)
    return float(within / (n1 + n2))

def all_metrics_to_compare_two_regions(mask1,mask2,tau=2.0):
    arr_1 = sitk.GetArrayViewFromImage(mask1)
    arr_2 = sitk.GetArrayViewFromImage(mask2)
    results = {"DSC": metric_to_compare_two_regions_dsc(arr_1, arr_2)}

    if _is_empty(mask1) or _is_empty(mask2):
        results.update({"ASSD_mm": float("nan"),
                        "HD_mm": float("nan"),  # <-- aquesta
                        "HD95_mm": float("nan"),
                        "NSD": float("nan")})
        return results
    # Càlcul de distàncies; reutilitzat per ASSD, HD95 i NSD.
    d_1to2, d_2to1 = distancia_superficies(mask1, mask2)

    results["ASSD_mm"] = metric_to_compare_two_regions_assd(d_1to2, d_2to1)
    results["HD_mm"] = metric_to_compare_two_regions_hd(d_1to2,d_2to1)
    results["HD95_mm"] = metric_to_compare_two_regions_hd95(d_1to2, d_2to1)
    results["NSD"] = metric_to_compare_two_regions_nsd(d_1to2, d_2to1, tau)
    return results

