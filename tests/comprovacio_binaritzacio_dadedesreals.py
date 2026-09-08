import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
from scipy.ndimage import affine_transform
from src import dataset, utils, transformation, metrics
from pathlib import Path

from config import DATASET_PATH, SEGMENTATIONS_PATH
from config import TAULA_COMPLEMENTARIA_PATH as TAULA

ds = dataset.load_dataset(DATASET_PATH, TAULA, {"HCC_054", "HCC_089"})
sample = next(s for s in ds if s["name"] == "HCC_025")
fases = utils.carrega_fases_pacient(sample, SEGMENTATIONS_PATH)

mask_fixa, mask_mobil, img_f, _ = utils.resample_to_common_grid(
    fases["pv"]["mask"], fases["pv"]["dcm"],
    fases["arterial"]["mask"], fases["arterial"]["dcm"],
    is_mask=True, return_sitk=True)

def transform_correcte(mask, theta):
    M, offset = transformation.matriu_rot_inv_offset(mask.shape, theta)
    camp = affine_transform(mask.astype(np.float64), M, offset=offset,
                            order=1, mode='constant', cval=0.0)
    return (camp > 0.5).astype(np.uint8)

x0 = transformation.transformacio_centroides(mask_fixa, mask_mobil)

print(f"{'theta':>28} | {'DSC actual':>10} | {'DSC correcte':>12} | {'dif':>7}")
for delta in [0.0, 0.25, 0.5, 0.75, 1.0]:
    theta = x0.copy()
    theta[0] += delta          # petites variacions en tx
    a = metrics.metric_to_compare_two_regions_dsc(mask_fixa, transformation.transform_region(mask_mobil, theta))
    c = metrics.metric_to_compare_two_regions_dsc(mask_fixa, transform_correcte(mask_mobil, theta))
    print(f"  x0 + [{delta:.2f},0,0,0,0,0]        | {a:10.4f} | {c:12.4f} | {a-c:+7.4f}")