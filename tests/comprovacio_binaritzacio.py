import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
from scipy.ndimage import affine_transform
from src import transformation

# Cub sintetic: 1000 voxels de "fetge"
mask = np.zeros((20, 20, 20), dtype=np.uint8)
mask[5:15, 5:15, 5:15] = 1
print("Volum original:", mask.sum())

# Translacio de MIG voxel en x (theta = tx,ty,tz,rz,ry,rx)
theta = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

# 1) El que fa el teu codi ACTUAL
res_actual = transformation.transform_region(mask, theta)
print("transform_region (codi actual):", res_actual.sum())

# 2) El que HAURIA de fer: interpolar en float, DESPRES llindar a 0.5
M, offset = transformation.matriu_rot_inv_offset(mask.shape, theta)
camp = affine_transform(mask.astype(np.float64), M, offset=offset,
                        order=1, mode='constant', cval=0.0)
res_correcte = (camp > 0.5).astype(np.uint8)
print("Interpolant en float i llindant a 0.5:", res_correcte.sum())

# 3) La PROVA del problema: quins valors retorna affine_transform?
camp_uint8 = affine_transform(mask, M, offset=offset,
                              order=1, mode='constant', cval=0.0)
print("\ndtype de sortida amb input uint8:", camp_uint8.dtype)
print("valors unics que retorna:", np.unique(camp_uint8))
print("valors unics interpolant en float:", np.unique(camp)[:8], "...")