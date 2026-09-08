import numpy as np
from scipy.ndimage import affine_transform
import os


def matriu_rotacio(rx,ry,rz):
    """
    Donats els angles:
    rx: sobre l'eix x
    ry: sobre l'eix y
    rz: sobre l'eix z
    Cream la matiu de rotació 3D
    """

    Rx = np.array([[1,0, 0],
                   [0, np.cos(rx),-np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry),0,np.sin(ry)],
                   [0, 1 ,0],
                   [-np.sin(ry),0,np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0,0,1]])
    return Rz @ Ry @ Rx
def matriu_rot_inv_offset(shape, parametres):
    """
    Funció per garantir que apliquem la mateixa transformació al CT com a la segmentació.
    Paràmetres = (tx,ty,tz,rz,ry,rx)
    """
    tx, ty, tz, rz, ry, rx = parametres
    #affine_transform espera la inversa
    matriu_rot_inv = matriu_rotacio(rx,ry,rz).T
    centre = np.array(shape)/2
    translacio = np.array([tx,ty,tz])
    offset = centre - matriu_rot_inv @ centre - matriu_rot_inv @ translacio
    return matriu_rot_inv, offset

def transform_region(mask, parameters, order=1):
    """
    Apliquem la transformació rígida a una màscara binària
    """

    matriu_rot_inv, offset = matriu_rot_inv_offset(mask.shape,parameters)
    mask_transformada = affine_transform(mask, matriu_rot_inv, offset=offset, order= order, mode='constant', cval=0.0)
    #ens hem d'assegurar que retornem una màscara BINÀRIA
    return (mask_transformada > 0.5).astype(np.uint8)

def transform_region_continu(mask, parameters, order=3):
    """
    Camp interpolat SENSE binaritzar: continu en els parametres theta.

    """
    matriu_rot_inv, offset = matriu_rot_inv_offset(mask.shape, parameters)
    return affine_transform(mask.astype(np.float64), matriu_rot_inv, offset=offset,
                            order=order, mode='constant', cval=0.0)

#def transform_region_continu(mask, parameters, order=None):
    """
    Camp interpolat SENSE binaritzar: continu en els parametres theta.
    order=1 (trilineal) -> C0 en theta, derivada amb salts.
    order=3 (spline cubic) -> C2 en theta, apte per a metodes de gradient.
    """
    if order is None:
        order = int(os.environ.get("TFG_INTERP_ORDER", 1))
    matriu_rot_inv, offset = matriu_rot_inv_offset(mask.shape, parameters)
    return affine_transform(mask.astype(np.float64), matriu_rot_inv, offset=offset,
                            order=order, mode='constant', cval=0.0)

def transform_volume(volume, parameters, cval=-1000.0):
    """
    Transformació rígida a un VOLUM CT (valors continus en HU).

    """
    matriu_rot_inv, offset = matriu_rot_inv_offset(volume.shape, parameters)

    return affine_transform(volume.astype(np.float64), matriu_rot_inv, offset=offset,
        order=3, mode='constant', cval=cval)

def centroide(mask):
    """
    Calculam el centroide d'un volum codificat en una mascara binària.
    c = (1/ |M|) sum_{(i,j,k) in M} (i,j,k)
    on M és el conjunt de voxels del volum
    """
    mask_bool= mask.astype(bool) #true on hi havia 1 i false on hi havia 0
    coordenades_voxels  = np.argwhere(mask_bool)
    return coordenades_voxels.mean(axis=0)

def transformacio_centroides(mask_fixa, mask_mobil):
    """
    Retorna la transalció que s'ha de fer per centar un volum amb un altre
    """
    c_mask_fixa = centroide(mask_fixa)
    c_mask_mobil = centroide(mask_mobil)

    translacio = c_mask_fixa - c_mask_mobil
    theta = np.zeros(6)
    theta[0:3] = translacio
    return theta

def transform_region_inversa(mask, parameters, order=1):
    """
    Inversa de la transformacio afi que aplica affine_transform:
        affine_transform fa  x_in = M x_out + offset
        la inversa es        x_in = M^-1 x_out - M^-1 offset
    Com que M (matriu_rot_inv) es una rotacio, M^-1 = M.T (exacte, sense errors).
    """
    matriu_rot_inv, offset = matriu_rot_inv_offset(mask.shape, parameters)
    M_inv = matriu_rot_inv.T
    offset_inv = -M_inv @ offset
    out = affine_transform(mask, M_inv, offset=offset_inv,
                           order=order, mode='constant', cval=0.0)
    return (out > 0.5).astype(np.uint8)

