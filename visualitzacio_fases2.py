"""
Donat un pacient, mostrem els tres talls (axial, coronal, sagital) de cada fase,
SENSE mascara.
"""

import os
import itertools
import numpy as np
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from src import dataset, utils

BASE_PATH = Path(__file__).parent

# Rutes centralitzades a config.py (les dades externes es configuren amb la
# variable d'entorn TFG_DATA; veure config.py i el README).
from config import DATASET_PATH, SEGMENTATIONS_PATH
from config import TAULA_COMPLEMENTARIA_PATH as TAULA_SUPL_PATH

CHECK_VIS_PATH = BASE_PATH / "visualitzacio_fases2"
DESCARTS_MANUALS = {'HCC_054', 'HCC_089'}
NOMES_PACIENTS = ["HCC_045"]   # tres digits! buit ([]) per processar tots

# Finestra HU FIXA (nivell, amplada). Abdominal estandard: L=40, W=400 -> [-160, 240].
HU_WINDOW = (40, 400)


def axial_index_at_z(dcm_slices, target_z):
    """
    Retorna l'index del tall axial la Z fisica del qual es mes propera a target_z.
    dcm_slices ve de utils.load_ct (ja ordenats de major a menor Z).
    """
    zs = [float(s.ImagePositionPatient[2]) for s in dcm_slices]
    return int(np.argmin([abs(z - target_z) for z in zs]))


def visualize_ct_fixed(img, spatial_resolution=(1, 1, 1), slice_idx=(None, None, None),
                       window=(40, 400), path_to_save=None, show=True):
    """
    Igual que utils.visualize pero SENSE mascara i amb finestra HU FIXA.

    window = (level, width) en HU. El contrast NO depen del volum: aixi totes les
    fases es veuen amb la mateixa escala de grisos i son comparables.
    slice_idx = (k_axial, i_coronal, j_sagital); None -> tall central d'aquell eix.
    """
    level, width = window
    vmin = level - width / 2.0
    vmax = level + width / 2.0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # Index de tall: el que ens passin o el central de cada eix
    resolved = list(slice_idx)
    for ax in range(3):
        if resolved[ax] is None:
            resolved[ax] = img.shape[ax] // 2

    # Mides fisiques per no distorsionar (mateixa logica que utils.visualize)
    dz = spatial_resolution[0] if spatial_resolution[0] else 1.0
    dy = spatial_resolution[1] if spatial_resolution[1] else 1.0
    dx = spatial_resolution[2] if spatial_resolution[2] else dy
    phys_ax = np.array([img.shape[0] * dz, img.shape[1] * dy, img.shape[2] * dx])
    scale = 6.0 / max(phys_ax[1], phys_ax[2])
    fig_w = max(phys_ax[2] * scale * 2, 6.0)
    fig_h = max(phys_ax[1] * scale + phys_ax[0] * scale, 4.0)

    fig, fig_axes = plt.subplots(
        2, 2, layout='compressed', figsize=(fig_w, fig_h),
        gridspec_kw={'height_ratios': [max(phys_ax[0], 1), max(phys_ax[1], 1)]}
    )
    fig.delaxes(fig_axes[0, 1])
    for i, j in itertools.product(range(2), range(2)):
        fig_axes[i, j].get_xaxis().set_visible(False)
        fig_axes[i, j].get_yaxis().set_visible(False)

    ax_np2fig = {0: (0, 0), 1: (1, 0), 2: (1, 1)}
    for ax in range(3):
        img_slice = img.take(resolved[ax], axis=ax)
        img_rgb = matplotlib.colormaps['bone'](norm(img_slice))[..., :3]
        px = np.delete(np.array([dz, dy, dx]), ax)
        fig_axes[ax_np2fig[ax]].imshow(img_rgb, aspect=float(px[0]) / float(px[1]))

    if path_to_save is not None:
        os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
        plt.savefig(path_to_save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":
    my_dataset = dataset.load_dataset(DATASET_PATH, TAULA_SUPL_PATH, DESCARTS_MANUALS)

    n_ok = 0
    n_error = 0

    for sample in my_dataset:
        name = sample["name"]
        if NOMES_PACIENTS and name not in NOMES_PACIENTS:
            continue

        all_ct_infos = sample["list_ct_paths"]
        print(f"Pacient: {name}")
        print(f"Fases al dataset: {[ct['phase'] for ct in all_ct_infos]}")

        try:
            # Fase de referencia = la primera. Fixem la Z fisica del seu tall central
            # i despres busquem a cada fase el tall mes proper a aquesta Z.
            ref_dcm, ref_ct = utils.load_ct(all_ct_infos[0])
            target_z = float(ref_dcm[len(ref_dcm) // 2].ImagePositionPatient[2])
            print(f"  Z de referencia (tall central de {all_ct_infos[0]['phase']}): {target_z:.1f} mm")

            for ct_info in all_ct_infos:
                phase = ct_info["phase"]
                print(f"  --- Fase: {phase} ---")

                dcm_slices, ct = utils.load_ct(ct_info)
                spatial_res = utils.get_spatial_resolution(dcm_slices[0])

                # Tall axial per Z fisica; coronal i sagital, al centre.
                k = axial_index_at_z(dcm_slices, target_z)
                slice_idx = (k, ct.shape[1] // 2, ct.shape[2] // 2)

                out_jpg = CHECK_VIS_PATH / f"Fases_{name}_{phase}.jpg"
                visualize_ct_fixed(
                    ct,
                    spatial_resolution=spatial_res,
                    slice_idx=slice_idx,
                    window=HU_WINDOW,
                    path_to_save=str(out_jpg),
                    show=True,   # False -> nomes desa el JPG, sense obrir finestres
                )
                print(f"    Tall axial index {k} (Z~{target_z:.1f} mm) desat a: {out_jpg}")

            n_ok += 1
            print()

        except Exception as e:
            print(f"  Pacient {name} error: {e}")
            n_error += 1
            print()
            continue

    print(f"\nFinalitzat. Pacients OK: {n_ok} | amb error: {n_error}")
    print(f"Visualitzacions desades a: {CHECK_VIS_PATH}")
