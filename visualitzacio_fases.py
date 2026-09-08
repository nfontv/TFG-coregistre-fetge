"""
Donat un pacient, mostrem per pantalla els tres talls (axial, coronal, sagital)
de cada fase, SENSE la mascara.
"""

import numpy as np
from pathlib import Path
from src import dataset, utils

BASE_PATH = Path(__file__).parent

# Rutes centralitzades a config.py (les dades externes es configuren amb la
# variable d'entorn TFG_DATA; veure config.py i el README).
from config import DATASET_PATH, SEGMENTATIONS_PATH
from config import TAULA_COMPLEMENTARIA_PATH as TAULA_SUPL_PATH

# Carpeta on desem les visualitzacions de les fases (opcional)
CHECK_VIS_PATH = BASE_PATH / "visualitzacio_fases"
DESCARTS_MANUALS = {'HCC_054', 'HCC_089'}
NOMES_PACIENTS = ["HCC_045"]   # deixa-ho buit ([]) per processar tots els pacients

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
            for ct_info in all_ct_infos:
                phase = ct_info["phase"]
                print(f"  --- Fase: {phase} ---")

                # Carreguem el CT d'aquesta fase. load_ct rep el diccionari complet
                # i retorna la llista de dcm i el volum en HU.
                dcm_slices, ct = utils.load_ct(ct_info)

                # Resolucio espacial (dz, dy, dx) per no distorsionar les imatges.
                spatial_res = utils.get_spatial_resolution(dcm_slices[0])

                # Com que NO volem mascara, en passem una de zeros amb la mateixa
                # forma que el CT. visualize() no dibuixara cap superposicio i,
                # com que no hi ha voxels marcats, agafara el tall central de cada eix.
                no_mask = np.zeros_like(ct, dtype=np.uint8)

                out_jpg = CHECK_VIS_PATH / f"Fases_{name}_{phase}.jpg"
                utils.visualize(
                    ct, no_mask,
                    spatial_resolution=spatial_res,
                    path_to_save=str(out_jpg),  # posa None si no vols desar-ho
                    show=True,                  # True -> obre la finestra per pantalla
                )

                print(f"    Visualitzacio desada a: {out_jpg}")

            n_ok += 1
            print()

        except Exception as e:
            print(f"  Pacient {name} error: {e}")
            n_error += 1
            print()
            continue

    print(f"\nFinalitzat. Pacients OK: {n_ok} | amb error: {n_error}")
    print(f"Visualitzacions desades a: {CHECK_VIS_PATH}")