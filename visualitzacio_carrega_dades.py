"""
Verificacio mitjançant la visualització de la carrega de dades
  - una imatge JPG amb la mascara superposada sobre el CT (3 vistes)
"""
import numpy as np
from pathlib import Path
from src import dataset, utils

BASE_PATH = Path(__file__).parent
DATASET_PATH = Path("/Volumes/noemifont/TFG DATA/manifest-1774974966100/HCC-TACE-Seg")
TAULA_SUPL_PATH = Path("/Users/noemifontvoorhoeve/Desktop/CoregistreRigidFetge-main/41597_2023_1928_MOESM1_ESM.xlsx")
SEGMENTATIONS_PATH = Path("/Users/noemifontvoorhoeve/Desktop/CoregistreRigidFetge-main/segmentations_MedSam2")

# Carpeta on desem les visualitzacions de comprovacio
CHECK_VIS_PATH = BASE_PATH / "comprovacio_dataset_visualitzacions_prova2"
DESCARTS_MANUALS = {'HCC_054', 'HCC_089'}
NOMES_PACIENTS = ["HCC_43","HCC_056","HCC_084","HCC_075","HCC_093","HCC_092","HCC_095","HCC_099","HCC_104"]

if __name__ == "__main__":
    my_dataset = dataset.load_dataset(DATASET_PATH,TAULA_SUPL_PATH,DESCARTS_MANUALS)

    sample = next(s for s in my_dataset)
    n_ok =0
    n_error = 0


    for sample in my_dataset:
        name = sample["name"]
        if NOMES_PACIENTS and name not in NOMES_PACIENTS:
            continue

        all_ct_infos = sample["list_ct_paths"]

        print(f"Pacient: {name}")
        print(f"Fases al dataset: {[ct['phase'] for ct in all_ct_infos]}")

        try:
            # Guarda: si no hi ha SEG, ho diem clarament i saltem el pacient.
            if sample["segmentation_path"] is None:
                print(f"  [AVIS] {name}: sense fitxer de segmentacio (SEG). Es salta.")
                print()
                n_error += 1
                continue

            # Carreguem la SEG ground truth una sola vegada per pacient i la
            # corregim d'orientacio (mateixa cadena que segmentacio__pacients).
            dicomseg, mask_loaded_raw = utils.load_segmentation(sample["segmentation_path"])
            orient_seg = utils.get_orientacio_segmentacio(dicomseg)
            mask_loaded_raw = utils.corregir_orientacio(mask_loaded_raw, orient_seg)

            for ct_info in all_ct_infos:
                phase = ct_info["phase"]
                print(f"  --- Fase: {phase} ---")

                # Carreguem el CT d'aquesta fase. load_ct rep el diccionari complet.
                dcm_slices, ct = utils.load_ct(ct_info)

                # Alineem la mascara ground truth a la geometria d'aquest CT.
                mask_aligned = utils.align_mask_to_ct(mask_loaded_raw, dicomseg, dcm_slices)

                # Avis si l'alineament ha quedat buit (fase amb Z fora de tolerancia)
                if int(mask_aligned.sum()) == 0:
                    print(f"    [AVIS] mascara alineada BUIDA per la fase {phase} "
                          f"(possible desalineament Z entre SEG i aquest CT)")

                # Comprovacio visual: superposem ground truth sobre el CT i desem JPG.
                spatial_res = utils.get_spatial_resolution(dcm_slices[0])
                out_jpg = CHECK_VIS_PATH / f"Visualize_{name}_{phase}.jpg"
                utils.visualize(
                    ct, mask_aligned,
                    spatial_resolution=spatial_res,
                    path_to_save=str(out_jpg),
                    show=False,  # no bloquejant; en M2 sense GUI evitem obrir finestres
                )
                print(f"    Visualitzacio desada a: {out_jpg}")

                print(f"    Obrint visualitzador interactiu de {name} / {phase} "
                      f"(tanca la finestra per continuar)...")
                utils.visualize_interactive(
                    ct, mask_aligned,
                    spatial_resolution=spatial_res,
                )

            n_ok += 1
            print()

        except Exception as e:
            print(f"  Pacient {name} error: {e}")
            n_error += 1
            print()
            continue

    print(f"\nFinalitzat. Pacients OK: {n_ok} | amb error: {n_error}")
    print(f"Visualitzacions desades a: {CHECK_VIS_PATH}")
