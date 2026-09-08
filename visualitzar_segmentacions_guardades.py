"""
Genera els .jpg per visualitzar les mascares segmentades per medsam2 i guardades.
"""
import os
from pathlib import Path



from src import dataset, utils


BASE_PATH = Path(__file__).parent
DATASET_PATH = Path("/Volumes/noemifont/TFG DATA/manifest-1774974966100/HCC-TACE-Seg")
SEGMENTATIONS_PATH =  Path("/Users/noemifontvoorhoeve/Desktop/CoregistreRigidFetge-main/segmentations_MedSam2_prova_box_n_adapt3")
OUTPUT_DIR = BASE_PATH / "visualitzacio_seg_MedSam2_prova_box_n_adapt3"

DESCARTS_MANUALS = {'HCC_054',"HCC_089"}
TAULA_COMPLEMENTARIA_PATH = Path('/Users/noemifontvoorhoeve/Desktop/CoregistreRigidFetge-main/41597_2023_1928_MOESM1_ESM.xlsx')

#NOMES_PACIENTS = ["HCC_024","HCC_029","HCC_043","HCC_056","HCC_057","HCC_060","HCC_062","HCC_084","HCC_075", "HCC_089","HCC_093","HCC_092","HCC_095","HCC_099","HCC_104"]
NOMES_PACIENTS = ["HCC_029"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    my_dataset = dataset.load_dataset(DATASET_PATH,TAULA_COMPLEMENTARIA_PATH,DESCARTS_MANUALS)

    n_ok = 0
    n_skip = 0
    n_error = 0
    for sample in my_dataset:
        name = sample["name"]

        if NOMES_PACIENTS and name not in NOMES_PACIENTS:
            continue

        patient_dir = SEGMENTATIONS_PATH / name

        if not patient_dir.exists():
            print(f"[skip] No hi ha cache per {name}")
            n_skip += 1
            continue

        ct_paths = sample["list_ct_paths"]

        for ct_info in ct_paths:
            fase = ct_info["phase"]
            # Cada segmentacio es guarda com "{fase}_segmentation.npy"
            npy_path = patient_dir / f"{fase}_segmentation.npy"

            # Si no existeix la cache d'aquesta fase, la saltem
            if not npy_path.exists():
                print(f"[skip] {name}: sense cache per la fase '{fase}'")
                n_skip += 1
                continue

            out_jpg = OUTPUT_DIR / f"Visualize_{name}_{fase}.jpg"

            # Salt si ja s'ha generat (idempotent)
            if out_jpg.exists():
                print(f"[salt] Ja existeix: {out_jpg.name}")
                continue

            try:
                dcm_slices, ct = utils.load_ct(ct_info)
                mask = utils.load_mask_npy(npy_path)
                spatial_resolution = utils.get_spatial_resolution(dcm_slices[0])

                utils.visualize(
                    img=ct,
                    mask=mask,
                    mask_color_rgb=(0, 0, 1),
                    spatial_resolution=spatial_resolution,
                    path_to_save=out_jpg,
                    show=False,
                )
                print(f"[ok]   Desat: {out_jpg.name}  "
                      f"(voxels fetge: {int(mask.sum()):,})")
                n_ok += 1
            except Exception as e:
                print(f"[err]  {name} {fase}: {type(e).__name__}: {e}")
                n_error += 1


    print(f"\n=== Resum ===")
    print(f"Imatges generades: {n_ok}")
    print(f"Saltats (sense cache): {n_skip}")
    print(f"Errors: {n_error}")
    print(f"Output: {OUTPUT_DIR}")