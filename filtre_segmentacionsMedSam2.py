"""
Filtre de les segmentacions tretes amb el model, abans del co-registre.
Retorna cvs amb les distancies entre cada fase amb la màscara groundtruth i les distàncies entre fases de cada pacient.
També retorna un png de cada fase amb la seva màscara per el filtre manual

"""

import csv
import numpy as np
from pathlib import Path

from src import dataset, utils, metrics
BASE_PATH = Path(__file__).parent

# Rutes centralitzades a config.py (les dades externes es configuren amb la
# variable d'entorn TFG_DATA; veure config.py i el README).
# SEGMENTATIONS_PATH ha de coincidir amb la carpeta on segmentacio_pacients.py
# va desar les segmentacions; CSV_PATH es el mateix fitxer que llegeix run.py.
from config import (
    DATASET_PATH,
    TAULA_COMPLEMENTARIA_PATH,
    SEGMENTATIONS_PATH,
    FILTRE_SEGMENTACIONS_CSV_PATH as CSV_PATH,
)

DESCARTS_MANUALS = {"HCC_054", "HCC_089"}
OUTPUT_PATH = CSV_PATH.parent

# None = tots els pacients. Llista per provar-ne uns quants.
NOMES_PACIENTS = None
#NOMES_PACIENTS = ["HCC_024","HCC_029","HCC_043","HCC_056","HCC_057","HCC_060","HCC_062","HCC_084","HCC_075", "HCC_089","HCC_093","HCC_092","HCC_095","HCC_099","HCC_104"]
#NOMES_PACIENTS = ["HCC_029"]





# Carrega de fases d'un pacient


PARELLES = [("pv", "arterial"), ("pv", "pre-contrast"), ("arterial", "pre-contrast")]

def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    my_dataset = dataset.load_dataset(DATASET_PATH, TAULA_COMPLEMENTARIA_PATH, DESCARTS_MANUALS)

    files_csv = []

    for sample in my_dataset:
        name = sample["name"]
        if NOMES_PACIENTS and name not in NOMES_PACIENTS:
            continue
        if not (SEGMENTATIONS_PATH / name).exists():
            continue

        print(f"\n=== {name} ===")
        fases = utils.carrega_fases_pacient(sample,SEGMENTATIONS_PATH)
        if len(fases) < 1:
            print("  sense fases cachejades; salto")
            continue

        # GroundTruth alineada a cada fase i RVD/DSC vs GroundTruth
        # El groundtruth es sobre PV; per a PRE/AP es una aproximació geometrica.
        groundtruth_ = {}
        if sample["segmentation_path"] is not None:
            try:
                dicomseg, mask_gt_raw = utils.load_segmentation(sample["segmentation_path"])
                for fase, data in fases.items():
                    gt_aligned = utils.resample_mask_to_ct(mask_gt_raw, dicomseg, data["dcm"])
                    groundtruth_[fase] = {
                        "rvd": metrics.metric_to_compare_two_regions_rvd(data["mask"], gt_aligned),  # seg respecte GT
                        "dsc": metrics.metric_to_compare_two_regions_dsc(data["mask"], gt_aligned),
                    }
                    flag = "" if fase == "pv" else "  (aprox: GT no es propi d'aquesta fase)"
                    print(f"  [{fase} vs GT] RVD={groundtruth_[fase]['rvd']:+.3f}  DSC={groundtruth_[fase]['dsc']:.3f}{flag}")
            except Exception as e:
                print(f"  [avis] GT no comparable: {e}")

        # Parelles de fases entre elles
        comparacio_entre_fases = {}
        for fa, fb in PARELLES:
            if fa in fases and fb in fases:
                try:
                    r, d = utils.compara_parella(fases[fa], fases[fb])
                    comparacio_entre_fases[(fa, fb)] = {"rvd": r, "dsc": d}
                    print(f"  [{fa} vs {fb}] RVD={r:+.3f}  DSC={d:.3f}")
                except Exception as e:
                    print(f"  [avis] parella {fa}-{fb} fallida: {e}")

        # Score de sospita per ordenar la revisió visual
        # Com més baix el DSC inter-fase i mes alt l'|RVD| inter-fase, més sospitos.
        dsc_entre_parelles = [v["dsc"] for v in comparacio_entre_fases.values() if not np.isnan(v["dsc"])]
        rvd_entre_parelles = [abs(v["rvd"]) for v in comparacio_entre_fases.values() if not np.isnan(v["rvd"])]
        min_dsc_entre_parelles = min(dsc_entre_parelles) if dsc_entre_parelles else float("nan")
        max_abs_rvd_entre_parelles = max(rvd_entre_parelles) if rvd_entre_parelles else float("nan")

        #Visualització
        utils.visualitza_fases(name, fases, OUTPUT_PATH)

        # Fila del CSV
        fila = {
            "pacient": name,
            "fases": "|".join(sorted(fases.keys())),
            "min_dsc_entre_parelles": round(min_dsc_entre_parelles, 4) if not np.isnan(min_dsc_entre_parelles) else "",
            "max_abs_rvd_entre_parelles": round(max_abs_rvd_entre_parelles, 4) if not np.isnan(max_abs_rvd_entre_parelles) else "",
        }
        for fase in ("pre-contrast", "arterial", "pv"):
            fila[f"rvd_{fase}_vs_gt"] = round(groundtruth_[fase]["rvd"], 4) if fase in groundtruth_ else ""
            fila[f"dsc_{fase}_vs_gt"] = round(groundtruth_[fase]["dsc"], 4) if fase in groundtruth_ else ""
        for fa, fb in PARELLES:
            key = (fa, fb)
            fila[f"rvd_{fa}_{fb}"] = round(comparacio_entre_fases[key]["rvd"], 4) if key in comparacio_entre_fases else ""
            fila[f"dsc_{fa}_{fb}"] = round(comparacio_entre_fases[key]["dsc"], 4) if key in comparacio_entre_fases else ""
        files_csv.append(fila)

    # Escriure CSV ordenat per sospita (DSC inter-fase mes baix primer)
    if files_csv:
        def clau_ordre(f):
            v = f["min_dsc_entre_parelles"]
            return v if isinstance(v, float) else 1.0  # sense dada -> al final
        files_csv.sort(key=clau_ordre)

        camps = list(files_csv[0].keys())
        with open(CSV_PATH, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=camps)
            writer.writeheader()
            writer.writerows(files_csv)
        print(f"\nCSV desat a: {CSV_PATH}  ({len(files_csv)} pacients)")
    print(f"PNGs a: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()