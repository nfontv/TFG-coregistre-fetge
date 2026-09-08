"""
Co-registre rigid del fetge: TAULA COMPLETA (totes les perdues x tots els optimitzadors) en PARAL.LEL sobre N nuclis.

Us:
    python run.py                 # graella completa, tots els nuclis
    python run.py --n_jobs 2 --maxiter 20   # prova petita

Cada parella (pacient, fase_mobil) executa les 12 configs. PV es SEMPRE la fase fixa.
La taula mestra (format llarg) s'escriu UN COP al final.
Historial per config: matriu (n_feval, 2) -> col 0 = loss, col 1 = DSC.
"""

# ---- Control de fils: HA d'anar ABANS d'importar numpy/scipy/sitk ----
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import csv
import argparse
import itertools
from datetime import datetime
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import SimpleITK as sitk

from src import dataset, utils, transformation, metrics, optimitzadors


BASE_PATH = Path(__file__).parent

# dataset del pen drive
DATASET_PATH = Path(r"C:\Users\Noemí Font\Desktop\CoregistreRigidFetge-main\data tgf noemi\manifest-1783073007564\HCC-TACE-Seg")
# camí de windows
SEGMENTATIONS_PATH = Path(r"C:\Users\Noemí Font\Desktop\CoregistreRigidFetge-main\segmentations_MedSam2")
TAULA_COMPLEMENTARIA_PATH = Path(r"C:\Users\Noemí Font\Desktop\CoregistreRigidFetge-main\41597_2023_1928_MOESM1_ESM.xlsx")
FILTRE_SEGMENTACIONS_CVS_PATH = Path(r"C:\Users\Noemí Font\Desktop\CoregistreRigidFetge-main\filtre_segmentacions\metriques_filtre.csv")

DESCARTS_MANUALS = {"HCC_054", "HCC_089"}

FASE_FIXA = "pv"
FASES_MOBILS = ["arterial", "pre-contrast"]

RESULTS_BASE = BASE_PATH / "results_coregistre"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)

# None -> tots els pacients. NOMES_PACIENTS = None
# Prova NOMES_PACIENTS = ["HCC_041", "HCC_025"]
#NOMES_PACIENTS = ["HCC_057", "HCC_019", "HCC_080", "HCC_060"]
NOMES_PACIENTS = None

#NOMES_PACIENTS = ["HCC_057", "HCC_019", "HCC_080", "HCC_060", "HCC_062", "HCC_050"]

#OPTIMIZERS = ["lbfgsb"]

#Totes les combinacions perdua x optimitzador.
LOSSES = ["soft_dice","assd", "hd95","dsc"]
OPTIMIZERS = ["powell", "lbfgsb", "cmaes"]


#  Funcions de perdua
def make_loss(loss_name):
    if loss_name == "dsc":
        return metrics.loss_function_for_optimizer_dice
    if loss_name == "soft_dice":
        return metrics.loss_function_for_optimizer_soft_dice
    if loss_name in ("assd", "hd95"):
        return loss_name  # marcador; loss_superficie l'usa
    raise ValueError(f"Perdua desconeguda: {loss_name!r}")



def loss_overlap(params, mask_fixa, mask_mobil, loss_pure, continu=False):
    """Perdua basada en solapament (dsc / soft_dice)."""
    if continu:
        mobil_t = transformation.transform_region_continu(mask_mobil, params)
    else:
        mobil_t = transformation.transform_region(mask_mobil, params)
    return loss_pure(mask_fixa, mobil_t)

def loss_superficie(params, mask_fixa, mask_mobil, spacing, loss_name):
    """Perdua de superficie (assd / hd95), dins la bounding box conjunta."""
    mobil_t = transformation.transform_region(mask_mobil, params)
    if int(mobil_t.sum()) == 0:
        return 1e6
    d_1to2, d_2to1 = metrics.distancia_superficies_roi(mask_fixa, mobil_t, spacing)
    if loss_name == "assd":
        return metrics.metric_to_compare_two_regions_assd(d_1to2, d_2to1)
    elif loss_name == "hd95":
        return metrics.metric_to_compare_two_regions_hd95(d_1to2, d_2to1)
    raise ValueError(f"loss_name de superficie desconegut: {loss_name!r}")


# Taula de tots els experiments
def append_to_master(files_csv, master_path):
    """Actualitza el CSV de tots els experiments; substitueix les files de les configs presents."""
    df_nou = pd.DataFrame(files_csv)
    configs_noves = set(df_nou["config"].unique())
    if os.path.exists(master_path):
        df_vell = pd.read_csv(master_path)
        df_vell = df_vell[~df_vell["config"].isin(configs_noves)]
        df_total = pd.concat([df_vell, df_nou], ignore_index=True)
    else:
        df_total = df_nou
    df_total.to_csv(master_path, index=False)
    return master_path


# UNA parella, TOTES les configs
def process_one_pair(task, config_grid, maxiter, out_dir):
    """Executat en un proces separat. Retorna una llista de files (una per config)."""
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    name = task["name"]
    fase_mobil = task["fase_mobil"]
    sample = task["sample"]

    fases = utils.carrega_fases_pacient(sample, SEGMENTATIONS_PATH)
    if FASE_FIXA not in fases or fase_mobil not in fases:
        return []

    mask_fixa_np, mask_mobil_np, img_fixa, img_mobil = utils.resample_to_common_grid(
        fases[FASE_FIXA]["mask"], fases[FASE_FIXA]["dcm"],
        fases[fase_mobil]["mask"], fases[fase_mobil]["dcm"],
        is_mask=True, return_sitk=True)

    x0 = transformation.transformacio_centroides(mask_fixa_np, mask_mobil_np)
    spacing = img_fixa.GetSpacing()

    def numpy_a_sitk(arr):
        img = sitk.GetImageFromArray(arr.astype(np.uint8))
        img.CopyInformation(img_fixa)
        return img

    def avalua_totes_metriques(mobil_transformada):
        return metrics.all_metrics_to_compare_two_regions(img_fixa, numpy_a_sitk(mobil_transformada))

    # Linies base (independents de la config): UN COP.
    metrics_identitat = avalua_totes_metriques(transformation.transform_region(mask_mobil_np, np.zeros(6)))
    metrics_centroides = avalua_totes_metriques(transformation.transform_region(mask_mobil_np, x0))

    # Monitor comu: DSC de la mascara transformada respecte la fixa (per avaluacio).
    def monitor_dsc(params):
        mobil_t = transformation.transform_region(mask_mobil_np, params)
        return metrics.metric_to_compare_two_regions_dsc(mask_fixa_np, mobil_t)

    hist_dir = out_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for loss_name, optimizer in config_grid:
        config = f"{loss_name}_{optimizer}"
        loss_pure = make_loss(loss_name)


        if loss_name in ("dsc", "soft_dice"):
            loss_fn = partial(loss_overlap,
                              mask_fixa=mask_fixa_np, mask_mobil=mask_mobil_np,
                              loss_pure=loss_pure,
                              continu=(loss_name == "soft_dice"))
        else:
            loss_fn = partial(loss_superficie,
                              mask_fixa=mask_fixa_np, mask_mobil=mask_mobil_np,
                              spacing=spacing, loss_name=loss_name)

        opt = optimitzadors.run_optimizer(
            loss_fn=loss_fn, x0=x0, optimizer=optimizer,
            maxiter=maxiter, monitor_fn=monitor_dsc)

        hist_path = hist_dir / f"{name}_{fase_mobil}_{config}.npy"
        np.save(hist_path, np.array(opt.history, dtype=float))

        metrics_final = avalua_totes_metriques(transformation.transform_region(mask_mobil_np, opt.x))

        files.append({
            "pacient": name,
            "fase_fixa": FASE_FIXA,
            "fase_mobil": fase_mobil,
            "parella": f"{FASE_FIXA}<-{fase_mobil}",
            "loss_name": loss_name,
            "optimizer": optimizer,
            "config": config,
            "dsc_final": metrics_final.get("DSC", ""),
            "assd_final": metrics_final.get("ASSD_mm", ""),
            "hd_final": metrics_final.get("HD_mm", ""),
            "hd95_final": metrics_final.get("HD95_mm", ""),
            "nsd_final": metrics_final.get("NSD", ""),
            "dsc_identitat": metrics_identitat.get("DSC", ""),
            "assd_identitat": metrics_identitat.get("ASSD_mm", ""),
            "hd95_identitat": metrics_identitat.get("HD95_mm", ""),
            "dsc_centroides": metrics_centroides.get("DSC", ""),
            "assd_centroides": metrics_centroides.get("ASSD_mm", ""),
            "hd95_centroides": metrics_centroides.get("HD95_mm", ""),
            "loss_final": round(float(opt.fun), 6),
            "n_iter": opt.n_iter,
            "n_feval": opt.n_feval,
            "convergence_ok": opt.converged,
            "temps_s": round(opt.time_s, 3),
            "params_opt": np.array2string(opt.x, precision=4, separator=","),
            "history_path": str(hist_path),
        })

    return files


#  Filtre de segmentacions 
def construeix_fases_filtrades():
    fases_filtrades = {}
    df_filtre = pd.read_csv(FILTRE_SEGMENTACIONS_CVS_PATH)
    for _, row in df_filtre.iterrows():
        pacient = row["pacient"]
        fases = set()
        if row["dsc_pv_pre-contrast"] >= 0.8:
            fases.update(["pv", "pre-contrast"])
        if row["dsc_arterial_pre-contrast"] >= 0.8:
            fases.update(["arterial", "pre-contrast"])
        if fases:
            fases_filtrades[pacient] = sorted(fases)
    return fases_filtrades

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=None,
                        help="Nombre de processos. Per defecte: tots els nuclis.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    out_dir = RESULTS_BASE / f"taula_resultats_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    my_dataset = dataset.load_dataset(DATASET_PATH, TAULA_COMPLEMENTARIA_PATH, DESCARTS_MANUALS)
    fases_filtrades = construeix_fases_filtrades()

    tasks = []
    for sample in my_dataset:
        name = sample["name"]
        if NOMES_PACIENTS and name not in NOMES_PACIENTS:
            continue
        if name not in fases_filtrades or FASE_FIXA not in fases_filtrades[name]:
            continue
        if not (SEGMENTATIONS_PATH / name).exists():
            continue
        for fase_mobil in FASES_MOBILS:
            if fase_mobil not in fases_filtrades[name]:
                continue
            tasks.append({"name": name, "fase_mobil": fase_mobil, "sample": sample})

    config_grid = list(itertools.product(LOSSES, OPTIMIZERS))
    print(f"{len(tasks)} parelles x {len(config_grid)} configs = "
          f"{len(tasks) * len(config_grid)} registres. n_jobs={args.n_jobs or os.cpu_count()}")

    worker = partial(process_one_pair, config_grid=config_grid,
                     maxiter=args.maxiter, out_dir=out_dir)

    totes_files = []
    with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                files = fut.result()
                totes_files.extend(files)
                print(f"[{i}/{len(tasks)}] {t['name']} {t['fase_mobil']}: {len(files)} configs")
            except Exception as e:
                import traceback
                print(f"[{i}/{len(tasks)}] {t['name']} {t['fase_mobil']}: ERROR -> {e}")
                traceback.print_exc()

    if not totes_files:
        print("Cap registre generat (revisa camins / filtre / cache).")
        return

    csv_path = out_dir / "results_grid.csv"
    camps = list(totes_files[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=camps)
        writer.writeheader()
        writer.writerows(totes_files)
    print(f"\nCSV desat a: {csv_path}  ({len(totes_files)} registres)")

    #master_path = RESULTS_BASE / "tots_els_experiments.csv"
    #append_to_master(totes_files, str(master_path))
    #print(f"Taula mestra actualitzada: {master_path}")


if __name__ == "__main__":
    main()