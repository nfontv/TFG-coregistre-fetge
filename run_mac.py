"""
Co-registre rigid del fetge, una CONFIG (perdua + optimitzador) per execucio.

Us:
    python run_mac.py --loss dsc      --optimizer powell
    python run_mac.py --loss dsc      --optimizer cmaes
    python run_mac.py --loss soft_dice --optimizer lbfgsb
    python run_mac.py --loss assd     --optimizer powell
    python run_mac.py --loss hd95     --optimizer powell

Cada execució escriu UN CSV (una fila per registre: pacient x parella de fases).

Disseny de dades (format llarg, tidy — Wickham 2014):
    una fila = (pacient, fase_mobil, config). PV es SEMPRE la fase fixa.
    Les mètriques d'avaluació (dsc/assd/hd/hd95/nsd) es calculen igual per a tots
    els experiments, independentment de la pèrdua usada.

Origen de les mascares: .npy ja guardades en (com fa filtre_segmentacionsMedSam2.py).
"""

import os
from datetime import datetime
import itertools
from pathlib import Path

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.widgets import Slider
from pandas import read_csv

from pydicom.uid import generate_uid

import pandas as pd


from src import dataset, utils, transformation, metrics, optimitzadors

from scipy.optimize import minimize
import SimpleITK as sitk

from functools import partial

import csv
import argparse

BASE_PATH = Path(__file__).parent

# Rutes centralitzades a config.py (les dades externes es configuren amb la
# variable d'entorn TFG_DATA; veure config.py i el README).
from config import (
    DATASET_PATH,
    SEGMENTATIONS_PATH,
    TAULA_COMPLEMENTARIA_PATH,
    FILTRE_SEGMENTACIONS_CSV_PATH as FILTRE_SEGMENTACIONS_CVS_PATH,
)

DESCARTS_MANUALS = {"HCC_054", "HCC_089"}


FASE_FIXA = "pv"
FASES_MOBILS = ["arterial", "pre-contrast"]


RESULTS_BASE = BASE_PATH / "results_coregistre"
RESULTS_BASE.mkdir(parents=True, exist_ok=True)
NOMES_PACIENTS=["HCC_041","HCC_025","HCC_036","HCC_023"]
#NOMES_PACIENTS=["HCC_004","HCC_005","HCC_006","HCC_009"]


def append_to_master(files_csv, config, master_path):
    """
    Afegeix les files d'aquesta execucio a un CSV mestre acumulat.

    si la taula ja conte files amb la mateixa 'config', les
    elimina abans d'afegir les noves (re-executar un experiment el SUBSTITUEIX
    en lloc de duplicar-lo).
    """
    df_nou = pd.DataFrame(files_csv)
    if os.path.exists(master_path):
        df_vell = pd.read_csv(master_path)
        df_vell = df_vell[df_vell["config"] != config]  # treu versio anterior
        df_total = pd.concat([df_vell, df_nou], ignore_index=True)
    else:
        df_total = df_nou
    df_total.to_csv(master_path, index=False)
    return master_path


def make_loss(loss_name):
    """
    Retorna la funcio de pèrdua corresponent al nom.
    Totes reben dues mascares (numpy 3D, mateixa shape) i retornen un float a minimitzar.

    NOTA: assd i hd95 necessiten geometria fisica (mm) necessiten els sitk.Image per calcular
    distancies de superficie reals.
    """
    if loss_name == "dsc":
        return metrics.loss_function_for_optimizer_dice

    if loss_name == "soft_dice":
        return metrics.loss_function_for_optimizer_soft_dice

    if loss_name in ("assd", "hd95"):
        # Aquestes necessiten sitk.Image, retornem un marcador.
        return loss_name

    raise ValueError(f"Pèrdua desconeguda: {loss_name!r}")

def loss_overlap(params, mask_fixa, mask_mobil, loss_pure):
    """Pèrdua basada en solapament (dsc / soft_dice): nomes numpy."""
    mobil_t = transformation.transform_region(mask_mobil, params)
    return loss_pure(mask_fixa, mobil_t)

def loss_superficie(params, mask_fixa, mask_mobil, spacing, loss_name):
    """Pèrdua de superfície (assd / hd95), calculada dins la bounding box conjunta."""
    mobil_t = transformation.transform_region(mask_mobil, params)

    if int(mobil_t.sum()) == 0:   # la fixa és no buida per construcció
        return 1e6

    d_1to2, d_2to1 = metrics.distancia_superficies_roi(mask_fixa, mobil_t, spacing)

    if loss_name == "assd":
        return metrics.metric_to_compare_two_regions_assd(d_1to2, d_2to1)
    elif loss_name == "hd95":
        return metrics.metric_to_compare_two_regions_hd95(d_1to2, d_2to1)
    else:
        raise ValueError(f"loss_name de superficie desconegut: {loss_name!r}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", required=True,
                        choices=["dsc", "soft_dice", "assd", "hd95"])
    parser.add_argument("--optimizer", required=True,
                        choices=["powell", "lbfgsb", "cmaes"])
    parser.add_argument("--maxiter", type=int, default=100)
    args = parser.parse_args()

    config = f"{args.loss}_{args.optimizer}"
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    out_dir = RESULTS_BASE / f"{config}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"results_{config}.csv"

    my_dataset = dataset.load_dataset(DATASET_PATH, TAULA_COMPLEMENTARIA_PATH, DESCARTS_MANUALS)

    FASES_FILTRADES = {}
    df_filtre = pd.read_csv(FILTRE_SEGMENTACIONS_CVS_PATH)

    # Iterem per cada pacient (cada fila del dataframe)
    for index, row in df_filtre.iterrows():
        pacient = row['pacient']
        fases_guardades = set()  # Utilitzem un set per evitar fases duplicades

        # Comprovem cada parella
        if row['dsc_pv_pre-contrast'] >= 0.8:
            fases_guardades.update(['pv', 'pre-contrast'])

        if row['dsc_arterial_pre-contrast'] >= 0.8:
            fases_guardades.update(['arterial', 'pre-contrast'])


        # Si s'ha superat el llindar en alguna de les parelles, hi haurà fases a la variable
        # Ens assegurem de no afegir pacients buits al diccionari
        if fases_guardades:
            # Convertim el set en una llista (i l'ordenem per tenir un resultat més net)
            FASES_FILTRADES[pacient] = sorted(list(fases_guardades))


    loss_pure = make_loss(args.loss)
    files_csv = []

    for sample in my_dataset:
        name = sample["name"]
        if NOMES_PACIENTS and name not in NOMES_PACIENTS:
            continue
        if FASES_FILTRADES and name not in FASES_FILTRADES:
            print(f"{name} no ha superat cap filtre")
            continue

        if not (SEGMENTATIONS_PATH / name).exists():
            continue

        fases = utils.carrega_fases_pacient(sample, SEGMENTATIONS_PATH)
        if FASE_FIXA not in fases:
            print(f"[salto] {name}: no te fase fixa {FASE_FIXA}")
            continue
        if FASE_FIXA not in FASES_FILTRADES[name]:
            print(f"La fase {FASE_FIXA} de {name} no ha superat el filtre")
            continue


        for fase_mobil in FASES_MOBILS:
            if fase_mobil not in fases:
                continue
            if fase_mobil not in FASES_FILTRADES[name]:
                print(f"La fase {fase_mobil} de {name} no ha superat el filtre.")
                continue

            print(f"\n=== {name} | {FASE_FIXA} <- {fase_mobil} | {config} ===")

            # Portem les dues mascares a una graella fisica comuna isotropica.
            # return_sitk=True ens dona tambe els sitk.Image (necessaris per a assd/hd95 en mm).
            mask_fixa_np, mask_mobil_np, img_fixa, img_mobil = utils.resample_to_common_grid(
                fases[FASE_FIXA]["mask"], fases[FASE_FIXA]["dcm"],
                fases[fase_mobil]["mask"], fases[fase_mobil]["dcm"],
                is_mask=True, return_sitk=True)

            # Inicialització  per centroides.
            x0 = transformation.transformacio_centroides(mask_fixa_np, mask_mobil_np)

            #Construim la funcio de pèrdua concreta.
            if args.loss in ("dsc", "soft_dice"):
                loss_fn = partial(loss_overlap,
                                  mask_fixa=mask_fixa_np,
                                  mask_mobil=mask_mobil_np,
                                  loss_pure=loss_pure)
            else:  # assd / hd95
                loss_fn = partial(loss_superficie,
                                  mask_fixa=mask_fixa_np,
                                  mask_mobil=mask_mobil_np,
                                  spacing=img_fixa.GetSpacing(),
                                  loss_name=args.loss)

            def numpy_a_sitk(arr):
                img = sitk.GetImageFromArray(arr.astype(np.uint8))
                img.CopyInformation(img_fixa)  # spacing + origin + direction de la graella comuna
                return img

            def avalua_totes_metriques(mobil_transformada):
                img_m = numpy_a_sitk(mobil_transformada)
                return metrics.all_metrics_to_compare_two_regions(img_fixa, img_m)

            mobil_identitat = transformation.transform_region(mask_mobil_np, np.zeros(6))
            metrics_identitat = avalua_totes_metriques(mobil_identitat)

            mobil_centroides = transformation.transform_region(mask_mobil_np, x0)
            print(img_fixa.GetPixelIDTypeAsString())  # hauria de dir "8-bit unsigned integer"
            #calculam la millora desde la inicialització per centroides
            metrics_centroides = avalua_totes_metriques(mobil_centroides)

            #Optimitzacio
            #sense bounds
            opt = optimitzadors.run_optimizer(
                loss_fn=loss_fn,
                x0=x0,
                optimizer=args.optimizer,
                maxiter=args.maxiter)
            # Desar l'historial de convergència en un .npy a part
            hist_dir = out_dir / "history"
            hist_dir.mkdir(parents=True, exist_ok=True)
            hist_path = hist_dir / f"{name}_{fase_mobil}.npy"
            np.save(hist_path, np.array(opt.history, dtype=float))

            # Avaluació final
            mobil_opt = transformation.transform_region(mask_mobil_np, opt.x)
            metrics_final = avalua_totes_metriques(mobil_opt)



            # 9) Fila de la taula llarga.
            fila = {
                # --- Identificacio ---
                "pacient": name,
                "fase_fixa": FASE_FIXA,
                "fase_mobil": fase_mobil,
                "parella": f"{FASE_FIXA}<-{fase_mobil}",
                "loss_name": args.loss,
                "optimizer": args.optimizer,
                "config": config,
                # Avaluacio FINAL (despres d'optimitzar)
                "dsc_final": metrics_final.get("DSC", ""),
                "assd_final": metrics_final.get("ASSD_mm", ""),
                "hd_final": metrics_final.get("HD_mm", ""),
                "hd95_final": metrics_final.get("HD95_mm", ""),
                "nsd_final": metrics_final.get("NSD", ""),
                # IDENTITAT (per a la millora real total)
                "dsc_identitat": metrics_identitat.get("DSC", ""),
                "assd_identitat": metrics_identitat.get("ASSD_mm", ""),
                "hd95_identitat": metrics_identitat.get("HD95_mm", ""),
                # CENTROIDES / x0 (per a l'aportacio de l'optimitzador)
                "dsc_centroides": metrics_centroides.get("DSC", ""),
                "assd_centroides": metrics_centroides.get("ASSD_mm", ""),
                "hd95_centroides": metrics_centroides.get("HD95_mm", ""),
                # Diagnostic de l'optimitzacio
                "loss_final": round(float(opt.fun), 6),
                "n_iter": opt.n_iter,
                "n_feval": opt.n_feval,
                "convergence_ok": opt.converged,
                "temps_s": round(opt.time_s, 3),
                "params_opt": np.array2string(opt.x, precision=4, separator=","),
                "history_path": str(hist_path),
            }
            files_csv.append(fila)
            print(f"  loss_final={opt.fun:.4f}  n_feval={opt.n_feval}  t={opt.time_s:.1f}s")

    # Escriure CSV
    if files_csv:
        camps = list(files_csv[0].keys())
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=camps)
            writer.writeheader()
            writer.writerows(files_csv)
        print(f"\nCSV desat a: {csv_path}  ({len(files_csv)} registres)")
        master_path = RESULTS_BASE / "tots_els_experiments.csv"
        append_to_master(files_csv, config, str(master_path))
        print(f"Taula mestra actualitzada: {master_path}")
    else:
        print("\nCap registre generat (revisa NOMES_PACIENTS / cache).")



if __name__ == "__main__":
    main()
