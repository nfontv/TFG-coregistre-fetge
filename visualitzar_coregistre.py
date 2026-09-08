""""
visualitza_resultat_coregistre.py

Visualitzacio DIAGNOSTICA a posteriori del resultat del co-registre, per a un
pacient + fase mobil + config concrets. Treballa a la GRAELLA COMUNA (l'espai on
es va calcular la transformacio).

Fons (a les dues figures): CT de la fase MOBIL (arterial/pre-contrast) reprojectat
a la graella comuna, SENSE moure, amb la seva segmentacio MedSAM2 a sobre (verd).

  FIGURA 1 — PV sense transformar:  + mascara PV tal qual (vermell)
             Mostra el desalineament de partida entre PV i la fase mobil.

  FIGURA 2 — PV transformada amb la INVERSA dels params optims (vermell)
             La transformacio optima porta mobil -> PV; la seva inversa porta
             PV -> mobil, de manera que la PV es mou cap al CT mobil quiet.
             Mostra l'alineament aconseguit pel registre.

  (verd = seg. mobil  |  vermell = PV  |  groc = coincidencia)

params_opt es LLEGEIX del CSV de resultats (no es recalcula).

Us:
    python visualitzar_coregistre.py --pacient HCC_004 --fase_mobil pre-contrast --csv results_coregistre/tots_els_experiments.csv --config hd95_powell
"""

import os
import ast
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk

from src import dataset, utils, transformation




# --- Camins (ajusta'ls als teus; mateixos que run_nou.py) ---
BASE_PATH = Path(__file__).parent
#DATASET_PATH = Path("/Volumes/noemifont/TFG DATA/manifest-1774974966100/HCC-TACE-Seg")
DATASET_PATH = Path("/Users/noemifontvoorhoeve/Desktop/TFG/manifest-1758635350325/HCC-TACE-Seg")
TAULA_COMPLEMENTARIA_PATH = BASE_PATH / "41597_2023_1928_MOESM1_ESM.xlsx"
DESCARTS_MANUALS = {"HCC_054", "HCC_089"}
SEGMENTATIONS_PATH = BASE_PATH / "segmentations_MedSam2"
OUT_DIR = BASE_PATH / "results_coregistre" / "visualitzacions_resultat_004_hd_95_powell_precontrast"

FASE_FIXA = "pv"


# ---------------------------------------------------------------------------
def llegeix_params_opt(csv_path, pacient, fase_mobil, config):
    """Recupera params_opt (vector 6) del CSV de resultats per a la fila demanada."""
    df = pd.read_csv(csv_path)
    fila = df[(df.pacient == pacient) &
              (df.fase_mobil == fase_mobil) &
              (df.config == config)]
    if fila.empty:
        raise SystemExit(f"No hi ha fila per ({pacient}, {fase_mobil}, {config}) a {csv_path}")
    text = fila.iloc[0]["params_opt"]
    valors = ast.literal_eval(text.replace(" ", ""))
    return np.array(valors, dtype=float)


def llegeix_fila(csv_path, pacient, fase_mobil, config):
    """Retorna la fila sencera del CSV (per accedir a history_path, etc.)."""
    df = pd.read_csv(csv_path)
    fila = df[(df.pacient == pacient) &
              (df.fase_mobil == fase_mobil) &
              (df.config == config)]
    if fila.empty:
        raise SystemExit(f"No hi ha fila per ({pacient}, {fase_mobil}, {config}) a {csv_path}")
    return fila.iloc[0]


def dibuixa(ct_fons, seg_verd, mask_vermell, titol, path_to_save, slice_idx=None):
    """CT de fons (gris) + seg_verd (verd) + mask_vermell (vermell); solapament groc."""
    if slice_idx is None:
        cobertura = seg_verd.sum(axis=(1, 2))
        slice_idx = int(np.argmax(cobertura)) if cobertura.any() else seg_verd.shape[0] // 2

    base = ct_fons[slice_idx]
    g = (seg_verd[slice_idx] > 0)
    r = (mask_vermell[slice_idx] > 0)

    vmin, vmax = np.percentile(base, 1), np.percentile(base, 99)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax)

    alpha = 0.4
    overlay = np.zeros((*base.shape, 4), dtype=float)
    overlay[r, 0] = 1.0; overlay[r, 3] = alpha
    overlay[g, 1] = 1.0; overlay[g, 3] = alpha
    overlay[g & r, :3] = [1.0, 1.0, 0.0]; overlay[g & r, 3] = alpha
    ax.imshow(overlay)

    ax.set_title(titol)
    ax.axis("off")
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    fig.savefig(path_to_save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura desada a: {path_to_save}")


def dibuixa_convergencia(history, titol, path_to_save):
    """
    Corba de convergencia: punts crus (totes les avaluacions) + millor-fins-ara.
    Eix Y centrat en la zona de convergencia (rang ajustat perque sigui interpretable).
    """
    history = np.asarray(history, dtype=float)
    if history.size == 0:
        print("[avis] history buit, no es grafica la convergencia.")
        return
    millor = np.minimum.accumulate(history)
    x = np.arange(1, len(history) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, history, s=8, color="#888780", alpha=0.45, label="avaluacions (cru)")
    ax.plot(x, millor, color="#534AB7", lw=2.0, label="millor fins ara")

    # Centrar l'eix Y en la zona de convergencia (basat en millor-fins-ara).
    y_min = millor.min()
    y_ref = millor[len(millor) // 5]   # valor despres del primer 20% (ja ha baixat)
    marge = (y_ref - y_min) * 0.5 + 1e-3
    ax.set_ylim(y_min - marge, y_ref + marge)

    ax.set_xlabel("Avaluacions funció de pèrdua (n_feval)")
    ax.set_ylabel("Valor de la pèrdua")
    ax.set_title(titol)
    ax.grid(True, alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    fig.savefig(path_to_save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura desada a: {path_to_save}")



#def dibuixa_convergencia(history, titol, path_to_save):
#    """
#    Corba de convergencia: punts crus (totes les avaluacions) + millor-fins-ara.
#    L'eix Y està fixat sempre entre 0 i 0.20 per poder comparar entre experiments.
#    """
#    history = np.asarray(history, dtype=float)
#    if history.size == 0:
#        print("[avis] history buit, no es grafica la convergencia.")
#        return
#    millor = np.minimum.accumulate(history)
#    x = np.arange(1, len(history) + 1)
#
#    fig, ax = plt.subplots(figsize=(7, 4.5))
#    ax.scatter(x, history, s=8, color="#888780", alpha=0.45, label="avaluacions (cru)")
#    ax.plot(x, millor, color="#534AB7", lw=2.0, label="millor fins ara")
#
#    # Fixar l'eix Y de 0.0 a 0.20 de forma permanent
#    ax.set_ylim(0.0, 0.20)
#
#    ax.set_xlabel("Avaluacions funcio de perdua (n_feval)")
#    ax.set_ylabel("Valor de la perdua")
#    ax.set_title(titol)
#    ax.grid(True, alpha=0.3)
#    ax.legend()
#
#    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
#    fig.savefig(path_to_save, dpi=150, bbox_inches="tight")
#    plt.close(fig)
#    print(f"Figura desada a: {path_to_save}")
#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pacient", required=True)
    parser.add_argument("--fase_mobil", required=True, choices=["arterial", "pre-contrast"])
    parser.add_argument("--csv", required=True, help="CSV de resultats")
    parser.add_argument("--config", required=True, help="p. ex. dsc_powell")
    args = parser.parse_args()

    my_dataset = dataset.load_dataset(DATASET_PATH, TAULA_COMPLEMENTARIA_PATH, DESCARTS_MANUALS)
    sample = next((s for s in my_dataset if s["name"] == args.pacient), None)
    if sample is None:
        raise SystemExit(f"Pacient {args.pacient} no trobat.")

    fases = utils.carrega_fases_pacient(sample, SEGMENTATIONS_PATH)
    for f in (FASE_FIXA, args.fase_mobil):
        if f not in fases:
            raise SystemExit(f"{args.pacient}: falta la fase {f}.")

    params_opt = llegeix_params_opt(args.csv, args.pacient, args.fase_mobil, args.config)

    # Graella comuna: mascares fixa (PV) i mobil, + sitk de referencia (com run_nou.py).
    mask_pv_c, mask_mobil_c, img_fixa_c, _ = utils.resample_to_common_grid(
        fases[FASE_FIXA]["mask"], fases[FASE_FIXA]["dcm"],
        fases[args.fase_mobil]["mask"], fases[args.fase_mobil]["dcm"],
        is_mask=True, return_sitk=True,
    )

    # CT mobil reprojectat a la graella comuna (fons quiet a les dues figures).
    ct_mobil_c = sitk.GetArrayFromImage(
        sitk.Resample(
            utils.build_sitk_image(fases[args.fase_mobil]["ct"], fases[args.fase_mobil]["dcm"], is_mask=False),
            img_fixa_c, sitk.Transform(), sitk.sitkLinear, -1000.0
        )
    )

    # FIGURA 1: PV sense transformar.
    path1 = OUT_DIR / f"{args.pacient}_{args.fase_mobil}_{args.config}_1_pv_precoregistre.png"
    dibuixa(ct_mobil_c, mask_mobil_c, mask_pv_c,
            titol=("PV sense transformar (graella comuna)\n"
                   "verd = seg. mobil  |  vermell = PV  |  groc = intersecció"),
            path_to_save=str(path1))

    # FIGURA 2: PV moguda amb la INVERSA dels params optims (PV -> mobil).
    pv_inversa = transformation.transform_region_inversa(mask_pv_c, params_opt)
    path2 = OUT_DIR / f"{args.pacient}_{args.fase_mobil}_{args.config}_2_pv_postcoregistre.png"
    dibuixa(ct_mobil_c, mask_mobil_c, pv_inversa,
            titol=(f"PV transformada ({args.config})\n"
                   "verd = seg. mobil  |  vermell = PV moguda  |  groc = intersecció"),
            path_to_save=str(path2))

    # FIGURA 3: corba de convergencia (de l'history .npy referenciat al CSV).
    fila = llegeix_fila(args.csv, args.pacient, args.fase_mobil, args.config)
    if "history_path" not in fila.index or pd.isna(fila.get("history_path", np.nan)):
        print("[avis] El CSV no te 'history_path'; salto la convergencia. "
              "Cal desar l'history a run_nou.py.")
    else:
        hist_path = Path(fila["history_path"])
        if not hist_path.exists():
            print(f"[avis] No es troba l'history: {hist_path}; salto la convergencia.")
        else:
            history = np.load(hist_path)
            path3 = OUT_DIR / f"{args.pacient}_{args.fase_mobil}_{args.config}_3_convergencia.png"
            dibuixa_convergencia(
                history,
                titol=f"Convergencia - {args.pacient}_{args.fase_mobil}_{args.config}",
                path_to_save=str(path3))


if __name__ == "__main__":
    main()