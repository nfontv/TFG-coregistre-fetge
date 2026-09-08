"""

Cada figura mostra TRES talls: axial, coronal i sagital

params_opt es LLEGEIX del CSV de resultats (no es recalcula).

Us:
    python visualitzar_coregistre_3talls.py --pacient HCC_057 --fase_mobil pre-contrast --csv results_coregistre/tots_els_experiments.csv --config soft_dice_powell
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



BASE_PATH = Path(__file__).parent

# Rutes centralitzades a config.py (les dades externes es configuren amb la
# variable d'entorn TFG_DATA; veure config.py i el README).
from config import DATASET_PATH, SEGMENTATIONS_PATH, TAULA_COMPLEMENTARIA_PATH

DESCARTS_MANUALS = {"HCC_054", "HCC_089"}
OUT_DIR = BASE_PATH / "results_coregistre" / "visualitzacions_resultat"

FASE_FIXA = "pv"


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


#  Visualitzacio de talls
def _tall_amb_index(vol, eix, idx):
    """Llesca 2D del volum per l'eix (0=axial, 1=coronal, 2=sagital). Convencio (Z,Y,X)."""
    if eix == 0:
        return vol[idx, :, :]
    elif eix == 1:
        return vol[:, idx, :]
    else:
        return vol[:, :, idx]


def _panell(ax, base, g, r, aspect, titol):
    """Un panell: CT gris + overlay verd (seg. mobil) / vermell (PV) / groc (interseccio)."""
    vmin, vmax = np.percentile(base, 1), np.percentile(base, 99)
    ax.imshow(base, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect, origin="lower")
    alpha = 0.4
    overlay = np.zeros((*base.shape, 4), dtype=float)
    overlay[r, 0] = 1.0; overlay[r, 3] = alpha
    overlay[g, 1] = 1.0; overlay[g, 3] = alpha
    overlay[g & r, :3] = [1.0, 1.0, 0.0]; overlay[g & r, 3] = alpha
    ax.imshow(overlay, aspect=aspect, origin="lower")
    ax.set_title(titol, fontsize=10)
    ax.axis("off")


def dibuixa(ct_fons, seg_verd, mask_vermell, titol, path_to_save, spacing):
    """
    Tres talls (axial, coronal, sagital) amb CT de fons + seg_verd (verd) +
    mask_vermell (vermell); solapament en groc.

    spacing: (sx, sy, sz) en mm, tal com el retorna img.GetSpacing() de SITK.
    Volums en convencio (Z, Y, X).
    """
    sx, sy, sz = spacing

    # Index de cada tall: centre de massa de la segmentacio en cada eix.
    if seg_verd.any():
        zc, yc, xc = (int(round(c)) for c in np.argwhere(seg_verd > 0).mean(axis=0))
    else:
        zc, yc, xc = (s // 2 for s in seg_verd.shape)

    # (eix, index, aspect, nom). aspect = mida_fisica_vertical / mida_fisica_horitzontal.
    #   axial   (Z fix): pla Y-X -> aspect = sy/sx (~1)
    #   coronal (Y fix): pla Z-X -> vertical Z (sz), horitzontal X (sx) -> sz/sx
    #   sagital (X fix): pla Z-Y -> vertical Z (sz), horitzontal Y (sy) -> sz/sy
    plans = [
        (0, zc, sy / sx, "Axial"),
        (1, yc, sz / sx, "Coronal"),
        (2, xc, sz / sy, "Sagital"),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    for ax, (eix, idx, aspect, nom) in zip(axs, plans):
        base = _tall_amb_index(ct_fons, eix, idx)
        g = _tall_amb_index(seg_verd, eix, idx) > 0
        r = _tall_amb_index(mask_vermell, eix, idx) > 0
        _panell(ax, base, g, r, aspect, f"{nom} (tall {idx})")

    fig.suptitle(titol, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    fig.savefig(path_to_save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura desada a: {path_to_save}")


def dibuixa_convergencia(history, titol, path_to_save):

    history = np.asarray(history, dtype=float)
    if history.size == 0:
        print("[avis] history buit, no es grafica la convergencia.")
        return

    if history.ndim == 2 and history.shape[1] >= 2:
        loss = history[:, 0]
        dsc = history[:, 1]
    else:
        loss = history.ravel()
        dsc = None

    x = np.arange(1, len(loss) + 1)
    millor_loss = np.minimum.accumulate(loss)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, loss, s=8, color="#888780", alpha=0.45, label="perdua (cru)")
    ax.plot(x, millor_loss, color="#534AB7", lw=2.0, label="perdua (millor fins ara)")
    ax.set_xlabel("Avaluacions funcio de perdua (n_feval)")
    ax.set_ylabel("Valor de la perdua")

    # Centrar l'eix Y de la perdua en la zona de convergencia.
    y_min = millor_loss.min()
    y_ref = millor_loss[len(millor_loss) // 5]
    marge = (y_ref - y_min) * 0.5 + 1e-3
    ax.set_ylim(y_min - marge, y_ref + marge)

    if dsc is not None:
        millor_dsc = np.maximum.accumulate(dsc)
        ax2 = ax.twinx()
        ax2.plot(x, millor_dsc, color="#C1440E", lw=2.0, label="DSC (millor fins ara)")
        ax2.set_ylabel("DSC (monitor)")
        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc="best", fontsize=8)
    else:
        ax.legend(loc="best", fontsize=8)

    ax.set_title(titol)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
    fig.savefig(path_to_save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figura desada a: {path_to_save}")



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

    # Graella comuna: mascares fixa (PV) i mobil, + sitk de referencia.
    mask_pv_c, mask_mobil_c, img_fixa_c, _ = utils.resample_to_common_grid(
        fases[FASE_FIXA]["mask"], fases[FASE_FIXA]["dcm"],
        fases[args.fase_mobil]["mask"], fases[args.fase_mobil]["dcm"],
        is_mask=True, return_sitk=True,
    )

    spacing = img_fixa_c.GetSpacing()   # (sx, sy, sz)

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
                   "verd = seg. mobil  |  vermell = PV  |  groc = interseccio"),
            path_to_save=str(path1),
            spacing=spacing)

    # FIGURA 2: PV moguda amb la INVERSA dels params optims (PV -> mobil).
    pv_inversa = transformation.transform_region_inversa(mask_pv_c, params_opt)
    path2 = OUT_DIR / f"{args.pacient}_{args.fase_mobil}_{args.config}_2_pv_postcoregistre.png"
    dibuixa(ct_mobil_c, mask_mobil_c, pv_inversa,
            titol=(f"PV transformada ({args.config})\n"
                   "verd = seg. mobil  |  vermell = PV moguda  |  groc = interseccio"),
            path_to_save=str(path2),
            spacing=spacing)

    # FIGURA 3: corba de convergencia (de l'history .npy referenciat al CSV).
    fila = llegeix_fila(args.csv, args.pacient, args.fase_mobil, args.config)
    if "history_path" not in fila.index or pd.isna(fila.get("history_path", np.nan)):
        print("[avis] El CSV no te 'history_path'; salto la convergencia.")
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