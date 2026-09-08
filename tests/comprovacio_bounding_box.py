import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

"""
Comprovacio visual de les BOXES de prompt que passem a MedSAM2.

Objectiu: veure EXACTAMENT que li donem al model abans de llancar la
inferencia completa. Per cada tall amb box, dibuixa el rectangle [x_min, y_min,
x_max, y_max] sobre la llesca axial del CT (mateix preprocessat que la
inferencia). Aixi es detecten a ull els errors tipics:
  - eixos x/y intercanviats (la box surt transposada),
  - origen equivocat (la box surt desplacada),
  - box que no envolta el fetge a les llesques dels extrems.

Reutilitza:
  - ai.preproces           (mateix windowing que la inferencia)
  - ai.extract_prompt_boxes (les boxes REALS que entraran al model)
  - dataset.load_dataset / utils.load_ct / utils.load_mask_npy

Genera un PNG per pacient/fase amb una graella de les llesques que tenen box.
No crida MedSAM2; nomes visualitza els prompts.
"""

import numpy as np
import matplotlib



matplotlib.use("Agg")  # backend sense finestra: nomes desem PNG
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

from src import dataset, utils


from config import DATASET_PATH, TAULA_COMPLEMENTARIA_PATH

BASE_PATH = Path(__file__).parent
OUTPUT_PATH = BASE_PATH / "comprovacio_box_prompt"

# Descarts manuals: ha de coincidir amb el que passes a load_dataset al teu pipeline.
DESCARTS_MANUALS =  {'HCC_054',"HCC_089"}
NOMES_PACIENTS = ["HCC_024","HCC_029","HCC_043","HCC_056","HCC_057","HCC_060","HCC_062","HCC_084","HCC_075", "HCC_089","HCC_093","HCC_092","HCC_095","HCC_099","HCC_104"]

# Quants pacients revisar (None = tots). Per a una comprovacio rapida, posa 2-3.

EVERY_N = 10  # ha de coincidir amb el que faras servir a la inferencia
def preproces(ct,level,width):#MedSAM2 requereix passar les unitats de HU a unit8
    #ct ha de ser n arxiu amb modality == CT

    lower = level - width / 2
    upper = level + width / 2
    clipped = np.clip(ct, lower,upper)
    normalitzats = (clipped-lower)/(upper-lower) #valors entr
    ct_uint8 = (normalitzats*255.0).astype(np.uint8) #queda una mascara amb el fetge "resaltat"
    return ct_uint8

def mask_2d_to_bbox(m2d: np.ndarray) -> np.ndarray:
    """
    Donada una mascara 2D binaria (H, W), retorna la bounding box que la
    conte en format [x_min, y_min, x_max, y_max] (coordenades de pixel).

    Convencio d'eixos: en un array numpy (H, W), el primer eix son les FILES
    (direccio y) i el segon les COLUMNES (direccio x). SAM2 espera la box en
    ordre [x_min, y_min, x_max, y_max], per tant hem de mapar:
        x  <-> columnes (axis=1)
        y  <-> files    (axis=0)
    Aquest ordre es la font d'error tipica; el respectem explicitament.
    """
    files = np.any(m2d > 0, axis=1)   # quines files contenen fetge
    columnes = np.any(m2d > 0, axis=0)  # quines columnes contenen fetge

    y_min, y_max = np.where(files)[0][[0, -1]]
    x_min, x_max = np.where(columnes)[0][[0, -1]]

    # +1 a max perque la box sigui inclusiva del darrer pixel amb fetge
    return np.array([x_min, y_min, x_max + 1, y_max + 1], dtype=np.float32)


def extract_prompt_boxes(mask_loaded, every_n=10):
    """
    Retorna una llista de (slice_idx, bbox) per usar com a prompts de BOX.

    Parametres:
        mask_loaded: np.ndarray (n_slices, H, W), binaria (0/1), alineada al CT
                     de la fase de referencia (PV) i projectada a la graella comuna.
        every_n: cada quants talls amb fetge posem una box.

    Retorna: llista [(slice_idx, bbox), ...] amb bbox = [x_min, y_min, x_max, y_max].
    """
    pixels_per_slice = (mask_loaded > 0).sum(axis=(1, 2))

    if pixels_per_slice.sum() == 0:
        raise ValueError(
            "Mascara buida: cap voxel de fetge, no es pot extreure prompt "
            "(probable desalineament Z entre la SEG i el CT d'aquesta fase)."
        )

    slices_amb_fetge = np.where(pixels_per_slice > 0)[0]
    key_slice_idx = int(np.argmax(pixels_per_slice))

    z0, z1 = int(slices_amb_fetge[0]), int(slices_amb_fetge[-1])
    seleccionats = set(range(z0, z1 + 1, every_n))
    seleccionats.add(z0)
    seleccionats.add(z1)
    seleccionats.add(key_slice_idx)

    prompts = []
    for idx in sorted(seleccionats):
        m2d = (mask_loaded[idx] > 0).astype(np.uint8)
        if m2d.sum() == 0:
            continue  # saltem talls sense fetge dins el rang
        bbox = mask_2d_to_bbox(m2d)
        prompts.append((idx, bbox))

    return prompts


def dibuixa_boxes_pacient(name, ct, mask, prompts, phase, output_dir):
    """
    ct:      volum CT en HU (slices, H, W)
    mask:    mascara ground truth alineada (slices, H, W)
    prompts: llista [(slice_idx, bbox), ...] amb bbox=[x_min,y_min,x_max,y_max]
    """
    if not prompts:
        print(f"  [{phase}] sense boxes (mascara buida?), salto")
        return

    # Mateix windowing que la inferencia, per veure el que veu el model
    ct_uint8 = preproces(ct, level=60, width=150)

    n = len(prompts)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # aplanem per indexar facilment

    for k, (idx, bbox) in enumerate(prompts):
        ax = axes[k]
        ax.imshow(ct_uint8[idx], cmap="bone")

        # Contorn del fetge GT en aquella llesca, en vermell tenue, com a
        # referencia per veure si la box l'envolta correctament.
        m2d = (mask[idx] > 0).astype(np.uint8)
        if m2d.sum() > 0:
            ax.contour(m2d, levels=[0.5], colors="red", linewidths=0.8)

        x_min, y_min, x_max, y_max = bbox
        # Rectangle: (x_min, y_min) cantonada inferior-esquerra en coords imatge,
        # amplada = x_max - x_min, alcada = y_max - y_min.
        rect = Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=1.5, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)
        ax.set_title(f"slice {idx}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # Apaguem els eixos sobrants de la graella
    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"{name} - fase {phase} - boxes de prompt (verd) vs fetge GT (vermell)",
                 fontsize=11)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{name}_{phase}_boxes.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{phase}] desat: {out_path.name}  ({n} boxes)")


def main():
    my_dataset = dataset.load_dataset(DATASET_PATH, TAULA_COMPLEMENTARIA_PATH, DESCARTS_MANUALS)


    for sample in my_dataset:
        if NOMES_PACIENTS and sample["name"] not in NOMES_PACIENTS:
            continue

        name = sample["name"]
        seg_path = sample["segmentation_path"]
        print(f"Pacient: {name}")

        if seg_path is None:
            print("  sense segmentation_path; salto")
            print()
            continue

        # Carreguem la GT (DICOM-SEG) UN sol cop per pacient. Es la mateixa
        # segmentacio que es carrega amb les dades i que genera els prompts.
        # load_segmentation retorna (seg_object, mask_multilabel 1=Liver/2=Mass).
        seg, mask_gt = utils.load_segmentation(seg_path)

        for ct_info in sample["list_ct_paths"]:
            phase = ct_info["phase"]

            # CT d'aquesta fase i GT alineada a la seva graella Z.
            # align_mask_to_ct fa servir les Z dels dcm_slices del CT.
            dcm_slices, ct = utils.load_ct(ct_info)
            mask = utils.align_mask_to_ct(mask_gt, seg, dcm_slices)

            try:
                prompts = extract_prompt_boxes(mask, every_n=EVERY_N)
            except ValueError as e:
                print(f"  [{phase}] {e}")
                continue

            dibuixa_boxes_pacient(name, ct, mask, prompts, phase, OUTPUT_PATH)


        print()

    print(f"Fet. Revisa els PNG a: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()