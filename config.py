"""
Rutes del projecte, centralitzades en un sol lloc.

Els fitxers que formen part del repositori (taula complementaria de TCIA,
resultats dels experiments, figures) es resolen sols a partir de la ubicacio
d'aquest fitxer: no cal tocar res.

Per executar el pipeline complet cal indicar on sonles dades:

    export TFG_DATA="/Volumes/elmeudisc/tgf noemi"     # macOS / Linux
    set TFG_DATA=D:\\tgf noemi                          # Windows

Si no es defineix la variable, s'assumeix una carpeta "tgf noemi" al costat
del repositori. Alternativament, es pot editar DATA_ROOT aqui sota.

Estructura esperada dins DATA_ROOT:
    <DATA_ROOT>/data tgf noemi/<MANIFEST_DIR>/HCC-TACE-Seg/   # series DICOM (TCIA)
    <DATA_ROOT>/segmentations_MedSam2/                        # mascares .npy
    <DATA_ROOT>/filtre_segmentacions/metriques_filtre.csv     # filtre de qualitat
"""

import os
from pathlib import Path

# Arrel del repositori: la carpeta on hi ha aquest fitxer.
ROOT = Path(__file__).resolve().parent


# Fitxers inclosos al repositori 

# Taula complementaria del dataset HCC-TACE-Seg (Moawad et al., 2023).
TAULA_COMPLEMENTARIA_PATH = ROOT / "docs" / "41597_2023_1928_MOESM1_ESM.xlsx"

# Resultats dels experiments de co-registre.
RESULTATS_PATH = ROOT / "resultats"
TAULA_EXPERIMENTS_PATH = RESULTATS_PATH / "tots_els_experiments.csv"

FIGURES_PATH = ROOT / "figures"


# Dades externes (no incloses al repositori)

DATA_ROOT = Path(os.environ.get("TFG_DATA", ROOT.parent / "tgf noemi"))

# Nom de la carpeta "manifest-..." que genera la descarrega de TCIA. Canvia a
# cada descarrega, per aixo es configurable.
MANIFEST_DIR = os.environ.get("TFG_MANIFEST", "manifest-1783073007564")

DATASET_PATH = DATA_ROOT / "data tgf noemi" / MANIFEST_DIR / "HCC-TACE-Seg"
SEGMENTATIONS_PATH = DATA_ROOT / "segmentations_MedSam2"
FILTRE_SEGMENTACIONS_CSV_PATH = DATA_ROOT / "filtre_segmentacions" / "metriques_filtre.csv"


# Sortides de noves execucions

RESULTS_BASE = ROOT / "results_coregistre"
