import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pandas as pd

from config import TAULA_EXPERIMENTS_PATH

df = pd.read_csv(TAULA_EXPERIMENTS_PATH, encoding="latin-1")
print("Files totals:", len(df))
print("\nFiles per config:")
print(df.config.value_counts().sort_index())