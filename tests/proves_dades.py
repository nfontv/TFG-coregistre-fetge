import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import pandas as pd
df = pd.read_csv(r"C:\Users\Noemí Font\Desktop\CoregistreRigidFetge-main\results_coregistre\tots_els_experiments.csv", encoding="latin-1")
print("Files totals:", len(df))
print("\nFiles per config:")
print(df.config.value_counts().sort_index())