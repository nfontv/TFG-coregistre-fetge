import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
from pathlib import Path

base = Path(r"C:\Users\Noemí Font\Desktop\CoregistreRigidFetge-main\results_coregistre\taula_resultats_2026.07.14_10.32.00\history")

for cfg in ["soft_dice_lbfgsb", "soft_dice_powell"]:
    h = np.load(base / f"HCC_025_arterial_{cfg}.npy")
    loss, dsc = h[:, 0], h[:, 1]
    print(f"--- {cfg} ---")
    print(f"  loss:  inicial={loss[0]:.6f}  final={loss[-1]:.6f}  minima={loss.min():.6f}")
    print(f"  ha baixat la loss? {loss.min() < loss[0]}   (millora: {loss[0]-loss.min():.6e})")
    print(f"  DSC dur: inicial={dsc[0]:.6f}  maxim={dsc.max():.6f}  valors unics={len(np.unique(dsc))}")
    print()