import sys
import time

import numpy as np

sys.path.insert(0, "scripts/paper_figs")
from _fortran_compare import build_pch
from _treegrowth import grow_full_population_pch08_adaptive

M0 = 1.0e13
M_res = 1.0e9
z0 = 0.0
z_max = 5.0
checkpoints = [0.5 * i for i in range(1, 11)]  # 0.5 .. 5.0, matches FORTRAN levels 2..11
target_nupper = 0.1  # matches FORTRAN eps2=0.1

N_TREES = int(sys.argv[1]) if len(sys.argv) > 1 else 20

pch = build_pch()

t0 = time.time()
counts = {zc: [] for zc in checkpoints}
for seed in range(N_TREES):
    np.random.seed(seed)
    result = grow_full_population_pch08_adaptive(
        pch, M0, z0, z_max, M_res, checkpoints, target_nupper=target_nupper, dz_max=0.5, min_dz=1e-4
    )
    for zc in checkpoints:
        counts[zc].append(len(result[zc]))
elapsed = time.time() - t0

print(f"N_TREES={N_TREES}  elapsed={elapsed:.2f}s  ({elapsed/N_TREES:.3f}s/tree)")
print("checkpoint_z  mean_count  std  sem")
for zc in checkpoints:
    arr = np.array(counts[zc])
    print(f"{zc:6.2f}  {arr.mean():10.2f}  {arr.std():8.2f}  {arr.std()/np.sqrt(len(arr)):6.3f}")
