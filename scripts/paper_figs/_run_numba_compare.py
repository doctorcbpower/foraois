import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "scripts/paper_figs")
from _fortran_compare import build_pch

M0 = 1.0e13
M_res = 1.0e9
z0 = 0.0
z_max = 5.0
checkpoints = [0.5 * i for i in range(1, 11)]
target_nupper = 0.1

N_TREES = int(sys.argv[1]) if len(sys.argv) > 1 else 300

pch = build_pch()

np.random.seed(0)
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    results = pch.grow_full_population_numba_adaptive(
        M0, z0, z_max, M_res, checkpoints, n_trees=N_TREES, target_nupper=target_nupper, dz_max=0.5, min_dz=1e-4,
        max_stack=20_000, max_out=4_000,
    )
elapsed = time.time() - t0

counts = {zc: [] for zc in checkpoints}
for pops in results:
    for zc in checkpoints:
        counts[zc].append(len(pops[zc]))

print(f"N_TREES={N_TREES}  elapsed={elapsed:.2f}s  ({elapsed/N_TREES*1000:.3f}ms/tree)")
print("checkpoint_z  mean_count  std  sem")
for zc in checkpoints:
    arr = np.array(counts[zc])
    print(f"{zc:6.2f}  {arr.mean():10.2f}  {arr.std():8.2f}  {arr.std()/np.sqrt(len(arr)):6.3f}")
