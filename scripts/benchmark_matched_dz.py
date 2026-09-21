#!/usr/bin/env python3
"""
Matched-timestep cost comparison of the PCH08 and Zhang-Hui (closed-form, flat-barrier) forest builders.

Both implementations (numpy and numba) of both algorithms are timed on the same halo set and step size on
one machine: M0 = 1e12 Msun/h, M_res = 1e9 Msun/h, z = 0 -> 5, real CAMB power spectrum
(config/planck2018_camb.yml), N halos. For each step size the script also reports the largest mean-field
PCH08 N_upper along the trajectory and the largest EPS expected number of splits per step, so the table shows
which steps satisfy the single-split-per-step requirement (<~ 0.1).

This is a cost comparison at a fixed step size. It is not a statement that either algorithm is faster in
general: the step size a given accuracy needs is set by the timestep criterion, and it differs.
The general numerical (Volterra) first-crossing solver is not benchmarked here.

Usage
-----
    python scripts/benchmark_matched_dz.py [--n 10000] [--repeats 3] [--csv benchmark_matched_dz.csv]
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from foraois import CosmoData, PCHMergerTree, ZhangHuiMergerTree
from foraois.diagnostics import expected_eps_splits_per_step, expected_splits_per_step
from foraois.utils import io

CONFIG = Path(__file__).resolve().parents[1] / "config" / "planck2018_camb.yml"
M0, Z0, Z_MAX, M_RES = 1.0e12, 0.0, 5.0, 1.0e9
DZ_VALUES = [0.2, 0.01, 0.003, 0.001]


def timed(fn, repeats):
    fn()  # warm-up (numba compilation, caches)
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.mean(t)), float(np.std(t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dz", type=float, nargs="*", default=DZ_VALUES)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    rp = io.get_params(a.config)
    cd = CosmoData(rp, redshift=[0.0])
    pch = PCHMergerTree(cd, rp)
    zh = ZhangHuiMergerTree(cd, rp, model="cdm", rng=np.random.default_rng(1))
    M0s = np.full(a.n, M0)

    rows = []
    print(f"N={a.n}, M0={M0:.0e}, M_res={M_RES:.0e}, z {Z0} -> {Z_MAX}; times in s (mean of {a.repeats})")
    print(f"{'dz':>7} {'max Nupper':>10} {'max E_EPS':>10} | {'PCH numpy':>10} {'PCH numba':>10} {'ZH numpy':>10} {'ZH numba':>10} | ZH/PCH numpy, numba")
    for dz in a.dz:
        _, nup, _ = expected_splits_per_step(pch, M0, Z0, Z_MAX, M_RES, dz=dz)
        zs = np.arange(Z0, Z_MAX + 0.5 * dz, dz)
        e_eps = max(expected_eps_splits_per_step(cd, M0, zs[j], zs[j + 1], M_RES) for j in range(0, len(zs) - 1, max(1, len(zs) // 60)))
        t = {}
        for name, gen in (("pch", pch), ("zh", zh)):
            for be in ("numpy", "numba"):
                fn = getattr(gen, f"build_forest_{be}")
                t[(name, be)] = timed(lambda fn=fn: fn(M0_array=M0s, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=dz), a.repeats)
        r_np = t[("zh", "numpy")][0] / t[("pch", "numpy")][0]
        r_nb = t[("zh", "numba")][0] / t[("pch", "numba")][0]
        print(f"{dz:7g} {np.nanmax(nup):10.3g} {e_eps:10.3g} | "
              f"{t[('pch','numpy')][0]:10.2f} {t[('pch','numba')][0]:10.2f} {t[('zh','numpy')][0]:10.2f} {t[('zh','numba')][0]:10.2f} | {r_np:.2f}, {r_nb:.2f}", flush=True)
        rows.append({"dz": dz, "n": a.n, "max_nupper": float(np.nanmax(nup)), "max_eps_splits": e_eps,
                     **{f"{n}_{b}_s": t[(n, b)][0] for (n, b) in t}, **{f"{n}_{b}_std": t[(n, b)][1] for (n, b) in t},
                     "zh_over_pch_numpy": r_np, "zh_over_pch_numba": r_nb})
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
