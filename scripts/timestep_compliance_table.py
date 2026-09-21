#!/usr/bin/env python3
"""
Timestep compliance of the single-split-per-step tree builders, on the main-progenitor mass at z = 1.

For each (M_res/M0, dz) the script reports

  * max N_upper: the largest mean-field PCH08 N_upper along the trajectory (PCH08 requires <~ 0.1),
  * PCH08 and Zhang-Hui median M(z=1)/M0 (and the PCH08 16-84 per cent range) over N trees.

Compliance (N_upper small), numerical convergence (statistic stable as dz decreases) and physical
interpretation (no N-body reference here) are separate questions; the table addresses the first two.
Replaces scripts/diagnose_pch08_high_z.py. See docs/PCH08_HIGH_Z_DIAGNOSTIC.md.

Usage
-----
    python scripts/timestep_compliance_table.py [--n 300] [--seed 3]
"""
import argparse
from pathlib import Path

import numpy as np

from foraois import CosmoData, PCHMergerTree, ZhangHuiMergerTree
from foraois.diagnostics import expected_splits_per_step
from foraois.utils import io

CONFIG = Path(__file__).resolve().parents[1] / "config" / "planck2018_camb.yml"
CASES = [(1e-2, 0.05), (1e-2, 0.0005), (1e-4, 0.05), (1e-4, 0.01), (1e-4, 0.002), (1e-4, 0.0005)]


def median_ratio(gen, seed, n, M0, z0, z_target, M_res, dz):
    np.random.seed(seed)
    if hasattr(gen, "rng"):
        gen.rng = np.random.default_rng(seed)
    out = gen.build_forest_numba(M0_array=np.full(n, M0), z0=z0, z_max=z_target, M_res=M_res, dz=dz)
    mh, zs = out[0], out[1]
    z = np.asarray(zs)[1:]  # mass_history excludes the z0 column
    r = mh[:, int(np.argmin(abs(z - z_target)))] / M0
    return r[r > 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--M0", type=float, default=1e12)
    a = ap.parse_args()
    rp = io.get_params(a.config)
    cd = CosmoData(rp, redshift=[0.0])
    pch = PCHMergerTree(cd, rp)
    zh = ZhangHuiMergerTree(cd, rp, model="cdm")
    print(f"M0={a.M0:.0e} Msun/h, N={a.n} trees, seed={a.seed}; statistic: main-progenitor M(z=1)/M0 (median)")
    print(f"{'Mres/M0':>8} {'dz':>7} {'max Nupper':>10} {'PCH08 median':>13} {'[16,84]':>14} {'ZH median':>10}")
    for frac, dz in CASES:
        Mres = a.M0 * frac
        _, nup, _ = expected_splits_per_step(pch, a.M0, 0.0, 1.5, Mres, dz=dz)
        rp_ = median_ratio(pch, a.seed, a.n, a.M0, 0.0, 1.0, Mres, dz)
        rz_ = median_ratio(zh, a.seed, a.n, a.M0, 0.0, 1.0, Mres, dz)
        q = np.percentile(rp_, [16, 84])
        print(f"{frac:8.0e} {dz:7g} {np.nanmax(nup):10.2f} {np.median(rp_):13.3f} [{q[0]:.2f}, {q[1]:.2f}] {np.median(rz_):10.3f}", flush=True)


if __name__ == "__main__":
    main()
