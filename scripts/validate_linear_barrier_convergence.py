#!/usr/bin/env python3
"""
Development diagnostic: resolution dependence of the general first-crossing path for the linear barrier
delta_c(M, z) = delta_sc(z) + beta*sigma^2(M).

Builds the barrier from the config (config/planck2018_linear_barrier.yml by default), takes one step z0 -> z1 with
``first_crossing_step`` (the path ``ZhangHuiMergerTree.build_tree`` uses), and compares the numerical first-crossing
density with the analytic solution for B(S) = a + beta*S, a = delta_sc(z1) - delta_sc(z0)
(``first_crossing.linear_barrier_first_crossing``). For each N_grid it prints dS/a^2 (grid spacing over the barrier
scale, the quantity the accuracy is set by), the peak-normalised error of f(S) over 0.05 < S <= S_res, and the error
of the crossing probability at S_res (what ``build_tree`` uses as p_res).

The tests (tests/test_zhang_hui_validation.py) run the same comparison on a synthetic power spectrum. This script uses
the real CAMB spectrum.

Usage
-----
    python scripts/validate_linear_barrier_convergence.py [--config config/planck2018_linear_barrier.yml]
"""
import argparse
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

from foraois import CosmoData, ZhangHuiMergerTree
from foraois.collapse import barrier_settings
from foraois.first_crossing import linear_barrier_first_crossing
from foraois.utils import io
from foraois.zhang_hui_trees import first_crossing_step


def cdf_exact(S, a, b):
    """Exact first-crossing CDF for B(S) = a + b*S."""
    return norm.cdf((-a - b * S) / np.sqrt(S)) + np.exp(-2.0 * a * b) * norm.cdf((-a + b * S) / np.sqrt(S))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/planck2018_linear_barrier.yml")
    ap.add_argument("--M0", type=float, default=1e12)
    ap.add_argument("--M-res", type=float, default=1e11)
    ap.add_argument("--z1", type=float, nargs="*", default=[0.5, 2.0])
    ap.add_argument("--n-grid", type=int, nargs="*", default=[100, 200, 400, 800, 1600])
    ap.add_argument("--s-max-factor", type=float, default=2.0)
    a = ap.parse_args()
    warnings.filterwarnings("ignore")

    rp = io.get_params(a.config)
    cd = CosmoData(rp, redshift=[0.0])
    ZhangHuiMergerTree(cd, rp)  # builds the sigma(M) table
    name, beta = barrier_settings(cd.run_params)
    print(f"config={a.config}  barrier={name}  beta={beta}  M0={a.M0:.0e}  M_res={a.M_res:.0e}")
    print(f"S_max = {a.s_max_factor} S_res")
    for z1 in a.z1:
        aa = float(cd.delta_col_at_z(z1) - cd.delta_col_at_z(0.0))
        print(f"\nz0 -> z1 = {z1}:  a^2 = (delta_sc(z1) - delta_sc(z0))^2 = {aa**2:.4f}")
        print(f"  {'N_grid':>6} {'dS/a^2':>8} {'peak-norm err':>14} {'CDF err @S_res':>15} {'p_res':>9}")
        for N in a.n_grid:
            st = first_crossing_step(a.M0, 0.0, z1, a.M_res, cd, N_grid=N, S_max_factor=a.s_max_factor)
            S, f, Sres = st["S_grid"], st["f"], st["S_res"]
            fe = linear_barrier_first_crossing(S, aa, beta)
            m = (S > 0.05) & (S <= Sres)
            peak = np.abs(f[m] - fe[m]).max() / fe[m].max()
            cdf_numeric = float(np.interp(Sres, S, cumulative_trapezoid(f, S, initial=0.0)))
            cdf = abs(cdf_numeric - float(cdf_exact(Sres, aa, beta)))
            print(f"  {N:6d} {S[1] / aa**2:8.3f} {peak:14.2e} {cdf:15.2e} {st['p_res']:9.5f}")


if __name__ == "__main__":
    main()
