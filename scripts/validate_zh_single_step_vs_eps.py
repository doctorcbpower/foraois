#!/usr/bin/env python3
"""
Single-step check of the Zhang-Hui binary-per-step sampler against the analytic EPS progenitor distribution.

For one halo mass and one step z0 -> z1 the script draws the ZH split many times (vectorised copy of the
closed-form flat-barrier draw in ``foraois.zhang_hui_trees.draw_progenitor_mass_zh_flat``) and compares the
sampled expected number of smaller fragments per step, in bins of ln(M2/M0), with the analytic EPS number
density (1/q) f(dS, d_omega) |dS/dlnM2|. It also prints the total ZH split probability, the analytic EPS
expected number of splits (``foraois.diagnostics.expected_eps_splits_per_step``) and the PCH08 target rate
(``foraois.diagnostics.true_split_probability``).

This tests one configuration only. It does not show that the multi-step tree is exact.

Usage
-----
    python scripts/validate_zh_single_step_vs_eps.py [--M0 1e12] [--mres-frac 1e-2] [--dz 0.02] [--n-draws 2000000]
"""
import argparse
from pathlib import Path

import numpy as np

from foraois import CosmoData, PCHMergerTree, ZhangHuiMergerTree
from foraois.collapse import delta_c
from foraois.diagnostics import expected_eps_splits_per_step, true_split_probability
from foraois.utils import io
from foraois.zhang_hui_trees import _S_at_mass, _flat_barrier_cdf, _flat_barrier_sample, _mass_at_S

CONFIG = Path(__file__).resolve().parents[1] / "config" / "planck2018_camb.yml"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--M0", type=float, default=1e12)
    ap.add_argument("--mres-frac", type=float, default=1e-2)
    ap.add_argument("--dz", type=float, default=0.02)
    ap.add_argument("--z0", type=float, default=0.0)
    ap.add_argument("--n-draws", type=int, default=2_000_000)
    ap.add_argument("--n-bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rp = io.get_params(a.config)
    cd = CosmoData(rp, redshift=[0.0])
    zh = ZhangHuiMergerTree(cd, rp, model="cdm")
    pch = PCHMergerTree(cd, rp)
    M0, Mres, z0, z1 = a.M0, a.M0 * a.mres_frac, a.z0, a.z0 + a.dz
    d0, d1 = float(delta_c(Mres, z0, "cdm", cd)), float(delta_c(Mres, z1, "cdm", cd))
    dw = d1 - d0
    s0 = float(cd.sigma_at_logmass(np.log10(M0))) ** 2
    S_res = float(cd.sigma_at_logmass(np.log10(Mres))) ** 2 - s0

    # ZH closed-form step, following draw_progenitor_mass_zh_flat
    p_res = float(_flat_barrier_cdf(S_res, dw))
    M_cont = M0 * p_res
    S_lower = float(np.clip(_S_at_mass(M_cont - Mres, s0, cd), 0.0, S_res))
    p_lower = float(_flat_barrier_cdf(S_lower, dw)) if S_lower > 0 else 1.0
    p_split = max(p_res - p_lower, 0.0)
    rng = np.random.default_rng(a.seed)
    split = rng.random(a.n_draws) < p_split
    v = rng.uniform(p_lower, p_res, split.sum())
    M2 = _mass_at_S(_flat_barrier_sample(v, dw), s0, cd)
    q2 = np.minimum(M2, M_cont - M2) / M0  # smaller fragment, as in the EPS binary-split count

    edges = np.linspace(np.log(a.mres_frac), np.log(0.5), a.n_bins + 1)
    h, _ = np.histogram(np.log(q2), edges)
    sampled = h / a.n_draws

    def eps_number(lnq):
        q = np.exp(lnq)
        lg = np.log10(M0 * q)
        sg = cd.sigma_at_logmass(lg)
        dS = sg**2 - s0
        f = dw * np.exp(-0.5 * dw**2 / dS) / (np.sqrt(2 * np.pi) * dS**1.5)
        return f * 2 * sg**2 * cd.dlogsigma_at_logmass(lg) / q

    eps = np.array([np.trapezoid(eps_number(x), x) for x in (np.linspace(lo, hi, 400) for lo, hi in zip(edges[:-1], edges[1:]))])
    print(f"M0={M0:.2e}  M_res/M0={a.mres_frac:g}  z0={z0}  dz={a.dz}  draws={a.n_draws}")
    print(f"  M_continuing/M0 = {p_res:.3f}")
    print(f"  {'bin lo':>9} {'bin hi':>9} {'ZH sampled':>11} {'EPS':>9} {'ratio':>6}")
    for lo, hi, s, e in zip(edges[:-1], edges[1:], sampled, eps):
        print(f"  {np.exp(lo):9.4f} {np.exp(hi):9.4f} {s:11.5f} {e:9.5f} {s / e:6.2f}")
    print(f"  ZH split probability p_split           = {p_split:.4f}")
    print(f"  EPS expected splits per step (q<=1/2)  = {expected_eps_splits_per_step(cd, M0, z0, z1, Mres):.4f}")
    print(f"  PCH08 target rate (true split prob.)   = {true_split_probability(pch, M0, Mres, d0, dw):.4f}  "
          f"(ratio to EPS {true_split_probability(pch, M0, Mres, d0, dw) / expected_eps_splits_per_step(cd, M0, z0, z1, Mres):.2f})")


if __name__ == "__main__":
    main()
