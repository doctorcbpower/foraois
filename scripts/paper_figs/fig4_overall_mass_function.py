#!/usr/bin/env python3
"""
Reproduces the structure of Parkinson, Cole & Helly (2008, "PCH08")
Figure 4: a self-consistency check between a merger-tree algorithm's
progenitor statistics and the Sheth-Tormen (2002) halo mass function.

The idea (PCH08 Sec. 3.4): if you draw a population of z=0 ("root") haloes
with number density following dn/dlogM (Sheth-Tormen), and then follow
every branch of every one of their merger trees back to an earlier
redshift z1, the resulting ensemble of progenitor masses -- reweighted by
how many real haloes each root mass represents -- should ALSO reproduce
the Sheth-Tormen mass function evaluated at (M1, z1), *if* the tree
algorithm's branching statistics are consistent with excursion-set theory.
PCH08 show this holds much better for their new algorithm ("New Trees",
what foraois calls PCH08) than for the original GALFORM algorithm at low
progenitor masses.

This is the heaviest of the four fig1-4 analogue scripts: it needs the
same full-branch-population growth as fig1/fig2 (_treegrowth.py), but
repeated over a whole GRID of root masses spanning several decades (not
just the 3 M2 values fig1/fig2 use), since building an ensemble that
represents the full halo population requires sampling the root-mass axis
too. Expect this to be slow, especially for the Zhang-Hui backend at the
high-M2 end (deepest dynamic range down to M_res => most branching
generations => most Volterra solves per tree). Start with the small
defaults below purely to confirm the script runs; scale up --n-trees and
--n-m2-bins (and refine --dz) for a publication-quality run.

As with fig1-3, there is no Millennium N-body reference available, and
Nadler et al. (2023)'s constrained sampler is out of scope for the same
reason given in fig1_conditional_mass_function.py's docstring: it is built
to efficiently sample one specified rare branch, not to characterise the
ordinary ensemble statistics this figure checks.

Method
------
1. Build a log-spaced grid of root masses M2 spanning [--m2-min, --m2-max].
2. For each grid bin, compute the comoving number density it represents,
   N_bin = integral of dn/dlogM2 (Sheth-Tormen, at z=0) over the bin, using
   the bin's geometric-mean mass as the single representative root mass
   grown into --n-trees realizations (a standard single-mass-per-bin
   approximation -- fine as long as bins are reasonably narrow).
3. Grow each realization's full branch population (as in fig1/fig2) to
   each z1 checkpoint, and reweight every realization's contribution by
   N_bin / n_trees, so each root-mass bin contributes its correct share of
   the total number density.
4. Histogram the reweighted progenitor masses into an ABSOLUTE mass grid
   (not M1/M2 as in fig1/fig2 -- here the whole point is comparing
   progenitors of different roots on the same absolute-mass axis) to get
   dn/dlog10(M1) at z1, and compare directly to the Sheth-Tormen curve
   evaluated at (M1, z1) with the same cosmo_data instance.

Usage
-----
    python scripts/paper_figs/fig4_overall_mass_function.py \
        [--m2-min 3e11] [--m2-max 3e15] [--n-m2-bins 8] \
        [--n-trees 10] [--dz 0.05] [--n-grid 40] \
        [--z1-values 0.5,1,2,4] [--output fig4_mass_function.png]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _treegrowth import (  # noqa: E402
    grow_full_population_pch08,
    grow_full_population_pch08_batch,
    grow_full_population_zh_batch,
    sheth_tormen_dn_dlogm,
)

from foraois.cosmo_utils import CosmoData  # noqa: E402
from foraois.pch_trees import PCHMergerTree  # noqa: E402
from foraois.utils import io as foraois_io  # noqa: E402

Z0 = 0.0
M_RES_FRAC = 1.0e-4  # M_res = M_RES_FRAC * (root mass in that bin), as in fig1/fig2


def build_m2_bins(m2_min, m2_max, n_bins, cosmo_data):
    """Log-spaced root-mass bin edges/representative masses/number densities."""
    edges = np.geomspace(m2_min, m2_max, n_bins + 1)
    reps = np.sqrt(edges[:-1] * edges[1:])  # geometric-mean representative mass
    n_bin = np.array(
        [
            quad(lambda logM: sheth_tormen_dn_dlogm(10**logM, Z0, cosmo_data), np.log10(lo), np.log10(hi))[0]
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )
    return reps, n_bin


def run_backend(grow_fn, reps, n_bin, checkpoints, dz, n_trees, m_res_frac, **grow_kwargs):
    """Grow n_trees realizations per root-mass bin; return, per checkpoint,
    a list of (weight, mass) pairs pooled across every bin -- weight is
    N_bin / n_trees, the number density each single realization represents."""
    pooled = {z1: {"masses": [], "weights": []} for z1 in checkpoints}
    z_max = max(checkpoints)
    for M2, Nb in zip(reps, n_bin):
        M_res = M2 * m_res_frac
        w = Nb / n_trees
        for _ in range(n_trees):
            pops = grow_fn(M0=M2, z0=Z0, z_max=z_max, M_res=M_res, dz=dz, checkpoints=checkpoints, **grow_kwargs)
            for z1 in checkpoints:
                masses = pops[z1]
                pooled[z1]["masses"].extend(masses)
                pooled[z1]["weights"].extend([w] * len(masses))
    return pooled


def run_backend_pch08_batch(pch, reps, n_bin, checkpoints, dz, n_trees, m_res_frac, target_nupper=0.5):
    """PCH08-specific equivalent of run_backend: grows all n_trees
    realizations for a given root-mass bin in one batched call
    (grow_full_population_pch08_batch, adaptive per-branch stepping with dz
    as the step-size ceiling) instead of run_backend's generic per-tree
    Python loop -- uses the ~400-600x faster numba kernel when available
    (see grow_full_population_pch08_batch's own docstring). This is the
    heaviest of the four fig1-4 scripts (a whole grid of root masses, not
    just 3), so it benefits the most from batching. target_nupper=None
    falls back to run_backend's old fixed-dz grow_full_population_pch08
    loop."""
    if target_nupper is None:
        return run_backend(
            lambda M0, z0, z_max, M_res, dz, checkpoints: grow_full_population_pch08(pch, M0, z0, z_max, M_res, dz, checkpoints),
            reps, n_bin, checkpoints, dz, n_trees, m_res_frac,
        )

    pooled = {z1: {"masses": [], "weights": []} for z1 in checkpoints}
    z_max = max(checkpoints)
    for M2, Nb in zip(reps, n_bin):
        M_res = M2 * m_res_frac
        w = Nb / n_trees
        pops_list = grow_full_population_pch08_batch(pch, M2, Z0, z_max, M_res, checkpoints, n_trees, target_nupper=target_nupper, dz_max=dz)
        for pops in pops_list:
            for z1 in checkpoints:
                masses = pops[z1]
                pooled[z1]["masses"].extend(masses)
                pooled[z1]["weights"].extend([w] * len(masses))
    return pooled


def run_backend_zh_batch(cosmo_data, reps, n_bin, checkpoints, dz, n_trees, m_res_frac, model="cdm", rng=None, n_grid=40, s_max_factor=8.0):
    """Zhang-Hui equivalent of run_backend_pch08_batch: grows all n_trees
    realizations for a given root-mass bin in one batched call
    (grow_full_population_zh_batch) instead of run_backend's generic
    per-tree Python loop -- uses the numba closed-form kernel when
    available (ZhangHuiMergerTree.grow_full_population_numba, no
    solve_first_crossing at all -- see that method's own docstring)."""
    pooled = {z1: {"masses": [], "weights": []} for z1 in checkpoints}
    z_max = max(checkpoints)
    for M2, Nb in zip(reps, n_bin):
        M_res = M2 * m_res_frac
        w = Nb / n_trees
        pops_list = grow_full_population_zh_batch(cosmo_data, M2, Z0, z_max, M_res, dz, checkpoints, n_trees, model=model, rng=rng, N_grid=n_grid, S_max_factor=s_max_factor)
        for pops in pops_list:
            for z1 in checkpoints:
                masses = pops[z1]
                pooled[z1]["masses"].extend(masses)
                pooled[z1]["weights"].extend([w] * len(masses))
    return pooled


def weighted_dndlogm(pooled_z1, log_edges):
    masses = np.asarray(pooled_z1["masses"])
    weights = np.asarray(pooled_z1["weights"])
    if masses.size == 0:
        centers = 0.5 * (log_edges[:-1] + log_edges[1:])
        return centers, np.full(len(centers), np.nan)
    counts, _ = np.histogram(np.log10(masses), bins=log_edges, weights=weights)
    dlogm = log_edges[1] - log_edges[0]
    dndlogm = counts / dlogm
    centers = 0.5 * (log_edges[:-1] + log_edges[1:])
    with np.errstate(divide="ignore"):
        log_dndlogm = np.log10(dndlogm)
    log_dndlogm[counts == 0] = np.nan
    return centers, log_dndlogm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--m2-min", type=float, default=3.0e11)
    parser.add_argument("--m2-max", type=float, default=3.0e15)
    parser.add_argument("--n-m2-bins", type=int, default=8)
    parser.add_argument("--n-trees", type=int, default=10)
    parser.add_argument("--dz", type=float, default=0.05)
    parser.add_argument("--n-grid", type=int, default=40)
    parser.add_argument("--s-max-factor", type=float, default=8.0)
    parser.add_argument("--z1-values", default="0.5,1,2,4")
    parser.add_argument("--n-mass-bins", type=int, default=24, help="log10(M1) histogram bins for the output curves")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--target-nupper",
        type=float,
        default=0.5,
        help="PCH08's adaptive-step target for Nupper (per-step split probability); --dz is used as the "
        "per-branch maximum step size. See fig1_conditional_mass_function.py's --target-nupper help for "
        "why fixed-dz stepping is not Nupper-compliant at these M_res/M2 ratios. Pass a negative value to "
        "fall back to the old fixed-dz grow_full_population_pch08 behaviour.",
    )
    parser.add_argument("--output", default="fig4_mass_function.png")
    args = parser.parse_args()
    checkpoints = [float(z) for z in args.z1_values.split(",")]
    target_nupper = None if args.target_nupper < 0 else args.target_nupper

    run_params = foraois_io.get_params(args.config)
    cosmo_data = CosmoData(run_params, redshift=[Z0])
    pch = PCHMergerTree(cosmo_data, run_params)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    reps, n_bin = build_m2_bins(args.m2_min, args.m2_max, args.n_m2_bins, cosmo_data)
    print(f"Root-mass bins (Msun/h): {reps}")
    print(f"Number density per bin: {n_bin}")

    print("Growing PCH08 ensemble ...", flush=True)
    pooled_pch = run_backend_pch08_batch(
        pch,
        reps,
        n_bin,
        checkpoints,
        args.dz,
        args.n_trees,
        M_RES_FRAC,
        target_nupper=target_nupper,
    )

    print("Growing Zhang-Hui ensemble ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pooled_zh = run_backend_zh_batch(
            cosmo_data,
            reps,
            n_bin,
            checkpoints,
            args.dz,
            args.n_trees,
            M_RES_FRAC,
            model="cdm",
            rng=rng,
            n_grid=args.n_grid,
            s_max_factor=args.s_max_factor,
        )

    import matplotlib.pyplot as plt

    log_edges = np.linspace(np.log10(args.m2_min * M_RES_FRAC), np.log10(args.m2_max), args.n_mass_bins + 1)

    fig, axes = plt.subplots(1, len(checkpoints), figsize=(4.3 * len(checkpoints), 4.2), sharey=True)
    if len(checkpoints) == 1:
        axes = [axes]

    for ax, z1 in zip(axes, checkpoints):
        c_pch, log_pch = weighted_dndlogm(pooled_pch[z1], log_edges)
        c_zh, log_zh = weighted_dndlogm(pooled_zh[z1], log_edges)
        log_st = np.log10(sheth_tormen_dn_dlogm(10**c_pch, z1, cosmo_data))

        ax.plot(c_pch, log_st, "-", color="0.3", lw=1.6, label="Sheth-Tormen (direct)")
        ax.step(c_pch, log_pch, where="mid", color="#1f77b4", lw=1.6, label="PCH08 (tree ensemble)")
        ax.step(c_zh, log_zh, where="mid", color="#9467bd", lw=1.6, label="Zhang-Hui (tree ensemble)")

        ax.set_title(f"$z_1={z1}$", fontsize=10)
        ax.set_xlabel(r"$\log_{10}(M_1\,/\,M_\odot h^{-1})$")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(r"$\log_{10}\,dn/d\log_{10}M_1$")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "PCH08 Fig. 4 analogue: progenitor ensemble vs Sheth-Tormen mass function\n"
        "(self-consistency check; no N-body reference available)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
