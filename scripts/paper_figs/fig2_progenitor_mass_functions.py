#!/usr/bin/env python3
"""
Reproduces the structure of Parkinson, Cole & Helly (2008, "PCH08")
Figure 2: the mass distributions of the *first* and *second* most-massive
progenitors, f_1st and f_2nd -- the contributions to the conditional mass
function (Fig. 1's quantity) provided specifically by the largest and
second-largest branch of the full tree at each z1 -- on the same 4 (z1) x
3 (M2) grid as Fig 1.

This is a genuinely different test from Fig 1: it does not just ask "how
much mass ends up at each M1", it asks "how often is *the* largest
progenitor a given mass" -- something the EPS/PCH08/Zhang-Hui rate alone
does not directly predict, since it depends on how the whole tree's
branching history combines (PCH08's own Section 3.2 makes the same point).
See fig1_conditional_mass_function.py's docstring for what is and is not
reproduced here relative to PCH08's own Fig 2 (no Millennium N-body
reference; same shared-dz-grid, full-branch-population approach; Nadler
et al. 2023's constrained sampler is out of scope for the same reason
given there).

Usage
-----
    python scripts/paper_figs/fig2_progenitor_mass_functions.py \
        [--n-trees 40] [--dz 0.05] [--n-grid 40] [--output fig2_progenitors.png]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _treegrowth import grow_full_population_pch08, grow_full_population_pch08_batch, grow_full_population_zh_batch  # noqa: E402

from foraois.cosmo_utils import CosmoData  # noqa: E402
from foraois.pch_trees import PCHMergerTree  # noqa: E402
from foraois.utils import io as foraois_io  # noqa: E402

M2_VALUES = [1.0e12, 3.16e13, 1.0e15]
Z1_VALUES = [4.0, 2.0, 1.0, 0.5]
Z0 = 0.0


def m_res_for(M2):
    return M2 * 1.0e-4


def top_two(masses):
    """Return (largest, second_largest) as a fraction of nothing in particular
    -- caller divides by M2. second_largest is None if fewer than 2 branches
    survived to this checkpoint (can't have a second progenitor)."""
    if not masses:
        return None, None
    s = sorted(masses, reverse=True)
    return s[0], (s[1] if len(s) > 1 else None)


def histogram_1st_2nd(top1_list, top2_list, M2, n_bins=20, log_range=(-4.5, 0.05)):
    edges = np.linspace(log_range[0], log_range[1], n_bins + 1)
    bins = M2 * 10 ** edges
    centers = 0.5 * (edges[:-1] + edges[1:])

    n1 = np.array([m for m in top1_list if m is not None])
    n2 = np.array([m for m in top2_list if m is not None])
    n_real = len(top1_list)

    count1, _ = np.histogram(n1, bins=bins)
    count2, _ = np.histogram(n2, bins=bins) if len(n2) else (np.zeros(n_bins), None)

    d_ln_ratio = (edges[1] - edges[0]) * np.log(10.0)
    f1 = count1 / (n_real * d_ln_ratio)
    f2 = count2 / (n_real * d_ln_ratio)

    with np.errstate(divide="ignore"):
        log_f1 = np.log10(f1)
        log_f2 = np.log10(f2)
    log_f1[count1 == 0] = np.nan
    log_f2[count2 == 0] = np.nan
    return centers, log_f1, log_f2


def run_column(pch, cosmo_data, M2, checkpoints, dz, n_trees, model, rng, n_grid, s_max_factor, target_nupper=0.5):
    """target_nupper (None falls back to the old fixed-dz behaviour) drives
    PCH08's population growth through grow_full_population_pch08_batch
    (adaptive per-branch stepping, dz as the step-size ceiling; uses the
    ~400-600x faster numba kernel when available -- see that function's
    docstring) instead of a fixed-dz per-tree Python loop."""
    M_res = m_res_for(M2)
    z_max = max(checkpoints)

    pch_top1 = {z1: [] for z1 in checkpoints}
    pch_top2 = {z1: [] for z1 in checkpoints}
    if target_nupper is not None:
        pch_pops = grow_full_population_pch08_batch(pch, M2, Z0, z_max, M_res, checkpoints, n_trees, target_nupper=target_nupper, dz_max=dz)
    else:
        pch_pops = [grow_full_population_pch08(pch, M2, Z0, z_max, M_res, dz, checkpoints) for _ in range(n_trees)]
    for pops in pch_pops:
        for z1 in checkpoints:
            t1, t2 = top_two(pops[z1])
            pch_top1[z1].append(t1)
            pch_top2[z1].append(t2)

    zh_top1 = {z1: [] for z1 in checkpoints}
    zh_top2 = {z1: [] for z1 in checkpoints}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        zh_pops = grow_full_population_zh_batch(
            cosmo_data, M2, Z0, z_max, M_res, dz, checkpoints, n_trees, model=model, rng=rng, N_grid=n_grid, S_max_factor=s_max_factor
        )
    for pops in zh_pops:
        for z1 in checkpoints:
            t1, t2 = top_two(pops[z1])
            zh_top1[z1].append(t1)
            zh_top2[z1].append(t2)

    return (pch_top1, pch_top2), (zh_top1, zh_top2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--n-trees", type=int, default=40)
    parser.add_argument("--dz", type=float, default=0.05)
    parser.add_argument("--n-grid", type=int, default=40)
    parser.add_argument("--s-max-factor", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--target-nupper",
        type=float,
        default=0.5,
        help="PCH08's adaptive-step target for Nupper (per-step split probability); --dz is used as the "
        "per-branch maximum step size. See fig1_conditional_mass_function.py's --target-nupper help for "
        "why fixed-dz stepping is not Nupper-compliant at this figure's M_res/M2 ratio. Pass a negative "
        "value to fall back to the old fixed-dz grow_full_population_pch08 behaviour.",
    )
    parser.add_argument("--output", default="fig2_progenitors.png")
    args = parser.parse_args()
    target_nupper = None if args.target_nupper < 0 else args.target_nupper

    run_params = foraois_io.get_params(args.config)
    cosmo_data = CosmoData(run_params, redshift=[Z0])
    pch = PCHMergerTree(cosmo_data, run_params)
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(Z1_VALUES), len(M2_VALUES), figsize=(11, 12), sharex="col")

    for col, M2 in enumerate(M2_VALUES):
        print(f"M2={M2:.2e} ...", flush=True)
        (pch1, pch2), (zh1, zh2) = run_column(
            pch, cosmo_data, M2, Z1_VALUES, args.dz, args.n_trees, "cdm", rng, args.n_grid, args.s_max_factor, target_nupper=target_nupper
        )
        for row, z1 in enumerate(Z1_VALUES):
            ax = axes[row, col]

            c, log_pch1, log_pch2 = histogram_1st_2nd(pch1[z1], pch2[z1], M2)
            _, log_zh1, log_zh2 = histogram_1st_2nd(zh1[z1], zh2[z1], M2)

            ax.step(c, log_pch1, where="mid", color="#1f77b4", lw=2.0, label=r"PCH08 $f_{\rm 1st}$")
            ax.step(c, log_pch2, where="mid", color="#1f77b4", lw=1.0, ls="--", label=r"PCH08 $f_{\rm 2nd}$")
            ax.step(c, log_zh1, where="mid", color="#9467bd", lw=2.0, label=r"Zhang-Hui $f_{\rm 1st}$")
            ax.step(c, log_zh2, where="mid", color="#9467bd", lw=1.0, ls="--", label=r"Zhang-Hui $f_{\rm 2nd}$")

            ax.set_ylim(-2, 0.5)
            ax.set_xlim(-4.5, 0.05)
            if row == 0:
                ax.set_title(f"$M_2={M2:.2e}\\,M_\\odot/h$", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"$z_1={z1}$\n" + r"$\log_{10} f_{\rm 1st,2nd}$", fontsize=9)
            if row == len(Z1_VALUES) - 1:
                ax.set_xlabel(r"$\log_{10}(M_1/M_2)$")
            ax.grid(alpha=0.3)

    axes[0, -1].legend(fontsize=7, loc="upper left")
    fig.suptitle("PCH08 Fig. 2 analogue: 1st/2nd most-massive progenitor (no N-body reference available)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
