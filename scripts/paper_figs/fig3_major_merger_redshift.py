#!/usr/bin/env python3
"""
Reproduces the structure of Parkinson, Cole & Helly (2008, "PCH08")
Figure 3: the redshift distribution of the most-recent major merger of
the main progenitor, for haloes of different final masses M2. A major
merger is defined, as in PCH08, as an event where the smaller of the two
resulting fragments is at least f_major=0.3 times the mass of the larger.

Unlike fig1/fig2, this only needs the *main-progenitor* history each
backend's own build_tree() already returns (no need for the full-branch
population machinery in _treegrowth.py) -- "most recent major merger of
the main progenitor" is, by definition, a main-branch-only question. This
also makes it much cheaper to run than fig1/fig2, including for
Zhang-Hui: it's one draw_progenitor_mass_zh call per step per tree, not
one per *branch* per step.

As with fig1/fig2, there is no Millennium N-body reference available, so
this compares foraois's PCH08 and Zhang-Hui backends to each other, not
to N-body; and Nadler et al. (2023)'s constrained sampler is out of scope
for the same reason given in fig1_conditional_mass_function.py's
docstring -- it targets efficient sampling of one specified rare outcome,
not the ordinary merger-rate statistics this figure characterizes.

Axis convention: PCH08's own Figure 3 bins and plots this in log10(1+z),
not linear z -- x-axis log10(1+z), y-axis dn/dlog10(1+z) (their axis
label reads "dn/dlog_10(1+z)"; easy to misread the subscript as
"dn/(1+z)" at a glance). This script matches that convention rather than
plotting a plain dN/dz, since they are visually different quantities (the
log stretches low-z and compresses high-z, changing where the
distribution appears to peak) -- do not "fix" an apparent low-z-vs-high-z
peak discrepancy against intuition without first checking which variable
is being plotted. Note that PCH08's own Figure 3 histograms also rise
toward higher z in all three of their mass panels, not toward z=0, so a
rising-with-z shape here is not by itself evidence of a bug.

Runtime and dz
--------------
Unlike fig1/fig2/fig4, this does NOT need a full branch-population growth
-- one build_tree() call per tree, either backend -- so it stays cheap
even at large n_trees; a Zhang-Hui tree at dz=0.05 takes order 0.1-0.3s
here. The tree loop is still embarrassingly parallel (--n-jobs), which
matters more once you push n_trees into the thousands. The same "do not
go far below dz=0.05 at the default --n-grid=40" caveat documented in
fig1_conditional_mass_function.py's docstring applies here too (Zhang-Hui's
first_crossing_step under-resolves the barrier shift for very fine steps,
regardless of which script is calling it) -- this script warns if you
combine a very small --dz with the default --n-grid.

Usage
-----
    python scripts/paper_figs/fig3_major_merger_redshift.py \
        [--n-trees 5000] [--dz 0.05] [--z-max 4.0] [--n-grid 40] \
        [--f-major 0.3] [--n-jobs 1] [--output fig3_major_mergers.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _treegrowth import run_pch08_mainbranch_ensemble, run_zh_mainbranch_ensemble  # noqa: E402

M2_VALUES = [1.0e12, 3.16e13, 1.0e15]
Z0 = 0.0


def m_res_for(M2):
    return M2 * 1.0e-4


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--n-trees", type=int, default=5000)
    parser.add_argument("--dz", type=float, default=0.05, help="Do not go far below 0.05 at the default --n-grid for Zhang-Hui; see module docstring")
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--n-grid", type=int, default=40)
    parser.add_argument("--s-max-factor", type=float, default=8.0)
    parser.add_argument("--f-major", type=float, default=0.3)
    parser.add_argument(
        "--target-nupper",
        type=float,
        default=0.1,
        help=(
            "PCH08's own adaptive-step target for Nupper (per-step split probability; "
            "PCH08 sec. 2.1 requires P<<1, ~0.1 in practice). The PCH08 backend uses "
            "adaptive dz (with --dz as the *ceiling* step size) chosen to keep Nupper "
            "at or below this value at every step, instead of PCHMergerTree.build_tree's "
            "fixed-dz stepping -- see _treegrowth.py's 'Adaptive-dz PCH08 stepping' "
            "section for why this is necessary. Pass a negative value to fall back to "
            "the old fixed-dz PCHMergerTree.build_tree behaviour (kept for comparison)."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="worker processes for the (embarrassingly parallel) tree loop")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="fig3_major_mergers.png")
    args = parser.parse_args()

    if args.dz < 0.02 and args.n_grid <= 40:
        print(
            f"WARNING: --dz={args.dz} with --n-grid={args.n_grid} will badly under-resolve the Zhang-Hui "
            "first-crossing solve at small steps -- results may be both much slower AND numerically "
            "unreliable. Consider --dz>=0.05 or a much larger --n-grid.",
            file=sys.stderr,
        )

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(M2_VALUES), figsize=(13, 4), sharey=True)
    # PCH08's own Fig. 3 bins and plots in log10(1+z), not linear z (x-axis
    # log10(1+z); y-axis dn/dlog10(1+z)) -- matched here so the shape is
    # comparable to theirs, not a different quantity in a different
    # variable. Binning the raw log10(1+z) values directly with
    # density=True gives dn/dlog10(1+z) automatically (density normalizes
    # by the bin width in whatever variable was actually binned).
    bins = np.linspace(0.0, np.log10(1.0 + args.z_max), 21)

    for col, M2 in enumerate(M2_VALUES):
        print(f"M2={M2:.2e} ...", flush=True)
        M_res = m_res_for(M2)

        target_nupper = None if args.target_nupper < 0 else args.target_nupper
        pch_raw = run_pch08_mainbranch_ensemble(
            args.config, M2, Z0, args.z_max, M_res, args.dz, args.f_major, args.n_trees, n_jobs=args.n_jobs, seed0=args.seed + col * 100_000,
            label=f"PCH08 M2={M2:.2e}", target_nupper=target_nupper,
        )
        pch_z = [z for z in pch_raw if z is not None]

        zh_raw = run_zh_mainbranch_ensemble(
            args.config, M2, Z0, args.z_max, M_res, args.dz, args.f_major, args.n_trees,
            model="cdm", N_grid=args.n_grid, S_max_factor=args.s_max_factor, n_jobs=args.n_jobs, seed0=args.seed + col * 100_000 + 1_000_000,
            label=f"Zhang-Hui M2={M2:.2e}",
        )
        zh_z = [z for z in zh_raw if z is not None]
        pch_x = np.log10(1.0 + np.asarray(pch_z))
        zh_x = np.log10(1.0 + np.asarray(zh_z))

        ax = axes[col]
        if pch_z:
            ax.hist(pch_x, bins=bins, histtype="step", density=True, color="#1f77b4", lw=1.8, label="PCH08")
        if zh_z:
            ax.hist(zh_x, bins=bins, histtype="step", density=True, color="#9467bd", lw=1.8, label="Zhang-Hui")
        ax.set_title(
            f"$M_2={M2:.2e}\\,M_\\odot/h$\n"
            f"(major merger found in {len(pch_z)}/{args.n_trees} PCH08, {len(zh_z)}/{args.n_trees} ZH trees)",
            fontsize=9,
        )
        ax.set_xlabel(r"$\log_{10}(1+z)$")
        if col == 0:
            ax.set_ylabel(r"$dn/d\log_{10}(1+z)$")
        ax.grid(alpha=0.3)

    axes[0].legend(fontsize=9)
    fig.suptitle(
        f"PCH08 Fig. 3 analogue: major-merger ($f_{{\\rm major}}={args.f_major}$) redshift (no N-body reference available)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
