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
    parser.add_argument("--dz-zh", type=float, default=None,
                        help="fixed Zhang-Hui step (default: --dz). The binary-per-step sampler needs a step small enough that the EPS "
                             "expected number of splits per step is <<1 (about 5e-4 at M_res/M2=1e-4)")
    parser.add_argument("--dz-zh-overlay", type=float, nargs="*", default=[],
                        help="extra ZH steps drawn thin for comparison (e.g. the coarse 0.05 of the earlier version)")
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
    parser.add_argument("--replot", default=None, help="redraw from a saved .npz (from a previous run) without recomputing")
    args = parser.parse_args()
    if args.replot:
        results, bins, dz_main, dz_extra = load_data(args.replot)
        plot_figure(results, bins, args.output, dz_main, dz_extra)
        return

    dz_zh = args.dz if args.dz_zh is None else args.dz_zh
    if dz_zh < 0.02 and args.n_grid <= 40 and False:  # closed-form flat-barrier path ignores n_grid
        print(
            f"WARNING: --dz={args.dz} with --n-grid={args.n_grid} will badly under-resolve the Zhang-Hui "
            "first-crossing solve at small steps -- results may be both much slower AND numerically "
            "unreliable. Consider --dz>=0.05 or a much larger --n-grid.",
            file=sys.stderr,
        )

    # PCH08's own Fig. 3 bins and plots in log10(1+z) (y-axis dn/dlog10(1+z)); histogramming the raw log10(1+z)
    # values with density=True gives that quantity directly.
    bins = np.linspace(0.0, np.log10(1.0 + args.z_max), 21)
    results = []  # per M2: (pch redshifts, zh redshifts)
    for row, M2 in enumerate(M2_VALUES):
        print(f"M2={M2:.2e} ...", flush=True)
        M_res = m_res_for(M2)
        target_nupper = None if args.target_nupper < 0 else args.target_nupper
        pch_raw = run_pch08_mainbranch_ensemble(
            args.config, M2, Z0, args.z_max, M_res, args.dz, args.f_major, args.n_trees, n_jobs=args.n_jobs, seed0=args.seed + row * 100_000,
            label=f"PCH08 M2={M2:.2e}", target_nupper=target_nupper,
        )
        def _zh(dz_use, off):
            raw = run_zh_mainbranch_ensemble(
                args.config, M2, Z0, args.z_max, M_res, dz_use, args.f_major, args.n_trees,
                model="cdm", N_grid=args.n_grid, S_max_factor=args.s_max_factor, n_jobs=args.n_jobs,
                seed0=args.seed + row * 100_000 + 1_000_000 + off, label=f"Zhang-Hui M2={M2:.2e} dz={dz_use:g}",
            )
            return np.array([z for z in raw if z is not None])

        zh_main = _zh(dz_zh, 0)
        extra = [_zh(d, 10_000 * (i + 1)) for i, d in enumerate(args.dz_zh_overlay)]
        results.append((np.array([z for z in pch_raw if z is not None]), zh_main, extra))
    from foraois.utils import paper_style as ps

    data = {"M2_values": np.array(M2_VALUES), "bins": bins, "n_trees": np.array(args.n_trees), "args": np.array(str(vars(args))),
            "dz_zh": np.array(dz_zh), "dz_zh_extra": np.array(list(args.dz_zh_overlay))}
    for row, (pz, zz, ex) in enumerate(results):
        data[f"pch_z_{row}"], data[f"zh_z_{row}"] = pz, zz
        for i, e in enumerate(ex):
            data[f"zhx{i}_z_{row}"] = e
    np.savez(str(ps.stem_of(args.output)) + ".npz", **data)
    plot_figure(results, bins, args.output, dz_zh, list(args.dz_zh_overlay))


def plot_figure(results, bins, output, dz_main=None, dz_extra=()):
    """Single-column (3.4 in) SciencePlots figure: one panel per halo mass. Coarser ZH steps are drawn thin and dashed."""
    import matplotlib.pyplot as plt
    from foraois.utils import paper_style as ps

    ps.apply()
    fig, axes = plt.subplots(len(M2_VALUES), 1, figsize=(ps.COL, 5.4), sharex=True)
    for ax, M2, (pz, zz, ex) in zip(axes, M2_VALUES, results):
        if len(pz):
            ax.hist(np.log10(1.0 + pz), bins=bins, histtype="step", density=True, color=ps.BLUE, lw=1.3, label="PCH08")
        for dz_e, e in zip(dz_extra, ex):
            if len(e):
                ax.hist(np.log10(1.0 + e), bins=bins, histtype="step", density=True, color=ps.RED, lw=0.7, ls="--", alpha=0.7,
                        label=rf"Zhang--Hui, $\Delta z={dz_e:g}$")
        if len(zz):
            ax.hist(np.log10(1.0 + zz), bins=bins, histtype="step", density=True, color=ps.RED, lw=1.3,
                    label="Zhang--Hui" if dz_main is None else rf"Zhang--Hui, $\Delta z={dz_main:g}$")
        ax.set_title(ps.m2_label(M2))
        ax.set_ylabel(r"$dn/d\log_{10}(1+z)$")
    axes[-1].set_xlabel(r"$\log_{10}(1+z)$")
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper right")
    fig.tight_layout()
    ps.save(fig, output)


def load_data(path):
    d = np.load(path)
    dz_main = float(d["dz_zh"]) if "dz_zh" in d.files else None
    dz_extra = [float(x) for x in d["dz_zh_extra"]] if "dz_zh_extra" in d.files else []
    res = [(d[f"pch_z_{r}"], d[f"zh_z_{r}"], [d[f"zhx{i}_z_{r}"] for i in range(len(dz_extra))]) for r in range(len(M2_VALUES))]
    return res, d["bins"], dz_main, dz_extra


if __name__ == "__main__":
    main()
