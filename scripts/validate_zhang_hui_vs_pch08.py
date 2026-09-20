#!/usr/bin/env python3
"""
Compare ZhangHuiMergerTree's CDM output against PCHMergerTree's, on the
same real Planck 2018 CAMB cosmology (config/planck2018_camb.yml).

N23 explicitly defer recalibrating their unconstrained (exact-rate) trees
against N-body/PCH08 (see ROADMAP.md) -- so this script's job is to
characterize and report any discrepancy, not assert tight agreement. Both
backends are run over the same N independent trees at fixed (M0, z0,
z_max, M_res, dz); survival fraction and the surviving trees' final
main-progenitor mass distribution are compared.

Uses build_forest_numpy for both backends (Zhang-Hui's closed-form
flat-barrier sampler -- see zhang_hui_trees.py's
_build_forest_flat_barrier_numpy docstring -- makes this both exact and
fast enough that --n-trees can be large).

With --output-figure, also produces the surviving-mass histogram figure.

Usage
-----
    python scripts/validate_zhang_hui_vs_pch08.py [--n-trees 20000]
    python scripts/validate_zhang_hui_vs_pch08.py --output-figure pch08_vs_zhanghui
"""

import argparse

import numpy as np

from foraois.cosmo_utils import CosmoData
from foraois.pch_trees import PCHMergerTree
from foraois.utils import io as foraois_io
from foraois.zhang_hui_trees import ZhangHuiMergerTree


def run_comparison(
    config="config/planck2018_camb.yml",
    M0=1.0e12,
    M_res=1.0e10,
    z0=0.0,
    z_max=1.0,
    dz=0.005,
    n_trees=20_000,
    seed=42,
    check_nupper=True,
):
    """Run matched PCH08/Zhang-Hui forests; return their surviving-mass arrays and summary stats.

    dz's default changed from 0.2 to 0.005 (2025 correction): PCHMergerTree
    uses a single fixed dz shared by the whole forest rather than PCH08's
    own adaptive per-halo step (see PCHMergerTree's class docstring), so
    Nupper -- the per-step split probability PCH08 designs to keep <<1,
    ~0.1 in practice (their sec. 2.1) -- is NOT bounded small by
    construction. At dz=0.2 (this script's old default) Nupper reaches
    3.5 for this (M0, M_res) pair, three-and-a-half times the collapse
    algorithm's own design assumption, not merely "a bit coarse". This
    silently flipped the sign of the reported comparison: at dz=0.2 the
    Zhang-Hui backend's mean surviving mass looked ~11% BELOW PCH08's; at
    dz=0.005 (max Nupper=0.082, safely compliant; see
    foraois.diagnostics.expected_splits_per_step) it is ~22% ABOVE -- a
    sign flip, not a refinement. A second, independent bug (a sign error
    in the Nupper/S_coeff normalization itself, see _branching_rate_terms
    and this repo's own fix history) was found and corrected after that,
    via cross-validation against Parkinson's reference FORTRAN
    implementation; with it fixed, this comparison moves from ~22% to
    ~11% (still above PCH08, so a magnitude correction rather than a
    second sign flip), stable from dz=0.01 down to dz=0.002 (12.2%,
    11.4%, 11.1% respectively) -- i.e. converged, not just "less wrong".
    Do not lower dz further than necessary for Nupper compliance purely
    for its own sake: Zhang-Hui's own first_crossing_step under-resolves
    at very fine dz (see fig1_conditional_mass_function.py's docstring)
    -- dz=0.005 is a deliberate middle point that satisfies both
    constraints for this (M0, M_res, z_max) configuration, not a
    universal default.
    """
    run_params = foraois_io.get_params(config)
    cosmo_data = CosmoData(run_params, redshift=[z0])

    pch_tree = PCHMergerTree(cosmo_data, run_params)

    if check_nupper:
        from foraois.diagnostics import expected_splits_per_step

        _, Nupper, _ = expected_splits_per_step(pch_tree, M0, z0, z_max, M_res, dz=dz)
        max_nupper = float(np.nanmax(Nupper))
        if max_nupper > 0.15:
            import warnings

            warnings.warn(
                f"max Nupper={max_nupper:.3f} at dz={dz} for this (M0, M_res, z_max) -- "
                "PCH08's own design target is <<1 (~0.1); the PCH08-vs-Zhang-Hui comparison "
                "below may not be trustworthy at this dz. Reduce --dz (or, for a different "
                "M0/M_res/z_max choice, re-check with foraois.diagnostics.expected_splits_per_step "
                "before trusting the result).",
                stacklevel=2,
            )
    zh_tree = ZhangHuiMergerTree(
        cosmo_data,
        run_params,
        model="cdm",
        rng=np.random.default_rng(seed),
    )

    M0_array = np.full(n_trees, M0)

    np.random.seed(seed)  # PCHMergerTree.build_forest_numpy uses the global np.random state
    pch_mh, *_ = pch_tree.build_forest_numpy(M0_array, z0, z_max, M_res, dz)
    zh_mh, *_ = zh_tree.build_forest_numpy(M0_array, z0, z_max, M_res, dz)

    pch_finals = pch_mh[:, -1]
    zh_finals = zh_mh[:, -1]

    pch_alive = pch_finals[pch_finals > 0]
    zh_alive = zh_finals[zh_finals > 0]

    stats = {
        "M0": M0,
        "z0": z0,
        "z_max": z_max,
        "M_res": M_res,
        "dz": dz,
        "n_trees": n_trees,
        "pch_frac_alive": float((pch_finals > 0).mean()),
        "zh_frac_alive": float((zh_finals > 0).mean()),
        "pch_mean": float(pch_alive.mean()) if pch_alive.size else float("nan"),
        "zh_mean": float(zh_alive.mean()) if zh_alive.size else float("nan"),
    }
    stats["mean_mass_discrepancy_pct"] = (
        (stats["zh_mean"] / stats["pch_mean"] - 1.0) * 100 if pch_alive.size and zh_alive.size else float("nan")
    )
    return pch_alive, zh_alive, stats


def build_figure(pch_alive, zh_alive, stats, file_name="pch08_vs_zhanghui"):
    """Plot the two surviving-mass distributions as overlaid step histograms (single-column, SciencePlots)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, NullFormatter

    from foraois.utils import paper_style as ps

    ps.apply()
    fig, ax = plt.subplots(figsize=(ps.COL, 2.7))
    bins = np.logspace(
        np.log10(min(pch_alive.min(), zh_alive.min())),
        np.log10(max(pch_alive.max(), zh_alive.max())),
        40,
    )
    ax.hist(pch_alive, bins=bins, histtype="step", lw=1.3, color=ps.BLUE, label="PCH08 (fitted rate)")
    ax.hist(zh_alive, bins=bins, histtype="step", lw=1.3, color=ps.RED, label="Zhang--Hui (exact rate)")
    ax.axvline(pch_alive.mean(), color=ps.BLUE, ls="--", lw=0.9)
    ax.axvline(zh_alive.mean(), color=ps.RED, ls="--", lw=0.9)
    ax.set_xscale("log")
    ticks = [t for t in (5e10, 1e11, 3e11) if bins[0] <= t <= bins[-1] * 1.05]
    ax.set_xticks(ticks)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"${ps.sci(x, 0)}$".replace("1\\times", "")))
    ax.set_xlabel(rf"surviving main-progenitor mass at $z_{{\rm max}}={stats['z_max']:g}$ [$M_\odot/h$]")
    ax.set_ylabel(r"$N_{\rm trees}$")
    ax.text(0.03, 0.95, f"$n=$ {stats['n_trees']:,}\nmean-mass difference $= {stats['mean_mass_discrepancy_pct']:+.1f}\\%$",
            transform=ax.transAxes, ha="left", va="top")
    ax.legend(loc="center left", frameon=False)
    fig.tight_layout()
    ps.save(fig, file_name)
    return f"{file_name}.pdf/.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--M0", type=float, default=1.0e12, help="Msun/h")
    parser.add_argument("--M-res", type=float, default=1.0e10, help="Msun/h")
    parser.add_argument("--z0", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=1.0)
    parser.add_argument("--dz", type=float, default=0.005, help="see run_comparison's docstring for why this default changed from 0.2")
    parser.add_argument("--n-trees", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-figure",
        default=None,
        help="base name (no extension) for the surviving-mass histogram figure; "
        "if omitted, only the printed statistics are produced",
    )
    parser.add_argument("--replot", default=None, help="redraw from a saved .pkl (from a previous run) without recomputing")
    args = parser.parse_args()
    if args.replot:
        from foraois.utils import paper_style as ps

        build_figure(*ps.load_data(args.replot), file_name=args.output_figure or "pch08_vs_zhanghui")
        return

    pch_alive, zh_alive, stats = run_comparison(
        config=args.config,
        M0=args.M0,
        M_res=args.M_res,
        z0=args.z0,
        z_max=args.z_max,
        dz=args.dz,
        n_trees=args.n_trees,
        seed=args.seed,
    )

    print(
        f"M0={stats['M0']:.2e} Msun/h, z0={stats['z0']} -> z_max={stats['z_max']}, "
        f"M_res={stats['M_res']:.2e} Msun/h, dz={stats['dz']}, n_trees={stats['n_trees']}"
    )
    print("\nPCH08 (fitted rate)")
    print(f"  fraction with M > M_res at z_max={stats['z_max']}: {stats['pch_frac_alive']:.4f}")
    print(f"  surviving-tree final mass: mean={stats['pch_mean']:.4e} Msun/h")
    print("\nZhang-Hui (exact rate)")
    print(f"  fraction with M > M_res at z_max={stats['z_max']}: {stats['zh_frac_alive']:.4f}")
    print(f"  surviving-tree final mass: mean={stats['zh_mean']:.4e} Msun/h")

    print("\nDiscrepancy (Zhang-Hui vs PCH08, not asserted to be small -- see ROADMAP.md):")
    print(f"  survival fraction: {stats['zh_frac_alive'] - stats['pch_frac_alive']:+.4f}")
    print(f"  mean surviving mass: {stats['mean_mass_discrepancy_pct']:+.2f}%")

    if args.output_figure:
        from foraois.utils import paper_style as ps

        ps.save_data(args.output_figure, (pch_alive, zh_alive, stats))
        out = build_figure(pch_alive, zh_alive, stats, file_name=args.output_figure)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
