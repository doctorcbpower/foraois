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
    dz=0.01,
    n_trees=20_000,
    seed=42,
):
    """Run matched PCH08/Zhang-Hui forests; return their surviving-mass arrays and summary stats."""
    run_params = foraois_io.get_params(config)
    cosmo_data = CosmoData(run_params, redshift=[z0])

    pch_tree = PCHMergerTree(cosmo_data, run_params)
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
    """Plot the two surviving-mass distributions as overlaid step histograms."""
    import matplotlib.pyplot as plt

    # Wide enough, with a short title, that the mean-mass-difference number
    # in the title doesn't get clipped at the figure's right edge -- this
    # clipped previously at figsize=(6, 4.5) with the fuller title text.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.logspace(
        np.log10(min(pch_alive.min(), zh_alive.min())),
        np.log10(max(pch_alive.max(), zh_alive.max())),
        40,
    )
    ax.hist(
        pch_alive,
        bins=bins,
        histtype="step",
        lw=1.6,
        color="#4C72B0",
        label="PCH08 (fitted rate)",
    )
    ax.hist(
        zh_alive,
        bins=bins,
        histtype="step",
        lw=1.6,
        color="#C44E52",
        label="Zhang-Hui (exact rate)",
    )
    ax.axvline(pch_alive.mean(), color="#4C72B0", ls="--", lw=1)
    ax.axvline(zh_alive.mean(), color="#C44E52", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel(
        rf"surviving main-progenitor mass at $z_{{\rm max}}={stats['z_max']:g}$ "
        r"[$M_\odot/h$]"
    )
    ax.set_ylabel("N trees")
    ax.set_title(
        f"PCH08 vs. barrier-agnostic solver (n={stats['n_trees']:,})\n"
        rf"mean-mass difference = {stats['mean_mass_discrepancy_pct']:+.1f}%",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{file_name}.png", dpi=150)
    return f"{file_name}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--M0", type=float, default=1.0e12, help="Msun/h")
    parser.add_argument("--M-res", type=float, default=1.0e10, help="Msun/h")
    parser.add_argument("--z0", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=1.0)
    parser.add_argument("--dz", type=float, default=0.01)
    parser.add_argument("--n-trees", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-figure",
        default=None,
        help="base name (no extension) for the surviving-mass histogram figure; "
        "if omitted, only the printed statistics are produced",
    )
    args = parser.parse_args()

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
        out = build_figure(pch_alive, zh_alive, stats, file_name=args.output_figure)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
