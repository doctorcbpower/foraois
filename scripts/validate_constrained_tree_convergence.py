#!/usr/bin/env python3
"""
Full-tree-level validation of build_constrained_tree, analogous to N23's
own Appendix B convergence tests (their Fig. B1: constrained solutions
converge to the unconstrained distribution for non-extreme constraint
parameters).

Rather than the literal M1 -> 0 (S1 -> infinity) limit N23 use -- not
practical here, since M1 can't meaningfully go below M_res for a real
tree-building constraint -- this checks the same underlying property in
the way that's actually usable: does a "typical" M1 (close to what an
unconstrained tree naturally reaches at z1 anyway) leave the branch's
*earlier* growth history statistically indistinguishable from
unconstrained trees at the same earlier redshift, while an "extreme" M1
(a genuine rare-outlier constraint) visibly shifts growth earlier --
confirming both that a weak constraint doesn't spuriously bias the tree,
and that the constraint mechanism has a real, correctly-directed effect
when it's supposed to (matching N23's own Fig. 6 finding: larger M1
shifts growth histories systematically earlier).

z_max = z1 throughout (no unconstrained continuation) -- this script
validates the constrained portion specifically; the continuation reuses
ZhangHuiMergerTree.build_tree unchanged.

With --output-figure, also produces a typical-vs-extreme growth-history
figure, drawing its representative individual histories from the same
typical/extreme M1 values reported in the printed statistics.

`first_crossing_step`'s own grid-resolution warning is expected at
moderate --n-grid; it is not silenced here so it stays visible when this
script is run directly.

Usage
-----
    python scripts/validate_constrained_tree_convergence.py [--n-trees 150]
    python scripts/validate_constrained_tree_convergence.py --output-figure constrained_vs_unconstrained
"""

import argparse

import numpy as np

from foraois.cosmo_utils import CosmoData
from foraois.pch_trees import PCHMergerTree
from foraois.utils import io as foraois_io
from foraois.zhang_hui_constrained_trees import build_constrained_tree
from foraois.zhang_hui_trees import ZhangHuiMergerTree


def mass_at_redshift(tree, M0, z_target):
    """Interpolate a build_tree/build_constrained_tree history's main-progenitor mass at z_target."""
    zs = [0.0] + [e["redshift"] for e in tree]
    masses = [M0] + [max(e["progenitors"]) for e in tree]
    return float(np.interp(z_target, zs, masses))


def history(tree, M0):
    """Return the (z, M) polyline of a build_tree/build_constrained_tree history, for plotting."""
    zs = [0.0] + [e["redshift"] for e in tree]
    ms = [M0] + [max(e["progenitors"]) if e["progenitors"] else 0.0 for e in tree]
    return zs, ms


def run_validation(
    config="config/planck2018_camb.yml",
    M0=1.0e12,
    M_res=1.0e10,
    z0=0.0,
    z1=1.0,
    z_earlier=0.5,
    dz=0.2,
    n_trees=150,
    n_grid=60,
    dS=0.05,
    seed=42,
):
    """Sample unconstrained trees, derive typical/extreme M1 targets, and validate constrained trees against them."""
    run_params = foraois_io.get_params(config)
    cosmo_data = CosmoData(run_params, redshift=[z0])
    PCHMergerTree(cosmo_data, run_params)  # side effect: populates the sigma(M) grid

    rng = np.random.default_rng(seed)
    zh_tree = ZhangHuiMergerTree(cosmo_data, run_params, model="cdm", rng=rng, N_grid=n_grid)

    unconstrained_at_z1 = np.empty(n_trees)
    unconstrained_at_earlier = np.empty(n_trees)
    unconstrained_histories = []
    for i in range(n_trees):
        tree = zh_tree.build_tree(M0, z0, z1, M_res, dz=dz)
        unconstrained_at_z1[i] = mass_at_redshift(tree, M0, z1)
        unconstrained_at_earlier[i] = mass_at_redshift(tree, M0, z_earlier)
        unconstrained_histories.append(history(tree, M0))

    M1_typical = float(np.median(unconstrained_at_z1))
    M1_extreme = float(np.percentile(unconstrained_at_z1, 99))

    results = {
        "M0": M0,
        "z0": z0,
        "z1": z1,
        "z_earlier": z_earlier,
        "M_res": M_res,
        "n_trees": n_trees,
        "M1_typical": M1_typical,
        "M1_extreme": M1_extreme,
        "unconstrained_earlier_mean": float(unconstrained_at_earlier.mean()),
        "unconstrained_histories": unconstrained_histories,
        "constrained": {},
    }

    rng2 = np.random.default_rng(seed + 1)
    for label, M1 in [("typical (median)", M1_typical), ("extreme (p99)", M1_extreme)]:
        earlier = np.empty(n_trees)
        histories = []
        for i in range(n_trees):
            tree = build_constrained_tree(
                M0,
                z0,
                M1,
                z1,
                z1,
                M_res,
                cosmo_data,
                model="cdm",
                dS=dS,
                rng=rng2,
                N_grid=n_grid,
                dz=dz,
            )
            earlier[i] = mass_at_redshift(tree, M0, z_earlier)
            histories.append(history(tree, M0))

        results["constrained"][label] = {
            "M1": M1,
            "earlier_mean": float(earlier.mean()),
            "earlier_median": float(np.median(earlier)),
            "ratio_to_unconstrained_mean": float(earlier.mean() / unconstrained_at_earlier.mean()),
            "histories": histories,
        }

    return results


def build_figure(results, n_show=12, file_name="constrained_vs_unconstrained"):
    """Plot a subsample of unconstrained, typical-constrained, and extreme-constrained histories."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for zs, ms in results["unconstrained_histories"][:n_show]:
        ax.plot(zs, ms, color="#4C72B0", alpha=0.30, lw=1.6)
    ax.plot([], [], color="#4C72B0", label="unconstrained")

    colors = {"typical (median)": "#55A868", "extreme (p99)": "#C44E52"}
    for label, block in results["constrained"].items():
        for zs, ms in block["histories"][:n_show]:
            ax.plot(zs, ms, color=colors[label], alpha=0.55, lw=1.8)
        ax.plot(
            [],
            [],
            color=colors[label],
            label=f"constrained, {label}: $M_1={block['M1']:.2e}$",
        )
        ax.scatter(
            [results["z1"]],
            [block["M1"]],
            color=colors[label],
            zorder=5,
            marker="*",
            s=90,
            edgecolor="k",
            linewidth=0.8,
        )

    ax.set_yscale("log")
    ax.set_xlabel("redshift $z$", fontsize=15)
    ax.set_ylabel(r"main-progenitor mass [$M_\odot/h$]", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{file_name}.png", dpi=150)
    return f"{file_name}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--M0", type=float, default=1.0e12, help="Msun/h")
    parser.add_argument("--M-res", type=float, default=1.0e10, help="Msun/h")
    parser.add_argument("--z0", type=float, default=0.0)
    parser.add_argument("--z1", type=float, default=1.0)
    parser.add_argument("--z-earlier", type=float, default=0.5)
    parser.add_argument("--dz", type=float, default=0.2)
    parser.add_argument("--n-trees", type=int, default=150)
    parser.add_argument(
        "--n-grid",
        type=int,
        default=60,
        help="solve_first_crossing grid resolution for the unconstrained backend",
    )
    parser.add_argument(
        "--dS",
        type=float,
        default=0.05,
        help="bridge path step size for the constrained backend",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-figure",
        default=None,
        help="base name (no extension) for the typical-vs-extreme growth-history "
        "figure; if omitted, only the printed statistics are produced",
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=12,
        help="number of individual histories per category drawn in the figure",
    )
    args = parser.parse_args()

    results = run_validation(
        config=args.config,
        M0=args.M0,
        M_res=args.M_res,
        z0=args.z0,
        z1=args.z1,
        z_earlier=args.z_earlier,
        dz=args.dz,
        n_trees=args.n_trees,
        n_grid=args.n_grid,
        dS=args.dS,
        seed=args.seed,
    )

    print(
        f"M0={results['M0']:.2e}, z0={results['z0']} -> z1={results['z1']}, "
        f"M_res={results['M_res']:.2e}, n_trees={results['n_trees']}"
    )
    print(f"\nUnconstrained mass at z1: median={results['M1_typical']:.3e}, p99={results['M1_extreme']:.3e}")
    print(f"Unconstrained mass at z={results['z_earlier']}: mean={results['unconstrained_earlier_mean']:.3e}")

    for label, block in results["constrained"].items():
        print(
            f"\nM1={label}={block['M1']:.3e}: constrained mass at z={results['z_earlier']}: "
            f"mean={block['earlier_mean']:.3e}, median={block['earlier_median']:.3e} "
            f"(ratio to unconstrained mean: {block['ratio_to_unconstrained_mean']:.3f})"
        )

    if args.output_figure:
        out = build_figure(results, n_show=args.n_show, file_name=args.output_figure)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
