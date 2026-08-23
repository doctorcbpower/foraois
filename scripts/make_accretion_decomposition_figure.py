#!/usr/bin/env python3
"""
Produce a smooth-accretion-vs-merger-mass illustration figure: a single
representative tree's cumulative smooth-accretion and merger-mass
channels, stacked against the total accreted mass, on a real Planck 2018
CAMB cosmology (config/planck2018_camb.yml).

This is a generic illustration of the decomposition mechanism -- the two
channels sum exactly to the total mass by construction.

The reference "total mass accreted" line MUST be M0 minus the per-step
mass_history, not mass_history[idx, 0] minus mass_history[idx, :] --
mass_history's index 0 is already one step *away* from M0 (the tree's
actual root mass), so anchoring on mass_history[idx, 0] silently drops
the first step's accretion and produces a total that no longer matches
cumsum(smooth_accretion + merger_mass), the exact identity
tests/test_pch_trees.py's test_smooth_accretion_and_merger_mass_conserve_mass_*
checks to 1e-6. This function asserts that identity itself before
plotting, so a regression here (in this script, or in the underlying
code) fails loudly instead of silently producing a mismatched figure.

Usage
-----
    python scripts/make_accretion_decomposition_figure.py [--output smooth_vs_merger]
"""

import argparse

import numpy as np

from foraois.cosmo_utils import CosmoData
from foraois.pch_trees import PCHMergerTree
from foraois.utils import io as foraois_io


def build_figure(
    config="config/planck2018_camb.yml",
    M0=1.0e12,
    z0=0.0,
    z_max=3.0,
    M_res=1.0e10,
    dz=0.2,
    n_trees=2000,
    seed=42,
    file_name="smooth_vs_merger",
):
    """Build a small forest, pick a survivor, and plot its cumulative accretion decomposition."""
    import matplotlib.pyplot as plt

    run_params = foraois_io.get_params(config)
    cosmo_data = CosmoData(run_params, redshift=[z0])
    tree_generator = PCHMergerTree(cosmo_data, run_params)

    np.random.seed(seed)  # build_forest_numpy uses the global np.random state
    M0_array = np.full(n_trees, M0)
    mass_history, _split_events, z_steps, smooth_accretion, merger_mass = tree_generator.build_forest_numpy(
        M0_array, z0=z0, z_max=z_max, M_res=M_res, dz=dz
    )

    idx = int(np.argmax(mass_history[:, -1] > 0))  # a tree that survives to z_max
    # z_steps includes z0 as its first element; the per-step accretion arrays
    # start at the first step past z0, so drop that leading entry to align them.
    z_steps_post = np.asarray(z_steps)[1:]
    cum_smooth = np.cumsum(smooth_accretion[idx])
    cum_merger = np.cumsum(merger_mass[idx])
    # M0, not mass_history[idx, 0] -- see docstring above.
    total = M0 - mass_history[idx, :]

    channel_sum = cum_smooth + cum_merger
    if not np.allclose(total, channel_sum, rtol=1e-8, atol=1e-6):
        max_dev = np.max(np.abs(total - channel_sum))
        raise AssertionError(
            f"mass-conservation identity violated for tree idx={idx}: "
            f"max|total - (cum_smooth+cum_merger)| = {max_dev:.4e} Msun/h "
            "(exceeds the 1e-6 tolerance tests/test_pch_trees.py checks) -- "
            "this points to a real bug, not a plotting issue; do not paper over it here."
        )

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.stackplot(
        z_steps_post,
        cum_smooth,
        cum_merger,
        labels=["smooth accretion (cumulative)", "merger mass (cumulative)"],
        colors=["#4C72B0", "#DD8452"],
        alpha=0.85,
    )
    ax.plot(z_steps_post, total, "k--", lw=1.2, label="total mass accreted (M0 - M(z))")
    ax.set_xlabel("redshift $z$")
    ax.set_ylabel(r"accreted mass [$M_\odot/h$]")
    ax.set_title("Smooth accretion vs. discrete mergers (single tree, CDM)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{file_name}.png", dpi=150)
    return f"{file_name}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="smooth_vs_merger")
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--M0", type=float, default=1.0e12, help="Msun/h")
    parser.add_argument("--M-res", type=float, default=1.0e10, help="Msun/h")
    parser.add_argument("--z0", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--dz", type=float, default=0.2)
    parser.add_argument("--n-trees", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = build_figure(
        config=args.config,
        M0=args.M0,
        z0=args.z0,
        z_max=args.z_max,
        M_res=args.M_res,
        dz=args.dz,
        n_trees=args.n_trees,
        seed=args.seed,
        file_name=args.output,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
