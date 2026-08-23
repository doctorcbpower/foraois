#!/usr/bin/env python3
"""
Produce the branching-rate sampling-validation figure for the Software
Release Paper's Validation section (utils/plot.py's
plot_branching_rate_validation), using a real Planck 2018 CAMB cosmology
(config/planck2018_camb.yml) rather than the synthetic power-law tests use,
so the figure reflects an actual cosmology rather than a toy P(k).

Usage
-----
    python scripts/make_validation_figure.py [--output validation_branching_rate.png]
"""

import argparse

from foraois.cosmo_utils import CosmoData
from foraois.pch_trees import PCHMergerTree
from foraois.utils import io as foraois_io
from foraois.utils import plot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # base name, no extension -- utils/plot.py's _save_or_show appends .png
    parser.add_argument("--output", default="validation_branching_rate")
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    # A moderately deep step (large d_omega) at a mass ratio well above
    # M_res, so q's full [qres, 0.5] range is resolved and Nupper stays
    # comfortably below 1 (required by check_sampling_consistency).
    parser.add_argument("--M2", type=float, default=1.0e12, help="Msun/h")
    parser.add_argument("--M-res", type=float, default=1.0e10, help="Msun/h")
    parser.add_argument("--z0", type=float, default=0.0)
    # Small enough that Nupper stays comfortably below 1 for the default
    # M2/M_res ratio (checked directly: Nupper ~ 0.33 here) -- see the
    # module docstring's note on why a coarser step doesn't work.
    parser.add_argument("--z1", type=float, default=0.02)
    parser.add_argument("--n-trials", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_params = foraois_io.get_params(args.config)
    cosmo_data = CosmoData(run_params, redshift=[args.z0])
    tree_generator = PCHMergerTree(cosmo_data, run_params)

    delta0 = float(cosmo_data.delta_col_at_z(args.z0))
    delta1 = float(cosmo_data.delta_col_at_z(args.z1))
    d_omega = delta1 - delta0

    terms = tree_generator.branching_rate_terms(args.M2, args.M_res, delta0, d_omega)
    print(f"Nupper = {float(terms['Nupper']):.4f} (must be < 1)")

    result = plot.plot_branching_rate_validation(
        tree_generator,
        args.M2,
        args.M_res,
        delta0,
        d_omega,
        n_trials=args.n_trials,
        seed=args.seed,
        file_name=args.output,
    )
    print(result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
