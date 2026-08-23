#!/usr/bin/env python3
"""
Produce an illustrative single-tree dendrogram figure
(utils/plot.py's plot_dendrogram), built with build_full_tree (every
progenitor, not just the main branch) on a real Planck 2018 CAMB
cosmology (config/planck2018_camb.yml), in the style of Nadler et al.
(2023)'s own tree figures.

Usage
-----
    python scripts/make_tree_dendrogram_figure.py [--output tree_dendrogram]
"""

import argparse

from foraois.cosmo_utils import CosmoData
from foraois.pch_trees import PCHMergerTree
from foraois.utils import io as foraois_io
from foraois.utils import plot


def build_figure(
    config="config/planck2018_camb.yml",
    M0=1.0e13,
    z0=0.0,
    z_max=4.0,
    M_res=1.0e11,
    dz=0.2,
    file_name="tree_dendrogram",
):
    """Build a single full (all-progenitor) tree and render it as a dendrogram."""
    run_params = foraois_io.get_params(config)
    cosmo_data = CosmoData(run_params, redshift=[z0])
    tree_generator = PCHMergerTree(cosmo_data, run_params)
    nodes = tree_generator.build_full_tree(M0=M0, z0=z0, z_max=z_max, M_res=M_res, dz=dz)
    result = plot.plot_dendrogram(nodes, cosmo_data, file_name=file_name)
    return nodes, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tree_dendrogram")
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--M0", type=float, default=1.0e13, help="Msun/h")
    parser.add_argument("--M-res", type=float, default=1.0e11, help="Msun/h")
    parser.add_argument("--z0", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=4.0)
    parser.add_argument("--dz", type=float, default=0.2)
    args = parser.parse_args()

    nodes, result = build_figure(
        config=args.config,
        M0=args.M0,
        z0=args.z0,
        z_max=args.z_max,
        M_res=args.M_res,
        dz=args.dz,
        file_name=args.output,
    )
    print(f"{len(nodes)} nodes")
    print(result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
