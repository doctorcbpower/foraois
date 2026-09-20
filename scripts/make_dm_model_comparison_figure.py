#!/usr/bin/env python3
"""
Produce a CDM/WDM/FDM power-spectrum-and-variance comparison figure
(utils/plot.py's plot_dm_model_comparison), using real Planck 2018 CAMB
cosmologies (config/planck2018_camb.yml and its WDM/FDM variants) rather
than synthetic P(k): switching dark matter model is a configuration-file
change, not a code change.

Usage
-----
    python scripts/make_dm_model_comparison_figure.py [--output dm_model_comparison]
"""

import argparse

from foraois.cosmo_utils import CosmoData
from foraois.utils import io as foraois_io
from foraois.utils import plot

DEFAULT_CONFIGS = [
    ("CDM", "config/planck2018_camb.yml"),
    ("WDM (3 keV)", "config/planck2018_wdm.yml"),
    ("FDM (1e-22 eV)", "config/planck2018_fdm.yml"),
]


def build_figure(configs=DEFAULT_CONFIGS, reference="CDM", z0=0.0, file_name="dm_model_comparison"):
    """Build the DM-model P(k)/sigma(M) comparison figure and save it to file_name.png."""
    models = {}
    for label, cfg in configs:
        run_params = foraois_io.get_params(cfg)
        cosmo_data = CosmoData(run_params, redshift=[z0])
        pk_data = cosmo_data.get_power_spectrum()
        cosmo_data._prepare_sigma_grid(pk_data)
        models[label] = (cosmo_data, pk_data)
    return plot.plot_dm_model_comparison(models, reference=reference, file_name=file_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # base name, no extension -- utils/plot.py's _save_or_show appends .pdf/.png
    parser.add_argument("--output", default="dm_model_comparison")
    parser.add_argument("--reference", default="CDM")
    parser.add_argument("--z0", type=float, default=0.0)
    args = parser.parse_args()
    plot.use_paper_style("full")  # SciencePlots, printed size

    result = build_figure(reference=args.reference, z0=args.z0, file_name=args.output)
    print(result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
