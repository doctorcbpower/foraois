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


def build_figure(configs=DEFAULT_CONFIGS, reference="CDM", z0=0.0, file_name="dm_model_comparison", pk_kmax=None):
    """Build the DM-model P(k)/sigma(M) comparison figure and save it to file_name.png.

    pk_kmax : float or None
        If given, overrides each config's own Code.pk_kmax in memory (the files on disk are untouched).
        The shipped configs' pk_kmax=100 is only accurate (1%) for sigma(M)/alpha(M) above ~8e7 Msun/h
        (docs/MODELS.md's "Numerical validity" section), while plot_dm_model_comparison's mass grid
        starts at log10(M)=6 -- so the default configs alone give sigma(M)/sigma_CDM(M) about 7 per cent
        high at that end. Pass e.g. 1000 (accurate to 1% down to ~2e4 Msun/h) to remove this.
    """
    models = {}
    for label, cfg in configs:
        run_params = foraois_io.get_params(cfg)
        if pk_kmax is not None:
            run_params["Code"]["pk_kmax"] = float(pk_kmax)
        cosmo_data = CosmoData(run_params, redshift=[z0])
        pk_data = cosmo_data.get_power_spectrum()
        cosmo_data.prepare_sigma_grid(pk_data)
        models[label] = (cosmo_data, pk_data)
    return plot.plot_dm_model_comparison(models, reference=reference, file_name=file_name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # base name, no extension -- utils/plot.py's _save_or_show appends .pdf/.png
    parser.add_argument("--output", default="dm_model_comparison")
    parser.add_argument("--reference", default="CDM")
    parser.add_argument("--z0", type=float, default=0.0)
    parser.add_argument("--pk-kmax", type=float, default=None, help="override each config's own pk_kmax (see build_figure's docstring)")
    args = parser.parse_args()
    plot.use_paper_style("full")  # SciencePlots, printed size

    result = build_figure(reference=args.reference, z0=args.z0, file_name=args.output, pk_kmax=args.pk_kmax)
    print(result)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
