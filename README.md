## Foraois
![Foraois banner](./foraois_banner.jpg)

[![CI](https://github.com/doctorcbpower/foraois/actions/workflows/ci.yml/badge.svg)](https://github.com/doctorcbpower/foraois/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23077021.svg)](https://doi.org/10.5281/zenodo.23077021)

### Monte Carlo dark matter halo merger trees, for a range of dark matter models

Foraois (Irish, "forest") generates Monte Carlo merger trees for dark matter
halo growth histories, with two interchangeable algorithms:

- **`PCHMergerTree`** -- Parkinson, Cole & Helly (2008)'s fitted branching-rate
  approach, with three interchangeable performance backends (`serial`/`numpy`/`numba`).
- **`ZhangHuiMergerTree`** -- builds binary-per-step trees from the Zhang &
  Hui (2006) first-crossing distribution rather than a fitted rate. The
  first-crossing solver accepts a general barrier. The vectorised builders use
  the fixed barrier only; the serial builder also supports an illustrative
  scale-dependent barrier (`barrier: linear`, see docs/MODELS.md). Includes a working implementation of
  Nadler, Benson, Driskell, Du & Gluscevic (2023)'s Brownian-bridge-constrained
  excursions: merger trees guaranteed to reach a specified progenitor mass at
  a specified redshift, useful for cheaply sampling rare/outlier growth
  histories without down-sampling a huge unconstrained ensemble.

Both support CDM, WDM, and FDM cosmologies via power-spectrum transfer
functions and window-function choice. See [docs/MODELS.md](docs/MODELS.md)
for the governing equations behind every module, and [ROADMAP.md](ROADMAP.md)
for what's not yet covered.

Cosmology-specific physics (the linear power spectrum, the mass variance
sigma(M), and the linear collapse threshold delta_col(z)) is computed once in
`CosmoData` and handed to the tree-growth kernels as precomputed lookup
tables. Because the fast kernels only ever consume those tables -- never
`P(k)` directly -- any dark matter model that can be expressed as a modified
power spectrum/window function (e.g. WDM, fuzzy/scalar-field DM) plugs in at
the `CosmoData` layer without touching the tree-building code itself.

## Installation

```
git clone git@github.com:doctorcbpower/foraois.git
cd foraois
pip install -e .
```

Core dependencies (`numpy`, `scipy`, `pyyaml`, `matplotlib`) install automatically. A linear power spectrum is computed via CLASS, CAMB, or your own tabulated table (`mode: user`, no extra dependency at all -- see `config/planck2018_user.yml`); the JIT-parallel `numba` backend is an optional extra -- CLASS/CAMB/`numba` are all optional, install whichever you need:

```
pip install -e .[class]   # requires classy (compiled from source)
pip install -e .[camb]    # pip-installable
pip install -e .[numba]   # build_forest_numba (PCHMergerTree, ZhangHuiMergerTree); not needed for build_tree/build_forest_numpy
```

The example configs under `config/` (used throughout this README and the demo notebook) are part of the git checkout, not shipped with the package itself -- the commands below assume you're running from the repository root. Not yet published to PyPI.

## Quick start

**PCH08, via the CLI** -- the fastest way to generate a large forest of trees:

```
python -m foraois.main --params_file config/planck2018.yml --n_trees 1000 --backend numpy
```

(`--backend numba` is faster still for large `N`, but needs `pip install -e .[numba]` first -- see Installation above.)

**PCH08, via Python** -- for programmatic use, the same algorithm:

```python
from foraois import CosmoData, PCHMergerTree
from foraois.utils import io

run_params = io.get_params("config/planck2018.yml")
cosmo_data = CosmoData(run_params, redshift=[0.0])
tree_generator = PCHMergerTree(cosmo_data, run_params)

tree = tree_generator.build_tree(M0=1e12, z0=0.0, z_max=5.0, M_res=1e9, dz=0.002)
```

(`dz` this small keeps `Nupper` -- the expected splits per step, see `foraois.diagnostics.expected_splits_per_step` -- comfortably below 1 for this mass ratio; PCH08's own single-split-per-step architecture assumes `Nupper` stays small, and a coarser `dz` here would silently violate that.)

**The Zhang-Hui algorithm** -- same tree-of-dicts shape as `build_tree` above (binary-per-step approximation; constant barriers in the vectorised builders):

```python
from foraois import ZhangHuiMergerTree

tree_generator = ZhangHuiMergerTree(cosmo_data, run_params, model="cdm")
tree = tree_generator.build_tree(M0=1e12, z0=0.0, z_max=5.0, M_res=1e9, dz=0.2)
```

**A constrained tree**, guaranteed to reach a specified `(M1, z1)`:

```python
from foraois import build_constrained_tree

tree = build_constrained_tree(
    M0=1e12, z0=0.0, M1=1e11, z1=4.0, z_max=8.0, M_res=1e9,
    cosmo_data=cosmo_data, model="cdm",
)
```

See [notebooks/foraois_demo.ipynb](notebooks/foraois_demo.ipynb) for a full interactive walkthrough of all three, and [docs/MODELS.md](docs/MODELS.md) for the API/equations behind each.

## Documentation

- [docs/MODELS.md](docs/MODELS.md) -- the governing equations, config reference (cosmology, dark matter models, window functions), and full CLI reference (`--algorithm`/`--backend` support matrix).
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) -- module-by-module package structure, and a pipeline map from configuration to trees.
- [docs/TESTING.md](docs/TESTING.md) -- what the test suite covers and how to run it.
- [docs/PAPER_FIGURES.md](docs/PAPER_FIGURES.md) -- which script reproduces which figure in the software-release paper, and the exact command used for each.
- [docs/PCH08_HIGH_Z_DIAGNOSTIC.md](docs/PCH08_HIGH_Z_DIAGNOSTIC.md) -- open question: PCH08 main-progenitor histories at small `M_res / M0` differ from Zhang-Hui; read before using PCH08 at `M_res / M0` below ~1e-2.
- [ROADMAP.md](ROADMAP.md) -- known gaps and open research questions.

## Interactive exploration

`notebooks/foraois_demo.ipynb` walks through cosmology setup, both tree-building algorithms and all their backends, the built-in visualisations, WDM/FDM/sharp-k comparisons, and Brownian-bridge-constrained trees. Open it with `jupyter notebook notebooks/foraois_demo.ipynb` after `pip install -e .[camb] jupyter`.

## References

This code implements the algorithms in:

* Parkinson, Cole & Helly 2008, **Generating dark matter halo merger trees**, _MNRAS_, 383, 557. [DOI: 10.1111/j.1365-2966.2007.12517.x](https://doi.org/10.1111/j.1365-2966.2007.12517.x)
* Nadler, Benson, Driskell, Du & Gluscevic 2023, **Growing the first galaxies' merger trees**, _MNRAS_, 521, 3201. [arXiv:2212.08584](https://arxiv.org/abs/2212.08584)
* Viel, Lesgourgues, Haehnelt, Matarrese & Riotto 2005, **Constraining warm dark matter candidates including sterile neutrinos and light gravitinos with WMAP and the Lyman-alpha forest**, _Phys. Rev. D_, 71, 063534 (thermal-relic WDM transfer function, eq. 6-7). [arXiv:astro-ph/0501562](https://arxiv.org/abs/astro-ph/0501562)
* Hu, Barkana & Gruzinov 2000, **Fuzzy Cold Dark Matter: The Wave Properties of Ultralight Particles**, _Phys. Rev. Lett._, 85, 1158 (FDM transfer function, eq. 8-9). [arXiv:astro-ph/0003365](https://arxiv.org/abs/astro-ph/0003365)
* Benson, Farahi, Cole, Moustakas, Jenkins, Lovell, Kennedy, Helly & Frenk 2013, **Dark matter halo merger histories beyond cold dark matter: I. Methods and application to warm dark matter**, _MNRAS_, 428, 1774 (sharp-k window calibration, `alpha=2.5`). [DOI: 10.1093/mnras/sts159](https://doi.org/10.1093/mnras/sts159)
* Kulkarni & Ostriker 2022, **What is the halo mass function in a fuzzy dark matter cosmology?**, _MNRAS_, 510, 1425 (sharp-k window applied to FDM, `alpha=2.5`, cross-checked against Benson et al. 2013 and Lacey & Cole 1994). [arXiv:2011.02116](https://arxiv.org/abs/2011.02116)
* Zhang & Hui 2006, **On Random Walks with a General Moving Barrier**, (first-crossing distribution via a Volterra integral equation, eq. 5, solved by forward substitution; exact for a Markov walk, and the Markov approximation when applied to a top-hat `S(M)`). [arXiv:astro-ph/0508384](https://arxiv.org/abs/astro-ph/0508384)
