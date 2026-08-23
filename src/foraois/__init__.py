"""
foraois: Monte Carlo dark-matter halo merger trees.

Public API
----------
CosmoData          -- cosmology/model layer: power spectrum, sigma(M),
                       delta_col(z) (see foraois.cosmo_utils)
PCHMergerTree       -- Parkinson, Cole & Helly (2008) fitted-rate trees,
                       with serial/numpy/numba backends
ZhangHuiMergerTree  -- exact, barrier-agnostic Zhang & Hui (2006) trees
build_constrained_tree -- Nadler et al. (2023) Brownian-bridge-constrained
                       trees, guaranteed to reach a chosen (M1, z1)
TreeAlgorithm       -- the build_tree/build_forest_numpy Protocol both
                       tree classes satisfy
MassFunctions       -- Press-Schechter / Sheth-Tormen halo mass functions

Plotting (foraois.utils.plot), validation diagnostics (foraois.diagnostics),
config loading (foraois.utils.io), and the collapse-barrier/transfer-function
layers (foraois.collapse, foraois.transfer_functions) are reached via their
own submodules rather than re-exported here -- see README.md and
docs/MODELS.md.
"""

from foraois.cosmo_utils import CosmoData
from foraois.mass_function_utils import MassFunctions
from foraois.pch_trees import PCHMergerTree
from foraois.tree_algorithm import TreeAlgorithm
from foraois.zhang_hui_constrained_trees import build_constrained_tree
from foraois.zhang_hui_trees import ZhangHuiMergerTree

__all__ = [
    "CosmoData",
    "MassFunctions",
    "PCHMergerTree",
    "TreeAlgorithm",
    "ZhangHuiMergerTree",
    "build_constrained_tree",
]
