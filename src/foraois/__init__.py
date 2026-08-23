# """
# foraois: A tool to generate Monte Carlo merger trees in general dark matter models.
# """

# Expose core classes and functions for cleaner imports
from foraois.cosmo_utils import CosmoData
from foraois.mass_function_utils import MassFunctions
from foraois.pch_trees import PCHMergerTree

__all__ = [
    "CosmoData",
    "MassFunctions",
    "PCHMergerTree",
]
