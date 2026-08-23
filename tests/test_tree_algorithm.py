"""
Tests for foraois.tree_algorithm -- verifies PCHMergerTree and
ZhangHuiMergerTree actually satisfy the shared TreeAlgorithm Protocol,
rather than just asserting it in a docstring.
"""

import pytest

from foraois.tree_algorithm import TreeAlgorithm
from foraois.zhang_hui_trees import ZhangHuiMergerTree

# build_tree's N_grid=60 below is chosen for test speed, not production
# accuracy -- same rationale/suppression as tests/test_zhang_hui_trees.py.
pytestmark = pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")


def test_pch_merger_tree_satisfies_tree_algorithm(tree_generator):
    assert isinstance(tree_generator, TreeAlgorithm)


def test_zhang_hui_merger_tree_satisfies_tree_algorithm(zh_tree_generator):
    zh_tree = ZhangHuiMergerTree(zh_tree_generator.cosmo_data, model="cdm")
    assert isinstance(zh_tree, TreeAlgorithm)


def test_plain_object_does_not_satisfy_tree_algorithm():
    # Sanity check the Protocol is actually discriminating, not vacuously
    # true for anything -- an object with neither method shouldn't pass.
    class NotATreeAlgorithm:
        pass

    assert not isinstance(NotATreeAlgorithm(), TreeAlgorithm)


def test_build_tree_return_shapes_match_across_backends(tree_generator, zh_tree_generator):
    # The Protocol itself only checks method names exist (see its own
    # docstring) -- this checks the documented return-value contract
    # directly: both backends' build_tree entries carry at least
    # {"redshift", "parent_mass", "progenitors"}.
    zh_tree = ZhangHuiMergerTree(zh_tree_generator.cosmo_data, model="cdm", N_grid=60)

    pch_result = tree_generator.build_tree(M0=1e12, z0=0.0, z_max=1.0, M_res=1e10, dz=0.2)
    zh_result = zh_tree.build_tree(M0=1e12, z0=0.0, z_max=1.0, M_res=1e10, dz=0.2)

    for result in (pch_result, zh_result):
        assert isinstance(result, list)
        for entry in result:
            assert {"redshift", "parent_mass", "progenitors"} <= entry.keys()
