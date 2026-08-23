"""
A shared interface for foraois' tree-building algorithms.

`foraois` has two structurally different tree-building algorithms --
`pch_trees.PCHMergerTree` (PCH08's fitted rate) and
`zhang_hui_trees.ZhangHuiMergerTree` (the exact Zhang-Hui/N23 rate) -- and
room for future backends (e.g. a generative-ML approach). `build_tree(M0,
z0, z_max, M_res, dz=0.1)` and `build_forest_numpy(M0_array, z0, z_max,
M_res, dz=0.1)` have identical signatures on both classes, and
`ZhangHuiMergerTree.build_forest_numpy` matches `PCHMergerTree.build_forest_numpy`'s
return shape exactly, so this `Protocol` documents and type-checks an
interface the two implementations already share.

`build_forest_numba` and `build_full_tree` are deliberately *not* part of
this Protocol -- `PCHMergerTree` has both, `ZhangHuiMergerTree` has
neither (no numba backend, no full-branching-structure builder). They
are real, useful capabilities, just not ones every backend is expected to
provide -- callers that need them should check for the concrete class or
use `hasattr`, not assume every `TreeAlgorithm` has them.

Uses `typing.Protocol` (structural typing) rather than an ABC both
classes explicitly inherit from: `PCHMergerTree` and `ZhangHuiMergerTree`
have genuinely different constructor signatures (the latter takes
`model`/`rng`/`N_grid`/`S_max_factor`, none of which apply to PCH08's
fitted rate), and they only share their tree-building methods, not their
construction or internal state. A `Protocol` gets the same type-checking
and self-documentation benefit (and lets a future backend conform
without inheriting from anything) without forcing a shared base class.

**Return-value contract** (not mechanically enforced by `Protocol`,
documented here instead): `build_tree` returns a list of dicts, each with
at least `"redshift"`, `"parent_mass"`, `"progenitors"` keys
(`ZhangHuiMergerTree`'s also carry `"smooth_accretion"`/`"merger_mass"` --
a superset, not a conflict). `build_forest_numpy` returns a 5-tuple
`(mass_history, split_events, z_steps, smooth_accretion, merger_mass)`,
identical shapes/semantics on both classes -- see either concrete class's
own docstring for the full field-by-field description and the
mass-conservation identity both satisfy.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class TreeAlgorithm(Protocol):
    """
    Structural interface satisfied by `PCHMergerTree` and
    `ZhangHuiMergerTree` (see `tests/test_tree_algorithm.py`).
    `@runtime_checkable` means `isinstance(obj, TreeAlgorithm)` works, but
    only checks that the named methods exist -- not their signatures or
    return shapes (a `Protocol` limitation); see the module docstring's
    "Return-value contract" for what `isinstance` alone doesn't verify.
    """

    def build_tree(self, M0, z0, z_max, M_res, dz=0.1): ...

    def build_forest_numpy(self, M0_array, z0, z_max, M_res, dz=0.1): ...
