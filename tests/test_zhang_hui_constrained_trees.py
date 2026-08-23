"""
Tests for foraois.zhang_hui_constrained_trees -- the constrained branch's
growth history, built on first_crossing_constrained.py's barrier-free
bridge-path simulator, and the full constrained tree grafting it onto
ordinary unconstrained continuation.
"""

import numpy as np
import pytest

from foraois.first_crossing_constrained import (
    simulate_bridge_path,
    solve_first_crossing_constrained,
)
from foraois.zhang_hui_constrained_trees import (
    _extract_successive_maxima,
    build_constrained_tree,
    constrained_branch_growth_history,
)
from foraois.zhang_hui_trees import ZhangHuiMergerTree

# build_constrained_tree's unconstrained continuation uses N_grid values
# here chosen for test speed, not production accuracy, which triggers
# zhang_hui_trees.py's own grid-resolution warning at these settings --
# same rationale/suppression as tests/test_zhang_hui_trees.py.
pytestmark = pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")


# ---------------------------------------------------------------------------
# _extract_successive_maxima -- pure array logic, no cosmology needed
# ---------------------------------------------------------------------------


def test_successive_maxima_extraction():
    S_grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    delta_path = np.array([0.0, 0.5, 0.3, 0.8, 0.6, 0.8])  # record at 0,1,3 (5 ties 3's value)
    S_max, delta_max = _extract_successive_maxima(S_grid, delta_path)
    assert list(S_max) == [0.0, 1.0, 3.0]
    assert list(delta_max) == [0.0, 0.5, 0.8]


# ---------------------------------------------------------------------------
# simulate_bridge_path validated against solve_first_crossing_constrained --
# the core sampling machinery constrained_branch_growth_history is built on.
# ---------------------------------------------------------------------------


def test_bridge_path_first_crossing_matches_solver():
    # For a flat barrier, delta1=B(S) for all S (by construction), so
    # "the first S at which the raw path reaches delta1" across many
    # simulated paths should match solve_first_crossing_constrained's own
    # f(S) -- an independent cross-check of simulate_bridge_path, sharing
    # no code with either the solver or its own MC cross-check (which uses
    # rejection sampling on unconstrained walks, not direct bridge
    # sampling).
    delta_c = 1.686
    S1 = 5.0
    delta1 = delta_c
    B = lambda S: np.full_like(np.asarray(S, dtype=float), delta_c)

    rng = np.random.default_rng(3)
    n_paths = 3000
    first_S = np.empty(n_paths)
    for i in range(n_paths):
        S_grid, delta_path = simulate_bridge_path(0.0, 0.0, S1, delta1, dS=0.02, rng=rng)
        crossed = np.where(delta_path >= delta1)[0]
        first_S[i] = S_grid[crossed[0]] if len(crossed) else S1

    S_mid, f = solve_first_crossing_constrained(B, 0.0, 0.0, S1, delta1, N=500)
    F_cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(S_mid))])

    for S_test in [1.0, 2.0, 3.0, 4.0]:
        cum_path = np.mean(first_S <= S_test)
        cum_solver = float(np.interp(S_test, S_mid, F_cum))
        assert cum_solver == pytest.approx(cum_path, abs=0.05)


# ---------------------------------------------------------------------------
# constrained_branch_growth_history
# ---------------------------------------------------------------------------


def test_growth_history_reaches_constraint_exactly(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    for seed in range(5):
        rng = np.random.default_rng(seed)
        history = constrained_branch_growth_history(
            1e14,
            0.0,
            1e12,
            8.0,
            1e10,
            cosmo_data,
            model="cdm",
            dS=0.05,
            rng=rng,
        )
        assert history[0] == {"redshift": 0.0, "mass": 1e14}
        assert history[-1]["redshift"] == pytest.approx(8.0)
        assert history[-1]["mass"] == pytest.approx(1e12, rel=1e-6)


def test_growth_history_monotonic(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    rng = np.random.default_rng(11)
    history = constrained_branch_growth_history(
        1e14,
        0.0,
        1e12,
        8.0,
        1e10,
        cosmo_data,
        model="cdm",
        dS=0.05,
        rng=rng,
    )
    redshifts = [h["redshift"] for h in history]
    masses = [h["mass"] for h in history]
    assert redshifts == sorted(redshifts)
    assert masses == sorted(masses, reverse=True)
    assert all(1e10 <= m <= 1e14 for m in masses)


def test_growth_history_stops_at_resolution_limit(zh_tree_generator):
    # M_res set well above M1 (but below M0) should truncate the history
    # before it reaches the full (M1, z1) constraint -- the branch stops
    # once a collapse event would drop below M_res.
    cosmo_data = zh_tree_generator.cosmo_data
    M_res = 5e13  # between M1=1e12 and M0=1e14
    rng = np.random.default_rng(13)
    history = constrained_branch_growth_history(
        1e14,
        0.0,
        1e12,
        8.0,
        M_res,
        cosmo_data,
        model="cdm",
        dS=0.05,
        rng=rng,
    )
    assert history[-1]["mass"] >= M_res
    assert history[-1]["mass"] > 1e12  # truncated well short of the full constraint
    assert history[-1]["redshift"] < 8.0


def test_growth_history_raises_on_invalid_inputs(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    with pytest.raises(ValueError):
        constrained_branch_growth_history(1e12, 0.0, 1e12, 8.0, 1e10, cosmo_data)
    with pytest.raises(ValueError):
        constrained_branch_growth_history(1e14, 8.0, 1e12, 0.0, 1e10, cosmo_data)


def test_growth_history_extends_delta_col_table_beyond_default_z_max(zh_tree_generator):
    # CosmoData.precompute_delta_col_table's own default only tabulates out
    # to z_max=15 (see cosmo_utils.py); _invert_z_for_delta extends it on
    # demand (the same guard PCHMergerTree/ZhangHuiMergerTree's own
    # _ensure_delta_col_covers uses -- see
    # test_pch_trees.py::test_build_forest_numba_extends_delta_col_table_beyond_z15
    # and test_zhang_hui_trees.py::test_ensure_delta_col_covers_extends_table_when_needed
    # for the equivalent checks on those two backends). Not previously
    # covered for the constrained-tree backend specifically.
    cosmo_data = zh_tree_generator.cosmo_data
    original_max = cosmo_data._dc_z_grid[-1]
    assert original_max < 20.0  # sanity: the default really doesn't already cover this

    z1 = 20.0
    rng = np.random.default_rng(17)
    history = constrained_branch_growth_history(
        1e14,
        0.0,
        1e12,
        z1,
        1e10,
        cosmo_data,
        model="cdm",
        dS=0.05,
        rng=rng,
    )
    # The table must actually have been extended to cover z1 ...
    assert cosmo_data._dc_z_grid[-1] >= z1
    # ... and the branch must have genuinely reached the constraint out
    # there, not silently frozen at the table's old boundary (the failure
    # mode ensure_delta_col_covers's own docstring describes: a clamped
    # delta_col(z) table makes d_omega identically 0 beyond it, freezing
    # mass/redshift progress rather than erroring).
    assert history[-1]["redshift"] == pytest.approx(z1)
    assert history[-1]["mass"] == pytest.approx(1e12, rel=1e-6)
    # At least one intermediate collapse event genuinely lies beyond the
    # table's original z_max=15, not just the final pinned endpoint.
    assert any(entry["redshift"] > original_max for entry in history[:-1])


# ---------------------------------------------------------------------------
# build_constrained_tree: grafts the constrained branch onto ordinary
# unconstrained continuation.
# ---------------------------------------------------------------------------


def test_build_constrained_tree_structure_and_transition(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    rng = np.random.default_rng(9)
    tree = build_constrained_tree(
        1e14,
        0.0,
        1e12,
        8.0,
        10.0,
        1e10,
        cosmo_data,
        model="cdm",
        dS=0.05,
        rng=rng,
        N_grid=60,
        dz=0.2,
    )
    assert len(tree) > 0

    redshifts = [e["redshift"] for e in tree]
    assert redshifts == sorted(redshifts)  # monotonic increasing

    for entry in tree:
        assert entry["parent_mass"] <= 1e14 + 1e-6
        for m in entry["progenitors"]:
            assert 0 < m <= entry["parent_mass"]

    # The transition point: some entry lands exactly at (z1, M1).
    at_constraint = [e for e in tree if e["redshift"] == pytest.approx(8.0)]
    assert len(at_constraint) == 1
    assert at_constraint[0]["progenitors"] == [pytest.approx(1e12, rel=1e-6)]


def _trees_equal(tree_a, tree_b):
    if len(tree_a) != len(tree_b):
        return False
    for a, b in zip(tree_a, tree_b, strict=True):
        if a.keys() != b.keys():
            return False
        for key in a:
            if key == "progenitors":
                if len(a[key]) != len(b[key]) or not np.allclose(a[key], b[key]):
                    return False
            elif a[key] != pytest.approx(b[key]):
                return False
    return True


def test_build_constrained_tree_reproducible_with_same_rng(zh_tree_generator):
    # Same pattern as test_pch_trees.py's PCHMergerTree reproducibility
    # check and test_zhang_hui_trees.py's build_tree one, applied to
    # build_constrained_tree -- both the constrained branch's own path
    # simulation (rng passed to simulate_bridge_path) and the unconstrained
    # continuation (rng passed through to a fresh ZhangHuiMergerTree) must
    # be reproducible from the same rng, not just one half of the tree.
    cosmo_data = zh_tree_generator.cosmo_data
    args = (1e14, 0.0, 1e12, 8.0, 10.0, 1e10, cosmo_data)
    kwargs = dict(model="cdm", dS=0.05, N_grid=60, dz=0.2)

    tree_a = build_constrained_tree(*args, rng=np.random.default_rng(9), **kwargs)
    tree_b = build_constrained_tree(*args, rng=np.random.default_rng(9), **kwargs)
    assert _trees_equal(tree_a, tree_b)


def test_build_constrained_tree_differs_with_different_rng(zh_tree_generator):
    # Independent seeds should be independently valid (both still satisfy
    # test_build_constrained_tree_structure_and_transition's own invariants
    # -- monotonic redshifts, exact (M1, z1) transition point) but
    # generally produce a different realization, both before and after z1.
    cosmo_data = zh_tree_generator.cosmo_data
    args = (1e14, 0.0, 1e12, 8.0, 10.0, 1e10, cosmo_data)
    kwargs = dict(model="cdm", dS=0.05, N_grid=60, dz=0.2)

    tree_a = build_constrained_tree(*args, rng=np.random.default_rng(1), **kwargs)
    tree_b = build_constrained_tree(*args, rng=np.random.default_rng(2), **kwargs)
    assert not _trees_equal(tree_a, tree_b)

    # Both realizations still hit the exact constraint point.
    for tree in (tree_a, tree_b):
        at_constraint = [e for e in tree if e["redshift"] == pytest.approx(8.0)]
        assert len(at_constraint) == 1
        assert at_constraint[0]["progenitors"] == [pytest.approx(1e12, rel=1e-6)]


def test_build_constrained_tree_schema_unified_before_and_after_z1(zh_tree_generator):
    # smooth_accretion/merger_mass are carried by every entry, both the
    # constrained portion (z <= z1) and the unconstrained continuation
    # (z > z1) -- see constrained_branch_growth_history's own docstring
    # for what smooth_accretion means on the constrained side: the
    # running-maximum trajectory's own mass loss between collapse events,
    # not a discrete merger -- merger_mass is honestly 0.0 there.
    cosmo_data = zh_tree_generator.cosmo_data
    rng = np.random.default_rng(9)
    z1 = 8.0
    tree = build_constrained_tree(
        1e14,
        0.0,
        1e12,
        z1,
        9.0,
        1e10,
        cosmo_data,
        model="cdm",
        dS=0.05,
        rng=rng,
        N_grid=60,
        dz=0.2,
    )
    assert len(tree) > 0
    for entry in tree:
        assert "smooth_accretion" in entry
        assert "merger_mass" in entry
        if entry["redshift"] <= z1:
            assert entry["merger_mass"] == 0.0


def test_build_constrained_tree_mass_conservation_end_to_end(zh_tree_generator):
    # sum(smooth_accretion + merger_mass) + final mass == M0, mirroring
    # ZhangHuiMergerTree.build_tree's own end-to-end conservation check --
    # now checkable across the whole constrained+unconstrained tree since
    # every entry carries both channels.
    cosmo_data = zh_tree_generator.cosmo_data
    rng = np.random.default_rng(9)
    M0 = 1e14
    tree = build_constrained_tree(
        M0,
        0.0,
        1e12,
        8.0,
        9.0,
        1e10,
        cosmo_data,
        model="cdm",
        dS=0.05,
        rng=rng,
        N_grid=60,
        dz=0.2,
    )
    total_lost = sum(e["smooth_accretion"] + e["merger_mass"] for e in tree)
    final_mass = max(tree[-1]["progenitors"])
    assert np.isclose(total_lost + final_mass, M0, rtol=1e-6, atol=1e-6)


def test_build_constrained_tree_no_continuation_when_z_max_equals_z1(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    rng = np.random.default_rng(9)
    tree = build_constrained_tree(
        1e14,
        0.0,
        1e12,
        8.0,
        8.0,
        1e10,
        cosmo_data,
        model="cdm",
        dS=0.05,
        rng=rng,
        N_grid=60,
        dz=0.2,
    )
    assert tree[-1]["redshift"] == pytest.approx(8.0)
    assert all("smooth_accretion" in e for e in tree)


def test_build_constrained_tree_raises_when_z_max_below_z1(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    with pytest.raises(ValueError):
        build_constrained_tree(1e14, 0.0, 1e12, 8.0, 5.0, 1e10, cosmo_data)


def test_build_constrained_tree_weak_constraint_roughly_matches_unconstrained(
    zh_tree_generator,
):
    # The full-scale version of this check lives in
    # scripts/validate_constrained_tree_convergence.py (150 trees on a
    # real cosmology) -- this is a fast, small-n regression
    # check, not a statistical characterization. A "typical" M1 (close to
    # what an unconstrained tree naturally reaches at z1 anyway) shouldn't
    # spuriously bias the branch's earlier growth history relative to
    # unconstrained trees at the same earlier redshift.
    cosmo_data = zh_tree_generator.cosmo_data
    M0, z0, z1, z_earlier, M_res, dz = 1e12, 0.0, 1.0, 0.5, 1e10, 0.2
    n_trees = 25

    def mass_at(tree, z_target):
        zs = [z0] + [e["redshift"] for e in tree]
        masses = [M0] + [max(e["progenitors"]) for e in tree]
        return float(np.interp(z_target, zs, masses))

    rng = np.random.default_rng(0)
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=rng, N_grid=60)

    at_z1 = np.empty(n_trees)
    at_earlier_unconstrained = np.empty(n_trees)
    for i in range(n_trees):
        tree = zh_tree.build_tree(M0, z0, z1, M_res, dz=dz)
        at_z1[i] = mass_at(tree, z1)
        at_earlier_unconstrained[i] = mass_at(tree, z_earlier)

    M1_typical = float(np.median(at_z1))

    rng2 = np.random.default_rng(1)
    at_earlier_constrained = np.empty(n_trees)
    for i in range(n_trees):
        tree = build_constrained_tree(
            M0,
            z0,
            M1_typical,
            z1,
            z1,
            M_res,
            cosmo_data,
            model="cdm",
            dS=0.05,
            rng=rng2,
            N_grid=60,
            dz=dz,
        )
        at_earlier_constrained[i] = mass_at(tree, z_earlier)

    ratio = at_earlier_constrained.mean() / at_earlier_unconstrained.mean()
    assert 0.7 < ratio < 1.3
