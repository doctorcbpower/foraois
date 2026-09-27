"""
Tests for foraois.zhang_hui_trees -- the single-progenitor Zhang & Hui
sampler.
"""

import warnings

import numpy as np
import pytest

from foraois.first_crossing import flat_barrier_first_crossing
from foraois.zhang_hui_trees import (
    ZhangHuiMergerTree,
    draw_progenitor_mass_zh,
    first_crossing_step,
)

# This file's tests deliberately use fast, coarse N_grid values (60-400) to
# check structural/self-consistency invariants (bounds, mass conservation,
# tree shape) cheaply -- not first_crossing_step's absolute p_res/F_zh
# accuracy, which its own grid-resolution warning correctly flags at these
# settings. That accuracy is validated separately, at a much higher
# N_grid, in tests/test_zhang_hui_validation.py.
pytestmark = pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")

# Shared constants for the ZhangHuiMergerTree.build_tree tests.
BT_M0 = 1.0e12  # Msun/h
BT_Z0 = 0.0
BT_Z_MAX = 2.0
BT_M_RES = 1.0e10  # Msun/h
BT_DZ = 0.2
BT_N_GRID = 60  # see test_progenitor_masses_are_resolved_and_bounded


# ---------------------------------------------------------------------------
# first_crossing_step
# ---------------------------------------------------------------------------


def test_cdm_step_reduces_to_flat_barrier_solution(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    M0, z0, z1, M_res = 1e12, 0.0, 0.5, 1e10

    step = first_crossing_step(M0, z0, z1, M_res, cosmo_data, model="cdm")

    d_omega = cosmo_data.delta_col_at_z(z1) - cosmo_data.delta_col_at_z(z0)
    expected = flat_barrier_first_crossing(step["S_grid"], d_omega)

    # Exclude S=0 (both are exactly 0 there by construction) and compare
    # the shape of the two curves; solve_first_crossing is a numerical
    # scheme so exact-to-machine-precision agreement isn't expected, only
    # that it reproduces the closed-form CDM answer to solver accuracy.
    assert np.allclose(step["f"][1:], expected[1:], rtol=5e-3, atol=1e-8)


def test_p_res_and_F_zh_are_valid_probabilities(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    step = first_crossing_step(1e12, 0.0, 0.5, 1e10, cosmo_data, model="cdm")

    assert 0.0 <= step["p_res"] <= 1.0
    assert 0.0 <= step["F_zh"] <= 1.0
    assert step["p_res"] + step["F_zh"] <= 1.0 + 1e-8


def test_raises_when_M_res_not_below_M0(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    with pytest.raises(ValueError):
        first_crossing_step(1e10, 0.0, 0.5, 1e10, cosmo_data, model="cdm")
    with pytest.raises(ValueError):
        first_crossing_step(1e10, 0.0, 0.5, 1e11, cosmo_data, model="cdm")


# ---------------------------------------------------------------------------
# draw_progenitor_mass_zh
# ---------------------------------------------------------------------------


def test_returns_empty_when_M_res_at_or_above_M0(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    assert draw_progenitor_mass_zh(1e10, 0.0, 0.5, 1e10, cosmo_data) == []
    assert draw_progenitor_mass_zh(1e10, 0.0, 0.5, 2e10, cosmo_data) == []


def test_progenitor_masses_are_resolved_and_bounded(zh_tree_generator):
    # N_grid deliberately small here (solve_first_crossing is O(N_grid^2),
    # and this test calls it dozens of times): correctness of bounds/budget
    # doesn't need production-accuracy resolution, only the single-solve
    # precision test above does.
    cosmo_data = zh_tree_generator.cosmo_data
    M0, M_res = 1e12, 1e10
    rng = np.random.default_rng(0)

    n_resolved_splits = 0
    for _ in range(40):
        progenitors = draw_progenitor_mass_zh(M0, 0.0, 0.5, M_res, cosmo_data, model="cdm", rng=rng, N_grid=60)
        assert all(M_res <= m <= M0 for m in progenitors)
        if len(progenitors) == 2:
            n_resolved_splits += 1

    # Sanity check the branch that actually exercises the inverse-CDF mass
    # draw got hit at least once over the trials, not just the "nothing
    # resolved this step" branch every time.
    assert n_resolved_splits > 0


def test_resolved_split_never_leaves_a_sub_resolution_complement(zh_tree_generator):
    # Regression test for N23's own footnote 3: their progenitor mass M'
    # is drawn from [M_res, M-M_res], keeping *both* M' and M-M' >= M_res
    # by construction.
    cosmo_data = zh_tree_generator.cosmo_data
    M0, M_res = 1e12, 1e10
    rng = np.random.default_rng(2)

    n_resolved_splits = 0
    for _ in range(60):
        progenitors = draw_progenitor_mass_zh(M0, 0.0, 0.5, M_res, cosmo_data, model="cdm", rng=rng, N_grid=60)
        if len(progenitors) == 2:
            n_resolved_splits += 1
            assert min(progenitors) >= M_res

    assert n_resolved_splits > 0


def test_mass_budget_conserved(zh_tree_generator):
    """
    sum(progenitors) + F_zh*M0 == M0, mirroring
    pch_trees.draw_progenitor_masses's own mass-conservation invariant
    (M0 = M1 + M2 + F*M0) -- F_zh is deterministic given (M0,z0,z1,M_res),
    computed independently via first_crossing_step for the check.
    """
    cosmo_data = zh_tree_generator.cosmo_data
    M0, z0, z1, M_res = 1e12, 0.0, 0.5, 1e10
    N_grid = 60  # see test_progenitor_masses_are_resolved_and_bounded

    F_zh = first_crossing_step(M0, z0, z1, M_res, cosmo_data, model="cdm", N_grid=N_grid)["F_zh"]

    rng = np.random.default_rng(1)
    for _ in range(20):
        progenitors = draw_progenitor_mass_zh(M0, z0, z1, M_res, cosmo_data, model="cdm", rng=rng, N_grid=N_grid)
        # progenitors below M_res are dropped from the returned list (same
        # as pch_trees), so only check the budget when both survive it.
        if len(progenitors) == 2:
            assert np.isclose(sum(progenitors) + F_zh * M0, M0, rtol=1e-6)


# ---------------------------------------------------------------------------
# ZhangHuiMergerTree.build_tree
# ---------------------------------------------------------------------------


def test_build_tree_structure(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(5), N_grid=BT_N_GRID)
    tree = zh_tree.build_tree(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)

    assert len(tree) > 0  # sanity: at least one split happened over this range
    for entry in tree:
        assert entry["parent_mass"] <= BT_M0 + 1e-6
        for progenitor in entry["progenitors"]:
            assert 0 < progenitor <= entry["parent_mass"]


def test_build_tree_mass_non_increasing(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(6), N_grid=BT_N_GRID)
    tree = zh_tree.build_tree(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)

    masses = [BT_M0] + [max(entry["progenitors"]) for entry in tree]
    assert all(masses[i + 1] <= masses[i] + 1e-6 for i in range(len(masses) - 1))


def test_build_tree_stops_below_resolution(zh_tree_generator):
    # A tiny M0 close to M_res, over a long baseline, should terminate the
    # loop (via the M_cur < M_res break) rather than running every step.
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(9), N_grid=BT_N_GRID)
    tree = zh_tree.build_tree(M0=1.5e10, z0=BT_Z0, z_max=BT_Z_MAX, M_res=1e10, dz=BT_DZ)
    for entry in tree:
        for progenitor in entry["progenitors"]:
            assert progenitor >= 1e10


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


def test_build_tree_reproducible_with_same_rng(zh_tree_generator):
    # Same pattern test_pch_trees.py uses for PCHMergerTree's numpy backend
    # (test_forest_numpy_reproducible_with_seed), applied to
    # ZhangHuiMergerTree.build_tree -- its rng is an explicit constructor
    # argument (np.random.Generator) rather than global np.random state, so
    # "same seed" here means constructing two generators with the same seed.
    cosmo_data = zh_tree_generator.cosmo_data
    zh_a = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(123), N_grid=BT_N_GRID)
    zh_b = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(123), N_grid=BT_N_GRID)
    tree_a = zh_a.build_tree(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    tree_b = zh_b.build_tree(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    assert _trees_equal(tree_a, tree_b)


def test_build_tree_differs_with_different_rng(zh_tree_generator):
    # Independent seeds should be independently valid (both still satisfy
    # every structural invariant test_build_tree_structure/_mass_non_
    # increasing check) but generally produce a different realization --
    # this is the complementary check to same-rng reproducibility above,
    # guarding against e.g. an rng argument silently not being threaded
    # through to the actual sampling calls.
    cosmo_data = zh_tree_generator.cosmo_data
    zh_a = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(1), N_grid=BT_N_GRID)
    zh_b = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(2), N_grid=BT_N_GRID)
    tree_a = zh_a.build_tree(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    tree_b = zh_b.build_tree(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    assert not _trees_equal(tree_a, tree_b)


def test_ensure_delta_col_covers_extends_table_when_needed(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm")

    original_max = cosmo_data._dc_z_grid[-1]
    zh_tree._ensure_delta_col_covers(original_max + 5.0)
    assert cosmo_data._dc_z_grid[-1] >= original_max + 5.0


def test_build_tree_smooth_accretion_and_merger_mass_per_step(zh_tree_generator):
    # Per-step identity: parent_mass == max(progenitors) + smooth_accretion
    # + merger_mass, by construction (see build_tree's docstring) -- this
    # pins that construction down as a regression check, and confirms
    # neither channel goes negative for a range of seeds (folding a
    # sub-M_res leftover into smooth_accretion could in principle drive it
    # negative -- same check build_forest_numpy's own test makes).
    cosmo_data = zh_tree_generator.cosmo_data
    for seed in range(10):
        zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(seed), N_grid=BT_N_GRID)
        tree = zh_tree.build_tree(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
        for entry in tree:
            assert entry["smooth_accretion"] >= -1e-6
            assert entry["merger_mass"] >= 0.0
            reconstructed = max(entry["progenitors"]) + entry["smooth_accretion"] + entry["merger_mass"]
            assert np.isclose(reconstructed, entry["parent_mass"], rtol=1e-8, atol=1e-6)


def test_build_tree_mass_conservation_end_to_end(zh_tree_generator):
    # Summed over every recorded step, smooth_accretion + merger_mass +
    # the final tracked mass should equal M0 exactly (a direct
    # consequence of the per-step identity above being telescoping) --
    # the same conservation build_forest_numpy's own test checks, just
    # for the serial single-halo path.
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(21), N_grid=BT_N_GRID)
    tree = zh_tree.build_tree(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    assert len(tree) > 0

    total_lost = sum(e["smooth_accretion"] + e["merger_mass"] for e in tree)
    final_mass = max(tree[-1]["progenitors"])
    assert np.isclose(total_lost + final_mass, BT_M0, rtol=1e-8, atol=1e-6)


# ---------------------------------------------------------------------------
# ZhangHuiMergerTree.build_forest_numpy
# ---------------------------------------------------------------------------


def test_build_forest_numpy_shape_and_bounds(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(10))
    N = 500
    mass_history, split_events, z_steps, smooth_accretion, merger_mass = zh_tree.build_forest_numpy(
        M0_array=np.full(N, BT_M0),
        z0=BT_Z0,
        z_max=BT_Z_MAX,
        M_res=BT_M_RES,
        dz=BT_DZ,
    )
    n_steps = int((BT_Z_MAX - BT_Z0) / BT_DZ)
    assert mass_history.shape == (N, n_steps)
    assert z_steps.shape == (n_steps + 1,)
    assert smooth_accretion.shape == (N, n_steps)
    assert merger_mass.shape == (N, n_steps)
    assert np.all((mass_history <= BT_M0 + 1e-6) & (mass_history >= 0.0))
    assert np.all(smooth_accretion >= -1e-6)
    assert np.all(merger_mass >= 0.0)


def test_build_forest_numpy_mass_non_increasing(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(11))
    N = 500
    mass_history, _, _, _, _ = zh_tree.build_forest_numpy(
        M0_array=np.full(N, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ
    )
    deltas = np.diff(mass_history, axis=1)
    alive_next = mass_history[:, 1:] > 0
    assert np.all(deltas[alive_next] <= 1e-6)
    dead = mass_history[:, :-1] == 0
    assert np.all(mass_history[:, 1:][dead] == 0)


def test_build_forest_numpy_mass_conservation(zh_tree_generator):
    # Same identity pch_trees's own
    # test_smooth_accretion_and_merger_mass_conserve_mass_numpy checks:
    # forward_gain == smooth_accretion + merger_mass while a halo is
    # resolved on both sides of a step. See _build_forest_flat_barrier_
    # numpy's docstring for why smooth_accretion is defined as the
    # conservation residual (not simply (1-p_res)*M_safe).
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(12))
    N = 2000
    mass_history, _, _, smooth_accretion, merger_mass = zh_tree.build_forest_numpy(
        M0_array=np.full(N, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ
    )
    full_history = np.concatenate([np.full((N, 1), BT_M0), mass_history], axis=1)
    forward_gain = full_history[:, :-1] - full_history[:, 1:]
    channel_sum = smooth_accretion + merger_mass

    both_resolved = (full_history[:, 1:] > 0) & (full_history[:, :-1] > 0)
    assert both_resolved.sum() > 0
    assert np.allclose(forward_gain[both_resolved], channel_sum[both_resolved], rtol=1e-8, atol=1e-6)


def test_build_forest_numpy_split_events_never_leave_a_sub_resolution_complement(
    zh_tree_generator,
):
    # Regression test for the mass-budget fix (see
    # test_resolved_split_never_leaves_a_sub_resolution_complement's own
    # docstring for the literature citation) -- every recorded split
    # event's M1 and M2 should both be >= M_res now, not just M2.
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(3))
    N = 5000
    _, split_events, _, _, _ = zh_tree.build_forest_numpy(
        M0_array=np.full(N, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ
    )
    assert len(split_events) > 0
    for event in split_events:
        assert np.all(event["M1"] >= BT_M_RES)
        assert np.all(event["M2"] >= BT_M_RES)


def test_build_forest_numpy_reproducible_with_same_rng(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    zh_a = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(7))
    zh_b = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(7))
    mh_a, *_ = zh_a.build_forest_numpy(np.full(200, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    mh_b, *_ = zh_b.build_forest_numpy(np.full(200, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    assert np.array_equal(mh_a, mh_b)


# ---------------------------------------------------------------------------
# ZhangHuiMergerTree.build_forest_numba
# ---------------------------------------------------------------------------


def test_build_forest_numba_shape_and_bounds(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm")
    N = 500
    mass_history, z_steps, smooth_accretion, merger_mass = zh_tree.build_forest_numba(
        M0_array=np.full(N, BT_M0),
        z0=BT_Z0,
        z_max=BT_Z_MAX,
        M_res=BT_M_RES,
        dz=BT_DZ,
    )
    n_steps = int((BT_Z_MAX - BT_Z0) / BT_DZ)
    assert mass_history.shape == (N, n_steps)
    assert z_steps.shape == (n_steps + 1,)
    assert smooth_accretion.shape == (N, n_steps)
    assert merger_mass.shape == (N, n_steps)
    assert np.all((mass_history <= BT_M0 + 1e-6) & (mass_history >= 0.0))
    assert np.all(smooth_accretion >= -1e-6)
    assert np.all(merger_mass >= 0.0)


def test_build_forest_numba_mass_conservation(zh_tree_generator):
    # Same identity test_build_forest_numpy_mass_conservation checks for
    # the numpy backend -- both implement the same closed-form algorithm
    # (_build_forest_flat_barrier_numpy/_kernel), just vectorised
    # differently.
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm")
    N = 2000
    mass_history, _, smooth_accretion, merger_mass = zh_tree.build_forest_numba(
        M0_array=np.full(N, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ
    )
    full_history = np.concatenate([np.full((N, 1), BT_M0), mass_history], axis=1)
    forward_gain = full_history[:, :-1] - full_history[:, 1:]
    channel_sum = smooth_accretion + merger_mass

    both_resolved = (full_history[:, 1:] > 0) & (full_history[:, :-1] > 0)
    assert both_resolved.sum() > 0
    assert np.allclose(forward_gain[both_resolved], channel_sum[both_resolved], rtol=1e-6, atol=1e-6)


def test_build_forest_numba_not_reproducible_even_with_matched_seed(zh_tree_generator):
    # Regression/documentation test for a genuine numba limitation found
    # while adding this backend: parallel=True/nb.prange gives each worker
    # thread its own internal random stream that is *not* deterministically
    # tied to np.random.seed(), whether called before this method or (also
    # checked directly, separately) as the very first statement inside the
    # jitted kernel itself. See build_forest_numba's own docstring. This
    # test exists to catch it if a future numba version (or a refactor
    # away from prange) changes that -- not to assert the limitation is
    # desirable.
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm")

    np.random.seed(7)
    mh_a, *_ = zh_tree.build_forest_numba(np.full(200, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    np.random.seed(7)
    mh_b, *_ = zh_tree.build_forest_numba(np.full(200, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    assert not np.array_equal(mh_a, mh_b)


def test_build_forest_numba_raises_actionable_error_without_numba(zh_tree_generator, monkeypatch):
    # numba is an optional dependency (pip install foraois[numba]) --
    # simulate it being absent (rather than actually uninstalling it,
    # which the rest of this test module needs) and check the resulting
    # error tells the user what to do, matching pch_trees.py's own
    # equivalent test.
    import foraois.zhang_hui_trees as zh_module

    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm")
    monkeypatch.setattr(zh_module, "_HAVE_NUMBA", False)
    with pytest.raises(ImportError, match=r"foraois\[numba\]"):
        zh_tree.build_forest_numba(np.full(10, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)


def test_numba_and_numpy_backends_statistically_consistent(zh_tree_generator):
    # Same rationale as pch_trees's own equivalent test: the two backends
    # draw randoms in different orders (numpy: one batch of N draws per
    # step; numba: prange over trees, order not fixed), so exact equality
    # isn't expected even with matched seeding -- check instead that they
    # agree on the *distribution* of final masses.
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="cdm", rng=np.random.default_rng(3))
    N = 20000

    mh_np, *_ = zh_tree.build_forest_numpy(np.full(N, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    np.random.seed(3)
    mh_nb, *_ = zh_tree.build_forest_numba(np.full(N, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)

    frac_alive_np = np.mean(mh_np[:, -1] > 0)
    frac_alive_nb = np.mean(mh_nb[:, -1] > 0)
    assert frac_alive_np == pytest.approx(frac_alive_nb, abs=0.03)

    mean_mass_np = mh_np[:, -1][mh_np[:, -1] > 0].mean()
    mean_mass_nb = mh_nb[:, -1][mh_nb[:, -1] > 0].mean()
    assert mean_mass_np == pytest.approx(mean_mass_nb, rel=0.1)


def test_erfcinv_numba_matches_scipy():
    # scipy.special.erfcinv isn't callable from inside @njit code -- the
    # numba kernel implements its own (Winitzki approximation + Newton-
    # Raphson refinement against math.erf, see the module comment above
    # _erfcinv_numba). Checked directly against scipy's implementation
    # here. Two tolerance tiers, reflecting the approximation's actual
    # measured accuracy profile: tight (rel=1e-3, actually ~1e-6 over most
    # of the range and degrading only as v approaches the 1e-15 edge of
    # this band) for any v not absurdly close to 0/2 -- the only regime
    # real cosmological (d_omega, S) values ever produce -- and loose
    # (~1%) for the deep tail (v below ~1e-15), included only to confirm
    # it stays finite and correctly signed rather than crashing or
    # blowing up, not because that tail is ever physically reached.
    from scipy.special import erfcinv as erfcinv_scipy

    from foraois.zhang_hui_trees import _erfcinv_numba

    realistic_vs = np.concatenate(
        [np.logspace(-15, -1, 60), np.linspace(0.11, 1.89, 100), 2.0 - np.logspace(-15, -1, 60)]
    )
    for v in realistic_vs:
        exact = erfcinv_scipy(v)
        approx = _erfcinv_numba(v)
        assert approx == pytest.approx(exact, rel=1e-3, abs=1e-6)

    deep_tail_vs = np.logspace(-300, -16, 60)
    for v in deep_tail_vs:
        exact = erfcinv_scipy(v)
        approx = _erfcinv_numba(v)
        if np.isinf(exact):
            assert np.isinf(approx) and np.sign(approx) == np.sign(exact)
            continue
        assert np.isfinite(approx)
        assert approx == pytest.approx(exact, rel=1e-2)

    assert _erfcinv_numba(0.0) == np.inf
    assert _erfcinv_numba(2.0) == -np.inf


# ---------------------------------------------------------------------------
# collapse barrier: independent of the model= argument, and of the DM model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["wdm", "fdm", "sidm", "anything"])
def test_model_argument_does_not_select_the_barrier(zh_tree_generator, model):
    # model= is retained for backwards compatibility only. It must neither change the barrier nor trigger the FDM
    # placeholder warning or the SIDM error, which now belong to the dark-matter (CosmoData) layer.
    cosmo_data = zh_tree_generator.cosmo_data
    reference = first_crossing_step(1e12, 0.0, 0.5, 1e10, cosmo_data, model="cdm", N_grid=60)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        step = first_crossing_step(1e12, 0.0, 0.5, 1e10, cosmo_data, model=model, N_grid=60)
    assert not [w for w in caught if "placeholder" in str(w.message)]
    assert step["p_res"] == reference["p_res"]
    assert step["F_zh"] == reference["F_zh"]


# ---------------------------------------------------------------------------
# scale-dependent (linear) barrier: supported only by the general serial path
# ---------------------------------------------------------------------------

LB_M0, LB_M_RES = 1.0e12, 1.0e11  # Msun/h; M_res/M0 = 0.1 keeps a serial tree cheap


def test_serial_tree_with_linear_barrier_has_consistent_bookkeeping(zh_tree_generator, set_barrier):
    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear")  # default beta = 0.15
    zh_tree = ZhangHuiMergerTree(cosmo_data, rng=np.random.default_rng(3), N_grid=400, S_max_factor=2.0)
    tree = zh_tree.build_tree(M0=LB_M0, z0=0.0, z_max=1.0, M_res=LB_M_RES, dz=0.5)

    assert len(tree) > 0
    parent = LB_M0
    for entry in tree:
        assert np.isclose(entry["parent_mass"], parent)
        assert all(LB_M_RES * (1 - 1e-9) <= m <= entry["parent_mass"] for m in entry["progenitors"])
        assert entry["merger_mass"] >= 0.0 and entry["smooth_accretion"] >= -1e-6
        main_plus_channels = max(entry["progenitors"]) + entry["smooth_accretion"] + entry["merger_mass"]
        assert np.isclose(main_plus_channels, entry["parent_mass"], rtol=1e-8)
        parent = max(entry["progenitors"])


def test_beta_changes_the_resolved_crossing_probability(zh_tree_generator, set_barrier):
    # A deterministic quantity (no random tree realisation involved).
    cosmo_data = zh_tree_generator.cosmo_data
    p_res = {}
    for beta in (0.0, 0.15, 0.4):
        set_barrier(cosmo_data, "linear", beta=beta)
        p_res[beta] = first_crossing_step(LB_M0, 0.0, 2.0, LB_M_RES, cosmo_data, N_grid=200, S_max_factor=2.0)["p_res"]
    assert p_res[0.0] - p_res[0.15] > 0.05
    assert p_res[0.15] - p_res[0.4] > 0.05


def test_zero_beta_step_is_identical_to_the_fixed_barrier(zh_tree_generator, set_barrier):
    cosmo_data = zh_tree_generator.cosmo_data
    fixed = first_crossing_step(LB_M0, 0.0, 0.5, LB_M_RES, cosmo_data, N_grid=100)
    set_barrier(cosmo_data, "linear", beta=0.0)
    linear0 = first_crossing_step(LB_M0, 0.0, 0.5, LB_M_RES, cosmo_data, N_grid=100)
    assert np.array_equal(fixed["f"], linear0["f"])
    assert fixed["p_res"] == linear0["p_res"] and fixed["F_zh"] == linear0["F_zh"]


_FLAT_ONLY_ENTRY_POINTS = [
    ("build_forest_numpy", {}),
    ("build_forest_numba", {}),
    ("grow_full_population_numba", {"checkpoints": [0.4], "n_trees": 2}),
]


def _call_flat_only(zh_tree, name, extra):
    args = dict(M0=BT_M0, z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
    if name == "grow_full_population_numba":
        return getattr(zh_tree, name)(**args, **extra)
    return getattr(zh_tree, name)(np.full(10, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)


@pytest.mark.parametrize("name,extra", _FLAT_ONLY_ENTRY_POINTS)
def test_flat_barrier_backends_reject_a_scale_dependent_barrier(zh_tree_generator, set_barrier, name, extra):
    if name != "build_forest_numpy":
        pytest.importorskip("numba")
    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.15)
    zh_tree = ZhangHuiMergerTree(cosmo_data, rng=np.random.default_rng(0))
    with pytest.raises(NotImplementedError, match=r"fixed collapse barrier.*barrier='linear'.*0\.15"):
        _call_flat_only(zh_tree, name, extra)


def test_closed_form_scalar_sampler_rejects_a_scale_dependent_barrier(zh_tree_generator, set_barrier):
    from foraois.zhang_hui_trees import draw_progenitor_mass_zh_flat

    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.15)
    with pytest.raises(NotImplementedError, match="draw_progenitor_mass_zh_flat"):
        draw_progenitor_mass_zh_flat(1e12, 0.0, 0.5, 1e10, cosmo_data, rng=np.random.default_rng(0))


@pytest.mark.parametrize("name,extra", _FLAT_ONLY_ENTRY_POINTS)
def test_flat_barrier_backends_accept_zero_beta(zh_tree_generator, set_barrier, name, extra):
    if name != "build_forest_numpy":
        pytest.importorskip("numba")
    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.0)
    zh_tree = ZhangHuiMergerTree(cosmo_data, rng=np.random.default_rng(0))
    _call_flat_only(zh_tree, name, extra)  # must not raise


def test_zero_beta_forest_is_identical_to_the_fixed_barrier_forest(zh_tree_generator, set_barrier):
    # beta = 0 is the fixed barrier, so with the same seed the closed-form NumPy forest is bit-identical.
    cosmo_data = zh_tree_generator.cosmo_data
    fixed = ZhangHuiMergerTree(cosmo_data, rng=np.random.default_rng(11)).build_forest_numpy(
        np.full(200, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ
    )
    set_barrier(cosmo_data, "linear", beta=0.0)
    linear0 = ZhangHuiMergerTree(cosmo_data, rng=np.random.default_rng(11)).build_forest_numpy(
        np.full(200, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ
    )
    assert np.array_equal(fixed[0], linear0[0])
