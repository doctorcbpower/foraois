"""
Tests for foraois.zhang_hui_trees -- the single-progenitor Zhang & Hui
sampler.
"""

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


def test_fdm_model_warns(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    with pytest.warns(UserWarning, match="placeholder"):
        draw_progenitor_mass_zh(
            1e12,
            0.0,
            0.5,
            1e10,
            cosmo_data,
            model="fdm",
            rng=np.random.default_rng(0),
            N_grid=60,
        )


def test_sidm_model_raises(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    with pytest.raises(NotImplementedError):
        draw_progenitor_mass_zh(
            1e12,
            0.0,
            0.5,
            1e10,
            cosmo_data,
            model="sidm",
            rng=np.random.default_rng(0),
        )


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


def test_build_forest_numpy_raises_for_sidm(zh_tree_generator):
    cosmo_data = zh_tree_generator.cosmo_data
    zh_tree = ZhangHuiMergerTree(cosmo_data, model="sidm", rng=np.random.default_rng(0))
    with pytest.raises(NotImplementedError):
        zh_tree.build_forest_numpy(np.full(10, BT_M0), z0=BT_Z0, z_max=BT_Z_MAX, M_res=BT_M_RES, dz=BT_DZ)
