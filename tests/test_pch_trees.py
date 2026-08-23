"""
Regression tests for PCHMergerTree's three interchangeable backends
(build_tree, build_forest_numpy, build_forest_numba).

Built on a synthetic power-law P(k) (see conftest.py) so these run without
CLASS/CAMB installed. The point is to pin down structural invariants that
must hold regardless of the underlying cosmology/DM model -- exactly the
kind of safety net the audit recommended before the WDM/FDM transfer-
function work (Section 7) touches this code.
"""

import numpy as np
import pytest

M0 = 1.0e12  # Msun/h
Z0 = 0.0
Z_MAX = 5.0
M_RES = 1.0e9  # Msun/h
DZ = 0.2


def test_forest_numpy_shape_and_bounds(tree_generator):
    np.random.seed(0)
    N = 200
    mass_history, split_events, z_steps, smooth_accretion, merger_mass = tree_generator.build_forest_numpy(
        M0_array=np.full(N, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    n_steps = int((Z_MAX - Z0) / DZ)
    assert mass_history.shape == (N, n_steps)
    assert z_steps.shape == (n_steps + 1,)
    assert smooth_accretion.shape == (N, n_steps)
    assert merger_mass.shape == (N, n_steps)
    # main-progenitor mass can never exceed the initial mass, and dead trees
    # are represented as exactly zero
    assert np.all((mass_history <= M0 + 1e-6) & (mass_history >= 0.0))


def test_forest_numpy_mass_non_increasing(tree_generator):
    np.random.seed(1)
    N = 300
    mass_history, _, _, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(N, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    # main-progenitor mass only ever decreases (splits) or stays flat
    # (no split that step); once a tree hits zero (dropped below M_res)
    # it must stay at zero.
    deltas = np.diff(mass_history, axis=1)
    alive_next = mass_history[:, 1:] > 0
    assert np.all(deltas[alive_next] <= 1e-6)
    dead = mass_history[:, :-1] == 0
    assert np.all(mass_history[:, 1:][dead] == 0)


def test_forest_numpy_reproducible_with_seed(tree_generator):
    np.random.seed(7)
    mh_a, _, _, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(50, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    np.random.seed(7)
    mh_b, _, _, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(50, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    assert np.array_equal(mh_a, mh_b)


def test_forest_numba_shape_and_bounds(tree_generator):
    N = 200
    mass_history, z_steps, smooth_accretion, merger_mass = tree_generator.build_forest_numba(
        M0_array=np.full(N, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    n_steps = int((Z_MAX - Z0) / DZ)
    assert mass_history.shape == (N, n_steps)
    assert np.all((mass_history <= M0 + 1e-6) & (mass_history >= 0.0))
    assert smooth_accretion.shape == (N, n_steps)
    assert merger_mass.shape == (N, n_steps)


def test_build_forest_numba_raises_actionable_error_without_numba(tree_generator, monkeypatch):
    # numba is an optional dependency (pip install foraois[numba]) --
    # simulate it being absent (rather than actually uninstalling it,
    # which the rest of this test module needs) and check the resulting
    # error tells the user what to do instead of an AttributeError/
    # NameError from calling a None kernel.
    import foraois.pch_trees as pch_trees_module

    monkeypatch.setattr(pch_trees_module, "_HAVE_NUMBA", False)
    with pytest.raises(ImportError, match=r"foraois\[numba\]"):
        tree_generator.build_forest_numba(M0_array=np.full(10, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)


def test_numba_and_numpy_backends_statistically_consistent(tree_generator):
    # The two backends draw randoms in different orders (numpy: one batch of
    # N draws per step; numba: prange over trees, order not fixed), so exact
    # equality isn't expected even with the same seed. Check instead that
    # the two backends agree on the *distribution* of final masses -- i.e.
    # neither backend has a bug that systematically biases the physics.
    N = 20000
    np.random.seed(3)
    mh_np, _, _, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(N, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    mh_nb, _, _, _ = tree_generator.build_forest_numba(M0_array=np.full(N, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)

    frac_alive_np = np.mean(mh_np[:, -1] > 0)
    frac_alive_nb = np.mean(mh_nb[:, -1] > 0)
    assert frac_alive_np == pytest.approx(frac_alive_nb, abs=0.03)

    mean_mass_np = mh_np[:, -1][mh_np[:, -1] > 0].mean()
    mean_mass_nb = mh_nb[:, -1][mh_nb[:, -1] > 0].mean()
    assert mean_mass_np == pytest.approx(mean_mass_nb, rel=0.1)


def test_build_tree_single_tree_structure(tree_generator):
    np.random.seed(5)
    tree = tree_generator.build_tree(M0=M0, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    for entry in tree:
        assert entry["parent_mass"] <= M0 + 1e-6
        for progenitor in entry["progenitors"]:
            assert 0 < progenitor <= entry["parent_mass"]


def test_build_full_tree_structure(tree_generator):
    # build_full_tree grows every branch (not just the main progenitor),
    # recording a node only at real events (splits or a branch's endpoint)
    # -- see plot.plot_dendrogram, which consumes this directly.
    np.random.seed(13)
    nodes = tree_generator.build_full_tree(M0=M0, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    by_id = {n["id"]: n for n in nodes}

    roots = [n for n in nodes if n["descendant_id"] is None]
    assert len(roots) == 1
    root = roots[0]
    assert root["mass"] == pytest.approx(M0)
    assert root["redshift"] == pytest.approx(Z0)
    assert root["is_main"] is True

    for n in nodes:
        assert n["mass"] > 0
        assert n["mass"] <= M0 + 1e-6
        if n["descendant_id"] is not None:
            # ids are assigned in increasing order as the tree is grown, so
            # a node's descendant always has a strictly smaller id -- this
            # also guarantees the graph is acyclic
            assert n["descendant_id"] < n["id"]
            descendant = by_id[n["descendant_id"]]
            # mass only decreases moving away from the root (towards
            # higher redshift / earlier times)
            assert n["mass"] <= descendant["mass"] + 1e-6
            assert n["redshift"] > descendant["redshift"]

    # following is_main from the root traces a single well-defined chain
    children = {}
    for n in nodes:
        children.setdefault(n["descendant_id"], []).append(n)
    current = root
    visited = {current["id"]}
    while True:
        main_kids = [k for k in children.get(current["id"], []) if k["is_main"]]
        assert len(main_kids) <= 1
        if not main_kids:
            break
        current = main_kids[0]
        assert current["id"] not in visited  # no cycles
        visited.add(current["id"])


def test_build_full_tree_max_nodes_guard(tree_generator):
    np.random.seed(17)
    with pytest.raises(RuntimeError, match="max_nodes"):
        tree_generator.build_full_tree(
            M0=M0,
            z0=Z0,
            z_max=Z_MAX,
            M_res=M_RES,
            dz=DZ,
            max_nodes=1,
        )


def test_build_forest_numpy_extends_delta_col_table_beyond_z15(tree_generator):
    # Regression test for a real bug: CosmoData.__init__ builds the
    # delta_col(z) table with its own default z_max=15 (see
    # precompute_delta_col_table), independent of whatever z_max a caller
    # later requests here. delta_col_at_z() looks the table up via
    # np.interp, which silently *clamps* beyond the table's range rather
    # than erroring -- so d_omega = delta_col(z1)-delta_col(z0) went
    # identically 0 for any step entirely beyond z=15, freezing every
    # tree's mass there with no warning. PCHMergerTree.build_forest_numpy
    # must extend the table (via _ensure_delta_col_covers) before
    # requesting z_max > 15.
    z_max = 18.0
    assert tree_generator.cosmo_data._dc_z_grid[-1] < z_max  # precondition: default table doesn't already cover this

    tree_generator.build_forest_numpy(
        M0_array=np.full(5, M0),
        z0=Z0,
        z_max=z_max,
        M_res=M_RES,
        dz=0.5,
    )  # unpacking not needed -- called only for its side effect on the delta_col table

    assert tree_generator.cosmo_data._dc_z_grid[-1] >= z_max
    # the direct symptom: delta_col actually keeps growing past the old
    # table edge, rather than being flat/clamped there
    assert tree_generator.cosmo_data.delta_col_at_z(z_max) != tree_generator.cosmo_data.delta_col_at_z(15.0)


def test_build_forest_numba_extends_delta_col_table_beyond_z15(tree_generator):
    z_max = 18.0
    tree_generator.build_forest_numba(
        M0_array=np.full(5, M0),
        z0=Z0,
        z_max=z_max,
        M_res=M_RES,
        dz=0.5,
    )
    assert tree_generator.cosmo_data._dc_z_grid[-1] >= z_max
    assert tree_generator.cosmo_data.delta_col_at_z(z_max) != tree_generator.cosmo_data.delta_col_at_z(15.0)


def test_build_forest_numpy_mass_growth_not_frozen_beyond_z15(tree_generator):
    # The actual observable symptom of the bug this guards against: without
    # the table extension, every tree's mass would freeze at an identical
    # value/step the instant both z0 and z1 of a step exceed 15 (since
    # Nupper and F both go identically 0 there) -- here, with a low-mass
    # M0 close to M_res so trees are forced deep enough to actually probe
    # z > 15, different trees' random walks must diverge instead of all
    # freezing in lockstep.
    np.random.seed(3)
    mass_history, _, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(20, 10.0 * M_RES),
        z0=Z0,
        z_max=18.0,
        M_res=M_RES,
        dz=0.2,
    )
    # steps entirely beyond z=15 (the old default table edge)
    beyond_15 = np.where(z_steps[1:] > 15.0)[0]
    assert len(beyond_15) > 0
    # not every tree can be frozen (unchanged from the previous step) at
    # every one of these steps if the table extension worked
    changed = np.diff(mass_history[:, beyond_15[0] - 1 :], axis=1) != 0
    assert changed.any()


def test_split_events_mass_conservation_direction(tree_generator):
    # For every recorded split, M1 (the retained largest progenitor) must be
    # >= M2 by construction (mass_history takes max(M1, M2)), and both must
    # be positive and no larger than the pre-split mass isn't directly
    # available here, but must be < M0 (can't grow via a split).
    np.random.seed(11)
    _, split_events, _, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(500, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    assert len(split_events) > 0, "expected at least one split in 500 trees over 5 steps"
    for event in split_events:
        assert np.all(event["M1"] > 0)
        assert np.all(event["M2"] >= 0)
        assert np.all(event["M1"] < M0)


def test_smooth_accretion_and_merger_mass_conserve_mass_numpy(tree_generator):
    # The core invariant smooth_accretion/merger_mass must satisfy: summed
    # together, chronologically, they account for exactly the mass gained
    # between consecutive *resolved* steps (see build_forest_numpy's
    # docstring). mass_history/smooth_accretion/merger_mass are all stored
    # in foraois' native backward (high-z-increasing) order, so index j+1
    # is chronologically *earlier* than index j; the mass gained going
    # forward in time (j+1 -> j) is mass_history[j] - mass_history[j+1].
    np.random.seed(23)
    N = 2000
    mass_history, _, _, smooth_accretion, merger_mass = tree_generator.build_forest_numpy(
        M0_array=np.full(N, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    # M0 itself is the (implicit) value before the first recorded step
    full_history = np.concatenate([np.full((N, 1), M0), mass_history], axis=1)

    forward_gain = full_history[:, :-1] - full_history[:, 1:]
    channel_sum = smooth_accretion + merger_mass

    # excludes exactly the resolution-loss transition step (both sides must
    # be resolved -- see docstring), per build_forest_numpy's documented
    # exception.
    both_resolved = (full_history[:, 1:] > 0) & (full_history[:, :-1] > 0)
    assert both_resolved.sum() > 0, "expected at least one resolved step in this sample"
    assert np.allclose(forward_gain[both_resolved], channel_sum[both_resolved], rtol=1e-8, atol=1e-6)

    # sanity: both channels are non-negative (pure mass gain, never a sink)
    assert np.all(smooth_accretion >= 0.0)
    assert np.all(merger_mass >= 0.0)
    # merger_mass is genuinely sparse (event-like), not a dense rate
    assert 0.0 < np.mean(merger_mass > 0) < 0.5


def test_smooth_accretion_and_merger_mass_conserve_mass_numba(tree_generator):
    # Same identity as the numpy-backend test above, but the numba kernel's
    # docstring claims *no* exception at the resolution-loss step (it never
    # zeroes mass_history the way build_forest_numpy does) -- checked here
    # directly rather than assumed from reading the code.
    np.random.seed(29)
    N = 2000
    mass_history, _, smooth_accretion, merger_mass = tree_generator.build_forest_numba(
        M0_array=np.full(N, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    full_history = np.concatenate([np.full((N, 1), M0), mass_history], axis=1)

    forward_gain = full_history[:, :-1] - full_history[:, 1:]
    channel_sum = smooth_accretion + merger_mass

    # No masking/exception here -- the identity is claimed to hold at every
    # single step for this backend.
    assert np.allclose(forward_gain, channel_sum, rtol=1e-8, atol=1e-6)
    assert np.all(smooth_accretion >= 0.0)
    assert np.all(merger_mass >= 0.0)
