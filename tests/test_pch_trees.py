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


def test_forest_numba_not_reproducible_even_with_matched_seed(tree_generator):
    # Documentation/regression test for a genuine numba limitation (see
    # build_forest_numba's own docstring): parallel=True/nb.prange gives
    # each worker thread its own internal random stream that is not
    # deterministically tied to np.random.seed(), unlike build_forest_numpy
    # above. This test exists to catch it if a future numba version (or a
    # refactor away from prange) changes that -- not to assert the
    # limitation is desirable.
    np.random.seed(7)
    mh_a, _, _, _ = tree_generator.build_forest_numba(M0_array=np.full(200, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    np.random.seed(7)
    mh_b, _, _, _ = tree_generator.build_forest_numba(M0_array=np.full(200, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    assert not np.array_equal(mh_a, mh_b)


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


def test_build_forest_numba_unresolved_accretion_matches_reference_constant(tree_generator):
    # Regression test for a real bug: _build_forest_kernel and
    # _draw_progenitors_scalar both used `1.0 / np.sqrt(2.0 * np.pi)`
    # (~0.3989) as F's (the unresolved-accretion mass-loss fraction)
    # prefactor instead of the correct `np.sqrt(2.0 / np.pi)` (~0.7979) that
    # _unresolved_accretion_fraction (the shared numpy/scalar machinery)
    # uses -- exactly half the correct value. This silently halved every
    # step's mass loss in BOTH numba kernels, compounding into up to a ~6x
    # branch-count excess with tree depth in the adaptive-stepping kernel
    # (see grow_full_population_numba_adaptive's docstring) -- yet
    # test_numba_and_numpy_backends_statistically_consistent above, with its
    # rel=0.1 tolerance (sized for RNG-order differences, not a tight
    # physics check), did not catch it in _build_forest_kernel.
    #
    # Isolate F from splitting entirely by using M_res just above M0/2, so
    # qres = M_res/M0 >= 0.5 for every halo at every step (no split can ever
    # produce two resolved fragments -- Nupper is exactly 0 by construction,
    # see _branching_rate_terms's own no_split_possible branch). Every
    # step's mass change is then pure, fully deterministic unresolved
    # accretion (no randomness involved at all), so build_forest_numba's
    # output can be checked directly against an analytically-compounded
    # reference F -- a machine-precision check, not a statistical one, that
    # this specific wrong-constant bug class cannot silently reappear in.
    from foraois.pch_trees import _branching_rate_terms, _unresolved_accretion_fraction

    pch = tree_generator
    M0_np = 1.0e12
    M_res_np = 0.6 * M0_np  # qres = 0.6 >= 0.5 for the whole run (mass only decreases)
    dz = 0.2
    z0, z_max = 0.0, 1.0

    mh_nb, z_steps, _, _ = pch.build_forest_numba(M0_array=np.full(5, M0_np), z0=z0, z_max=z_max, M_res=M_res_np, dz=dz)

    m = M0_np
    delta_prev = pch._delta_col_at_z(z_steps[0])
    for z_next in z_steps[1:]:
        delta_next = pch._delta_col_at_z(z_next)
        d_omega = delta_next - delta_prev
        terms = _branching_rate_terms(
            np.asarray(m), M_res_np, pch.cosmo_data.sigma_at_logmass, pch.cosmo_data.dlogsigma_at_logmass,
            delta_prev, d_omega, pch.G0, pch.gamma1, pch.gamma2,
        )
        assert float(terms["Nupper"]) == 0.0  # confirms qres>=0.5 held throughout, as designed
        F = float(
            _unresolved_accretion_fraction(m, terms, delta_prev, d_omega, pch.G0, pch.gamma2, pch._j_u_grid, pch._j_values)
        )
        m = m * (1.0 - F) if m * (1.0 - F) >= M_res_np else 0.0
        delta_prev = delta_next

    assert mh_nb[:, -1] == pytest.approx(m, rel=1e-10)


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


# ---------------------------------------------------------------------------
# collapse barrier: PCH08 reads delta_col(z) directly, so it supports only the fixed barrier
# ---------------------------------------------------------------------------


def test_pch08_rejects_a_scale_dependent_barrier(tree_generator, set_barrier):
    from foraois.pch_trees import PCHMergerTree

    cosmo_data = tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.15)
    with pytest.raises(NotImplementedError, match=r"PCHMergerTree.*fixed collapse barrier.*barrier='linear'"):
        PCHMergerTree(cosmo_data, {})


def test_pch08_accepts_linear_barrier_with_zero_beta(tree_generator, set_barrier):
    from foraois.pch_trees import PCHMergerTree

    cosmo_data = tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.0)
    PCHMergerTree(cosmo_data, {})  # beta = 0 is the fixed barrier PCH08 already implements


def test_pch08_accepts_the_duck_typed_cosmodata_of_the_fortran_cross_check(monkeypatch):
    # scripts/paper_figs/_fortran_compare.py builds PCHMergerTree on a stand-in that has none of CosmoData's config
    # (no run_params). Such an object carries no barrier request and must be accepted as the fixed barrier.
    from pathlib import Path

    from foraois.collapse import barrier_is_flat, delta_c

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts" / "paper_figs"))
    from _fortran_compare import FakeCosmoData, build_pch

    assert not hasattr(FakeCosmoData(), "run_params")
    pch = build_pch()  # raised AttributeError when the flat-barrier guard assumed run_params
    assert barrier_is_flat(pch.cosmo_data)
    assert delta_c(1e12, 0.5, pch.cosmo_data) == pch.cosmo_data.delta_col_at_z(0.5)
    assert np.isfinite(pch._sigma_at_mass(1e13))

