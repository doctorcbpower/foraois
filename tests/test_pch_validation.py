"""
Validation battery for PCHMergerTree's Appendix A branching-rate/rejection-
sampling machinery (Parkinson, Cole & Helly 2008).

This is deliberately independent of any specific dark-matter model -- every
check here only exercises PCHMergerTree.branching_rate_terms() and the
foraois.diagnostics module, both of which only touch sigma(M)/alpha(M)
through CosmoData's generic lookups. The same tests should keep passing
unchanged once sigma(M)/alpha(M) come from a WDM/FDM transfer function
instead of CDM -- that's the point of building this now, ahead of the
SIDM/FDM work.
"""

import numpy as np
import pytest

from foraois import diagnostics
from foraois.pch_trees import _build_j_table


def test_j_table_matches_closed_form_at_gamma1_zero():
    # J(u) = integral_0^u (1+1/u'^2)^(gamma1/2) du' reduces to J(u) = u
    # exactly when gamma1 = 0 (PCH08's own stated special case, "the
    # original GALFORM algorithm").
    u_grid, J = _build_j_table(0.0, n_grid=50)
    assert np.allclose(J, u_grid)


def test_j_table_positive_and_increasing_for_pch08_gamma1():
    u_grid, J = _build_j_table(0.38, n_grid=100)
    assert np.all(J >= 0)
    assert np.all(np.diff(J) >= 0)


def test_expected_splits_decreases_with_finer_dz(tree_generator):
    # Nupper is (to leading order) proportional to d_omega, which shrinks
    # with the step size -- a finer dz should give a smaller (safer) Nupper
    # for the same halo history.
    _, Nupper_coarse, _ = diagnostics.expected_splits_per_step(
        tree_generator, M0=1e12, z0=0.0, z_max=1.0, M_res=1e9, dz=0.2
    )
    _, Nupper_fine, _ = diagnostics.expected_splits_per_step(
        tree_generator, M0=1e12, z0=0.0, z_max=1.0, M_res=1e9, dz=0.01
    )
    assert np.nanmax(Nupper_fine) < np.nanmax(Nupper_coarse)
    # and it should scale roughly linearly with dz (allow a factor of ~3
    # for the mean-field mass decay approximation inside the diagnostic)
    ratio = np.nanmax(Nupper_coarse) / np.nanmax(Nupper_fine)
    assert 0.2 / 0.01 / 3 < ratio < 0.2 / 0.01 * 3


def test_expected_splits_flags_a_too_coarse_step(tree_generator):
    # A deliberately extreme case (large halo, fine resolution, coarse dz)
    # should trip well above PCH08's own recommended adaptive-stepping
    # target of ~0.1 -- this is exactly the situation the diagnostic exists
    # to catch before trusting build_forest_* output.
    _, Nupper, _ = diagnostics.expected_splits_per_step(tree_generator, M0=1e12, z0=0.0, z_max=1.0, M_res=1e9, dz=0.2)
    assert np.nanmax(Nupper) > 1.0


@pytest.mark.parametrize(
    "M_res_frac,d_omega",
    [
        (0.1, 0.01),
        (0.01, 0.01),
        (0.001, 0.01),
        (0.01, 0.05),
    ],
)
def test_sampling_consistency_matches_quadrature(tree_generator, M_res_frac, d_omega):
    # The core correctness check: the actual rejection-sampling code path
    # (_draw_progenitor_ratio + _rejection_ratio, exercised via Monte Carlo)
    # must reproduce the analytically (quadrature) integrated true target
    # density S(q)*R(q) -- not just Nupper, which is only an upper bound.
    # This can catch a real bug in either the sampler or the rejection
    # ratio, since the two are computed independently here.
    M2 = 1e12
    result = diagnostics.check_sampling_consistency(
        tree_generator,
        M2=M2,
        M_res=M2 * M_res_frac,
        delta0=1.68,
        d_omega=d_omega,
        n_trials=200_000,
        seed=hash((M_res_frac, d_omega)) % (2**31),
    )
    assert result["passed"], (
        f"empirical={result['empirical_rate']:.5f} vs true={result['true_rate']:.5f} ({result['n_sigma']:.1f} sigma)"
    )


def test_true_split_probability_never_exceeds_nupper(tree_generator):
    # Nupper is the envelope integral (an upper bound per PCH08 eq. A5);
    # the true target density's integral must never exceed it.
    for M_res_frac in [0.1, 0.01, 0.001]:
        M2 = 1e12
        M_res = M2 * M_res_frac
        terms = tree_generator.branching_rate_terms(M2, M_res, delta0=1.68, d_omega=0.05)
        true_rate = diagnostics.true_split_probability(tree_generator, M2, M_res, delta0=1.68, d_omega=0.05)
        assert true_rate <= terms["Nupper"] + 1e-10


def test_sampling_consistency_rejects_nupper_above_one(tree_generator):
    # Found via the demo notebook: with a large enough d_omega, Nupper >= 1
    # and the single-split-per-step architecture's r1 <= Nupper comparison
    # is implicitly clipped to a probability of 1 by r1's Uniform(0,1)
    # support, silently underestimating the true (unclipped) target rate by
    # a factor of Nupper -- the same breakdown expected_splits_per_step
    # warns about, not a sampler bug. check_sampling_consistency should
    # refuse to run (not report a confusing "failure") in that regime.
    terms = tree_generator.branching_rate_terms(1e12, 1e9, delta0=1.68, d_omega=0.2)
    assert terms["Nupper"] >= 1.0  # sanity: this test needs that regime

    with pytest.raises(ValueError, match="Nupper"):
        diagnostics.check_sampling_consistency(
            tree_generator,
            M2=1e12,
            M_res=1e9,
            delta0=1.68,
            d_omega=0.2,
            n_trials=1000,
        )


def test_forest_numpy_no_warnings_when_some_trees_die(tree_generator):
    # Dead trees used to be filled in with M_safe == M_res exactly, a
    # degenerate input (qres=1, sigma_res == sigma2) that divides by zero
    # in V(q)'s denominator -- caught by actually running the demo notebook
    # against a real (non-power-law) P(k), where enough trees died mid-run
    # to hit it. Force some trees to resolution quickly here (tiny M0,
    # relatively large M_res) so this exercises the same dead-tree code path.
    import warnings

    M0_array = np.concatenate([np.full(50, 1.2e8), np.full(50, 1e13)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        mass_history, _, _, _, _ = tree_generator.build_forest_numpy(
            M0_array=M0_array,
            z0=0.0,
            z_max=5.0,
            M_res=1e8,
            dz=0.1,
        )
    assert np.any(mass_history[:50, -1] == 0)  # some of the small haloes did die
    assert np.all(np.isfinite(mass_history))


def test_forest_numpy_and_numba_handle_qres_above_half(tree_generator):
    # A halo with M2 < 2*M_res (qres >= 0.5) has an empty valid range for q
    # -- no split can yield two resolved fragments. Found via the demo
    # notebook: this used to NaN (log(0.5/qres) with qres>0.5) rather than
    # correctly treating it as "no split possible this step" (Nupper=0).
    # Checked on both backends since the numba kernel has its own inlined
    # copy of this branch.
    import warnings

    M0_array = np.full(20, 1.4e8)  # M_res=1e8 -> qres ~ 0.71, comfortably >= 0.5
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        mh_np, _, _, _, _ = tree_generator.build_forest_numpy(
            M0_array=M0_array,
            z0=0.0,
            z_max=2.0,
            M_res=1e8,
            dz=0.1,
        )
        mh_nb, _, _, _ = tree_generator.build_forest_numba(
            M0_array=M0_array,
            z0=0.0,
            z_max=2.0,
            M_res=1e8,
            dz=0.1,
        )
    assert np.all(np.isfinite(mh_np))
    assert np.all(np.isfinite(mh_nb))
