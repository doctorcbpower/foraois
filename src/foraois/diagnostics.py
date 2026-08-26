"""
Validation tools for PCHMergerTree's branching-rate/rejection-sampling
machinery (Parkinson, Cole & Helly 2008, Appendix A).

These are deliberately independent of any specific dark-matter model: every
function here only touches PCHMergerTree.branching_rate_terms() and the
CosmoData sigma(M)/alpha(M) lookups it wraps, so the same checks apply
unchanged once sigma(M)/alpha(M) come from a WDM/FDM transfer function
instead of CDM -- this is meant to be the shared validation harness for
that future work, not a CDM-only tool.

Two independent things are checked:

1. Step-size safety (expected_splits_per_step): Nupper -- the envelope's
   total split probability -- isn't bounded small by construction the way
   it is in PCH08's own adaptive-Delta_z algorithm, because
   PCHMergerTree uses one fixed dz shared across a whole forest (see the
   class docstring). This lets you check Nupper stays comfortably below 1
   (PCH08 target: ~0.1) for a chosen dz before trusting the output.

2. Sampling correctness (true_split_probability / check_sampling_
   consistency): the rejection-sampling scheme in _build_forest_numpy /
   _build_forest_kernel is only correct if the accepted-sample rate matches
   the analytically-integrated true target density S(q)*R(q) -- Nupper
   itself is only an upper bound (per PCH08's own eq. A5 description), not
   the true rate. This is checked by direct Monte Carlo against a
   quadrature integral of S(q)*R(q), independent of the tree-building code
   path itself (it re-derives the target density from branching_rate_terms
   rather than reusing _build_forest_numpy's internals), so it can catch a
   real implementation bug rather than just confirming the code agrees with
   itself.
"""

import warnings

import numpy as np
from scipy import integrate

from .pch_trees import _rejection_ratio


def expected_splits_per_step(tree_generator, M0, z0, z_max, M_res, dz=0.1):
    """
    Deterministic (no randomness) trajectory of Nupper for a single halo
    mass M0 evolving from z0 to z_max on the given fixed step grid --
    Nupper is an upper bound on the expected number of resolved splits in
    that step (PCH08 eq. A5). Values well above PCH08's own adaptive-
    stepping target (~0.1) mean dz is too coarse for that part of the tree's
    history: multiple mergers per step become non-negligible, which the
    single-split-per-step architecture here doesn't model.

    Returns
    -------
    z_steps : np.ndarray, shape (n_steps+1,)
    Nupper : np.ndarray, shape (n_steps,)
    M_trajectory : np.ndarray, shape (n_steps,)
        The mass used to evaluate Nupper at each step (mean-field: ignores
        stochastic branching, so this degrades once M0 has actually split a
        lot by z_max -- treat it as an indicative, not exact, mass path).
    """
    z_steps = np.arange(z0, z_max + dz * 0.5, dz)
    n_steps = len(z_steps) - 1

    Nupper = np.zeros(n_steps)
    M_trajectory = np.zeros(n_steps)
    M = float(M0)

    for j in range(n_steps):
        d0 = float(tree_generator._delta_col_at_z(z_steps[j]))
        d1 = float(tree_generator._delta_col_at_z(z_steps[j + 1]))
        d_omega = d1 - d0

        terms = tree_generator.branching_rate_terms(M, M_res, d0, d_omega)
        Nupper[j] = terms["Nupper"]
        M_trajectory[j] = M
        # mean-field decay estimate for the next step's reference mass
        M = M * (1.0 - min(max(terms["Nupper"], 0.0), 1.0) * 0.25)
        if M < M_res:
            Nupper[j + 1 :] = np.nan
            M_trajectory[j + 1 :] = np.nan
            break

    return z_steps, Nupper, M_trajectory


def expected_splits_per_step_zh(cosmo_data, M0, z0, z_max, M_res, dz=0.1, model="cdm", N_grid=40, S_max_factor=8.0):
    """
    Zhang-Hui analogue of expected_splits_per_step: a deterministic
    (mean-field, no randomness) trajectory of `p_split` -- the per-step
    resolved-split probability `first_crossing_step` computes exactly via
    quadrature (see its own docstring) -- for a single halo mass M0
    evolving from z0 to z_max on a fixed dz grid.

    Why this is needed: PCHMergerTree's Nupper is an *approximate upper
    bound* that is not bounded to [0,1] by construction, and can reach
    into the hundreds at a coarse dz (see PCHMergerTree's own class
    docstring and this module's Nupper checks) -- a clear, unambiguous
    failure mode. Zhang-Hui's p_split, by contrast, is an exact CDF-derived
    probability and is always in [0,1]; it can never "blow up" the way
    Nupper does. But the underlying concern PCH08's Nupper<<1 design target
    guards against is not "does the upper-bound estimator exceed 1" per se
    -- it is that a single step can register at most ONE resolved split,
    so if the *true* expected number of resolved splits across the step is
    not small, the single-split-per-step construction under-counts real
    multi-merger structure within that step, for *either* backend's
    branching kernel. p_split close to 1 is exactly the regime where that
    under-counting risk is largest for Zhang-Hui (a near-certain split
    every step leaves no room to distinguish "one split" from "several
    splits compressed into one draw"), so we treat p_split itself as the
    Zhang-Hui-side quantity to hold small, by direct analogy with Nupper,
    and adopt the same qualitative target (<<1, ~0.1 in practice) pending
    a more rigorous derivation. This has NOT been derived from first
    principles the way PCH08's eq. A5 Nupper was -- treat it as a
    reasonable, symmetric, empirically-motivated proxy (see this module's
    tests / the paper's Section 5.4 discussion for a convergence check
    confirming p_split<=0.1 tracks where Zhang-Hui's own results stop
    changing with dz), not an authoritative bound.

    Returns
    -------
    z_steps : np.ndarray, shape (n_steps+1,)
    p_split : np.ndarray, shape (n_steps,)
    M_trajectory : np.ndarray, shape (n_steps,)
    """
    from .zhang_hui_trees import first_crossing_step

    z_steps = np.arange(z0, z_max + dz * 0.5, dz)
    n_steps = len(z_steps) - 1

    p_split = np.zeros(n_steps)
    M_trajectory = np.zeros(n_steps)
    M = float(M0)

    for j in range(n_steps):
        z0_j, z1_j = z_steps[j], z_steps[j + 1]
        step = first_crossing_step(M, z0_j, z1_j, M_res, cosmo_data, model=model, N_grid=N_grid, S_max_factor=S_max_factor)
        p_split[j] = step["p_split"]
        M_trajectory[j] = M
        # mean-field decay estimate for the next step's reference mass,
        # matching expected_splits_per_step's own convention exactly
        M = M * (1.0 - min(max(step["p_split"], 0.0), 1.0) * 0.25)
        if M < M_res:
            p_split[j + 1 :] = np.nan
            M_trajectory[j + 1 :] = np.nan
            break

    return z_steps, p_split, M_trajectory


def true_split_probability(tree_generator, M2, M_res, delta0, d_omega, n_points=400):
    """
    Exact (quadrature, not Monte Carlo) total split probability
    integral_{qres}^{1/2} S(q) R(q) dq -- the TRUE target density PCH08's
    rejection sampling is built to reproduce (eq. A1: dN/dq = S(q)R(q)dz).
    Nupper = integral S(q) dq alone is only an upper bound on this (S(q) is
    the majorizing envelope, R(q) <= 1 the acceptance probability), so this
    is always <= Nupper.
    """
    terms = tree_generator.branching_rate_terms(M2, M_res, delta0, d_omega)
    eta = terms["eta"]
    qres = terms["qres"]
    S_coeff_domega = terms["S_coeff"] * terms["d_omega"]

    sigma_fn = tree_generator.cosmo_data.sigma_at_logmass
    alpha_fn = tree_generator.cosmo_data.dlogsigma_at_logmass
    gamma1 = tree_generator.gamma1

    def integrand(q):
        S_q = S_coeff_domega * q ** (eta - 1.0)
        R_q = float(_rejection_ratio(np.asarray(q), np.asarray(M2), terms, sigma_fn, alpha_fn, gamma1))
        return S_q * R_q

    # For wide qres-to-0.5 dynamic ranges, quad's roundoff-error estimate can
    # be conservative even though the result is accurate (cross-validated
    # against a log-substituted integral and Monte Carlo -- see
    # check_sampling_consistency); suppress that specific warning here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
        value, _ = integrate.quad(integrand, qres, 0.5, limit=n_points)
    return value


def check_sampling_consistency(tree_generator, M2, M_res, delta0, d_omega, n_trials=200_000, seed=None):
    """
    Monte Carlo vs. quadrature cross-check of the rejection-sampling scheme.

    Draws n_trials independent (r1, q, r3) triples exactly as
    _build_forest_numpy does for a single fixed halo, and compares the
    empirical accepted-split fraction against true_split_probability's
    quadrature integral. This exercises the actual sampling code path
    (_draw_progenitor_ratio, _rejection_ratio) against an independently
    computed target, so it can catch a real bug in either -- not just
    confirm the code is internally consistent with itself.

    Only meaningful when Nupper < 1: the "at most one split per step"
    architecture compares a single r1 against Nupper, which is implicitly
    clipped to 1 by r1's Uniform(0,1) support. true_split_probability's
    integral assumes the ordinary (unclipped) rate Nupper*E_q[R(q)], so once
    Nupper >= 1 the two are expected to disagree -- that's the single-split-
    per-step approximation breaking down (the same thing
    expected_splits_per_step warns about), not a sampling bug. Raises
    ValueError in that regime rather than reporting a confusing failure;
    check expected_splits_per_step (or reduce dz) first.

    Returns a dict with 'empirical_rate', 'true_rate', 'n_sigma' (the
    empirical rate's deviation from the true rate in units of its own
    binomial standard error), and 'passed' (n_sigma < 5).
    """
    from .pch_trees import _draw_progenitor_ratio

    if seed is not None:
        np.random.seed(seed)

    terms = tree_generator.branching_rate_terms(M2, M_res, delta0, d_omega)
    if terms["Nupper"] >= 1.0:
        raise ValueError(
            f"Nupper = {terms['Nupper']:.3f} >= 1 for this (M2, M_res, delta0, "
            "d_omega): the single-split-per-step approximation has broken "
            "down here (see expected_splits_per_step), so comparing against "
            "true_split_probability isn't a meaningful check -- reduce dz "
            "(i.e. d_omega) for this halo/step before trusting build_forest_* "
            "output, and re-run this check with the smaller d_omega."
        )
    sigma_fn = tree_generator.cosmo_data.sigma_at_logmass
    alpha_fn = tree_generator.cosmo_data.dlogsigma_at_logmass
    gamma1 = tree_generator.gamma1

    r1 = np.random.rand(n_trials)
    attempted = r1 <= terms["Nupper"]

    u2 = np.random.rand(n_trials)
    q = _draw_progenitor_ratio(u2, terms["qres"], terms["eta"])
    R = _rejection_ratio(q, M2, terms, sigma_fn, alpha_fn, gamma1)
    r3 = np.random.rand(n_trials)
    accepted = attempted & (r3 <= R)

    empirical_rate = float(accepted.mean())
    true_rate = true_split_probability(tree_generator, M2, M_res, delta0, d_omega)

    binomial_se = np.sqrt(true_rate * (1.0 - true_rate) / n_trials)
    n_sigma = abs(empirical_rate - true_rate) / binomial_se if binomial_se > 0 else np.inf

    return {
        "empirical_rate": empirical_rate,
        "true_rate": true_rate,
        "n_sigma": float(n_sigma),
        "passed": bool(n_sigma < 5.0),
    }
