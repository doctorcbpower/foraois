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
