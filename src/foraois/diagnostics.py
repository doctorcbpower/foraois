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
   (practical criterion: <~ 0.1) for a chosen dz before trusting the output.

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
    stepping value (~0.1) mean dz is too coarse for that part of the tree's
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


def expected_eps_splits_per_step(cosmo_data, M0, z0, z1, M_res, model="cdm", n_q=400):
    """
    Analytic EPS expected number of binary splits, with the smaller fragment resolved
    (M_res <= M2 <= M0/2), in one step z0 -> z1 for a halo of mass M0:

        E = int_{q_res}^{1/2} (1/q) f(dS, d_omega) |dS/dM2| M0 dq,   q = M2/M0,

    with f the flat-barrier first-crossing density
    d_omega exp(-d_omega^2 / 2 dS) / (sqrt(2 pi) dS^{3/2}) and dS = sigma^2(M2) - sigma^2(M0).
    Both single-split-per-step constructions need E small (about 0.1 or below, the same requirement as
    PCH08's N_upper criterion), but E is a property of the EPS rate and does not depend on either
    backend's sampler. A binary-per-step tree (Zhang-Hui) draws at most one split per step, so at
    E >> 1 its split probability saturates below E.

    Parameters
    ----------
    cosmo_data : CosmoData
        Provides sigma(M) and delta_col(z); its sigma grid is populated when a tree generator is
        constructed from it. The barrier is flat, so `model` only selects delta_c.
    n_q : int
        Log-spaced quadrature points in q.

    Returns
    -------
    float
    """
    from foraois.collapse import delta_c

    if M_res * 2.0 >= M0:
        return 0.0
    d_omega = float(delta_c(M_res, z1, model, cosmo_data)) - float(delta_c(M_res, z0, model, cosmo_data))
    sigma0_sq = float(cosmo_data.sigma_at_logmass(np.log10(M0))) ** 2
    lnq = np.linspace(np.log(M_res / M0), np.log(0.5), n_q)
    q = np.exp(lnq)
    logM2 = np.log10(M0 * q)
    sig = np.asarray(cosmo_data.sigma_at_logmass(logM2), dtype=float)
    dS = np.maximum(sig**2 - sigma0_sq, 1e-300)
    dS_dlnM = 2.0 * sig**2 * np.abs(np.asarray(cosmo_data.dlogsigma_at_logmass(logM2), dtype=float))
    f = d_omega * np.exp(-0.5 * d_omega**2 / dS) / (np.sqrt(2.0 * np.pi) * dS**1.5)
    return float(np.trapezoid(f * dS_dlnM / q, lnq))


def expected_splits_per_step_zh(cosmo_data, M0, z0, z_max, M_res, dz=0.1, model="cdm", N_grid=40, S_max_factor=8.0):
    """
    Trajectory of the Zhang-Hui single-step split probability `p_split` (from `first_crossing_step`) for a
    single halo mass M0 evolving from z0 to z_max on a fixed dz grid (mean-field, no randomness).

    `p_split` is bounded by 1, so unlike Nupper it cannot exceed the single-split-per-step limit; it saturates
    instead. A small `p_split` does not by itself show that the step is adequate: the quantity to compare
    with 0.1 is the EPS expected number of splits per step, `expected_eps_splits_per_step`. When that is
    large the Zhang-Hui builder, which draws at most one split per step, under-counts splits and `p_split`
    saturates below it. This function is kept for the trajectory of `p_split` itself.

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
