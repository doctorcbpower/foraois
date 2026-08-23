"""
A solver for N23's constrained (Brownian-bridge) first-crossing integral
equation -- Nadler, Benson, Driskell, Du & Gluscevic 2023 (N23), MNRAS
521, 3201, their Eq. A5.

This is deliberately a *separate* module and numerical method from
first_crossing.py, not an extension of it. first_crossing.py's
solve_first_crossing implements Zhang & Hui (2006)'s own derivation,
which first differentiates their integral equation into a
`g1(S) + integral f*g2` kernel form; N23 do *not* do this for either
their unconstrained or constrained case. They discretize the raw
"mass-conservation" integral equations directly (their Eq. A1 and A5)
via Du et al. (2017)'s midpoint method -- a genuinely different
numerical scheme. Differentiating Eq. A5 to force it into
first_crossing.py's g1/g2 framework would be a real derivation risk
(its S1,S0-dependent variance terms are more involved than the
unconstrained case's); this module instead follows N23's own validated
method against the equation as given.

Both the unconstrained (Eq. A1) and constrained (Eq. A5) equations share
the same structure:

    1 = integral_0^S f(Ŝ) dŜ + F(S) - integral_0^S f(Ŝ) K(S, Ŝ) dŜ

`_solve_mass_conservation_midpoint` solves this generic form via a
midpoint quadrature: f is represented at the *midpoint* of each
subinterval (not at grid edges, unlike first_crossing.py's f), so the
kernel K(S_i, Ŝ) is never evaluated at Ŝ=S_i exactly for interior i (the
midpoint of the interval ending at S_i is always a full half-step away)
-- this sidesteps first_crossing.py's own near-diagonal singularity
handling (_g2_near_diagonal) for those points, by construction. The
*last* interval of the constrained solver (S -> S1) is a genuine
exception -- see solve_first_crossing_constrained's own comments -- and
does need special handling, just a different (simpler) one: the bridge's
total probability over [S0, S1] is exactly 1 by construction, which
pins the last unknown analytically instead of evaluating a 0/0 limit.

`solve_first_crossing_unconstrained_midpoint` applies the same generic
solver to the *unconstrained* equation (Eq. A1) purely as a validation
tool: it has a known closed-form answer (first_crossing.py's
flat_barrier_first_crossing) independent of solve_first_crossing's own
implementation, so matching it validates this module's numerical method
itself before trusting it on the constrained case, which has no
closed-form ground truth (only the S1 -> infinity convergence-to-
unconstrained check N23 themselves use, Appendix B1). It also confirms
that Zhang & Hui's forward-substitution scheme and N23's own midpoint
method agree (to ~1e-4 relative error away from the near-S=0 tail, at
N~500).

The constrained kernel's conditional propagator (`_K_constrained` below)
follows standard Brownian-bridge conditioning theory: conditional on the
bridge having touched the barrier at an earlier point `(Ŝ, B(Ŝ))`, the
segment from there to the fixed endpoint `(S1, delta1)` is itself a
bridge, with drift `B(Ŝ) + (S-Ŝ)/(S1-Ŝ)*(delta1-B(Ŝ))`. This is
verifiable analytically: it reduces to the unconstrained kernel exactly
as `S1 -> infinity`.
"""

import numpy as np
from scipy.special import erf


def _solve_mass_conservation_midpoint(F_func, K_func, S_max, N):
    """
    Generic solver for `1 = integral_0^S f dŜ + F(S) - integral_0^S f*K(S,Ŝ) dŜ`,
    via Du et al. (2017)-style midpoint discretization (see module
    docstring). Shared core for both the unconstrained and constrained
    cases below -- neither differs in numerical method, only in
    `F_func`/`K_func`.

    Parameters
    ----------
    F_func : callable
        F(S) -> float or ndarray, the forcing term (evaluated at grid
        edges S_1..S_N).
    K_func : callable
        K(S, Ŝ) -> float or ndarray, the integral kernel (evaluated at
        grid edges S for the first argument, subinterval midpoints Ŝ for
        the second).
    S_max : float
        Upper end of the grid.
    N : int
        Number of subintervals (and thus midpoints/unknowns).

    Returns
    -------
    S_mid : np.ndarray, shape (N,)
        Midpoint of each subinterval -- where f is represented (not at
        grid edges, unlike first_crossing.py's S_grid/f).
    f : np.ndarray, shape (N,)
    """
    dS = S_max / N
    S_edges = np.arange(N + 1) * dS  # S_edges[0]=0 .. S_edges[N]=S_max
    S_mid = S_edges[:-1] + 0.5 * dS  # midpoint of subinterval [S_edges[i], S_edges[i+1]]

    F_vals = F_func(S_edges[1:])  # F(S_1) .. F(S_N)

    f = np.zeros(N)
    for i in range(N):
        Si = S_edges[i + 1]
        # K(S_i, Ŝ_j) for every already-known midpoint j <= i (including
        # j=i itself, the newest unknown -- isolated below).
        Kvals = K_func(Si, S_mid[: i + 1])
        weight = dS * (1.0 - Kvals)
        known_sum = np.dot(f[:i], weight[:i]) if i > 0 else 0.0
        # weight[i] multiplies the unknown f[i] itself.
        f[i] = (1.0 - F_vals[i] - known_sum) / weight[i]

    return S_mid, f


def _F_unconstrained(S, B):
    with np.errstate(divide="ignore", invalid="ignore"):
        return erf(B(S) / np.sqrt(2.0 * S))


def _K_unconstrained(S, Sp, B):
    with np.errstate(divide="ignore", invalid="ignore"):
        return erf((B(S) - B(Sp)) / np.sqrt(2.0 * (S - Sp)))


def solve_first_crossing_unconstrained_midpoint(B, S_max, N):
    """
    Solve N23's Eq. A1 (the unconstrained case, same physics as
    first_crossing.py's solve_first_crossing) via this module's midpoint
    method -- purely a cross-check of the numerical method itself (see
    module docstring): compare against flat_barrier_first_crossing
    (closed form) or solve_first_crossing (a different numerical method
    for the same equation) before trusting this module's constrained
    solver, which has no closed-form ground truth to check against.

    Parameters
    ----------
    B : callable
        Barrier function B(S) -- same convention as
        first_crossing.py's solve_first_crossing.
    S_max : float
    N : int

    Returns
    -------
    S_mid, f : np.ndarray, shape (N,) each -- f represented at
        subinterval midpoints, not grid edges.
    """
    return _solve_mass_conservation_midpoint(
        F_func=lambda S: _F_unconstrained(S, B),
        K_func=lambda S, Sp: _K_unconstrained(S, Sp, B),
        S_max=S_max,
        N=N,
    )


def _F_constrained(S, B, S0, delta0, S1, delta1):
    # mu(S) = delta0 + (S-S0)/(S1-S0)*(delta1-delta0) -- the bridge's own
    # drift (main text Eq. 15, independently verified as standard
    # Brownian-bridge theory). F(S) = erf[(B(S)-mu(S))/sqrt(2*Var(S))],
    # Var(S) = (S1-S)(S-S0)/(S1-S0) (Eq. 16 at Ŝ=S).
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = delta0 + (S - S0) / (S1 - S0) * (delta1 - delta0)
        var = (S1 - S) * (S - S0) / (S1 - S0)
        return erf((B(S) - mu) / np.sqrt(2.0 * var))


def _K_constrained(S, Sp, B, S1, delta1):
    # Conditional on the bridge having touched the barrier at an earlier
    # point (Ŝ=Sp, B(Sp)), the segment from there to S1 is *itself* a
    # Brownian bridge (the recursive self-similarity N23 note after their
    # Eq. 18): mu_cond(S) = B(Sp) + (S-Sp)/(S1-Sp)*(delta1-B(Sp)),
    # Var_cond(S) = (S1-S)(S-Sp)/(S1-Sp). This reduces to the
    # *unconstrained* kernel exactly as S1 -> infinity (mu_cond(S) ->
    # B(Sp), Var_cond(S) -> S-Sp), an analytically verifiable check.
    with np.errstate(divide="ignore", invalid="ignore"):
        Bp = B(Sp)
        mu_cond = Bp + (S - Sp) / (S1 - Sp) * (delta1 - Bp)
        var_cond = (S1 - S) * (S - Sp) / (S1 - Sp)
        return erf((B(S) - mu_cond) / np.sqrt(2.0 * var_cond))


def solve_first_crossing_constrained(B, S0, delta0, S1, delta1, N):
    """
    Solve N23's Eq. A5 -- the first-crossing distribution for a Brownian
    bridge excursion pinned to pass through `(S1, delta1)`, starting from
    `(S0, delta0)` -- via direct midpoint discretization (see module
    docstring; NOT first_crossing.py's g1/g2 scheme).

    Origin-shift convention (matching zhang_hui_trees.py): in normal use,
    `S0=0, delta0=0` (the walk always starts at its own current point),
    with `B` already the *shifted* barrier `delta_c(M(S),z) - delta_c(M0,z0)`
    the way zhang_hui_trees.py's own `first_crossing_step` builds it.
    `S0, delta0` are kept as explicit parameters (not hardcoded to 0) to
    match N23's own general Eq. A5 and to make the S0=0 assumption
    checkable, not assumed.

    Parameters
    ----------
    B : callable
        The underlying (unconstrained) barrier B(S), same as passed to
        first_crossing.py's solve_first_crossing -- NOT the "effective"
        bridge-adjusted quantity; the bridge machinery is applied
        internally.
    S0, delta0 : float
        The bridge's starting point.
    S1, delta1 : float
        The fixed point the bridge is conditioned to pass through
        (S1 > S0). Physically, delta1 should equal B(S1) (the shifted
        barrier's own value at S1 -- that's what it means for M1 to be a
        genuine progenitor mass at z1), but is kept as an explicit,
        independent parameter (not computed internally as B(S1)) to match
        N23's own general treatment and keep this solver usable for
        testing/validation with an arbitrary constraint point.
    N : int
        Number of subintervals between S0 and S1 -- the bridge is only
        defined on [S0, S1]; beyond S1, switch to the unconstrained
        solver (see zhang_hui_constrained_trees.py's constrained-branch
        grower).

    Returns
    -------
    S_mid, f : np.ndarray, shape (N,) each -- f represented at
        subinterval midpoints of [S0, S1].
    """
    if S1 <= S0:
        raise ValueError(f"S1={S1} must be > S0={S0}.")

    dS = (S1 - S0) / N
    S_edges = S0 + np.arange(N + 1) * dS
    S_mid = S_edges[:-1] + 0.5 * dS

    F_vals = _F_constrained(S_edges[1:], B, S0, delta0, S1, delta1)

    f = np.zeros(N)
    # Last interval (i=N-1) is skipped in this loop and handled below --
    # both F(S1) and K(S1, Ŝ) hit a genuine 0/0 limit as S->S1 (numerator
    # and denominator both vanish simultaneously, since delta1=B(S1) is
    # exactly where the bridge is guaranteed to be by construction), so
    # the ordinary recursion can't be evaluated there directly.
    for i in range(N - 1):
        Si = S_edges[i + 1]
        Kvals = _K_constrained(Si, S_mid[: i + 1], B, S1, delta1)
        weight = dS * (1.0 - Kvals)
        known_sum = np.dot(f[:i], weight[:i]) if i > 0 else 0.0
        f[i] = (1.0 - F_vals[i] - known_sum) / weight[i]

    # The bridge reaches delta1 by S1 with probability exactly 1 (that's
    # what being conditioned on the constraint means), so the total
    # integral over [S0, S1] is known analytically -- use it to fix the
    # last interval instead of evaluating the singular S=S1 point.
    f[N - 1] = 1.0 / dS - np.sum(f[: N - 1])

    return S_mid, f


def simulate_constrained_random_walk_first_crossings(
    B,
    S1,
    delta1,
    dS,
    n_walks,
    tol,
    rng=None,
):
    """
    Direct Monte Carlo cross-check for the constrained first-crossing
    distribution, independent of solve_first_crossing_constrained's own
    internals -- the constrained analog of first_crossing.py's
    simulate_random_walk_first_crossings.

    Simulates `n_walks` genuinely *unconstrained* Wiener-process random
    walks from `(S=0, delta=0)` to `S1` (uncorrelated Gaussian increments,
    same convention as first_crossing.py's own simulator), then keeps only
    walks landing within `tol` of `delta1` at `S1` -- rejection sampling
    for the Brownian-bridge condition, rather than directly simulating a
    bridge process (deliberately: this way the check doesn't share any
    machinery with either the analytic bridge formulas or the solver
    being validated). For each *kept* walk, records the first S at which
    it crosses `B(S)`; a walk that reaches `S1` without having crossed
    earlier is recorded as crossing at `S1` itself (since `delta1=B(S1)`
    by construction, arriving at the target point *is* a crossing, absent
    an earlier one -- matches the analytic solver's own last-interval
    treatment, and where its `S1`-adjacent probability mass concentration
    comes from).

    Parameters
    ----------
    B : callable
    S1, delta1 : float
        The constraint point (see solve_first_crossing_constrained).
    dS : float
        Step size for the walk's own S-grid.
    n_walks : int
        Number of *attempted* (not accepted) walks -- the number surviving
        the `tol` window will be considerably smaller (the acceptance
        fraction is roughly the free walk's own N(0,S1) density at delta1,
        times 2*tol -- pick n_walks/tol so enough walks survive for
        useful statistics; too small a tol makes this prohibitively slow).
    tol : float
        Half-width of the acceptance window around delta1 at S1.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        The first-crossing S value for each *accepted* walk (length <=
        n_walks, generally much smaller -- see `tol` above).
    """
    rng = rng if rng is not None else np.random.default_rng()

    S_grid = np.arange(0.0, S1 + dS, dS)
    delta = np.zeros(n_walks)
    crossed = np.zeros(n_walks, dtype=bool)
    first_crossing_S = np.full(n_walks, np.nan)

    for i in range(1, len(S_grid)):
        step = rng.normal(0.0, np.sqrt(dS), size=n_walks)
        delta += step  # unconstrained -- every walk keeps accumulating,
        # unlike first_crossing.py's own simulator, since we need the
        # endpoint value at S1 for every walk to decide acceptance, not
        # just walks that haven't crossed yet.
        newly_crossed = (~crossed) & (delta >= B(S_grid[i]))
        first_crossing_S[newly_crossed] = S_grid[i]
        crossed |= newly_crossed

    accepted = np.abs(delta - delta1) <= tol
    result = first_crossing_S[accepted]
    # Walks that reach S1 within the acceptance window without having
    # crossed earlier count as crossing at S1 itself (see docstring).
    result = np.where(np.isnan(result), S1, result)
    return result


def simulate_bridge_path(S0, delta0, S1, delta1, dS, rng=None):
    """
    Sample one realization of a Brownian bridge pinned to `(S0,delta0)`
    and `(S1,delta1)`, via sequential conditional sampling -- deliberately
    *barrier-free*: a bridge's own statistics depend only on its two
    endpoints, not on any collapse barrier. This is what makes it possible
    to build the constrained tree-growing machinery
    (zhang_hui_constrained_trees.py) on path simulation rather than
    extending `solve_first_crossing_constrained` to a per-step-varying
    barrier redshift.

    At each new grid point S, given the already-sampled current point
    `(Ŝ, delta_current)`, the conditional distribution of `delta(S)` is
    exactly the "bridge segment from `(Ŝ,delta_current)` to `(S1,delta1)`"
    -- the same `mu_cond`/`var_cond` forms as `_K_constrained` (the
    recursive self-similarity property: a bridge segment starting from an
    already-reached point is itself a bridge to the same fixed endpoint).
    This is the standard sequential/Markovian construction of a Brownian
    bridge, reusing already-verified math rather than a new derivation.

    Parameters
    ----------
    S0, delta0 : float
        Starting point.
    S1, delta1 : float
        Fixed endpoint (S1 > S0) -- the path is guaranteed (up to floating
        point) to end exactly at delta1 when S reaches S1.
    dS : float
        Step size for the path's own S-grid.
    rng : np.random.Generator, optional

    Returns
    -------
    S_grid : np.ndarray, shape (M+1,)
    delta_path : np.ndarray, shape (M+1,)
        delta_path[0] = delta0, delta_path[-1] = delta1 exactly.
    """
    if S1 <= S0:
        raise ValueError(f"S1={S1} must be > S0={S0}.")
    rng = rng if rng is not None else np.random.default_rng()

    S_grid = np.arange(S0, S1, dS)
    S_grid = np.append(S_grid, S1)  # ensure the path ends exactly at S1

    delta_path = np.empty(len(S_grid))
    delta_path[0] = delta0

    Sp, deltap = S0, delta0
    for i in range(1, len(S_grid)):
        S = S_grid[i]
        if S >= S1:
            # Final point -- no sampling needed, the bridge is pinned here
            # by construction (also avoids the S1-Sp -> 0 denominator).
            delta_path[i] = delta1
            break
        mu_cond = deltap + (S - Sp) / (S1 - Sp) * (delta1 - deltap)
        var_cond = (S1 - S) * (S - Sp) / (S1 - Sp)
        deltap = rng.normal(mu_cond, np.sqrt(var_cond))
        delta_path[i] = deltap
        Sp = S

    return S_grid, delta_path
