"""
Exact first-crossing distribution for excursion-set random walks with an
arbitrary moving barrier B(S) (Zhang & Hui 2006, "On Random Walks with a
General Moving Barrier", arXiv:astro-ph/0508384).

This solves the *Markov* (Brownian) first-crossing problem, i.e. a walk with
uncorrelated increments in S, the mass variance. The solution is exact for
such a walk. The sharp-k excursion-set construction (see
utils/window_function.py's sharp_k_window and its docstring) is the standard
one that gives Markov walks. The solver itself takes only a barrier B(S) and
knows nothing about the window function, so with any other S(M) -- including
the default real-space top-hat, whose walks are correlated -- it is the
Markov approximation applied to that S(M), not an exact treatment of the
correlated-walk problem (which needs the harder machinery of e.g. Musso &
Sheth 2013, arXiv:1303.0337). No non-Markovian correction is implemented

Zhang & Hui show f(S), the probability density of first crossing the
barrier between S and S+dS, satisfies the Volterra integral equation

    f(S) = g1(S) + integral_0^S f(S') g2(S, S') dS'                (eq. 5)

    g1(S)      = (B(S)/S - 2 dB/dS) * P0(B(S), S)                  (eq. 6)
    g2(S, S')  = [2 dB/dS - (B(S)-B(S'))/(S-S')] * P0(B(S)-B(S'), S-S')
                                                                     (eq. 7)
    P0(delta, S) = 1/sqrt(2 pi S) * exp(-delta^2 / (2 S))           (eq. 2)

which reduces to a triangular linear system on a uniform S-grid, solved by
forward substitution. It reproduces the classic flat/linear-barrier analytic 
solutions, to discretisation error, as a special case (Section 2.2, eq. 18), 
which is the main thing this module's tests check it against, together with 
a direct Monte Carlo random walk (reproducing the paper's Figure 2 cross-check).

This is a general-purpose replacement for PCH08's fitted (gamma1, gamma2,
G0) branching-rate machinery: PCH08's fit is specifically calibrated for a
constant (CDM) barrier under a top-hat window; f(S) here is a numerical 
solution (discretisation error only) for any barrier shape (constant, linear,
ellipsoidal-collapse, mass-dependent SIDM/FDM barriers, ...) as long as the 
walk is Markovian. PCHMergerTree/pch_trees.py does not use this module (it keeps
its fitted rate); ZhangHuiMergerTree does (see the navigation section below).
It is a standalone, independently-validated building block.

Where things live (navigation)
------------------------------
This module holds only the general mathematics and has no foraois imports: the Volterra solver
(`solve_first_crossing`), the analytic first-crossing densities (`flat_barrier_first_crossing`,
`linear_barrier_first_crossing`) and a Monte Carlo reference. It knows nothing about cosmology, barriers
configured on `CosmoData`, or trees. The Zhang-Hui merger trees do use `solve_first_crossing`; the pieces that
turn it into a tree step live in `zhang_hui_trees.py`:

* `first_crossing_step` builds the shifted barrier B(S) = delta_c(M(S), z1) - delta_c(M0, z0) from
  `foraois.collapse.delta_c`, sizes the S-grid, calls `solve_first_crossing` and integrates f(S). It then goes on
  to the Zhang-Hui / Nadler et al. (2023) mass-budget quantities (`F_zh`, `S_lower`, `p_split`). It deliberately
  combines the two: the general half has no other user, and it needs `cosmo_data`, which this module avoids.
* The grid resolution machinery (`N_grid`, `S_max_factor`, the coarse-grid warning) belongs to that step, which
  builds the grid from the resolution mass; `solve_first_crossing` takes `S_max` and `N` as given.
* `_flat_barrier_cdf` / `_flat_barrier_sample` are the closed-form CDF and quantile of
  `flat_barrier_first_crossing`; the closed-form ZH sampling paths use them (the Numba kernels carry their own
  inlined equivalent).
* `first_crossing_constrained.py` is a separate solver for the bridge-conditioned (constrained) problem.
"""

import numpy as np
from scipy import integrate


def P0(delta, S):
    """
    Free (no-barrier) Gaussian propagator: the probability density of an
    unconstrained Markovian walk being at displacement `delta` after
    variance `S` has accumulated (Zhang & Hui eq. 2). S=0 is handled as
    the delta-function limit (nonzero only exactly at delta=0), though
    callers here never evaluate P0 at S=0 in practice (S=0 corresponds to
    the start of the walk, before any barrier crossing is possible).
    """
    S = np.asarray(S, dtype=float)
    delta = np.asarray(delta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            S > 0,
            np.exp(-(delta**2) / (2.0 * np.where(S > 0, S, 1.0))) / np.sqrt(2.0 * np.pi * np.where(S > 0, S, 1.0)),
            np.where(delta == 0, np.inf, 0.0),
        )
    return result


def _numeric_dBdS(B, S, h_frac=1e-6):
    """Central-difference dB/dS, used when the caller doesn't supply one analytically."""
    S = np.asarray(S, dtype=float)
    h = np.maximum(np.abs(S), 1.0) * h_frac
    return (B(S + h) - B(S - h)) / (2.0 * h)


def g1(S, B, dBdS):
    """Zhang & Hui eq. 6, the inhomogeneous term of the integral equation."""
    Bv = B(S)
    return (Bv / S - 2.0 * dBdS(S)) * P0(Bv, S)


def g2(S, Sp, B, dBdS):
    """
    Zhang & Hui eq. 7, the integral equation's kernel. Only valid for
    S != Sp -- it diverges as S' -> S (like 1/sqrt(S-S'), an integrable
    singularity); the near-diagonal discretized term needs the regularized
    average in _g2_near_diagonal instead of a pointwise call here (see
    that function's docstring and eq. 24).
    """
    Bs, Bsp = B(S), B(Sp)
    dS = S - Sp
    return (2.0 * dBdS(S) - (Bs - Bsp) / dS) * P0(Bs - Bsp, dS)


def _g2_near_diagonal(Si, dS_step, B, dBdS):
    """
    The width-averaged g2(Si, S') over S' in [Si - dS_step, Si] (Zhang &
    Hui eq. 24), used for the one discretized term (Delta_{i,i}) that
    would otherwise require evaluating g2 within dS_step/2 of its S'->S
    singularity -- every other discretized term in the sum is at least
    1.5*dS_step away from the singularity and needs no special treatment
    (see this module's docstring for the gap bookkeeping).

    Substituting u = sqrt(Si - S') turns the 1/sqrt(Si-S') singularity
    into a removable one (the integrand becomes ~finite as u->0), so
    plain adaptive quadrature converges cleanly instead of struggling with
    an endpoint singularity.
    """

    def integrand(u):
        Sp = Si - u**2
        return g2(Si, Sp, B, dBdS) * 2.0 * u

    u_max = np.sqrt(dS_step)
    val, _ = integrate.quad(integrand, 0.0, u_max, limit=100)
    return val / dS_step


def solve_first_crossing(B, S_max, N, dBdS=None):
    """
    Solve Zhang & Hui (2006)'s integral equation (eq. 5) for the
    first-crossing distribution f(S) of a Markovian excursion-set random
    walk with barrier B(S), via the forward-substitution scheme in their
    eq. 19-22 (a triangular linear system -- no matrix inversion needed).

    Parameters
    ----------
    B : callable
        Barrier function, B(S) -> float or ndarray. Must be differentiable
        and, physically, B(0) > 0 (the walk starts at delta=0, S=0, below
        any sensible barrier).
    S_max : float
        Upper end of the S-grid to solve out to.
    N : int
        Number of grid intervals (N+1 grid points, S_0=0 .. S_N=S_max).
        Larger N resolves sharper barrier features and the near-S=0
        behaviour better, at O(N^2) cost (the forward-substitution sum at
        each S_i touches all previous grid points).
    dBdS : callable, optional
        dB/dS as a function of S. If omitted, computed via central finite
        differences (_numeric_dBdS) -- fine for smooth barriers, but an
        analytic dBdS should be preferred when available (e.g. for a flat
        or linear barrier, cheaper and exact).

    Returns
    -------
    S_grid : np.ndarray, shape (N+1,)
    f : np.ndarray, shape (N+1,)
        f(S_grid[0]) = 0 by construction (eq. 22) -- the walk cannot have
        already crossed at S=0.
    """
    if dBdS is None:
        dBdS = lambda S: _numeric_dBdS(B, S)

    dS = S_max / N
    S_grid = np.arange(N + 1) * dS

    f = np.zeros(N + 1)
    g1_vals = np.zeros(N + 1)
    g1_vals[1:] = g1(S_grid[1:], B, dBdS)

    for i in range(1, N + 1):
        Si = S_grid[i]

        Delta_ii = 0.5 * dS * _g2_near_diagonal(Si, dS, B, dBdS)

        if i == 1:
            running_sum = 0.0
        else:
            j = np.arange(1, i)  # 1 .. i-1
            Sj = S_grid[j]
            Delta_ij = 0.5 * dS * g2(Si, Sj - 0.5 * dS, B, dBdS)
            Delta_ij1 = 0.5 * dS * g2(Si, S_grid[j + 1] - 0.5 * dS, B, dBdS)
            running_sum = np.sum(f[j] * (Delta_ij + Delta_ij1))

        f[i] = (g1_vals[i] + running_sum) / (1.0 - Delta_ii)

    return S_grid, f


def flat_barrier_first_crossing(S, delta_c):
    """
    Analytic first-crossing distribution for a constant barrier B(S)=delta_c
    (the standard Press-Schechter case) -- Zhang & Hui eq. 18 with a=delta_c,
    b=0:

        f(S) = delta_c / (S sqrt(2 pi S)) * exp(-delta_c^2 / (2 S))

    Used as a closed-form ground truth for solve_first_crossing's
    validation tests, independent of the general numerical solver.
    """
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            S > 0,
            delta_c / (S * np.sqrt(2.0 * np.pi * S)) * np.exp(-(delta_c**2) / (2.0 * np.where(S > 0, S, 1.0))),
            0.0,
        )


def linear_barrier_first_crossing(S, a, b):
    """
    Analytic first-crossing distribution for a linear barrier B(S)=a+b*S --
    the general inverse Gaussian solution, Zhang & Hui eq. 18 (their B(0)
    is just `a` for a linear barrier):

        f(S) = a / (S sqrt(2 pi S)) * exp(-(a + b*S)^2 / (2 S))

    Reduces to flat_barrier_first_crossing when b=0.
    """
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(
            S > 0,
            a / (S * np.sqrt(2.0 * np.pi * S)) * np.exp(-((a + b * S) ** 2) / (2.0 * np.where(S > 0, S, 1.0))),
            0.0,
        )


def simulate_random_walk_first_crossings(B, S_max, dS, n_walks, rng=None):
    """
    Direct Monte Carlo cross-check, independent of solve_first_crossing:
    generate n_walks independent Markovian random walks (uncorrelated
    Gaussian increments in S, i.e. a Wiener process in S -- matching what
    a sharp-k-windowed excursion-set trajectory actually is) and record
    the S at which each first crosses the barrier B(S). Mirrors Zhang &
    Hui's own validation (their Figure 2: exact solution vs Monte Carlo).

    NORMALIZATION WARNING -- read before computing a crossing probability
    from this function's output. The returned array's LENGTH is the
    number of walks that crossed before S_max, not n_walks: any walk that
    never crosses within the truncated S_max range is silently omitted
    (this is deliberate -- there is no first-crossing S to report for it
    -- but it makes the *count* of returned values a biased denominator).
    To estimate P(cross by some S0 <= S_max) from this output, divide by
    n_walks (the true, full-ensemble denominator), NOT by
    len(simulate_random_walk_first_crossings(...)):

        first_crossings = simulate_random_walk_first_crossings(B, S_max, dS, n_walks)
        p_cross_by_S0 = np.sum(first_crossings <= S0) / n_walks   # correct
        p_cross_by_S0 = np.sum(first_crossings <= S0) / len(first_crossings)  # WRONG -- inflates
                                                                                # the estimate by
                                                                                # 1/P(cross by S_max)

    This is not a hypothetical trap: an earlier validation script made
    exactly this mistake (dividing by the crossed-only count) and got a
    ~1.65x-inflated crossing probability that looked like a genuine
    codebase bug until traced back to the normalization, at a barrier/S_max
    combination where only ~60% of walks crossed before truncation.

    Parameters
    ----------
    B : callable
    S_max : float
    dS : float
        Step size for the walk's own S-grid (independent of any grid used
        by solve_first_crossing -- finer generally gives a more accurate
        first-crossing S per walk, at higher cost).
    n_walks : int
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        The first-crossing S value for each walk that crossed before
        S_max (walks that never cross within S_max are omitted -- see the
        NORMALIZATION WARNING above before dividing by this array's
        length to estimate a probability).
    """
    rng = rng if rng is not None else np.random.default_rng()

    S_grid = np.arange(0.0, S_max + dS, dS)
    delta = np.zeros(n_walks)
    crossed = np.zeros(n_walks, dtype=bool)
    first_crossing_S = np.full(n_walks, np.nan)

    for i in range(1, len(S_grid)):
        step = rng.normal(0.0, np.sqrt(dS), size=n_walks)
        active = ~crossed
        delta[active] += step[active]
        newly_crossed = active & (delta >= B(S_grid[i]))
        first_crossing_S[newly_crossed] = S_grid[i]
        crossed |= newly_crossed

    # NaN entries are walks that never crossed within S_max -- dropping
    # them here means len(return value) != n_walks; see this function's
    # docstring ("NORMALIZATION WARNING") before dividing by the returned
    # array's length to estimate a crossing probability -- divide by the
    # caller's own n_walks instead.
    return first_crossing_S[~np.isnan(first_crossing_S)]
