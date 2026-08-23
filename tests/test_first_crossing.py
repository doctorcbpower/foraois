"""
Tests for first_crossing.py -- Zhang & Hui (2006)'s exact first-crossing
distribution for Markovian excursion-set random walks with an arbitrary
moving barrier.

Validation strategy (deliberately independent of any of this module's own
internals): compare solve_first_crossing's numerical solution against (1)
two closed-form analytic solutions (flat and linear barriers, both
special cases with known exact answers), and (2) a direct Monte Carlo
random walk simulation for a genuinely nonlinear barrier, reproducing
Zhang & Hui's own Figure 2 cross-check. No CLASS/CAMB or cosmology
backend is needed anywhere in this file -- it's pure random-walk math.
"""

import numpy as np
import pytest

from foraois.first_crossing import (
    flat_barrier_first_crossing,
    linear_barrier_first_crossing,
    simulate_random_walk_first_crossings,
    solve_first_crossing,
)


def _flat_barrier(delta_c):
    B = lambda S: np.full_like(np.asarray(S, dtype=float), delta_c)
    dBdS = lambda S: np.zeros_like(np.asarray(S, dtype=float))
    return B, dBdS


def _linear_barrier(a, b):
    B = lambda S: a + b * np.asarray(S, dtype=float)
    dBdS = lambda S: np.full_like(np.asarray(S, dtype=float), b)
    return B, dBdS


def test_flat_barrier_matches_press_schechter_analytic():
    # The flat barrier is the standard Press-Schechter case, with a known
    # closed-form solution (Zhang & Hui eq. 18, a=delta_c, b=0) -- the
    # strongest possible check, since it pins down the exact numbers, not
    # just self-consistency.
    delta_c = 1.686
    B, dBdS = _flat_barrier(delta_c)
    S_grid, f = solve_first_crossing(B, S_max=5.0, N=500, dBdS=dBdS)
    f_exact = flat_barrier_first_crossing(S_grid, delta_c)

    mask = S_grid > 0.05  # skip S~0 where both curves are tiny and float noise dominates
    rel_err = np.abs(f[mask] - f_exact[mask]) / f_exact[mask]
    assert rel_err.max() < 1e-10


def test_flat_barrier_f_is_zero_at_S0():
    B, dBdS = _flat_barrier(1.686)
    S_grid, f = solve_first_crossing(B, S_max=5.0, N=200, dBdS=dBdS)
    assert f[0] == 0.0


def test_flat_barrier_integral_approaches_one_as_S_max_grows():
    # The Press-Schechter first-crossing distribution integrates to
    # exactly 1 over S in [0, infinity) -- every walk eventually crosses a
    # flat barrier (this is the historical "missing factor of 2" result
    # resolved by Bond et al. 1991's excursion-set derivation). The
    # integral should grow monotonically toward 1 as S_max increases.
    B, dBdS = _flat_barrier(1.686)
    totals = []
    for S_max in [5.0, 20.0, 40.0]:
        N = int(S_max * 50)
        S_grid, f = solve_first_crossing(B, S_max=S_max, N=N, dBdS=dBdS)
        totals.append(np.trapezoid(f, S_grid))
    assert np.all(np.diff(totals) > 0)
    assert totals[-1] > 0.75
    assert totals[-1] < 1.0 + 1e-6


def test_linear_barrier_matches_analytic_inverse_gaussian():
    a, b = 1.2, 0.15
    B, dBdS = _linear_barrier(a, b)
    S_grid, f = solve_first_crossing(B, S_max=5.0, N=2000, dBdS=dBdS)
    f_exact = linear_barrier_first_crossing(S_grid, a, b)

    mask = S_grid > 0.05
    rel_err = np.abs(f[mask] - f_exact[mask]) / f_exact[mask]
    assert rel_err.max() < 5e-3


def test_linear_barrier_converges_with_finer_grid():
    # The forward-substitution scheme's error should shrink as N grows
    # (checked against the analytic solution, independent of any Monte
    # Carlo noise) -- a basic sanity check on the numerical method itself.
    a, b = 1.2, 0.15
    B, dBdS = _linear_barrier(a, b)
    errs = []
    for N in [250, 1000]:
        S_grid, f = solve_first_crossing(B, S_max=5.0, N=N, dBdS=dBdS)
        f_exact = linear_barrier_first_crossing(S_grid, a, b)
        mask = S_grid > 0.05
        errs.append(np.max(np.abs(f[mask] - f_exact[mask]) / f_exact[mask]))
    assert errs[1] < errs[0]


def test_numeric_dBdS_fallback_matches_analytic_derivative():
    # dBdS is optional (finite-difference fallback) -- check it gives
    # essentially the same answer as supplying the exact derivative, for
    # a barrier where both are available.
    a, b = 1.2, 0.15
    B, dBdS_exact = _linear_barrier(a, b)
    S_grid_exact, f_exact = solve_first_crossing(B, S_max=5.0, N=500, dBdS=dBdS_exact)
    S_grid_fd, f_fd = solve_first_crossing(B, S_max=5.0, N=500, dBdS=None)

    mask = S_grid_exact > 0.05
    rel_err = np.abs(f_fd[mask] - f_exact[mask]) / np.maximum(np.abs(f_exact[mask]), 1e-12)
    assert rel_err.max() < 1e-4


def test_nonlinear_barrier_matches_monte_carlo_cumulative():
    # Zhang & Hui's own Figure 2 test case: B(S) = 1 + 0.3 S + 0.3 S^2 --
    # no closed-form solution exists for a nonlinear barrier, so this
    # cross-checks against an independent Monte Carlo random walk instead
    # (mirroring the paper's own validation). Comparing the CUMULATIVE
    # crossing probability (rather than per-bin density) avoids both
    # methods' known small-S/tail noise (the paper notes the same
    # per-bin discrepancy in their own Fig. 2) and tests the physically
    # meaningful quantity directly.
    B = lambda S: 1.0 + 0.3 * np.asarray(S, dtype=float) + 0.3 * np.asarray(S, dtype=float) ** 2
    dBdS = lambda S: 0.3 + 0.6 * np.asarray(S, dtype=float)

    S_grid, f = solve_first_crossing(B, S_max=3.0, N=1500, dBdS=dBdS)
    cum_solver = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(S_grid))])

    n_walks = 100_000
    rng = np.random.default_rng(42)
    mc = simulate_random_walk_first_crossings(B, S_max=3.0, dS=0.002, n_walks=n_walks, rng=rng)

    for S_test in [0.5, 1.0, 1.5, 2.0, 2.5]:
        cum_mc = np.sum(mc <= S_test) / n_walks
        cum_solver_at = np.interp(S_test, S_grid, cum_solver)
        assert cum_solver_at == pytest.approx(cum_mc, abs=0.02)


def test_solve_first_crossing_never_strongly_negative():
    # f(S) is a probability density and must be >= 0 everywhere in
    # principle; the forward-substitution scheme can produce small
    # negative numerical noise in the far tail where the true value is
    # tiny (a real, convergent-with-N discretization artifact -- see this
    # test's tolerance), but nothing resembling a real (non-noise)
    # negative excursion.
    B = lambda S: 1.0 + 0.3 * np.asarray(S, dtype=float) + 0.3 * np.asarray(S, dtype=float) ** 2
    dBdS = lambda S: 0.3 + 0.6 * np.asarray(S, dtype=float)
    S_grid, f = solve_first_crossing(B, S_max=3.0, N=3000, dBdS=dBdS)
    # Observed noise floor at this resolution is ~-2e-3 (shrinks with N --
    # see the module docstring); this bounds it an order of magnitude
    # looser, catching a real regression without being a tight tripwire.
    assert f.min() > -2e-2


def test_barrier_must_be_positive_at_S0_for_sensible_walk():
    # Not a hard requirement enforced by the code (B is caller-supplied),
    # but flat_barrier_first_crossing/linear_barrier_first_crossing both
    # assume B(0) > 0 -- a walk starting at delta=0 below a barrier at or
    # below 0 has already "crossed" trivially, which isn't a meaningful
    # excursion-set setup. This documents the assumption via example
    # rather than asserting it's enforced.
    S = np.array([0.5, 1.0])
    f = flat_barrier_first_crossing(S, delta_c=1.686)
    assert np.all(f > 0)
