"""
Tests for foraois.first_crossing_constrained -- the
Brownian-bridge-constrained first-crossing solver (N23's Eq. A5), solved
via a direct midpoint discretization independent of first_crossing.py's
own g1/g2 scheme.
"""

import numpy as np
import pytest

from foraois.first_crossing import flat_barrier_first_crossing
from foraois.first_crossing_constrained import (
    simulate_constrained_random_walk_first_crossings,
    solve_first_crossing_constrained,
    solve_first_crossing_unconstrained_midpoint,
)


def _flat_barrier(delta_c):
    return lambda S: np.full_like(np.asarray(S, dtype=float), delta_c)


# ---------------------------------------------------------------------------
# Unconstrained midpoint solver -- validates the shared numerical method
# against a known closed form, independent of solve_first_crossing.
# ---------------------------------------------------------------------------


def test_unconstrained_midpoint_matches_closed_form():
    delta_c = 1.686
    B = _flat_barrier(delta_c)
    S_mid, f = solve_first_crossing_unconstrained_midpoint(B, S_max=5.0, N=500)
    f_exact = flat_barrier_first_crossing(S_mid, delta_c)

    # Exclude the near-S=0 tail (float-noise regime, same exclusion
    # first_crossing.py's own tests use) -- away from it, agreement with
    # the closed form should be tight.
    mask = S_mid > 0.3
    rel_err = np.abs(f[mask] - f_exact[mask]) / f_exact[mask]
    assert rel_err.max() < 2e-4


# ---------------------------------------------------------------------------
# Constrained solver
# ---------------------------------------------------------------------------


def test_constrained_raises_when_S1_not_greater_than_S0():
    B = _flat_barrier(1.686)
    with pytest.raises(ValueError):
        solve_first_crossing_constrained(B, 0.0, 0.0, 0.0, 1.686, N=100)
    with pytest.raises(ValueError):
        solve_first_crossing_constrained(B, 5.0, 0.0, 3.0, 1.686, N=100)


def test_constrained_normalizes_to_one():
    # The bridge reaches delta1 by S1 with probability exactly 1 --
    # checked for several (S1, N) combinations, not just the one used
    # elsewhere.
    delta_c = 1.686
    B = _flat_barrier(delta_c)
    for S1, N in [(5.0, 200), (10.0, 500), (50.0, 800)]:
        _, f = solve_first_crossing_constrained(B, 0.0, 0.0, S1, delta_c, N=N)
        assert np.sum(f) * (S1 / N) == pytest.approx(1.0, abs=1e-8)


def test_constrained_no_nan_and_non_negative():
    delta_c = 1.686
    B = _flat_barrier(delta_c)
    _, f = solve_first_crossing_constrained(B, 0.0, 0.0, 10.0, delta_c, N=500)
    assert not np.isnan(f).any()
    assert (f >= -1e-8).all()


def test_constrained_peaks_near_S1():
    # Figure 2 of N23: the constrained rate is similar to the
    # unconstrained one at small S, and peaks/cuts off near S1 -- a
    # qualitative shape check independent of the exact numbers.
    delta_c = 1.686
    B = _flat_barrier(delta_c)
    S_mid, f = solve_first_crossing_constrained(B, 0.0, 0.0, 10.0, delta_c, N=500)
    assert S_mid[np.argmax(f)] > 8.0


def test_constrained_converges_to_unconstrained_as_S1_grows():
    # The key physics check (N23's own Appendix B1 convergence test):
    # for S well below S1, the constrained distribution should approach
    # the unconstrained one as S1 -> infinity. Grid spacing (dS) is held
    # fixed across S1 values (N scaled with S1): letting dS grow with S1
    # at fixed N would instead be a pure resolution artifact, not a real
    # divergence.
    delta_c = 1.686
    B = _flat_barrier(delta_c)
    S_test = 2.0

    S_mid_u, f_u = solve_first_crossing_unconstrained_midpoint(B, S_max=5.0, N=500)
    f_u_at_test = np.interp(S_test, S_mid_u, f_u)

    dS_fixed = 0.05
    ratios = []
    for S1 in [50.0, 200.0, 1000.0]:
        N = int(S1 / dS_fixed)
        delta1 = float(B(np.array(S1)))
        S_mid_c, f_c = solve_first_crossing_constrained(B, 0.0, 0.0, S1, delta1, N=N)
        f_c_at_test = np.interp(S_test, S_mid_c, f_c)
        ratios.append(f_c_at_test / f_u_at_test)

    # Deviation from 1 should shrink monotonically as S1 grows, and the
    # largest S1 should be very close to the unconstrained value --
    # checked directly: [1.050, 1.012, 1.002] for these S1 values, so a
    # loose bound on the first point (still far from S1=infinity) and a
    # tight one on the last is the right shape of check, not a uniform
    # tolerance across all three.
    deviations = [abs(r - 1.0) for r in ratios]
    assert deviations[0] < 0.1
    assert deviations[-1] < 0.01
    assert deviations == sorted(deviations, reverse=True)


def test_constrained_matches_direct_monte_carlo():
    # Independent cross-check, unrelated to the S1->infinity convergence
    # test above and sharing no machinery with either the analytic bridge
    # formulas or the solver itself: simulate genuinely unconstrained
    # random walks, keep only ones landing near (S1, delta1) at S1
    # (rejection sampling for the bridge condition), and compare their
    # first-crossing-S distribution to the solver's.
    #
    # Checked directly (not just at this tolerance): the residual shrinks
    # with finer MC step size / tighter acceptance window (e.g. dS=0.005,
    # tol=0.05, n_walks=3e6 gives ~0.01-0.017 absolute agreement instead
    # of ~0.02) -- consistent with discretization/window noise shrinking
    # as expected, not a persistent discrepancy. abs=0.03 here (looser
    # than first_crossing.py's own abs=0.02 MC tolerance, since this
    # check also carries the rejection-window's own smoothing bias on
    # top of ordinary MC noise) is set from the settings that actually
    # ran in a few seconds, not tuned to just barely pass.
    delta_c = 1.686
    B = _flat_barrier(delta_c)
    S1 = 5.0
    delta1 = float(B(np.array(S1)))

    rng = np.random.default_rng(42)
    mc = simulate_constrained_random_walk_first_crossings(
        B,
        S1,
        delta1,
        dS=0.01,
        n_walks=1_000_000,
        tol=0.1,
        rng=rng,
    )
    assert len(mc) > 1000  # sanity: the rejection window actually accepted enough walks

    S_mid, f = solve_first_crossing_constrained(B, 0.0, 0.0, S1, delta1, N=500)
    F_cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(S_mid))])

    for S_test in [0.5, 1.0, 2.0, 3.0, 4.0, 4.5]:
        cum_mc = np.mean(mc <= S_test)
        cum_solver = float(np.interp(S_test, S_mid, F_cum))
        assert cum_solver == pytest.approx(cum_mc, abs=0.03)
