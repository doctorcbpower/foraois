"""
Validate zhang_hui_trees.py's actual barrier construction (collapse.delta_c
fed into solve_first_crossing, the same way first_crossing_step builds it)
against a direct Monte Carlo random walk, independent of
solve_first_crossing's own internals.

This is deliberately narrower than "a full tree": build_tree re-roots the
walk's origin at delta_c(M_cur, z_cur) after every step (the same
approximation PCH08's own algorithm makes -- see zhang_hui_trees.py's
module docstring), so each step is itself the thing to validate, and a
correct single step, applied repeatedly, is what build_tree already is.
scripts/validate_zhang_hui_vs_pch08.py separately characterizes the
resulting full-tree statistics against PCH08.

Deliberately does not go through first_crossing_step's own S_max=
S_res*S_max_factor grid sizing: that ties the grid to M_res, which for a
large M0/M_res ratio can be far coarser than the first-crossing
distribution's actual width (~B(0)^2) -- exactly the failure mode
first_crossing_step's own grid-resolution warning catches. This test
instead solves on a small, appropriately-resolved S_max chosen for *this*
comparison's accuracy, decoupling "does the barrier-construction + solver
pipeline match Monte Carlo" from that separate M_res/S_res grid-sizing
concern.
"""

import numpy as np
import pytest

from foraois.first_crossing import (
    simulate_random_walk_first_crossings,
    solve_first_crossing,
)


def test_cdm_step_cumulative_probability_matches_direct_monte_carlo(zh_tree_generator):
    # Same pattern as test_first_crossing.py's own
    # test_nonlinear_barrier_matches_monte_carlo_cumulative, but against
    # the real barrier construction zhang_hui_trees.py actually builds
    # (via collapse.delta_c/CosmoData), not a synthetic one -- an
    # independent check of the whole pipeline, not just the solver.
    cosmo_data = zh_tree_generator.cosmo_data
    z0, z1 = 0.0, 0.5
    d_omega = float(cosmo_data.delta_col_at_z(z1) - cosmo_data.delta_col_at_z(z0))
    B = lambda S: np.full_like(np.asarray(S, dtype=float), d_omega)
    dBdS = lambda S: np.zeros_like(np.asarray(S, dtype=float))

    # S_max=30 comfortably covers where most of the flat-barrier crossing
    # probability accumulates for this d_omega (checked directly: cumulative
    # already exceeds 0.9 by S=27); N_grid=3000 gives dS=0.01, well below
    # B(0)^2 ~= d_omega^2 ~= 0.25.
    S_max, N_grid = 30.0, 3000
    S_grid, f = solve_first_crossing(B, S_max, N_grid, dBdS=dBdS)
    F_cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(S_grid))])

    # The MC simulator has its own discretization bias, set by its dS
    # relative to the same B(0)^2 scale -- checked directly that dS=0.01
    # (matching solve_first_crossing's own grid) still carries several-
    # percent bias here; dS=0.002 was needed to bring it under this test's
    # tolerance. n_walks kept modest (dS this fine is the expensive part)
    # since statistical noise at 100k walks (~0.003) is subdominant to the
    # residual discretization bias anyway.
    rng = np.random.default_rng(123)
    n_walks = 100_000
    mc = simulate_random_walk_first_crossings(B, S_max=S_max, dS=0.002, n_walks=n_walks, rng=rng)

    # Excludes frac=0.1 (S=3): both methods carry the largest residual bias
    # closest to S=0 (same near-S0 noise test_first_crossing.py's own
    # nonlinear-barrier test documents and excludes), where the walk has
    # had the least variance to average over.
    for frac in [0.3, 0.5, 0.7, 0.9]:
        S_test = frac * S_max
        cum_mc = np.sum(mc <= S_test) / n_walks
        cum_solver = float(np.interp(S_test, S_grid, F_cum))
        assert cum_solver == pytest.approx(cum_mc, abs=0.015)


def test_eps_expected_splits_scale_linearly_with_small_steps(zh_tree_generator):
    """E is the EPS rate times the step: halving the step halves it (small-step limit), and a lower M_res raises it."""
    from foraois.diagnostics import expected_eps_splits_per_step

    cd = zh_tree_generator.cosmo_data
    M0 = 1e12
    e1 = expected_eps_splits_per_step(cd, M0, 0.0, 0.002, M0 * 1e-3)
    e2 = expected_eps_splits_per_step(cd, M0, 0.0, 0.001, M0 * 1e-3)
    assert e1 > 0.0
    assert e2 == pytest.approx(0.5 * e1, rel=0.05)
    assert expected_eps_splits_per_step(cd, M0, 0.0, 0.002, M0 * 1e-4) > e1
    assert expected_eps_splits_per_step(cd, M0, 0.0, 0.002, 0.6 * M0) == 0.0
