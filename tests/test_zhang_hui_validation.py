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


# ---------------------------------------------------------------------------
# Linear (scale-dependent) barrier: the general first-crossing path against the analytic solution
#
# delta_c(M, z) = delta_sc(z) + beta*sigma^2(M) gives, with S = sigma^2(M) - sigma^2(M0), the shifted barrier
# B(S) = a + beta*S with a = delta_sc(z1) - delta_sc(z0), whose first-crossing density is known exactly
# (first_crossing.linear_barrier_first_crossing). first_crossing_step is the path build_tree uses.
#
# What this establishes (development finding, see docs/MODELS.md): the numerical error is a resolution effect, set by
# the grid spacing dS = S_max/N_grid relative to a^2 = B(0)^2, and it converges at about order 1/2 in N_grid. It is
# proportional to beta and vanishes for beta = 0 (where g2 = 0). The pointwise relative error is not used as the
# metric because it is dominated by the tail and can change sign as N_grid grows; the peak-normalised error and the
# error of the crossing probability at S_res (the quantity p_res that build_tree uses) are.
# ---------------------------------------------------------------------------

from scipy.integrate import cumulative_trapezoid  # noqa: E402
from scipy.stats import norm  # noqa: E402

from foraois.collapse import barrier_settings  # noqa: E402
from foraois.first_crossing import linear_barrier_first_crossing  # noqa: E402
from foraois.zhang_hui_trees import first_crossing_step  # noqa: E402

_LB_M0, _LB_M_RES = 1.0e12, 1.0e11


def _linear_barrier_cdf(S, a, b):
    """Exact first-crossing CDF for B(S) = a + b*S (checked against the integral of the analytic density)."""
    S = np.asarray(S, dtype=float)
    return norm.cdf((-a - b * S) / np.sqrt(S)) + np.exp(-2.0 * a * b) * norm.cdf((-a + b * S) / np.sqrt(S))


def _step_errors(cosmo_data, z1, N_grid):
    """(peak-normalised error of f, |error| of the crossing probability at S_res, dS/a^2) for one step."""
    beta = barrier_settings(cosmo_data.run_params)[1]
    a = float(cosmo_data.delta_col_at_z(z1) - cosmo_data.delta_col_at_z(0.0))
    step = first_crossing_step(_LB_M0, 0.0, z1, _LB_M_RES, cosmo_data, N_grid=N_grid, S_max_factor=2.0)
    S, f, S_res = step["S_grid"], step["f"], step["S_res"]
    f_exact = linear_barrier_first_crossing(S, a, beta)
    m = (S > 0.05) & (S <= S_res)
    peak_norm = np.abs(f[m] - f_exact[m]).max() / f_exact[m].max()
    cdf_numeric = float(np.interp(S_res, S, cumulative_trapezoid(f, S, initial=0.0)))
    cdf_err = abs(cdf_numeric - float(_linear_barrier_cdf(S_res, a, beta)))
    return peak_norm, cdf_err, S[1] / a**2


@pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")
def test_linear_barrier_step_matches_analytic_solution_when_resolved(zh_tree_generator, set_barrier):
    # Tested regime: z0 -> z1 = 2 (a^2 = 5.5), M0 = 1e12, M_res = 1e11 Msun/h, S_max = 2*S_res, beta = 0.15 (the
    # default), N_grid = 400, i.e. dS/a^2 = 0.005. Measured on this fixture: peak-normalised error 3.5e-3, CDF error
    # 8.1e-4; the bounds below leave a modest margin.
    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear")
    peak_norm, cdf_err, ratio = _step_errors(cosmo_data, 2.0, 400)
    assert ratio < 0.01
    assert peak_norm < 5e-3
    assert cdf_err < 1.5e-3


@pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")
def test_linear_barrier_error_decreases_with_resolution(zh_tree_generator, set_barrier):
    # The error falls monotonically as N_grid doubles (measured: about a factor 0.71 per doubling, i.e. order ~1/2),
    # so the discrepancy is discretisation error that goes away with resolution, not an implementation error.
    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear")
    errors = [_step_errors(cosmo_data, 2.0, N)[:2] for N in (100, 200, 400, 800)]
    for metric in (0, 1):
        series = [e[metric] for e in errors]
        assert all(later < earlier for earlier, later in zip(series, series[1:], strict=False))
        assert series[0] / series[-1] > 2.0  # 8x the resolution; order 1/2 predicts ~2.8


@pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")
def test_zero_beta_reproduces_the_flat_barrier_solution_to_machine_precision(zh_tree_generator, set_barrier):
    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.0)
    peak_norm, _, _ = _step_errors(cosmo_data, 2.0, 200)
    assert peak_norm < 1e-12


@pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")
def test_linear_barrier_error_depends_on_grid_spacing_relative_to_barrier_scale(zh_tree_generator, set_barrier):
    # The same physical problem at a smaller barrier offset (z1 = 0.5, a^2 = 0.25): when dS/a^2 is large the
    # crossing probability is badly wrong (order 0.1 at dS/a^2 ~ 0.9), and it becomes accurate as the grid resolves
    # the peak of f(S). This documents the resolution dependence; it is a numerical requirement of the general
    # solver, not a restriction imposed by the code.
    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear")
    coarse = _step_errors(cosmo_data, 0.5, 50)
    medium = _step_errors(cosmo_data, 0.5, 100)
    fine = _step_errors(cosmo_data, 0.5, 800)
    assert coarse[2] > 0.8 and medium[2] > 0.4 and fine[2] < 0.06
    assert coarse[1] > 0.1
    assert medium[1] > 10 * fine[1]
    assert fine[1] < 5e-3


# ---------------------------------------------------------------------------
# Diagnostics that assume a flat barrier
# ---------------------------------------------------------------------------


def test_eps_expected_splits_rejects_a_scale_dependent_barrier(zh_tree_generator, set_barrier):
    from foraois.diagnostics import expected_eps_splits_per_step

    cosmo_data = zh_tree_generator.cosmo_data
    reference = expected_eps_splits_per_step(cosmo_data, 1e12, 0.0, 0.05, 1e10)
    set_barrier(cosmo_data, "linear", beta=0.0)
    assert expected_eps_splits_per_step(cosmo_data, 1e12, 0.0, 0.05, 1e10) == reference  # beta = 0 is fixed
    set_barrier(cosmo_data, "linear", beta=0.15)
    with pytest.raises(NotImplementedError, match="expected_eps_splits_per_step"):
        expected_eps_splits_per_step(cosmo_data, 1e12, 0.0, 0.05, 1e10)


# ---------------------------------------------------------------------------
# The linear barrier beyond the sigma(M) table
#
# With a finite pk_kmax, sigma^2(M) saturates at small mass, and the sigma(M) table stops at 100 Msun/h, so only S up
# to sigma^2(table floor) - sigma^2(M0) is representable. first_crossing_step's grid runs to S_max_factor*S_res, which
# can exceed that. A mass-independent barrier never looks at the mass there, but the linear barrier needs sigma^2(M):
# without a floor the mass mapping underflows, delta_c becomes inf and the step returned a negative p_res and NaN F_zh.
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")
def test_linear_barrier_step_stays_finite_when_the_grid_exceeds_the_sigma_table(zh_tree_generator, set_barrier):
    cosmo_data = zh_tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear")
    s0 = float(cosmo_data.sigma_at_logmass(np.log10(1e12))) ** 2
    s_table_max = float(cosmo_data.sigma_at_logmass(cosmo_data._logmass[0])) ** 2 - s0
    step = first_crossing_step(1e12, 0.0, 0.5, 1e10, cosmo_data, N_grid=100, S_max_factor=8.0)
    assert step["S_grid"][-1] > s_table_max  # the grid does extend past what the table represents
    # Finiteness is what is asserted. This grid is deliberately coarse (dS/a^2 ~ 5, far outside the resolved regime
    # tested above, and first_crossing_step warns), so f and p_res are not accurate here and p_res can even be
    # slightly negative; that is the documented under-resolved behaviour, not the table-floor problem.
    assert np.all(np.isfinite(step["f"]))
    assert np.isfinite(step["p_res"]) and np.isfinite(step["F_zh"])
    assert 0.0 <= step["F_zh"] <= 1.0  # first_crossing_step clips the unresolved fraction
