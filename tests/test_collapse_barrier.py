"""
Tests for foraois.collapse -- the collapse-barrier prescription delta_c(M, z, cosmo_data), which is selected by the
config keys ``barrier`` / ``barrier_parameter`` and is independent of the dark-matter model.

``fixed``: delta_c = delta_sc(z). ``linear``: delta_c = delta_sc(z) + beta * sigma^2(M), an illustrative
scale-dependent barrier (not a calibrated collapse model), with beta = ``barrier_parameter`` (default 0.15).
"""

import numpy as np
import pytest

from foraois.collapse import barrier_is_flat, barrier_settings, delta_c, require_flat_barrier
from foraois.collapse.barrier import DEFAULT_BARRIER, DEFAULT_BARRIER_PARAMETER
from foraois.cosmo_utils import CosmoData
from foraois.utils import io

MASSES = np.array([1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15])
REDSHIFTS = np.array([0.0, 1.0, 5.0])

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_defaults_are_fixed_barrier_and_beta_0p15():
    assert DEFAULT_BARRIER == "fixed"
    assert DEFAULT_BARRIER_PARAMETER == 0.15
    assert barrier_settings({}) == ("fixed", 0.15)


# ---------------------------------------------------------------------------
# fixed barrier
# ---------------------------------------------------------------------------


def test_fixed_equals_spherical_collapse_threshold(cosmo_data):
    # No barrier keys at all (a hand-built config): the default is the fixed barrier.
    assert np.array_equal(delta_c(MASSES[:3], REDSHIFTS, cosmo_data), cosmo_data.delta_col_at_z(REDSHIFTS))


def test_fixed_ignores_mass(cosmo_data):
    assert delta_c(1e6, 5.0, cosmo_data) == delta_c(1e15, 5.0, cosmo_data)


def test_fixed_scalar_z_returns_scalar(cosmo_data):
    result = delta_c(1e12, 5.0, cosmo_data)
    assert np.isscalar(result) or (hasattr(result, "shape") and result.shape == ())


def test_fixed_ignores_barrier_parameter(cosmo_data, set_barrier):
    reference = delta_c(MASSES[:3], REDSHIFTS, cosmo_data)
    set_barrier(cosmo_data, "fixed", beta=5.0)
    assert np.array_equal(delta_c(MASSES[:3], REDSHIFTS, cosmo_data), reference)


# ---------------------------------------------------------------------------
# linear barrier: delta_c = delta_sc(z) + beta * sigma^2(M)
# ---------------------------------------------------------------------------


def _sigma_sq(cosmo_data, M):
    return cosmo_data.sigma_at_logmass(np.log10(M)) ** 2


def test_linear_matches_formula_with_default_beta(tree_generator, set_barrier):
    cosmo_data = tree_generator.cosmo_data  # tree_generator builds the sigma(M) table
    set_barrier(cosmo_data, "linear")  # barrier_parameter not given: the default applies
    expected = cosmo_data.delta_col_at_z(2.0) + 0.15 * _sigma_sq(cosmo_data, MASSES)
    assert np.allclose(delta_c(MASSES, 2.0, cosmo_data), expected, rtol=1e-12)


def test_linear_rises_towards_smaller_mass(tree_generator, set_barrier):
    cosmo_data = tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.15)
    values = delta_c(MASSES, 1.0, cosmo_data)  # MASSES ascending
    assert np.all(np.diff(values) < 0.0)
    assert np.all(values > cosmo_data.delta_col_at_z(1.0))


def test_changing_beta_changes_the_barrier(tree_generator, set_barrier):
    cosmo_data = tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.15)
    low = delta_c(MASSES, 1.0, cosmo_data)
    set_barrier(cosmo_data, "linear", beta=0.30)
    high = delta_c(MASSES, 1.0, cosmo_data)
    assert np.all(high > low)
    assert np.allclose(high - low, 0.15 * _sigma_sq(cosmo_data, MASSES), rtol=1e-10)


def test_linear_with_zero_beta_equals_fixed(tree_generator, set_barrier):
    cosmo_data = tree_generator.cosmo_data
    fixed = delta_c(MASSES, REDSHIFTS[:, None], cosmo_data)
    set_barrier(cosmo_data, "linear", beta=0.0)
    assert np.array_equal(delta_c(MASSES, REDSHIFTS[:, None], cosmo_data), fixed)


def test_linear_barrier_below_the_sigma_table_is_held_at_the_floor_value(tree_generator, set_barrier):
    cosmo_data = tree_generator.cosmo_data
    set_barrier(cosmo_data, "linear", beta=0.15)
    floor = 10.0 ** cosmo_data._logmass[0]
    below = delta_c(np.array([0.0, 1e-30, 1.0, 0.5 * floor]), 1.0, cosmo_data)
    assert np.all(np.isfinite(below))
    assert np.allclose(below, delta_c(floor, 1.0, cosmo_data))


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_unknown_barrier_name_raises(cosmo_data, set_barrier):
    set_barrier(cosmo_data, "flat")  # the WIP branch's old name for 'fixed'
    with pytest.raises(ValueError, match="Unknown barrier"):
        delta_c(1e12, 1.0, cosmo_data)


def test_negative_beta_raises_for_linear_only(cosmo_data, set_barrier):
    set_barrier(cosmo_data, "linear", beta=-0.1)
    with pytest.raises(ValueError, match="barrier_parameter >= 0"):
        barrier_settings(cosmo_data.run_params)
    set_barrier(cosmo_data, "fixed", beta=-0.1)  # ignored for the fixed barrier
    assert barrier_settings(cosmo_data.run_params)[0] == "fixed"


# ---------------------------------------------------------------------------
# the DM-model label no longer selects the barrier (legacy 4-argument call)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["cdm", "wdm", "fdm", "sidm", "anything"])
def test_legacy_model_argument_is_accepted_and_ignored(cosmo_data, model):
    assert np.array_equal(delta_c(MASSES[:3], 1.0, model, cosmo_data), delta_c(MASSES[:3], 1.0, cosmo_data))


def test_delta_c_wrong_argument_count_raises(cosmo_data):
    with pytest.raises(TypeError):
        delta_c(1e12, 1.0)


def test_object_without_configured_barrier_is_the_fixed_barrier():
    # A historical duck-typed stand-in has no run_params at all; it must not be assumed to have one.
    class Bare:
        def delta_col_at_z(self, z):
            return 1.686 * (1.0 + np.asarray(z))

    bare = Bare()
    assert barrier_is_flat(bare)
    require_flat_barrier(bare, "SomeAlgorithm")
    assert np.array_equal(delta_c(1e12, np.array([0.0, 1.0]), bare), bare.delta_col_at_z(np.array([0.0, 1.0])))


# ---------------------------------------------------------------------------
# flatness predicate and guard
# ---------------------------------------------------------------------------


def test_flatness_predicate(cosmo_data, set_barrier):
    assert barrier_is_flat(cosmo_data)
    set_barrier(cosmo_data, "linear", beta=0.0)
    assert barrier_is_flat(cosmo_data)
    set_barrier(cosmo_data, "linear", beta=0.15)
    assert not barrier_is_flat(cosmo_data)


def test_require_flat_barrier_error_names_request_and_algorithm(cosmo_data, set_barrier):
    set_barrier(cosmo_data, "linear", beta=0.15)
    with pytest.raises(NotImplementedError, match=r"SomeAlgorithm.*fixed.*barrier='linear'.*0\.15"):
        require_flat_barrier(cosmo_data, "SomeAlgorithm")
    set_barrier(cosmo_data, "linear", beta=0.0)
    require_flat_barrier(cosmo_data, "SomeAlgorithm")  # beta = 0 is the fixed barrier: accepted


# ---------------------------------------------------------------------------
# configuration controls the barrier used in the calculation
# ---------------------------------------------------------------------------

RAW = {
    "Run": {"mode": "camb", "barrier": "linear", "barrier_parameter": 0.3},
    "Cosmology": {"H0": 67.66, "OmegaM": 0.3111, "OmegaBar": 0.049, "As": 2.1e-9, "ns": 0.965, "tau_reio": 0.056},
    "camb": {},
}


def test_config_values_control_the_barrier(synthetic_pk_data):
    cosmo_data = CosmoData(io.params_from_dict(RAW), redshift=[0.0])
    cosmo_data._prepare_sigma_grid(synthetic_pk_data)
    expected = cosmo_data.delta_col_at_z(1.0) + 0.3 * _sigma_sq(cosmo_data, MASSES)
    assert np.allclose(delta_c(MASSES, 1.0, cosmo_data), expected, rtol=1e-12)


def test_from_params_barrier_keywords_reach_the_config():
    cosmo_data = CosmoData.from_params(
        H0=67.66, OmegaM=0.3111, OmegaBar=0.049, As=2.1e-9, ns=0.965, tau=0.056, barrier="linear", barrier_parameter=0.2
    )
    assert barrier_settings(cosmo_data.run_params) == ("linear", 0.2)
    default = CosmoData.from_params(H0=67.66, OmegaM=0.3111, OmegaBar=0.049, As=2.1e-9, ns=0.965, tau=0.056)
    assert barrier_settings(default.run_params) == ("fixed", 0.15)
