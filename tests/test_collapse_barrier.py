"""
Tests for foraois.collapse -- the delta_c(M, z, model, cosmo_data)
barrier interface.
"""

import warnings

import numpy as np
import pytest

from foraois.collapse import delta_c
from foraois.collapse.cdm import delta_c_cdm
from foraois.collapse.fdm import delta_c_fdm
from foraois.collapse.sidm import delta_c_sidm
from foraois.collapse.wdm import delta_c_wdm

# ---------------------------------------------------------------------------
# cdm
# ---------------------------------------------------------------------------


def test_cdm_matches_delta_col_at_z_directly(cosmo_data):
    z = np.array([0.0, 1.0, 5.0, 10.0])
    M = np.array([1e10, 1e11, 1e12, 1e13])  # arbitrary, must be ignored
    expected = cosmo_data.delta_col_at_z(z)
    assert np.array_equal(delta_c_cdm(M, z, cosmo_data), expected)


def test_cdm_ignores_mass(cosmo_data):
    z = 5.0
    lo = delta_c_cdm(1e6, z, cosmo_data)
    hi = delta_c_cdm(1e15, z, cosmo_data)
    assert lo == hi


def test_cdm_scalar_z_returns_scalar(cosmo_data):
    result = delta_c_cdm(1e12, 5.0, cosmo_data)
    assert np.isscalar(result) or (hasattr(result, "shape") and result.shape == ())


# ---------------------------------------------------------------------------
# wdm -- currently identical to cdm (Open Question 1, disclosed not verified)
# ---------------------------------------------------------------------------


def test_wdm_currently_identical_to_cdm(cosmo_data):
    z = np.array([0.0, 3.0, 8.0])
    M = np.array([1e9, 1e11, 1e13])
    assert np.array_equal(delta_c_wdm(M, z, cosmo_data), delta_c_cdm(M, z, cosmo_data))


# ---------------------------------------------------------------------------
# fdm -- warned placeholder falling back to cdm
# ---------------------------------------------------------------------------


def test_fdm_falls_back_to_cdm_value(cosmo_data):
    z = np.array([2.0, 6.0])
    M = np.array([1e8, 1e10])
    with pytest.warns(UserWarning, match="placeholder"):
        result = delta_c_fdm(M, z, cosmo_data)
    assert np.array_equal(result, delta_c_cdm(M, z, cosmo_data))


def test_fdm_warns_every_call_not_just_once(cosmo_data):
    # warnings.warn's default "once per location" filter could otherwise
    # hide this from a caller who genuinely wants to know every time --
    # explicitly reset the filter to catch a second call too.
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        with pytest.warns(UserWarning):
            delta_c_fdm(1e10, 5.0, cosmo_data)
        with pytest.warns(UserWarning):
            delta_c_fdm(1e10, 5.0, cosmo_data)


# ---------------------------------------------------------------------------
# sidm -- not implemented
# ---------------------------------------------------------------------------


def test_sidm_raises_not_implemented(cosmo_data):
    with pytest.raises(NotImplementedError, match="SIDM"):
        delta_c_sidm(1e10, 5.0, cosmo_data)


# ---------------------------------------------------------------------------
# delta_c dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["cdm", "wdm"])
def test_delta_c_dispatches_to_correct_model(cosmo_data, model):
    z = np.array([1.0, 4.0])
    M = np.array([1e10, 1e12])
    expected = {"cdm": delta_c_cdm, "wdm": delta_c_wdm}[model](M, z, cosmo_data)
    assert np.array_equal(delta_c(M, z, model, cosmo_data), expected)


def test_delta_c_dispatches_fdm_with_warning(cosmo_data):
    with pytest.warns(UserWarning):
        result = delta_c(1e10, 5.0, "fdm", cosmo_data)
    assert result == delta_c_cdm(1e10, 5.0, cosmo_data)


def test_delta_c_dispatches_sidm_raises(cosmo_data):
    with pytest.raises(NotImplementedError):
        delta_c(1e10, 5.0, "sidm", cosmo_data)


def test_delta_c_unknown_model_raises_value_error(cosmo_data):
    with pytest.raises(ValueError, match="Unknown model"):
        delta_c(1e10, 5.0, "not_a_real_model", cosmo_data)
