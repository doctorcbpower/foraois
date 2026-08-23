"""
Tests for foraois.mass_function_utils.MassFunctions -- regression coverage
for two bugs found in a pre-release audit: fst()/dlnnu_dlnm() returned a
ValueError instance instead of raising it on invalid input, and __init__'s
redshift guard (`np.any(self.redshift) < 0`, a bool compared to an int,
always False) never actually fired.
"""

import numpy as np
import pytest

from foraois.mass_function_utils import MassFunctions


@pytest.fixture
def mass_functions(tree_generator):
    return MassFunctions(tree_generator.cosmo_data, redshift=[0.0])


def test_negative_redshift_raises_in_constructor(tree_generator):
    with pytest.raises(ValueError):
        MassFunctions(tree_generator.cosmo_data, redshift=[-1.0])


def test_fps_rejects_non_positive_mass(mass_functions):
    with pytest.raises(ValueError):
        mass_functions.fps(0.0, 0.0)


def test_fps_rejects_negative_redshift(mass_functions):
    with pytest.raises(ValueError):
        mass_functions.fps(1e12, -1.0)


def test_fst_raises_not_returns_on_non_positive_mass(mass_functions):
    # Regression: fst() used `return ValueError(...)` instead of `raise`,
    # so this call used to hand back an unraised exception object.
    with pytest.raises(ValueError):
        mass_functions.fst(0.0, 0.0)


def test_fst_raises_not_returns_on_negative_redshift(mass_functions):
    with pytest.raises(ValueError):
        mass_functions.fst(1e12, -1.0)


def test_dlnnu_dlnm_raises_not_returns_on_non_positive_mass(mass_functions):
    # Regression: dlnnu_dlnm() had the same return-instead-of-raise bug.
    with pytest.raises(ValueError):
        mass_functions.dlnnu_dlnm(0.0, 0.0)


def test_dlnnu_dlnm_raises_not_returns_on_negative_redshift(mass_functions):
    with pytest.raises(ValueError):
        mass_functions.dlnnu_dlnm(1e12, -1.0)


def test_fps_matches_closed_form(mass_functions):
    # f_PS(nu) = sqrt(2/pi) * nu * exp(-nu^2/2), nu = delta_col/sigma(M) --
    # checked against the closed form directly (via the class's own
    # _get_nu), rather than assumed monotonic in mass: the fixture's
    # synthetic power-law P(k) gives an unrealistically large sigma(M), so
    # nu << 1 at every mass in this test's range (below the multiplicity
    # function's nu=1 peak), which would make a "decreasing with mass"
    # assumption fixture-dependent rather than a property of fps() itself.
    masses = np.array([1e11, 1e12, 1e13, 1e14, 1e15])
    fnu = mass_functions.fps(masses, 0.0)
    nu = mass_functions._get_nu(masses, 0.0)
    expected = np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu * nu)
    assert np.all(fnu > 0)
    assert np.allclose(fnu, expected)


def test_fst_positive(mass_functions):
    masses = np.array([1e11, 1e12, 1e13, 1e14])
    fnu = mass_functions.fst(masses, 0.0)
    assert np.all(fnu > 0)


def test_get_massfunc_unknown_type_raises(mass_functions):
    with pytest.raises(ValueError):
        mass_functions.get_massfunc(0.0, mass_function_type="not_a_real_type")


@pytest.mark.parametrize("mass_function_type", ["press_schechter", "sheth_tormen"])
def test_get_massfunc_shape_and_positivity(mass_functions, mass_function_type):
    logmass, dndlogm = mass_functions.get_massfunc(0.0, mass_function_type=mass_function_type)
    assert logmass.shape == dndlogm.shape
    assert np.all(dndlogm >= 0)
    # Number density should fall off at the high-mass end (exponential
    # cutoff in the multiplicity function dominates the 1/M**2 prefactor).
    assert dndlogm[-1] < dndlogm[len(dndlogm) // 2]
