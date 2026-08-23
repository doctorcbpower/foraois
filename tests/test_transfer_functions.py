"""
Tests for transfer_functions.py and its wiring into
CosmoData.get_power_spectrum() (dm_model='wdm'/'fdm').

The pure-math tests need no cosmology backend at all. The integration
tests need a real P(k) (CosmoData's synthetic-power-law test fixture
monkeypatches get_power_spectrum() entirely, which would bypass the
transfer-function application logic being tested here) -- they use real
CAMB via pytest.importorskip, so they're skipped gracefully rather than
failing where CAMB isn't installed.
"""

import numpy as np
import pytest

from foraois import transfer_functions as tf
from foraois.cosmo_utils import CosmoData

M_WDM = 3.0
OMEGA_WDM = 0.3
H = 0.7


def test_T_CDM_is_identity():
    k = np.logspace(-4, 2, 50)
    assert np.allclose(tf.T_CDM(k), 1.0)


def test_T_WDM_bounded_in_zero_one():
    k = np.logspace(-4, 3, 200)
    T = tf.T_WDM(k, M_WDM, OMEGA_WDM, H)
    assert np.all(T >= 0.0)
    assert np.all(T <= 1.0 + 1e-12)


def test_T_WDM_goes_to_one_as_k_goes_to_zero():
    T = tf.T_WDM(np.array([1e-8]), M_WDM, OMEGA_WDM, H)
    assert T[0] == pytest.approx(1.0, abs=1e-6)


def test_T_WDM_monotonically_decreasing_with_k():
    k = np.logspace(-3, 2, 200)
    T = tf.T_WDM(k, M_WDM, OMEGA_WDM, H)
    assert np.all(np.diff(T) <= 1e-12)


def test_T_WDM_heavier_particle_suppresses_less():
    # A heavier thermal relic behaves more like CDM at any fixed k --
    # physically, heavier WDM free-streams less.
    k = np.logspace(-2, 2, 100)
    T_light = tf.T_WDM(k, 1.0, OMEGA_WDM, H)
    T_heavy = tf.T_WDM(k, 10.0, OMEGA_WDM, H)
    assert np.all(T_heavy >= T_light - 1e-12)
    assert np.any(T_heavy > T_light + 1e-6)  # not trivially identical


def test_wdm_half_mode_k_matches_definition():
    k_hm = tf.wdm_half_mode_k(M_WDM, OMEGA_WDM, H)
    T_at_khm = tf.T_WDM(np.array([k_hm]), M_WDM, OMEGA_WDM, H)
    assert T_at_khm[0] == pytest.approx(0.5, rel=1e-8)


def test_dm_transfer_function_defaults_to_cdm(cosmo_data):
    # cosmo_data fixture's Code dict has no dm_model key at all --
    # .get('dm_model', 'cdm') must default cleanly, matching every config
    # written before this feature existed.
    k = np.logspace(-3, 1, 20)
    assert np.allclose(cosmo_data._dm_transfer_function(k), 1.0)


def test_dm_transfer_function_wdm_dispatch(cosmo_data):
    cosmo_data.run_params["dm_model"] = "wdm"
    cosmo_data.run_params["dm_model_mass"] = M_WDM
    k = np.logspace(-3, 2, 20)
    T = cosmo_data._dm_transfer_function(k)
    h = cosmo_data.cosmo_params.get("h", cosmo_data.cosmo_params["H0"] / 100.0)
    expected = tf.T_WDM(k, M_WDM, cosmo_data.cosmo_params["OmegaM"], h)
    assert np.allclose(T, expected)


def test_dm_transfer_function_wdm_requires_mass(cosmo_data):
    cosmo_data.run_params["dm_model"] = "wdm"
    cosmo_data.run_params["dm_model_mass"] = None
    with pytest.raises(ValueError, match="dm_model_mass"):
        cosmo_data._dm_transfer_function(np.array([1.0]))


def test_dm_transfer_function_rejects_unknown_model(cosmo_data):
    cosmo_data.run_params["dm_model"] = "not_a_real_model"
    with pytest.raises(ValueError, match="Unknown dm_model"):
        cosmo_data._dm_transfer_function(np.array([1.0]))


# ---------------------------------------------------------------------------
# FDM (Hu, Barkana & Gruzinov 2000)
# ---------------------------------------------------------------------------

M_A22 = 1.0  # 1e-22 eV


def test_T_FDM_goes_to_one_as_k_goes_to_zero():
    T = tf.T_FDM(np.array([1e-8]), M_A22, H)
    assert T[0] == pytest.approx(1.0, abs=1e-6)


def test_T_FDM_monotonically_decreasing_near_half_mode():
    # T_F(k) = cos(x^3)/(1+x^8) is NOT globally monotonic -- it oscillates
    # via cos(x^3) at large k (per HBG00, this is real ringing, not a fit
    # artifact) -- so unlike WDM this can only be checked within the first,
    # physically-relevant lobe (x up to (pi/2)^(1/3), where cos(x^3) first
    # hits zero). fdm_half_mode_k relies on exactly this monotonic range.
    k_hm = tf.fdm_half_mode_k(M_A22, H)
    k = np.linspace(1e-6, k_hm * 1.29, 200)  # stays within the safe monotonic lobe
    T = tf.T_FDM(k, M_A22, H)
    assert np.all(np.diff(T) <= 1e-10)


def test_T_FDM_heavier_particle_suppresses_less():
    # Heavier FDM particles have a larger de Broglie/Jeans scale suppression
    # pushed to higher k -- i.e. less suppression at any fixed k -- the FDM
    # analogue of WDM's "heavier free-streams less".
    k = np.linspace(1e-3, 3.0, 100)
    T_light = tf.T_FDM(k, 0.1, H)
    T_heavy = tf.T_FDM(k, 10.0, H)
    assert np.all(T_heavy >= T_light - 1e-9)
    assert np.any(T_heavy > T_light + 1e-6)


def test_fdm_half_mode_k_matches_definition():
    k_hm = tf.fdm_half_mode_k(M_A22, H)
    T_at_khm = tf.T_FDM(np.array([k_hm]), M_A22, H)
    assert T_at_khm[0] == pytest.approx(0.5, rel=1e-6)


def test_dm_transfer_function_fdm_dispatch(cosmo_data):
    cosmo_data.run_params["dm_model"] = "fdm"
    cosmo_data.run_params["dm_model_mass"] = M_A22
    k = np.logspace(-3, 1, 20)
    T = cosmo_data._dm_transfer_function(k)
    h = cosmo_data.cosmo_params.get("h", cosmo_data.cosmo_params["H0"] / 100.0)
    expected = tf.T_FDM(k, M_A22, h)
    assert np.allclose(T, expected)


def test_dm_transfer_function_fdm_requires_mass(cosmo_data):
    cosmo_data.run_params["dm_model"] = "fdm"
    cosmo_data.run_params["dm_model_mass"] = None
    with pytest.raises(ValueError, match="dm_model_mass"):
        cosmo_data._dm_transfer_function(np.array([1.0]))


# ---------------------------------------------------------------------------
# Real-CAMB integration tests (skipped if camb isn't installed)
# ---------------------------------------------------------------------------

CAMB_PARAMS = {
    "Code": {
        "mode": "camb",
        "pk_kmin": 1e-4,
        "pk_kmax": 1e2,
        "pk_npoints": 500,
        "CAMB": {},
    },
    "Cosmology": {
        "H0": 67.66,
        "OmegaM": 0.3111,
        "OmegaK": 0.0,
        "OmegaLambda": 0.6889,
        "omega_b": 0.02242,
        "omega_cdm": 0.1193,
        "omk": 0.0,
        "As": 2.1e-9,
        "ns": 0.9665,
        "tau": 0.0561,
        "mnu": 0.0,
        "num_massive_neutrinos": 0,
        "h": 0.6766,
    },
}


def test_wdm_power_spectrum_matches_cdm_at_large_scales():
    pytest.importorskip("camb")

    cosmo_cdm = CosmoData(CAMB_PARAMS, redshift=[0.0])
    pk_cdm = cosmo_cdm.get_power_spectrum()

    wdm_params = {
        **CAMB_PARAMS,
        "Code": {**CAMB_PARAMS["Code"], "dm_model": "wdm", "dm_model_mass": 1.0},
    }
    cosmo_wdm = CosmoData(wdm_params, redshift=[0.0])
    pk_wdm = cosmo_wdm.get_power_spectrum()

    k = pk_cdm["k"]
    ratio = pk_wdm["Pk"][0] / pk_cdm["Pk"][0]

    assert ratio[np.argmin(np.abs(k - 1e-3))] > 0.99  # unsuppressed at large scales
    assert ratio[np.argmin(np.abs(k - 50))] < 0.3  # strongly suppressed at small scales


def test_wdm_sigma_of_mass_converges_to_cdm_at_high_mass():
    pytest.importorskip("camb")

    cosmo_cdm = CosmoData(CAMB_PARAMS, redshift=[0.0])
    pk_cdm = cosmo_cdm.get_power_spectrum()
    cosmo_cdm._prepare_sigma_grid(pk_cdm, logmass_min=6.0, logmass_max=15.0, dlogmass=0.5)

    wdm_params = {
        **CAMB_PARAMS,
        "Code": {**CAMB_PARAMS["Code"], "dm_model": "wdm", "dm_model_mass": 1.0},
    }
    cosmo_wdm = CosmoData(wdm_params, redshift=[0.0])
    pk_wdm = cosmo_wdm.get_power_spectrum()
    cosmo_wdm._prepare_sigma_grid(pk_wdm, logmass_min=6.0, logmass_max=15.0, dlogmass=0.5)

    # low-mass sigma is suppressed relative to CDM ...
    ratio_low = cosmo_wdm.sigma_at_logmass(6.0) / cosmo_cdm.sigma_at_logmass(6.0)
    assert ratio_low < 0.8
    # ... and high-mass sigma is essentially unaffected
    ratio_high = cosmo_wdm.sigma_at_logmass(14.0) / cosmo_cdm.sigma_at_logmass(14.0)
    assert ratio_high == pytest.approx(1.0, abs=0.01)


def test_wdm_tree_building_unchanged_by_dm_model():
    # PCHMergerTree never touches P(k) directly -- only sigma(M)/delta_col(z)
    # via CosmoData -- so building a forest with dm_model='wdm' configured
    # should just work, no pch_trees.py changes needed. This is the whole
    # point of the CosmoData/tree-kernel separation; check it actually holds.
    pytest.importorskip("camb")
    from foraois.pch_trees import PCHMergerTree

    wdm_params = {
        **CAMB_PARAMS,
        "Code": {**CAMB_PARAMS["Code"], "dm_model": "wdm", "dm_model_mass": 2.0},
    }
    cosmo_wdm = CosmoData(wdm_params, redshift=[0.0])
    tree_generator = PCHMergerTree(cosmo_wdm, wdm_params)

    np.random.seed(0)
    mass_history, _, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(200, 1e12),
        z0=0.0,
        z_max=3.0,
        M_res=1e9,
        dz=0.05,
    )
    assert mass_history.shape == (200, len(z_steps) - 1)
    assert np.all(mass_history >= 0.0)


def test_wdm_tree_building_with_M_res_inside_suppression_scale():
    # docs/MODELS.md's own caveat: "M_res should stay well above whatever
    # suppression scale the chosen dm_model + window combination imposes:
    # sigma(M) genuinely flattens out below it, and the PCH08 branching-
    # rate algebra is poorly conditioned there." This was documented but
    # not previously exercised by a test -- check it degrades *sensibly*
    # (finite, non-negative, no crash) rather than silently producing NaN/
    # negative masses or an unbounded/undefined branching rate when M_res
    # is chosen deep inside the flat regime (dm_model_mass=1.0 keV already
    # gives sigma(1e6)/sigma_CDM(1e6) < 0.8 per the convergence test above;
    # M_res=1e5 here is another decade further into the suppressed tail).
    pytest.importorskip("camb")
    from foraois.pch_trees import PCHMergerTree

    wdm_params = {
        **CAMB_PARAMS,
        "Code": {**CAMB_PARAMS["Code"], "dm_model": "wdm", "dm_model_mass": 1.0},
    }
    cosmo_wdm = CosmoData(wdm_params, redshift=[0.0])
    tree_generator = PCHMergerTree(cosmo_wdm, wdm_params)

    np.random.seed(0)
    mass_history, _, z_steps, smooth_accretion, merger_mass = tree_generator.build_forest_numpy(
        M0_array=np.full(200, 1e10),
        z0=0.0,
        z_max=3.0,
        M_res=1e5,
        dz=0.05,
    )
    assert mass_history.shape == (200, len(z_steps) - 1)
    assert np.all(np.isfinite(mass_history))
    assert np.all(mass_history >= 0.0)
    assert np.all(np.isfinite(smooth_accretion)) and np.all(smooth_accretion >= -1e-6)
    assert np.all(np.isfinite(merger_mass)) and np.all(merger_mass >= 0.0)
    # Mass conservation must still hold exactly in this regime, same
    # identity test_pch_trees.py checks for the ordinary (unsuppressed) case.
    full_history = np.concatenate([np.full((200, 1), 1e10), mass_history], axis=1)
    forward_gain = full_history[:, :-1] - full_history[:, 1:]
    channel_sum = smooth_accretion + merger_mass
    both_resolved = (full_history[:, 1:] > 0) & (full_history[:, :-1] > 0)
    assert both_resolved.sum() > 0
    assert np.allclose(forward_gain[both_resolved], channel_sum[both_resolved], rtol=1e-6, atol=1e-6)


def test_fdm_power_spectrum_matches_cdm_at_large_scales():
    pytest.importorskip("camb")

    cosmo_cdm = CosmoData(CAMB_PARAMS, redshift=[0.0])
    pk_cdm = cosmo_cdm.get_power_spectrum()

    fdm_params = {
        **CAMB_PARAMS,
        "Code": {**CAMB_PARAMS["Code"], "dm_model": "fdm", "dm_model_mass": 1.0},
    }
    cosmo_fdm = CosmoData(fdm_params, redshift=[0.0])
    pk_fdm = cosmo_fdm.get_power_spectrum()

    k = pk_cdm["k"]
    ratio = pk_fdm["Pk"][0] / pk_cdm["Pk"][0]

    assert ratio[np.argmin(np.abs(k - 1e-3))] > 0.99  # unsuppressed at large scales
    assert ratio[np.argmin(np.abs(k - 20))] < 0.3  # strongly suppressed near/past cutoff


def test_fdm_sigma_of_mass_converges_to_cdm_at_high_mass():
    pytest.importorskip("camb")

    cosmo_cdm = CosmoData(CAMB_PARAMS, redshift=[0.0])
    pk_cdm = cosmo_cdm.get_power_spectrum()
    cosmo_cdm._prepare_sigma_grid(pk_cdm, logmass_min=6.0, logmass_max=15.0, dlogmass=0.5)

    fdm_params = {
        **CAMB_PARAMS,
        "Code": {**CAMB_PARAMS["Code"], "dm_model": "fdm", "dm_model_mass": 1.0},
    }
    cosmo_fdm = CosmoData(fdm_params, redshift=[0.0])
    pk_fdm = cosmo_fdm.get_power_spectrum()
    cosmo_fdm._prepare_sigma_grid(pk_fdm, logmass_min=6.0, logmass_max=15.0, dlogmass=0.5)

    ratio_low = cosmo_fdm.sigma_at_logmass(6.0) / cosmo_cdm.sigma_at_logmass(6.0)
    assert ratio_low < 0.8
    ratio_high = cosmo_fdm.sigma_at_logmass(14.0) / cosmo_cdm.sigma_at_logmass(14.0)
    assert ratio_high == pytest.approx(1.0, abs=0.01)


def test_fdm_tree_building_unchanged_by_dm_model():
    pytest.importorskip("camb")
    from foraois.pch_trees import PCHMergerTree

    fdm_params = {
        **CAMB_PARAMS,
        "Code": {**CAMB_PARAMS["Code"], "dm_model": "fdm", "dm_model_mass": 1.0},
    }
    cosmo_fdm = CosmoData(fdm_params, redshift=[0.0])
    tree_generator = PCHMergerTree(cosmo_fdm, fdm_params)

    np.random.seed(0)
    mass_history, _, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(200, 1e12),
        z0=0.0,
        z_max=3.0,
        M_res=1e9,
        dz=0.05,
    )
    assert mass_history.shape == (200, len(z_steps) - 1)
    assert np.all(mass_history >= 0.0)
