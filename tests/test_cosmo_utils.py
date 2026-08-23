"""
Regression tests for CosmoData's pure-math methods (growth factor, delta_col,
sigma(M)/mass variance, radius<->mass conversions).

These deliberately avoid CLASS/CAMB: get_power_spectrum() (the only method
that touches those backends) is never called here. Instead a synthetic
power-law P(k) = A * k^n stands in, exercising exactly the same downstream
code path (get_mass_variance, dlogsigma_dlogmass, _prepare_sigma_grid) that
a real cosmology would.
"""

import numpy as np
import pytest


def test_delta_col_increases_with_redshift(cosmo_data):
    z = np.array([0.0, 1.0, 5.0, 10.0])
    delta_col = cosmo_data.delta_col_at_z(z)
    assert delta_col[0] == pytest.approx(1.686, rel=1e-3)
    assert np.all(np.diff(delta_col) > 0)


def test_delta_col_table_matches_direct_integration(cosmo_data):
    # delta_col_at_z is a table lookup; cross-check it against the full
    # growth-factor integration it was built from.
    z = 3.7
    _, _, delta_col_direct = cosmo_data.get_linear_growth_and_collapse(redshift=z)
    delta_col_table = cosmo_data.delta_col_at_z(z)
    assert delta_col_table == pytest.approx(delta_col_direct, rel=1e-3)


def test_radius_mass_roundtrip(cosmo_data):
    mass = np.array([1e8, 1e10, 1e12, 1e14])
    radius = cosmo_data.get_radius(mass)
    mass_back = cosmo_data.get_mass(radius)
    assert np.allclose(mass_back, mass, rtol=1e-10)


def test_omega_matter_at_z0_equals_omega_m(cosmo_data):
    assert cosmo_data.get_omega_matter(0.0) == pytest.approx(cosmo_data.cosmo_params["OmegaM"], rel=1e-6)


def test_negative_redshift_rejected(cosmo_data):
    with pytest.raises(ValueError):
        cosmo_data.get_hubble_parameter(-1.0)


def test_sigma_decreases_with_mass(cosmo_data, synthetic_pk_data):
    # sigma(M) is the RMS density-field variance smoothed on scale R(M):
    # for any physically reasonable P(k) it must fall monotonically as M
    # (and hence the smoothing scale) grows.
    cosmo_data._prepare_sigma_grid(synthetic_pk_data, logmass_min=8.0, logmass_max=15.0, dlogmass=0.1)
    assert np.all(np.diff(cosmo_data._sigma) < 0)


def test_sigma_at_logmass_matches_grid(cosmo_data, synthetic_pk_data):
    cosmo_data._prepare_sigma_grid(synthetic_pk_data, logmass_min=8.0, logmass_max=15.0, dlogmass=0.1)
    # interpolant should reproduce grid points closely
    idx = 20
    logm = cosmo_data._logmass[idx]
    assert cosmo_data.sigma_at_logmass(logm) == pytest.approx(cosmo_data._sigma[idx], rel=1e-6)


def test_window_function_type_set_before_get_mass_variance(cosmo_data):
    # Regression test for a real bug: _prepare_sigma_grid used to reference
    # self.window_function_type before it was ever assigned (it was only set
    # as a side effect of calling get_mass_variance() with an explicit
    # window_function_type kwarg). CosmoData.__init__ now defaults it.
    assert cosmo_data.window_function_type == "top_hat"


def test_cosmic_time_decreases_with_redshift(cosmo_data):
    # Used by plot.plot_dendrogram for edge-length/y-axis positioning
    # (Nadler et al. 2023, Figure 4 style): cosmic time must fall
    # monotonically with increasing redshift.
    z = np.array([0.0, 1.0, 5.0, 10.0, 15.0])
    t = cosmo_data.cosmic_time_at_z(z)
    assert np.all(np.diff(t) < 0)


def test_cosmic_time_table_matches_direct_integration(cosmo_data):
    z = 4.2
    t_direct = cosmo_data.get_cosmic_time(z)
    t_table = cosmo_data.cosmic_time_at_z(z)
    assert t_table == pytest.approx(t_direct, rel=1e-3)


def test_cosmic_time_at_z0_is_positive(cosmo_data):
    assert cosmo_data.cosmic_time_at_z(0.0) > 0.0


def test_cosmic_time_at_z_extends_table_beyond_default_z_max_15(cosmo_data):
    # Regression test for a real bug, same class as pch_trees'
    # delta_col-table fix (_ensure_delta_col_covers): the table built
    # here defaults to z_max=15 (precompute_cosmic_time_table), and
    # np.interp silently clamps beyond that rather than erroring, so any
    # z > 15 used to return the same (wrong) cosmic time as z=15.
    assert cosmo_data.cosmic_time_at_z(15.0) != cosmo_data.cosmic_time_at_z(25.0)
    assert cosmo_data._t_z_grid[-1] >= 25.0
    # and it must still agree with direct integration at the extended z
    t_direct = cosmo_data.get_cosmic_time(20.0)
    t_table = cosmo_data.cosmic_time_at_z(20.0)
    assert t_table == pytest.approx(t_direct, rel=1e-3)


# ---------------------------------------------------------------------------
# Sharp-k window (Stage B of WDM/FDM support: Benson et al. 2013's
# k0 = alpha/R cutoff, alpha=2.5, verified against Kulkarni & Ostriker 2022
# eq. 7 and surrounding text -- the same source already verified for T_FDM)
# ---------------------------------------------------------------------------


def test_window_function_type_propagates_to_wf(cosmo_data, synthetic_pk_data):
    # Regression test for a real bug: get_mass_variance()/dlogsigma_dlogmass()
    # only set CosmoData.window_function_type, never WindowFunctions
    # (cosmo_data.wf).window_function_type -- the attribute that actually
    # drives dispatch inside WindowFunctions.window_function(). Any
    # window_function_type other than the hardcoded default ('top_hat', set
    # once in CosmoData.__init__) silently had no effect until this was
    # fixed.
    cosmo_data.get_mass_variance(synthetic_pk_data, radius=8.0, window_function_type="gaussian")
    assert cosmo_data.wf.window_function_type == "gaussian"


def test_prepare_windows_cache_respects_window_type(cosmo_data, synthetic_pk_data):
    # Regression test for a real bug: WindowFunctions._prepare_windows()
    # cached W/dWdR keyed on radii alone -- switching window_function_type
    # for the same radii array (e.g. comparing top_hat vs sharp_k at the
    # same mass, as plot_dm_model_comparison-style code would) silently
    # returned the first window type's cached result.
    r = 8.0
    S_top_hat = cosmo_data.get_mass_variance(synthetic_pk_data, radius=r, window_function_type="top_hat")
    S_sharp_k = cosmo_data.get_mass_variance(synthetic_pk_data, radius=r, window_function_type="sharp_k")
    S_top_hat_again = cosmo_data.get_mass_variance(synthetic_pk_data, radius=r, window_function_type="top_hat")
    assert S_sharp_k != pytest.approx(S_top_hat)
    assert S_top_hat_again == pytest.approx(S_top_hat)


def test_sharp_k_window_is_a_step_function(cosmo_data):
    alpha = cosmo_data.wf.sharp_k_alpha
    R = 2.0
    k_inside = np.array([0.01, alpha / R * 0.99])
    k_outside = np.array([alpha / R * 1.01, 100.0])
    assert np.all(cosmo_data.wf.sharp_k_window(k_inside, R) == 1.0)
    assert np.all(cosmo_data.wf.sharp_k_window(k_outside, R) == 0.0)


def test_sharp_k_window_deriv_raises_not_implemented(cosmo_data):
    # The window is discontinuous -- its R-derivative is a Dirac delta, not
    # something the generic W*dW/dR route can represent. See
    # dlogsigma_dlogmass's sharp_k branch for the closed-form alternative.
    cosmo_data.wf.window_function_type = "sharp_k"
    with pytest.raises(NotImplementedError):
        cosmo_data.wf.window_function_deriv(1.0, 2.0)


def test_sharp_k_sigma_exceeds_top_hat_at_fixed_mass(cosmo_data, synthetic_pk_data):
    # The sharp-k window includes small-scale power right up to its cutoff
    # with no rolloff, whereas the top-hat's oscillatory tails suppress it
    # -- so sigma(M) is systematically larger for sharp-k at fixed mass
    # (matches Kulkarni & Ostriker 2022 Figure 2, sigma(R): FDM/CDM sharp-k
    # lies above the corresponding top-hat curve).
    R = cosmo_data.get_radius(1e12)
    S_top_hat = cosmo_data.get_mass_variance(synthetic_pk_data, radius=R, window_function_type="top_hat")
    S_sharp_k = cosmo_data.get_mass_variance(synthetic_pk_data, radius=R, window_function_type="sharp_k")
    assert S_sharp_k > S_top_hat


def test_dlogsigma_dlogmass_sharp_k_matches_power_law_exponent(cosmo_data, synthetic_pk_data):
    # synthetic_pk_data is a pure power law P(k) = A k^n (n=-2). For the
    # sharp-k window, sigma^2(R) = 1/(2 pi^2) integral_0^{alpha/R} k^2 P(k) dk
    # is analytically a power law in R (hence in M), giving an exact,
    # mass-independent answer for d ln(sigma^2)/d ln M = -(n+3)/3 -- a
    # stronger check than a finite-difference comparison, since it pins
    # down the actual value, not just self-consistency.
    n = -2.0
    expected = -(n + 3.0) / 3.0
    # masses chosen so k0 = alpha/R stays comfortably below pk_kmax=10
    # (PLANCK_LIKE fixture) -- see the separate pk_kmax-clamp test below for
    # what happens once it doesn't.
    for mass in [1e11, 1e12, 1e13]:
        result = cosmo_data.dlogsigma_dlogmass(synthetic_pk_data, mass=mass, window_function_type="sharp_k")
        assert result == pytest.approx(expected, rel=1e-3)


def test_dlogsigma_dlogmass_sharp_k_matches_finite_difference(cosmo_data, synthetic_pk_data):
    mass = 1e11  # k0 = alpha/R comfortably below pk_kmax=10 (PLANCK_LIKE fixture)
    eps = 1e-5
    R0 = cosmo_data.get_radius(mass)
    R1 = cosmo_data.get_radius(mass * (1 + eps))
    S0 = cosmo_data.get_mass_variance(synthetic_pk_data, radius=R0, window_function_type="sharp_k")
    S1 = cosmo_data.get_mass_variance(synthetic_pk_data, radius=R1, window_function_type="sharp_k")
    fd = (np.log(S1) - np.log(S0)) / eps

    analytic = cosmo_data.dlogsigma_dlogmass(synthetic_pk_data, mass=mass, window_function_type="sharp_k")
    assert analytic == pytest.approx(fd, rel=1e-3)


def test_dlogsigma_dlogmass_sharp_k_zero_beyond_pk_kmax(cosmo_data, synthetic_pk_data):
    # Once k0 = alpha/R exceeds pk_kmax, _sigma2_sharp_k's integration limit
    # saturates at pk_kmax -- sigma^2 stops changing with R there, so the
    # true derivative is exactly zero, not whatever the unclamped closed
    # form would extrapolate to.
    pk_kmax = cosmo_data.run_params["pk_kmax"]
    alpha = cosmo_data.wf.sharp_k_alpha
    tiny_mass = cosmo_data.get_mass(alpha / (pk_kmax * 10))  # k0 = 10x kmax
    result = cosmo_data.dlogsigma_dlogmass(synthetic_pk_data, mass=tiny_mass, window_function_type="sharp_k")
    assert result == pytest.approx(0.0, abs=1e-10)


def test_window_function_type_config_wiring():
    # config/planck2018_fdm_sharpk.yml sets window_function_type: sharp_k
    # and sharp_k_alpha: 2.5 in the Run section -- check io.get_params()
    # threads both through to CosmoData correctly, and that the default
    # config (planck2018.yml, no window_function_type key at all) still
    # resolves to top_hat, matching every config written before this
    # feature existed.
    from foraois.cosmo_utils import CosmoData
    from foraois.utils import io

    run_params = io.get_params("config/planck2018_fdm_sharpk.yml")
    assert run_params["Code"]["window_function_type"] == "sharp_k"
    assert run_params["Code"]["sharp_k_alpha"] == pytest.approx(2.5)

    cosmo = CosmoData(
        {"Code": run_params["Code"], "Cosmology": run_params["Cosmology"]},
        redshift=[0.0],
    )
    assert cosmo.window_function_type == "sharp_k"
    assert cosmo.wf.window_function_type == "sharp_k"
    assert cosmo.wf.sharp_k_alpha == pytest.approx(2.5)

    default_run_params = io.get_params("config/planck2018.yml")
    assert default_run_params["Code"]["window_function_type"] == "top_hat"


def test_sharp_k_tree_building_unchanged_by_window_type(synthetic_pk_data):
    # PCHMergerTree only ever consumes sigma(M)/delta_col(z) through
    # CosmoData's lookup tables -- switching window_function_type before
    # building the sigma grid should just work, no pch_trees.py changes
    # needed (same architectural point already checked for WDM/FDM).
    from foraois.cosmo_utils import CosmoData
    from foraois.pch_trees import PCHMergerTree

    params = {
        "Code": {"mode": "camb", "pk_kmin": 1e-4, "pk_kmax": 10.0, "pk_npoints": 500},
        "Cosmology": {
            "H0": 67.66,
            "OmegaM": 0.3111,
            "OmegaK": 0.0,
            "OmegaLambda": 0.6889,
        },
    }
    cosmo_sk = CosmoData(params, redshift=[0.0])
    cosmo_sk.window_function_type = "sharp_k"
    cosmo_sk.get_power_spectrum = lambda: synthetic_pk_data
    tree_generator = PCHMergerTree(cosmo_sk, params)

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


# ---------------------------------------------------------------------------
# parameter_names() / get_power_spectrum() real-backend integration checks
# (skipped if the backend isn't installed)
# ---------------------------------------------------------------------------

_CAMB_BASE_PARAMS = {
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


def test_parameter_names_returns_internal_to_backend_mapping():
    # Regression test for a real, previously-undiscovered bug: this used
    # to return only the backend-side names (e.g. ['ombh2', 'omch2', ...]
    # for CAMB), which get_power_spectrum() then used to look values up
    # *in self.cosmo_params* -- whose keys are the internal names
    # ('omega_b', 'omega_cdm', ...), not the backend ones. Every backend
    # parameter whose internal and backend names differ (omega_b/ombh2,
    # omega_cdm/omch2, tau/tau_reio for CLASS) was silently dropped by the
    # `if name in self.cosmo_params` filter, for every real CAMB/CLASS run
    # ever made with this class -- both backends silently fell back to
    # their own default baryon/CDM density (and, for CLASS, amplitude,
    # tilt, and optical depth too) instead of the configured values. Now
    # returns {internal: external}, so both names are available together.
    from foraois.cosmo_utils import CosmoData

    cosmo_data = CosmoData(_CAMB_BASE_PARAMS, redshift=[0.0])
    mapping = cosmo_data.parameter_names()
    assert mapping["omega_b"] == "ombh2"
    assert mapping["omega_cdm"] == "omch2"
    assert mapping["tau"] == "tau"  # CAMB's own kwarg name, not CLASS's 'tau_reio'
    assert set(mapping) <= set(cosmo_data.cosmo_params)  # every key must be a real internal name


def test_camb_power_spectrum_actually_uses_configured_omega_b_and_tau():
    # The direct, symptom-level check: changing omega_b/omega_cdm/tau in
    # the config must actually change the resulting P(k) -- under the bug
    # this fixes, it didn't (CAMB silently used its own defaults for
    # these regardless of what was configured), so this test would have
    # failed before the fix and is the guard against it recurring.
    pytest.importorskip("camb")
    from foraois.cosmo_utils import CosmoData

    cosmo_baseline = CosmoData(_CAMB_BASE_PARAMS, redshift=[0.0])
    pk_baseline = cosmo_baseline.get_power_spectrum()

    perturbed_params = {
        "Code": dict(_CAMB_BASE_PARAMS["Code"]),
        "Cosmology": dict(_CAMB_BASE_PARAMS["Cosmology"]),
    }
    perturbed_params["Cosmology"]["omega_b"] = _CAMB_BASE_PARAMS["Cosmology"]["omega_b"] * 3.0
    perturbed_params["Cosmology"]["omega_cdm"] = _CAMB_BASE_PARAMS["Cosmology"]["omega_cdm"] * 0.3
    perturbed_params["Cosmology"]["tau"] = 0.2

    cosmo_perturbed = CosmoData(perturbed_params, redshift=[0.0])
    pk_perturbed = cosmo_perturbed.get_power_spectrum()

    assert not np.allclose(pk_baseline["Pk"], pk_perturbed["Pk"], rtol=1e-3)


def test_camb_sigma8_matches_target_for_menon_power_2024_config():
    # End-to-end sanity check against a real, independently-derived target:
    # config/menon_power_2024.yml's As was solved (by exploiting sigma8^2
    # being linear in As) to give sigma8=0.815 at z=0 -- confirms the
    # parameter-mapping fix actually reaches CAMB's computed P(k), not
    # just that the dict looks right in isolation.
    pytest.importorskip("camb")
    from pathlib import Path

    from foraois.cosmo_utils import CosmoData
    from foraois.utils import io

    config_path = Path(__file__).resolve().parents[1] / "config" / "menon_power_2024.yml"
    run_params = io.get_params(str(config_path))
    cosmo_data = CosmoData(run_params, redshift=[0.0])
    pk_data = cosmo_data.get_power_spectrum()
    sigma8 = np.sqrt(cosmo_data.get_mass_variance(pk_data, radius=8.0, window_function_type="top_hat"))
    assert sigma8 == pytest.approx(0.815, abs=1e-3)
