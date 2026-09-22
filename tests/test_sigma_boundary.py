"""
Regression tests for the v0.1.2 sigma(M) table/pk_kmax numerical-validity
fix (audit, 2026-09; see foraois_sigma_audit/REPORT.md and REPORT_kmax.md
in the maintainer's audit checkout for the full derivation).

Two independent problems, addressed together by `CosmoData.check_M_res`
(cosmo_utils.py) and `_prepare_sigma_grid`'s new default `logmass_min`:

* P1 (bug): the compiled (numba) tree kernels read sigma(M) from a table
  that used to start at 1e5 Msun/h and silently *clamp* below it, while the
  scipy interpolants used by the NumPy/serial paths *extrapolate* -- so a
  compiled build with M_res below 1e5 Msun/h silently used the wrong
  sigma(M_res) regardless of the requested M_res. Fixed by extending the
  table (`logmass_min=2.0`) and raising instead of clamping if a build still
  falls outside it.
* P2 (numerical-validity gap, not a bug): sigma(M)/alpha(M) are integrals of
  P(k) truncated at `pk_kmax`; below a mass set by `pk_kmax * R(M) ~ 4`
  (calibrated against a k_max=3e4 reference) this truncation itself makes
  both quantities too small. `check_M_res` warns rather than raises, since
  this degrades gracefully rather than silently returning nonsense.

Most tests use a synthetic power-law P(k) (no CLASS/CAMB dependency),
following the convention of the rest of this suite (see conftest.py); a few
that specifically check the *shipped, documented* configs need `camb`
(guarded with `pytest.importorskip`).
"""

import numpy as np
import pytest

from foraois import CosmoData, PCHMergerTree, ZhangHuiMergerTree
from foraois.utils import io

# A config with a large enough pk_kmax that M_res down to ~100 Msun/h is
# numerically supportable (used only to isolate the P1 table-clamp fix from
# P2's separate, coarser pk_kmax-truncation warning) -- 3000 was the
# smallest value the audit's tree-level tests found to pass cleanly at
# M_res=1e4 Msun/h for a demanding (z0=5-anchored, z_max=25) configuration.
_WIDE_KMAX = {
    "Code": {"mode": "camb", "pk_kmin": 1e-4, "pk_kmax": 3000.0, "pk_npoints": 2000},
    "Cosmology": {"H0": 67.66, "OmegaM": 0.3111, "OmegaK": 0.0, "OmegaLambda": 0.6889},
}


def _synthetic_pk(kmax, n=2000):
    """Power-law P(k) = A k^-2 on [1e-4, kmax], standing in for CLASS/CAMB (see conftest.py)."""
    k = np.logspace(-4, np.log10(kmax), n)
    return {"k": k, "Pk": (2.0e4 * k**-2.0).reshape(1, -1), "z": [0.0]}


@pytest.fixture(scope="module")
def wide_cosmo():
    """CosmoData/PCHMergerTree/ZhangHuiMergerTree with pk_kmax=3000, synthetic P(k), no CAMB needed."""
    cd = CosmoData(_WIDE_KMAX, redshift=[0.0])
    cd.get_power_spectrum = lambda: _synthetic_pk(3000.0)
    return _WIDE_KMAX, cd, PCHMergerTree(cd, _WIDE_KMAX), ZhangHuiMergerTree(cd, _WIDE_KMAX, model="cdm")


# ------------------------------------------------------------------ table extent / no silent clamp


def test_default_table_covers_100_msun_per_h(wide_cosmo):
    """logmass_min default is 2.0 (was 5.0 through v0.1.1): the table now reaches 100 Msun/h."""
    _, cd, _, _ = wide_cosmo
    assert cd._logmass[0] == pytest.approx(2.0)
    assert 10 ** cd._logmass[0] == pytest.approx(100.0, rel=1e-6)


@pytest.mark.parametrize("M_res", [1e3, 3e3, 1e4, 3e4])
def test_pch_numba_matches_numpy_below_old_1e5_boundary(wide_cosmo, M_res):
    """The old bug: build_forest_numba silently clamped sigma(M) below 1e5 Msun/h, giving results that
    disagreed with build_forest_numpy by tens of per cent. With the fix, both paths use the same sigma(M)
    and agree within Monte Carlo noise. N chosen so the tolerance is well above the expected sampling error."""
    _, cd, tg, _ = wide_cosmo
    N = 4000
    a = tg.build_forest_numpy(np.full(N, 1e8), 0.0, 10.0, M_res, dz=0.01)
    b = tg.build_forest_numba(np.full(N, 1e8), 0.0, 10.0, M_res, dz=0.01)
    j = -1  # last step (z closest to 10)
    ma, mb = a[0][:, j], b[0][:, j]
    assert np.median(mb[mb > 0]) / 1e8 == pytest.approx(np.median(ma[ma > 0]) / 1e8, rel=0.10)
    assert (mb > 0).mean() == pytest.approx((ma > 0).mean(), abs=0.05)


@pytest.mark.parametrize("M_res", [1e3, 3e3, 1e4, 3e4])
def test_zh_numba_matches_numpy_below_old_1e5_boundary(wide_cosmo, M_res):
    """As above, for the Zhang-Hui compiled kernel (which additionally clamped its *inverse* sigma->M map)."""
    _, cd, _, zh = wide_cosmo
    N = 4000
    a = zh.build_forest_numpy(np.full(N, 1e8), 0.0, 10.0, M_res, dz=0.01)
    b = zh.build_forest_numba(np.full(N, 1e8), 0.0, 10.0, M_res, dz=0.01)
    j = -1
    aa, bb = a[0][:, j], b[0][:, j]
    ra, rb = (aa > 0).mean(), (bb > 0).mean()
    assert rb == pytest.approx(ra, abs=0.05)
    if ra > 0.05 and rb > 0.05:
        assert np.median(bb[bb > 0]) / 1e8 == pytest.approx(np.median(aa[aa > 0]) / 1e8, rel=0.15)


def test_zh_inverse_lookup_no_longer_clamps(wide_cosmo):
    """Old bug: the compiled inverse map (sigma -> log M) clamped to log10(1e5)=5.0 for any sigma above
    sigma(1e5). It must now resolve a mass below the old 1e5 boundary."""
    from foraois.zhang_hui_trees import _interp_sorted_zh

    _, cd, _, _ = wide_cosmo
    target = float(cd.sigma_at_logmass(3.5))  # sigma of a 3162 Msun/h halo
    got = _interp_sorted_zh(target, cd._sigma[::-1].copy(), cd._logmass[::-1].copy())
    assert got == pytest.approx(3.5, abs=0.02)


# ------------------------------------------------------------------ check_M_res: raise below the table


def test_check_M_res_raises_below_table_compiled(wide_cosmo):
    _, cd, _, _ = wide_cosmo
    with pytest.raises(ValueError, match="tabulated range"):
        cd.check_M_res(1.0, lookup_factor=0.5, compiled=True)  # 0.5 Msun/h << table floor of 100


def test_check_M_res_raises_below_table_noncompiled_too(wide_cosmo):
    """Extrapolation below the (now-extended) table was never validated either: even the NumPy/serial
    path must refuse, not silently extrapolate arbitrarily far."""
    _, cd, _, _ = wide_cosmo
    with pytest.raises(ValueError, match="tabulated range"):
        cd.check_M_res(1.0, compiled=False)


def test_check_M_res_lookup_factor_distinguishes_pch_from_zh(wide_cosmo):
    """PCH08's own smallest lookup is sigma(0.5*M_res); Zhang-Hui's is sigma(M_res). At an M_res whose
    half falls below the table but whose full value does not, only the PCH (lookup_factor=0.5) call raises."""
    _, cd, _, _ = wide_cosmo
    M_res = 150.0  # inside the table (100) itself, but 0.5*M_res=75 is not
    cd.check_M_res(M_res, lookup_factor=1.0, compiled=True)  # ZH: fine
    with pytest.raises(ValueError):
        cd.check_M_res(M_res, lookup_factor=0.5, compiled=True)  # PCH: raises


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("build_tree", dict(M0=1e8, z0=0.0, z_max=3.0, M_res=1.0, dz=0.1)),
        ("build_full_tree", dict(M0=1e8, z0=0.0, z_max=3.0, M_res=1.0, dz=0.1)),
        ("build_forest_numpy", dict(M0_array=np.full(2, 1e8), z0=0.0, z_max=3.0, M_res=1.0, dz=0.1)),
        ("build_forest_numba", dict(M0_array=np.full(2, 1e8), z0=0.0, z_max=3.0, M_res=1.0, dz=0.1)),
    ],
)
def test_pch_every_entry_point_is_guarded(wide_cosmo, method, kwargs):
    """Every public PCH08 tree-building method must refuse an unsupported M_res, not just build_forest_numba."""
    _, cd, tg, _ = wide_cosmo
    with pytest.raises(ValueError):
        getattr(tg, method)(**kwargs)


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("build_tree", dict(M0=1e8, z0=0.0, z_max=3.0, M_res=1.0, dz=0.1)),
        ("build_forest_numpy", dict(M0_array=np.full(2, 1e8), z0=0.0, z_max=3.0, M_res=1.0, dz=0.1)),
        ("build_forest_numba", dict(M0_array=np.full(2, 1e8), z0=0.0, z_max=3.0, M_res=1.0, dz=0.1)),
    ],
)
def test_zh_every_entry_point_is_guarded(wide_cosmo, method, kwargs):
    _, cd, _, zh = wide_cosmo
    with pytest.raises(ValueError):
        getattr(zh, method)(**kwargs)


# ------------------------------------------------------------------ check_M_res: pk_kmax warning (P2)


def test_kmax_warning_fires_for_cdm_low_resolution(wide_cosmo):
    _, cd, _, _ = wide_cosmo
    with pytest.warns(UserWarning, match="pk_kmax"):
        cd.check_M_res(500.0, compiled=False)  # kmax=3000: kR(500) ~ 3000*0.0087 ~ 2.6 < 4.1


def test_kmax_warning_does_not_claim_invalidity_and_states_3_5pct_thresholds(wide_cosmo):
    """The warning must say 1% accuracy is unestablished (not that the calculation is invalid), and must give
    the pk_kmax needed for 3%/5% so the reader can judge whether their own looser requirement is already met."""
    import warnings

    _, cd, _, _ = wide_cosmo
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cd.check_M_res(500.0, compiled=False)
    msg = str(w[0].message)
    assert "does not by itself mean the calculation is invalid" in msg  # explicit disclaimer, not merely absence of the word
    assert "1%" in msg and "3%" in msg and "5%" in msg
    assert "necessary but not sufficient" in msg or "necessary, not sufficient" in msg


def test_kmax_warning_absent_for_well_resolved_mass(wide_cosmo):
    import warnings

    _, cd, _, _ = wide_cosmo
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.check_M_res(1e8, compiled=False)  # comfortably inside kmax=3000's validity


def test_kmax_warning_absent_for_wdm_and_fdm(wide_cosmo):
    """Thermal-relic WDM and standard (top-hat) FDM transfer functions suppress P(k) at the k this
    criterion probes, so the CDM-calibrated threshold would over-warn; skipped for both."""
    import warnings

    cfg, cd, _, _ = wide_cosmo
    for model in ("wdm", "fdm"):
        p = {"Code": dict(cfg["Code"], dm_model=model), "Cosmology": cfg["Cosmology"]}
        cd2 = CosmoData(p, redshift=[0.0])
        cd2._logmass, cd2._sigma = cd._logmass, cd._sigma  # reuse the already-built table
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cd2.check_M_res(500.0, compiled=False)


def test_kmax_warning_absent_for_sharp_k(wide_cosmo):
    """The sharp-k window already forces alpha=0 below M(k0=pk_kmax) (docs/MODELS.md); the generic
    pk_kmax warning would be redundant/confusing on top of that explicit signal."""
    import warnings

    cfg, cd, _, _ = wide_cosmo
    p = {"Code": dict(cfg["Code"], window_function_type="sharp_k"), "Cosmology": cfg["Cosmology"]}
    cd2 = CosmoData(p, redshift=[0.0])
    cd2.window_function_type = "sharp_k"
    cd2._logmass, cd2._sigma = cd._logmass, cd._sigma
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd2.check_M_res(500.0, compiled=False)


# ------------------------------------------------------------------ shipped-config regressions (need camb)


def test_shipped_planck2018_camb_warns_at_ashvini_scale_resolution():
    """planck2018_camb.yml (pk_kmax=100, unchanged in this release) is documented as adequate only for
    M_res >~ 1.6e7 Msun/h (5% criterion); requesting M_res=1e4 must warn."""
    pytest.importorskip("camb")
    p = io.get_params("config/planck2018_camb.yml")
    cd = CosmoData(p, redshift=[0.0])
    cd.get_power_spectrum = lambda: _synthetic_pk(100.0)  # avoid a real CAMB call; same pk_kmax as the config
    cd._prepare_sigma_grid(cd.get_power_spectrum())
    with pytest.warns(UserWarning, match="pk_kmax"):
        cd.check_M_res(1e4, compiled=False)


def test_shipped_menon_config_no_longer_warns_at_its_own_use_case():
    """menon_power_2024.yml's pk_kmax was raised from 100 to 3000 in this release specifically to support
    its documented use case (Ashvini production, M_res ~ 1e4 Msun/h): that call must not warn."""
    pytest.importorskip("camb")
    import warnings

    p = io.get_params("config/menon_power_2024.yml")
    # pk_kmax loads as the string '3.e3' (a pre-existing PyYAML 1.1 quirk shared by every "N.eM"-style
    # value in these configs, e.g. the shipped "1.e2" configs) -- consumed via float()/np.float32()
    # everywhere in the library (get_kvals, check_M_res), so this cast matches actual usage.
    assert float(p["Code"]["pk_kmax"]) == pytest.approx(3000.0)
    cd = CosmoData(p, redshift=[0.0])
    cd.get_power_spectrum = lambda: _synthetic_pk(3000.0)
    cd._prepare_sigma_grid(cd.get_power_spectrum())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.check_M_res(1e4, compiled=False)


# ------------------------------------------------------------------ determinism (unaffected by the fix)


def test_numpy_paths_remain_deterministic_at_low_M_res(wide_cosmo):
    _, cd, tg, zh = wide_cosmo
    np.random.seed(7)
    a = tg.build_forest_numpy(np.full(50, 1e8), 0.0, 5.0, 1e4, 0.02)[0]
    np.random.seed(7)
    b = tg.build_forest_numpy(np.full(50, 1e8), 0.0, 5.0, 1e4, 0.02)[0]
    assert np.array_equal(a, b)

    a = ZhangHuiMergerTree(cd, _WIDE_KMAX, model="cdm", rng=np.random.default_rng(7)).build_forest_numpy(
        np.full(50, 1e8), 0.0, 5.0, 1e4, 0.02
    )[0]
    b = ZhangHuiMergerTree(cd, _WIDE_KMAX, model="cdm", rng=np.random.default_rng(7)).build_forest_numpy(
        np.full(50, 1e8), 0.0, 5.0, 1e4, 0.02
    )[0]
    assert np.array_equal(a, b)
