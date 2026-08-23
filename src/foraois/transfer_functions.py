"""
Transfer functions T(k) = sqrt(P_DM(k) / P_CDM(k)) for non-standard dark
matter models.

CosmoData.get_power_spectrum() applies T(k)^2 multiplicatively to a CDM
linear P(k) (from CLASS/CAMB) when a non-CDM `dm_model` is configured.
Because PCHMergerTree's tree-building kernels only ever touch sigma(M)
and delta_col(z) through CosmoData's generic lookup tables (never P(k)
directly), suppressing P(k) here is enough to get correct WDM/FDM
behaviour throughout the whole pipeline with no changes to pch_trees.py
at all -- verified directly for both (see test_transfer_functions.py).

CDM, thermal-relic WDM (T_WDM), and FDM's *linear* suppression (T_FDM)
are implemented here. What's deliberately NOT here: FDM's quantum pressure
keeps acting
during nonlinear collapse (not just linear growth the way WDM's free-
streaming does), setting soliton cores and a genuine minimum halo mass;
there's evidence in the literature (e.g. Du et al. 2017, discussed in
Kulkarni & Ostriker 2022) that reproducing FDM N-body/Schrodinger-Poisson
mass functions at the low-mass end needs a mass-dependent collapse
barrier delta_c(M), not just T_FDM(k) applied on top of the standard
constant CDM barrier. That's a separate research question -- it needs a
specific physically-motivated ansatz pulled from the literature and
checked, not a guess -- tracked as a follow-up, not attempted here. The
same papers also use a sharp-k (not top-hat) window function to avoid
the well-known "cloud-in-cloud" artifact a truncated P(k) produces under
a top-hat window; that's tracked as a separate, immediately-following
piece of work (it touches shared window_function.py code used by every
model, and the R<->M calibration for a sharp-k window needs its own
verification pass) rather than being rushed in alongside this module.
"""

import numpy as np
from scipy.optimize import brentq


def T_CDM(k):
    """Trivial identity transfer function: T(k) = 1 for all k."""
    return np.ones_like(np.asarray(k, dtype=float))


def T_WDM(k, m_wdm_kev, omega_wdm, h):
    """
    Thermal-relic WDM transfer function (Viel, Lesgourgues, Haehnelt,
    Matarrese & Riotto 2005, Phys. Rev. D 71, 063534,
    arXiv:astro-ph/0501562, eq. 6-7):

        T(k) = [1 + (alpha*k)^(2*nu)]^(-5/nu),   nu = 1.12

        alpha = 0.049 * (m_wdm/1 keV)^-1.11 * (Omega_wdm/0.25)^0.11
                       * (h/0.7)^1.22   [h^-1 Mpc]

    Calibrated against CAMB/CMBFAST WDM power spectra for k < 5 h/Mpc;
    treat results outside that range with caution. k is expected in
    h/Mpc, matching alpha's h^-1 Mpc units and foraois' own P(k)
    convention throughout.

    Parameters
    ----------
    k : array-like, h/Mpc
    m_wdm_kev : float
        Thermal-relic WDM particle mass, in keV.
    omega_wdm : float
        WDM density parameter Omega_wdm (dimensionless, i.e. Omega_x --
        *not* the physical density Omega_wdm*h^2). For a pure-WDM model
        with no separate CDM component, this is just Omega_m.
    h : float

    Returns
    -------
    np.ndarray, same shape as k
    """
    k = np.asarray(k, dtype=float)
    nu = 1.12
    alpha = 0.049 * m_wdm_kev ** (-1.11) * (omega_wdm / 0.25) ** 0.11 * (h / 0.7) ** 1.22
    return (1.0 + (alpha * k) ** (2.0 * nu)) ** (-5.0 / nu)


def wdm_half_mode_k(m_wdm_kev, omega_wdm, h):
    """
    The half-mode wavenumber k_hm where T_WDM(k_hm) = 0.5, i.e. where the
    WDM power spectrum is suppressed to half the CDM value. Solved
    directly from T_WDM's functional form: (alpha*k)^(2*nu) = 2^(nu/5) - 1.
    """
    nu = 1.12
    alpha = 0.049 * m_wdm_kev ** (-1.11) * (omega_wdm / 0.25) ** 0.11 * (h / 0.7) ** 1.22
    return (2.0 ** (nu / 5.0) - 1.0) ** (1.0 / (2.0 * nu)) / alpha


def T_FDM(k, m_a22, h):
    """
    Fuzzy dark matter (ultralight scalar-field DM) transfer function
    (Hu, Barkana & Gruzinov 2000, Phys. Rev. Lett. 85, 1158,
    arXiv:astro-ph/0003365, eq. 8-9; cross-checked against Kulkarni &
    Ostriker 2022, MNRAS 510, 1425, eq. 6, who use the identical formula
    inside an extended Press-Schechter halo mass function calculation):

        T_F(k) = cos(x^3) / (1 + x^8)
        x = 1.61 * m_a22^(1/18) * k_phys / k_Jeq
        k_Jeq = 9 * m_a22^0.5   [Mpc^-1, physical -- NOT h/Mpc]

    m_a22 = m_a / 1e-22 eV, the FDM particle mass in units of 1e-22 eV.

    Unlike Viel et al.'s WDM alpha (already in h^-1 Mpc), k_Jeq here is in
    *physical* Mpc^-1 -- verified against two independent sources, since
    this is an easy unit mismatch to get wrong. k is converted from
    foraois' standard h/Mpc convention internally (k_phys = k * h) so
    callers don't need to remember this.

    T_F itself oscillates and can go negative at large k (from cos(x^3));
    HBG00 note this reflects real transfer-function ringing beyond the
    main cutoff, not a fitting artifact -- and since P_FDM = T_F(k)^2 *
    P_CDM(k) is always used squared (see CosmoData._dm_transfer_function),
    the sign doesn't matter for P(k) itself, only for e.g. the half-mode
    solve below, which stays within T_F's first (monotonic, positive)
    lobe.

    Parameters
    ----------
    k : array-like, h/Mpc
    m_a22 : float
    h : float

    Returns
    -------
    np.ndarray, same shape as k
    """
    k = np.asarray(k, dtype=float)
    k_phys = k * h  # h/Mpc -> Mpc^-1 (HBG00's k_Jeq has no h in it)
    k_Jeq = 9.0 * m_a22**0.5
    x = 1.61 * m_a22 ** (1.0 / 18.0) * k_phys / k_Jeq
    return np.cos(x**3) / (1.0 + x**8)


def fdm_half_mode_k(m_a22, h):
    """
    The half-mode wavenumber k_hm where T_FDM(k_hm) = 0.5 (same
    definition as wdm_half_mode_k -- the standard "half-mode" convention,
    e.g. Schneider et al. 2012 -- NOT the "half-power" point T_F^2 = 0.5
    HBG00's own eq. 9 quotes, which is a different, larger k).

    T_F(k) isn't algebraically invertible (unlike T_WDM), so this solves
    it numerically -- but only within x in [0, (pi/2)^(1/3)], where
    cos(x^3) decreases monotonically from 1 to 0 and 1+x^8 is strictly
    increasing, so T_F is guaranteed strictly decreasing from 1 to 0 and
    the root is unique; that range comfortably brackets T_F=0.5 (which
    empirically falls around x~0.9) without risking picking up one of
    T_F's later oscillations.
    """
    x_max = (np.pi / 2.0) ** (1.0 / 3.0)
    x_hm = brentq(lambda x: np.cos(x**3) / (1.0 + x**8) - 0.5, 1e-8, x_max)
    k_Jeq = 9.0 * m_a22**0.5
    k_phys_hm = x_hm * k_Jeq / (1.61 * m_a22 ** (1.0 / 18.0))
    return k_phys_hm / h  # Mpc^-1 -> h/Mpc
