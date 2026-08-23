import numpy as np
from scipy import integrate

from foraois.cosmo_utils import ensure_delta_col_covers

try:
    import numba as nb

    _HAVE_NUMBA = True
except ImportError:
    nb = None
    _HAVE_NUMBA = False

# ---------------------------------------------------------------------------
# J(u) lookup table -- Parkinson, Cole & Helly (2008) Appendix A, eq. A7
# ---------------------------------------------------------------------------
# J(u_res) = integral_0^u_res (1 + 1/u^2)^(gamma1/2) du feeds the unresolved-
# accretion fraction F (eq. A6). It depends only on gamma1 (fixed for a
# given run), not on halo mass/redshift, so it's tabulated once here and
# interpolated per halo per step -- the same pattern CosmoData already uses
# for sigma(M) and delta_col(z). The integrand is (integrably) singular at
# u=0 for gamma1 != 0; scipy.integrate.quad handles this without issue
# (verified against the gamma1=0 closed form J(u)=u, where it's exact to
# machine precision).
# ---------------------------------------------------------------------------


def _build_j_table(gamma1, u_max=50.0, n_grid=400):
    if gamma1 == 0.0:
        u_grid = np.linspace(0.0, u_max, n_grid)
        return u_grid, u_grid.copy()

    def integrand(u):
        return (1.0 + 1.0 / u**2) ** (gamma1 / 2.0)

    u_grid = np.concatenate([[0.0], np.geomspace(1e-8, u_max, n_grid - 1)])
    J = np.zeros(n_grid)
    for i in range(1, n_grid):
        J[i], _ = integrate.quad(integrand, 0.0, u_grid[i], limit=200)
    return u_grid, J


# ---------------------------------------------------------------------------
# Shared (non-numba) branching-rate machinery -- PCH08 Appendix A
# ---------------------------------------------------------------------------
# Used by both the scalar draw_progenitor_masses() and the vectorised
# _build_forest_numpy(): every operation below is a plain elementwise numpy
# expression, so it works unchanged whether M2 etc. are scalars or arrays.
# The numba kernel below has its own inlined copy since it can't call
# scipy-backed sigma_fn/alpha_fn from inside @njit.
#
# Only gamma1 >= 0 is implemented -- eq. A8's mu = alpha_h branch. This
# covers PCH08's own best-fit gamma1 = 0.38; the gamma1 < 0 branch (a
# mass-resolution-dependent log-ratio for mu) is left unimplemented rather
# than guessed at without a reference case to validate against.
# ---------------------------------------------------------------------------


def _branching_rate_terms(M2, M_res, sigma_fn, alpha_fn, delta0, d_omega, G0, gamma1, gamma2):
    if gamma1 < 0:
        raise NotImplementedError(
            "gamma1 < 0 is not implemented (Appendix A eq. A8's mu definition "
            "for this branch hasn't been validated here); PCH (2008)'s own "
            "best-fit gamma1=0.38 > 0 doesn't need it."
        )

    qres = M_res / M2

    # When M2 < 2*M_res (qres >= 0.5), q's valid range [qres, 0.5] is empty:
    # no split can produce two resolved fragments (q and 1-q would both need
    # to be >= qres, impossible once qres > 0.5). Nupper must be exactly 0
    # there, not the NaN log(0.5/qres) with qres>0.5 would otherwise give.
    # qres_safe (clamped below 0.5) is used only to keep the B/beta/eta
    # algebra finite for every array element; the clamped entries' Nupper is
    # overwritten to 0 afterwards regardless of what that algebra produces.
    no_split_possible = qres >= 0.5
    # 0.25 (not e.g. 0.499): any well-conditioned placeholder works here since
    # the whole computation is discarded for these entries via the Nupper
    # override below, but something close to the 0.5 boundary makes beta's
    # own log(0.5/qres) denominator blow up.
    qres_safe = np.where(no_split_possible, 0.25, qres)

    sigma2 = sigma_fn(np.log10(M2))
    sigma_h = sigma_fn(np.log10(M2 / 2.0))
    sigma_res = sigma_fn(np.log10(M_res))
    alpha_h = alpha_fn(np.log10(M2 / 2.0))

    V_res = sigma_res**2 / (sigma_res**2 - sigma2**2) ** 1.5
    V_half = sigma_h**2 / (sigma_h**2 - sigma2**2) ** 1.5

    beta = np.log(V_half / V_res) / np.log(0.5 / qres_safe)
    B = V_half / 0.5**beta
    mu = alpha_h  # gamma1 >= 0 branch (eq. A8)
    eta = beta - 1.0 - gamma1 * mu

    # PCH08 eq. 12: S(q) = sqrt(2/pi)*B*alpha_h*q^(eta-1) * [G0/2^(mu*gamma1)]
    # * (delta/sigma2)^gamma2 * (sigma_h/sigma2)^gamma1 -- G0 is *divided* by
    # 2^(mu*gamma1) (confirmed against the paper's raw LaTeX source,
    # \frac{G_0}{2^{\mu\gamma_1}}), not multiplied. This factor is meant to
    # cancel against R(q)'s own +2^(mu*gamma1) (eq. 13's (2q)^mu term,
    # raised to gamma1) in the product S(q)*R(q) -- verified by expanding
    # S(q)*R(q) against eq. 1-3/6-7's original (unperturbed-EPS x G-modifier)
    # target density: the two factors must have opposite sign to cancel,
    # leaving no explicit 2^(mu*gamma1) dependence in the true target.
    # Multiplying instead of dividing here left them adding instead of
    # cancelling, inflating the accepted merger rate by 2^(2*mu*gamma1) --
    # a real, undetected bug until this term was checked against eq. 12's
    # own raw LaTeX source rather than trusted from memory (see
    # tests/test_pch_validation.py's independent-of-S(q)/R(q) regression
    # test, which recomputes the target density straight from eq. 1-3/6-7
    # rather than re-using this function's own S_coeff/eta/beta machinery).
    S_coeff = (
        np.sqrt(2.0 / np.pi)
        * B
        * alpha_h
        * G0
        * (2.0 ** (-mu * gamma1))
        * (delta0 / sigma2) ** gamma2
        * (sigma_h / sigma2) ** gamma1
    )

    small_eta = np.abs(eta) < 1e-8
    eta_safe = np.where(small_eta, 1.0, eta)
    integral_q = np.where(small_eta, np.log(0.5 / qres_safe), (0.5**eta - qres_safe**eta) / eta_safe)
    Nupper = np.where(no_split_possible, 0.0, S_coeff * d_omega * integral_q)

    return {
        "qres": qres,
        "sigma2": sigma2,
        "sigma_h": sigma_h,
        "sigma_res": sigma_res,
        "alpha_h": alpha_h,
        "B": B,
        "beta": beta,
        "mu": mu,
        "eta": eta,
        "Nupper": Nupper,
        "S_coeff": S_coeff,
        "d_omega": d_omega,
        "no_split_possible": no_split_possible,
    }


def _draw_progenitor_ratio(u, qres, eta):
    small_eta = np.abs(eta) < 1e-8
    eta_safe = np.where(small_eta, 1.0, eta)
    q_general = (qres**eta + u * (0.5**eta - qres**eta)) ** (1.0 / eta_safe)
    q_loguniform = qres * (0.5 / qres) ** u
    return np.where(small_eta, q_loguniform, q_general)


def _rejection_ratio(q, M2, terms, sigma_fn, alpha_fn, gamma1):
    sigma1_q = sigma_fn(np.log10(q * M2))
    alpha1_q = alpha_fn(np.log10(q * M2))
    V_q = sigma1_q**2 / (sigma1_q**2 - terms["sigma2"] ** 2) ** 1.5
    return (
        (alpha1_q / terms["alpha_h"])
        * (V_q / (terms["B"] * q ** terms["beta"]))
        * ((2.0 * q) ** terms["mu"] * sigma1_q / terms["sigma_h"]) ** gamma1
    )


def _unresolved_accretion_fraction(M2, terms, delta0, d_omega, G0, gamma2, j_u_grid, j_values):
    u_res = terms["sigma2"] / np.sqrt(np.maximum(terms["sigma_res"] ** 2 - terms["sigma2"] ** 2, 1e-300))
    J_u_res = np.interp(u_res, j_u_grid, j_values)
    F = np.sqrt(2.0 / np.pi) * J_u_res * (G0 / terms["sigma2"]) * (delta0 / terms["sigma2"]) ** gamma2 * d_omega
    return np.clip(F, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Numba JIT kernel
# ---------------------------------------------------------------------------
# Compiled once on first call, then cached.  Uses nb.prange to parallelise
# over trees across all available CPU cores (set NUMBA_NUM_THREADS or let
# Numba auto-detect).
#
# All array arguments must be plain NumPy arrays of float64 — no dicts, no
# class instances.  The wrapper ``PCHMergerTree.build_forest_numba`` handles
# the translation between Python objects and these raw arrays.
#
# Implements the same PCH08 Appendix A algorithm as _build_forest_numpy, but
# inlined into scalar per-halo operations since numba can't call the
# scipy-backed helpers above. gamma1 < 0 is not supported here either (see
# _branching_rate_terms).
# ---------------------------------------------------------------------------


if _HAVE_NUMBA:

    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _build_forest_kernel(
        M0_array,  # (N,)   initial halo masses
        z_steps,  # (S,)   redshift grid edges
        dc_z_grid,  # (Nz,) precomputed redshift grid for delta_col table
        dc_dc_grid,  # (Nz,) delta_col values on that grid
        logmass_grid,  # (Nm,) log10(M) grid for sigma table
        sigma_grid,  # (Nm,) sigma(M) values
        alpha_grid,  # (Nm,) alpha(M) = |dlnsigma/dlnM| values, same grid
        j_u_grid,  # (Nj,) u grid for the J(u) lookup table
        j_values,  # (Nj,) J(u) values on that grid
        G0,  # scalar PCH parameter
        gamma1,  # scalar PCH parameter (must be >= 0)
        gamma2,  # scalar PCH parameter
        M_res,  # scalar resolution mass
    ):
        """
        Core Numba kernel.  Evolves N trees independently across S-1 redshift
        steps.  Returns the main-progenitor mass history: shape (N, S-1), plus
        two more (N, S-1) arrays decomposing each step's mass change into a
        smooth and a merger-driven channel -- see the module docstring note
        above _build_forest_numpy for what these mean and the exact identity
        they satisfy relative to mass_history.

        Each tree index is processed by a separate thread (nb.prange) so there
        is zero inter-thread communication and no locking overhead.
        """
        N = M0_array.shape[0]
        S = z_steps.shape[0]
        n_steps = S - 1

        mass_history = np.zeros((N, n_steps), dtype=np.float64)
        smooth_accretion = np.zeros((N, n_steps), dtype=np.float64)
        merger_mass = np.zeros((N, n_steps), dtype=np.float64)
        inv_sqrt2pi = 1.0 / np.sqrt(2.0 * np.pi)

        for i in nb.prange(N):  # <-- parallel over trees
            m = M0_array[i]

            for j in range(n_steps):
                z0 = z_steps[j]
                z1 = z_steps[j + 1]
                d0 = _interp_sorted(z0, dc_z_grid, dc_dc_grid)
                d1 = _interp_sorted(z1, dc_z_grid, dc_dc_grid)
                d_omega = d1 - d0

                if m < M_res:
                    mass_history[i, j] = m
                    continue

                qres = M_res / m
                sigma2 = _interp_sorted(np.log10(m), logmass_grid, sigma_grid)
                sigma_h = _interp_sorted(np.log10(m / 2.0), logmass_grid, sigma_grid)
                sigma_res = _interp_sorted(np.log10(M_res), logmass_grid, sigma_grid)
                alpha_h = _interp_sorted(np.log10(m / 2.0), logmass_grid, alpha_grid)

                # M2 < 2*M_res (qres >= 0.5): q's valid range [qres, 0.5] is
                # empty, no split can produce two resolved fragments -- Nupper
                # is exactly 0 (not the NaN log(0.5/qres) with qres>0.5 would
                # otherwise give). Skip straight to unresolved accretion.
                if qres >= 0.5:
                    beta = 0.0
                    B = 0.0
                    mu = alpha_h
                    eta = 1.0
                    Nupper = 0.0
                else:
                    V_res = sigma_res**2 / (sigma_res**2 - sigma2**2) ** 1.5
                    V_half = sigma_h**2 / (sigma_h**2 - sigma2**2) ** 1.5

                    beta = np.log(V_half / V_res) / np.log(0.5 / qres)
                    B = V_half / 0.5**beta
                    mu = alpha_h  # gamma1 >= 0 branch (eq. A8)
                    eta = beta - 1.0 - gamma1 * mu

                    # See _branching_rate_terms's own comment on this
                    # line -- eq. 12 divides G0 by 2^(mu*gamma1).
                    S_coeff = (
                        np.sqrt(2.0 / np.pi)
                        * B
                        * alpha_h
                        * G0
                        * (2.0 ** (-mu * gamma1))
                        * (d0 / sigma2) ** gamma2
                        * (sigma_h / sigma2) ** gamma1
                    )

                    if abs(eta) < 1e-8:
                        integral_q = np.log(0.5 / qres)
                    else:
                        integral_q = (0.5**eta - qres**eta) / eta
                    Nupper = S_coeff * d_omega * integral_q

                # unresolved accretion fraction (applies regardless of split)
                u_res = sigma2 / np.sqrt(max(sigma_res**2 - sigma2**2, 1e-300))
                J_u_res = _interp_sorted(u_res, j_u_grid, j_values)
                F = inv_sqrt2pi * J_u_res * (G0 / sigma2) * (d0 / sigma2) ** gamma2 * d_omega
                if F < 0.0:
                    F = 0.0
                elif F > 1.0:
                    F = 1.0

                r1 = np.random.rand()
                if r1 > Nupper:
                    smooth_accretion[i, j] = F * m
                    mass_history[i, j] = m * (1.0 - F)
                    m = m * (1.0 - F)
                    continue

                u2 = np.random.rand()
                if abs(eta) < 1e-8:
                    q = qres * (0.5 / qres) ** u2
                else:
                    q = (qres**eta + u2 * (0.5**eta - qres**eta)) ** (1.0 / eta)

                sigma1_q = _interp_sorted(np.log10(q * m), logmass_grid, sigma_grid)
                alpha1_q = _interp_sorted(np.log10(q * m), logmass_grid, alpha_grid)
                V_q = sigma1_q**2 / (sigma1_q**2 - sigma2**2) ** 1.5
                R = (alpha1_q / alpha_h) * (V_q / (B * q**beta)) * ((2.0 * q) ** mu * sigma1_q / sigma_h) ** gamma1

                r3 = np.random.rand()
                if r3 > R:
                    smooth_accretion[i, j] = F * m
                    mass_history[i, j] = m * (1.0 - F)
                    m = m * (1.0 - F)
                    continue

                M1 = q * m
                M2 = m * (1.0 - F - q)
                smooth_accretion[i, j] = F * m
                merger_mass[i, j] = M1 if M1 < M2 else M2
                m = M1 if M1 >= M2 else M2
                mass_history[i, j] = m

            # end step loop
        # end tree loop (prange)

        return mass_history, smooth_accretion, merger_mass

    @nb.njit(cache=True)
    def _interp_sorted(x, xp, fp):
        """
        Scalar linear interpolation assuming xp is sorted ascending.
        Equivalent to np.interp(x, xp, fp) but callable from inside @njit.
        Clamps to boundary values outside the range.
        """
        n = xp.shape[0]

        if x <= xp[0]:
            return fp[0]
        if x >= xp[n - 1]:
            return fp[n - 1]

        # Binary search
        lo = 0
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) >> 1
            if xp[mid] <= x:
                lo = mid
            else:
                hi = mid

        t = (x - xp[lo]) / (xp[hi] - xp[lo])
        return fp[lo] + t * (fp[hi] - fp[lo])
else:
    _build_forest_kernel = None
    _interp_sorted = None


# ---------------------------------------------------------------------------
# NumPy vectorised implementation (no Numba dependency)
# ---------------------------------------------------------------------------
# Runs all N trees through the same redshift step simultaneously using
# array operations.  Faster than the serial Python loop for N > ~100, and
# immediately usable without a JIT warm-up cost.
# ---------------------------------------------------------------------------


def _build_forest_numpy(
    M0_array,
    z_steps,
    dc_z_grid,
    dc_dc_grid,
    sigma_fn,
    alpha_fn,
    j_u_grid,
    j_values,
    G0,
    gamma1,
    gamma2,
    M_res,
):
    """
    Vectorised NumPy implementation, using the PCH08 Appendix A branching-
    rate/rejection-sampling algorithm (see module docstring above).

    Returns
    -------
    mass_history : np.ndarray, shape (N, n_steps)
        Main-progenitor mass at the end of each redshift step for each tree.
    split_events : list of dicts
        One entry per step that had at least one split, for later analysis.
    smooth_accretion : np.ndarray, shape (N, n_steps)
        Per-halo, per-step mass gained via unresolved (below-M_res) smooth
        accretion, expressed in the *chronological* (forward-time) sense --
        i.e. this is F*M evaluated at the step's earlier (lower-z) endpoint,
        not a quantity you difference out of mass_history yourself. F (eq.
        A6) is a fully analytic, deterministic rate already computed inside
        every step regardless of whether a split occurs; this just exposes
        it instead of discarding it. Together with merger_mass below, it
        satisfies, for every halo/step resolved (mass_history != 0) on
        *both* sides of the step (chronological/forward-time index k, k+1):
            M_chronological[k+1] - M_chronological[k]
                == smooth_accretion[..., k] + merger_mass[..., k]
        The one exception is the single step a halo first drops below
        M_res: mass_history is zeroed there (this backend's "0 = not
        resolved" convention), and smooth_accretion/merger_mass are zeroed
        alongside it for consistency, so the identity isn't meaningful for
        that specific transition step (see tests/test_pch_trees.py's
        mass-conservation check, which excludes it explicitly).
        build_forest_numba's identity has no such exception -- see its own
        docstring. Intended
        use: a smooth cosmological-accretion rate (smooth_accretion/dt) for
        driving continuous gas inflow in a galaxy-formation model, as
        distinct from the discrete merger_mass events below -- differencing
        the total mass_history instead conflates "smooth accretion" with
        "real, discrete merger events" into one noisy-looking derivative.
    merger_mass : np.ndarray, shape (N, n_steps)
        Per-halo, per-step mass gained via a resolved merger (the smaller
        of the two split progenitors, M1/M2, merging into the tracked main
        branch) -- 0 at every step without an accepted split. A sparse,
        event-like channel: nonzero entries are exactly the tree's merger
        history, without needing to post-process split_events.
    """
    N = len(M0_array)
    n_steps = len(z_steps) - 1

    M = M0_array.copy().astype(np.float64)
    alive = M >= M_res
    mass_history = np.zeros((N, n_steps), dtype=np.float64)
    smooth_accretion = np.zeros((N, n_steps), dtype=np.float64)
    merger_mass = np.zeros((N, n_steps), dtype=np.float64)
    split_events = []

    for j in range(n_steps):
        z0 = z_steps[j]
        z1 = z_steps[j + 1]
        d0 = float(np.interp(z0, dc_z_grid, dc_dc_grid))
        d1 = float(np.interp(z1, dc_z_grid, dc_dc_grid))
        d_omega = d1 - d0

        # Dummy positive value for dead trees (masked out below via `alive`).
        # Must not equal M_res exactly: qres = M_res/M_safe = 1 there, making
        # sigma_res == sigma2 and dividing by zero in V(q)'s (sigma1^2 -
        # sigma2^2) denominator.
        M_safe = np.where(alive, M, 2.0 * M_res)
        terms = _branching_rate_terms(M_safe, M_res, sigma_fn, alpha_fn, d0, d_omega, G0, gamma1, gamma2)

        F = _unresolved_accretion_fraction(M_safe, terms, d0, d_omega, G0, gamma2, j_u_grid, j_values)

        r1 = np.random.rand(N)
        split_mask = alive & (r1 <= terms["Nupper"])

        u2 = np.random.rand(N)
        q = _draw_progenitor_ratio(u2, terms["qres"], terms["eta"])
        R = _rejection_ratio(q, M_safe, terms, sigma_fn, alpha_fn, gamma1)
        r3 = np.random.rand(N)
        accepted = split_mask & (r3 <= R)

        M1 = q * M_safe
        M2 = M_safe * (1.0 - F - q)
        M_after_split = np.where(M1 >= M2, M1, M2)

        M_next = M_safe * (1.0 - F)  # default: no split (rejected or not triggered)
        M_next = np.where(accepted, M_after_split, M_next)
        M = np.where(alive, M_next, M)

        alive = alive & (M >= M_res)
        mass_history[:, j] = np.where(alive, M, 0.0)

        # Masked with the *post*-step alive mask, matching mass_history's
        # own "0 means not resolved" convention -- a tree that drops below
        # M_res this step reports 0 for all three arrays at this step, not
        # just mass_history, so the conservation identity in the docstring
        # above only needs to (and does) hold while a halo remains resolved
        # on both sides of a step (see tests/test_pch_trees.py).
        smooth_accretion[:, j] = np.where(alive, F * M_safe, 0.0)
        merger_mass[:, j] = np.where(alive & accepted, np.minimum(M1, M2), 0.0)

        n_splits = int(accepted.sum())
        if n_splits > 0:
            split_events.append(
                {
                    "z_step": (z0, z1),
                    "n_splits": n_splits,
                    "M1": M1[accepted].copy(),
                    "M2": M2[accepted].copy(),
                    "tree_ids": np.where(accepted)[0].copy(),
                }
            )

    return mass_history, split_events, smooth_accretion, merger_mass


# ---------------------------------------------------------------------------
# PCHMergerTree class
# ---------------------------------------------------------------------------


class PCHMergerTree:
    """
    Parkinson, Cole & Helly (2008) merger tree algorithm.

    Branching probability and progenitor-mass sampling implement the exact
    Appendix A algorithm from the paper (power-law envelope + rejection
    sampling against the full 2-parameter modifier function), rather than
    the coarser single-point heuristic (P = 5*A*G evaluated at a single mass,
    then a mass-ratio drawn from an unrelated fixed power law) used
    previously. See _branching_rate_terms/_draw_progenitor_ratio/
    _rejection_ratio above for the numpy/scalar implementation and
    _build_forest_kernel for the numba equivalent.

    One architectural simplification kept relative to the paper: PCH08's
    algorithm adaptively chooses the redshift step size Delta_z per halo per
    step to keep the expected number of splits small (their eq. A5-A8
    procedure). This class instead uses a single, fixed dz shared by every
    halo in a forest, because that shared grid is what makes
    build_forest_numpy/build_forest_numba vectorisable across haloes at all.
    This means Nupper (the per-step split probability) isn't bounded small
    by construction the way it is in the adaptive original -- use
    foraois.diagnostics to check Nupper stays comfortably below 1 for a
    given dz before trusting the output.

    Performance backends
    ---------------------
    * ``build_tree()``  — serial single-tree method retained for testing /
      debugging; internally uses the precomputed ``delta_col`` table from
      ``CosmoData`` (O(log N) lookup instead of full ODE integration per
      step).
    * ``build_forest_numpy()``  — vectorised NumPy implementation; runs N
      trees simultaneously through each redshift step using array operations.
      ~100–1000× faster than calling ``build_tree`` N times.
    * ``build_forest_numba()`` — Numba JIT + parallel implementation; uses
      all CPU cores.  ~8–32× faster than the NumPy version for large N.
      Requires the optional ``numba`` dependency (``pip install
      foraois[numba]``) -- raises ``ImportError`` with that instruction if
      it isn't installed; ``build_tree``/``build_forest_numpy`` need no
      such extra.
    """

    # PCH (2008) best-fit parameters (Parkinson, Cole & Helly 2008, MNRAS
    # 383, 557, Section 3, "Allowing all three parameters to vary,
    # consistently good fits are found with G0 = 0.57, gamma1 = 0.38 and
    # gamma2 = -0.01"). gamma1 was previously 0.1 here, which doesn't match
    # the paper it's attributed to.
    G0 = 0.57
    gamma1 = 0.38
    gamma2 = -0.01

    def __init__(self, cosmo_data, params):
        """
        Parameters
        ----------
        cosmo_data : foraois.cosmo_utils.CosmoData
            Builds the sigma(M) grid as a side effect of construction
            (calls ``cosmo_data._prepare_sigma_grid()``) -- other code
            that needs that grid (e.g. ``MassFunctions``) should be
            constructed after this, not before.
        params : dict
            Currently unused beyond being stored as ``self.params`` --
            PCH08's own rate parameters (G0, gamma1, gamma2) are fixed
            class attributes, not read from ``params``.
        """
        self.cosmo_data = cosmo_data
        self.params = params
        self.pk_data = cosmo_data.get_power_spectrum()

        # Build sigma(M) table (calls _prepare_sigma_grid once)
        self.cosmo_data._prepare_sigma_grid(self.pk_data)

        # Convenience references to the raw grids (backward-compatible)
        self.logmass_grid = self.cosmo_data._logmass
        self.sigma_grid = self.cosmo_data._sigma
        self.dlogsigma_grid = self.cosmo_data._dlogsigma_dlogmass

        # alpha(M) = |dln sigma/dln M|, on the same log-mass grid -- used by
        # the numba kernel (which can't call CosmoData.dlogsigma_at_logmass
        # directly). Note self.dlogsigma_grid is the *signed* raw derivative
        # (negative, since sigma decreases with M); alpha must be positive.
        self.alpha_grid = np.abs(self.dlogsigma_grid)

        # Convenience references to delta_col table built in CosmoData.__init__
        self._dc_z_grid = self.cosmo_data._dc_z_grid
        self._dc_dc_grid = self.cosmo_data._dc_dc_grid

        # J(u) lookup table for the unresolved-accretion fraction (Appendix
        # A, eq. A7) -- depends only on gamma1, built once here.
        self._j_u_grid, self._j_values = _build_j_table(self.gamma1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sigma_at_mass(self, M):
        """Vectorised sigma(M) lookup using the scipy interpolant."""
        return self.cosmo_data.sigma_at_logmass(np.log10(M))

    def _delta_col_at_z(self, z):
        """Cheap delta_col(z) lookup using the precomputed table."""
        return self.cosmo_data.delta_col_at_z(z)

    def _ensure_delta_col_covers(self, z_max):
        """
        Extend the delta_col(z) table if this tree-building call's z_max
        exceeds what's currently tabulated -- see
        cosmo_utils.ensure_delta_col_covers's docstring for the
        silent-clamping failure mode this avoids. Also refreshes this
        class's own cached copy of the grid arrays (used by the numba
        kernel, which needs raw arrays rather than a CosmoData method).
        """
        ensure_delta_col_covers(self.cosmo_data, z_max)
        self._dc_z_grid = self.cosmo_data._dc_z_grid
        self._dc_dc_grid = self.cosmo_data._dc_dc_grid

    # ------------------------------------------------------------------
    # Public methods: dndM_prob / compute_g_modifier
    # (kept for API compatibility; describe only the 2-term (gamma1,
    # gamma2) modifier evaluated at a single mass point -- draw_progenitor_
    # masses/build_forest_* below implement the full Appendix A machinery,
    # of which this is only one ingredient)
    # ------------------------------------------------------------------

    def compute_g_modifier(self, sigma_M1, sigma_M0, delta_z0):
        """
        PCH08's own 2-parameter modifier g(sigma1/sigma0, delta_z0/sigma0)
        = G0 * (sigma1/sigma0)^gamma1 * (delta_z0/sigma0)^gamma2, evaluated
        at a single (M1, M0) point. Only one ingredient of the full
        Appendix A branching-rate/rejection-sampling algorithm -- see
        draw_progenitor_masses/build_forest_numpy for the actual
        tree-building machinery.
        """
        x = sigma_M1 / sigma_M0
        y = delta_z0 / sigma_M0
        return self.G0 * (x**self.gamma1) * (y**self.gamma2)

    def get_mass_variance_at_mass(self, mass):
        """sigma(M) at the given mass [Msun/h], top-hat window."""
        radius = self.cosmo_data.get_radius(mass)
        return self.cosmo_data.get_mass_variance(self.pk_data, radius=radius, window_function_type="top_hat")

    def dndM_prob(self, M1, M0, delta_0, delta_1, sigma_M0):
        """
        Unperturbed-EPS-times-g_modifier progenitor mass distribution at a
        single mass point M1, i.e. eps_term(M1, M0, delta_0, delta_1) *
        compute_g_modifier's g(M1, M0, delta_0) -- the pointwise quantity
        Appendix A's branching rate is built from, not a normalized
        probability density on its own. Returns 0.0 if sigma(M1) <=
        sigma(M0) (M1 not a valid progenitor at this step).
        """
        logM1 = np.log10(M1)
        sigma_M1 = self.cosmo_data.sigma_at_logmass(logM1)

        delta_omega = delta_1 - delta_0
        delta_sigma2 = sigma_M1**2 - sigma_M0**2

        if delta_sigma2 <= 0:
            return 0.0

        dlogsigma_dlogm = self.cosmo_data.dlogsigma_at_logmass(logM1)

        eps_term = (
            (1.0 / np.sqrt(2.0 * np.pi))
            * (delta_omega / delta_sigma2**1.5)
            * dlogsigma_dlogm
            * np.exp(-(delta_omega**2) / (2.0 * delta_sigma2))
        )

        g_mod = self.G0 * (sigma_M1 / sigma_M0) ** self.gamma1 * (delta_0 / sigma_M0) ** self.gamma2

        return eps_term * g_mod

    def branching_rate_terms(self, M2, M_res, delta0, d_omega):
        """Public wrapper for diagnostics (see foraois.diagnostics)."""
        return _branching_rate_terms(
            np.asarray(M2, dtype=np.float64),
            M_res,
            self.cosmo_data.sigma_at_logmass,
            self.cosmo_data.dlogsigma_at_logmass,
            delta0,
            d_omega,
            self.G0,
            self.gamma1,
            self.gamma2,
        )

    def draw_progenitor_masses(self, M0, delta_0, delta_1, sigma_M0, M_res):
        """
        Single-halo version of the Appendix A algorithm used by
        build_forest_numpy/build_forest_numba. sigma_M0 is accepted for
        backward compatibility but not used directly -- sigma(M0) is
        recomputed internally alongside sigma(M0/2), sigma(M_res), so that
        this scalar path shares exactly the same machinery (and thus the
        same numerical behaviour) as the vectorised backends.
        """
        d_omega = delta_1 - delta_0
        if d_omega <= 0:
            return []

        terms = _branching_rate_terms(
            M0,
            M_res,
            self.cosmo_data.sigma_at_logmass,
            self.cosmo_data.dlogsigma_at_logmass,
            delta_0,
            d_omega,
            self.G0,
            self.gamma1,
            self.gamma2,
        )
        F = float(
            _unresolved_accretion_fraction(
                M0,
                terms,
                delta_0,
                d_omega,
                self.G0,
                self.gamma2,
                self._j_u_grid,
                self._j_values,
            )
        )

        if np.random.rand() > terms["Nupper"]:
            return [M0 * (1.0 - F)] if M0 * (1.0 - F) >= M_res else []

        u2 = np.random.rand()
        q = float(_draw_progenitor_ratio(u2, terms["qres"], terms["eta"]))
        R = float(
            _rejection_ratio(
                q,
                M0,
                terms,
                self.cosmo_data.sigma_at_logmass,
                self.cosmo_data.dlogsigma_at_logmass,
                self.gamma1,
            )
        )

        if np.random.rand() > R:
            return [M0 * (1.0 - F)] if M0 * (1.0 - F) >= M_res else []

        M1 = q * M0
        M2 = M0 * (1.0 - F - q)

        progenitors = []
        if M1 >= M_res:
            progenitors.append(M1)
        if M2 >= M_res:
            progenitors.append(M2)
        return progenitors

    # ------------------------------------------------------------------
    # build_tree  —  single-tree serial method (debugging / reference)
    # ------------------------------------------------------------------

    def build_tree(self, M0, z0, z_max, M_res, dz=0.1):
        """
        Recursively step a single tree from z0 to z_max.

        Now uses the precomputed delta_col table (``_delta_col_at_z``) instead
        of calling ``get_linear_growth_and_collapse`` at every step, giving a
        ~100× speedup for the growth-factor part.  The overall method is still
        serial; use ``build_forest_numpy`` or ``build_forest_numba`` for
        large N.
        """
        self._ensure_delta_col_covers(z_max)

        tree = []
        current_z = z0
        M_cur = M0

        sigma_M0 = self._sigma_at_mass(M_cur)
        delta_0 = self._delta_col_at_z(current_z)

        z_steps = np.arange(z0, z_max + dz * 0.5, dz)
        max_steps = len(z_steps)

        for k in range(max_steps - 1):
            if M_cur < M_res:
                break

            z_next = z_steps[k + 1]
            delta_1 = self._delta_col_at_z(z_next)

            progenitors = self.draw_progenitor_masses(M_cur, delta_0, delta_1, sigma_M0, M_res)

            if progenitors:
                tree.append(
                    {
                        "redshift": z_next,
                        "parent_mass": M_cur,
                        "progenitors": progenitors,
                    }
                )
                M_cur = max(progenitors)
                sigma_M0 = self._sigma_at_mass(M_cur)

            delta_0 = delta_1

        return tree

    # ------------------------------------------------------------------
    # build_full_tree  —  full branching structure for one tree (illustration)
    # ------------------------------------------------------------------

    def build_full_tree(self, M0, z0, z_max, M_res, dz=0.1, max_nodes=200_000):
        """
        Grow the *entire* merger tree for one halo -- every progenitor at
        every split, recursively, down to M_res -- not just the main
        progenitor branch build_tree()/build_forest_*() track. This is what
        a Figure-4-style (Nadler et al. 2023) dendrogram plot needs: that
        figure shows every branch fanning out from the root, not a single
        mass-growth line.

        This is necessarily serial and grows one node at a time (the whole
        point is the branching structure itself), so it's for illustrating
        a single example tree, not for building statistical samples --
        use build_forest_numpy/build_forest_numba for those, which are
        60-1000x faster but only return the main-progenitor mass history.

        Node count grows roughly as (M0/M_res) for a realistic mass
        function, so this can get large fast; max_nodes is a safety cap
        (raises RuntimeError if exceeded) rather than silently running for
        a very long time or exhausting memory -- raise M_res or lower
        max_nodes deliberately if you hit it.

        A node is only recorded at an actual split (or at the branch's
        endpoint -- M_res reached or z_max reached); steps where nothing
        happens but unresolved-accretion mass loss (draw_progenitor_masses
        returning a single "continuing" fragment) are absorbed silently
        into the running mass rather than creating a node of their own.
        Recording a node at every dz grid step regardless produces a
        visually misleading plot -- long runs of near-coincident points
        between real branch events -- and doesn't match how Nadler et al.
        2023's own Figure 4 (or any DHalo-style tree, one node per real
        event) looks.

        Returns
        -------
        list of dict, one per node, each with keys:
            id : int -- unique within this tree
            mass : float
            redshift : float
            descendant_id : int or None -- None only for the root (at z0)
            is_main : bool -- True if this node is the more massive of the
                two progenitors at its split (or the sole progenitor, if
                the other fragment fell below M_res). Following is_main
                links from the root traces out the single "main branch"
                the other build_* methods track.
        """
        self._ensure_delta_col_covers(z_max)

        z_steps = np.arange(z0, z_max + dz * 0.5, dz)
        n_steps = len(z_steps)

        nodes = [
            {
                "id": 0,
                "mass": float(M0),
                "redshift": float(z0),
                "descendant_id": None,
                "is_main": True,
            }
        ]
        next_id = 1

        # Stack of (node_id, mass, step_index) branches still to grow, where
        # node_id is the most recently *recorded* node for that branch.
        stack = [(0, float(M0), 0)]

        while stack:
            if len(nodes) > max_nodes:
                raise RuntimeError(
                    f"build_full_tree exceeded max_nodes={max_nodes} -- raise "
                    "M_res (fewer, larger-mass-ratio-resolved branches) or "
                    "raise max_nodes deliberately if you really need this many."
                )

            node_id, m_cur, step_idx = stack.pop()
            sigma_M0 = self._sigma_at_mass(m_cur)
            delta_0 = self._delta_col_at_z(z_steps[step_idx])
            z_cur = z_steps[step_idx]
            just_split = False

            for k in range(step_idx, n_steps - 1):
                if m_cur < M_res:
                    break

                z_next = z_steps[k + 1]
                delta_1 = self._delta_col_at_z(z_next)
                progenitors = self.draw_progenitor_masses(m_cur, delta_0, delta_1, sigma_M0, M_res)

                if len(progenitors) == 2:
                    m1, m2 = progenitors
                    id1, id2 = next_id, next_id + 1
                    next_id += 2
                    nodes.append(
                        {
                            "id": id1,
                            "mass": m1,
                            "redshift": float(z_next),
                            "descendant_id": node_id,
                            "is_main": m1 >= m2,
                        }
                    )
                    nodes.append(
                        {
                            "id": id2,
                            "mass": m2,
                            "redshift": float(z_next),
                            "descendant_id": node_id,
                            "is_main": m2 > m1,
                        }
                    )
                    # continue the larger branch inline; push the smaller one
                    # to grow later (both eventually get grown -- order
                    # doesn't matter, this just bounds recursion depth to
                    # O(n_steps) instead of O(n_nodes))
                    if m1 >= m2:
                        stack.append((id2, m2, k + 1))
                        node_id, m_cur = id1, m1
                    else:
                        stack.append((id1, m1, k + 1))
                        node_id, m_cur = id2, m2
                    sigma_M0 = self._sigma_at_mass(m_cur)
                    z_cur, just_split = z_next, True
                elif len(progenitors) == 1:
                    # no split this step -- absorb the (unresolved-accretion-
                    # reduced) mass silently, no new node
                    m_cur = progenitors[0]
                    sigma_M0 = self._sigma_at_mass(m_cur)
                    z_cur, just_split = z_next, False
                else:
                    break  # both fragments fell below M_res

                delta_0 = delta_1

            if not just_split:
                # branch ended (M_res or z_max reached) without its last
                # event being a split already recorded above -- record its
                # terminal state as a leaf so the final edge reflects where
                # it actually stopped
                nodes.append(
                    {
                        "id": next_id,
                        "mass": m_cur,
                        "redshift": float(z_cur),
                        "descendant_id": node_id,
                        "is_main": True,
                    }
                )
                next_id += 1

        return nodes

    # ------------------------------------------------------------------
    # build_forest_numpy  —  vectorised NumPy (N trees, no Numba)
    # ------------------------------------------------------------------

    def build_forest_numpy(self, M0_array, z0, z_max, M_res, dz=0.1):
        """
        Evolve N merger trees simultaneously using vectorised NumPy.

        Parameters
        ----------
        M0_array : array-like, shape (N,)
            Halo masses at z0 in Msun/h.
        z0 : float
            Starting redshift.
        z_max : float
            Maximum redshift.
        M_res : float
            Mass resolution threshold; trees with M < M_res are stopped.
        dz : float
            Redshift step size (default 0.1).

        Returns
        -------
        mass_history : np.ndarray, shape (N, n_steps)
            Main-progenitor mass at each step for each tree.
            Zero indicates the tree has dropped below M_res.
        split_events : list of dict
            Merger events keyed by step; each dict contains:
            ``z_step``, ``n_splits``, ``M1``, ``M2``, ``tree_ids``.
        z_steps : np.ndarray, shape (n_steps+1,)
            The redshift grid used (useful for plotting).
        smooth_accretion : np.ndarray, shape (N, n_steps)
            Per-halo, per-step mass gained via unresolved smooth accretion
            (analytic, deterministic -- see ``_build_forest_numpy``'s
            docstring for the exact forward-time bookkeeping and the
            conservation identity relative to ``mass_history``). Use
            ``smooth_accretion / dt`` for a smooth accretion-rate driver
            instead of differencing ``mass_history``.
        merger_mass : np.ndarray, shape (N, n_steps)
            Per-halo, per-step mass gained via a resolved merger; 0 except
            at accepted-split steps. The discrete-event counterpart to
            ``smooth_accretion`` above.
        """
        self._ensure_delta_col_covers(z_max)

        M0_array = np.asarray(M0_array, dtype=np.float64)
        z_steps = np.arange(z0, z_max + dz * 0.5, dz)

        mass_history, split_events, smooth_accretion, merger_mass = _build_forest_numpy(
            M0_array=M0_array,
            z_steps=z_steps,
            dc_z_grid=self._dc_z_grid,
            dc_dc_grid=self._dc_dc_grid,
            sigma_fn=self.cosmo_data.sigma_at_logmass,
            alpha_fn=self.cosmo_data.dlogsigma_at_logmass,
            j_u_grid=self._j_u_grid,
            j_values=self._j_values,
            G0=self.G0,
            gamma1=self.gamma1,
            gamma2=self.gamma2,
            M_res=float(M_res),
        )
        return mass_history, split_events, z_steps, smooth_accretion, merger_mass

    # ------------------------------------------------------------------
    # build_forest_numba  —  JIT + parallel (N trees, all CPU cores)
    # ------------------------------------------------------------------

    def build_forest_numba(self, M0_array, z0, z_max, M_res, dz=0.1):
        """
        Evolve N merger trees using the Numba JIT kernel with thread-level
        parallelism (``nb.prange``).

        The first call will trigger JIT compilation (~2–5 s).  Subsequent
        calls reuse the cached binary.  For large N (≥ 10 000) the Numba
        version is typically 8–32× faster than ``build_forest_numpy``.

        Parameters
        ----------
        Same as ``build_forest_numpy``.

        Returns
        -------
        mass_history : np.ndarray, shape (N, n_steps)
        z_steps : np.ndarray, shape (n_steps+1,)
        smooth_accretion : np.ndarray, shape (N, n_steps)
            Per-halo, per-step mass gained via unresolved smooth accretion
            -- see ``build_forest_numpy``'s docstring for the exact
            forward-time bookkeeping and conservation identity relative to
            ``mass_history`` (which this backend's identity satisfies
            exactly at every step, unlike ``build_forest_numpy``'s, which
            has one exception at the step a tree drops below ``M_res`` --
            see that method's docstring).
        merger_mass : np.ndarray, shape (N, n_steps)
            Per-halo, per-step mass gained via a resolved merger; 0 except
            at accepted-split steps.

        Notes
        -----
        The Numba kernel does not return ``split_events`` because allocating
        variable-length Python lists inside ``@njit`` is expensive -- use
        ``merger_mass`` above (its nonzero entries *are* the per-halo merger
        history) instead of post-processing ``mass_history`` by comparing
        consecutive columns.

        **Not reproducible via any seed**, unlike ``build_forest_numpy``
        (checked directly): numba's ``parallel=True``/``nb.prange``
        combination gives each worker thread its own internal random
        stream that is not deterministically tied to ``np.random.seed()``,
        whether called before this method or as the jitted kernel's own
        first statement. A real, disclosed limitation of numba's parallel
        RNG, not something this codebase controls -- use
        ``build_forest_numpy`` instead if reproducibility matters more
        than the extra speed here.

        Raises
        ------
        ImportError
            If numba is not installed -- it's an optional dependency
            (``pip install foraois[numba]``); use ``build_forest_numpy``
            instead (same API/return shape, no numba needed) if you don't
            need the extra speed.
        """
        if not _HAVE_NUMBA:
            raise ImportError(
                "build_forest_numba requires numba, which is not installed. "
                "Install it with `pip install foraois[numba]` (or `pip install "
                "numba` directly), or use build_forest_numpy instead -- same "
                "API and return shape, no numba needed."
            )
        self._ensure_delta_col_covers(z_max)

        M0_array = np.asarray(M0_array, dtype=np.float64)
        z_steps = np.arange(z0, z_max + dz * 0.5, dz)

        mass_history, smooth_accretion, merger_mass = _build_forest_kernel(
            M0_array=M0_array,
            z_steps=z_steps,
            dc_z_grid=self._dc_z_grid,
            dc_dc_grid=self._dc_dc_grid,
            logmass_grid=self.logmass_grid,
            sigma_grid=self.sigma_grid,
            alpha_grid=self.alpha_grid,
            j_u_grid=self._j_u_grid,
            j_values=self._j_values,
            G0=self.G0,
            gamma1=self.gamma1,
            gamma2=self.gamma2,
            M_res=float(M_res),
        )
        return mass_history, z_steps, smooth_accretion, merger_mass
