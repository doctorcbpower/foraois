"""
Zhang & Hui (2006)-based progenitor-mass sampling: a barrier-agnostic replacement for PCH08's
fitted branching rate, built on the Markov first-crossing distribution.
That distribution is exact for a Markov walk; with a non-sharp-k S(M), such as
the default real-space top-hat, it is the Markov approximation applied to that
S(M) (see `first_crossing.py`).

Where this module sits (general mathematics -> ZH step -> tree):

    first_crossing.py          general first-crossing mathematics: `solve_first_crossing`, analytic densities
                               (no cosmology, no trees)
    collapse/barrier.py        the barrier, `delta_c(M, z, cosmo_data)`
    this module:
      `first_crossing_step`    one (M0, z0) -> z1 step. Its first part is general excursion-set work (shifted
                               barrier, S-grid, `solve_first_crossing`, cumulative f); its second part is
                               Zhang-Hui / Nadler et al. (2023) specific (`F_zh`, `S_lower`, `p_split`). The
                               `N_grid` / `S_max_factor` resolution machinery and its coarse-grid warning belong
                               to this step, not to the solver.
      `_flat_barrier_cdf`,     closed-form CDF / quantile of the flat-barrier first-crossing law (the integral and
      `_flat_barrier_sample`   inverse of `first_crossing.flat_barrier_first_crossing`), used by the flat-barrier
                               NumPy sampling paths
      `draw_progenitor_mass_zh` progenitor sampling from a step; `ZhangHuiMergerTree` builds trees from it

`draw_progenitor_mass_zh` is the single-progenitor analog of
`pch_trees.PCHMergerTree.draw_progenitor_masses`: given a halo of mass
`M0` at `z0`, and a target redshift `z1`, it draws the progenitor(s) at
`z1` using the Markov first-crossing distribution (`first_crossing.py`'s
`solve_first_crossing`, evaluated against a barrier from
`foraois.collapse.delta_c`) instead of PCH08's fitted rate.

Origin-shift convention (standard excursion-set/conditional-mass-function
setup, e.g. Bond, Cole, Efstathiou & Kaiser 1991): the walk is tracked
relative to the parent halo's own trajectory point, `S0 = sigma(M0)^2`,
`delta0 = delta_c(M0, z0, cosmo_data)`. A progenitor of mass
`M(S) < M0` at `z1` corresponds to first crossing the shifted barrier
`B(S) = delta_c(M(S), z1, cosmo_data) - delta0` at variance
`S = sigma(M(S))^2 - S0` (S increases as progenitor mass decreases).
The barrier is the one configured on `cosmo_data` (`barrier` in the
config; see `foraois.collapse.barrier`), not chosen by the dark-matter
model. For the default `fixed` barrier, `B(S)` reduces to the constant
`delta_c(z1) - delta_c(z0)` -- exactly the flat-barrier case
`first_crossing.py`'s own `flat_barrier_first_crossing` validates
independently, which is this module's main internal consistency check
(see tests/test_zhang_hui_trees.py). For the `linear` barrier,
`B(S) = delta_c(z1) - delta_c(z0) + beta*S`, which
`first_crossing.linear_barrier_first_crossing` solves analytically
(see tests/test_zhang_hui_validation.py for the resolution it needs).

The unresolved-accretion mass fraction (PCH08's `F`) is not free from
`first_crossing.py`'s existing machinery -- it is computed here as its
own integral of `f(S)` over the sub-`M_res` tail (`S > S_res`), mirroring
`pch_trees._unresolved_accretion_fraction`'s role but evaluated against
the Markov first-crossing rate. The resolved-crossing probability and mass draw both
come from the same `f(S)` solve, so this needs exactly one
`solve_first_crossing` call per step, not two.

Mass-budget structure mirrors `PCHMergerTree.draw_progenitor_masses`
exactly (same return convention: a list of 0-2 progenitor masses
satisfying `M0 = sum(progenitors) + F_zh*M0`), so `ZhangHuiMergerTree.build_tree`
can drive this function the same way `pch_trees.py`'s `build_tree` drives
`draw_progenitor_masses` -- deliberately, so the two backends stay
directly comparable and don't need different tree-walking logic.

`ZhangHuiMergerTree.build_tree` drives this function one call per
z-step, serial, single halo. `ZhangHuiMergerTree.build_forest_numpy`
below uses a separate, closed-form vectorised path instead (see its own
docstring for why).
"""

import math
import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfc, erfcinv

from foraois.collapse import barrier_is_flat, delta_c, require_flat_barrier
from foraois.cosmo_utils import ensure_delta_col_covers
from foraois.first_crossing import solve_first_crossing

try:
    import numba as nb

    _HAVE_NUMBA = True
except ImportError:
    nb = None
    _HAVE_NUMBA = False


def _mass_at_S(S, sigma0_sq, cosmo_data):
    """
    Invert S = sigma(M)^2 - sigma0_sq back to M, via CosmoData's
    logmass_at_sigma (added alongside this module -- see cosmo_utils.py).
    S is clipped to >= 0 before the sqrt: forward-substitution solves of
    solve_first_crossing only ever evaluate B(S) at grid points >= 0, but
    floating-point round-off at S=0 can otherwise produce a tiny negative
    sigma^2 argument.
    """
    sigma_target = np.sqrt(np.maximum(S, 0.0) + sigma0_sq)
    logmass = cosmo_data.logmass_at_sigma(sigma_target)
    return 10.0**logmass


def _S_at_mass(M, sigma0_sq, cosmo_data):
    """
    Inverse of `_mass_at_S`: S = sigma(M)^2 - sigma0_sq. Used to bound the
    resolved-split sampling range at *both* ends (see
    `first_crossing_step`'s `S_lower`/`p_split`) -- the same formula
    `S_res` itself already uses, factored out so both call sites share it.
    """
    sigma_M = cosmo_data.sigma_at_logmass(np.log10(M))
    return sigma_M**2 - sigma0_sq


def _flat_barrier_cdf(S, b):
    """
    Exact CDF of the flat-barrier (constant B(S)=b) first-crossing
    distribution. Mathematically this belongs beside
    `first_crossing.flat_barrier_first_crossing`; it lives here because the
    ZH flat-barrier sampling paths are its only users. The Numba kernels do
    not call it (they inline their own equivalent, see `_erfcinv_numba`).
    It is the closed-form integral of
    first_crossing.py's flat_barrier_first_crossing (Zhang & Hui eq. 18's
    b=0, constant-barrier case), a Levy distribution with scale b^2:

        F(S) = erfc(b / sqrt(2*S))

    Checked directly against numerical integration of
    flat_barrier_first_crossing to machine precision. This is what makes
    build_forest_numpy (below) avoid solve_first_crossing's O(N_grid^2)
    solve entirely for a flat barrier (`fixed`, or `linear` with beta=0)
    -- see _assert_flat_barrier.
    """
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore"):
        return erfc(b / np.sqrt(2.0 * S))


def _flat_barrier_sample(u, b):
    """
    Inverse of _flat_barrier_cdf: given u = F(S) in [0, 1], returns S, via
    the standard Levy-distribution quantile function
    S = b^2 / (2 * erfcinv(u)^2). The compiled paths do not call this; they
    use `_erfcinv_numba` for the same quantile. u=0 (erfcinv(0)=inf) maps to S=0 rather
    than raising -- harmless here since build_forest_numpy only samples
    u=0 for entries it masks out afterward (see its own comments).
    """
    u = np.asarray(u, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return b**2 / (2.0 * erfcinv(u) ** 2)


def _assert_flat_barrier(model, z_ref, cosmo_data, algorithm="the closed-form flat-barrier sampler"):
    """
    The closed-form Levy sampling used by `build_forest_numpy`,
    `build_forest_numba`, `grow_full_population_numba` and
    `draw_progenitor_mass_zh_flat` applies only when delta_c does not
    depend on mass. The requested barrier is checked explicitly from the
    config (`require_flat_barrier`, which raises `NotImplementedError` for
    a scale-dependent barrier) and then, defensively, numerically: delta_c
    is compared at two very different masses, so a barrier that is flat
    by configuration but not in fact fails loudly instead of being
    silently mis-sampled. `model` is accepted for signature compatibility
    and does not select the barrier.
    """
    require_flat_barrier(cosmo_data, algorithm)
    M_lo, M_hi = 1e6, 1e16
    d_lo = float(delta_c(M_lo, z_ref, cosmo_data))
    d_hi = float(delta_c(M_hi, z_ref, cosmo_data))
    if not np.isclose(d_lo, d_hi, rtol=1e-8, atol=1e-12):
        raise NotImplementedError(
            f"{algorithm} requires a mass-independent (flat) collapse barrier, but delta_c is not flat at "
            f"z={z_ref} (delta_c({M_lo:.0e})={d_lo!r} != delta_c({M_hi:.0e})={d_hi!r})."
        )


def first_crossing_step(
    M0,
    z0,
    z1,
    M_res,
    cosmo_data,
    model="cdm",
    N_grid=400,
    S_max_factor=8.0,
):
    """
    Shared setup for one (M0,z0)->z1 step: solves `f(S)` via
    `solve_first_crossing` and derives the quantities
    `draw_progenitor_mass_zh` samples from. Split out from
    `draw_progenitor_mass_zh` (rather than kept as a closure inside it) so
    tests can check the resolved-crossing probability / unresolved
    fraction / CDM-flat-barrier-reduction directly, without needing to
    reverse-engineer them from many repeated random draws.

    Returns
    -------
    dict with keys:
        S_grid, f : the raw solve_first_crossing output
        F_cum     : cumulative trapezoidal integral of f over S_grid
        S_res     : variance at M_res (relative to M0's own S0)
        p_res     : P(resolved first crossing, i.e. S <= S_res)
        F_zh      : unresolved-accretion mass fraction (S > S_res tail),
                    clipped to [0, 1]. For a flat barrier this is the full
                    tail (the grid integral out to S_max plus the closed-form
                    tail beyond it); for a scale-dependent barrier it is the
                    tail integrated out to S_max only (see the comment at
                    its definition).
        S_lower   : variance at `M_avail - M_res` (M_avail = M0*(1-F_zh)),
                    the *upper* mass bound on a resolved split -- see
                    `p_split` below.
        p_split   : P(S in [S_lower, S_res]) -- the probability of a
                    resolved split whose *complement* is also guaranteed
                    resolved, mirroring N23's own [M_res, M-M_res]
                    sampling range for their progenitor mass M' (their
                    Eq. 24 and footnote 3: "every branching event adds
                    two progenitors... the other with mass M-M' to ensure
                    mass conservation" -- their M' is drawn from
                    [M_res, M-M_res], which by construction keeps *both*
                    M' and M-M' >= M_res). `draw_progenitor_mass_zh` draws
                    from this restricted range, not the full `[0, S_res]`
                    `p_res` covers, so both fragments of a resolved split
                    are guaranteed >= M_res. 0 when `M_avail <= 2*M_res`
                    (no mass-conserving resolved split is possible at
                    all, same guard PCH08's own `no_split_possible` uses
                    for `M2 < 2*M_res`).
        sigma0_sq : sigma(M0)^2, needed to convert a sampled S back to a
                    progenitor mass via _mass_at_S
    """
    if M_res >= M0:
        raise ValueError(f"M_res={M_res} must be < M0={M0}.")

    delta0 = float(delta_c(M0, z0, cosmo_data))
    sigma0_sq = float(cosmo_data.sigma_at_logmass(np.log10(M0))) ** 2
    sigma_res_sq = float(cosmo_data.sigma_at_logmass(np.log10(M_res))) ** 2
    S_res = sigma_res_sq - sigma0_sq
    if S_res <= 0:
        raise ValueError(
            f"M_res={M_res} does not give sigma(M_res) > sigma(M0={M0}) -- "
            "M_res must be strictly below M0 on the sigma(M) relation."
        )

    S_max = S_res * S_max_factor

    def B(S):
        # np.asarray (not atleast_1d) preserves S's dimensionality -- 0-d in
        # (scalar S from solve_first_crossing's internal scipy.integrate.quad
        # calls) stays 0-d, matching the barrier-function convention already
        # used by first_crossing.py's own tests (np.full_like/np.asarray),
        # which quad's C backend requires (it rejects genuine ndim>=1 arrays
        # from a scalar-input integrand).
        S = np.asarray(S, dtype=float)
        M_S = _mass_at_S(S, sigma0_sq, cosmo_data)
        # cdm/wdm's delta_c ignores M and returns a bare scalar regardless of
        # M_S's shape; multiplying by ones_like(S) broadcasts it to match S
        # (0-d or 1-d) the same way, while still working unchanged once a
        # genuinely mass-dependent delta_c (fdm/sidm) already returns
        # S-shaped output.
        dc = np.asarray(delta_c(M_S, z1, cosmo_data), dtype=float)
        return dc * np.ones_like(S) - delta0

    # Grid-resolution sanity check: S_max is sized off S_res (a
    # sigma(M)-derived scale), but the first-crossing distribution's width
    # is set by B(0)^2, a completely different scale. When sigma(M) >>
    # B(0) (e.g. an unrealistically-normalized synthetic power spectrum --
    # harmless for PCH08's scale-covariant rate math, but not for this
    # uniform-S-grid solver), the grid can have order one point inside the
    # region carrying nearly all the crossing probability, silently
    # producing badly wrong p_res/F_zh despite solve_first_crossing itself
    # returning numerically consistent values at the (too sparse) points
    # it was asked to evaluate. This is a coarse order-of-magnitude check,
    # not a guaranteed-accurate threshold (dS a factor of ~2 below
    # B(0)^2 can still carry ~5% error) -- and uses a fixed uniform grid
    # (like solve_first_crossing itself), so it's a warning, not an
    # automatic fix.
    B0_sq = float(B(0.0)) ** 2
    dS_grid = S_max / N_grid
    if dS_grid > B0_sq:
        warnings.warn(
            f"first_crossing_step's grid spacing (dS={dS_grid:.3g}) is coarser "
            f"than the shifted-barrier scale B(0)^2={B0_sq:.3g} -- the "
            "first-crossing distribution's peak likely falls between grid "
            "points, and p_res/F_zh from this step may be badly inaccurate. "
            "Increase N_grid, or check whether sigma(M) is unrealistically "
            "large relative to delta_c for this cosmology/power spectrum.",
            stacklevel=2,
        )

    S_grid, f = solve_first_crossing(B, S_max, N_grid)
    F_cum = cumulative_trapezoid(f, S_grid, initial=0.0)

    p_res = float(np.interp(S_res, S_grid, F_cum))
    if barrier_is_flat(cosmo_data):
        # A flat barrier has unit total crossing probability, so the unresolved fraction is the full tail beyond
        # S_res, as in the closed-form flat-barrier paths. The grid only reaches S_max, so F_cum[-1] - p_res alone
        # would hand the tail beyond S_max to the resolved progenitor; add that tail from the closed-form CDF. The
        # numerical difference is kept (rather than 1 - p_res) because p_res and F_cum[-1] share the same grid error
        # near S = 0, which cancels in the difference but not in p_res alone when dS exceeds B(0)^2.
        F_zh = float(np.clip(F_cum[-1] - p_res + (1.0 - _flat_barrier_cdf(S_max, float(B(0.0)))), 0.0, 1.0))
    else:
        # A scale-dependent barrier need not be crossed with unit probability, so adding a tail beyond S_max (or
        # using 1 - p_res) would also assign the non-crossing probability to the unresolved population, which is a
        # change of prescription, not a numerical correction. The tail is kept as the integral out to S_max.
        F_zh = float(np.clip(F_cum[-1] - p_res, 0.0, 1.0))

    # Restrict the resolved-split range so the complement is guaranteed
    # resolved too (see the docstring's p_split entry) -- mirrors N23's
    # own [M_res, M-M_res] sampling range for their progenitor mass.
    M_avail = M0 * (1.0 - F_zh)
    if M_avail > 2.0 * M_res:
        S_lower = float(_S_at_mass(M_avail - M_res, sigma0_sq, cosmo_data))
        S_lower = max(S_lower, 0.0)
        if S_lower < S_res:
            p_lower = float(np.interp(S_lower, S_grid, F_cum))
            p_split = max(p_res - p_lower, 0.0)
        else:
            S_lower = S_res
            p_split = 0.0
    else:
        S_lower = S_res
        p_split = 0.0

    return {
        "S_grid": S_grid,
        "f": f,
        "F_cum": F_cum,
        "S_res": S_res,
        "p_res": p_res,
        "F_zh": F_zh,
        "S_lower": S_lower,
        "p_split": p_split,
        "sigma0_sq": sigma0_sq,
    }


def draw_progenitor_mass_zh(
    M0,
    z0,
    z1,
    M_res,
    cosmo_data,
    model="cdm",
    rng=None,
    N_grid=400,
    S_max_factor=8.0,
):
    """
    Draw the progenitor(s) of a halo of mass `M0` at `z0`, at target
    redshift `z1`, using the Zhang & Hui (2006) first-crossing
    distribution for the collapse barrier configured on `cosmo_data`
    (`foraois.collapse.delta_c`). This is the general path: it solves the
    first-crossing equation numerically and supports a scale-dependent
    barrier. For such a barrier the solver's grid must resolve the peak
    of f(S), whose width is set by B(0)^2 = (delta_c(z1) - delta_c(z0))^2
    -- see docs/MODELS.md ("Resolution of the general first-crossing
    path") and tests/test_zhang_hui_validation.py.

    Parameters
    ----------
    M0 : float
        Parent halo mass (Msun/h) at z0.
    z0, z1 : float
        Starting and target redshift (z1 > z0).
    M_res : float
        Mass resolution limit (Msun/h).
    cosmo_data : CosmoData
        Owns the sigma(M) table, built on first use or explicitly with `cosmo_data.prepare_sigma_grid(...)`.
    model : str
        Retained for backwards compatibility; it does not select the
        collapse barrier (set by the `barrier` config value, see
        `foraois.collapse.barrier`) or the dark-matter model (`dm_model`).
    rng : np.random.Generator, optional
    N_grid : int
        Grid resolution for `solve_first_crossing` (see its own
        docstring for the O(N^2) cost this trades off against accuracy).
    S_max_factor : float
        The `f(S)` grid is solved out to `S_max_factor * S_res` (S_res
        being the variance at M_res) so the unresolved-accretion tail
        integral (F_zh, below) has somewhere to integrate over beyond
        the resolved range -- too small underestimates F_zh by truncating
        real tail probability.

    Returns
    -------
    list of float
        0, 1, or 2 progenitor masses (Msun/h), each >= M_res, satisfying
        `M0 = sum(progenitors) + F_zh*M0` where F_zh is the unresolved-
        accretion fraction computed internally (same 0/1/2-progenitor
        count convention as `pch_trees.draw_progenitor_masses`). When two
        are returned the order is [continuing, drawn]: they are the
        complementary fragments of the resolved mass, the drawn mass can
        exceed the continuing mass, and callers should not infer an ordering
        by mass from the position in the list. Both progenitors are guaranteed
        >= M_res whenever two are returned -- the resolved-split draw is
        restricted to `first_crossing_step`'s `p_split`/`S_lower` range,
        which bounds the complement the same way the drawn progenitor
        itself is bounded, mirroring N23's own `[M_res, M-M_res]`
        sampling range (see `first_crossing_step`'s docstring for the
        literature citation this rests on).
    """
    rng = rng if rng is not None else np.random.default_rng()

    if M_res >= M0:
        return []

    step = first_crossing_step(M0, z0, z1, M_res, cosmo_data, model, N_grid, S_max_factor)
    p_split, F_zh = step["p_split"], step["F_zh"]

    M_continuing = M0 * (1.0 - F_zh)

    if rng.random() >= p_split:
        return [M_continuing] if M_continuing >= M_res else []

    v = rng.uniform(step["p_res"] - p_split, step["p_res"])
    S_star = float(np.interp(v, step["F_cum"], step["S_grid"]))
    # M2 is the drawn progenitor mass; it can exceed half of M_avail, and the other fragment is the complementary
    # mass M_avail - M2 (no ordering of the two is assumed).
    M2 = float(_mass_at_S(np.array([S_star]), step["sigma0_sq"], cosmo_data)[0])

    M_continuing -= M2

    progenitors = []
    if M_continuing >= M_res:
        progenitors.append(M_continuing)
    if M2 >= M_res:
        progenitors.append(M2)
    return progenitors


def draw_progenitor_mass_zh_flat(M0, z0, z1, M_res, cosmo_data, model="cdm", rng=None):
    """
    Closed-form flat-barrier equivalent of `draw_progenitor_mass_zh`, for
    a flat barrier only (`fixed`, or `linear` with beta=0; a scale-dependent
    barrier raises `NotImplementedError`). Same algorithm as
    `_build_forest_flat_barrier_numpy`'s per-halo math (Levy-distribution
    `_flat_barrier_cdf`/`_flat_barrier_sample`, and the same N23-footnote-3
    `p_split`/`S_lower` resolved-split restriction as
    `first_crossing_step`'s general path -- see that function's docstring
    for the citation), just as a single-halo scalar call instead of the
    vectorised forest builder.

    This exists because `draw_progenitor_mass_zh` (via `first_crossing_step`
    -> `solve_first_crossing`) solves the general Zhang & Hui Volterra
    integral equation numerically -- an O(N_grid) loop of
    `scipy.integrate.quad` calls (`_g2_near_diagonal`), each itself
    adaptive -- even though the flat-barrier case has an exact closed form
    requiring none of that. That numerical solve is what makes
    `grow_full_population_zh` (`_treegrowth.py`) so much more expensive
    than PCH08 per branch per step; this function lets a full-population
    grower use the same closed-form shortcut `build_forest_numpy`/
    `build_forest_numba` already use for main-progenitor-only trees,
    without the caller having to duplicate that vectorised code's
    per-halo math by hand. Does NOT check `_assert_flat_barrier` itself
    (a per-call check would defeat the point of avoiding per-step
    overhead) -- callers must check once up front, as
    `grow_full_population_zh_flat` does.

    Returns
    -------
    list of float
        Same 0/1/2-progenitor convention as `draw_progenitor_mass_zh`.
    """
    require_flat_barrier(cosmo_data, "draw_progenitor_mass_zh_flat")
    rng = rng if rng is not None else np.random.default_rng()

    if M_res >= M0:
        return []

    sigma0_sq = float(cosmo_data.sigma_at_logmass(np.log10(M0))) ** 2
    sigma_res_sq = float(cosmo_data.sigma_at_logmass(np.log10(M_res))) ** 2
    S_res = sigma_res_sq - sigma0_sq
    if S_res <= 0:
        raise ValueError(
            f"M_res={M_res} does not give sigma(M_res) > sigma(M0={M0}) -- "
            "M_res must be strictly below M0 on the sigma(M) relation."
        )

    # M_res is an arbitrary evaluation mass for a flat barrier -- delta_c
    # doesn't depend on it (that's exactly what _assert_flat_barrier
    # verifies, once, up front) -- same choice _build_forest_flat_barrier_numpy
    # makes.
    d0 = float(delta_c(M_res, z0, cosmo_data))
    d1 = float(delta_c(M_res, z1, cosmo_data))
    d_omega = d1 - d0

    p_res = float(_flat_barrier_cdf(S_res, d_omega))
    M_continuing = M0 * p_res

    can_split = M_continuing > 2.0 * M_res
    if can_split:
        S_lower = float(np.clip(_S_at_mass(M_continuing - M_res, sigma0_sq, cosmo_data), 0.0, S_res))
        p_lower = float(_flat_barrier_cdf(S_lower, d_omega)) if S_lower > 0.0 else 1.0
        p_split = max(p_res - p_lower, 0.0)
    else:
        p_lower = p_res
        p_split = 0.0

    if rng.random() >= p_split:
        return [M_continuing] if M_continuing >= M_res else []

    v = rng.uniform(p_lower, p_res)
    S_star = float(_flat_barrier_sample(v, d_omega))
    # M2 is the drawn progenitor mass (it can exceed M_continuing / 2); M1 is the complementary mass.
    M2 = float(_mass_at_S(np.array([S_star]), sigma0_sq, cosmo_data)[0])
    M1 = M_continuing - M2

    progenitors = []
    if M1 >= M_res:
        progenitors.append(M1)
    if M2 >= M_res:
        progenitors.append(M2)
    return progenitors


def _build_forest_flat_barrier_numpy(M0_array, z_steps, model, cosmo_data, M_res, rng):
    """
    Vectorised forest builder, exploiting that cdm/wdm/fdm's collapse
    barrier is mass-independent (checked once via _assert_flat_barrier)
    so the first-crossing distribution has the closed-form Levy solution
    (_flat_barrier_cdf/_flat_barrier_sample) instead of needing
    solve_first_crossing's O(N_grid^2) numerical solve per halo per step.
    This is exact (not truncated at some S_max), which is why this is a
    separate closed-form path instead of just vectorising
    first_crossing_step/draw_progenitor_mass_zh (which stay the general,
    non-flat-barrier-capable implementation driving build_tree). Not
    applicable once a genuine mass-dependent barrier exists (real FDM
    physics) -- that would need the general (and much more expensive)
    per-halo solve_first_crossing path this avoids.

    Mass-budget bookkeeping (the "M_continuing = M_safe*p_res, then
    subtract the drawn progenitor" logic, and which candidate becomes the
    tracked main-progenitor mass) mirrors draw_progenitor_mass_zh/
    build_tree's own single-halo convention. Unlike PCH08's binary split
    (where M1+M2 = M0*(1-F) exactly by construction of q's sampling
    range, bounded to keep *both* fragments resolved), a drawn progenitor
    mass M2 has no matching lower bound on its complement
    M1 = M_continuing - M2 unless the sampling range is restricted to
    match -- following N23's own footnote 3 (their progenitor mass M' is
    drawn from [M_res, M-M_res], which by construction keeps *both* M'
    and M-M' >= M_res), the resolved-split draw here is restricted to
    the matching range (`p_split`/`S_lower`, mirroring
    `first_crossing_step`'s own range; see its docstring for the
    citation), so M1 is guaranteed >= M_res whenever a split is drawn.
    smooth_accretion (below) is defined as the conservation *residual*
    (M_safe - M_next - merger_mass), which equals exactly
    (1-p_res)*M_safe given this sampling-range restriction -- kept as the
    residual formula anyway as a cheap built-in consistency check.

    Return convention/shapes mirror pch_trees._build_forest_numpy exactly
    (see that function's own docstring for the conservation identity and
    forward-time bookkeeping this follows) -- with one exception:
    Zhang & Hui's rate is exact, not a fitted rate needing PCH08's
    rejection-sampling correction step, so there is no "accepted vs.
    rejected" distinction here -- a resolved crossing this step is
    accepted unconditionally once drawn.
    """
    N = len(M0_array)
    n_steps = len(z_steps) - 1

    M = M0_array.copy().astype(np.float64)
    alive = M >= M_res
    mass_history = np.zeros((N, n_steps), dtype=np.float64)
    smooth_accretion = np.zeros((N, n_steps), dtype=np.float64)
    merger_mass = np.zeros((N, n_steps), dtype=np.float64)
    split_events = []

    sigma_res_sq = float(cosmo_data.sigma_at_logmass(np.log10(M_res))) ** 2

    for j in range(n_steps):
        z0, z1 = z_steps[j], z_steps[j + 1]
        # M_res is an arbitrary evaluation mass -- delta_c doesn't depend
        # on it for a flat barrier (that's exactly what _assert_flat_barrier
        # verified once, outside this loop), it's just already at hand.
        d0 = float(delta_c(M_res, z0, cosmo_data))
        d1 = float(delta_c(M_res, z1, cosmo_data))
        d_omega = d1 - d0

        # Dummy positive value for dead trees (masked out below via `alive`),
        # same pattern as pch_trees._build_forest_numpy's M_safe: must not
        # equal M_res exactly, or S_res=0 makes p_res ill-defined (0/0 in
        # the erfc argument) rather than just harmlessly small.
        M_safe = np.where(alive, M, 2.0 * M_res)
        sigma0_sq = cosmo_data.sigma_at_logmass(np.log10(M_safe)) ** 2
        S_res = sigma_res_sq - sigma0_sq

        p_res = _flat_barrier_cdf(S_res, d_omega)
        M_continuing = M_safe * p_res  # F_zh = 1-p_res exactly (closed form, no S_max truncation)

        # Restrict the resolved-split range so the complement is
        # guaranteed resolved too, mirroring N23's own [M_res, M-M_res]
        # sampling range for their progenitor mass (see
        # first_crossing_step's docstring for the same fix and its
        # literature citation) -- this is what makes the "M1 can land
        # below M_res" failure mode this function's docstring describes
        # no longer arise from the sampling itself.
        can_split = M_continuing > 2.0 * M_res
        M_avail_safe = np.where(can_split, M_continuing, 3.0 * M_res)  # dummy, avoids a negative S_at_mass argument
        S_lower = np.clip(_S_at_mass(M_avail_safe - M_res, sigma0_sq, cosmo_data), 0.0, S_res)
        p_lower = _flat_barrier_cdf(S_lower, d_omega)
        p_split = np.where(can_split, np.clip(p_res - p_lower, 0.0, None), 0.0)

        u1 = rng.random(N)
        split_mask = alive & (u1 < p_split)

        v = rng.uniform(p_lower, p_res, size=N)
        S_star = _flat_barrier_sample(v, d_omega)
        M2 = _mass_at_S(S_star, sigma0_sq, cosmo_data)
        M1 = M_continuing - M2  # guaranteed >= M_res whenever split_mask, up to floating point

        ok1 = M1 >= M_res
        ok2 = M2 >= M_res
        M_after_split = np.where(
            ok1 & ok2,
            np.maximum(M1, M2),
            np.where(ok1, M1, np.where(ok2, M2, 0.0)),
        )

        M_next = np.where(split_mask, M_after_split, M_continuing)
        M = np.where(alive, M_next, M)

        alive = alive & (M >= M_res)
        mass_history[:, j] = np.where(alive, M, 0.0)

        # smooth_accretion is defined as the conservation residual
        # (M_safe - M_next - merger_mass), not simply (1-p_res)*M_safe --
        # the two agree exactly given the p_split/S_lower sampling-range
        # restriction (both fragments guaranteed resolved, per N23's own
        # footnote 3 -- see this function's own docstring). Kept as the
        # residual formula anyway, as a cheap built-in consistency check.
        merger_mass[:, j] = np.where(alive & split_mask & ok1 & ok2, np.minimum(M1, M2), 0.0)
        smooth_accretion[:, j] = np.where(alive, M_safe - M_next - merger_mass[:, j], 0.0)

        accepted = split_mask & ok1 & ok2
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
# Numba JIT kernel -- parallel counterpart to _build_forest_flat_barrier_numpy
# above, same closed-form Levy sampling, same restriction to flat barriers.
# ---------------------------------------------------------------------------
# scipy.special.erfc/erfcinv (used by the numpy path) aren't callable from
# inside @njit code; math.erf/math.erfc are numba-supported, but there is no
# numba-compatible erfcinv anywhere -- _erfcinv_numba below implements one
# directly (Winitzki 2008's rational approximation as a Newton-Raphson
# starting point, refined against math.erf to ~1e-6 relative accuracy for
# any v not astronomically close to 0 or 1 -- checked directly against
# scipy.special.erfcinv over v in [1e-300, 1-1e-15], see
# tests/test_zhang_hui_trees.py). ln(1-x^2) is computed as
# ln(1-x) + ln(1+x) rather than ln(1-x*x) directly, to avoid the
# catastrophic-cancellation underflow the direct form suffers as x -> +-1
# (i.e. v -> 0 or 2).
#
# PCH08's own numba kernel (pch_trees._build_forest_kernel) inlines its own
# copy of a sorted-array binary-search interpolator for the same reason
# _interp_sorted_zh below duplicates it here rather than importing from
# pch_trees.py: these are two independent algorithm implementations by
# design (see tree_algorithm.py's module docstring), and this is a tiny,
# purely mechanical utility, not shared domain logic.
# ---------------------------------------------------------------------------

if _HAVE_NUMBA:

    @nb.njit(cache=True)
    def _interp_sorted_zh(x, xp, fp):
        """Scalar linear interpolation, xp ascending, clamped outside its range. Same as pch_trees._interp_sorted."""
        n = xp.shape[0]
        if x <= xp[0]:
            return fp[0]
        if x >= xp[n - 1]:
            return fp[n - 1]
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

    @nb.njit(cache=True)
    def _erfcinv_numba(v):
        """
        Numba-compatible erfcinv(v), v in [0, 2]. See module comment above
        for the method and its accuracy. v<=0 -> +inf, v>=2 -> -inf,
        matching scipy.special.erfcinv's own limiting behaviour (needed
        here since a fully-underflowed p_lower=0.0 is a real, expected
        input, not an error -- see _flat_barrier_sample's docstring).
        """
        if v <= 0.0:
            return np.inf
        if v >= 2.0:
            return -np.inf
        x = 1.0 - v
        ln1mx2 = np.log(v) + np.log(2.0 - v)
        a = 0.147
        term = 2.0 / (np.pi * a) + ln1mx2 / 2.0
        inner = term * term - ln1mx2 / a
        if inner < 0.0:
            inner = 0.0
        y = np.sqrt(np.sqrt(inner) - term)
        if x < 0.0:
            y = -y
        two_over_sqrtpi = 1.1283791670955126
        for _ in range(6):
            deriv = two_over_sqrtpi * np.exp(-y * y)
            if deriv == 0.0:
                break
            y = y - (math.erf(y) - x) / deriv
        return y

    @nb.njit(parallel=True, cache=True, fastmath=True)
    def _build_forest_flat_barrier_kernel(
        M0_array,  # (N,)   initial halo masses
        z_steps,  # (S,)   redshift grid edges
        dc_z_grid,  # (Nz,) precomputed redshift grid for delta_col table
        dc_dc_grid,  # (Nz,) delta_col values on that grid
        logmass_grid,  # (Nm,) log10(M) grid for sigma table, ascending
        sigma_grid,  # (Nm,) sigma(M) values on that grid
        sigma_grid_rev,  # (Nm,) sigma_grid reversed -- ascending in sigma
        logmass_grid_rev,  # (Nm,) logmass_grid reversed, paired with sigma_grid_rev
        M_res,  # scalar resolution mass
    ):
        """
        Core Numba kernel for the flat-barrier closed-form sampler -- see
        _build_forest_flat_barrier_numpy's own docstring for the algorithm
        (mass-budget bookkeeping, the N23-footnote-3 resolved-split
        sampling range, and the smooth_accretion/merger_mass conservation
        identity), which this reimplements per-halo/per-step instead of
        vectorised across halos. Uses the module-level erf/erfcinv-based
        Levy sampling exactly as that function does, not a different
        derivation.

        M_res is used both as the resolution floor and (like the numpy
        path) as the arbitrary mass point delta_c is evaluated at for a
        flat barrier (delta_c doesn't depend on it -- see
        _assert_flat_barrier, checked once by the caller before this runs).

        Each tree index is processed by a separate thread (nb.prange).
        """
        N = M0_array.shape[0]
        S = z_steps.shape[0]
        n_steps = S - 1

        mass_history = np.zeros((N, n_steps), dtype=np.float64)
        smooth_accretion = np.zeros((N, n_steps), dtype=np.float64)
        merger_mass = np.zeros((N, n_steps), dtype=np.float64)

        log10_M_res = np.log10(M_res)
        sigma_res = _interp_sorted_zh(log10_M_res, logmass_grid, sigma_grid)
        sigma_res_sq = sigma_res * sigma_res

        for i in nb.prange(N):  # <-- parallel over trees
            m = M0_array[i]
            alive = m >= M_res

            for j in range(n_steps):
                if not alive:
                    mass_history[i, j] = 0.0
                    continue

                z0 = z_steps[j]
                z1 = z_steps[j + 1]
                d0 = _interp_sorted_zh(z0, dc_z_grid, dc_dc_grid)
                d1 = _interp_sorted_zh(z1, dc_z_grid, dc_dc_grid)
                d_omega = d1 - d0

                sigma0 = _interp_sorted_zh(np.log10(m), logmass_grid, sigma_grid)
                sigma0_sq = sigma0 * sigma0
                S_res = sigma_res_sq - sigma0_sq

                p_res = math.erfc(d_omega / np.sqrt(2.0 * S_res))
                m_continuing = m * p_res

                can_split = m_continuing > 2.0 * M_res
                if can_split:
                    sigma_avail = _interp_sorted_zh(np.log10(m_continuing - M_res), logmass_grid, sigma_grid)
                    S_lower = sigma_avail * sigma_avail - sigma0_sq
                    if S_lower < 0.0:
                        S_lower = 0.0
                    elif S_lower > S_res:
                        S_lower = S_res
                    p_lower = math.erfc(d_omega / np.sqrt(2.0 * S_lower)) if S_lower > 0.0 else 1.0
                    p_split = p_res - p_lower
                    if p_split < 0.0:
                        p_split = 0.0
                else:
                    p_lower = p_res
                    p_split = 0.0

                u1 = np.random.random()
                do_split = can_split and (u1 < p_split)

                m1 = 0.0
                m2 = 0.0
                ok1 = False
                ok2 = False
                if do_split:
                    v = p_lower + np.random.random() * (p_res - p_lower)
                    S_star = (d_omega * d_omega) / (2.0 * _erfcinv_numba(v) ** 2)
                    sigma_target = np.sqrt(max(S_star, 0.0) + sigma0_sq)
                    logmass2 = _interp_sorted_zh(sigma_target, sigma_grid_rev, logmass_grid_rev)
                    m2 = 10.0**logmass2
                    m1 = m_continuing - m2
                    ok1 = m1 >= M_res
                    ok2 = m2 >= M_res

                if not do_split:
                    m_next = m_continuing
                elif ok1 and ok2:
                    merger_mass[i, j] = min(m1, m2)
                    m_next = max(m1, m2)
                elif ok1:
                    m_next = m1
                elif ok2:
                    m_next = m2
                else:
                    # Both fragments landed below M_res -- the p_split/
                    # S_lower restriction is specifically designed to
                    # prevent this (see _build_forest_flat_barrier_numpy's
                    # docstring), so this should only ever be hit at a
                    # floating-point boundary; matches that function's own
                    # nested np.where, which also gives 0.0 here (a split
                    # was drawn, but neither fragment survived it).
                    m_next = 0.0

                smooth_accretion[i, j] = m - m_next - merger_mass[i, j]
                m = m_next
                alive = m >= M_res
                mass_history[i, j] = m if alive else 0.0
                if not alive:
                    smooth_accretion[i, j] = 0.0
                    merger_mass[i, j] = 0.0

        return mass_history, smooth_accretion, merger_mass

    # -----------------------------------------------------------------------
    # Numba full-population kernel -- extends _build_forest_flat_barrier_
    # kernel's per-halo/per-step closed-form Levy sampling (above) from
    # "track only the main progenitor" to "track every branch", the same
    # way pch_trees._grow_tree_pch08_adaptive_scalar/
    # _grow_forest_pch08_adaptive_kernel extend PCH08's own build_forest_
    # numba. Unlike PCH08, no adaptive step-size search is needed here:
    # Zhang & Hui's rate is an exact, already-bounded probability
    # (p_split <= 1 by construction, not PCH08's Nupper upper-bound
    # approximation that needs to be kept small), so every branch simply
    # walks the same shared, fixed-dz z_steps grid the caller already
    # uses for _build_forest_flat_barrier_kernel -- no per-branch adaptive
    # stepping, no _pick_adaptive_step-style search.
    #
    # This is the numba counterpart of scripts/paper_figs/_treegrowth.py's
    # grow_full_population_zh_flat (itself the closed-form counterpart of
    # grow_full_population_zh -- see draw_progenitor_mass_zh_flat's own
    # docstring for why the closed form exists at all): same per-branch
    # math, same {checkpoint: [masses]} output convention, just compiled
    # and parallelized across trees instead of a serial Python loop.
    # -----------------------------------------------------------------------

    @nb.njit(cache=True)
    def _grow_tree_zh_flat_scalar(
        M0, M_res, z_steps, checkpoint_step_idx,
        dc_z_grid, dc_dc_grid, logmass_grid, sigma_grid, sigma_grid_rev, logmass_grid_rev,
        out_masses, out_counts, max_stack,
    ):
        """Single-tree full-population growth on the fixed z_steps grid,
        using the same closed-form flat-barrier Levy sampling as
        _build_forest_flat_barrier_kernel's per-halo body, extended to
        every branch via an explicit stack (mirrors
        pch_trees._grow_tree_pch08_adaptive_scalar's stack pattern).
        checkpoint_step_idx holds each checkpoint's index into z_steps
        (computed by the Python wrapper, mirroring _treegrowth.py's
        _checkpoint_indices); out_masses/out_counts are this tree's
        preallocated (n_cp, max_out) / (n_cp,) output slices."""
        n_cp = checkpoint_step_idx.shape[0]
        max_out = out_masses.shape[1]
        n_steps = z_steps.shape[0] - 1

        mass_stack = np.empty(max_stack, dtype=np.float64)
        step_idx_stack = np.empty(max_stack, dtype=np.int64)

        log10_M_res = np.log10(M_res)
        sigma_res = _interp_sorted_zh(log10_M_res, logmass_grid, sigma_grid)
        sigma_res_sq = sigma_res * sigma_res

        stack_ptr = 1
        mass_stack[0] = M0
        step_idx_stack[0] = 0

        while stack_ptr > 0:
            stack_ptr -= 1
            m = mass_stack[stack_ptr]
            j = step_idx_stack[stack_ptr]

            while j < n_steps and m >= M_res:
                z0 = z_steps[j]
                z1 = z_steps[j + 1]
                d0 = _interp_sorted_zh(z0, dc_z_grid, dc_dc_grid)
                d1 = _interp_sorted_zh(z1, dc_z_grid, dc_dc_grid)
                d_omega = d1 - d0

                sigma0 = _interp_sorted_zh(np.log10(m), logmass_grid, sigma_grid)
                sigma0_sq = sigma0 * sigma0
                S_res = sigma_res_sq - sigma0_sq

                p_res = math.erfc(d_omega / np.sqrt(2.0 * S_res))
                m_continuing = m * p_res

                can_split = m_continuing > 2.0 * M_res
                if can_split:
                    sigma_avail = _interp_sorted_zh(np.log10(m_continuing - M_res), logmass_grid, sigma_grid)
                    S_lower = sigma_avail * sigma_avail - sigma0_sq
                    if S_lower < 0.0:
                        S_lower = 0.0
                    elif S_lower > S_res:
                        S_lower = S_res
                    p_lower = math.erfc(d_omega / np.sqrt(2.0 * S_lower)) if S_lower > 0.0 else 1.0
                    p_split = p_res - p_lower
                    if p_split < 0.0:
                        p_split = 0.0
                else:
                    p_lower = p_res
                    p_split = 0.0

                u1 = np.random.random()
                do_split = can_split and (u1 < p_split)

                m1 = 0.0
                m2 = 0.0
                ok1 = False
                ok2 = False
                if do_split:
                    v = p_lower + np.random.random() * (p_res - p_lower)
                    S_star = (d_omega * d_omega) / (2.0 * _erfcinv_numba(v) ** 2)
                    sigma_target = np.sqrt(max(S_star, 0.0) + sigma0_sq)
                    logmass2 = _interp_sorted_zh(sigma_target, sigma_grid_rev, logmass_grid_rev)
                    m2 = 10.0**logmass2
                    m1 = m_continuing - m2
                    ok1 = m1 >= M_res
                    ok2 = m2 >= M_res

                if not do_split:
                    m_next = m_continuing
                elif ok1 and ok2:
                    m_next = m1
                elif ok1:
                    m_next = m1
                elif ok2:
                    m_next = m2
                else:
                    # Both fragments landed below M_res -- see
                    # _build_forest_flat_barrier_kernel's own comment on
                    # this same branch: the p_split/S_lower restriction is
                    # designed to prevent this, so it's only a
                    # floating-point-boundary case.
                    m_next = 0.0

                # The non-continuing fragment becomes a new branch,
                # pushed to resume its own walk from step j+1.
                if do_split and ok1 and ok2 and stack_ptr < max_stack:
                    mass_stack[stack_ptr] = m2
                    step_idx_stack[stack_ptr] = j + 1
                    stack_ptr += 1
                    for k in range(n_cp):
                        if checkpoint_step_idx[k] == j + 1:
                            c = out_counts[k]
                            if c < max_out:
                                out_masses[k, c] = m2
                                out_counts[k] = c + 1
                            break

                m = m_next
                j += 1

                if m >= M_res:
                    for k in range(n_cp):
                        if checkpoint_step_idx[k] == j:
                            c = out_counts[k]
                            if c < max_out:
                                out_masses[k, c] = m
                                out_counts[k] = c + 1
                            break

    @nb.njit(parallel=True, cache=True)
    def _grow_forest_zh_flat_kernel(
        M0_array, M_res, z_steps, checkpoint_step_idx,
        dc_z_grid, dc_dc_grid, logmass_grid, sigma_grid, sigma_grid_rev, logmass_grid_rev,
        max_stack, max_out,
    ):
        """nb.prange-parallel wrapper: one independent full-population
        growth per tree index, each writing into its own disjoint slice of
        the preallocated output arrays -- mirrors
        pch_trees._grow_forest_pch08_adaptive_kernel's own prange-over-trees
        pattern."""
        N = M0_array.shape[0]
        n_cp = checkpoint_step_idx.shape[0]
        out_masses = np.zeros((N, n_cp, max_out), dtype=np.float64)
        out_counts = np.zeros((N, n_cp), dtype=np.int64)

        for i in nb.prange(N):
            _grow_tree_zh_flat_scalar(
                M0_array[i], M_res, z_steps, checkpoint_step_idx,
                dc_z_grid, dc_dc_grid, logmass_grid, sigma_grid, sigma_grid_rev, logmass_grid_rev,
                out_masses[i], out_counts[i], max_stack,
            )

        return out_masses, out_counts

else:
    _interp_sorted_zh = None
    _erfcinv_numba = None
    _build_forest_flat_barrier_kernel = None
    _grow_tree_zh_flat_scalar = None
    _grow_forest_zh_flat_kernel = None


class ZhangHuiMergerTree:
    """
    The barrier-agnostic, exact-rate counterpart to
    `pch_trees.PCHMergerTree`, built on `draw_progenitor_mass_zh`.

    `build_tree` (serial, single-halo) deliberately mirrors
    `PCHMergerTree.build_tree`'s own structure/signature (same z-stepping
    loop, same tree-of-dicts return shape) so downstream code
    (`utils/plot.py`) needs no special-casing to accept either backend's
    tree; use it for a single illustrative tree, or where a scale-dependent
    barrier is configured (`barrier: linear`; see `foraois.collapse`) --
    the only ZH path that supports one.

    `build_forest_numpy`/`build_forest_numba` are the vectorised/parallel
    forest-scale backends -- but only for a flat barrier (`fixed`, or
    `linear` with beta=0; other barriers raise `NotImplementedError`; see
    `_build_forest_flat_barrier_numpy`'s docstring for why this is a
    closed-form path, not a vectorised `first_crossing_step`), matching
    `PCHMergerTree.build_forest_numpy`/`build_forest_numba`'s own
    signatures and return shapes. Requires the optional `numba` dependency
    (`pip install foraois[numba]`) -- raises `ImportError` with that
    instruction if it isn't installed.
    """

    def __init__(
        self,
        cosmo_data,
        params=None,
        model="cdm",
        rng=None,
        N_grid=400,
        S_max_factor=8.0,
    ):
        """
        Parameters
        ----------
        cosmo_data : foraois.cosmo_utils.CosmoData
            Builds the sigma(M) grid as a side effect of construction (see
            PCHMergerTree.__init__'s docstring for the same note).
        params : dict, optional
            Currently unused; accepted for signature parity with
            PCHMergerTree.
        model : str
            Retained for backwards compatibility. It no longer selects the
            collapse barrier, which is set by the `barrier` config value
            (see foraois.collapse.barrier); the dark-matter model is set by
            `dm_model` and acts on P(k) inside CosmoData.
        rng : np.random.Generator, optional
            Default: a fresh np.random.default_rng() (unseeded).
        N_grid : int
            solve_first_crossing's grid resolution -- see
            first_crossing_step's own grid-resolution warning for how to
            choose this per problem.
        S_max_factor : float
            first_crossing_step's S_max = S_max_factor * S_res, sizing the
            solver's grid off the resolution mass.
        """
        self.cosmo_data = cosmo_data
        self.params = params
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()
        self.N_grid = N_grid
        self.S_max_factor = S_max_factor

        # sigma(M) is owned by CosmoData (built on first use, or by prepare_sigma_grid); the tree only reads it.
        self.pk_data = cosmo_data.get_power_spectrum()

    def _ensure_delta_col_covers(self, z_max):
        """
        Same guard as `PCHMergerTree._ensure_delta_col_covers` (see
        cosmo_utils.ensure_delta_col_covers's docstring for the
        silent-clamping failure mode this avoids) -- applies equally here,
        since `draw_progenitor_mass_zh` also reaches
        `cosmo_data.delta_col_at_z` indirectly, through
        `collapse.delta_c`. No cached grid copy to
        refresh afterward, unlike PCHMergerTree's version -- this class
        reads `cosmo_data.delta_col_at_z()` live instead.
        """
        ensure_delta_col_covers(self.cosmo_data, z_max)

    def build_tree(self, M0, z0, z_max, M_res, dz=0.1):
        """
        Recursively step a single tree from z0 to z_max using the exact
        Zhang & Hui rate (via `draw_progenitor_mass_zh`) instead of
        PCH08's fitted one. Serial only -- see class docstring.

        Each tree entry carries `smooth_accretion`/`merger_mass`, for
        parity with `build_forest_numpy`'s own channels. Computed here
        purely from `parent_mass` and `progenitors`
        (`smooth_accretion = parent_mass - max(progenitors) - merger_mass`,
        the same conservation-*residual* definition `build_forest_numpy`
        uses, not a separately derived quantity) -- so
        `draw_progenitor_mass_zh`'s own return contract doesn't need to
        change for this.
        """
        self._ensure_delta_col_covers(z_max)
        self.cosmo_data.check_M_res(M_res, compiled=False)

        tree = []
        M_cur = M0

        z_steps = np.arange(z0, z_max + dz * 0.5, dz)
        max_steps = len(z_steps)

        for k in range(max_steps - 1):
            if M_cur < M_res:
                break

            z_cur = z_steps[k]
            z_next = z_steps[k + 1]

            progenitors = draw_progenitor_mass_zh(
                M_cur,
                z_cur,
                z_next,
                M_res,
                self.cosmo_data,
                model=self.model,
                rng=self.rng,
                N_grid=self.N_grid,
                S_max_factor=self.S_max_factor,
            )

            if progenitors:
                M_next = max(progenitors)
                merger_mass = min(progenitors) if len(progenitors) == 2 else 0.0
                smooth_accretion = M_cur - M_next - merger_mass
                tree.append(
                    {
                        "redshift": z_next,
                        "parent_mass": M_cur,
                        "progenitors": progenitors,
                        "smooth_accretion": smooth_accretion,
                        "merger_mass": merger_mass,
                    }
                )
                M_cur = M_next

        return tree

    def build_forest_numpy(self, M0_array, z0, z_max, M_res, dz=0.1):
        """
        Evolve N merger trees simultaneously using the closed-form
        flat-barrier sampler (`_build_forest_flat_barrier_numpy` -- see
        its docstring for why this is a closed-form path rather than a
        vectorised `first_crossing_step`, and its mass-budget caveat).

        Only supports a mass-independent barrier (`fixed`, or `linear` with
        beta=0): a scale-dependent barrier raises `NotImplementedError`
        (explicit guard, then `_assert_flat_barrier` as a defensive
        numerical check at `z_max`), rather than silently building an
        incorrect forest.

        Parameters/returns deliberately match
        `PCHMergerTree.build_forest_numpy` exactly (`M0_array, z0, z_max,
        M_res, dz` in; `mass_history, split_events, z_steps,
        smooth_accretion, merger_mass` out, same shapes/semantics) --
        see that method's own docstring for the full field-by-field
        description and the mass-conservation identity, which this
        backend satisfies the same way (same forward-time, "0 means not
        resolved" bookkeeping).
        """
        self._ensure_delta_col_covers(z_max)
        self.cosmo_data.check_M_res(M_res, compiled=False)
        _assert_flat_barrier(self.model, z_max, self.cosmo_data, "ZhangHuiMergerTree.build_forest_numpy")

        M0_array = np.asarray(M0_array, dtype=np.float64)
        z_steps = np.arange(z0, z_max + dz * 0.5, dz)

        mass_history, split_events, smooth_accretion, merger_mass = _build_forest_flat_barrier_numpy(
            M0_array=M0_array,
            z_steps=z_steps,
            model=self.model,
            cosmo_data=self.cosmo_data,
            M_res=float(M_res),
            rng=self.rng,
        )
        return mass_history, split_events, z_steps, smooth_accretion, merger_mass

    def build_forest_numba(self, M0_array, z0, z_max, M_res, dz=0.1):
        """
        Evolve N merger trees using the Numba JIT kernel with thread-level
        parallelism (`nb.prange`) -- the parallel counterpart to
        `build_forest_numpy`, same closed-form flat-barrier Levy sampling
        (`_build_forest_flat_barrier_kernel`; see the module comment above
        it for how it reimplements `scipy.special.erfcinv`, which isn't
        callable from inside `@njit` code).

        Only supports a mass-independent barrier, guarded exactly as
        `build_forest_numpy` is; raises `NotImplementedError` otherwise.

        Randomness note: unlike `build_tree`/`build_forest_numpy` (which
        use `self.rng`, a `np.random.Generator`), this uses `np.random`
        calls inside `@njit` code, since numba's JIT cannot accept a
        `np.random.Generator` instance -- the only option available, and
        the same one `PCHMergerTree.build_forest_numba` already uses.
        **Not reproducible via any seed, `self.rng`'s or otherwise**: numba's
        `parallel=True`/`nb.prange` combination gives each worker thread
        its own internal random stream that is not deterministically tied
        to `np.random.seed()`, whether called before this method or from
        inside the kernel itself -- checked directly, calling
        `np.random.seed(s)` immediately before *and* as the jitted
        kernel's own first statement both still gave different output run
        to run. This is a real, disclosed limitation of numba's parallel
        RNG, not something this codebase controls; use `build_forest_numpy`
        instead if reproducibility matters more than the extra speed here.

        Parameters/returns match `build_forest_numpy` except for the
        absence of `split_events` -- same as
        `PCHMergerTree.build_forest_numba`'s own docstring explains for
        the identical reason (allocating variable-length Python lists
        inside `@njit` is expensive); use `merger_mass`'s nonzero entries
        instead.

        Raises
        ------
        ImportError
            If numba is not installed -- it's an optional dependency
            (`pip install foraois[numba]`); use `build_forest_numpy`
            instead (same closed-form algorithm, no numba needed).
        """
        if not _HAVE_NUMBA:
            raise ImportError(
                "build_forest_numba requires numba, which is not installed. "
                "Install it with `pip install foraois[numba]` (or `pip install "
                "numba` directly), or use build_forest_numpy instead -- same "
                "algorithm, no numba needed."
            )
        self._ensure_delta_col_covers(z_max)
        self.cosmo_data.check_M_res(M_res, lookup_factor=1.0, compiled=True)
        _assert_flat_barrier(self.model, z_max, self.cosmo_data, "ZhangHuiMergerTree.build_forest_numba")

        M0_array = np.asarray(M0_array, dtype=np.float64)
        z_steps = np.arange(z0, z_max + dz * 0.5, dz)

        logmass_grid = self.cosmo_data._logmass
        sigma_grid = self.cosmo_data._sigma
        mass_history, smooth_accretion, merger_mass = _build_forest_flat_barrier_kernel(
            M0_array=M0_array,
            z_steps=z_steps,
            dc_z_grid=self.cosmo_data._dc_z_grid,
            dc_dc_grid=self.cosmo_data._dc_dc_grid,
            logmass_grid=logmass_grid,
            sigma_grid=sigma_grid,
            sigma_grid_rev=sigma_grid[::-1].copy(),
            logmass_grid_rev=logmass_grid[::-1].copy(),
            M_res=float(M_res),
        )
        return mass_history, z_steps, smooth_accretion, merger_mass

    # ------------------------------------------------------------------
    # grow_full_population_numba -- JIT + parallel full-branch-population
    # growth (see the "Numba full-population kernel" block above
    # build_forest_numba for the algorithm)
    # ------------------------------------------------------------------

    def grow_full_population_numba(self, M0, z0, z_max, M_res, checkpoints, n_trees, dz=0.05, max_stack=20_000, max_out=4_000):
        """
        Grow n_trees independent full-branch-population realizations of a
        single initial mass M0, using the closed-form flat-barrier Levy
        sampler on a fixed dz grid (see the "Numba full-population kernel"
        module comment above for why no adaptive stepping is needed here,
        unlike PCH08's own `grow_full_population_numba_adaptive`).

        Drop-in numba replacement for scripts/paper_figs/_treegrowth.py's
        grow_full_population_zh_flat, looped n_trees times -- same
        {checkpoint_z: [branch masses]} return convention (one dict per
        tree), same closed-form math, just compiled and parallelized
        across trees (nb.prange) instead of a serial per-tree Python loop.
        Only supports a flat barrier (guarded as `build_forest_numpy` is);
        raises NotImplementedError otherwise.

        Parameters
        ----------
        M0 : float
            Single initial mass (Msun/h) -- every realization starts here
            (see PCHMergerTree.grow_full_population_numba_adaptive's own
            docstring for why this is kept simple rather than per-tree).
        checkpoints : sequence of float
            Redshifts at which to record the branch-mass population; each
            must land exactly on the z0 + n*dz grid (to within 1e-3, same
            tolerance as _treegrowth._checkpoint_indices), or this raises
            ValueError -- choose dz accordingly.
        max_stack, max_out : int
            Same silent-truncation-guard semantics as
            PCHMergerTree.grow_full_population_numba_adaptive's own
            max_stack/max_out (see that method's docstring) -- checked
            here via the same out_counts>=max_out warning.

        Returns
        -------
        list of dict
            One {checkpoint_z: [branch masses]} dict per tree.
        """
        if not _HAVE_NUMBA:
            raise ImportError(
                "grow_full_population_numba requires numba, which is not installed. "
                "Install it with `pip install foraois[numba]`, or use "
                "scripts/paper_figs/_treegrowth.py's grow_full_population_zh_flat "
                "instead (same closed-form algorithm, pure Python, no numba needed)."
            )
        self._ensure_delta_col_covers(z_max)
        self.cosmo_data.check_M_res(M_res, lookup_factor=1.0, compiled=True)
        _assert_flat_barrier(self.model, z_max, self.cosmo_data, "ZhangHuiMergerTree.grow_full_population_numba")

        z_steps = np.arange(z0, z_max + dz * 0.5, dz)
        checkpoints_sorted = sorted(set(checkpoints))
        checkpoint_step_idx = np.empty(len(checkpoints_sorted), dtype=np.int64)
        for k, zc in enumerate(checkpoints_sorted):
            idx = int(np.argmin(np.abs(z_steps - zc)))
            if abs(z_steps[idx] - zc) > max(1e-8, 1e-3):
                raise ValueError(
                    f"checkpoint z={zc} is not on the dz grid (nearest grid point is "
                    f"z={z_steps[idx]:.4f}) -- choose dz so that z0 + n*dz hits every "
                    f"checkpoint exactly."
                )
            checkpoint_step_idx[k] = idx

        M0_array = np.full(int(n_trees), float(M0), dtype=np.float64)
        logmass_grid = self.cosmo_data._logmass
        sigma_grid = self.cosmo_data._sigma

        out_masses, out_counts = _grow_forest_zh_flat_kernel(
            M0_array=M0_array,
            M_res=float(M_res),
            z_steps=z_steps,
            checkpoint_step_idx=checkpoint_step_idx,
            dc_z_grid=self.cosmo_data._dc_z_grid,
            dc_dc_grid=self.cosmo_data._dc_dc_grid,
            logmass_grid=logmass_grid,
            sigma_grid=sigma_grid,
            sigma_grid_rev=sigma_grid[::-1].copy(),
            logmass_grid_rev=logmass_grid[::-1].copy(),
            max_stack=int(max_stack),
            max_out=int(max_out),
        )

        if np.any(out_counts >= max_out):
            n_hit = int(np.sum(np.any(out_counts >= max_out, axis=1)))
            warnings.warn(
                f"grow_full_population_numba: {n_hit}/{n_trees} realizations hit "
                f"max_out={max_out} for at least one checkpoint -- their branch population at "
                "that checkpoint was silently truncated. Increase max_out and re-run before "
                "trusting statistics built from this output.",
                stacklevel=2,
            )

        results = []
        for i in range(len(M0_array)):
            pops = {}
            for k, zc in enumerate(checkpoints_sorted):
                c = int(out_counts[i, k])
                pops[float(zc)] = out_masses[i, k, :c].tolist()
            results.append(pops)
        return results
