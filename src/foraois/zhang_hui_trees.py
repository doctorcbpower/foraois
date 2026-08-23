"""
Zhang & Hui (2006)-based progenitor-mass sampling: the exact,
barrier-agnostic replacement for PCH08's fitted branching rate.

`draw_progenitor_mass_zh` is the single-progenitor analog of
`pch_trees.PCHMergerTree.draw_progenitor_masses`: given a halo of mass
`M0` at `z0`, and a target redshift `z1`, it draws the progenitor(s) at
`z1` using the *exact* first-crossing distribution (`first_crossing.py`'s
`solve_first_crossing`, evaluated against a barrier from
`foraois.collapse.delta_c`) instead of PCH08's fitted rate.

Origin-shift convention (standard excursion-set/conditional-mass-function
setup, e.g. Bond, Cole, Efstathiou & Kaiser 1991): the walk is tracked
relative to the parent halo's own trajectory point, `S0 = sigma(M0)^2`,
`delta0 = delta_c(M0, z0, model, cosmo_data)`. A progenitor of mass
`M(S) < M0` at `z1` corresponds to first crossing the shifted barrier
`B(S) = delta_c(M(S), z1, model, cosmo_data) - delta0` at variance
`S = sigma(M(S))^2 - S0` (S increases as progenitor mass decreases).
For the CDM/WDM models (`foraois.collapse`'s barrier is constant in M
for both), `B(S)` reduces to the constant `delta_c(z1) - delta_c(z0)` --
i.e. exactly the flat-barrier case `first_crossing.py`'s own
`flat_barrier_first_crossing` already validates independently, which is
this module's main internal consistency check (see
tests/test_zhang_hui_trees.py).

The unresolved-accretion mass fraction (PCH08's `F`) is not free from
`first_crossing.py`'s existing machinery -- it is computed here as its
own integral of `f(S)` over the sub-`M_res` tail (`S > S_res`), mirroring
`pch_trees._unresolved_accretion_fraction`'s role but evaluated against
the exact rate. The resolved-crossing probability and mass draw both
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

import warnings

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erfc, erfcinv

from foraois.collapse import delta_c
from foraois.first_crossing import solve_first_crossing


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
    distribution -- the closed-form integral of
    first_crossing.py's flat_barrier_first_crossing (Zhang & Hui eq. 18's
    b=0, constant-barrier case), a Levy distribution with scale b^2:

        F(S) = erfc(b / sqrt(2*S))

    Checked directly against numerical integration of
    flat_barrier_first_crossing to machine precision. This is what makes
    build_forest_numpy (below) avoid solve_first_crossing's O(N_grid^2)
    solve entirely for the currently-only-supported flat-barrier models
    (cdm, wdm, fdm's placeholder) -- see _assert_flat_barrier.
    """
    S = np.asarray(S, dtype=float)
    with np.errstate(divide="ignore"):
        return erfc(b / np.sqrt(2.0 * S))


def _flat_barrier_sample(u, b):
    """
    Inverse of _flat_barrier_cdf: given u = F(S) in [0, 1], returns S, via
    the standard Levy-distribution quantile function
    S = b^2 / (2 * erfcinv(u)^2). u=0 (erfcinv(0)=inf) maps to S=0 rather
    than raising -- harmless here since build_forest_numpy only samples
    u=0 for entries it masks out afterward (see its own comments).
    """
    u = np.asarray(u, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return b**2 / (2.0 * erfcinv(u) ** 2)


def _assert_flat_barrier(model, z_ref, cosmo_data):
    """
    build_forest_numpy's closed-form Levy sampling only applies when
    delta_c(M, z, model, cosmo_data) doesn't depend on M -- true today for
    cdm, wdm, and fdm's placeholder (all three literally return the same
    constant-in-M value regardless of the M passed in; see collapse/*.py),
    false for sidm (which raises before this check would even matter) and
    for any future genuinely mass-dependent barrier (real FDM physics).
    Checked at runtime against two very different masses, rather than
    hardcoded against model names, so this keeps working automatically if
    a model's flatness changes, and fails loudly instead of silently
    mis-sampling if it isn't flat.
    """
    M_lo, M_hi = 1e6, 1e16
    d_lo = float(delta_c(M_lo, z_ref, model, cosmo_data))
    d_hi = float(delta_c(M_hi, z_ref, model, cosmo_data))
    if not np.isclose(d_lo, d_hi, rtol=1e-8, atol=1e-12):
        raise NotImplementedError(
            f"build_forest_numpy's closed-form sampling requires a "
            f"mass-independent (flat) collapse barrier; model={model!r} "
            f"is not flat at z={z_ref} (delta_c({M_lo:.0e})={d_lo!r} != "
            f"delta_c({M_hi:.0e})={d_hi!r}). Not yet supported -- see "
            "ROADMAP.md."
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
                    clipped to [0, 1]
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

    delta0 = float(delta_c(M0, z0, model, cosmo_data))
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
        dc = np.asarray(delta_c(M_S, z1, model, cosmo_data), dtype=float)
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
    redshift `z1`, using the exact Zhang & Hui (2006) first-crossing
    distribution for `model`'s collapse barrier (`foraois.collapse`).

    Parameters
    ----------
    M0 : float
        Parent halo mass (Msun/h) at z0.
    z0, z1 : float
        Starting and target redshift (z1 > z0).
    M_res : float
        Mass resolution limit (Msun/h).
    cosmo_data : CosmoData
        Must already have `_prepare_sigma_grid` run (a side effect of
        constructing a `PCHMergerTree` against it today -- see
        `pch_trees.PCHMergerTree.__init__`; calling it directly is also
        fine: `cosmo_data._prepare_sigma_grid(cosmo_data.get_power_spectrum())`).
    model : {'cdm', 'wdm', 'fdm', 'sidm'}
        Passed straight through to `foraois.collapse.delta_c` -- 'fdm'
        will warn (disclosed placeholder), 'sidm' will raise (not
        implemented); see collapse/__init__.py.
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
        accretion fraction computed internally (mirrors
        `pch_trees.draw_progenitor_masses`'s [M1, M2] / [M_continuing] /
        [] return convention exactly). Both progenitors are guaranteed
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
    M2 = float(_mass_at_S(np.array([S_star]), step["sigma0_sq"], cosmo_data)[0])

    M_continuing -= M2

    progenitors = []
    if M_continuing >= M_res:
        progenitors.append(M_continuing)
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
        d0 = float(delta_c(M_res, z0, model, cosmo_data))
        d1 = float(delta_c(M_res, z1, model, cosmo_data))
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


class ZhangHuiMergerTree:
    """
    The barrier-agnostic, exact-rate counterpart to
    `pch_trees.PCHMergerTree`, built on `draw_progenitor_mass_zh`.

    `build_tree` (serial, single-halo) deliberately mirrors
    `PCHMergerTree.build_tree`'s own structure/signature (same z-stepping
    loop, same tree-of-dicts return shape) so downstream code
    (`utils/plot.py`) needs no special-casing to accept either backend's
    tree; use it for a single illustrative tree, or where a genuinely
    mass-dependent barrier is needed (once one exists -- see
    `_assert_flat_barrier`).

    `build_forest_numpy` is the vectorised, forest-scale backend -- but
    only for the currently-flat-barrier models (cdm/wdm/fdm's
    placeholder; see `_build_forest_flat_barrier_numpy`'s docstring for
    why this is a closed-form path, not a vectorised
    `first_crossing_step`), matching `PCHMergerTree.build_forest_numpy`'s
    signature and return shape. No `build_forest_numba` yet (see
    ROADMAP.md).
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
        self.cosmo_data = cosmo_data
        self.params = params
        self.model = model
        self.rng = rng if rng is not None else np.random.default_rng()
        self.N_grid = N_grid
        self.S_max_factor = S_max_factor

        # Same side effect PCHMergerTree.__init__ performs: CosmoData's
        # sigma(M) interpolants (and, since cosmo_utils.py's change
        # alongside this module, their inverse) aren't built until this is
        # called -- see cosmo_utils.py's _prepare_sigma_grid docstring.
        self.pk_data = cosmo_data.get_power_spectrum()
        self.cosmo_data._prepare_sigma_grid(self.pk_data)

    def _ensure_delta_col_covers(self, z_max):
        """
        Same guard as `PCHMergerTree._ensure_delta_col_covers` (see that
        method's docstring for the silent-clamping failure mode this
        avoids) -- applies equally here, since `draw_progenitor_mass_zh`
        also reaches `cosmo_data.delta_col_at_z` indirectly, through
        `collapse.delta_c_cdm`/`delta_c_wdm`.
        """
        current_z_max = self.cosmo_data._dc_z_grid[-1]
        if z_max <= current_z_max:
            return
        self.cosmo_data.precompute_delta_col_table(z_max=z_max)

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

        Only supports models whose barrier is mass-independent (checked
        once via `_assert_flat_barrier`, at `z_max` -- true today for
        `cdm`/`wdm`/`fdm`'s placeholder, not `sidm`); raises
        `NotImplementedError` otherwise, rather than silently building an
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
        _assert_flat_barrier(self.model, z_max, self.cosmo_data)

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
