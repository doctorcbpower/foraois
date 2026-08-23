"""
The constrained branch's growth history -- ties together
`first_crossing_constrained.py`'s barrier-free `simulate_bridge_path`
with `foraois.collapse.delta_c` to produce an actual `(redshift, mass)`
sequence for a branch guaranteed to reach `(M1, z1)`.

This module exists separately from `zhang_hui_trees.py` and
`first_crossing_constrained.py` rather than folding into either: it needs
both `first_crossing_constrained`'s bridge math *and*
`cosmo_data`/`delta_c`-aware mass<->variance/barrier conversions the way
`zhang_hui_trees.py` has, but is conceptually a different operation (path
simulation + successive-maxima extraction, not first-crossing-density
sampling) from what either of those already do.

Algorithm:

1. Simulate one Brownian-bridge sample path `delta(S)` from the halo's own
   origin `(S=0, delta=0)` to the fixed constraint endpoint `(S1, delta1)`
   (`simulate_bridge_path` -- barrier-free by construction).
2. Track the *running maximum* of `delta(S)` as S increases -- the
   standard EPS "only the running maximum is physical" logic (Bond, Cole,
   Efstathiou & Kaiser 1991's cloud-in-cloud argument): a region that has
   already collapsed stays collapsed, so a later downward fluctuation in
   delta doesn't "un-collapse" it. This is also what makes N23's explicit
   reflection-principle step unnecessary here -- an early up-crossing of
   delta1 simply becomes (and stays) the running maximum regardless of
   what the raw path does afterward, which is exactly the physical
   behaviour the reflection principle exists to guarantee in their
   different (integral-equation-based) derivation.
3. Each point where the running maximum *increases* is a genuine collapse
   event: invert `delta_c(M(S), z, model, cosmo_data) = delta0_abs +
   delta_max(S)` for `z`, given `S` (hence `M(S)`) known -- a plain 1D
   root-find against whatever `delta_c` returns, which is what makes this
   approach barrier-agnostic: it works unchanged for a future
   mass-dependent barrier.
4. Filter to `M(S) >= M_res`, matching every other backend's convention.
"""

import numpy as np
from scipy.optimize import brentq

from foraois.collapse import delta_c
from foraois.first_crossing_constrained import simulate_bridge_path
from foraois.zhang_hui_trees import ZhangHuiMergerTree, _mass_at_S


def _extract_successive_maxima(S_grid, delta_path):
    """
    Running-maximum extraction (see module docstring, point 2): returns
    the `(S, delta)` pairs at which `delta_path` sets a new running-max
    record -- these, and only these, correspond to genuine collapse
    events. The very first point (S=0, delta=0) always counts (it's the
    halo's own starting point).
    """
    running_max = np.maximum.accumulate(delta_path)
    is_new_record = np.concatenate([[True], running_max[1:] > running_max[:-1]])
    return S_grid[is_new_record], delta_path[is_new_record]


def _invert_z_for_delta(target_delta_absolute, M, z_lo, cosmo_data, model):
    """
    Solve `delta_c(M, z, model, cosmo_data) = target_delta_absolute` for
    `z >= z_lo`, via `scipy.optimize.brentq` with an expanding upper
    bracket (no a priori bound on how high z might need to go for an
    extreme constraint) -- a plain root-find against whatever `delta_c`
    returns, deliberately not special-cased for CDM/WDM's mass-independent
    barrier, so this keeps working unchanged once a genuinely
    mass-dependent barrier exists (see module docstring).

    `cosmo_data`'s precomputed delta_col(z) table is extended as needed
    (same guard `zhang_hui_trees.ZhangHuiMergerTree._ensure_delta_col_covers`
    uses) so `delta_c`'s underlying `delta_col_at_z` lookup doesn't
    silently clamp beyond the table's current range.
    """

    def f(z):
        return float(delta_c(M, z, model, cosmo_data)) - target_delta_absolute

    z_hi = z_lo + 5.0
    while f(z_hi) < 0:
        z_hi *= 1.5
        if z_hi > 1000.0:
            raise RuntimeError(
                f"Could not bracket a root for delta_c(M={M:.3e}, z)="
                f"{target_delta_absolute} below z=1000 -- constraint may be "
                "unphysically extreme, or delta_c is not monotonically "
                "increasing in z for this model."
            )
        current_z_max = cosmo_data._dc_z_grid[-1]
        if z_hi > current_z_max:
            cosmo_data.precompute_delta_col_table(z_max=z_hi)

    return brentq(f, z_lo, z_hi)


def constrained_branch_growth_history(
    M0,
    z0,
    M1,
    z1,
    M_res,
    cosmo_data,
    model="cdm",
    dS=0.01,
    rng=None,
):
    """
    Simulate the growth history of a single branch guaranteed to reach
    mass `M1` at redshift `z1`, starting from `(M0, z0)`.

    Parameters
    ----------
    M0, z0 : float
        The branch's own starting point (M0 > M1, z0 < z1).
    M1, z1 : float
        The fixed constraint the branch must satisfy exactly.
    M_res : float
        Mass resolution limit -- collapse events below this are dropped.
    cosmo_data : CosmoData
        Must already have `_prepare_sigma_grid` run (see
        `zhang_hui_trees.ZhangHuiMergerTree.__init__`).
    model : {'cdm', 'wdm', 'fdm'}
        'sidm' will raise (not implemented) via `collapse.delta_c`.
    dS : float
        Step size for `simulate_bridge_path`'s own S-grid.
    rng : np.random.Generator, optional

    Returns
    -------
    list of dict, each `{"redshift": z, "mass": M, "smooth_accretion": ...,
    "merger_mass": 0.0}` (the first entry, the root at z0, has no
    `smooth_accretion`/`merger_mass` keys -- there is nothing "before" it
    to have accreted from), in order of *increasing* redshift (i.e. going
    backward in time, matching `build_tree`'s own tree-of-dicts ordering)
    -- the last entry is guaranteed to be `{"redshift": z1, "mass": M1}`
    exactly (up to floating point and the z-inversion's own tolerance),
    unless M1 < M_res, in which case entries below M_res are dropped and
    the branch simply ends at the last resolved collapse event.

    `smooth_accretion` here is `mass[i-1] - mass[i]` -- the running-
    maximum trajectory's own mass loss between two successive collapse
    events. This is *not* a discrete "other progenitor" the way
    `build_tree`'s own `merger_mass` is (which is why `merger_mass` is
    always exactly `0.0` here, honestly, not "unknown"): the path between
    two recorded maxima is a continuous random-walk excursion, not a
    binary split with a tracked complement, so there is no specific other
    halo to attribute this mass to -- it is genuinely smooth-accretion-like
    bookkeeping, matching the physical distinction `pch_trees.py`'s own
    `smooth_accretion`/`merger_mass` channels draw. *Real* secondary
    progenitor sub-trees branching off the constrained portion itself
    (not just this smooth-accretion accounting) are a separate, larger
    piece of work -- see `build_constrained_tree`'s own docstring for why
    that's still deferred, not attempted here.
    """
    if M1 >= M0:
        raise ValueError(f"M1={M1} must be < M0={M0}.")
    if z1 <= z0:
        raise ValueError(f"z1={z1} must be > z0={z0}.")

    delta0_abs = float(delta_c(M0, z0, model, cosmo_data))
    delta1_abs = float(delta_c(M1, z1, model, cosmo_data))
    sigma0_sq = float(cosmo_data.sigma_at_logmass(np.log10(M0))) ** 2
    sigma1_sq = float(cosmo_data.sigma_at_logmass(np.log10(M1))) ** 2
    S1 = sigma1_sq - sigma0_sq
    if S1 <= 0:
        raise ValueError(
            f"M1={M1} does not give sigma(M1) > sigma(M0={M0}) -- "
            "M1 must be strictly below M0 on the sigma(M) relation."
        )
    delta1 = delta1_abs - delta0_abs  # shifted convention

    S_grid, delta_path = simulate_bridge_path(0.0, 0.0, S1, delta1, dS=dS, rng=rng)
    S_max, delta_max = _extract_successive_maxima(S_grid, delta_path)

    history = [{"redshift": z0, "mass": M0}]
    z_prev = z0
    m_prev = M0
    for S, delta in zip(S_max[1:], delta_max[1:], strict=True):
        # The running maximum can reach delta1 *before* S1 -- the path is
        # only pinned at the two endpoints, not barrier-constrained in
        # between, so an early upward excursion past delta1 followed by a
        # dip back down (to still end exactly at delta1 at S1) is a real,
        # expected occurrence, not a bug. Per N23's own reflection-
        # principle description (Section 3.2.2): once delta1 is reached
        # at all, the constraint's own guarantee takes over -- force the
        # node to be exactly (M1, z1) right here, rather than tracking
        # the overshoot point's own (necessarily smaller) mass as though
        # it were the literal answer. This is what makes the explicit
        # reflection-principle simulation N23 need for their different
        # (integral-equation-based) derivation unnecessary here (see
        # module docstring, point 2).
        if delta >= delta1 - 1e-9:
            history.append(
                {
                    "redshift": z1,
                    "mass": M1,
                    "smooth_accretion": m_prev - M1,
                    "merger_mass": 0.0,
                }
            )
            break
        M = float(_mass_at_S(np.array([S]), sigma0_sq, cosmo_data)[0])
        if M < M_res:
            break
        z = _invert_z_for_delta(delta0_abs + delta, M, z_prev, cosmo_data, model)
        history.append(
            {
                "redshift": z,
                "mass": M,
                "smooth_accretion": m_prev - M,
                "merger_mass": 0.0,
            }
        )
        z_prev, m_prev = z, M
    else:
        # Defensive fallback -- the path is pinned to delta1 exactly at
        # S1 by construction, so the loop should always hit the delta>=
        # delta1 branch above before exhausting S_max/delta_max; this
        # only guards against an unexpected floating-point miss.
        history.append(
            {
                "redshift": z1,
                "mass": M1,
                "smooth_accretion": m_prev - M1,
                "merger_mass": 0.0,
            }
        )

    return history


def build_constrained_tree(
    M0,
    z0,
    M1,
    z1,
    z_max,
    M_res,
    cosmo_data,
    model="cdm",
    dS=0.01,
    rng=None,
    N_grid=400,
    S_max_factor=8.0,
    dz=0.1,
):
    """
    A full constrained tree -- grafts `constrained_branch_growth_history`'s
    constrained branch (`z0` to `z1`, guaranteed to reach `M1`) onto
    ordinary unconstrained sampling (`ZhangHuiMergerTree.build_tree`) for
    the continuation from `(M1, z1)` out to `z_max`, matching N23's own
    description: "the remainder of the constrained branch... is
    constructed using the unconstrained first-crossing rate distribution."

    Returns the same tree-of-dicts shape `ZhangHuiMergerTree.build_tree`
    uses (`{"redshift", "parent_mass", "progenitors"}` per entry, ordered
    by increasing redshift) so downstream code doesn't need to
    special-case a constrained tree. Every entry (both the constrained
    portion, `z <= z1`, and the unconstrained continuation, `z > z1`)
    carries `smooth_accretion`/`merger_mass` too -- see
    `constrained_branch_growth_history`'s own docstring for what
    `smooth_accretion` means for the constrained portion specifically
    (the running-maximum trajectory's own mass loss between collapse
    events, not a discrete "other progenitor" -- `merger_mass` is
    honestly `0.0` there, not "unknown").

    **Not implemented, disclosed rather than silently approximated:
    *real* "secondary" branches.** N23's own algorithm also grows
    unconstrained sub-trees off every branching event *along* the
    constrained branch itself, not just from `(M1,z1)` onward -- what
    `smooth_accretion` tracks (above) is honest bookkeeping for the mass
    involved, not a substitute for actually growing those sub-trees.
    Doing that properly needs a `ZhangHuiMergerTree.build_full_tree` (a
    version of `PCHMergerTree.build_full_tree` for the exact-rate
    backend) as a prerequisite -- it doesn't exist yet either, for the
    *unconstrained* case, so this isn't a constrained-tree-specific gap
    so much as a larger piece of work neither backend has yet
    (`ZhangHuiMergerTree.build_tree`/`PCHMergerTree.build_tree` both make
    the same "main branch only" simplification, as opposed to
    `build_full_tree`). See ROADMAP.md.

    Parameters
    ----------
    M0, z0, M1, z1 : float
        See `constrained_branch_growth_history`.
    z_max : float
        End redshift for the unconstrained continuation beyond `z1`
        (must be >= `z1`; if `M1 < M_res`, no continuation is grown
        regardless, since the branch has already ended).
    M_res : float
    cosmo_data : CosmoData
    model : {'cdm', 'wdm', 'fdm'}
    dS : float
        Step size for the constrained branch's own path simulation (see
        `constrained_branch_growth_history`).
    rng : np.random.Generator, optional
    N_grid, S_max_factor : passed through to the unconstrained
        continuation's `ZhangHuiMergerTree` (see its own docstring).
    dz : float
        Redshift step for the unconstrained continuation.

    Returns
    -------
    list of dict -- see above.
    """
    if z_max < z1:
        raise ValueError(f"z_max={z_max} must be >= z1={z1}.")

    rng = rng if rng is not None else np.random.default_rng()

    history = constrained_branch_growth_history(M0, z0, M1, z1, M_res, cosmo_data, model, dS, rng)

    tree = []
    for prev, cur in zip(history[:-1], history[1:], strict=True):
        tree.append(
            {
                "redshift": cur["redshift"],
                "parent_mass": prev["mass"],
                "progenitors": [cur["mass"]],
                "smooth_accretion": cur["smooth_accretion"],
                "merger_mass": cur["merger_mass"],
            }
        )

    reached_z1 = history[-1]["redshift"] == z1
    if reached_z1 and z_max > z1 and history[-1]["mass"] >= M_res:
        zh_tree = ZhangHuiMergerTree(
            cosmo_data,
            model=model,
            rng=rng,
            N_grid=N_grid,
            S_max_factor=S_max_factor,
        )
        tree.extend(zh_tree.build_tree(history[-1]["mass"], z1, z_max, M_res, dz=dz))

    return tree
