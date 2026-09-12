"""
Shared full-branch tree-growth helper for the PCH08-paper-style diagnostic
scripts in this directory (fig1/fig2/fig3/fig4).

foraois's own ``PCHMergerTree.build_full_tree`` grows the complete
branching structure of a *single* PCH08 tree, for illustration (e.g. a
dendrogram), but only records a node when something happens (a split, or
a branch's terminal state) -- it does not give you "every branch's mass
at a fixed target redshift" directly, and it has no Zhang-Hui equivalent
at all (ZhangHuiMergerTree only exposes main-progenitor tracking via
build_tree/build_forest_numpy; see ROADMAP.md's "secondary progenitor
branches" gap).

Reproducing Parkinson, Cole & Helly (2008)'s Figs 1-2 needs exactly that
"every branch's mass at a fixed target redshift" population, for *both*
backends on equal footing, so this module grows it directly: every active
branch is stepped through the same dz grid (the same grid for both
backends, so any difference in the resulting statistics is attributable
to the branching-rate law itself, not to a different number of
generations of splitting), calling the single-halo, single-step draw
function each backend already exposes:

  * PCH08:      PCHMergerTree.draw_progenitor_masses(M, delta_0, delta_1,
                sigma_M, M_res) -- the same fitted-rate primitive
                build_tree/build_full_tree already use internally.
  * Zhang-Hui:  draw_progenitor_mass_zh(M, z0, z1, M_res, cosmo_data,
                model, rng, N_grid, S_max_factor) -- the same exact-rate
                primitive ZhangHuiMergerTree.build_tree already uses.

Both return a list of 0 (both fragments below M_res), 1 (no split, mass
reduced by unresolved accretion), or 2 (a resolved binary split) masses,
so the branch-population bookkeeping below is written once and shared by
both backends via a small adapter.

This is deliberately *not* vectorised -- branch counts vary tree to tree
and step to step, which does not lend itself to build_forest_numpy/numba's
fixed-shape-array approach. It is also, for the Zhang-Hui backend
specifically, much more expensive per branch than PCH08 (each call solves
a Volterra system rather than evaluating a closed-form rate) -- see each
fig script's own docstring for the resulting runtime guidance.
"""

import multiprocessing as mp
import sys
import time

import numpy as np

from foraois.zhang_hui_trees import _assert_flat_barrier, draw_progenitor_mass_zh, draw_progenitor_mass_zh_flat

# ----------------------------------------------------------------------
# Progress reporting.
#
# These scripts are slow (see the multiprocessing note below and each fig
# script's own "Runtime" docstring section), and a run is typically
# launched with nohup/sbatch on an HPC node rather than watched
# interactively, so progress is reported as plain periodic lines (no
# carriage-return progress bar, which does not render sensibly in a log
# file) rather than left silent until the whole ensemble finishes. Lines
# are throttled by elapsed time (default: every ~10s), not by tree count,
# so a run stays quiet at the expected cadence whether trees are cheap
# (PCH08) or expensive (Zhang-Hui).
# ----------------------------------------------------------------------


def _progress_map(job_fn, jobs, label, n_jobs=1, pool_ctx=None, initializer=None, initargs=(), report_every_s=10.0):
    """Run job_fn over jobs (serially if n_jobs<=1, else via a pool_ctx.Pool
    of n_jobs workers using imap_unordered), printing a periodic progress
    line to stderr. Order of the returned results is NOT guaranteed to
    match `jobs` (every caller in this module only aggregates results, so
    this is safe) -- imap_unordered lets the progress line update as soon
    as ANY worker finishes a tree, rather than waiting for a synchronized
    batch."""
    n_total = len(jobs)
    t0 = time.time()
    last_print = t0
    results = []

    def _report(done, force=False):
        nonlocal last_print
        now = time.time()
        if not force and (now - last_print) < report_every_s:
            return
        elapsed = now - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (n_total - done) / rate if rate > 0 else float("nan")
        print(
            f"  [{label}] {done}/{n_total} trees ({100.0 * done / max(n_total, 1):.0f}%)"
            f" | elapsed {elapsed:.0f}s | rate {rate:.2f} trees/s | ETA {eta:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        last_print = now

    if n_total == 0:
        return results

    if n_jobs <= 1:
        for i, j in enumerate(jobs, 1):
            results.append(job_fn(j))
            _report(i)
        _report(n_total, force=True)
        return results

    with pool_ctx.Pool(processes=n_jobs, initializer=initializer, initargs=initargs) as pool:
        for i, r in enumerate(pool.imap_unordered(job_fn, jobs), 1):
            results.append(r)
            _report(i)
    _report(n_total, force=True)
    return results

# ----------------------------------------------------------------------
# Multiprocessing support.
#
# Tree realizations are independent draws, so this is an embarrassingly
# parallel Monte Carlo problem -- the natural fix for the wall-clock cost
# noted throughout this module's docstring (especially for Zhang-Hui) is
# more cores, not a smaller n_trees. This sandbox only has 2 CPUs, but on
# an HPC login/compute node with --n-jobs=32 (say), the wall-clock cost
# drops by roughly that factor.
#
# Each worker process builds its own CosmoData/PCHMergerTree ONCE (via
# _worker_init), then reuses them across every tree assigned to that
# worker -- CosmoData construction (~2s, dominated by the sigma(M) table
# build) would otherwise dwarf a single tree's own cost if repeated per
# tree. Worker state lives in module-level globals rather than being
# passed through pickling, since PCHMergerTree/CosmoData instances are
# not guaranteed cheap (or safe) to pickle repeatedly.
# ----------------------------------------------------------------------
_worker_state = {}


def _worker_init(config_path):
    import warnings

    from foraois.cosmo_utils import CosmoData
    from foraois.pch_trees import PCHMergerTree
    from foraois.utils import io as foraois_io

    # A spawned worker gets its own default warnings filters -- the parent
    # process's `with warnings.catch_warnings(): simplefilter("ignore", ...)`
    # around the ensemble call does not reach here, so first_crossing_step's
    # expected coarse-grid warning (see this module's docstring) would
    # otherwise print once per step per tree in every worker.
    warnings.simplefilter("ignore", UserWarning)

    run_params = foraois_io.get_params(config_path)
    cosmo_data = CosmoData(run_params, redshift=[0.0])
    pch = PCHMergerTree(cosmo_data, run_params)
    _worker_state["cosmo_data"] = cosmo_data
    _worker_state["pch"] = pch


def _pch08_tree_job(args):
    M0, z0, z_max, M_res, dz, checkpoints, seed, target_nupper = args
    np.random.seed(seed)  # draw_progenitor_masses uses the global numpy RNG
    pch = _worker_state["pch"]
    if target_nupper is not None:
        return grow_full_population_pch08_adaptive(pch, M0, z0, z_max, M_res, checkpoints, target_nupper=target_nupper, dz_max=dz)
    return grow_full_population_pch08(pch, M0, z0, z_max, M_res, dz, checkpoints)


def _zh_tree_job(args):
    M0, z0, z_max, M_res, dz, checkpoints, model, N_grid, S_max_factor, seed = args
    cosmo_data = _worker_state["cosmo_data"]
    rng = np.random.default_rng(seed)
    return grow_full_population_zh_single(cosmo_data, M0, z0, z_max, M_res, dz, checkpoints, model=model, rng=rng, N_grid=N_grid, S_max_factor=S_max_factor)


def run_pch08_ensemble(config_path, M0, z0, z_max, M_res, dz, checkpoints, n_trees, n_jobs=1, seed0=0, label="PCH08", target_nupper=None):
    """Grow n_trees independent PCH08 full-branch-population realizations.
    Returns a list of n_trees dicts, each {checkpoint_z: [branch masses]}
    (one per grow_full_population_pch08 call).

    If target_nupper is given (the adaptive-stepping case every fig1/2/4
    caller now uses -- see each script's own docstring), this dispatches
    to grow_full_population_pch08_batch, which uses
    PCHMergerTree.grow_full_population_numba_adaptive when numba is
    installed: that kernel is JIT + prange-parallel *across trees itself*
    (~400-600x faster per tree than this module's serial Python loop --
    see grow_full_population_pch08_batch's own docstring), so it is run as
    one single-process batched call rather than spread across n_jobs
    worker processes (stacking multiprocessing on top would oversubscribe
    cores for no benefit, and numba's own RNG state isn't controlled by
    the per-tree seed0+i scheme used below). n_jobs/seed0 are then unused;
    they still apply to the target_nupper=None (fixed-dz, numba-unsupported)
    and numba-not-installed fallback paths below, both spread across
    n_jobs worker processes exactly as before.

    Falls back to the pre-existing serial-per-tree, optionally
    multiprocessed path (grow_full_population_pch08_adaptive/
    grow_full_population_pch08 via _pch08_tree_job) if target_nupper is
    None or numba is not installed."""
    from foraois.pch_trees import _HAVE_NUMBA

    if target_nupper is not None and _HAVE_NUMBA:
        _worker_init(config_path)
        pch = _worker_state["pch"]
        return grow_full_population_pch08_batch(pch, M0, z0, z_max, M_res, checkpoints, n_trees, target_nupper=target_nupper, dz_max=dz)

    jobs = [(M0, z0, z_max, M_res, dz, checkpoints, seed0 + i, target_nupper) for i in range(n_trees)]
    if n_jobs <= 1:
        _worker_init(config_path)
    ctx = mp.get_context("spawn")  # fork risks a BLAS-after-fork deadlock once the parent has touched
    return _progress_map(  # numpy/scipy's threaded linear algebra (e.g. building the sigma(M) table)
        _pch08_tree_job, jobs, label, n_jobs=n_jobs, pool_ctx=ctx, initializer=_worker_init, initargs=(config_path,)
    )


def run_zh_ensemble(config_path, M0, z0, z_max, M_res, dz, checkpoints, n_trees, model="cdm", N_grid=40, S_max_factor=8.0, n_jobs=1, seed0=0, label="Zhang-Hui"):
    """Zhang-Hui equivalent of run_pch08_ensemble. For a flat-barrier model
    (cdm/wdm/fdm's placeholder) with numba installed, dispatches to
    grow_full_population_zh_batch's numba path -- JIT + prange-parallel
    *across trees itself* (no solve_first_crossing at all -- see
    ZhangHuiMergerTree.grow_full_population_numba's own docstring), run as
    one single-process batched call rather than spread across n_jobs
    worker processes (same reasoning as run_pch08_ensemble's own numba
    dispatch: stacking multiprocessing on top would oversubscribe cores
    for no benefit). n_jobs/seed0 are then unused; they still apply to the
    non-flat-barrier and numba-not-installed fallback paths below, spread
    across n_jobs worker processes exactly as before (now running the
    closed-form grow_full_population_zh_flat per tree when the model is
    flat but numba isn't installed, still much cheaper than the general
    solver -- see _zh_tree_job/grow_full_population_zh_single)."""
    from foraois.zhang_hui_trees import _HAVE_NUMBA, _assert_flat_barrier

    _worker_init(config_path)
    cosmo_data = _worker_state["cosmo_data"]
    try:
        _assert_flat_barrier(model, z_max, cosmo_data)
        flat = True
    except NotImplementedError:
        flat = False

    if flat and _HAVE_NUMBA:
        return grow_full_population_zh_batch(cosmo_data, M0, z0, z_max, M_res, dz, checkpoints, n_trees, model=model)

    jobs = [(M0, z0, z_max, M_res, dz, checkpoints, model, N_grid, S_max_factor, seed0 + i) for i in range(n_trees)]
    ctx = mp.get_context("spawn")
    return _progress_map(_zh_tree_job, jobs, label, n_jobs=n_jobs, pool_ctx=ctx, initializer=_worker_init, initargs=(config_path,))


# ----------------------------------------------------------------------
# Main-progenitor-only ensembles (fig3's use case: cheap enough that a
# single, mostly Python-loop-driven build_tree() call per realization is
# fine -- no full branch population needed -- but still parallelized
# since fig3 is the one script where a "just crank up n_trees, it's cheap
# per tree" run can still add up to a lot of wall-clock at large n_trees.
# ----------------------------------------------------------------------


def most_recent_major_merger_z(tree, f_major):
    """tree is the list of step dicts build_tree() returns, in increasing-z
    order; return the redshift of the first (lowest-z) entry with two
    progenitors whose mass ratio is >= f_major, or None if none qualify."""
    for entry in tree:
        progenitors = entry["progenitors"]
        if len(progenitors) == 2:
            m1, m2 = progenitors
            ratio = min(m1, m2) / max(m1, m2)
            if ratio >= f_major:
                return entry["redshift"]
    return None


# ----------------------------------------------------------------------
# Adaptive-dz PCH08 stepping
# ----------------------------------------------------------------------
#
# PCHMergerTree.build_tree/build_forest_numpy/build_forest_numba all use a
# single, fixed dz shared across every step (and, for the forest builders,
# every halo) -- an explicit, documented architectural simplification
# relative to PCH08's own algorithm, which instead adaptively picks the
# redshift step delta_z1 per halo per step "such that P << 1" (PCH08 sec.
# 2.1, page 558) where P is what PCHMergerTree.branching_rate_terms calls
# Nupper -- their eq. A5 upper bound on the per-step split probability.
# PCH08's target is Nupper ~ 0.1; nothing in the fixed-dz path enforces
# this, and diagnostics.expected_splits_per_step shows it is badly
# violated (Nupper of order 1-300, not <<1) at every dz/M0/M_res
# combination used in this paper's actual figures and in the existing
# validate_zhang_hui_vs_pch08.py Section-5.4 benchmark -- see the
# fig3_major_merger_redshift.py module docstring for the concrete numbers.
#
# The functions below restore PCH08's adaptive step-size choice for the
# two *serial*, per-branch-Python-loop code paths in this file
# (build_tree-style main-branch growth, and grow_full_population_pch08's
# per-branch population growth) -- neither of those is the vectorised
# build_forest_numpy/numba path the class docstring's "shared grid" caveat
# is actually about, so adaptive stepping is a straightforward drop-in
# here without touching foraois's library code or its vectorised forest
# builders (which would need a materially bigger rewrite -- a per-halo,
# per-step-varying grid does not vectorise the same way; not attempted
# here, see the fig3 docstring for how this bears on the Section 5.4
# benchmark specifically).
#
# Mechanism: at each step, bisect on the target redshift z_next in
# (z_cur, z_ceiling] for the largest step whose Nupper (evaluated once, at
# the *start*-of-step mass/delta, exactly as PCHMergerTree itself does)
# does not exceed target_nupper. Nupper is monotonically increasing in the
# step size (domega = delta(z_next) - delta(z_cur), for fixed M_cur), so
# bisection is well posed. z_ceiling caps how big a single step is allowed
# to grow when Nupper happens to stay small for a long stretch (e.g. once
# a branch is far from any imminent split) -- this only affects step
# *coarseness* in quiet stretches, not correctness.


def _nupper_at(pch, M_cur, M_res, delta_0, domega):
    if domega <= 0:
        return 0.0
    terms = pch.branching_rate_terms(M_cur, M_res, delta_0, domega)
    return float(terms["Nupper"])


def _eps1_domega_cap(pch, M_cur, eps1):
    """PCH08's own *second*, independent step-size constraint (split_PCH.F90,
    the reference FORTRAN implementation's ``dw < eps1*sfac`` check), missed
    entirely by earlier versions of this module's adaptive stepper.

    The Nupper/eps2 cap (see _pick_adaptive_step) only bounds the *total*
    expected number of splits per step small; it says nothing about whether
    dn/dq itself stays proportional to the step size dw=domega within that
    step, which is a separate linearization the whole rejection-sampling
    scheme (S(q)*dw as the split rate) assumes. PCH08's own code guards this
    with an independent absolute cap on domega:

        domega <= eps1 * sqrt(2*(sigma(M2/2)^2 - sigma(M2)^2))

    which does not depend on M_res/qres at all (unlike the eps2 cap), so it
    binds hardest for extreme M_res/M2 ratios where the eps2 cap alone would
    otherwise allow a much larger domega. Cross-checked against an actual
    compiled run of the reference FORTRAN (Parkinson's own split_PCH.F90):
    omitting this cap produced a robust, statistically significant ~7-12%
    over-count of resolved branches at every checkpoint relative to the
    FORTRAN's own output, at fixed eps2/target_nupper -- see this repo's
    paper-validation notes for the full comparison.
    """
    sigma_m2 = float(pch._sigma_at_mass(M_cur))
    sigma_half = float(pch._sigma_at_mass(0.5 * M_cur))
    diff = sigma_half**2 - sigma_m2**2
    if diff <= 0.0:
        # Degenerate (M_cur so small sigma(M/2)<=sigma(M) numerically) --
        # match split_PCH.F90's own implicit behaviour of falling back to
        # the other cap(s) by returning +inf here (no eps1 constraint).
        return float("inf")
    return eps1 * np.sqrt(2.0 * diff)


def _pick_adaptive_step(pch, z_cur, delta_0, z_ceiling, M_cur, M_res, target_nupper, min_dz, eps1=0.1):
    """Return (z_next, delta_next) for a step in (z_cur, z_ceiling] with
    Nupper close to (at or just below) target_nupper, i.e. the largest step
    PCH08's own "P << 1" criterion permits -- AND satisfying PCH08's second,
    independent eps1 linearity constraint (see _eps1_domega_cap), exactly as
    the reference FORTRAN's split_PCH.F90 does (``dw = min(eps1*sfac,
    dw_eps2, dwmax)``).

    Nupper is monotone increasing in the step size domega = delta(z_next) -
    delta_0 for fixed M_cur/delta_0, and empirically close to linear in
    domega for the small-domega regime this function lives in (Nupper
    scaling checks in this module's development: ~50x change in Nupper for
    a ~50x change in domega) -- so instead of blind bisection (which needs
    dozens of _branching_rate_terms calls, the dominant cost of adaptive
    stepping, to reach useful precision), warm-start from that linear
    scaling and refine with a handful of secant-style corrections, only
    falling back to a bisection safety net if the warm start misbehaves.
    """
    delta_ceiling = pch._delta_col_at_z(z_ceiling)
    domega_ceiling = delta_ceiling - delta_0

    domega_eps1_cap = _eps1_domega_cap(pch, M_cur, eps1)
    domega_full = min(domega_ceiling, domega_eps1_cap)

    if domega_full <= 0.0:
        # eps1 cap collapsed the step to (numerically) zero -- take the
        # smallest allowed step rather than stalling.
        z_next = min(z_cur + min_dz, z_ceiling)
        return z_next, pch._delta_col_at_z(z_next)

    Nupper_full = _nupper_at(pch, M_cur, M_res, delta_0, domega_full)
    if Nupper_full <= target_nupper:
        if domega_full >= domega_ceiling:
            return z_ceiling, delta_ceiling
        # eps1, not eps2, is the binding constraint: still need to convert
        # domega_full back to a z (same bisection as the general path below).
        delta_target = delta_0 + domega_full
        lo, hi = z_cur, z_ceiling
        for _ in range(20):
            if hi - lo <= min_dz:
                break
            mid = 0.5 * (lo + hi)
            if pch._delta_col_at_z(mid) <= delta_target:
                lo = mid
            else:
                hi = mid
        z_next = max(lo, z_cur + min_dz)
        z_next = min(z_next, z_ceiling)
        return z_next, pch._delta_col_at_z(z_next)

    # Linear warm start, then a few multiplicative (secant-like) corrections
    # in domega space -- cheap because it avoids ever re-bisecting a wide
    # bracket. Converges in ~3-5 _nupper_at calls in practice.
    domega = domega_full * (target_nupper / Nupper_full)
    for _ in range(6):
        domega = min(max(domega, 0.0), domega_full)
        Nupper = _nupper_at(pch, M_cur, M_res, delta_0, domega)
        if Nupper <= 1e-300:
            break
        if 0.85 * target_nupper <= Nupper <= 1.02 * target_nupper:
            break
        domega = domega * (target_nupper / Nupper)

    delta_target = delta_0 + domega
    # Convert the accepted domega back to a redshift via bisection on z
    # (delta_col_at_z is monotone increasing in z) -- a handful of
    # iterations suffices since we already have a good domega estimate and
    # only need a z precise enough that min_dz-scale steps aren't lost.
    lo, hi = z_cur, z_ceiling
    for _ in range(20):
        if hi - lo <= min_dz:
            break
        mid = 0.5 * (lo + hi)
        if pch._delta_col_at_z(mid) <= delta_target:
            lo = mid
        else:
            hi = mid

    z_next = max(lo, z_cur + min_dz)
    z_next = min(z_next, z_ceiling)
    delta_next = pch._delta_col_at_z(z_next)

    # Safety net: the domega->z inversion above assumes delta_col_at_z is
    # well-behaved (it is, in practice), but re-check Nupper at the chosen
    # z_next in case of pathological warm-start behaviour (e.g. Nupper(0)
    # already resolved to a value >1e-300 but the secant loop above still
    # didn't converge within tolerance) -- fall back to full bisection on z
    # directly against target_nupper if so.
    Nupper_check = _nupper_at(pch, M_cur, M_res, delta_0, delta_next - delta_0)
    if Nupper_check > 1.5 * target_nupper:
        lo, hi = z_cur, z_next
        for _ in range(40):
            if hi - lo <= min_dz:
                break
            mid = 0.5 * (lo + hi)
            delta_mid = pch._delta_col_at_z(mid)
            if _nupper_at(pch, M_cur, M_res, delta_0, delta_mid - delta_0) <= target_nupper:
                lo = mid
            else:
                hi = mid
        z_next = max(lo, z_cur + min_dz)
        z_next = min(z_next, z_ceiling)
        delta_next = pch._delta_col_at_z(z_next)

    return z_next, delta_next


def build_tree_pch08_adaptive(pch, M0, z0, z_max, M_res, target_nupper=0.1, dz_max=0.5, min_dz=1e-4, max_steps=2_000_000):
    """Adaptive-dz replacement for PCHMergerTree.build_tree: same return
    convention (list of {"redshift", "parent_mass", "progenitors"} dicts,
    increasing z), but each step's size is chosen (via _pick_adaptive_step)
    so Nupper stays <= target_nupper, honoring PCH08's own "P << 1" design
    (target ~0.1) instead of using one fixed dz for the whole tree."""
    pch._ensure_delta_col_covers(z_max)

    tree = []
    z_cur = z0
    M_cur = M0
    delta_cur = pch._delta_col_at_z(z_cur)
    n_steps = 0

    while z_cur < z_max - 1e-9 and M_cur >= M_res:
        n_steps += 1
        if n_steps > max_steps:
            raise RuntimeError(
                f"build_tree_pch08_adaptive exceeded max_steps={max_steps} "
                f"(M_cur={M_cur:.3e}, z_cur={z_cur:.4f}) -- target_nupper "
                "too small relative to min_dz for this M_res, or a bug."
            )
        z_ceiling = min(z_cur + dz_max, z_max)
        z_next, delta_next = _pick_adaptive_step(pch, z_cur, delta_cur, z_ceiling, M_cur, M_res, target_nupper, min_dz)

        sigma_M0 = pch._sigma_at_mass(M_cur)
        progenitors = pch.draw_progenitor_masses(M_cur, delta_cur, delta_next, sigma_M0, M_res)

        if progenitors:
            tree.append({"redshift": z_next, "parent_mass": M_cur, "progenitors": progenitors})
            M_cur = max(progenitors)

        z_cur, delta_cur = z_next, delta_next

    return tree


def _pch08_mainbranch_job(args):
    M0, z0, z_max, M_res, dz, f_major, seed, target_nupper = args
    np.random.seed(seed)
    pch = _worker_state["pch"]
    if target_nupper is not None:
        tree = build_tree_pch08_adaptive(pch, M0, z0, z_max, M_res, target_nupper=target_nupper, dz_max=dz)
    else:
        tree = pch.build_tree(M0, z0, z_max, M_res, dz=dz)
    return most_recent_major_merger_z(tree, f_major)


def _zh_mainbranch_job(args):
    M0, z0, z_max, M_res, dz, model, N_grid, S_max_factor, f_major, seed = args
    from foraois.zhang_hui_trees import ZhangHuiMergerTree

    cosmo_data = _worker_state["cosmo_data"]
    run_params = _worker_state["run_params"]
    rng = np.random.default_rng(seed)
    zh = ZhangHuiMergerTree(cosmo_data, run_params, model=model, rng=rng, N_grid=N_grid, S_max_factor=S_max_factor)
    tree = zh.build_tree(M0, z0, z_max, M_res, dz=dz)
    return most_recent_major_merger_z(tree, f_major)


def _worker_init_with_params(config_path):
    """Like _worker_init, but also stashes run_params (needed to build a
    ZhangHuiMergerTree, which _worker_init's plain PCH08-focused version
    does not keep around)."""
    from foraois.cosmo_utils import CosmoData
    from foraois.pch_trees import PCHMergerTree
    from foraois.utils import io as foraois_io

    import warnings

    warnings.simplefilter("ignore", UserWarning)

    run_params = foraois_io.get_params(config_path)
    cosmo_data = CosmoData(run_params, redshift=[0.0])
    pch = PCHMergerTree(cosmo_data, run_params)
    _worker_state["cosmo_data"] = cosmo_data
    _worker_state["run_params"] = run_params
    _worker_state["pch"] = pch


def run_pch08_mainbranch_ensemble(config_path, M0, z0, z_max, M_res, dz, f_major, n_trees, n_jobs=1, seed0=0, label="PCH08", target_nupper=None):
    """Grow n_trees independent main-progenitor-only PCH08 trees; return a
    list of n_trees major-merger redshifts (None where none qualified).

    If target_nupper is given, uses PCHMergerTree.major_merger_redshifts_
    numba_adaptive when numba is installed -- a JIT + prange-parallel
    kernel that walks the adaptive-dz main branch directly (dz becomes the
    *maximum* step size, dz_max) instead of PCHMergerTree.build_tree's
    fixed dz, without materializing a step-by-step tree list -- see that
    method's own docstring for the benchmarked speedup (many-hundred-x at
    PCH08's own target_nupper=0.1, the expensive case
    fig3_major_merger_redshift.py's own default hits). Run as one
    single-process batched call, same reasoning as run_pch08_ensemble's own
    numba dispatch (stacking multiprocessing on top would oversubscribe
    cores for no benefit). n_jobs/seed0 are then unused; they still apply
    to the target_nupper=None (fixed-dz, numba-unsupported) and
    numba-not-installed fallback paths below, both spread across n_jobs
    worker processes exactly as before (via build_tree_pch08_adaptive/
    _pch08_mainbranch_job, still pure Python -- no progress reporting is
    needed on the numba path since it typically finishes before the first
    ~10s progress line would even print)."""
    from foraois.pch_trees import _HAVE_NUMBA

    if target_nupper is not None and _HAVE_NUMBA:
        _worker_init_with_params(config_path)
        pch = _worker_state["pch"]
        return pch.major_merger_redshifts_numba_adaptive(M0, z0, z_max, M_res, f_major, n_trees, target_nupper=target_nupper, dz_max=dz)

    jobs = [(M0, z0, z_max, M_res, dz, f_major, seed0 + i, target_nupper) for i in range(n_trees)]
    if n_jobs <= 1:
        _worker_init_with_params(config_path)
    ctx = mp.get_context("spawn")
    return _progress_map(_pch08_mainbranch_job, jobs, label, n_jobs=n_jobs, pool_ctx=ctx, initializer=_worker_init_with_params, initargs=(config_path,))


def major_merger_redshifts_zh_batch(cosmo_data, run_params, M0, z0, z_max, M_res, dz, f_major, n_trees, model="cdm"):
    """Batch-compute n_trees Zhang-Hui main-progenitor major-merger
    redshifts at once, using the already-existing closed-form flat-barrier
    forest builder (ZhangHuiMergerTree.build_forest_numba, falling back to
    build_forest_numpy if numba isn't installed) instead of looping
    ZhangHuiMergerTree.build_tree's general Volterra-solver path
    (draw_progenitor_mass_zh -> solve_first_crossing) per tree -- the same
    O(N_grid)-quad-calls-per-branch-per-step cost identified in
    draw_progenitor_mass_zh_flat's own docstring, which build_tree (unlike
    grow_full_population_zh_flat) was never routed around.

    mass_history[i, j] is the larger of the two split fragments at step j
    (or the sole continuing mass if no split that step); merger_mass[i, j]
    is the smaller fragment (0 if no split) -- together they give the
    split ratio directly (merger_mass/mass_history), without needing the
    full per-step progenitor-list scan most_recent_major_merger_z does.
    Scans columns in increasing-z order so the first (lowest-z) qualifying
    split recorded per tree matches most_recent_major_merger_z's own
    "first entry" convention.

    Returns None (not a list) if model isn't flat-barrier -- the caller
    should fall back to the general per-tree build_tree path in that case,
    same "check once, dispatch" pattern as grow_full_population_zh_batch."""
    from foraois.zhang_hui_trees import ZhangHuiMergerTree, _HAVE_NUMBA, _assert_flat_barrier

    try:
        _assert_flat_barrier(model, z_max, cosmo_data)
    except NotImplementedError:
        return None

    zh = ZhangHuiMergerTree(cosmo_data, run_params, model=model)
    M0_array = np.full(int(n_trees), float(M0), dtype=np.float64)

    if _HAVE_NUMBA:
        mass_history, z_steps, _smooth_accretion, merger_mass = zh.build_forest_numba(M0_array, z0, z_max, M_res, dz=dz)
    else:
        mass_history, _split_events, z_steps, _smooth_accretion, merger_mass = zh.build_forest_numpy(M0_array, z0, z_max, M_res, dz=dz)

    N, n_steps = mass_history.shape
    results = [None] * N
    for j in range(n_steps):
        mh = mass_history[:, j]
        mm = merger_mass[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(mh > 0, mm / mh, 0.0)
        hit = (mm > 0) & (ratio >= f_major)
        if not np.any(hit):
            continue
        z_next = float(z_steps[j + 1])
        for i in np.nonzero(hit)[0]:
            if results[i] is None:
                results[i] = z_next
    return results


def run_zh_mainbranch_ensemble(config_path, M0, z0, z_max, M_res, dz, f_major, n_trees, model="cdm", N_grid=40, S_max_factor=8.0, n_jobs=1, seed0=0, label="Zhang-Hui"):
    """Zhang-Hui equivalent of run_pch08_mainbranch_ensemble. For a
    flat-barrier model, dispatches to major_merger_redshifts_zh_batch --
    the already-existing closed-form forest builder, run as one batched
    call -- instead of looping the general Volterra-solver build_tree per
    tree across n_jobs worker processes (this general path is the actual
    bottleneck fig3_major_merger_redshift.py's own Zhang-Hui column hits;
    see major_merger_redshifts_zh_batch's own docstring). Falls back to
    the general per-tree path (spread across n_jobs worker processes, as
    before) for a non-flat-barrier model."""
    _worker_init_with_params(config_path)
    cosmo_data = _worker_state["cosmo_data"]
    run_params = _worker_state["run_params"]

    batch_result = major_merger_redshifts_zh_batch(cosmo_data, run_params, M0, z0, z_max, M_res, dz, f_major, n_trees, model=model)
    if batch_result is not None:
        return batch_result

    jobs = [(M0, z0, z_max, M_res, dz, model, N_grid, S_max_factor, f_major, seed0 + i) for i in range(n_trees)]
    ctx = mp.get_context("spawn")
    return _progress_map(_zh_mainbranch_job, jobs, label, n_jobs=n_jobs, pool_ctx=ctx, initializer=_worker_init_with_params, initargs=(config_path,))


def _checkpoint_indices(z_steps, checkpoints, atol=1e-8):
    """Map each requested checkpoint redshift to its nearest z_steps grid index."""
    idx = {}
    for zc in checkpoints:
        i = int(np.argmin(np.abs(z_steps - zc)))
        if abs(z_steps[i] - zc) > max(atol, 1e-3):
            raise ValueError(
                f"checkpoint z={zc} is not on the dz grid (nearest grid point is "
                f"z={z_steps[i]:.4f}) -- choose dz so that z0 + n*dz hits every "
                f"checkpoint exactly, e.g. dz=0.05 with checkpoints at 0.5,1,2,4."
            )
        idx[zc] = i
    return idx


def grow_full_population_pch08(pch, M0, z0, z_max, M_res, dz, checkpoints):
    """
    Grow every branch of a single PCH08 tree from z0 to z_max on a fixed
    dz grid, recording the full list of active branch masses at each
    requested checkpoint redshift.

    Parameters
    ----------
    pch : foraois.pch_trees.PCHMergerTree
    M0, z0, z_max, M_res, dz : as in PCHMergerTree.build_tree
    checkpoints : sequence of float
        Redshifts at which to snapshot the branch population; must each
        land exactly on the z0 + n*dz grid (to within 1e-3).

    Returns
    -------
    dict[float, list[float]]
        checkpoint redshift -> list of branch masses (Msun/h) alive at
        that redshift (i.e. not yet fallen below M_res).
    """
    pch._ensure_delta_col_covers(z_max)
    z_steps = np.arange(z0, z_max + dz * 0.5, dz)
    cp_idx = _checkpoint_indices(z_steps, checkpoints)

    masses = [float(M0)]
    result = {}
    delta_0 = pch._delta_col_at_z(z_steps[0])

    for k in range(len(z_steps) - 1):
        delta_1 = pch._delta_col_at_z(z_steps[k + 1])
        next_masses = []
        for m in masses:
            if m < M_res:
                continue
            sigma_m = pch._sigma_at_mass(m)
            progenitors = pch.draw_progenitor_masses(m, delta_0, delta_1, sigma_m, M_res)
            next_masses.extend(progenitors)
        masses = next_masses
        delta_0 = delta_1

        for zc, idx in cp_idx.items():
            if idx == k + 1:
                result[zc] = list(masses)

    return result


def grow_full_population_zh(
    cosmo_data,
    M0,
    z0,
    z_max,
    M_res,
    dz,
    checkpoints,
    model="cdm",
    rng=None,
    N_grid=40,
    S_max_factor=8.0,
):
    """
    Zhang-Hui equivalent of grow_full_population_pch08 -- same dz grid,
    same bookkeeping, but each branch's step uses the exact
    draw_progenitor_mass_zh rate instead of PCH08's fitted one.

    Substantially more expensive per branch/step than the PCH08 version
    (each call solves a Volterra system of size N_grid rather than
    evaluating a closed-form rate) -- keep N_grid modest (30-60) and
    prefer fewer, larger M_res-to-M0 dynamic ranges (fewer branches) when
    exploring interactively; see each fig script's runtime notes.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    z_steps = np.arange(z0, z_max + dz * 0.5, dz)
    cp_idx = _checkpoint_indices(z_steps, checkpoints)

    masses = [float(M0)]
    result = {}

    for k in range(len(z_steps) - 1):
        z_cur, z_next = z_steps[k], z_steps[k + 1]
        next_masses = []
        for m in masses:
            if m < M_res:
                continue
            progenitors = draw_progenitor_mass_zh(
                m, z_cur, z_next, M_res, cosmo_data, model=model, rng=rng, N_grid=N_grid, S_max_factor=S_max_factor
            )
            next_masses.extend(progenitors)
        masses = next_masses

        for zc, idx in cp_idx.items():
            if idx == k + 1:
                result[zc] = list(masses)

    return result


def grow_full_population_zh_flat(
    cosmo_data,
    M0,
    z0,
    z_max,
    M_res,
    dz,
    checkpoints,
    model="cdm",
    rng=None,
):
    """
    Closed-form-flat-barrier equivalent of grow_full_population_zh: same
    dz grid, same bookkeeping, but each branch's step uses
    draw_progenitor_mass_zh_flat (exact Levy-distribution sampling, no
    solve_first_crossing Volterra solve) instead of draw_progenitor_mass_zh.
    Only valid for flat-barrier models (cdm/wdm/fdm's placeholder) --
    callers should check _assert_flat_barrier once (as
    grow_full_population_zh_batch does) rather than per call.

    Eliminates Zhang-Hui full-population growth's dominant cost (an
    O(N_grid) loop of scipy.integrate.quad calls per branch per step, see
    draw_progenitor_mass_zh_flat's own docstring) -- benchmarked in this
    repo's sandbox at ~1-3ms/branch-step here vs ~15-40ms/branch-step for
    the general grow_full_population_zh at N_grid=40, a ~15-20x per-step
    speedup that compounds with every branch/step in a full population."""
    if rng is None:
        rng = np.random.default_rng(0)

    z_steps = np.arange(z0, z_max + dz * 0.5, dz)
    cp_idx = _checkpoint_indices(z_steps, checkpoints)

    masses = [float(M0)]
    result = {}

    for k in range(len(z_steps) - 1):
        z_cur, z_next = z_steps[k], z_steps[k + 1]
        next_masses = []
        for m in masses:
            if m < M_res:
                continue
            progenitors = draw_progenitor_mass_zh_flat(m, z_cur, z_next, M_res, cosmo_data, model=model, rng=rng)
            next_masses.extend(progenitors)
        masses = next_masses

        for zc, idx in cp_idx.items():
            if idx == k + 1:
                result[zc] = list(masses)

    return result


def grow_full_population_zh_single(cosmo_data, M0, z0, z_max, M_res, dz, checkpoints, model="cdm", rng=None, N_grid=40, S_max_factor=8.0):
    """Single-tree dispatch: uses grow_full_population_zh_flat (closed-form,
    ~15-20x faster per branch-step -- see its own docstring) for
    flat-barrier models, checked once via _assert_flat_barrier; falls back
    to the general grow_full_population_zh (solve_first_crossing) for any
    model that isn't flat (not currently reachable for cdm/wdm/fdm, but
    kept so this stays correct if a genuinely mass-dependent barrier is
    ever added -- see ROADMAP.md). See grow_full_population_zh_batch for
    the n_trees-at-once version (uses the numba kernel when available;
    this single-tree function stays pure Python and is what that batch
    function falls back to when numba isn't installed)."""
    try:
        _assert_flat_barrier(model, z_max, cosmo_data)
        flat = True
    except NotImplementedError:
        flat = False

    if flat:
        return grow_full_population_zh_flat(cosmo_data, M0, z0, z_max, M_res, dz, checkpoints, model=model, rng=rng)
    return grow_full_population_zh(cosmo_data, M0, z0, z_max, M_res, dz, checkpoints, model=model, rng=rng, N_grid=N_grid, S_max_factor=S_max_factor)


def grow_full_population_zh_batch(cosmo_data, M0, z0, z_max, M_res, dz, checkpoints, n_trees, model="cdm", rng=None, N_grid=40, S_max_factor=8.0, max_stack=20_000, max_out=4_000):
    """Grow n_trees independent Zhang-Hui full-branch-population
    realizations at once. Dispatch order:

    1. Flat-barrier model + numba installed: ZhangHuiMergerTree.
       grow_full_population_numba -- JIT + prange-parallel *across trees
       itself* (no solve_first_crossing at all, exact closed-form Levy
       sampling -- see that method's own docstring), run as one
       single-process batched call.
    2. Flat-barrier, no numba: grow_full_population_zh_flat looped
       n_trees times (still the closed-form math, ~15-20x faster per
       branch-step than the general solver, just not JIT-compiled).
    3. Not flat-barrier: grow_full_population_zh looped n_trees times
       (the general numerical solver -- only reachable once a genuinely
       mass-dependent barrier model exists; see ROADMAP.md).

    Returns a list of n_trees {checkpoint_z: [branch masses]} dicts, same
    convention as grow_full_population_zh(_flat), so this is a drop-in
    replacement for a `[grow_full_population_zh_single(...) for _ in
    range(n_trees)]` loop anywhere in this module."""
    from foraois.zhang_hui_trees import _HAVE_NUMBA, ZhangHuiMergerTree

    try:
        _assert_flat_barrier(model, z_max, cosmo_data)
        flat = True
    except NotImplementedError:
        flat = False

    if flat and _HAVE_NUMBA:
        zh = ZhangHuiMergerTree(cosmo_data, model=model)
        return zh.grow_full_population_numba(M0, z0, z_max, M_res, checkpoints, n_trees=n_trees, dz=dz, max_stack=max_stack, max_out=max_out)

    return [
        grow_full_population_zh_single(cosmo_data, M0, z0, z_max, M_res, dz, checkpoints, model=model, rng=rng, N_grid=N_grid, S_max_factor=S_max_factor)
        for _ in range(n_trees)
    ]


def grow_full_population_pch08_adaptive(pch, M0, z0, z_max, M_res, checkpoints, target_nupper=0.5, dz_max=0.5, min_dz=1e-4, max_steps_per_branch=500_000):
    """Adaptive-dz replacement for grow_full_population_pch08: same return
    convention ({checkpoint_z: [branch masses]}), but each branch's step
    size is chosen independently (via _pick_adaptive_step) so its own
    Nupper stays <= target_nupper, instead of every branch sharing one
    fixed dz grid -- see this module's 'Adaptive-dz PCH08 stepping'
    section for why fixed dz badly violates PCH08's own <<1 design target
    at the M_res/M0 ratios fig1 uses (Nupper of order 1e2-1e3, not the
    handful of steps a modestly-finer fixed dz could absorb the way it did
    for the Section-5.4 main-progenitor-only benchmark).

    Implemented as an explicit stack of (mass, z, delta) branches rather
    than grow_full_population_pch08's lock-step "advance every branch by
    one shared dz" loop, because branches now take different-sized steps
    -- a branch is popped, walked forward on its own adaptive grid until
    it either dies, reaches z_max, or reaches a split, and a split pushes
    its second fragment as a new stack entry to be walked independently
    later (the first fragment continues inline). Checkpoints are still hit
    exactly for every branch (each branch's per-step ceiling is capped by
    the next unreached checkpoint), so results remain directly comparable
    to the fixed-dz version's checkpointed snapshots."""
    pch._ensure_delta_col_covers(z_max)
    checkpoints_sorted = sorted(set(checkpoints))
    result = {zc: [] for zc in checkpoints_sorted}

    delta0 = pch._delta_col_at_z(z0)
    stack = [(float(M0), z0, delta0)]

    while stack:
        M_cur, z_cur, delta_cur = stack.pop()
        n_steps = 0
        while M_cur >= M_res and z_cur < z_max - 1e-9:
            n_steps += 1
            if n_steps > max_steps_per_branch:
                raise RuntimeError(
                    f"grow_full_population_pch08_adaptive: a branch exceeded "
                    f"max_steps_per_branch={max_steps_per_branch} (M_cur={M_cur:.3e}, "
                    f"z_cur={z_cur:.4f}) -- target_nupper too small relative to min_dz, or a bug."
                )
            next_cp = next((zc for zc in checkpoints_sorted if zc > z_cur + 1e-9), None)
            ceiling = z_max if next_cp is None else next_cp
            z_step_ceiling = min(z_cur + dz_max, ceiling)

            z_next, delta_next = _pick_adaptive_step(pch, z_cur, delta_cur, z_step_ceiling, M_cur, M_res, target_nupper, min_dz)

            sigma_m = pch._sigma_at_mass(M_cur)
            progenitors = pch.draw_progenitor_masses(M_cur, delta_cur, delta_next, sigma_m, M_res)

            landed_cp = next((zc for zc in checkpoints_sorted if abs(zc - z_next) < 1e-6), None)

            if not progenitors:
                M_cur = 0.0
                break

            if len(progenitors) == 2:
                stack.append((progenitors[1], z_next, delta_next))
                if landed_cp is not None:
                    result[landed_cp].append(progenitors[1])

            M_cur = progenitors[0]
            z_cur, delta_cur = z_next, delta_next
            if landed_cp is not None:
                result[landed_cp].append(M_cur)

    return result


def grow_full_population_pch08_batch(
    pch, M0, z0, z_max, M_res, checkpoints, n_trees, target_nupper=0.5, dz_max=0.5, min_dz=1e-4,
    eps1=0.1, max_stack=20_000, max_out=4_000,
):
    """Grow n_trees independent PCH08 full-branch-population realizations at
    once, using PCHMergerTree.grow_full_population_numba_adaptive (JIT +
    prange-parallel over trees) when numba is installed -- ~400-600x faster
    than looping grow_full_population_pch08_adaptive per tree (see that
    numba method's own docstring for the benchmarked speedup; confirmed in
    this repo's sandbox: ~6.9s/tree serial vs ~0.012s/tree warm-numba at a
    comparable configuration). Falls back to the serial per-tree Python
    loop -- functionally identical, just slow -- if numba is not installed,
    so this is always safe to call.

    Returns a list of n_trees {checkpoint_z: [branch masses]} dicts, same
    convention as grow_full_population_pch08(_adaptive), so this is a
    drop-in replacement for a `[grow_full_population_pch08_adaptive(...)
    for _ in range(n_trees)]` loop anywhere in this module."""
    from foraois.pch_trees import _HAVE_NUMBA

    if _HAVE_NUMBA:
        return pch.grow_full_population_numba_adaptive(
            M0, z0, z_max, M_res, checkpoints, n_trees=n_trees,
            target_nupper=target_nupper, dz_max=dz_max, min_dz=min_dz, eps1=eps1,
            max_stack=max_stack, max_out=max_out,
        )
    return [
        grow_full_population_pch08_adaptive(pch, M0, z0, z_max, M_res, checkpoints, target_nupper=target_nupper, dz_max=dz_max, min_dz=min_dz)
        for _ in range(n_trees)
    ]


def cmf_histogram(all_masses_by_realization, M2, n_bins=20, log_range=(-4.5, 0.05), min_count=30):
    """
    Build a PCH08-Fig-1-style conditional mass function: log10(f_cmf) vs
    log10(M1/M2), where f_cmf per bin is the total mass in that bin's
    progenitors, summed over all realizations, divided by (M2 * n_realizations
    * bin width in ln(M1/M2)) -- i.e. dF/dlnM1, the mass-weighted analogue
    of a number-weighted histogram, matching PCH08's own f_cmf definition
    (Eq. 1: a mass FRACTION density, not a number density).

    Being mass-weighted, this estimator has a heavy right tail: near
    M1~M2, a single tree that happens to retain almost all of M2's mass in
    one branch dominates that bin's sum outright, so a bin's value can
    swing by an order of magnitude between otherwise-equivalent runs even
    at large --n-trees (confirmed empirically: at n_trees=2000, M2=1e12,
    z1=4, PCH08 has just 8 progenitors landing in the second-to-last
    nonempty bin and 0 beyond it, while a 10x increase in n_trees moved
    that bin's value further rather than converging it). min_count masks
    (to NaN) any bin whose raw progenitor COUNT (not mass) falls below
    this threshold, so a caller doesn't plot a curve segment set by a
    handful of rare events as if it meant something -- this was previously
    only a docstring claim ("restrict to bins with >=30 pooled progenitor
    counts") with nothing in the code actually enforcing it.

    Parameters
    ----------
    all_masses_by_realization : list[list[float]]
        One inner list of progenitor masses per realization (tree).
    M2 : float
        The descendant mass all realizations were grown from.
    n_bins, log_range : histogram binning in log10(M1/M2).
    min_count : int
        Bins with fewer than this many pooled progenitors are masked to
        NaN, regardless of how much mass they happen to carry. Pass 0 to
        disable masking (the previous behaviour, masking only truly empty
        bins).

    Returns
    -------
    bin_centers, log10_f_cmf, counts : arrays (log10_f_cmf is -inf-safe:
        empty or count-starved bins are masked to NaN rather than plotted
        as -inf or as a rare-event-dominated value; counts is the raw,
        unmasked per-bin progenitor count, for callers that want to report
        or further filter on it).
    """
    all_masses = np.concatenate([np.asarray(m) for m in all_masses_by_realization if len(m)])
    log_ratio = np.log10(all_masses / M2)

    edges = np.linspace(log_range[0], log_range[1], n_bins + 1)
    mass_per_bin, _ = np.histogram(all_masses, bins=M2 * 10 ** edges, weights=all_masses)
    counts, _ = np.histogram(log_ratio, bins=edges)

    n_real = len(all_masses_by_realization)
    d_ln_ratio = (edges[1] - edges[0]) * np.log(10.0)
    f_cmf = mass_per_bin / (M2 * n_real * d_ln_ratio)

    centers = 0.5 * (edges[:-1] + edges[1:])
    with np.errstate(divide="ignore"):
        log_f_cmf = np.log10(f_cmf)
    log_f_cmf[counts < max(min_count, 1)] = np.nan
    return centers, log_f_cmf, counts


def eps_analytic_cmf(M2, z2, z1, cosmo_data, log_ratio_grid):
    """
    The unmodified-EPS conditional mass function (Eq. 1 of PCH08, i.e.
    G=1 in Eq. 6/7), evaluated analytically (no Monte Carlo) at each
    point of log_ratio_grid = log10(M1/M2), for reference -- shows how far
    PCH08's empirical correction and the exact Zhang-Hui rate each pull
    the raw EPS prediction, which the paper's own Fig 1 does not show
    directly (it only shows GALFORM/New-Trees/N-body).
    """
    M1 = M2 * 10 ** log_ratio_grid
    sigma1 = cosmo_data.sigma_at_logmass(np.log10(M1))
    sigma2 = cosmo_data.sigma_at_logmass(np.log10(M2))
    delta1 = cosmo_data.delta_col_at_z(z1)
    delta2 = cosmo_data.delta_col_at_z(z2)

    dS = sigma1**2 - sigma2**2
    dS = np.where(dS > 0, dS, np.nan)
    dbarrier = delta1 - delta2

    # EPS conditional distribution dN/dlnM1 * M1/M2 (a mass-fraction
    # density in ln(M1/M2)), i.e. Eq. 1 with the |dlnsigma/dlnM1| Jacobian
    # folded in (dlogsigma_at_logmass already returns the absolute value).
    dlnsigma_dlnM = cosmo_data.dlogsigma_at_logmass(np.log10(M1))

    f = (
        np.sqrt(2.0 / np.pi)
        * sigma1**2
        * dbarrier
        / dS**1.5
        * np.exp(-0.5 * dbarrier**2 / dS)
        * np.abs(dlnsigma_dlnM)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log10(f)


# Sheth-Tormen (2002) mass-function fit parameters, as quoted in PCH08's own
# Fig 4 caption.
ST_A = 0.322
ST_A_SMALL = 0.707
ST_P = 0.3


def sheth_tormen_nu_f_nu(nu, A=ST_A, a=ST_A_SMALL, p=ST_P):
    """
    nu*f(nu) in the Sheth & Tormen (2002) normalisation, i.e. the quantity
    whose integral over dln(nu) is unity -- equivalently, the fraction of
    mass per dln(nu) locked up in haloes of that peak height. This is
    exactly the f_ST(nu) formula quoted in PCH08 Fig. 4's caption.
    """
    return A * np.sqrt(2.0 * a / np.pi) * (1.0 + (1.0 / (a * nu**2)) ** p) * nu * np.exp(-0.5 * a * nu**2)


def sheth_tormen_dn_dlogm(M, z, cosmo_data, A=ST_A, a=ST_A_SMALL, p=ST_P):
    """
    Sheth-Tormen comoving number density per unit log10(M), dn/dlog10(M),
    at redshift z, built from nu*f(nu) via the standard
        dn/dlnM = (rho_mean/M) * [nu f(nu)] * |dlnsigma/dlnM|
    identity (nu = delta_c(z)/sigma(M), rho_mean = OmegaM * rho_crit,0).
    Units follow cosmo_data's own convention (M in Msun/h, rho_crit0 in
    Msun/Mpc^3/h^2), so this is only meant to be compared internally
    against Monte Carlo tree ensembles built with the same cosmo_data
    instance -- not against an externally-quoted number density.
    """
    logM = np.log10(M)
    sigma = cosmo_data.sigma_at_logmass(logM)
    delta_c = cosmo_data.delta_col_at_z(z)
    nu = delta_c / sigma
    nuf = sheth_tormen_nu_f_nu(nu, A, a, p)
    rho_mean = cosmo_data.cosmo_params["OmegaM"] * cosmo_data.rhocrit0
    dlnsigma_dlnM = cosmo_data.dlogsigma_at_logmass(logM)
    dn_dlnM = (rho_mean / M) * nuf * dlnsigma_dlnM
    return dn_dlnM * np.log(10.0)  # dn/dlnM -> dn/dlog10M
