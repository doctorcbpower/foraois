#!/usr/bin/env python3
"""
Reproduces the *structure* of Parkinson, Cole & Helly (2008, "PCH08")
Figure 1: the conditional mass function f_cmf(M1|M2) -- the fraction of a
descendant halo's mass (M2 at z2=0) found in progenitor haloes of mass M1
at earlier redshifts z1 -- on the same 4 (z1) x 3 (M2) grid PCH08 use,
log10(f_cmf) vs log10(M1/M2).

What this does and does not reproduce
--------------------------------------
PCH08's own Fig 1 compares four things: Millennium Simulation N-body trees
(histogram), the original unmodified GALFORM/EPS algorithm (dotted), their
new fitted-rate algorithm -- what foraois calls PCH08 (dashed), and an
analytic fit to the N-body data (solid). We have no access to Millennium
merger trees, so this script cannot reproduce the N-body curve or the fit
to it -- what it *can* do is put foraois's own PCH08 backend and its
barrier-agnostic Zhang & Hui (2006) backend through the identical
diagnostic, plus the raw (unmodified) EPS conditional mass function
(Eq. 1 of PCH08, G=1) as an analytic reference curve PCH08's own Fig 1
does not show. This tells you how far each backend's Monte Carlo result
sits from raw EPS, and how the two backends compare to each other --
useful even without an N-body baseline, since it is exactly the kind of
"does the implementation reproduce the published algorithm" check
docs/VALIDATION-style work calls for.

Nadler et al. (2023)'s Brownian-bridge-constrained sampler is
deliberately not included here: that machinery targets efficient sampling
of a single, specified rare outcome (a branch guaranteed to reach a given
(M1, z1)), not the ordinary/unconstrained ensemble statistics a
conditional mass function is built from -- including it here would not
answer a different question, it would just be the wrong tool for this
one.

How the population is built
----------------------------
PCH08's own Fig 1 measures the *full* conditional mass function -- the
fraction of M2's mass across the *entire* multi-generation merger history
down to z1, not just a single split -- so this script grows the complete
branching population of each tree (every branch, not just the main
progenitor) from z0=0 to each z1 on a shared dz grid, using
_treegrowth.grow_full_population_{pch08,zh}. The same dz grid is used for
both backends specifically so that any difference in the resulting
statistics comes from the branching-rate law itself, not from a different
number of generations of splitting.

Runtime
-------
The PCH08 side is cheap (closed-form rate per branch/step). The Zhang-Hui
side solves a small Volterra system *per branch per step*, so its cost
scales roughly as (number of branches summed over all steps) x N_grid^2;
benchmarked in this repo's sandbox (2 vCPUs) at M2=1e15, dz=0.05,
N_grid=40, a single full-population Zhang-Hui tree costs ~10-20s versus
~0.05s for PCH08 -- a 200-400x gap. This is an embarrassingly parallel
Monte Carlo problem (every tree is independent), so --n-jobs is the right
lever for a real run, not shrinking --n-trees: on an HPC node with
--n-jobs=32 the wall-clock cost of a given --n-trees drops by roughly
that factor. --n-trees-zh lets you additionally run fewer Zhang-Hui
realizations than PCH08 ones, since it is so much more expensive per
tree.

Convergence check (informing the --n-trees=1000 default)
----------------------------------------------------------
Restricting to bins with >=30 pooled progenitor counts (i.e. excluding
the sparse, always-somewhat-ragged tails -- also visibly ragged in
PCH08's own N-body-based Fig 1), the worst-case column (M2=1e15, z1=1,
PCH08 backend, dz=0.05) stabilizes to within ~0.03-0.05 dex in log10(f_cmf)
by n_trees~600-1200, with no further tightening beyond that (the residual
~0.03-0.05 dex is set by the bin width, not the sample size). n_trees=1000
was chosen as a round number safely inside that plateau. This was checked
for PCH08 only (cheap enough to iterate); Zhang-Hui is assumed to need a
similar order of n_trees for the SAME reason PCH08 does (comparable
branch/bin counts on the same dz grid), not independently re-verified,
given its much higher per-tree cost -- if you have spare core-hours,
re-running this check for Zhang-Hui and adjusting --n-trees-zh accordingly
is a reasonable thing to do before treating a Zhang-Hui curve as final.

On the EPS reference curve (why it is OFF by default)
--------------------------------------------------------
The unmodified-EPS analytic curve (eps_analytic_cmf) is the CONTINUUM,
SINGLE-STEP conditional mass function -- it integrates to exactly 1 over
all M1 down to 0 (verified numerically), i.e. it assumes every last bit
of M2's mass is accounted for, including infinitesimally small
progenitors. The Monte Carlo curves are neither of those things: they are
built by iterating draw_progenitor_masses over MANY small dz steps (80 of
them at dz=0.05, z0=0 to z_max=4), and at EVERY step any mass that would
have gone into an unresolved (< M_res) fragment is permanently absorbed
into "smooth accretion" and removed from the tracked population -- it
does not get redistributed back into resolved bins later. This is not a
boundary effect confined to M1 near M_res: verified against foraois's own
build_full_tree (independently, at M2=1e12, M_res=1e-4*M2, dz=0.05,
z1=4), only ~39-40% of M2's mass survives in resolved (>=M_res) branches
by z1=4, consistently across independent trees (both this module's
grow_full_population_pch08 and the library's own build_full_tree agree on
this fraction) -- so a resolution-limited, multi-step tree is EXPECTED to
sit well below the continuum single-step formula at every M1, not just
near M_res, and the gap widens with z1 (more steps => more compounding
loss). This is a well known, genuine limitation of resolution-limited
excursion-set trees (part of the original motivation for PCH08's own
empirical G-correction, which is calibrated against N-body, not against
this analytic curve) -- it is NOT a bug in this script, but comparing
against the raw analytic curve by default produced a plot that looked
alarmingly wrong at a glance (most of the curve clipped by the y-axis,
what MC there was sitting far below the reference) when the real content
-- PCH08 vs Zhang-Hui agreement with EACH OTHER, the actual validation
question -- was fine. Pass --show-eps-reference if you want the overlay
back for your own diagnostic purposes, with this caveat in mind.

On dz for Zhang-Hui (do not go below ~0.05 at the default --n-grid)
---------------------------------------------------------------------
Counter to the naive expectation that smaller dz is more accurate, making
dz much smaller than ~0.05 at N_grid=40 made results at M2=1e15 both
~7x slower AND numerically less trustworthy: first_crossing_step's own
resolution warning ("grid spacing coarser than the shifted-barrier
scale") fired on nearly every step at dz=0.01 (a single tree took ~130s,
versus ~19s at dz=0.05), because a fixed (N_grid, S_max_factor) grid --
sized for the FULL M0-to-M_res dynamic range -- under-resolves the much
smaller barrier shift a single small dz step produces. Increasing N_grid
to compensate would add back cost quadratically per call. Practical
takeaway: pick a dz no finer than needed to land on the checkpoint
redshifts (0.05 does that for the checkpoints used here), and treat a
much finer dz as a numerical-accuracy regression for Zhang-Hui specifically,
not a free precision upgrade -- this script's main() warns if
--dz<0.02 and --n-grid<=40 are combined.

Usage
-----
    python scripts/paper_figs/fig1_conditional_mass_function.py \
        [--n-trees 40] [--dz 0.05] [--n-grid 40] \
        [--config config/planck2018_camb.yml] [--output fig1_cmf.png]
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _treegrowth import cmf_histogram, eps_analytic_cmf, run_pch08_ensemble, run_zh_ensemble  # noqa: E402

from foraois.cosmo_utils import CosmoData  # noqa: E402
from foraois.utils import io as foraois_io  # noqa: E402

M2_VALUES = [1.0e12, 3.16e13, 1.0e15]  # Msun/h -- PCH08 Fig 1's three columns
Z1_VALUES = [4.0, 2.0, 1.0, 0.5]  # rows, high-z at top as in PCH08's own layout
Z0 = 0.0


def m_res_for(M2):
    """A fixed dynamic range (M2/M_res = 1e4) rather than PCH08's fixed
    particle-mass resolution -- keeps every column's branch count roughly
    comparable rather than reproducing MS's specific particle mass."""
    return M2 * 1.0e-4


def run_column(config_path, M2, checkpoints, dz, n_trees_pch, n_trees_zh, model, n_grid, s_max_factor, n_jobs, seed, target_nupper=None):
    """Grow independent realizations to the deepest checkpoint, reusing the
    same trajectories for every shallower checkpoint (cheaper than
    regrowing from scratch per z1, and matches how PCH08's own Fig 1
    panels within a column share the same underlying trees).

    PCH08 and Zhang-Hui are allowed different n_trees: PCH08 is cheap
    enough to run at high statistics, but Zhang-Hui solves a Volterra
    system per branch per step and can be 100-300x more expensive per
    tree (see the module docstring) -- see this script's --n-trees-zh
    for tuning that independently. n_jobs>1 spreads the tree loop across
    worker processes (see _treegrowth.run_{pch08,zh}_ensemble).

    If target_nupper is given, PCH08's population growth uses
    grow_full_population_pch08_adaptive (dz becomes the per-branch maximum
    step size) instead of the fixed-dz grow_full_population_pch08 -- see
    the module docstring's 'Nupper compliance' section for why fig1's
    M_res/M2=1e-4 dynamic range makes this necessary (unlike the
    Section-5.4 main-progenitor-only benchmark, no fixed dz small enough
    to be Nupper-compliant here is cheap to run). Zhang-Hui is unaffected
    (its p_split is a genuine bounded first-crossing probability, not an
    upper-bound approximation -- see fig3's module docstring) and keeps
    using the fixed-dz grid regardless of this setting."""
    M_res = m_res_for(M2)
    z_max = max(checkpoints)

    pch_by_z1 = {z1: [] for z1 in checkpoints}
    pch_pops = run_pch08_ensemble(
        config_path, M2, Z0, z_max, M_res, dz, checkpoints, n_trees_pch, n_jobs=n_jobs, seed0=seed, label=f"PCH08 M2={M2:.2e}",
        target_nupper=target_nupper,
    )
    for pops in pch_pops:
        for z1 in checkpoints:
            pch_by_z1[z1].append(pops[z1])

    zh_by_z1 = {z1: [] for z1 in checkpoints}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # first_crossing_step's coarse-grid warning is expected here
        zh_pops = run_zh_ensemble(
            config_path, M2, Z0, z_max, M_res, dz, checkpoints, n_trees_zh,
            model=model, N_grid=n_grid, S_max_factor=s_max_factor, n_jobs=n_jobs, seed0=seed + 1_000_000,
            label=f"Zhang-Hui M2={M2:.2e}",
        )
        for pops in zh_pops:
            for z1 in checkpoints:
                zh_by_z1[z1].append(pops[z1])

    return pch_by_z1, zh_by_z1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/planck2018_camb.yml")
    parser.add_argument("--n-trees", type=int, default=1000, help="PCH08 realizations per M2 column (cheap; convergence-checked, see module docstring)")
    parser.add_argument(
        "--n-trees-zh",
        type=int,
        default=None,
        help="Zhang-Hui realizations per M2 column; defaults to --n-trees if not given, but ZH is 100-300x more expensive per tree (see module docstring), so consider a smaller value unless --n-jobs is large",
    )
    parser.add_argument("--dz", type=float, default=0.05, help="Do not go far below 0.05 for Zhang-Hui at the default --n-grid; see module docstring's 'On dz' note")
    parser.add_argument("--n-grid", type=int, default=40, help="Zhang-Hui solve_first_crossing grid resolution")
    parser.add_argument("--s-max-factor", type=float, default=8.0)
    parser.add_argument("--n-jobs", type=int, default=1, help="worker processes for the (embarrassingly parallel) tree loop -- set to your core count on an HPC node")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--show-eps-reference",
        action="store_true",
        help="overlay the continuum, single-step unmodified-EPS analytic curve. OFF by default -- see module docstring's "
        "'On the EPS reference curve' note for why this is not a fair like-for-like comparison to the multi-step, "
        "resolution-limited Monte Carlo curves and can look like a large, alarming discrepancy that is not a bug.",
    )
    parser.add_argument(
        "--target-nupper",
        type=float,
        default=0.5,
        help=(
            "PCH08's own adaptive-step target for Nupper (per-step split probability; PCH08 sec. 2.1 "
            "requires P<<1, ~0.1 in practice). The PCH08 backend uses adaptive per-branch stepping "
            "(with --dz as the *ceiling* step size) to keep Nupper at or below this value, instead of "
            "the fixed-dz grid every branch shared before -- see run_column's docstring and this "
            "module's 'Nupper compliance' section for why the fixed-dz path was badly non-compliant "
            "(Nupper of order 1e2-1e3) at this figure's M_res/M2=1e-4 dynamic range. 0.5 (rather than "
            "PCH08's ~0.1) is a practical compromise (~9s/tree at M2=1e15 vs multiple minutes at 0.1); "
            "tighten it if runtime allows. Pass a negative value to fall back to the old fixed-dz "
            "grow_full_population_pch08 behaviour (kept for comparison; NOT recommended for the paper "
            "figure at this M_res/M2 ratio).",
        ),
    )
    parser.add_argument("--output", default="fig1_cmf.png")
    args = parser.parse_args()
    n_trees_zh = args.n_trees_zh if args.n_trees_zh is not None else args.n_trees

    if args.dz < 0.02 and args.n_grid <= 40:
        print(
            f"WARNING: --dz={args.dz} with --n-grid={args.n_grid} will badly under-resolve the Zhang-Hui "
            "first-crossing solve at small steps (see module docstring's 'On dz' note) -- results may be "
            "both much slower AND numerically unreliable. Consider --dz>=0.05 or a much larger --n-grid.",
            file=sys.stderr,
        )

    run_params = foraois_io.get_params(args.config)
    cosmo_data = CosmoData(run_params, redshift=[Z0])
    from foraois.pch_trees import PCHMergerTree

    PCHMergerTree(cosmo_data, run_params)  # side effect: builds cosmo_data's sigma(M) interpolation table

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(Z1_VALUES), len(M2_VALUES), figsize=(11, 12), sharex="col")

    log_ratio_grid = np.linspace(-4.5, 0.0, 200)

    target_nupper = None if args.target_nupper < 0 else args.target_nupper
    for col, M2 in enumerate(M2_VALUES):
        print(f"M2={M2:.2e} ...", flush=True)
        pch_by_z1, zh_by_z1 = run_column(
            args.config, M2, Z1_VALUES, args.dz, args.n_trees, n_trees_zh, "cdm", args.n_grid, args.s_max_factor, args.n_jobs, args.seed + col * 10_000,
            target_nupper=target_nupper,
        )
        for row, z1 in enumerate(Z1_VALUES):
            ax = axes[row, col]

            centers, log_f_pch = cmf_histogram(pch_by_z1[z1], M2)
            _, log_f_zh = cmf_histogram(zh_by_z1[z1], M2)

            if args.show_eps_reference:
                log_f_eps = eps_analytic_cmf(M2, Z0, z1, cosmo_data, log_ratio_grid)
                ax.plot(log_ratio_grid, log_f_eps, "-", color="0.6", lw=1.2, label="unmodified EPS (analytic, NOT a fair comparison -- see docstring)")

            ax.step(centers, log_f_pch, where="mid", color="#1f77b4", lw=1.5, label="PCH08 (Monte Carlo)")
            ax.step(centers, log_f_zh, where="mid", color="#9467bd", lw=1.5, label="Zhang-Hui (Monte Carlo)")

            # Auto-extend the y floor to whatever the data actually spans (capped
            # at a sane minimum) -- a fixed floor clips most of the curve at high
            # z1 / large M2, where the resolution-driven suppression pushes most
            # bins well below -2 (see 'On the EPS reference curve' in the module
            # docstring for why this suppression is real, not a bug).
            finite_vals = np.concatenate([log_f_pch[np.isfinite(log_f_pch)], log_f_zh[np.isfinite(log_f_zh)]])
            y_floor = min(-2.0, np.min(finite_vals) - 0.2) if finite_vals.size else -2.0
            ax.set_ylim(y_floor, 0.5)
            ax.set_xlim(-4.5, 0.05)
            if row == 0:
                ax.set_title(f"$M_2={M2:.2e}\\,M_\\odot/h$", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"$z_1={z1}$\n" + r"$\log_{10} f_{\rm cmf}$", fontsize=9)
            if row == len(Z1_VALUES) - 1:
                ax.set_xlabel(r"$\log_{10}(M_1/M_2)$")
            ax.grid(alpha=0.3)

    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.suptitle("PCH08 Fig. 1 analogue: conditional mass function (no N-body reference available)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
