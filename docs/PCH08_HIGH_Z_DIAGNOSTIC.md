# Timestep compliance of PCH08 histories at small M_res / M0

Status: explained. Near-deterministic PCH08 main-progenitor histories at small `M_res / M0` occur when the fixed step `dz`
violates the single-split-per-step requirement (`N_upper` of order 1 or larger), and disappear as `dz` is reduced.

## Observation

With `PCHMergerTree.build_forest_numba` (and `build_forest_numpy`) at a fixed `dz`, the main-progenitor histories at small
`M_res / M0` can be nearly deterministic (a 16-84 per cent range of a few hundredths in `M(z=1)/M0`) and assemble earlier than
those of the Zhang & Hui generator. Earlier versions of this note recorded this at `dz = 0.05` with `M_res` down to 1e-8 of the
halo mass and left its origin open.

## Cause

PCH08 (their section 2.1) requires the expected number of resolved splits in a step to be small (`N_upper` of order 0.1) because a
step can register at most one split. We adopt `N_upper <~ 0.1` as a practical criterion for the single-split construction; it is
not a mathematical validity boundary. `PCHMergerTree` uses one fixed `dz` for the whole forest and does not adapt it to `N_upper`.
`N_upper` grows as `M_res / M0` falls, so a step that is compliant at `M_res / M0 = 1e-2` is not at `1e-4`. Once `N_upper >> 1`
the step splits almost every time, the mass lost to the unresolved side is fixed by the mean rate, and the history becomes close
to deterministic.

## Table

`scripts/timestep_compliance_table.py`: `M0 = 1e12 Msun/h`, `z0 = 0`, 3000 trees, main-progenitor `M(z=1)/M0`. `max N_upper` is the
largest mean-field PCH08 `N_upper` along the trajectory. Sampling error on a median is about 0.01.

| M_res/M0 | dz | max N_upper | PCH08 median [16, 84] | Zhang & Hui median |
|---|---|---|---|---|
| 1e-2 | 0.05   | 0.32 | 0.55 [0.36, 0.68] | 0.45 |
| 1e-2 | 0.0005 | 0.00 | 0.56 [0.37, 0.69] | 0.42 |
| 1e-4 | 0.05   | 14.4 | 0.81 [0.79, 0.81] | 0.49 |
| 1e-4 | 0.01   | 2.8  | 0.76 [0.59, 0.79] | 0.46 |
| 1e-4 | 0.002  | 0.57 | 0.56 [0.36, 0.69] | 0.45 |
| 1e-4 | 0.0005 | 0.14 | 0.56 [0.37, 0.69] | 0.44 |

At `M_res / M0 = 1e-4` the PCH08 statistic moves from 0.81 to 0.56 as `N_upper` falls from 14 to 0.14, and its range opens up.
The last two rows agree with each other, so the statistic is approaching stability. `N_upper = 0.14` is close to, not below, the
0.1 criterion.

## What this does and does not show

* Compliance, numerical convergence and physical interpretation are separate. The table addresses the first two. There is no
  N-body reference here, so it says nothing about which algorithm is closer to simulations.
* At small `dz` a difference between the two algorithms remains: the PCH08 median `M(z=1)/M0` is about 1.3 times the Zhang & Hui
  value at both resolution ratios. Its origin is not established. Its sign is consistent with the PCH08 split rate lying below
  the EPS rate, but that has not been quantified as the cause.
* The Zhang & Hui builder also draws at most one split per step. At `M_res / M0 = 1e-4`, `dz = 0.05` the EPS expected number of splits
  per step is about 7.5 (`foraois.diagnostics.expected_eps_splits_per_step`), so its split probability saturates and the same
  timestep requirement applies. Its steps must be compared with the EPS expected split count, not with `N_upper`. We adopt `E <~ 0.1`
  in the same practical sense.
* Zhang & Hui is an approximation, not an exact sampler of the EPS tree: in one configuration (`M0 = 1e12`, `M_res / M0 = 1e-2`,
  `dz = 0.02`, `scripts/validate_zh_single_step_vs_eps.py`) its single-step smaller-fragment density agrees with the analytic EPS
  density to about 10 per cent over most of the resolved range and is lower by up to 45 per cent in the bins nearest `M_res`.

## Practical guidance

Choose `dz` so that the largest `N_upper` (PCH08, `foraois.diagnostics.expected_splits_per_step`) or EPS expected splits per step
(both algorithms, `foraois.diagnostics.expected_eps_splits_per_step`) along the trajectory is below about 0.1, and check that the
statistic of interest is stable when `dz` is reduced further. The paper's PCH08 conditional-mass-function and major-merger runs use
`scripts/paper_figs/_treegrowth.py`, which adapts the step to `N_upper`.
