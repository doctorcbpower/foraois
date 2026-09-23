# Changelog

Format loosely follows [Keep a Changelog](https://keepachangelog.com/); this package predates a formal release process, so entries start at the version this file was introduced.

## v0.1.2 (unreleased)

Numerical-validity fix and audit (2026-09), no algorithmic changes.

### Fixed

- **Low-mass `sigma(M)`/`alpha(M)` table clamp (P1).** The compiled (Numba) tree kernels
  (`PCHMergerTree.build_forest_numba`, `grow_full_population_numba_adaptive`,
  `major_merger_redshifts_numba_adaptive`; `ZhangHuiMergerTree.build_forest_numba`,
  `grow_full_population_numba`) read `sigma(M)`/`alpha(M)` from a table that began at
  `1e5 Msun/h` and *clamped* below it, silently reusing `sigma(1e5)` for any smaller
  mass regardless of the `M_res` requested -- an undocumented, unintended resolution
  floor, independent of the NumPy/serial paths (which extrapolate instead, via
  `sigma_at_logmass`/`dlogsigma_at_logmass`). The Zhang--Hui kernel's *inverse*
  sigma-to-mass map had the same clamp. `_prepare_sigma_grid`'s default `logmass_min`
  is now `2.0` (`100 Msun/h`), and every public tree-building method calls
  `CosmoData.check_M_res(M_res, ...)` before building, which raises `ValueError`
  instead of silently clamping or extrapolating below a range that has not been
  validated.

### Added

- `CosmoData.check_M_res(M_res, lookup_factor, compiled)`: validates a requested tree
  mass resolution against both the sigma(M) table (raises if unsupported) and
  `pk_kmax` (warns if the resulting truncation of `sigma(M)`/`alpha(M)` is likely to
  exceed ~1 per cent at `M_res`; skipped for WDM/FDM top-hat, where it would be
  spuriously conservative, and for the sharp-k window, which already signals this
  explicitly). See `docs/MODELS.md`'s "Numerical validity of sigma(M) and alpha(M) at
  low mass" for the full derivation and a worked table of `pk_kmax` against the
  minimum resolvable mass.
- `tests/test_sigma_boundary.py` (28 tests): regression coverage for the above --
  no-silent-clamp at the old `1e5 Msun/h` boundary, inverse-sigma consistency,
  NumPy/compiled agreement down to `M_res=1e3 Msun/h`, every public entry point
  refusing an out-of-range `M_res`, the `pk_kmax` warning firing for CDM and not for
  WDM/FDM/sharp-k, the shipped `menon_power_2024.yml`/`planck2018_camb.yml`
  configs behaving as newly documented, and the warning message itself stating the
  1%/3%/5% thresholds without claiming the calculation is invalid.
- `--pk-kmax` option on `scripts/make_dm_model_comparison_figure.py` (overrides each
  config's own value in memory, for figures reaching lower mass than a config's
  default `pk_kmax` supports).

### Changed

- `config/menon_power_2024.yml`: `pk_kmax` `100` to `3000` -- this config's
  documented use case (Ashvini's MVM production, and the SHMR calculation in
  Section 9 of the companion paper) reaches `M_res ~ 1e4 Msun/h`, well below where
  `pk_kmax=100` is numerically adequate. No other shipped config changed; their
  existing documented uses (`M_res >~ 1e8`-`1e9 Msun/h` throughout the README, quick
  start and test suite) were already unaffected by either fix.

### Audited, no source change (alpha(M) sign convention)

A downstream project (`Stochastic_Occupation`) reconciling its own tree layer against this
release's `check_M_res`/table fix built its own `alpha(M)` table by passing
`CosmoData._dlogsigma_dlogmass` -- the raw, signed `d ln(sigma)/d ln M` -- directly into
`foraois`'s compiled PCH08 kernels, instead of going through `PCHMergerTree.alpha_grid`'s
`np.abs(...)`. Symptom there: every tested seed produced a single unbranched lineage.

Audited whether `foraois` itself has the same sign error. It does not: `PCHMergerTree.__init__`
already builds `self.alpha_grid = np.abs(self.dlogsigma_grid)`, and every compiled-kernel call
site (`build_forest_numba`, `grow_full_population_numba_adaptive`,
`major_merger_redshifts_numba_adaptive`) passes only `alpha_grid`, never the signed
`dlogsigma_grid`. The non-compiled reference path (`branching_rate_terms`, used by
`draw_progenitor_masses` and `diagnostics`) reads `CosmoData.dlogsigma_at_logmass`, whose
interpolant is built from `np.abs(...)` and is therefore already positive by construction.
`ZhangHuiMergerTree` does not consume `alpha(M)` at all. No `foraois` source file changed; no
version bump. `_dlogsigma_dlogmass`/`dlogsigma_dlogmass()` are confirmed, not changed, to be the
correct general-purpose signed API (`mass_function_utils.py`'s own consumer relies on that sign).

What changed: only test coverage. The top-hat branch of `dlogsigma_dlogmass` had no direct sign
test (only the sharp-k branch did); `tests/test_cosmo_utils.py` and `tests/test_pch_validation.py`
gained four tests closing that gap, including one that reproduces the downstream failure mode
directly through `branching_rate_terms`'s `Nupper` (a negative, and therefore unphysical and
never-accepted, expected split count) rather than merely checking that `abs()` was called
somewhere.

### Notes for anyone who generated trees with v0.1.1 at `M_res < 1e5 Msun/h`

Results from `build_forest_numba`, `grow_full_population_numba_adaptive`,
`major_merger_redshifts_numba_adaptive` (PCH08) or `build_forest_numba`,
`grow_full_population_numba` (Zhang--Hui) at a mass resolution below
`1e5 Msun/h` used the wrong `sigma(M)` and should be regenerated. NumPy/serial
paths (`build_tree`, `build_full_tree`, `build_forest_numpy`) were not affected by
the clamp, but were, like every path, subject to the separate `pk_kmax` truncation
above -- check `CosmoData.check_M_res` (or the table in `docs/MODELS.md`) against
the `pk_kmax` your config used.

## v0.1.1 and earlier

Not recorded in this file; see the git history.
