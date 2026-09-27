# Roadmap

Known gaps and open questions, in no particular order.

- **SIDM.** No collapse barrier exists or has even been scoped for
  self-interacting dark matter. Unlike WDM/FDM, SIDM's best-studied effects
  (core formation, subhalo evaporation) are primarily *nonlinear* structural
  changes, so it isn't yet established that this problem has the same shape
  (derive a modified linear collapse threshold, plug into the excursion-set
  formalism) as WDM/FDM's. This needs its own literature-grounded research
  before any implementation work.

- **WDM's collapse barrier.** Currently identical to CDM's (constant in
  mass). The sharp-k-window literature (Benson et al. 2013) implicitly treats
  this as sufficient, but whether accuracy also requires a WDM-specific
  refit of the branching-rate parameters (as Benson et al. 2013 did for
  PCH08's own fit) hasn't been checked here.

- **FDM's collapse barrier.** `dm_model: fdm` is a disclosed placeholder: the
  collapse barrier is the configured one (fixed unless `barrier: linear`), and
  `CosmoData` warns that only the power-spectrum suppression is modelled -- the real
  physics is a genuinely mass-dependent moving barrier arising from
  Schrodinger-Poisson quantum pressure, which is still open research. Current
  FDM support only captures the *linear* power-spectrum suppression.

- **A general calibration-to-N-body framework.** PCH08's own
  `(G0, gamma1, gamma2)` fit and Benson et al. (2013)'s WDM refit are each
  one-time, paper-specific exercises. There's no reusable pipeline here that
  takes "N-body reference data + a candidate rate-function family" and
  produces best-fit parameters for a new dark matter model.

- **Secondary progenitor branches on constrained trees.**
  `ZhangHuiMergerTree` has `build_full_tree` for growing every progenitor of
  an *unconstrained* tree (not just the main branch); the constrained
  backend (`zhang_hui_constrained_trees.py`) doesn't yet have an equivalent
  -- it only tracks the main (running-maximum) branch.

- **`pk_kmax` requirement beyond the two tested tree configurations.**
  `check_M_res`'s `pk_kmax * R(M_res) < 4.1` warning is calibrated from a
  Planck 2018 CDM, top-hat sigma(M)/alpha(M) table only (see MODELS.md's
  "Numerical validity" section); it is not applied to WDM/FDM (assumed less
  demanding, not separately tree-tested) and the size of the margin
  tree-level statistics need above this pointwise criterion was checked at
  only two `(z0, z_max, dz)` configurations. A systematic tree-level
  `pk_kmax` scan across dark-matter models and tree configurations would
  let the warning threshold (and `menon_power_2024.yml`'s `pk_kmax=3000`)
  be tightened or loosened with evidence rather than margin.

