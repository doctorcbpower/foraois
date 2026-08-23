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

- **FDM's collapse barrier.** `delta_c_fdm` is a disclosed placeholder that
  falls back to the constant CDM value (and warns on every call) -- the real
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

- **A numba backend for the exact Zhang-Hui algorithm.**
  `ZhangHuiMergerTree.build_forest_numpy` already uses a closed-form solution
  for the flat-barrier models (competitive with PCH08's own numpy backend),
  but there's no JIT-compiled/parallel backend analogous to
  `PCHMergerTree.build_forest_numba`.
