"""
delta_c(M, z, model, cosmo_data) -- the single public interface
tree-building code should depend on for the collapse barrier, dispatching
to the per-model implementation (cdm.py/wdm.py/fdm.py/sidm.py).

Not wired into pch_trees.py's own branching-rate machinery: PCH08's
Appendix A algorithm is derived for a barrier that depends only on z (a
genuine constant-in-M assumption baked into its rate formulas, not
implementation laziness -- see pch_trees.py's own docstrings), so it has
nothing to gain from routing through a mass-dependent interface, while
CDM/WDM (the only models with real barrier content right now) are
mass-independent anyway. `zhang_hui_trees.py`/`zhang_hui_constrained_trees.py`
use this interface directly, since their algorithm is barrier-agnostic by
construction.
"""

_MODEL_FUNCS = {}


def _get_model_funcs():
    # Imported lazily (not at module load) so importing foraois.collapse
    # doesn't require importing every per-model module up front -- cheap
    # here, but keeps the pattern consistent for when fdm.py/sidm.py grow
    # heavier dependencies (e.g. a real quantum-pressure solver).
    if not _MODEL_FUNCS:
        from .cdm import delta_c_cdm
        from .fdm import delta_c_fdm
        from .sidm import delta_c_sidm
        from .wdm import delta_c_wdm

        _MODEL_FUNCS.update(
            {
                "cdm": delta_c_cdm,
                "wdm": delta_c_wdm,
                "fdm": delta_c_fdm,
                "sidm": delta_c_sidm,
            }
        )
    return _MODEL_FUNCS


def delta_c(M, z, model, cosmo_data):
    """
    Collapse barrier delta_c(M, z) for the given dark matter model.

    Parameters
    ----------
    M : float or array-like
        Halo mass (Msun/h). Ignored by every model currently implemented
        (cdm, wdm both use a constant-in-M barrier; fdm's placeholder
        falls back to the same constant value; sidm isn't implemented) --
        accepted now so callers and this interface don't need to change
        once a genuinely mass-dependent barrier (FDM's real physics, or a
        future SIDM one) exists.
    z : float or array-like
        Redshift.
    model : {'cdm', 'wdm', 'fdm', 'sidm'}
    cosmo_data : CosmoData

    Returns
    -------
    float or np.ndarray, matching z's shape.

    Raises
    ------
    ValueError
        Unknown `model`.
    NotImplementedError
        model='sidm' (see collapse.sidm).
    """
    funcs = _get_model_funcs()
    try:
        func = funcs[model]
    except KeyError:
        raise ValueError(f"Unknown model '{model}' (supported: {sorted(funcs)})") from None
    return func(M, z, cosmo_data)
