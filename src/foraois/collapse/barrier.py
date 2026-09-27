"""
Collapse-barrier prescription: the single place delta_c(M, z) is defined.

The barrier is a separate concept from the dark-matter model. The dark-matter model (``dm_model`` in the config) only
modifies P(k), and hence sigma(M), inside ``CosmoData``; the barrier is chosen by the config keys ``barrier`` and
``barrier_parameter`` and is read from ``cosmo_data.run_params``. Tree-building code asks for ``delta_c(M, z,
cosmo_data)`` and does not need to know how the barrier is built.

Two prescriptions are provided.

``fixed`` (default)
    delta_c(M, z) = delta_sc(z), the spherical-collapse threshold, independent of mass.

``linear``
    delta_c(M, z) = delta_sc(z) + beta * sigma^2(M), with beta = ``barrier_parameter`` (default 0.15). With
    S = sigma^2(M) this is linear in the variance, which is the linear barrier B(S) = a + b S that
    ``first_crossing.linear_barrier_first_crossing`` solves analytically. It rises towards small masses. It is a
    phenomenological, illustrative barrier for testing the general first-crossing path, not a physically calibrated
    collapse model. beta must be >= 0.

Which algorithms support which barrier is explicit: the closed-form flat-barrier samplers, PCH08, the constrained
sampler and the flat-barrier diagnostics call ``require_flat_barrier`` and raise ``NotImplementedError`` for a
scale-dependent barrier. Only the general serial Zhang-Hui path (``ZhangHuiMergerTree.build_tree``) supports it.
"""

import numpy as np

BARRIERS = ("fixed", "linear")
DEFAULT_BARRIER = "fixed"
DEFAULT_BARRIER_PARAMETER = 0.15


def barrier_settings(code_params):
    """
    Validated ``(barrier, barrier_parameter)`` from a ``run_params["Code"]`` dict, applying the defaults. This is
    the only place the config keys are interpreted. ``barrier_parameter`` is used only by ``linear``.

    Raises
    ------
    ValueError
        Unknown barrier name, or a negative ``barrier_parameter`` for ``linear``.
    """
    name = code_params.get("barrier", DEFAULT_BARRIER)
    if name not in BARRIERS:
        raise ValueError(f"Unknown barrier {name!r} (supported: {list(BARRIERS)}).")
    beta = code_params.get("barrier_parameter", DEFAULT_BARRIER_PARAMETER)
    if beta is None:
        beta = DEFAULT_BARRIER_PARAMETER
    if name == "linear":
        beta = float(beta)
        if not beta >= 0.0:
            raise ValueError(f"barrier='linear' requires barrier_parameter >= 0 (got {beta}).")
    return name, beta


def _barrier_config(cosmo_data):
    """
    The ``run_params["Code"]`` dict of a configured ``CosmoData``. An object that has none (a historical duck-typed
    stand-in, such as the one the FORTRAN cross-check uses) carries no barrier request and is the fixed barrier.
    """
    return getattr(cosmo_data, "run_params", None) or {}


def delta_c(M, z, *args):
    """
    Collapse barrier delta_c(M, z) for the barrier configured on ``cosmo_data``.

    Call as ``delta_c(M, z, cosmo_data)``. The earlier form ``delta_c(M, z, model, cosmo_data)`` is still accepted so
    that existing callers keep working; its ``model`` argument is ignored, because the dark-matter model no longer
    selects the barrier.

    Parameters
    ----------
    M : float or array-like
        Halo mass (Msun/h). Ignored by ``fixed``. For ``linear``, masses below the floor of the sigma(M) table
        (100 Msun/h by default) are evaluated at the floor, so ``delta_c`` stays finite for any ``M``.
    z : float or array-like
        Redshift.
    cosmo_data : CosmoData
        Supplies ``delta_col_at_z`` (delta_sc), ``sigma_at_logmass`` (for ``linear``; needs the sigma(M) table,
        built when a tree generator is constructed) and the ``barrier`` settings in ``run_params``.

    Returns
    -------
    float or np.ndarray
    """
    if len(args) == 1:
        (cosmo_data,) = args
    elif len(args) == 2:
        _, cosmo_data = args
    else:
        raise TypeError("delta_c expects (M, z, cosmo_data) or the legacy (M, z, model, cosmo_data)")
    name, beta = barrier_settings(_barrier_config(cosmo_data))
    delta_sc = cosmo_data.delta_col_at_z(z)
    if name == "fixed" or beta == 0.0:
        # beta = 0 is the fixed barrier, so it is returned as such (same value and shape, no sigma(M) needed).
        return delta_sc
    # sigma(M) is tabulated down to a floor mass. Below it sigma^2 is held at its floor value: the first-crossing grid
    # can extend to variances the table does not represent (sigma^2 saturates at small mass because P(k) is cut at
    # pk_kmax), where the inverse mass map underflows and would otherwise make the barrier inf/NaN.
    M = np.maximum(np.asarray(M, dtype=float), 10.0 ** cosmo_data._logmass[0])
    sigma = cosmo_data.sigma_at_logmass(np.log10(M))
    return delta_sc + beta * sigma**2


def barrier_is_flat(cosmo_data):
    """True if the configured barrier is independent of mass: ``fixed``, or ``linear`` with beta = 0."""
    name, beta = barrier_settings(_barrier_config(cosmo_data))
    return name == "fixed" or float(beta) == 0.0


def require_flat_barrier(cosmo_data, algorithm):
    """
    Raise ``NotImplementedError`` if the configured barrier is scale dependent. Called by every algorithm that
    assumes delta_c does not depend on mass, so that a request for a scale-dependent barrier is never silently
    replaced by the fixed one.
    """
    if not barrier_is_flat(cosmo_data):
        name, beta = barrier_settings(_barrier_config(cosmo_data))
        raise NotImplementedError(
            f"{algorithm} supports only the fixed collapse barrier (or barrier='linear' with "
            f"barrier_parameter=0), but barrier={name!r} with barrier_parameter={beta} was requested. "
            "Use ZhangHuiMergerTree.build_tree for a scale-dependent barrier."
        )
