"""
SIDM collapse barrier -- not implemented.

Unlike WDM/FDM, it isn't established that SIDM's effects (core formation,
subhalo evaporation -- primarily *nonlinear*, post-collapse structural
changes) even correspond to a modified *linear* collapse threshold the
way WDM/FDM's do. This needs its own literature-grounded research before
any code -- raising here rather than guessing at a placeholder (unlike
fdm.py, which at least has a documented, if inadequate, fallback).
"""


def delta_c_sidm(M, z, cosmo_data):
    """
    delta_c(M, z) for SIDM -- always raises. No barrier has been derived
    or even scoped for SIDM yet; see module docstring.
    """
    raise NotImplementedError(
        "SIDM has no collapse barrier implemented (or even scoped) yet. "
        "This needs its own literature-grounded research plan before any "
        "code, not a placeholder guess."
    )
