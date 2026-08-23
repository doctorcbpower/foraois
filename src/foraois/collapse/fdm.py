"""
FDM collapse barrier -- NOT the real physics yet.

FDM's Schrodinger-Poisson quantum pressure is expected to modify the
*linear* collapse threshold into a genuinely mass-dependent delta_c(M,z),
not the constant CDM value -- but deriving that (a collapse profile, the
quantum-pressure force law, the nonlinear and linear collapse
calculation, an analytic fit) is still open research (see ROADMAP.md).

delta_c_fdm falls back to the constant CDM barrier as a known-inadequate
placeholder -- existing FDM tree-building (transfer_functions.T_FDM's
power-spectrum suppression, applied with this barrier) already implicitly
relies on this fallback; this function just makes it an explicit,
warned-about choice instead of a silent one. Every call warns (not just
the first -- see warnings.warn's stacklevel) so this can't be mistaken
for validated physics.
"""

import warnings

from .cdm import delta_c_cdm


def delta_c_fdm(M, z, cosmo_data):
    """
    delta_c(M, z) for FDM -- currently just the constant CDM barrier (see
    module docstring). Always warns: this is a disclosed placeholder, not
    a result to build science conclusions on.
    """
    warnings.warn(
        "delta_c_fdm is a placeholder: it returns the constant CDM barrier, "
        "not a real FDM-specific collapse threshold -- the mass-dependent "
        "moving-barrier physics is still open research, not implemented. "
        "Existing FDM results only capture the linear power-spectrum "
        "suppression, not this effect.",
        stacklevel=2,
    )
    return delta_c_cdm(M, z, cosmo_data)
