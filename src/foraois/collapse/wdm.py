"""
WDM collapse barrier.

The literature (Benson et al. 2013's sharp-k window calibration) treats
WDM's departure from CDM as adequately captured by the *power spectrum*
suppression (transfer_functions.T_WDM) plus a sharp-k window, keeping the
same constant CDM barrier -- but whether that's sufficient *without* also
refitting PCH08's own (G0, gamma1, gamma2) rate parameters for WDM
specifically (as Benson et al. 2013 did) is an open question (see
ROADMAP.md). Until that's investigated, this is the same constant barrier
as CDM, not a placeholder pending real physics the way fdm.py's is.
"""

from .cdm import delta_c_cdm


def delta_c_wdm(M, z, cosmo_data):
    """
    delta_c(M, z) for WDM -- identical to the CDM barrier (see module
    docstring for the caveat this rests on). `M` unused, same as
    delta_c_cdm.
    """
    return delta_c_cdm(M, z, cosmo_data)
