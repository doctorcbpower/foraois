"""
CDM collapse barrier: the standard constant (mass-independent) spherical-
collapse threshold delta_col(z) = 1.686/D(z), already implemented and
tabulated in CosmoData (see cosmo_utils.py's precompute_delta_col_table/
delta_col_at_z).
"""


def delta_c_cdm(M, z, cosmo_data):
    """
    delta_c(M, z) for CDM -- constant in M by construction (the standard
    EPS assumption: a fixed, mass-independent barrier). `M` is accepted
    for interface uniformity with delta_c(M, z, model, cosmo_data) but
    genuinely unused here, not merely unvalidated -- do not add a
    no-op-but-implies-relevance check on it.

    Parameters
    ----------
    M : float or array-like
        Halo mass (Msun/h) -- unused.
    z : float or array-like
        Redshift.
    cosmo_data : CosmoData
        Must already have precompute_delta_col_table() run (done
        automatically in CosmoData.__init__).

    Returns
    -------
    float or np.ndarray, matching z's shape.
    """
    return cosmo_data.delta_col_at_z(z)
