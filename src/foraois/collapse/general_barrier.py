"""
General collapse barrier: allows for a collapse barrier that depends on both
smoothing mass/radius and redshift. Used for illustrative purposes.
"""


def delta_c_general(M, z, cosmo_data):
    """
    delta_c(M, z) for the general moving barrier case. Used to illustrate that
    foraois can construct trees with general moving barriers. As of 25/09/2026, 
    all DM variants assume a constant barrier at a given z.

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
    beta=0.15

    return cosmo_data.delta_col_at_z(z) + beta * cosmo_data.sigma(M)**2
