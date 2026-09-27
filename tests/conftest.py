import numpy as np
import pytest

from foraois.cosmo_utils import CosmoData
from foraois.pch_trees import PCHMergerTree

PLANCK_LIKE = {
    "Code": {
        "mode": "camb",  # never actually triggers CAMB in these tests --
        # get_power_spectrum() is not called; we build pk_data synthetically.
        "pk_kmin": 1e-4,
        "pk_kmax": 10.0,
        "pk_npoints": 500,
    },
    "Cosmology": {
        "H0": 67.66,
        "OmegaM": 0.3111,
        "OmegaK": 0.0,
        "OmegaLambda": 0.6889,
    },
}


@pytest.fixture
def cosmo_data():
    """
    A CosmoData instance built with no CLASS/CAMB dependency.

    __init__ only needs get_hubble_parameter/get_omega_matter (pure Python
    math on the Cosmology params) plus precompute_delta_col_table(), none of
    which touch the boltzmann-code backends -- those are only invoked inside
    get_power_spectrum(), which none of these tests call.
    """
    return CosmoData(PLANCK_LIKE, redshift=[0.0])


@pytest.fixture
def synthetic_pk_data():
    """
    A synthetic power-law P(k) = A * k^n, standing in for a real CLASS/CAMB
    linear power spectrum. Good enough to exercise get_mass_variance,
    dlogsigma_dlogmass and the PCHMergerTree machinery built on top of them
    without needing either boltzmann code installed.
    """
    k = np.logspace(-4, 1, 500)
    A, n = 2.0e4, -2.0
    Pk = A * k**n
    return {"k": k, "Pk": Pk.reshape(1, -1), "z": [0.0]}


@pytest.fixture
def tree_generator(cosmo_data, synthetic_pk_data):
    """
    A PCHMergerTree built on the synthetic-P(k) CosmoData above.
    get_power_spectrum() is monkeypatched so construction never touches
    CLASS/CAMB.
    """
    cosmo_data.get_power_spectrum = lambda: synthetic_pk_data
    return PCHMergerTree(cosmo_data, PLANCK_LIKE)


@pytest.fixture
def zh_synthetic_pk_data():
    """
    Same functional form as synthetic_pk_data, but with an amplitude tuned
    to give a roughly realistic sigma(M) normalization (sigma(1e12
    Msun/h) ~ 2.2, close to a real Planck cosmology's ~2.2 -- checked
    directly against config/planck2018_camb.yml) rather than
    synthetic_pk_data's much larger sigma(M) ~ 37.

    zhang_hui_trees.py's first_crossing_step solves on a uniform S-grid
    sized off sigma(M_res)^2 - sigma(M0)^2; when sigma(M) is wildly larger
    than delta_c (~1.7) -- as with synthetic_pk_data's amplitude -- that
    grid can badly under-resolve the actual first-crossing distribution
    (its width is set by delta_c^2, a completely different scale), giving
    numerically meaningless p_res/F_zh despite solve_first_crossing itself
    behaving correctly at the (too sparse) points it evaluates -- see
    first_crossing_step's own grid-resolution warning. PCH08's own tests
    don't need this fix (its rate math is
    scale-covariant in sigma(M)'s absolute normalization), which is why
    synthetic_pk_data above is left alone rather than retuned.
    """
    k = np.logspace(-4, 1, 500)
    A, n = 70.0, -2.0
    Pk = A * k**n
    return {"k": k, "Pk": Pk.reshape(1, -1), "z": [0.0]}


@pytest.fixture
def zh_tree_generator(cosmo_data, zh_synthetic_pk_data):
    """
    A PCHMergerTree built on zh_synthetic_pk_data's realistic-sigma(M)
    amplitude (see that fixture's docstring) -- used by
    tests/test_zhang_hui_trees.py and tests/test_zhang_hui_validation.py.
    Still a PCHMergerTree (not a ZhangHuiMergerTree) only for historical
    reasons: cosmo_data now builds its own sigma(M) table on first use, so
    the tree is not needed for that. Callers only use .cosmo_data,
    constructing their own ZhangHuiMergerTree instances with whatever
    model/rng/N_grid the test needs.
    """
    cosmo_data.get_power_spectrum = lambda: zh_synthetic_pk_data
    return PCHMergerTree(cosmo_data, PLANCK_LIKE)


@pytest.fixture
def set_barrier(monkeypatch):
    """
    Returns ``set_barrier(cosmo_data, name, beta=None)``, which sets the collapse-barrier config keys on
    ``cosmo_data.run_params`` for the duration of one test. ``run_params`` is the same dict object as
    ``PLANCK_LIKE["Code"]``, shared by every test, so the keys are set through ``monkeypatch`` (restored after the
    test) and never assigned directly.
    """

    def _set(cosmo_data, name, beta=None):
        monkeypatch.setitem(cosmo_data.run_params, "barrier", name)
        if beta is not None:
            monkeypatch.setitem(cosmo_data.run_params, "barrier_parameter", beta)

    return _set
