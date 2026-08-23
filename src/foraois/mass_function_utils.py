import numpy as np
from scipy import interpolate


class MassFunctions:
    """
    A class to generate mass functions and their associated quantities.
    """

    def __init__(self, cosmo_data, redshift=None):
        if redshift is None:
            redshift = [0.0]
        self.redshift = np.asarray(redshift)
        if np.any(self.redshift < 0):
            raise ValueError("Redshift must be a non-negative float or array of non-negative floats.")
        self.cosmo_data = cosmo_data
        self.rhomean = cosmo_data.get_omega_matter(redshift) * 2.7755e11

        # Precompute interpolators
        self._sigma_interp = interpolate.interp1d(
            cosmo_data._logmass,
            np.log10(cosmo_data._sigma),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        self._dsigma_interp = interpolate.interp1d(
            cosmo_data._logmass,
            np.log10(np.abs(cosmo_data._dlogsigma_dlogmass)),
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

    def _get_sigma(self, masses):
        return 10 ** self._sigma_interp(np.log10(masses))

    def _get_dlogsigma_dlogm(self, masses):
        return -(10 ** self._dsigma_interp(np.log10(masses)))

    def _get_nu(self, masses, redshift):
        sigma_vals = self._get_sigma(masses)
        _, _, delta_col = self.cosmo_data.get_linear_growth_and_collapse(redshift)
        return delta_col / sigma_vals

    def get_massfunc(self, redshift, mass_function_type="press_schechter"):
        """
        Compute the halo mass function dn/dlogM at a given redshift.

        Assumes the mass function can be written in the form

            n(M,z) = rho_mean/M**2 * f_XX(nu) * |dln nu/dln M|

        where n is the comoving number density, M is halo mass, rho_mean
        is the cosmic mean density, nu = delta_col/sigma(M), and f_XX(nu)
        is the multiplicity function for the chosen parameterisation
        (Press-Schechter or Sheth-Tormen). Since nu = delta_col/sigma(M),
        dln nu = -dln sigma, so |dln nu/dln M| = |dln sigma/dln M|.

        Parameters
        ----------
        redshift : float
        mass_function_type : {"press_schechter", "sheth_tormen"}

        Returns
        -------
        logmass : np.ndarray
            log10(M) grid the mass function is evaluated on, Msun/h.
        dndlogm : np.ndarray
            n(M,z) at each logmass grid point.
        """
        redshift = np.atleast_1d(redshift)
        if np.any(redshift < 0):
            raise ValueError("Redshift cannot be less than zero")

        logmass = np.arange(8, 15, 0.1)
        masses = 10**logmass

        if mass_function_type == "press_schechter":
            fnu = self.fps(masses, redshift)
        elif mass_function_type == "sheth_tormen":
            fnu = self.fst(masses, redshift)
        else:
            raise ValueError(
                f"Unknown mass_function_type={mass_function_type!r} (supported: 'press_schechter', 'sheth_tormen')."
            )

        dlnnu_dlnm = self.dlnnu_dlnm(masses, redshift)

        return logmass, (self.rhomean / masses) * fnu * dlnnu_dlnm

    def fps(self, mass, redshift):
        """
        This is Press Schechter multiplicity function. It has the form,

        f_PS(nu)= sqrt(2/pi) x nu x exp(-nu**2/2)

        where nu=delta_col/sigma(M)
        """
        masses = np.atleast_1d(mass)
        if np.any(masses <= 0):
            raise ValueError("Cannot have masses less than or equal to zero")

        redshifts = np.atleast_1d(redshift)
        if np.any(redshifts < 0):
            raise ValueError("Cannot have redshifts less than zero")

        nu = self._get_nu(masses, redshift)

        return np.sqrt(2 / np.pi) * nu * np.exp(-0.5 * nu * nu)

    def fst(self, mass, redshift):
        """
        This is Sheth Tormen multiplicity function. It has the form,

        f_ST(nu')= 0.322*(1+nu'^-0.6)*fps(nu')

        where nu'=0.84 x nu with nu=delta_col/sigma(M), and fps is the
        Press-Schechter multiplicity function.
        """
        masses = np.atleast_1d(mass)
        if np.any(masses <= 0):
            raise ValueError("Cannot have masses less than or equal to zero")

        redshifts = np.atleast_1d(redshift)
        if np.any(redshifts < 0):
            raise ValueError("Cannot have redshifts less than zero")

        nu = self._get_nu(masses, redshift)
        nuprime = 0.84 * nu
        fps_vals = np.sqrt(2 / np.pi) * nuprime * np.exp(-0.5 * nuprime * nuprime)

        return 0.322 * (1.0 + nuprime**-0.6) * fps_vals

    def dlnnu_dlnm(self, mass, redshift):
        """
        This is the logarithmic derivative of nu with respect to mass, M. Note that
             dln nu/dln M = -dln sigma/dln M
        and so we can that derivative, which we compute in cosmo_utils.

        Parameters:
        mass (np.ndarray): masses at which to compute the derivative
        redshift (np.ndarray): redshift at which to compute the derivative

        Returns:
        dlnudlm (np.ndarray): array of values of the derivative
        """
        masses = np.atleast_1d(mass)
        if np.any(masses <= 0):
            raise ValueError("Cannot have masses less than or equal to zero")

        redshifts = np.atleast_1d(redshift)
        if np.any(redshifts < 0):
            raise ValueError("Cannot have redshifts less than zero")

        return np.abs(self._get_dlogsigma_dlogm(masses))
