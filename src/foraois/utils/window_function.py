import numpy as np


class WindowFunctions:
    """
    A class to generate window functions for computations of mass variances and mass functions.
    """

    def __init__(self, window_function_type="top_hat", **kwargs):
        """
        Parameters
        ----------
        window_function_type : {"top_hat", "sharp_k", "gaussian"}
            Real-space top-hat (default), Fourier-space sharp-k step
            function (k0 = sharp_k_alpha/R), or Gaussian.
        use_spherical_bessel : bool, keyword-only
            Top-hat only: use the spherical-Bessel-function form of the
            window integral instead of the closed form. Default True.
        sharp_k_alpha : float, keyword-only
            Sharp-k window's k0 = alpha/R calibration constant (Benson
            et al. 2013). Default 2.5.
        """
        self.window_function_type = window_function_type
        self.use_spherical_bessel = kwargs.get("use_spherical_bessel", True)
        # Sharp-k window cutoff: W(kR) = Theta(alpha - kR), i.e. k0 = alpha/R.
        # alpha=2.5 follows Benson et al. 2013 (calibrated against N-body
        # WDM simulations), the same value Kulkarni & Ostriker 2022 use for
        # FDM; also close to the alpha=2.42 approximate-integral estimate of
        # Lacey & Cole 1994. See sharp_k_window()'s docstring.
        self.sharp_k_alpha = kwargs.get("sharp_k_alpha", 2.5)
        self._window_cache = {}  # initialize cache

    def _prepare_windows(self, radii, k_grid):
        """
        Precompute and cache window function W and derivative for given radii.

        Parameters:
        radii (np.ndarray): Radii in Mpc/h
        k_grid (np.ndarray): The k values in h/Mpc

        Returns:
        W, dWdR (np.ndarray): Window function and derivative
        """
        radii = np.atleast_1d(radii)  # Ensure radii is always an array

        # window_function_type is included in the cache key -- not just
        # radii -- because different callers (e.g. WDM using top_hat vs FDM
        # using sharp_k) can request different window types for the same
        # radii array, and a radii-only key would silently return another
        # window type's cached W/dWdR.
        radii_key = (self.window_function_type, tuple(np.round(radii, 12)))
        if hasattr(self, "_window_cache") and radii_key in self._window_cache:
            return self._window_cache[radii_key]

        W = self.window_function(k_grid[None, :], radii[:, None])
        try:
            dWdR = self.window_function_deriv(k_grid[None, :], radii[:, None])
        except NotImplementedError:
            # sharp_k: dW/dR is a Dirac delta, not representable on this
            # grid. Callers that only need W (e.g. get_mass_variance) still
            # work; callers needing dW/dR (e.g. the generic branch of
            # dlogsigma_dlogmass) must not reach here for sharp_k -- that
            # method special-cases it with a closed-form derivative instead.
            dWdR = None

        self._window_cache[radii_key] = (W, dWdR)
        return W, dWdR

    def window_function(self, k, R):
        """
        Calculate the window function for a given k and radius R.

        Parameters:
        k (np.ndarray): The k values in k/Mpc.
        R (float): The radius in Mpc/h.

        Returns:
        np.ndarray: The window function values.
        """
        if self.window_function_type == "top_hat":
            return self.top_hat_window(k, R)
        elif self.window_function_type == "gaussian":
            return self.gaussian_window(k, R)
        elif self.window_function_type == "sharp_k":
            return self.sharp_k_window(k, R)
        else:
            raise ValueError(f"Unknown window function type '{self.window_function_type}'.")

    def window_function_deriv(self, k, R):
        """
        Calculate the derivative of the window function with respect to R for a given k and R

        Parameters:
        k (np.ndarray): The k values in h/Mpc.
        R (float): The radius in Mpc/h.

        Returns:
        np.ndarray: The window function derivative values.
        """
        if self.window_function_type == "top_hat":
            return self.top_hat_window_deriv(k, R)
        elif self.window_function_type == "gaussian":
            return self.gaussian_window(k, R)
        elif self.window_function_type == "sharp_k":
            raise NotImplementedError(
                "sharp_k_window's derivative w.r.t. R is a Dirac delta "
                "function (the window is a step function), not something "
                "expressible via this generic W*dW/dR route. "
                "CosmoData.dlogsigma_dlogmass() special-cases 'sharp_k' with "
                "a closed-form d ln(sigma^2)/d ln M instead of calling this."
            )
        else:
            raise ValueError(f"Unknown window function type '{self.window_function_type}'.")

    def top_hat_window(self, k, R):
        """
        Calculate the top-hat window function.

        Parameters:
        k (np.ndarray): The k values, in h/Mpc.
        R (float): The radius in Mpc/h.

        Returns:
        np.ndarray: The top-hat window function values.
        """

        k = np.atleast_1d(k)
        R = np.atleast_1d(R)

        x = k * R

        if self.use_spherical_bessel:
            from scipy.special import spherical_jn

            return 3 * spherical_jn(1, x) / (x)

        # Mask for small arguments where we use series expansion
        small = x < 1e-3
        W = np.empty_like(x)

        # Series expansion: j1(x) ~ x/3 - x**2/30 + x**5/840 - ...
        # So j1(x)/x ~ 1/3 - x**2/30 + x**4/840 - ...

        if np.any(small):
            x2 = x[small] ** 2
            j1_over_x = (1.0 / 3.0) * (1 - x2 / 10.0 + (x2**2) / 280.0)
            W[small] = 3.0 * j1_over_x

        if np.any(~small):
            y = x[~small]
            sin_y = np.sin(y)
            cos_y = np.cos(y)
            j1_val = (sin_y - y * cos_y) / (y * y)
            W[~small] = 3 * j1_val / y

        if np.ndim(k) == 0:
            return W.item()

        return W

    def top_hat_window_deriv(self, k, R):
        """
        Calculate the top-hat window function derivative.

        Parameters:
        k (np.ndarray): The k values, in h/Mpc.
        R (float): The radius in Mpc/h.

        Returns:
        np.ndarray: Derivative of the top-hat window function values.
        """

        k = np.asarray(k)
        R = np.asarray(R)
        x = k * R

        if self.use_spherical_bessel:
            from scipy.special import spherical_jn

            return (3 / R) * (spherical_jn(0, x) - 3 * spherical_jn(1, x) / (x))

        # Series expansion:
        # j0(x) ~ 1 - x**2/6 + x**4/120 - ...
        # j1(x) ~ x/3 - x**2/30 + x**5/840 - ... so j1(x)/x ~ 1/3 - x**2/30 + x**4/840 - ...
        # Return (3/R)*(j0(x)-3*j1(x)/x)

        x2 = x**2

        j0 = 1.0 - (x2 / 6.0) + (x2 * x2 / 120.0)
        j1_over_x = (1.0 / 3.0) * (1 - x2 / 10.0 + x2 * x2 / 280.0)
        dW_small = (3.0 / R) * (j0 - 3 * j1_over_x)

        # Use full expression
        # j0(x) = sin(x)/x
        # j1(x) = (sin(x) - x cos(x)) / x^2
        # So j1(x)/x = (sin(x) - x cos(x)) / x^3
        # Return (3/R)*(j0(x)-3*j1(x)/x)
        # Handle potential division by zero

        sin_x = np.sin(x)
        cos_x = np.cos(x)
        j0 = sin_x / x
        j1_val = (sin_x - x * cos_x) / (x * x)
        dW = (3.0 / R) * (j0 - 3 * j1_val / x)

        # Identify where the transition from small-x approxmation happens and adjust value accordingly
        dW = np.where(x < 1.0e-3, dW_small, dW)

        # Handle the case of x=0
        dW = np.where(x == 0, 0.0, dW)

        # Return scalar if input was scalar
        if np.ndim(k) == 0 and np.ndim(R) == 0:
            return float(dW)
        return dW

    def sharp_k_window(self, k, R):
        """
        Sharp-k (Fourier-space top-hat) window function:

            W(k, R) = 1  for k <= k0 = alpha/R
            W(k, R) = 0  for k >  k0

        Unlike the real-space top-hat, this window has no real-space profile
        with finite volume (its real-space transform diverges -- Maggiore &
        Riotto 2010), so its "radius" R is not literally an enclosed-mass
        radius; alpha calibrates the relationship between the k-space cutoff
        and the same mass-radius relation used for top-hat (M = 4/3 pi
        rho_bar R^3, via CosmoData.get_radius/get_mass -- unchanged). This
        follows Benson et al. 2013 and Kulkarni & Ostriker 2022 (the same
        source verified for T_FDM in transfer_functions.py), who fit
        alpha=2.5 to N-body/simulation results and use exactly this
        R (not a rescaled one) inside k0=alpha/R -- confirmed directly from
        Kulkarni & Ostriker 2022 eq. 7 and the surrounding text.

        Commonly used for WDM/FDM to avoid the "cloud-in-cloud" artifact a
        truncated P(k) produces under a top-hat window (the top-hat's
        oscillatory sinc-like tails let small-scale power leak back in
        above the free-streaming/quantum-pressure cutoff; sharp-k removes
        that power exactly).

        Parameters
        ----------
        k : array-like, h/Mpc
        R : array-like, Mpc/h

        Returns
        -------
        np.ndarray, 1.0 or 0.0, broadcast shape of k and R
        """
        x = k * R
        return np.where(x <= self.sharp_k_alpha, 1.0, 0.0)

    def gaussian_window(self, k, R):
        """
        Calculate the Gaussian window function.

        Parameters:
        k (np.ndarray): The k values.
        R (float): The radius in Mpc/h.

        Returns:
        np.ndarray: The Gaussian window function values.
        """
        return np.exp(-0.5 * (k * R) ** 2)

    def gaussian_window_deriv(self, k, R):
        """
        Calculate the derivative of the Gaussian window function.

        Parameters:
        k (np.ndarray): The k values.
        R (float): The radius in Mpc/h.

        Returns:
        np.ndarray: The Gaussian window function derivative values.
        """
        return -k * k * R * np.exp(-0.5 * (k * R) ** 2)
