import warnings

import numpy as np
from scipy import integrate, interpolate

from foraois import transfer_functions
from foraois.utils import io, window_function


class CosmoData:
    """
    A class to calculate and store cosmological data.

    Construct either directly (``params`` in the nested
    ``{"Cosmology": ..., "Code": ...}`` shape ``foraois.utils.io.get_params``
    produces from a YAML file) or via ``CosmoData.from_params(...)`` for
    plain-keyword-argument construction with no config file needed.

    Key performance changes vs. original
    -------------------------------------
    * ``precompute_delta_col_table()`` builds a dense z-grid of delta_col once
      so every downstream caller can use a cheap ``np.interp`` instead of
      re-integrating the growth ODE.
    * ``_prepare_sigma_grid()`` now stores a ``scipy`` linear interpolant
      (``assume_sorted=True``) so vectorised sigma lookups avoid Python-level
      loop overhead.
    * Both tables are built automatically at the end of ``__init__`` so callers
      don't have to remember to call them manually.
    """

    def __init__(self, params, redshift=None):
        """
        Parameters
        ----------
        params : dict
            ``{"Cosmology": {...}, "Code": {...}}``, e.g. from
            ``foraois.utils.io.get_params(yaml_path)``. Prefer
            ``CosmoData.from_params(...)`` for direct construction from
            plain cosmological-parameter keyword arguments.
        redshift : list of float, optional
            Redshift(s) at which ``get_power_spectrum()`` will later
            evaluate ``P(k, z)``. Default ``[0.0]``.
        """
        if redshift is None:
            redshift = [0.0]
        self.run_params = params["Code"]
        self.mode = params["Code"]["mode"]
        self.redshift = redshift
        self.cosmo_params = params["Cosmology"]

        use_spherical_bessel_on = params["Code"].get("use_spherical_bessel", False)

        self.linear_growth_factor_z0 = 1.0
        self.linear_growth_factor_z0, _, _ = self.get_linear_growth_and_collapse(redshift=0.0)
        self.rhocrit0 = 2.775e11  # Critical density at z=0 in Msun/Mpc^3/h^2

        default_window_function_type = params["Code"].get("window_function_type", "top_hat")

        self.wf = window_function.WindowFunctions(
            window_function_type=default_window_function_type,
            use_spherical_bessel=use_spherical_bessel_on,
            sharp_k_alpha=params["Code"].get("sharp_k_alpha", 2.5),
        )
        # get_mass_variance() sets this as a side effect when called with an
        # explicit window_function_type, but _prepare_sigma_grid() (called
        # from PCHMergerTree.__init__) reads it before that ever happens, so
        # it must have a default here too -- kept in sync with self.wf's own
        # default above (see get_mass_variance/dlogsigma_dlogmass's comments
        # on why these are two separate attributes that must be kept equal).
        self.window_function_type = default_window_function_type

        # ------------------------------------------------------------------
        # Precompute delta_col(z) lookup table immediately so it is always
        # available without any extra call from the user.
        # ------------------------------------------------------------------
        self.precompute_delta_col_table()

    @classmethod
    def from_params(
        cls,
        H0,
        OmegaM,
        OmegaBar,
        As,
        ns,
        tau,
        OmegaK=0.0,
        mnu=0.0,
        num_massive_neutrinos=0.0,
        mode="camb",
        dm_model="cdm",
        dm_model_mass=None,
        window_function_type="top_hat",
        sharp_k_alpha=2.5,
        pk_kmin=1.0e-4,
        pk_kmax=1.0e2,
        pk_npoints=1000,
        use_spherical_bessel=False,
        redshift=None,
        class_output="mPk",
        class_pk_max_1_per_mpc=1000.0,
        pk_file=None,
    ):
        """
        Construct a ``CosmoData`` directly from cosmological-parameter
        keyword arguments -- no YAML file or hand-built nested dict needed.

        Builds the same ``{"Run": ..., "Cosmology": ...}`` shape a YAML
        config parses to, then reuses ``foraois.utils.io.params_from_dict``
        (the same derivation logic ``get_params`` uses, e.g. omega_b/
        omega_cdm from OmegaBar/OmegaM/h) to produce the nested dict
        ``CosmoData.__init__`` expects.

        Parameters
        ----------
        H0 : float
            Hubble constant today, km/s/Mpc.
        OmegaM, OmegaBar, OmegaK : float
            Matter, baryon, and curvature density parameters today
            (dimensionless). ``OmegaLambda`` is derived as ``1 - OmegaM``.
        As : float
            Primordial scalar power spectrum amplitude. Unused (any value
            works) if ``mode="user"``, since P(k) is supplied directly
            rather than generated from primordial parameters.
        ns : float
            Scalar spectral index. Unused if ``mode="user"`` (see ``As``).
        tau : float
            Reionization optical depth. Unused if ``mode="user"`` (see ``As``).
        mnu : float
            Sum of neutrino masses, eV.
        num_massive_neutrinos : int
            Number of massive neutrino species (CAMB only).
        mode : {"camb", "class", "user"}
            Linear power spectrum backend. ``"camb"`` (default) is
            pip-installable (``pip install foraois[camb]``); ``"class"``
            requires ``classy`` (``pip install foraois[class]``); ``"user"``
            needs no extra dependency at all, but requires ``pk_file``.
        dm_model : {"cdm", "wdm", "fdm"}
            Dark matter model; see ``foraois.transfer_functions``.
        dm_model_mass : float, optional
            Required if ``dm_model`` is ``"wdm"`` (keV) or ``"fdm"``
            (units of 1e-22 eV).
        window_function_type : {"top_hat", "sharp_k", "gaussian"}
            Window function for ``sigma(M)``; see
            ``foraois.utils.window_function.WindowFunctions``.
        sharp_k_alpha : float
            Sharp-k window's ``k0 = alpha/R`` calibration constant
            (Benson et al. 2013), only used if ``window_function_type="sharp_k"``.
        pk_kmin, pk_kmax : float
            ``P(k)` grid bounds, h/Mpc.
        pk_npoints : int
            Number of ``P(k)`` grid points.
        use_spherical_bessel : bool
            Use the spherical-Bessel-function form of the top-hat window
            integral instead of the closed form.
        redshift : list of float, optional
            Passed through to ``CosmoData.__init__``. Default ``[0.0]``.
        class_output, class_pk_max_1_per_mpc :
            CLASS-specific settings, only used if ``mode="class"``.
        pk_file : str, optional
            Path to a two-column (k [h/Mpc], P(k) [(Mpc/h)^3]) text table,
            required if ``mode="user"`` -- see
            ``CosmoData._load_user_power_spectrum``'s docstring for the
            exact file format and interpolation behaviour.

        Returns
        -------
        CosmoData
        """
        if mode == "class":
            mode_block = {"output": class_output, "P_k_max_1/Mpc": class_pk_max_1_per_mpc}
        elif mode == "user":
            mode_block = {"pk_file": pk_file}
        else:
            mode_block = {}

        raw = {
            "Run": {
                "mode": mode,
                "pk_kmin": pk_kmin,
                "pk_kmax": pk_kmax,
                "pk_npoints": pk_npoints,
                "use_spherical_bessel": use_spherical_bessel,
                "dm_model": dm_model,
                "dm_model_mass": dm_model_mass,
                "window_function_type": window_function_type,
                "sharp_k_alpha": sharp_k_alpha,
            },
            "Cosmology": {
                "H0": H0,
                "OmegaM": OmegaM,
                "OmegaBar": OmegaBar,
                "OmegaK": OmegaK,
                "As": As,
                "ns": ns,
                "tau_reio": tau,
                "mnu": mnu,
                "num_massive_neutrinos": num_massive_neutrinos,
            },
            mode: mode_block,
        }
        run_params = io.params_from_dict(raw, source="CosmoData.from_params() arguments")
        return cls(run_params, redshift=redshift)

    # ------------------------------------------------------------------
    # Cosmological utility methods (unchanged from original)
    # ------------------------------------------------------------------

    def parameter_names(self):
        """
        Mapping {internal_name: backend_name} from this class's own
        cosmology parameter names (as stored in self.cosmo_params, e.g.
        'omega_b', 'tau') to the names the selected backend (CLASS or
        CAMB) expects for the same quantity (e.g. 'omega_b'/'ombh2',
        'tau_reio'), restricted to keys actually present in
        self.cosmo_params.

        Returns a dict, not a list: get_power_spectrum() needs both the
        internal name (to look values up in self.cosmo_params) and the
        backend name (the kwarg CLASS/CAMB expects) simultaneously.
        Returning only backend names here previously broke both call
        sites silently -- see the docstring note in get_power_spectrum's
        CAMB/CLASS branches for the bug this replaced.
        """
        param_names = {}
        param_names["class"] = {
            "omega_b": "omega_b",
            "omega_cdm": "omega_cdm",
            "h": "h",
            "As": "A_s",
            "ns": "n_s",
            "tau": "tau_reio",
        }
        param_names["camb"] = {
            "omega_b": "ombh2",
            "omega_cdm": "omch2",
            "omk": "omk",
            "H0": "H0",
            "As": "As",
            "ns": "ns",
            "tau": "tau",
            "mnu": "mnu",
            "num_massive_neutrinos": "num_massive_neutrinos",
        }
        return {
            internal: external for internal, external in param_names[self.mode].items() if internal in self.cosmo_params
        }

    def get_hubble_parameter(self, redshift=0.0):
        redshift = np.asarray(redshift)
        if np.any(redshift < 0):
            raise ValueError(f"Redshift must be non-negative; received redshift={redshift}.")
        H0 = self.cosmo_params["H0"]
        aexp = 1.0 / (1.0 + redshift)
        return H0 * np.sqrt(
            self.cosmo_params["OmegaM"] * aexp**-3
            + self.cosmo_params["OmegaK"] * aexp**-2
            + self.cosmo_params["OmegaLambda"]
        )

    def get_omega_matter(self, redshift=0.0):
        redshift = np.asarray(redshift)
        if np.any(redshift < 0):
            raise ValueError(f"Redshift must be non-negative; received redshift={redshift}.")
        H0 = self.cosmo_params["H0"]
        Omega0 = self.cosmo_params["OmegaM"]
        aexp = 1.0 / (1.0 + redshift)
        H_of_a = self.get_hubble_parameter(redshift)
        return (Omega0 / aexp**3) / (H_of_a / H0) ** 2

    def get_linear_growth_and_collapse(self, redshift, **kwargs):
        """
        Compute linear growth factor D(z), growth rate f(z), and the linear
        collapse threshold delta_col(z).  This performs the full trapezoid
        integration and should be called **once** (or a small number of times)
        to build lookup tables.  Use ``delta_col_at_z()`` for cheap per-call
        lookups inside loops.
        """
        a_min = kwargs.get("a_min", 1.0e-4)
        n_a = kwargs.get("n_a", 1000)

        delta_col_z0 = 1.686
        redshifts = np.atleast_1d(redshift)
        if np.any(redshifts < 0):
            raise ValueError(f"Redshift must be >= 0; received redshift={redshift}.")

        aexp = 1.0 / (1.0 + redshifts)
        a_grid = np.linspace(a_min, 1.0, n_a)
        H_grid = self.get_hubble_parameter(1.0 / a_grid - 1.0)

        integrand_grid = 1.0 / (a_grid * H_grid) ** 3
        integral_cum = integrate.cumulative_trapezoid(integrand_grid, a_grid, initial=0.0)
        integral_vals = np.interp(aexp, a_grid, integral_cum)

        H_of_z = self.get_hubble_parameter(redshifts)
        OmegaM_of_z = self.get_omega_matter(redshifts)

        x = aexp * H_of_z
        D_of_z = (5.0 / 2.0) * OmegaM_of_z * x**3 * integral_vals
        g = D_of_z / aexp
        f_of_z = -1.0 - OmegaM_of_z / 2.0 + (1.0 - OmegaM_of_z) + 5.0 * OmegaM_of_z / (2.0 * g)

        D_of_z /= self.linear_growth_factor_z0
        delta_col = delta_col_z0 / D_of_z

        if np.isscalar(redshift):
            return float(D_of_z[0]), float(f_of_z[0]), float(delta_col[0])
        return D_of_z, f_of_z, delta_col

    # ------------------------------------------------------------------
    # NEW: fast delta_col lookup table
    # ------------------------------------------------------------------

    def precompute_delta_col_table(self, z_max=15.0, nz=5000):
        """
        Build a dense redshift grid and evaluate delta_col(z) once via the
        full growth-factor integration.  Subsequent callers use
        ``delta_col_at_z(z)`` which is a single ``np.interp`` call — O(log N)
        per element, with no integration overhead.

        Parameters
        ----------
        z_max : float
            Maximum redshift to tabulate (default 15 covers all typical
            merger-tree applications).
        nz : int
            Number of grid points.  5000 gives sub-0.01% interpolation error
            for smooth D(z).
        """
        z_grid = np.linspace(0.0, z_max, nz)
        _, _, dc_grid = self.get_linear_growth_and_collapse(redshift=z_grid)

        self._dc_z_grid = z_grid  # shape (nz,)
        self._dc_dc_grid = dc_grid  # shape (nz,)

    def delta_col_at_z(self, z):
        """
        Return delta_col at redshift *z* (scalar or array) via fast table
        lookup.  Requires ``precompute_delta_col_table()`` to have been called
        (done automatically in ``__init__``).

        Parameters
        ----------
        z : float or array-like

        Returns
        -------
        float or np.ndarray
        """
        scalar_input = np.isscalar(z)
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        out = np.interp(z_arr, self._dc_z_grid, self._dc_dc_grid)
        return float(out[0]) if scalar_input else out

    # ------------------------------------------------------------------
    # NEW: cosmic time t(z), used only for dendrogram-style tree plots
    # (edge length proportional to log of elapsed cosmic time between
    # successive redshifts, following Nadler et al. 2023, Figure 4)
    # ------------------------------------------------------------------

    def get_cosmic_time(self, redshift, a_min=1e-4, n_a=2000):
        """
        Cosmic time t(a) = integral_0^a da'/(a' H(a')), in units of 1/H0
        (whatever units cosmo_params['H0'] carries -- e.g. km/s/Mpc gives t
        in Mpc*s/km). No further unit conversion is applied because the only
        use of this so far (dendrogram edge lengths) only needs relative
        values (differences/ratios of t), which are unaffected by an overall
        constant scaling.
        """
        redshifts = np.atleast_1d(np.asarray(redshift, dtype=float))
        if np.any(redshifts < 0):
            raise ValueError(f"Redshift must be non-negative; received redshift={redshift}.")

        aexp = 1.0 / (1.0 + redshifts)
        a_grid = np.linspace(a_min, 1.0, n_a)
        H_grid = self.get_hubble_parameter(1.0 / a_grid - 1.0)

        integrand = 1.0 / (a_grid * H_grid)
        integral_cum = integrate.cumulative_trapezoid(integrand, a_grid, initial=0.0)
        t = np.interp(aexp, a_grid, integral_cum)

        return float(t[0]) if np.isscalar(redshift) else t

    def precompute_cosmic_time_table(self, z_max=15.0, nz=2000):
        """Build a dense lookup table for cosmic_time_at_z(); see delta_col's
        precompute_delta_col_table for the equivalent pattern. Not called
        automatically in __init__ (unlike delta_col) since it's only needed
        by dendrogram-style tree plots, not the core tree-building path."""
        z_grid = np.linspace(0.0, z_max, nz)
        self._t_z_grid = z_grid
        self._t_t_grid = self.get_cosmic_time(z_grid)

    def cosmic_time_at_z(self, z):
        """
        Fast table lookup of cosmic_time; builds the table lazily on first
        call (see precompute_cosmic_time_table), and extends it if the
        requested z exceeds the table's current range.

        Same bug class as pch_trees.PCHMergerTree's delta_col table fix
        (_ensure_delta_col_covers): precompute_cosmic_time_table()'s own
        default z_max=15 is independent of whatever z a caller later asks
        for here, and np.interp silently *clamps* to the table's edge
        rather than erroring -- so any z > 15 (increasingly common now
        that deep trees are supported -- see the delta_col fix) would
        silently return the same (wrong) cosmic time as z=15, with no
        warning. Guarded here directly rather than relying on every
        caller to remember to precompute_cosmic_time_table(z_max=...)
        first.
        """
        z_arr = np.atleast_1d(np.asarray(z, dtype=float))
        requested_z_max = float(np.max(z_arr)) if z_arr.size else 0.0

        if not hasattr(self, "_t_z_grid") or requested_z_max > self._t_z_grid[-1]:
            self.precompute_cosmic_time_table(z_max=max(15.0, requested_z_max))

        scalar_input = np.isscalar(z)
        out = np.interp(z_arr, self._t_z_grid, self._t_t_grid)
        return float(out[0]) if scalar_input else out

    # ------------------------------------------------------------------
    # Power spectrum helpers (unchanged from original)
    # ------------------------------------------------------------------

    def get_kvals(self):
        """
        log10(k) grid the linear P(k) will be evaluated on, h/Mpc, sized
        from run_params['pk_kmin']/'pk_kmax'/'pk_npoints' (uniform in
        log10 k). Sets self.log10kmin/log10kmax/npoints as a side effect,
        consumed by get_power_spectrum()'s CLASS/CAMB branches.
        """
        if "pk_kmin" not in self.run_params or "pk_kmax" not in self.run_params:
            raise KeyError("Run parameters must contain 'pk_kmin' and 'pk_kmax'.")
        self.log10kmin = np.log10(np.float32(self.run_params["pk_kmin"]))
        self.log10kmax = np.log10(np.float32(self.run_params["pk_kmax"]))
        self.npoints = self.run_params.get("pk_npoints", 1000)
        log10k = np.linspace(self.log10kmin, self.log10kmax, self.npoints)
        return log10k

    def get_power_spectrum(self):
        """
        Get cosmology instance from CLASS or CAMB and return the linear P(k).
        """
        log10k = self.get_kvals()
        pk_data = {"k": None, "z": self.redshift, "Pk": None}

        if self.mode == "class":
            import classy

            # Regression fix for a real bug: parameter_names() used to
            # return only the CLASS-side names (e.g. 'tau_reio'), which
            # this dict comprehension then looked up *in self.cosmo_params*
            # -- whose keys are the internal names ('tau'), not the CLASS
            # ones. 'tau_reio' (and 'A_s', 'n_s') were never actually in
            # self.cosmo_params, so `if name in self.cosmo_params` silently
            # dropped them: CLASS ran with its own default amplitude,
            # tilt, and optical depth instead of the configured values,
            # for every mode='class' run ever made with this class.
            # parameter_names() now returns {internal: external}, so both
            # names are available where they're each needed.
            class_param_map = self.parameter_names()
            class_cosmo_params = {
                external: self.cosmo_params[internal] for internal, external in class_param_map.items()
            }
            class_parameters = self.run_params["CLASS"]
            cosmo = classy.Class()
            cosmo.set(class_cosmo_params)
            cosmo.set(class_parameters)
            cosmo.compute()

            pk_data["redshift"] = self.redshift
            tmp = []
            for lk in log10k:
                k = 10**lk * cosmo.h()
                tmp.append(cosmo.pk_lin(k, self.redshift[0]))
            pk_data["k"] = 10**log10k
            pk_data["Pk"] = np.array(tmp).reshape(1, -1) * cosmo.h() ** 3

            cosmo.struct_cleanup()
            cosmo.empty()

        elif self.mode == "camb":
            import camb

            # Same bug/fix as the CLASS branch above: 'ombh2'/'omch2'/
            # 'tau_reio' were never keys of self.cosmo_params (whose keys
            # are 'omega_b'/'omega_cdm'/'tau'), so they were silently
            # dropped -- CAMB ran with its own default baryon/CDM density
            # and optical depth for every mode='camb' run ever made with
            # this class (only H0/As/ns/omk/mnu -- the params whose
            # internal and backend names happen to coincide -- actually
            # reached CAMB).
            camb_param_map = self.parameter_names()
            camb_cosmo_params = {external: self.cosmo_params[internal] for internal, external in camb_param_map.items()}
            camb_cosmo_params.pop("As")
            camb_cosmo_params.pop("ns")
            camb_init_params = {name: self.cosmo_params[name] for name in ["As", "ns"]}

            pars = camb.CAMBparams()
            pars.set_cosmology(**camb_cosmo_params)
            pars.InitPower.set_params(**camb_init_params)
            pars.set_matter_power(redshifts=[self.redshift[0]], kmax=10 ** log10k[-1])
            pars.NonLinear = camb.model.NonLinear_none
            cosmo = camb.get_results(pars)

            pk_data["k"], pk_data["reshift"], pk_data["Pk"] = cosmo.get_matter_power_spectrum(
                minkh=10 ** log10k[0],
                maxkh=10 ** log10k[-1],
                npoints=self.npoints,
            )
            pk_data["Pk"] = pk_data["Pk"].reshape(1, -1)

        elif self.mode == "user":
            pk_data["k"], pk_data["Pk"] = self._load_user_power_spectrum(log10k)
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")

        pk_data["Pk"] = pk_data["Pk"] * self._dm_transfer_function(pk_data["k"])[None, :] ** 2

        return pk_data

    def _load_user_power_spectrum(self, log10k):
        """
        Load a user-supplied linear P(k) table (mode="user") and log-log
        interpolate it onto the pk_kmin..pk_kmax grid (log10k) the
        class/camb branches of get_power_spectrum() also use, so
        downstream code (sigma(M), etc.) never needs to know which
        backend produced P(k).

        The table (run_params['USER']['pk_file']) must be a plain text
        file with two whitespace/comma-separated columns, no header:
        k (h/Mpc), P(k) ((Mpc/h)^3) -- the same convention CLASS/CAMB
        results are already put in elsewhere in this file. Interpolated
        linearly in log10(k)/log10(P(k)), appropriate for a P(k) that is
        smooth and positive over many decades in k, as every physical
        linear power spectrum is.

        Requesting k outside the table's own tabulated range raises
        rather than extrapolating: unlike delta_col_at_z's np.interp-
        based clamping (see ensure_delta_col_covers's docstring for the
        silent-failure-mode bug that one caused), there is no automatic
        "extend the table" fallback possible here -- extending a
        *measured/supplied* P(k) table needs new data, not more
        computation -- so failing clearly is the only honest option;
        widen pk_kmin/pk_kmax or the pk_file's own range instead.

        Returns
        -------
        k : np.ndarray, shape (len(log10k),), h/Mpc
        Pk : np.ndarray, shape (1, len(log10k)), (Mpc/h)^3
        """
        pk_file = self.run_params.get("USER", {}).get("pk_file")
        if not pk_file:
            raise ValueError(
                "mode='user' requires run_params['USER']['pk_file'] (a path to a "
                "k, P(k) table) -- see docs/MODELS.md's config reference."
            )
        table = np.loadtxt(pk_file)
        if table.ndim != 2 or table.shape[1] != 2:
            raise ValueError(f"pk_file={pk_file!r} must be a two-column (k, P(k)) table; got shape {table.shape}.")
        k_table, Pk_table = table[:, 0], table[:, 1]
        if np.any(k_table <= 0) or np.any(Pk_table <= 0):
            raise ValueError(
                f"pk_file={pk_file!r}: k and P(k) must both be strictly positive "
                "(this method interpolates in log-log space)."
            )

        k = 10**log10k
        k_min_table, k_max_table = k_table.min(), k_table.max()
        if k.min() < k_min_table or k.max() > k_max_table:
            raise ValueError(
                f"pk_file={pk_file!r} only covers k in [{k_min_table:.3e}, {k_max_table:.3e}] h/Mpc, "
                f"but pk_kmin/pk_kmax requests [{k.min():.3e}, {k.max():.3e}] h/Mpc -- widen the table "
                "or narrow pk_kmin/pk_kmax in the config."
            )

        order = np.argsort(k_table)
        log_interp = interpolate.interp1d(
            np.log10(k_table[order]), np.log10(Pk_table[order]), kind="linear", bounds_error=True
        )
        Pk = 10 ** log_interp(np.log10(k))
        return k, Pk.reshape(1, -1)

    def _dm_transfer_function(self, k):
        """
        T(k) for the configured dm_model ('cdm' by default -- see
        transfer_functions.py). Applied as T(k)^2 to P(k) in
        get_power_spectrum() above, regardless of which Boltzmann code
        produced the underlying CDM spectrum.
        """
        dm_model = self.run_params.get("dm_model", "cdm")
        if dm_model == "cdm":
            return transfer_functions.T_CDM(k)
        elif dm_model == "wdm":
            m_wdm = self.run_params.get("dm_model_mass")
            if m_wdm is None:
                raise ValueError(
                    "dm_model='wdm' requires 'dm_model_mass' (the WDM particle "
                    "mass in keV) to be set in the Code/Run config."
                )
            # 'h' is always H0/100 by definition; io.get_params() already
            # stores it explicitly, but fall back to computing it in case
            # cosmo_params was built some other way (e.g. a hand-written
            # dict that only set H0).
            h = self.cosmo_params.get("h", self.cosmo_params["H0"] / 100.0)
            return transfer_functions.T_WDM(k, m_wdm, self.cosmo_params["OmegaM"], h)
        elif dm_model == "fdm":
            m_a22 = self.run_params.get("dm_model_mass")
            if m_a22 is None:
                raise ValueError(
                    "dm_model='fdm' requires 'dm_model_mass' (the FDM particle "
                    "mass in units of 1e-22 eV) to be set in the Code/Run config."
                )
            h = self.cosmo_params.get("h", self.cosmo_params["H0"] / 100.0)
            return transfer_functions.T_FDM(k, m_a22, h)
        else:
            raise ValueError(f"Unknown dm_model '{dm_model}' (supported: 'cdm', 'wdm', 'fdm').")

    # ------------------------------------------------------------------
    # Internal P(k) grid cache (unchanged from original)
    # ------------------------------------------------------------------

    def _prepare_pk_grid(self, pk_data):
        cache_key = id(pk_data["k"]) ^ id(pk_data["Pk"])
        if getattr(self, "_pk_cache_key", None) == cache_key:
            return

        Pk_array = pk_data["Pk"]
        if Pk_array.ndim > 1:
            Pk_array = Pk_array[0]

        logk = np.log(pk_data["k"])
        logP = np.log(Pk_array)
        pk_interp_log = interpolate.interp1d(
            logk,
            logP,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )

        lkmin = np.log(np.float32(self.run_params["pk_kmin"]))
        lkmax = np.log(np.float32(self.run_params["pk_kmax"]))
        N = 2**8 + 1
        lk_grid = np.linspace(lkmin, lkmax, N)
        k_grid = np.exp(lk_grid)
        dx = np.diff(lk_grid)[0]
        Pk_vals = np.exp(pk_interp_log(lk_grid))

        self._pk_cache_key = cache_key
        self._lk_grid = lk_grid
        self._k_grid = k_grid
        self._Pk_vals = Pk_vals
        self._dx = dx
        self._pk_interp_log = pk_interp_log
        self._window_cache = {}

    # ------------------------------------------------------------------
    # NEW: optimised sigma grid with scipy interpolant
    # ------------------------------------------------------------------

    def _prepare_sigma_grid(self, pk_data, logmass_min=5.0, logmass_max=16.0, dlogmass=0.01):
        """
        Precompute and cache sigma(M) and d log sigma / d log M on a uniform
        log-mass grid, then build fast ``scipy`` interpolants.

        Changes vs. original
        --------------------
        * Accepts ``logmass_min``, ``logmass_max``, and ``dlogmass`` as
          arguments so the caller can tune resolution.
        * Stores ``scipy.interpolate.interp1d`` objects
          (``_sigma_interp`` and ``_dlogsigma_interp``) with
          ``assume_sorted=True`` for O(log N) vectorised lookups that avoid
          any Python-level looping.
        * Raw arrays (``_logmass``, ``_sigma``, ``_dlogsigma_dlogmass``) are
          still stored for backward compatibility.
        """
        logmass = np.arange(logmass_min, logmass_max, dlogmass)
        radii = self.get_radius(10**logmass)

        sigma = np.sqrt(self.get_mass_variance(pk_data, radius=radii, window_function_type=self.window_function_type))
        dlogsigmadlogmass = self.dlogsigma_dlogmass(
            pk_data,
            mass=10**logmass,
            window_function_type=self.window_function_type,
        )

        # Raw arrays (backward-compatible)
        self._logmass = logmass
        self._sigma = sigma
        self._dlogsigma_dlogmass = dlogsigmadlogmass

        # Fast scipy interpolants (assume_sorted avoids repeated sort checks)
        _kw = dict(
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        self._sigma_interp = interpolate.interp1d(logmass, sigma, **_kw)
        self._dlogsigma_interp = interpolate.interp1d(logmass, np.abs(dlogsigmadlogmass), **_kw)
        # Inverse of sigma_at_logmass (logmass as a function of sigma), needed
        # by zhang_hui_trees.py to convert a sampled first-crossing S back to
        # a progenitor mass. sigma decreases monotonically with logmass, so
        # reversing both arrays gives sigma in ascending order, as interp1d
        # requires of its x-values.
        self._logmass_interp_inv = interpolate.interp1d(sigma[::-1], logmass[::-1], **_kw)

    def sigma_at_logmass(self, logmass):
        """
        Fast vectorised lookup of sigma(M) using the precomputed interpolant.

        Parameters
        ----------
        logmass : float or array-like   log10(M / [Msun/h])

        Returns
        -------
        float or np.ndarray
        """
        return self._sigma_interp(logmass)

    def dlogsigma_at_logmass(self, logmass):
        """
        Fast vectorised lookup of |d log sigma / d log M| using the
        precomputed interpolant.

        Parameters
        ----------
        logmass : float or array-like   log10(M / [Msun/h])

        Returns
        -------
        float or np.ndarray
        """
        return self._dlogsigma_interp(logmass)

    def logmass_at_sigma(self, sigma):
        """
        Inverse of sigma_at_logmass: given sigma(M), return log10(M).

        Used by zhang_hui_trees.py to convert a sampled first-crossing
        variance S back to a progenitor mass (M = 10**logmass_at_sigma(sqrt(S))).

        Parameters
        ----------
        sigma : float or array-like

        Returns
        -------
        float or np.ndarray   log10(M / [Msun/h])
        """
        return self._logmass_interp_inv(sigma)

    # ------------------------------------------------------------------
    # Mass variance (unchanged from original)
    # ------------------------------------------------------------------

    def get_mass_variance(self, pk_data, radius=8.0, window_function_type="top_hat"):
        """
        sigma^2(R) = integral k^2 P(k) W(kR)^2 dk / (2 pi^2), the mass
        variance at radius R (Mpc/h) for the linear P(k) in pk_data --
        note this returns sigma^2, not sigma (e.g. sigma8 = sqrt(
        get_mass_variance(pk_data, radius=8.0))). Sets
        self.window_function_type as a side effect (see the comment
        below on why this must be kept in sync with self.wf's own copy).

        Parameters
        ----------
        pk_data : dict
            {"k": ..., "Pk": ...} as returned by get_power_spectrum().
        radius : float or array-like
            Radius in Mpc/h.
        window_function_type : {"top_hat", "sharp_k", "gaussian"}

        Returns
        -------
        float or np.ndarray, same shape as radius
        """
        # self.wf.window_function_type is what actually drives dispatch
        # inside WindowFunctions.window_function()/_prepare_windows() --
        # self.window_function_type (CosmoData's own attribute, used e.g. by
        # dlogsigma_dlogmass to pick the sharp_k branch) must be kept in
        # sync with it explicitly; they are not the same attribute.
        self.window_function_type = window_function_type
        self.wf.window_function_type = window_function_type

        radii = np.asarray(radius)
        if np.any(radius <= 0):
            raise ValueError(f"All radii must be positive; received radius={radius}.")
        if "k" not in pk_data or "Pk" not in pk_data:
            raise KeyError("pk_data must contain 'k' and 'Pk' keys.")

        self._prepare_pk_grid(pk_data)
        k_grid, Pk_vals, dx = self._k_grid, self._Pk_vals, self._dx

        if window_function_type == "sharp_k":
            sigma = self._sigma2_sharp_k(radii)
            return sigma if sigma.size > 1 else sigma[0]

        W, _ = self.wf._prepare_windows(radii, k_grid)
        integrand_vals = k_grid[None, :] ** 3 * Pk_vals[None, :] * W * W
        sigma = (1.0 / 2.0 / np.pi**2) * integrate.romb(integrand_vals, dx=dx, show=False)

        mass_variance = sigma if sigma.size > 1 else sigma[0]
        return mass_variance

    def _sigma2_sharp_k(self, radii):
        """
        sigma^2(R) for the sharp-k window, computed as an exact definite
        integral with a variable upper limit k0 = alpha/R (via scipy.quad
        against the true log-log-interpolated P(k), _pk_interp_log) rather
        than the generic W(kR)^2-weighted Romberg sum (used for top_hat/
        gaussian) over the fixed 257-point log-k grid built by
        _prepare_pk_grid.

        That generic path is inaccurate here: Romberg integration assumes a
        smooth integrand and extrapolates accordingly, but W^2 for a sharp-k
        window is a step function whose discontinuity almost never lands on
        a grid point -- direct comparison against a quad reference showed
        up to ~3% error growing at high mass (low k0, where the [0, k0]
        integration range is poorly resolved by a grid spaced for the full
        [pk_kmin, pk_kmax] range). Since sigma(M) errors at that level would
        propagate into every downstream tree-building statistic, this uses
        quad's adaptive refinement around the k0 endpoint instead.
        """
        radii = np.atleast_1d(radii).astype(float)
        alpha = self.wf.sharp_k_alpha
        lk_min = self._lk_grid[0]
        lk_max = self._lk_grid[-1]

        def integrand(lnk):
            return np.exp(3.0 * lnk + self._pk_interp_log(lnk))

        sigma = np.empty_like(radii)
        with warnings.catch_warnings():
            # quad's IntegrationWarning here just means its adaptive
            # subdivision couldn't squeeze out the last few digits of
            # epsrel -- expected for a real (BAO-wiggled) P(k), whose
            # piecewise-linear-in-log-log interpolant has small kinks at
            # every source grid point. epsrel=1e-6 is already far tighter
            # than sigma(M) needs (the ~3% error this method fixes was the
            # actual problem; sub-1e-6 precision was never the goal), so
            # the achieved accuracy is fine -- suppress the warning rather
            # than chase an unnecessary tolerance.
            warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
            for i, R in enumerate(radii):
                lk0 = min(np.log(alpha / R), lk_max)
                if lk0 <= lk_min:
                    sigma[i] = 0.0
                    continue
                val, _ = integrate.quad(integrand, lk_min, lk0, limit=200, epsabs=0.0, epsrel=1e-6)
                sigma[i] = val / (2.0 * np.pi**2)

        return sigma

    def dlogsigma_dlogmass(self, pk_data, mass, window_function_type="top_hat"):
        """
        d ln(sigma) / d ln(M) at the given mass -- a numerical derivative
        for top_hat/gaussian windows, a closed form for sharp_k (see
        _sigma2_sharp_k's own docstring for why). Negative, since sigma
        decreases with mass; PCHMergerTree's alpha_grid takes abs() of
        this.

        Both paths compute d ln(sigma^2)/d ln(M) internally (that's what
        falls out naturally from differentiating the sigma^2(R) integral)
        and then halve it, since d ln(S) = 2 d ln(sigma) for S = sigma^2.
        That halving was missing prior to a fix verified by direct
        finite-difference comparison against sigma_at_logmass (this
        function returned exactly 2x the true d ln(sigma)/d ln(M) at every
        mass tested beforehand) -- see the top-hat/gaussian path's own
        inline comment and _dlogsigma_dlogmass_sharp_k's docstring for the
        derivation.

        Parameters
        ----------
        pk_data : dict
            {"k": ..., "Pk": ...} as returned by get_power_spectrum().
        mass : float or array-like, Msun/h
        window_function_type : {"top_hat", "sharp_k", "gaussian"}

        Returns
        -------
        float or np.ndarray, same shape as mass
        """
        # See get_mass_variance's comment: self.window_function_type must be
        # kept in sync with self.wf.window_function_type explicitly, and
        # unconditionally -- self.window_function_type is never actually
        # None after __init__ (it defaults to "top_hat"), so the previous
        # "if self.window_function_type is None" guard here was dead code
        # that silently made any explicit window_function_type override a
        # no-op whenever a window type had already been set once.
        self.window_function_type = window_function_type
        self.wf.window_function_type = window_function_type

        masses = np.atleast_1d(mass).astype(float)
        if np.any(masses <= 0):
            raise ValueError(f"All masses must be positive; received mass={mass}.")
        radii = self.get_radius(masses)

        if "k" not in pk_data or "Pk" not in pk_data:
            raise KeyError("pk_data must contain 'k' and 'Pk' keys.")

        self._prepare_pk_grid(pk_data)
        k_grid, Pk_vals, dx = self._k_grid, self._Pk_vals, self._dx

        if self.window_function_type == "sharp_k":
            return self._dlogsigma_dlogmass_sharp_k(pk_data, radii)

        W, dWdR = self.wf._prepare_windows(radii, k_grid)

        # NOTE: despite the variable name, "sigma" here is actually
        # S = sigma^2 (the mass variance itself, not its square root) --
        # see get_mass_variance's own (1/2pi^2) prefactor, matched here.
        # "dsigmadr" is correspondingly dS/dR, not d(sigma)/dR (chain rule
        # on W^2 inside the S(R) integral gives the extra factor of 2 that
        # cancels the get_mass_variance prefactor's 1/2). d ln S/d ln R
        # = R*(dS/dR)/S is therefore what (1/3)*radii*dsigmadr/sigma below
        # computes -- correct for d ln S/d ln M, but this function is
        # named/documented (and consumed throughout pch_trees.py's
        # alpha_grid) as d ln(sigma)/d ln(M) = (1/2) d ln S/d ln M, hence
        # the extra factor of 1/2 (an undiagnosed factor-of-2 bug prior to
        # this fix -- verified by direct finite-difference comparison
        # against sigma_at_logmass: the unfixed formula returned exactly
        # 2x the true d ln(sigma)/d ln(M) at every mass tested).
        integrand_vals = k_grid[None, :] ** 3 * Pk_vals[None, :] * W * dWdR
        dsigmadr = (1.0 / np.pi**2) * integrate.romb(integrand_vals, dx=dx, show=False)

        integrand_vals = k_grid[None, :] ** 3 * Pk_vals[None, :] * W * W
        sigma = (1.0 / 2.0 / np.pi**2) * integrate.romb(integrand_vals, dx=dx, show=False)

        result = (1.0 / 6.0) * radii * dsigmadr / sigma

        return result if result.size > 1 else result[0]

    def _dlogsigma_dlogmass_sharp_k(self, pk_data, radii):
        """
        Closed-form d ln(sigma) / d ln M for the sharp-k window, matching
        this module's documented convention (dlogsigma_dlogmass's own
        docstring, and every consumer in pch_trees.py's alpha_grid) --
        verified empirically for the top-hat case via finite differences
        against sigma_at_logmass.

        With W(kR) = Theta(alpha - kR), sigma^2(R) collapses to a plain
        definite integral with a variable upper limit k0 = alpha/R:

            S(R) = 1/(2 pi^2) * integral_0^{k0} k^2 P(k) dk

        so Leibniz's rule gives an exact closed form -- no numerical
        differentiation needed, and no need to face the delta-function
        derivative of the window itself (see
        WindowFunctions.window_function_deriv's sharp_k branch):

            dS/dR = -k0^3 P(k0) / (2 pi^2 R)
            d ln S / d ln M = (1/3) d ln S / d ln R = -k0^3 P(k0) / (6 pi^2 S)
            d ln sigma / d ln M = (1/2) d ln S / d ln M = -k0^3 P(k0) / (12 pi^2 S)

        (S = sigma^2, so d ln S = 2 d ln sigma -- this halving step was
        missing prior to this fix, a factor-of-2 bug shared with the
        top-hat/gaussian path above; see that function's own comment for
        the numerical verification.)

        P(k0) uses the same true log-log P(k) interpolant (_pk_interp_log)
        that _sigma2_sharp_k integrates against, rather than the coarser
        257-point resampled grid used by the top_hat/gaussian path -- for
        consistency with S itself, which also comes from _sigma2_sharp_k
        rather than the generic Romberg integral (see its own docstring for
        why that generic path isn't accurate enough for a step-function
        window).
        """
        self._prepare_pk_grid(pk_data)

        radii = np.atleast_1d(radii).astype(float)
        alpha = self.wf.sharp_k_alpha
        k0 = alpha / radii
        k_max = self._k_grid[-1]

        P_at_k0 = np.exp(self._pk_interp_log(np.log(np.minimum(k0, k_max))))
        S = self._sigma2_sharp_k(radii)

        result = -(k0**3 * P_at_k0) / (12.0 * np.pi**2 * S)

        # Once k0 = alpha/R exceeds pk_kmax, _sigma2_sharp_k clamps the
        # integral there too -- S stops changing with R at all in that
        # regime, so the true derivative is exactly zero, not whatever the
        # unclamped closed form (which assumes the integration limit tracks
        # R) would extrapolate to.
        result = np.where(k0 > k_max, 0.0, result)

        return result if result.size > 1 else result[0]

    # ------------------------------------------------------------------
    # Mass ↔ radius conversions (unchanged from original)
    # ------------------------------------------------------------------

    def get_radius(self, mass):
        mass = np.asarray(mass)
        if np.any(mass <= 0):
            raise ValueError(f"Mass must be positive; received mass={mass}.")
        mean_density = self.cosmo_params["OmegaM"] * self.rhocrit0
        return (3 * mass / (4.0 * np.pi * mean_density)) ** (1.0 / 3.0)

    def get_mass(self, radius):
        radius = np.asarray(radius)
        if np.any(radius <= 0):
            raise ValueError(f"Radius must be positive; received radius={radius}.")
        mean_density = self.cosmo_params["OmegaM"] * self.rhocrit0
        return (4.0 * np.pi / 3.0) * mean_density * radius**3


def ensure_delta_col_covers(cosmo_data, z_max):
    """
    Extend cosmo_data's delta_col(z) table if z_max exceeds what's
    currently tabulated -- shared by PCHMergerTree._ensure_delta_col_covers
    and ZhangHuiMergerTree._ensure_delta_col_covers (each also does its own
    bookkeeping afterward: PCHMergerTree refreshes its cached copy of the
    grid arrays for the numba kernel, ZhangHuiMergerTree needs no such
    refresh since it reads cosmo_data.delta_col_at_z() live instead of
    caching the grids itself).

    CosmoData.__init__ builds this table with its own default z_max=15
    (see precompute_delta_col_table), independent of whatever z_max a
    later build_tree/build_full_tree/build_forest_numpy/build_forest_numba
    call actually needs. delta_col_at_z() looks the table up via
    np.interp, which *silently clamps* to the table's boundary value for
    z beyond it -- so requesting z_max > 15 without this check doesn't
    error or warn, it just returns a flat delta_col(z)=delta_col(15) for
    every z beyond 15. Since delta_col only ever enters the branching-rate
    machinery via d_omega = delta_col(z1) - delta_col(z0), a flat/clamped
    table makes d_omega identically 0 for any step entirely beyond the
    table's range, which makes both the split probability (Nupper) and
    the unresolved-accretion fraction (F) identically 0 too -- freezing
    every tree's mass at whatever value it had at the table's edge, for
    every subsequent (higher-z) step. This is a real, silent-failure-mode
    bug this guard exists to close: any caller building trees to
    z_max > 15 without it gets a silently corrupted tree.
    """
    current_z_max = cosmo_data._dc_z_grid[-1]
    if z_max <= current_z_max:
        return
    cosmo_data.precompute_delta_col_table(z_max=z_max)
