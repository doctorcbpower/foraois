import yaml


def get_params(file_path):
    """
    Open a YAML file containing run parameters.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    dict: A dictionary containing the run parameters.
    """
    with open(file_path) as file:
        params = yaml.safe_load(file)
    if not params:
        raise ValueError(f"Parameters file '{file_path}' is empty or invalid.")
    if not isinstance(params, dict):
        raise ValueError(f"Parameters file '{file_path}' does not contain a valid dictionary.")
    if "Run" not in params:
        raise KeyError(f"Parameters file '{file_path}' does not contain 'Run' key.")
    if "Cosmology" not in params:
        raise KeyError(f"Parameters file '{file_path}' does not contain 'Cosmology' key.")
    if params["Run"]["mode"] not in params:
        raise KeyError(f"Parameters file '{file_path}' does not contain {params['Run']['mode']} key.")

    run_params = {
        "Cosmology": {},
        "Code": {"mode": "", "CLASS": {}, "CAMB": {}, "USER": {}},
    }  # Store the parameters for the run

    h = params["Cosmology"]["H0"] / 100.0
    OmegaDM = params["Cosmology"]["OmegaM"] - params["Cosmology"]["OmegaBar"]

    run_params["Cosmology"]["H0"] = params["Cosmology"]["H0"]
    run_params["Cosmology"]["h"] = h
    run_params["Cosmology"]["OmegaLambda"] = 1.0 - params["Cosmology"]["OmegaM"]
    run_params["Cosmology"]["OmegaM"] = params["Cosmology"]["OmegaM"]
    run_params["Cosmology"]["OmegaK"] = params["Cosmology"].get("OmegaK", 0.0)
    run_params["Cosmology"]["omega_b"] = params["Cosmology"]["OmegaBar"] * h * h
    run_params["Cosmology"]["omega_cdm"] = OmegaDM * h * h
    run_params["Cosmology"]["omk"] = params["Cosmology"].get("OmegaK", 0.0)
    run_params["Cosmology"]["As"] = params["Cosmology"]["As"]
    run_params["Cosmology"]["ns"] = params["Cosmology"]["ns"]
    run_params["Cosmology"]["tau"] = params["Cosmology"]["tau_reio"]
    run_params["Cosmology"]["mnu"] = params["Cosmology"].get("mnu", 0.0)
    run_params["Cosmology"]["num_massive_neutrinos"] = params["Cosmology"].get("num_massive_neutrinos", 0.0)

    run_params["Code"]["mode"] = params["Run"]["mode"]
    if run_params["Code"]["mode"] == "class":
        run_params["Code"]["CLASS"]["output"] = params["class"]["output"]
        run_params["Code"]["CLASS"]["P_k_max_1/Mpc"] = params["class"]["P_k_max_1/Mpc"]
    elif run_params["Code"]["mode"] == "camb":
        run_params["Code"]["CAMB"]["output"] = None
    else:
        run_params["Code"]["USER"]["output"] = None

    # Non-standard dark matter: 'cdm' (default) leaves P(k) untouched;
    # 'wdm'/'fdm' multiply it by transfer_functions.T_WDM/T_FDM(k, dm_model_mass, ...)^2
    # inside CosmoData.get_power_spectrum() -- see that module's docstring.
    # dm_model_mass is the WDM particle mass in keV, or the FDM particle mass
    # in units of 1e-22 eV; required if dm_model is 'wdm' or 'fdm'.
    run_params["Code"]["dm_model"] = params["Run"].get("dm_model", "cdm")
    run_params["Code"]["dm_model_mass"] = params["Run"].get("dm_model_mass", None)

    run_params["Code"]["pk_kmin"] = params["Run"].get("pk_kmin", 1.0e-4)
    run_params["Code"]["pk_kmax"] = params["Run"].get("pk_kmax", 1.0e2)
    run_params["Code"]["pk_npoints"] = params["Run"].get("pk_npoints", 1.0e3)
    run_params["Code"]["plot_pk"] = params["Run"].get("plot_pk", False)
    run_params["Code"]["plot_mvar"] = params["Run"].get("plot_mvar", False)
    run_params["Code"]["pk_file_name"] = params["Run"].get("pk_file_name", None)
    run_params["Code"]["mvar_file_name"] = params["Run"].get("mvar_file_name", None)
    run_params["Code"]["use_spherical_bessel"] = params["Run"].get("use_spherical_bessel", False)

    # Window function for sigma(M): 'top_hat' (default) or 'sharp_k' (a
    # Fourier-space step-function window, k0=alpha/R -- see
    # WindowFunctions.sharp_k_window's docstring). Commonly paired with
    # dm_model='wdm'/'fdm' to avoid the top-hat's "cloud-in-cloud" artifact
    # from a truncated P(k), but this is independent of dm_model -- neither
    # setting implies the other.
    run_params["Code"]["window_function_type"] = params["Run"].get("window_function_type", "top_hat")
    run_params["Code"]["sharp_k_alpha"] = params["Run"].get("sharp_k_alpha", 2.5)

    return run_params
