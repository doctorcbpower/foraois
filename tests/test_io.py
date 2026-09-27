"""
Tests for foraois.utils.io -- get_params() (YAML config loading) and
params_from_dict() (the shared derivation logic get_params() and
CosmoData.from_params() both build on).
"""

import pytest

from foraois.utils import io

RAW_PARAMS = {
    "Run": {
        "mode": "camb",
        "pk_kmin": 1.0e-4,
        "pk_kmax": 1.0e2,
        "pk_npoints": 500,
    },
    "Cosmology": {
        "H0": 67.66,
        "OmegaM": 0.3111,
        "OmegaBar": 0.0490,
        "OmegaK": 0.0,
        "As": 2.105e-9,
        "ns": 0.9665,
        "tau_reio": 0.0561,
    },
    "camb": {},
}


def test_params_from_dict_derives_omega_b_and_omega_cdm():
    params = io.params_from_dict(RAW_PARAMS)
    h = RAW_PARAMS["Cosmology"]["H0"] / 100.0
    expected_omega_b = RAW_PARAMS["Cosmology"]["OmegaBar"] * h * h
    expected_omega_cdm = (RAW_PARAMS["Cosmology"]["OmegaM"] - RAW_PARAMS["Cosmology"]["OmegaBar"]) * h * h
    assert params["Cosmology"]["h"] == pytest.approx(h)
    assert params["Cosmology"]["omega_b"] == pytest.approx(expected_omega_b)
    assert params["Cosmology"]["omega_cdm"] == pytest.approx(expected_omega_cdm)
    assert params["Cosmology"]["OmegaLambda"] == pytest.approx(1.0 - RAW_PARAMS["Cosmology"]["OmegaM"])


def test_params_from_dict_carries_backend_mode_through():
    params = io.params_from_dict(RAW_PARAMS)
    assert params["Code"]["mode"] == "camb"


def test_params_from_dict_defaults_dm_model_to_cdm():
    params = io.params_from_dict(RAW_PARAMS)
    assert params["Code"]["dm_model"] == "cdm"
    assert params["Code"]["dm_model_mass"] is None


def test_params_from_dict_rejects_empty():
    with pytest.raises(ValueError, match="empty or invalid"):
        io.params_from_dict({})


def test_params_from_dict_rejects_non_dict():
    with pytest.raises(ValueError, match="valid dictionary"):
        io.params_from_dict(["not", "a", "dict"])


def test_params_from_dict_rejects_missing_run_key():
    with pytest.raises(KeyError, match="'Run'"):
        io.params_from_dict({"Cosmology": {}})


def test_params_from_dict_rejects_missing_cosmology_key():
    with pytest.raises(KeyError, match="'Cosmology'"):
        io.params_from_dict({"Run": {"mode": "camb"}})


def test_params_from_dict_rejects_missing_backend_block():
    bad = {"Run": {"mode": "camb"}, "Cosmology": RAW_PARAMS["Cosmology"]}
    with pytest.raises(KeyError, match="camb"):
        io.params_from_dict(bad)


def test_params_from_dict_user_mode_requires_pk_file():
    bad = {"Run": {"mode": "user"}, "Cosmology": RAW_PARAMS["Cosmology"], "user": {}}
    with pytest.raises(KeyError, match="pk_file"):
        io.params_from_dict(bad)


def test_params_from_dict_user_mode_carries_pk_file_through():
    good = {
        "Run": {"mode": "user"},
        "Cosmology": RAW_PARAMS["Cosmology"],
        "user": {"pk_file": "some/table.txt"},
    }
    params = io.params_from_dict(good)
    assert params["Code"]["mode"] == "user"
    assert params["Code"]["USER"]["pk_file"] == "some/table.txt"


def test_params_from_dict_rejects_unknown_mode():
    bad = {"Run": {"mode": "not_a_real_mode"}, "Cosmology": RAW_PARAMS["Cosmology"], "not_a_real_mode": {}}
    with pytest.raises(ValueError, match="unknown Run.mode"):
        io.params_from_dict(bad)


def test_get_params_missing_file_raises_actionable_error(tmp_path):
    missing = tmp_path / "does_not_exist.yml"
    with pytest.raises(FileNotFoundError, match="not found"):
        io.get_params(str(missing))


def test_get_params_loads_real_config_file():
    params = io.get_params("config/planck2018_camb.yml")
    assert params["Code"]["mode"] == "camb"
    assert params["Cosmology"]["H0"] == pytest.approx(67.556)


# ---------------------------------------------------------------------------
# collapse-barrier config: independent of dm_model
# ---------------------------------------------------------------------------


def test_params_from_dict_barrier_defaults_to_fixed_with_beta_0p15():
    code = io.params_from_dict(RAW_PARAMS)["Code"]
    assert code["barrier"] == "fixed"
    assert code["barrier_parameter"] == pytest.approx(0.15)


def test_params_from_dict_carries_barrier_settings_through_independently_of_dm_model():
    raw = {**RAW_PARAMS, "Run": {**RAW_PARAMS["Run"], "barrier": "linear", "barrier_parameter": 0.3, "dm_model": "cdm"}}
    code = io.params_from_dict(raw)["Code"]
    assert (code["barrier"], code["barrier_parameter"], code["dm_model"]) == ("linear", 0.3, "cdm")


def test_params_from_dict_rejects_unknown_barrier():
    raw = {**RAW_PARAMS, "Run": {**RAW_PARAMS["Run"], "barrier": "flat"}}
    with pytest.raises(ValueError, match="Unknown barrier"):
        io.params_from_dict(raw)


def test_params_from_dict_rejects_negative_beta_for_linear_barrier():
    raw = {**RAW_PARAMS, "Run": {**RAW_PARAMS["Run"], "barrier": "linear", "barrier_parameter": -0.1}}
    with pytest.raises(ValueError, match="barrier_parameter >= 0"):
        io.params_from_dict(raw)


def test_shared_default_config_uses_the_fixed_barrier():
    code = io.get_params("config/planck2018_camb.yml")["Code"]
    assert (code["barrier"], code["barrier_parameter"]) == ("fixed", pytest.approx(0.15))


def test_linear_barrier_config_differs_from_the_default_only_in_the_barrier():
    default = io.get_params("config/planck2018_camb.yml")
    linear = io.get_params("config/planck2018_linear_barrier.yml")
    assert (linear["Code"]["barrier"], linear["Code"]["barrier_parameter"]) == ("linear", pytest.approx(0.15))
    assert default["Cosmology"] == linear["Cosmology"]
    assert {k: v for k, v in default["Code"].items() if k != "barrier"} == {
        k: v for k, v in linear["Code"].items() if k != "barrier"
    }
