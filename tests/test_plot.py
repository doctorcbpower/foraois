"""
Smoke tests for foraois.utils.plot -- these just check every function
runs without error and produces a non-empty PNG, using the synthetic
power-law P(k) fixtures (no CLASS/CAMB required). They don't inspect pixel
content; that's exercised visually via notebooks/foraois_demo.ipynb.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless, no display needed for tests

import numpy as np
import pytest

from foraois.utils import plot

M0 = 1.0e12
Z0 = 0.0
Z_MAX = 5.0
M_RES = 1.0e9
DZ = 0.2


def _assert_png_written(path):
    full = f"{path}.png"
    assert os.path.exists(full)
    assert os.path.getsize(full) > 0
    os.remove(full)


def test_plot_power_spectrum(synthetic_pk_data, tmp_path):
    out = str(tmp_path / "pk")
    plot.plot_power_spectrum(
        synthetic_pk_data,
        file_name=out,
        axes_limits={"xmin": 1e-4, "xmax": 1e1, "ymin": 1e-2, "ymax": 1e5},
    )
    _assert_png_written(out)


def test_plot_mass_variance(cosmo_data, synthetic_pk_data, tmp_path):
    out = str(tmp_path / "mvar")
    plot.plot_mass_variance(
        cosmo_data,
        synthetic_pk_data,
        file_name=out,
        axes_limits={"xmin": 0.1, "xmax": 20.0, "ymin": 0.0, "ymax": 20.0},
    )
    _assert_png_written(out)


def test_plot_merger_history(tree_generator, tmp_path):
    np.random.seed(1)
    tree = tree_generator.build_tree(M0=M0, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    out = str(tmp_path / "merger_history")
    plot.plot_merger_history(tree, file_name=out)
    _assert_png_written(out)


def test_plot_forest_summary(tree_generator, tmp_path):
    np.random.seed(2)
    mass_history, split_events, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(500, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    out = str(tmp_path / "forest_summary")
    plot.plot_forest_summary(mass_history, split_events, z_steps, M_res=M_RES, file_name=out)
    _assert_png_written(out)


def test_plot_mass_history(tree_generator, tmp_path):
    np.random.seed(7)
    mass_history, _, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(500, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    out = str(tmp_path / "mass_history")
    plot.plot_mass_history(mass_history, z_steps, n_show=50, M_res=M_RES, file_name=out)
    _assert_png_written(out)


def test_plot_merger_rate(tree_generator, tmp_path):
    np.random.seed(9)
    _, split_events, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(500, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    out = str(tmp_path / "merger_rate")
    plot.plot_merger_rate(split_events, z_steps, N_trees=500, file_name=out)
    _assert_png_written(out)


def test_plot_mass_function(tree_generator, tmp_path):
    np.random.seed(10)
    mass_history, _, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(2000, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    out = str(tmp_path / "mass_function")
    plot.plot_mass_function(mass_history, z_steps, z_targets=[0.5, 1.0], M_res=M_RES, file_name=out)
    _assert_png_written(out)


def test_plot_tree_graph(tree_generator, tmp_path):
    np.random.seed(3)
    mass_history, split_events, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(50, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    out = str(tmp_path / "tree_graph")
    plot.plot_tree_graph(
        mass_history,
        z_steps,
        tree_idx=0,
        M_res=M_RES,
        split_events=split_events,
        file_name=out,
    )
    _assert_png_written(out)


def test_plot_dendrogram(tree_generator, cosmo_data, tmp_path):
    np.random.seed(4)
    nodes = tree_generator.build_full_tree(M0=M0, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    out = str(tmp_path / "dendrogram")
    plot.plot_dendrogram(nodes, cosmo_data, file_name=out)
    _assert_png_written(out)


def test_plot_dendrogram_with_min_mass_display(tree_generator, cosmo_data, tmp_path):
    np.random.seed(5)
    nodes = tree_generator.build_full_tree(M0=M0, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    out = str(tmp_path / "dendrogram_pruned")
    plot.plot_dendrogram(nodes, cosmo_data, M_min_display=1e11, mark_node_id=0, file_name=out)
    _assert_png_written(out)


def test_plot_dendrogram_empty_after_pruning_does_not_crash(tree_generator, cosmo_data):
    np.random.seed(6)
    nodes = tree_generator.build_full_tree(M0=M0, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    # M0 itself is 1e12, so a threshold above it prunes everything, including
    # the root -- this must be handled gracefully (print + return), not crash
    plot.plot_dendrogram(nodes, cosmo_data, M_min_display=1e15, file_name=None)


def test_plot_excursion_trajectory(tree_generator, cosmo_data, tmp_path):
    np.random.seed(8)
    mass_history, _, z_steps, _, _ = tree_generator.build_forest_numpy(
        M0_array=np.full(20, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ
    )
    masses = np.concatenate([[M0], mass_history[0]])
    out = str(tmp_path / "excursion")
    plot.plot_excursion_trajectory(masses, z_steps, cosmo_data, file_name=out)
    _assert_png_written(out)


def test_plot_excursion_trajectory_too_few_points_does_not_crash(cosmo_data):
    # Fewer than 2 valid (mass > 0) points must be handled gracefully
    # (print + return), not crash on e.g. S[0] indexing.
    plot.plot_excursion_trajectory([1e12], [0.0], cosmo_data, file_name=None)
    plot.plot_excursion_trajectory([1e12, 0.0], [0.0, 1.0], cosmo_data, file_name=None)


def test_plot_dm_model_comparison(cosmo_data, synthetic_pk_data, tmp_path):
    from foraois import transfer_functions as tf
    from foraois.cosmo_utils import CosmoData

    cosmo_data._prepare_sigma_grid(synthetic_pk_data, logmass_min=6.0, logmass_max=15.0, dlogmass=0.5)

    # A second CosmoData instance with a "WDM-like" P(k) -- built by hand
    # (T_WDM applied to the same synthetic P(k)) rather than real CAMB, to
    # keep this test dependency-free like the rest of the suite. Same
    # minimal Cosmology block the cosmo_data fixture itself uses.
    params = {
        "Code": {"mode": "camb", "pk_kmin": 1e-4, "pk_kmax": 10.0, "pk_npoints": 500},
        "Cosmology": {
            "H0": 67.66,
            "OmegaM": 0.3111,
            "OmegaK": 0.0,
            "OmegaLambda": 0.6889,
        },
    }
    cosmo_wdm = CosmoData(params, redshift=[0.0])
    h = cosmo_wdm.cosmo_params.get("h", cosmo_wdm.cosmo_params["H0"] / 100.0)
    T = tf.T_WDM(synthetic_pk_data["k"], 3.0, cosmo_wdm.cosmo_params["OmegaM"], h)
    pk_wdm = {**synthetic_pk_data, "Pk": synthetic_pk_data["Pk"] * T[None, :] ** 2}
    cosmo_wdm._prepare_sigma_grid(pk_wdm, logmass_min=6.0, logmass_max=15.0, dlogmass=0.5)

    out = str(tmp_path / "dm_comparison")
    plot.plot_dm_model_comparison(
        {"CDM": (cosmo_data, synthetic_pk_data), "WDM": (cosmo_wdm, pk_wdm)},
        reference="CDM",
        file_name=out,
    )
    _assert_png_written(out)


def test_plot_dm_model_comparison_rejects_unknown_reference(cosmo_data, synthetic_pk_data):
    with pytest.raises(KeyError):
        plot.plot_dm_model_comparison({"CDM": (cosmo_data, synthetic_pk_data)}, reference="WDM")


def test_plot_branching_rate_validation(tree_generator, tmp_path):
    # Same (M2, M_res, delta0, d_omega) as
    # test_pch_validation.py::test_sampling_consistency_matches_quadrature,
    # so this is known to satisfy Nupper < 1 (required by
    # check_sampling_consistency, which this plot calls internally).
    out = str(tmp_path / "branching_rate_validation")
    result = plot.plot_branching_rate_validation(
        tree_generator,
        M2=1e12,
        M_res=1e10,
        delta0=1.68,
        d_omega=0.01,
        n_trials=20_000,
        seed=11,
        file_name=out,
    )
    _assert_png_written(out)
    assert result["passed"]
