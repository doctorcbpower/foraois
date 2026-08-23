import argparse
import time

import numpy as np

from foraois import PCHMergerTree, cosmo_utils
from foraois.utils import io, plot


def main():
    parser = argparse.ArgumentParser(description="Generate Monte Carlo merger trees.")
    parser.add_argument(
        "--params_file",
        dest="params_file",
        type=str,
        required=True,
        help="Path to YAML parameters file",
    )
    parser.add_argument(
        "--n_trees",
        dest="n_trees",
        type=int,
        default=1000,
        help="Number of trees to generate (default: 1000)",
    )
    parser.add_argument(
        "--backend",
        dest="backend",
        type=str,
        default="numpy",
        choices=["serial", "numpy", "numba"],
        help=(
            "Tree-building backend: "
            '"serial" (original single-tree loop, for testing), '
            '"numpy" (vectorised, no extra deps, default), '
            '"numba" (JIT + parallel, fastest for large N, needs '
            "`pip install foraois[numba]`)"
        ),
    )
    args = parser.parse_args()

    run_params = io.get_params(args.params_file)

    # ------------------------------------------------------------------
    # Cosmology setup
    # Note: CosmoData.__init__ now calls precompute_delta_col_table()
    # automatically, so delta_col lookups are always fast.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    cosmology_data = cosmo_utils.CosmoData(run_params, redshift=[0.0])
    print(f"CosmoData initialised in {time.perf_counter() - t0:.2f} s (includes delta_col table)")

    print(f"Using '{run_params['Code'].get('mode')}' power spectrum")

    pk_data = cosmology_data.get_power_spectrum()

    if run_params["Code"]["plot_pk"]:
        axes_limits = {
            "xmin": run_params["Code"].get("pk_xmin", 1.0e-4),
            "xmax": run_params["Code"].get("pk_xmax", 3.0e0),
            "ymin": run_params["Code"].get("pk_ymin", 3.0e-0),
            "ymax": run_params["Code"].get("pk_ymax", 4.0e4),
        }
        file_name = run_params["Code"].get("pk_file_name", None)
        plot.plot_power_spectrum(pk_data, file_name, axes_limits)

    cosmology_data.sigma8 = cosmology_data.get_mass_variance(pk_data, radius=8.0, window_function_type="top_hat")
    print(f"sigma_8 = {np.sqrt(cosmology_data.sigma8):.4f}")

    if run_params["Code"]["plot_mvar"]:
        axes_limits = {"xmin": 0.1, "xmax": 20.0, "ymin": 0.0, "ymax": 20.0}
        file_name = run_params["Code"].get("mvar_file_name", None)
        plot.plot_mass_variance(cosmology_data, pk_data, file_name, axes_limits)

    growth_factor, f_growth_factor, delta_col = cosmology_data.get_linear_growth_and_collapse(redshift=[0.0, 1.0])
    print(f"Linear growth factor:       {growth_factor}")
    print(f"Dimensionless growth rate:  {f_growth_factor}")
    print(f"Critical collapse threshold:{delta_col}")

    # ------------------------------------------------------------------
    # Tree generator
    # PCHMergerTree.__init__ calls _prepare_sigma_grid once, which builds
    # the fast scipy interpolants for sigma(M) and d log sigma / d log M.
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    tree_generator = PCHMergerTree(cosmology_data, run_params)
    print(f"PCHMergerTree initialised (sigma grid) in {time.perf_counter() - t0:.2f} s")

    # ------------------------------------------------------------------
    # Tree parameters
    # ------------------------------------------------------------------
    M0 = 1.0e12  # Msun/h — halo mass at z=0
    z0 = 0.0
    z_max = 15.0
    M_res = 1.0e4  # Msun/h — mass resolution
    dz = 0.1
    N = args.n_trees

    # ------------------------------------------------------------------
    # Run in the chosen backend
    # ------------------------------------------------------------------
    backend = args.backend
    print(f"\nBuilding {N:,} trees with backend='{backend}' ...")

    if backend == "serial":
        # ---- Original serial loop (kept for regression testing) --------
        t0 = time.perf_counter()
        trees = []
        for _ in range(N):
            trees.append(tree_generator.build_tree(M0=M0, z0=z0, z_max=z_max, M_res=M_res, dz=dz))
        elapsed = time.perf_counter() - t0
        print(f"Serial:  {elapsed:.2f} s for {N:,} trees ({elapsed / N * 1e3:.2f} ms / tree)")

        # Single-tree mass history (list-of-dicts format from build_tree)
        plot.plot_merger_history(trees[0], file_name="merger_history_serial")

    elif backend == "numpy":
        # ---- Vectorised NumPy (no Numba) --------------------------------
        M0_array = np.full(N, M0)
        t0 = time.perf_counter()
        mass_history, split_events, z_steps, _smooth_accretion, _merger_mass = tree_generator.build_forest_numpy(
            M0_array=M0_array, z0=z0, z_max=z_max, M_res=M_res, dz=dz
        )
        elapsed = time.perf_counter() - t0
        print(f"NumPy:   {elapsed:.2f} s for {N:,} trees ({elapsed / N * 1e6:.1f} µs / tree)")
        print(f"  mass_history shape: {mass_history.shape}")
        print(f"  total merger events recorded: {sum(e['n_splits'] for e in split_events):,}")

        n_alive = int((mass_history[:, -1] >= M_res).sum())
        print(f"  trees alive at z={z_steps[-1]:.1f}: {n_alive:,} / {N:,} ({100 * n_alive / N:.1f}%)")

        # ---- Plots -------------------------------------------------------
        print("\nGenerating plots ...")

        # 1. Mass tracks for a random subset of trees
        plot.plot_mass_history(
            mass_history,
            z_steps,
            n_show=200000,
            M_res=M_res,
            file_name="mass_tracks_numpy",
        )

        # 2. Merger rate dN/dz per tree
        plot.plot_merger_rate(
            split_events,
            z_steps,
            N_trees=N,
            file_name="merger_rate_numpy",
        )

        # 3. Progenitor mass function at several redshifts
        plot.plot_mass_function(
            mass_history,
            z_steps,
            z_targets=[0.5, 1.0, 2.0, 3.0],
            M_res=M_res,
            file_name="mass_function_numpy",
        )

        # 4. Branching diagram for tree 0
        plot.plot_tree_graph(
            mass_history,
            z_steps,
            tree_idx=0,
            M_res=M_res,
            split_events=split_events,
            file_name="tree_graph_numpy",
        )

        # 5. All-in-one 2×2 summary panel
        plot.plot_forest_summary(
            mass_history,
            split_events,
            z_steps,
            M_res=M_res,
            N_show=150,
            file_name="forest_summary_numpy",
        )

    elif backend == "numba":
        # ---- Numba JIT + parallel ----------------------------------------
        M0_array = np.full(N, M0)

        # Warm-up / compile pass (small N so it's fast)
        print("  Triggering Numba JIT compilation (warm-up) ...")
        t_warmup = time.perf_counter()
        tree_generator.build_forest_numba(M0_array=np.full(10, M0), z0=z0, z_max=z_max, M_res=M_res, dz=dz)
        print(f"  JIT warm-up: {time.perf_counter() - t_warmup:.2f} s")

        # Timed run
        t0 = time.perf_counter()
        mass_history, z_steps, _smooth_accretion, _merger_mass = tree_generator.build_forest_numba(
            M0_array=M0_array, z0=z0, z_max=z_max, M_res=M_res, dz=dz
        )
        elapsed = time.perf_counter() - t0
        print(f"Numba:   {elapsed:.2f} s for {N:,} trees ({elapsed / N * 1e6:.1f} µs / tree)")
        print(f"  mass_history shape: {mass_history.shape}")

        n_alive = int((mass_history[:, -1] >= M_res).sum())
        print(f"  trees alive at z={z_steps[-1]:.1f}: {n_alive:,} / {N:,} ({100 * n_alive / N:.1f}%)")

        # ---- Plots -------------------------------------------------------
        # Note: build_forest_numba does not return split_events (too expensive
        # inside @njit).  We skip the merger-rate and tree-graph plots here,
        # or re-run a small chunk with the numpy backend to get split_events.
        print("\nGenerating plots ...")

        plot.plot_mass_history(
            mass_history,
            z_steps,
            n_show=200,
            M_res=M_res,
            file_name="mass_tracks_numba",
        )

        plot.plot_mass_function(
            mass_history,
            z_steps,
            z_targets=[0.5, 1.0, 2.0, 3.0],
            M_res=M_res,
            file_name="mass_function_numba",
        )

        # Run a small numpy pass to get split_events for the remaining plots
        print("  Running small numpy pass for merger-event data ...")
        n_sample = min(N, 5000)
        _, split_events_sample, _, _, _ = tree_generator.build_forest_numpy(
            M0_array=np.full(n_sample, M0),
            z0=z0,
            z_max=z_max,
            M_res=M_res,
            dz=dz,
        )

        plot.plot_merger_rate(
            split_events_sample,
            z_steps,
            N_trees=n_sample,
            file_name="merger_rate_numba",
        )

        plot.plot_tree_graph(
            mass_history,
            z_steps,
            tree_idx=0,
            M_res=M_res,
            split_events=split_events_sample,
            file_name="tree_graph_numba",
        )

        plot.plot_forest_summary(
            mass_history,
            split_events_sample,
            z_steps,
            M_res=M_res,
            N_show=150,
            file_name="forest_summary_numba",
        )


if __name__ == "__main__":
    """
    Usage examples
    --------------
    # 1 000 trees with vectorised NumPy (default, no extra deps)
    python main.py --params_file params.yaml --n_trees 1000

    # 1 000 000 trees with Numba + all CPU cores (needs `pip install foraois[numba]`)
    python main.py --params_file params.yaml --n_trees 1000000 --backend numba

    # Single-tree serial loop for debugging
    python main.py --params_file params.yaml --n_trees 100 --backend serial
    """
    main()
