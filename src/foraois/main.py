import argparse
import time

import numpy as np

from foraois import PCHMergerTree, ZhangHuiMergerTree, build_constrained_tree, cosmo_utils
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
        "--algorithm",
        dest="algorithm",
        type=str,
        default="pch08",
        choices=["pch08", "zhang-hui", "constrained"],
        help=(
            "Tree-building algorithm: "
            '"pch08" (Parkinson, Cole & Helly 2008 fitted rate, default), '
            '"zhang-hui" (binary-per-step trees from the Zhang & Hui 2006 first-crossing distribution), '
            '"constrained" (Nadler et al. 2023 Brownian-bridge-constrained '
            "branch guaranteed to reach --M1 at --z1)"
        ),
    )
    parser.add_argument(
        "--backend",
        dest="backend",
        type=str,
        default="numpy",
        choices=["serial", "numpy", "numba"],
        help=(
            "Tree-building backend (ignored for --algorithm constrained, which "
            "has no vectorised backend yet -- see ROADMAP.md): "
            '"serial" (single-tree loop, for testing), '
            '"numpy" (vectorised, no extra deps, default), '
            '"numba" (JIT + parallel, fastest for large N, needs '
            "`pip install foraois[numba]`)"
        ),
    )
    parser.add_argument(
        "--M1",
        dest="M1",
        type=float,
        default=1.0e11,
        help="Constraint mass at --z1, Msun/h (only used for --algorithm constrained; default: 1e11)",
    )
    parser.add_argument(
        "--z1",
        dest="z1",
        type=float,
        default=4.0,
        help="Constraint redshift (only used for --algorithm constrained; default: 4.0)",
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
    # Tree parameters
    # ------------------------------------------------------------------
    M0 = 1.0e12  # Msun/h — halo mass at z=0
    z0 = 0.0
    # M_res=1e4 with dz=0.1 previously gave Nupper (expected splits per
    # step, see foraois.diagnostics.expected_splits_per_step) in the
    # millions -- PCH08's single-split-per-step architecture targets
    # Nupper~0.1; an 8-order-of-magnitude M0/M_res ratio at this dz was
    # never a valid regime for it. M_res=1e10, dz=0.005 keeps Nupper < 0.11
    # across the whole z0..z_max=15 range for M0=1e12 -- used for PCH08
    # (any backend) and ZhangHuiMergerTree.build_forest_numpy (both cheap:
    # PCH08's rate is a closed-form-per-step formula, and build_forest_numpy
    # uses the closed-form Levy sampler for flat barriers, not a numerical
    # solve -- see zhang_hui_trees.py's own docstring).
    M_res = 1.0e10  # Msun/h — mass resolution
    N = args.n_trees
    dm_model = run_params["Code"].get("dm_model", "cdm")

    algorithm = args.algorithm
    backend = args.backend
    print(f"\nBuilding {N:,} trees with algorithm='{algorithm}', backend='{backend}' ...")

    # ZhangHuiMergerTree.build_tree (unlike build_forest_numpy) calls
    # first_crossing_step's O(N_grid^2)-per-step numerical solve at every
    # step -- genuinely expensive per tree, unlike every other path here.
    # This affects both --algorithm zhang-hui --backend serial directly,
    # and --algorithm constrained's unconstrained continuation beyond z1
    # (build_constrained_tree grafts a ZhangHuiMergerTree.build_tree call
    # onto the constrained branch -- see zhang_hui_constrained_trees.py).
    # A coarser dz/z_max/N_grid than PCH08's above keeps a handful of
    # trees tractable; this is not a backend for bulk (N large) generation
    # the way build_forest_numpy is -- warned about explicitly below.
    zh_build_tree_z_max = 8.0
    zh_build_tree_dz = 0.2
    zh_build_tree_N_grid = 60

    if algorithm == "zhang-hui" and backend == "serial" and N > 20:
        print(
            f"  Note: --algorithm zhang-hui --backend serial calls an O(N_grid^2) "
            f"solve at every step -- building {N:,} trees this way will be slow. "
            "Use --backend numpy for bulk generation (closed-form, no such cost)."
        )
    if algorithm == "constrained" and N > 20:
        print(
            f"  Note: --algorithm constrained has no vectorised backend yet (see "
            f"ROADMAP.md) -- building {N:,} trees serially will be slow."
        )

    # ------------------------------------------------------------------
    # Constrained trees (Nadler et al. 2023): a structurally different
    # operation from PCH08/Zhang-Hui's build_tree/build_forest_numpy --
    # build_constrained_tree grows one branch at a time (no vectorised
    # forest backend exists yet, see ROADMAP.md), so this loops serially
    # regardless of --backend.
    # ------------------------------------------------------------------
    if algorithm == "constrained":
        M1, z1 = args.M1, args.z1
        z_max = zh_build_tree_z_max
        # build_constrained_tree needs cosmology_data's sigma(M) grid, which
        # PCHMergerTree/ZhangHuiMergerTree's own __init__ builds as a side
        # effect (see their docstrings) -- neither is constructed on this
        # path, so it must be built explicitly here instead.
        cosmology_data._prepare_sigma_grid(pk_data)
        if M1 >= M0:
            raise ValueError(f"--M1={M1} must be < M0={M0}.")
        if z1 <= z0 or z1 >= z_max:
            raise ValueError(f"--z1={z1} must satisfy z0={z0} < z1 < z_max={z_max} (see --z1's own help).")

        rng = np.random.default_rng()
        t0 = time.perf_counter()
        trees = [
            build_constrained_tree(
                M0,
                z0,
                M1,
                z1,
                z_max,
                M_res,
                cosmology_data,
                model=dm_model,
                rng=rng,
                dz=zh_build_tree_dz,
                N_grid=zh_build_tree_N_grid,
            )
            for _ in range(N)
        ]
        elapsed = time.perf_counter() - t0
        print(f"Constrained (serial loop): {elapsed:.2f} s for {N:,} trees ({elapsed / N * 1e3:.2f} ms / tree)")
        print(f"  each tree guaranteed to reach M1={M1:.3e} Msun/h at z1={z1:g}")

        # Single-tree mass history (same tree-of-dicts shape build_tree
        # uses, so the same plotting function works unchanged).
        plot.plot_merger_history(trees[0], file_name="merger_history_constrained")
        return

    # ------------------------------------------------------------------
    # PCH08 / Zhang-Hui: both satisfy the same TreeAlgorithm interface
    # (build_tree/build_forest_numpy, identical signatures and return
    # shapes -- see tree_algorithm.py), and both now also provide
    # build_forest_numba, so the backend-dispatch logic below is shared
    # between them unconditionally.
    # ------------------------------------------------------------------

    # build_forest_numpy/build_forest_numba are cheap for both algorithms (closed-form for
    # Zhang-Hui, see above) so z_max=15/dz=0.005 is fine there; only
    # zhang-hui's *serial* build_tree needs the coarser parameters.
    z_max = zh_build_tree_z_max if (algorithm == "zhang-hui" and backend == "serial") else 15.0
    dz = zh_build_tree_dz if (algorithm == "zhang-hui" and backend == "serial") else 0.005

    t0 = time.perf_counter()
    if algorithm == "pch08":
        tree_generator = PCHMergerTree(cosmology_data, run_params)
    else:
        tree_generator = ZhangHuiMergerTree(cosmology_data, run_params, model=dm_model, N_grid=zh_build_tree_N_grid)
    print(f"{type(tree_generator).__name__} initialised (sigma grid) in {time.perf_counter() - t0:.2f} s")

    if backend == "serial":
        # ---- Single-tree loop (kept for debugging/testing) --------------
        t0 = time.perf_counter()
        trees = []
        for _ in range(N):
            trees.append(tree_generator.build_tree(M0=M0, z0=z0, z_max=z_max, M_res=M_res, dz=dz))
        elapsed = time.perf_counter() - t0
        print(f"Serial:  {elapsed:.2f} s for {N:,} trees ({elapsed / N * 1e3:.2f} ms / tree)")

        # Single-tree mass history (list-of-dicts format from build_tree)
        plot.plot_merger_history(trees[0], file_name=f"merger_history_serial_{algorithm}")

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
            file_name=f"mass_tracks_numpy_{algorithm}",
        )

        # 2. Merger rate dN/dz per tree
        plot.plot_merger_rate(
            split_events,
            z_steps,
            N_trees=N,
            file_name=f"merger_rate_numpy_{algorithm}",
        )

        # 3. Progenitor mass function at several redshifts
        plot.plot_mass_function(
            mass_history,
            z_steps,
            z_targets=[0.5, 1.0, 2.0, 3.0],
            M_res=M_res,
            file_name=f"mass_function_numpy_{algorithm}",
        )

        # 4. Branching diagram for tree 0
        plot.plot_tree_graph(
            mass_history,
            z_steps,
            tree_idx=0,
            M_res=M_res,
            split_events=split_events,
            file_name=f"tree_graph_numpy_{algorithm}",
        )

        # 5. All-in-one 2×2 summary panel
        plot.plot_forest_summary(
            mass_history,
            split_events,
            z_steps,
            M_res=M_res,
            n_show=150,
            file_name=f"forest_summary_numpy_{algorithm}",
        )

    elif backend == "numba":
        # ---- Numba JIT + parallel -----------------------------------------
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
            n_show=150,
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

    # The exact Zhang-Hui algorithm instead of PCH08's fitted rate
    python main.py --params_file params.yaml --n_trees 1000 --algorithm zhang-hui

    # Brownian-bridge-constrained trees, guaranteed to reach M1 at z1
    python main.py --params_file params.yaml --n_trees 100 --algorithm constrained --M1 1e11 --z1 4.0
    """
    main()
