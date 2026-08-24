#!/usr/bin/env python3
"""
Benchmark PCHMergerTree's three tree-building backends (serial, numpy,
numba) plus the unconstrained Zhang-Hui backend's vectorized (numpy)
implementation, across a range of forest sizes N, for the Software
Release Paper's Performance section (this is the single source-of-truth
script for both the wall-clock figure the paper embeds and the
preliminary PCH08-vs-Zhang-Hui speed comparison quoted in the text).

Uses a synthetic power-law P(k) (same pattern as tests/conftest.py) so this
runs with no CLASS/CAMB dependency and no network access -- the point is
backend scaling behaviour, not any particular cosmology.

Usage
-----
    python scripts/benchmark_backends.py [--output benchmark.png] [--csv benchmark.csv] [--repeats N]

--repeats > 1 re-times each (N, backend) point within the same warm process
and reports/plots the mean and standard deviation rather than a single-run
value -- use this to settle close calls (e.g. Zhang-Hui numpy vs. PCH08
numba, which have been observed to swap ordering between individual runs)
rather than quoting whichever single run happened to be run first.

Serial is capped at a much smaller N than the vectorized backends (see
MAX_N_SERIAL below) -- it calls build_tree() once per halo in a Python
loop and is ~1000x slower per tree, so timing it out to the same N as the
vectorised backends would take hours for no additional information (its
asymptotic per-tree cost doesn't change with N; only the vectorised
backends' per-tree cost does, which is the actual scaling result worth
plotting).
"""

import argparse
import csv
import time

import numpy as np

from foraois.cosmo_utils import CosmoData
from foraois.pch_trees import PCHMergerTree
from foraois.zhang_hui_trees import ZhangHuiMergerTree

PLANCK_LIKE = {
    "Code": {"mode": "camb", "pk_kmin": 1e-4, "pk_kmax": 10.0, "pk_npoints": 500},
    "Cosmology": {"H0": 67.66, "OmegaM": 0.3111, "OmegaK": 0.0, "OmegaLambda": 0.6889},
}

M0 = 1.0e12  # Msun/h
Z0, Z_MAX, DZ = 0.0, 5.0, 0.2
M_RES = 1.0e9  # Msun/h

N_VALUES = [10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
MAX_N_SERIAL = 300  # see module docstring


def build_tree_generators(seed=1):
    cosmo_data = CosmoData(PLANCK_LIKE, redshift=[0.0])
    k = np.logspace(-4, 1, 500)
    Pk = 2.0e4 * k**-2.0
    pk_data = {"k": k, "Pk": Pk.reshape(1, -1), "z": [0.0]}
    cosmo_data.get_power_spectrum = lambda: pk_data
    pch = PCHMergerTree(cosmo_data, PLANCK_LIKE)
    zh = ZhangHuiMergerTree(cosmo_data, PLANCK_LIKE, model="cdm", rng=np.random.default_rng(seed))
    return pch, zh


def time_serial(tree_generator, n):
    t0 = time.perf_counter()
    for _ in range(n):
        tree_generator.build_tree(M0=M0, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    return time.perf_counter() - t0


def time_numpy(tree_generator, n):
    M0_array = np.full(n, M0)
    t0 = time.perf_counter()
    tree_generator.build_forest_numpy(M0_array=M0_array, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    return time.perf_counter() - t0


def time_numba(tree_generator, n, warm=False):
    M0_array = np.full(n, M0)
    if not warm:
        # JIT compilation (~2-5s) is a one-time cost, not part of per-N
        # scaling -- warm the cache with a tiny call first, timed separately.
        tree_generator.build_forest_numba(M0_array=np.full(10, M0), z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    t0 = time.perf_counter()
    tree_generator.build_forest_numba(M0_array=M0_array, z0=Z0, z_max=Z_MAX, M_res=M_RES, dz=DZ)
    return time.perf_counter() - t0


def run_benchmark(repeats=1):
    """Time serial/numpy/numba PCH08 and numpy Zhang-Hui across N_VALUES; return the results dict.

    With repeats > 1, each (N, backend) timing is repeated independently
    within a single warm process (JIT compilation is a one-time cost, timed
    separately, not repeated) and the reported/plotted value is the mean
    over repeats, with the per-repeat standard deviation also printed --
    so run-to-run variance (e.g. numba JIT/scheduling noise) is reported
    honestly rather than quoted from a single run.
    """
    pch, zh = build_tree_generators()

    print("Warming up Numba JIT cache...")
    t_warmup = time.perf_counter()
    time_numba(pch, 10, warm=False)
    print(f"  JIT warm-up: {time.perf_counter() - t_warmup:.2f} s (one-time cost, excluded below)\n")

    results = {"N": [], "serial": [], "numpy": [], "numba": [], "zh_numpy": []}
    results_std = {"serial": [], "numpy": [], "numba": [], "zh_numpy": []}

    for n in N_VALUES:
        row = {"serial": [], "numpy": [], "numba": [], "zh_numpy": []}

        for _ in range(repeats):
            row["serial"].append(time_serial(pch, n) if n <= MAX_N_SERIAL else None)
            row["numpy"].append(time_numpy(pch, n))
            row["numba"].append(time_numba(pch, n, warm=True))
            row["zh_numpy"].append(time_numpy(zh, n))

        def _fmt(key):
            vals = [v for v in row[key] if v is not None]
            if not vals:
                return "   n/a  "
            mean = np.mean(vals)
            if repeats > 1:
                return f"{mean:.4f}±{np.std(vals):.4f}s ({mean / n * 1e6:.2f} us/tree)"
            return f"{mean:.4f}s ({mean / n * 1e6:.2f} us/tree)"

        print(
            f"N={n:>7,}  serial={_fmt('serial')}  numpy={_fmt('numpy')}  "
            f"numba={_fmt('numba')}  zh_numpy={_fmt('zh_numpy')}"
        )

        results["N"].append(n)
        for key in ("serial", "numpy", "numba", "zh_numpy"):
            vals = [v for v in row[key] if v is not None]
            results[key].append(float(np.mean(vals)) if vals else None)
            results_std[key].append(float(np.std(vals)) if len(vals) > 1 else 0.0)

    results["_std"] = results_std
    results["_repeats"] = repeats
    return results


def write_csv(results, csv_path):
    std = results.get("_std", {"serial": [], "numpy": [], "numba": [], "zh_numpy": []})
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "serial_s",
                "numpy_s",
                "numba_s",
                "zh_numpy_s",
                "serial_std_s",
                "numpy_std_s",
                "numba_std_s",
                "zh_numpy_std_s",
                "repeats",
            ]
        )
        for i in range(len(results["N"])):
            writer.writerow(
                [
                    results["N"][i],
                    results["serial"][i],
                    results["numpy"][i],
                    results["numba"][i],
                    results["zh_numpy"][i],
                    std["serial"][i] if std["serial"] else "",
                    std["numpy"][i] if std["numpy"] else "",
                    std["numba"][i] if std["numba"] else "",
                    std["zh_numpy"][i] if std["zh_numpy"] else "",
                    results.get("_repeats", 1),
                ]
            )


def build_figure(results, output_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5))
    N_arr = np.array(results["N"])
    std = results.get("_std", {})
    repeats = results.get("_repeats", 1)

    def _std_or_none(key):
        if repeats > 1 and std.get(key):
            return np.array(std[key])
        return None

    serial_mask = np.array([v is not None for v in results["serial"]])
    if serial_mask.any():
        serial_std = _std_or_none("serial")
        ax.errorbar(
            N_arr[serial_mask],
            np.array(results["serial"])[serial_mask],
            yerr=serial_std[serial_mask] if serial_std is not None else None,
            fmt="o-",
            color="#d62728",
            capsize=2,
            label="PCH08 serial",
        )

    ax.errorbar(
        N_arr, results["numpy"], yerr=_std_or_none("numpy"), fmt="s-", color="#1f77b4", capsize=2, label="PCH08 numpy"
    )
    ax.errorbar(
        N_arr,
        results["numba"],
        yerr=_std_or_none("numba"),
        fmt="^-",
        color="#2ca02c",
        capsize=2,
        label="PCH08 numba (JIT warm)",
    )
    ax.errorbar(
        N_arr,
        results["zh_numpy"],
        yerr=_std_or_none("zh_numpy"),
        fmt="d-",
        color="#9467bd",
        capsize=2,
        label="Zhang-Hui numpy (closed-form)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$ trees")
    ax.set_ylabel("wall time [s]")
    title = f"Backend scaling ($z_0={Z0}$-$z_{{\\rm max}}={Z_MAX}$, $dz={DZ}$)"
    if repeats > 1:
        title += f", mean $\\pm$ std of {repeats} repeats"
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="benchmark_backends.png")
    parser.add_argument("--csv", default="benchmark_backends.csv")
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each (N, backend) timing this many times within one warm process and report the mean "
        "and standard deviation, rather than a single-run value -- use this to settle close calls "
        "(e.g. Zhang-Hui numpy vs. PCH08 numba) that can flip sign between individual runs.",
    )
    args = parser.parse_args()

    results = run_benchmark(repeats=args.repeats)

    write_csv(results, args.csv)
    print(f"\nWrote {args.csv}")

    build_figure(results, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
