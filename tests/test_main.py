"""
Smoke tests for foraois.main's CLI.

Nothing else in the test suite exercised main.py at all -- that's exactly
how two real bugs went undetected until manual end-to-end runs surfaced
them: plot_forest_summary() being called with the wrong keyword (N_show
instead of n_show, crashing every --backend numpy/numba run before its
final plot), and build_constrained_tree being called before
cosmology_data._prepare_sigma_grid() had ever run (crashing every
--algorithm constrained run immediately). These tests use the smallest
practical --n_trees -- --algorithm zhang-hui --backend serial and
--algorithm constrained are both genuinely slow per tree (see main.py's
own comments on why), not something to run at scale here.
"""

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless, no display needed for tests

from foraois.main import main  # noqa: E402

CONFIG = str(Path(__file__).resolve().parents[1] / "config" / "planck2018_camb.yml")

# main.py's zhang-hui/constrained paths use N_grid=60 (fast, not production
# accuracy -- see main.py's own comments) at a mass/z combination that
# triggers first_crossing_step's grid-resolution warning -- expected here,
# same rationale/suppression as tests/test_zhang_hui_trees.py.
pytestmark = pytest.mark.filterwarnings("ignore:first_crossing_step's grid spacing.*:UserWarning")


def _run(monkeypatch, tmp_path, extra_args):
    monkeypatch.chdir(tmp_path)  # plots are written relative to CWD
    monkeypatch.setattr(sys, "argv", ["foraois-run", "--params_file", CONFIG, "--n_trees", "2", *extra_args])
    main()


@pytest.mark.parametrize(
    "algorithm,backend",
    [
        ("pch08", "serial"),
        ("pch08", "numpy"),
        ("pch08", "numba"),
        ("zhang-hui", "serial"),
        ("zhang-hui", "numpy"),
        ("zhang-hui", "numba"),
    ],
)
def test_cli_runs_without_error(monkeypatch, tmp_path, algorithm, backend):
    _run(monkeypatch, tmp_path, ["--algorithm", algorithm, "--backend", backend])


def test_cli_constrained_runs_without_error(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, ["--algorithm", "constrained", "--M1", "1e11", "--z1", "4.0"])


def test_cli_constrained_rejects_M1_above_M0(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match=r"--M1"):
        _run(monkeypatch, tmp_path, ["--algorithm", "constrained", "--M1", "2e12", "--z1", "4.0"])


def test_cli_constrained_rejects_z1_out_of_range(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match=r"--z1"):
        _run(monkeypatch, tmp_path, ["--algorithm", "constrained", "--M1", "1e11", "--z1", "50.0"])
