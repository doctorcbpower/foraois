"""
Figure style for the software-release paper.

Uses SciencePlots ('science' + 'bright') when it is installed (``pip install SciencePlots``, or the ``paper`` extra) and
falls back to a plain serif style otherwise. Figures are drawn at the size at which they are printed (COL = single
column, FULL = text width of the two-column class), so the font sizes set here are the printed sizes. Every figure is
saved as a vector PDF and a 300 dpi PNG.
"""
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

COL, FULL = 3.4, 7.1  # inches

BLUE, RED, GREEN, YELLOW, CYAN, PURPLE, GREY = "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"
PALETTE = [BLUE, RED, GREEN, YELLOW, CYAN, PURPLE, GREY]


def apply():
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science", "bright"])
        if not shutil.which("latex"):
            mpl.rcParams["text.usetex"] = False
    except Exception:
        plt.style.use("default")
        mpl.rcParams["font.family"] = "serif"
    mpl.rcParams.update({
        "axes.grid": False, "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "legend.title_fontsize": 7,
        "lines.linewidth": 1.2, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def stem_of(name):
    """'a/b.png' -> Path('a/b'); a bare stem is returned unchanged."""
    p = Path(name)
    return p.with_suffix("") if p.suffix in {".png", ".pdf"} else p


def save(fig, name):
    stem = stem_of(name)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    print(f"  saved -> {stem}.pdf / .png")


def m2_label(M2):
    """Panel label '$M_2 = 3.16 x 10^13 M_sun/h$' from a mass, without Python's 'e+13' formatting."""
    import numpy as np

    exp = int(np.floor(np.log10(M2)))
    mant = M2 / 10.0**exp
    body = rf"10^{{{exp}}}" if abs(mant - 1) < 1e-6 else rf"{mant:.2f}\times10^{{{exp}}}"
    return rf"$M_2={body}\,M_\odot/h$"


def sci(x, digits=2):
    """LaTeX 'a\\times10^{b}' for a positive number (no Python 'e+11' formatting in figure text)."""
    import numpy as np

    exp = int(np.floor(np.log10(x)))
    mant = x / 10.0**exp
    return rf"{mant:.{digits}f}\times10^{{{exp}}}"


def save_data(name, obj):
    """Pickle the inputs of a figure next to it (stem.pkl), so it can be restyled without rerunning."""
    import pickle

    with open(str(stem_of(name)) + ".pkl", "wb") as f:
        pickle.dump(obj, f)


def load_data(path):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)
