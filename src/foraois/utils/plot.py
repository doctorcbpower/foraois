"""
foraois/utils/plot.py
-----------------------
Visualisation utilities for foraois.

Existing functions (signatures unchanged)
-----------------------------------------
plot_power_spectrum(pk_data, file_name, axes_limits)
plot_mass_variance(cosmology_data, pk_data, file_name, axes_limits)
plot_merger_history(tree, file_name)

New functions for forest output
--------------------------------
plot_mass_history(mass_history, z_steps, ...)
    Main-progenitor mass tracks M(z) for a sample of trees.

plot_merger_rate(split_events, z_steps, ...)
    Mean merger rate dN/dz per tree vs redshift.

plot_mass_function(mass_history, z_steps, ...)
    Progenitor mass function at one or more target redshifts.

plot_tree_graph(mass_history, z_steps, tree_idx, ...)
    Main-progenitor mass track for a single tree from a forest, coloured
    by mass. Only shows the branch build_forest_* actually tracks -- for
    the full multi-branch structure, use plot_dendrogram with
    PCHMergerTree.build_full_tree() output instead.

plot_forest_summary(mass_history, split_events, z_steps, ...)
    2x2 summary panel combining the four diagnostics above.

plot_dendrogram(nodes, cosmo_data, ...)
    Full branching-tree diagram from PCHMergerTree.build_full_tree(), in
    the style of Nadler et al. (2023) Figure 4: node size ~ log(mass),
    edge length ~ log(elapsed cosmic time), main branch highlighted.

plot_excursion_trajectory(masses, redshifts, cosmo_data, ...)
    A halo growth history re-expressed as an excursion-set random walk
    (Delta_delta vs Delta_S) alongside the familiar M vs z view, in the
    style of Nadler et al. (2023) Figure 1's unconstrained (grey) line --
    foraois doesn't implement their constrained excursions yet, so
    there's nothing to draw for the coloured constrained lines.

plot_dm_model_comparison(models, reference, ...)
    P(k) and sigma(M)/sigma_reference(M) across dark matter models (e.g.
    CDM vs WDM), showing the small-scale suppression directly.

All functions accept ``file_name``:
    None  → display interactively (plt.show)
    str   → save to <file_name>.png and close
"""

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Shared style -- publication/journal-figure look: white background, black
# text, a colourblind-safe categorical palette (Okabe & Ito 2008), thin
# light-grey gridlines. Every function below just refers to these names, so
# the whole module's look changes from here.
# ---------------------------------------------------------------------------

_BG = "#ffffff"
_EDGE = "#4d4d4d"
_FG = "#1a1a1a"
_DIM = "#8a8a8a"
_GRID = "#e6e6e6"

_BLUE = "#0072B2"
_ORANGE = "#E69F00"
_GREEN = "#009E73"
_PURPLE = "#CC79A7"
_RED = "#D55E00"
_YELLOW = "#F0E442"
_SKY = "#56B4E9"
_INK = "#1a1a1a"

# Colormap: blue → green → orange  (low → high mass); reads cleanly on white
_CMAP = mcolors.LinearSegmentedColormap.from_list("pch_mass", [_BLUE, _GREEN, _ORANGE])

_RC = {
    "figure.facecolor": _BG,
    "axes.facecolor": _BG,
    "axes.edgecolor": _EDGE,
    "axes.labelcolor": _FG,
    "axes.titlecolor": _FG,
    "axes.grid": True,
    "grid.color": _GRID,
    "grid.linewidth": 0.6,
    "xtick.color": _DIM,
    "ytick.color": _DIM,
    "xtick.labelcolor": _FG,
    "ytick.labelcolor": _FG,
    "text.color": _FG,
    "legend.facecolor": "#ffffff",
    "legend.edgecolor": _EDGE,
    "legend.labelcolor": _FG,
    "legend.framealpha": 0.9,
    "lines.linewidth": 2.2,
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}


def _style_on():
    mpl.rcParams.update(_RC)


def _save_or_show(fig, file_name):
    fig.tight_layout()
    if file_name is None:
        plt.show()
    else:
        fig.savefig(
            f"{file_name}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        print(f"  saved → {file_name}.png")
    plt.close(fig)


def _decorate(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, labelpad=5, fontsize=15)
    ax.set_ylabel(ylabel, labelpad=5, fontsize=15)
    if title:
        ax.set_title(title, pad=7)
    ax.tick_params(labelsize=12)


# ---------------------------------------------------------------------------
# Existing functions — signatures and behaviour unchanged
# ---------------------------------------------------------------------------


def plot_power_spectrum(pk_data, file_name, axes_limits):
    """
    Plot the linear matter power spectrum P(k).

    Parameters
    ----------
    pk_data : dict
        Keys 'k' and 'Pk'.
    file_name : str or None
        Output filename (no extension). None → interactive display.
    axes_limits : dict
        Keys: 'xmin', 'xmax', 'ymin', 'ymax'.
    """
    _style_on()
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.loglog(pk_data["k"][:], pk_data["Pk"][0][:], color=_BLUE, lw=1.6, label="P(k) linear")

    ax.set_xlim(axes_limits["xmin"], axes_limits["xmax"])
    ax.set_ylim(axes_limits["ymin"], axes_limits["ymax"])
    _decorate(
        ax,
        r"$k\;\;[h\,\mathrm{Mpc}^{-1}]$",
        r"$P(k)\;\;[\mathrm{Mpc}^3\,h^{-3}]$",
        "Linear matter power spectrum",
    )
    ax.legend(fontsize=8)

    _save_or_show(fig, file_name)


def plot_mass_variance(cosmology_data, pk_data, file_name, axes_limits):
    """
    Plot sigma(R) vs radius R and mark the sigma_8 point.

    Parameters
    ----------
    cosmology_data : CosmoData
    pk_data : dict
    file_name : str or None
    axes_limits : dict
        Keys: 'xmin', 'xmax', 'ymin', 'ymax'.
    """
    _style_on()
    fig, ax = plt.subplots(figsize=(6, 4))

    logr_vals = np.linspace(-2, 1.3, 100)
    mass_variance = np.array(
        [
            cosmology_data.get_mass_variance(pk_data, radius=10**logr, window_function_type="top_hat")
            for logr in logr_vals
        ]
    )

    # sigma8 isn't set anywhere in CosmoData.__init__ -- it's only ever a
    # side effect of some earlier get_mass_variance(radius=8.0) call, so a
    # caller who hasn't happened to make one first hits an AttributeError
    # here. Compute it directly instead of assuming it's already there.
    if not hasattr(cosmology_data, "sigma8"):
        cosmology_data.sigma8 = cosmology_data.get_mass_variance(pk_data, radius=8.0, window_function_type="top_hat")

    ax.plot(10**logr_vals, np.sqrt(mass_variance), color=_ORANGE, lw=1.6)
    ax.axvline(8.0, color=_RED, ls="--", lw=0.9, alpha=0.8, label="8 Mpc/h")
    ax.axhline(
        np.sqrt(cosmology_data.sigma8),
        color=_RED,
        ls="--",
        lw=0.9,
        alpha=0.8,
        label=r"$\sigma_8$",
    )

    ax.set_xlim(axes_limits["xmin"], axes_limits["xmax"])
    ax.set_ylim(axes_limits["ymin"], axes_limits["ymax"])
    _decorate(ax, r"$R\;\;[\mathrm{Mpc}\,h^{-1}]$", r"$\sigma(R)$", "Mass variance")
    ax.legend(fontsize=8)

    _save_or_show(fig, file_name)


def plot_merger_history(tree, file_name=None):
    """
    Plot the main-progenitor mass and fragment masses from a single serial
    ``build_tree`` result.

    Parameters
    ----------
    tree : list of dict
        Each dict has keys 'redshift', 'parent_mass', 'progenitors'.
    file_name : str or None
    """
    if not tree:
        print("plot_merger_history: tree is empty.")
        return

    _style_on()
    fig, ax = plt.subplots(figsize=(8, 5))

    redshifts = [step["redshift"] for step in tree]
    parent_masses = [step["parent_mass"] for step in tree]

    for step in tree:
        for prog in step["progenitors"]:
            ax.scatter(
                step["redshift"],
                np.log10(prog),
                color=_BLUE,
                alpha=0.7,
                s=35,
                edgecolors=_EDGE,
                linewidths=0.5,
                zorder=4,
            )

    ax.plot(
        redshifts,
        np.log10(parent_masses),
        color=_INK,
        lw=2.0,
        ls="--",
        label="main progenitor",
        zorder=5,
    )

    _decorate(
        ax,
        r"Redshift $z$",
        r"$\log_{10}(M\,/\,[M_\odot\,h^{-1}])$",
        "Merger tree assembly history",
    )
    ax.legend(fontsize=8)

    _save_or_show(fig, file_name)


# ---------------------------------------------------------------------------
# New functions for vectorised forest output
# ---------------------------------------------------------------------------


def plot_mass_history(mass_history, z_steps, n_show=200, M_res=None, file_name=None):
    """
    Plot main-progenitor mass tracks M(z) for a random sample of trees from
    the vectorised forest, with the median and 16-84th percentile band.

    Parameters
    ----------
    mass_history : np.ndarray, shape (N, n_steps)
        Output of ``build_forest_numpy`` or ``build_forest_numba``.
        Zeros mark trees that have dropped below M_res.
    z_steps : np.ndarray, shape (n_steps+1,)
        Redshift grid edges returned alongside mass_history.
    n_show : int
        Number of randomly sampled individual tracks to draw (default 200).
    M_res : float or None
        Resolution mass; drawn as a dashed horizontal line if given.
    file_name : str or None
    """
    _style_on()
    fig, ax = plt.subplots(figsize=(9, 5))

    N, n_steps = mass_history.shape
    z_mid = 0.5 * (z_steps[:-1] + z_steps[1:])

    # Replace dead-tree zeros with NaN so lines terminate cleanly
    mh = mass_history.astype(float).copy()
    mh[mh <= 0] = np.nan

    # Colour each track by its z=0 mass
    m0 = mass_history[:, 0]
    m0_pos = m0[m0 > 0]
    logm_min = np.log10(m0_pos.min()) if m0_pos.size else 8.0
    logm_max = np.log10(m0_pos.max()) if m0_pos.size else 14.0
    norm = mcolors.Normalize(vmin=logm_min, vmax=logm_max)

    idx = np.random.choice(N, size=min(n_show, N), replace=False)
    for i in idx:
        track = np.log10(np.where(mh[i] > 0, mh[i], np.nan))
        c = _CMAP(norm(np.log10(m0[i])) if m0[i] > 0 else 0.0)
        ax.plot(z_mid, track, color=c, alpha=0.25, lw=0.7)

    # Median and percentile envelope
    with np.errstate(all="ignore"):
        log_mh = np.where(mh > 0, np.log10(mh), np.nan)

    median = np.nanmedian(log_mh, axis=0)
    p16 = np.nanpercentile(log_mh, 16, axis=0)
    p84 = np.nanpercentile(log_mh, 84, axis=0)

    ax.plot(z_mid, median, color=_INK, lw=2.2, zorder=10, label="median")
    ax.fill_between(z_mid, p16, p84, color=_INK, alpha=0.10, label="16–84th pctile")

    if M_res is not None:
        ax.axhline(
            np.log10(M_res),
            color=_RED,
            lw=0.9,
            ls="--",
            alpha=0.7,
            label=f"$M_{{\\rm res}}={M_res:.0e}$",
        )

    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label(r"$\log_{10}(M_0\,/\,[M_\odot\,h^{-1}])$", fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=_FG)

    _decorate(
        ax,
        r"Redshift $z$",
        r"$\log_{10}(M_{\rm main}\,/\,[M_\odot\,h^{-1}])$",
        f"Main-progenitor mass tracks  (N={N:,}, showing {min(n_show, N):,})",
    )
    ax.legend(fontsize=8, loc="upper right")

    _save_or_show(fig, file_name)


def plot_merger_rate(split_events, z_steps, N_trees=None, file_name=None):
    """
    Plot the mean merger rate dN_merge/dz per tree vs redshift.

    Parameters
    ----------
    split_events : list of dict
        Output of ``build_forest_numpy``.  Each dict has keys
        ``z_step`` (tuple of floats) and ``n_splits`` (int).
    z_steps : np.ndarray, shape (n_steps+1,)
    N_trees : int or None
        Total number of trees; normalises the rate to per-tree if given.
    file_name : str or None
    """
    _style_on()
    fig, ax = plt.subplots(figsize=(7, 4))

    n_steps = len(z_steps) - 1
    dz = np.diff(z_steps)
    z_mid = 0.5 * (z_steps[:-1] + z_steps[1:])
    counts = np.zeros(n_steps)

    for event in split_events:
        z0, _ = event["z_step"]
        j = int(np.searchsorted(z_steps[:-1], z0, side="right")) - 1
        j = max(0, min(j, n_steps - 1))
        counts[j] += event["n_splits"]

    rate = counts / dz
    if N_trees:
        rate = rate / N_trees

    ax.step(z_mid, rate, where="mid", color=_ORANGE, lw=1.6)
    ax.fill_between(z_mid, 0, rate, step="mid", color=_ORANGE, alpha=0.15)
    ax.set_ylim(bottom=0)

    ylabel = r"$dN_{\rm merge}/dz$ per tree" if N_trees else r"$dN_{\rm merge}/dz$ (total)"
    _decorate(ax, r"Redshift $z$", ylabel, "Merger rate vs redshift")

    _save_or_show(fig, file_name)


def plot_mass_function(mass_history, z_steps, z_targets=None, n_bins=30, M_res=None, file_name=None):
    """
    Plot the progenitor mass function dN/d log M at one or more redshifts,
    obtained by slicing ``mass_history`` at the nearest available step.

    Parameters
    ----------
    mass_history : np.ndarray, shape (N, n_steps)
    z_steps : np.ndarray, shape (n_steps+1,)
    z_targets : list of float or None
        Redshifts to evaluate.  Default: [0.5, 1.0, 2.0, 3.0].
    n_bins : int
        Log-mass histogram bins (default 30).
    M_res : float or None
        Vertical dashed line marking the resolution limit.
    file_name : str or None
    """
    _style_on()
    fig, ax = plt.subplots(figsize=(7, 5))

    if z_targets is None:
        z_targets = [0.5, 1.0, 2.0, 3.0]

    z_mid = 0.5 * (z_steps[:-1] + z_steps[1:])
    palette = [_BLUE, _ORANGE, _GREEN, _PURPLE, _RED, _YELLOW]
    N = mass_history.shape[0]

    for iz, zt in enumerate(z_targets):
        j = int(np.argmin(np.abs(z_mid - zt)))
        m_slice = mass_history[:, j]
        m_alive = m_slice[m_slice > 0]

        if m_alive.size < 2:
            continue

        logm = np.log10(m_alive)
        edges = np.linspace(logm.min(), logm.max(), n_bins + 1)
        cts, _ = np.histogram(logm, bins=edges)
        dlogm = np.diff(edges)
        phi = cts / dlogm / N  # per tree per dex

        c = palette[iz % len(palette)]
        ax.step(
            edges[:-1],
            np.where(phi > 0, np.log10(np.where(phi > 0, phi, 1.0)), np.nan),
            where="post",
            color=c,
            lw=1.5,
            label=f"$z = {z_mid[j]:.2f}$",
        )

    if M_res is not None:
        ax.axvline(
            np.log10(M_res),
            color=_RED,
            lw=0.9,
            ls="--",
            alpha=0.7,
            label=r"$M_{\rm res}$",
        )

    _decorate(
        ax,
        r"$\log_{10}(M\,/\,[M_\odot\,h^{-1}])$",
        r"$\log_{10}(\mathrm{d}N/\mathrm{d}\log M)$ per tree",
        "Progenitor mass function",
    )
    ax.legend(fontsize=8)

    _save_or_show(fig, file_name)


def plot_tree_graph(mass_history, z_steps, tree_idx=0, M_res=None, split_events=None, file_name=None):
    """
    Draw a branching diagram for a single tree.

    The main-progenitor track is rendered as a LineCollection whose colour
    and linewidth both encode log10(M).  Merger events involving this tree
    (from split_events) are marked as scatter points.

    Parameters
    ----------
    mass_history : np.ndarray, shape (N, n_steps)
    z_steps : np.ndarray, shape (n_steps+1,)
    tree_idx : int
        Which tree in the forest to draw (default 0).
    M_res : float or None
    split_events : list of dict or None
        If provided, mergers belonging to tree_idx are highlighted.
    file_name : str or None
    """
    _style_on()
    fig, ax = plt.subplots(figsize=(10, 5))

    z_mid = 0.5 * (z_steps[:-1] + z_steps[1:])
    n_steps = len(z_mid)

    track = mass_history[tree_idx].astype(float).copy()
    track[track <= 0] = np.nan
    log_track = np.where(track > 0, np.log10(track), np.nan)

    valid = ~np.isnan(log_track)
    if valid.sum() < 2:
        print(f"plot_tree_graph: tree {tree_idx} has fewer than 2 valid steps.")
        plt.close(fig)
        return

    logm_vals = log_track[valid]
    vmin, vmax = logm_vals.min(), logm_vals.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    width_range = vmax - vmin if vmax > vmin else 1.0

    segs, widths, seg_colors = [], [], []
    for j in range(n_steps - 1):
        y0, y1 = log_track[j], log_track[j + 1]
        if np.isnan(y0) or np.isnan(y1):
            continue
        mid_logm = 0.5 * (y0 + y1)
        segs.append([(z_mid[j], y0), (z_mid[j + 1], y1)])
        widths.append(1.0 + 3.5 * (mid_logm - vmin) / width_range)
        seg_colors.append(_CMAP(norm(mid_logm)))

    lc = LineCollection(segs, linewidths=widths, colors=seg_colors, zorder=3)
    ax.add_collection(lc)

    # Merger events for this tree
    if split_events is not None:
        mz, mm = [], []
        for event in split_events:
            if tree_idx in event.get("tree_ids", []):
                z0_ev, z1_ev = event["z_step"]
                z_ev = 0.5 * (z0_ev + z1_ev)
                j = int(np.argmin(np.abs(z_mid - z_ev)))
                if not np.isnan(log_track[j]):
                    mz.append(z_ev)
                    mm.append(log_track[j])
        if mz:
            ax.scatter(
                mz,
                mm,
                color=_ORANGE,
                s=40,
                zorder=6,
                edgecolors=_INK,
                linewidths=0.6,
                label="merger event",
            )

    if M_res is not None:
        ax.axhline(
            np.log10(M_res),
            color=_RED,
            lw=0.9,
            ls="--",
            alpha=0.6,
            label=f"$M_{{\\rm res}}={M_res:.0e}$",
        )

    ax.set_xlim(z_steps[0], z_steps[-1])
    pad = 0.3
    ax.set_ylim(logm_vals.min() - pad, logm_vals.max() + pad)

    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label(r"$\log_{10}(M\,/\,[M_\odot\,h^{-1}])$", fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=_FG)

    ax.legend(fontsize=8, loc="upper right")
    _decorate(
        ax,
        r"Redshift $z$",
        r"$\log_{10}(M\,/\,[M_\odot\,h^{-1}])$",
        f"Merger tree  (tree index {tree_idx})",
    )

    _save_or_show(fig, file_name)


def plot_forest_summary(mass_history, split_events, z_steps, M_res=None, n_show=150, file_name=None):
    """
    2×2 summary panel for a full forest run.

    Top-left     Mass tracks for n_show random trees + median/percentile band
    Top-right    Merger rate dN_merge/dz per tree
    Bottom-left  Progenitor mass function at z ≈ 0.5, 1, 2, 3
    Bottom-right Fraction of trees still alive (M ≥ M_res) vs redshift

    Parameters
    ----------
    mass_history : np.ndarray, shape (N, n_steps)
    split_events : list of dict
    z_steps : np.ndarray, shape (n_steps+1,)
    M_res : float or None
    n_show : int
        Trees shown in the track panel (default 150).
    file_name : str or None
    """
    _style_on()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.38, wspace=0.30)

    N, _n_steps = mass_history.shape
    z_mid = 0.5 * (z_steps[:-1] + z_steps[1:])
    dz = np.diff(z_steps)

    mh = mass_history.astype(float).copy()
    mh[mh <= 0] = np.nan

    with np.errstate(all="ignore"):
        log_mh = np.where(mh > 0, np.log10(mh), np.nan)

    # ------------------------------------------------------------------ [0,0]
    ax = axes[0, 0]

    m0 = mass_history[:, 0]
    m0_pos = m0[m0 > 0]
    logm_min = np.log10(m0_pos.min()) if m0_pos.size else 8.0
    logm_max = np.log10(m0_pos.max()) if m0_pos.size else 14.0
    norm = mcolors.Normalize(vmin=logm_min, vmax=logm_max)

    idx = np.random.choice(N, size=min(n_show, N), replace=False)
    for i in idx:
        c = _CMAP(norm(np.log10(m0[i])) if m0[i] > 0 else 0.0)
        ax.plot(z_mid, log_mh[i], color=c, alpha=0.2, lw=0.6)

    median = np.nanmedian(log_mh, axis=0)
    p16 = np.nanpercentile(log_mh, 16, axis=0)
    p84 = np.nanpercentile(log_mh, 84, axis=0)
    ax.plot(z_mid, median, color=_INK, lw=2.0, zorder=5)
    ax.fill_between(z_mid, p16, p84, color=_INK, alpha=0.10)
    if M_res is not None:
        ax.axhline(np.log10(M_res), color=_RED, lw=0.8, ls="--", alpha=0.6)
    _decorate(ax, r"$z$", r"$\log_{10}(M)$", "Mass tracks")

    n_steps = len(z_steps) - 1

    # ------------------------------------------------------------------ [0,1]
    ax = axes[0, 1]
    counts = np.zeros(n_steps)
    for event in split_events:
        z0, _ = event["z_step"]
        j = int(np.searchsorted(z_steps[:-1], z0, side="right")) - 1
        j = max(0, min(j, n_steps - 1))
        counts[j] += event["n_splits"]

    rate = counts / dz / N
    ax.step(z_mid, rate, where="mid", color=_ORANGE, lw=1.4)
    ax.fill_between(z_mid, 0, rate, step="mid", color=_ORANGE, alpha=0.15)
    ax.set_ylim(bottom=0)
    _decorate(ax, r"$z$", r"$dN_{\rm merge}/dz$ per tree", "Merger rate")

    # ------------------------------------------------------------------ [1,0]
    ax = axes[1, 0]
    z_targets = [0.5, 1.0, 2.0, 3.0]
    palette = [_BLUE, _ORANGE, _GREEN, _PURPLE]
    n_bins = 25

    for zt, c in zip(z_targets, palette, strict=True):
        j = int(np.argmin(np.abs(z_mid - zt)))
        m_slice = mass_history[:, j]
        m_alive = m_slice[m_slice > 0]
        if m_alive.size < 2:
            continue
        logm = np.log10(m_alive)
        edges = np.linspace(logm.min(), logm.max(), n_bins + 1)
        cts, _ = np.histogram(logm, bins=edges)
        dlogm = np.diff(edges)
        phi = cts / dlogm / N
        ax.step(
            edges[:-1],
            np.where(phi > 0, np.log10(np.where(phi > 0, phi, 1.0)), np.nan),
            where="post",
            color=c,
            lw=1.3,
            label=f"$z={z_mid[j]:.1f}$",
        )

    if M_res is not None:
        ax.axvline(np.log10(M_res), color=_RED, lw=0.8, ls="--", alpha=0.6)
    ax.legend(fontsize=7, ncol=2)
    _decorate(ax, r"$\log_{10}(M)$", r"$\log_{10}(dN/d\log M)$", "Mass function")

    # ------------------------------------------------------------------ [1,1]
    ax = axes[1, 1]
    threshold = M_res if M_res is not None else 0.0
    alive_frac = np.mean(mass_history > threshold, axis=0) * 100.0
    ax.plot(z_mid, alive_frac, color=_GREEN, lw=1.5)
    ax.fill_between(z_mid, 0, alive_frac, color=_GREEN, alpha=0.12)
    ax.set_ylim(0, 105)
    _decorate(ax, r"$z$", "Trees alive  (%)", "Survival fraction")

    fig.suptitle(f"Merger forest summary  —  N = {N:,} trees", fontsize=11, color=_FG, y=1.01)

    _save_or_show(fig, file_name)


# ---------------------------------------------------------------------------
# Full branching-tree diagram (Nadler et al. 2023, Figure 4 style)
# ---------------------------------------------------------------------------


def plot_dendrogram(
    nodes,
    cosmo_data,
    M_min_display=None,
    highlight_color=None,
    mark_node_id=None,
    z_ticks=None,
    file_name=None,
):
    """
    Draw the full branching structure of one tree from
    PCHMergerTree.build_full_tree(), styled after Nadler et al. (2023),
    Figure 4: node size scales with log10(mass); edge length is
    proportional to log(elapsed cosmic time) between a node and its
    descendant, not linear in redshift (so the tree visibly compresses at
    high z, where a fixed dz spans much less cosmic time); the main
    branch (found by following is_main links from the root) is
    highlighted.

    Parameters
    ----------
    nodes : list of dict
        Output of PCHMergerTree.build_full_tree().
    cosmo_data : CosmoData
        Needed for cosmic_time_at_z() -- the edge-length/y-axis transform.
    M_min_display : float or None
        Only draw nodes with mass >= this threshold. The underlying tree
        can be (and often is) resolved to a much lower M_res than you want
        to draw -- this mirrors Nadler et al.'s own approach of computing
        down to M_min=1e6 Msun but only plotting M >= 1e12 Msun. Since mass
        is monotonically non-increasing along any path from the root
        towards higher redshift (a descendant's mass is always >= either
        progenitor's), this is a simple filter, not a re-pruning: every
        surviving node's ancestors back to the root survive too.
    highlight_color : str or None
        Colour for the main branch; defaults to the module's blue.
    mark_node_id : int or None
        If given, draw a star marker on this node (e.g. a Brownian-bridge
        constraint point, or any node of particular interest).
    z_ticks : list of float or None
        Redshift values to label on the y-axis. Default: 5 values evenly
        spaced across the redshift range present in `nodes`.
    file_name : str or None
    """
    if highlight_color is None:
        highlight_color = _BLUE

    if M_min_display is not None:
        nodes = [n for n in nodes if n["mass"] >= M_min_display]
    if not nodes:
        print("plot_dendrogram: no nodes to plot (check M_min_display).")
        return

    by_id = {n["id"]: n for n in nodes}
    children = {}
    for n in nodes:
        children.setdefault(n["descendant_id"], []).append(n)

    try:
        root = next(n for n in nodes if n["descendant_id"] is None)
    except StopIteration:
        print("plot_dendrogram: no root node (descendant_id=None) survived M_min_display.")
        return

    # ---- x positions: simple tidy-tree layout, leaves left-to-right ----
    x_pos = {}
    _next_x = [0.0]

    def _assign_x(node):
        kids = [k for k in children.get(node["id"], []) if k["id"] in by_id]
        if not kids:
            x_pos[node["id"]] = _next_x[0]
            _next_x[0] += 1.0
        else:
            for k in kids:
                _assign_x(k)
            x_pos[node["id"]] = np.mean([x_pos[k["id"]] for k in kids])

    _assign_x(root)

    # ---- y positions: y(z) = log10(t(z=0) / t(z)) -- 0 at the root,
    # increasing with redshift, with spacing that shrinks at high z to
    # match how little cosmic time a fixed dz spans there ----
    t0 = cosmo_data.cosmic_time_at_z(0.0)
    all_z = np.array([n["redshift"] for n in nodes])
    t_of_z = cosmo_data.cosmic_time_at_z(all_z)
    y_of = dict(
        zip(
            (n["id"] for n in nodes),
            np.log10(t0 / t_of_z),
            strict=True,
        )
    )

    # ---- draw ----
    _style_on()
    fig, ax = plt.subplots(figsize=(9, 7))

    mass_vals = np.array([n["mass"] for n in nodes])
    logm_min, logm_max = np.log10(mass_vals.min()), np.log10(mass_vals.max())
    span = max(logm_max - logm_min, 1e-6)

    def _size(mass, small=6.0, large=140.0):
        return small + (large - small) * (np.log10(mass) - logm_min) / span

    # edges first (so nodes draw on top)
    edge_segs = []
    for n in nodes:
        if n["descendant_id"] is None:
            continue
        d = by_id[n["descendant_id"]]
        edge_segs.append([(x_pos[d["id"]], y_of[d["id"]]), (x_pos[n["id"]], y_of[n["id"]])])
    ax.add_collection(LineCollection(edge_segs, colors=_DIM, linewidths=1.0, zorder=1))

    # main branch: follow is_main links from the root
    main_ids = set()
    current = root
    while True:
        main_ids.add(current["id"])
        kids = [k for k in children.get(current["id"], []) if k["id"] in by_id and k["is_main"]]
        if not kids:
            break
        current = kids[0]

    xs = np.array([x_pos[n["id"]] for n in nodes])
    ys = np.array([y_of[n["id"]] for n in nodes])
    sizes = _size(mass_vals)
    is_main_arr = np.array([n["id"] in main_ids for n in nodes])

    ax.scatter(
        xs[~is_main_arr],
        ys[~is_main_arr],
        s=sizes[~is_main_arr],
        facecolors="white",
        edgecolors=_EDGE,
        linewidths=1.0,
        zorder=2,
    )
    ax.scatter(
        xs[is_main_arr],
        ys[is_main_arr],
        s=sizes[is_main_arr],
        facecolors=highlight_color,
        edgecolors=_EDGE,
        linewidths=1.0,
        zorder=3,
    )

    if mark_node_id is not None and mark_node_id in by_id:
        ax.scatter(
            [x_pos[mark_node_id]],
            [y_of[mark_node_id]],
            marker="*",
            s=400,
            facecolors=_YELLOW,
            edgecolors=_EDGE,
            linewidths=1.2,
            zorder=4,
        )

    ax.set_xticks([])
    ax.set_xlim(-1, max(x_pos.values()) + 1)

    if z_ticks is None:
        z_lo, z_hi = all_z.min(), all_z.max()
        z_ticks = np.linspace(z_lo, z_hi, 5)
    z_ticks = np.asarray(sorted(set(np.round(z_ticks, 6))))
    y_ticks = np.log10(t0 / cosmo_data.cosmic_time_at_z(z_ticks))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"$z={z:g}$" for z in z_ticks])

    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)

    _save_or_show(fig, file_name)


# ---------------------------------------------------------------------------
# Excursion-set trajectory (Nadler et al. 2023, Figure 1 style)
# ---------------------------------------------------------------------------


def plot_excursion_trajectory(masses, redshifts, cosmo_data, file_name=None):
    """
    Nadler et al. (2023), Figure 1 style: left panel is the excursion-set
    random walk Delta_delta = delta_c(z) - delta_c(z0) vs
    Delta_S = S(M) - S(M0) (S = sigma(M)^2, the mass variance) traced out
    by a halo's growth history; right panel is the same trajectory as the
    more familiar M vs z. A tree's mass at each recorded redshift is, by
    construction, exactly where the excursion trajectory first crosses the
    barrier delta_c(z) at that redshift -- so any (M, z) sequence already
    *is* this trajectory, just two different changes of variable on the
    same underlying walk; nothing needs separate simulating.

    Only unconstrained trajectories are meaningful here: foraois doesn't
    yet implement Nadler et al.'s Brownian-bridge constrained excursions
    (the "first-crossing solver" item in the code audit roadmap), so
    there's no equivalent of their coloured constrained lines to draw --
    this reproduces their grey unconstrained-excursion line only.

    Parameters
    ----------
    masses : array-like, shape (n,)
        Main-progenitor mass at each redshift, ordered starting from the
        root (M0 at z0, index 0) outward to higher redshift. E.g. for a
        single build_forest_numpy/numba tree, prepend M0 to that tree's
        mass_history row (mass_history doesn't include the z0 initial
        condition itself): [M0, *mass_history[tree_idx]], paired with
        z_steps (which does include z0 at index 0). Zero entries (a tree
        that dropped below M_res) are dropped automatically.
    redshifts : array-like, shape (n,)
        Matching redshifts.
    cosmo_data : CosmoData
    file_name : str or None
    """
    masses = np.asarray(masses, dtype=float)
    redshifts = np.asarray(redshifts, dtype=float)
    mask = masses > 0
    masses, redshifts = masses[mask], redshifts[mask]

    if masses.size < 2:
        print("plot_excursion_trajectory: fewer than 2 valid points to plot.")
        return

    S = cosmo_data.sigma_at_logmass(np.log10(masses)) ** 2
    delta_c = cosmo_data.delta_col_at_z(redshifts)
    dS = S - S[0]
    ddelta = delta_c - delta_c[0]

    _style_on()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(dS, ddelta, color=_DIM, lw=1.2)
    ax.scatter([dS[0]], [ddelta[0]], color=_BLUE, s=55, zorder=5, label=r"$(M_0, z_0)$")
    ax.scatter([dS[-1]], [ddelta[-1]], color=_ORANGE, s=55, zorder=5, label=r"$(M_1, z_1)$")
    _decorate(
        ax,
        r"$\Delta S = S[M] - S[M_0]$",
        r"$\Delta\delta = \delta_c(z) - \delta_c(z_0)$",
        "Excursion trajectory",
    )
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(redshifts, masses, color=_DIM, lw=1.2)
    ax2.scatter([redshifts[0]], [masses[0]], color=_BLUE, s=55, zorder=5)
    ax2.scatter([redshifts[-1]], [masses[-1]], color=_ORANGE, s=55, zorder=5)
    ax2.set_yscale("log")
    ax2.invert_xaxis()  # z decreasing left to right, z0 on the right -- matches Fig. 1's right panel
    _decorate(ax2, r"$z$", r"$M\;\;[M_\odot\,h^{-1}]$", "Halo growth history")

    fig.suptitle(
        "Unconstrained excursion trajectory (cf. Nadler et al. 2023, Fig. 1)",
        fontsize=11,
        y=1.03,
    )

    _save_or_show(fig, file_name)


# ---------------------------------------------------------------------------
# Dark-matter-model comparison (e.g. CDM vs WDM)
# ---------------------------------------------------------------------------


def plot_dm_model_comparison(models, reference, file_name=None):
    """
    Compare P(k) and sigma(M) across dark matter models -- e.g. CDM vs
    WDM (or, later, FDM). Left panel: P(k) for each model. Right panel:
    each model's sigma(M) relative to the reference model's, which shows
    the suppression directly (flat at 1 where a model matches the
    reference, dropping at low mass for e.g. WDM).

    Parameters
    ----------
    models : dict[str, tuple[CosmoData, dict]]
        {label: (cosmo_data, pk_data)} for every model to plot, including
        the reference itself. Each cosmo_data must already have
        _prepare_sigma_grid(pk_data) called (or sigma_at_logmass will
        raise) -- this function only reads, it doesn't build the grid.
    reference : str
        Key into `models` to divide sigma(M) by (e.g. "CDM").
    file_name : str or None
    """
    if reference not in models:
        raise KeyError(f"reference '{reference}' not in models {list(models)}")

    _style_on()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    palette = [_BLUE, _ORANGE, _GREEN, _PURPLE, _RED, _YELLOW]

    logm = np.linspace(6.0, 15.0, 200)
    ref_cosmo, _ = models[reference]
    ref_sigma = ref_cosmo.sigma_at_logmass(logm)

    ax = axes[0]
    for i, (label, (_cosmo_data, pk_data)) in enumerate(models.items()):
        c = palette[i % len(palette)]
        ax.loglog(pk_data["k"], pk_data["Pk"][0], color=c, lw=2.3, label=label)
    _decorate(
        ax,
        r"$k\;\;[h\,\mathrm{Mpc}^{-1}]$",
        r"$P(k)\;\;[\mathrm{Mpc}^3\,h^{-3}]$",
    )
    ax.legend(fontsize=10)

    ax2 = axes[1]
    for i, (label, (cosmo_data, _pk_data)) in enumerate(models.items()):
        c = palette[i % len(palette)]
        sigma = cosmo_data.sigma_at_logmass(logm)
        ax2.plot(logm, sigma / ref_sigma, color=c, lw=2.3, label=label)
    ax2.axhline(1.0, color=_DIM, lw=1.2, ls="--")
    _decorate(
        ax2,
        r"$\log_{10}(M\,/\,[M_\odot\,h^{-1}])$",
        rf"$\sigma(M)\,/\,\sigma_{{\rm {reference}}}(M)$",
    )
    ax2.legend(fontsize=10)

    _save_or_show(fig, file_name)


# ---------------------------------------------------------------------------
# Branching-rate validation (Monte Carlo vs. quadrature)
# ---------------------------------------------------------------------------


def plot_branching_rate_validation(
    tree_generator,
    M2,
    M_res,
    delta0,
    d_omega,
    n_trials=500_000,
    n_bins=40,
    seed=None,
    file_name=None,
):
    """
    Publication figure validating the rejection-sampling scheme
    (foraois.diagnostics.check_sampling_consistency's distributional
    counterpart): histograms the progenitor mass ratio q from n_trials
    accepted Monte Carlo draws (the same _draw_progenitor_ratio/
    _rejection_ratio code path build_forest_numpy/numba actually use) and
    overlays the independently-computed analytic differential rate density
    S(q)*R(q) (PCH08 eq. A1), normalised to the same accepted-fraction
    integral. Agreement in shape, not just check_sampling_consistency's
    single aggregate rate number, is the point of this figure -- a shape
    mismatch with a matching aggregate rate would indicate a real bug that
    the scalar check alone could miss.

    Parameters
    ----------
    tree_generator : PCHMergerTree
    M2, M_res, delta0, d_omega : float
        Same meaning as foraois.diagnostics.check_sampling_consistency.
    n_trials : int
    n_bins : int
    seed : int or None
    file_name : str or None

    Returns
    -------
    dict with 'empirical_rate', 'true_rate', 'n_sigma' (from
    check_sampling_consistency, run internally at the same n_trials) --
    the same numbers the figure visualises, for use in a caption/table.
    """
    from ..diagnostics import check_sampling_consistency
    from ..pch_trees import _draw_progenitor_ratio, _rejection_ratio

    if seed is not None:
        np.random.seed(seed)

    terms = tree_generator.branching_rate_terms(M2, M_res, delta0, d_omega)
    qres, eta = terms["qres"], terms["eta"]

    r1 = np.random.rand(n_trials)
    attempted = r1 <= terms["Nupper"]
    u2 = np.random.rand(n_trials)
    q = _draw_progenitor_ratio(u2, qres, eta)
    sigma_fn = tree_generator.cosmo_data.sigma_at_logmass
    alpha_fn = tree_generator.cosmo_data.dlogsigma_at_logmass
    R = _rejection_ratio(q, M2, terms, sigma_fn, alpha_fn, tree_generator.gamma1)
    r3 = np.random.rand(n_trials)
    accepted = attempted & (r3 <= R)
    q_accepted = q[accepted]

    consistency = check_sampling_consistency(
        tree_generator,
        M2,
        M_res,
        delta0,
        d_omega,
        n_trials=n_trials,
        seed=None,
    )

    S_coeff_domega = terms["S_coeff"] * terms["d_omega"]

    def rate_density(q_arr):
        S_q = S_coeff_domega * q_arr ** (eta - 1.0)
        R_q = _rejection_ratio(q_arr, M2, terms, sigma_fn, alpha_fn, tree_generator.gamma1)
        return S_q * R_q

    _style_on()
    fig, ax = plt.subplots(figsize=(7, 5))

    # log-spaced bins: S(q) ~ q^(eta-1) is a power law over several decades
    # of q near qres, so linear bins waste most of their resolution on the
    # flat high-q tail and under-sample the steep low-q rise where most of
    # the accepted mass actually is.
    edges = np.geomspace(qres, 0.5, n_bins + 1)
    counts, _ = np.histogram(q_accepted, bins=edges)
    widths = np.diff(edges)
    empirical_density = counts / widths / n_trials  # per unit q, per trial
    # zero-count bins (possible in the sparsely-sampled high-q tail) can't
    # be shown on a log scale -- masked to nan rather than plotted as 0.
    empirical_density = np.where(counts > 0, empirical_density, np.nan)

    ax.step(
        edges[:-1],
        empirical_density,
        where="post",
        color=_BLUE,
        lw=2.3,
        label=f"Monte Carlo ($N={n_trials:,}$)",
    )

    q_fine = np.geomspace(qres, 0.5, 400)
    ax.plot(
        q_fine,
        rate_density(q_fine),
        color=_ORANGE,
        lw=2.6,
        ls="--",
        label="analytic $S(q)R(q)$ (quadrature)",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    _decorate(
        ax,
        r"$q = M_1/M_2$",
        r"$\mathrm{d}N_{\rm accepted}/\mathrm{d}q$ per trial",
    )
    ax.legend(fontsize=10)
    ax.text(
        0.97,
        0.95,
        rf"$n_\sigma = {consistency['n_sigma']:.2f}$"
        f"\nempirical rate $= {consistency['empirical_rate']:.4g}$"
        f"\ntrue rate $= {consistency['true_rate']:.4g}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=_DIM,
    )

    _save_or_show(fig, file_name)

    return consistency
