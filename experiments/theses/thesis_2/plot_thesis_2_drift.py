"""
Plots for the thesis-2 drift-adaptation sweep (5 epsilon x 5 decay = 25 runs).

Reads the per-combo CSVs written by exp_thesis_2_drift.py / run_sweep.py and
produces:
  - thesis_2_band_*.png       single normal-width chart per metric (ghost
                               clusters, purity, silhouette, outlier-ratio
                               drift signal): mean line + shaded min-max range
                               across all loaded combos, static vs Evostream
                               overlaid. These are the figures meant for the
                               thesis document - no wide multi-panel grids.
  - thesis_2_param_convergence_heatmap.png
                               where (eps, decay) settle after adaptation,
                               regardless of starting point
  - thesis_2_trajectory_example.png
                               epsilon/decay trajectory for one representative
                               run, as two stacked single-axis panels (a
                               dual-axis/twinx chart is a well-known
                               readability anti-pattern, so this replaces it)
  - thesis_2_summary_purity_gain.png
                               aggregate static-vs-hotswap comparison across
                               all combos
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from experiments.theses.thesis_2.exp_thesis_2_drift import (
    BATCH_SIZE,
    DRIFT_POINT,
    SAMPLES_PER_PHASE,
    SWEEP_DECAY_VALUES as DECAY_VALUES,
    SWEEP_EPSILON_VALUES as EPSILON_VALUES,
    INITIAL_WARMUP_SIZE as WARMUP_END,
)
from src.core.config import config

RESULTS_DIR = "experiments/theses/thesis_2/results"
DEFAULT_COMBO = (0.10, 0.005)

# Full stream length (phase 1 + phase 2). Used as a fixed, known x-limit
# instead of reading back matplotlib's autoscaled range: with sharex=True
# across small-multiple facets, an axis pinned from an EMPTY facet (no data
# for that combo yet) propagates its tiny default range to every other
# facet sharing that axis group, silently truncating facets that do have
# data plotted after it. A fixed bound sidesteps that ordering trap entirely.
STREAM_LENGTH = 2 * SAMPLES_PER_PHASE

WARMUP_COLOR = "#b2ebf2"
DRIFT_LINE_COLOR = "#1a1a1a"
STATIC_COLOR = "#e74c3c"
HOTSWAP_COLOR = "#27ae60"
DETECTION_COLOR = "#c2185b"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "figure.titlesize": 15,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "legend.fontsize": 10,
    }
)


def load_all() -> Dict[Tuple[float, float], pd.DataFrame]:
    dfs = {}
    for eps in EPSILON_VALUES:
        for decay in DECAY_VALUES:
            path = f"{RESULTS_DIR}/thesis_2_drift_results_eps_{eps}_decay_{decay}.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                for c in df.columns:
                    if "purity" in c or "silhouette" in c:
                        df[c] = df[c].ffill()
                dfs[(eps, decay)] = df
    return dfs


def _style_axis(ax):
    ax.axvspan(0, WARMUP_END, color=WARMUP_COLOR, alpha=0.9, zorder=0)
    ax.axvline(DRIFT_POINT, color=DRIFT_LINE_COLOR, linestyle=(0, (1, 1)), lw=1.8, zorder=1)
    ax.grid(True, linestyle="--", alpha=0.4)
    # Fixed, known bound (see STREAM_LENGTH comment above) - safe to set here,
    # before any data is plotted, since it never depends on autoscale.
    ax.set_xlim(0, STREAM_LENGTH)


def _base_legend_handles() -> List:
    return [
        mpatches.Patch(facecolor=WARMUP_COLOR, label="Rozgrzewka (IPCA i DenStream)"),
        Line2D([0], [0], color=DRIFT_LINE_COLOR, ls=(0, (1, 1)), lw=1.8, label=f"Zaplanowany dryf pojęć (t={DRIFT_POINT})"),
    ]


def _detection_counts(dfs: Dict[Tuple[float, float], pd.DataFrame]) -> Dict[float, int]:
    """How many loaded combos' detectors fired at each sample_idx (not just
    the true, purposely-induced drift at DRIFT_POINT)."""
    counts: Dict[float, int] = {}
    for df in dfs.values():
        if "drift_detected" not in df.columns:
            continue
        for t in df.loc[df["drift_detected"], "sample_idx"]:
            counts[t] = counts.get(t, 0) + 1
    return counts


def _plot_detection_density(ax, dfs: Dict[Tuple[float, float], pd.DataFrame]):
    """Thin bar-chart strip: bar height = number of combos whose detector
    fired at that sample_idx. A separate panel (sharing the main chart's
    x-axis) rather than overlaying bands on the data itself, so "how common
    was this detection point" reads directly off bar height instead of
    guessing at stacked alpha, and the main trend lines stay uncluttered.
    """
    ax.axvspan(0, WARMUP_END, color=WARMUP_COLOR, alpha=0.9, zorder=0)
    ax.axvline(DRIFT_POINT, color=DRIFT_LINE_COLOR, linestyle=(0, (1, 1)), lw=1.8, zorder=1)
    ax.set_xlim(0, STREAM_LENGTH)

    counts = _detection_counts(dfs)
    if counts:
        xs = list(counts.keys())
        heights = list(counts.values())
        ax.bar(xs, heights, width=BATCH_SIZE * 0.9, color=DETECTION_COLOR, alpha=0.85, zorder=2)
        max_count = max(heights)
        step = max(1, max_count // 4)
        ax.set_yticks(range(0, max_count + 1, step))
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Detekcje", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_xlabel("Liczba przetworzonych dokumentów")


def plot_metric_band(
    dfs: Dict[Tuple[float, float], pd.DataFrame],
    ylabel: str,
    static_col: Optional[str],
    hot_col: str,
    filename: str,
    y_lim: Optional[Tuple[float, float]] = None,
    threshold: Optional[float] = None,
):
    """Single normal-width chart: mean line + shaded min-max range across all
    loaded combos, for static and Evostream overlaid in different colors with
    enough transparency that both stay visible where they overlap. Replaces
    the wide small-multiples grid as the figure meant for the thesis document
    itself - one chart per metric instead of a 5-column facet grid.

    Every combo shares the same sample_idx grid (same document stream, same
    batch size, regardless of eps/decay), so the per-timestep mean/min/max
    across combos is just a row-wise aggregate - no interpolation needed.
    """
    hot_stack = pd.concat([df.set_index("sample_idx")[hot_col] for df in dfs.values()], axis=1)
    x = hot_stack.index

    fig, (ax, ax_det) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.06},
    )
    _style_axis(ax)
    ax.tick_params(labelbottom=False)
    if y_lim:
        ax.set_ylim(y_lim)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", lw=1.6, zorder=1)

    # Fill stays translucent (so overlap is a visible blend, not one color
    # blotting out the other), but each band also gets a near-opaque edge
    # stroke in its own color tracing its min/max boundary - that edge stays
    # legible even inside the overlap region, so you can always tell which
    # band's extent you're looking at regardless of how the fills blend.
    if static_col:
        static_stack = pd.concat([df.set_index("sample_idx")[static_col] for df in dfs.values()], axis=1)
        ax.fill_between(
            x, static_stack.min(axis=1), static_stack.max(axis=1),
            facecolor=to_rgba(STATIC_COLOR, 0.16), edgecolor=to_rgba(STATIC_COLOR, 0.85),
            linewidth=1.0, zorder=1,
        )
        ax.plot(x, static_stack.mean(axis=1), color=STATIC_COLOR, lw=2.2, zorder=3)

    ax.fill_between(
        x, hot_stack.min(axis=1), hot_stack.max(axis=1),
        facecolor=to_rgba(HOTSWAP_COLOR, 0.16), edgecolor=to_rgba(HOTSWAP_COLOR, 0.85),
        linewidth=1.0, zorder=2,
    )
    ax.plot(x, hot_stack.mean(axis=1), color=HOTSWAP_COLOR, lw=2.4, zorder=4)

    ax.set_ylabel(ylabel)

    _plot_detection_density(ax_det, dfs)

    handles = _base_legend_handles()
    if threshold is not None:
        handles.append(Line2D([0], [0], color="red", ls="--", lw=1.6, label=f"Próg detekcji dryfu ({threshold * 100:.0f}%)"))
    if static_col:
        handles += [
            Line2D([0], [0], color=STATIC_COLOR, lw=2.2, label="Model statyczny (średnia)"),
            mpatches.Patch(facecolor=STATIC_COLOR, alpha=0.18, label="Model statyczny (zakres min-max)"),
        ]
    handles += [
        Line2D([0], [0], color=HOTSWAP_COLOR, lw=2.4, label="Evostream (średnia)"),
        mpatches.Patch(facecolor=HOTSWAP_COLOR, alpha=0.18, label="Evostream (zakres min-max)"),
        mpatches.Patch(facecolor=DETECTION_COLOR, alpha=0.85, label="Detekcje dryfu (liczba kombinacji, panel poniżej)"),
    ]
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)

    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/{filename}"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_convergence_heatmaps(dfs: Dict[Tuple[float, float], pd.DataFrame], tail_docs: int = 1000):
    eps_grid = np.full((len(EPSILON_VALUES), len(DECAY_VALUES)), np.nan)
    decay_grid = np.full((len(EPSILON_VALUES), len(DECAY_VALUES)), np.nan)

    for i, eps in enumerate(EPSILON_VALUES):
        for j, decay in enumerate(DECAY_VALUES):
            df = dfs.get((eps, decay))
            if df is None:
                continue
            tail = df[df["sample_idx"] >= df["sample_idx"].max() - tail_docs]
            eps_grid[i, j] = tail["eps_adapted"].mean()
            decay_grid[i, j] = tail["decay_adapted"].mean()

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = [
        (axs[0], eps_grid, rf"Uśredniony $\epsilon(t)$ po ustabilizowaniu (ost. {tail_docs} dok.)", "{:.3f}"),
        (axs[1], decay_grid, rf"Uśredniony $\lambda(t)$ po ustabilizowaniu (ost. {tail_docs} dok.)", "{:.3f}"),
    ]
    for ax, grid, subtitle, fmt in panels:
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(DECAY_VALUES)))
        ax.set_xticklabels([str(d) for d in DECAY_VALUES])
        ax.set_yticks(range(len(EPSILON_VALUES)))
        ax.set_yticklabels([str(e) for e in EPSILON_VALUES])
        ax.set_xlabel(r"Startowe $\lambda_0$")
        ax.set_ylabel(r"Startowe $\epsilon_0$")
        ax.set_title(subtitle, fontsize=11)
        finite_vals = grid[~np.isnan(grid)]
        norm_mid = (finite_vals.max() + finite_vals.min()) / 2 if finite_vals.size else 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                val = grid[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > norm_mid else "black"
                    ax.text(j, i, fmt.format(val), ha="center", va="center", color=text_color, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/thesis_2_param_convergence_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_trajectory_example(dfs: Dict[Tuple[float, float], pd.DataFrame], combo: Tuple[float, float] = DEFAULT_COMBO):
    eps, decay = combo
    df = dfs.get(combo)
    if df is None:
        print(f"Combo {combo} missing, skipping trajectory example.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    _style_axis(axs[0])
    _style_axis(axs[1])

    axs[0].plot(df["sample_idx"], df["eps_adapted"], color="#2980b9", lw=2.2)
    axs[0].axhline(eps, color="#2980b9", ls=":", lw=1.4)
    axs[0].set_ylabel(r"Promień mikroklastra $\epsilon(t)$")

    axs[1].plot(df["sample_idx"], df["decay_adapted"], color="#8e44ad", lw=2.2)
    axs[1].axhline(decay, color="#8e44ad", ls=":", lw=1.4)
    axs[1].set_ylabel(r"Współczynnik wygaszania $\lambda(t)$")
    axs[1].set_xlabel("Liczba przetworzonych dokumentów")

    handles = _base_legend_handles() + [
        Line2D([0], [0], color="#2980b9", lw=2.2, label=r"Adaptacyjny $\epsilon(t)$"),
        Line2D([0], [0], color="#2980b9", ls=":", lw=1.4, label=rf"Start $\epsilon_0={eps}$"),
        Line2D([0], [0], color="#8e44ad", lw=2.2, label=r"Adaptacyjny $\lambda(t)$"),
        Line2D([0], [0], color="#8e44ad", ls=":", lw=1.4, label=rf"Start $\lambda_0={decay}$"),
    ]
    plt.tight_layout()
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.0))
    out_path = f"{RESULTS_DIR}/thesis_2_trajectory_example.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_summary_gain(dfs: Dict[Tuple[float, float], pd.DataFrame], tail_docs: int = 1000):
    rows = []
    for (eps, decay), df in dfs.items():
        tail = df[df["sample_idx"] >= df["sample_idx"].max() - tail_docs]
        rows.append(
            {
                "eps": eps,
                "decay": decay,
                "static_purity": tail["static_purity"].mean(),
                "hotswap_purity": tail["hotswap_purity"].mean(),
            }
        )
    summary = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    means = [summary["static_purity"].mean(), summary["hotswap_purity"].mean()]
    stds = [summary["static_purity"].std(), summary["hotswap_purity"].std()]
    colors = [STATIC_COLOR, HOTSWAP_COLOR]
    labels = ["Model statyczny", "Evostream (hot-swap)"]
    x = np.arange(2)
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Czystość po ustabilizowaniu\n(średnia ± odch. std. z {len(summary)} kombinacji)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/thesis_2_summary_purity_gain.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    dfs = load_all()
    n_expected = len(EPSILON_VALUES) * len(DECAY_VALUES)
    print(f"Loaded {len(dfs)}/{n_expected} combinations.")
    if not dfs:
        print("No result CSVs found. Run run_sweep.py (or exp_thesis_2_drift.py) first.")
        return

    # One normal-width chart per metric: mean + min-max band across all
    # loaded combos. These are the figures meant for the thesis document.
    plot_metric_band(
        dfs, "Liczba mikroklastrów", "static_micro", "hotswap_micro", "thesis_2_band_ghost_clusters.png",
    )
    plot_metric_band(
        dfs, "Czystość", "static_purity", "hotswap_purity", "thesis_2_band_purity.png", y_lim=(0.0, 1.05),
    )
    plot_metric_band(
        dfs, "Wskaźnik sylwetki", "static_silhouette", "hotswap_silhouette", "thesis_2_band_silhouette.png", y_lim=(-0.1, 0.3),
    )
    plot_metric_band(
        dfs, "Przesunięcie centroidu (odl. euklidesowa)", None, "centroid_shift", "thesis_2_band_centroid_shift.png",
        threshold=config.drift.centroid_shift_threshold,
    )

    plot_convergence_heatmaps(dfs)
    plot_trajectory_example(dfs)
    plot_summary_gain(dfs)

    print("All thesis-2 plots generated.")


if __name__ == "__main__":
    main()
