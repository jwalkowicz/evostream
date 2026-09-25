# Figures for the thesis 2 sweep, drawn from the per-run CSVs

import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from experiments.plot_style import use_polish_number_format
from experiments.theses.thesis_2.exp_thesis_2_drift import (
    BATCH_SIZE,
    DRIFT_POINT,
    SAMPLES_PER_PHASE,
)
from experiments.theses.thesis_2.exp_thesis_2_drift import (
    INITIAL_WARMUP_SIZE as WARMUP_END,
)
from experiments.theses.thesis_2.exp_thesis_2_drift import (
    SWEEP_DECAY_VALUES as DECAY_VALUES,
)
from experiments.theses.thesis_2.exp_thesis_2_drift import (
    SWEEP_EPSILON_VALUES as EPSILON_VALUES,
)
from src.core.config import config

RESULTS_DIR = "experiments/theses/thesis_2/results"

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


def load_all() -> dict[tuple[float, float], pd.DataFrame]:
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
    ax.set_xlim(0, STREAM_LENGTH)


def _base_legend_handles() -> list:
    return [
        mpatches.Patch(facecolor=WARMUP_COLOR, label="Rozgrzewka (IPCA i DenStream)"),
        Line2D(
            [0], [0], color=DRIFT_LINE_COLOR, ls=(0, (1, 1)), lw=1.8, label=f"Zaplanowany dryf pojęć (t={DRIFT_POINT})"
        ),
    ]


def _detection_counts(dfs: dict[tuple[float, float], pd.DataFrame]) -> dict[float, int]:
    counts: dict[float, int] = {}
    for df in dfs.values():
        if "drift_detected" not in df.columns:
            continue
        for t in df.loc[df["drift_detected"], "sample_idx"]:
            counts[t] = counts.get(t, 0) + 1
    return counts


def _plot_detection_density(ax, dfs: dict[tuple[float, float], pd.DataFrame]):
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
    ax.set_ylabel("Alarmy", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_xlabel("Liczba przetworzonych dokumentów")


def plot_metric_band(
    dfs: dict[tuple[float, float], pd.DataFrame],
    ylabel: str,
    static_col: str | None,
    hot_col: str,
    filename: str,
    y_lim: tuple[float, float] | None = None,
    threshold: float | None = None,
):
    hot_stack = pd.concat([df.set_index("sample_idx")[hot_col] for df in dfs.values()], axis=1)
    x = hot_stack.index

    fig, (ax, ax_det) = plt.subplots(
        2,
        1,
        figsize=(11, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [5, 1], "hspace": 0.06},
    )
    _style_axis(ax)
    ax.tick_params(labelbottom=False)
    if y_lim:
        ax.set_ylim(y_lim)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", lw=1.6, zorder=1)

    if static_col:
        static_stack = pd.concat([df.set_index("sample_idx")[static_col] for df in dfs.values()], axis=1)
        ax.fill_between(
            x,
            static_stack.min(axis=1),
            static_stack.max(axis=1),
            facecolor=to_rgba(STATIC_COLOR, 0.16),
            edgecolor=to_rgba(STATIC_COLOR, 0.85),
            linewidth=1.0,
            zorder=1,
        )
        ax.plot(x, static_stack.mean(axis=1), color=STATIC_COLOR, lw=2.2, zorder=3)

    ax.fill_between(
        x,
        hot_stack.min(axis=1),
        hot_stack.max(axis=1),
        facecolor=to_rgba(HOTSWAP_COLOR, 0.16),
        edgecolor=to_rgba(HOTSWAP_COLOR, 0.85),
        linewidth=1.0,
        zorder=2,
    )
    ax.plot(x, hot_stack.mean(axis=1), color=HOTSWAP_COLOR, lw=2.4, zorder=4)

    ax.set_ylabel(ylabel)

    _plot_detection_density(ax_det, dfs)

    handles = _base_legend_handles()
    if threshold is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color="red",
                ls="--",
                lw=1.6,
                label=f"Próg sygnału przesunięcia centroidów (δc = {threshold:.2f})".replace(".", ","),
            )
        )
    if static_col:
        handles += [
            Line2D([0], [0], color=STATIC_COLOR, lw=2.2, label="Model statyczny (średnia)"),
            mpatches.Patch(facecolor=STATIC_COLOR, alpha=0.18, label="Model statyczny (zakres min-max)"),
        ]
    handles += [
        Line2D([0], [0], color=HOTSWAP_COLOR, lw=2.4, label="Model adaptacyjny (średnia)"),
        mpatches.Patch(facecolor=HOTSWAP_COLOR, alpha=0.18, label="Model adaptacyjny (zakres min-max)"),
        mpatches.Patch(
            facecolor=DETECTION_COLOR, alpha=0.85, label="Alarmy detektora (liczba przebiegów, panel poniżej)"
        ),
    ]
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)

    plt.tight_layout()
    out_path = f"{RESULTS_DIR}/{filename}"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    use_polish_number_format()
    dfs = load_all()
    n_expected = len(EPSILON_VALUES) * len(DECAY_VALUES)
    print(f"Loaded {len(dfs)}/{n_expected} combinations.")
    if not dfs:
        print("No result CSVs found. Run run_sweep.py (or exp_thesis_2_drift.py) first.")
        return

    plot_metric_band(
        dfs,
        "Wskaźnik sylwetki",
        "static_silhouette",
        "hotswap_silhouette",
        "thesis_2_band_silhouette.png",
        y_lim=(-0.1, 0.3),
    )
    plot_metric_band(
        dfs,
        "Przesunięcie centroidu (odl. euklidesowa)",
        None,
        "centroid_shift",
        "thesis_2_band_centroid_shift.png",
        threshold=config.drift.centroid_shift_threshold,
    )

    print("All thesis-2 plots generated.")


if __name__ == "__main__":
    main()
