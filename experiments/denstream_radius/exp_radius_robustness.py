"""
Robustness of the thesis 1 conclusions to the DenStream radius formula.

Runs the thesis 1 streaming simulation with river's radius and with the
corrected one (see src/domain/clustering.py and
https://github.com/online-ml/river/issues/2004) over every dimensionality,
the thesis 1 epsilon grid and several stream orders. Each seed shuffles the
phase-1 stream differently, which also changes the documents IPCA is fitted
on, so the results show how much of each difference is noise.

Runs are independent processes (each has its own RNG state and its own
radius setting). Latencies measured with several workers in parallel are
only indicative - use thesis 1 for timing.
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from experiments.theses.thesis_1.exp_thesis_1_ipca import (
    EPSILON_GRID,
    PCA_DIMS,
    STREAM_SEEDS,
    load_phase1_stream,
    run_streaming_simulation,
    shuffle_stream,
)
from src.domain.clustering import set_river_radius_fix

RESULTS_DIR = "experiments/denstream_radius/results"
RADIUS_MODES = ["river", "fixed"]


def run_one(seed: int, radius: str, pca_dim: Optional[int], epsilon: float) -> dict:
    set_river_radius_fix(radius == "fixed")
    _, embeddings, labels = load_phase1_stream()
    df = run_streaming_simulation(*shuffle_stream(embeddings, labels, seed), pca_dim, epsilon)
    return {
        "seed": seed,
        "radius": radius,
        "pca_dim": df["pca_dim"].iloc[0],
        "epsilon": epsilon,
        "mean_purity": df["purity"].mean(),
        "mean_nmi": df["nmi"].mean(),
        "mean_ari": df["ari"].mean(),
        "mean_micro_clusters": df["n_micro_clusters"].mean(),
        "mean_outlier_clusters": df["n_outlier_clusters"].mean(),
        "mean_stream_ms": df["stream_ms"].mean(),
    }


def best_epsilon_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (radius, dimension): the epsilon with the best purity averaged
    over seeds, with mean and std across seeds at that epsilon."""
    by_eps = (
        runs.groupby(["radius", "pca_dim", "epsilon"])
        .agg(
            purity=("mean_purity", "mean"),
            purity_std=("mean_purity", "std"),
            nmi=("mean_nmi", "mean"),
            nmi_std=("mean_nmi", "std"),
            micro_clusters=("mean_micro_clusters", "mean"),
            stream_ms=("mean_stream_ms", "mean"),
        )
        .reset_index()
    )
    best = by_eps.loc[by_eps.groupby(["radius", "pca_dim"])["purity"].idxmax()]
    return best.sort_values(["pca_dim", "radius"], ascending=[False, True])


def plot_purity_vs_epsilon(runs: pd.DataFrame, out_path: str):
    plt.rcParams.update({"font.size": 10, "font.family": "serif"})
    dims = sorted(runs["pca_dim"].unique(), reverse=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    labels = {"river": "promień river", "fixed": "promień poprawiony"}

    for ax, dim in zip(axes.flat, dims):
        for radius in RADIUS_MODES:
            sub = runs[(runs["pca_dim"] == dim) & (runs["radius"] == radius)]
            stats = sub.groupby("epsilon")["mean_purity"].agg(["mean", "std"])
            ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o", capsize=3, label=labels[radius])
        ax.set_title("Pełne SBERT (384d)" if dim == 384 else f"IPCA (d={dim})")
        ax.set_xlabel("ε")
        ax.grid(True, linestyle="--", alpha=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("Średnia czystość")
    axes.flat[0].legend()

    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    grid = list(itertools.product(STREAM_SEEDS, RADIUS_MODES, PCA_DIMS, EPSILON_GRID))
    print(f"Running {len(grid)} simulations with {args.workers} worker process(es)...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, *zip(*grid)))

    runs = pd.DataFrame(rows)
    runs.to_csv(f"{RESULTS_DIR}/radius_robustness_runs.csv", index=False)

    best = best_epsilon_table(runs)
    best.to_csv(f"{RESULTS_DIR}/radius_robustness_best_epsilon.csv", index=False)
    plot_purity_vs_epsilon(runs, f"{RESULTS_DIR}/radius_robustness_purity_vs_epsilon.pdf")

    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
