"""
Search-space bounds for the DenStream epsilon parameter (thesis section 3.4).

Streams the thesis 1 document stream (the pre-drift phase of thesis 2:
6 categories, 5000 documents, d = 16) through the full two-phase clusterer
for a grid of epsilon values and records clustering quality and the number
of p-micro-clusters. Every epsilon is run at both ends of the decaying-factor
search range, so the chosen bounds hold whatever lambda NSGA-II picks, and
on several stream orders to separate real effects from noise. Only pre-drift
data is used, so the bounds are set without seeing the post-drift topics the
thesis 2 adaptation is evaluated on.
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from experiments.theses.thesis_1.exp_thesis_1_ipca import (
    STREAM_SEEDS,
    load_phase1_stream,
    run_streaming_simulation,
    shuffle_stream,
)
from src.core.config import config

RESULTS_DIR = "experiments/param_bounds/epsilon/results"
PCA_DIM = 16
EPSILON_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
DECAY_VALUES = list(config.evolution.param_bounds["decaying_factor"])


def run_one(seed: int, decay: float, epsilon: float) -> dict:
    _, embeddings, labels = load_phase1_stream()
    df = run_streaming_simulation(
        *shuffle_stream(embeddings, labels, seed), PCA_DIM, epsilon, decaying_factor=decay
    )
    return {
        "seed": seed,
        "decaying_factor": decay,
        "epsilon": epsilon,
        "mean_purity": df["purity"].mean(),
        "mean_nmi": df["nmi"].mean(),
        "mean_micro_clusters": df["n_micro_clusters"].mean(),
        "min_micro_clusters": df["n_micro_clusters"].min(),
        "mean_outlier_clusters": df["n_outlier_clusters"].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    grid = list(itertools.product(STREAM_SEEDS, DECAY_VALUES, EPSILON_GRID))
    print(f"Running {len(grid)} simulations with {args.workers} worker process(es)...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = pd.DataFrame(pool.map(run_one, *zip(*grid)))
    runs.to_csv(f"{RESULTS_DIR}/epsilon_bounds_runs.csv", index=False)

    summary = (
        runs.groupby(["decaying_factor", "epsilon"])
        .agg(
            purity=("mean_purity", "mean"),
            purity_std=("mean_purity", "std"),
            nmi=("mean_nmi", "mean"),
            nmi_std=("mean_nmi", "std"),
            micro_clusters=("mean_micro_clusters", "mean"),
            min_micro_clusters=("min_micro_clusters", "min"),
            outlier_clusters=("mean_outlier_clusters", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(f"{RESULTS_DIR}/epsilon_bounds_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
