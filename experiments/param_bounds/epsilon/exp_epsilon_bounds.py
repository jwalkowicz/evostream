# Search-space bounds for epsilon on the validation stream (thesis section 3.4)

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from experiments.param_bounds.common.validation_stream import (
    VALIDATION_SAMPLES_PER_PHASE,
    load_validation_stream,
)
from experiments.theses.thesis_1.exp_thesis_1_ipca import (
    STREAM_SEEDS,
    run_streaming_simulation,
    shuffle_stream,
)
from src.core.config import config

RESULTS_DIR = "experiments/param_bounds/epsilon/results"
PCA_DIM = 16
EPSILON_GRID = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
DECAY_VALUES = list(config.evolution.param_bounds["decaying_factor"])


def run_one(seed: int, decay: float, epsilon: float) -> dict:
    embeddings, labels = load_validation_stream()
    embeddings = embeddings[:VALIDATION_SAMPLES_PER_PHASE]
    labels = labels[:VALIDATION_SAMPLES_PER_PHASE]
    df = run_streaming_simulation(*shuffle_stream(embeddings, labels, seed), PCA_DIM, epsilon, decaying_factor=decay)
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
