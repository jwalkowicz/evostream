"""
Reruns the thesis 1 simulation with river's radius formula and with the
corrected one, for every dimension, epsilon and stream order. Runs are
parallel processes, so the timings here are only indicative.
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

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


def run_one(seed: int, radius: str, pca_dim: int | None, epsilon: float) -> dict:
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

    print(best.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
