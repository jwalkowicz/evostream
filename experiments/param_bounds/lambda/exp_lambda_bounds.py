"""
Search-space bounds for the DenStream decaying factor lambda (thesis
section 3.4), measured on the system's own time scale.

River's DenStream advances its clock by one tick every `stream_speed`
documents (the system uses the default, 100), and a micro-cluster that
receives no new points loses half its weight every 1 / lambda ticks, i.e.
every stream_speed / lambda documents. The earlier version of this
experiment used stream_speed = 1, a clock 100x faster than the system's.

The validation stream (4 categories that no thesis experiment uses, abrupt
switch to 4 other unused categories after 3000 documents, d = 16, IPCA
fitted on the first documents; see
experiments/param_bounds/common/validation_stream.py) is processed by a
static clusterer - no drift detection or adaptation - for a grid of lambda
values, on several stream orders. The bounds are therefore chosen on data
completely separate from the data the theses are evaluated on. For each
lambda it records:
  - clustering quality before the switch (does forgetting hurt a stable
    stream?) and after it (does it help the model let go of old topics?),
  - how many of the p-micro-clusters alive at the switch survive 500, 1000,
    2000 and 3000 documents later.
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from experiments.param_bounds.common.validation_stream import (
    VALIDATION_PHASE1_CATEGORIES,
    VALIDATION_SAMPLES_PER_PHASE as SAMPLES_PER_PHASE,
    load_validation_stream,
)
from experiments.theses.thesis_1.exp_thesis_1_ipca import STREAM_SEEDS, shuffle_stream
from experiments.theses.thesis_2.exp_thesis_2_drift import BATCH_SIZE, INITIAL_WARMUP_SIZE
from src.core.config import config
from src.domain.clustering import StreamClusterer

RESULTS_DIR = "experiments/param_bounds/lambda/results"
PCA_DIM = 16
LAMBDA_GRID = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
SURVIVAL_CHECKPOINTS = [500, 1000, 2000, 3000]  # documents after the topic switch


def load_drift_stream(seed: int):
    """Both phases of the validation stream, each shuffled with the seed; the
    topic switch stays at document SAMPLES_PER_PHASE."""
    embeddings, labels = load_validation_stream()
    parts = [
        shuffle_stream(embeddings[start : start + SAMPLES_PER_PHASE], labels[start : start + SAMPLES_PER_PHASE], seed)
        for start in (0, SAMPLES_PER_PHASE)
    ]
    return np.vstack([parts[0][0], parts[1][0]]), parts[0][1] + parts[1][1]


def run_one(seed: int, decay: float) -> Tuple[dict, pd.DataFrame]:
    embeddings, labels = load_drift_stream(seed)
    ipca = IncrementalPCA(n_components=PCA_DIM).fit(embeddings[:INITIAL_WARMUP_SIZE])
    clusterer = StreamClusterer(
        epsilon=config.denstream.epsilon,
        mu=config.denstream.mu,
        beta=config.denstream.beta,
        decaying_factor=decay,
        n_samples_init=config.denstream.n_samples_init,
        window_size=config.denstream.window_size,
        expected_macro_clusters=len(VALIDATION_PHASE1_CATEGORIES),
    )
    clusterer.warm_start(normalize(ipca.transform(embeddings[:INITIAL_WARMUP_SIZE])))

    records = []
    # River replaces a micro-cluster with an updated copy whenever it absorbs
    # a point, so old structures are tracked by their creation time, which
    # the copies keep: a p-micro-cluster is "old" if it was created before
    # the topic switch.
    switch_time = None
    n_old_at_switch = 0
    for start in range(INITIAL_WARMUP_SIZE, len(embeddings), BATCH_SIZE):
        end = start + BATCH_SIZE
        clusterer.update(normalize(ipca.transform(embeddings[start:end])), labels=labels[start:end])
        metrics = clusterer.get_metrics()
        alive = list(clusterer.model.p_micro_clusters.values())
        if end == SAMPLES_PER_PHASE:
            switch_time = clusterer.model.timestamp
            n_old_at_switch = len(alive)
        survivors = None
        # Undefined when the model had already forgotten everything by the switch.
        if n_old_at_switch and end > SAMPLES_PER_PHASE:
            survivors = sum(mc.creation_time < switch_time for mc in alive) / n_old_at_switch
        records.append(
            {
                "seed": seed,
                "decaying_factor": decay,
                "samples_seen": end,
                "purity": metrics["purity"],
                "nmi": metrics["nmi"],
                "n_micro_clusters": metrics["n_micro_clusters"],
                "old_micro_clusters_alive": survivors,
            }
        )

    ts = pd.DataFrame(records)
    phase1 = ts[ts["samples_seen"] <= SAMPLES_PER_PHASE]
    phase2 = ts[ts["samples_seen"] > SAMPLES_PER_PHASE]
    summary = {
        "seed": seed,
        "decaying_factor": decay,
        "half_life_docs": clusterer.model.stream_speed / decay,
        "purity_before_switch": phase1["purity"].mean(),
        "nmi_before_switch": phase1["nmi"].mean(),
        "purity_after_switch": phase2["purity"].mean(),
        "nmi_after_switch": phase2["nmi"].mean(),
        "micro_clusters": ts["n_micro_clusters"].mean(),
        "micro_clusters_at_switch": n_old_at_switch,
    }
    for docs in SURVIVAL_CHECKPOINTS:
        row = phase2[phase2["samples_seen"] == SAMPLES_PER_PHASE + docs]
        summary[f"old_alive_after_{docs}"] = row["old_micro_clusters_alive"].iloc[0]
    return summary, ts


def plot_quality(summary: pd.DataFrame, out_path: str):
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in [("purity_before_switch", "Czystość przed zmianą tematów"), ("purity_after_switch", "Czystość po zmianie tematów")]:
        ax.errorbar(summary["decaying_factor"], summary[f"{col}_mean"], yerr=summary[f"{col}_std"], marker="o", capsize=4, lw=2, label=label)
    for bound in config.evolution.param_bounds["decaying_factor"]:
        ax.axvline(bound, color="grey", linestyle="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Współczynnik wygaszania λ (skala logarytmiczna)")
    ax.set_ylabel("Średnia czystość")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_purity_over_time(timeseries: pd.DataFrame, out_path: str):
    """Purity of the static model over the stream for the two most extreme
    lambda values (mean over stream orders): the drop at the topic switch
    and the lack of recovery, whatever the speed of forgetting. Warm-up and
    drift markers follow the thesis 2 figures."""
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvspan(0, INITIAL_WARMUP_SIZE, color="#b2ebf2", alpha=0.9, zorder=0, label="Rozgrzewka IPCA")
    extremes = [
        (min(LAMBDA_GRID), "#2980b9", "-", 3.0),
        (max(LAMBDA_GRID), "#e67e22", "--", 2.0),
    ]
    for decay, color, style, width in extremes:
        curve = timeseries[timeseries["decaying_factor"] == decay].groupby("samples_seen")["purity"].mean()
        ax.plot(curve.index, curve.values, color=color, linestyle=style, lw=width,
                label=f"λ = {str(decay).replace('.', ',')}")
    ax.axvline(SAMPLES_PER_PHASE, color="#1a1a1a", linestyle=(0, (1, 1)), lw=1.8,
               label=f"Zaplanowany dryf pojęć (t={SAMPLES_PER_PHASE})")
    ax.set_xlabel("Liczba przetworzonych dokumentów")
    ax.set_ylabel("Czystość")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, timeseries["samples_seen"].max())
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    grid = list(itertools.product(STREAM_SEEDS, LAMBDA_GRID))
    print(f"Running {len(grid)} simulations with {args.workers} worker process(es)...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_one, *zip(*grid)))

    runs = pd.DataFrame([summary for summary, _ in results])
    timeseries = pd.concat([ts for _, ts in results], ignore_index=True)
    metrics = [c for c in runs.columns if c not in ("seed", "decaying_factor")]
    summary = runs.groupby("decaying_factor")[metrics].agg(["mean", "std"])
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    summary = summary.reset_index()

    runs.to_csv(f"{RESULTS_DIR}/lambda_bounds_runs.csv", index=False)
    timeseries.to_csv(f"{RESULTS_DIR}/lambda_bounds_timeseries.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/lambda_bounds_summary.csv", index=False)
    plot_quality(summary, f"{RESULTS_DIR}/lambda_bounds_quality.png")
    plot_purity_over_time(timeseries, f"{RESULTS_DIR}/lambda_bounds_purity_over_time.png")

    cols = ["decaying_factor", "half_life_docs_mean", "purity_before_switch_mean", "purity_before_switch_std",
            "purity_after_switch_mean", "purity_after_switch_std", "micro_clusters_mean"] + [f"old_alive_after_{d}_mean" for d in SURVIVAL_CHECKPOINTS]
    print(summary[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
