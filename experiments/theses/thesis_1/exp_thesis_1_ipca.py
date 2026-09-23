"""
Thesis 1 experiment: SBERT + IPCA vs. full-dimensional SBERT in the DenStream
online phase - clustering quality vs. per-document processing time.

The environment mirrors thesis 2 so the two experiments are directly
comparable: the same phase-1 document stream (6 categories, 5000 documents,
same cleaning, shuffle and SBERT embeddings), the same batch size, the same
IPCA warmup and the same DenStream settings from config.

Every dimensionality is run over one shared epsilon grid and reported at its
best epsilon, so no variant is judged at a value tuned for another one.
Each configuration is repeated on several shuffled orders of the stream
(STREAM_SEEDS), and results are reported as mean +/- std across orders.

The micro-cluster radius formula is selectable (--radius), see
src/domain/clustering.py and https://github.com/online-ml/river/issues/2004.
Each formula writes its own result files.
"""

import argparse
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from experiments.theses.thesis_2.exp_thesis_2_drift import (
    BATCH_SIZE,
    INITIAL_WARMUP_SIZE,
    PHASE1_CATEGORIES,
    SAMPLES_PER_PHASE,
    _load_or_build_dataset,
    _load_or_compute_embeddings,
)
from src.core.common import set_seed
from src.core.config import config
from src.core.logger import logger
from src.domain.clustering import StreamClusterer, set_river_radius_fix

RESULTS_DIR = "experiments/theses/thesis_1/results"

# None = full 384-dimensional SBERT embeddings, no projection.
PCA_DIMS: List[Optional[int]] = [None, 128, 64, 32, 16, 8]
EPSILON_GRID = [0.05, 0.10, 0.20, 0.30, 0.50]
# Each seed shuffles the stream differently, which also changes the documents
# IPCA is fitted on - the spread across seeds is the run-to-run noise.
STREAM_SEEDS = [0, 1, 2]
SBERT_TIMING_SAMPLE = 500


def load_phase1_stream():
    """The first SAMPLES_PER_PHASE documents of the thesis 2 stream, i.e.
    exactly the pre-drift part of that experiment."""
    texts, labels = _load_or_build_dataset()
    embeddings = _load_or_compute_embeddings(texts)
    return (
        texts[:SAMPLES_PER_PHASE],
        embeddings[:SAMPLES_PER_PHASE],
        labels[:SAMPLES_PER_PHASE],
    )


def shuffle_stream(embeddings: np.ndarray, labels: List[str], seed: int):
    order = np.random.default_rng(seed).permutation(len(embeddings))
    return embeddings[order], [labels[i] for i in order]


def measure_sbert_latency_ms(texts: List[str]) -> float:
    """SBERT cost per document is the same for every variant, so it is
    measured once on a sample instead of re-encoding the whole stream."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(config.ml.embedding_model, device="cpu")
    sample = texts[:SBERT_TIMING_SAMPLE]
    encoder.encode(sample[:32], batch_size=32)  # warm-up, excluded from timing
    t0 = time.perf_counter()
    encoder.encode(sample, batch_size=128, normalize_embeddings=True)
    return (time.perf_counter() - t0) * 1000.0 / len(sample)


def run_streaming_simulation(
    embeddings: np.ndarray,
    labels: List[str],
    pca_dim: Optional[int],
    epsilon: float,
    decaying_factor: Optional[float] = None,
) -> pd.DataFrame:
    """Streams the documents through (IPCA ->) DenStream batch by batch.

    IPCA is fitted once on the first INITIAL_WARMUP_SIZE documents and then
    frozen, as in thesis 2. Those warmup documents are not clustered or
    scored. Latency covers only the per-batch stream work (projection +
    clustering); the one-off IPCA fit is reported separately. The decaying
    factor defaults to the config value.
    """
    set_seed(config.seed)
    n_categories = len(set(labels))

    ipca_fit_ms = 0.0
    ipca = None
    if pca_dim is not None:
        ipca = IncrementalPCA(n_components=pca_dim)
        t0 = time.perf_counter()
        ipca.partial_fit(embeddings[:INITIAL_WARMUP_SIZE])
        ipca_fit_ms = (time.perf_counter() - t0) * 1000.0

    clusterer = StreamClusterer(
        epsilon=epsilon,
        mu=config.denstream.mu,
        beta=config.denstream.beta,
        decaying_factor=decaying_factor or config.denstream.decaying_factor,
        n_samples_init=config.denstream.n_samples_init,
        window_size=config.denstream.window_size,
        expected_macro_clusters=n_categories,
    )

    records = []
    for start in range(INITIAL_WARMUP_SIZE, len(embeddings), BATCH_SIZE):
        batch = embeddings[start : start + BATCH_SIZE]
        batch_labels = labels[start : start + BATCH_SIZE]

        t0 = time.perf_counter()
        vectors = normalize(ipca.transform(batch)) if ipca is not None else batch
        transform_ms = (time.perf_counter() - t0) * 1000.0 / len(batch)

        t0 = time.perf_counter()
        clusterer.update(vectors, labels=batch_labels)
        cluster_ms = (time.perf_counter() - t0) * 1000.0 / len(batch)

        metrics = clusterer.get_metrics()
        records.append(
            {
                "pca_dim": pca_dim or embeddings.shape[1],
                "epsilon": epsilon,
                "samples_seen": start + len(batch),
                "transform_ms": transform_ms,
                "cluster_ms": cluster_ms,
                "stream_ms": transform_ms + cluster_ms,
                "ipca_fit_ms": ipca_fit_ms,
                "n_micro_clusters": metrics["n_micro_clusters"],
                "n_outlier_clusters": metrics["n_outlier_clusters"],
                "purity": metrics["purity"],
                "nmi": metrics["nmi"],
                "ari": metrics["ari"],
                "silhouette": metrics["silhouette"],
            }
        )

    df = pd.DataFrame(records)
    logger.info(
        f"d={df['pca_dim'].iloc[0]}, eps={epsilon}: "
        f"purity={df['purity'].mean():.3f}, nmi={df['nmi'].mean():.3f}, "
        f"stream={df['stream_ms'].mean():.3f} ms/doc"
    )
    return df


def summarize(timeseries: pd.DataFrame, sbert_ms: float) -> pd.DataFrame:
    """One row per (dimension, epsilon): each run is first averaged over the
    stream, then mean and std are taken across stream seeds. The best
    epsilon (by mean purity) is flagged for each dimension. Speed-up is
    relative to full SBERT at its own best epsilon."""
    per_run = (
        timeseries.groupby(["pca_dim", "epsilon", "seed"])
        .agg(
            purity=("purity", "mean"),
            nmi=("nmi", "mean"),
            ari=("ari", "mean"),
            silhouette=("silhouette", "mean"),
            micro_clusters=("n_micro_clusters", "mean"),
            stream_ms=("stream_ms", "mean"),
            ipca_fit_ms=("ipca_fit_ms", "first"),
        )
        .reset_index()
    )
    summary = (
        per_run.groupby(["pca_dim", "epsilon"])
        .agg(
            mean_purity=("purity", "mean"),
            std_purity=("purity", "std"),
            mean_nmi=("nmi", "mean"),
            std_nmi=("nmi", "std"),
            mean_ari=("ari", "mean"),
            mean_silhouette=("silhouette", "mean"),
            mean_micro_clusters=("micro_clusters", "mean"),
            mean_stream_ms=("stream_ms", "mean"),
            std_stream_ms=("stream_ms", "std"),
            ipca_fit_ms=("ipca_fit_ms", "mean"),
        )
        .reset_index()
    )
    summary["mean_total_ms"] = summary["mean_stream_ms"] + sbert_ms

    best_idx = summary.groupby("pca_dim")["mean_purity"].idxmax()
    summary["is_best_epsilon"] = summary.index.isin(best_idx)

    full_dim = summary["pca_dim"].max()
    full_best = summary[(summary["pca_dim"] == full_dim) & summary["is_best_epsilon"]].iloc[0]
    summary["stream_speedup"] = full_best["mean_stream_ms"] / summary["mean_stream_ms"]
    summary["purity_vs_full"] = summary["mean_purity"] / full_best["mean_purity"]
    return summary.sort_values(["pca_dim", "epsilon"], ascending=[False, True])


def plot_best_purity(timeseries: pd.DataFrame, summary: pd.DataFrame, out_path: str):
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(10, 6))

    best = summary[summary["is_best_epsilon"]]
    for _, row in best.iterrows():
        sub = timeseries[
            (timeseries["pca_dim"] == row["pca_dim"])
            & (timeseries["epsilon"] == row["epsilon"])
        ]
        name = "Pełne SBERT (384d)" if row["pca_dim"] == 384 else f"IPCA (d={row['pca_dim']})"
        mean_over_seeds = sub.groupby("samples_seen")["purity"].mean()
        ax.plot(
            mean_over_seeds.index,
            mean_over_seeds.ewm(span=5).mean(),
            label=f"{name}, ε={row['epsilon']}",
            lw=2.0,
        )

    ax.set_title("Czystość klastrów tematycznych (najlepsze ε, średnia z kolejności strumienia)")
    ax.set_xlabel("Liczba przetworzonych dokumentów")
    ax.set_ylabel("Czystość")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--radius",
        choices=["river", "fixed"],
        default="fixed" if config.denstream.fix_river_radius else "river",
        help="Micro-cluster radius formula (default: the one set in config).",
    )
    args = parser.parse_args()
    set_river_radius_fix(args.radius == "fixed")

    logger.add("logs/thesis_1_ipca.log", rotation="500 MB")
    logger.info(
        f"Thesis 1 | radius={args.radius} | categories={PHASE1_CATEGORIES} | "
        f"docs={SAMPLES_PER_PHASE} | batch={BATCH_SIZE} | warmup={INITIAL_WARMUP_SIZE} | "
        f"seeds={STREAM_SEEDS}"
    )

    texts, embeddings, labels = load_phase1_stream()
    sbert_ms = measure_sbert_latency_ms(texts)
    logger.info(f"SBERT latency: {sbert_ms:.2f} ms/doc")

    runs = []
    for seed in STREAM_SEEDS:
        seed_embeddings, seed_labels = shuffle_stream(embeddings, labels, seed)
        for pca_dim in PCA_DIMS:
            for eps in EPSILON_GRID:
                run = run_streaming_simulation(seed_embeddings, seed_labels, pca_dim, eps)
                runs.append(run.assign(seed=seed))
    timeseries = pd.concat(runs, ignore_index=True)
    summary = summarize(timeseries, sbert_ms)

    suffix = f"radius_{args.radius}"
    timeseries.to_csv(f"{RESULTS_DIR}/thesis_1_timeseries_{suffix}.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/thesis_1_summary_{suffix}.csv", index=False)
    plot_best_purity(timeseries, summary, f"{RESULTS_DIR}/thesis_1_purity_{suffix}.pdf")

    print(summary[summary["is_best_epsilon"]].to_string(index=False))


if __name__ == "__main__":
    main()
