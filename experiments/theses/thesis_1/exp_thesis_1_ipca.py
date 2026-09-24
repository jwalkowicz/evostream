"""
Thesis 1 experiment: clustering quality and per-document time of DenStream
on full SBERT embeddings vs. IPCA projections of several dimensions.

Uses the first phase of the thesis 2 stream with the same settings. Every
dimension runs over the same epsilon grid and is reported at its best
epsilon; each configuration is repeated on several stream orders
(STREAM_SEEDS). --radius selects the micro-cluster radius formula
(river's or the corrected one, see src/domain/clustering.py).
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
# Each seed gives a different document order (and so a different warm-up sample).
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

    The first INITIAL_WARMUP_SIZE documents form the warm-up buffer: IPCA is
    fitted on them once and then frozen, and DenStream is warm-started on
    them - the same procedure as after a model swap in thesis 2. They are not
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
    warmup = embeddings[:INITIAL_WARMUP_SIZE]
    clusterer.warm_start(normalize(ipca.transform(warmup)) if ipca is not None else warmup)

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


def plot_purity_band(timeseries: pd.DataFrame, summary: pd.DataFrame, out_path: str):
    """Purity over the stream, one panel per dimension: the line is the best
    epsilon, the band the min-max range over the whole epsilon grid (each
    curve averaged over stream orders), so the figure also shows how
    sensitive every dimension is to epsilon and where the model collapses."""
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    dims = sorted(timeseries["pca_dim"].unique(), reverse=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    best_eps = summary[summary["is_best_epsilon"]].set_index("pca_dim")["epsilon"]
    grid = "; ".join(f"{e:.2f}".replace(".", ",") for e in EPSILON_GRID)

    for ax, dim in zip(axes.flat, dims):
        curves = (
            timeseries[timeseries["pca_dim"] == dim]
            .groupby(["epsilon", "samples_seen"])["purity"].mean()
            .unstack("epsilon")
            .ewm(span=5).mean()
        )
        ax.axvspan(0, INITIAL_WARMUP_SIZE, color="#b2ebf2", alpha=0.9, zorder=0,
                   label="Rozgrzewka (IPCA i DenStream)")
        ax.fill_between(curves.index, curves.min(axis=1), curves.max(axis=1), color="#2980b9",
                        alpha=0.25, lw=0, label=f"Zakres dla ε ∈ {{{grid}}}")
        ax.plot(curves.index, curves[best_eps[dim]], color="#2980b9", lw=2,
                label="Najlepsza wartość ε")
        ax.set_title("d = 384 (bez IPCA)" if dim == 384 else f"d = {dim}", fontsize=11)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0, timeseries["samples_seen"].max())
        ax.grid(True, linestyle="--", alpha=0.6)

    for ax in axes[1]:
        ax.set_xlabel("Liczba przetworzonych dokumentów")
    for ax in axes[:, 0]:
        ax.set_ylabel("Czystość")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    plt.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
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
    plot_purity_band(timeseries, summary, f"{RESULTS_DIR}/thesis_1_purity_{suffix}.png")

    print(summary[summary["is_best_epsilon"]].to_string(index=False))


if __name__ == "__main__":
    main()
