"""Thesis 3 experiment: Pareto fronts and compromise solutions of NSGA-II for several projection dimensions."""

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river import stream
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import normalize

from experiments.plot_style import use_polish_number_format
from experiments.theses.thesis_1.exp_thesis_1_ipca import STREAM_SEEDS, shuffle_stream
from experiments.theses.thesis_2.exp_thesis_2_drift import (
    HOTSWAP_BUFFER_SIZE,
    PHASE1_CATEGORIES,
    SAMPLES_PER_PHASE,
    _load_or_build_dataset,
    _load_or_compute_embeddings,
)
from src.core.config import config
from src.core.logger import logger
from src.domain.clustering import StreamClusterer, purity_score
from src.domain.evolution import NSGAIIOptimizer, evaluate_parameters

RESULTS_DIR = "experiments/theses/thesis_3/results"
PCA_DIMS = [8, 16, 32, 64, 128]
PHASES = [1, 2]
N_MACRO_CLUSTERS = len(PHASE1_CATEGORIES)
EXAMPLE_RUN = (1, STREAM_SEEDS[0])  # (phase, seed) shown in the Pareto-front figure


def load_buffer(phase: int, seed: int) -> tuple[np.ndarray, list[str]]:
    """HOTSWAP_BUFFER_SIZE consecutive documents of one stream phase, after
    shuffling that phase with the given seed."""
    texts, labels = _load_or_build_dataset()
    embeddings = _load_or_compute_embeddings(texts)
    start = (phase - 1) * SAMPLES_PER_PHASE
    phase_slice = slice(start, start + SAMPLES_PER_PHASE)
    shuffled, shuffled_labels = shuffle_stream(embeddings[phase_slice], labels[phase_slice], seed)
    return shuffled[:HOTSWAP_BUFFER_SIZE], shuffled_labels[:HOTSWAP_BUFFER_SIZE]


def score_against_labels(params: dict, buffer: np.ndarray, labels: list[str]) -> tuple[float, float]:
    """Purity and NMI of a DenStream model with the given parameters, trained
    on the buffer the same way as in a model swap."""
    clusterer = StreamClusterer(expected_macro_clusters=N_MACRO_CLUSTERS, window_size=len(buffer))
    clusterer.hot_swap_model(new_params=params, window_data=buffer)
    preds = [clusterer.predict_one(x) for x, _ in stream.iter_array(buffer)]
    return purity_score(labels, preds), float(normalized_mutual_info_score(labels, preds))


def run_one(pca_dim: int, phase: int, seed: int) -> tuple[list[dict], dict]:
    raw_buffer, labels = load_buffer(phase, seed)
    ipca = IncrementalPCA(n_components=pca_dim)
    ipca.partial_fit(raw_buffer)
    buffer = normalize(ipca.transform(raw_buffer))

    optimizer = NSGAIIOptimizer(
        n_macro_clusters=N_MACRO_CLUSTERS,
        population_size=config.evolution.population_size,
        generations=config.evolution.generations,
        crossover_rate=config.evolution.crossover_rate,
        crossover_eta=config.evolution.crossover_eta,
        mutation_rate=config.evolution.mutation_rate,
        mutation_eta=config.evolution.mutation_eta,
        param_bounds=config.evolution.param_bounds,
        fixed_mu=config.denstream.mu,
        n_samples_init=config.denstream.n_samples_init,
        seed=config.evolution.seed + seed,
    )
    t0 = time.perf_counter()
    compromise, front, _ = optimizer.evolve(data_buffer=buffer)
    optimization_s = time.perf_counter() - t0

    # Parameters are rounded to 4 decimals, so distinct front members can
    # coincide - keep one of each.
    unique_front = list({(ind.params["epsilon"], ind.params["decaying_factor"]): ind for ind in front}.values())

    run_id = {"pca_dim": pca_dim, "phase": phase, "seed": seed}
    dict_buffer = [dict(enumerate(row)) for row in buffer]
    front_rows = []
    for ind in unique_front:
        _, _, n_micro = evaluate_parameters(
            dict_buffer,
            epsilon=ind.params["epsilon"],
            decaying_factor=ind.params["decaying_factor"],
            n_macro_clusters=N_MACRO_CLUSTERS,
            mu=config.denstream.mu,
            n_samples_init=config.denstream.n_samples_init,
        )
        front_rows.append(
            {
                **run_id,
                "epsilon": ind.params["epsilon"],
                "decaying_factor": ind.params["decaying_factor"],
                "quality": ind.quality_score,
                "complexity": ind.complexity_score,
                "n_micro_clusters": n_micro,
                "micro_macro_ratio": n_micro / N_MACRO_CLUSTERS,
                "is_compromise": ind.params == compromise.params,
            }
        )

    compromise_row = next(r for r in front_rows if r["is_compromise"])
    purity, nmi = score_against_labels(compromise.params, buffer, labels)
    summary_row = {
        **run_id,
        "front_size": len(unique_front),
        "optimization_s": optimization_s,
        **{
            k: compromise_row[k]
            for k in ("epsilon", "decaying_factor", "quality", "complexity", "n_micro_clusters", "micro_macro_ratio")
        },
        "purity": purity,
        "nmi": nmi,
    }
    logger.info(
        f"d={pca_dim}, phase={phase}, seed={seed}: front={len(unique_front)}, "
        f"compromise eps={summary_row['epsilon']}, decay={summary_row['decaying_factor']}, "
        f"ratio={summary_row['micro_macro_ratio']:.2f}, purity={purity:.3f}, nmi={nmi:.3f}"
    )
    return front_rows, summary_row


def aggregate_by_dimension(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and std of the compromise solution across all repetitions
    (both phases x all seeds) for each dimension."""
    metrics = [
        "front_size",
        "epsilon",
        "decaying_factor",
        "quality",
        "complexity",
        "micro_macro_ratio",
        "purity",
        "nmi",
        "optimization_s",
    ]
    agg = runs.groupby("pca_dim")[metrics].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


def _dimension_axis(ax, ylabel: str):
    ax.set_xscale("log", base=2)
    ax.set_xticks(PCA_DIMS)
    ax.set_xticklabels(PCA_DIMS)
    ax.set_xlabel("Wymiar projekcji IPCA (d)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)


def plot_fronts(fronts: pd.DataFrame, out_path: str):
    use_polish_number_format()
    phase, seed = EXAMPLE_RUN
    fig, ax = plt.subplots(figsize=(9, 6))
    for d in PCA_DIMS:
        sub = fronts[(fronts["pca_dim"] == d) & (fronts["phase"] == phase) & (fronts["seed"] == seed)]
        sub = sub.sort_values("complexity")
        (line,) = ax.plot(sub["complexity"], sub["quality"], "-o", markersize=4, lw=1.5, alpha=0.8, label=f"d = {d}")
        comp = sub[sub["is_compromise"]]
        ax.scatter(
            comp["complexity"], comp["quality"], s=180, marker="*", color=line.get_color(), edgecolor="black", zorder=5
        )
    ax.scatter([], [], s=180, marker="*", color="white", edgecolor="black", label="rozwiązanie kompromisowe")
    ax.set_xlabel("Złożoność strukturalna $f_2$")
    ax.set_ylabel("Jakość $f_1$ (wskaźnik sylwetki środków mikroklastrów)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_compromise_quality(summary: pd.DataFrame, out_path: str):
    use_polish_number_format()
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label in [("purity", "Czystość"), ("nmi", "NMI")]:
        ax.errorbar(
            summary["pca_dim"],
            summary[f"{metric}_mean"],
            yerr=summary[f"{metric}_std"],
            marker="o",
            capsize=4,
            lw=2,
            label=label,
        )
    _dimension_axis(ax, "Wartość miary")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_compromise_structure(summary: pd.DataFrame, out_path: str):
    use_polish_number_format()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        summary["pca_dim"],
        summary["micro_macro_ratio_mean"],
        yerr=summary["micro_macro_ratio_std"],
        marker="o",
        capsize=4,
        lw=2,
    )
    _dimension_axis(ax, "$N_{micro} / N_{macro}$")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_compromise_params(summary: pd.DataFrame, out_path: str):
    use_polish_number_format()
    bounds = config.evolution.param_bounds
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, param, label in [(axes[0], "epsilon", "$\\varepsilon^*$"), (axes[1], "decaying_factor", "$\\lambda^*$")]:
        ax.errorbar(
            summary["pca_dim"], summary[f"{param}_mean"], yerr=summary[f"{param}_std"], marker="o", capsize=4, lw=2
        )
        ax.set_ylim(*bounds[param])
        _dimension_axis(ax, label)
    axes[0].set_xlabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logger.add("logs/thesis_3_pareto.log", rotation="500 MB")
    grid = list(itertools.product(PCA_DIMS, PHASES, STREAM_SEEDS))
    logger.info(
        f"Thesis 3 | dims={PCA_DIMS} | phases={PHASES} | seeds={STREAM_SEEDS} | "
        f"buffer={HOTSWAP_BUFFER_SIZE} | k={N_MACRO_CLUSTERS} | bounds={config.evolution.param_bounds}"
    )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_one, *zip(*grid)))

    fronts = pd.DataFrame([row for front_rows, _ in results for row in front_rows])
    runs = pd.DataFrame([summary_row for _, summary_row in results])
    summary = aggregate_by_dimension(runs)

    fronts.to_csv(f"{RESULTS_DIR}/thesis_3_pareto_fronts.csv", index=False)
    runs.to_csv(f"{RESULTS_DIR}/thesis_3_compromise_runs.csv", index=False)
    summary.to_csv(f"{RESULTS_DIR}/thesis_3_summary.csv", index=False)

    plot_fronts(fronts, f"{RESULTS_DIR}/thesis_3_pareto_fronts.png")
    plot_compromise_quality(summary, f"{RESULTS_DIR}/thesis_3_compromise_quality.png")
    plot_compromise_structure(summary, f"{RESULTS_DIR}/thesis_3_compromise_structure.png")
    plot_compromise_params(summary, f"{RESULTS_DIR}/thesis_3_compromise_params.png")

    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
