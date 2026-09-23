"""
A-priori analysis of the SBERT embedding space (thesis section 3.3), run on
the same 5000-document stream as thesis 1 (the pre-drift phase of thesis 2):

  - intrinsic dimensionality, estimated with the Levina-Bickel maximum
    likelihood estimator (MLE) over k = 10..20 nearest neighbours,
  - cumulative variance explained by the first d principal components,
  - distance contrast ratio DCR = (max - min) / mean pairwise distance,
    after projecting to d components and L2-normalising (as in the system);
    it drops as distances concentrate in high dimensions.

Standard PCA on the whole sample is used here (not IPCA) to obtain exact
explained-variance values.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from experiments.theses.thesis_1.exp_thesis_1_ipca import load_phase1_stream

RESULTS_DIR = "experiments/dimensionality/results"
DIMS = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 384]
MARKED_DIM = 16
DCR_SAMPLE_SIZE = 2000
SEED = 42


def intrinsic_dimension_mle(X: np.ndarray, k1: int = 10, k2: int = 20) -> float:
    """Levina-Bickel MLE, averaged over points and over k = k1..k2."""
    distances, _ = NearestNeighbors(n_neighbors=k2 + 1).fit(X).kneighbors(X)
    distances = distances[:, 1:]  # drop each point's zero distance to itself
    estimates = []
    for k in range(k1, k2 + 1):
        log_ratios = np.log(distances[:, k - 1 : k]) - np.log(distances[:, : k - 1])
        estimates.append(np.mean((k - 1) / log_ratios.sum(axis=1)))
    return float(np.mean(estimates))


def distance_contrast_ratio(X: np.ndarray) -> float:
    d = pdist(X)
    d = d[d > 1e-8]
    return float((d.max() - d.min()) / d.mean())


def plot_curve(dims, values, ylabel, title, out_path):
    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims, values, color="#2980b9", linewidth=2.5, marker="o")
    ax.axvline(x=MARKED_DIM, color="#e74c3c", linestyle="--", linewidth=2, label=f"d = {MARKED_DIM}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims)
    ax.set_xticklabels(dims)
    ax.set_title(title)
    ax.set_xlabel("Wymiarowość przestrzeni (skala logarytmiczna)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    _, embeddings, labels = load_phase1_stream()
    print(f"Documents: {len(embeddings)}, categories: {len(set(labels))}")

    id_mle = intrinsic_dimension_mle(embeddings)
    print(f"Intrinsic dimensionality (MLE): {id_mle:.2f}")

    pca = PCA(n_components=embeddings.shape[1]).fit(embeddings)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    sample = np.random.default_rng(SEED).choice(len(embeddings), DCR_SAMPLE_SIZE, replace=False)

    rows = []
    for d in DIMS:
        if d == embeddings.shape[1]:
            projected = embeddings[sample]
        else:
            projected = normalize(pca.transform(embeddings[sample])[:, :d])
        rows.append(
            {
                "d": d,
                "explained_variance_pct": 100.0 * cumulative_variance[d - 1],
                "dcr": distance_contrast_ratio(projected),
            }
        )
        print(f"d={d:>3} | variance: {rows[-1]['explained_variance_pct']:5.1f}% | DCR: {rows[-1]['dcr']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/dimensionality_metrics.csv", index=False)
    with open(f"{RESULTS_DIR}/dimensionality_summary.json", "w") as f:
        json.dump(
            {"n_documents": len(embeddings), "n_categories": len(set(labels)), "intrinsic_dimensionality_mle": id_mle},
            f,
            indent=4,
        )

    plot_curve(
        DIMS, df["explained_variance_pct"], "Skumulowana wariancja wyjaśniona [%]",
        "Skumulowana wariancja wyjaśniona przez składowe główne",
        f"{RESULTS_DIR}/dimensionality_variance.png",
    )
    plot_curve(
        DIMS, df["dcr"], "Współczynnik kontrastu odległości (DCR)",
        "Kontrast odległości w funkcji wymiarowości",
        f"{RESULTS_DIR}/dimensionality_dcr.png",
    )


if __name__ == "__main__":
    main()
