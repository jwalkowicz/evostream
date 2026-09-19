import json
import logging
import os

import numpy as np
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def estimate_intrinsic_dimensionality_mle(
    X: np.ndarray, k1: int = 10, k2: int = 20
) -> float:
    """
    Maximum Likelihood Estimation of Intrinsic Dimensionality.
    Uses the distance to the k-nearest neighbors to estimate the manifold dimension.
    """

    logger.info("Estimating Intrinsic Dimensionality (MLE) using k-NN...")
    nn = NearestNeighbors(n_neighbors=k2 + 1, algorithm="auto", metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    distances = distances[:, 1:]

    id_estimates = []
    for k in range(k1, k2 + 1):
        r_k = distances[:, k - 1]

        valid_idx = r_k > 1e-8

        if np.sum(valid_idx) == 0:
            continue

        r_k_valid = r_k[valid_idx]
        d_valid = distances[valid_idx, : k - 1]

        log_r_k = np.log(r_k_valid).reshape(-1, 1)
        log_r_i = np.log(d_valid + 1e-8)

        sum_log_ratio = np.sum(log_r_k - log_r_i, axis=1)
        m_hat = (k - 1) / (sum_log_ratio + 1e-8)

        id_estimates.append(np.mean(m_hat))

    if not id_estimates:
        return float("nan")

    return float(np.mean(id_estimates))


def compute_distance_contrast_ratio(X: np.ndarray, sample_size: int = 2000) -> float:
    """
    Computes the Distance Contrast Ratio (DCR).
    As D -> infinity, DCR -> 0 (all distances become equal).
    """
    if len(X) > sample_size:
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    distances = pdist(X_sample, metric="euclidean")

    if len(distances) == 0:
        return 0.0

    d_max = np.max(distances)
    d_min = np.min(distances[distances > 1e-8]) if np.any(distances > 1e-8) else 0.0
    d_mean = np.mean(distances)

    return float((d_max - d_min) / (d_mean + 1e-8))


def run_dimensionality_experiment():
    """
    Runs the full dimensionality justification experiment.
    """
    logger.info("Fetching 20 Newsgroups data (subset for experiment)...")
    categories = ["sci.space", "comp.graphics", "rec.autos", "talk.politics.mideast"]
    dataset = fetch_20newsgroups(
        subset="train", categories=categories, remove=("headers", "footers", "quotes")
    )

    texts = dataset.data[:2000]
    texts = [t.strip() for t in texts if len(t.strip()) > 50]
    logger.info(f"Loaded {len(texts)} valid documents.")

    logger.info("Generating SBERT embeddings (all-MiniLM-L6-v2) in 384D...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_384d = model.encode(
        texts, show_progress_bar=True, normalize_embeddings=True
    )
    embeddings_384d = np.array(embeddings_384d)

    id_mle = estimate_intrinsic_dimensionality_mle(embeddings_384d)
    logger.info(f"==> Estimated Intrinsic Dimensionality (Manifold): {id_mle:.2f}")

    dimensions_to_test = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 384]
    results = []

    logger.info("Computing metrics across projection dimensions...")

    full_pca = PCA(n_components=min(len(texts), 384))
    full_pca.fit(embeddings_384d)
    cumulative_evr_all = np.cumsum(full_pca.explained_variance_ratio_)

    for d in dimensions_to_test:
        if d == 384:
            X_proj = embeddings_384d
            evr = 1.0
        else:
            pca = PCA(n_components=d)
            X_proj = pca.fit_transform(embeddings_384d)
            norms = np.linalg.norm(X_proj, axis=1, keepdims=True)
            X_proj = X_proj / np.maximum(norms, 1e-8)
            evr = float(cumulative_evr_all[d - 1])

        dcr = compute_distance_contrast_ratio(X_proj)

        logger.info(f" d={d:<3} | EVR: {evr * 100:>5.1f}% | DCR: {dcr:.4f}")

        results.append({"d": d, "evr": evr, "dcr": dcr})

    output_data = {
        "dataset_size": len(texts),
        "intrinsic_dimensionality": id_mle,
        "projection_metrics": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results/dimensionality_results.json",
    )
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"Experiment complete! Results saved to {out_path}")


if __name__ == "__main__":
    run_dimensionality_experiment()
