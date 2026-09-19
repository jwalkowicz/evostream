import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

plt.style.use("default")
plt.rcParams["font.family"] = "DejaVu Sans"


def main():
    print("Loading data and model...")
    device = "cpu"
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    cats = ["sci.space", "sci.med", "rec.autos"]
    raw = fetch_20newsgroups(
        subset="all", categories=cats, remove=("headers", "footers", "quotes")
    )

    texts = [t for t in raw.data if len(t.split()) >= 10][:1500]

    print(f"Encoding {len(texts)} documents...")
    embeddings = encoder.encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )

    print("Computing PCA Explained Variance...")
    pca_full = PCA(n_components=384)
    pca_full.fit(embeddings)

    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(2, 385), cum_var[1:] * 100, color="#2980b9", linewidth=2.5)

    ax.set_xscale("log", base=2)
    ticks = [2, 4, 8, 16, 32, 64, 128, 384]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_xlim(left=2, right=384)

    ax.axvline(
        x=16,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label="d = 16",
    )

    ax.set_title(
        "Skumulowana wariancja wyjaśniona przez główne składowe PCA", fontsize=13
    )
    ax.set_xlabel("Wymiarowość przestrzeni (skala logarytmiczna)", fontsize=11)
    ax.set_ylabel("Skumulowana wariancja wyjaśniona [%]", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("experiments/dimensionality/results/pca_variance.png", dpi=300)

    # 2. Clustering Quality, Time & DCR vs Dimensionality
    dims_to_test = [2, 4, 8, 16, 32, 64, 128, 384]
    sil_scores = []
    time_scores = []
    dcr_scores = []

    def estimate_intrinsic_dimensionality_mle(X, k1=10, k2=20):

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
        return float(np.mean(id_estimates)) if id_estimates else float("nan")

    def compute_distance_contrast_ratio(X, sample_size=1000):
        X_sample = X[np.random.choice(len(X), min(sample_size, len(X)), replace=False)]
        distances = pdist(X_sample, metric="euclidean")
        if len(distances) == 0:
            return 0.0
        d_max, d_min, d_mean = (
            np.max(distances),
            np.min(distances[distances > 1e-8]) if np.any(distances > 1e-8) else 0.0,
            np.mean(distances),
        )
        return float((d_max - d_min) / (d_mean + 1e-8))

    id_mle = estimate_intrinsic_dimensionality_mle(embeddings)
    print(
        f"\nEstimated Intrinsic Dimensionality (MLE) of raw 384D space: {id_mle:.2f}\n"
    )

    print("Evaluating clustering quality, speed, and DCR across dimensions...")
    for d in dims_to_test:
        if d == 384:
            X_red = embeddings
        else:
            pca = PCA(n_components=d, random_state=42)
            X_red = normalize(pca.fit_transform(embeddings))

        dcr = compute_distance_contrast_ratio(X_red)
        dcr_scores.append(dcr)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

        start_time = time.perf_counter()
        preds = kmeans.fit_predict(X_red)
        end_time = time.perf_counter()

        exec_time = (end_time - start_time) * 1000  # to milliseconds
        sil = silhouette_score(X_red, preds)

        sil_scores.append(sil)
        time_scores.append(exec_time)
        print(f"Dim {d:>3} -> Silhouette: {sil:.4f} | Time: {exec_time:.2f} ms")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dims_to_test, sil_scores, color="#2980b9", linewidth=2.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims_to_test)
    ax.set_xticklabels(dims_to_test)
    ax.set_xlim(left=2, right=384)

    ax.axvline(x=16, color="#e74c3c", linestyle="--", linewidth=2, label="d = 16")

    ax.set_title("Wpływ redukcji wymiarowości na jakość separacji", fontsize=13)
    ax.set_xlabel("Wymiarowość przestrzeni (skala logarytmiczna)", fontsize=11)
    ax.set_ylabel("Indeks sylwetki", fontsize=11)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("experiments/dimensionality/results/pca_silhouette.png", dpi=300)

    # 3. Processing Time Plot
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(dims_to_test, time_scores, color="#2980b9", linewidth=2.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(dims_to_test)
    ax2.set_xticklabels(dims_to_test)
    ax2.set_xlim(left=2, right=384)

    ax2.axvline(x=16, color="#e74c3c", linestyle="--", linewidth=2, label="d = 16")

    ax2.set_title("Wpływ wymiarowości reprezentacji na czas klasteryzacji", fontsize=13)
    ax2.set_xlabel("Wymiarowość przestrzeni (skala logarytmiczna)", fontsize=11)
    ax2.set_ylabel("Czas wykonania [ms]", fontsize=11)
    ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("experiments/dimensionality/results/pca_time.png", dpi=300)

    # 4. DCR Plot
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(dims_to_test, dcr_scores, color="#2980b9", linewidth=2.5)
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(dims_to_test)
    ax3.set_xticklabels(dims_to_test)
    ax3.set_xlim(left=2, right=384)

    ax3.axvline(x=16, color="#e74c3c", linestyle="--", linewidth=2, label="d = 16")

    ax3.set_title(
        "Degradacja kontrastu odległości (DCR) w funkcji wymiarowości", fontsize=13
    )
    ax3.set_xlabel("Wymiarowość przestrzeni (skala logarytmiczna)", fontsize=11)
    ax3.set_ylabel("Współczynnik kontrastu odległości (DCR)", fontsize=11)
    ax3.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("experiments/dimensionality/results/pca_dcr.png", dpi=300)

    # Save results to CSV
    csv_path = "experiments/dimensionality/results/pca_metrics_results.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Dimension",
                "Explained_Variance_Percent",
                "Silhouette_Score",
                "Processing_Time_ms",
                "DCR",
            ]
        )
        for i, d in enumerate(dims_to_test):
            evr_val = 100.0 if d == 384 else cum_var[d - 1] * 100
            writer.writerow(
                [
                    d,
                    f"{evr_val:.2f}",
                    f"{sil_scores[i]:.4f}",
                    f"{time_scores[i]:.2f}",
                    f"{dcr_scores[i]:.4f}",
                ]
            )

if __name__ == "__main__":
    main()
