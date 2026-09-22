import collections

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river import cluster
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

from src.apps.web_ui import load_encoder_and_data


def calculate_purity(y_true, y_pred):
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0
    clusters = collections.defaultdict(list)
    for t, p in zip(y_true, y_pred):
        if p != -1:
            clusters[p].append(t)
    if not clusters:
        return 0.0
    correct = 0
    total = sum(len(v) for v in clusters.values())
    for cluster_labels in clusters.values():
        counts = collections.Counter(cluster_labels)
        correct += counts.most_common(1)[0][1]
    return correct / total if total > 0 else 0.0


def main():
    encoder, p1, p2, p3 = load_encoder_and_data()

    dataset = p1
    texts = [x[0] for x in dataset]
    labels_true = [x[1] for x in dataset]

    epsilon = 0.1
    mu = 2
    beta = 0.75
    decay = 0.005
    denstream = cluster.DenStream(
        epsilon=epsilon, mu=mu, beta=beta, decaying_factor=decay, n_samples_init=1
    )

    ipca = IncrementalPCA(n_components=16)
    ipca_fitted = False

    batch_size = 150
    metrics_log = []

    window_embeddings = []
    window_labels = []
    window_preds = []

    total = len(texts)

    for i in range(0, total, batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_labels = labels_true[i : i + batch_size]

        emb = encoder.encode(batch_texts, show_progress_bar=False)

        if not ipca_fitted:
            ipca.fit(emb)
            ipca_fitted = True

        emb_pca = ipca.transform(emb)
        emb_norm = normalize(emb_pca, norm="l2")

        for j, vec in enumerate(emb_norm):
            v_dict = {k: float(v) for k, v in enumerate(vec)}
            denstream.learn_one(v_dict)
            pred = denstream.predict_one(v_dict)

            window_embeddings.append(vec)
            window_labels.append(batch_labels[j])
            window_preds.append(pred)

        if len(window_embeddings) > 300:
            window_embeddings = window_embeddings[-300:]
            window_labels = window_labels[-300:]
            window_preds = window_preds[-300:]

        n_p = len(denstream.p_micro_clusters)
        n_macro = len(denstream.clusters)

        radii = []
        centers = []
        t = denstream.timestamp
        for pmc in denstream.p_micro_clusters.values():
            radii.append(pmc.calc_radius(t))
            c = pmc.calc_center(t)
            centers.append([c.get(dim, 0.0) for dim in range(16)])

        mean_r = np.mean(radii) if radii else 0.0

        min_d = 0.0
        mean_d = 0.0
        max_d = 0.0

        if len(centers) > 1:
            dists = []
            for k1 in range(len(centers)):
                for k2 in range(k1 + 1, len(centers)):
                    d = np.linalg.norm(np.array(centers[k1]) - np.array(centers[k2]))
                    dists.append(d)
            min_d = np.min(dists)
            mean_d = np.mean(dists)
            max_d = np.max(dists)

        ari = adjusted_rand_score(window_labels, window_preds)
        nmi = normalized_mutual_info_score(window_labels, window_preds)
        pur = calculate_purity(window_labels, window_preds)

        sil = 0.0
        if len(set(window_preds)) > 1:
            sil = silhouette_score(window_embeddings, window_preds, metric="euclidean")

        metrics_log.append(
            {
                "batch": i // batch_size + 1,
                "n_p_micro": n_p,
                "n_macro": n_macro,
                "mean_radius": mean_r,
                "min_d": min_d,
                "mean_d": mean_d,
                "max_d": max_d,
                "2_eps": 2 * epsilon,
                "ari": ari,
                "nmi": nmi,
                "purity": pur,
                "silhouette": sil,
            }
        )

    df = pd.DataFrame(metrics_log)
    out_path = (
        "experiments/denstream_offline_failure/results/denstream_failure_metrics.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"Metrics saved to {out_path}")


def generate_chart():
    df = pd.read_csv(
        "experiments/denstream_offline_failure/results/denstream_failure_metrics.csv"
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(df["batch"], df["n_p_micro"], label="p-micro-clusters", marker="o")
    axes[0].plot(
        df["batch"], df["n_macro"], label="macro-clusters (River native)", marker="x"
    )
    axes[0].set_ylabel("Liczba klastrów")
    axes[0].set_title("Liczba p-mikroklastrów a liczba makroklastrów")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        df["batch"],
        df["min_d"],
        label="Minimalny dystans euklidesowy",
        color="red",
        linestyle="--",
    )
    axes[1].plot(
        df["batch"], df["mean_d"], label="Średni dystans euklidesowy", color="orange"
    )
    axes[1].axhline(
        y=df["2_eps"].iloc[0],
        color="black",
        linestyle=":",
        label="Próg scalania DBSCAN (2*eps)",
    )

    ax2 = axes[1].twinx()
    ax2.set_ylabel("ARI / NMI")

    axes[1].set_xlabel("Batch (nr partii)")
    axes[1].set_ylabel("Dystans euklidesowy")
    axes[1].set_title("Dystanse vs próg scalania")

    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right")
    axes[1].grid(True)

    plt.tight_layout()
    out_path = (
        "experiments/denstream_offline_failure/results/denstream_failure_proof.png"
    )
    plt.savefig(out_path)
    print(f"Wykres został wygenerowany w '{out_path}'")


if __name__ == "__main__":
    main()
    generate_chart()
