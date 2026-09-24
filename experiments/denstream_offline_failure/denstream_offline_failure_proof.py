"""
Why DenStream's native offline phase (DBSCAN over p-micro-cluster centres,
merging centres closer than 2 * epsilon) fails in this system (thesis
section 2.3.2). Records, after each batch of the thesis stream, the number of
p-micro-clusters, the number of macro-clusters found by river and the
distances between centres compared with 2 * epsilon.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from river import cluster, stream
from scipy.spatial.distance import pdist
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from experiments.plot_style import use_polish_number_format
from experiments.theses.thesis_1.exp_thesis_1_ipca import load_phase1_stream
from experiments.theses.thesis_2.exp_thesis_2_drift import BATCH_SIZE, INITIAL_WARMUP_SIZE
from src.core.config import config
from src.domain.clustering import micro_cluster_centers

RESULTS_DIR = "experiments/denstream_offline_failure/results"
PCA_DIM = 16


def main():
    _, embeddings, _ = load_phase1_stream()
    ipca = IncrementalPCA(n_components=PCA_DIM).fit(embeddings[:INITIAL_WARMUP_SIZE])
    model = cluster.DenStream(
        epsilon=config.denstream.epsilon,
        mu=config.denstream.mu,
        beta=config.denstream.beta,
        decaying_factor=config.denstream.decaying_factor,
        n_samples_init=config.denstream.n_samples_init,
    )
    for x, _ in stream.iter_array(normalize(ipca.transform(embeddings[:INITIAL_WARMUP_SIZE]))):
        model.learn_one(x)

    records = []
    for start in range(INITIAL_WARMUP_SIZE, len(embeddings), BATCH_SIZE):
        batch = normalize(ipca.transform(embeddings[start : start + BATCH_SIZE]))
        for x, _ in stream.iter_array(batch):
            model.learn_one(x)
        # predict_one runs river's own offline phase and fills model.clusters.
        model.predict_one(dict(enumerate(batch[-1])))

        _, centers = micro_cluster_centers(model)
        distances = pdist(centers) if len(centers) > 1 else np.array([np.nan])
        records.append(
            {
                "samples_seen": start + len(batch),
                "n_p_micro": len(centers),
                "n_macro_native": len(model.clusters),
                "min_d": float(np.min(distances)),
                "mean_d": float(np.mean(distances)),
                "max_d": float(np.max(distances)),
                "2_eps": 2 * config.denstream.epsilon,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(f"{RESULTS_DIR}/denstream_failure_metrics.csv", index=False)
    print(
        f"p-micro-clusters: {df['n_p_micro'].min()}-{df['n_p_micro'].max()} | "
        f"native macro-clusters: {df['n_macro_native'].min()}-{df['n_macro_native'].max()} | "
        f"batches with macro == micro: {(df['n_macro_native'] == df['n_p_micro']).sum()}/{len(df)}\n"
        f"min distance: {df['min_d'].min():.3f}-{df['min_d'].max():.3f} | "
        f"mean distance: {df['mean_d'].min():.3f}-{df['mean_d'].max():.3f} | "
        f"2*eps = {2 * config.denstream.epsilon:.2f}"
    )


def generate_chart():
    """Distances between p-micro-cluster centres against the offline-phase
    merge threshold 2*epsilon (thesis section 2.3.2, Figure 3)."""
    use_polish_number_format()
    df = pd.read_csv(f"{RESULTS_DIR}/denstream_failure_metrics.csv")

    plt.rcParams.update({"font.size": 11, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["samples_seen"], df["mean_d"], lw=2, color="#2980b9",
            label="Średnia odległość między środkami p-mikroklastrów")
    ax.plot(df["samples_seen"], df["min_d"], lw=2, color="#e67e22",
            label="Najmniejsza odległość między środkami p-mikroklastrów")
    ax.axhline(df["2_eps"].iloc[0], color="#c0392b", linestyle="--", lw=2,
               label="Próg łączenia mikroklastrów w fazie offline (2ε)")

    ax.set_xlabel("Liczba przetworzonych dokumentów")
    ax.set_ylabel("Odległość euklidesowa")
    ax.set_ylim(0, 1.1)
    ax.set_xlim(df["samples_seen"].min(), df["samples_seen"].max())
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=False)

    out_path = f"{RESULTS_DIR}/denstream_failure_proof.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
    generate_chart()
