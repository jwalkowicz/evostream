import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from river import cluster
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from src.core.logger import logger
from src.domain.clustering import StreamClusterer
from src.domain.preprocessing import TextPreprocessor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_benchmark_data(categories: List[str], max_samples: int = 3000) -> Tuple[List[str], List[str]]:
    logger.info(f"Fetching 20 Newsgroups data for categories: {categories}")
    raw_data = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    preprocessor = TextPreprocessor()
    texts, labels = [], []
    for text, target_idx in zip(raw_data.data, raw_data.target):
        cleaned = preprocessor.clean(text)
        if len(cleaned.split()) >= 10:  # Minimum 10 tokens
            label = raw_data.target_names[target_idx]
            texts.append(cleaned)
            labels.append(label)

    # Deterministic shuffle
    combined = list(zip(texts, labels))
    random.seed(42)
    random.shuffle(combined)
    if max_samples and len(combined) > max_samples:
        combined = combined[:max_samples]

    shuffled_texts = [c[0] for c in combined]
    shuffled_labels = [c[1] for c in combined]
    logger.info(f"Loaded {len(shuffled_texts)} clean benchmark text samples.")
    return shuffled_texts, shuffled_labels


def run_streaming_simulation(
    name: str,
    raw_embeddings: np.ndarray,
    labels: List[str],
    pca_dim: int | None,
    sbert_avg_latency_ms: float,
    batch_size: int = 64,
) -> pd.DataFrame:
    logger.info(f"=== Running Experiment: {name} (pca_dim={pca_dim}) ===")
    set_seed(42)

    # Initialize online IPCA if dimensionality reduction is enabled
    pca = IncrementalPCA(n_components=pca_dim) if pca_dim is not None else None

    # Clustering hyperparameters
    denstream = cluster.DenStream(
        decaying_factor=0.01,
        epsilon=0.45,
        mu=2,
    )
    clusterer = StreamClusterer(
        model=denstream,
        offline_eps=0.55,
        offline_min_samples=1,
        window_size=500,
    )

    records = []
    n_samples = len(raw_embeddings)
    n_batches = int(np.ceil(n_samples / batch_size))
    process = psutil.Process()

    warmup_buffer = []
    is_pca_fitted = False

    for b_idx in range(n_batches):
        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, n_samples)
        batch_raw = raw_embeddings[start_i:end_i]
        batch_labels = labels[start_i:end_i]

        t0 = time.perf_counter()
        if pca is not None:
            if not is_pca_fitted:
                warmup_buffer.extend(batch_raw)
                if len(warmup_buffer) >= pca.n_components:
                    pca.partial_fit(np.array(warmup_buffer))
                    is_pca_fitted = True
                    batch_vectors = normalize(pca.transform(batch_raw))
                else:
                    batch_vectors = batch_raw[:, :pca.n_components]
            else:
                if len(batch_raw) >= pca.n_components:
                    pca.partial_fit(batch_raw)
                batch_vectors = normalize(pca.transform(batch_raw))
        else:
            batch_vectors = batch_raw

        t_transform = time.perf_counter() - t0

        t1 = time.perf_counter()
        clusterer.update(batch_vectors, labels=batch_labels)
        t_cluster = time.perf_counter() - t1

        transform_ms_per_doc = (t_transform * 1000.0) / len(batch_raw)
        cluster_ms_per_doc = (t_cluster * 1000.0) / len(batch_raw)
        total_latency_per_doc = sbert_avg_latency_ms + transform_ms_per_doc + cluster_ms_per_doc

        ram_mb = process.memory_info().rss / (1024 * 1024)
        metrics = clusterer.get_metrics()

        records.append({
            "variant": name,
            "pca_dim": pca_dim if pca_dim is not None else 384,
            "batch": b_idx + 1,
            "samples_seen": metrics["n_samples_seen"],
            "latency_ms_per_doc": total_latency_per_doc,
            "sbert_time_ms": sbert_avg_latency_ms,
            "transform_time_ms": transform_ms_per_doc,
            "cluster_time_ms": cluster_ms_per_doc,
            "pure_stream_latency_ms": transform_ms_per_doc + cluster_ms_per_doc,
            "ram_usage_mb": ram_mb,
            "n_micro_clusters": metrics["n_micro_clusters"],
            "n_macro_clusters": metrics["n_macro_clusters"],
            "micro_macro_ratio": metrics["micro_macro_ratio"],
            "purity": metrics["purity"],
            "silhouette": metrics["silhouette"],
            "davies_bouldin": metrics["davies_bouldin"],
            "ari": metrics["ari"],
            "nmi": metrics["nmi"],
        })

    df = pd.DataFrame(records)
    logger.success(
        f"[{name}] Done | "
        f"Pure Stream Latency: {df['pure_stream_latency_ms'].mean():.4f} ms/doc | "
        f"Total Latency: {df['latency_ms_per_doc'].mean():.2f} ms/doc | "
        f"Mean Purity: {df['purity'].dropna().mean()*100:.1f}% | "
        f"Peak RAM: {df['ram_usage_mb'].max():.1f} MB"
    )
    return df


def generate_thesis_plots(all_results: pd.DataFrame, summary_df: pd.DataFrame):
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "font.family": "serif",
        "figure.titlesize": 14,
    })

    fig, axs = plt.subplots(2, 2, figsize=(16, 11))
    fig.tight_layout(pad=4.5)

    variants = all_results["variant"].unique()
    colors = {
        "Full SBERT (384d)": "#1f77b4",
        "IPCA (d=128)": "#2ca02c",
        "IPCA (d=64)": "#ff7f0e",
        "IPCA (d=32)": "#d62728",
        "IPCA (d=16)": "#9467bd",
        "IPCA (d=8)": "#8c564b",
    }

    # Plot (a): Pure Stream Latency per Document
    for var in variants:
        sub = all_results[all_results["variant"] == var]
        smoothed = sub["pure_stream_latency_ms"].ewm(span=5).mean()
        axs[0, 0].plot(sub["samples_seen"], smoothed, label=var, color=colors.get(var, "gray"), lw=2.0)
    axs[0, 0].set_title("(a) Czas operacji strumieniowych (Transformacja + Klastrowanie)", fontweight="bold")
    axs[0, 0].set_xlabel("Liczba przetworzonych dokumentów")
    axs[0, 0].set_ylabel("Czas online [ms / dokument]")
    axs[0, 0].grid(True, linestyle="--", alpha=0.6)
    axs[0, 0].legend(loc="upper right", frameon=True)

    # Plot (b): RAM Memory Consumption
    for var in variants:
        sub = all_results[all_results["variant"] == var]
        axs[0, 1].plot(sub["samples_seen"], sub["ram_usage_mb"], label=var, color=colors.get(var, "gray"), lw=2.0)
    axs[0, 1].set_title("(b) Zużycie pamięci RAM w trakcie przetwarzania", fontweight="bold")
    axs[0, 1].set_xlabel("Liczba przetworzonych dokumentów")
    axs[0, 1].set_ylabel("Pamięć RAM [MB]")
    axs[0, 1].grid(True, linestyle="--", alpha=0.6)
    axs[0, 1].legend(loc="lower right", frameon=True)

    # Plot (c): Cluster Purity over Stream Time
    for var in variants:
        sub = all_results[all_results["variant"] == var].dropna(subset=["purity"])
        smoothed = sub["purity"].ewm(span=5).mean()
        axs[1, 0].plot(sub["samples_seen"], smoothed, label=var, color=colors.get(var, "gray"), lw=2.0)
    axs[1, 0].set_title("(c) Czystość klastrów tematycznych (Cluster Purity)", fontweight="bold")
    axs[1, 0].set_xlabel("Liczba przetworzonych dokumentów")
    axs[1, 0].set_ylabel("Czystość (Purity) [0 - 1.0]")
    axs[1, 0].set_ylim(0.2, 1.0)
    axs[1, 0].grid(True, linestyle="--", alpha=0.6)
    axs[1, 0].legend(loc="lower right", frameon=True)

    # Plot (d): Speedup vs. Purity Retention Trade-off
    dims = summary_df["pca_dim"].values
    speedup = summary_df["stream_speedup_factor"].values
    purity_ret = summary_df["purity_retention_pct"].values

    ax_d = axs[1, 1]
    ax_d2 = ax_d.twinx()

    l1 = ax_d.bar([str(d) for d in dims], speedup, width=0.4, color="#3498db", alpha=0.85, label="Przyspieszenie fazy online (Speedup X)")
    l2 = ax_d2.plot([str(d) for d in dims], purity_ret, color="#e74c3c", marker="o", lw=2.5, markersize=8, label="Retencja czystości (% Purity)")

    ax_d.set_title("(d) Kompromis wymiarowości: Przyspieszenie vs. Retencja jakości", fontweight="bold")
    ax_d.set_xlabel("Wymiar projekcji (d)")
    ax_d.set_ylabel("Przyspieszenie fazy online [krotnie]", color="#2980b9")
    ax_d2.set_ylabel("Retencja czystości klastrów [%]", color="#c0392b")
    ax_d2.set_ylim(80, 110)
    ax_d.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Combined legend
    lines = [l1, l2[0]]
    labels = ["Przyspieszenie online [x]", "Retencja czystości [%]"]
    ax_d.legend(lines, labels, loc="upper left", frameon=True)

    # Save outputs
    png_path = "charts/thesis_1_ipca_benchmark.png"
    pdf_path = "charts/thesis_1_ipca_benchmark.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    logger.success(f"Saved Thesis 1 figures to {png_path} and {pdf_path}")
    plt.close()


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Wyniki ewaluacji empirycznej dla Tezy 1: Porownanie przetwarzania w pelnowymiarowej przestrzeni SBERT (384d) z inkrementalna projekcja IPCA.}",
        "\\label{tab:thesis_1_ipca_results}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Wariant} & \\textbf{Wymiar ($d$)} & \\textbf{Czas online [ms]} & \\textbf{Przysp. online} & \\textbf{Purity [\\%]} & \\textbf{Silhouette} & \\textbf{RAM [MB]} \\\\",
        "\\midrule",
    ]
    for _, row in summary_df.iterrows():
        purity_str = f"{row['mean_purity']*100:.1f} \\pm {row['std_purity']*100:.1f}" if pd.notna(row['mean_purity']) else "--"
        sil_str = f"{row['mean_silhouette']:.3f}" if pd.notna(row['mean_silhouette']) else "--"
        line = (
            f"{row['variant']} & {row['pca_dim']} & "
            f"{row['mean_pure_stream_ms']:.4f} $\\pm$ {row['std_pure_stream_ms']:.4f} & "
            f"{row['stream_speedup_factor']:.2f}\\times & "
            f"{purity_str} & "
            f"{sil_str} & "
            f"{row['peak_ram_mb']:.1f} \\\\"
        )
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    latex = "\n".join(lines)
    with open("experiments/thesis_1_table.tex", "w", encoding="utf-8") as f:
        f.write(latex)
    logger.success("Saved LaTeX summary table to experiments/thesis_1_table.tex")
    return latex


def main():
    logger.info("Initializing Thesis 1 Experiment Suite...")
    categories = [
        "sci.space",
        "sci.med",
        "rec.autos",
        "rec.sport.baseball",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
    ]
    texts, labels = load_benchmark_data(categories=categories, max_samples=3000)

    # Use hardware acceleration if available
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SBERT model (all-MiniLM-L6-v2) on device: {device}...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    logger.info("Pre-computing SBERT embeddings for the benchmark stream...")
    t0 = time.perf_counter()
    raw_embeddings = encoder.encode(
        texts,
        batch_size=128,
        device=device,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    sbert_total_sec = time.perf_counter() - t0
    sbert_avg_latency_ms = (sbert_total_sec * 1000.0) / len(texts)
    logger.success(f"SBERT Encoding completed: {len(raw_embeddings)} vectors | Avg SBERT Latency: {sbert_avg_latency_ms:.2f} ms/doc")

    configs = [
        ("Full SBERT (384d)", None),
        ("IPCA (d=128)", 128),
        ("IPCA (d=64)", 64),
        ("IPCA (d=32)", 32),
        ("IPCA (d=16)", 16),
        ("IPCA (d=8)", 8),
    ]

    all_results = []
    for name, dim in configs:
        df_res = run_streaming_simulation(
            name=name,
            raw_embeddings=raw_embeddings,
            labels=labels,
            pca_dim=dim,
            sbert_avg_latency_ms=sbert_avg_latency_ms,
            batch_size=64,
        )
        all_results.append(df_res)

    full_df = pd.concat(all_results, ignore_index=True)
    full_df.to_csv("experiments/thesis_1_raw_timeseries.csv", index=False)

    # Compute baseline for speedup
    baseline_stream_latency = full_df[full_df["variant"] == "Full SBERT (384d)"]["pure_stream_latency_ms"].mean()
    baseline_purity = full_df[full_df["variant"] == "Full SBERT (384d)"]["purity"].dropna().mean()

    summary_rows = []
    for name, dim in configs:
        sub = full_df[full_df["variant"] == name]
        mean_stream_lat = sub["pure_stream_latency_ms"].mean()
        std_stream_lat = sub["pure_stream_latency_ms"].std()
        mean_total_lat = sub["latency_ms_per_doc"].mean()
        mean_pur = sub["purity"].dropna().mean()
        std_pur = sub["purity"].dropna().std()
        mean_sil = sub["silhouette"].dropna().mean()
        peak_ram = sub["ram_usage_mb"].max()

        speedup = baseline_stream_latency / mean_stream_lat if mean_stream_lat > 0 else 1.0
        purity_ret = (mean_pur / baseline_purity * 100.0) if baseline_purity > 0 else 100.0

        summary_rows.append({
            "variant": name,
            "pca_dim": dim if dim is not None else 384,
            "mean_pure_stream_ms": mean_stream_lat,
            "std_pure_stream_ms": std_stream_lat,
            "stream_speedup_factor": speedup,
            "mean_total_latency_ms": mean_total_lat,
            "mean_purity": mean_pur,
            "std_purity": std_pur,
            "purity_retention_pct": purity_ret,
            "mean_silhouette": mean_sil,
            "peak_ram_mb": peak_ram,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("experiments/thesis_1_summary.csv", index=False)

    print("\n=================== THESIS 1 BENCHMARK SUMMARY ===================")
    print(summary_df.to_string(index=False))
    print("===================================================================\n")

    generate_thesis_plots(all_results=full_df, summary_df=summary_df)
    latex_table = generate_latex_table(summary_df)
    print(latex_table)


if __name__ == "__main__":
    main()

