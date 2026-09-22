import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from src.domain.evolution import NSGAIIOptimizer, Individual
from src.core.config import config
from src.domain.preprocessing import TextPreprocessor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset_embeddings(categories: List[str], max_samples: int = 2000) -> np.ndarray:
    raw_data = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
    preprocessor = TextPreprocessor()
    texts = []
    for text in raw_data.data:
        cleaned = preprocessor.clean(text)
        if len(cleaned.split()) >= 10:
            texts.append(cleaned)
    random.seed(42)
    random.shuffle(texts)
    texts = texts[:max_samples]

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Encoding {len(texts)} samples with SBERT on {device}...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = encoder.encode(texts, batch_size=128, device=device, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings


def run_pareto_analysis():
    set_seed(42)
    categories = [
        "sci.space",
        "sci.med",
        "rec.autos",
        "rec.sport.baseball",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
    ]
    raw_embeddings = load_dataset_embeddings(categories=categories, max_samples=2000)

    dims_to_test = [8, 16, 32, 64, 128]
    all_fronts_records = []
    knee_summaries = []

    logger.info("=== Running Multi-Objective Pareto Analysis for Thesis 3 ===")

    for d in dims_to_test:
        logger.info(f"Running NSGA-II Pareto extraction for IPCA dimension d={d}...")
        ipca = IncrementalPCA(n_components=d)
        warmup_n = max(d + 50, 200)
        ipca.partial_fit(raw_embeddings[:warmup_n])
        reduced_data = normalize(ipca.transform(raw_embeddings))

        # Run NSGA-II optimizer with larger population & generations for smooth front
        optimizer = NSGAIIOptimizer(population_size=config.evolution.population_size, generations=config.evolution.generations, crossover_rate=config.evolution.crossover_rate, mutation_rate=config.evolution.mutation_rate, param_bounds=config.evolution.param_bounds, fixed_mu=config.denstream.mu, n_samples_init=config.denstream.n_samples_init)
        best_knee, pareto_front, history = optimizer.evolve(data_buffer=reduced_data[:500])

        for ind in pareto_front:
            all_fronts_records.append({
                "pca_dim": d,
                "epsilon": ind.params["epsilon"],
                "mu": ind.params["mu"],
                "decay": ind.params["decaying_factor"],
                "offline_eps": ind.params["offline_eps"],
                "quality_score": ind.quality_score,
                "complexity_score": ind.complexity_score,
                "is_knee_point": (ind.params == best_knee.params),
            })

        knee_summaries.append({
            "pca_dim": d,
            "knee_epsilon": best_knee.params["epsilon"],
            "knee_mu": best_knee.params["mu"],
            "knee_decay": best_knee.params["decaying_factor"],
            "knee_offline_eps": best_knee.params["offline_eps"],
            "knee_quality": best_knee.quality_score,
            "knee_complexity": best_knee.complexity_score,
            "pareto_front_size": len(pareto_front),
        })

    df_fronts = pd.DataFrame(all_fronts_records)
    df_fronts.to_csv("experiments/thesis_3_pareto_fronts.csv", index=False)

    df_summary = pd.DataFrame(knee_summaries)
    df_summary.to_csv("experiments/thesis_3_summary.csv", index=False)

    logger.success("Saved Pareto fronts to experiments/thesis_3_pareto_fronts.csv")

    # Generate Publication Plots
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "font.family": "serif",
    })

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(left=0.07, right=0.92, top=0.94, bottom=0.08, wspace=0.38, hspace=0.36)

    colors = {8: "#8c564b", 16: "#9467bd", 32: "#d62728", 64: "#ff7f0e", 128: "#2ca02c"}

    # Plot (a): Pareto Fronts
    for d in dims_to_test:
        sub = df_fronts[df_fronts["pca_dim"] == d].sort_values("complexity_score")
        axs[0, 0].plot(sub["complexity_score"], sub["quality_score"], "-o", color=colors[d], label=f"Front Pareto (d={d})", alpha=0.75, markersize=5, lw=1.8)
        knee = sub[sub["is_knee_point"] == True]
        if len(knee) > 0:
            axs[0, 0].scatter(knee["complexity_score"], knee["quality_score"], color=colors[d], s=180, edgecolor="black", zorder=5, marker="*", label=f"Punkt kolana (d={d})")

    axs[0, 0].set_title("(a) Fronty Pareto: Jakość klastrowania vs. Złożoność", fontweight="bold")
    axs[0, 0].set_xlabel("Złożoność znormalizowana (Liczba mikroklastrów)")
    axs[0, 0].set_ylabel("Jakość klastrowania (Silhouette Score)")
    axs[0, 0].set_xlim(-0.02, 0.88)
    axs[0, 0].set_ylim(-0.55, 0.35)
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=8.5, ncol=2)

    # Plot (b): Knee Point Quality vs. Dimension d
    axs[0, 1].plot(df_summary["pca_dim"], df_summary["knee_quality"], color="#2980b9", marker="s", lw=2.5, markersize=9, label="Jakość w punkcie kolana")
    for _, row in df_summary.iterrows():
        axs[0, 1].annotate(f"{row['knee_quality']:.3f}", xy=(row["pca_dim"], row["knee_quality"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#1565c0")
    axs[0, 1].set_title("(b) Jakość separacji w punkcie kolana (Knee Point)", fontweight="bold")
    axs[0, 1].set_xlabel("Wymiar projekcji IPCA (d)")
    axs[0, 1].set_ylabel("Jakość w punkcie kolana (Silhouette)", color="#2980b9", fontweight="bold")
    axs[0, 1].set_ylim(-0.55, 0.35)
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend(loc="upper right", frameon=True, framealpha=0.95)

    # Plot (c): Knee Point Complexity vs. Dimension d
    axs[1, 0].plot(df_summary["pca_dim"], df_summary["knee_complexity"], color="#e74c3c", marker="^", lw=2.5, markersize=9, label="Złożoność w punkcie kolana")
    for _, row in df_summary.iterrows():
        axs[1, 0].annotate(f"{row['knee_complexity']:.3f}", xy=(row["pca_dim"], row["knee_complexity"]), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, fontweight="bold", color="#c62828")
    axs[1, 0].set_title("(c) Złożoność struktur mikroklastrów w punkcie kolana", fontweight="bold")
    axs[1, 0].set_xlabel("Wymiar projekcji IPCA (d)")
    axs[1, 0].set_ylabel("Złożoność w punkcie kolana (Complexity Score)", color="#e74c3c", fontweight="bold")
    axs[1, 0].set_ylim(0.0, 0.90)
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)
    axs[1, 0].legend(loc="upper left", frameon=True, framealpha=0.95)

    # Plot (d): Hyperparameter convergence
    x_pos = np.arange(len(dims_to_test))
    w = 0.35
    b_eps = axs[1, 1].bar(x_pos - w/2, df_summary["knee_epsilon"], width=w, label=r"Promień $\epsilon^*$", color="#3498db", alpha=0.85)
    b_dec = axs[1, 1].bar(x_pos + w/2, df_summary["knee_decay"] * 10, width=w, label=r"Wygaszanie $\lambda^* \times 10$", color="#9b59b6", alpha=0.85)

    for rect in b_eps:
        h = rect.get_height()
        axs[1, 1].annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1565c0")
    for rect in b_dec:
        h = rect.get_height()
        axs[1, 1].annotate(f"{h/10:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#6a1b9a")

    axs[1, 1].set_title(r"(d) Wartości optymalnych parametrów ($\epsilon^*, \lambda^*$) w punkcie kolana", fontweight="bold")
    axs[1, 1].set_xlabel("Wymiar projekcji IPCA (d)")
    axs[1, 1].set_ylabel("Wartość parametru")
    axs[1, 1].set_xticks(x_pos)
    axs[1, 1].set_xticklabels([f"d={d}" for d in dims_to_test])
    axs[1, 1].set_ylim(0.0, 0.80)
    axs[1, 1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axs[1, 1].legend(loc="upper right", frameon=True, framealpha=0.95)

    png_path = "charts/thesis_3_pareto_tradeoff.png"
    pdf_path = "charts/thesis_3_pareto_tradeoff.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    logger.success(f"Saved Thesis 3 figures to {png_path} and {pdf_path}")
    plt.close()

    # Generate LaTeX Summary Table
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Wyniki analizy wielokryterialnej dla Tezy 3: Charakterystyka punktow kolana (Knee Points) na frontach Pareto w zaleznosci od wymiaru projekcji IPCA.}",
        "\\label{tab:thesis_3_pareto_results}",
        "\\begin{tabular}{ccccccc}",
        "\\toprule",
        "\\textbf{Wymiar ($d$)} & \\textbf{Rozmiar frontu} & \\textbf{Jakosc (Knee)} & \\textbf{Zlozonosc (Knee)} & $\\epsilon$ (Knee) & $\\mu$ (Knee) & $\\lambda$ (Knee) \\\\",
        "\\midrule",
    ]
    for _, row in df_summary.iterrows():
        line = (
            f"{int(row['pca_dim'])} & {int(row['pareto_front_size'])} & "
            f"{row['knee_quality']:.4f} & {row['knee_complexity']:.4f} & "
            f"{row['knee_epsilon']:.4f} & {int(row['knee_mu'])} & {row['knee_decay']:.4f} \\\\"
        )
        lines.append(line)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    latex_table = "\n".join(lines)
    with open("experiments/thesis_3_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    logger.success("Saved LaTeX summary table to experiments/thesis_3_table.tex")
    print("\n" + latex_table)


if __name__ == "__main__":
    run_pareto_analysis()
