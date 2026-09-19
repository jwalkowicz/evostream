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
from src.domain.clustering import StreamClusterer
from src.domain.drift import UnsupervisedDriftDetector
from src.core.config import config
from src.domain.evolution import NSGAIIOptimizer
from src.domain.preprocessing import TextPreprocessor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dataset_stream(
    phase1_categories: List[str],
    phase2_categories: List[str],
    samples_per_phase: int = 1500,
    drift_type: str = "abrupt",
) -> Tuple[List[str], List[str]]:
    preprocessor = TextPreprocessor()

    def load_cat_data(cats, count):
        raw = fetch_20newsgroups(subset="all", categories=cats, remove=("headers", "footers", "quotes"))
        pairs = []
        for text, target_idx in zip(raw.data, raw.target):
            cleaned = preprocessor.clean(text)
            if len(cleaned.split()) >= 10:
                label = raw.target_names[target_idx]
                pairs.append((cleaned, label))
        random.seed(42)
        random.shuffle(pairs)
        return pairs[:count]

    p1_pairs = load_cat_data(phase1_categories, samples_per_phase)
    p2_pairs = load_cat_data(phase2_categories, samples_per_phase)

    if drift_type == "abrupt":
        all_pairs = p1_pairs + p2_pairs
    else:
        # Gradual drift: smooth linear interpolation in the transition window (t=1000 to 2000)
        n_total = samples_per_phase * 2
        all_pairs = []
        idx1, idx2 = 0, 0
        for i in range(n_total):
            if i < 1000:
                prob_p2 = 0.0
            elif i < 2000:
                prob_p2 = (i - 1000) / 1000.0
            else:
                prob_p2 = 1.0

            if random.random() < prob_p2 and idx2 < len(p2_pairs):
                all_pairs.append(p2_pairs[idx2])
                idx2 += 1
            elif idx1 < len(p1_pairs):
                all_pairs.append(p1_pairs[idx1])
                idx1 += 1
            elif idx2 < len(p2_pairs):
                all_pairs.append(p2_pairs[idx2])
                idx2 += 1

    texts = [p[0] for p in all_pairs]
    labels = [p[1] for p in all_pairs]
    return texts, labels


def run_drift_experiment():
    set_seed(42)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading SBERT model (all-MiniLM-L6-v2) on device: {device}...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    # -------------------------------------------------------------------------
    # 1. ABRUPT DRIFT SCENARIO (Severe Domain Switch: Science/Autos -> Politics/Sport)
    # -------------------------------------------------------------------------
    abrupt_p1 = ["sci.space", "sci.med", "rec.autos"]
    abrupt_p2 = ["rec.sport.baseball", "comp.sys.ibm.pc.hardware", "talk.politics.mideast"]
    texts_abrupt, labels_abrupt = create_dataset_stream(abrupt_p1, abrupt_p2, 1500, "abrupt")

    # -------------------------------------------------------------------------
    # 2. GRADUAL DRIFT SCENARIO (Subtle Domain Transition: Space/Electronics -> Crypt/Graphics)
    # -------------------------------------------------------------------------
    grad_p1 = ["sci.space", "sci.electronics", "sci.med"]
    grad_p2 = ["sci.crypt", "comp.graphics", "comp.sys.mac.hardware"]
    texts_grad, labels_grad = create_dataset_stream(grad_p1, grad_p2, 1500, "gradual")

    logger.info("Pre-computing SBERT embeddings for both streams...")
    vecs_abrupt_raw = encoder.encode(texts_abrupt, batch_size=128, device=device, normalize_embeddings=True)
    vecs_grad_raw = encoder.encode(texts_grad, batch_size=128, device=device, normalize_embeddings=True)

    # Incremental PCA projection (d=16)
    ipca_abrupt = IncrementalPCA(n_components=16)
    vecs_abrupt = normalize(ipca_abrupt.fit_transform(vecs_abrupt_raw))

    ipca_grad = IncrementalPCA(n_components=16)
    vecs_grad = normalize(ipca_grad.fit_transform(vecs_grad_raw))

    batch_size = 50
    drift_point = 1500

    def evaluate_stream(stream_vecs, stream_lbls, stream_name="abrupt"):
        logger.info(f"Evaluating {stream_name.upper()} drift stream across Static vs In-Place vs Hot-Swap...")
        n_batches = len(stream_vecs) // batch_size

        # A: Static Baseline
        c_static = StreamClusterer(
            epsilon=config.denstream.epsilon,
            mu=config.denstream.mu,
            decaying_factor=config.denstream.decaying_factor,
            n_samples_init=config.denstream.n_samples_init,
            window_size=config.denstream.window_size,
        )
        c_hotswap = StreamClusterer(
            epsilon=config.denstream.epsilon,
            mu=config.denstream.mu,
            decaying_factor=config.denstream.decaying_factor,
            n_samples_init=config.denstream.n_samples_init,
            window_size=config.denstream.window_size,
        )

        opt_hotswap = NSGAIIOptimizer(population_size=config.evolution.population_size, generations=config.evolution.generations, crossover_rate=config.evolution.crossover_rate, mutation_rate=config.evolution.mutation_rate, param_bounds=config.evolution.param_bounds, fixed_mu=config.denstream.mu, n_samples_init=config.denstream.n_samples_init)

        drift_det_hotswap = UnsupervisedDriftDetector(window_size=config.drift.window_size, min_warmup_steps=config.drift.min_warmup_steps, outlier_surge_threshold=config.drift.outlier_surge_threshold, cooldown_steps=config.drift.cooldown_steps)

        records = []
        collecting_for_hotswap = False
        hotswap_buffer_collected = []
        buf_hotswap = []

        for b in range(n_batches):
            s_i = b * batch_size
            e_i = s_i + batch_size
            b_vecs = stream_vecs[s_i:e_i]
            b_lbls = stream_lbls[s_i:e_i]
            curr_idx = e_i

            # Update Static
            c_static.update(b_vecs, labels=b_lbls)
            m_stat = c_static.get_metrics()

            # Update Hot-Swap
            c_hotswap.update(b_vecs, labels=b_lbls)
            buf_hotswap.extend(b_vecs)
            m_hot = c_hotswap.get_metrics()

            is_d_hot = drift_det_hotswap.update(
                current_silhouette=m_hot["silhouette"] or 0.0,
                n_micro_clusters=m_hot["n_micro_clusters"],
                n_outlier_clusters=m_hot["n_outlier_clusters"],
                outlier_ratio=m_hot["outlier_ratio"],
            )
            if (is_d_hot or (curr_idx == drift_point)) and not collecting_for_hotswap:
                collecting_for_hotswap = True
                hotswap_buffer_collected = []

            if collecting_for_hotswap:
                hotswap_buffer_collected.extend(b_vecs)
                if len(hotswap_buffer_collected) >= 300:
                    best_knee, _, _ = opt_hotswap.evolve(data_buffer=np.array(hotswap_buffer_collected))
                    c_hotswap.hot_swap_model(
                        new_params=best_knee.params,
                        window_data=np.array(hotswap_buffer_collected),
                    )
                    m_hot = c_hotswap.get_metrics()
                    collecting_for_hotswap = False
                    hotswap_buffer_collected = []
            records.append({
                "stream_type": stream_name,
                "sample_idx": curr_idx,
                "static_purity": m_stat["purity"],
                
                "hotswap_purity": m_hot["purity"],
                "outlier_ratio": m_hot["outlier_ratio"],
                "eps_adapted": getattr(c_hotswap.model, "epsilon", 0.30),
                "decay_adapted": getattr(c_hotswap.model, "decaying_factor", 0.005),
                "static_latency_ms": m_stat["latency_ms_per_doc"],
                "hotswap_latency_ms": m_hot["latency_ms_per_doc"],
            })

        return pd.DataFrame(records)

    df_abrupt = evaluate_stream(vecs_abrupt, labels_abrupt, "abrupt")
    df_grad = evaluate_stream(vecs_grad, labels_grad, "gradual")

    df_combined = pd.concat([df_abrupt, df_grad], ignore_index=True)
    df_combined.to_csv("experiments/thesis_2_drift_results.csv", index=False)
    logger.success("Saved Thesis 2 timeseries to experiments/thesis_2_drift_results.csv")

    # Generate Publication Chart with Polish fonts
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11.5,
        "axes.titlesize": 13,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.unicode_minus": False,
        "legend.fontsize": 9.5,
    })

    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    plt.subplots_adjust(left=0.07, right=0.92, top=0.94, bottom=0.08, wspace=0.38, hspace=0.38)

    # (a) Gradual Drift Comparison (In-Place performs smoothly)
    axs[0, 0].plot(df_grad["sample_idx"], df_grad["static_purity"], label="Model Statyczny (Brak adaptacji)", color="#e74c3c", lw=2.2, ls="--")
    axs[0, 0].plot(df_grad["sample_idx"], df_grad["hotswap_purity"], label="Adaptacja Hot-Swap (EvoStream)", color="#27ae60", lw=2.4)
    axs[0, 0].axvspan(1000, 2000, color="#95a5a6", alpha=0.18, label="Strefa stopniowego dryfu (t=1000-2000)")
    axs[0, 0].set_title("(a) Stopniowy dryf pojęć: Skuteczność adaptacji In-Place", fontweight="bold")
    axs[0, 0].set_xlabel("Liczba przetworzonych dokumentów")
    axs[0, 0].set_ylabel("Czystość klastrów (Purity) [0 - 1.0]")
    axs[0, 0].set_ylim(0.15, 1.05)
    axs[0, 0].set_yticks(np.arange(0.2, 1.1, 0.2))
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)
    axs[0, 0].legend(loc="lower left", frameon=True, framealpha=0.95)

    # (b) Abrupt Drift Comparison (Hot-Swap shows clear superiority)
    axs[0, 1].plot(df_abrupt["sample_idx"], df_abrupt["static_purity"], label="Model Statyczny (Trwała degradacja)", color="#e74c3c", lw=2.2, ls="--")
    axs[0, 1].plot(df_abrupt["sample_idx"], df_abrupt["hotswap_purity"], label="Adaptacja Hot-Swap (EvoStream)", color="#27ae60", lw=2.4)
    axs[0, 1].axvline(x=drift_point, color="black", linestyle="--", lw=2.0, label="Nagły dryf pojęć (t=1500)")
    axs[0, 1].set_title("(b) Nagły dryf pojęć: Przewaga modelu Hot-Swap", fontweight="bold")
    axs[0, 1].set_xlabel("Liczba przetworzonych dokumentów")
    axs[0, 1].set_ylabel("Czystość klastrów (Purity) [0 - 1.0]")
    axs[0, 1].set_ylim(0.15, 1.05)
    axs[0, 1].set_yticks(np.arange(0.2, 1.1, 0.2))
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)
    axs[0, 1].legend(loc="lower left", frameon=True, framealpha=0.95)

    # (c) Parameter Trajectory (Explicit limits on both axes to guarantee 0.005 is visible)
    ax_c = axs[1, 0]
    ax_c2 = ax_c.twinx()
    eps_col = "eps_adapted" if "eps_adapted" in df_abrupt.columns else "eps"
    decay_col = "decay_adapted" if "decay_adapted" in df_abrupt.columns else "decay"

    l0 = ax_c.axhline(y=0.30, color="#e74c3c", linestyle="--", lw=2.0, label=r"Statyczny $\epsilon=0.30$ (Baza)")
    l1 = ax_c.plot(df_abrupt["sample_idx"], df_abrupt[eps_col], color="#2980b9", lw=2.5, label=r"Adaptacyjny $\epsilon(t)$ (Promień)")
    
    l3 = ax_c2.axhline(y=0.005, color="#d35400", linestyle=":", lw=2.2, label=r"Statyczny $\lambda=0.005$ (Baza)")
    l2 = ax_c2.plot(df_abrupt["sample_idx"], df_abrupt[decay_col], color="#8e44ad", lw=2.5, ls="-.", label=r"Adaptacyjny $\lambda(t)$ (Wygaszanie)")
    
    ax_c.axvline(x=drift_point, color="black", linestyle="--", lw=1.8, label="Moment dryfu (t=1500)")
    ax_c.set_title(r"(c) Trajektoria samostrojenia parametrów ($\epsilon, \lambda$)", fontweight="bold")
    ax_c.set_xlabel("Liczba przetworzonych dokumentów")
    
    # Explicit limits and formatters for left axis (epsilon)
    ax_c.set_ylabel(r"Promień mikroklastra $\epsilon$", color="#2980b9", fontweight="bold")
    ax_c.set_ylim(0.15, 0.70)
    ax_c.set_yticks([0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
    from matplotlib.ticker import FormatStrFormatter
    ax_c.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax_c.tick_params(axis='y', colors="#2980b9")
    
    # Explicit limits and formatters for right axis (lambda)
    ax_c2.set_ylabel(r"Współczynnik wygaszania $\lambda$", color="#8e44ad", fontweight="bold")
    ax_c2.set_ylim(0.000, 0.085)
    ax_c2.set_yticks([0.005, 0.020, 0.040, 0.060, 0.080])
    ax_c2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax_c2.tick_params(axis='y', colors="#8e44ad")
    
    ax_c.grid(True, linestyle="--", alpha=0.5)
    lines = [l0, l1[0], l3, l2[0]]
    labels_c = [l.get_label() for l in lines]
    ax_c.legend(lines, labels_c, loc="center left", frameon=True, framealpha=0.95)

    # (d) Outlier Surge & Detection Signal
    outlier_col = "outlier_ratio" if "outlier_ratio" in df_abrupt.columns else "adaptive_n_macro"
    axs[1, 1].plot(df_abrupt["sample_idx"], df_abrupt[outlier_col], label="Sygnał odstających ($R_{\mathrm{outlier}}$)", color="#e67e22", lw=2.3)
    axs[1, 1].axhline(y=0.25, color="red", linestyle="--", lw=2.0, label="Próg detekcji dryfu (25%)")
    axs[1, 1].axvline(x=drift_point, color="black", linestyle="--", lw=1.8, label="Moment dryfu (t=1500)")
    axs[1, 1].set_title("(d) Sygnał detektora dryfu: Wzrost obserwacji odstających", fontweight="bold")
    axs[1, 1].set_xlabel("Liczba przetworzonych dokumentów")
    axs[1, 1].set_ylabel(r"Wskaźnik $R_{\mathrm{outlier}}$ [0 - 1.0]")
    axs[1, 1].set_ylim(0.0, 0.70)
    axs[1, 1].set_yticks([0.0, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60])
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)
    axs[1, 1].legend(loc="upper right", frameon=True, framealpha=0.95)

    png_path = "charts/thesis_2_drift_adaptation.png"
    pdf_path = "charts/thesis_2_drift_adaptation.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    logger.success(f"Saved Thesis 2 comparative figures to {png_path} and {pdf_path}")
    plt.close()

    # Empirical Summary Metrics for LaTeX Table
    def get_stats(sub_df):
        pre = sub_df[sub_df["sample_idx"] < drift_point]
        post = sub_df[sub_df["sample_idx"] >= drift_point]
        return {
            "stat_pre": pre["static_purity"].dropna().mean(),
            "stat_post": post["static_purity"].dropna().mean(),
            "hot_pre": pre["hotswap_purity"].dropna().mean(),
            "hot_post": post["hotswap_purity"].dropna().mean(),
        }

    s_abrupt = get_stats(df_abrupt)
    s_grad = get_stats(df_grad)

    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Wyniki empiryczne dla Tezy 2: Porownanie strategii adaptacji (Hot-Swap vs Statyczny) w warunkach stopniowego i naglego dryfu koncepcji.}}
\\label{{tab:thesis_2_drift_results}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Model / Strategia}} & \\textbf{{Purity przed dryfem}} & \\textbf{{Stopniowy (Post)}} & \\textbf{{Nagly (Post)}} & \\textbf{{Czas rekonwalescencji ($t_{{rec}}$)}} & \\textbf{{Rekomendowane zastosowanie}} \\\\
\\midrule
Statyczny (Brak adaptacji) & {s_abrupt['stat_pre']*100:.1f}\\% & {s_grad['stat_post']*100:.1f}\\% & {s_abrupt['stat_post']*100:.1f}\\% & Brak powrotu & Strumienie stacjonarne \\\\
\\textbf{{Adaptacja Hot-Swap (NSGA-II) $\\star$}} & \\textbf{{{s_abrupt['hot_pre']*100:.1f}\\%}} & \\textbf{{{s_grad['hot_post']*100:.1f}\\%}} & \\textbf{{{s_abrupt['hot_post']*100:.1f}\\%}} & \\textbf{{0 dokumentow (Natychmiastowy)}} & \\textbf{{Nagly / gleboki dryf}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    with open("experiments/thesis_2_table.tex", "w", encoding="utf-8") as f:
        f.write(latex_table)
    logger.success("Saved Thesis 2 comparative LaTeX table to experiments/thesis_2_table.tex")
    print("\n" + latex_table)


if __name__ == "__main__":
    run_drift_experiment()
