import random
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path when launched via Streamlit CLI
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from river import stream
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from src.core.config import config
from src.domain.clustering import StreamClusterer
from src.domain.drift import UnsupervisedDriftDetector
from src.domain.evolution import NSGAIIOptimizer
from src.domain.preprocessing import TextPreprocessor
from src.model.schemas import CLUSTERING_RESULTS_SCHEMA, MODEL_PARAMETERS_SCHEMA

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="evostream",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 26px;
        font-weight: 700;
        color: #1e3d59;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 15px;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3498db;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Cached Dataset & Embedding Loader
# ---------------------------------------------------------
@st.cache_resource
def load_encoder_and_data():
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    encoder = SentenceTransformer(
        config.ml.embedding_model if config.ml else "all-MiniLM-L6-v2", device=device
    )
    preprocessor = TextPreprocessor()

    p1_cats = config.dataset.categories_concept_a
    p2_cats = config.dataset.categories_concept_b
    p3_cats = config.dataset.categories_concept_c
    max_samples = getattr(config.dataset, "max_samples_per_concept", 1000)

    def load_domain(cats, phase_name):
        raw = fetch_20newsgroups(
            subset="all", categories=cats, remove=("headers", "footers", "quotes")
        )
        pairs = []
        for text, target_idx in zip(raw.data, raw.target):
            cleaned = preprocessor.clean(text)
            if len(cleaned.split()) >= 10:
                label = raw.target_names[target_idx]
                pairs.append((cleaned, label, phase_name))
        random.seed(config.seed)
        random.shuffle(pairs)
        return pairs[:max_samples]

    p1 = load_domain(p1_cats, "Koncept A: Nauka i motoryzacja (sci/rec)")
    p2 = load_domain(p2_cats, "Koncept B: Sport, IT i polityka (rec/comp/talk)")
    p3 = load_domain(
        p3_cats, "Koncept C: Grafika, religia i kryptografia (comp/soc/sci)"
    )

    return encoder, p1, p2, p3


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "clusterer" not in st.session_state:
    st.session_state.clusterer = StreamClusterer(
        epsilon=config.denstream.epsilon,
        mu=config.denstream.mu,
        beta=config.denstream.beta,
        decaying_factor=config.denstream.decaying_factor,
        n_samples_init=config.denstream.n_samples_init,
        window_size=config.denstream.window_size,
    )
    st.session_state.drift_detector = UnsupervisedDriftDetector(
        window_size=config.drift.window_size,
        min_warmup_steps=config.drift.min_warmup_steps,
        quality_drop_sigma=config.drift.quality_drop_sigma,
        outlier_surge_threshold=config.drift.outlier_surge_threshold,
        cooldown_steps=config.drift.cooldown_steps,
    )
    st.session_state.optimizer = NSGAIIOptimizer(
        population_size=config.evolution.population_size,
        generations=config.evolution.generations,
        crossover_rate=config.evolution.crossover_rate,
        mutation_rate=config.evolution.mutation_rate,
        param_bounds={k: tuple(v) for k, v in config.evolution.param_bounds.items()},
        fixed_mu=config.denstream.mu,
        n_samples_init=config.denstream.n_samples_init,
        min_eval_buffer=config.evolution.min_eval_buffer,
    )
    st.session_state.ipca = IncrementalPCA(
        n_components=config.ml.pca_components_num if config.ml else 16
    )
    st.session_state.ipca_fitted = False

    st.session_state.current_concept = "A"
    st.session_state.is_streaming = False
    st.session_state.stream_idx_a = 0
    st.session_state.stream_idx_b = 0
    st.session_state.stream_idx_c = 0
    st.session_state.total_docs_processed = 0

    st.session_state.history_records = []
    st.session_state.recent_2d_points = []
    st.session_state.recent_labels = []
    st.session_state.recent_texts = []
    st.session_state.recent_macro_preds = []
    st.session_state.recent_micro_preds = []
    st.session_state.recent_vectors_buffer = []
    st.session_state.latest_pareto_front = []
    st.session_state.latest_knee_point = None
    st.session_state.drift_events = []


encoder, data_a, data_b, data_c = load_encoder_and_data()
TOPICS_A = config.dataset.categories_concept_a
TOPICS_B = config.dataset.categories_concept_b
TOPICS_C = config.dataset.categories_concept_c


# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------

if st.session_state.current_concept == "A":
    active_concept_str = "Koncept A: Nauka i motoryzacja (sci/rec)"
    active_topics_list = TOPICS_A
elif st.session_state.current_concept == "B":
    active_concept_str = "Koncept B: Sport, IT i polityka (rec/comp/talk)"
    active_topics_list = TOPICS_B
else:
    active_concept_str = "Koncept C: Grafika, religia i kryptografia (comp/soc/sci)"
    active_topics_list = TOPICS_C

st.sidebar.markdown(f"**Aktywna domena strumienia:**\n`{active_concept_str}`")
st.sidebar.markdown(
    "**Bieżące kategorie w strumieniu:**\n"
    + "".join([f"- `{t}`\n" for t in active_topics_list])
)

# Live Streaming & Pace Controls
st.sidebar.markdown("---")
st.sidebar.markdown("### Przepływ i tempo strumienia")

batch_size = 150  # Hardcoded to prevent breaking the batch-based warmup logic
stream_delay = 0.7  # Hardcoded stream delay

# Auto-Stream Play / Pause Button
if st.session_state.get("is_streaming", False):
    if st.sidebar.button("Zatrzymaj strumień", type="primary", width="stretch"):
        st.session_state.is_streaming = False
        st.rerun()
else:
    if st.sidebar.button("Uruchom ciągły strumień", type="primary", width="stretch"):
        st.session_state.is_streaming = True
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Wstrzykiwanie dryfu pojęć")
abrupt_drift_clicked = st.sidebar.button("Wymuś nagły dryf", width="stretch")


disable_adaptation = st.sidebar.checkbox(
    "Zablokuj adaptację (tylko detekcja)",
    value=False,
    help="Algorytm wykryje dryf, ale nie uruchomi ewolucji (świetne do pokazywania na prezentacji jak model degraduje bez pomocy NSGA-II).",
)
manual_ga_clicked = st.sidebar.button("Uruchom optymalizację NSGA-II", width="stretch")
reset_clicked = st.sidebar.button("Zresetuj stan strumienia", width="stretch")


# ---------------------------------------------------------
# Actions Handling
# ---------------------------------------------------------
if reset_clicked:
    st.session_state.clusterer = StreamClusterer(
        epsilon=config.denstream.epsilon,
        mu=config.denstream.mu,
        beta=config.denstream.beta,
        decaying_factor=config.denstream.decaying_factor,
        n_samples_init=config.denstream.n_samples_init,
        window_size=config.denstream.window_size,
    )
    st.session_state.drift_detector = UnsupervisedDriftDetector(
        window_size=config.drift.window_size,
        min_warmup_steps=config.drift.min_warmup_steps,
        quality_drop_sigma=config.drift.quality_drop_sigma,
        outlier_surge_threshold=config.drift.outlier_surge_threshold,
        cooldown_steps=config.drift.cooldown_steps,
    )
    st.session_state.optimizer = NSGAIIOptimizer(
        population_size=config.evolution.population_size,
        generations=config.evolution.generations,
        crossover_rate=config.evolution.crossover_rate,
        mutation_rate=config.evolution.mutation_rate,
        param_bounds={k: tuple(v) for k, v in config.evolution.param_bounds.items()},
        fixed_mu=config.denstream.mu,
        n_samples_init=config.denstream.n_samples_init,
        min_eval_buffer=config.evolution.min_eval_buffer,
    )
    st.session_state.current_concept = "A"
    st.session_state.is_streaming = False
    st.session_state.stream_idx_a = 0
    st.session_state.stream_idx_b = 0
    st.session_state.stream_idx_c = 0
    st.session_state.total_docs_processed = 0
    st.session_state.history_records = []
    st.session_state.recent_2d_points = []
    st.session_state.recent_labels = []
    st.session_state.recent_texts = []
    st.session_state.recent_macro_preds = []
    st.session_state.recent_micro_preds = []
    st.session_state.recent_vectors_buffer = []
    st.session_state.latest_pareto_front = []
    st.session_state.latest_knee_point = None
    st.session_state.drift_events = []
    st.rerun()

if abrupt_drift_clicked:
    next_concept_map = {"A": "B", "B": "C", "C": "A"}
    st.session_state.current_concept = next_concept_map.get(
        st.session_state.current_concept, "B"
    )
    st.session_state.drift_events.append(st.session_state.total_docs_processed)
    st.sidebar.success(
        f"Wstrzyknięto nagły dryf pojęć. Przełączono na: Koncept {st.session_state.current_concept}"
    )

if manual_ga_clicked:
    if len(st.session_state.recent_vectors_buffer) >= config.evolution.min_eval_buffer:
        with st.spinner("Optymalizacja wielokryterialna NSGA-II..."):
            best_knee, front, _ = st.session_state.optimizer.evolve(
                data_buffer=np.array(st.session_state.recent_vectors_buffer),
                current_params={
                    "epsilon": float(st.session_state.clusterer.model.epsilon),
                    "decaying_factor": float(
                        st.session_state.clusterer.model.decaying_factor
                    ),
                    "mu": int(st.session_state.clusterer.model.mu),
                },
            )
            st.session_state.latest_pareto_front = front
            st.session_state.latest_knee_point = best_knee
            st.session_state.clusterer.hot_swap_model(
                best_knee.params, np.array(st.session_state.recent_vectors_buffer)
            )
            st.sidebar.success(
                f"Wymuszono Hot-Swap: eps={best_knee.params['epsilon']:.3f}, decay={best_knee.params['decaying_factor']:.3f}"
            )
    else:
        st.sidebar.warning(
            f"Niewystarczająca liczba dokumentów w buforze (wymagane min. {config.evolution.min_eval_buffer})."
        )


def _slice_circular(data_list, start_idx, size):
    n = len(data_list)
    if n == 0:
        return [], 0
    if start_idx + size <= n:
        items = data_list[start_idx : start_idx + size]
    else:
        items = data_list[start_idx:] + data_list[: (start_idx + size) % n]
    new_idx = (start_idx + size) % n
    return items, new_idx


def ingest_batch(current_b_size):
    # Select batch from current active concept
    if st.session_state.current_concept == "A":
        batch_items, st.session_state.stream_idx_a = _slice_circular(
            data_a, st.session_state.stream_idx_a, current_b_size
        )
    elif st.session_state.current_concept == "B":
        batch_items, st.session_state.stream_idx_b = _slice_circular(
            data_b, st.session_state.stream_idx_b, current_b_size
        )
    else:
        batch_items, st.session_state.stream_idx_c = _slice_circular(
            data_c, st.session_state.stream_idx_c, current_b_size
        )

    batch_texts = [item[0] for item in batch_items]
    batch_labels = [item[1] for item in batch_items]

    # SBERT Encoding
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    sbert_vecs = encoder.encode(
        batch_texts,
        batch_size=len(batch_texts),
        device=device,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # IPCA Dimensionality Reduction
    if not st.session_state.ipca_fitted:
        if len(sbert_vecs) >= config.ml.pca_components_num:
            st.session_state.ipca.partial_fit(sbert_vecs)
            st.session_state.ipca_fitted = True

    reduced_vecs = normalize(st.session_state.ipca.transform(sbert_vecs))

    # Cluster Update
    batch_macro_preds = st.session_state.clusterer.update(
        reduced_vecs, labels=batch_labels
    )
    if hasattr(st.session_state.clusterer, "last_batch_micro_preds"):
        batch_micro_preds = st.session_state.clusterer.last_batch_micro_preds
    else:
        batch_micro_preds = [
            st.session_state.clusterer.model.predict_one(x)
            for x, _ in stream.iter_array(reduced_vecs)
        ]

    metrics = st.session_state.clusterer.get_metrics()
    st.session_state.total_docs_processed += len(batch_texts)

    # Buffer & Drift Evaluation
    st.session_state.recent_vectors_buffer.extend(reduced_vecs)
    st.session_state.recent_2d_points.extend(reduced_vecs[:, :2])
    st.session_state.recent_labels.extend(batch_labels)
    st.session_state.recent_texts.extend(batch_texts)
    st.session_state.recent_macro_preds.extend(batch_macro_preds)
    st.session_state.recent_micro_preds.extend(batch_micro_preds)

    w_size = config.denstream.window_size
    if len(st.session_state.recent_vectors_buffer) > w_size:
        st.session_state.recent_vectors_buffer = st.session_state.recent_vectors_buffer[
            -w_size:
        ]
        st.session_state.recent_2d_points = st.session_state.recent_2d_points[-w_size:]
        st.session_state.recent_labels = st.session_state.recent_labels[-w_size:]
        st.session_state.recent_texts = st.session_state.recent_texts[-w_size:]
        st.session_state.recent_macro_preds = st.session_state.recent_macro_preds[
            -w_size:
        ]
        st.session_state.recent_micro_preds = st.session_state.recent_micro_preds[
            -w_size:
        ]

    # Compute a reactive outlier ratio based on how many documents in this batch
    # were unassigned (micro_pred == -1) and fell into the outlier buffer.
    # This is much more accurate than n_o / (n_p + n_o) because o-micro-clusters
    # get promoted to p-micro-clusters within the batch loop.
    n_p = metrics["n_micro_clusters"]
    n_o = metrics["n_outlier_clusters"]
    raw_outlier_ratio = float(n_o / max(1, n_p + n_o))

    # --- CONSOLE LOG: batch snapshot (printed every batch for easy debugging) ---
    concept = st.session_state.get("current_concept", "A")
    drift_active = f"Concept {concept}"
    sil = metrics.get("silhouette") or 0.0
    pur = (metrics.get("purity") or 0.0) * 100
    print(
        f"[BATCH #{st.session_state.total_docs_processed:>5}] "
        f"{drift_active} | "
        f"micro={n_p} macro={metrics['n_macro_clusters']} o={n_o} | "
        f"Purity={pur:.1f}% Sil={sil:.4f} R_out={raw_outlier_ratio:.3f} | "
        f"ε={getattr(st.session_state.clusterer.model, 'epsilon', 0):.3f} "
        f"λ={getattr(st.session_state.clusterer.model, 'decaying_factor', 0):.4f}"
    )

    is_drift = st.session_state.drift_detector.update(
        current_silhouette=metrics["silhouette"] or 0.0,
        n_micro_clusters=n_p,
        n_outlier_clusters=n_o,
        outlier_ratio=raw_outlier_ratio,
        n_macro_clusters=metrics.get("n_macro_clusters"),
    )

    if is_drift:
        if disable_adaptation:
            st.toast("🚨 DRYF WYKRYTY! (Ale adaptacja zablokowana)", icon="🛑")
            print(f"\n{'=' * 70}")
            print(
                f"🚨 DRIFT DETECTED @ doc #{st.session_state.total_docs_processed} [ADAPTATION BLOCKED]"
            )
            print(f"{'=' * 70}\n")
        else:
            st.toast("🚨 DRYF POJĘĆ WYKRYTY! Uruchamianie NSGA-II...", icon="🚨")
            print(f"\n{'=' * 70}")
            print(f"🚨 DRIFT DETECTED @ doc #{st.session_state.total_docs_processed}")
            print(
                f"   Sil={sil:.4f} R_out={raw_outlier_ratio:.3f} "
                f"n_macro={metrics['n_macro_clusters']}"
            )
            print(f"{'=' * 70}\n")

    if (
        is_drift
        and not disable_adaptation
        and len(st.session_state.recent_vectors_buffer)
        >= config.evolution.min_eval_buffer
    ):
        print(
            f"⚙️  NSGA-II starting (buffer={len(st.session_state.recent_vectors_buffer)} docs, "
            f"eps_bounds=[{config.evolution.param_bounds['epsilon'][0]}, "
            f"{config.evolution.param_bounds['epsilon'][1]}])..."
        )
        with st.spinner("⚙️ Optymalizacja NSGA-II w toku..."):
            best_knee, front, _ = st.session_state.optimizer.evolve(
                data_buffer=np.array(st.session_state.recent_vectors_buffer),
                current_params={
                    "epsilon": float(st.session_state.clusterer.model.epsilon),
                    "decaying_factor": float(
                        st.session_state.clusterer.model.decaying_factor
                    ),
                    "mu": int(st.session_state.clusterer.model.mu),
                },
            )
        st.session_state.latest_pareto_front = front
        st.session_state.latest_knee_point = best_knee
        print(
            f"✅ NSGA-II done | Front size={len(front)} | "
            f"Knee: ε={best_knee.params['epsilon']:.4f} "
            f"λ={best_knee.params['decaying_factor']:.4f} "
            f"Quality={best_knee.quality_score:.4f} Complexity={best_knee.complexity_score:.4f}"
        )

        # Apply Adaptation Strategy
        st.session_state.clusterer.hot_swap_model(
            best_knee.params, np.array(st.session_state.recent_vectors_buffer)
        )
        st.toast("✅ Zastosowano strategię: Window-Trained Hot-Swap!")
        print("🔄 ADAPTATION: Hot-Swap")

        st.session_state.drift_detector.steps_since_last_drift = 0
        st.session_state.drift_detector.history_quality.clear()

        metrics = st.session_state.clusterer.get_metrics()
        print(
            f"   Post-adaptation: micro={metrics['n_micro_clusters']} "
            f"macro={metrics['n_macro_clusters']} "
            f"ε={getattr(st.session_state.clusterer.model, 'epsilon', 0):.4f}\n"
        )

    st.session_state.history_records.append(
        {
            "docs": st.session_state.total_docs_processed,
            "purity": metrics["purity"],
            "silhouette": metrics["silhouette"],
            "n_micro": metrics["n_micro_clusters"],
            "n_macro": metrics["n_macro_clusters"],
            "outlier_ratio": raw_outlier_ratio,
            "sil_threshold": getattr(
                st.session_state.drift_detector, "current_threshold", None
            ),
            "latency_ms": metrics["latency_ms_per_doc"],
            "eps": getattr(
                st.session_state.clusterer.model, "epsilon", config.denstream.epsilon
            ),
            "mu": getattr(st.session_state.clusterer.model, "mu", config.denstream.mu),
            "decay": getattr(
                st.session_state.clusterer.model,
                "decaying_factor",
                config.denstream.decaying_factor,
            ),
        }
    )


# Ingest if auto-streaming is active
if st.session_state.get("is_streaming", False):
    ingest_batch(batch_size)


# ---------------------------------------------------------
# Top Summary KPI Cards & Active Topics Banner
# ---------------------------------------------------------

curr_m = st.session_state.clusterer.get_metrics()
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Przetworzono", f"{st.session_state.total_docs_processed} dok.")
c2.metric("Mikroklastry", curr_m["n_micro_clusters"])
c3.metric("Makroklastry", curr_m["n_macro_clusters"])
purity_val = f"{curr_m['purity'] * 100:.1f}%" if curr_m["purity"] is not None else "--"
c4.metric("Czystość (Purity)", purity_val)
sil_val = f"{curr_m['silhouette']:.3f}" if curr_m["silhouette"] is not None else "--"
c5.metric("Sylwetka (iCVI)", sil_val)
c6.metric("Opóźnienie", f"{curr_m['latency_ms_per_doc']:.2f} ms/dok.")

# Top Domain Banner
with st.container():
    if st.session_state.current_concept == "A":
        active_concept_str = "Koncept A: Nauka i motoryzacja"
        active_topics_list = TOPICS_A
    elif st.session_state.current_concept == "B":
        active_concept_str = "Koncept B: Sport, IT i polityka"
        active_topics_list = TOPICS_B
    else:
        active_concept_str = "Koncept C: Grafika, religia i kryptografia"
        active_topics_list = TOPICS_C

    # Active Concept Banner
    st.markdown(
        f"""
        <div style="background-color: #f0f7fb; border-left: 5px solid #2980b9; padding: 12px 16px; border-radius: 6px; margin: 10px 0 20px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: bold; color: #2c3e50; font-size: 15px;">Aktywna domena strumienia:</span>
                    <span style="background-color: #2980b9; color: white; padding: 3px 8px; border-radius: 4px; font-weight: bold; margin-left: 8px;">{active_concept_str}</span>
                </div>
            </div>
            <div style="margin-top: 6px; font-size: 13px; color: #34495e;">
                <b>Kategorie 20 Newsgroups w strumieniu:</b> {" &nbsp;•&nbsp; ".join([f"<code>{t}</code>" for t in active_topics_list])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()


# ---------------------------------------------------------
# Main Tabs View
# ---------------------------------------------------------
selected_view = st.radio(
    "Wybierz widok panelu głównego:",
    [
        "Przestrzeń semantyczna i makroklastry (2D)",
        "Telemetria i detekcja dryfu",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

if selected_view == "Przestrzeń semantyczna i makroklastry (2D)":
    st.subheader("Przestrzeń semantyczna i makroklastry (2D)")

    if len(st.session_state.recent_2d_points) > 0:
        import plotly.graph_objects as go
        
        fig_scatter = go.Figure()
        
        # Opcja A: Rysujemy TYLKO mikroklastry i makroklastry (bez surowych dokumentów i mylących kół)
        structs = st.session_state.clusterer.get_cluster_structures()
        micro_list = []
        for k, v in structs.get("p_micro_clusters", {}).items():
            micro_list.append({
                "type": "p-micro",
                "center": v["center"][:2], # Bierzemy tylko 2 pierwsze wymiary centrum
                "key": k,
                "macro_id": v.get("macro_id", -1),
                "weight": v["weight"],
            })
        for k, v in structs.get("o_micro_clusters", {}).items():
            micro_list.append({
                "type": "o-micro",
                "center": v["center"][:2],
                "key": k,
                "macro_id": -1,
                "weight": v["weight"],
            })
            
        macro_clusters = {}
        for m in micro_list:
            if m["type"] == "p-micro" and m["macro_id"] != -1:
                mid = m["macro_id"]
                if mid not in macro_clusters:
                    macro_clusters[mid] = []
                macro_clusters[mid].append(m["center"])
                
        macro_colors = ["#8e44ad", "#2980b9", "#27ae60", "#d35400", "#c0392b", "#f39c12", "#16a085", "#2c3e50"]
        
        # 1. Rysowanie Linii (powiązania mikroklastrów w makroklaster)
        for mid, centers in macro_clusters.items():
            m_color = macro_colors[mid % len(macro_colors)]
            mean_center = __import__("numpy").mean(centers, axis=0).tolist()
            
            for c in centers:
                fig_scatter.add_trace(go.Scatter(
                    x=[mean_center[0], c[0]],
                    y=[mean_center[1], c[1]],
                    mode="lines",
                    line=dict(color=m_color, width=2, dash="dot"),
                    opacity=0.6,
                    showlegend=False,
                    hoverinfo="none"
                ))
            
            # Punkt centralny Makroklastra (bez ogromnego kółka)
            fig_scatter.add_trace(go.Scatter(
                x=[mean_center[0]],
                y=[mean_center[1]],
                mode="text",
                text=[f"<b>Makro #{mid}</b>"],
                textposition="top center",
                textfont=dict(color=m_color, size=14),
                showlegend=False,
                hoverinfo="none"
            ))

        # 2. Rysowanie Mikroklastrów P-Micro
        for mid, centers in macro_clusters.items():
            m_color = macro_colors[mid % len(macro_colors)]
            keys = [m["key"] for m in micro_list if m["type"] == "p-micro" and m["macro_id"] == mid]
            weights = [m["weight"] for m in micro_list if m["type"] == "p-micro" and m["macro_id"] == mid]
            
            fig_scatter.add_trace(go.Scatter(
                x=[c[0] for c in centers],
                y=[c[1] for c in centers],
                mode="markers+text",
                text=[f"mc{k}" for k in keys],
                textposition="bottom right",
                marker=dict(
                    size=[min(30, max(10, w * 1.5)) for w in weights], # Skalowanie po wadze
                    color=m_color,
                    line=dict(width=2, color="#ffffff"),
                ),
                name=f"Makro #{mid}",
                hovertext=[f"Mikroklaster {k} (Waga: {w:.1f})" for k, w in zip(keys, weights)],
                hoverinfo="text"
            ))
            
        # 3. Rysowanie Szumu O-Micro
        o_micros = [m for m in micro_list if m["type"] == "o-micro"]
        if o_micros:
            fig_scatter.add_trace(go.Scatter(
                x=[m["center"][0] for m in o_micros],
                y=[m["center"][1] for m in o_micros],
                mode="markers",
                marker=dict(size=8, color="#e74c3c", symbol="x", line=dict(width=2, color="#e74c3c")),
                name="Szum (O-Micro)",
                hovertext=[f"Szum MC_{m['key']}" for m in o_micros],
                hoverinfo="text"
            ))

        fig_scatter.update_layout(
            title="Architektura klastrów online (Mikroklastry & Makroklastry)",
            height=580,
            xaxis_title="IPCA Składowa 1",
            yaxis_title="IPCA Składowa 2",
            showlegend=True,
            plot_bgcolor="whitesmoke"
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        # Sekcja z przykładowymi dokumentami
        st.markdown("### 📝 Przykładowe dokumenty z ostatniej partii strumienia")
        if len(st.session_state.recent_texts) > 0:
            sample_texts = st.session_state.recent_texts[-3:]
            sample_labels = st.session_state.recent_labels[-3:]
            sample_macros = st.session_state.recent_macro_preds[-3:]
            
            cols = st.columns(3)
            for i, (txt, lbl, mac) in enumerate(zip(sample_texts, sample_labels, sample_macros)):
                with cols[i]:
                    mac_str = f"Makro {mac}" if mac != -1 else "Szum"
                    color = macro_colors[mac % len(macro_colors)] if mac != -1 else "#e74c3c"
                    st.markdown(f'''
                    <div style="border-top: 4px solid {color}; padding: 10px; background-color: #f8f9fa; border-radius: 5px; height: 100%;">
                        <div style="font-size: 11px; color: #7f8c8d; margin-bottom: 5px;">Kategoria: <b>{lbl}</b> | Przypisanie: <b>{mac_str}</b></div>
                        <div style="font-size: 13px; color: #2c3e50; font-style: italic;">"{txt[:180]}..."</div>
                    </div>
                    ''', unsafe_allow_html=True)


    else:
        st.info(
            "Brak punktów w buforze przestrzeni dwuwymiarowej. Kliknij przycisk 'Uruchom ciągły strumień', aby rozpocząć przetwarzanie danych."
        )

elif selected_view == "Telemetria i detekcja dryfu":
    st.subheader("Wskaźniki telemetryczne i sygnały detekcji dryfu pojęć")
    if len(st.session_state.history_records) > 0:
        df_hist = pd.DataFrame(st.session_state.history_records)

        fig_pur = go.Figure()
        fig_pur.add_trace(
            go.Scatter(
                x=df_hist["docs"],
                y=df_hist["purity"],
                mode="lines",
                name="Czystość klastrów (Purity)",
                line=dict(color="#2ecc71", width=2),
            )
        )
        for d in st.session_state.drift_events:
            fig_pur.add_vline(
                x=d,
                line_dash="dash",
                line_color="red",
                annotation_text="Dryf pojęć",
                annotation_position="top right",
            )
        fig_pur.update_layout(
            title="Czystość klastrów (Purity) w czasie",
            xaxis_title="Liczba przetworzonych dokumentów",
            yaxis_title="Purity",
            height=250,
        )
        st.plotly_chart(fig_pur, width="stretch")

        fig_sil = go.Figure()
        fig_sil.add_trace(
            go.Scatter(
                x=df_hist["docs"],
                y=df_hist["silhouette"],
                mode="lines",
                name="Wskaźnik sylwetki (iCVI)",
                line=dict(color="#3498db", width=2),
            )
        )
        if (
            "sil_threshold" in df_hist.columns
            and not df_hist["sil_threshold"].isnull().all()
        ):
            fig_sil.add_trace(
                go.Scatter(
                    x=df_hist["docs"],
                    y=df_hist["sil_threshold"],
                    mode="lines",
                    name="Próg alarmowy (Dynamiczny)",
                    line=dict(color="#e74c3c", width=1.5, dash="dash"),
                )
            )
        for d in st.session_state.drift_events:
            fig_sil.add_vline(
                x=d,
                line_dash="dash",
                line_color="red",
                annotation_text="Dryf pojęć",
                annotation_position="top right",
            )
        fig_sil.update_layout(
            title="Wskaźnik sylwetki (iCVI) w czasie",
            xaxis_title="Liczba przetworzonych dokumentów",
            yaxis_title="iCVI",
            height=250,
            yaxis=dict(range=[-0.1, 0.2]),
        )
        st.plotly_chart(fig_sil, width="stretch")

        fig_out = go.Figure()
        fig_out.add_trace(
            go.Scatter(
                x=df_hist["docs"],
                y=df_hist["outlier_ratio"],
                mode="lines",
                name="Wskaźnik odstających (R_outlier)",
                line=dict(color="#e67e22", width=2),
            )
        )
        fig_out.add_hline(
            y=config.drift.outlier_surge_threshold,
            line_dash="dash",
            line_color="#333333",
            annotation_text=f"Próg alarmowy dryfu R_outlier ({config.drift.outlier_surge_threshold * 100:.0f}%)",
        )
        for d in st.session_state.drift_events:
            fig_out.add_vline(x=d, line_dash="dash", line_color="red")
        fig_out.update_layout(
            title="Wskaźnik obserwacji odstających R_outlier (Sygnał dryfu)",
            xaxis_title="Liczba przetworzonych dokumentów",
            yaxis_title="Wskaźnik R_outlier",
            height=300,
        )
        st.plotly_chart(fig_out, width="stretch")

        fig_params = go.Figure()
        fig_params.add_trace(
            go.Scatter(
                x=df_hist["docs"],
                y=df_hist["eps"],
                mode="lines",
                name="Promień ε",
                line=dict(color="#3498db", width=2),
            )
        )
        fig_params.add_trace(
            go.Scatter(
                x=df_hist["docs"],
                y=df_hist["decay"],
                mode="lines",
                name="Czynnik wygaszania λ",
                line=dict(color="#9b59b6", width=2),
            )
        )
        fig_params.update_layout(
            title="Trajektoria samostrojenia parametrów",
            xaxis_title="Liczba przetworzonych dokumentów",
            yaxis_title="Wartość parametru",
            height=300,
        )
        st.plotly_chart(fig_params, width="stretch")

        fig_lat = px.line(
            df_hist,
            x="docs",
            y="latency_ms",
            title="Opóźnienie przetwarzania strumieniowego na dokument [ms]",
        )
        fig_lat.update_layout(
            xaxis_title="Liczba przetworzonych dokumentów",
            yaxis_title="Opóźnienie [ms]",
            height=300,
        )
        st.plotly_chart(fig_lat, width="stretch")
    else:
        st.info(
            "Brak zarejestrowanych danych telemetrycznych. Rozpocznij strumieniowanie z panelu bocznego."
        )

elif selected_view == "Baza danych (PostgreSQL)":
    st.subheader("Baza danych PostgreSQL: Telemetria i historia eksperymentu")
    if getattr(st.session_state, "db", None) and st.session_state.db.is_connected():
        st.success(
            f"**Połączenie aktywne:** PostgreSQL `{config.postgres.host}:{config.postgres.port}` | Baza: `{config.postgres.db}`"
        )
        try:
            df_sql = pd.read_sql(
                f"SELECT * FROM {config.postgres.results_table} ORDER BY timestamp DESC LIMIT 200;",
                st.session_state.db.engine,
            )
            if len(df_sql) > 0:
                st.dataframe(df_sql.head(50))
            else:
                st.info(
                    "Tabela `clustering_results` w PostgreSQL jest obecnie pusta. Rozpocznij strumieniowanie, aby automatycznie zapisywać telemetrię."
                )
        except Exception as e:
            st.error(f"Błąd odczytu z bazy: {e}")
    else:
        st.error(
            "**Baza danych PostgreSQL jest obecnie offline.**\nAplikacja działa w niezależnym trybie pamięciowym (In-Memory)."
        )

if st.session_state.get("is_streaming", False):
    import time

    time.sleep(stream_delay)
    st.rerun()
