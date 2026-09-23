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
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

from src.core.config import config
from src.domain.clustering import StreamClusterer
from src.domain.drift import UnsupervisedDriftDetector
from src.domain.evolution import NSGAIIOptimizer
from src.domain.preprocessing import StreamProjector, TextPreprocessor
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


encoder, data_a, data_b, data_c = load_encoder_and_data()
N_MACRO_CLUSTERS = len(config.dataset.categories_concept_a)


def create_projector() -> StreamProjector:
    """IPCA fitted once on the first documents of concept A and then frozen,
    as in the thesis experiments; it is re-fitted only on a model swap."""
    projector = StreamProjector(n_components=config.ml.pca_components_num)
    warmup_texts = [item[0] for item in data_a[: config.ml.ipca_warmup_size]]
    projector.fit(
        encoder.encode(warmup_texts, normalize_embeddings=True, convert_to_numpy=True)
    )
    return projector


def create_clusterer() -> StreamClusterer:
    return StreamClusterer(
        epsilon=config.denstream.epsilon,
        mu=config.denstream.mu,
        beta=config.denstream.beta,
        decaying_factor=config.denstream.decaying_factor,
        n_samples_init=config.denstream.n_samples_init,
        window_size=config.denstream.window_size,
        expected_macro_clusters=N_MACRO_CLUSTERS,
    )


def create_detector() -> UnsupervisedDriftDetector:
    return UnsupervisedDriftDetector(
        window_size=config.drift.window_size,
        min_warmup_steps=config.drift.min_warmup_steps,
        quality_drop_sigma=config.drift.quality_drop_sigma,
        cooldown_steps=config.drift.cooldown_steps,
        consecutive_drops_required=config.drift.consecutive_drops_required,
        centroid_shift_threshold=config.drift.centroid_shift_threshold,
        quality_absolute_floor=config.drift.quality_absolute_floor,
        centroid_shift_min_warmup_steps=config.drift.centroid_shift_min_warmup_steps,
    )


def create_optimizer() -> NSGAIIOptimizer:
    return NSGAIIOptimizer(
        n_macro_clusters=N_MACRO_CLUSTERS,
        population_size=config.evolution.population_size,
        generations=config.evolution.generations,
        crossover_rate=config.evolution.crossover_rate,
        crossover_eta=config.evolution.crossover_eta,
        mutation_rate=config.evolution.mutation_rate,
        mutation_eta=config.evolution.mutation_eta,
        param_bounds={k: tuple(v) for k, v in config.evolution.param_bounds.items()},
        fixed_mu=config.denstream.mu,
        n_samples_init=config.denstream.n_samples_init,
        min_eval_buffer=config.evolution.min_eval_buffer,
        seed=config.evolution.seed,
    )


def reset_session_state():
    st.session_state.clusterer = create_clusterer()
    st.session_state.drift_detector = create_detector()
    st.session_state.optimizer = create_optimizer()
    st.session_state.projector = create_projector()

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
    st.session_state.recent_vectors_buffer = []
    st.session_state.recent_raw_buffer = []
    st.session_state.collecting_for_swap = False
    st.session_state.swap_buffer = []
    st.session_state.latest_pareto_front = []
    st.session_state.latest_compromise = None
    st.session_state.drift_events = []
    st.session_state.detection_events = []


def swap_model(raw_buffer: np.ndarray):
    """Model swap as in the thesis 2 experiment: re-fits IPCA on the raw
    embeddings of the buffer, runs NSGA-II on the projected buffer and swaps
    in the compromise-solution model trained on it."""
    st.session_state.projector.fit(raw_buffer)
    projected = st.session_state.projector.transform(raw_buffer)
    st.session_state.recent_vectors_buffer = list(projected[-config.denstream.window_size :])
    st.session_state.recent_2d_points = list(projected[-config.denstream.window_size :, :2])

    compromise, front, _ = st.session_state.optimizer.evolve(
        data_buffer=projected,
        current_params={
            "epsilon": float(st.session_state.clusterer.model.epsilon),
            "decaying_factor": float(st.session_state.clusterer.model.decaying_factor),
            "mu": int(st.session_state.clusterer.model.mu),
        },
    )
    st.session_state.latest_pareto_front = front
    st.session_state.latest_compromise = compromise
    st.session_state.clusterer.hot_swap_model(compromise.params, projected)
    print(
        f"✅ NSGA-II done | Front size={len(front)} | "
        f"Compromise: ε={compromise.params['epsilon']:.4f} "
        f"λ={compromise.params['decaying_factor']:.4f} "
        f"Quality={compromise.quality_score:.4f} Complexity={compromise.complexity_score:.4f}"
    )
    return compromise


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "clusterer" not in st.session_state:
    reset_session_state()


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

# Same batch size as the thesis experiments - the detector's windows and
# warm-up periods are counted in batches.
batch_size = config.ml.batch_size
stream_delay = 0.2

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

st.sidebar.markdown("---")
st.sidebar.markdown("### Stan adaptacji")
_warmup = config.drift.centroid_shift_min_warmup_steps
_steps = st.session_state.drift_detector.total_steps_seen
if _steps < _warmup:
    st.sidebar.caption(f"Rozgrzewka sygnału przesunięcia centroidów: {_steps}/{_warmup} partii")
else:
    st.sidebar.caption("Detektor dryfu aktywny (oba sygnały)")
if st.session_state.collecting_for_swap:
    _collected = len(st.session_state.swap_buffer)
    _needed = config.evolution.hotswap_buffer_size
    st.sidebar.progress(
        min(1.0, _collected / _needed),
        text=f"Zbieranie dokumentów po dryfie: {_collected}/{_needed}",
    )


# ---------------------------------------------------------
# Actions Handling
# ---------------------------------------------------------
if reset_clicked:
    reset_session_state()
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
    # Manual trigger for demonstrations: swaps immediately on the most recent
    # window of documents instead of waiting for a drift alarm.
    if len(st.session_state.recent_raw_buffer) >= config.evolution.min_eval_buffer:
        with st.spinner("Optymalizacja wielokryterialna NSGA-II..."):
            compromise = swap_model(np.array(st.session_state.recent_raw_buffer))
        st.sidebar.success(
            f"Wymieniono model: ε={compromise.params['epsilon']:.3f}, λ={compromise.params['decaying_factor']:.3f}"
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

    # IPCA projection (frozen between model swaps)
    reduced_vecs = st.session_state.projector.transform(sbert_vecs)

    # Cluster Update
    batch_macro_preds = st.session_state.clusterer.update(
        reduced_vecs, labels=batch_labels
    )
    metrics = st.session_state.clusterer.get_metrics()
    st.session_state.total_docs_processed += len(batch_texts)

    # Recent-window buffers (plots, sample documents, manual NSGA-II)
    w_size = config.denstream.window_size
    st.session_state.recent_vectors_buffer = (st.session_state.recent_vectors_buffer + list(reduced_vecs))[-w_size:]
    st.session_state.recent_raw_buffer = (st.session_state.recent_raw_buffer + list(sbert_vecs))[-w_size:]
    st.session_state.recent_2d_points = (st.session_state.recent_2d_points + list(reduced_vecs[:, :2]))[-w_size:]
    st.session_state.recent_labels = (st.session_state.recent_labels + batch_labels)[-w_size:]
    st.session_state.recent_texts = (st.session_state.recent_texts + batch_texts)[-w_size:]
    st.session_state.recent_macro_preds = (st.session_state.recent_macro_preds + batch_macro_preds)[-w_size:]

    sil = metrics.get("silhouette") or 0.0
    shift = metrics.get("centroid_shift")
    pur = (metrics.get("purity") or 0.0) * 100
    print(
        f"[BATCH #{st.session_state.total_docs_processed:>5}] "
        f"Concept {st.session_state.current_concept} | "
        f"micro={metrics['n_micro_clusters']} macro={metrics['n_macro_clusters']} | "
        f"Purity={pur:.1f}% Sil={sil:.4f} Shift={shift if shift is None else round(shift, 3)} | "
        f"ε={st.session_state.clusterer.model.epsilon:.3f} "
        f"λ={st.session_state.clusterer.model.decaying_factor:.4f}"
    )

    # Drift detection and model swap, as in the thesis 2 experiment: an alarm
    # starts collecting a buffer of post-drift documents; once it is full,
    # IPCA is re-fitted, NSGA-II runs and the new model is swapped in.
    is_drift = st.session_state.drift_detector.update(
        current_silhouette=sil, centroid_shift=shift
    )
    if is_drift:
        st.session_state.detection_events.append(st.session_state.total_docs_processed)
        print(f"🚨 DRIFT DETECTED @ doc #{st.session_state.total_docs_processed}")
        if disable_adaptation:
            st.toast("🚨 Wykryto dryf (adaptacja zablokowana)", icon="🛑")
        elif not st.session_state.collecting_for_swap:
            st.toast(
                f"🚨 Wykryto dryf! Zbieranie {config.evolution.hotswap_buffer_size} nowych dokumentów przed optymalizacją NSGA-II...",
                icon="🚨",
            )
            st.session_state.collecting_for_swap = True
            st.session_state.swap_buffer = []

    if st.session_state.collecting_for_swap:
        st.session_state.swap_buffer.extend(sbert_vecs)
        if len(st.session_state.swap_buffer) >= config.evolution.hotswap_buffer_size:
            with st.spinner("⚙️ Optymalizacja NSGA-II w toku..."):
                swap_model(np.array(st.session_state.swap_buffer))
            st.session_state.collecting_for_swap = False
            st.session_state.swap_buffer = []
            st.toast("✅ Wdrożono nowy model (IPCA + DenStream)")
            metrics = st.session_state.clusterer.get_metrics()
    else:
        st.session_state.clusterer.ease_decaying_factor(config.denstream.decaying_factor)

    st.session_state.history_records.append(
        {
            "docs": st.session_state.total_docs_processed,
            "purity": metrics["purity"],
            "silhouette": metrics["silhouette"],
            "centroid_shift": metrics["centroid_shift"],
            "n_micro": metrics["n_micro_clusters"],
            "n_macro": metrics["n_macro_clusters"],
            "sil_threshold": st.session_state.drift_detector.current_threshold,
            "latency_ms": metrics["latency_ms_per_doc"],
            "eps": st.session_state.clusterer.model.epsilon,
            "decay": st.session_state.clusterer.model.decaying_factor,
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

        fig_shift = go.Figure()
        fig_shift.add_trace(
            go.Scatter(
                x=df_hist["docs"],
                y=df_hist["centroid_shift"],
                mode="lines",
                name="Przesunięcie centroidów makroklastrów",
                line=dict(color="#e67e22", width=2),
            )
        )
        fig_shift.add_hline(
            y=config.drift.centroid_shift_threshold,
            line_dash="dash",
            line_color="#333333",
            annotation_text=f"Próg alarmowy ({config.drift.centroid_shift_threshold})",
        )
        for d in st.session_state.drift_events:
            fig_shift.add_vline(x=d, line_dash="dash", line_color="red")
        for d in st.session_state.detection_events:
            fig_shift.add_vline(x=d, line_dash="dot", line_color="#8e44ad")
        fig_shift.update_layout(
            title="Przesunięcie centroidów makroklastrów (sygnał dryfu)",
            xaxis_title="Liczba przetworzonych dokumentów",
            yaxis_title="Średnie przesunięcie",
            height=300,
        )
        st.plotly_chart(fig_shift, width="stretch")
        st.caption(
            f"Czerwone linie – wstrzyknięty dryf, fioletowe – wykrycie dryfu. Sygnał jest aktywny po "
            f"{config.drift.centroid_shift_min_warmup_steps} partiach "
            f"({config.drift.centroid_shift_min_warmup_steps * batch_size} dokumentach) od startu."
        )

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
