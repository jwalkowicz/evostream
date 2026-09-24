import random
import sys
import time
from pathlib import Path

# Streamlit runs this file as a script, so the project root has to be added to the path.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

from src.core.config import config
from src.core.logger import logger
from src.domain.clustering import StreamClusterer
from src.domain.drift import UnsupervisedDriftDetector
from src.domain.evolution import NSGAIIOptimizer
from src.domain.preprocessing import StreamProjector, TextPreprocessor

# The two topic sets of the thesis 2 stream; "Wymuś nagły dryf" switches between them.
CONCEPTS = {
    "A": ("Tematy przed zmianą (faza 1)", config.dataset.categories_concept_a),
    "B": ("Tematy po zmianie (faza 2)", config.dataset.categories_concept_b),
}
NEXT_CONCEPT = {"A": "B", "B": "A"}
N_MACRO_CLUSTERS = len(config.dataset.categories_concept_a)
BATCH_SIZE = config.ml.batch_size
WINDOW = config.denstream.window_size
STREAM_DELAY_S = 0.2
# Starting epsilon of the demo: on the thesis 2 stream it detects the topic change
# after 6 batches, wherever the change is forced (no alarm before it at any epsilon).
START_EPSILON = 0.05
MACRO_COLORS = ["#8e44ad", "#2980b9", "#27ae60", "#d35400", "#c0392b", "#f39c12", "#16a085", "#2c3e50"]
NOISE_COLOR = "#e74c3c"


@st.cache_resource
def load_encoder_and_streams():
    """SBERT model and the cleaned, shuffled documents of each concept."""
    encoder = SentenceTransformer(config.ml.embedding_model, device="cpu")
    preprocessor = TextPreprocessor()
    streams = {}
    for key, (_, categories) in CONCEPTS.items():
        raw = fetch_20newsgroups(subset="all", categories=categories, remove=("headers", "footers", "quotes"))
        docs = []
        for text, target in zip(raw.data, raw.target):
            cleaned = preprocessor.clean(text)
            if len(cleaned.split()) >= 10:
                docs.append((cleaned, raw.target_names[target]))
        random.seed(config.seed)
        random.shuffle(docs)
        streams[key] = docs[: config.dataset.max_samples_per_concept]
    return encoder, streams


encoder, streams = load_encoder_and_streams()


def fmt(value: float, digits: int) -> str:
    """Number with a decimal comma, as in the thesis."""
    return f"{value:.{digits}f}".replace(".", ",")


def encode(texts: list[str]) -> np.ndarray:
    return encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def reset_state() -> None:
    """New model warm-started on the first documents of concept A, as in the experiments."""
    warmup = encode([text for text, _ in streams["A"][: config.ml.ipca_warmup_size]])
    state = st.session_state
    state.projector = StreamProjector(n_components=config.ml.pca_components_num)
    state.projector.fit(warmup)
    state.clusterer = StreamClusterer(
        epsilon=START_EPSILON,
        mu=config.denstream.mu,
        beta=config.denstream.beta,
        decaying_factor=config.denstream.decaying_factor,
        n_samples_init=config.denstream.n_samples_init,
        window_size=WINDOW,
        expected_macro_clusters=N_MACRO_CLUSTERS,
    )
    state.clusterer.warm_start(state.projector.transform(warmup))
    state.detector = UnsupervisedDriftDetector(
        window_size=config.drift.window_size,
        min_warmup_steps=config.drift.min_warmup_steps,
        quality_drop_sigma=config.drift.quality_drop_sigma,
        cooldown_steps=config.drift.cooldown_steps,
        consecutive_drops_required=config.drift.consecutive_drops_required,
        centroid_shift_threshold=config.drift.centroid_shift_threshold,
        quality_absolute_floor=config.drift.quality_absolute_floor,
        centroid_shift_min_warmup_steps=config.drift.centroid_shift_min_warmup_steps,
    )
    state.optimizer = NSGAIIOptimizer(
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
    state.concept = "A"
    state.stream_pos = {"A": config.ml.ipca_warmup_size, "B": 0, "C": 0}
    state.streaming = False
    state.docs_processed = 0
    state.history = []
    state.recent_raw = []
    state.recent_docs = []
    state.collecting_for_swap = False
    state.swap_buffer = []
    state.injected_drifts = []
    state.detected_drifts = []


def swap_model(raw_buffer: np.ndarray):
    """Model swap as in the thesis 2 experiment: IPCA refitted on the buffer,
    NSGA-II on the projected buffer, new DenStream trained on it."""
    state = st.session_state
    state.projector.fit(raw_buffer)
    projected = state.projector.transform(raw_buffer)
    compromise, front, _ = state.optimizer.evolve(data_buffer=projected)
    state.clusterer.hot_swap_model(compromise.params, projected)
    logger.info(
        f"Model swapped: front size {len(front)}, eps={compromise.params['epsilon']:.4f}, "
        f"lambda={compromise.params['decaying_factor']:.4f}"
    )
    return compromise


def next_batch() -> list[tuple[str, str]]:
    """Next documents of the active concept; the stream wraps around at the end."""
    state = st.session_state
    docs = streams[state.concept]
    start = state.stream_pos[state.concept]
    batch = [docs[(start + i) % len(docs)] for i in range(BATCH_SIZE)]
    state.stream_pos[state.concept] = (start + BATCH_SIZE) % len(docs)
    return batch


def process_batch(adaptation_enabled: bool) -> None:
    state = st.session_state
    batch = next_batch()
    texts, labels = [text for text, _ in batch], [label for _, label in batch]
    raw = encode(texts)
    predictions = state.clusterer.update(state.projector.transform(raw), labels=labels)
    state.docs_processed += len(batch)
    state.recent_raw = (state.recent_raw + list(raw))[-WINDOW:]
    state.recent_docs = (state.recent_docs + list(zip(texts, labels, predictions)))[-WINDOW:]

    # As in the thesis 2 experiment: an alarm starts collecting a buffer, and
    # once it is full IPCA is refitted, NSGA-II runs and the model is swapped.
    metrics = state.clusterer.get_metrics()
    if state.detector.update(current_silhouette=metrics["silhouette"] or 0.0, centroid_shift=metrics["centroid_shift"]):
        state.detected_drifts.append(state.docs_processed)
        if not adaptation_enabled:
            st.toast("Wykryto dryf (adaptacja wyłączona)")
        elif not state.collecting_for_swap:
            st.toast(f"Wykryto dryf – zbieranie {config.evolution.hotswap_buffer_size} dokumentów do wymiany modelu")
            state.collecting_for_swap = True
            state.swap_buffer = []

    if state.collecting_for_swap:
        state.swap_buffer.extend(raw)
        if len(state.swap_buffer) >= config.evolution.hotswap_buffer_size:
            with st.spinner("Optymalizacja NSGA-II..."):
                swap_model(np.array(state.swap_buffer))
            state.collecting_for_swap = False
            state.swap_buffer = []
            st.toast("Wdrożono nowy model")
    else:
        state.clusterer.ease_decaying_factor(config.denstream.decaying_factor)

    state.history.append(
        {
            "docs": state.docs_processed,
            "purity": metrics["purity"],
            "silhouette": metrics["silhouette"],
            "silhouette_threshold": state.detector.current_threshold,
            "centroid_shift": metrics["centroid_shift"],
            "epsilon": state.clusterer.model.epsilon,
            "decay": state.clusterer.model.decaying_factor,
            "latency_ms": metrics["latency_ms_per_doc"],
        }
    )


def render_sidebar() -> bool:
    """Controls; returns whether adaptation is enabled."""
    state = st.session_state
    sidebar = st.sidebar
    name, categories = CONCEPTS[state.concept]
    sidebar.markdown(f"**Aktywne tematy:** {name}")
    sidebar.markdown("\n".join(f"- `{c}`" for c in categories))
    sidebar.divider()

    if state.streaming:
        if sidebar.button("Zatrzymaj strumień", type="primary", width="stretch"):
            state.streaming = False
            st.rerun()
    elif sidebar.button("Uruchom strumień", type="primary", width="stretch"):
        state.streaming = True
        st.rerun()

    if sidebar.button("Wymuś nagły dryf", width="stretch"):
        state.concept = NEXT_CONCEPT[state.concept]
        state.injected_drifts.append(state.docs_processed)
        sidebar.success(f"Przełączono na: {CONCEPTS[state.concept][0]}")

    adaptation_enabled = not sidebar.checkbox(
        "Wyłącz adaptację (tylko detekcja)",
        help="Detektor nadal zgłasza dryf, ale model nie jest wymieniany.",
    )

    if sidebar.button("Uruchom NSGA-II teraz", width="stretch"):
        if len(state.recent_raw) >= config.evolution.min_eval_buffer:
            with st.spinner("Optymalizacja NSGA-II..."):
                compromise = swap_model(np.array(state.recent_raw))
            sidebar.success(
                f"Nowy model: ε = {fmt(compromise.params['epsilon'], 3)}, λ = {fmt(compromise.params['decaying_factor'], 3)}"
            )
        else:
            sidebar.warning(f"Za mało dokumentów (potrzeba co najmniej {config.evolution.min_eval_buffer}).")

    if sidebar.button("Resetuj", width="stretch"):
        reset_state()
        st.rerun()

    sidebar.divider()
    warmup_steps = config.drift.centroid_shift_min_warmup_steps
    steps = state.detector.total_steps_seen
    if steps < warmup_steps:
        sidebar.caption(f"Rozgrzewka sygnału przesunięcia centroidów: {steps}/{warmup_steps} partii")
    else:
        sidebar.caption("Detektor dryfu aktywny (oba sygnały)")
    if state.collecting_for_swap:
        needed = config.evolution.hotswap_buffer_size
        collected = len(state.swap_buffer)
        sidebar.progress(min(1.0, collected / needed), text=f"Bufor wymiany modelu: {collected}/{needed}")
    return adaptation_enabled


def render_metrics() -> None:
    state = st.session_state
    metrics = state.clusterer.get_metrics()
    purity, silhouette = metrics["purity"], metrics["silhouette"]
    cols = st.columns(6)
    cols[0].metric("Przetworzone dokumenty", state.docs_processed)
    cols[1].metric("Mikroklastry", metrics["n_micro_clusters"])
    cols[2].metric("Makroklastry", metrics["n_macro_clusters"])
    cols[3].metric("Czystość", "–" if pd.isna(purity) else fmt(purity, 3))
    cols[4].metric("Wskaźnik sylwetki", "–" if pd.isna(silhouette) else fmt(silhouette, 3))
    cols[5].metric("Czas [ms/dok.]", fmt(metrics["latency_ms_per_doc"], 2))


def render_cluster_view() -> None:
    """Micro-cluster centres on the first two IPCA components, coloured by macro-cluster."""
    state = st.session_state
    structures = state.clusterer.get_cluster_structures()
    by_macro: dict[int, list[tuple]] = {}
    for key, mc in structures["p_micro_clusters"].items():
        by_macro.setdefault(mc["macro_id"], []).append((key, mc["center"][:2], mc["weight"]))

    fig = go.Figure()
    for macro_id, members in sorted(by_macro.items()):
        color = MACRO_COLORS[macro_id % len(MACRO_COLORS)]
        centres = np.array([centre for _, centre, _ in members])
        mean = centres.mean(axis=0)
        for x, y in centres:
            fig.add_trace(
                go.Scatter(
                    x=[mean[0], x],
                    y=[mean[1], y],
                    mode="lines",
                    line=dict(color=color, width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=centres[:, 0],
                y=centres[:, 1],
                mode="markers",
                name=f"Makroklaster {macro_id}",
                marker=dict(
                    size=[min(30, max(10, w * 1.5)) for _, _, w in members],
                    color=color,
                    line=dict(width=1, color="white"),
                ),
                hovertext=[f"Mikroklaster {key}, waga {w:.1f}" for key, _, w in members],
                hoverinfo="text",
            )
        )
    outliers = list(structures["o_micro_clusters"].values())
    if outliers:
        fig.add_trace(
            go.Scatter(
                x=[mc["center"][0] for mc in outliers],
                y=[mc["center"][1] for mc in outliers],
                mode="markers",
                name="o-mikroklastry",
                marker=dict(size=8, color=NOISE_COLOR, symbol="x"),
            )
        )
    fig.update_layout(separators=", ", height=560, xaxis_title="Składowa IPCA 1", yaxis_title="Składowa IPCA 2")
    st.plotly_chart(fig, width="stretch")

    if state.recent_docs:
        st.markdown("#### Ostatnie dokumenty")
        for col, (text, label, macro_id) in zip(st.columns(3), state.recent_docs[-3:]):
            assigned = "szum" if macro_id == -1 else f"makroklaster {macro_id}"
            col.caption(f"Kategoria: {label} | przypisanie: {assigned}")
            col.write(text[:180] + "…")


def line_chart(df: pd.DataFrame, series: dict[str, str], title: str, threshold: float | None = None) -> go.Figure:
    fig = go.Figure()
    for column, name in series.items():
        fig.add_trace(go.Scatter(x=df["docs"], y=df[column], mode="lines", name=name))
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#333333")
    for docs in st.session_state.injected_drifts:
        fig.add_vline(x=docs, line_dash="dash", line_color="red")
    for docs in st.session_state.detected_drifts:
        fig.add_vline(x=docs, line_dash="dot", line_color="#8e44ad")
    fig.update_layout(separators=", ", title=title, height=280, xaxis_title="Przetworzone dokumenty", margin=dict(t=40))
    return fig


def render_telemetry_view() -> None:
    df = pd.DataFrame(st.session_state.history)
    st.caption(
        "Czerwone linie – wymuszony dryf, fioletowe – dryf wykryty przez detektor. Po wymianie modelu "
        f"przesunięcie centroidów jest liczone od nowa, więc przez {st.session_state.clusterer.centroid_shift_lookback_batches} partii nie ma wartości."
    )
    st.plotly_chart(line_chart(df, {"purity": "Czystość"}, "Czystość"), width="stretch")
    st.plotly_chart(
        line_chart(
            df, {"silhouette": "Wskaźnik sylwetki", "silhouette_threshold": "Próg względny"}, "Wskaźnik sylwetki"
        ),
        width="stretch",
    )
    st.plotly_chart(
        line_chart(
            df,
            {"centroid_shift": "Przesunięcie centroidów"},
            "Przesunięcie centroidów makroklastrów",
            threshold=config.drift.centroid_shift_threshold,
        ),
        width="stretch",
    )
    st.plotly_chart(line_chart(df, {"epsilon": "ε", "decay": "λ"}, "Parametry modelu"), width="stretch")
    st.plotly_chart(
        line_chart(df, {"latency_ms": "Czas przetwarzania"}, "Czas przetwarzania [ms/dok.]"), width="stretch"
    )


st.set_page_config(page_title="evostream", layout="wide")
if "clusterer" not in st.session_state:
    reset_state()

adaptation_enabled = render_sidebar()
if st.session_state.streaming:
    process_batch(adaptation_enabled)

st.title("evostream")
render_metrics()
cluster_tab, telemetry_tab = st.tabs(["Mikroklastry i makroklastry", "Metryki i detekcja dryfu"])
with cluster_tab:
    render_cluster_view()
with telemetry_tab:
    if st.session_state.history:
        render_telemetry_view()
    else:
        st.info("Brak danych – uruchom strumień w panelu bocznym.")

if st.session_state.streaming:
    time.sleep(STREAM_DELAY_S)
    st.rerun()
