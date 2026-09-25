# Thesis 2 experiment: static vs. adaptive DenStream on a stream with an abrupt topic change

import argparse
import copy
import os
import pickle
import random

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from src.core.common import set_seed
from src.core.config import config
from src.core.logger import logger
from src.domain.clustering import StreamClusterer
from src.domain.drift import UnsupervisedDriftDetector
from src.domain.evolution import NSGAIIOptimizer
from src.domain.preprocessing import TextPreprocessor

RESULTS_DIR = "experiments/theses/thesis_2/results"
DATASET_CACHE_PATH = f"{RESULTS_DIR}/cached_dataset.pkl"
EMBEDDINGS_CACHE_PATH = f"{RESULTS_DIR}/cached_embeddings.npy"

# Two disjoint sets of six categories
PHASE1_CATEGORIES = [
    "sci.space",
    "sci.med",
    "rec.autos",
    "sci.electronics",
    "comp.graphics",
    "sci.crypt",
]
PHASE2_CATEGORIES = [
    "rec.sport.baseball",
    "comp.sys.ibm.pc.hardware",
    "talk.politics.mideast",
    "rec.sport.hockey",
    "rec.motorcycles",
    "talk.politics.guns",
]
SAMPLES_PER_PHASE = 5000
DRIFT_POINT = SAMPLES_PER_PHASE
BATCH_SIZE = config.ml.batch_size
INITIAL_WARMUP_SIZE = config.ml.ipca_warmup_size
HOTSWAP_BUFFER_SIZE = config.evolution.hotswap_buffer_size
PCA_COMPONENTS = 16
LAMBDA_EASING_RATE = 0.90

# Starting parameters: a 5 x 5 grid over the NSGA-II search space.
SWEEP_EPSILON_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40]
SWEEP_DECAY_VALUES = [0.005, 0.02, 0.04, 0.06, 0.08]


def create_dataset_stream(
    phase1_categories: list[str],
    phase2_categories: list[str],
    samples_per_phase: int = SAMPLES_PER_PHASE,
) -> tuple[list[str], list[str]]:
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
    all_pairs = p1_pairs + p2_pairs

    texts = [p[0] for p in all_pairs]
    labels = [p[1] for p in all_pairs]
    return texts, labels


def _load_or_build_dataset() -> tuple[list[str], list[str]]:
    if os.path.exists(DATASET_CACHE_PATH):
        with open(DATASET_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    texts, labels = create_dataset_stream(PHASE1_CATEGORIES, PHASE2_CATEGORIES, SAMPLES_PER_PHASE)
    with open(DATASET_CACHE_PATH, "wb") as f:
        pickle.dump((texts, labels), f)
    return texts, labels


def _load_or_compute_embeddings(texts: list[str]) -> np.ndarray:
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        logger.info("Loading cached SBERT embeddings...")
        return np.load(EMBEDDINGS_CACHE_PATH)

    logger.info("Pre-computing SBERT embeddings for the concatenated stream...")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = encoder.encode(texts, batch_size=128, device="cpu", normalize_embeddings=True)
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    return embeddings


def run_drift_experiment(initial_eps: float = 0.10, initial_decay: float = 0.005) -> pd.DataFrame:
    set_seed(42)

    assert len(PHASE1_CATEGORIES) == len(PHASE2_CATEGORIES)
    n_expected_macro = len(PHASE1_CATEGORIES)

    texts, labels_abrupt = _load_or_build_dataset()
    vecs_abrupt_raw = _load_or_compute_embeddings(texts)

    n_batches = len(vecs_abrupt_raw) // BATCH_SIZE

    warmup_raw = vecs_abrupt_raw[:INITIAL_WARMUP_SIZE]
    start_batch = INITIAL_WARMUP_SIZE // BATCH_SIZE
    logger.info(f"Using the first {INITIAL_WARMUP_SIZE} documents to fit IPCA and warm-start DenStream...")

    ipca_static = IncrementalPCA(n_components=PCA_COMPONENTS)
    ipca_static.partial_fit(warmup_raw)
    ipca_hotswap = copy.deepcopy(ipca_static)

    common_kwargs = dict(
        epsilon=initial_eps,
        decaying_factor=initial_decay,
        mu=config.denstream.mu,
        n_samples_init=config.denstream.n_samples_init,
        window_size=config.denstream.window_size,
        expected_macro_clusters=n_expected_macro,
    )
    c_static = StreamClusterer(**common_kwargs)
    c_hotswap = StreamClusterer(**common_kwargs)

    warmup_proj = normalize(ipca_static.transform(warmup_raw))
    c_static.warm_start(warmup_proj)
    c_hotswap.warm_start(warmup_proj)

    opt_hotswap = NSGAIIOptimizer(
        n_macro_clusters=n_expected_macro,
        population_size=config.evolution.population_size,
        generations=config.evolution.generations,
        crossover_rate=config.evolution.crossover_rate,
        mutation_rate=config.evolution.mutation_rate,
        param_bounds=config.evolution.param_bounds,
        fixed_mu=config.denstream.mu,
        n_samples_init=config.denstream.n_samples_init,
    )

    drift_det_hotswap = UnsupervisedDriftDetector(
        window_size=config.drift.window_size,
        min_warmup_steps=config.drift.min_warmup_steps,
        quality_drop_sigma=config.drift.quality_drop_sigma,
        cooldown_steps=config.drift.cooldown_steps,
        consecutive_drops_required=config.drift.consecutive_drops_required,
        centroid_shift_threshold=config.drift.centroid_shift_threshold,
        quality_absolute_floor=config.drift.quality_absolute_floor,
        centroid_shift_min_warmup_steps=config.drift.centroid_shift_min_warmup_steps,
    )

    records = []
    collecting_for_hotswap = False
    hotswap_buffer_collected_raw: list[np.ndarray] = []

    trigger_sample_idx: int | None = None  # first alarm
    swap_sample_idx: int | None = None  # first deployed replacement model

    for b in range(start_batch, n_batches):
        s_i = b * BATCH_SIZE
        e_i = s_i + BATCH_SIZE
        b_raw = vecs_abrupt_raw[s_i:e_i]
        b_lbls = labels_abrupt[s_i:e_i]
        curr_idx = e_i

        b_vecs_static = normalize(ipca_static.transform(b_raw))
        b_vecs_hot = normalize(ipca_hotswap.transform(b_raw))

        c_static.update(b_vecs_static, labels=b_lbls)
        m_stat = c_static.get_metrics()

        c_hotswap.update(b_vecs_hot, labels=b_lbls)
        m_hot = c_hotswap.get_metrics()

        is_d_hot = drift_det_hotswap.update(
            current_silhouette=m_hot["silhouette"] or 0.0,
            centroid_shift=m_hot["centroid_shift"],
        )

        if is_d_hot and not collecting_for_hotswap:
            collecting_for_hotswap = True
            hotswap_buffer_collected_raw = []
            if trigger_sample_idx is None:
                trigger_sample_idx = curr_idx
                logger.info(f"First hot-swap triggered by detector at sample_idx={curr_idx}")

        if collecting_for_hotswap:
            hotswap_buffer_collected_raw.extend(b_raw)
            if len(hotswap_buffer_collected_raw) >= HOTSWAP_BUFFER_SIZE:
                logger.info("Swap buffer full, refitting IPCA and optimising DenStream parameters")
                new_ipca = IncrementalPCA(n_components=PCA_COMPONENTS)
                new_ipca.partial_fit(np.array(hotswap_buffer_collected_raw))
                new_buffer_proj = normalize(new_ipca.transform(np.array(hotswap_buffer_collected_raw)))
                compromise, _, _ = opt_hotswap.evolve(data_buffer=new_buffer_proj)
                c_hotswap.hot_swap_model(new_params=compromise.params, window_data=new_buffer_proj)
                ipca_hotswap = new_ipca

                m_hot = c_hotswap.get_metrics()
                collecting_for_hotswap = False
                hotswap_buffer_collected_raw = []
                if swap_sample_idx is None:
                    swap_sample_idx = curr_idx

        # After a swap, a lambda above the starting value is eased back to it.
        if not collecting_for_hotswap:
            curr_decay = c_hotswap.model.decaying_factor
            if curr_decay > initial_decay:
                c_hotswap.model.decaying_factor = max(initial_decay, curr_decay * LAMBDA_EASING_RATE)

        records.append(
            {
                "sample_idx": curr_idx,
                "static_purity": m_stat["purity"],
                "hotswap_purity": m_hot["purity"],
                "static_silhouette": m_stat["silhouette"],
                "hotswap_silhouette": m_hot["silhouette"],
                "outlier_ratio": m_hot["outlier_ratio"],
                "centroid_shift": m_hot["centroid_shift"],
                "eps_adapted": c_hotswap.model.epsilon,
                "decay_adapted": c_hotswap.model.decaying_factor,
                "static_latency_ms": m_stat["latency_ms_per_doc"],
                "hotswap_latency_ms": m_hot["latency_ms_per_doc"],
                "static_micro": m_stat["n_micro_clusters"],
                "hotswap_micro": m_hot["n_micro_clusters"],
                "drift_detected": is_d_hot,
            }
        )

    if trigger_sample_idx is None:
        logger.warning("The detector never fired, no swap happened in this run.")

    df = pd.DataFrame(records)
    df["trigger_sample_idx"] = trigger_sample_idx
    df["swap_sample_idx"] = swap_sample_idx
    out_path = f"{RESULTS_DIR}/thesis_2_drift_results_eps_{initial_eps}_decay_{initial_decay}.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"Saved results to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Static vs. hot-swap DenStream drift-adaptation experiment.")
    parser.add_argument("--eps", type=float, default=0.10)
    parser.add_argument("--decay", type=float, default=0.005)
    args = parser.parse_args()

    logger.add("logs/thesis_2_drift.log", rotation="500 MB")
    run_drift_experiment(initial_eps=args.eps, initial_decay=args.decay)
