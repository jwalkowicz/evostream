"""
Thesis 2 experiment: static DenStream vs. EvoStream (NSGA-II hot-swap) under
an abrupt concept drift on 20-newsgroups text streams.

Runs a single (initial_eps, initial_decay) configuration end-to-end and saves
the per-batch metrics to CSV. Plotting lives in plot_thesis_2_drift.py, which
consumes these CSVs across a full parameter sweep (see run_sweep.py).
"""

import argparse
import copy
import os
import pickle
import random
from typing import List, Optional, Tuple

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

# Post-swap decay cooldown-easing behavior. Three variants under test:
#   "asymmetric_own" (baseline/default): ease decay back down toward THIS
#       RUN's own initial_decay, only when the evolved value is above it.
#       Never eases back up when the evolved value is already below it.
#       Only makes sense if initial_decay represents a sensible target - for
#       sweep combos where initial_decay is a deliberate stress-test value
#       (e.g. 0.08), "return to it" isn't a meaningful goal.
#   "none": don't ease at all - let NSGA-II's evolved decay stand until the
#       next swap. Doesn't second-guess the optimizer with an arbitrary target.
#   "anchor_default": ease toward config.denstream.decaying_factor (the
#       fixed practitioner baseline) instead of each combo's own sweep
#       starting point, so "cooldown" always means "return to a sensible
#       value," even for combos that deliberately started elsewhere.
# Set via env var so the same code can be re-run under each variant without
# hand-editing, and each variant's results are written to separate files
# (see RESULT_SUFFIX below) so the validated baseline sweep is never overwritten.
COOLDOWN_MODE = os.environ.get("COOLDOWN_MODE", "asymmetric_own")

# 6 categories per phase: a semantically disjoint topic set switches in at
# DRIFT_POINT, simulating an abrupt concept drift.
PHASE1_CATEGORIES = [
    "sci.space", "sci.med", "rec.autos",
    "sci.electronics", "comp.graphics", "sci.crypt",
]
PHASE2_CATEGORIES = [
    "rec.sport.baseball", "comp.sys.ibm.pc.hardware", "talk.politics.mideast",
    "rec.sport.hockey", "rec.motorcycles", "talk.politics.guns",
]
SAMPLES_PER_PHASE = 5000
DRIFT_POINT = SAMPLES_PER_PHASE
BATCH_SIZE = config.ml.batch_size
# Two separate buffer sizes that used to share one constant:
# - INITIAL_WARMUP_SIZE: documents sacrificed at stream start to fit the
#   first IPCA, before either model starts clustering at all.
# - HOTSWAP_BUFFER_SIZE: documents collected after a drift detection before
#   the new IPCA + DenStream can be retrained. Larger means a more
#   representative retrain sample (less overfit to one narrow slice of
#   post-drift content) but also a longer detection-to-deployment lag where
#   the OLD model keeps running, and slower NSGA-II evaluation (each of its
#   384 evaluations re-fits DenStream over the whole buffer). Kept separate
#   so tuning one doesn't also change the unrelated initial cold-start.
INITIAL_WARMUP_SIZE = config.ml.ipca_warmup_size
# Validated: buf=500 vs the original buf=300 gave identical pre-swap behavior
# (buffer size doesn't affect anything before the first swap) but, once
# triggered, cut ε=0.075's re-trigger count from 5 to 3 and roughly tripled
# the interval between later re-triggers (1100-1450 docs -> 3450 docs) at
# effectively no cost to final purity (0.660 -> 0.651) - promoted to the
# default. Still overridable for isolated comparisons against the old value.
HOTSWAP_BUFFER_SIZE = int(os.environ.get("HOTSWAP_BUFFER_SIZE", config.evolution.hotswap_buffer_size))

# Warmup/pretraining strategy for both IPCA and DenStream:
#   "sequential_prefix" (default): use the first INITIAL_WARMUP_SIZE
#       documents of the stream, in order, to fit IPCA. DenStream is NOT
#       pretrained at all - it sees these documents for the first time right
#       when real per-batch tracking begins (start_batch skips ahead past
#       them), so it's deployed with zero micro-clusters at t=INITIAL_WARMUP_SIZE.
#   "random_sample_pretrain": randomly sample INITIAL_WARMUP_SIZE documents
#       from anywhere in phase 1 (not just the first N) to fit IPCA AND to
#       pretrain DenStream - fed through the same batch-by-batch update()
#       calls real traffic uses (not one giant dump, which was tried first
#       and made things worse: a single 300-point update() call produced 14
#       small, immature micro-clusters at once and measurably hurt early
#       purity/silhouette vs. no pretraining at all), just without
#       recording/scoring those batches. The main loop then processes the
#       ENTIRE sequential stream from document 0 - a few documents may have
#       already been seen during the random pretraining sample, which a
#       streaming model handles the same as any repeated exposure.
WARMUP_STRATEGY = os.environ.get("WARMUP_STRATEGY", "sequential_prefix")

# Built after all env-var-driven settings above so a run under non-default
# settings never overwrites the validated baseline sweep's result files.
RESULT_SUFFIX = ""
if COOLDOWN_MODE != "asymmetric_own":
    RESULT_SUFFIX += f"__{COOLDOWN_MODE}"
if HOTSWAP_BUFFER_SIZE != config.evolution.hotswap_buffer_size:
    RESULT_SUFFIX += f"__buf{HOTSWAP_BUFFER_SIZE}"
if WARMUP_STRATEGY != "sequential_prefix":
    RESULT_SUFFIX += f"__{WARMUP_STRATEGY}"

# Sweep grid: 5 epsilon x 5 decay = 25 combinations, spanning the full
# NSGA-II search space defined in config.evolution.param_bounds - epsilon
# [0.05, 0.40], decaying_factor [0.005, 0.08] - and staying within those
# bounds on both ends for both parameters (not below, not above). The
# epsilon values match the thesis 1 grid inside these bounds.
SWEEP_EPSILON_VALUES = [0.05, 0.10, 0.20, 0.30, 0.40]
SWEEP_DECAY_VALUES = [0.005, 0.02, 0.04, 0.06, 0.08]
PCA_COMPONENTS = 16


def create_dataset_stream(
    phase1_categories: List[str],
    phase2_categories: List[str],
    samples_per_phase: int = SAMPLES_PER_PHASE,
) -> Tuple[List[str], List[str]]:
    preprocessor = TextPreprocessor()

    def load_cat_data(cats, count):
        raw = fetch_20newsgroups(
            subset="all", categories=cats, remove=("headers", "footers", "quotes")
        )
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


def _load_or_build_dataset() -> Tuple[List[str], List[str]]:
    """Text cleaning is identical across every (eps, decay) combo in a sweep,
    so cache it once instead of re-running BeautifulSoup over 10k docs per run.
    """
    if os.path.exists(DATASET_CACHE_PATH):
        with open(DATASET_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    texts, labels = create_dataset_stream(
        PHASE1_CATEGORIES, PHASE2_CATEGORIES, SAMPLES_PER_PHASE
    )
    with open(DATASET_CACHE_PATH, "wb") as f:
        pickle.dump((texts, labels), f)
    return texts, labels


def _load_or_compute_embeddings(texts: List[str]) -> np.ndarray:
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        logger.info("Loading cached SBERT embeddings...")
        return np.load(EMBEDDINGS_CACHE_PATH)

    logger.info("Pre-computing SBERT embeddings for the concatenated stream...")
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    embeddings = encoder.encode(
        texts, batch_size=128, device="cpu", normalize_embeddings=True
    )
    np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    return embeddings


def run_drift_experiment(
    initial_eps: float = 0.10, initial_decay: float = 0.005
) -> pd.DataFrame:
    set_seed(42)

    assert len(PHASE1_CATEGORIES) == len(PHASE2_CATEGORIES), (
        "Phase category lists must match in length so the true number of "
        "macro-clusters stays constant across the drift."
    )
    n_expected_macro = len(PHASE1_CATEGORIES)

    texts, labels_abrupt = _load_or_build_dataset()
    vecs_abrupt_raw = _load_or_compute_embeddings(texts)

    n_batches = len(vecs_abrupt_raw) // BATCH_SIZE

    if WARMUP_STRATEGY == "random_sample_pretrain":
        # A random sample spanning all of phase 1, not just whichever
        # documents happen to be shuffled first, to fit IPCA and pretrain
        # DenStream. The main loop then processes the entire sequential
        # stream from document 0 - warmup no longer consumes a fixed prefix.
        warmup_rng = np.random.default_rng(42)
        warmup_indices = np.sort(
            warmup_rng.choice(SAMPLES_PER_PHASE, size=INITIAL_WARMUP_SIZE, replace=False)
        )
        warmup_raw = vecs_abrupt_raw[warmup_indices]
        warmup_labels = [labels_abrupt[i] for i in warmup_indices]
        start_batch = 0
        logger.info(
            f"Randomly sampling {INITIAL_WARMUP_SIZE} documents from across phase 1 "
            "to train initial IPCA..."
        )
    else:
        warmup_raw = vecs_abrupt_raw[:INITIAL_WARMUP_SIZE]
        warmup_labels = labels_abrupt[:INITIAL_WARMUP_SIZE]
        start_batch = INITIAL_WARMUP_SIZE // BATCH_SIZE
        logger.info(f"Sacrificing first {INITIAL_WARMUP_SIZE} documents to train initial IPCA...")

    ipca_static = IncrementalPCA(n_components=PCA_COMPONENTS)
    ipca_static.partial_fit(warmup_raw)
    ipca_hotswap = copy.deepcopy(ipca_static)

    # Both static and hot-swap start from the SAME swept (initial_eps,
    # initial_decay) for this run - the grid tests both models across the
    # full range of starting combinations, not just the hot-swap side against
    # one fixed reference.
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

    if WARMUP_STRATEGY == "random_sample_pretrain":
        # Same batch-by-batch update() calls real traffic uses, not one
        # giant dump (see WARMUP_STRATEGY docstring above for why that
        # matters), just without recording/scoring these batches.
        for i in range(0, len(warmup_raw), BATCH_SIZE):
            chunk_raw = warmup_raw[i : i + BATCH_SIZE]
            chunk_labels = warmup_labels[i : i + BATCH_SIZE]
            c_static.update(normalize(ipca_static.transform(chunk_raw)), labels=chunk_labels)
            c_hotswap.update(normalize(ipca_hotswap.transform(chunk_raw)), labels=chunk_labels)
        logger.info(
            f"Pretrained both DenStream instances on {len(warmup_raw)} randomly-sampled "
            f"phase-1 documents, in {BATCH_SIZE}-doc chunks."
        )

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
    hotswap_buffer_collected_raw: List[np.ndarray] = []
    # The hot-swap is triggered exclusively by the detector's own signals
    # (quality drop / centroid shift) - no hardcoded `curr_idx == DRIFT_POINT`
    # fallback. DRIFT_POINT is only the point at which the underlying data
    # stream switches topics (ground truth for evaluation/plotting); the
    # detector doesn't get to see it, so this is genuinely blind detection.
    trigger_sample_idx: Optional[int] = None
    # Sample index where the first hot-swap actually COMPLETES (new model
    # deployed), as opposed to trigger_sample_idx (when drift was first
    # detected). Between the two, the hot-swap model is still architecturally
    # identical to the static one - it needs HOTSWAP_BUFFER_SIZE fresh post-drift
    # documents before it can retrain, so it necessarily performs just as
    # badly as static during that window. That's a real, unavoidable
    # detection-to-deployment lag, not the new model underperforming.
    swap_sample_idx: Optional[int] = None

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
                logger.info(
                    "Drift buffer full! Initiating Full Pipeline Hot-Swap (IPCA + DenStream)..."
                )

                # 1. Re-fit the feature extractor on the new concept.
                new_ipca = IncrementalPCA(n_components=PCA_COMPONENTS)
                new_ipca.partial_fit(np.array(hotswap_buffer_collected_raw))
                new_buffer_proj = normalize(
                    new_ipca.transform(np.array(hotswap_buffer_collected_raw))
                )

                # 2. Evolve DenStream params for the new embedding space.
                compromise, _, _ = opt_hotswap.evolve(data_buffer=new_buffer_proj)

                # 3. Swap in the newly-evolved DenStream instance.
                c_hotswap.hot_swap_model(
                    new_params=compromise.params,
                    window_data=new_buffer_proj,
                )

                # 4. Swap in the newly-fitted feature extractor.
                ipca_hotswap = new_ipca

                m_hot = c_hotswap.get_metrics()
                collecting_for_hotswap = False
                hotswap_buffer_collected_raw = []
                if swap_sample_idx is None:
                    swap_sample_idx = curr_idx

        # Cooldown: see COOLDOWN_MODE above for the three variants under test.
        if not collecting_for_hotswap and COOLDOWN_MODE != "none":
            anchor = initial_decay if COOLDOWN_MODE == "asymmetric_own" else config.denstream.decaying_factor
            curr_decay = getattr(c_hotswap.model, "decaying_factor", anchor)
            if curr_decay > anchor:
                c_hotswap.model.decaying_factor = max(anchor, curr_decay * 0.90)

        records.append(
            {
                "sample_idx": curr_idx,
                "static_purity": m_stat["purity"],
                "hotswap_purity": m_hot["purity"],
                "static_silhouette": m_stat["silhouette"],
                "hotswap_silhouette": m_hot["silhouette"],
                "outlier_ratio": m_hot["outlier_ratio"],
                "centroid_shift": m_hot["centroid_shift"],
                "eps_adapted": getattr(c_hotswap.model, "epsilon", initial_eps),
                "decay_adapted": getattr(c_hotswap.model, "decaying_factor", initial_decay),
                "static_latency_ms": m_stat["latency_ms_per_doc"],
                "hotswap_latency_ms": m_hot["latency_ms_per_doc"],
                "static_micro": m_stat["n_micro_clusters"],
                "hotswap_micro": m_hot["n_micro_clusters"],
                "drift_detected": is_d_hot,
            }
        )

    if trigger_sample_idx is None:
        logger.warning(
            "Detector never fired during the whole stream - no hot-swap happened this run."
        )

    df = pd.DataFrame(records)
    df["trigger_sample_idx"] = trigger_sample_idx
    df["swap_sample_idx"] = swap_sample_idx
    out_path = f"{RESULTS_DIR}/thesis_2_drift_results_eps_{initial_eps}_decay_{initial_decay}{RESULT_SUFFIX}.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"Saved results to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Static vs. hot-swap DenStream drift-adaptation experiment."
    )
    parser.add_argument("--eps", type=float, default=0.10)
    parser.add_argument("--decay", type=float, default=0.005)
    args = parser.parse_args()

    logger.add("logs/thesis_2_drift.log", rotation="500 MB")
    run_drift_experiment(initial_eps=args.eps, initial_decay=args.decay)
