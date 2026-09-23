"""
Validation stream for choosing the NSGA-II search-space bounds (thesis
section 3.4).

It is built only from the eight 20 Newsgroups categories that no thesis
experiment uses, so the bounds are chosen on data completely separate from
the data the theses are evaluated on. Two groups of four categories form an
abrupt topic change after VALIDATION_SAMPLES_PER_PHASE documents; the text
cleaning, filtering, shuffling and SBERT encoding are exactly the same as
for the thesis stream.
"""

import os
import pickle
from typing import List, Tuple

import numpy as np

from experiments.theses.thesis_2.exp_thesis_2_drift import create_dataset_stream
from src.core.config import config

VALIDATION_PHASE1_CATEGORIES = [
    "comp.windows.x", "misc.forsale", "soc.religion.christian", "talk.politics.misc",
]
VALIDATION_PHASE2_CATEGORIES = [
    "comp.os.ms-windows.misc", "comp.sys.mac.hardware", "alt.atheism", "talk.religion.misc",
]
VALIDATION_SAMPLES_PER_PHASE = 3000

CACHE_DIR = "experiments/param_bounds/common"
DATASET_CACHE_PATH = f"{CACHE_DIR}/validation_dataset.pkl"
EMBEDDINGS_CACHE_PATH = f"{CACHE_DIR}/validation_embeddings.npy"


def load_validation_stream() -> Tuple[np.ndarray, List[str]]:
    """Both phases of the validation stream (SBERT embeddings and category
    labels), cached after the first call."""
    if os.path.exists(DATASET_CACHE_PATH):
        with open(DATASET_CACHE_PATH, "rb") as f:
            texts, labels = pickle.load(f)
    else:
        texts, labels = create_dataset_stream(
            VALIDATION_PHASE1_CATEGORIES, VALIDATION_PHASE2_CATEGORIES, VALIDATION_SAMPLES_PER_PHASE
        )
        with open(DATASET_CACHE_PATH, "wb") as f:
            pickle.dump((texts, labels), f)

    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        embeddings = np.load(EMBEDDINGS_CACHE_PATH)
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(config.ml.embedding_model, device="cpu")
        embeddings = encoder.encode(texts, batch_size=128, device="cpu", normalize_embeddings=True)
        np.save(EMBEDDINGS_CACHE_PATH, embeddings)
    return embeddings, labels
