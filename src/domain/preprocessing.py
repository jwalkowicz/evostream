import re
import string
from typing import List, Optional

import numpy as np
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import IncrementalPCA


class TextPreprocessor:
    """Handles low-level text cleaning and normalization."""

    def clean(self, text: str) -> str:
        """Removes HTML, URLs, punctuation, and normalizes whitespace."""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\W+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Cleans a batch of texts."""
        return [self.clean(t) for t in texts]


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for text semantic encoding via SBERT and optional
    incremental dimensionality reduction via IncrementalPCA.
    """

    def __init__(self, encoder, pca: Optional[IncrementalPCA] = None):
        """
        Args:
            encoder: A sentence encoding model (e.g., SentenceTransformer).
            pca: An optional IncrementalPCA model for online projection.
        """
        self.encoder = encoder
        self.pca = pca
        self.warmup_buffer: List[np.ndarray] = []
        self._is_pca_fitted = False

    @property
    def output_dim(self) -> int:
        """Returns the dimensionality of the transformed representations."""
        if self.pca is not None:
            return self.pca.n_components
        if hasattr(self.encoder, "get_sentence_embedding_dimension"):
            return self.encoder.get_sentence_embedding_dimension()
        return -1

    def fit(self, X, y=None):
        return self

    def _update_pca_with_buffer(self, embeddings: np.ndarray):
        """Buffers embeddings until reaching n_components before the first partial_fit."""
        if self.pca is None:
            return

        if not self._is_pca_fitted:
            self.warmup_buffer.extend(embeddings)
            if len(self.warmup_buffer) >= self.pca.n_components:
                buffer_array = np.array(self.warmup_buffer)
                self.pca.partial_fit(buffer_array)
                self._is_pca_fitted = True
                self.warmup_buffer.clear()
        else:
            if len(embeddings) >= self.pca.n_components:
                self.pca.partial_fit(embeddings)

    def transform(self, X: List[str]) -> np.ndarray:
        """Encodes texts to dense embeddings and applies PCA if configured."""
        if not X:
            return np.empty((0, self.output_dim))

        embeddings = self.encoder.encode(
            X, show_progress_bar=False, convert_to_numpy=True
        )
        if self.pca is not None:
            if not self._is_pca_fitted:
                self._update_pca_with_buffer(embeddings)
                if not self._is_pca_fitted:
                    return embeddings[:, : self.pca.n_components]
            return self.pca.transform(embeddings)

        return embeddings

    def fit_transform(self, X: List[str], y=None, **fit_params) -> np.ndarray:
        """Encodes texts, incrementally updates PCA, and returns transformed embeddings."""
        if not X:
            return np.empty((0, self.output_dim))

        embeddings = self.encoder.encode(
            X, show_progress_bar=False, convert_to_numpy=True
        )
        if self.pca is not None:
            self._update_pca_with_buffer(embeddings)
            if self._is_pca_fitted:
                return self.pca.transform(embeddings)
            return embeddings[:, : self.pca.n_components]

        return embeddings
