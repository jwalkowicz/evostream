import re
import string
from typing import List, Optional

import numpy as np
from bs4 import BeautifulSoup
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize


class TextPreprocessor:
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
        return [self.clean(t) for t in texts]


class StreamProjector:
    """IPCA projection of SBERT embeddings, fitted on a buffer and then
    frozen until the next model swap (refitting it continuously would rotate
    the space under existing micro-clusters). Outputs are L2-normalised.
    n_components=None passes the embeddings through unchanged."""

    def __init__(self, n_components: Optional[int]):
        self.n_components = n_components
        self.ipca: Optional[IncrementalPCA] = None

    @property
    def is_fitted(self) -> bool:
        return self.n_components is None or self.ipca is not None

    def fit(self, embeddings: np.ndarray) -> None:
        """Fits a fresh IPCA on the given embeddings, replacing any previous one."""
        if self.n_components is None:
            return
        ipca = IncrementalPCA(n_components=self.n_components)
        ipca.partial_fit(np.asarray(embeddings))
        self.ipca = ipca

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = np.asarray(embeddings)
        if self.n_components is None:
            return normalize(embeddings)
        if self.ipca is None:
            raise RuntimeError("StreamProjector.fit() must be called before transform().")
        return normalize(self.ipca.transform(embeddings))
