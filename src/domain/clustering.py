import collections
import math
import time
from typing import Any

import numpy as np
from river import cluster, stream
from river.cluster.denstream import DenStreamMicroCluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.core.config import config
from src.core.logger import logger

_RIVER_CALC_RADIUS = DenStreamMicroCluster.calc_radius


def _rms_radius(self: DenStreamMicroCluster, timestamp: int) -> float:
    """Micro-cluster radius as defined by Cao et al. (2006); river's formula underestimates it
    (https://github.com/online-ml/river/issues/2004)."""
    fading = self.fading_function(timestamp - self.last_edit_time)
    weight = self._weight(fading)
    mean_sq_norm = sum(fading * ss for ss in self.squared_sum.values()) / weight
    centre_sq_norm = sum((fading * ls / weight) ** 2 for ls in self.linear_sum.values())
    variance = mean_sq_norm - centre_sq_norm
    return math.sqrt(variance) if variance > 0 else 0.0


def set_river_radius_fix(enabled: bool) -> None:
    """Switches every DenStream micro-cluster between the corrected radius
    and river's original one (kept for before/after comparisons)."""
    DenStreamMicroCluster.calc_radius = _rms_radius if enabled else _RIVER_CALC_RADIUS


set_river_radius_fix(config.denstream.fix_river_radius)


def micro_cluster_centers(model: cluster.DenStream) -> tuple[list[Any], np.ndarray]:
    """Keys and centre vectors of the model's current p-micro-clusters."""
    t = getattr(model, "timestamp", 0)
    p_mcs = getattr(model, "p_micro_clusters", {})
    keys = list(p_mcs.keys())
    centers = []
    for k in keys:
        c = p_mcs[k].calc_center(t)
        centers.append([c[dim] for dim in range(len(c))])
    return keys, np.asarray(centers, dtype=np.float64)


def group_micro_clusters(centers: np.ndarray, n_macro_clusters: int) -> np.ndarray:
    """Offline phase: average-linkage agglomerative grouping of p-micro-cluster centres into macro-clusters."""
    if len(centers) < n_macro_clusters:
        return np.arange(len(centers))
    agg = AgglomerativeClustering(n_clusters=n_macro_clusters, metric="euclidean", linkage="average")
    return agg.fit_predict(centers)


def purity_score(y_true: list[Any], y_pred: list[int]) -> float | None:
    """Share of documents belonging to the dominant true category of their
    predicted cluster. Documents predicted as noise (-1) are left out."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return None
    clusters = collections.defaultdict(list)
    for t, p in zip(y_true, y_pred):
        if p != -1:
            clusters[p].append(t)
    if not clusters:
        return 0.0
    total = sum(len(v) for v in clusters.values())
    correct = sum(collections.Counter(v).most_common(1)[0][1] for v in clusters.values())
    return correct / total


class StreamClusterer:
    def __init__(
        self,
        epsilon: float = 0.10,
        mu: int = 2,
        beta: float = 0.75,
        decaying_factor: float = 0.005,
        n_samples_init: int = 1,
        window_size: int = 300,
        expected_macro_clusters: int = 5,
        centroid_shift_lookback_batches: int = 10,
    ):
        self.model = cluster.DenStream(
            epsilon=epsilon,
            mu=mu,
            beta=beta,
            decaying_factor=decaying_factor,
            n_samples_init=n_samples_init,
        )
        self.window_size = window_size
        self.expected_macro_clusters = expected_macro_clusters

        self.window_embeddings = collections.deque(maxlen=window_size)
        self.window_macro_preds = collections.deque(maxlen=window_size)
        self.window_true_labels = collections.deque(maxlen=window_size)

        self.n_samples_seen = 0

        self.micro_to_macro = {}
        self.n_macro_clusters = 0

        # Macro-cluster centroids after each batch, for the centroid-shift signal.
        self.centroid_shift_lookback_batches = centroid_shift_lookback_batches
        self.centroid_history = collections.deque(maxlen=centroid_shift_lookback_batches + 1)

    def _cluster_offline(self, n_macro_clusters: int):
        keys, centers = micro_cluster_centers(self.model)
        labels = group_micro_clusters(centers, n_macro_clusters)
        self.micro_to_macro = {k: int(label) for k, label in zip(keys, labels)}
        self.n_macro_clusters = len(set(self.micro_to_macro.values()))
        logger.info(f"Offline phase: {len(keys)} p-micro-clusters grouped into {self.n_macro_clusters} macro-clusters")

    def predict_one(self, x: dict) -> int:
        """Macro-cluster of the nearest p-micro-cluster, or -1 if there are
        none yet. Epsilon bounds a micro-cluster's radius, not the distance of
        a single point to its centre, so it is not used as a noise threshold."""
        if not self.model.p_micro_clusters:
            return -1

        t = getattr(self.model, "timestamp", 0)
        best_mc = None
        best_dist = float("inf")

        for mc_id, mc in self.model.p_micro_clusters.items():
            c = mc.calc_center(t)
            dist = self.model._distance(x, c)
            if dist < best_dist:
                best_dist = dist
                best_mc = mc_id

        return self.micro_to_macro.get(best_mc, -1)

    def _compute_macro_centroids(self) -> dict[int, np.ndarray]:
        """Weighted mean of the p-micro-cluster centres of each macro-cluster."""
        t = getattr(self.model, "timestamp", 0)
        p_mcs = getattr(self.model, "p_micro_clusters", {})

        weighted_sums: dict[int, np.ndarray] = {}
        weight_totals: dict[int, float] = {}
        for mc_id, mc in p_mcs.items():
            macro_label = self.micro_to_macro.get(mc_id)
            if macro_label is None or macro_label == -1:
                continue
            c = mc.calc_center(t)
            if not c:
                continue
            w = mc.calc_weight(t)
            dim = max(c.keys()) + 1
            vec = np.array([c.get(i, 0.0) for i in range(dim)], dtype=np.float64)
            if macro_label not in weighted_sums:
                weighted_sums[macro_label] = np.zeros_like(vec)
                weight_totals[macro_label] = 0.0
            weighted_sums[macro_label] += vec * w
            weight_totals[macro_label] += w

        return {
            label: weighted_sums[label] / weight_totals[label] for label in weighted_sums if weight_totals[label] > 0
        }

    def get_centroid_shift(self) -> float | None:
        """Mean distance from each macro-cluster centroid to the nearest centroid from a few batches ago."""
        lookback = self.centroid_shift_lookback_batches
        if len(self.centroid_history) <= lookback:
            return None
        current = self.centroid_history[-1]
        reference = self.centroid_history[-1 - lookback]
        if not current or not reference:
            return None
        ref_vecs = list(reference.values())
        distances = [
            min(float(np.linalg.norm(cur_vec - ref_vec)) for ref_vec in ref_vecs) for cur_vec in current.values()
        ]
        return float(np.mean(distances)) if distances else None

    def update(
        self,
        embeddings: np.ndarray | list[list[float]],
        labels: list[str | int] | None = None,
    ) -> list[int]:
        """Learns a batch and returns its macro-cluster assignments."""
        if len(embeddings) == 0:
            return []

        embeddings_arr = np.asarray(embeddings, dtype=np.float32)
        start_time = time.perf_counter()
        batch_macro_preds = []

        for x, _ in stream.iter_array(embeddings_arr):
            self.model.learn_one(x)

        self._cluster_offline(n_macro_clusters=self.expected_macro_clusters)
        self.centroid_history.append(self._compute_macro_centroids())

        for i, (x, _) in enumerate(stream.iter_array(embeddings_arr)):
            macro_pred = self.predict_one(x)
            batch_macro_preds.append(macro_pred)

            self.window_embeddings.append(embeddings_arr[i])
            self.window_macro_preds.append(macro_pred)
            if labels is not None:
                self.window_true_labels.append(labels[i])

        self.n_samples_seen += len(embeddings_arr)
        end_time = time.perf_counter()
        self.last_batch_latency_ms = ((end_time - start_time) / max(1, len(embeddings_arr))) * 1000.0
        return batch_macro_preds

    def get_cluster_structures(self) -> dict[str, Any]:
        """Current micro-clusters and their macro-cluster assignment, for plotting."""
        t = getattr(self.model, "timestamp", 0)
        p_mcs = getattr(self.model, "p_micro_clusters", {})
        o_mcs = getattr(self.model, "o_micro_clusters", {})

        def extract_mc(mc_dict, map_to_macro=False):
            out = {}
            for k, mc in mc_dict.items():
                w = mc.calc_weight(t)
                c_dict = mc.calc_center(t)
                r = mc.calc_radius(t)
                mc_data = {
                    "weight": w,
                    "center": [c_dict[i] for i in range(len(c_dict))],
                    "radius": r,
                }
                if map_to_macro:
                    mc_data["macro_id"] = self.micro_to_macro.get(k, -1)
                out[k] = mc_data
            return out

        return {
            "p_micro_clusters": extract_mc(p_mcs, map_to_macro=True),
            "o_micro_clusters": extract_mc(o_mcs, map_to_macro=False),
            "macro_clusters": self.micro_to_macro,
        }

    def get_metrics(self) -> dict:
        t = getattr(self.model, "timestamp", 0)
        p_mcs = getattr(self.model, "p_micro_clusters", {})
        o_mcs = getattr(self.model, "o_micro_clusters", {})
        n_p_mc = len(p_mcs)
        n_o_mc = len(o_mcs)
        n_noise_in_window = sum(1 for p in self.window_macro_preds if p == -1)

        # Share of the total micro-cluster weight held by o-micro-clusters.
        p_weight = sum(mc.calc_weight(t) for mc in p_mcs.values())
        o_weight = sum(mc.calc_weight(t) for mc in o_mcs.values())
        total_weight = p_weight + o_weight
        outlier_ratio = (o_weight / total_weight) if total_weight > 0 else 0.0

        metrics_dict = {
            "n_micro_clusters": n_p_mc,
            "n_outlier_clusters": n_o_mc,
            "outlier_ratio": outlier_ratio,
            "centroid_shift": self.get_centroid_shift(),
            "latency_ms_per_doc": getattr(self, "last_batch_latency_ms", 0.0),
            "n_macro_clusters": getattr(self, "n_macro_clusters", 0),
            "n_noise_pmcs": n_o_mc,
            "n_samples_seen": getattr(self, "n_samples_seen", 0),
            "n_noise_docs_window": n_noise_in_window,
            "purity": np.nan,
            "silhouette": np.nan,
            "ari": 0.0,
            "nmi": 0.0,
        }

        if len(self.window_embeddings) >= 10:
            X_valid = []
            y_valid = []
            for x, y in zip(self.window_embeddings, self.window_macro_preds):
                if y != -1:
                    X_valid.append(x)
                    y_valid.append(y)

            if len(set(y_valid)) > 1:
                try:
                    metrics_dict["silhouette"] = round(float(silhouette_score(X_valid, y_valid)), 4)
                except Exception:
                    pass

        if len(self.window_true_labels) == len(self.window_macro_preds) and len(self.window_true_labels) >= 10:
            y_true = list(self.window_true_labels)
            y_pred = list(self.window_macro_preds)
            purity = purity_score(y_true, y_pred)
            if purity is not None:
                metrics_dict["purity"] = round(purity, 4)
            try:
                metrics_dict["ari"] = round(float(adjusted_rand_score(y_true, y_pred)), 4)
            except Exception:
                pass
            try:
                metrics_dict["nmi"] = round(float(normalized_mutual_info_score(y_true, y_pred)), 4)
            except Exception:
                pass

        return metrics_dict

    def ease_decaying_factor(self, target: float, rate: float = 0.9) -> None:
        """Lowers a decaying factor raised by a swap back towards `target`,
        by `rate` per batch."""
        current = self.model.decaying_factor
        if current > target:
            self.model.decaying_factor = max(target, current * rate)

    def warm_start(self, embeddings: np.ndarray | list[list[float]]) -> None:
        """Trains the model on a buffer before it serves the stream, at the
        start of the stream and after every swap. The buffer is not scored."""
        for x, _ in stream.iter_array(np.asarray(embeddings, dtype=np.float32)):
            self.model.learn_one(x)
        self._cluster_offline(n_macro_clusters=self.expected_macro_clusters)

        self.window_embeddings.clear()
        self.window_macro_preds.clear()
        self.window_true_labels.clear()
        self.n_samples_seen = 0
        # Start the centroid history from this model. Comparing with the old model's centroids
        # (a different model in a different IPCA space) would show the swap itself as a shift.
        self.centroid_history.clear()
        self.centroid_history.append(self._compute_macro_centroids())

    def hot_swap_model(
        self,
        new_params: dict[str, Any],
        window_data: np.ndarray | list[list[float]],
    ):
        eps = float(new_params.get("epsilon", 0.10))
        mu = max(int(new_params.get("mu", 2)), 2)
        decay = float(new_params.get("decaying_factor", 0.005))

        self.model = cluster.DenStream(
            epsilon=eps,
            mu=mu,
            decaying_factor=decay,
            beta=0.75,
            n_samples_init=1,
        )
        self.warm_start(window_data)

        logger.info(
            f"Model swapped: epsilon={eps}, mu={mu}, decay={decay} | "
            f"p-micro-clusters: {len(self.model.p_micro_clusters)}, macro-clusters: {self.n_macro_clusters}"
        )
