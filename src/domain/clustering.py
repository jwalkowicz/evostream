import collections
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
from river import cluster, stream
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from src.core.logger import logger


class StreamClusterer:
    def __init__(
        self,
        epsilon: float = 0.10,
        mu: int = 2,
        beta: float = 0.75,
        decaying_factor: float = 0.005,
        n_samples_init: int = 1,
        window_size: int = 300,
    ):
        self.model = cluster.DenStream(
            epsilon=epsilon,
            mu=mu,
            beta=beta,
            decaying_factor=decaying_factor,
            n_samples_init=n_samples_init,
        )
        self.window_size = window_size

        self.window_embeddings = collections.deque(maxlen=window_size)
        self.window_macro_preds = collections.deque(maxlen=window_size)
        self.window_true_labels = collections.deque(maxlen=window_size)

        self.n_samples_seen = 0
        self.last_batch_noise_ratio = 0.0

    def update(
        self,
        embeddings: Union[np.ndarray, List[List[float]]],
        labels: Optional[List[Union[str, int]]] = None,
    ) -> List[int]:
        """
        Updates the model with new embeddings and returns macro-cluster assignments.
        """
        if len(embeddings) == 0:
            return []

        embeddings_arr = np.asarray(embeddings, dtype=np.float32)
        start_time = time.perf_counter()
        batch_macro_preds = []
        noise_count = 0

        for i, (x, _) in enumerate(stream.iter_array(embeddings_arr)):
            self.model.learn_one(x)
            macro_pred = self.model.predict_one(x)

            if macro_pred == -1:
                noise_count += 1

            batch_macro_preds.append(macro_pred)

            self.window_embeddings.append(embeddings_arr[i])
            self.window_macro_preds.append(macro_pred)
            if labels is not None:
                self.window_true_labels.append(labels[i])

        self.n_samples_seen += len(embeddings_arr)
        self.last_batch_noise_ratio = noise_count / len(embeddings_arr)
        end_time = time.perf_counter()
        self.last_batch_latency_ms = (
            (end_time - start_time) / max(1, len(embeddings_arr))
        ) * 1000.0
        return batch_macro_preds

    def get_cluster_structures(self) -> Dict[str, Any]:
        """
        Returns a physical dump of the model's memory: current positions of p-micro-clusters,
        o-micro-clusters (noise), and their mapping to macro-clusters.
        """
        t = getattr(self.model, "timestamp", 0)
        p_mcs = getattr(self.model, "p_micro_clusters", {})
        o_mcs = getattr(self.model, "o_micro_clusters", {})

        micro_to_macro = {}
        if hasattr(self.model, "clusters") and self.model.clusters:
            for k, mc in p_mcs.items():
                try:
                    c_dict = mc.calc_center(t)
                    macro_id = self.model._get_closest_cluster_key(c_dict, self.model.clusters)
                    micro_to_macro[k] = macro_id
                except Exception:
                    micro_to_macro[k] = -1

        def extract_mc(mc_dict):
            out = {}
            for k, mc in mc_dict.items():
                w = mc.calc_weight(t)
                c_dict = mc.calc_center(t)
                r = mc.calc_radius(t)
                out[k] = {
                    "weight": w,
                    "center": [c_dict[i] for i in range(len(c_dict))],
                    "radius": r,
                    "macro_id": micro_to_macro.get(k, -1),
                }
            return out

        return {
            "p_micro_clusters": extract_mc(p_mcs),
            "o_micro_clusters": extract_mc(o_mcs),
            "macro_clusters": micro_to_macro,
        }

    def _calculate_purity(
        self, y_true: List[Any], y_pred: List[int]
    ) -> Optional[float]:
        if not y_true or not y_pred or len(y_true) != len(y_pred):
            return None
        clusters = collections.defaultdict(list)
        for t, p in zip(y_true, y_pred):
            if p != -1:
                clusters[p].append(t)
        if not clusters:
            return 0.0
        correct = 0
        total = sum(len(v) for v in clusters.values())
        for cluster_labels in clusters.values():
            counts = collections.Counter(cluster_labels)
            correct += counts.most_common(1)[0][1]
        return correct / total if total > 0 else 0.0

    def get_metrics(self) -> dict:
        metrics_dict = {
            "n_micro_clusters": len(getattr(self.model, "p_micro_clusters", {})),
            "n_outlier_clusters": len(getattr(self.model, "o_micro_clusters", {})),
            "outlier_ratio": getattr(self, "last_batch_noise_ratio", 0.0),
            "latency_ms_per_doc": getattr(self, "last_batch_latency_ms", 0.0),
            "n_macro_clusters": len(getattr(self.model, "clusters", {})),
            "n_noise_pmcs": len(getattr(self.model, "o_micro_clusters", {})),
            "n_noise_docs_window": sum(1 for p in self.window_macro_preds if p == -1),
            "purity": 0.0,
            "silhouette": 0.0,
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
                    metrics_dict["silhouette"] = round(
                        float(silhouette_score(X_valid, y_valid)), 4
                    )
                except Exception:
                    pass

        if (
            len(self.window_true_labels) == len(self.window_macro_preds)
            and len(self.window_true_labels) >= 10
        ):
            y_true = list(self.window_true_labels)
            y_pred = list(self.window_macro_preds)
            purity = self._calculate_purity(y_true, y_pred)
            if purity is not None:
                metrics_dict["purity"] = round(purity, 4)
            try:
                metrics_dict["ari"] = round(
                    float(adjusted_rand_score(y_true, y_pred)), 4
                )
            except Exception:
                pass
            try:
                metrics_dict["nmi"] = round(
                    float(normalized_mutual_info_score(y_true, y_pred)), 4
                )
            except Exception:
                pass

        return metrics_dict

    def hot_swap_model(
        self,
        new_params: Dict[str, Any],
        window_data: Union[np.ndarray, List[List[float]]],
    ):
        """
        Initializes a new DenStream model with optimal parameters found by NSGA-II
        and pre-trains (warms up) the model on the historical document window.
        Initialization phase (n_samples_init) is disabled to ensure immediate responsiveness.
        """
        eps = float(new_params.get("epsilon", 0.10))
        mu = max(int(new_params.get("mu", 2)), 2)
        decay = float(new_params.get("decaying_factor", 0.005))

        window_arr = np.asarray(window_data, dtype=np.float32)

        n_init = 1

        new_model = cluster.DenStream(
            epsilon=eps,
            mu=mu,
            decaying_factor=decay,
            beta=0.75,
            n_samples_init=n_init,
        )

        for x, _ in stream.iter_array(window_arr):
            new_model.learn_one(x)
            new_model.predict_one(x)

        self.model = new_model

        self.window_embeddings.clear()
        self.window_macro_preds.clear()
        self.window_true_labels.clear()
        self.n_samples_seen = 0
        self.last_batch_noise_ratio = 0.0

        logger.info(
            f"Model Hot-Swapped with fresh window-trained instance: "
            f"epsilon={eps}, mu={mu}, decay={decay} | "
            f"Active Micro-Clusters: {len(new_model.p_micro_clusters)}, "
            f"Macro-Clusters: {len(getattr(new_model, 'clusters', {}))}"
        )
