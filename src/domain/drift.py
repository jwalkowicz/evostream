import collections
from typing import Optional

import numpy as np

from src.core.logger import logger


class UnsupervisedDriftDetector:
    """
    Real-time unsupervised concept drift detector for text stream clustering.
    Monitors three complementary spatial signals:
    1. Outlier buffer proliferation rate (Surge in novel topic candidates).
    2. Statistical degradation of rolling cluster cohesion (Silhouette / iCVI).
    """

    def __init__(
        self,
        window_size: int = 20,
        min_warmup_steps: int = 10,
        quality_drop_sigma: float = 2.0,
        outlier_surge_threshold: float = 0.35,
        cooldown_steps: int = 10,
        centroid_shift_threshold: float = 0.20,
    ):
        self.window_size = window_size
        self.min_warmup_steps = min_warmup_steps
        self.quality_drop_sigma = quality_drop_sigma
        self.outlier_surge_threshold = outlier_surge_threshold
        self.cooldown_steps = cooldown_steps
        self.centroid_shift_threshold = centroid_shift_threshold

        self.history_quality = collections.deque(maxlen=window_size)
        self.history_cms = collections.deque(maxlen=window_size)
        self.previous_n_macro = 0

        self.total_steps_seen = 0
        self.steps_since_last_drift = 0
        self.total_drifts_detected = 0
        self.current_threshold = None

    def update(
        self,
        current_silhouette: float,
        n_micro_clusters: int,
        n_outlier_clusters: int,
        outlier_ratio: Optional[float] = None,
        n_macro_clusters: Optional[int] = None,
    ) -> bool:
        """
        Evaluates current stream step. Returns True if concept drift is detected.

        Args:
            current_silhouette: Current silhouette score (internal cluster quality).
            n_micro_clusters: Number of active p-micro-clusters.
            n_outlier_clusters: Number of o-micro-clusters (outlier buffer).
            outlier_ratio: Pre-computed outlier ratio (optional; computed from counts if None).
            centroid_mass_shift: CMS metric from StreamClusterer — ratio of centroid
                                migration speed to inter-centroid separation.
            n_macro_clusters: Current number of macro-clusters for novelty detection.
        """
        self.total_steps_seen += 1
        self.steps_since_last_drift += 1

        if outlier_ratio is None:
            total_clusters = n_micro_clusters + n_outlier_clusters
            outlier_ratio = (
                float(n_outlier_clusters / total_clusters)
                if total_clusters > 0
                else 0.0
            )

        is_drift = False
        drift_reasons = []

        if (
            self.total_steps_seen >= self.min_warmup_steps
            and self.steps_since_last_drift >= self.cooldown_steps
        ):
            # Signal 1: outlier surge
            if outlier_ratio >= self.outlier_surge_threshold:
                is_drift = True
                drift_reasons.append(
                    f"Outlier Surge (R_outlier: {outlier_ratio * 100:.1f}% >= Threshold: {self.outlier_surge_threshold * 100:.1f}%)"
                )

            # Signal 2: significant quality degradation
            if len(self.history_quality) >= max(3, self.min_warmup_steps // 2):
                mean_q = float(np.mean(self.history_quality))
                std_q = float(np.std(self.history_quality))
                threshold_q = mean_q - self.quality_drop_sigma * max(std_q, 0.015)
                self.current_threshold = threshold_q

                if current_silhouette < threshold_q:
                    is_drift = True
                    drift_reasons.append(
                        f"Quality Drop (Silhouette: {current_silhouette:.3f} < Baseline: {threshold_q:.3f})"
                    )

        if is_drift:
            self.total_drifts_detected += 1
            self.steps_since_last_drift = 0
            logger.warning(
                f"[DRIFT DETECTED] #{self.total_drifts_detected} -> Reasons: {' | '.join(drift_reasons)}"
            )

        if current_silhouette is not None and current_silhouette > 0:
            self.history_quality.append(current_silhouette)
        if n_macro_clusters is not None:
            self.previous_n_macro = n_macro_clusters

        return is_drift
