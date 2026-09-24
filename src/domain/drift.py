import collections
from typing import Optional

import numpy as np

from src.core.logger import logger


class UnsupervisedDriftDetector:
    """Unsupervised concept drift detector with two signals:

    1. quality drop: the silhouette falls below both its recent baseline and
       an absolute floor for several consecutive batches;
    2. centroid shift: the macro-cluster centroids move further than a
       threshold in the embedding space.
    """

    def __init__(
        self,
        window_size: int = 20,
        min_warmup_steps: int = 10,
        quality_drop_sigma: float = 2.0,
        cooldown_steps: int = 10,
        centroid_shift_threshold: float = 0.20,
        consecutive_drops_required: int = 2,
        quality_absolute_floor: float = 0.08,
        centroid_shift_min_warmup_steps: int = 40,
    ):
        self.window_size = window_size
        self.min_warmup_steps = min_warmup_steps
        self.quality_drop_sigma = quality_drop_sigma
        self.cooldown_steps = cooldown_steps
        self.centroid_shift_threshold = centroid_shift_threshold
        self.consecutive_drops_required = consecutive_drops_required
        self._consecutive_quality_drops = 0
        # Ignores a model that settles on a slightly lower but still good level.
        self.quality_absolute_floor = quality_absolute_floor
        # Right after a start the centroids still move a lot on their own.
        self.centroid_shift_min_warmup_steps = centroid_shift_min_warmup_steps

        self.history_quality = collections.deque(maxlen=window_size)

        self.total_steps_seen = 0
        self.steps_since_last_drift = 0
        self.total_drifts_detected = 0
        self.current_threshold = None

    def update(self, current_silhouette: float, centroid_shift: Optional[float] = None) -> bool:
        """Processes one batch; returns True if drift is detected."""
        self.total_steps_seen += 1
        self.steps_since_last_drift += 1

        is_drift = False
        drift_reasons = []
        can_trigger = (
            self.total_steps_seen >= self.min_warmup_steps
            and self.steps_since_last_drift >= self.cooldown_steps
        )

        # The counter of consecutive drops is updated during the cooldown too.
        if len(self.history_quality) >= max(3, self.min_warmup_steps // 2):
            mean_q = float(np.mean(self.history_quality))
            std_q = float(np.std(self.history_quality))
            threshold_q = mean_q - self.quality_drop_sigma * max(std_q, 0.015)
            self.current_threshold = threshold_q

            if current_silhouette < threshold_q and current_silhouette < self.quality_absolute_floor:
                self._consecutive_quality_drops += 1
            else:
                self._consecutive_quality_drops = 0

            if can_trigger and self._consecutive_quality_drops >= self.consecutive_drops_required:
                is_drift = True
                drift_reasons.append(
                    f"Quality drop (silhouette: {current_silhouette:.3f} < baseline: {threshold_q:.3f} "
                    f"and < absolute floor: {self.quality_absolute_floor:.3f}, "
                    f"{self._consecutive_quality_drops} consecutive batches)"
                )

        centroid_shift_ready = self.total_steps_seen >= self.centroid_shift_min_warmup_steps
        if can_trigger and centroid_shift_ready and centroid_shift is not None and centroid_shift >= self.centroid_shift_threshold:
            is_drift = True
            drift_reasons.append(
                f"Centroid shift ({centroid_shift:.3f} >= threshold: {self.centroid_shift_threshold:.3f})"
            )

        if is_drift:
            self.total_drifts_detected += 1
            self.steps_since_last_drift = 0
            self._consecutive_quality_drops = 0
            logger.warning(
                f"Drift detected {self.total_drifts_detected}. Reason(s): {' | '.join(drift_reasons)}"
            )

        if current_silhouette is not None and current_silhouette > 0:
            self.history_quality.append(current_silhouette)

        return is_drift
