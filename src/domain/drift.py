import collections
from typing import Optional

import numpy as np

from src.core.logger import logger


class UnsupervisedDriftDetector:
    """
    Unsupervised concept drift detector. Monitors two signals:
    1. Degradation of rolling cluster score (silhouette), relative to its own
       recent history AND below an absolute floor, sustained for several
       consecutive batches.
    2. Macro-cluster centroids moving in the embedding space (purely
       geometric - independent of clustering quality).

    An earlier version also monitored the share of outlier micro-cluster
    weight. It never contributed a detection: on the thesis 2 stream that
    share stayed at or below ~2% of the total weight even during the engineered
    drift, far under any reasonable threshold, so it was removed.
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
        # A single anomalous batch can transiently crash the silhouette score
        # (e.g. one awkward batch of documents) even when the underlying
        # concept hasn't actually shifted; real drift stays low for many
        # batches in a row. Requiring the quality-drop signal to persist for
        # this many consecutive batches filters out one-off blips without
        # slowing down reaction to genuine, sustained drift.
        self.consecutive_drops_required = consecutive_drops_required
        self._consecutive_quality_drops = 0
        # A model can also gently settle onto a new, slightly lower but
        # still perfectly healthy silhouette baseline over many batches
        # (clustering quality stays fine in absolute terms, e.g. purity
        # unaffected) - that shouldn't count as drift just because it moved
        # relative to its own recent history. Requiring the ABSOLUTE
        # silhouette to also be below this floor (real drift crashes it
        # toward 0/negative) filters that case out without needing a longer
        # persistence requirement that would also slow down real detection.
        self.quality_absolute_floor = quality_absolute_floor
        # The centroid-shift signal is unreliable for a while after a
        # (re)start: the very first clusters are still forming, so centroids
        # swing a lot with no relation to real drift. This settling period is
        # longer than the quality signal needs, so it gets its own, later
        # warmup gate rather than sharing min_warmup_steps.
        self.centroid_shift_min_warmup_steps = centroid_shift_min_warmup_steps

        self.history_quality = collections.deque(maxlen=window_size)

        self.total_steps_seen = 0
        self.steps_since_last_drift = 0
        self.total_drifts_detected = 0
        self.current_threshold = None

    def update(self, current_silhouette: float, centroid_shift: Optional[float] = None) -> bool:
        """
        Evaluates current stream step. Returns True if concept drift is detected.
        """
        self.total_steps_seen += 1
        self.steps_since_last_drift += 1

        is_drift = False
        drift_reasons = []
        can_trigger = (
            self.total_steps_seen >= self.min_warmup_steps
            and self.steps_since_last_drift >= self.cooldown_steps
        )

        # Signal 1: quality degradation. Tracked as a persistence counter
        # (updated every step, regardless of cooldown) so genuine sustained
        # drift is recognized promptly once cooldown lifts, but a single
        # anomalous batch - which can transiently crash the silhouette score
        # even without any real concept shift - can't fire this alone.
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

        # Signal 2: macro-cluster centroids have moved by more than an
        # absolute distance threshold in the embedding space. Unlike signal
        # 1, this is a fixed absolute cutoff, not a rolling-baseline
        # comparison - it doesn't care whether clustering quality is good or
        # bad, only whether the cluster centers themselves have relocated.
        # Gated by its own, later warmup: centroids are still settling for a
        # while after a (re)start, long after signal 1 is already trusted.
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
