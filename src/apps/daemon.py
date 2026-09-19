import json
import signal
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.core.config import config
from src.core.logger import logger
from src.domain.clustering import StreamClusterer
from src.domain.drift import UnsupervisedDriftDetector
from src.domain.evolution import NSGAIIOptimizer
from src.domain.preprocessing import EmbeddingTransformer, TextPreprocessor
from src.infrastructure.postgres.client import DBAdmin


@dataclass
class DaemonPrototype:
    batch_size: int
    timeout: float
    text_column: str
    label_column: str = "label"
    results_table: str = "clustering_results"
    adaptation_mode: str = "hot_swap"


class ClusteringDaemon:
    """
    Consumes raw text data from Kafka, applies preprocessing & embedding transformation,
    executes online two-phase stream clustering, detects concept drift, and autonomously
    adapts parameters via NSGA-II.
    """

    def __init__(
        self,
        consumer,
        storage: Optional[DBAdmin],
        preprocessor: TextPreprocessor,
        transformer: EmbeddingTransformer,
        clusterer: StreamClusterer,
        prototype: DaemonPrototype,
    ):
        self.consumer = consumer
        self.storage = storage
        self.preprocessor = preprocessor
        self.transformer = transformer
        self.clusterer = clusterer
        self.prototype = prototype
        self.running = True

        self.drift_detector = UnsupervisedDriftDetector(
            window_size=config.drift.window_size,
            min_warmup_steps=config.drift.min_warmup_steps,
            quality_drop_sigma=config.drift.quality_drop_sigma,
            outlier_surge_threshold=config.drift.outlier_surge_threshold,
            cooldown_steps=config.drift.cooldown_steps,
        )
        self.optimizer = NSGAIIOptimizer(
            population_size=config.evolution.population_size,
            generations=config.evolution.generations,
            crossover_rate=config.evolution.crossover_rate,
            crossover_eta=config.evolution.crossover_eta,
            mutation_rate=config.evolution.mutation_rate,
            mutation_eta=config.evolution.mutation_eta,
            fixed_mu=config.evolution.fixed_mu,
            n_samples_init=config.evolution.n_samples_init,
            min_eval_buffer=config.evolution.min_eval_buffer,
            seed=config.evolution.seed,
            param_bounds={
                k: tuple(v) for k, v in config.evolution.param_bounds.items()
            },
        )
        self.recent_vectors_buffer = []

    def _handle_shutdown(self, sig, frame):
        logger.warning("Shutdown signal received. Stopping Daemon...")
        self.running = False

    def run(self):
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        logger.info("Core Clustering Daemon is running...")

        try:
            while self.running:
                kafka_batch = self.consumer.consume(
                    batch_size=self.prototype.batch_size, timeout=self.prototype.timeout
                )
                if not kafka_batch:
                    continue

                raw_texts = []
                labels = []

                for msg in kafka_batch:
                    try:
                        data = json.loads(msg.value().decode("utf-8"))
                        text = data.get(self.prototype.text_column, "")
                        label = data.get(self.prototype.label_column, None)
                        if text:
                            raw_texts.append(text)
                            labels.append(label)
                    except Exception as e:
                        logger.error(f"Failed to parse message: {e}")

                if not raw_texts:
                    continue

                cleaned_texts = self.preprocessor.clean_batch(raw_texts)

                embeddings = self.transformer.fit_transform(cleaned_texts)

                preds = self.clusterer.update(
                    embeddings, labels=labels if any(labels) else None
                )

                metrics = self.clusterer.get_metrics()
                metrics["pca_components"] = self.transformer.output_dim

                self.recent_vectors_buffer.extend(embeddings)
                if len(self.recent_vectors_buffer) > 500:
                    self.recent_vectors_buffer = self.recent_vectors_buffer[-500:]

                is_drift = self.drift_detector.update(
                    current_silhouette=metrics["silhouette"] or 0.0,
                    n_micro_clusters=metrics["n_micro_clusters"],
                    n_outlier_clusters=metrics["n_outlier_clusters"],
                    outlier_ratio=metrics.get("outlier_ratio"),
                    n_macro_clusters=metrics.get("n_macro_clusters"),
                )

                if is_drift and len(self.recent_vectors_buffer) >= 50:
                    logger.warning(
                        f"Unsupervised Concept Drift flagged! Triggering reactive NSGA-II "
                        f"optimization (mode: {self.prototype.adaptation_mode})..."
                    )

                    best_knee, _, _ = self.optimizer.evolve(
                        data_buffer=np.array(self.recent_vectors_buffer),
                        current_params={
                            "epsilon": float(self.clusterer.model.epsilon),
                            "decaying_factor": float(
                                self.clusterer.model.decaying_factor
                            ),
                        },
                    )
                    self.clusterer.hot_swap_model(
                        new_params=best_knee.params,
                        window_data=np.array(self.recent_vectors_buffer),
                    )

                if self.storage:
                    try:
                        self.storage.insert(
                            table=self.prototype.results_table,
                            data=metrics,
                        )
                    except Exception as e:
                        logger.error(f"Failed to insert metrics into DB: {e}")

                self.consumer.commit()
                logger.info(
                    f"Processed batch of {len(raw_texts)} docs | "
                    f"Micro: {metrics['n_micro_clusters']} | "
                    f"Macro: {metrics['n_macro_clusters']} | "
                    f"Ratio: {metrics['micro_macro_ratio']:.2f} | "
                    f"Purity: {metrics.get('purity', 0.0)} | "
                    f"Silhouette: {metrics.get('silhouette', 0.0)} | "
                    f"Latency: {metrics.get('latency_ms_per_doc', 0.0)}ms/doc"
                )

        finally:
            self.consumer.close()
            if self.storage:
                self.storage.close()
            logger.info("Daemon shutdown complete.")
