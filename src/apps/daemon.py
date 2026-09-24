import json
import signal
from dataclasses import dataclass

import numpy as np

from src.core.config import config
from src.core.logger import logger
from src.domain.clustering import StreamClusterer
from src.domain.drift import UnsupervisedDriftDetector
from src.domain.evolution import NSGAIIOptimizer
from src.domain.preprocessing import StreamProjector, TextPreprocessor
from src.infrastructure.postgres.client import DBAdmin


@dataclass
class DaemonPrototype:
    batch_size: int
    timeout: float
    text_column: str
    label_column: str = "label"
    results_table: str = "clustering_results"
    params_table: str = "model_parameters"


class ClusteringDaemon:
    """Reads documents from Kafka and runs the same pipeline as the thesis 2
    experiment: SBERT, IPCA, DenStream, drift detection and, after an alarm,
    a model swap with NSGA-II on a buffer of new documents."""

    def __init__(
        self,
        consumer,
        storage: DBAdmin | None,
        preprocessor: TextPreprocessor,
        encoder,
        projector: StreamProjector,
        clusterer: StreamClusterer,
        prototype: DaemonPrototype,
    ):
        self.consumer = consumer
        self.storage = storage
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.projector = projector
        self.clusterer = clusterer
        self.prototype = prototype
        self.running = True

        self.drift_detector = UnsupervisedDriftDetector(
            window_size=config.drift.window_size,
            min_warmup_steps=config.drift.min_warmup_steps,
            quality_drop_sigma=config.drift.quality_drop_sigma,
            cooldown_steps=config.drift.cooldown_steps,
            consecutive_drops_required=config.drift.consecutive_drops_required,
            centroid_shift_threshold=config.drift.centroid_shift_threshold,
            quality_absolute_floor=config.drift.quality_absolute_floor,
            centroid_shift_min_warmup_steps=config.drift.centroid_shift_min_warmup_steps,
        )
        self.optimizer = NSGAIIOptimizer(
            n_macro_clusters=clusterer.expected_macro_clusters,
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
            param_bounds={k: tuple(v) for k, v in config.evolution.param_bounds.items()},
        )

        self.warmup_buffer: list[np.ndarray] = []
        self.swap_buffer: list[np.ndarray] = []
        self.collecting_for_swap = False
        self.docs_processed = 0
        self.swap_count = 0

    def _handle_shutdown(self, sig, frame):
        logger.warning("Shutdown signal received. Stopping Daemon...")
        self.running = False

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def _collect_warmup(self, embeddings: np.ndarray) -> None:
        """Buffers the first documents; once there are enough, fits IPCA on
        them and warm-starts DenStream on them - the same procedure as after a
        model swap. As in the experiments, these documents are not scored."""
        self.warmup_buffer.extend(embeddings)
        if len(self.warmup_buffer) >= config.ml.ipca_warmup_size:
            warmup = np.array(self.warmup_buffer)
            self.projector.fit(warmup)
            self.clusterer.warm_start(self.projector.transform(warmup))
            logger.info(f"IPCA fitted and DenStream warm-started on {len(warmup)} documents.")
            self.warmup_buffer = []

    def _swap_model(self) -> None:
        """Re-fits IPCA on the post-drift buffer, evolves DenStream parameters
        on the projected buffer and swaps in the new projection and model."""
        raw_buffer = np.array(self.swap_buffer)
        self.projector.fit(raw_buffer)
        projected = self.projector.transform(raw_buffer)

        compromise, pareto_front, _ = self.optimizer.evolve(
            data_buffer=projected,
            current_params={
                "epsilon": float(self.clusterer.model.epsilon),
                "decaying_factor": float(self.clusterer.model.decaying_factor),
            },
        )
        self.clusterer.hot_swap_model(new_params=compromise.params, window_data=projected)
        self.swap_count += 1
        self._save_parameters(compromise, pareto_front_size=len(pareto_front))

        self.collecting_for_swap = False
        self.swap_buffer = []

    def _save_parameters(self, compromise, pareto_front_size: int) -> None:
        """Records the deployed compromise solution - one row per model swap."""
        if not self.storage:
            return
        try:
            self.storage.insert(
                table=self.prototype.params_table,
                data={
                    "swap_number": self.swap_count,
                    "docs_processed": self.docs_processed,
                    "epsilon": compromise.params["epsilon"],
                    "mu": compromise.params["mu"],
                    "decay_factor": compromise.params["decaying_factor"],
                    "fitness_quality": compromise.quality_score,
                    "fitness_complexity": compromise.complexity_score,
                    "pareto_front_size": pareto_front_size,
                },
            )
        except Exception as e:
            logger.error(f"Failed to insert model parameters into DB: {e}")

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

                embeddings = self._encode(self.preprocessor.clean_batch(raw_texts))
                self.docs_processed += len(embeddings)

                if not self.projector.is_fitted:
                    self._collect_warmup(embeddings)
                    self.consumer.commit()
                    continue

                self.clusterer.update(
                    self.projector.transform(embeddings),
                    labels=labels if any(labels) else None,
                )
                metrics = self.clusterer.get_metrics()
                metrics["pca_components"] = self.projector.n_components or embeddings.shape[1]

                is_drift = self.drift_detector.update(
                    current_silhouette=metrics["silhouette"] or 0.0,
                    centroid_shift=metrics["centroid_shift"],
                )

                if is_drift and not self.collecting_for_swap:
                    logger.warning(
                        "Unsupervised concept drift flagged! Collecting "
                        f"{config.evolution.hotswap_buffer_size} post-drift documents "
                        "before re-fitting IPCA and running NSGA-II..."
                    )
                    self.collecting_for_swap = True
                    self.swap_buffer = []

                if self.collecting_for_swap:
                    self.swap_buffer.extend(embeddings)
                    if len(self.swap_buffer) >= config.evolution.hotswap_buffer_size:
                        self._swap_model()
                else:
                    self.clusterer.ease_decaying_factor(config.denstream.decaying_factor)

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
                    f"Purity: {metrics.get('purity', 0.0)} | "
                    f"Silhouette: {metrics.get('silhouette', 0.0)} | "
                    f"Latency: {metrics.get('latency_ms_per_doc', 0.0):.3f}ms/doc"
                )

        finally:
            self.consumer.close()
            if self.storage:
                self.storage.close()
            logger.info("Daemon shutdown complete.")
