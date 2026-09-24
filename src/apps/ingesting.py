import random
import signal
import time
from dataclasses import dataclass

from sklearn.datasets import fetch_20newsgroups

from src.core.logger import logger


@dataclass
class IngesterPrototype:
    topic: str
    text_column: str
    label_column: str = "label"
    batch_size: int = 64
    batch_interval: float = 0.5
    drift_step: int = 3000
    drift_type: str = "sudden"


class IngesterApp:
    """Sends 20 Newsgroups documents to Kafka with an abrupt topic change,
    each message carrying its category for evaluation."""

    def __init__(self, producer, prototype: IngesterPrototype):
        self.producer = producer
        self.prototype = prototype
        self.running = True
        self.message_count = 0

    def _load_data_with_labels(self, categories: list[str]) -> list[tuple[str, str]]:
        logger.info(f"Loading data for categories: {categories}")
        dataset = fetch_20newsgroups(
            subset="all",
            categories=categories,
            remove=("headers", "footers", "quotes"),
        )
        samples = []
        for text, target_idx in zip(dataset.data, dataset.target):
            cleaned = text.strip()
            if len(cleaned) > 10:
                label_name = dataset.target_names[target_idx]
                samples.append((cleaned, label_name))

        random.shuffle(samples)
        logger.info(f"Loaded {len(samples)} valid samples for {categories}")
        return samples

    def handle_shutdown(self, sig, frame):
        logger.warning("Shutdown signal received. Stopping ingestion...")
        self.running = False

    def run(self):
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        phase1_categories = ["sci.space", "sci.med", "rec.autos"]
        phase1_data = self._load_data_with_labels(phase1_categories)

        phase2_categories = [
            "rec.sport.baseball",
            "comp.sys.ibm.pc.hardware",
            "talk.politics.mideast",
        ]
        phase2_data = self._load_data_with_labels(phase2_categories)

        logger.info(f"Starting ingestion stream to topic '{self.prototype.topic}'...")
        drift_logged = False

        try:
            while self.running:
                if self.prototype.drift_type == "sudden":
                    if self.message_count < self.prototype.drift_step:
                        current_pool = phase1_data
                    else:
                        if not drift_logged:
                            logger.warning(f"Concept drift triggered at message count {self.message_count}")
                            drift_logged = True
                        current_pool = phase2_data

                batch = []
                for _ in range(self.prototype.batch_size):
                    text, label = random.choice(current_pool)
                    batch.append(
                        {
                            self.prototype.text_column: text,
                            self.prototype.label_column: label,
                            "timestamp": time.time(),
                            "msg_id": self.message_count,
                        }
                    )
                    self.message_count += 1

                for msg in batch:
                    self.producer.send(topic=self.prototype.topic, value=msg)

                logger.info(f"Sent batch of {len(batch)} messages. Total sent: {self.message_count}")
                time.sleep(self.prototype.batch_interval)

        except Exception as e:
            logger.error(f"An error occurred during ingestion: {e}")

        finally:
            self.producer.close()
            logger.info("Ingester shutdown complete.")
