import json

from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, KafkaError, KafkaException, NewTopic

from src.core.logger import logger


class StreamAdmin:
    """Kafka administrator client for managing topics."""

    def __init__(self, bootstrap_servers: str):
        self.admin = AdminClient({"bootstrap.servers": bootstrap_servers})

    def setup_topic(self, name, num_partitions, replication_factor):
        topic = NewTopic(
            topic=name,
            num_partitions=num_partitions,
            replication_factor=replication_factor,
        )
        self.create_topics([topic])

    def create_topics(self, topics: list):
        topics_futures = self.admin.create_topics(topics)

        for topic_name, future in topics_futures.items():
            try:
                future.result()
                logger.success(f"Topic '{topic_name}' has been successfully created.")
            except KafkaException as e:
                if e.args[0].code() == KafkaError.TOPIC_ALREADY_EXISTS:
                    logger.info(f"Topic '{topic_name}' already exists. Skipping initialization.")
                else:
                    logger.error(f"Kafka failed to create topic '{topic_name}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error creating topic '{topic_name}': {e}")


class StreamProducer:
    """Kafka producer client for sending messages."""

    def __init__(self, bootstrap_servers: str):
        self.producer = Producer({"bootstrap.servers": bootstrap_servers})
        logger.info(f"Kafka Producer initialized at {bootstrap_servers}")

    def _acked(self, err, msg):
        """Internal callback for delivery reports."""
        if err is not None:
            logger.error(f"Failed to deliver message: {err}")

    def send(self, topic: str, value: dict):
        value_encoded = json.dumps(value).encode("utf-8")
        self.producer.produce(topic, value=value_encoded, callback=self._acked)
        self.producer.poll(0)

    def close(self):
        logger.info("Flushing remaining messages...")
        self.producer.flush()
        logger.info("Producer successfully closed.")


class StreamConsumer:
    """Kafka consumer client for receiving messages."""

    def __init__(self, bootstrap_servers: str, group_id: str, topics: list, offset_reset: str):
        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": offset_reset,
                "enable.auto.commit": False,
            }
        )
        self.consumer.subscribe(topics)

    def consume(self, batch_size: int, timeout: float):
        return self.consumer.consume(batch_size, timeout=timeout)

    def commit(self):
        self.consumer.commit(asynchronous=True)

    def close(self):
        self.consumer.close()
        logger.info("Consumer successfully closed.")
