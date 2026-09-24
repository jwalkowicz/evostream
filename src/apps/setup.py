from dataclasses import dataclass

from src.core.logger import logger


@dataclass
class TopicPrototype:
    name: str
    num_partitions: int
    replication_factor: int


@dataclass
class TablePrototype:
    name: str
    schema_sql: str


class InfrastructureSetup:
    """Creates the Kafka topics and database tables."""

    def __init__(self, messaging_admin, storage_admin):
        self.messaging_admin = messaging_admin
        self.storage_admin = storage_admin

    def setup_messaging(self, topics: list[TopicPrototype]):
        logger.info("Initializing messaging channels...")
        for topic in topics:
            self.messaging_admin.setup_topic(
                name=topic.name,
                num_partitions=topic.num_partitions,
                replication_factor=topic.replication_factor,
            )

    def setup_storage(self, tables: list[TablePrototype]):
        logger.info("Initializing PostgreSQL schemas...")
        for table in tables:
            self.storage_admin.create_table(
                name=table.name,
                schema_sql=table.schema_sql,
            )

    def run_all(
        self,
        topic_prototypes: list[TopicPrototype],
        table_prototypes: list[TablePrototype],
    ):
        self.setup_messaging(topic_prototypes)
        self.setup_storage(table_prototypes)
