from dataclasses import dataclass
from typing import List

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
    """
    Orchestrates the initialization of all required infrastructure components.
    """

    def __init__(self, messaging_admin, storage_admin):
        self.messaging_admin = messaging_admin
        self.storage_admin = storage_admin

    def setup_messaging(self, topics: List[TopicPrototype]):
        logger.info("Initializing messaging channels...")
        for topic in topics:
            self.messaging_admin.setup_topic(
                name=topic.name,
                num_partitions=topic.num_partitions,
                replication_factor=topic.replication_factor,
            )

    def setup_storage(self, tables: List[TablePrototype]):
        logger.info("Initializing PostgreSQL schemas...")
        for table in tables:
            self.storage_admin.create_table(
                name=table.name,
                schema_sql=table.schema_sql,
            )

    def run_all(
        self,
        topic_prototypes: List[TopicPrototype],
        table_prototypes: List[TablePrototype],
    ):
        self.setup_messaging(topic_prototypes)
        self.setup_storage(table_prototypes)
