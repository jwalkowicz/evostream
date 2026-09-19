import typer
from river import cluster
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA

from src.apps.daemon import ClusteringDaemon, DaemonPrototype
from src.apps.ingesting import IngesterApp, IngesterPrototype
from src.apps.setup import InfrastructureSetup, TablePrototype, TopicPrototype
from src.core.config import config
from src.domain.clustering import StreamClusterer
from src.domain.preprocessing import EmbeddingTransformer, TextPreprocessor
from src.infrastructure.kafka.client import StreamAdmin, StreamConsumer, StreamProducer
from src.infrastructure.postgres.client import DBAdmin
from src.model import schemas

app = typer.Typer(help="evoStream")


def get_db_admin():
    """Factory for Postgres administration and storage client."""
    return DBAdmin(
        dbname=config.postgres.db,
        user=config.postgres.user,
        host=config.postgres.host,
        password=config.postgres.password,
    )


@app.command(name="setup")
def setup_command():
    """Initialize system infrastructure (Kafka topics, etc.)"""
    typer.echo("Initializing setup...")
    stream_admin = StreamAdmin(bootstrap_servers=config.kafka.bootstrap_servers)
    storage_admin = get_db_admin()
    setup_manager = InfrastructureSetup(stream_admin, storage_admin)

    topic_prototypes = [
        TopicPrototype(
            config.kafka.topic.raw_messages,
            config.kafka.num_partitions,
            config.kafka.replication_factor,
        ),
        TopicPrototype(
            config.kafka.topic.embeddings,
            config.kafka.num_partitions,
            config.kafka.replication_factor,
        ),
    ]
    table_prototypes = [
        TablePrototype(
            config.postgres.tables.results, schemas.CLUSTERING_RESULTS_SCHEMA
        ),
        TablePrototype(config.postgres.tables.params, schemas.MODEL_PARAMETERS_SCHEMA),
    ]

    try:
        setup_manager.run_all(topic_prototypes, table_prototypes)
    finally:
        storage_admin.close()


@app.command(name="ingest")
def ingest_command(
    batch_size: int = typer.Option(config.kafka.batch_size, help="Batch size for Kafka messages"),
    interval: float = typer.Option(0.5, help="Interval in seconds between batches"),
    drift_step: int = typer.Option(3000, help="Message index where concept drift occurs"),
    drift_type: str = typer.Option("sudden", help="Type of drift: sudden, gradual, recurring"),
):
    """Start the data ingestion process."""
    typer.echo(f"Starting ingestion (drift_type={drift_type}, drift_step={drift_step})...")
    producer = StreamProducer(bootstrap_servers=config.kafka.bootstrap_servers)
    prototype = IngesterPrototype(
        topic=config.kafka.topic.raw_messages,
        text_column=config.dataset.text_column,
        label_column=config.kafka.event.text_column if hasattr(config.kafka.event, "label_column") else "label",
        batch_size=batch_size,
        batch_interval=interval,
        drift_step=drift_step,
        drift_type=drift_type,
    )
    app_instance = IngesterApp(producer=producer, prototype=prototype)
    app_instance.run()


@app.command(name="daemon")
def run_daemon_command(
    use_pca: bool = typer.Option(config.ml.use_pca, help="Enable IncrementalPCA dimensionality reduction"),
    pca_dim: int = typer.Option(config.ml.pca_components_num, help="Target PCA dimensions"),
):
    """
    Core clustering daemon.
    Performs preprocessing + online embedding transformation + two-phase stream clustering.
    """
    typer.echo(f"Starting clustering daemon (use_pca={use_pca}, pca_dim={pca_dim})...")

    consumer = StreamConsumer(
        bootstrap_servers=config.kafka.bootstrap_servers,
        group_id=config.kafka.consumers.clusterer_group,
        topics=[config.kafka.topic.raw_messages],
        offset_reset=config.kafka.consumers.offset_reset,
    )
    storage = get_db_admin()

    preprocessor = TextPreprocessor()
    pca_instance = IncrementalPCA(n_components=pca_dim) if use_pca else None
    transformer = EmbeddingTransformer(
        encoder=SentenceTransformer(config.ml.embedding_model),
        pca=pca_instance,
    )

    denstream_model = cluster.DenStream(
        decaying_factor=config.denstream.decaying_factor,
        epsilon=config.denstream.epsilon,
        mu=config.denstream.mu,
    )
    clusterer = StreamClusterer(
        model=denstream_model,
        offline_eps=config.denstream.offline_eps,
        offline_min_samples=config.denstream.offline_min_samples,
    )

    prototype = DaemonPrototype(
        batch_size=config.kafka.batch_size,
        timeout=config.kafka.timeout,
        text_column=config.dataset.text_column,
        label_column="label",
        results_table=config.postgres.tables.results,
    )

    daemon = ClusteringDaemon(
        consumer=consumer,
        storage=storage,
        preprocessor=preprocessor,
        transformer=transformer,
        clusterer=clusterer,
        prototype=prototype,
    )
    daemon.run()


@app.command(name="ui")
def run_ui_command():
    """Launch the interactive Streamlit defense presentation UI."""
    import subprocess
    import sys
    typer.echo("Launching interactive Streamlit live defense presentation...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/apps/web_ui.py"])


@app.command(name="benchmark-thesis-1")
def benchmark_thesis_1_command():
    """Run Thesis 1 benchmark: SBERT + IPCA vs Full Dimensionality."""
    from experiments.exp_thesis_1_ipca import main as run_exp1
    run_exp1()


@app.command(name="benchmark-thesis-2")
def benchmark_thesis_2_command():
    """Run Thesis 2 benchmark: NSGA-II Reactive Concept Drift Self-Adaptation."""
    from experiments.exp_thesis_2_drift import run_drift_experiment
    run_drift_experiment()


@app.command(name="benchmark-thesis-3")
def benchmark_thesis_3_command():
    """Run Thesis 3 benchmark: Multi-Objective Pareto Front & Knee Point Analysis."""
    from experiments.exp_thesis_3_pareto import run_pareto_analysis
    run_pareto_analysis()


@app.command(name="hello")
def hello_command():
    typer.echo("Hello evoStream!")


if __name__ == "__main__":
    app()


