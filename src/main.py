import typer
from sentence_transformers import SentenceTransformer

from src.apps.daemon import ClusteringDaemon, DaemonPrototype
from src.apps.ingesting import IngesterApp, IngesterPrototype
from src.apps.setup import InfrastructureSetup, TablePrototype, TopicPrototype
from src.core.config import config
from src.domain.clustering import StreamClusterer
from src.domain.preprocessing import StreamProjector, TextPreprocessor
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
    """Create the Kafka topics and database tables."""
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
        TablePrototype(config.postgres.tables.results, schemas.CLUSTERING_RESULTS_SCHEMA),
        TablePrototype(config.postgres.tables.params, schemas.MODEL_PARAMETERS_SCHEMA),
    ]

    try:
        setup_manager.run_all(topic_prototypes, table_prototypes)
    finally:
        storage_admin.close()


@app.command(name="ingest")
def ingest_command(
    batch_size: int = typer.Option(config.kafka.batch_size),
    interval: float = typer.Option(0.5),
    drift_step: int = typer.Option(3000),
):
    """Send documents to Kafka, switching topics after drift_step messages."""
    typer.echo(f"Starting ingestion (topic change after {drift_step} messages)...")
    producer = StreamProducer(bootstrap_servers=config.kafka.bootstrap_servers)
    prototype = IngesterPrototype(
        topic=config.kafka.topic.raw_messages,
        text_column=config.dataset.text_column,
        batch_size=batch_size,
        batch_interval=interval,
        drift_step=drift_step,
    )
    app_instance = IngesterApp(producer=producer, prototype=prototype)
    app_instance.run()


@app.command(name="daemon")
def run_daemon_command(
    use_pca: bool = typer.Option(config.ml.use_pca),
    pca_dim: int = typer.Option(config.ml.pca_components_num),
):
    """Run the clustering daemon."""
    typer.echo(f"Starting clustering daemon (use_pca={use_pca}, pca_dim={pca_dim})...")

    consumer = StreamConsumer(
        bootstrap_servers=config.kafka.bootstrap_servers,
        group_id=config.kafka.consumers.clusterer_group,
        topics=[config.kafka.topic.raw_messages],
        offset_reset=config.kafka.consumers.offset_reset,
    )
    storage = get_db_admin()

    preprocessor = TextPreprocessor()
    encoder = SentenceTransformer(config.ml.embedding_model)
    projector = StreamProjector(n_components=pca_dim if use_pca else None)

    clusterer = StreamClusterer(
        epsilon=config.denstream.epsilon,
        mu=config.denstream.mu,
        beta=config.denstream.beta,
        decaying_factor=config.denstream.decaying_factor,
        n_samples_init=config.denstream.n_samples_init,
        window_size=config.denstream.window_size,
        expected_macro_clusters=len(config.dataset.categories_concept_a),
    )

    prototype = DaemonPrototype(
        batch_size=config.ml.batch_size,
        timeout=config.kafka.timeout,
        text_column=config.dataset.text_column,
        label_column="label",
        results_table=config.postgres.tables.results,
        params_table=config.postgres.tables.params,
    )

    daemon = ClusteringDaemon(
        consumer=consumer,
        storage=storage,
        preprocessor=preprocessor,
        encoder=encoder,
        projector=projector,
        clusterer=clusterer,
        prototype=prototype,
    )
    daemon.run()


@app.command(name="ui")
def run_ui_command():
    """Run the Streamlit demo panel."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/apps/web_ui.py"])


@app.command(name="benchmark-thesis-1")
def benchmark_thesis_1_command():
    """Run the thesis 1 experiment."""
    from experiments.theses.thesis_1.exp_thesis_1_ipca import main as run_exp1

    run_exp1()


@app.command(name="benchmark-thesis-2")
def benchmark_thesis_2_command():
    """Run the thesis 2 experiment."""
    from experiments.theses.thesis_2.exp_thesis_2_drift import run_drift_experiment

    run_drift_experiment()


@app.command(name="benchmark-thesis-3")
def benchmark_thesis_3_command():
    """Run the thesis 3 experiment."""
    from experiments.theses.thesis_3.exp_thesis_3_pareto import main as run_pareto_analysis

    run_pareto_analysis()


if __name__ == "__main__":
    app()
