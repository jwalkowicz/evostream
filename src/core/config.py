from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

class KafkaSettings(BaseModel):
    topic: KafkaTopic
    event: KafkaEventSchema
    consumers: KafkaConsumer

    bootstrap_servers: str
    num_partitions: int
    replication_factor: int
    timeout: float
    batch_size: int


class KafkaTopic(BaseModel):
    raw_messages: str
    embeddings: str


class KafkaEventSchema(BaseModel):
    text_column: str
    vector_column: str


class KafkaConsumer(BaseModel):
    preprocessor_group: str
    clusterer_group: str
    offset_reset: str

class DatasetSettings(BaseModel):
    name: str = "20newsgroups"
    categories_concept_a: List[str] = ["sci.space", "sci.med", "rec.autos"]
    categories_concept_b: List[str] = [
        "rec.sport.baseball",
        "comp.sys.ibm.pc.hardware",
        "talk.politics.mideast",
    ]
    categories_concept_c: List[str] = [
        "comp.graphics",
        "soc.religion.christian",
        "sci.crypt",
    ]
    max_samples_per_concept: int = 1000
    text_column: str = "text"

class MLSettings(BaseModel):
    embedding_model: str
    pca_components_num: int
    use_pca: bool = True


class DenStreamSettings(BaseModel):
    epsilon: float = 0.10
    mu: int = 2
    beta: float = 0.75
    decaying_factor: float = 0.005
    offline_eps: float = 0.70
    offline_min_samples: int = 1
    window_size: int = 300
    n_samples_init: int = 1
    adaptive_eps: bool = True
    eps_percentile: float = 0.40
    eps_scale: float = 1.8


class DriftSettings(BaseModel):
    window_size: int = 20
    min_warmup_steps: int = 5
    quality_drop_sigma: float = 2.0
    outlier_surge_threshold: float = 0.30
    cooldown_steps: int = 8
    divergence_threshold: float = 0.25


class EvolutionSettings(BaseModel):
    seed: int = 42
    population_size: int = 32
    generations: int = 12
    crossover_rate: float = 0.8
    crossover_eta: float = 15.0
    mutation_rate: float = 0.2
    mutation_eta: float = 20.0
    fixed_mu: int = 2
    beta: float = 0.75
    n_samples_init: int = 1
    min_eval_buffer: int = 40
    param_bounds: Dict[str, Any] = {
        "epsilon": [0.05, 0.15],
        "decaying_factor": [0.005, 0.08],
    }


class PostgresSettings(BaseModel):
    user: str = "user"
    password: str = "password"
    db: str = "evostream-db"
    host: str = "localhost"
    port: int = 5432
    tables: PostgresTableNames


class PostgresTableNames(BaseModel):
    params: str = "model_parameters"
    results: str = "clustering_results"


class Settings(BaseSettings):
    seed: int = 42
    kafka: Optional[KafkaSettings] = None
    dataset: Optional[DatasetSettings] = None
    ml: Optional[MLSettings] = None
    denstream: DenStreamSettings = DenStreamSettings()
    drift: DriftSettings = DriftSettings()
    evolution: EvolutionSettings = EvolutionSettings()
    postgres: PostgresSettings

    model_config = SettingsConfigDict(
        env_file=".env", env_nested_delimiter="_", extra="ignore"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:

        yaml_path = "config/config.yaml"
        yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path)

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
        )

    @property
    def denstream_params(self) -> dict:
        return self.denstream.model_dump()


config = Settings()
