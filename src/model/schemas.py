CLUSTERING_RESULTS_SCHEMA = """
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    n_samples_seen INTEGER,
    n_micro_clusters INTEGER,
    n_outlier_clusters INTEGER,
    n_macro_clusters INTEGER,
    micro_macro_ratio FLOAT,
    outlier_ratio FLOAT,
    noise_ratio FLOAT,
    silhouette FLOAT,
    davies_bouldin FLOAT,
    purity FLOAT,
    ari FLOAT,
    nmi FLOAT,
    latency_ms_per_doc FLOAT,
    ram_usage_mb FLOAT,
    pca_components INTEGER
"""

# One row per model swap: the compromise solution NSGA-II deployed after a
# detected drift.
MODEL_PARAMETERS_SCHEMA = """
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    swap_number INTEGER,
    docs_processed INTEGER,
    epsilon FLOAT,
    mu INTEGER,
    decay_factor FLOAT,
    fitness_quality FLOAT,
    fitness_complexity FLOAT,
    pareto_front_size INTEGER
"""
