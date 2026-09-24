"""
Time of one NSGA-II run at d = 16 on a 500-document buffer, measured with
nothing else running (the "ok. 11 s" in thesis sections 2.1, 3.5 and 4.2).
"""

import time

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from experiments.theses.thesis_3.exp_thesis_3_pareto import N_MACRO_CLUSTERS, load_buffer
from src.core.config import config
from src.domain.evolution import NSGAIIOptimizer

PCA_DIM = 16
BUFFERS = [(1, 0), (2, 0), (2, 1)]  # (stream phase, stream order)


def main():
    times = []
    for phase, seed in BUFFERS:
        raw, _ = load_buffer(phase, seed)
        buffer = normalize(IncrementalPCA(n_components=PCA_DIM).fit(raw).transform(raw))
        optimizer = NSGAIIOptimizer(
            n_macro_clusters=N_MACRO_CLUSTERS,
            population_size=config.evolution.population_size,
            generations=config.evolution.generations,
            crossover_rate=config.evolution.crossover_rate,
            crossover_eta=config.evolution.crossover_eta,
            mutation_rate=config.evolution.mutation_rate,
            mutation_eta=config.evolution.mutation_eta,
            param_bounds=config.evolution.param_bounds,
            fixed_mu=config.denstream.mu,
            n_samples_init=config.denstream.n_samples_init,
            seed=config.evolution.seed + seed,
        )
        start = time.perf_counter()
        optimizer.evolve(data_buffer=buffer)
        times.append(time.perf_counter() - start)
    print("NSGA-II time at d = 16 [s]:", ", ".join(f"{t:.1f}" for t in times))


if __name__ == "__main__":
    main()
