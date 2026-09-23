import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from river import cluster
from sklearn.metrics import silhouette_score

from src.core.config import config
from src.core.logger import logger
from src.domain.clustering import group_micro_clusters, micro_cluster_centers


@dataclass
class Individual:
    params: Dict[str, float]
    objectives: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rank: int = 0
    crowding_dist: float = 0.0
    quality_score: float = 0.0
    complexity_score: float = 0.0

    def dominates(self, other: "Individual") -> bool:
        return bool(
            np.all(self.objectives <= other.objectives)
            and np.any(self.objectives < other.objectives)
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return False
        return self.params == other.params and bool(
            np.allclose(self.objectives, other.objectives)
        )


def evaluate_parameters(
    dict_buffer: List[Dict[int, float]],
    epsilon: float,
    decaying_factor: float,
    n_macro_clusters: int,
    mu: int,
    n_samples_init: int,
) -> Tuple[float, float, int]:
    """
    Fitness of one candidate (epsilon, decaying_factor): trains a fresh
    DenStream on the buffer, groups its p-micro-clusters into macro-clusters
    with the system's offline phase and returns (quality, complexity,
    number of p-micro-clusters):

      quality    = silhouette of the p-micro-cluster centres, labelled by
                   their macro-cluster,
      complexity = N_micro / |buffer| + 0.05 * ln(1 + N_micro / N_macro).

    Degenerate candidates get penalty values: (-1, 1) with fewer than two
    p-micro-clusters, (-0.5, 0.8) with too few to form the macro-clusters.
    """
    try:
        model = cluster.DenStream(
            epsilon=epsilon,
            mu=mu,
            decaying_factor=decaying_factor,
            beta=config.denstream.beta,
            n_samples_init=n_samples_init,
        )
        # River's micro-clusters keep a reference to the dict they were
        # created from and add later points into it, so each candidate gets
        # its own copies - otherwise it would corrupt the shared buffer for
        # every candidate evaluated after it.
        for row in dict_buffer:
            model.learn_one(dict(row))

        _, centers = micro_cluster_centers(model)
        n_micro = len(centers)

        if n_micro < 2:
            return -1.0, 1.0, n_micro
        if n_micro <= n_macro_clusters:
            return -0.5, 0.8, n_micro

        macro_labels = group_micro_clusters(centers, n_macro_clusters)
        quality = float(silhouette_score(centers, macro_labels))
        complexity = n_micro / len(dict_buffer) + 0.05 * float(np.log1p(n_micro / n_macro_clusters))
        return quality, complexity, n_micro
    except Exception:
        # Fallback penalty for parameters the model cannot run with
        return -1.0, 1.0, 0


class StreamClusteringOptimizationProblem(ElementwiseProblem):
    """
    Multi-objective optimization problem for streaming text clustering.
    2D chromosome: theta = (epsilon, decaying_factor) in R^2.

    Objectives (both minimised by pymoo): f1 = -quality, f2 = complexity,
    as computed by evaluate_parameters() - candidates are scored with the
    same offline phase the system deploys.
    """

    def __init__(
        self,
        data_buffer: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        n_macro_clusters: int,
        fixed_mu: int = 3,
        n_samples_init: int = 30,
    ):
        super().__init__(n_var=2, n_obj=2, xl=xl, xu=xu)
        self.n_macro_clusters = n_macro_clusters
        self.fixed_mu = fixed_mu
        self.n_samples_init = n_samples_init
        self.dict_buffer = [dict(enumerate(row)) for row in data_buffer]

    def _evaluate(self, x, out, *args, **kwargs):
        quality, complexity, _ = evaluate_parameters(
            self.dict_buffer,
            epsilon=float(x[0]),
            decaying_factor=float(x[1]),
            n_macro_clusters=self.n_macro_clusters,
            mu=self.fixed_mu,
            n_samples_init=self.n_samples_init,
        )
        out["F"] = [-quality, complexity]


class NSGAIIOptimizer:
    def __init__(
        self,
        n_macro_clusters: int,
        population_size: int = 16,
        generations: int = 6,
        crossover_rate: float = 0.8,
        crossover_eta: float = 15.0,
        mutation_rate: float = 0.2,
        mutation_eta: float = 20.0,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fixed_mu: int = 3,
        n_samples_init: int = 30,
        min_eval_buffer: int = 40,
        seed: Optional[int] = None,
    ):
        self.n_macro_clusters = n_macro_clusters
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.crossover_eta = crossover_eta
        self.mutation_rate = mutation_rate
        self.mutation_eta = mutation_eta
        self.fixed_mu = fixed_mu
        self.n_samples_init = n_samples_init
        self.min_eval_buffer = min_eval_buffer
        self.seed = seed if seed is not None else getattr(config.evolution, "seed", 42)

        self.param_bounds = param_bounds or {
            "epsilon": (0.15, 0.65),
            "decaying_factor": (0.005, 0.08),
        }
        self.param_keys = list(self.param_bounds.keys())
        self.xl = np.array(
            [self.param_bounds[k][0] for k in self.param_keys], dtype=np.float64
        )
        self.xu = np.array(
            [self.param_bounds[k][1] for k in self.param_keys], dtype=np.float64
        )

    def _create_individual(self, x: np.ndarray, f: np.ndarray) -> Individual:
        eps = round(float(x[0]), 4)
        decay = round(float(x[1]), 4)
        param_dict = {
            "epsilon": eps,
            "decaying_factor": decay,
            "mu": self.fixed_mu,
            "offline_eps": float(getattr(config.denstream, "offline_eps", 0.40)),
        }

        quality_score = float(-f[0])
        complexity_score = float(f[1])
        return Individual(
            params=param_dict,
            objectives=np.array(f, dtype=np.float32),
            quality_score=quality_score,
            complexity_score=complexity_score,
        )

    def _get_fallback_params(
        self, current_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        if current_params is not None:
            return dict(current_params)
        return {
            "epsilon": config.denstream.epsilon,
            "decaying_factor": config.denstream.decaying_factor,
            "mu": self.fixed_mu,
            "offline_eps": config.denstream.offline_eps,
        }

    def select_compromise_solution(self, pareto_front: List[Individual]) -> Individual:
        """
        Selects the compromise solution from the Pareto front with the
        pseudo-weights method (pymoo): the solution whose pseudo-weight vector
        is closest to equal weights for both objectives.
        """
        if len(pareto_front) == 0:
            return Individual(params=self._get_fallback_params())
        if len(pareto_front) <= 2:
            return pareto_front[0]

        objs = np.array([ind.objectives for ind in pareto_front], dtype=np.float64)
        weights = np.ones(objs.shape[1], dtype=np.float64) / objs.shape[1]

        best_idx = int(PseudoWeights(weights).do(objs))
        return pareto_front[best_idx]

    def evolve(
        self,
        data_buffer: np.ndarray,
        current_params: Optional[Dict[str, float]] = None,
    ) -> Tuple[Individual, List[Individual], List[Dict[str, Any]]]:
        """
        Executes NSGA-II multi-objective evolution on the data buffer.
        """
        t_start = time.perf_counter()

        if len(data_buffer) < self.min_eval_buffer:
            logger.warning(
                f"Data buffer size ({len(data_buffer)}) below minimum evaluation threshold ({self.min_eval_buffer}). "
            )
            fallback_params = self._get_fallback_params(current_params)
            fallback_ind = Individual(params=fallback_params)
            return fallback_ind, [], []

        problem = StreamClusteringOptimizationProblem(
            data_buffer=data_buffer,
            xl=self.xl,
            xu=self.xu,
            n_macro_clusters=self.n_macro_clusters,
            fixed_mu=self.fixed_mu,
            n_samples_init=self.n_samples_init,
        )

        algorithm = NSGA2(
            pop_size=self.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.crossover_rate, eta=self.crossover_eta),
            mutation=PM(prob=self.mutation_rate, eta=self.mutation_eta),
            eliminate_duplicates=True,
        )

        res = minimize(
            problem,
            algorithm,
            termination=("n_gen", self.generations),
            seed=self.seed,
            verbose=False,
        )

        t_end = time.perf_counter()
        opt_latency_ms = round((t_end - t_start) * 1000, 2)

        pareto_front: List[Individual] = []
        if res.X is not None and res.F is not None:
            X_arr = np.atleast_2d(res.X)
            F_arr = np.atleast_2d(res.F)
            for x_val, f_val in zip(X_arr, F_arr):
                pareto_front.append(self._create_individual(x_val, f_val))

        if not pareto_front:
            fallback_params = self._get_fallback_params(current_params)
            compromise = Individual(params=fallback_params)
        else:
            compromise = self.select_compromise_solution(pareto_front)

        history = [
            {
                "generation": self.generations,
                "pareto_front_size": len(pareto_front),
                "avg_quality": float(
                    np.mean([ind.quality_score for ind in pareto_front])
                )
                if pareto_front
                else 0.0,
                "avg_complexity": float(
                    np.mean([ind.complexity_score for ind in pareto_front])
                )
                if pareto_front
                else 0.0,
                "optimization_latency_ms": opt_latency_ms,
            }
        ]

        logger.success(
            f"Pymoo NSGA-II Complete in {opt_latency_ms:.1f}ms | Front Size: {len(pareto_front)} | "
            f"Compromise solution: eps={compromise.params['epsilon']}, "
            f"mu={compromise.params['mu']}, "
            f"decay={compromise.params['decaying_factor']}, "
            f"Quality Surrogate: {compromise.quality_score:.4f}, "
            f"Complexity Proxy: {compromise.complexity_score:.4f}"
        )

        return compromise, pareto_front, history
