import time
import warnings
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

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class Individual:
    """Represents a candidate parameter configuration in the population."""

    params: Dict[str, float]
    objectives: np.ndarray = field(default_factory=lambda: np.zeros(2))
    rank: int = 0
    crowding_dist: float = 0.0
    quality_score: float = 0.0
    complexity_score: float = 0.0

    def dominates(self, other: "Individual") -> bool:
        """Pareto domination test (Minimization for both objectives)."""
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


class StreamClusteringOptimizationProblem(ElementwiseProblem):
    """
    Pymoo multi-objective optimization problem for streaming text clustering.
    Strictly 2D Chromosome: theta = (epsilon, decaying_factor) in R^2.

    Objectives:
      1. f1(theta) = -Q_tilde(theta)
      2. f2(theta) = C_struct(theta)
    """

    def __init__(
        self,
        data_buffer: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
        fixed_mu: int = 3,
        n_samples_init: int = 30,
        labels_buffer: Optional[List[Any]] = None,
    ):
        super().__init__(n_var=2, n_obj=2, xl=xl, xu=xu)
        self.data_buffer = data_buffer
        self.fixed_mu = fixed_mu
        self.n_samples_init = n_samples_init
        self.labels_buffer = labels_buffer
        self.dict_buffer = [dict(enumerate(row)) for row in data_buffer]

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            eps = float(x[0])
            decay = float(x[1])
            mu = self.fixed_mu

            model = cluster.DenStream(
                epsilon=eps,
                mu=mu,
                decaying_factor=decay,
                beta=config.denstream.beta,
                n_samples_init=self.n_samples_init,
            )
            for row in self.dict_buffer:
                model.learn_one(row)
                model.predict_one(row)

            p_mcs = getattr(model, "p_micro_clusters", {})
            clusters = getattr(model, "clusters", {})
            n_micro = len(p_mcs)
            n_macro = len(clusters)
            t = getattr(model, "timestamp", 0)

            quality_score = -1.0
            complexity_score = 1.0

            if n_micro >= 2:
                centers = []
                macro_labels = []

                for macro_id, mc_list in clusters.items():
                    for mc in mc_list:
                        c_dict = mc.calc_center(t)
                        centers.append([c_dict[i] for i in range(len(c_dict))])
                        macro_labels.append(macro_id)

                centers_np = np.array(centers, dtype=np.float32)

                if 2 <= n_macro < len(centers_np):
                    try:
                        quality_score = float(
                            silhouette_score(centers_np, macro_labels)
                        )
                    except Exception:
                        quality_score = 0.0
                    ratio = float(n_micro / max(n_macro, 1))
                    norm_micro_count = float(n_micro / max(len(self.data_buffer), 1))
                    complexity_score = norm_micro_count + 0.05 * np.log1p(ratio)
                else:
                    quality_score = -0.5
                    complexity_score = 0.8
            else:
                quality_score = -1.0
                complexity_score = 1.0

            out["F"] = [-quality_score, complexity_score]
        except Exception:
            # Fallback penalty for bad parameters
            out["F"] = [1.0, 1.0]


class NSGAIIOptimizer:
    """
    Pymoo-backed Non-dominated Sorting Genetic Algorithm II (NSGA-II) for multi-criteria
    optimization.
    """

    def __init__(
        self,
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

        logger.info(
            f"NSGA-II (Pymoo Engine) initialized: pop_size={population_size}, generations={generations}, "
            f"fixed mu={fixed_mu}, n_samples_init={n_samples_init}, min_eval_buffer={min_eval_buffer}"
        )

    def compute_divergence(
        self,
        current_params: Dict[str, float],
        new_params: Dict[str, float],
    ) -> float:
        """
        Computes normalized parametric divergence between
        current parameters and newly proposed NSGA-II parameters.
        """
        sum_sq = 0.0
        n_params = 0
        for k in ["epsilon", "decaying_factor"]:
            if k in self.param_bounds and k in current_params and k in new_params:
                low, high = self.param_bounds[k]
                rng = max(high - low, 1e-6)
                diff = (float(new_params[k]) - float(current_params[k])) / rng
                sum_sq += diff**2
                n_params += 1
        return float(np.sqrt(sum_sq / max(n_params, 1)))

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

    def detect_knee_point(self, pareto_front: List[Individual]) -> Individual:
        """
        Selects the balanced compromise Knee Point solution on the non-dominated Pareto front.
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
        labels_buffer: Optional[List[Any]] = None,
        current_params: Optional[Dict[str, float]] = None,
    ) -> Tuple[Individual, List[Individual], List[Dict[str, Any]]]:
        """
        Executes NSGA-II multi-objective evolution on the data buffer.
        Returns:
            - best_knee_individual: Compromise Knee Point solution for self-adaptation.
            - pareto_front: Set of non-dominated individuals on Front 0.
            - history: Log of convergence statistics and optimization wall-clock latency.
        """
        t_start = time.perf_counter()

        if len(data_buffer) < self.min_eval_buffer:
            logger.warning(
                f"Data buffer size ({len(data_buffer)}) below minimum evaluation threshold ({self.min_eval_buffer}). "
                f"Maintaining status quo operational parameters."
            )
            fallback_params = self._get_fallback_params(current_params)
            fallback_ind = Individual(params=fallback_params)
            return fallback_ind, [], []

        problem = StreamClusteringOptimizationProblem(
            data_buffer=data_buffer,
            xl=self.xl,
            xu=self.xu,
            fixed_mu=self.fixed_mu,
            n_samples_init=self.n_samples_init,
            labels_buffer=labels_buffer,
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

        # Extract Pareto front
        pareto_front: List[Individual] = []
        if res.X is not None and res.F is not None:
            X_arr = np.atleast_2d(res.X)
            F_arr = np.atleast_2d(res.F)
            for x_val, f_val in zip(X_arr, F_arr):
                pareto_front.append(self._create_individual(x_val, f_val))

        if not pareto_front:
            fallback_params = self._get_fallback_params(current_params)
            best_knee_point = Individual(params=fallback_params)
        else:
            best_knee_point = self.detect_knee_point(pareto_front)

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
            f"Knee Point: eps={best_knee_point.params['epsilon']}, "
            f"mu={best_knee_point.params['mu']}, "
            f"decay={best_knee_point.params['decaying_factor']}, "
            f"Quality Surrogate: {best_knee_point.quality_score:.4f}, "
            f"Complexity Proxy: {best_knee_point.complexity_score:.4f}"
        )

        return best_knee_point, pareto_front, history
