"""
Runs the (epsilon x decaying_factor) sweep for the thesis-2 drift experiment
directly in-process - no shelling out to `uv run python -m ... --eps ...`
per combo - then regenerates all plots.

Concurrency is via a thread pool calling run_drift_experiment() as a normal
Python function. Note: run_drift_experiment() calls set_seed(42) at the top
of each run, which reseeds the shared global numpy/random/torch RNG state.
With more than one worker thread, two runs can genuinely interleave on that
shared state, so which random draws each run consumes is timing-dependent -
harmless for a quick sanity check, but for a final, reproducible full sweep
prefer running with --workers 1, or move to separate processes (each has its
own isolated RNG state) once the experiment logic itself is settled.
"""

import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from experiments.theses.thesis_2.exp_thesis_2_drift import (
    SWEEP_DECAY_VALUES,
    SWEEP_EPSILON_VALUES,
    run_drift_experiment,
)
from experiments.theses.thesis_2.plot_thesis_2_drift import main as generate_all_plots
from src.core.logger import logger

# Small subset for validating the pipeline before committing to the full grid:
# the default combo, plus the opposite corner of the (in-bounds) grid.
QUICK_COMBOS: List[Tuple[float, float]] = [(0.10, 0.005), (0.15, 0.08)]

DEFAULT_WORKERS = 4


def run_combo(eps: float, decay: float) -> None:
    logger.info(f"[eps={eps}, decay={decay}] starting...")
    run_drift_experiment(initial_eps=eps, initial_decay=decay)
    logger.info(f"[eps={eps}, decay={decay}] done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--quick", action="store_true",
        help="Run only the small QUICK_COMBOS validation set instead of the full 5x5 grid.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    logger.add("logs/thesis_2_drift.log", rotation="500 MB")

    combos = QUICK_COMBOS if args.quick else list(itertools.product(SWEEP_EPSILON_VALUES, SWEEP_DECAY_VALUES))
    print(f"Running {len(combos)} combination(s) with {args.workers} worker thread(s)...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_combo, eps, decay): (eps, decay) for eps, decay in combos}
        for future in as_completed(futures):
            eps, decay = futures[future]
            future.result()  # re-raises on failure, with the (eps, decay) known above

    print("\nAll simulations finished.")

    if not args.skip_plots:
        print("Generating plots...")
        generate_all_plots()
        print("Plots written to experiments/theses/thesis_2/results/")


if __name__ == "__main__":
    main()
