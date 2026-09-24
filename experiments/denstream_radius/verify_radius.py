"""Compares river's micro-cluster radius and the corrected one with the radius computed from the points."""

import numpy as np
import pandas as pd
from river.cluster.denstream import DenStreamMicroCluster
from scipy.stats import ortho_group
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import normalize

from experiments.theses.thesis_2.exp_thesis_2_drift import (
    EMBEDDINGS_CACHE_PATH,
    INITIAL_WARMUP_SIZE,
    SAMPLES_PER_PHASE,
)
from src.domain.clustering import set_river_radius_fix

RESULTS_PATH = "experiments/denstream_radius/results/radius_verification.csv"
GROUP_SIZE = 20


def micro_cluster_of(points: np.ndarray) -> DenStreamMicroCluster:
    mc = DenStreamMicroCluster(x=dict(enumerate(points[0])), timestamp=0, decaying_factor=0.01)
    for p in points[1:]:
        mc.insert(dict(enumerate(p)), timestamp=0)
    return mc


def brute_force_radius(points: np.ndarray) -> float:
    return float(np.sqrt(((points - points.mean(axis=0)) ** 2).sum(axis=1).mean()))


def compare(check: str, case: str, points: np.ndarray) -> dict:
    mc = micro_cluster_of(points)
    set_river_radius_fix(False)
    river_radius = mc.calc_radius(timestamp=0)
    set_river_radius_fix(True)
    fixed_radius = mc.calc_radius(timestamp=0)
    return {
        "check": check,
        "case": case,
        "n_points": len(points),
        "dim": points.shape[1],
        "brute_force_radius": brute_force_radius(points),
        "river_radius": river_radius,
        "fixed_radius": fixed_radius,
    }


def main():
    rng = np.random.default_rng(0)
    rows = []

    for centre in [(0, 0), (1000, 0), (0, 1000), (1000, 1000), (-23000, 1700)]:
        cloud = rng.normal(centre, 1.0, size=(2000, 2))
        rows.append(compare("A_issue_2004", f"centre={centre}", cloud))

    embeddings = np.load(EMBEDDINGS_CACHE_PATH)[:SAMPLES_PER_PHASE]
    ipca = IncrementalPCA(n_components=16).fit(embeddings[:INITIAL_WARMUP_SIZE])
    projected = normalize(ipca.transform(embeddings))
    group = projected[rng.choice(len(projected), GROUP_SIZE, replace=False)]
    padded = np.hstack([group, np.zeros((GROUP_SIZE, 384 - 16))])
    rows.append(compare("B_rotation", "16d", group))
    rows.append(compare("B_rotation", "16d rotated", group @ ortho_group.rvs(16, random_state=1)))
    rows.append(compare("B_rotation", "zero-padded to 384d", padded))
    rows.append(compare("B_rotation", "zero-padded to 384d, rotated", padded @ ortho_group.rvs(384, random_state=2)))

    for i in range(5):
        docs = embeddings[rng.choice(len(embeddings), GROUP_SIZE, replace=False)]
        rows.append(compare("C_raw_sbert", f"random group {i + 1}", docs))

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
