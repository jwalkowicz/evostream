import numpy as np
import pandas as pd
from river.cluster import DenStream

MU = 2
BETA = 0.75
DECAYING_FACTOR = 0.005
N_SAMPLES_INIT = 1
STREAM_SPEED = 1

EPSILONS = [
    0.02,
    0.05,
    0.08,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]


def main():
    data = np.load("experiments/param_bounds/common/shared_data_16d.npz")
    X_16d = data["X_16d"]
    block_sizes = data["block_sizes"]
    print(f"Shape: {X_16d.shape}, Block sizes: {block_sizes.tolist()}")

    results = []

    print(
        f"\nFixed parameters: mu={MU}, beta={BETA}, lambda={DECAYING_FACTOR}, "
        f"n_samples_init={N_SAMPLES_INIT}, stream_speed={STREAM_SPEED}"
    )

    for eps in EPSILONS:
        ds = DenStream(
            epsilon=eps,
            mu=MU,
            beta=BETA,
            decaying_factor=DECAYING_FACTOR,
            n_samples_init=N_SAMPLES_INIT,
            stream_speed=STREAM_SPEED,
        )

        for vector in X_16d:
            x_dict = {i: val for i, val in enumerate(vector)}
            ds.learn_one(x_dict)

        num_pmc = len(ds.p_micro_clusters)
        num_omc = len(ds.o_micro_clusters)

        results.append(
            {"Epsilon": eps, "P-Microclusters": num_pmc, "O-Microclusters": num_omc}
        )

        print(f"Eps: {eps:.2f} | P-MC: {num_pmc:>3} | O-MC: {num_omc:>3}")

    df = pd.DataFrame(results)
    csv_path = "experiments/param_bounds/epsilon/results/epsilon_bounds_proof.csv"
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
