import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river.cluster import DenStream

MU = 2
BETA = 0.75
EPSILON = 0.10
DECAYING_FACTOR = 0.005
STREAM_SPEED = 1
N_SAMPLES_LIMIT = 1

LAMBDAS = [
    0.001,
    0.0025,
    0.005,
    0.01,
    0.02,
    0.04,
    0.08,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.25,
    0.30,
]


def main():
    data = np.load("experiments/param_bounds/common/shared_data_16d.npz")
    X_16d = data["X_16d"]
    block_sizes = data["block_sizes"]

    block_boundaries = {
        "A_end": int(block_sizes[0]),
        "B_start": int(block_sizes[0]),
        "B_end": int(block_sizes[0] + block_sizes[1]),
        "C_start": int(block_sizes[0] + block_sizes[1]),
    }

    print("Theoretical T_p:")
    for lam in LAMBDAS:
        tp = math.ceil((1 / lam) * math.log((MU * BETA) / (MU * BETA - 1)))
        for N_example in [5, 10, 50]:
            if N_example > MU * BETA:
                decay_time = math.log2(N_example / (MU * BETA)) / lam
            else:
                decay_time = 0
            print(
                f"lambda={lam:.4f}: T_p={tp:>6}, P-MC with N={N_example} decays in {decay_time:.0f} ticks"
            )

    results = []
    history_all = {}
    weight_traces = {}

    for lam in LAMBDAS:
        print(f"Running lambda = {lam}...")
        ds = DenStream(
            epsilon=EPSILON,
            beta=BETA,
            mu=MU,
            decaying_factor=lam,
            n_samples_init=N_SAMPLES_LIMIT,
            stream_speed=STREAM_SPEED,
        )

        history = []
        prev_pmc_ids = set()
        prev_omc_ids = set()
        promotions = 0
        deletions_pmc = 0
        deletions_omc = 0
        mc_weights = {}

        for t, vector in enumerate(X_16d):
            x_dict = {i: val for i, val in enumerate(vector)}
            ds.learn_one(x_dict)

            curr_pmc_ids = set(ds.p_micro_clusters.keys())
            curr_omc_ids = set(ds.o_micro_clusters.keys())

            missing_omcs = prev_omc_ids - curr_omc_ids
            for oid in missing_omcs:
                if oid in curr_pmc_ids:
                    promotions += 1
                else:
                    deletions_omc += 1

            missing_pmcs = prev_pmc_ids - curr_pmc_ids
            for pid in missing_pmcs:
                if pid not in curr_omc_ids:
                    deletions_pmc += 1

            for pid, pmc in ds.p_micro_clusters.items():
                w = pmc.calc_weight(ds.timestamp)
                if pid not in mc_weights:
                    mc_weights[pid] = []
                mc_weights[pid].append((t, w))

            history.append(
                {
                    "t": t,
                    "n_pmc": len(curr_pmc_ids),
                    "n_omc": len(curr_omc_ids),
                    "n_total": len(curr_pmc_ids) + len(curr_omc_ids),
                    "timestamp": ds.timestamp,
                    "promotions_cumul": promotions,
                    "del_pmc_cumul": deletions_pmc,
                    "del_omc_cumul": deletions_omc,
                }
            )

            prev_pmc_ids = curr_pmc_ids
            prev_omc_ids = curr_omc_ids

        history_all[lam] = history
        weight_traces[lam] = mc_weights

        totals = [h["n_total"] for h in history]
        results.append(
            {
                "Lambda": lam,
                "Końcowe P-MC": len(curr_pmc_ids),
                "Końcowe O-MC": len(curr_omc_ids),
                "Usunięte O-MC": deletions_omc,
                "Usunięte P-MC": deletions_pmc,
                "Promocje (O->P)": promotions,
                "Śr. Liczba MC": round(np.mean(totals), 1),
                "Max Liczba MC": max(totals),
            }
        )

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(
        "experiments/param_bounds/lambda/results/lambda_detailed_proof.csv", index=False
    )

    # Charts
    repr_LAMBDAS = [0.005, 0.08, 0.16, 0.30]
    fig, axes = plt.subplots(
        len(repr_LAMBDAS), 1, figsize=(12, 3.5 * len(repr_LAMBDAS)), sharex=True
    )
    for idx, lam in enumerate(repr_LAMBDAS):
        h = history_all[lam]
        ts = [d["t"] for d in h]
        axes[idx].plot(ts, [d["n_pmc"] for d in h], label="P-MC", color="#2980b9")
        axes[idx].plot(
            ts, [d["n_omc"] for d in h], label="O-MC", color="#e67e22", alpha=0.8
        )
        axes[idx].axvline(
            x=block_boundaries["A_end"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Koniec bloku A",
        )
        axes[idx].axvline(
            x=block_boundaries["B_end"],
            color="green",
            linestyle="--",
            alpha=0.5,
            label="Koniec bloku B",
        )
        axes[idx].set_ylabel("Liczba MC")
        axes[idx].set_title(f"λ = {lam}")
        axes[idx].legend(loc="upper right", fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Numer dokumentu w strumieniu")
    plt.suptitle("Ewolucja mikroklastrów w czasie (abrupt drift A→B→C)", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        "experiments/param_bounds/lambda/results/lambda_timeseries.png", dpi=300
    )

    trace_LAMBDAS = [0.005, 0.08, 0.16, 0.30]
    fig, axes = plt.subplots(
        len(trace_LAMBDAS), 1, figsize=(12, 3.5 * len(trace_LAMBDAS)), sharex=True
    )
    for idx, lam in enumerate(trace_LAMBDAS):
        traces = weight_traces[lam]
        sorted_mcs = sorted(traces.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]
        for mc_id, wt_list in sorted_mcs:
            ts_w = [x[0] for x in wt_list]
            ws = [x[1] for x in wt_list]
            axes[idx].plot(ts_w, ws, label=f"MC {mc_id}", alpha=0.8)
        axes[idx].axvline(
            x=block_boundaries["A_end"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Koniec A",
        )
        axes[idx].axvline(
            x=block_boundaries["B_end"],
            color="green",
            linestyle="--",
            alpha=0.5,
            label="Koniec B",
        )
        axes[idx].set_ylabel("Waga MC")
        axes[idx].set_title(f"Wagi mikroklastrów (λ = {lam})")
        axes[idx].legend(loc="upper right", fontsize=7)
        axes[idx].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Numer dokumentu w strumieniu")
    plt.suptitle("Wygaszanie wag mikroklastrów po zmianie tematu", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        "experiments/param_bounds/lambda/results/lambda_weight_traces.png", dpi=300
    )

    fig, axes = plt.subplots(
        len(repr_LAMBDAS), 1, figsize=(12, 3.5 * len(repr_LAMBDAS)), sharex=True
    )
    for idx, lam in enumerate(repr_LAMBDAS):
        h = history_all[lam]
        ts = [d["t"] for d in h]
        axes[idx].plot(
            ts, [d["del_pmc_cumul"] for d in h], label="Usunięte P-MC", color="#c0392b"
        )
        axes[idx].plot(
            ts, [d["del_omc_cumul"] for d in h], label="Usunięte O-MC", color="#e67e22"
        )
        axes[idx].plot(
            ts,
            [d["promotions_cumul"] for d in h],
            label="Promocje O→P",
            color="#27ae60",
        )
        axes[idx].axvline(
            x=block_boundaries["A_end"], color="red", linestyle="--", alpha=0.3
        )
        axes[idx].axvline(
            x=block_boundaries["B_end"], color="green", linestyle="--", alpha=0.3
        )
        axes[idx].set_ylabel("Skumulowana liczba")
        axes[idx].set_title(f"λ = {lam}")
        axes[idx].legend(loc="upper left", fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Numer dokumentu w strumieniu")
    plt.suptitle(
        "Skumulowane zdarzenia: usunięcia i promocje mikroklastrów", fontsize=13
    )
    plt.tight_layout()
    plt.savefig("experiments/param_bounds/lambda/results/lambda_events.png", dpi=300)

    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(df))
    width = 0.25
    ax.bar(
        [p - width for p in x_pos],
        df["Końcowe P-MC"],
        width,
        label="Końcowe P-MC",
        color="#2980b9",
    )
    ax.bar(x_pos, df["Usunięte P-MC"], width, label="Usunięte P-MC", color="#c0392b")
    ax.bar(
        [p + width for p in x_pos],
        df["Usunięte O-MC"],
        width,
        label="Usunięte O-MC",
        color="#e67e22",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(l) for l in df["Lambda"]], rotation=45)
    ax.set_xlabel("Wartość λ")
    ax.set_ylabel("Liczba mikroklastrów")
    ax.set_title("Podsumowanie struktury mikroklastrów dla różnych wartości λ")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        "experiments/param_bounds/lambda/results/lambda_summary_bar.png", dpi=300
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        df["Lambda"],
        df["Śr. Liczba MC"],
        marker="o",
        label="Średnia liczba MC",
        color="#2980b9",
    )
    ax.plot(
        df["Lambda"],
        df["Max Liczba MC"],
        marker="s",
        label="Maksymalna liczba MC",
        color="#c0392b",
    )
    ax.set_xlabel("Wartość λ")
    ax.set_ylabel("Liczba mikroklastrów")
    ax.set_title("Średnia i maksymalna liczba mikroklastrów w funkcji λ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiments/param_bounds/lambda/results/lambda_mc_count.png", dpi=300)

    a_end = block_boundaries["A_end"]
    checkpoints = [10, 25, 50, 100, 200, 500, 1000]

    decay_results = []
    for lam in LAMBDAS:
        traces = weight_traces[lam]
        mcs_at_a_end = {}
        for mc_id, wt_list in traces.items():
            t_to_w = {t: w for t, w in wt_list}
            if a_end - 1 in t_to_w:
                mcs_at_a_end[mc_id] = t_to_w

        if not mcs_at_a_end:
            decay_results.append(
                {"Lambda": lam, "MCs at A_end": 0, "Survived to stream end": "N/A"}
            )
            continue

        survived_count = 0
        row = {"Lambda": lam, "MCs at A_end": len(mcs_at_a_end)}

        best_mc_id = max(
            mcs_at_a_end.keys(), key=lambda mid: mcs_at_a_end[mid].get(a_end - 1, 0)
        )
        best_trace = mcs_at_a_end[best_mc_id]
        w_at_a = best_trace.get(a_end - 1, 0)
        row["W at A_end"] = round(w_at_a, 2)

        for cp in checkpoints:
            t_check = a_end + cp
            if t_check in best_trace:
                row[f"W at +{cp}"] = round(best_trace[t_check], 2)
            else:
                row[f"W at +{cp}"] = "removed"

        stream_end = len(X_16d) - 1
        for mc_id, t_to_w in mcs_at_a_end.items():
            if stream_end in t_to_w:
                survived_count += 1
        row["Survived to end"] = survived_count

        # Theoretical check
        if w_at_a > MU * BETA:
            theoretical_decay = math.log2(w_at_a / (MU * BETA)) / lam
            row["Theoretical decay (ticks)"] = round(theoretical_decay, 0)
        else:
            row["Theoretical decay (ticks)"] = 0

        decay_results.append(row)

    df_decay = pd.DataFrame(decay_results)
    print(df_decay.to_string(index=False))
    df_decay.to_csv(
        "experiments/param_bounds/lambda/results/lambda_decay_analysis.csv", index=False
    )


if __name__ == "__main__":
    main()
