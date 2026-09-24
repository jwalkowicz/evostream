"""
Numbers quoted in thesis section 4.2, computed from the per-run CSVs of the
thesis 2 sweep: purity per stream segment, alarms and their signals, the
results per starting epsilon (Table 5) and the effect of the extra swaps.
"""

import glob
import re

import pandas as pd

from experiments.theses.thesis_2.exp_thesis_2_drift import DRIFT_POINT, RESULTS_DIR
from src.core.config import config

SEGMENTS = [(300, DRIFT_POINT), (DRIFT_POINT, 6000), (6000, 9000), (9000, 10000)]


def load_runs():
    runs = {}
    for path in sorted(glob.glob(f"{RESULTS_DIR}/thesis_2_drift_results_eps_*_decay_*.csv")):
        eps, decay = map(float, re.findall(r"eps_([\d.]+)_decay_([\d.]+)\.csv", path)[0])
        runs[(eps, decay)] = pd.read_csv(path)
    return runs


def segment_mean(df, column, start, end):
    return df[(df.sample_idx > start) & (df.sample_idx <= end)][column].mean()


def summarize_run(df):
    alarms = df[df.drift_detected]
    after = alarms[alarms.sample_idx > DRIFT_POINT]
    first = after.iloc[0] if len(after) else None
    signal = None
    if first is not None:
        signal = "centroid" if first.centroid_shift >= config.drift.centroid_shift_threshold else "silhouette"
    row = {
        "alarms_before_change": int((alarms.sample_idx <= DRIFT_POINT).sum()),
        "alarms_after_change": len(after),
        "first_alarm_delay": None if first is None else int(first.sample_idx - DRIFT_POINT),
        "first_alarm_signal": signal,
        "final_eps": df.eps_adapted.iloc[-1],
        "final_decay": df.decay_adapted.iloc[-1],
    }
    for start, end in SEGMENTS:
        row[f"static_{start + 1}_{end}"] = segment_mean(df, "static_purity", start, end)
        row[f"adaptive_{start + 1}_{end}"] = segment_mean(df, "hotswap_purity", start, end)
    for start, end in [(6000, 10000), (6000, 9000), (9000, 10000)]:
        row[f"static_silhouette_{start + 1}_{end}"] = segment_mean(df, "static_silhouette", start, end)
        row[f"adaptive_silhouette_{start + 1}_{end}"] = segment_mean(df, "hotswap_silhouette", start, end)
    return row


def extra_swaps(df, swap_delay=450, window=500):
    """Purity of the adaptive model in the `window` documents before and after
    every swap that followed a second or later alarm after the change."""
    alarms = df[df.drift_detected & (df.sample_idx > DRIFT_POINT)].sample_idx.tolist()[1:]
    out = []
    for alarm in alarms:
        swap = alarm + swap_delay
        before = segment_mean(df, "hotswap_purity", swap - window, swap)
        after = segment_mean(df, "hotswap_purity", swap, swap + window)
        out.append((swap, before, after))
    return out


def main():
    runs = load_runs()
    table = pd.DataFrame([{"eps": e, "decay": d, **summarize_run(df)} for (e, d), df in runs.items()])
    pd.set_option("display.width", 200)

    print(f"Runs: {len(table)}")
    print(f"Alarms before the change: {table.alarms_before_change.sum()}")
    delays = table.first_alarm_delay
    print(f"First alarm after the change: median {delays.median():.0f}, range {delays.min()}-{delays.max()} documents")
    print(f"First-alarm signal: {table.first_alarm_signal.value_counts().to_dict()}")
    print(f"Alarms after the change per run: mean {table.alarms_after_change.mean():.1f}, "
          f"range {table.alarms_after_change.min()}-{table.alarms_after_change.max()}")

    print("\nPurity per stream segment (mean ± std over runs):")
    for start, end in SEGMENTS:
        s, a = table[f"static_{start + 1}_{end}"], table[f"adaptive_{start + 1}_{end}"]
        print(f"  {start + 1:>5}-{end:<5}  static {s.mean():.3f} ± {s.std():.3f} | "
              f"adaptive {a.mean():.3f} ± {a.std():.3f} | adaptive better in {(a > s).sum()}/{len(table)}")
    print("Silhouette (mean over runs):")
    for start, end in [(6000, 10000), (6000, 9000), (9000, 10000)]:
        s, a = table[f"static_silhouette_{start + 1}_{end}"], table[f"adaptive_silhouette_{start + 1}_{end}"]
        print(f"  {start + 1:>5}-{end:<5}  static {s.mean():.3f} | adaptive {a.mean():.3f}")

    print("\nTable 5 - results per starting epsilon (identical for every starting lambda):")
    per_eps = table.groupby("eps")[["static_9001_10000", "adaptive_9001_10000", "first_alarm_delay", "final_eps"]].mean()
    print(per_eps.round(3).to_string())
    print(f"\nFinal epsilon: mean {table.final_eps.mean():.3f} ± {table.final_eps.std():.3f}, "
          f"range {table.final_eps.min():.3f}-{table.final_eps.max():.3f}")

    print("\nExtra swaps (second and later alarms), one run per starting epsilon:")
    for eps in sorted(table.eps.unique()):
        df = runs[(eps, min(d for e, d in runs if e == eps))]
        swaps = [f"{swap}: {b:.2f} -> {a:.2f}" for swap, b, a in extra_swaps(df)]
        print(f"  eps={eps}: {', '.join(swaps) or 'none'}")


if __name__ == "__main__":
    main()
