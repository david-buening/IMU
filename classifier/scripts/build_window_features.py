import math
from pathlib import Path

import pandas as pd


# -----------------------------
# Adjustable parameters
# -----------------------------
IMU_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = CLASSIFIER_ROOT / "results"
INPUT_CSV = IMU_ROOT / "data" / "preprocessed.csv"
OUTPUT_CSV = RESULTS_DIR / "features_windows_1s_all_experiments.csv"

# Set to None to process all reconstructed experiments.
MAX_EXPERIMENTS = None

WINDOW_SIZE_S = 1.0
OVERLAP = 0.50
MIN_SAMPLES_PER_HAND = 30
TIME_REFERENCE_COL = "time_s"  # Use "time_s" or "server_time_s".

# Variant A: add one final full-size window shifted to the phase end.
ADD_SHIFTED_END_WINDOW = True
MIN_SHIFTED_END_WINDOW_START_DIFF_S = WINDOW_SIZE_S / 4

PHASE_ORDER = ["Aufheben", "Laufen", "Absetzen"]
SENSOR_COLS = ["AX", "AY", "AZ", "GX", "GY", "GZ"]
ACC_COLS = ["AX", "AY", "AZ"]
GYRO_COLS = ["GX", "GY", "GZ"]
EPS = 1e-9


def split_contiguous_sensor_runs(df: pd.DataFrame) -> list[pd.DataFrame]:
    """Split preprocessed.csv back into contiguous original sensor files."""
    meta_cols = ["subject", "box_size", "sensor_hand", "surface", "one_handed_carry"]
    runs = []
    start = 0

    for i in range(1, len(df)):
        meta_changed = any(df.at[i, col] != df.at[i - 1, col] for col in meta_cols)
        time_reset = df.at[i, "time_s"] < df.at[i - 1, "time_s"]

        if meta_changed or time_reset:
            runs.append(df.iloc[start:i].copy())
            start = i

    runs.append(df.iloc[start:].copy())
    return runs


def pair_left_right_runs(runs: list[pd.DataFrame]) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Pair neighboring L/R sensor files that belong to the same experiment."""
    experiments = []
    i = 0

    while i < len(runs) - 1:
        first = runs[i]
        second = runs[i + 1]

        shared_cols = ["subject", "box_size", "surface", "one_handed_carry"]
        same_experiment = all(first[col].iloc[0] == second[col].iloc[0] for col in shared_cols)
        different_hands = {first["sensor_hand"].iloc[0], second["sensor_hand"].iloc[0]} == {"L", "R"}

        if same_experiment and different_hands:
            left = first if first["sensor_hand"].iloc[0] == "L" else second
            right = second if second["sensor_hand"].iloc[0] == "R" else first
            experiments.append((left.copy(), right.copy()))
            i += 2
        else:
            print("Could not pair run, skipping:")
            print(first[["subject", "box_size", "sensor_hand", "surface", "one_handed_carry"]].iloc[0].to_dict())
            i += 1

    return experiments


def add_derived_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(TIME_REFERENCE_COL).copy()
    df["acc_mag"] = (df["AX"] ** 2 + df["AY"] ** 2 + df["AZ"] ** 2).apply(math.sqrt)
    df["gyro_mag"] = (df["GX"] ** 2 + df["GY"] ** 2 + df["GZ"] ** 2).apply(math.sqrt)

    acc_diff = df[ACC_COLS].diff().fillna(0.0)
    gyro_diff = df[GYRO_COLS].diff().fillna(0.0)
    df["acc_jerk_mag"] = ((acc_diff**2).sum(axis=1)).apply(math.sqrt)
    df["gyro_jerk_mag"] = ((gyro_diff**2).sum(axis=1)).apply(math.sqrt)
    return df


def make_phase_windows(phase_start: float, phase_end: float) -> list[tuple[float, float]]:
    """Create windows inside one phase only, including a shifted final window."""
    step_s = WINDOW_SIZE_S * (1.0 - OVERLAP)
    duration = phase_end - phase_start

    if duration <= 0:
        return []

    if duration <= WINDOW_SIZE_S:
        return [(phase_start, phase_end)]

    starts = []
    current = phase_start
    while current + WINDOW_SIZE_S <= phase_end + EPS:
        starts.append(round(current, 6))
        current += step_s

    if ADD_SHIFTED_END_WINDOW:
        final_start = round(phase_end - WINDOW_SIZE_S, 6)
        if not starts or abs(starts[-1] - final_start) >= MIN_SHIFTED_END_WINDOW_START_DIFF_S:
            starts.append(final_start)

    deduped_starts = []
    for start in starts:
        if not deduped_starts or abs(deduped_starts[-1] - start) > EPS:
            deduped_starts.append(start)

    return [(start, start + WINDOW_SIZE_S) for start in deduped_starts]


def basic_stats(series: pd.Series) -> dict[str, float]:
    values = [float(value) for value in series]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    minimum = min(values)
    maximum = max(values)
    return {
        "mean": mean,
        "std": math.sqrt(variance),
        "min": minimum,
        "max": maximum,
        "range": maximum - minimum,
        "energy": sum(value**2 for value in values) / len(values),
    }


def extract_window_features(left_window: pd.DataFrame, right_window: pd.DataFrame) -> dict[str, float]:
    features = {}
    derived_cols = ["acc_mag", "gyro_mag", "acc_jerk_mag", "gyro_jerk_mag"]

    for side, window in [("L", left_window), ("R", right_window)]:
        for col in SENSOR_COLS:
            stats = basic_stats(window[col])
            features[f"{side}_{col}_mean"] = stats["mean"]
            features[f"{side}_{col}_std"] = stats["std"]

        for col in derived_cols:
            stats = basic_stats(window[col])
            for stat_name, value in stats.items():
                features[f"{side}_{col}_{stat_name}"] = value

        features[f"{side}_n_samples"] = int(len(window))

    for col in derived_cols:
        for stat_name in ["mean", "std", "range", "energy"]:
            left_value = features[f"L_{col}_{stat_name}"]
            right_value = features[f"R_{col}_{stat_name}"]
            features[f"{col}_{stat_name}_absdiff"] = abs(left_value - right_value)
            features[f"{col}_{stat_name}_ratio_L_over_R"] = left_value / (right_value + EPS)

    for col in SENSOR_COLS:
        left_std = features[f"L_{col}_std"]
        right_std = features[f"R_{col}_std"]
        features[f"{col}_std_absdiff"] = abs(left_std - right_std)
        features[f"{col}_std_ratio_L_over_R"] = left_std / (right_std + EPS)

    return features


def build_feature_rows(left: pd.DataFrame, right: pd.DataFrame, experiment_id: int) -> list[dict]:
    rows = []
    left = add_derived_signals(left)
    right = add_derived_signals(right)

    meta = left[["subject", "box_size", "surface", "one_handed_carry"]].iloc[0].to_dict()

    print(f"\nExperiment {experiment_id}")
    print(meta)

    for phase in PHASE_ORDER:
        left_phase = left[left["phase"] == phase]
        right_phase = right[right["phase"] == phase]

        if left_phase.empty or right_phase.empty:
            print(f"  {phase}: skipped, phase missing on one hand")
            continue

        # Use the shared time range so every window can contain both hands.
        # time_s = watch-local recording time since Start.
        # server_time_s = laptop receive time.
        phase_start = max(left_phase[TIME_REFERENCE_COL].min(), right_phase[TIME_REFERENCE_COL].min())
        phase_end = min(left_phase[TIME_REFERENCE_COL].max(), right_phase[TIME_REFERENCE_COL].max())
        windows = make_phase_windows(phase_start, phase_end)

        print(
            f"  {phase}: shared {TIME_REFERENCE_COL} {phase_start:.3f}-{phase_end:.3f} "
            f"({phase_end - phase_start:.3f}s), candidate windows: {len(windows)}"
        )

        kept = 0
        for window_index, (start_s, end_s) in enumerate(windows, start=1):
            left_window = left_phase[
                (left_phase[TIME_REFERENCE_COL] >= start_s) & (left_phase[TIME_REFERENCE_COL] <= end_s)
            ]
            right_window = right_phase[
                (right_phase[TIME_REFERENCE_COL] >= start_s) & (right_phase[TIME_REFERENCE_COL] <= end_s)
            ]

            if len(left_window) < MIN_SAMPLES_PER_HAND or len(right_window) < MIN_SAMPLES_PER_HAND:
                print(
                    f"    skip window {window_index}: {start_s:.3f}-{end_s:.3f}, "
                    f"L={len(left_window)}, R={len(right_window)}"
                )
                continue

            kept += 1
            print(
                f"    keep window {window_index}: {start_s:.3f}-{end_s:.3f}, "
                f"L={len(left_window)}, R={len(right_window)}"
            )

            row = {
                "experiment_id": experiment_id,
                "phase": phase,
                "subject": meta["subject"],
                "box_size": meta["box_size"],
                "surface": meta["surface"],
                "one_handed_carry": bool(meta["one_handed_carry"]),
                "time_reference": TIME_REFERENCE_COL,
                "window_index_in_phase": window_index,
                "window_start_s": round(start_s, 3),
                "window_end_s": round(end_s, 3),
            }
            row.update(extract_window_features(left_window, right_window))
            rows.append(row)

        print(f"    kept windows: {kept}")

    return rows


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    runs = split_contiguous_sensor_runs(df)
    experiments = pair_left_right_runs(runs)

    if MAX_EXPERIMENTS is not None:
        experiments = experiments[:MAX_EXPERIMENTS]

    all_rows = []
    for experiment_id, (left, right) in enumerate(experiments, start=1):
        all_rows.extend(build_feature_rows(left, right, experiment_id))

    features = pd.DataFrame(all_rows)
    features.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(features)} rows and {len(features.columns)} columns to:")
    print(f"  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
