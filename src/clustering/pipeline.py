"""
Main feature-extraction pipeline.

Reads  : data/preprocessed.csv
Writes :
  data/features/feature_matrix_single.csv  — one row per window per sensor
  data/features/feature_matrix_paired.csv  — one row per window, both wrists

Run from project root:
    python src/clustering/pipeline.py

Steps
-----
1. Load preprocessed.csv
2. Assign trial_ids (detect time_s resets in `both`-condition groups)
3. Extract single-sensor windows and compute time-domain features  → single CSV
4. Align L/R streams per trial, extract paired windows, compute time-domain +
   asymmetry features                                               → paired CSV
5. Sanity checks: report NaN/inf counts and final shapes
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features.windows    import assign_trial_ids, extract_windows, SENSOR_COLS
from features.align      import align_all_trials, SENSOR_COLS as ALIGN_SENSOR_COLS
from features.time_domain import compute_features
from features.asymmetry  import compute_asymmetry


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR    = PROJECT_ROOT / "data"
INPUT_CSV   = DATA_DIR / "preprocessed.csv"
FEATURES_DIR = DATA_DIR / "features"
OUT_SINGLE  = FEATURES_DIR / "feature_matrix_single.csv"
OUT_PAIRED  = FEATURES_DIR / "feature_matrix_paired.csv"


# ---------------------------------------------------------------------------
# Helper: lookup phase for a server-time interval
# ---------------------------------------------------------------------------

def _phase_for_window(
    source_df: pd.DataFrame,
    t_start: float,
    t_end: float,
    subject: str,
    box_size: str,
    surface: str,
    carrying_hand: str,
    trial_id: int,
) -> str:
    """Return the majority phase label for rows in a given time range."""
    mask = (
        (source_df["subject"]       == subject) &
        (source_df["box_size"]      == box_size) &
        (source_df["surface"]       == surface) &
        (source_df["carrying_hand"] == carrying_hand) &
        (source_df["trial_id"]      == trial_id) &
        (source_df["server_time_s"] >= t_start) &
        (source_df["server_time_s"] <= t_end)
    )
    phases = source_df.loc[mask, "phase"]
    if phases.empty:
        return "unknown"
    return phases.value_counts().idxmax()


# ---------------------------------------------------------------------------
# Single-sensor feature matrix
# ---------------------------------------------------------------------------

def build_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-domain features for each window of each single wrist sensor.

    Returns a DataFrame with one row per (window, sensor_hand).
    """
    print("  Extracting single-sensor windows …")
    windows = extract_windows(df)
    print(f"  → {len(windows):,} windows extracted")

    rows = []
    for w in windows:
        feats = compute_features(w["data"])

        row = {
            "subject":        w["subject"],
            "box_size":       w["box_size"],
            "sensor_hand":    w["sensor_hand"],
            "surface":        w["surface"],
            "carrying_hand":  w["carrying_hand"],
            "one_handed_carry": w["one_handed_carry"],
            "trial_id":       w["trial_id"],
            "win_idx":        w["win_idx"],
            "phase":          w["phase"],
            "start_t":        w["start_t"],
            "end_t":          w["end_t"],
            "n_samples":      w["n_samples"],
        }
        row.update(feats)
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired feature matrix
# ---------------------------------------------------------------------------

def build_paired(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each trial, align L and R, slide windows over the aligned timeline,
    compute features for both sides, then compute asymmetry features.

    Returns a DataFrame with one row per aligned window (both wrists combined).
    """
    print("  Aligning L/R streams …")
    aligned_trials = align_all_trials(df)
    print(f"  → {len(aligned_trials)} aligned trials")

    rows = []
    trial_group_cols = ["subject", "box_size", "surface", "carrying_hand", "trial_id"]

    for trial_key, aligned in aligned_trials.items():
        meta = dict(zip(trial_group_cols, trial_key))

        n = len(aligned)
        step   = 50
        window = 100
        win_idx = 0

        start = 0
        while start < n:
            end   = start + window
            chunk = aligned.iloc[start : min(end, n)]
            if len(chunk) < 50:
                break

            t_start = chunk["server_time_s"].iloc[0]
            t_end   = chunk["server_time_s"].iloc[-1]

            # extract raw arrays for each side
            l_cols = [f"{ax}_L" for ax in ALIGN_SENSOR_COLS]
            r_cols = [f"{ax}_R" for ax in ALIGN_SENSOR_COLS]
            data_L = chunk[l_cols].to_numpy(dtype=float)
            data_R = chunk[r_cols].to_numpy(dtype=float)

            # time-domain features for each side
            feats_L = compute_features(data_L)
            feats_R = compute_features(data_R)

            # asymmetry features
            feats_asym = compute_asymmetry(feats_L, feats_R, data_L, data_R)

            # look up the majority phase from the original (L) stream
            phase = _phase_for_window(
                df,
                t_start, t_end,
                meta["subject"], meta["box_size"], meta["surface"],
                meta["carrying_hand"], meta["trial_id"],
            )

            row = {
                **meta,
                "one_handed_carry": meta["carrying_hand"] != "both",
                "win_idx":  win_idx,
                "phase":    phase,
                "start_t":  t_start,
                "end_t":    t_end,
                "n_samples": len(chunk),
            }
            # prefix L and R feature dicts to distinguish them
            row.update({f"L_{k}": v for k, v in feats_L.items()})
            row.update({f"R_{k}": v for k, v in feats_R.items()})
            row.update(feats_asym)

            rows.append(row)
            start  += step
            win_idx += 1

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def _check(df: pd.DataFrame, name: str) -> None:
    n_nan = df.select_dtypes(include="number").isna().sum().sum()
    n_inf = np.isinf(df.select_dtypes(include="number").to_numpy()).sum()
    print(f"  {name}: shape={df.shape}  NaN={n_nan}  Inf={n_inf}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading preprocessed.csv …")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} rows loaded")

    print("Assigning trial IDs …")
    df = assign_trial_ids(df)
    n_trials = df.groupby(
        ["subject", "box_size", "surface", "carrying_hand", "trial_id"]
    ).ngroups
    print(f"  → {n_trials} unique trials")

    # --- single-sensor path -------------------------------------------------
    print("\nBuilding single-sensor feature matrix …")
    single = build_single(df)
    _check(single, "single")
    single.to_csv(OUT_SINGLE, index=False)
    print(f"  Saved → {OUT_SINGLE}")

    # --- paired path --------------------------------------------------------
    print("\nBuilding paired feature matrix …")
    paired = build_paired(df)
    _check(paired, "paired")
    paired.to_csv(OUT_PAIRED, index=False)
    print(f"  Saved → {OUT_PAIRED}")

    print("\nDone.")


if __name__ == "__main__":
    main()
