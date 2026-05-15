"""
Trial boundary detection and sliding window generation.

Trial boundary detection
------------------------
Only `carrying_hand == "both"` groups can contain multiple trials per metadata key
(two recordings per condition). A new trial starts whenever `time_s` resets
(i.e., the next value is smaller than the current value), because each recording
restarts the watch-local clock from 0.

For all other carrying conditions each group is already a single trial.

Sliding windows
---------------
Windows are 1 s long (100 samples at ~100 Hz) with 50% overlap (50-sample step).
Windows with fewer than 50 samples are discarded (handles trial-end fragments).
Phase labels are NOT used during windowing; they are attached as metadata by
taking the majority phase in each window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Trial boundary detection
# ---------------------------------------------------------------------------

TRIAL_GROUP_COLS = ["subject", "box_size", "surface", "carrying_hand", "sensor_hand"]


def assign_trial_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a `trial_id` integer column to *df*.

    Within each metadata group the trial counter increments whenever
    `time_s` resets (next < current).  Groups with `carrying_hand != "both"`
    always have a single trial (trial_id = 0).

    Parameters
    ----------
    df : pd.DataFrame
        The full preprocessed dataset (47 k rows).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an additional `trial_id` column (int, 0-based).
    """
    df = df.copy()
    df["trial_id"] = 0

    for keys, group_idx in df.groupby(TRIAL_GROUP_COLS).groups.items():
        # Sort by original row order within the group
        idx = group_idx  # already in insertion order after groupby
        time_vals = df.loc[idx, "time_s"].to_numpy()

        trial_id = 0
        trial_ids = np.zeros(len(time_vals), dtype=int)

        for i in range(1, len(time_vals)):
            if time_vals[i] < time_vals[i - 1]:
                trial_id += 1
            trial_ids[i] = trial_id

        df.loc[idx, "trial_id"] = trial_ids

    return df


# ---------------------------------------------------------------------------
# Sliding windows
# ---------------------------------------------------------------------------

SENSOR_COLS = ["GX", "GY", "GZ", "AX", "AY", "AZ"]
META_COLS = ["subject", "box_size", "sensor_hand", "surface",
             "carrying_hand", "one_handed_carry", "trial_id"]

WINDOW_SIZE = 100   # samples  (~1 s at 100 Hz)
STEP_SIZE   = 50    # samples  (50% overlap)
MIN_SAMPLES = 50    # discard shorter fragments


def _majority(series: pd.Series) -> str:
    """Return the most common value in *series*."""
    return series.value_counts().idxmax()


def extract_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    min_samples: int = MIN_SAMPLES,
) -> list[dict]:
    """
    Slide windows over every trial and collect metadata + raw-signal arrays.

    The DataFrame must already have a `trial_id` column (from `assign_trial_ids`).

    Parameters
    ----------
    df : pd.DataFrame
    window_size : int
    step_size   : int
    min_samples : int

    Returns
    -------
    list[dict]
        Each dict has keys:
          - all META_COLS values (scalar metadata)
          - "phase"   : majority phase label in the window
          - "start_t" : server_time_s of the first sample
          - "end_t"   : server_time_s of the last sample
          - "data"    : np.ndarray shape (n, 6) for the six sensor axes
          - "server_time_s_arr" : 1-D array of server timestamps in the window
    """
    window_group_cols = META_COLS  # group so each trial+sensor is isolated
    windows = []

    for keys, grp in df.groupby(window_group_cols):
        grp = grp.sort_values("server_time_s").reset_index(drop=True)
        n = len(grp)
        meta_vals = dict(zip(window_group_cols, keys))

        start = 0
        win_idx = 0
        while start < n:
            end = start + window_size
            chunk = grp.iloc[start:min(end, n)]
            if len(chunk) < min_samples:
                break

            win = {**meta_vals,
                   "win_idx":          win_idx,
                   "phase":            _majority(chunk["phase"]),
                   "start_t":          chunk["server_time_s"].iloc[0],
                   "end_t":            chunk["server_time_s"].iloc[-1],
                   "n_samples":        len(chunk),
                   "data":             chunk[SENSOR_COLS].to_numpy(dtype=float),
                   "server_time_s_arr": chunk["server_time_s"].to_numpy(dtype=float)}

            windows.append(win)
            start += step_size
            win_idx += 1

    return windows
