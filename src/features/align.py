"""
L/R sensor alignment via server_time_s interpolation.

Both wrists share the same server clock, so `server_time_s` is a reliable
common reference. For a given trial we:
  1. Determine the overlapping time range [t_min, t_max] where both L and R
     have data.
  2. Build a regular 100 Hz grid over that range.
  3. Linearly interpolate every sensor axis for each wrist onto the grid.

The output is a single DataFrame with columns:
  <axis>_L, <axis>_R  for each of GX GY GZ AX AY AZ
plus  server_time_s  (the shared grid).

Usage in pipeline
-----------------
Called per trial after `assign_trial_ids`.  The result feeds the paired
feature path.  Single-sensor windows are extracted from the pre-alignment
per-sensor DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SENSOR_COLS = ["GX", "GY", "GZ", "AX", "AY", "AZ"]
FS = 100.0   # target sampling frequency (Hz)


def align_trial(
    left: pd.DataFrame,
    right: pd.DataFrame,
    fs: float = FS,
) -> pd.DataFrame | None:
    """
    Align one trial's L and R streams onto a shared 100 Hz grid.

    Parameters
    ----------
    left  : rows where sensor_hand == "L" for this trial
    right : rows where sensor_hand == "R" for this trial
    fs    : target sampling frequency

    Returns
    -------
    pd.DataFrame with columns [server_time_s, GX_L, …, AZ_L, GX_R, …, AZ_R]
    or None if either side has fewer than 2 samples (cannot interpolate).
    """
    if len(left) < 2 or len(right) < 2:
        return None

    left  = left.sort_values("server_time_s")
    right = right.sort_values("server_time_s")

    t_min = max(left["server_time_s"].min(), right["server_time_s"].min())
    t_max = min(left["server_time_s"].max(), right["server_time_s"].max())

    if t_max <= t_min:
        return None

    n_points = int(round((t_max - t_min) * fs)) + 1
    grid = np.linspace(t_min, t_max, n_points)

    result = pd.DataFrame({"server_time_s": grid})

    for side_label, side_df in [("L", left), ("R", right)]:
        t = side_df["server_time_s"].to_numpy(dtype=float)
        for col in SENSOR_COLS:
            vals = side_df[col].to_numpy(dtype=float)
            result[f"{col}_{side_label}"] = np.interp(grid, t, vals)

    return result


def align_all_trials(df: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    """
    Run `align_trial` for every unique (subject, box_size, surface,
    carrying_hand, trial_id) group that has both L and R data.

    Returns
    -------
    dict mapping trial key tuple → aligned DataFrame
    """
    trial_group_cols = ["subject", "box_size", "surface", "carrying_hand", "trial_id"]
    aligned_trials: dict[tuple, pd.DataFrame] = {}

    for keys, grp in df.groupby(trial_group_cols):
        left  = grp[grp["sensor_hand"] == "L"]
        right = grp[grp["sensor_hand"] == "R"]

        aligned = align_trial(left, right)
        if aligned is not None:
            # carry metadata forward
            for col, val in zip(trial_group_cols, keys):
                aligned[col] = val
            # majority phase per timestamp is not meaningful here;
            # phase will be looked up per window after alignment
            aligned_trials[keys] = aligned

    return aligned_trials
