"""
Time-domain feature extraction for a single sensor window.

All features are computed on a 2-D numpy array of shape (n_samples, 6),
where the 6 columns correspond to [GX, GY, GZ, AX, AY, AZ].

Feature set
-----------
For each of the 6 raw axes + derived Amag (accel magnitude) + Gmag (gyro
magnitude):
  mean, std, rms, range, iqr, skewness, kurtosis,
  zero_crossing_rate, energy, mean_abs_dev

Plus:
  signal_magnitude_area (SMA) — accel only (one scalar per window)
  jerk_std, jerk_rms           — from diff(Amag)/dt

Total: 11 features × 8 channels + SMA + 2 Jerk ≈ 91 features per window.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis as sp_kurtosis, skew as sp_skew


# Column indices in the raw data array [GX GY GZ AX AY AZ]
_GX, _GY, _GZ = 0, 1, 2
_AX, _AY, _AZ = 3, 4, 5

FS = 100.0   # nominal sampling frequency


# ---------------------------------------------------------------------------
# Primitive statistics
# ---------------------------------------------------------------------------

def _mean(x: np.ndarray) -> float:
    return float(np.mean(x))

def _std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))

def _range(x: np.ndarray) -> float:
    return float(np.max(x) - np.min(x))

def _iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))

def _skewness(x: np.ndarray) -> float:
    # Guard: scipy raises RuntimeWarning and returns NaN when std ≈ 0
    if len(x) <= 2 or np.std(x) < 1e-10:
        return 0.0
    val = float(sp_skew(x, bias=False))
    return val if np.isfinite(val) else 0.0

def _kurtosis(x: np.ndarray) -> float:
    # Fisher's definition (excess kurtosis = 0 for normal)
    if len(x) <= 3 or np.std(x) < 1e-10:
        return 0.0
    val = float(sp_kurtosis(x, fisher=True, bias=False))
    return val if np.isfinite(val) else 0.0

def _zero_crossing_rate(x: np.ndarray) -> float:
    signs = np.sign(x - np.mean(x))   # centre before counting crossings
    crossings = np.sum(np.diff(signs) != 0)
    return float(crossings / max(len(x) - 1, 1))

def _energy(x: np.ndarray) -> float:
    return float(np.mean(x ** 2))

def _mean_abs_dev(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x - np.mean(x))))


# Ordered list of (name, function)
_STAT_FUNCS = [
    ("mean",               _mean),
    ("std",                _std),
    ("rms",                _rms),
    ("range",              _range),
    ("iqr",                _iqr),
    ("skewness",           _skewness),
    ("kurtosis",           _kurtosis),
    ("zero_crossing_rate", _zero_crossing_rate),
    ("energy",             _energy),
    ("mean_abs_dev",       _mean_abs_dev),
]


def _channel_features(signal: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute all stats for one 1-D signal with a given name prefix."""
    return {f"{prefix}_{name}": fn(signal) for name, fn in _STAT_FUNCS}


# ---------------------------------------------------------------------------
# Window-level feature function
# ---------------------------------------------------------------------------

def compute_features(data: np.ndarray, fs: float = FS) -> dict[str, float]:
    """
    Compute all time-domain features for a single window.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, 6)
        Columns: [GX, GY, GZ, AX, AY, AZ]
    fs   : float
        Sampling frequency in Hz (used for Jerk scaling).

    Returns
    -------
    dict mapping feature_name → float value
    """
    feats: dict[str, float] = {}

    # --- raw axes -----------------------------------------------------------
    axis_names = ["GX", "GY", "GZ", "AX", "AY", "AZ"]
    for i, name in enumerate(axis_names):
        feats.update(_channel_features(data[:, i], name))

    # --- derived magnitudes -------------------------------------------------
    accel = data[:, 3:6]   # AX AY AZ
    gyro  = data[:, 0:3]   # GX GY GZ

    Amag = np.sqrt(np.sum(accel ** 2, axis=1))
    Gmag = np.sqrt(np.sum(gyro  ** 2, axis=1))

    feats.update(_channel_features(Amag, "Amag"))
    feats.update(_channel_features(Gmag, "Gmag"))

    # --- signal magnitude area (accel only) ---------------------------------
    feats["SMA"] = float(np.mean(np.sum(np.abs(accel), axis=1)))

    # --- jerk (derivative of Amag) ------------------------------------------
    if len(Amag) > 1:
        jerk = np.diff(Amag) * fs   # units: g/s
        feats["jerk_std"] = _std(jerk)
        feats["jerk_rms"] = _rms(jerk)
    else:
        feats["jerk_std"] = 0.0
        feats["jerk_rms"] = 0.0

    return feats
