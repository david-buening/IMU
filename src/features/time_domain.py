"""
Time-domain feature extraction for a single sensor window.

All features are computed on a 2-D numpy array of shape (n_samples, 6),
where the 6 columns correspond to [GX, GY, GZ, AX, AY, AZ].

Feature set
-----------
For each of the 10 channels below:
  mean, std, rms, range, iqr, skewness, kurtosis,
  zero_crossing_rate, mean_abs_dev

Channels
--------
  GX GY GZ AX AY AZ — raw sensor axes
  Amag  — L2 norm of accel  sqrt(AX² + AY² + AZ²)
  Gmag  — L2 norm of gyro   sqrt(GX² + GY² + GZ²)
  Ajerk — vector accel jerk magnitude: sqrt(sum(diff(accel)², axis=1))
  Gjerk — vector gyro  jerk magnitude: sqrt(sum(diff(gyro)², axis=1))

Plus:
  SMA — signal magnitude area, accel only (one scalar per window)

Total: 9 stats × 10 channels + SMA = 91 features per window.

Notes
-----
- energy (= rms²) removed — redundant with rms.
- Ajerk/Gjerk are NOT scaled by fs (matches classifier jerk convention).
- Ajerk replaces the old scalar jerk_std / jerk_rms features.
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
    return float(np.mean(x)) if len(x) > 0 else 0.0

def _std(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2))) if len(x) > 0 else 0.0

def _range(x: np.ndarray) -> float:
    return float(np.max(x) - np.min(x)) if len(x) > 0 else 0.0

def _iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) > 0 else 0.0

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
    if len(x) < 2:
        return 0.0
    signs = np.sign(x - np.mean(x))   # centre before counting crossings
    crossings = np.sum(np.diff(signs) != 0)
    return float(crossings / (len(x) - 1))

def _mean_abs_dev(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x - np.mean(x)))) if len(x) > 0 else 0.0


# Ordered list of (name, function) — 9 stats per channel
_STAT_FUNCS = [
    ("mean",               _mean),
    ("std",                _std),
    ("rms",                _rms),
    ("range",              _range),
    ("iqr",                _iqr),
    ("skewness",           _skewness),
    ("kurtosis",           _kurtosis),
    ("zero_crossing_rate", _zero_crossing_rate),
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
        Sampling frequency in Hz (kept for API compatibility; not used).

    Returns
    -------
    dict mapping feature_name → float value  (91 entries)
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

    # --- jerk channels (vector jerk magnitude, not scaled by fs) ------------
    # Ajerk: frame-to-frame L2 norm of accel difference
    # Gjerk: frame-to-frame L2 norm of gyro difference
    if len(accel) > 1:
        Ajerk = np.sqrt(np.sum(np.diff(accel, axis=0) ** 2, axis=1))
        Gjerk = np.sqrt(np.sum(np.diff(gyro,  axis=0) ** 2, axis=1))
    else:
        Ajerk = np.array([0.0])
        Gjerk = np.array([0.0])

    feats.update(_channel_features(Ajerk, "Ajerk"))
    feats.update(_channel_features(Gjerk, "Gjerk"))

    return feats
