"""
Cross-wrist asymmetry features for the paired-sensor path.

Given feature dicts for the L and R wrists (from time_domain.compute_features),
this module computes three types of asymmetry per shared feature:

  abs_diff        = |f_R - f_L|
  symmetry_index  = |f_R - f_L| / ((|f_R| + |f_L|) / 2) × 100
                    (Robinson et al., 1987; undefined when both sides ≈ 0)
  log_ratio       = log(|f_R| / (|f_L| + ε) + ε)
                    (symmetric at 0; handles multiplicative asymmetry)

Plus two signal-level features computed from the raw window arrays:

  pearson_corr_<axis>  — Pearson r between L and R for each of the 6 axes
  xcorr_max_Amag       — peak normalised cross-correlation of L vs R Amag

References
----------
Robinson et al. (1987). Use of force platform variables to quantify the effects
    of chiropractic manipulation on gait symmetry.  J Manipulative Physiol Ther.
Moe-Nilssen (1998). A new method for evaluating motor control in gait under
    real-life environmental conditions.  Part 1: The instrument.
Kavanagh & Menz (2008). Accelerometry: A technique for quantifying movement
    patterns during walking.  Gait & Posture.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


_EPS = 1e-9   # prevents log(0) and division by zero


def compute_asymmetry(
    feats_L: dict[str, float],
    feats_R: dict[str, float],
    data_L: np.ndarray,
    data_R: np.ndarray,
) -> dict[str, float]:
    """
    Compute all asymmetry features.

    Parameters
    ----------
    feats_L, feats_R : dict
        Feature dicts from time_domain.compute_features for L and R wrists.
        Must have identical keys.
    data_L, data_R : np.ndarray, shape (n, 6)
        Raw window arrays [GX GY GZ AX AY AZ] for each wrist.
        Lengths may differ slightly after interpolation; truncated to the
        shorter of the two.

    Returns
    -------
    dict mapping feature_name → float
    """
    asym: dict[str, float] = {}

    # --- per-feature scalar asymmetry --------------------------------------
    shared_keys = [k for k in feats_L if k in feats_R]
    for key in shared_keys:
        f_l = feats_L[key]
        f_r = feats_R[key]

        asym[f"abs_diff_{key}"]  = abs(f_r - f_l)

        denom = (abs(f_r) + abs(f_l)) / 2.0
        # Symmetry index is only meaningful when at least one side has a
        # detectable signal.  When both sides are near zero (e.g. mean ≈ 0
        # after DC-offset removal), the ratio is numerically unstable and
        # physically meaningless — clamp to 0.
        # Robinson et al. (1987): theoretical max = 200 (one side zero).
        if denom < 1e-4:
            asym[f"symm_idx_{key}"] = 0.0
        else:
            asym[f"symm_idx_{key}"] = min(
                (abs(f_r - f_l) / (denom + _EPS)) * 100.0, 200.0
            )

        asym[f"log_ratio_{key}"] = float(np.log(abs(f_r) / (abs(f_l) + _EPS) + _EPS))

    # --- signal-level features ---------------------------------------------
    n = min(len(data_L), len(data_R))
    data_L, data_R = data_L[:n], data_R[:n]

    axis_names = ["GX", "GY", "GZ", "AX", "AY", "AZ"]
    for i, name in enumerate(axis_names):
        col_l = data_L[:, i]
        col_r = data_R[:, i]
        if np.std(col_l) > _EPS and np.std(col_r) > _EPS:
            r, _ = pearsonr(col_l, col_r)
            val = float(r)
            asym[f"pearson_corr_{name}"] = val if np.isfinite(val) else 0.0
        else:
            asym[f"pearson_corr_{name}"] = 0.0

    # normalised cross-correlation of Amag (peak value)
    Amag_L = np.sqrt(np.sum(data_L[:, 3:6] ** 2, axis=1))
    Amag_R = np.sqrt(np.sum(data_R[:, 3:6] ** 2, axis=1))

    asym["xcorr_max_Amag"] = _xcorr_max(Amag_L, Amag_R)

    return asym


def _xcorr_max(a: np.ndarray, b: np.ndarray) -> float:
    """
    Peak of the normalised cross-correlation between vectors *a* and *b*.

    Normalisation: divide by sqrt(sum(a²) * sum(b²)) so the result is in [-1, 1].
    """
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom < _EPS:
        return 0.0
    xcorr = np.correlate(a - a.mean(), b - b.mean(), mode="full") / denom
    return float(np.max(np.abs(xcorr)))
