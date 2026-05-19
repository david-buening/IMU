"""
Cross-wrist asymmetry features for the paired-sensor path.

Given feature dicts for the L and R wrists (from time_domain.compute_features),
this module computes two types of asymmetry per shared feature:

  abs_diff  = |f_R - f_L|
  ratio     = f_R / (f_L + ε)    — signed, linear scale

Plus Pearson correlation per raw axis (6 features):

  pearson_corr_<axis>  — Pearson r between L and R for each of the 6 axes

Total: 2 × 91 + 6 = 188 asymmetry features per paired window.

Design notes
------------
- symm_idx removed: numerically unstable when both sides ≈ 0 (e.g. near-zero
  mean features), producing physically meaningless outliers.
- log_ratio removed: loses sign information for signed features (e.g. mean of
  AY which is negative on one wrist).
- xcorr_max_Amag removed: redundant with Pearson r at lag-0.
- ratio keeps sign and scale, matching the classifier's intent.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr


_EPS = 1e-9   # prevents division by zero


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
    dict mapping feature_name → float  (188 entries)
    """
    asym: dict[str, float] = {}

    # --- per-feature scalar asymmetry --------------------------------------
    shared_keys = [k for k in feats_L if k in feats_R]
    for key in shared_keys:
        f_l = feats_L[key]
        f_r = feats_R[key]

        asym[f"abs_diff_{key}"] = abs(f_r - f_l)
        asym[f"ratio_{key}"]    = f_r / (f_l + _EPS)

    # --- signal-level Pearson correlations ---------------------------------
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

    return asym
