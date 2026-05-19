"""
Feature scaling for the clustering pipeline.

Clustering is performed directly in the scaled feature space (no PCA).
A 2-D PCA is provided separately for scatter plot visualisation only.

Usage
-----
    from src.clustering.reduce import fit_scale, pca_2d

    X_scaled, scaler, feat_cols = fit_scale(feature_df, meta_cols)
    X_viz = pca_2d(X_scaled)   # for scatter plots only

Design choices
--------------
- RobustScaler (median/IQR) because gyro and accel have a ~100× scale
  difference and gyro contains extreme outliers (±527 °/s).
- Clip to ±50 after scaling to suppress the few extreme values that survive
  RobustScaler (mainly ratio features when denominator ≈ 0).
- PCA removed from the main pipeline: clustering in the full scaled space
  avoids the risk that PCA discards low-variance but high-signal dimensions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def fit_scale(
    df: pd.DataFrame,
    meta_cols: list[str],
) -> tuple[np.ndarray, RobustScaler, list[str]]:
    """
    Scale numeric feature columns with RobustScaler.

    Parameters
    ----------
    df        : feature DataFrame (meta + numeric feature columns)
    meta_cols : column names to exclude from scaling

    Returns
    -------
    X_scaled  : np.ndarray, shape (n_windows, n_features)
    scaler    : fitted RobustScaler (for later transform of new data)
    feat_cols : list of feature column names (column order matches X_scaled)
    """
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].to_numpy(dtype=float)

    # Replace residual NaN/Inf with column median (guard against edge-case windows)
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = col_medians[j]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Clip to ±50 and zero out any remaining non-finite values.
    # ratio features can produce extreme values when denominator ≈ 0;
    # RobustScaler divides by near-zero IQR → ±Inf before clipping.
    X_scaled = np.clip(X_scaled, -50.0, 50.0)
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)

    print(f"  Scaled: {X_scaled.shape[0]} windows × {X_scaled.shape[1]} features")

    return X_scaled, scaler, feat_cols


# ---------------------------------------------------------------------------
# 2-D PCA for visualisation only
# ---------------------------------------------------------------------------

def pca_2d(X_scaled: np.ndarray) -> np.ndarray:
    """
    Project *X_scaled* to 2 principal components for scatter plot visualisation.

    Clustering is performed in the full scaled feature space; this projection
    is only used to produce 2-D scatter plots.

    Returns
    -------
    np.ndarray, shape (n_windows, 2)
    """
    n_components = min(2, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    X_2d = pca.fit_transform(X_scaled)

    # Pad with zeros if fewer than 2 components were possible
    if X_2d.shape[1] < 2:
        X_2d = np.column_stack([X_2d, np.zeros(len(X_2d))])

    return X_2d
