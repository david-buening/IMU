"""
Dimensionality reduction: RobustScaler + PCA.

Usage
-----
    from src.clustering.reduce import fit_transform, scree_plot

    X_pca, scaler, pca = fit_transform(feature_df, meta_cols)
    scree_plot(pca, save_path="data/features/scree_single.png")

Design choices
--------------
- RobustScaler (median/IQR) before PCA because gyro and accel have a ~100×
  scale difference and gyro contains extreme outliers (±527 °/s).
- PCA retains enough components to explain 95% of variance.
- Scaler is fit only on the feature columns; meta columns are kept separate.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


# ---------------------------------------------------------------------------
# Fit / transform
# ---------------------------------------------------------------------------

def fit_transform(
    df: pd.DataFrame,
    meta_cols: list[str],
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, RobustScaler, PCA]:
    """
    Scale and reduce the numeric feature columns.

    Parameters
    ----------
    df                 : feature DataFrame (meta + numeric feature columns)
    meta_cols          : column names to exclude from scaling/PCA
    variance_threshold : cumulative explained variance to retain (default 0.95)

    Returns
    -------
    X_pca   : np.ndarray, shape (n_windows, n_components)
    scaler  : fitted RobustScaler (for later transform of new data)
    pca     : fitted PCA object
    """
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].to_numpy(dtype=float)

    # Replace any residual NaN/Inf with column median (should not occur after
    # the pipeline, but guard against edge-case windows)
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        bad = ~np.isfinite(X[:, j])
        if bad.any():
            X[bad, j] = col_medians[j]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Some asymmetry features (log_ratio) can produce extreme values when IQR ≈ 0
    # (RobustScaler divides by near-zero → ±Inf).  Clip to ±50 σ-equivalents and
    # zero out any remaining non-finite cells so PCA is well-conditioned.
    X_scaled = np.clip(X_scaled, -50.0, 50.0)
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)

    # Fit PCA on all components first to determine n_components
    pca_full = PCA(n_components=min(X_scaled.shape))
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"  PCA: {X_scaled.shape[1]} features → {n_components} components "
          f"({cumvar[n_components - 1] * 100:.1f}% variance retained)")

    return X_pca, scaler, pca


# ---------------------------------------------------------------------------
# Scree plot
# ---------------------------------------------------------------------------

def scree_plot(
    pca: PCA,
    save_path: str | Path | None = None,
    title: str = "PCA Scree Plot",
) -> None:
    """
    Plot explained variance ratio per component and cumulative variance.

    Parameters
    ----------
    pca       : fitted PCA object
    save_path : if given, save figure to this path instead of showing
    title     : plot title
    """
    evr  = pca.explained_variance_ratio_
    cumv = np.cumsum(evr)
    comps = np.arange(1, len(evr) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(comps, evr * 100, color="steelblue", alpha=0.7, label="Per-component")
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance (%)")

    ax2 = ax1.twinx()
    ax2.plot(comps, cumv * 100, "r-o", markersize=3, label="Cumulative")
    ax2.axhline(95, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("Cumulative variance (%)")
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Scree plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Loadings helper
# ---------------------------------------------------------------------------

def loadings_df(pca: PCA, feature_names: list[str]) -> pd.DataFrame:
    """
    Return a DataFrame of PCA component loadings (components × features).
    Useful for interpreting which original features dominate each PC.
    """
    n_comp = pca.n_components_
    cols = feature_names[: pca.components_.shape[1]]
    return pd.DataFrame(
        pca.components_,
        index=[f"PC{i + 1}" for i in range(n_comp)],
        columns=cols,
    )
