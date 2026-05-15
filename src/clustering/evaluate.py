"""
Cluster validation: internal metrics, external metrics, and visualisations.

Internal metrics (no labels required)
--------------------------------------
  Silhouette score         — higher is better (+1 perfect, 0 random)
  Davies-Bouldin index     — lower is better (0 perfect)
  Calinski-Harabász index  — higher is better

External metrics (require ground-truth labels)
-----------------------------------------------
  ARI  (Adjusted Rand Index)           — chance-corrected cluster overlap
  AMI  (Adjusted Mutual Information)   — information-theoretic overlap
  Homogeneity / Completeness / V-measure

Validation targets: one_handed_carry (binary), carrying_hand (3-class), phase (3-class)

Visualisations
--------------
  scatter_2d        — PCA scatter coloured by cluster, then by label
  dendrogram_plot   — Ward linkage dendrogram (subsampled for readability)
  centroid_heatmap  — mean feature value per cluster (unscaled)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Internal metrics
# ---------------------------------------------------------------------------

def internal_metrics(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    Compute internal validity metrics.

    Parameters
    ----------
    X      : feature matrix (post-PCA)
    labels : cluster assignments (noise points -1 are excluded from metrics)

    Returns
    -------
    dict with keys silhouette, davies_bouldin, calinski_harabasz
    """
    mask = labels != -1   # exclude DBSCAN noise
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}

    X_clean = X[mask]
    l_clean = labels[mask]

    return {
        "silhouette":        silhouette_score(X_clean, l_clean),
        "davies_bouldin":    davies_bouldin_score(X_clean, l_clean),
        "calinski_harabasz": calinski_harabasz_score(X_clean, l_clean),
    }


# ---------------------------------------------------------------------------
# External metrics
# ---------------------------------------------------------------------------

def external_metrics(
    labels_pred: np.ndarray,
    labels_true: np.ndarray,
    label_name: str = "label",
) -> dict[str, float]:
    """
    Compute external validity metrics against a ground-truth label column.

    Parameters
    ----------
    labels_pred : predicted cluster assignments
    labels_true : ground-truth class labels (str or int)
    label_name  : descriptive name used as dict key prefix

    Returns
    -------
    dict with keys {label_name}_ari, {label_name}_ami, {label_name}_homogeneity,
                   {label_name}_completeness, {label_name}_vmeasure
    """
    # exclude noise points
    mask = labels_pred != -1
    lp = labels_pred[mask]
    lt = np.asarray(labels_true)[mask]

    h, c, v = homogeneity_completeness_v_measure(lt, lp)

    return {
        f"{label_name}_ari":          adjusted_rand_score(lt, lp),
        f"{label_name}_ami":          adjusted_mutual_info_score(lt, lp),
        f"{label_name}_homogeneity":  h,
        f"{label_name}_completeness": c,
        f"{label_name}_vmeasure":     v,
    }


def evaluate_all_labels(
    labels_pred: np.ndarray,
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run external_metrics against one_handed_carry, carrying_hand, and phase.

    Returns a DataFrame with one row per validation target.
    """
    targets = {
        "one_handed_carry": meta_df["one_handed_carry"].astype(str),
        "carrying_hand":    meta_df["carrying_hand"],
        "phase":            meta_df["phase"],
    }
    rows = []
    for name, series in targets.items():
        m = external_metrics(labels_pred, series.to_numpy(), label_name=name)
        rows.append(m)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def scatter_2d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "PCA 2D",
    save_path: str | Path | None = None,
    colour_by: np.ndarray | None = None,
    colour_label: str = "label",
) -> None:
    """
    2-D scatter plot of the first two PCA components.

    If *colour_by* is given, points are coloured by that array instead of
    *labels* (useful for labelling by ground-truth after clustering).

    Parameters
    ----------
    X           : post-PCA feature matrix (at least 2 columns)
    labels      : cluster assignments (used for marker shape / noise marking)
    title       : plot title
    save_path   : save figure here if given
    colour_by   : override colouring (e.g. meta_df["carrying_hand"])
    colour_label: legend / colourbar label when using colour_by
    """
    # Project to 2D; pad with zeros if only 1 component survived PCA
    if X.shape[1] > 2:
        pca2 = PCA(n_components=2, random_state=42)
        Xp = pca2.fit_transform(X)
    elif X.shape[1] == 1:
        Xp = np.column_stack([X[:, 0], np.zeros(len(X))])
    else:
        Xp = X

    colour_vals = colour_by if colour_by is not None else labels
    unique_vals = sorted(set(colour_vals))
    palette = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, val in enumerate(unique_vals):
        mask = colour_vals == val
        marker = "x" if val == -1 else "o"
        ax.scatter(Xp[mask, 0], Xp[mask, 1],
                   c=[palette[i % len(palette)]],
                   label=str(val), marker=marker,
                   s=12, alpha=0.6, linewidths=0.5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(title=colour_label, markerscale=2,
              loc="best", fontsize=7, ncol=2)
    plt.tight_layout()

    _save_or_show(save_path)


def dendrogram_plot(
    X: np.ndarray,
    max_samples: int = 300,
    save_path: str | Path | None = None,
    title: str = "Ward Dendrogram",
) -> None:
    """
    Ward linkage dendrogram.  Subsampled to *max_samples* for readability.
    """
    if len(X) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X = X[idx]

    Z = linkage(X, method="ward")

    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=0)
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Ward distance")
    plt.tight_layout()
    _save_or_show(save_path)


def centroid_heatmap(
    X_raw: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    max_features: int = 30,
    save_path: str | Path | None = None,
    title: str = "Cluster centroid heatmap",
) -> None:
    """
    Heatmap of mean (unscaled) feature values per cluster.

    Shows only the *max_features* features with the highest between-cluster
    variance to keep the plot readable.
    """
    mask   = labels != -1
    X_use  = X_raw[mask]
    l_use  = labels[mask]
    unique = sorted(set(l_use))

    centroids = np.array([X_use[l_use == k].mean(axis=0) for k in unique])

    # Select top features by between-cluster variance
    between_var = centroids.var(axis=0)
    top_idx     = np.argsort(between_var)[::-1][:max_features]
    centroids   = centroids[:, top_idx]
    names       = [feature_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.35), 4))
    sns.heatmap(
        centroids,
        xticklabels=names,
        yticklabels=[f"C{k}" for k in unique],
        cmap="RdBu_r", center=0,
        ax=ax, annot=False,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    _save_or_show(save_path)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def metrics_summary(
    results: dict[str, dict],
) -> pd.DataFrame:
    """
    Collect internal + external metrics from a results dict and format as a
    readable DataFrame.

    Parameters
    ----------
    results : {algo_name: {"internal": dict, "external": DataFrame}}

    Returns
    -------
    pd.DataFrame with one row per algorithm
    """
    rows = []
    for algo, res in results.items():
        row = {"algorithm": algo}
        row.update(res.get("internal", {}))
        ext_df = res.get("external")
        if ext_df is not None:
            for _, r in ext_df.iterrows():
                row.update(r.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _save_or_show(save_path: str | Path | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close()
