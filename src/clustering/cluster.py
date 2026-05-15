"""
Clustering algorithms for the IMU carrying-behaviour analysis.

Four complementary algorithms are implemented to cover different geometric
assumptions about cluster shape:

  K-Means          — spherical clusters, Euclidean distance
  Agglomerative    — hierarchical (Ward linkage), no shape assumption
  DBSCAN           — density-based, finds arbitrary shapes + noise
  GMM              — probabilistic, soft assignments, handles ellipsoidal clusters

Each function accepts the PCA-reduced feature matrix X (np.ndarray) and returns
a labels array (int, -1 = noise for DBSCAN).

Helper utilities
----------------
  kmeans_selection   : fit k=2..k_max, compute inertia and silhouette; return best k
  dbscan_epsilon     : k-distance plot to guide ε selection
  gmm_bic_selection  : fit GMM for a range of components, return best by BIC
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# K-Means
# ---------------------------------------------------------------------------

def run_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
    """Fit K-Means with *k* clusters and return integer labels."""
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    return km.fit_predict(X)


def kmeans_selection(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> tuple[int, dict]:
    """
    Fit K-Means for k in [k_min, k_max] and select the best k using silhouette.

    Returns
    -------
    best_k  : int
    results : dict mapping k → {"inertia": float, "silhouette": float, "labels": ndarray}
    """
    results = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else float("nan")
        results[k] = {"inertia": km.inertia_, "silhouette": sil, "labels": labels}
        print(f"    K-Means k={k:2d}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}")

    # select k with highest silhouette
    best_k = max(results, key=lambda k: results[k]["silhouette"])
    print(f"  → Best k = {best_k} (silhouette = {results[best_k]['silhouette']:.4f})")

    if save_path is not None:
        _plot_kmeans_selection(results, k_min, k_max, save_path)

    return best_k, results


def _plot_kmeans_selection(
    results: dict,
    k_min: int,
    k_max: int,
    save_path: str | Path,
) -> None:
    ks = list(range(k_min, k_max + 1))
    inertias   = [results[k]["inertia"]    for k in ks]
    silhouettes = [results[k]["silhouette"] for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(ks, inertias, "b-o")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia (within-cluster SSE)")
    ax1.set_title("Elbow plot")
    ax1.set_xticks(ks)

    ax2.bar(ks, silhouettes, color="steelblue")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette vs k")
    ax2.set_xticks(ks)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  K-Means selection plot saved → {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Agglomerative / Hierarchical (Ward)
# ---------------------------------------------------------------------------

def run_agglomerative(
    X: np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
) -> np.ndarray:
    """Fit Agglomerative Clustering and return labels."""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return agg.fit_predict(X)


# ---------------------------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------------------------

def dbscan_epsilon_plot(
    X: np.ndarray,
    k: int = 5,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    """
    k-distance plot to guide ε selection for DBSCAN.

    The elbow of the sorted k-th nearest-neighbour distances is a good
    starting estimate for ε.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dists = np.sort(distances[:, -1])[::-1]

    plt.figure(figsize=(8, 4))
    plt.plot(k_dists)
    plt.xlabel("Points sorted by distance (descending)")
    plt.ylabel(f"{k}-th nearest neighbour distance")
    plt.title(title if title is not None else f"k-distance plot (k={k}) — find the elbow to choose ε")
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"  k-distance plot saved → {save_path}")
    else:
        plt.show()
    plt.close()


def run_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
) -> np.ndarray:
    """
    Fit DBSCAN and return labels (-1 = noise).

    Parameters
    ----------
    eps        : neighbourhood radius
    min_samples: minimum points to form a core point
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    print(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points "
          f"({n_noise / len(labels) * 100:.1f}%)")
    return labels


# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------

def gmm_bic_selection(
    X: np.ndarray,
    n_min: int = 2,
    n_max: int = 8,
    random_state: int = 42,
    save_path: str | Path | None = None,
) -> tuple[int, dict]:
    """
    Fit GMM for n_components in [n_min, n_max]; select best by BIC.

    Returns
    -------
    best_n  : int
    results : dict mapping n → {"bic": float, "aic": float, "labels": ndarray}
    """
    results = {}
    for n in range(n_min, n_max + 1):
        gmm = GaussianMixture(
            n_components=n, covariance_type="full",
            max_iter=200, random_state=random_state
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        results[n] = {"bic": gmm.bic(X), "aic": gmm.aic(X), "labels": labels}
        print(f"    GMM n={n:2d}  BIC={gmm.bic(X):.1f}  AIC={gmm.aic(X):.1f}")

    best_n = min(results, key=lambda n: results[n]["bic"])
    print(f"  → Best n = {best_n} (BIC = {results[best_n]['bic']:.1f})")

    if save_path is not None:
        _plot_gmm_selection(results, n_min, n_max, save_path)

    return best_n, results


def _plot_gmm_selection(
    results: dict,
    n_min: int,
    n_max: int,
    save_path: str | Path,
) -> None:
    ns   = list(range(n_min, n_max + 1))
    bics = [results[n]["bic"] for n in ns]
    aics = [results[n]["aic"] for n in ns]

    plt.figure(figsize=(7, 4))
    plt.plot(ns, bics, "b-o", label="BIC")
    plt.plot(ns, aics, "r--s", label="AIC")
    plt.xlabel("n_components")
    plt.ylabel("Information criterion")
    plt.title("GMM model selection (lower is better)")
    plt.legend()
    plt.xticks(ns)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  GMM selection plot saved → {save_path}")
    plt.close()


def run_gmm(
    X: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> np.ndarray:
    """Fit GMM with *n_components* and return hard-assigned labels."""
    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        max_iter=200, random_state=random_state
    )
    gmm.fit(X)
    return gmm.predict(X)
