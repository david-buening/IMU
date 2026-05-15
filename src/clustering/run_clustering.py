"""
Full clustering analysis for IMU one-sided carrying data.

Reads:
    data/features/feature_matrix_single.csv
    data/features/feature_matrix_paired.csv

Writes to data/features/plots/{subset}/{tag}/:
    scree_{tag}.png
    kmeans_{tag}.png               elbow + silhouette (k=2..8)
    kdist_{tag}.png                DBSCAN k-distance plot
    gmm_bic_{tag}.png              GMM BIC/AIC model selection
    scatter_cluster_{algo}_{tag}.png   cluster colouring (kmeans: k=2, best_k, k=8 only)
    scatter_{label}_{tag}.png      ground-truth label overlay (3 plots, generated once)
    dendrogram_{tag}.png
    heatmap_kmeans_{tag}.png

Filtered paired plots go to data/features/plots/{subset}_filtered/paired/.

Writes to data/features/:
    metrics_summary.csv           all metrics (pooled + stratified, unfiltered + filtered)
    feature_matrix_paired_filtered.csv   paired matrix after kurtosis-outlier removal

Run from project root:
    python src/clustering/run_clustering.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clustering.reduce   import fit_transform, scree_plot
from clustering.cluster  import (
    kmeans_selection,
    run_agglomerative,
    dbscan_epsilon_plot, run_dbscan,
    gmm_bic_selection,
)
from clustering.evaluate import (
    internal_metrics, evaluate_all_labels,
    scatter_2d, dendrogram_plot, centroid_heatmap,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PLOTS_DIR    = FEATURES_DIR / "plots"
OUT_METRICS  = FEATURES_DIR / "metrics_summary.csv"

SINGLE_CSV          = FEATURES_DIR / "feature_matrix_single.csv"
PAIRED_CSV          = FEATURES_DIR / "feature_matrix_paired.csv"
PAIRED_FILTERED_CSV = FEATURES_DIR / "feature_matrix_paired_filtered.csv"

SINGLE_META = [
    "subject", "box_size", "sensor_hand", "surface",
    "carrying_hand", "one_handed_carry",
    "trial_id", "win_idx", "phase",
    "start_t", "end_t", "n_samples",
]
PAIRED_META = [
    "subject", "box_size", "surface",
    "carrying_hand", "one_handed_carry",
    "trial_id", "win_idx", "phase",
    "start_t", "end_t", "n_samples",
]

LABEL_COLS = ["one_handed_carry", "carrying_hand", "phase"]


# ---------------------------------------------------------------------------
# Auto-eps for DBSCAN (knee of sorted k-distance curve)
# ---------------------------------------------------------------------------

def _knee_eps(X: np.ndarray, k: int = 5) -> float:
    """Knee of the k-distance plot via max perpendicular distance heuristic."""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_dists = np.sort(distances[:, -1])

    n = len(k_dists)
    x = np.linspace(0, 1, n)
    y = (k_dists - k_dists.min()) / (k_dists.max() - k_dists.min() + 1e-12)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1

    pts = np.column_stack([x, y]) - p1
    dists = np.abs(pts[:, 0] * line_vec[1] - pts[:, 1] * line_vec[0]) / (
        np.linalg.norm(line_vec) + 1e-12
    )
    knee_idx = int(np.argmax(dists))
    eps = float(k_dists[knee_idx])
    print(f"    Auto-eps (knee at index {knee_idx}): {eps:.4f}")
    return eps


# ---------------------------------------------------------------------------
# Kurtosis-outlier filter for the paired matrix
# ---------------------------------------------------------------------------

def filter_paired_outliers(
    df: pd.DataFrame,
    meta_cols: list[str],
    kurtosis_threshold: float = 30.0,
) -> pd.DataFrame:
    """
    Remove windows where raw sensor kurtosis on either wrist is extreme.

    Checks only the L_* and R_* kurtosis features (direct excess kurtosis of
    each sensor axis within the window).  A window is removed if any raw
    kurtosis exceeds *kurtosis_threshold*.

    Rationale: for 100-sample windows of steady carrying motion, excess
    kurtosis is typically < 5 (99th percentile across the dataset = ~15).
    Values above 30 indicate a momentary impact on one wrist (table contact,
    accidental drop), creating extreme cross-wrist asymmetry.  These ~14
    windows (2.3%) dominate the paired k=2 clustering (silhouette ~0.78)
    without representing the experimental conditions of interest.

    Only L_*/R_* columns are checked, not abs_diff/symm_idx/log_ratio
    kurtosis features which operate on different scales.

    Parameters
    ----------
    df                  : paired feature DataFrame (meta + feature columns)
    meta_cols           : column names to exclude from filtering
    kurtosis_threshold  : excess kurtosis above which a window is removed
                          (default 30; 99th pct of the dataset is ~15)

    Returns
    -------
    Filtered DataFrame with outlier windows removed (index reset).
    """
    feat_cols = [c for c in df.columns if c not in meta_cols]
    kurt_cols = [
        c for c in feat_cols
        if "kurtosis" in c and (c.startswith("L_") or c.startswith("R_"))
    ]

    if not kurt_cols:
        print("  [filter] No L_/R_ kurtosis columns found — returning unfiltered.")
        return df

    X_kurt = df[kurt_cols].to_numpy(dtype=float)
    outlier_mask = np.any(X_kurt > kurtosis_threshold, axis=1)
    n_removed = int(outlier_mask.sum())
    n_total = len(df)

    print(f"  [filter] Raw-kurtosis outlier filter (threshold={kurtosis_threshold}):")
    print(f"    {n_removed} windows removed / {n_total} total "
          f"({100 * n_removed / n_total:.1f}%)")
    if n_removed > 0:
        subj_counts = df.loc[outlier_mask, "subject"].value_counts()
        print(f"    By subject: {subj_counts.to_dict()}")
        triggers = (X_kurt > kurtosis_threshold)[outlier_mask]
        col_counts = triggers.sum(axis=0)
        top_idx = np.argsort(col_counts)[::-1][:5]
        print("    Top triggering columns:")
        for i in top_idx:
            if col_counts[i] > 0:
                print(f"      {kurt_cols[i]}: {col_counts[i]} windows  "
                      f"(max={X_kurt[:, i].max():.1f})")

    return df[~outlier_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-matrix analysis
# ---------------------------------------------------------------------------

def run_for_matrix(
    df: pd.DataFrame,
    meta_cols: list[str],
    tag: str,
    subset: str = "all",
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run the full reduction → clustering → validation pipeline for one
    feature matrix (or a subject-filtered slice of it).

    Parameters
    ----------
    df        : feature DataFrame (meta + numeric features)
    meta_cols : columns to exclude from PCA
    tag       : "single", "paired", or "paired_filtered"
    subset    : "all", "David", or "Viktor" — used for labels and file names
    out_dir   : directory for plots; defaults to PLOTS_DIR / subset / tag

    Returns
    -------
    pd.DataFrame with one metrics row per algorithm, including `subset` column
    """
    if out_dir is None:
        out_dir = PLOTS_DIR / subset / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = [c for c in df.columns if c not in meta_cols]
    label = f"{tag} [{subset}]"

    # ------------------------------------------------------------------
    # 1. Reduce: RobustScaler + PCA
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Matrix: {label}  ({df.shape[0]} windows × {len(feat_cols)} features)")
    print(f"{'='*60}")
    print("\n[1] Dimensionality reduction")
    X_pca, scaler, pca = fit_transform(df, meta_cols)
    scree_plot(pca,
               save_path=out_dir / f"scree_{tag}.png",
               title=f"PCA scree — {label}")

    # Clipped-scaled features for the centroid heatmap (mirrors what PCA sees)
    X_scaled = np.clip(
        scaler.transform(df[feat_cols].to_numpy(dtype=float)),
        -50.0, 50.0,
    )
    X_scaled = np.where(np.isfinite(X_scaled), X_scaled, 0.0)

    # ------------------------------------------------------------------
    # 2. K-Means sweep k=2..8
    # ------------------------------------------------------------------
    print(f"\n[2] K-Means sweep (k=2..8)")
    best_k, km_results = kmeans_selection(
        X_pca,
        save_path=out_dir / f"kmeans_{tag}.png",
    )

    # ------------------------------------------------------------------
    # 3. DBSCAN
    # ------------------------------------------------------------------
    print(f"\n[3] DBSCAN")
    dbscan_epsilon_plot(
        X_pca, k=5,
        save_path=out_dir / f"kdist_{tag}.png",
        title=f"k-distance plot (k=5) — {label}",
    )
    eps_auto = _knee_eps(X_pca, k=5)
    labels_db = run_dbscan(X_pca, eps=eps_auto, min_samples=5)

    # ------------------------------------------------------------------
    # 4. GMM BIC selection
    # ------------------------------------------------------------------
    print(f"\n[4] GMM BIC selection")
    best_n, gmm_results = gmm_bic_selection(
        X_pca,
        save_path=out_dir / f"gmm_bic_{tag}.png",
    )

    # ------------------------------------------------------------------
    # 5. Final algorithm set
    # ------------------------------------------------------------------
    algorithms: dict[str, np.ndarray] = {}
    for k, res in km_results.items():
        algorithms[f"kmeans_k{k}"] = res["labels"]

    print(f"\n[5] Agglomerative Ward (k={best_k})")
    algorithms[f"ward_k{best_k}"] = run_agglomerative(X_pca, best_k)
    algorithms[f"dbscan_eps{eps_auto:.3f}"] = labels_db
    algorithms[f"gmm_n{best_n}"] = gmm_results[best_n]["labels"]

    # K-Means k values for which we generate scatter_cluster plots.
    # Intermediate k (3–7) add little visual value; keep the baseline (2),
    # the recommended solution (best_k), and the upper bound (8).
    km_scatter_ks = {2, best_k, max(km_results.keys())}

    # ------------------------------------------------------------------
    # 6. Label scatters — generated ONCE per subset (independent of algo)
    # ------------------------------------------------------------------
    dummy_labels = np.zeros(len(X_pca), dtype=int)
    for lbl in LABEL_COLS:
        if lbl in df.columns:
            scatter_2d(
                X_pca, dummy_labels,
                colour_by=df[lbl].astype(str).to_numpy(),
                colour_label=lbl,
                title=f"{lbl} — {label}",
                save_path=out_dir / f"scatter_{lbl}_{tag}.png",
            )

    # ------------------------------------------------------------------
    # 7. Metrics + cluster-assignment scatter plots
    # ------------------------------------------------------------------
    print(f"\n[6] Evaluating {len(algorithms)} algorithm configurations …")
    summary_rows: list[dict] = []

    for algo_name, labels in algorithms.items():
        n_cl    = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        print(f"\n  {algo_name}  ({n_cl} clusters, {n_noise} noise)")

        int_m  = internal_metrics(X_pca, labels)
        ext_df = evaluate_all_labels(labels, df[list(meta_cols)])

        # Scatter coloured by cluster assignment — skip intermediate K-Means k
        is_intermediate_km = (
            algo_name.startswith("kmeans_k")
            and int(algo_name.split("kmeans_k")[1]) not in km_scatter_ks
        )
        if not is_intermediate_km:
            scatter_2d(
                X_pca, labels,
                title=f"{algo_name} clusters — {label}",
                save_path=out_dir / f"scatter_cluster_{algo_name}_{tag}.png",
                colour_label="cluster",
            )

        row: dict = {
            "subset":     subset,
            "matrix":     tag,
            "algorithm":  algo_name,
            "n_windows":  len(df),
            "n_clusters": n_cl,
            "n_noise":    n_noise,
        }
        row.update(int_m)
        for _, r in ext_df.iterrows():
            row.update({k: v for k, v in r.to_dict().items() if pd.notna(v)})
        summary_rows.append(row)

    # ------------------------------------------------------------------
    # 8. Dendrogram
    # ------------------------------------------------------------------
    dendrogram_plot(
        X_pca, max_samples=300,
        save_path=out_dir / f"dendrogram_{tag}.png",
        title=f"Ward dendrogram — {label}",
    )

    # ------------------------------------------------------------------
    # 9. Centroid heatmap (best K-Means, clipped-scaled features)
    # ------------------------------------------------------------------
    centroid_heatmap(
        X_scaled,
        km_results[best_k]["labels"],
        feat_cols,
        max_features=30,
        save_path=out_dir / f"heatmap_kmeans_{tag}.png",
        title=f"K-Means (k={best_k}) centroid heatmap — {label}",
    )

    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Within-subject stratified analysis
# ---------------------------------------------------------------------------

def run_stratified_by_subject(
    df: pd.DataFrame,
    meta_cols: list[str],
    tag: str,
) -> pd.DataFrame:
    """
    Run the full clustering pipeline separately for each subject.

    Rationale: pooled analysis is dominated by between-subject variation
    (different gait, arm swing, sensor placement).  Within-subject clustering
    removes that confounder so condition-driven structure (if it exists) has
    a chance to emerge.

    Parameters
    ----------
    df        : full feature DataFrame
    meta_cols : metadata columns to exclude from PCA
    tag       : "single", "paired", or "paired_filtered"

    Returns
    -------
    pd.DataFrame with metrics for all subjects combined
    """
    subjects = sorted(df["subject"].unique())
    summaries = []

    for subj in subjects:
        sub_df = df[df["subject"] == subj].reset_index(drop=True)
        print(f"\n{'#'*60}")
        print(f"  STRATIFIED: {subj} / {tag}  ({len(sub_df)} windows)")
        print(f"{'#'*60}")

        summary = run_for_matrix(
            sub_df,
            meta_cols,
            tag=tag,
            subset=f"{subj}_filtered" if "filtered" in tag else subj,
        )
        summaries.append(summary)

    return pd.concat(summaries, ignore_index=True)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

_PRINT_COLS = [
    "subset", "matrix", "algorithm", "n_windows", "n_clusters",
    "silhouette", "davies_bouldin",
    "one_handed_carry_ari", "carrying_hand_ari", "phase_ari",
]


def _print_summary(df: pd.DataFrame) -> None:
    sub = df[[c for c in _PRINT_COLS if c in df.columns]].copy()
    for col in sub.select_dtypes(include="number").columns:
        sub[col] = sub[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "nan")
    print("\n" + "=" * 100)
    print("METRICS SUMMARY")
    print("=" * 100)
    print(sub.to_string(index=False))
    print("=" * 100)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries: list[pd.DataFrame] = []

    # -----------------------------------------------------------------------
    # 1. Pooled analysis — single-sensor
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  POOLED ANALYSIS — SINGLE-SENSOR")
    print("="*60)
    df_single = pd.read_csv(SINGLE_CSV)
    all_summaries.append(
        run_for_matrix(df_single, SINGLE_META, "single", subset="all")
    )

    # -----------------------------------------------------------------------
    # 2. Pooled analysis — paired (unfiltered, for comparison)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  POOLED ANALYSIS — PAIRED (unfiltered)")
    print("="*60)
    df_paired = pd.read_csv(PAIRED_CSV)
    all_summaries.append(
        run_for_matrix(df_paired, PAIRED_META, "paired", subset="all")
    )

    # -----------------------------------------------------------------------
    # 3. Pooled analysis — paired (kurtosis-outlier filtered)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  POOLED ANALYSIS — PAIRED (filtered)")
    print("="*60)
    df_paired_filt = filter_paired_outliers(df_paired, PAIRED_META, kurtosis_threshold=30.0)
    df_paired_filt.to_csv(PAIRED_FILTERED_CSV, index=False)
    print(f"  Filtered matrix saved → {PAIRED_FILTERED_CSV}")
    all_summaries.append(
        run_for_matrix(
            df_paired_filt, PAIRED_META, "paired_filtered", subset="all_filtered"
        )
    )

    # -----------------------------------------------------------------------
    # 4. Stratified analysis — single-sensor
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  STRATIFIED ANALYSIS — SINGLE-SENSOR (per subject)")
    print("="*60)
    all_summaries.append(
        run_stratified_by_subject(df_single, SINGLE_META, "single")
    )

    # -----------------------------------------------------------------------
    # 5. Stratified analysis — paired (unfiltered)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  STRATIFIED ANALYSIS — PAIRED unfiltered (per subject)")
    print("="*60)
    all_summaries.append(
        run_stratified_by_subject(df_paired, PAIRED_META, "paired")
    )

    # -----------------------------------------------------------------------
    # 6. Stratified analysis — paired (filtered)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  STRATIFIED ANALYSIS — PAIRED filtered (per subject)")
    print("="*60)
    all_summaries.append(
        run_stratified_by_subject(df_paired_filt, PAIRED_META, "paired_filtered")
    )

    # -----------------------------------------------------------------------
    # Save and print
    # -----------------------------------------------------------------------
    full_summary = pd.concat(all_summaries, ignore_index=True)
    full_summary.to_csv(OUT_METRICS, index=False)
    print(f"\nMetrics saved → {OUT_METRICS}")

    _print_summary(full_summary)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        main()
