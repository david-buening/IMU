"""
Full clustering analysis for IMU one-sided carrying data.

Four analysis passes per subject (all stratified by subject — no pooled analysis,
which would be dominated by between-subject variation):

  Pass A  — full paired feature space (~370 features, scaled, no PCA)
             K-Means k=2..5, Ward (k=best_K), GMM-diag n=2..5
             DBSCAN skipped (ε is meaningless in ~370 dims)
             Tests H1 baseline.

  Pass B  — classifier top-10 features (scaled, no PCA)
             K-Means k=2..5, Ward, GMM-diag n=2..5, DBSCAN
             Tests H1: can unsupervised clustering recover carrying condition?

  Pass B_ph — Pass B repeated per motion phase (Aufheben / Laufen / Absetzen)
               Tests AH1: does walking produce clearest asymmetry signal?

  Pass B_box — Pass B repeated per box size (big / small)
                Tests AH2: is big box easier to cluster than small?

Reads:
    data/features/feature_matrix_paired.csv

Writes to data/features/plots/{subject}/{pass}/:
    kmeans_{pass}.png              elbow + silhouette
    kdist_{pass}.png               DBSCAN k-distance plot (low-dim passes only)
    gmm_bic_{pass}.png             GMM BIC/AIC model selection
    scatter_cluster_{algo}_{pass}.png
    scatter_{label}_{pass}.png     ground-truth label overlays
    dendrogram_{pass}.png
    heatmap_kmeans_{pass}.png

Writes to data/features/:
    metrics_summary.csv
    feature_matrix_paired_filtered.csv

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
from sklearn.preprocessing import RobustScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from clustering.reduce  import fit_scale, pca_2d
from clustering.cluster import (
    kmeans_selection,
    run_agglomerative,
    dbscan_epsilon_plot, run_dbscan,
    gmm_bic_selection,
)
from clustering.evaluate import (
    internal_metrics, evaluate_all_labels,
    scatter_2d, dendrogram_plot, centroid_heatmap,
    label_purity_table,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURES_DIR        = PROJECT_ROOT / "data" / "features"
PLOTS_DIR           = FEATURES_DIR / "plots"
OUT_METRICS         = FEATURES_DIR / "metrics_summary.csv"
PAIRED_CSV          = FEATURES_DIR / "feature_matrix_paired.csv"
PAIRED_FILTERED_CSV = FEATURES_DIR / "feature_matrix_paired_filtered.csv"

PAIRED_META = [
    "subject", "box_size", "surface",
    "carrying_hand", "one_handed_carry",
    "trial_id", "win_idx", "phase",
    "start_t", "end_t", "n_samples",
]

LABEL_COLS = ["one_handed_carry", "carrying_hand", "phase"]

PHASES    = ["Aufheben", "Laufen", "Absetzen"]
BOX_SIZES = ["big", "small"]

# Top-10 features from the colleague's Random Forest classifier
# (ranked by mean decrease in impurity; F1 ≈ 0.88–0.92 for one_handed_carry).
# abs_diff_Gjerk_mean now correctly maps to the gyro jerk magnitude asymmetry
# feature — previously incorrectly mapped to abs_diff_jerk_std (accel-based).
CLASSIFIER_TOP_FEATURES = [
    "L_AX_mean", "L_AY_mean", "L_AZ_mean",   # RF top-3: static L-wrist tilt
    "R_AX_mean", "R_AY_mean", "R_AZ_mean",   # RF features 4–6: static R-wrist tilt
    "abs_diff_AY_std",                         # cross-wrist AY variability asymmetry
    "abs_diff_GY_std",                         # cross-wrist GY variability asymmetry
    "abs_diff_Gmag_mean",                      # cross-wrist gyro magnitude asymmetry
    "abs_diff_Gjerk_mean",                     # cross-wrist gyro jerk magnitude asymmetry
]


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
# Kurtosis-outlier filter
# ---------------------------------------------------------------------------

def filter_paired_outliers(
    df: pd.DataFrame,
    meta_cols: list[str],
    kurtosis_threshold: float = 30.0,
) -> pd.DataFrame:
    """
    Remove windows where raw sensor kurtosis on either wrist is extreme.

    Checks only L_* and R_* kurtosis features.  A window is removed if any
    raw kurtosis exceeds *kurtosis_threshold*.

    Threshold=30 is a heuristic: for 100-sample windows of steady carrying
    motion, excess kurtosis is typically < 5 (99th pct ≈ 15 across dataset).
    Values above 30 indicate a momentary impact (table contact, drop) creating
    extreme cross-wrist asymmetry unrelated to the experimental condition.
    Kurtosis is mean-invariant, so this filter is unaffected by DC offset.
    """
    feat_cols = [c for c in df.columns if c not in meta_cols]
    # Exclude jerk channels: Ajerk/Gjerk are impulsive by construction,
    # so their kurtosis is naturally high even during normal motion.
    # Only check raw axes and derived magnitudes (Amag, Gmag).
    kurt_cols = [
        c for c in feat_cols
        if "kurtosis" in c
        and (c.startswith("L_") or c.startswith("R_"))
        and "jerk" not in c.lower()
    ]

    if not kurt_cols:
        print("  [filter] No L_/R_ kurtosis columns found — returning unfiltered.")
        return df

    X_kurt = df[kurt_cols].to_numpy(dtype=float)
    outlier_mask = np.any(X_kurt > kurtosis_threshold, axis=1)
    n_removed = int(outlier_mask.sum())
    n_total = len(df)

    print(f"  [filter] Kurtosis outlier filter (threshold={kurtosis_threshold}):")
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
                print(f"      {kurt_cols[i]}: {col_counts[i]} windows "
                      f"(max={X_kurt[:, i].max():.1f})")

    return df[~outlier_mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Shared clustering runner
# ---------------------------------------------------------------------------

def _run_algorithms(
    X_scaled: np.ndarray,
    X_viz: np.ndarray,
    meta_df: pd.DataFrame,
    feat_cols: list[str],
    tag: str,
    label: str,
    out_dir: Path,
    run_dbscan_flag: bool = False,
    k_max: int = 5,
) -> list[dict]:
    """
    Run K-Means k=2..k_max, Ward, GMM-diag n=2..k_max, and optionally DBSCAN.

    Parameters
    ----------
    X_scaled       : scaled feature matrix (full dims — clustering space)
    X_viz          : 2-D PCA projection of X_scaled (for scatter plots)
    meta_df        : metadata DataFrame (same row order, includes label cols)
    feat_cols      : feature column names (for centroid heatmap labels)
    tag            : short pass identifier (e.g. "pass_a", "pass_b")
    label          : human-readable label for plot titles
    out_dir        : directory for saved plots
    run_dbscan_flag: whether to run DBSCAN (only meaningful in low dims)
    k_max          : upper bound for k/n sweep (default 5; lower for small subsets)

    Returns
    -------
    List of metric dicts, one per algorithm configuration.
    """
    # K-Means sweep k=2..k_max
    print(f"\n[1] K-Means sweep (k=2..{k_max})")
    best_k, km_results = kmeans_selection(
        X_scaled, k_min=2, k_max=k_max,
        save_path=out_dir / f"kmeans_{tag}.png",
    )

    # GMM diagonal n=2..k_max
    print(f"\n[2] GMM-diag sweep (n=2..{k_max})")
    best_n, gmm_results = gmm_bic_selection(
        X_scaled, n_min=2, n_max=k_max,
        save_path=out_dir / f"gmm_bic_{tag}.png",
    )

    algorithms: dict[str, np.ndarray] = {}
    for k, res in km_results.items():
        algorithms[f"kmeans_k{k}"] = res["labels"]

    print(f"\n[3] Ward agglomerative (k={best_k})")
    algorithms[f"ward_k{best_k}"] = run_agglomerative(X_scaled, best_k)
    algorithms[f"gmm_n{best_n}"]  = gmm_results[best_n]["labels"]

    # DBSCAN — only in Pass B (10 dims: ε is interpretable)
    if run_dbscan_flag:
        print(f"\n[4] DBSCAN (auto-eps)")
        dbscan_epsilon_plot(
            X_scaled, k=5,
            save_path=out_dir / f"kdist_{tag}.png",
            title=f"k-distance (k=5) — {label}",
        )
        eps_auto  = _knee_eps(X_scaled, k=5)
        labels_db = run_dbscan(X_scaled, eps=eps_auto, min_samples=5)
        algorithms[f"dbscan_eps{eps_auto:.3f}"] = labels_db

    # Ground-truth label scatters (once per pass)
    dummy_labels = np.zeros(len(X_viz), dtype=int)
    for lbl in LABEL_COLS:
        if lbl in meta_df.columns:
            scatter_2d(
                X_viz, dummy_labels,
                colour_by=meta_df[lbl].astype(str).to_numpy(),
                colour_label=lbl,
                title=f"{lbl} — {label}",
                save_path=out_dir / f"scatter_{lbl}_{tag}.png",
            )

    # Dendrogram (subsampled Ward linkage)
    dendrogram_plot(
        X_scaled, max_samples=300,
        save_path=out_dir / f"dendrogram_{tag}.png",
        title=f"Ward dendrogram — {label}",
    )

    # K-Means centroid heatmap
    centroid_heatmap(
        X_scaled,
        km_results[best_k]["labels"],
        feat_cols,
        max_features=min(30, len(feat_cols)),
        save_path=out_dir / f"heatmap_kmeans_{tag}.png",
        title=f"K-Means (k={best_k}) centroid heatmap — {label}",
    )

    # Metrics + scatter plots for each algorithm
    print(f"\nEvaluating {len(algorithms)} configurations …")
    km_scatter_ks = {2, best_k}
    rows: list[dict] = []

    for algo_name, labels in algorithms.items():
        n_cl    = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        print(f"\n  {algo_name}  ({n_cl} clusters, {n_noise} noise)")

        int_m  = internal_metrics(X_scaled, labels)
        ext_df = evaluate_all_labels(labels, meta_df)

        # Scatter plots: skip intermediate K-Means k values
        is_intermediate_km = (
            algo_name.startswith("kmeans_k")
            and int(algo_name.split("kmeans_k")[1]) not in km_scatter_ks
        )
        if not is_intermediate_km:
            scatter_2d(
                X_viz, labels,
                title=f"{algo_name} — {label}",
                save_path=out_dir / f"scatter_cluster_{algo_name}_{tag}.png",
                colour_label="cluster",
            )
            purity = label_purity_table(labels, meta_df)
            print(purity.to_string(index=False))

        row: dict = {
            "tag": tag,
            "algorithm": algo_name,
            "n_windows": len(meta_df),
            "n_clusters": n_cl,
            "n_noise": n_noise,
        }
        row.update(int_m)
        for _, r in ext_df.iterrows():
            row.update({k: v for k, v in r.to_dict().items() if pd.notna(v)})
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Pass A: full feature space
# ---------------------------------------------------------------------------

def run_pass_a(
    df_subj: pd.DataFrame,
    meta_cols: list[str],
    subj: str,
) -> list[dict]:
    """
    Cluster subject *subj* in the full scaled paired feature space (~370 features).

    DBSCAN is skipped because ε (neighbourhood radius) is not interpretable
    in ~370 dimensions (curse of dimensionality).
    """
    tag     = "pass_a"
    label   = f"Pass A [full features] — {subj}"
    out_dir = PLOTS_DIR / subj / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PASS A (full features): {subj}  ({len(df_subj)} windows)")
    print(f"{'='*60}")

    X_scaled, _, feat_cols = fit_scale(df_subj, meta_cols)
    X_viz = pca_2d(X_scaled)

    rows = _run_algorithms(
        X_scaled, X_viz, df_subj[list(meta_cols)],
        feat_cols, tag, label, out_dir,
        run_dbscan_flag=False,
    )
    for r in rows:
        r["subject"] = subj
        r["pass"]    = "A"
    return rows


# ---------------------------------------------------------------------------
# Pass B: classifier top-10 features
# ---------------------------------------------------------------------------

def run_pass_b(
    df_subj: pd.DataFrame,
    meta_cols: list[str],
    subj: str,
) -> list[dict]:
    """
    Cluster subject *subj* using only the top-10 features from the colleague's
    Random Forest classifier.

    Rationale: the classifier identifies mean accelerations and cross-wrist
    asymmetry as most discriminative (F1 ≈ 0.88–0.92).  Clustering in that
    10-feature space tests whether the same signal is geometrically compact.
    DBSCAN is included here because ε is interpretable in 10 dims.
    """
    tag     = "pass_b"
    label   = f"Pass B [clf top-10] — {subj}"
    out_dir = PLOTS_DIR / subj / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PASS B (clf features): {subj}  ({len(df_subj)} windows)")
    print(f"{'='*60}")

    available = [f for f in CLASSIFIER_TOP_FEATURES if f in df_subj.columns]
    missing   = [f for f in CLASSIFIER_TOP_FEATURES if f not in df_subj.columns]
    if missing:
        print(f"  WARNING: columns not found — {missing}")
    if not available:
        print("  No features available — skipping Pass B.")
        return []

    print(f"  Using {len(available)} features: {available}")

    # Scale only the selected features (pass empty meta_cols → all cols are features)
    X_scaled, _, feat_cols = fit_scale(df_subj[available], [])
    X_viz = pca_2d(X_scaled)

    rows = _run_algorithms(
        X_scaled, X_viz, df_subj[list(meta_cols)],
        feat_cols, tag, label, out_dir,
        run_dbscan_flag=True,   # 10 dims: DBSCAN is meaningful
    )
    for r in rows:
        r["subject"] = subj
        r["pass"]    = "B"
    return rows


# ---------------------------------------------------------------------------
# Pass B stratified: AH1 (per phase) and AH2 (per box size)
# ---------------------------------------------------------------------------

def run_pass_b_stratified(
    df_subj: pd.DataFrame,
    meta_cols: list[str],
    subj: str,
    stratify_col: str,
    stratify_val: str,
) -> list[dict]:
    """
    Pass B clustering on a subset of one subject's windows.

    Filters df_subj to rows where stratify_col == stratify_val, then runs
    the same top-10 classifier feature clustering as Pass B.

    Used for:
      AH1 — stratify_col="phase",    stratify_val in PHASES
      AH2 — stratify_col="box_size", stratify_val in BOX_SIZES

    The tag encodes the stratification so output dirs/files don't collide:
      pass_b_ph_Laufen, pass_b_box_big, etc.
    """
    short = {"phase": "ph", "box_size": "box"}.get(stratify_col, stratify_col)
    tag     = f"pass_b_{short}_{stratify_val}"
    label   = f"Pass B [{stratify_col}={stratify_val}] — {subj}"
    out_dir = PLOTS_DIR / subj / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    df_sub = df_subj[df_subj[stratify_col] == stratify_val].reset_index(drop=True)
    n = len(df_sub)

    print(f"\n{'='*60}")
    print(f"  PASS B [{stratify_col}={stratify_val}]: {subj}  ({n} windows)")
    print(f"{'='*60}")

    if n < 20:
        print(f"  Too few windows ({n} < 20) — skipping.")
        return []

    available = [f for f in CLASSIFIER_TOP_FEATURES if f in df_sub.columns]
    missing   = [f for f in CLASSIFIER_TOP_FEATURES if f not in df_sub.columns]
    if missing:
        print(f"  WARNING: columns not found — {missing}")
    if not available:
        print("  No features available — skipping.")
        return []

    # Cap k_max so we never request more clusters than windows // 10
    k_max = min(4, n // 10)
    if k_max < 2:
        print(f"  Too few windows for k>=2 clustering — skipping.")
        return []

    print(f"  Using {len(available)} features, k_max={k_max}")

    X_scaled, _, feat_cols = fit_scale(df_sub[available], [])
    X_viz = pca_2d(X_scaled)

    rows = _run_algorithms(
        X_scaled, X_viz, df_sub[list(meta_cols)],
        feat_cols, tag, label, out_dir,
        run_dbscan_flag=True,
        k_max=k_max,
    )
    for r in rows:
        r["subject"]       = subj
        r["pass"]          = f"B_{short}_{stratify_val}"
        r[stratify_col]    = stratify_val
    return rows


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

_PRINT_COLS = [
    "subject", "pass", "tag", "algorithm", "n_windows", "n_clusters",
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

    print("Loading paired feature matrix …")
    df_paired = pd.read_csv(PAIRED_CSV)
    print(f"  {len(df_paired):,} windows loaded")

    # Kurtosis outlier filter.  Threshold=30 is a heuristic (99th pct ≈ 15);
    # kurtosis is mean-invariant, so this filter is unaffected by DC-offset choices.
    print("\nApplying kurtosis outlier filter …")
    df_filt = filter_paired_outliers(df_paired, PAIRED_META, kurtosis_threshold=30.0)
    df_filt.to_csv(PAIRED_FILTERED_CSV, index=False)
    print(f"  Filtered matrix saved → {PAIRED_FILTERED_CSV}")

    subjects = sorted(df_filt["subject"].unique())
    all_rows: list[dict] = []

    for subj in subjects:
        df_subj = df_filt[df_filt["subject"] == subj].reset_index(drop=True)
        print(f"\n{'#'*60}")
        print(f"  SUBJECT: {subj}  ({len(df_subj)} windows after filter)")
        print(f"{'#'*60}")

        all_rows.extend(run_pass_a(df_subj, PAIRED_META, subj))
        all_rows.extend(run_pass_b(df_subj, PAIRED_META, subj))

        # AH1 — phase-stratified Pass B
        for phase in PHASES:
            all_rows.extend(
                run_pass_b_stratified(df_subj, PAIRED_META, subj, "phase", phase)
            )

        # AH2 — box-size-stratified Pass B
        for box_size in BOX_SIZES:
            all_rows.extend(
                run_pass_b_stratified(df_subj, PAIRED_META, subj, "box_size", box_size)
            )

    full_summary = pd.DataFrame(all_rows)
    full_summary.to_csv(OUT_METRICS, index=False)
    print(f"\nMetrics saved → {OUT_METRICS}")

    _print_summary(full_summary)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        main()
