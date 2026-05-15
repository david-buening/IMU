# Cluster Analysis — IMU One-Sided Carrying
## TU Wien ASM2 | Viktor Hoffmann

---

## 1. Objective

Apply **unsupervised clustering** to wrist-IMU data recorded during carrying tasks to discover whether natural groupings in motion patterns correspond to known experimental conditions:

| Validation target | Values | Hypothesis |
|---|---|---|
| `one_handed_carry` | True / False | One-sided carrying produces asymmetric motion that separates cleanly |
| `carrying_hand` | left / right / both | Left vs right creates a lateralised pattern even on the passive wrist |
| `phase` | Aufheben / Laufen / Absetzen | Pick-up, carry, and set-down phases have distinct dynamics |

This is a **cluster-first, validate-later** approach.  No labels are used during feature extraction or clustering; they are only consulted at the validation step.

---

## 2. Dataset

| Property | Value |
|---|---|
| Subjects | 2 (David, Viktor) |
| Box sizes | big, small |
| Surfaces | floor, table |
| Carrying conditions | left, right, both |
| Sensors | 2 × wrist IMU (left = L, right = R) |
| Total raw rows | 47,413 |
| Nominal sampling rate | ~100 Hz |
| Sensor axes | GX, GY, GZ (°/s), AX, AY, AZ (g) |
| Movement phases | Aufheben (pick-up), Laufen (carry), Absetzen (set-down) |

### 2.1 Known data quality issues

- **4 empty files** (David, big box, both hands, floor): skipped during preprocessing.
- **Gyro scale**: std 29–48 °/s, extremes ±527 °/s — necessitates `RobustScaler`.
- **Scale ratio**: gyro/accel ≈ 100× — if not normalised, gyro dominates all distance calculations.
- **L vs R mounting offset**: mean GY is −20 °/s (L) vs +2 °/s (R); AX is +0.5 g (L) vs −0.5 g (R).  This is a systematic mounting-orientation effect, not a motion difference.  It is removed by subtracting the per-trial mean from each axis before feature extraction (DC offset removal).
- **Shortest Laufen phase**: ~268 samples (~2.7 s) → minimum ~2 complete 1 s windows per trial.
- **`both` condition has two recordings per metadata key** (two repetitions).  The watch-local clock (`time_s`) resets at the start of each recording, so a reset indicates a trial boundary.

---

## 3. Pipeline Overview

```
data/preprocessed.csv  (47 k rows, raw sensor readings)
         │
         ▼
[1] Trial boundary detection
    Detect time_s resets within (subject, box, surface, carrying_hand=both, sensor)
    → assigns trial_id column (0-based integer per group)
         │
         ▼
[2] Sliding windows
    1 s window (100 samples), 50% overlap (50-sample step)
    Discard fragments < 50 samples
    Majority-vote phase label per window (metadata only)
         │
         ├─────────────────────────────────────────────────────────────┐
         ▼                                                             ▼
[3a] SINGLE-SENSOR PATH                                    [3b] PAIRED PATH
     Per window × sensor_hand:                              Align L/R via server_time_s
       DC-offset removal (subtract per-trial mean)          Build 100 Hz common grid
       Time-domain features (6 axes + Amag + Gmag + Jerk)  Per window:
     → feature_matrix_single.csv                             DC-offset removal
                                                             Time-domain features L
                                                             Time-domain features R
                                                             Asymmetry features
                                                           → feature_matrix_paired.csv
         │                                                             │
         └─────────────────────┬─────────────────────────────────────┘
                               ▼
[4] RobustScaler (median/IQR)
    Fit on data, transform all windows
         │
         ▼
[5] PCA — retain 95% explained variance
    Saves scree plot + component loadings
         │
         ▼
[6] Clustering (run independently on single and paired matrices)
    K-Means k=2..8        → elbow + silhouette to select k
    Agglomerative (Ward)  → with dendrogram
    DBSCAN                → ε from k-distance plot
    GMM                   → BIC for model selection
         │
         ▼
[7] Validation
    Internal: Silhouette, Davies-Bouldin, Calinski-Harabász
    External: ARI, AMI, Homogeneity vs {one_handed_carry, carrying_hand, phase}
    Visual:   2D PCA scatter, dendrogram, centroid heatmap
```

---

## 4. Feature Engineering

### 4.1 Time-domain features (per window, per channel)

Computed for each of the 8 channels: **GX, GY, GZ, AX, AY, AZ, Amag, Gmag**

| Feature | Formula | Rationale |
|---|---|---|
| `mean` | μ = Σx / N | DC posture offset |
| `std` | σ = √(Σ(x−μ)² / (N−1)) | Variability |
| `rms` | √(Σx² / N) | Signal energy (standard in ergonomics literature) |
| `range` | max − min | Peak-to-peak dynamics |
| `iqr` | Q75 − Q25 | Robust spread (insensitive to extremes) |
| `skewness` | 3rd standardised moment | Motion asymmetry within the window |
| `kurtosis` | 4th standardised moment (excess) | Impact/spike detection |
| `zero_crossing_rate` | # sign changes / (N−1) | Frequency proxy |
| `energy` | Σx² / N | Average signal power |
| `mean_abs_dev` | mean(\|x−μ\|) | Robust spread alternative |

### 4.2 Derived signals

| Signal | Formula | Features computed |
|---|---|---|
| Amag | √(AX²+AY²+AZ²) | All 10 above |
| Gmag | √(GX²+GY²+GZ²) | All 10 above |
| Jerk | diff(Amag) × fs | std, rms only |
| SMA  | mean(Σ\|Ai\|) per row | 1 scalar (accel-only motion intensity) |

**Total per window (single sensor): ~91 features**
(10 × 8 channels + SMA + jerk_std + jerk_rms)

### 4.3 DC offset removal

Before computing features, the per-trial mean of each axis is subtracted.
This eliminates mounting-orientation biases (the systematic L/R offsets noted in §2.1)
without removing genuine motion differences.

### 4.4 Asymmetry features (paired path only)

For every scalar feature f computed on both L and R:

| Feature | Formula | Reference |
|---|---|---|
| `abs_diff_f` | \|f_R − f_L\| | Absolute bilateral difference |
| `symm_idx_f` | \|f_R − f_L\| / ((|f_R|+|f_L|)/2) × 100 | Robinson et al., 1987 |
| `log_ratio_f` | log(\|f_R\| / (\|f_L\|+ε) + ε) | Symmetric at 0; multiplicative asymmetry |

Signal-level:

| Feature | Formula | Reference |
|---|---|---|
| `pearson_corr_<axis>` | Pearson r(L, R) over window | Synchrony measure |
| `xcorr_max_Amag` | peak of normalised cross-correlation(L Amag, R Amag) | Moe-Nilssen, 1998 |

**Total paired features: ~2 × 91 + 3 × 91 + 6 + 1 ≈ 460 features** (before PCA)

---

## 5. Alignment (L/R synchronisation)

Both wrists share a common server clock (`server_time_s`).  For each trial:

1. Find the **overlapping time range** [max(t_L_min, t_R_min), min(t_L_max, t_R_max)].
2. Build a **regular 100 Hz grid** over that range using `np.linspace`.
3. **Linearly interpolate** each axis of L and R onto the grid.

This produces a paired DataFrame with columns `GX_L, …, AZ_L, GX_R, …, AZ_R` indexed by a common `server_time_s`.  No resampling artefacts are introduced beyond what linear interpolation implies; the original sampling rate is already ≈100 Hz.

---

## 6. Normalisation

### RobustScaler
Applied to the full feature matrix before PCA.
Centres each feature at its **median** and scales by its **IQR** (inter-quartile range).

Chosen over StandardScaler because:
- Gyro data contains extreme outliers (±527 °/s vs IQR ~40 °/s).
- Accel and gyro differ by ~100× in raw magnitude.
- RobustScaler is resistant to both issues.

The scaler is **fit on the full feature matrix** (no train/test split, as this is exploratory clustering, not supervised learning).

---

## 7. Dimensionality Reduction (PCA)

Applied after RobustScaler.  Retains enough principal components to explain **≥95% of the cumulative variance**.

Rationale:
- ~91 (single) or ~460 (paired) features are too many for distance-based clustering (curse of dimensionality).
- Many features are correlated (e.g., std and rms for the same axis).
- PCA loadings reveal which original features drive each component, aiding interpretation.

Outputs saved:
- `data/features/scree_single.png` / `scree_paired.png` — explained variance per component
- Loadings accessible via `reduce.loadings_df(pca, feature_names)`

---

## 8. Clustering Algorithms

Four algorithms are used to cover different geometric assumptions:

### 8.1 K-Means (k = 2…8)
- **Assumption**: spherical, equal-variance clusters.
- **k selection**: silhouette score (primary) + inertia elbow (secondary).
- 20 random initialisations per k to avoid local minima.
- **Output**: `elbow_kmeans_*.png` showing both criteria.

### 8.2 Agglomerative Clustering (Ward linkage)
- **Assumption**: hierarchical, minimises within-cluster variance at each merge step.
- Same geometric assumption as K-Means but reveals the full hierarchy.
- Dendrogram is subsampled to 300 points for readability.
- **k**: set equal to the best K-Means k for comparability.

### 8.3 DBSCAN
- **Assumption**: density-based; identifies arbitrary-shape clusters and marks outliers as noise (label = −1).
- **ε selection**: k-distance plot (k=5); the elbow of the sorted 5-th nearest-neighbour distances is the recommended starting ε.
- `min_samples` default = 5.
- Noise fraction reported after fitting.

### 8.4 Gaussian Mixture Model (GMM)
- **Assumption**: ellipsoidal clusters with a probabilistic generative model.
- **n_components selection**: Bayesian Information Criterion (BIC); lower = better-fitting model with fewest parameters.
- AIC also reported as a secondary criterion.
- Full covariance matrix used (most expressive; feasible given PCA reduction).

---

## 9. Validation

### 9.1 Internal metrics

| Metric | Range | Better |
|---|---|---|
| Silhouette score | [−1, 1] | higher |
| Davies-Bouldin index | [0, ∞) | lower |
| Calinski-Harabász index | [0, ∞) | higher |

Noise points (DBSCAN label = −1) are excluded before computing these.

### 9.2 External metrics (cluster vs ground-truth)

Run for three label columns:

| Label | Type | Research question |
|---|---|---|
| `one_handed_carry` | binary | Does asymmetric load separate from bilateral? |
| `carrying_hand` | 3-class | Is left vs right distinguishable? |
| `phase` | 3-class | Are motion phases separable? |

| Metric | Description |
|---|---|
| ARI (Adjusted Rand Index) | Chance-corrected cluster-label overlap; 1 = perfect, 0 = random |
| AMI (Adjusted Mutual Info) | Information-theoretic; 1 = perfect, 0 = random |
| Homogeneity | Each cluster contains only one label class |
| Completeness | Each class is contained in only one cluster |
| V-measure | Harmonic mean of homogeneity and completeness |

**Original hypothesis**: Paired ARI > Single ARI for `one_handed_carry`, because asymmetry features directly encode the bilateral difference that distinguishes one-sided from two-sided carrying.

**Updated interpretation** (see §13): pooled analysis is dominated by between-subject variance; within-subject stratification is required to test the hypothesis cleanly.

### 9.3 Visualisations

| Plot | File | Description |
|---|---|---|
| Elbow + silhouette | `data/features/elbow_kmeans_*.png` | K-Means k selection |
| GMM BIC/AIC | `data/features/gmm_bic_*.png` | GMM component selection |
| k-distance | `data/features/kdist_*.png` | DBSCAN ε guidance |
| Scree | `data/features/scree_*.png` | PCA variance retained |
| 2D scatter (cluster) | `data/features/scatter_cluster_*.png` | Clusters in PC1–PC2 space |
| 2D scatter (label) | `data/features/scatter_label_*.png` | Labels in PC1–PC2 space |
| Dendrogram | `data/features/dendrogram_*.png` | Ward hierarchy |
| Centroid heatmap | `data/features/heatmap_*.png` | Top-variance features per cluster |

---

## 10. File Structure

```
IMU/
├── data/
│   ├── preprocessed.csv                  input: 47,413 rows
│   └── features/
│       ├── feature_matrix_single.csv     single-sensor windows + features
│       └── feature_matrix_paired.csv     paired windows + asymmetry features
├── src/
│   ├── features/
│   │   ├── windows.py                    trial boundary detection + sliding windows
│   │   ├── align.py                      L/R interpolation to shared 100 Hz grid
│   │   ├── time_domain.py                statistical feature extraction
│   │   └── asymmetry.py                  cross-wrist asymmetry features
│   └── clustering/
│       ├── pipeline.py                   orchestrates feature extraction
│       ├── reduce.py                     RobustScaler + PCA wrapper
│       ├── cluster.py                    K-Means, Ward, DBSCAN, GMM
│       └── evaluate.py                   metrics + plots
└── clustering.md                         this document
```

---

## 11. How to Run

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Extract features

```bash
python src/clustering/pipeline.py
```

Outputs `data/features/feature_matrix_single.csv` and `data/features/feature_matrix_paired.csv`.
Expected shapes:
- Single: `(913, 95)` — 913 windows × 83 features + 12 metadata columns
- Paired: `(601, 433)` — 601 windows × 422 features + 11 metadata columns

### Step 3 — Run full clustering + stratified analysis

```bash
python src/clustering/run_clustering.py
```

Runs four analyses in sequence:
1. Pooled single-sensor (all subjects)
2. Pooled paired (all subjects)
3. Within-subject single (David, Viktor separately)
4. Within-subject paired (David, Viktor separately)

All plots saved under `data/features/plots/{subset}/{tag}/`.
Full metrics table saved to `data/features/metrics_summary.csv`.

### Step 3 (alt) — Interactive / notebook

```python
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "src")

from clustering.reduce  import fit_transform, scree_plot, loadings_df
from clustering.cluster import kmeans_selection, run_kmeans, run_agglomerative, \
                                 dbscan_epsilon_plot, run_dbscan, gmm_bic_selection
from clustering.evaluate import internal_metrics, evaluate_all_labels, \
                                  scatter_2d, dendrogram_plot, centroid_heatmap

# Load single-sensor features
df = pd.read_csv("data/features/feature_matrix_single.csv")

META_COLS = ["subject","box_size","sensor_hand","surface","carrying_hand",
             "one_handed_carry","trial_id","win_idx","phase","start_t","end_t","n_samples"]

# Reduce
X, scaler, pca = fit_transform(df, META_COLS)
scree_plot(pca, save_path="data/features/scree_single.png")

# K-Means selection
best_k, km_results = kmeans_selection(X, save_path="data/features/elbow_kmeans_single.png")
labels_km = km_results[best_k]["labels"]

# Validate
print(internal_metrics(X, labels_km))
print(evaluate_all_labels(labels_km, df[META_COLS]))

# Visualise
scatter_2d(X, labels_km, title="K-Means clusters (single)",
           save_path="data/features/scatter_cluster_single.png")
scatter_2d(X, labels_km, colour_by=df["carrying_hand"].to_numpy(),
           colour_label="carrying_hand",
           title="Ground-truth carrying_hand (single)",
           save_path="data/features/scatter_label_single.png")
```

Repeat with `feature_matrix_paired.csv` for the paired analysis.

---

## 12. File Structure (updated, incl. filtered paired run)

```
IMU/
├── data/
│   ├── preprocessed.csv
│   └── features/
│       ├── feature_matrix_single.csv
│       ├── feature_matrix_paired.csv
│       ├── feature_matrix_paired_filtered.csv   kurtosis-outlier-removed (§15)
│       ├── metrics_summary.csv                  pooled + stratified metrics
│       ├── metrics_summary_filtered.csv         filtered paired metrics
│       └── plots/
│           ├── all/
│           │   ├── single/                 pooled single-sensor plots
│           │   └── paired/                 pooled paired plots (unfiltered)
│           ├── all_filtered/
│           │   └── paired/                 pooled paired plots (filtered)
│           ├── David/
│           │   ├── single/                 David-only single plots
│           │   └── paired/                 David-only paired plots
│           ├── David_filtered/
│           │   └── paired/                 David-only paired plots (filtered)
│           ├── Viktor/
│           │   ├── single/                 Viktor-only single plots
│           │   └── paired/                 Viktor-only paired plots
│           └── Viktor_filtered/
│               └── paired/                 Viktor-only paired plots (filtered)
├── src/
│   ├── features/
│   │   ├── windows.py
│   │   ├── align.py
│   │   ├── time_domain.py
│   │   └── asymmetry.py
│   └── clustering/
│       ├── pipeline.py
│       ├── run_clustering.py               main entry point (pooled + stratified + filtered)
│       ├── reduce.py
│       ├── cluster.py
│       └── evaluate.py
└── clustering.md
```

---

## 13. Results and Interpretation

### 13.1 Numerical fix: Symmetry Index instability

The original `symm_idx` formula is unstable when both sides are near zero — which always occurs for the `mean` feature after per-trial DC offset removal. This caused 3 windows to reach `symm_idx_AX_mean ≈ −4 × 10⁶`, producing a false k=2 silhouette=0.78 in the initial run.

**Fix applied in `asymmetry.py`:**
- If `(|f_R| + |f_L|)/2 < 1e-4`: return 0.0 (signal absent on both sides)
- Cap the result at 200.0 (theoretical maximum per Robinson et al., 1987)

After the fix, the paired matrix has 601 windows × 422 features, NaN=0, Inf=0.  However, extreme kurtosis asymmetry windows persist (see §13.5 and §15 for the follow-up filter).

---

### 13.2 Dimensionality Reduction (PCA)

| Subset | Matrix | Components (95% var) | PC1 (%) | PC2 (%) | Interpretation |
|---|---|---|---|---|---|
| All | Single | 32 | 25.0 | 14.4 | Smooth, no sharp elbow |
| All | Paired | 91 | 15.4 | 14.2 | Very flat — 420+ correlated features |
| David | Single | 31 | 28.9 | 14.3 | PC1 more dominant — consistent gait |
| Viktor | Single | 32 | 18.0 | 15.6 | More balanced — multidimensional motion |
| David | Paired | 76 | — | — | Still heavily distributed |
| Viktor | Paired | 70 | — | — | Still heavily distributed |

**Key observations:**
- Both subjects require ~31 components for 95%, indicating many independent variance directions with no obviously dominant cluster axis.
- The paired matrix needs ~91 components (pooled) — the 3× asymmetry representation per feature adds massive redundancy without adding separable structure.
- Viktor's PC1/PC2 are nearly equal (18%/16%), meaning K-Means (which cuts perpendicular to the leading axis) is structurally less well-suited than GMM for Viktor's data.

---

### 13.3 Single-sensor pooled results

#### K-Means and silhouette

| Algorithm | Silhouette | DB | one_handed_carry ARI | carrying_hand ARI | phase ARI |
|---|---|---|---|---|---|
| kmeans_k2 | **0.406** | 1.841 | 0.000 | −0.003 | 0.049 |
| ward_k2   | **0.472** | 1.696 | 0.005 | −0.012 | 0.053 |
| kmeans_k5 | 0.164 | 2.048 | 0.004 | 0.013 | **0.078** |
| kmeans_k7 | 0.078 | 2.204 | 0.019 | 0.030 | **0.110** |
| gmm_n3    | 0.086 | 2.662 | −0.001 | −0.003 | 0.092 |

Silhouette peaks sharply at k=2 (0.406) and drops to 0.20 at k=3, with no further elbow — a hallmark of a single continuous distribution with one dominant split.

#### What the k=2 split represents

- **C1 (~87%, 797 windows)**: compact dense mass at low PC1; near-zero values on all features → **steady locomotion (Laufen)**
- **C0 (~13%, 116 windows)**: scattered, high-PC1 points; high `jerk_std`, `jerk_rms`, `Amag_kurtosis`, `AX/AZ_range` → **impulsive transitions (Aufheben/Absetzen)**

The centroid heatmap confirms: C0 has feature values up to 3.5× RobustScaler units above median on all impulsive motion features, while C1 is near zero across all 30 highest-variance features.

#### Label overlays (PCA scatter)

- `one_handed_carry` True/False: uniformly interleaved in both clusters — no hint of spatial separation. **ARI = 0.00**.
- `carrying_hand` (both/left/right): all three conditions appear throughout the scatter. **ARI = −0.003** (essentially 0).
- `phase`: Laufen (green) concentrates in C1; Aufheben/Absetzen are more present in C0. Weak but real — **ARI = 0.049**.

#### GMM model selection

GMM BIC minimum at k=3, AIC continues decreasing through k=8 (disagreement typical of near-continuous distributions). The flat BIC landscape (90 000–95 000) means no k is strongly favoured; k=3 weakly suggests the three phases leave a faint imprint, but the effect is not decisive.

#### Ward dendrogram

Clean two-branch structure at Ward distance ~107; no sub-structure above distance 50. Confirms k=2 as the only meaningful partition in the pooled single space.

---

### 13.4 Single-sensor stratified results (within-subject)

#### David single

| Algorithm | Silhouette | DB | one_handed_carry ARI | carrying_hand ARI | phase ARI |
|---|---|---|---|---|---|
| kmeans_k2 | **0.459** | 1.554 | 0.038 | −0.011 | **0.102** |
| ward_k2   | **0.475** | 1.506 | 0.034 | −0.012 | 0.083 |
| kmeans_k7 | 0.082 | 2.013 | **0.058** | 0.045 | 0.116 |
| kmeans_k8 | 0.112 | 1.867 | **0.080** | 0.024 | 0.024 |
| gmm_n3    | 0.209 | 2.065 | 0.006 | −0.007 | 0.067 |

Within David, removing between-subject noise increases silhouette slightly (0.459 vs 0.406 pooled). Phase ARI doubles to **0.102** at k=2 — the phase structure is genuinely more visible within a single subject. One_handed_carry ARI rises to 0.038 (k=2) and 0.080 (k=8), remaining small but above chance.

The phase PCA scatter shows Laufen (green) concentrated in the low-PC1 dense cluster, while Aufheben/Absetzen appear more in the high-PC1 scatter — consistent with a motion-intensity split that partially aligns with phase. Carrying condition (both/left/right) remains fully interleaved with no spatial gradient.

The dendrogram shows a clean two-branch structure (distance ~106), with each branch internally subdividing into 4–5 sub-groups below distance 40 — consistent with the 3-phase × condition structure generating multiple sub-modes that are not yet separable at k=2.

#### Viktor single

| Algorithm | Silhouette | DB | one_handed_carry ARI | carrying_hand ARI | phase ARI |
|---|---|---|---|---|---|
| kmeans_k2 | **0.429** | 1.647 | −0.001 | −0.003 | 0.043 |
| ward_k2   | **0.453** | 1.551 | 0.000 | 0.003 | 0.035 |
| gmm_n2    | 0.196 | 3.015 | 0.014 | 0.030 | **0.160** |
| kmeans_k7 | 0.102 | 2.095 | 0.029 | 0.037 | 0.099 |

Viktor's K-Means k=2 silhouette (0.429) is marginally lower than David's (0.459) — Viktor's data has more spread across PC2, making the hard K-Means cut slightly less efficient. However, **GMM n=2 achieves phase ARI = 0.160**, the highest phase correspondence in the entire dataset.

This is explained by Viktor's scree: PC1 ≈ PC2 ≈ 18% (nearly equal), so the natural cluster shape is elliptical, not axis-aligned — GMM's elliptical Gaussian fits it better. The BIC curve minimum is sharply at k=2, confirming 2-component GMM is the correct generative model for Viktor's data.

The phase scatter (GMM n=2 colouring) shows: Laufen (green) at low PC1, Aufheben (orange) at mid-to-high PC1, Absetzen (blue) spread broadly. ARI=0.16 means phase accounts for 16% of cluster assignment variance — real signal, but far below what would be needed for reliable phase detection (would need ≥0.5). Viktor's centroid heatmap also includes `zero_crossing_rate` variants among top features (absent in David), suggesting Viktor's transition phases involve more oscillatory motion.

---

### 13.5 Paired pooled results — kurtosis outlier artefact confirmed

Despite the symm_idx fix, the pooled paired k=2 silhouette remains 0.778.

| Algorithm | Silhouette | DB | one_handed_carry ARI | carrying_hand ARI | phase ARI |
|---|---|---|---|---|---|
| kmeans_k2 | **0.778** | 0.636 | 0.001 | −0.004 | 0.001 |
| ward_k2   | **0.778** | 0.636 | 0.001 | −0.004 | 0.001 |
| gmm_n2    | 0.558 | 2.371 | −0.001 | 0.005 | 0.021 |
| dbscan    | 0.502 | 0.701 | −0.000 | 0.005 | 0.012 |

**Root cause**: The scatter plot (kmeans_k2_paired, pooled) shows 3 orange points at PC2 = 95, 133, 140 — far above the 598-window mass near PC2 ≈ 0. These 3 windows form "cluster 1". The centroid heatmap reveals their signature: `abs_diff_GX_kurtosis ≈ 47`, `R_AZ_kurtosis ≈ 35`, `R_GX_kurtosis ≈ 30` — all near the ±50 clip boundary. The Ward dendrogram confirms: these samples merge with the rest at Ward distance ~245, while all other merges occur below 110.

**Physical interpretation**: These are windows where one wrist received a momentary impact (table contact, box drop) creating extreme kurtosis on that wrist only. The `abs_diff_kurtosis` asymmetry feature amplifies this to the clip boundary. They are real events but physically atypical — they do not represent the carrying conditions of interest.

All external ARIs are ≈ 0. The paired pooled analysis is entirely uninformative until these windows are removed.

---

### 13.6 Paired stratified results — artefact persists per-subject

**David paired:**

| Algorithm | Silhouette | DB | one_handed_carry ARI | carrying_hand ARI | phase ARI |
|---|---|---|---|---|---|
| kmeans_k2 | **0.796** | 0.137 | 0.002 | −0.002 | 0.003 |
| ward_k2   | **0.796** | 0.137 | 0.002 | −0.002 | 0.003 |
| dbscan_eps29.9 | 0.562 | 0.505 | −0.010 | 0.008 | 0.015 |
| kmeans_k5 | 0.241 | 1.711 | −0.005 | 0.010 | **0.091** |
| kmeans_k7 | 0.208 | 1.556 | −0.003 | 0.011 | **0.101** |

The k=2 scatter shows a single orange outlier at PC1=143 — one extreme kurtosis window dominates the 2-cluster split. The DBSCAN plot (eps=29.9) is the most informative: DBSCAN correctly labels the extreme outliers as **noise (−1)** and recovers a single main cluster, with 2–3 borderline points forming a tiny second cluster. This is the appropriate interpretation: the data has one real cluster with a tail of outliers.

At higher k (k=5,7) where the outlier's influence is diluted, phase ARI reaches 0.091–0.101 — comparable to David single-sensor, suggesting the paired space does not add phase information beyond single-sensor once the artefact is controlled.

**Viktor paired:**

| Algorithm | Silhouette | DB | one_handed_carry ARI | carrying_hand ARI | phase ARI |
|---|---|---|---|---|---|
| kmeans_k2 | **0.767** | 0.606 | −0.001 | −0.005 | −0.000 |
| ward_k2   | **0.767** | 0.606 | −0.001 | −0.005 | −0.000 |
| gmm_n2    | **0.661** | 0.244 | −0.001 | −0.005 | 0.012 |
| kmeans_k3 | 0.552 | 1.031 | −0.001 | −0.007 | 0.012 |
| kmeans_k4 | 0.555 | 1.246 | −0.002 | −0.011 | 0.022 |
| kmeans_k8 | **0.522** | 0.738 | 0.001 | 0.007 | 0.032 |

Viktor's paired silhouette profile is unusual: values remain elevated at k=3,4,5,6 (0.51–0.56), unlike all other analyses where silhouette collapses to 0.10–0.20 above k=2. This may indicate secondary outlier structure at multiple scales — two extreme windows at PC2 ≈ 113, 135 form one artefact cluster; the broader spread of the Viktor paired space at intermediate k may represent additional impact windows.

All external ARIs ≈ 0 for all k configurations. There is no asymmetry signal.

---

### 13.7 Summary and hypothesis assessment

| Hypothesis | Result | Evidence |
|---|---|---|
| One-sided carrying produces separable asymmetric motion | **Not supported** | ARI for `one_handed_carry` ≤ 0.08 across all analyses; no spatial separation in any PCA scatter |
| Left vs right carrying creates lateralised patterns | **Not supported** | ARI for `carrying_hand` ≤ 0.08 across all analyses; carrying_hand scatter uniformly mixed |
| Phase (pick-up / carry / set-down) is partially recoverable | **Weakly supported** | Phase ARI up to 0.16 (Viktor GMM n=2); Laufen visually concentrates in low-PC1 cluster |
| Paired (asymmetry) features outperform single-sensor | **Not supported** | Paired ARIs ≤ single-sensor ARIs for all label columns; paired dominated by kurtosis artefacts |

**Why the carrying-condition hypothesis fails:**

1. **Wrist kinematics encode arm swing, not load side.** Both wrists swing freely during walking regardless of which hand grips the box. The wrist sensor does not directly measure grip force or joint load — only arm oscillation.
2. **DC offset removal destroys the mean.** The most informative asymmetry (is one wrist accelerating more on average?) is removed by the per-trial mean subtraction required to neutralise sensor mounting offsets.
3. **n=2 subjects.** Any genuine effect would need to exceed the combined variability from surface, box size, trial order, and individual gait — with only 2 subjects providing evidence.

**What does cluster:** Motion intensity (phase-correlated). Within-subject, the walking phase has detectably lower jerk and kurtosis than transition phases. This drives all cluster structures observed.

---

## 14. Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| n=2 subjects | Between-subject confounding dominates pooled PCA; within-subject results have no generalisability | Explicit stratification (§13.4, §13.6); report per-subject |
| DC offset removal destroys `mean` | `symm_idx_mean` and `log_ratio_mean` are always ~0; cannot test mean-level asymmetry | Accepted trade-off; asymmetry analysis uses std/rms/energy/kurtosis features |
| `symm_idx` instability (fixed) | Pre-fix: false k=2 silhouette=0.78; post-fix: residual kurtosis artefacts persist | Fixed in `asymmetry.py`; kurtosis-outlier filter added (§15) |
| Kurtosis asymmetry at impact windows | 2–3 windows per subject dominate paired k=2 clustering entirely | IQR-based filter removes them before running filtered paired analysis |
| 4 missing trials (David, big, floor, both) | David's both-condition data is incomplete for one cell | Noted; does not affect other conditions |
| Phase-boundary windows | ~50% of phase-boundary windows have mixed labels; majority-vote phase label is approximate | Inherent to label-agnostic windowing; majority vote is standard practice |
| 50% window overlap | Consecutive windows share 50 samples → temporal autocorrelation inflates effective n | Standard in HAR literature; no correction applied in this exploratory study |

---

## 15. Filtered Paired Analysis (kurtosis-outlier removal)

### 15.1 Motivation

The paired k=2 clustering is dominated by 2–3 windows per subject where one wrist received an extreme impact (kurtosis asymmetry features saturating the ±50 clip boundary). These windows are artefacts — not representative of the carrying conditions of interest. Removing them allows a fair test of whether any residual asymmetry structure corresponds to experimental labels.

### 15.2 Filter method

Only the raw sensor kurtosis features (`L_*_kurtosis` and `R_*_kurtosis`) are checked — asymmetry-derived columns (`abs_diff_*`, `symm_idx_*`, `log_ratio_*`) operate on different scales and are excluded. A window is removed if **any** raw kurtosis feature exceeds an absolute threshold:

```
threshold = 30   (excess kurtosis)
```

Justification: for 100-sample windows of steady carrying motion, excess kurtosis is typically < 5. The 99th percentile across the full dataset is ~15. Values above 30 indicate a momentary impact on one wrist (table contact, accidental drop). This threshold removes 14 windows (2.3% of 601) — far more targeted than IQR-based methods, which falsely remove 30% due to the heavy-tailed nature of kurtosis distributions.

The filtered matrix is saved to `data/features/feature_matrix_paired_filtered.csv`.

### 15.3 How to run

```bash
python src/clustering/run_clustering.py
```

The script now runs 6 analyses in sequence:
1. Pooled single (all subjects)
2. Pooled paired — **unfiltered** (for comparison)
3. Pooled paired — **filtered**
4. Stratified single (David, Viktor separately)
5. Stratified paired — unfiltered
6. Stratified paired — **filtered**

Filtered plots are saved under `data/features/plots/{subset}_filtered/paired/`.
Filtered metrics are appended to `metrics_summary.csv` with `matrix = "paired_filtered"`.

### 15.4 Results

#### Effect of filter

The filter removed 14 windows (2.3%): 9 from Viktor, 5 from David, all triggered by R-wrist raw kurtosis > 30. The impact on best-silhouette k=2:

| Subset | Unfiltered k=2 silhouette | Filtered best silhouette | Change |
|---|---|---|---|
| all | 0.778 | 0.551 (k=2) | −0.227 |
| Viktor | 0.767 | 0.566 (k=3 Ward) | −0.201 |
| David | 0.796 | 0.580 (DBSCAN) | −0.216 |

The ~0.22 drop confirms that the unfiltered paired k=2 structure was almost entirely artefactual. The remaining silhouette (~0.55) is genuine but modest.

#### Pooled filtered (all_filtered, n=587)

| Algorithm | Silhouette | carrying_hand ARI | one_handed_carry ARI | phase ARI |
|---|---|---|---|---|
| kmeans_k2 | **0.551** | −0.004 | 0.001 | 0.007 |
| kmeans_k3 | 0.510 | 0.008 | −0.001 | 0.021 |
| kmeans_k5 | 0.214 | 0.059 | 0.016 | 0.041 |
| kmeans_k6 | 0.166 | 0.101 | 0.048 | 0.014 |
| gmm_n3    | 0.534 | 0.006 | −0.001 | 0.018 |

The pooled filtered k=2 heatmap shows C1 (the larger cluster) has uniformly negative `log_ratio_*` — this reflects David's systematic L>R loading rather than a carrying-condition effect. Between-subject confounding persists in the pooled space even after filtering. All external ARIs at sensible silhouette values (k=2,3) are ≈ 0.

#### Viktor_filtered (n=325) — partial asymmetry signal

| Algorithm | Silhouette | carrying_hand ARI | one_handed_carry ARI | phase ARI |
|---|---|---|---|---|
| kmeans_k2 | 0.564 | −0.002 | −0.001 | 0.013 |
| kmeans_k3 | 0.566 | −0.005 | −0.001 | 0.022 |
| kmeans_k5 | 0.528 | 0.007 | 0.001 | 0.035 |
| kmeans_k6 | 0.207 | **0.166** | **0.102** | 0.017 |
| kmeans_k7 | 0.178 | **0.187** | **0.122** | 0.019 |
| kmeans_k8 | 0.111 | **0.417** | **0.346** | 0.014 |
| ward_k3   | **0.566** | −0.002 | −0.001 | 0.020 |
| dbscan    | 0.565 | 0.021 | 0.005 | 0.014 |

**Key finding — k=3 heatmap (intrinsic asymmetry structure):**

The silhouette elbow selects k=3 (Ward: 0.566, K-Means: 0.566). The centroid heatmap for k=3 reveals the top-30 most discriminative features are entirely `log_ratio_*` features, with a clear three-regime pattern:

- **C0**: all log_ratio features strongly **negative** → L-wrist feature magnitude > R-wrist (left-dominant motion windows)
- **C1**: all log_ratio features near **zero** → bilateral symmetry
- **C2**: all log_ratio features strongly **positive** → R-wrist > L-wrist (right-dominant)

This is genuine bilateral asymmetry structure. The three clusters represent left-dominant, balanced, and right-dominant motion windows — the conceptual segmentation the asymmetry hypothesis predicted. However, the k=3 ARI ≈ 0, meaning this L/R dominance structure does not cleanly partition by `carrying_hand` label. The labels are distributed across all three clusters: a left-carry trial produces some left-dominant windows (C0), some balanced windows (C1), and a few right-dominant windows (C2).

**High-k ARI (k=6–8):**

A sharp silhouette cliff occurs at k=5→6 (0.528→0.207). Beyond this point, internal validity is poor, but external ARI increases (carrying_hand: 0.17 at k=6, 0.19 at k=7, 0.42 at k=8). The k=8 scatter shows three condition-pure peripheral sub-populations:
1. **Bottom cluster** (PC2 ≈ −50): 100% `carrying_hand=left`, 100% `one_handed_carry=True`
2. **Left scatter** (PC1 < −40): 100% `carrying_hand=both`

These small pure clusters mechanically inflate ARI without reflecting coherent within-cluster structure. **The correct interpretation: a few specific windows are strongly identifiable by carrying condition; the majority are not distinguishable.** k=3 with silhouette=0.566 is the appropriate clustering solution for Viktor's filtered paired data.

#### David_filtered (n=262) — residual artefact, no signal

| Algorithm | Silhouette | carrying_hand ARI | one_handed_carry ARI | phase ARI |
|---|---|---|---|---|
| kmeans_k2 | 0.501 | 0.010 | −0.007 | 0.015 |
| kmeans_k3 | 0.307 | 0.004 | 0.001 | **0.077** |
| kmeans_k8 | 0.160 | 0.034 | 0.013 | 0.040 |
| dbscan    | **0.580** | 0.008 | −0.012 | 0.019 |

The k=2 scatter shows 3–4 residual outlier windows at PC2 ≈ −50. The heatmap identifies these as having extreme `symm_idx_AX_mean ≈ −50` — a mean-feature symmetry index artefact not caught by the kurtosis filter (the mean after DC removal is near zero but not identically zero, making the symmetry-index denominator unstable). DBSCAN appropriately labels these as noise. All external ARIs are ≈ 0 at valid silhouette values; David's filtered paired data carries no carrying-condition signal.

#### Revised hypothesis assessment

| Hypothesis | Pre-filter | Post-filter |
|---|---|---|
| One-sided carry separable | Not supported (ARI ≤ 0.08) | Weak partial support: Viktor k=8 ARI=0.35; driven by peripheral clusters, not general structure |
| Left vs right separable | Not supported | Weak partial support: Viktor k=8 ARI=0.42; same peripheral clusters |
| Phase partially recoverable | Weakly supported | Unchanged (max ARI 0.077 David k=3) |
| Asymmetry features add value | Not supported | **Conditional support**: Viktor k=3 log_ratio 3-cluster structure (L-dominant / balanced / R-dominant) is genuine asymmetry signal but too diffuse to align with label partitions |

**Subject asymmetry in results:** Viktor shows partial signal; David does not. Two plausible explanations: (1) David's sensor mounting was more mechanically symmetric, reducing recoverable lateralisation; (2) David's carrying kinematics are more bilateral, dampening the L/R difference. With n=2 subjects, neither explanation can be confirmed.

**Overall conclusion:** Filtering was necessary to expose genuine signal. The genuine signal is modest (k=3 log_ratio structure), subject-specific (Viktor only), and insufficient to support the original carrying-condition hypothesis in a generalisable way. The dominant clustering structure across all analyses remains motion-intensity and phase-correlated dynamics, not carrying lateralisation.

---

## 16. References

1. Robinson, R. O., Herzog, W., & Nigg, B. M. (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. *Journal of Manipulative and Physiological Therapeutics*, 10(4), 172–176.
2. Moe-Nilssen, R. (1998). A new method for evaluating motor control in gait under real-life environmental conditions. Part 1: The instrument. *Clinical Biomechanics*, 13(4–5), 320–327.
3. Kavanagh, J. J., & Menz, H. B. (2008). Accelerometry: A technique for quantifying movement patterns during walking. *Gait & Posture*, 28(1), 1–15.
4. Lara, O. D., & Labrador, M. A. (2013). A survey on human activity recognition using wearable sensors. *IEEE Communications Surveys & Tutorials*, 15(3), 1192–1209.
5. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
6. Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224–227.
7. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*, 96(34), 226–231.
