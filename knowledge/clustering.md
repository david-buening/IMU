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
- **L vs R mounting offset**: mean GY is −20 °/s (L) vs +2 °/s (R); AX is ±0.5 g.  This is a systematic mounting-orientation effect.  DC offset removal was considered but ultimately **not applied** (see §7 design decision).
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
       Time-domain features (10 channels, 91 total)         Build 100 Hz common grid
     → feature_matrix_single.csv                            Per window:
                                                              Time-domain features L (91)
                                                              Time-domain features R (91)
                                                              Asymmetry features   (188)
                                                            → feature_matrix_paired.csv
         │                                                             │
         └─────────────────────┬─────────────────────────────────────┘
                               ▼
[4] Kurtosis outlier filter
    Remove windows where any L/R raw sensor kurtosis > 30
    (excludes jerk kurtosis columns — see §8)
         │
         ▼
[5] RobustScaler (median/IQR) + clip ±50 + zero non-finite
    No PCA — clustering in full scaled space
    2-D PCA computed separately for scatter plot visualisation only
         │
         ▼
[6] Two analysis passes per subject (stratified; no pooled analysis)

    PASS A — full paired feature space (~370 features)
      K-Means k=2..5        → elbow + silhouette to select k
      Agglomerative (Ward)  → k = best K-Means k
      GMM (diag cov)        → BIC for n=2..5
      DBSCAN                → SKIPPED (ε meaningless in ~370 dims)
      Tests H1 baseline (full feature space)

    PASS B — classifier top-10 features
      K-Means k=2..5
      Agglomerative (Ward)
      GMM (diag cov) n=2..5
      DBSCAN                → auto-ε from k-distance knee
      Tests H1: can unsupervised clustering recover carrying condition?

    PASS B_ph_* — Pass B repeated per motion phase (Aufheben / Laufen / Absetzen)
      Same algorithms as Pass B, k_max capped by subset size
      Tests AH1: does walking (Laufen) produce the clearest asymmetry signal?

    PASS B_box_* — Pass B repeated per box size (big / small)
      Same algorithms as Pass B, k_max capped by subset size
      Tests AH2: is the big box easier to cluster than the small box?
         │
         ▼
[7] Validation
    Internal: Silhouette, Davies-Bouldin
    External: ARI, AMI vs {one_handed_carry, carrying_hand, phase}
    Label-purity table: dominant label + % per cluster
    Visual:   2D PCA scatter (viz only), dendrogram, centroid heatmap
```

---

## 4. Feature Engineering

### 4.1 Time-domain statistics (per channel)

9 statistics are computed for each of the 10 channels listed in §4.2:

| Feature | Formula | Rationale |
|---|---|---|
| `mean` | μ = Σx / N | DC posture offset |
| `std` | σ = √(Σ(x−μ)² / (N−1)) | Variability |
| `rms` | √(Σx² / N) | Signal energy |
| `range` | max − min | Peak-to-peak dynamics |
| `iqr` | Q75 − Q25 | Robust spread |
| `skewness` | 3rd standardised moment (bias-corrected) | Motion asymmetry within window |
| `kurtosis` | 4th standardised moment, Fisher (excess) | Impact / spike detection |
| `zero_crossing_rate` | # sign changes / (N−1), centred | Frequency proxy |
| `mean_abs_dev` | mean(|x − μ|) | Robust spread alternative |

Note: `energy` (= rms²) was removed as redundant with `rms`.

### 4.2 Signal channels

| Channel | Formula | Notes |
|---|---|---|
| GX, GY, GZ | raw gyro axes | °/s |
| AX, AY, AZ | raw accel axes | g |
| Amag | √(AX²+AY²+AZ²) | L2 norm of acceleration |
| Gmag | √(GX²+GY²+GZ²) | L2 norm of gyroscope |
| Ajerk | √(Σ diff(accel)²) frame-to-frame | Vector accel jerk magnitude, not scaled by fs |
| Gjerk | √(Σ diff(gyro)²) frame-to-frame | Vector gyro jerk magnitude, not scaled by fs |

**SMA** (Signal Magnitude Area): mean(Σ|Ai|) per row — 1 scalar, accel only.

**Total per window (single sensor): 9 × 10 + 1 (SMA) = 91 features**

### 4.3 Design decision: no DC offset removal

Per-trial mean subtraction was considered to neutralise the systematic L/R mounting
offset (GY: −20 vs +2 °/s; AX: ±0.5 g).  It was ultimately **not applied** because:

- It destroys the `mean` features, which are the top-3 most important features in the colleague's Random Forest classifier (`L/R_AX/AY/AZ_mean`).
- Kurtosis is mean-invariant, so the outlier filter is unaffected by this choice.
- The asymmetry `abs_diff_mean` and `ratio_mean` features still encode cross-wrist mean differences without removing them from within-wrist features.

### 4.4 Asymmetry features (paired path only)

For every scalar feature f computed on both L and R:

| Feature | Formula | Notes |
|---|---|---|
| `abs_diff_f` | \|f_R − f_L\| | Absolute bilateral difference |
| `ratio_f` | f_R / (f_L + ε) | Signed, linear scale; matches classifier intent |

Signal-level (per raw axis):

| Feature | Formula | Notes |
|---|---|---|
| `pearson_corr_<axis>` | Pearson r(L, R) over window | Synchrony measure; 6 features (GX GY GZ AX AY AZ) |

Removed from earlier versions: `symm_idx` (unstable when denominator ≈ 0),
`log_ratio` (loses sign for negative features), `xcorr_max_Amag` (redundant with pearson_corr at lag-0).

**Total paired features: 91 + 91 + 2×91 + 6 = 370**
(L features + R features + abs_diff + ratio + 6 pearson correlations)

---

## 5. Alignment (L/R synchronisation)

Both wrists share a common server clock (`server_time_s`).  For each trial:

1. Find the **overlapping time range** [max(t_L_min, t_R_min), min(t_L_max, t_R_max)].
2. Build a **regular 100 Hz grid** over that range using `np.linspace`.
3. **Linearly interpolate** each axis of L and R onto the grid.

This produces a paired DataFrame with columns `GX_L, …, AZ_L, GX_R, …, AZ_R` indexed by a common `server_time_s`.

---

## 6. Normalisation

### RobustScaler
Applied to the feature matrix before clustering.
Centres each feature at its **median** and scales by its **IQR**.

Chosen over StandardScaler because:
- Gyro data contains extreme outliers (±527 °/s vs IQR ~40 °/s).
- Accel and gyro differ by ~100× in raw magnitude.
- RobustScaler is resistant to both.

After scaling, values are **clipped to ±50** (suppresses extreme ratio features when
denominator ≈ 0) and any remaining non-finite values are zeroed.

---

## 7. Dimensionality Reduction

**PCA is not used for clustering.**

Removing PCA was a deliberate design decision after observing that PCA consistently
discarded the carrying-condition signal:
- The dominant variance in the feature space comes from motion intensity (jerk, kurtosis,
  range) which is correlated with movement phase, not carrying condition.
- PCA projects the discriminative direction into low-variance components and effectively
  discards it before clustering begins.
- Clustering directly in the scaled feature space (370 dims for Pass A, 10 dims for Pass B)
  avoids this information loss.

**A 2-component PCA is computed separately for scatter plot visualisation only.**
It is not used as input to any clustering algorithm.

---

## 8. Kurtosis Outlier Filter

### Motivation
In the full paired feature space, a small number of windows (2–3 per subject) where one
wrist received a momentary impact (table contact, box drop) generate extreme kurtosis
asymmetry values that dominate the k=2 clustering entirely, producing a false
silhouette of 0.78.

### Method
For each window, check the raw L/R kurtosis features only:
```
kurt_cols = [c for c in feat_cols
             if "kurtosis" in c
             and (c.startswith("L_") or c.startswith("R_"))
             and "jerk" not in c.lower()]
```

**Jerk channels are explicitly excluded** from the filter: Ajerk and Gjerk are impulsive
by construction (they measure frame-to-frame differences), so their kurtosis is
naturally high even during normal motion. Including them would remove ~55% of all windows.

Threshold: **excess kurtosis > 30** on any checked column → window removed.

Justification: For 100-sample windows of steady carrying motion, excess kurtosis is
typically < 5 (99th percentile ≈ 15 across the full dataset). Values above 30 indicate
a momentary mechanical event unrelated to the experimental condition.

### Effect
- Windows removed: **14 / 601 (2.3%)**
- David: 9 windows, Viktor: 5 windows
- Top triggering columns: R_GZ_kurtosis (max 56.0), R_AX_kurtosis (max 47.8), R_Gmag_kurtosis (max 90.2)

---

## 9. Clustering Algorithms

### 9.1 K-Means (k = 2…5)
- **Assumption**: spherical, equal-variance clusters.
- **k selection**: silhouette score (primary) + inertia elbow (secondary).
- 20 random initialisations per k to avoid local minima.

### 9.2 Agglomerative Clustering (Ward linkage)
- **Assumption**: hierarchical, minimises within-cluster variance at each merge step.
- k set equal to the best K-Means k for comparability.
- Dendrogram subsampled to 300 points for readability.

### 9.3 DBSCAN
- **Assumption**: density-based; marks outliers as noise (label = −1).
- **ε selection**: k-distance plot knee heuristic (k=5).
- `min_samples` = 5.
- **Run only in Pass B** (10-dimensional space — ε is interpretable).
- Skipped in Pass A: ε is not interpretable in ~370 dimensions.

### 9.4 Gaussian Mixture Model (GMM)
- **Assumption**: ellipsoidal clusters with a probabilistic generative model.
- `covariance_type="diag"` — diagonal covariance to avoid singular matrices in high dimensions.
- **n selection**: Bayesian Information Criterion (BIC) for n=2..5; lower = better fit.

---

## 10. Validation

### 10.1 Internal metrics

| Metric | Range | Better |
|---|---|---|
| Silhouette score | [−1, 1] | higher |
| Davies-Bouldin index | [0, ∞) | lower |

Noise points (DBSCAN label = −1) are excluded before computing these.

### 10.2 External metrics (cluster vs ground-truth)

Run for three label columns: `one_handed_carry`, `carrying_hand`, `phase`.

| Metric | Description |
|---|---|
| ARI (Adjusted Rand Index) | Chance-corrected cluster-label overlap; 1 = perfect, 0 = random |
| AMI (Adjusted Mutual Info) | Information-theoretic equivalent |

### 10.3 Label-purity table

For each cluster, the dominant label value and its percentage are reported.
This complements ARI by showing which condition a cluster "represents" even when
ARI is low due to small pure sub-clusters alongside a large mixed majority.

---

## 11. Classifier Top-10 Features (Pass B)

Features from a colleague's Random Forest classifier (F1 ≈ 0.88–0.92 for `one_handed_carry`),
mapped to the updated feature names after adding the Gjerk channel:

```python
CLASSIFIER_TOP_FEATURES = [
    "L_AX_mean", "L_AY_mean", "L_AZ_mean",    # RF top-3: static L-wrist tilt
    "R_AX_mean", "R_AY_mean", "R_AZ_mean",    # RF features 4–6: static R-wrist tilt
    "abs_diff_AY_std",                          # cross-wrist AY variability asymmetry
    "abs_diff_GY_std",                          # cross-wrist GY variability asymmetry
    "abs_diff_Gmag_mean",                       # cross-wrist gyro magnitude asymmetry
    "abs_diff_Gjerk_mean",                      # cross-wrist gyro jerk magnitude asymmetry
                                                # (was abs_diff_jerk_std — accel-only; now
                                                #  correctly mapped to the Gjerk channel)
]
```

---

## 12. File Structure

```
IMU/
├── data/
│   ├── preprocessed.csv                       generated: 47,413 rows
│   └── features/                              all generated — not tracked in git
│       ├── feature_matrix_paired.csv          paired windows before filter (601 × 381)
│       ├── feature_matrix_paired_filtered.csv kurtosis-outlier-removed (587 × 381)
│       ├── metrics_summary.csv                all clustering metrics (all passes)
│       └── plots/
│           ├── David/
│           │   ├── pass_a/                    Pass A — full 370 features
│           │   ├── pass_b/                    Pass B — clf top-10 features
│           │   ├── pass_b_ph_Aufheben/        AH1: pick-up phase only
│           │   ├── pass_b_ph_Laufen/          AH1: walking phase only
│           │   ├── pass_b_ph_Absetzen/        AH1: put-down phase only
│           │   ├── pass_b_box_big/            AH2: big box only
│           │   └── pass_b_box_small/          AH2: small box only
│           └── Viktor/
│               └── (same structure as David/)
├── src/
│   ├── features/
│   │   ├── windows.py                         trial boundary detection + sliding windows
│   │   ├── align.py                           L/R interpolation to shared 100 Hz grid
│   │   ├── time_domain.py                     91 features per window
│   │   └── asymmetry.py                       188 cross-wrist asymmetry features
│   └── clustering/
│       ├── pipeline.py                        orchestrates feature extraction
│       ├── reduce.py                          RobustScaler + pca_2d (viz only)
│       ├── cluster.py                         K-Means, Ward, DBSCAN, GMM
│       ├── evaluate.py                        metrics + plots + label_purity_table
│       └── run_clustering.py                  main entry point (all passes per subject)
└── knowledge/clustering.md                    this document
```

---

## 13. How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Extract feature matrices
```bash
python src/clustering/pipeline.py
```

Expected output shapes:
- Single: `(913, 103)` — 913 windows × 91 features + 12 metadata columns
- Paired: `(601, 381)` — 601 windows × 370 features + 11 metadata columns

### Step 3 — Run clustering analysis
```bash
python src/clustering/run_clustering.py
```

This runs in sequence for each subject (David, Viktor):
1. Kurtosis filter → saves `feature_matrix_paired_filtered.csv`
2. Pass A — full 370-feature space (K-Means, Ward, GMM; no DBSCAN)
3. Pass B — classifier top-10 features (K-Means, Ward, GMM, DBSCAN)
4. Pass B_ph_* — Pass B stratified per phase: Aufheben, Laufen, Absetzen (tests AH1)
5. Pass B_box_* — Pass B stratified per box size: big, small (tests AH2)

All plots saved to `data/features/plots/{subject}/{pass}/`.
Full metrics table saved to `data/features/metrics_summary.csv`.

---

## 14. Results

### 14.1 Pass A — Full 370-feature space

**Conclusion: strong internal structure, zero external relevance.**

| Subject | Algorithm | Silhouette | OHC ARI | CH ARI |
|---|---|---|---|---|
| David | K-Means k=2 | **0.831** | 0.001 | −0.001 |
| David | Ward k=2 | **0.831** | 0.001 | −0.001 |
| Viktor | K-Means k=2 | **0.814** | −0.001 | −0.003 |
| Viktor | GMM n=3 | 0.800 | −0.001 | −0.003 |

All carrying-condition ARIs are ≈ 0 across all k configurations.
The 370-feature scaled space clusters strongly — but it captures motion dynamics
(intensity, jerk variation across trials) rather than carrying condition.
The 2D PCA scatter visualisation is additionally distorted by extreme `ratio_*`
outliers that dominate PC1, compressing the main cluster to a single region.

### 14.2 Pass B — Classifier top-10 features

**Conclusion: real carrying-condition signal, subject-dependent strength.**

**Viktor (325 windows):**

| Algorithm | Silhouette | OHC ARI | CH ARI | Interpretation |
|---|---|---|---|---|
| K-Means k=2 | 0.253 | **0.404** | 0.233 | Clean binary split |
| K-Means k=5 | 0.269 | 0.299 | 0.423 | Sub-condition groups emerge |
| Ward k=5 | 0.254 | 0.329 | 0.409 | Similar to K-Means k=5 |
| **GMM n=5** | 0.211 | **0.387** | **0.519** | **Headline result** |

GMM n=5 (label-purity detail):

| Cluster | n | carrying_hand | % |
|---|---|---|---|
| 0 | 107 | both | **100%** |
| 1 | 58 | right | **100%** |
| 4 | 45 | left | **100%** |
| 2 | 63 | both | 87% |
| 3 | 52 | left | 88% |

Three of five clusters are perfectly pure.  GMM independently recovers the three
experimental carrying conditions as natural density modes in the 10-feature space.

**David (262 windows):**

| Algorithm | Silhouette | OHC ARI | CH ARI |
|---|---|---|---|
| K-Means k=2 | 0.328 | −0.005 | 0.055 |
| **K-Means k=3** | 0.245 | **0.353** | 0.205 |
| GMM n=5 | 0.198 | 0.179 | 0.218 |

k=2 fails for David (both clusters are majority one-handed).  At k=3, real signal
emerges (ARI = 0.353).  GMM n=5 also finds two pure both-handed clusters.
David's weaker result suggests more gradual kinematic differences between left and
right single-hand trials, not an absence of signal.

### 14.3 Pass B_ph — Phase-stratified results (AH1)

Phase-stratified Pass B per subject. Best OHC-ARI across algorithms:

**Viktor:**

| Phase | Best OHC-ARI | Best CH-ARI | Best algorithm |
|---|---|---|---|
| **Laufen** | **0.787** | **0.987** | GMM n=3 / Ward k=3 |
| Aufheben | 0.334 | 0.248 | GMM n=4 |
| Absetzen | 0.217 | 0.303 | GMM n=4 |

Viktor Laufen GMM n=3 cluster purities: 100% both-handed, 100% right-handed, 97% left-handed.
Three perfectly separated point clouds visible in the 2D PCA scatter plot.

**David:**

| Phase | Best OHC-ARI | Best CH-ARI |
|---|---|---|
| **Laufen** | **0.342** | **0.506** |
| Aufheben | 0.169 | 0.017 |
| Absetzen | 0.175 | 0.185 |

**AH1 conclusion:** Laufen produces the strongest cluster separation by a large margin for both subjects. Confirmed.

---

### 14.4 Pass B_box — Box-size-stratified results (AH2)

**Viktor:**

| Box size | Best OHC-ARI | Best CH-ARI |
|---|---|---|
| **big** | **0.373** | **0.421** |
| small | 0.318 | 0.388 |

**David:**

| Box size | Best OHC-ARI | Best CH-ARI |
|---|---|---|
| **big** | **0.489** | **0.795** |
| small | 0.263 | 0.270 |

David big box DBSCAN: OHC-ARI = 1.000 (perfect separation of one-handed vs both-handed on 82.6% of non-noise windows; 17.4% flagged as noise).

**AH2 conclusion:** Big box produces clearly stronger cluster separation for David. Viktor shows the same direction but a smaller gap. Weakly-to-moderately supported.

---

### 14.5 Hypothesis assessment (all passes)

| Hypothesis | Result | Best evidence |
|---|---|---|
| H1 — one-sided vs two-handed separable | **Supported** | Viktor K-Means k=2 OHC-ARI = 0.40 (purity: 92% / 75%); David k=3 ARI = 0.35 |
| H1 — carrying hand (L/R/both) separable | **Strongly supported** | Viktor GMM n=5 CH-ARI = 0.52; three clusters at 100% purity |
| AH1 — Laufen phase strongest | **Confirmed** | Viktor Laufen GMM n=3 OHC-ARI = 0.787, CH-ARI = 0.987 vs Aufheben 0.334 / Absetzen 0.217 |
| AH2 — big box easier | **Weakly-moderately supported** | David big ARI = 0.49 vs small 0.26; Viktor big 0.37 vs small 0.32 |
| PCA-based approach viable | **Not supported** | Pass A silhouette 0.81–0.83 but ARI ≈ 0 across all algorithms |

---

## 15. Lessons Learned

| Problem | Diagnosis | Fix |
|---|---|---|
| False high silhouette (paired k=2 = 0.78) | 2–3 impact windows form an artificial cluster | Kurtosis filter on raw L/R channels only |
| Filter removed 55% of windows | Ajerk/Gjerk kurtosis naturally high → over-triggered | Exclude jerk kurtosis from filter |
| PCA + full features → ARI ≈ 0 | PCA maximises variance (motion intensity), not class separability | Remove PCA from clustering; use classifier-selected features |
| `symm_idx` / `log_ratio` instability | Near-zero denominators, loss of sign | Replaced by linear `ratio = f_R / (f_L + ε)` |
| `abs_diff_jerk_std` mismatched classifier feature | Old jerk was accel-only; Gjerk channel did not exist | Added Gjerk channel; renamed to `abs_diff_Gjerk_mean` |

---

## 16. Known Limitations

| Limitation | Impact |
|---|---|
| n=2 subjects | Results cannot be generalised; Viktor vs David difference is uninterpretable without more subjects |
| 50% window overlap | Temporal autocorrelation inflates effective sample size |
| Pass A scatter plots | 2D PCA of 370 features is dominated by ratio-feature outliers; visualization is not representative |
| DBSCAN in 10D | Both subjects collapse to 1 cluster — no density gap between conditions; purely distributional differences |
| Phase-boundary windows | Majority-vote phase label introduces label noise at ~50% of phase-boundary windows |

---

## 17. References

1. Robinson, R. O., Herzog, W., & Nigg, B. M. (1987). Use of force platform variables to quantify the effects of chiropractic manipulation on gait symmetry. *Journal of Manipulative and Physiological Therapeutics*, 10(4), 172–176.
2. Moe-Nilssen, R. (1998). A new method for evaluating motor control in gait under real-life environmental conditions. *Clinical Biomechanics*, 13(4–5), 320–327.
3. Kavanagh, J. J., & Menz, H. B. (2008). Accelerometry: A technique for quantifying movement patterns during walking. *Gait & Posture*, 28(1), 1–15.
4. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
5. Davies, D. L., & Bouldin, D. W. (1979). A cluster separation measure. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, PAMI-1(2), 224–227.
6. Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD*, 96(34), 226–231.
