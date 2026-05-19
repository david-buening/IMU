# IMU Carrying Classification & Clustering

TU Wien ASM2 — Viktor Hoffmann, May 2026

Can two wrist-worn IMU sensors detect whether a box is carried with one hand or two?
This project answers that question via **supervised classification** and **unsupervised clustering**.

---

## Repository Structure

```
data/
  Big Box David/        raw sensor CSVs (original, tracked in git)
  Big Box Viktor/
  Smal Box David/
  Small Box Viktor/
  preprocessed.csv      generated — do not commit
  features/             generated — do not commit

classifier/
  scripts/              classifier pipeline scripts
  results/              generated CSVs + plots

src/
  preprocessing/        preprocessing.py
  features/             windows.py, align.py, time_domain.py, asymmetry.py
  clustering/           pipeline.py, run_clustering.py, cluster.py, reduce.py, evaluate.py

knowledge/
  classifier.md         detailed classifier documentation
  README_report.md      written report (classifier)
  README_presentation.md  presentation slide notes (classifier)
  clustering.md         detailed clustering documentation
  preprocessing_logic.md  preprocessing design notes

management_summary.tex  management summary (LaTeX)
requirements.txt
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below assume the **project root** as working directory and `.venv` active.

---

## Full Pipeline (from raw CSVs to all results)

### Step 1 — Preprocessing

Merges all 64 raw sensor CSVs into a single labelled dataset.

```bash
python src/preprocessing/preprocessing.py
```

Output: `data/preprocessed.csv` (47,413 rows × 15 columns)

---

### Step 2 — Classifier: Feature Extraction

Builds 1-second sliding windows with 50% overlap and computes 116 features per window.

```bash
python classifier/scripts/build_window_features.py
```

Output: `classifier/results/features_windows_1s_all_experiments.csv` (478 windows × 128 columns)

---

### Step 3 — Clustering: Feature Extraction

Builds paired L+R feature matrices (91 features per wrist + 188 asymmetry features = 370 total).

```bash
python src/clustering/pipeline.py
```

Outputs:
- `data/features/feature_matrix_single.csv` — single-sensor windows (913 × 103)
- `data/features/feature_matrix_paired.csv` — paired windows (601 × 381)

---

### Step 4 — Classifier: Model Training & Evaluation

Run scripts in order:

```bash
# 1. Baseline model comparison (4 classifiers, grouped 5-fold CV)
python classifier/scripts/cross_validate_models.py

# 2. Feature importance ranking (Random Forest, single 80/20 split)
python classifier/scripts/random_forest_feature_selection.py

# 3. Feature selection cross-validation (top 10/20/40 features, 5-fold CV)
python classifier/scripts/cross_validate_feature_selection.py

# 4. Phase-stratified cross-validation (AH1: Laufen vs Aufheben vs Absetzen)
python classifier/scripts/cross_validate_by_phase.py

# 5. Box-size-stratified cross-validation (AH2: big vs small)
python classifier/scripts/cross_validate_by_box_size.py

# 6. SHAP feature importance analysis
python classifier/scripts/random_forest_shap_analysis.py

# 7. Generate result plots
python classifier/scripts/create_result_plots.py
```

Key outputs in `classifier/results/`:
- `model_baseline_cross_validation_results.csv`
- `random_forest_feature_selection_cross_validation_results.csv`
- `random_forest_top10_by_phase_cross_validation_results.csv`
- `random_forest_top10_by_box_size_cross_validation_results.csv`
- `feature_importance_shap_random_forest.csv`
- `plots/*.png`

---

### Step 5 — Clustering Analysis

Runs all clustering passes and generates all plots and metrics.

```bash
python src/clustering/run_clustering.py
```

This runs in sequence for each subject (David, Viktor):
- **Pass A** — full 370-feature space (K-Means, Ward, GMM; no DBSCAN)
- **Pass B** — classifier top-10 features (K-Means, Ward, GMM, DBSCAN)
- **Pass B_ph_*** — Pass B stratified per phase (AH1: Aufheben / Laufen / Absetzen)
- **Pass B_box_*** — Pass B stratified per box size (AH2: big / small)

Outputs:
- `data/features/feature_matrix_paired_filtered.csv` — after kurtosis outlier removal
- `data/features/metrics_summary.csv` — all clustering metrics (ARI, silhouette, etc.)
- `data/features/plots/{subject}/{pass}/*.png` — scatter plots, dendrograms, heatmaps

---

## Key Results

### Classifier (Random Forest, Top 40 features, grouped 5-fold CV)

| Hypothesis | Result | Metric |
|---|---|---|
| H1 — IMU signals sufficient | Confirmed | Macro F1 = 0.879 ± 0.029 |
| AH1 — Walking phase strongest | Confirmed | Laufen F1 = 0.940 vs Aufheben 0.887 / Absetzen 0.882 |
| AH2 — Big box easier | Confirmed | Big F1 = 0.967 vs Small F1 = 0.763 |

### Clustering (Pass B, top-10 clf features, no labels)

| Subject | Best result | ARI |
|---|---|---|
| Viktor | GMM n=5, CH-ARI | 0.519 (3 clusters at 100% purity) |
| Viktor | K-Means k=2, OHC-ARI | 0.404 |
| Viktor Laufen | GMM n=3, CH-ARI | 0.987 (all 3 carrying conditions perfectly separated) |
| David | K-Means k=3, OHC-ARI | 0.353 |
| David big box | GMM n=4, CH-ARI | 0.795 |

---

## Documentation

| Document | Contents |
|---|---|
| `knowledge/clustering.md` | Full clustering pipeline, feature engineering, design decisions, results |
| `knowledge/README_report.md` | Written classifier report |
| `knowledge/README_presentation.md` | Presentation slide notes (classifier) |
| `knowledge/preprocessing_logic.md` | Preprocessing design notes |
| `management_summary.tex` | Management summary (LaTeX, both classifier + clustering) |
