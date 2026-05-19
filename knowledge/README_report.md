# IMU-Based Carrying Posture Classification — Report

## 1. Problem Statement

This section reports the supervised classification component of the project.
The goal is to determine whether one-handed and two-handed object carrying can
be distinguished automatically from the raw signals of two wrist-worn IMU sensors.

The classification target is binary:

- `one_handed_carry = True`: the object was carried with one hand
- `one_handed_carry = False`: the object was carried with both hands

Left-handed and right-handed single-arm carrying are intentionally merged into
the single positive class.

Three hypotheses guide the analysis:

1. IMU signals from both wrists are sufficient to classify carrying posture.
2. The walking phase (Laufen) produces a stronger discriminative signal than
   the pick-up (Aufheben) or put-down (Absetzen) phase.
3. A larger object (big box) is easier to classify than a smaller one (small box).

---

## 2. Data

The dataset consists of 30 reconstructed experiments across two subjects (David,
Viktor). Each experiment captures one continuous carry trial under one condition:

| Variable | Values |
|---|---|
| `subject` | David, Viktor |
| `box_size` | big, small |
| `surface` | table, floor |
| `one_handed_carry` | True, False |

Each experiment was recorded with two synchronised IMU sensors, one on each wrist.
The six raw channels per sensor are: `AX`, `AY`, `AZ` (accelerometer) and
`GX`, `GY`, `GZ` (gyroscope).

Each trial is divided into three sequential phases:

- **Aufheben**: picking up the object from a surface
- **Laufen**: carrying while walking (approximately 10 m)
- **Absetzen**: placing the object back down

Class distribution across experiments: 15 one-handed, 15 two-handed.

---

## 3. Feature Extraction

Feature extraction is implemented in `scripts/build_window_features.py`.

### 3.1 Windowing

Sensor data is segmented into sliding windows of 1 second duration with 50 %
overlap. Windows are created separately per experiment and per phase to prevent
cross-phase or cross-experiment contamination. An additional final window is
shifted to align with the phase end to avoid under-representing the last motion
segment. Windows with fewer than 30 samples per hand are discarded.

This yields 478 windows across 30 experiments.

### 3.2 Derived Signals

Four magnitude and jerk signals are computed per hand before windowing:

```
acc_mag      = sqrt(AX² + AY² + AZ²)
gyro_mag     = sqrt(GX² + GY² + GZ²)
acc_jerk_mag = ||diff(acc)||   (frame-to-frame acceleration change)
gyro_jerk_mag= ||diff(gyro)||
```

### 3.3 Per-Window Features

For each window, the following statistics are computed for both the left and
right wrist independently:

- Per-axis (AX–GZ): mean, standard deviation
- Per magnitude/jerk signal: mean, std, min, max, range, energy

Additionally, cross-wrist asymmetry features are computed:

- Absolute difference: |L − R| for mean, std, range, energy of each magnitude signal
- Ratio: L / R for the same statistics
- Per-axis standard deviation differences and ratios

**Total: 116 sensor-derived features per window.**

The following columns present in the feature CSV are explicitly excluded from
model inputs, as they represent experimental metadata or bookkeeping rather than
motion signal:

| Excluded column | Reason |
|---|---|
| `phase`, `box_size`, `surface` | Experimental metadata, not derivable from sensor data |
| `subject` | Person identity — would encourage person-specific overfitting |
| `L_n_samples`, `R_n_samples` | Depend on Bluetooth timing and packet loss, not motion |
| `experiment_id`, `window_*`, `time_reference` | Row bookkeeping |

---

## 4. Classification Approach

### 4.1 Models

Four classifiers are evaluated:

- Logistic Regression (`class_weight="balanced"`, `max_iter=2000`)
- K-Nearest Neighbours (`k=5`)
- Support Vector Machine (RBF kernel, `class_weight="balanced"`)
- Random Forest (300 trees, `class_weight="balanced"`)

All numeric features are median-imputed. Logistic Regression, KNN, and SVM also
apply robust scaling (RobustScaler, median/IQR). Random Forest does not require scaling.

### 4.2 Evaluation Strategy

All evaluation splits are grouped by `experiment_id`. This is critical because
overlapping 1-second windows from the same experiment are highly correlated. A
window-level split would leak near-duplicate samples into both train and test,
inflating all performance estimates.

The primary evaluation is **5-fold stratified grouped cross-validation**:

- Each fold contains approximately equal proportions of one-handed and
  two-handed experiments.
- All windows from one experiment always fall into the same fold.
- Performance is reported as mean ± standard deviation across the five folds.

An initial **grouped 80/20 train-test split** (24 train experiments, 6 test
experiments) is used for faster model and feature-set comparison before
committing to full cross-validation.

### 4.3 Feature Selection

Random Forest feature importance is used to rank the 116 features. Performance
is then evaluated with the top 10, top 20, and top 40 features across 5-fold
cross-validation to identify the optimal feature subset.

Feature importance is averaged across all CV folds (computed on the training
portion only within each fold) to produce a stable ranking and avoid information
leakage.

Cross-validated results show that **Top 40 features yield the best mean Macro F1
(0.879) and the lowest variance (±0.029)** among all evaluated subsets. Top 10
and Top 20 achieve similar mean performance (0.875) but with higher variance.
Top 40 features are therefore used for all hypothesis analyses (H1–H3).

The 10 most important features by CV-averaged RF importance:

```
L_AZ_mean               (vertical acceleration, left wrist)
AY_std_absdiff          (lateral acceleration variability asymmetry)
L_AX_mean               (forward/back acceleration, left wrist)
GY_std_absdiff          (gyro Y variability asymmetry)
gyro_jerk_mag_mean_absdiff
L_AY_mean               (lateral acceleration, left wrist)
R_AX_mean               (forward/back acceleration, right wrist)
R_AZ_mean               (vertical acceleration, right wrist)
R_AY_mean               (lateral acceleration, right wrist)
gyro_mag_mean_absdiff
```

---

## 5. Results

### 5.1 Model Comparison

Grouped 5-fold cross-validation, all features (116 numeric):

| Model | Accuracy mean | Accuracy std | Macro F1 mean | Macro F1 std |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.869** | **0.032** | **0.868** | **0.031** |
| SVM | 0.850 | 0.042 | 0.849 | 0.042 |
| Logistic Regression | 0.840 | 0.046 | 0.838 | 0.046 |
| KNN | 0.825 | 0.031 | 0.824 | 0.029 |

Random Forest achieves the highest mean Macro F1 and the second-lowest standard
deviation, indicating both the best average performance and reliable
generalisation. All subsequent analyses use Random Forest.

### 5.2 Feature Selection

Random Forest with varying feature set sizes, grouped 5-fold CV:

| Feature set | Accuracy mean | Accuracy std | Macro F1 mean | Macro F1 std |
|---|---:|---:|---:|---:|
| Top 40 features | 0.880 | 0.028 | 0.879 | 0.029 |
| Top 20 features | 0.876 | 0.041 | 0.875 | 0.040 |
| Top 10 features | 0.876 | 0.046 | 0.875 | 0.046 |
| All 116 features | 0.869 | 0.032 | 0.868 | 0.031 |

Feature selection provides a modest but consistent improvement. Top 10 through
Top 40 perform similarly, suggesting the most discriminative information is
concentrated in a small subset of features. **Top 40 is selected as the final
configuration** for all hypothesis analyses, as it achieves the best mean Macro
F1 (0.879) with the lowest variance (±0.029).

On the single 80/20 split, Top 10 features yield the highest single-split score
(Macro F1 0.918), though single-split results are more sensitive to the
particular experiments selected for testing.

### 5.3 Hypothesis 1 — IMU Signals Are Sufficient

The cross-validation result with the selected Top 40 feature set directly answers
the first hypothesis:

```
Random Forest, Top 40 features
Macro F1 mean = 0.879  (±0.029)
Accuracy mean = 0.880  (±0.028)
```

**Hypothesis 1 confirmed.** Wrist IMU signals alone — without any metadata about
box size, surface, or phase — are sufficient to classify carrying posture with
reliable generalisation across subjects and conditions.

### 5.4 Hypothesis 2 — Phase-Specific Performance

Separate Random Forest models (Top 40 features, 5-fold CV) trained and evaluated
for each phase independently:

| Phase | Accuracy mean | Accuracy std | Macro F1 mean | Macro F1 std |
|---|---:|---:|---:|---:|
| **Laufen** | **0.940** | **0.078** | **0.940** | **0.078** |
| Aufheben | 0.890 | 0.077 | 0.887 | 0.083 |
| Absetzen | 0.883 | 0.065 | 0.882 | 0.065 |

**Hypothesis 2 confirmed.** The walking phase achieves Macro F1 0.940, above
pick-up (0.887) and put-down (0.882). This is consistent with the biomechanical
expectation: during walking, the unloaded arm swings freely while the loaded arm
is held more statically, producing a large and consistent left-right asymmetry
signal.

The higher variance for Aufheben (std 0.083 vs. 0.065 for Absetzen) suggests
that the pick-up motion is more variable across trials, which may reflect
differences in starting position and individual technique.

### 5.5 Hypothesis 3 — Box-Size Effect

Separate Random Forest models (Top 40 features, 5-fold CV) trained and evaluated
per box size:

| Box size | Accuracy mean | Accuracy std | Macro F1 mean | Macro F1 std |
|---|---:|---:|---:|---:|
| **big** | **0.970** | **0.029** | **0.967** | **0.033** |
| small | 0.769 | 0.071 | 0.763 | 0.075 |

**Hypothesis 3 confirmed.** Classification performance is substantially higher for
the big box (Macro F1 0.967) than for the small box (0.763). This supports the
interpretation that a larger, heavier object forces more pronounced postural and
movement differences between one-handed and two-handed carrying, making the
asymmetry signal more detectable. For the small box, one-handed and two-handed
carrying may produce more similar wrist trajectories.

### 5.6 Feature Importance and SHAP Analysis

Top features by CV-averaged Random Forest importance:

```
L_AZ_mean, AY_std_absdiff, L_AX_mean, GY_std_absdiff, gyro_jerk_mag_mean_absdiff,
L_AY_mean, R_AX_mean, R_AZ_mean, R_AY_mean, gyro_mag_mean_absdiff,
gyro_jerk_mag_mean_ratio_L_over_R, ...
```

Top features by mean absolute SHAP value:

```
L_AZ_mean, L_AX_mean, AY_std_absdiff, GY_std_absdiff, R_AZ_mean,
gyro_mag_mean_absdiff, L_AY_mean, gyro_jerk_mag_mean_absdiff,
R_AX_mean, R_AY_mean
```

Both rankings are highly consistent. Two feature types dominate:

1. **Absolute wrist kinematics** — mean acceleration and rotation values per axis
   capture the overall posture and load distribution of each wrist independently.
2. **Left-right asymmetry** — absdiff and ratio features between the two wrists
   capture the relative difference in movement patterns, which directly reflects
   whether one arm is more loaded than the other.

The prominence of asymmetry features is consistent with the project hypothesis
that one-handed carrying creates detectable left-right differences in wrist motion.

---

## 6. Discussion

### Limitations

- **Dataset size**: 30 experiments across 2 subjects limits statistical power.
  Results should be interpreted as indicative rather than definitive.
- **Subject generalisation**: with only 2 subjects, it is unclear whether the
  trained model generalises to new individuals with different body sizes, gait
  patterns, or dominant hand preferences.
- **Phase labelling**: phase boundaries are manually annotated, which introduces
  some variability in where the walking phase begins and ends.
- **Controlled conditions**: all trials were conducted in a lab setting with a
  predefined route. Real-world carrying may involve more varied trajectories and
  interruptions.

### Interpretation of Box-Size Result

The large performance gap between big and small box (Macro F1 0.967 vs. 0.763)
raises the question of whether a single model trained across both sizes would
retain discriminative power. A combined model would face mixed signal quality
and would need to generalise across conditions where the asymmetry signal is
strong (big box) and weak (small box).

---

## 7. Conclusion

All three project hypotheses are confirmed by the cross-validation results:

1. Wrist IMU features alone classify carrying posture at Macro F1 0.879 (Top 40 features).
2. The walking phase is the most informative (Macro F1 0.940).
3. Larger objects produce stronger classification signal (Macro F1 0.967 vs. 0.763).

The Random Forest classifier with feature selection is the recommended choice.
The most important features combine absolute wrist kinematics with left-right
asymmetry, which aligns with the biomechanical rationale underlying the project.

A practical deployment would train a final model on all available data using the
top-ranked features from cross-validation and evaluate it on new subjects to
assess person-independent generalisation.
