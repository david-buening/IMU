# IMU Classifier — Presentation Notes

Slide outline for the classifier part of the presentation.
Assumes experiment setup, motivation, and study design are covered in earlier slides.

---

## Slide: Classification Goal

**One sentence:** Can two wrist-worn IMU sensors reliably tell apart one-handed and two-handed object carrying?

- Binary label: `one_handed_carry = True / False`
- Input: raw accelerometer + gyroscope signals from left and right wrist
- No additional metadata — sensor signals only

---

## Slide: Our Pipeline (4 Steps)

```
1. Preprocessing
   Raw CSV files → merged, phase-labelled dataset

2. Feature Extraction
   Time windows (1 s, 50 % overlap) → 116 numeric features per window

3. Model Selection
   4 classifiers compared on grouped 80/20 split

4. Evaluation
   5-fold grouped cross-validation → performance per model, phase, box size
```

Key design decision: windows from the same experiment are always kept together
(no leakage between train and test).

---

## Slide: Features

Two sensor streams (left wrist + right wrist), each producing:

- Per-axis mean and standard deviation (AX, AY, AZ, GX, GY, GZ)
- Magnitude signals: `acc_mag`, `gyro_mag`, `acc_jerk_mag`, `gyro_jerk_mag`
  - mean, std, min, max, range, energy

Cross-wrist asymmetry features:
- Absolute differences L − R
- Ratios L / R

**Total: 116 sensor-derived features per window.**

---

## Slide: Model Comparison

Grouped 5-fold cross-validation, all 4 classifiers:

| Model | Macro F1 (mean) | Macro F1 (std) |
|---|---:|---:|
| **Random Forest** | **0.868** | **0.031** |
| SVM | 0.849 | 0.042 |
| Logistic Regression | 0.838 | 0.046 |
| KNN | 0.824 | 0.029 |

→ **Random Forest selected** for all further analyses.

---

## Slide: Hypothesis 1 — IMU Signals Are Sufficient

> One-handed vs. two-handed carrying can be classified from wrist IMU data alone.

Random Forest cross-validation result:

```
Macro F1 = 0.868  (±0.031)
Accuracy = 0.869  (±0.032)
```

**Confirmed.** The model generalises across subjects, box sizes, and surfaces
using only raw sensor-derived features.

---

## Slide: Hypothesis 2 — Walking Phase Has Strongest Signal

> The walking phase (Laufen) produces clearer asymmetry than pick-up or put-down.

Phase-specific Random Forest (Top 10 features, 5-fold CV):

| Phase | Macro F1 (mean) |
|---|---:|
| **Laufen** | **0.943** |
| Aufheben | 0.869 |
| Absetzen | 0.862 |

**Confirmed.** During walking, the free arm swings naturally while the loaded arm
stays stabilised — creating a stronger and more consistent asymmetry signal.

---

## Slide: Hypothesis 3 — Big Box Is Easier to Classify

> Object size influences classification difficulty.

Box-size-specific Random Forest (Top 10 features, 5-fold CV):

| Box size | Macro F1 (mean) |
|---|---:|
| **big** | **0.965** |
| small | 0.763 |

**Confirmed.** A larger object forces more pronounced posture differences between
one-handed and two-handed carrying, making the asymmetry more detectable.

---

## Slide: Most Important Features (SHAP)

Top features ranked by mean absolute SHAP value:

```
L_AZ_mean                        ← vertical acceleration, left wrist
L_AX_mean                        ← forward/back acceleration, left wrist
AY_std_absdiff                   ← lateral acceleration variability asymmetry
GY_std_absdiff                   ← gyro Y variability asymmetry
R_AZ_mean                        ← vertical acceleration, right wrist
gyro_mag_mean_absdiff            ← rotation magnitude asymmetry
L_AY_mean                        ← lateral acceleration, left wrist
gyro_jerk_mag_mean_absdiff       ← rotation jerk asymmetry
R_AX_mean                        ← forward/back acceleration, right wrist
R_AY_mean                        ← lateral acceleration, right wrist
```

Pattern: the model relies on **absolute wrist orientation/movement** and
**left-right asymmetry** — exactly what we hypothesised.

---

## Slide: Takeaways

1. **IMU-based classification works** — Macro F1 ~0.87 from sensor signals alone
2. **Walking phase is most informative** — F1 0.94 vs. ~0.86 for other phases
3. **Box size matters** — big box F1 0.97 vs. small box F1 0.76
4. **Asymmetry features are key** — SHAP confirms left-right differences drive predictions
5. **Next step** would be a real-time deployment model trained on all data
