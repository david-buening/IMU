from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from cross_validate_models import make_stratified_group_folds
from random_forest_feature_selection import save_random_forest_importance, train_random_forest
from train_classifier import (
    CATEGORICAL_FEATURES,
    GROUP_COL,
    LABEL_COL,
    load_features,
    make_xy,
    validate_dataset,
)


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = CLASSIFIER_ROOT / "results"
PHASE_CV_RESULTS_CSV = RESULTS_DIR / "random_forest_top10_by_phase_cross_validation_results.csv"

PHASES = ["Aufheben", "Laufen", "Absetzen"]
TOP_K = 40


def evaluate_fold(phase: str, fold: int, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    X_train, X_test, y_train, y_test, numeric, categorical = make_xy(train_df, test_df)

    # Compute feature importance on this fold's training data only.
    rf_all = train_random_forest(X_train, y_train, numeric, categorical)
    importance = save_random_forest_importance(rf_all, numeric, categorical)
    selected_features = importance["original_feature"].head(TOP_K).tolist()

    selected_categorical = [col for col in CATEGORICAL_FEATURES if col in selected_features]
    selected_numeric = [col for col in selected_features if col not in selected_categorical]

    X_train_top = train_df[selected_features]
    X_test_top = test_df[selected_features]
    rf_top = train_random_forest(X_train_top, y_train, selected_numeric, selected_categorical)

    y_pred = rf_top.predict(X_test_top)
    y_prob = rf_top.predict_proba(X_test_top)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=[False, True])

    return {
        "phase": phase,
        "fold": fold,
        "n_train_windows": len(train_df),
        "n_test_windows": len(test_df),
        "n_train_experiments": train_df[GROUP_COL].nunique(),
        "n_test_experiments": test_df[GROUP_COL].nunique(),
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "f1_both_hands": report["False"]["f1-score"],
        "f1_one_hand": report["True"]["f1-score"],
        "mean_predicted_probability_one_hand": float(y_prob.mean()),
        "tn": matrix[0, 0],
        "fp": matrix[0, 1],
        "fn": matrix[1, 0],
        "tp": matrix[1, 1],
        "selected_features": ";".join(selected_features),
    }


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "macro_f1",
        "f1_both_hands",
        "f1_one_hand",
        "mean_predicted_probability_one_hand",
    ]
    summary = results.groupby("phase")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary.sort_values("macro_f1_mean", ascending=False)


def main() -> None:
    df = load_features()
    validate_dataset(df)

    all_rows = []
    for phase in PHASES:
        phase_df = df[df["phase"] == phase].copy()
        print(f"\n=== Phase: {phase} ===")
        print(f"Windows: {len(phase_df)}, experiments: {phase_df[GROUP_COL].nunique()}")
        print("Experiment labels:")
        print(phase_df.groupby(GROUP_COL)[LABEL_COL].first().value_counts().to_string())

        for fold, (train_idx, test_idx) in enumerate(make_stratified_group_folds(phase_df), start=1):
            train_df = phase_df.loc[train_idx].copy()
            test_df = phase_df.loc[test_idx].copy()

            overlap = set(train_df[GROUP_COL]) & set(test_df[GROUP_COL])
            if overlap:
                raise ValueError(f"Data leakage in phase {phase}, fold {fold}: {sorted(overlap)}")

            row = evaluate_fold(phase, fold, train_df, test_df)
            all_rows.append(row)
            print(
                f"  Fold {fold}: accuracy={row['accuracy']:.3f}, "
                f"macro_f1={row['macro_f1']:.3f}, "
                f"test_experiments={row['n_test_experiments']}"
            )

    fold_results = pd.DataFrame(all_rows)
    summary = aggregate_results(fold_results)
    output = fold_results.merge(summary, on="phase", suffixes=("", "_summary"))
    output.to_csv(PHASE_CV_RESULTS_CSV, index=False)

    print(f"\nSaved phase cross-validation results to {PHASE_CV_RESULTS_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
