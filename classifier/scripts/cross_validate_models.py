from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from train_classifier import (
    FEATURE_CSV,
    GROUP_COL,
    LABEL_COL,
    RANDOM_STATE,
    build_models,
    load_features,
    make_xy,
    validate_dataset,
)


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = CLASSIFIER_ROOT / "results"
BASELINE_CV_RESULTS_CSV = RESULTS_DIR / "model_baseline_cross_validation_results.csv"

N_SPLITS = 5


def make_stratified_group_folds(df: pd.DataFrame) -> list[tuple[list[int], list[int]]]:
    experiment_labels = df.groupby(GROUP_COL)[LABEL_COL].first()
    rng = np.random.default_rng(RANDOM_STATE)

    fold_groups = [[] for _ in range(N_SPLITS)]
    for _, label_series in experiment_labels.groupby(experiment_labels):
        group_ids = label_series.index.to_numpy()
        rng.shuffle(group_ids)
        for idx, group_id in enumerate(group_ids):
            fold_groups[idx % N_SPLITS].append(group_id)

    folds = []
    all_groups = set(experiment_labels.index)
    for test_groups in fold_groups:
        test_groups = set(test_groups)
        train_groups = all_groups - test_groups
        train_idx = df.index[df[GROUP_COL].isin(train_groups)].tolist()
        test_idx = df.index[df[GROUP_COL].isin(test_groups)].tolist()
        folds.append((train_idx, test_idx))
    return folds


def summarize_fold(model_name: str, fold: int, y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "model": model_name,
        "fold": fold,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "f1_both_hands": report["False"]["f1-score"],
        "f1_one_hand": report["True"]["f1-score"],
    }


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "f1_both_hands", "f1_one_hand"]
    summary = (
        results.groupby("model")[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary.sort_values("macro_f1_mean", ascending=False)


def main() -> None:
    df = load_features()
    validate_dataset(df)

    experiment_labels = df.groupby(GROUP_COL)[LABEL_COL].first()
    if experiment_labels.value_counts().min() < N_SPLITS:
        raise ValueError("N_SPLITS is larger than the number of experiments in one class")

    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(make_stratified_group_folds(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        overlap = set(train_df[GROUP_COL]) & set(test_df[GROUP_COL])
        if overlap:
            raise ValueError(f"Data leakage in fold {fold}: {sorted(overlap)}")

        print(f"\nFold {fold}")
        print(f"Train experiments: {train_df[GROUP_COL].nunique()}, test experiments: {test_df[GROUP_COL].nunique()}")
        print("Test experiment labels:")
        print(test_df.groupby(GROUP_COL)[LABEL_COL].first().value_counts().to_string())

        X_train, X_test, y_train, y_test, numeric, categorical = make_xy(train_df, test_df)
        models = build_models(numeric, categorical)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            row = summarize_fold(model_name, fold, y_test, y_pred)
            fold_rows.append(row)
            print(f"  {model_name}: accuracy={row['accuracy']:.3f}, macro_f1={row['macro_f1']:.3f}")

    fold_results = pd.DataFrame(fold_rows)
    summary = aggregate_results(fold_results)

    output = fold_results.merge(summary, on="model", suffixes=("", "_summary"))
    output.to_csv(BASELINE_CV_RESULTS_CSV, index=False)

    print(f"\nSaved baseline cross-validation results to {BASELINE_CV_RESULTS_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
