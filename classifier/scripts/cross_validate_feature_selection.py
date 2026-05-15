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
RF_SELECTION_CV_RESULTS_CSV = RESULTS_DIR / "random_forest_feature_selection_cross_validation_results.csv"

TOP_K_VALUES = [10, 20, 40]


def evaluate_feature_set(fold: int, feature_set: str, selected_features: list[str], train_df, test_df) -> dict:
    categorical = [col for col in CATEGORICAL_FEATURES if col in selected_features]
    numeric = [col for col in selected_features if col not in categorical]

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]
    y_train = train_df[LABEL_COL].astype(bool)
    y_test = test_df[LABEL_COL].astype(bool)

    model = train_random_forest(X_train, y_train, numeric, categorical)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=[False, True])

    return {
        "fold": fold,
        "feature_set": feature_set,
        "n_features": len(selected_features),
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "f1_both_hands": report["False"]["f1-score"],
        "f1_one_hand": report["True"]["f1-score"],
        "tn": matrix[0, 0],
        "fp": matrix[0, 1],
        "fn": matrix[1, 0],
        "tp": matrix[1, 1],
        "selected_features": ";".join(selected_features),
    }


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "macro_f1", "f1_both_hands", "f1_one_hand"]
    summary = results.groupby("feature_set")[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    return summary.sort_values("macro_f1_mean", ascending=False)


def main() -> None:
    df = load_features()
    validate_dataset(df)

    rows = []
    for fold, (train_idx, test_idx) in enumerate(make_stratified_group_folds(df), start=1):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        overlap = set(train_df[GROUP_COL]) & set(test_df[GROUP_COL])
        if overlap:
            raise ValueError(f"Data leakage in fold {fold}: {sorted(overlap)}")

        print(f"\nFold {fold}")
        X_train, X_test, y_train, y_test, numeric, categorical = make_xy(train_df, test_df)
        all_features = X_train.columns.tolist()

        # Importance is computed on the training part of this fold only.
        rf = train_random_forest(X_train, y_train, numeric, categorical)
        importance = save_random_forest_importance(rf, numeric, categorical)
        ranked_features = importance["original_feature"].tolist()

        rows.append(evaluate_feature_set(fold, "all_features", all_features, train_df, test_df))

        for top_k in TOP_K_VALUES:
            selected = ranked_features[:top_k]
            rows.append(evaluate_feature_set(fold, f"top_{top_k}_rf_importance", selected, train_df, test_df))

        for row in rows[-(len(TOP_K_VALUES) + 1):]:
            print(f"  {row['feature_set']}: accuracy={row['accuracy']:.3f}, macro_f1={row['macro_f1']:.3f}")

    fold_results = pd.DataFrame(rows)
    summary = aggregate_results(fold_results)
    output = fold_results.merge(summary, on="feature_set", suffixes=("", "_summary"))
    output.to_csv(RF_SELECTION_CV_RESULTS_CSV, index=False)

    print(f"\nSaved Random Forest feature-selection CV results to {RF_SELECTION_CV_RESULTS_CSV}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
