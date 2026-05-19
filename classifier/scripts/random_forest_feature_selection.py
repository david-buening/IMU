from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from cross_validate_models import make_stratified_group_folds
from train_classifier import (
    CATEGORICAL_FEATURES,
    FEATURE_CSV,
    GROUP_COL,
    LABEL_COL,
    RANDOM_STATE,
    SUBJECT_COL,
    build_models,
    load_features,
    make_preprocessor,
    make_train_test_split,
    make_xy,
    validate_dataset,
)


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = CLASSIFIER_ROOT / "results"
IMPORTANCE_CSV = RESULTS_DIR / "feature_importance_random_forest.csv"
SELECTION_RESULTS_CSV = RESULTS_DIR / "model_feature_selection_results.csv"
SHAP_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance_shap_random_forest.csv"

TOP_K_VALUES = [10, 20, 40]


def get_transformed_feature_names(model: Pipeline, numeric: list[str], categorical: list[str]) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    feature_names = []

    feature_names.extend(numeric)

    if categorical:
        onehot = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        if hasattr(onehot, "get_feature_names_out"):
            onehot_names = onehot.get_feature_names_out(categorical).tolist()
        else:
            onehot_names = onehot.get_feature_names(categorical).tolist()
        feature_names.extend(onehot_names)

    return feature_names


def transformed_to_original_feature(transformed_name: str, numeric: list[str], categorical: list[str]) -> str:
    if transformed_name in numeric:
        return transformed_name

    for cat_col in categorical:
        prefix = f"{cat_col}_"
        if transformed_name.startswith(prefix):
            return cat_col

    return transformed_name


def _importance_df(model: Pipeline, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Compute feature importance from a trained RF pipeline without saving."""
    transformed_names = get_transformed_feature_names(model, numeric, categorical)
    importances = model.named_steps["model"].feature_importances_

    transformed_importance = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "original_feature": [
                transformed_to_original_feature(name, numeric, categorical) for name in transformed_names
            ],
            "importance": importances,
        }
    )
    return (
        transformed_importance.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )


def compute_cv_averaged_importance(
    df: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> pd.DataFrame:
    """Average RF feature importances across CV folds for a stable ranking."""
    all_importances = []
    for train_idx, _ in make_stratified_group_folds(df):
        train_fold = df.iloc[train_idx]
        X_fold = train_fold[numeric + categorical]
        y_fold = train_fold[LABEL_COL].astype(bool)
        rf = train_random_forest(X_fold, y_fold, numeric, categorical)
        imp = _importance_df(rf, numeric, categorical)
        all_importances.append(imp.set_index("original_feature")["importance"])

    averaged = pd.concat(all_importances, axis=1).mean(axis=1).reset_index()
    averaged.columns = ["original_feature", "importance"]
    return averaged.sort_values("importance", ascending=False)


def train_random_forest(X_train, y_train, numeric: list[str], categorical: list[str]) -> Pipeline:
    model = Pipeline(
        [
            ("preprocess", make_preprocessor(numeric, categorical, scale_numeric=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def save_random_forest_importance(model: Pipeline, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    transformed_names = get_transformed_feature_names(model, numeric, categorical)
    importances = model.named_steps["model"].feature_importances_

    transformed_importance = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "original_feature": [
                transformed_to_original_feature(name, numeric, categorical) for name in transformed_names
            ],
            "importance": importances,
        }
    )

    # One-hot encoded categorical values are summed back to the original column.
    original_importance = (
        transformed_importance.groupby("original_feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    original_importance.to_csv(IMPORTANCE_CSV, index=False)

    print(f"\nSaved Random Forest feature importance to {IMPORTANCE_CSV}")
    print("Top 20 features:")
    print(original_importance.head(20).to_string(index=False))
    return original_importance


def try_save_shap_importance(model: Pipeline, X_test: pd.DataFrame, numeric: list[str], categorical: list[str]) -> None:
    try:
        import shap
    except ModuleNotFoundError:
        print("\nSHAP is not installed, skipping SHAP importance.")
        print("Install later with: python -m pip install shap")
        return

    transformed_names = get_transformed_feature_names(model, numeric, categorical)
    transformed_test = model.named_steps["preprocess"].transform(X_test)
    rf = model.named_steps["model"]
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(transformed_test)

    # For binary classifiers, SHAP returns either a list of two arrays (older
    # versions) or a single 3D array of shape (n_samples, n_features, n_classes).
    if isinstance(shap_values, list):
        class_values = shap_values[1]
    elif shap_values.ndim == 3:
        class_values = shap_values[:, :, 1]
    else:
        class_values = shap_values

    mean_abs_shap = np.abs(class_values).mean(axis=0)
    shap_df = pd.DataFrame(
        {
            "transformed_feature": transformed_names,
            "original_feature": [
                transformed_to_original_feature(name, numeric, categorical) for name in transformed_names
            ],
            "mean_abs_shap": mean_abs_shap,
        }
    )
    shap_df = (
        shap_df.groupby("original_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    shap_df.to_csv(SHAP_IMPORTANCE_CSV, index=False)
    print(f"\nSaved SHAP importance to {SHAP_IMPORTANCE_CSV}")


def evaluate_random_forest_feature_set(
    name: str,
    selected_features: list[str],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    categorical = [col for col in CATEGORICAL_FEATURES if col in selected_features]
    numeric = [col for col in selected_features if col not in categorical]

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]
    y_train = train_df[LABEL_COL].astype(bool)
    y_test = test_df[LABEL_COL].astype(bool)

    model = train_random_forest(X_train, y_train, numeric, categorical)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=[False, True])

    return {
        "feature_set": name,
        "n_features": len(selected_features),
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "f1_both_hands": report["False"]["f1-score"],
        "f1_one_hand": report["True"]["f1-score"],
        "tn": matrix[0, 0],
        "fp": matrix[0, 1],
        "fn": matrix[1, 0],
        "tp": matrix[1, 1],
        "selected_features": ";".join(selected_features),
    }


def main() -> None:
    df = load_features()
    validate_dataset(df)
    train_df, test_df = make_train_test_split(df)
    X_train, X_test, y_train, y_test, numeric, categorical = make_xy(train_df, test_df)

    # Average feature importances across CV folds for a stable ranking.
    importance = compute_cv_averaged_importance(df, numeric, categorical)
    importance.to_csv(IMPORTANCE_CSV, index=False)
    print(f"\nSaved averaged CV feature importance to {IMPORTANCE_CSV}")
    print("Top 20 features:")
    print(importance.head(20).to_string(index=False))

    # Train one RF on the full training split for SHAP analysis.
    full_rf = train_random_forest(X_train, y_train, numeric, categorical)
    try_save_shap_importance(full_rf, X_test, numeric, categorical)

    all_features = X_train.columns.tolist()
    ranked_features = importance["original_feature"].tolist()

    rows = [
        evaluate_random_forest_feature_set("all_features", all_features, train_df, test_df),
    ]

    for top_k in TOP_K_VALUES:
        selected = ranked_features[:top_k]
        rows.append(evaluate_random_forest_feature_set(f"top_{top_k}_rf_importance", selected, train_df, test_df))

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    results.to_csv(SELECTION_RESULTS_CSV, index=False)

    print(f"\nSaved feature selection comparison to {SELECTION_RESULTS_CSV}")
    print(results[["feature_set", "n_features", "accuracy", "macro_f1", "f1_both_hands", "f1_one_hand"]].to_string(index=False))


if __name__ == "__main__":
    main()
