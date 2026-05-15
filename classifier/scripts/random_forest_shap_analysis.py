from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from random_forest_feature_selection import get_transformed_feature_names, train_random_forest
from train_classifier import (
    load_features,
    make_train_test_split,
    make_xy,
    validate_dataset,
)


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = CLASSIFIER_ROOT / "results"
SHAP_IMPORTANCE_CSV = RESULTS_DIR / "feature_importance_shap_random_forest.csv"
SHAP_SUMMARY_PNG = RESULTS_DIR / "shap_summary_random_forest.png"

MAX_DISPLAY = 20


def get_positive_class_shap_values(shap_values):
    """Return SHAP values for class True / one-handed carrying."""
    if isinstance(shap_values, list):
        return shap_values[1]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        return shap_values[:, :, 1]

    return shap_values


def to_dense_array(values):
    if hasattr(values, "toarray"):
        return values.toarray()
    return np.asarray(values)


def main() -> None:
    df = load_features()
    validate_dataset(df)
    train_df, test_df = make_train_test_split(df)
    X_train, X_test, y_train, y_test, numeric, categorical = make_xy(train_df, test_df)

    model = train_random_forest(X_train, y_train, numeric, categorical)
    transformed_names = get_transformed_feature_names(model, numeric, categorical)
    transformed_test = to_dense_array(model.named_steps["preprocess"].transform(X_test))

    rf = model.named_steps["model"]
    explainer = shap.TreeExplainer(rf)
    shap_values = get_positive_class_shap_values(explainer.shap_values(transformed_test))

    if shap_values.shape[1] != len(transformed_names):
        raise ValueError(
            f"SHAP feature mismatch: {shap_values.shape[1]} values, "
            f"{len(transformed_names)} feature names"
        )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame(
            {
                "feature": transformed_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(SHAP_IMPORTANCE_CSV, index=False)

    transformed_test_df = pd.DataFrame(transformed_test, columns=transformed_names)
    shap.summary_plot(
        shap_values,
        transformed_test_df,
        max_display=MAX_DISPLAY,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PNG, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved SHAP feature importance to {SHAP_IMPORTANCE_CSV}")
    print(f"Saved SHAP summary plot to {SHAP_SUMMARY_PNG}")
    print("\nTop SHAP features:")
    print(importance.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
