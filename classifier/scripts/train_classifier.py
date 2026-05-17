from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


IMU_ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = CLASSIFIER_ROOT / "results"
FEATURE_CSV = RESULTS_DIR / "features_windows_1s_all_experiments.csv"
RESULTS_CSV = RESULTS_DIR / "model_baseline_results.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42

LABEL_COL = "one_handed_carry"
GROUP_COL = "experiment_id"

# Columns that describe the row or leak the label, but should not be model inputs.
DROP_COLS = [
    LABEL_COL,
    GROUP_COL,
    "time_reference",
    "window_index_in_phase",
    "window_start_s",
    "window_end_s",
    # Experiment metadata — not available from sensor signals in the real world.
    "phase",
    "box_size",
    "surface",
    # Sample counts depend on Bluetooth timing and dropout, not on motion.
    "L_n_samples",
    "R_n_samples",
]

CATEGORICAL_FEATURES = []

# Avoid using subject as an input feature for the baseline, because it can make
# the model learn person-specific patterns instead of carrying behavior.
SUBJECT_COL = "subject"


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURE_CSV)
    print(f"Loaded {len(df)} windows from {FEATURE_CSV}")
    print(f"Experiments: {df[GROUP_COL].nunique()}")
    print("Label distribution by window:")
    print(df[LABEL_COL].value_counts().to_string())
    print("Label distribution by experiment:")
    print(df.groupby(GROUP_COL)[LABEL_COL].first().value_counts().to_string())
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    required = {LABEL_COL, GROUP_COL, SUBJECT_COL, *CATEGORICAL_FEATURES}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[LABEL_COL].isna().any():
        raise ValueError(f"Label column {LABEL_COL} contains missing values")

    inconsistent_labels = df.groupby(GROUP_COL)[LABEL_COL].nunique()
    inconsistent_labels = inconsistent_labels[inconsistent_labels > 1]
    if not inconsistent_labels.empty:
        raise ValueError(f"Experiments with inconsistent labels: {inconsistent_labels.index.tolist()}")

    total_nans = int(df.isna().sum().sum())
    print(f"NaN check: {total_nans} missing values")


def make_train_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_labels = df.groupby(GROUP_COL)[LABEL_COL].first()
    rng = np.random.default_rng(RANDOM_STATE)

    test_groups = []
    for label_value, group_ids in experiment_labels.groupby(experiment_labels):
        ids = group_ids.index.to_numpy()
        n_test = max(1, round(len(ids) * TEST_SIZE))
        sampled = rng.choice(ids, size=n_test, replace=False)
        test_groups.extend(sampled.tolist())

    test_groups = set(test_groups)
    train_df = df[~df[GROUP_COL].isin(test_groups)].copy()
    test_df = df[df[GROUP_COL].isin(test_groups)].copy()

    train_groups = set(train_df[GROUP_COL])
    test_groups = set(test_df[GROUP_COL])
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError(f"Data leakage: experiments in both train and test: {sorted(overlap)}")

    print("\nTrain/test split by experiment_id")
    print(f"Train windows: {len(train_df)}, experiments: {len(train_groups)}")
    print(f"Test windows:  {len(test_df)}, experiments: {len(test_groups)}")
    print("Train experiment labels:")
    print(train_df.groupby(GROUP_COL)[LABEL_COL].first().value_counts().to_string())
    print("Test experiment labels:")
    print(test_df.groupby(GROUP_COL)[LABEL_COL].first().value_counts().to_string())
    return train_df, test_df


def make_xy(train_df: pd.DataFrame, test_df: pd.DataFrame):
    feature_df = train_df.drop(columns=[col for col in DROP_COLS if col in train_df.columns])
    feature_df = feature_df.drop(columns=[SUBJECT_COL], errors="ignore")
    feature_cols = feature_df.columns.tolist()

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[LABEL_COL].astype(bool)
    y_test = test_df[LABEL_COL].astype(bool)

    categorical = [col for col in CATEGORICAL_FEATURES if col in feature_cols]
    numeric = [col for col in feature_cols if col not in categorical]

    print("\nFeature setup")
    print(f"Numeric features: {len(numeric)}")
    print(f"Categorical features: {categorical}")
    print(f"Total model input columns before one-hot encoding: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test, numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ]
    )


def build_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocess", make_preprocessor(numeric, categorical, scale_numeric=True)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "KNN": Pipeline(
            [
                ("preprocess", make_preprocessor(numeric, categorical, scale_numeric=True)),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Random Forest": Pipeline(
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
        ),
        "SVM": Pipeline(
            [
                ("preprocess", make_preprocessor(numeric, categorical, scale_numeric=True)),
                ("model", SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    rows = []

    for name, model in models.items():
        print(f"\n=== {name} ===")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, y_pred, labels=[False, True])

        print(f"Accuracy: {accuracy:.3f}")
        print("Confusion matrix, rows=true, cols=pred, labels=[False, True]:")
        print(matrix)
        print(classification_report(y_test, y_pred, target_names=["both_hands", "one_hand"], zero_division=0))

        rows.append(
            {
                "model": name,
                "accuracy": accuracy,
                "precision_both_hands": report["False"]["precision"],
                "recall_both_hands": report["False"]["recall"],
                "f1_both_hands": report["False"]["f1-score"],
                "precision_one_hand": report["True"]["precision"],
                "recall_one_hand": report["True"]["recall"],
                "f1_one_hand": report["True"]["f1-score"],
                "macro_f1": report["macro avg"]["f1-score"],
                "weighted_f1": report["weighted avg"]["f1-score"],
                "tn": matrix[0, 0],
                "fp": matrix[0, 1],
                "fn": matrix[1, 0],
                "tp": matrix[1, 1],
            }
        )

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    results.to_csv(RESULTS_CSV, index=False)
    print(f"\nSaved model comparison to {RESULTS_CSV}")
    print(results[["model", "accuracy", "macro_f1", "f1_both_hands", "f1_one_hand"]].to_string(index=False))
    return results


def main() -> None:
    df = load_features()
    validate_dataset(df)
    train_df, test_df = make_train_test_split(df)
    X_train, X_test, y_train, y_test, numeric, categorical = make_xy(train_df, test_df)
    models = build_models(numeric, categorical)
    evaluate_models(models, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
