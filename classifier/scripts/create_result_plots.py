from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CLASSIFIER_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CLASSIFIER_ROOT / "results"
PLOT_DIR = DATA_DIR / "plots"


def save_bar_plot(df, x_col, y_col, err_col, title, ylabel, output_path, color="#4C78A8"):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(df))
    ax.bar(x, df[y_col], yerr=df[err_col] if err_col else None, capsize=5, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col], rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    for idx, value in enumerate(df[y_col]):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_cv():
    df = pd.read_csv(DATA_DIR / "model_baseline_cross_validation_results.csv")
    summary = (
        df[["model", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]]
        .drop_duplicates()
        .sort_values("macro_f1_mean", ascending=False)
    )
    save_bar_plot(
        summary,
        "model",
        "macro_f1_mean",
        "macro_f1_std",
        "Baseline Model Comparison (5-fold grouped CV)",
        "Macro F1",
        PLOT_DIR / "baseline_model_comparison_macro_f1.png",
        color="#4C78A8",
    )


def plot_feature_selection_cv():
    df = pd.read_csv(DATA_DIR / "random_forest_feature_selection_cross_validation_results.csv")
    summary = (
        df[["feature_set", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]]
        .drop_duplicates()
        .sort_values("macro_f1_mean", ascending=False)
    )
    labels = {
        "top_10_rf_importance": "Top 10",
        "top_20_rf_importance": "Top 20",
        "top_40_rf_importance": "Top 40",
        "all_features": "All",
    }
    summary["feature_set"] = summary["feature_set"].map(labels).fillna(summary["feature_set"])
    save_bar_plot(
        summary,
        "feature_set",
        "macro_f1_mean",
        "macro_f1_std",
        "Random Forest Feature Selection (5-fold grouped CV)",
        "Macro F1",
        PLOT_DIR / "feature_selection_macro_f1.png",
        color="#59A14F",
    )


def plot_phase_cv():
    df = pd.read_csv(DATA_DIR / "random_forest_top10_by_phase_cross_validation_results.csv")
    summary = (
        df[["phase", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]]
        .drop_duplicates()
        .sort_values("macro_f1_mean", ascending=False)
    )
    save_bar_plot(
        summary,
        "phase",
        "macro_f1_mean",
        "macro_f1_std",
        "Phase-specific Classification (RF Top 10, grouped CV)",
        "Macro F1",
        PLOT_DIR / "phase_comparison_macro_f1.png",
        color="#F28E2B",
    )


def plot_box_size_cv():
    df = pd.read_csv(DATA_DIR / "random_forest_top10_by_box_size_cross_validation_results.csv")
    summary = (
        df[["box_size", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"]]
        .drop_duplicates()
        .sort_values("macro_f1_mean", ascending=False)
    )
    save_bar_plot(
        summary,
        "box_size",
        "macro_f1_mean",
        "macro_f1_std",
        "Box-size-specific Classification (RF Top 10, grouped CV)",
        "Macro F1",
        PLOT_DIR / "box_size_comparison_macro_f1.png",
        color="#E15759",
    )


def plot_shap_top_features():
    df = pd.read_csv(DATA_DIR / "feature_importance_shap_random_forest.csv").head(15)
    df = df.sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(df["feature"], df["mean_abs_shap"], color="#B07AA1")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Top SHAP Features (Random Forest)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "shap_top_features_bar.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_baseline_cv()
    plot_feature_selection_cv()
    plot_phase_cv()
    plot_box_size_cv()
    plot_shap_top_features()
    print(f"Saved plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
