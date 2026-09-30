"""Run the complete supervised machine learning workflow for the project."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_preprocessing import clean_dataset, load_dataset, prepare_classification_data, prepare_regression_data, split_dataset
from src.evaluation import (
    calculate_classification_metrics,
    calculate_regression_metrics,
    create_classification_comparison_table,
    create_model_comparison_table,
    create_regression_comparison_table,
    make_confusion_matrix,
    save_results_dataframe,
)
from src.classification_models import (
    train_decision_tree_classifier,
    train_knn_classifier,
    train_logistic_regression,
    train_random_forest_classifier,
)
from src.regression_models import (
    train_decision_tree_regressor,
    train_linear_regression,
    train_random_forest_regressor,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "dataset.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def save_plot(fig, filename: str):
    """Save a matplotlib figure to the figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    file_path = FIGURES_DIR / filename
    fig.tight_layout()
    fig.savefig(file_path, dpi=300)
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame):
    """Plot the distribution of wine quality scores."""
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(df["quality"], bins=10, kde=True, color="steelblue", ax=ax)
    ax.set_title("Wine Quality Distribution")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    save_plot(fig, "target_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot a correlation heatmap for model features."""
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    save_plot(fig, "correlation_heatmap.png")


def plot_regression_actual_vs_predicted(y_true, y_pred, title_prefix=""):
    """Create an actual-vs-predicted scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5)
    ax.set_title(f"{title_prefix}Actual vs Predicted Quality")
    ax.set_xlabel("Actual Quality")
    ax.set_ylabel("Predicted Quality")
    save_plot(fig, f"{title_prefix.lower().replace(' ', '_')}_actual_vs_predicted.png")


def plot_regression_model_comparison(model_results):
    """Create a model comparison chart for regression performance."""
    comparison_df = create_regression_comparison_table(model_results)
    fig, ax = plt.subplots(figsize=(8, 6))
    summary = comparison_df.set_index("Model")["RMSE"].sort_values()
    summary.plot(kind="bar", color="darkgreen", ax=ax)
    ax.set_title("Regression Model RMSE Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    save_plot(fig, "regression_model_comparison.png")


def plot_classification_confusion_matrix(model_name: str, y_true, y_pred):
    """Plot a confusion matrix for a classification model."""
    cm = make_confusion_matrix(y_true, y_pred)
    labels = [0, 1]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    file_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    save_plot(fig, file_name)


def plot_classification_accuracy_comparison(model_results):
    """Compare accuracy scores for classification models."""
    comparison_df = create_classification_comparison_table(model_results)
    fig, ax = plt.subplots(figsize=(8, 6))
    comparison_df.set_index("Model")["Accuracy"].sort_values().plot(kind="bar", color="purple", ax=ax)
    ax.set_title("Classification Model Accuracy Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    save_plot(fig, "classification_accuracy_comparison.png")


def plot_classification_f1_comparison(model_results):
    """Compare F1 scores for classification models."""
    comparison_df = create_classification_comparison_table(model_results)
    fig, ax = plt.subplots(figsize=(8, 6))
    comparison_df.set_index("Model")["F1"].sort_values().plot(kind="bar", color="darkorange", ax=ax)
    ax.set_title("Classification Model F1 Score Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("F1 Score")
    save_plot(fig, "classification_f1_comparison.png")


def run_regression_workflow(X_train, X_test, y_train, y_test):
    """Train, evaluate, and save regression models."""
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Decision Tree Regressor": train_decision_tree_regressor(X_train, y_train, random_state=42),
        "Random Forest Regressor": train_random_forest_regressor(X_train, y_train, random_state=42, n_estimators=200),
    }

    results = {}
    pd_frames = []
    for name, model in models.items():
        predictions = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, predictions)
        results[name] = metrics
        pd_frames.append({"Model": name, **metrics})

        plot_regression_actual_vs_predicted(y_test.to_numpy(), predictions, title_prefix=f"{name} - ")

    regression_df = pd.DataFrame(pd_frames, columns=["Model", "MAE", "MSE", "RMSE", "R2"])
    save_results_dataframe(regression_df, RESULTS_DIR / "regression_results.csv")
    plot_regression_model_comparison(results)
    return regression_df, results


def run_classification_workflow(X_train, X_test, y_train, y_test):
    """Train, evaluate, and save classification models."""
    models = {
        "Logistic Regression": train_logistic_regression(X_train, y_train, random_state=42),
        "Decision Tree Classifier": train_decision_tree_classifier(X_train, y_train, random_state=42),
        "Random Forest Classifier": train_random_forest_classifier(X_train, y_train, random_state=42, n_estimators=200),
        "KNN Classifier": train_knn_classifier(X_train, y_train, random_state=42, n_neighbors=5),
    }

    results = {}
    rows = []
    for name, model in models.items():
        predictions = model.predict(X_test)
        metrics = calculate_classification_metrics(y_test, predictions)
        results[name] = metrics
        rows.append({"Model": name, **metrics})
        plot_classification_confusion_matrix(name, y_test, predictions)

    classification_df = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
    save_results_dataframe(classification_df, RESULTS_DIR / "classification_results.csv")
    plot_classification_accuracy_comparison(results)
    plot_classification_f1_comparison(results)
    return classification_df, results


def main():
    """Execute the end-to-end supervised ML workflow."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = clean_dataset(load_dataset(DATA_PATH))
    print("\n=== Dataset Overview ===")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(df.head().to_string(index=False))
    print(f"Null values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(df.describe().to_string())

    plot_target_distribution(df)
    plot_correlation_heatmap(df)

    X_reg, y_reg = prepare_regression_data(df)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_dataset(X_reg, y_reg, task="regression")

    X_cls, y_cls = prepare_classification_data(df, threshold=6)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = split_dataset(X_cls, y_cls, task="classification")

    print("\n=== Train/Test Split ===")
    print(f"Regression training size: {X_train_reg.shape[0]}")
    print(f"Regression testing size: {X_test_reg.shape[0]}")
    print(f"Classification training size: {X_train_cls.shape[0]}")
    print(f"Classification testing size: {X_test_cls.shape[0]}")

    regression_df, regression_results = run_regression_workflow(X_train_reg, X_test_reg, y_train_reg, y_test_reg)
    classification_df, classification_results = run_classification_workflow(X_train_cls, X_test_cls, y_train_cls, y_test_cls)

    combined_df = create_model_comparison_table(regression_results, classification_results)
    save_results_dataframe(combined_df, RESULTS_DIR / "model_comparison.csv")

    print("\n=== Regression Model Results ===")
    print(regression_df.to_string(index=False))
    print("\n=== Classification Model Results ===")
    print(classification_df.to_string(index=False))
    print("\n=== Result Storage ===")
    print(f"Regression results saved to: {RESULTS_DIR / 'regression_results.csv'}")
    print(f"Classification results saved to: {RESULTS_DIR / 'classification_results.csv'}")
    print(f"Model comparison saved to: {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Figures saved to: {FIGURES_DIR}")

    print("\nProject completed successfully. Different metrics evaluate different aspects of model performance.")


if __name__ == "__main__":
    main()
