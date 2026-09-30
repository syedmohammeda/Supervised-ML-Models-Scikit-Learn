"""Evaluation utilities for regression and classification tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


def calculate_regression_metrics(y_true, y_pred):
    """Compute standard regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def calculate_classification_metrics(y_true, y_pred):
    """Compute classification metrics for binary labels."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def make_confusion_matrix(y_true, y_pred):
    """Return the confusion matrix array."""
    return confusion_matrix(y_true, y_pred)


def save_results_dataframe(df: pd.DataFrame, output_path: str | Path):
    """Save a DataFrame to CSV and create parent folders if needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def create_regression_comparison_table(results_dict):
    """Combine model names and regression metric values in one DataFrame."""
    rows = []
    for model_name, metrics in results_dict.items():
        row = {"Model": model_name, "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "R2": metrics["R2"]}
        rows.append(row)
    return pd.DataFrame(rows, columns=["Model", "MAE", "RMSE", "R2"])


def create_classification_comparison_table(results_dict):
    """Combine model names and classification metric values in one DataFrame."""
    rows = []
    for model_name, metrics in results_dict.items():
        row = {
            "Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])


def create_model_comparison_table(regression_results, classification_results):
    """Merge regression and classification comparison tables into a single summary."""
    regression_rows = [{"Model": model_name, **metrics} for model_name, metrics in regression_results.items()]
    classification_rows = [{"Model": model_name, **metrics} for model_name, metrics in classification_results.items()]

    regression_df = pd.DataFrame(regression_rows)
    classification_df = pd.DataFrame(classification_rows)

    merged = regression_df.merge(classification_df, on="Model", how="outer")
    return merged
