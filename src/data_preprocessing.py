"""Utilities for loading, cleaning, and preparing the dataset for training."""

from __future__ import annotations

from pathlib import Path
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DATASET_PATH = DATA_DIR / "dataset.csv"


def ensure_dataset_file(path: Path | str = DATASET_PATH) -> Path:
    """Ensure a local copy of the dataset exists.

    The project prefers the UCI Wine Quality dataset. If the file is unavailable,
    the function downloads it from the official UCI archive. If that download fails,
    it falls back to the built-in scikit-learn wine dataset as a last resort.
    """
    dataset_path = Path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        return dataset_path

    uci_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        with urllib.request.urlopen(uci_url, timeout=30) as response:
            content = response.read()
        dataset_path.write_bytes(content)
        return dataset_path
    except Exception:
        try:
            from sklearn.datasets import load_wine

            wine_frame = load_wine(as_frame=True, return_X_y=False).frame.copy()
            wine_frame.rename(columns={"target": "quality"}, inplace=True)
            wine_frame["quality"] = wine_frame["quality"].map({0: 5, 1: 6, 2: 7})
            wine_frame.to_csv(dataset_path, index=False)
            return dataset_path
        except Exception as exc:  # pragma: no cover - fallback safety net
            raise RuntimeError(
                "The dataset could not be downloaded and the local fallback dataset is unavailable. "
                f"Original error: {exc}"
            ) from exc


def load_dataset(path: Path | str = DATASET_PATH) -> pd.DataFrame:
    """Load the local dataset into a pandas DataFrame and validate it."""
    dataset_path = ensure_dataset_file(path)

    with open(dataset_path, "r", encoding="utf-8") as file:
        first_line = file.readline().strip()

    delimiter = ";" if ";" in first_line else ","
    df = pd.read_csv(dataset_path, sep=delimiter)

    if df.empty:
        raise ValueError(f"The dataset at {dataset_path} is empty.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and fill missing numeric values with the column median."""
    cleaned_df = df.copy()
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    numeric_columns = cleaned_df.select_dtypes(include=["number"]).columns
    for column in numeric_columns:
        cleaned_df[column] = cleaned_df[column].fillna(cleaned_df[column].median())

    return cleaned_df


def create_classification_target(df: pd.DataFrame, quality_column: str = "quality", threshold: int = 6) -> pd.DataFrame:
    """Convert the numeric quality score into a binary classification target.

    Wines with quality >= 6 are treated as 'higher quality' and assigned 1, while
    wines with quality < 6 are assigned 0.
    """
    processed_df = df.copy()
    processed_df["quality_label"] = (processed_df[quality_column] >= threshold).astype(int)
    return processed_df


def split_dataset(X, y, task: str = "regression", test_size: float = 0.2, random_state: int = 42):
    """Split features and target into train/test partitions with reproducibility."""
    if task == "classification":
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def prepare_regression_data(df: pd.DataFrame):
    """Separate features and target for the regression task."""
    X = df.drop(columns=["quality"])
    y = df["quality"]
    return X, y


def prepare_classification_data(df: pd.DataFrame, threshold: int = 6):
    """Separate features and target for the classification task."""
    processed_df = create_classification_target(df, threshold=threshold)
    X = processed_df.drop(columns=["quality", "quality_label"])
    y = processed_df["quality_label"]
    return X, y
