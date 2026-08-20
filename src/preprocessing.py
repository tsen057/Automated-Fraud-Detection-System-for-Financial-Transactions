"""Data loading and preprocessing for the credit card transaction dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE, TEST_SIZE

logger = logging.getLogger(__name__)


def load_transactions(csv_path: str | Path) -> pd.DataFrame:
    """Load the raw transaction CSV.

    Raises FileNotFoundError with a clear message if the dataset (which is
    not committed to the repo, see README) is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download it from the Kaggle "
            "'Credit Card Fraud Detection' dataset and place it there."
        )
    logger.info("Loading transactions from %s", csv_path)
    return pd.read_csv(csv_path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the Amount column and drop columns not used for modeling.

    Returns a new DataFrame; does not mutate the input.
    """
    df = df.copy()
    if "Amount" not in df.columns:
        raise ValueError("Expected an 'Amount' column in the input data.")

    scaler = StandardScaler()
    df["normalizedAmount"] = scaler.fit_transform(df[["Amount"]])

    drop_cols = [c for c in ("Time", "Amount") if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def split_features_target(
    df: pd.DataFrame, target_col: str = "Class"
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Expected a '{target_col}' column in the input data.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

