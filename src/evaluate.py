"""Evaluation utilities: metrics computation and reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


def evaluate_model(model: Any, X_test, y_test) -> dict:
    """Compute standard classification metrics for the fraud class."""
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[1], zero_division=0
    )

    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "fraud_precision": float(precision[0]),
        "fraud_recall": float(recall[0]),
        "fraud_f1": float(f1[0]),
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", path)


def print_report(metrics: dict) -> None:
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(f"\nFraud class — precision: {metrics['fraud_precision']:.3f}, "
          f"recall: {metrics['fraud_recall']:.3f}, f1: {metrics['fraud_f1']:.3f}")

