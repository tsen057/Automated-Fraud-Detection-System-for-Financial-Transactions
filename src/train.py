"""Train and compare Random Forest and XGBoost fraud classifiers."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.config import MODEL_PATH, RANDOM_STATE

logger = logging.getLogger(__name__)


def build_candidate_models() -> dict[str, Any]:
    """Return the set of candidate models to train and compare.

    Both use class-imbalance handling since fraud is a small minority class.
    """
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            # Roughly offsets class imbalance; recomputed properly in train_and_select.
            scale_pos_weight=1,
            n_jobs=-1,
        ),
    }


def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[str, Any, dict[str, float]]:
    """Train all candidate models and return the best one by F1 on the minority class.

    Returns (best_model_name, best_model, {model_name: f1_score}).
    """
    models = build_candidate_models()

    # Give XGBoost a realistic imbalance weight based on the training split.
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    if n_pos > 0:
        models["xgboost"].set_params(scale_pos_weight=n_neg / n_pos)

    scores: dict[str, float] = {}
    fitted: dict[str, Any] = {}

    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        score = f1_score(y_val, preds, pos_label=1, zero_division=0)
        scores[name] = score
        fitted[name] = model
        logger.info("%s F1 (fraud class): %.4f", name, score)

    best_name = max(scores, key=scores.get)
    return best_name, fitted[best_name], scores


def save_model(model: Any, path=MODEL_PATH) -> None:
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path=MODEL_PATH) -> Any:
    return joblib.load(path)

