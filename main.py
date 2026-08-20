"""End-to-end fraud detection pipeline.

Usage:
    python main.py
"""

from __future__ import annotations

import logging

from src.config import (
    DASHBOARD_DATA_PATH,
    GRAPH_IMAGE_PATH,
    METRICS_PATH,
    RAW_DATA_PATH,
)
from src.evaluate import evaluate_model, print_report, save_metrics
from src.graph_network import build_fraud_network, save_network_plot
from src.preprocessing import (
    load_transactions,
    preprocess,
    split_features_target,
    train_test_split_data,
)
from src.train import save_model, train_and_select
from dashboard.export_dashboard_data import build_dashboard_table, export

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    raw_df = load_transactions(RAW_DATA_PATH)
    df = preprocess(raw_df)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    best_name, model, scores = train_and_select(X_train, y_train, X_test, y_test)
    logger.info("Selected model: %s (scores: %s)", best_name, scores)
    save_model(model)

    metrics = evaluate_model(model, X_test, y_test)
    metrics["model_selected"] = best_name
    metrics["model_comparison_f1"] = scores
    print_report(metrics)
    save_metrics(metrics, METRICS_PATH)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    table = build_dashboard_table(X_test, y_test, y_pred, y_pred_proba)
    export(table, DASHBOARD_DATA_PATH)

    graph = build_fraud_network(df)
    save_network_plot(graph, GRAPH_IMAGE_PATH)


if __name__ == "__main__":
    run_pipeline()

