"""Export model predictions and compliance-report insights into flat,
analyst-friendly tables that Power BI (or Tableau, Looker, etc.) can
connect to directly.

Design note: Power BI itself is a desktop/cloud BI tool, not something a
script produces — a .pbix file isn't a meaningful "generated" artifact on
its own. The actual engineering work is producing a clean, well-typed,
regularly refreshed data source. This script is that data source. Point
Power BI's "Get Data > Text/CSV" (or a scheduled Power BI dataflow) at
`assets/dashboard_data.csv`, or swap the CSV writer for a database/API
call in production.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_dashboard_table(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred,
    y_pred_proba,
) -> pd.DataFrame:
    """Flatten predictions into one row per transaction for BI consumption."""
    table = pd.DataFrame(
        {
            "transaction_index": X.index,
            "normalized_amount": X["normalizedAmount"].values if "normalizedAmount" in X.columns else None,
            "actual_class": y_true.values,
            "predicted_class": y_pred,
            "fraud_probability": y_pred_proba,
        }
    )
    table["flagged_for_review"] = table["predicted_class"] == 1
    table["risk_tier"] = pd.cut(
        table["fraud_probability"],
        bins=[-0.01, 0.3, 0.7, 1.0],
        labels=["low", "medium", "high"],
    )
    return table


def export(table: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    logger.info("Exported %d rows to %s", len(table), output_path)

