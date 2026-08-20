"""Central configuration for the fraud detection pipeline."""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
MODEL_PATH = MODELS_DIR / "fraud_detection_model.joblib"
METRICS_PATH = ASSETS_DIR / "metrics.json"
DASHBOARD_DATA_PATH = ASSETS_DIR / "dashboard_data.csv"
GRAPH_IMAGE_PATH = ASSETS_DIR / "fraud_network_graph.png"

# --- Modeling --------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- NLP / compliance report extraction ------------------------------------
# Terms that indicate elevated risk in unstructured compliance narratives.
RISK_KEYWORDS = [
    "suspicious", "unauthorized", "unusual pattern", "structuring",
    "shell company", "high-risk jurisdiction", "sanctions", "money laundering",
    "layering", "smurfing", "offshore", "politically exposed", "cash-intensive",
    "unverified source of funds", "rapid movement of funds", "dormant account",
]

for _dir in (DATA_DIR, ASSETS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

