# Automated Fraud Detection System for Financial Transactions

A production-structured ML pipeline for flagging fraudulent credit card
transactions, extracting risk signals from unstructured compliance
reports, and surfacing both to analysts through a live dashboard.

Built as a portfolio project demonstrating end-to-end ML engineering:
model comparison, testing, CI/CD, containerization, and a
decision-support interface — not just a notebook.

---

## What it does

Given a transaction dataset, the pipeline preprocesses features, trains
and compares **Random Forest** and **XGBoost** classifiers (selecting
the better one by F1 on the fraud class), evaluates it, builds a
similarity network of flagged transactions, and exports a clean data
table an analyst dashboard (or Power BI) can consume directly.

Separately, an NLP module scores unstructured compliance report text
against a bank of known risk phrases (structuring, shell companies,
sanctions exposure, etc.) using TF-IDF, and extracts structured fields
(amounts, dates, account/invoice references) via targeted regex — so
analysts can triage report volume instead of reading every report in
full.

---

## Architecture

```
data/creditcard.csv (not committed — see Dataset Access)
        │
        ▼
src/preprocessing.py   → normalize amounts, split features/target
        │
        ▼
src/train.py            → train RandomForest + XGBoost, select best by F1
        │
        ▼
src/evaluate.py          → confusion matrix, classification report → assets/metrics.json
        │
        ├── src/graph_network.py         → fraud similarity graph → assets/fraud_network_graph.png
        └── dashboard/export_dashboard_data.py → assets/dashboard_data.csv
                                                        │
                                                        ▼
                                        dashboard/app.py (Streamlit) or Power BI "Get Data"

src/nlp_extraction.py   → compliance report text → risk score + extracted entities
                           (independent of the transaction pipeline; run on report text directly)
```

---

## Tech Stack

| Layer                  | Technology                                  |
| ----------------------- | -------------------------------------------- |
| Data processing         | Pandas, NumPy                                |
| Modeling                | Scikit-learn (Random Forest), XGBoost        |
| NLP / risk scoring      | Scikit-learn TF-IDF + rule-based regex       |
| Graph analysis          | NetworkX, Matplotlib                         |
| Dashboard               | Streamlit (data also Power BI–compatible)    |
| Model persistence       | Joblib                                       |
| Testing                 | Pytest                                       |
| CI/CD                   | GitHub Actions                               |
| Containerization        | Docker                                       |

---

## Project Structure

```
fraud_detection/
├── src/
│   ├── config.py            # paths, model params, risk keyword list
│   ├── preprocessing.py     # load + clean transaction data
│   ├── train.py             # RandomForest + XGBoost training/selection
│   ├── evaluate.py          # metrics computation and reporting
│   ├── graph_network.py     # fraud similarity network
│   └── nlp_extraction.py    # compliance report risk scoring + entity extraction
├── dashboard/
│   ├── export_dashboard_data.py   # flat table for BI tools
│   └── app.py                     # Streamlit investigation dashboard
├── tests/                   # pytest suite (synthetic data, no real dataset needed)
├── assets/                  # generated: metrics.json, dashboard_data.csv, graph.png
├── models/                  # generated: fraud_detection_model.joblib
├── data/                    # place creditcard.csv here (not committed)
├── main.py                  # pipeline entrypoint
├── requirements.txt
├── Dockerfile
├── .github/workflows/ci.yml
└── README.md
```

---

## Getting Started

### 1. Create and activate a virtual environment

```
python -m venv env
env\Scripts\activate        # Windows
source env/bin/activate     # Mac / Linux
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Add the dataset

Download from Kaggle — [Credit Card Fraud
Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) — and place
`creditcard.csv` in `data/`.

### 4. Run the pipeline

```
python main.py
```

This trains both models, saves the better one, writes evaluation
metrics, builds the fraud network graph, and exports the dashboard
data table.

### 5. Launch the dashboard

```
streamlit run dashboard/app.py
```

---

## Testing

```
pytest tests/ -v
```

Tests run against synthetic data (see `tests/conftest.py`), so the full
suite runs in CI without needing the real (large, license-restricted)
dataset.

---

## Docker

```
docker build -t fraud-detection .
docker run --rm -v $(pwd)/data:/app/data fraud-detection
```

To serve the dashboard instead of running the training pipeline:

```
docker run --rm -p 8501:8501 fraud-detection \
  streamlit run dashboard/app.py --server.address 0.0.0.0
```

---

## Power BI Integration

`dashboard/export_dashboard_data.py` writes `assets/dashboard_data.csv`
— one row per scored transaction with `fraud_probability`,
`risk_tier`, and `flagged_for_review`. Point Power BI's **Get Data →
Text/CSV** at this file (or swap the CSV writer for a database/API
call for a live refresh in production) to build the same
investigation-prioritization views as the bundled Streamlit dashboard.

---

## Compliance Report NLP Module

Independent of the transaction pipeline — analyze free-text compliance
narratives directly:

```python
from src.nlp_extraction import analyze_report

insight = analyze_report(report_text)
insight.risk_score          # aggregate TF-IDF salience of known risk phrases
insight.matched_risk_terms  # which risk phrases were found, ranked
insight.amounts              # extracted dollar amounts
insight.dates                 # extracted dates
insight.references            # extracted account/invoice references
```

---

## Example Output

```
Confusion Matrix:
[[56862, 2], [26, 72]]

Fraud class — precision: 0.973, recall: 0.735, f1: 0.837
```

*(Illustrative — actual numbers depend on the model selected and the
specific train/test split.)*

---

## Limitations

- Dataset is heavily class-imbalanced; F1 on the fraud class, not
  accuracy, is the metric that matters.
- The NLP risk-scoring bank is a fixed keyword list — it will miss risk
  language it wasn't given, and would benefit from a labeled corpus for
  a supervised classifier in a real deployment.
- The similarity network uses a single feature (`V1`) as a proxy for
  transaction similarity; a production version would likely use a
  learned embedding or multiple features.

---

## Future Enhancements

- Real-time scoring endpoint (FastAPI) instead of batch pipeline runs
- Feature importance / SHAP explanations surfaced in the dashboard
- Swap the keyword-based NLP scorer for a supervised classifier trained
  on labeled compliance reports
- Incremental/streaming dashboard data updates

---

## Dataset Access

Due to size and license constraints, the dataset is not included in
this repository. Download it from Kaggle — [Credit Card Fraud
Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud).

---

## Author

Tejaswini Sengaonkar
[LinkedIn](https://linkedin.com/in/tejaswini-sengaonkar) | [GitHub](https://github.com/tsen057)

