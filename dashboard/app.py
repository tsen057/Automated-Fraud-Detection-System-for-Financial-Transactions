"""Interactive investigation-prioritization dashboard.

Reads the CSV produced by export_dashboard_data.py and lets an analyst
filter, sort, and prioritize flagged transactions. Run with:

    streamlit run dashboard/app.py

This is a genuinely runnable substitute for the Power BI dashboard
described in the project summary — Power BI itself isn't something that
can be generated as a code file, but the same underlying data feed
(assets/dashboard_data.csv) can be pointed at Power BI's "Get Data"
import just as easily as at this app.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "assets" / "dashboard_data.csv"

st.set_page_config(page_title="Fraud Investigation Dashboard", layout="wide")
st.title("Fraud Investigation Prioritization")

if not DATA_PATH.exists():
    st.error(
        f"No dashboard data found at {DATA_PATH}. "
        "Run the pipeline (main.py) first to generate predictions."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

col1, col2, col3 = st.columns(3)
col1.metric("Transactions scored", len(df))
col2.metric("Flagged for review", int(df["flagged_for_review"].sum()))
col3.metric(
    "High-risk tier",
    int((df["risk_tier"] == "high").sum()) if "risk_tier" in df.columns else 0,
)

st.divider()

risk_filter = st.multiselect(
    "Risk tier",
    options=sorted(df["risk_tier"].dropna().unique().tolist()) if "risk_tier" in df.columns else [],
    default=None,
)
flagged_only = st.checkbox("Show flagged transactions only", value=True)

view = df.copy()
if risk_filter:
    view = view[view["risk_tier"].isin(risk_filter)]
if flagged_only:
    view = view[view["flagged_for_review"]]

view = view.sort_values("fraud_probability", ascending=False)

st.subheader(f"Transactions ({len(view)})")
st.dataframe(view, use_container_width=True)

st.download_button(
    "Download filtered view as CSV",
    data=view.to_csv(index=False),
    file_name="filtered_fraud_review.csv",
    mime="text/csv",
)

