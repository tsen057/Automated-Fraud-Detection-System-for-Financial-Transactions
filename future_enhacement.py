"""
XGBoost addition for the Automated Fraud Detection System.

This follows the same structure as your existing Creditcard.py:
- same preprocessing approach (normalize Amount, drop irrelevant columns)
- same class-imbalance handling (fraud is a tiny fraction of transactions)
- same evaluation style (confusion matrix + classification report)
- same persistence approach (joblib)

Install if needed:  pip install xgboost --break-system-packages
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier

# ---- 1. Load data -----------------------------------------------------
DATA_PATH = "C:/Users/tejas/Downloads/Creditcard/creditcard.csv"  # matches your RF script
df = pd.read_csv(DATA_PATH)

# ---- 2. Preprocess (exact match to your Random Forest script) ---------
scaler = StandardScaler()
df['normalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df = df.drop(columns=['Time', 'Amount'])

X = df.drop(columns=['Class'])
y = df['Class']

# NOTE: your RF script does not use stratify= — using the identical
# split call here so X_test/y_test are the exact same rows, making the
# two models directly comparable.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 3. Train XGBoost with class imbalance handling --------------------
# Your RF script uses class_weight='balanced'. XGBoost doesn't have that
# exact parameter, so scale_pos_weight is the equivalent mechanism —
# same purpose (upweight the rare fraud class), different name/library.
fraud_count = y_train.sum()
normal_count = len(y_train) - fraud_count
scale_pos_weight = normal_count / fraud_count

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)
xgb_model.fit(X_train, y_train)

# ---- 4. Evaluate (same format as your Random Forest output) -----------
y_pred = xgb_model.predict(X_test)

print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred))

# ---- 5. Persist model ---------------------------------------------------
joblib.dump(xgb_model, "fraud_detection_model_xgboost.joblib")
print("\nModel saved to fraud_detection_model_xgboost.joblib")

# ---- 6. Feature importance (nice extra to mention on the call) --------
importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
print("\nTop 10 features by importance:")
print(importances.sort_values(ascending=False).head(10))
