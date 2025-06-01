import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load data
df = pd.read_csv("C:/Users/tejas/Downloads/Creditcard/creditcard.csv")

# Preprocessing
scaler = StandardScaler()
df['normalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(columns=['Time', 'Amount'])

X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Enhanced Fraud Graph (connect similar frauds by V1 similarity threshold)
fraud_df = df[df['Class'] == 1].copy()
fraud_indices = fraud_df.index.tolist()

G = nx.Graph()

# Add fraud nodes
G.add_nodes_from(fraud_indices)

# Connect nodes with similar V1 values
threshold = 0.5
for i in range(len(fraud_indices)):
    for j in range(i + 1, len(fraud_indices)):
        node_i = fraud_indices[i]
        node_j = fraud_indices[j]
        if abs(fraud_df.loc[node_i, 'V1'] - fraud_df.loc[node_j, 'V1']) < threshold:
            G.add_edge(node_i, node_j)

# Limit to top 50 for visual clarity
subgraph_nodes = list(G.nodes)[:50]
subG = G.subgraph(subgraph_nodes)

plt.figure(figsize=(12, 12))
nx.draw(subG, with_labels=True, node_size=100, font_size=8)
plt.title("Improved Network of Fraudulent Transactions")
plt.show()

# Save model
joblib.dump(model, 'fraud_detection_model.joblib')
