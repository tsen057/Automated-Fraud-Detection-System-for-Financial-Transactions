#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Install required libraries
#!pip install pandas numpy scikit-learn nltk spacy networkx seaborn matplotlib
#!pip install pyresparser # For NLP-based entity extraction
#!pip install plotly # For visualization


# In[6]:


import pandas as pd

# Load the credit card fraud detection dataset
df = pd.read_csv("C:/Users/tejas/Downloads/Creditcard/creditcard.csv")

# Check for the first few rows of the dataset
df.head()


# In[7]:


from sklearn.preprocessing import StandardScaler

# Check for missing values
df.isnull().sum()

# Standardize the data (since features are in different ranges)
scaler = StandardScaler()
df['normalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))

# Drop the columns that are not needed
df = df.drop(columns=['Time', 'Amount'])

# Check the data again after preprocessing
df.head()


# In[8]:


from sklearn.model_selection import train_test_split

# Split the data into features (X) and target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.shape, X_test.shape


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[10]:


import networkx as nx
import matplotlib.pyplot as plt

# Create a graph to represent fraudulent transactions
G = nx.Graph()

# Add nodes and edges based on fraudulent transactions
for i in range(len(df)):
    if df['Class'][i] == 1:  # Fraudulent transactions
        G.add_node(i)
        G.add_edge(i, df['V1'][i])  # Example of connecting based on features

# Plot the network
plt.figure(figsize=(10, 10))
nx.draw(G, with_labels=True, node_size=50, font_size=10)
plt.title('Network of Fraudulent Transactions')
plt.show()


# In[11]:


import spacy
from spacy import displacy

# Load SpaCy model for named entity recognition
nlp = spacy.load('en_core_web_sm')

# Example text (You can extend this to actual financial document parsing)
doc_text = """
Invoice ID: 12345
Vendor: ABC Corp
Amount: 5000 USD
Transaction Date: 01/01/2023
"""

# Process the document text with SpaCy
doc = nlp(doc_text)

# Display named entities in the document
for ent in doc.ents:
    print(f"{ent.text} - {ent.label_}")

# Visualize the entities in the document (optional)
displacy.render(doc, style='ent')


# In[12]:


import joblib

# Save the trained model
joblib.dump(model, 'fraud_detection_model.joblib')

# Load the model back
loaded_model = joblib.load('fraud_detection_model.joblib')

# Make predictions with the loaded model
y_pred_loaded = loaded_model.predict(X_test)

