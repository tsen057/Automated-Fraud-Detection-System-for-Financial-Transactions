# Automated Fraud Detection System for Financial Transactions

## Overview
This project uses machine learning and network graph analysis to identify potentially fraudulent credit card transactions. It also includes a lightweight regex-based module to extract structured financial information from text. This system is ideal for high-volume financial data environments.

## Key Features
- **Preprocessing**: Normalizes transaction amounts and drops irrelevant features.
- **Modeling**: Uses Random Forest with class balancing to address imbalanced classes.
- **Evaluation**: Prints confusion matrix and classification report.
- **Visualization**: Displays an improved fraud transaction network using feature similarity.
- **Entity Extraction**: Extracts invoice info using regular expressions (no heavy NLP libraries).
- **Model Persistence**: Saves and reloads trained model using `joblib`.

## Tech Stack

| Category           | Tools & Libraries         |
|--------------------|--------------------------|
| Programming        | Python                   |
| Data Processing    | Pandas, NumPy            |
| Machine Learning   | Scikit-learn             |
| Visualization      | NetworkX, Matplotlib     |
| Model Persistence  | Joblib                   |
| Other              | Regex                    |

## Project Structure
```
Automated-Fraud-Detection-System
├── Improved_Fraud_Graph_And_Model.py       # Main Python script
├── fraud_detection_model.joblib            # Saved trained model
├── assets/
│   └── fraud_network_graph.png             # Graph image 
├── README.md
└── requirements.txt
```

## Example Output

### Confusion Matrix:
```
[[56862     2]
 [   26    72]]
```

### Classification Report:
```
              precision    recall  f1-score   support
           0       1.00      1.00      1.00     56864
           1       0.97      0.73      0.84        98
```

### Graph Sample:
![Fraud Network](assets/fraud_network_graph.png)

## Dataset Access
Due to size constraints, the dataset is not included in this repository.

Download it from [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  

## Future Enhancements
- Add real-time fraud prediction module.
- Try advanced algorithms (XGBoost, LightGBM).
- Integrate Streamlit or Flask for live demos.

## Author
**Tejaswini Sengaonkar**  
[LinkedIn](https://www.linkedin.com/in/tejaswini-sengaonkar) | [GitHub](https://github.com/tsen057)
