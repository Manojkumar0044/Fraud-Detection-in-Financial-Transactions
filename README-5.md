# Fraud Detection in Financial Transactions

## Codec Technologies – Data Science Internship Project

### Project Title
**Fraud Detection in Financial Transactions Using Machine Learning**

### Objective
Build a machine-learning solution that identifies potentially fraudulent financial transactions and evaluates the model using metrics suitable for an imbalanced classification problem.

### Internship alignment
Codec Technologies' Data Science Internship syllabus covers data cleaning/preprocessing, exploratory data analysis, visualization, machine learning, model training/evaluation and a final project. This project follows that progression. 

### Tools & Technologies
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab
- GitHub

### Dataset
A reproducible 10,000-row financial transaction dataset is included in `data/financial_transactions.csv`. It is a synthetic learning dataset created for this project; it does not contain real customer or banking information.

### Features
- Transaction amount
- Transaction type
- Merchant risk score
- Account age
- Transaction frequency
- Failed attempts
- Device trust score
- Location mismatch
- International transaction flag
- Transaction hour
- Distance from home
- Previous chargebacks

### Workflow
1. Load and inspect data
2. Handle missing values
3. Explore fraud distribution and transaction patterns
4. Encode categorical variables
5. Standardize numerical variables
6. Split data using stratification
7. Train Logistic Regression and Random Forest
8. Evaluate accuracy, precision, recall, F1-score and ROC-AUC
9. Analyze the confusion matrix and ROC curve
10. Select the better model for fraud screening

### Final Model Result
The generated run selected **Logistic Regression** based on F1-score and ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6870 | 0.3036 | 0.6888 | 0.4214 | 0.7487 |
| Random Forest | 0.8035 | 0.3994 | 0.3716 | 0.3850 | 0.7356 |

### Important note
For fraud detection, accuracy alone can be misleading because fraudulent transactions are rare. Precision, recall, F1-score and ROC-AUC are therefore reported.

### Repository Structure
```text
Codec_Fraud_Detection_Project/
├── data/
│   └── financial_transactions.csv
├── notebooks/
│   └── fraud_detection.ipynb
├── outputs/
│   ├── 01_class_distribution.png
│   ├── 02_fraud_rate_by_type.png
│   ├── 03_confusion_matrix.png
│   ├── 04_roc_curves.png
│   └── model_results.csv
├── reports/
│   └── Fraud_Detection_Internship_Report.pdf
├── app.py
├── INTERNSHIP_LOGBOOK.md
├── fraud_detection.py
├── requirements.txt
└── README.md
```

### How to Run
```bash
pip install -r requirements.txt
python fraud_detection.py
```

Or upload the notebook and `data/financial_transactions.csv` to Google Colab and run all cells.

### Future Scope
- Real-time transaction scoring
- SMOTE/advanced imbalance handling
- Explainable AI with SHAP
- Threshold optimization based on financial cost
- Model drift monitoring
- API deployment for transaction screening


### Interactive Dashboard
Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard displays transaction KPIs, fraud distribution, fraud rate by transaction type, amount distribution, fraud activity by hour, and a high-risk transaction table.
