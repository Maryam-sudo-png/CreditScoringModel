# Credit Scoring Model

**Task 1 — Machine Learning**

## Objective
Predict an individual's creditworthiness (Good / Bad credit risk) using past financial data.

## Dataset
**Statlog (German Credit Data)** — UCI Machine Learning Repository
- 1000 loan applicants, 20 features (financial + demographic)
- Target: Good Risk (70%) vs Bad Risk (30%)
- Source: `german.csv` (from UCI, mirrored on GitHub)

## Approach
1. Encoded categorical features (checking account status, credit history, purpose, etc.)
2. Scaled numeric features for Logistic Regression
3. Trained 3 classifiers: **Logistic Regression**, **Decision Tree**, **Random Forest**
4. Used `class_weight="balanced"` to handle class imbalance
5. Evaluated with Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.80** | 0.68 | 0.63 | 0.66 | **0.81** |
| Logistic Regression | 0.71 | 0.51 | 0.73 | 0.60 | 0.79 |
| Decision Tree | 0.68 | 0.48 | 0.65 | 0.55 | 0.66 |

**Best model: Random Forest**

### Top predictive features
1. Checking account status
2. Credit amount
3. Loan duration (months)
4. Age
5. Savings status

## Files
- `credit_scoring_model.py` — full training/evaluation pipeline
- `german.csv` — dataset
- `credit_scoring_results.png` — ROC curves + feature importance chart
- `model_comparison_results.csv` — metrics table
- `best_credit_model.pkl`, `scaler.pkl`, `label_encoders.pkl` — saved artifacts

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
python credit_scoring_model.py
```

## Tools
Python, Pandas, Scikit-learn, Matplotlib, Seaborn
