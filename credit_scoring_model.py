"""
Task 1: Credit Scoring Model
========================================================
Objective : Predict an individual's creditworthiness (good/bad credit risk)
            using past financial data.
Dataset   : Statlog (German Credit Data) - UCI Machine Learning Repository
            1000 applicants, 20 features, binary target (Good/Bad credit risk)
Models    : Logistic Regression, Decision Tree, Random Forest
Metrics   : Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

Author: Maryam
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

RANDOM_STATE = 42

# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
COLUMN_NAMES = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment_since",
    "installment_rate_pct", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents",
    "telephone", "foreign_worker", "target"
]

df = pd.read_csv("german.csv", header=None, names=COLUMN_NAMES)

# Target: 1 = Good credit risk, 2 = Bad credit risk -> convert to 0/1 (1 = bad/default)
df["target"] = df["target"].map({1: 0, 2: 1})  # 0 = Good, 1 = Bad

print("Dataset shape:", df.shape)
print("\nClass balance (0=Good, 1=Bad):")
print(df["target"].value_counts(normalize=True).round(3))

# -----------------------------------------------------------------------
# 2. FEATURE ENGINEERING / PREPROCESSING
# -----------------------------------------------------------------------
categorical_cols = df.select_dtypes(include="object").columns.tolist()
numeric_cols = [c for c in df.columns if c not in categorical_cols + ["target"]]

# Label-encode categorical (tree models handle this fine; for LR we scale numerics)
encoders = {}
df_enc = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col])
    encoders[col] = le

X = df_enc.drop(columns=["target"])
y = df_enc["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------------------------
# 3. TRAIN MODELS
# -----------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
}

results = []
roc_curves = {}
fitted_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    fitted_models[name] = model
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)

    print(f"\n{'='*55}\n{name}\n{'='*55}")
    print(classification_report(y_test, y_pred, target_names=["Good Risk", "Bad Risk"]))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
print("\n\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison_results.csv", index=False)

# -----------------------------------------------------------------------
# 4. FEATURE IMPORTANCE (Random Forest)
# -----------------------------------------------------------------------
rf_model = fitted_models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 important features (Random Forest):")
print(importances.head(10))

# -----------------------------------------------------------------------
# 5. PLOTS
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ROC curves
for name, (fpr, tpr, auc) in roc_curves.items():
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves - Credit Scoring Models")
axes[0].legend(loc="lower right")

# Feature importance
importances.head(10).sort_values().plot(kind="barh", ax=axes[1], color="#2f5f8a")
axes[1].set_title("Top 10 Feature Importances (Random Forest)")
axes[1].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("credit_scoring_results.png", dpi=150)
print("\nSaved plot -> credit_scoring_results.png")
print("Saved results table -> model_comparison_results.csv")

# -----------------------------------------------------------------------
# 6. SAVE BEST MODEL
# -----------------------------------------------------------------------
import joblib
best_name = results_df.iloc[0]["Model"]
joblib.dump(fitted_models[best_name], "best_credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "label_encoders.pkl")
print(f"\nBest model ({best_name}) saved -> best_credit_model.pkl")
