import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import joblib

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("asthma_disease_data.csv")

print("Original Dataset Shape:", data.shape)

print("Diagnosis distribution:")
print(data["Diagnosis"].value_counts())

# -----------------------------
# Keep only children (<18)
# -----------------------------
data = data[data["Age"] < 18]

print("Dataset after Age Filter:", data.shape)

print("Diagnosis distribution after filter:")
print(data["Diagnosis"].value_counts())

# -----------------------------
# Select features
# -----------------------------
X = data[
[
"Age",
"BMI",
"PollutionExposure",
"Wheezing",
"ShortnessOfBreath",
"Coughing",
"NighttimeSymptoms"
]
]

y = data["Diagnosis"]

# -----------------------------
# Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train SVM (balanced)
# -----------------------------
svm_model = SVC(probability=True, class_weight="balanced")

svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_accuracy)

# -----------------------------
# Train XGBoost (balanced)
# -----------------------------
xgb_model = XGBClassifier(scale_pos_weight=18)

xgb_model.fit(X_train, y_train)

xgb_pred = xgb_model.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_pred)

print("XGBoost Accuracy:", xgb_accuracy)

# -----------------------------
# Hybrid model
# -----------------------------
svm_prob = svm_model.predict_proba(X_test)[:,1]
xgb_prob = xgb_model.predict_proba(X_test)[:,1]

hybrid_prob = (svm_prob + xgb_prob) / 2

hybrid_pred = (hybrid_prob > 0.5).astype(int)

hybrid_accuracy = accuracy_score(y_test, hybrid_pred)

print("Hybrid Accuracy:", hybrid_accuracy)

# -----------------------------
# Save models
# -----------------------------
joblib.dump(svm_model,"svm_model.pkl")
joblib.dump(xgb_model,"xgb_model.pkl")

print("Models saved successfully.")