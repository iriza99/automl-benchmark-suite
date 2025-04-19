from dataset_loader import (
    load_from_openml,
    load_from_csv_files,
    load_from_excel_files
)

import numpy as np
import pandas as pd
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

# 1) Load from OpenML (example: task_id=267, Diabetes)
X_train, y_train, X_test, y_test, label_column = load_from_openml(task_id=267)
print(f"[OpenML] {label_column=}, X_train={X_train.shape}, X_test={X_test.shape}")

# 2) Or load from CSV
# X_train, y_train, X_test, y_test, label_column = load_from_csv_files(
#     train_path="Oesophageal_train.csv",
#     test_path="Oesophageal_test.csv",
#     label_column="GroundTruth_bi",
#     sep=';'
# )
# print(f"[CSV] {label_column=}, X_train={X_train.shape}, X_test={X_test.shape}")

# 3) Or load from Excel
# X_train, y_train, X_test, y_test, label_column = load_from_excel_files(
#     train_path="Hereditary_train.xlsx",
#     test_path="Hereditary_test.xlsx",
#     label_column="Diagnoses",
#     sheet_name=0
# )
# print(f"[Excel] {label_column=}, X_train={X_train.shape}, X_test={X_test.shape}")

# ─── TPOT ───────────────────────────────────────────────────────────

# Time limit for TPOT (in minutes)
time_limit_mins = 5

tpot = TPOTClassifier(
    max_time_mins=time_limit_mins,
    random_state=1,
    verbosity=2
)
tpot.fit(X_train, y_train)

y_pred  = tpot.predict(X_test)
y_proba = tpot.predict_proba(X_test)

# Determine number of classes
n_classes = np.unique(y_test).shape[0]

if n_classes == 2:
    # Binary classification → ROC AUC
    auc = roc_auc_score(y_test, y_proba[:, 1])
    print(f"ROC AUC: {auc:.4f}")
else:
    # Multiclass → Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
