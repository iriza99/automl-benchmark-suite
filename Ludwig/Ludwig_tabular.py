from dataset_loader import (
    load_from_openml,
    load_from_csv_files,
    load_from_excel_files
)

import pandas as pd
import numpy as np
from ludwig.automl import auto_train
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

# ─── Ludwig ───────────────────────────────────────────────────────────

train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)


import ray

auto_train_results = auto_train(
    dataset=train_data,   
    target=label_column,       
    time_limit_s=300,     
    tune_for_memory=False  
)

# Evaluate the best model on the test data
evaluation_results = auto_train_results.best_model.evaluate(test_data)

# Extract the evaluation statistics
evaluation_statistics = evaluation_results[0] 

# Print all available metrics
print("All Evaluation Metrics:")
for metric_name, value in evaluation_statistics.items():
    print(f"{metric_name}: {value}")

ray.shutdown()