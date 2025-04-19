from dataset_loader import (
    load_from_openml,
    load_from_csv_files,
    load_from_excel_files
)

import pandas as pd
import h2o
from h2o.automl import H2OAutoML

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

# 1) Load from OpenML (example: task_id=267, diabetes classification)
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

# ─── H2O AutoML ────────────────────────────────────────────────────────────

h2o.init()

# Combine pandas DataFrames into H2OFrames
hf_train = h2o.H2OFrame(pd.concat([X_train, y_train], axis=1))
hf_test  = h2o.H2OFrame(pd.concat([X_test,  y_test ], axis=1))

# Identify response and feature columns
response   = label_column
feature_cols = [c for c in hf_train.columns if c != response]

# Convert response column to factor for classification
hf_train[response] = hf_train[response].asfactor()
hf_test[response]  = hf_test[response].asfactor()

aml = H2OAutoML(
    max_runtime_secs=300,  # 5-minute time limit
    seed=1
)
aml.train(x=feature_cols, y=response, training_frame=hf_train)

# Retrieve the best model
leader = aml.leader

print(f"AUC: {leader.model_performance(hf_test).auc():.4f}")

h2o.shutdown(prompt=False)