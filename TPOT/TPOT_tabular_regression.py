from dataset_loader import (
    load_from_openml,
    load_from_csv_files,
    load_from_excel_files
)

import pandas as pd
from tpot import TPOTRegressor
from sklearn.metrics import root_mean_squared_error

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

# 1) Load from OpenML (example: task_id=211690, Liver disorders)
X_train, y_train, X_test, y_test, label_column = load_from_openml(task_id=211690)
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

tpot = TPOTRegressor(
    max_time_mins=time_limit_mins,
    random_state=1,
    verbosity=2
)
tpot.fit(X_train, y_train)

y_pred = tpot.predict(X_test)

# Calculate RMSE
rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse:.4f}")