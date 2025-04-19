from dataset_loader import (
    load_from_openml,
    load_from_csv_files,
    load_from_excel_files
)

# Load data from OpenML (example: task_id=267 (diabetes))
X_train, y_train, X_test, y_test, label_column = load_from_openml(task_id=267)
print(f"[OpenML] {label_column=}, {X_train.shape=}, {X_test.shape=}")

# Load data from separate CSV files
# X_train, y_train, X_test, y_test, label_column = load_from_csv_files(
#     train_path="Oesophageal_train.csv",
#     test_path="Oesophageal_test.csv",
#     label_column="GroundTruth_bi",
#     sep=';'
# )
# print(f"[CSV] {label_column=}, {X_train.shape=}, {X_test.shape=}")

# Load data from separate Excel files
# X_train, y_train, X_test, y_test, label_column = load_from_excel_files(
#     train_path="Hereditary_train.xlsx",
#     test_path="Hereditary_test.xlsx",
#     label_column="Diagnoses",
#     sheet_name=0
# )
# print(f"[Excel] {label_column=}, {X_train.shape=}, {X_test.shape=}")



import pandas as pd
from autogluon.tabular import TabularPredictor

# Combine features and label into a single DataFrame for AutoGluon
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Time limit for training (in seconds)
time_limit = 300

# Train AutoGluon model
predictor = TabularPredictor(label=label_column).fit(train_data, time_limit=time_limit)

# Evaluate on test data
performance = predictor.evaluate(test_data, silent=True)

# Print results
print("Performance metrics:\n")

# Display selected metrics if available
for metric in ['accuracy', 'roc_auc', 'rmse']:
    if metric in performance:
        print(f"{metric.upper()}: {performance[metric]:.4f}")
