import pandas as pd
import openml


def load_from_openml(task_id):
    """
    Load exact train/test partitions defined by an OpenML task.

    Returns:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        label (str): Name of the target column.

    Parameters:
        task_id (int): OpenML task identifier.
    """
    # Retrieve the task and dataset, always download splits and data
    task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()

    # Extract features and target
    X, y, _, _ = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)

    # Get the exact train/test indices
    train_idx, test_idx = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)

    # Partition and reset indices
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    label = dataset.default_target_attribute
    return X_train, y_train, X_test, y_test, label


def load_from_csv_files(train_path, test_path, label_column, sep=',', index_col=None):
    """
    Load training and testing sets from separate CSV files.

    Returns:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        label_column (str): Name of the target column.

    Parameters:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.
        label_column (str): Name of the target column.
        sep (str): Column separator (e.g., ',' or ';').
        index_col (str or int, optional): Column to use as the index.
    """
    # Read DataFrames
    train_df = pd.read_csv(train_path, sep=sep, index_col=index_col)
    test_df = pd.read_csv(test_path, sep=sep, index_col=index_col)

    # Separate X and y
    X_train = train_df.drop(columns=[label_column])
    y_train = train_df[label_column]
    X_test = test_df.drop(columns=[label_column])
    y_test = test_df[label_column]

    return X_train, y_train, X_test, y_test, label_column


def load_from_excel_files(train_path, test_path, label_column, sheet_name=0, index_col=None):
    """
    Load training and testing sets from separate Excel files (.xlsx).
    
    Returns:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        label_column (str): Name of the target column.

    Parameters:
        train_path (str): Path to the training Excel file.
        test_path (str): Path to the testing Excel file.
        label_column (str): Name of the target column.
        sheet_name (str or int): Sheet name or index.
        index_col (str or int): Column to use as the index.
    """
    # Read DataFrames
    train_df = pd.read_excel(train_path, sheet_name=sheet_name, index_col=index_col)
    test_df = pd.read_excel(test_path, sheet_name=sheet_name, index_col=index_col)

    # Separate X and y
    X_train = train_df.drop(columns=[label_column])
    y_train = train_df[label_column]
    X_test = test_df.drop(columns=[label_column])
    y_test = test_df[label_column]

    return X_train, y_train, X_test, y_test, label_column