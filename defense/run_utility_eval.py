import xgboost as xgb
import pandas as pd
import numpy as np
import os
import glob
import json
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Any, Optional
import matplotlib.pyplot as plt
from xgboost.callback import TrainingCallback

def _base_params(use_gpu: bool):
    p = {
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 3.0,
        "tree_method": "hist",
    }
    if use_gpu:
        # Works on XGBoost 2.x; older versions will just ignore it
        p.update({"device": "cuda"})
        # Back-compat for 1.x (safe even if unused)
        p.setdefault("tree_method", "gpu_hist")
        p.setdefault("predictor", "gpu_predictor")
    return p

class TqdmCallback(TrainingCallback):
    def __init__(self, total_rounds: int, patience: int, eval_name: str = "valid"):
        self.total_rounds = int(total_rounds)
        self.patience = int(patience) if patience is not None else None
        self.eval_name = eval_name
        self.pbar = None
        self.metric_name = None
        self.maximize = None
        self.best = None
        self.best_iter = -1
        self.since_improve = 0

    def _is_improved(self, curr):
        if self.best is None:
            return True
        return (curr > self.best) if self.maximize else (curr < self.best)

    def before_training(self, model):
        self.pbar = tqdm(total=self.total_rounds, desc="Training XGBoost", ncols=110)
        return model

    def after_iteration(self, model, epoch: int, evals_log: Any):
        if self.pbar is not None:
            self.pbar.update(1)

        # Read the current metric directly from evals_log (works on 1.x and 2.x)
        curr_metric = None
        metric_name = self.metric_name
        try:
            if isinstance(evals_log, dict) and self.eval_name in evals_log:
                eval_dict = evals_log[self.eval_name]  # e.g. {'rmse': [..]} or {'auc': [..]}
                if isinstance(eval_dict, dict) and len(eval_dict) > 0:
                    metric_name = next(iter(eval_dict.keys()))
                    hist = eval_dict[metric_name]
                    if isinstance(hist, list) and hist:
                        curr_metric = float(hist[-1])
        except Exception:
            pass

        if curr_metric is None:
            return False  # keep going even if we couldn't parse this round

        if self.metric_name is None:
            self.metric_name = metric_name
            self.maximize = any(k in metric_name.lower() for k in ["auc", "map", "ndcg"])

        if self._is_improved(curr_metric):
            self.best = curr_metric
            self.best_iter = epoch
            self.since_improve = 0
        else:
            self.since_improve += 1

        projected_stop = (
            self.best_iter + self.patience if (self.patience is not None and self.best_iter >= 0) else None
        )

        if self.pbar is not None:
            postfix = {
                f"{self.eval_name}-{self.metric_name}": f"{curr_metric:.5f}",
                "best": f"{self.best:.5f}" if (self.best is not None) else "—",
                "since↑": self.since_improve,
            }
            if projected_stop is not None:
                postfix["stop@"] = projected_stop
            self.pbar.set_postfix(postfix, refresh=True)

        return False  # let XGBoost's own early stopping stop training

    def after_training(self, model):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        return model

# ---------------------- REGRESSION ----------------------
def train_xgb_regression(
    X, y,
    num_boost_round=2000,
    use_gpu=True,
    val_size=0.2,
    random_state=42,
    early_stopping_rounds=200,
    verbose_eval=False,
    eval_name="valid"
):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, shuffle=True
    )
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    params = _base_params(use_gpu)
    params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, eval_name)],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,  # tqdm handles display
        callbacks=[TqdmCallback(num_boost_round, early_stopping_rounds, eval_name=eval_name)]
    )

    print(f"\nBest iteration: {bst.best_iteration}")
    # print(f"Best {eval_name} score: {bst.attributes().get('best_score', 'N/A')}")
    return bst

# ---------------------- CLASSIFICATION ----------------------
def train_xgb_classification(
    X, y,
    num_boost_round=2000,
    use_gpu=True,
    val_size=0.2,
    random_state=42,
    early_stopping_rounds=200,
    verbose_eval=False,
    eval_name="valid"
):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dvalid = xgb.DMatrix(X_val, label=y_val)

    params = _base_params(use_gpu)
    num_classes = len(np.unique(y))
    if num_classes == 2:
        params.update({"objective": "binary:logistic", "eval_metric": "logloss"})  # add "auc" too if desired
    else:
        params.update({"objective": "multi:softprob", "num_class": num_classes, "eval_metric": "mlogloss"})

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, eval_name)],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,  # tqdm handles display
        callbacks=[TqdmCallback(num_boost_round, early_stopping_rounds, eval_name=eval_name)]
    )

    print(f"\nBest iteration: {bst.best_iteration}")
    # print(f"Best {eval_name} score: {bst.attributes().get('best_score', 'N/A')}")
    return bst



# def utility_measure_debug(base_dir):
#     """
#     Debugging function to check for overfitting by training XGBoost on train.csv
#     and testing on test.csv, printing both training and testing RMSE.
    
#     Args:
#         base_dir: Base directory path containing subdirectories and config JSON
#     """
#     # Load configuration from JSON file
#     config_path = os.path.join(base_dir, "config.json")
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
#     with open(config_path, 'r') as f:
#         config = json.load(f)
    
#     target_col = config.get("target_column")
#     if not target_col:
#         raise ValueError("target_column not specified in config.json")
    
#     task_type = config.get("task_type", "regression")  # default to regression
    
#     # Find subdirectories starting with "train_test_synth"
#     pattern = os.path.join(base_dir, "train_test_synth*")
#     subdirs = glob.glob(pattern)
#     subdirs = [d for d in subdirs if os.path.isdir(d)]
    
#     if not subdirs:
#         raise FileNotFoundError(f"No subdirectories starting with 'train_test_synth' found in {base_dir}")
    
#     for subdir in [subdirs[0]]:
#         subdir_name = os.path.basename(subdir)
#         print(f"\n=== Processing {subdir_name} ===")
        
#         # Find train.csv and test.csv
#         train_path = os.path.join(subdir, "train.csv")
#         test_path = os.path.join(subdir, "test.csv")
        
#         if not os.path.exists(train_path):
#             print(f"Warning: train.csv not found in {subdir}, skipping...")
#             continue
            
#         if not os.path.exists(test_path):
#             print(f"Warning: test.csv not found in {subdir}, skipping...")
#             continue
        
#         # Load train and test data
#         train_df = pd.read_csv(train_path)
#         test_df = pd.read_csv(test_path)
        
#         if target_col not in train_df.columns:
#             print(f"Warning: target column '{target_col}' not found in train.csv, skipping...")
#             continue
            
#         if target_col not in test_df.columns:
#             print(f"Warning: target column '{target_col}' not found in test.csv, skipping...")
#             continue
        
#         # Prepare training data
#         X_train = train_df.drop(columns=[target_col])
#         y_train = train_df[target_col]
        
#         # Prepare test data
#         X_test = test_df.drop(columns=[target_col])
#         y_test = test_df[target_col]
        
#         # Align features (use common features only)
#         common_features = list(set(X_train.columns) & set(X_test.columns))
#         X_train_aligned = X_train[common_features]
#         X_test_aligned = X_test[common_features]
        
#         print(f"Training samples: {len(X_train_aligned)}")
#         print(f"Test samples: {len(X_test_aligned)}")
#         print(f"Features: {len(common_features)}")
        
#         try:
#             if task_type == "regression":
#                 # Train XGBoost regression model
#                 model = train_xgb_regression(X_train_aligned, y_train, use_gpu=True)
                
#                 # Make predictions on training data
#                 dtrain = xgb.DMatrix(X_train_aligned)
#                 y_train_pred = model.predict(dtrain)
#                 train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                
#                 # Make predictions on test data
#                 dtest = xgb.DMatrix(X_test_aligned)
#                 y_test_pred = model.predict(dtest)
#                 test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                
#                 print(f"Training RMSE: {train_rmse:.4f}")
#                 print(f"Testing RMSE: {test_rmse:.4f}")
#                 print(f"Overfitting ratio (test/train): {test_rmse/train_rmse:.2f}")
                
#             else:
#                 print(f"Classification task detected, but debug function is set up for regression only")
                
#         except Exception as e:
#             print(f"Error processing {subdir_name}: {str(e)}")
#             continue


def utility_measure(base_dir):
    """
    Utility measure function that finds subdirectories starting with "train_test_synth",
    trains XGBoost models on CSV files, and evaluates them on test.csv.

    For synth_auc_*.csv files, extracts the number of rows, trains on that data,
    then compares with vanilla_auc_*.csv (same number of samples) and rest.csv (same number of samples).

    Args:
        base_dir: Base directory path containing subdirectories and config JSON

    Returns:
        pd.DataFrame: Results with columns for subdirectory, dataset, and metrics
    """
    results = []

    # Load configuration from JSON file
    config_path = os.path.join(base_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    target_col = config.get("target_column")
    if not target_col:
        raise ValueError("target_column not specified in config.json")

    task_type = config.get("task_type", "regression")  # default to regression

    # Find subdirectories starting with "train_test_synth"
    pattern = os.path.join(base_dir, "train_test_synth*")
    subdirs = glob.glob(pattern)
    subdirs = [d for d in subdirs if os.path.isdir(d)]

    if not subdirs:
        raise FileNotFoundError(f"No subdirectories starting with 'train_test_synth' found in {base_dir}")

    # Separate result collections for AUC and TPR
    auc_results = []
    tpr_results = []

    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        subdir_name = os.path.basename(subdir)

        # Find test.csv
        test_path = os.path.join(subdir, "test.csv")
        if not os.path.exists(test_path):
            print(f"Warning: test.csv not found in {subdir}, skipping...")
            continue

        # Load test data
        test_df = pd.read_csv(test_path)
        if target_col not in test_df.columns:
            print(f"Warning: target column '{target_col}' not found in test.csv of {subdir}, skipping...")
            continue

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Handle categorical target for classification
        le = None
        if task_type == "classification":
            le = LabelEncoder()
            y_test_encoded = le.fit_transform(y_test)
        else:
            y_test_encoded = y_test

        # Process synth_auc_*.csv and synth_tpr_*.csv files specifically
        synth_auc_files = []
        synth_tpr_files = []
        vanilla_auc_files = []
        vanilla_tpr_files = []
        rest_file = None
        other_files = []

        csv_files = glob.glob(os.path.join(subdir, "*.csv"))
        csv_files = [f for f in csv_files if os.path.basename(f) != "test.csv"]

        for csv_file in csv_files:
            basename = os.path.basename(csv_file)
            if basename.startswith("synth_auc_") and basename.endswith(".csv"):
                synth_auc_files.append(csv_file)
            elif basename.startswith("synth_tpr_") and basename.endswith(".csv"):
                synth_tpr_files.append(csv_file)
            elif basename.startswith("vanilla_auc_") and basename.endswith(".csv"):
                vanilla_auc_files.append(csv_file)
            elif basename.startswith("vanilla_tpr_") and basename.endswith(".csv"):
                vanilla_tpr_files.append(csv_file)
            elif basename == "rest.csv":
                rest_file = csv_file
            else:
                other_files.append(csv_file)

        # Process synth_auc files with comparative analysis
        for synth_file in synth_auc_files:
            basename = os.path.basename(synth_file)
            # Extract number from synth_auc_XXXX.csv
            import re
            match = re.search(r'synth_auc_(\d+)\.csv', basename)
            if not match:
                print(f"Warning: Could not extract number from {basename}, skipping...")
                continue

            sample_size = int(match.group(1))
            print(f"\nProcessing synth_auc file with {sample_size} samples: {basename}")

            # Process the synth_auc file
            synth_result = process_single_dataset(synth_file, "synth_auc_" + str(sample_size),
                                                subdir_name, target_col, task_type, le,
                                                sample_size, X_test, y_test_encoded)
            if synth_result:
                synth_result["data_type"] = "synthetic"
                if "auc" in subdir_name.lower():
                    auc_results.append(synth_result)
                elif "tpr" in subdir_name.lower():
                    tpr_results.append(synth_result)
                else:
                    results.append(synth_result)

            # Use the single vanilla file
            if vanilla_auc_files:
                print(f"Processing corresponding vanilla file: {os.path.basename(vanilla_auc_files[0])}")
                vanilla_result = process_single_dataset(vanilla_auc_files[0], "vanilla_auc_" + str(sample_size),
                                                      subdir_name, target_col, task_type, le,
                                                      sample_size, X_test, y_test_encoded)
                if vanilla_result:
                    vanilla_result["data_type"] = "vanilla"
                    if "auc" in subdir_name.lower():
                        auc_results.append(vanilla_result)
                    elif "tpr" in subdir_name.lower():
                        tpr_results.append(vanilla_result)
                    else:
                        results.append(vanilla_result)
            else:
                print(f"Warning: No corresponding vanilla file found for sample size {sample_size}")

            # Process rest.csv with same sample size
            if rest_file:
                print(f"Processing rest.csv with {sample_size} samples")
                rest_result = process_single_dataset(rest_file, "rest_" + str(sample_size),
                                                   subdir_name, target_col, task_type, le,
                                                   sample_size, X_test, y_test_encoded)
                if rest_result:
                    rest_result["data_type"] = "rest"
                    if "auc" in subdir_name.lower():
                        auc_results.append(rest_result)
                    elif "tpr" in subdir_name.lower():
                        tpr_results.append(rest_result)
                    else:
                        results.append(rest_result)
            else:
                print(f"Warning: rest.csv not found in {subdir}")

        # Process synth_tpr files with comparative analysis
        for synth_file in synth_tpr_files:
            basename = os.path.basename(synth_file)
            # Extract number from synth_tpr_XXXX.csv
            import re
            match = re.search(r'synth_tpr_(\d+)\.csv', basename)
            if not match:
                print(f"Warning: Could not extract number from {basename}, skipping...")
                continue

            sample_size = int(match.group(1))
            print(f"\nProcessing synth_tpr file with {sample_size} samples: {basename}")

            # Process the synth_tpr file
            synth_result = process_single_dataset(synth_file, "synth_tpr_" + str(sample_size),
                                                subdir_name, target_col, task_type, le,
                                                sample_size, X_test, y_test_encoded)
            if synth_result:
                synth_result["data_type"] = "synthetic"
                if "auc" in subdir_name.lower():
                    auc_results.append(synth_result)
                elif "tpr" in subdir_name.lower():
                    tpr_results.append(synth_result)
                else:
                    results.append(synth_result)

            # Use the single vanilla file
            if vanilla_tpr_files:
                print(f"Processing corresponding vanilla file: {os.path.basename(vanilla_tpr_files[0])}")
                vanilla_result = process_single_dataset(vanilla_tpr_files[0], "vanilla_tpr_" + str(sample_size),
                                                      subdir_name, target_col, task_type, le,
                                                      sample_size, X_test, y_test_encoded)
                if vanilla_result:
                    vanilla_result["data_type"] = "vanilla"
                    if "auc" in subdir_name.lower():
                        auc_results.append(vanilla_result)
                    elif "tpr" in subdir_name.lower():
                        tpr_results.append(vanilla_result)
                    else:
                        results.append(vanilla_result)
            else:
                print(f"Warning: No corresponding vanilla file found for sample size {sample_size}")

            # Process rest.csv with same sample size
            if rest_file:
                print(f"Processing rest.csv with {sample_size} samples")
                rest_result = process_single_dataset(rest_file, "rest_" + str(sample_size),
                                                   subdir_name, target_col, task_type, le,
                                                   sample_size, X_test, y_test_encoded)
                if rest_result:
                    rest_result["data_type"] = "rest"
                    if "auc" in subdir_name.lower():
                        auc_results.append(rest_result)
                    elif "tpr" in subdir_name.lower():
                        tpr_results.append(rest_result)
                    else:
                        results.append(rest_result)
            else:
                print(f"Warning: rest.csv not found in {subdir}")

        # Process other files (train.csv, etc.) normally
        # for csv_file in other_files:
        #     if os.path.basename(csv_file) == "train.csv":
        #         dataset_name = "train"
        #         result = process_single_dataset(csv_file, dataset_name, subdir_name,
        #                                       target_col, task_type, le, None, X_test, y_test_encoded)
        #         if result:
        #             result["data_type"] = "real"
        #             if "auc" in subdir_name.lower():
        #                 auc_results.append(result)
        #             elif "tpr" in subdir_name.lower():
        #                 tpr_results.append(result)
        #             else:
        #                 results.append(result)

    # Check if we have any valid results
    if not auc_results and not tpr_results and not results:
        raise ValueError("No valid results obtained from any dataset")

    # Create utility_summary directory
    summary_dir = os.path.join(base_dir, "utility_summary")
    os.makedirs(summary_dir, exist_ok=True)

    # Process AUC results if available
    if auc_results:
        auc_df = pd.DataFrame(auc_results)
        auc_output_path = os.path.join(summary_dir, "utility_auc.csv")
        auc_df.to_csv(auc_output_path, index=False)
        print(f"AUC results saved to: {auc_output_path}")
        # create_utility_visualization(auc_df, base_dir, task_type, "auc")

    # Process TPR results if available
    if tpr_results:
        tpr_df = pd.DataFrame(tpr_results)
        tpr_output_path = os.path.join(summary_dir, "utility_tpr.csv")
        tpr_df.to_csv(tpr_output_path, index=False)
        print(f"TPR results saved to: {tpr_output_path}")
        # create_utility_visualization(tpr_df, base_dir, task_type, "tpr")

    # Handle other results if any (fallback)
    # if results:
    #     other_df = pd.DataFrame(results)
    #     other_output_path = os.path.join(summary_dir, "utility_results.csv")
    #     other_df.to_csv(other_output_path, index=False)
    #     print(f"Other results saved to: {other_output_path}")
    #     create_utility_visualization(other_df, base_dir, task_type, "utility")

    # Return combined results for backward compatibility
    all_results = auc_results + tpr_results + results
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


def process_single_dataset(csv_file, dataset_name, subdir_name, target_col, task_type, le, sample_size, X_test, y_test_encoded):
    """
    Process a single dataset file and return evaluation results.

    Args:
        csv_file: Path to the CSV file
        dataset_name: Name for the dataset
        subdir_name: Name of the subdirectory
        target_col: Target column name
        task_type: "regression" or "classification"
        le: Label encoder (for classification)
        sample_size: Number of samples to use (None for all)
        X_test: Test features
        y_test_encoded: Encoded test targets

    Returns:
        dict: Result dictionary with metrics, or None if processing failed
    """
    try:
        # Load training data
        train_df = pd.read_csv(csv_file)
        if target_col not in train_df.columns:
            print(f"Warning: target column '{target_col}' not found in {dataset_name}, skipping...")
            return None

        # Sample if needed
        if sample_size is not None and len(train_df) > sample_size:
            train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        elif sample_size is not None and len(train_df) < sample_size:
            print(f"Warning: {dataset_name} has {len(train_df)} samples, fewer than requested {sample_size}. Using all available samples.")

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        # Ensure feature alignment
        common_features = X_train.columns.intersection(X_test.columns)
        if len(common_features) == 0:
            print(f"Warning: No common features between train and test for {dataset_name}, skipping...")
            return None

        X_train = X_train[common_features]
        X_test_aligned = X_test[common_features]

        # Handle categorical target for classification
        if task_type == "classification":
            y_train_encoded = le.transform(y_train)
        else:
            y_train_encoded = y_train

        # Train model
        if task_type == "classification":
            model = train_xgb_classification(X_train, y_train_encoded, use_gpu=True)
        else:
            model = train_xgb_regression(X_train, y_train_encoded, use_gpu=True)

        # Make predictions
        dtest = xgb.DMatrix(X_test_aligned)
        if task_type == "classification":
            y_pred_prob = model.predict(dtest)
            if len(np.unique(y_test_encoded)) == 2:  # binary classification
                y_pred = (y_pred_prob > 0.5).astype(int)
            else:  # multiclass
                y_pred = np.argmax(y_pred_prob.reshape(len(y_test_encoded), -1), axis=1)
        else:
            y_pred = model.predict(dtest)

        # Calculate metrics
        result = {
            "subdirectory": subdir_name,
            "dataset": dataset_name,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test_aligned),
            "n_features": len(common_features)
        }

        if task_type == "classification":
            result.update({
                "accuracy": accuracy_score(y_test_encoded, y_pred),
                "f1_score": f1_score(y_test_encoded, y_pred, average='weighted')
            })
        else:
            result.update({
                "rmse": np.sqrt(mean_squared_error(y_test_encoded, y_pred)),
                "r2_score": r2_score(y_test_encoded, y_pred)
            })

        return result

    except Exception as e:
        print(f"Error processing {dataset_name} in {subdir_name}: {str(e)}")
        return None


def create_visualization(base_dir):
    """
    Create bar plots for utility metrics from CSV files in utility_summary subdirectory.

    Args:
        base_dir (str): Base directory containing utility_summary subdirectory
    """
    # Define paths
    utility_summary_dir = os.path.join(base_dir, 'utility_summary')
    visualization_dir = os.path.join(base_dir, 'visualization')

    # Create visualization directory if it doesn't exist
    os.makedirs(visualization_dir, exist_ok=True)

    # File paths
    auc_file = os.path.join(utility_summary_dir, 'utility_auc.csv')
    tpr_file = os.path.join(utility_summary_dir, 'utility_tpr.csv')

    # Process each CSV file
    for csv_file, output_name in [
        (auc_file, 'utility_auc_plot.png'),
        (tpr_file, 'utility_tpr_plot.png')
    ]:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping...")
            continue

        # Read CSV
        df = pd.read_csv(csv_file)

        # Replace 'rest' with 'REAL-DATA', 'synthetic' with 'TLP', and 'vanilla' with 'VANILLA'
        df['data_type'] = df['data_type'].replace({'rest': 'REAL-DATA', 'synthetic': 'TLP', 'vanilla': 'VANILLA'})

        # Sort by n_train_samples to ensure left-to-right ordering
        df = df.sort_values('n_train_samples')

        # Determine which metric to plot
        if 'rmse' in df.columns:
            metric_col = 'rmse'
            metric_label = 'RMSE'
        elif 'accuracy' in df.columns:
            metric_col = 'accuracy'
            metric_label = 'Accuracy'
        else:
            print(f"Warning: Neither 'rmse' nor 'accuracy' column found in {csv_file}, skipping...")
            continue

        # Get unique values for grouping
        train_samples = sorted(df['n_train_samples'].unique())
        data_types = ['REAL-DATA', 'VANILLA', 'TLP']  # Specific order

        # Set up the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Calculate bar positions
        x = np.arange(len(train_samples))  # Group positions
        width = 0.25  # Width of each bar

        # Colors for each data type - better color scheme
        colors = {'REAL-DATA': "#E70C0C", 'VANILLA': "#1721DD", 'TLP': "#DDDD27"}

        # Create bars for each data type
        for i, data_type in enumerate(data_types):
            # Get metric values for this data type across all train samples
            values = []
            for train_sample in train_samples:
                subset = df[(df['n_train_samples'] == train_sample) &
                           (df['data_type'] == data_type)]
                if not subset.empty:
                    values.append(subset[metric_col].iloc[0])
                else:
                    values.append(0)  # Handle missing data

            # Calculate bar positions for this data type
            bar_positions = x + (i - 1) * width

            # Create bars
            bars = ax.bar(bar_positions, values, width,
                         label=data_type, color=colors[data_type], alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # Customize the plot
        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel(metric_label)
        # Remove title
        ax.set_xticks(x)
        ax.set_xticklabels(train_samples)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Set y-axis limits for better visualization
        ax.set_ylim(0, max(df[metric_col]) * 1.1)

        # Save the plot
        output_path = os.path.join(visualization_dir, output_name)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {output_name} to {visualization_dir}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python utility_measure.py <base_dir>")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    
    try:
        results = utility_measure(base_dir)
        print("\nUtility measurement completed successfully!")
        print(f"Results shape: {results.shape}")
        print("\nFirst few rows:")
        print(results.head())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)




    try:
        create_visualization(base_dir)
    except Exception as e:
        print(f"Error during visualization: {e}")
     