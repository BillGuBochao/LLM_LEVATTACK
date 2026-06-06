import xgboost as xgb
import pandas as pd
import numpy as np
import os
import re
import glob
import json
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm
from typing import Any, Optional, List, Tuple
import matplotlib.pyplot as plt
from xgboost.callback import TrainingCallback


def _base_params(use_gpu: bool):
    p = {
        "eta": 0.05,
        "max_depth": 8,
        "subsample": 1.0,
        "colsample_bytree": 0.8,  # adjusted per-dataset in training functions
        "lambda": 3.0,
        "tree_method": "hist",
    }
    if use_gpu:
        p.update({"device": "cuda"})
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

        curr_metric = None
        metric_name = self.metric_name
        try:
            if isinstance(evals_log, dict) and self.eval_name in evals_log:
                eval_dict = evals_log[self.eval_name]
                if isinstance(eval_dict, dict) and len(eval_dict) > 0:
                    metric_name = next(iter(eval_dict.keys()))
                    hist = eval_dict[metric_name]
                    if isinstance(hist, list) and hist:
                        curr_metric = float(hist[-1])
        except Exception:
            pass

        if curr_metric is None:
            return False

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

        return False

    def after_training(self, model):
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
        return model


# ---------------------- ENCODING ----------------------
def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Optional[OneHotEncoder], List[str]]:
    """
    Detect categorical columns and one-hot encode them.
    Numerical columns are passed through as-is.

    The encoder is fit on the union of train and test categories
    to avoid unknown-category errors at test time.

    Args:
        X_train: Training feature DataFrame
        X_test: Test feature DataFrame

    Returns:
        Tuple of (X_train_encoded, X_test_encoded, encoder_or_None, feature_names)
    """
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

    if not categorical_cols:
        feature_names = numerical_cols
        return X_train[numerical_cols].values, X_test[numerical_cols].values, None, feature_names

    print(f"  Encoding {len(categorical_cols)} categorical columns: {categorical_cols}")

    # Fit encoder on combined data so all categories are known
    combined_cat = pd.concat([X_train[categorical_cols], X_test[categorical_cols]], axis=0)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(combined_cat)

    train_cat_encoded = encoder.transform(X_train[categorical_cols])
    test_cat_encoded = encoder.transform(X_test[categorical_cols])

    ohe_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()

    if numerical_cols:
        X_train_encoded = np.hstack([X_train[numerical_cols].values, train_cat_encoded])
        X_test_encoded = np.hstack([X_test[numerical_cols].values, test_cat_encoded])
        feature_names = numerical_cols + ohe_feature_names
    else:
        X_train_encoded = train_cat_encoded
        X_test_encoded = test_cat_encoded
        feature_names = ohe_feature_names

    print(f"  Features after encoding: {len(feature_names)} (was {len(categorical_cols) + len(numerical_cols)})")
    return X_train_encoded, X_test_encoded, encoder, feature_names


def _adaptive_colsample(n_features: int) -> float:
    """Pick colsample_bytree based on the number of features.

    - <= 10 features : 1.0  (use all -- too few to subsample)
    - 11-50 features : 0.8
    - > 50 features  : 0.6
    """
    if n_features <= 10:
        return 1.0
    elif n_features <= 50:
        return 0.8
    else:
        return 0.6


# ---------------------- REGRESSION ----------------------
def train_xgb_regression(
    X, y,
    num_boost_round=120,
    use_gpu=True,
    verbose_eval=False,
    eval_name="train"
):
    """Train XGBoost regression on ALL provided data (no validation split).

    Uses a fixed number of boosting rounds since synthetic datasets are
    typically small (8-128 rows) and a validation split would waste data.
    colsample_bytree is set adaptively based on the number of features.
    """
    dtrain = xgb.DMatrix(X, label=y)

    params = _base_params(use_gpu)
    params["colsample_bytree"] = _adaptive_colsample(X.shape[1])
    params.update({"objective": "reg:squarederror", "eval_metric": "rmse"})

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, eval_name)],
        verbose_eval=verbose_eval,
        callbacks=[TqdmCallback(num_boost_round, patience=None, eval_name=eval_name)]
    )

    return bst


# ---------------------- CLASSIFICATION ----------------------
def train_xgb_classification(
    X, y,
    num_boost_round=120,
    use_gpu=True,
    verbose_eval=False,
    eval_name="train",
    num_classes=None,
):
    """Train XGBoost classification on ALL provided data (no validation split).

    Uses a fixed number of boosting rounds since synthetic datasets are
    typically small (8-128 rows) and a validation split would waste data.
    colsample_bytree is set adaptively based on the number of features.

    Args:
        num_classes: Total number of classes. Pass explicitly when small
                     training splits may not contain all classes.
    """
    dtrain = xgb.DMatrix(X, label=y)

    params = _base_params(use_gpu)
    params["colsample_bytree"] = _adaptive_colsample(X.shape[1])
    if num_classes is None:
        num_classes = len(np.unique(y))
    if num_classes == 2 and int(y.max()) <= 1:
        params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
    else:
        params.update({"objective": "multi:softprob", "num_class": num_classes, "eval_metric": "mlogloss"})

    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, eval_name)],
        verbose_eval=verbose_eval,
        callbacks=[TqdmCallback(num_boost_round, patience=None, eval_name=eval_name)]
    )

    return bst


# ---------------------- SINGLE DATASET ----------------------
def process_single_dataset(csv_file, dataset_name, subdir_name, target_col, task_type, le, sample_size, X_test, y_test_encoded):
    """
    Process a single dataset file with automatic one-hot encoding of categorical features.

    Args:
        csv_file: Path to the CSV file
        dataset_name: Name for the dataset
        subdir_name: Name of the subdirectory
        target_col: Target column name
        task_type: "regression" or "classification"
        le: Label encoder (for classification)
        sample_size: Number of samples to use (None for all)
        X_test: Test features (raw DataFrame, before encoding)
        y_test_encoded: Encoded test targets

    Returns:
        dict: Result dictionary with metrics, or None if processing failed
    """
    try:
        train_df = pd.read_csv(csv_file)
        if target_col not in train_df.columns:
            print(f"Warning: target column '{target_col}' not found in {dataset_name}, skipping...")
            return None

        if sample_size is not None and len(train_df) > sample_size:
            train_df = train_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        elif sample_size is not None and len(train_df) < sample_size:
            print(f"Warning: {dataset_name} has {len(train_df)} samples, fewer than requested {sample_size}. Using all available samples.")

        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        # Align on common columns first
        common_features = X_train.columns.intersection(X_test.columns)
        if len(common_features) == 0:
            print(f"Warning: No common features between train and test for {dataset_name}, skipping...")
            return None

        X_train = X_train[common_features]
        X_test_aligned = X_test[common_features]

        # One-hot encode categorical features
        X_train_enc, X_test_enc, encoder, feature_names = encode_features(X_train, X_test_aligned)

        # Handle categorical target for classification
        if task_type == "classification":
            y_train_encoded = le.transform(y_train)
        else:
            y_train_encoded = y_train

        # Train model
        if task_type == "classification":
            model = train_xgb_classification(X_train_enc, y_train_encoded, use_gpu=True)
        else:
            model = train_xgb_regression(X_train_enc, y_train_encoded, use_gpu=True)

        # Make predictions
        dtest = xgb.DMatrix(X_test_enc)
        if task_type == "classification":
            y_pred_prob = model.predict(dtest)
            if len(np.unique(y_test_encoded)) == 2:
                y_pred = (y_pred_prob > 0.5).astype(int)
            else:
                y_pred = np.argmax(y_pred_prob.reshape(len(y_test_encoded), -1), axis=1)
        else:
            y_pred = model.predict(dtest)

        # Calculate metrics
        result = {
            "subdirectory": subdir_name,
            "dataset": dataset_name,
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test_aligned),
            "n_features_raw": len(common_features),
            "n_features_encoded": len(feature_names)
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


# ---------------------- MAIN UTILITY MEASURE ----------------------
def utility_measure(base_dir):
    """
    Utility measure function that handles datasets with categorical features
    via OneHotEncoder. Finds subdirectories starting with "train_test_synth",
    trains XGBoost models on CSV files, and evaluates them on test.csv.

    Args:
        base_dir: Base directory path containing subdirectories and config JSON

    Returns:
        pd.DataFrame: Results with columns for subdirectory, dataset, and metrics
    """
    results = []

    config_path = os.path.join(base_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    target_col = config.get("target_column")
    if not target_col:
        raise ValueError("target_column not specified in config.json")

    task_type = config.get("task_type", "regression")

    pattern = os.path.join(base_dir, "train_test_synth*")
    subdirs = glob.glob(pattern)
    subdirs = [d for d in subdirs if os.path.isdir(d)]

    if not subdirs:
        raise FileNotFoundError(f"No subdirectories starting with 'train_test_synth' found in {base_dir}")

    auc_results = []
    tpr_results = []

    for subdir in tqdm(subdirs, desc="Processing subdirectories"):
        subdir_name = os.path.basename(subdir)

        test_path = os.path.join(subdir, "test.csv")
        if not os.path.exists(test_path):
            print(f"Warning: test.csv not found in {subdir}, skipping...")
            continue

        test_df = pd.read_csv(test_path)
        if target_col not in test_df.columns:
            print(f"Warning: target column '{target_col}' not found in test.csv of {subdir}, skipping...")
            continue

        # Keep X_test as a DataFrame (encoding happens inside process_single_dataset)
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        le = None
        if task_type == "classification":
            le = LabelEncoder()
            y_test_encoded = le.fit_transform(y_test)
        else:
            y_test_encoded = y_test

        # Categorize CSV files
        synth_auc_files = []
        synth_tpr_files = []
        vanilla_auc_files = []
        vanilla_tpr_files = []
        rest_file = None

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

        def _route_result(result, data_type, subdir_name):
            """Append result to the correct list based on subdirectory name."""
            if result is None:
                return
            result["data_type"] = data_type
            if "auc" in subdir_name.lower():
                auc_results.append(result)
            elif "tpr" in subdir_name.lower():
                tpr_results.append(result)
            else:
                results.append(result)

        def _process_synth_group(synth_files, prefix, vanilla_files):
            """Process a group of synth files with their vanilla and rest counterparts."""
            for synth_file in synth_files:
                basename = os.path.basename(synth_file)
                match = re.search(rf'{prefix}_(\d+)\.csv', basename)
                if not match:
                    print(f"Warning: Could not extract number from {basename}, skipping...")
                    continue

                sample_size = int(match.group(1))
                print(f"\nProcessing {prefix} file with {sample_size} samples: {basename}")

                synth_result = process_single_dataset(synth_file, f"{prefix}_{sample_size}",
                                                      subdir_name, target_col, task_type, le,
                                                      sample_size, X_test, y_test_encoded)
                _route_result(synth_result, "synthetic", subdir_name)

                if vanilla_files:
                    vanilla_name = prefix.replace("synth", "vanilla")
                    print(f"Processing corresponding vanilla file: {os.path.basename(vanilla_files[0])}")
                    vanilla_result = process_single_dataset(vanilla_files[0], f"{vanilla_name}_{sample_size}",
                                                            subdir_name, target_col, task_type, le,
                                                            sample_size, X_test, y_test_encoded)
                    _route_result(vanilla_result, "vanilla", subdir_name)

                if rest_file:
                    print(f"Processing rest.csv with {sample_size} samples")
                    rest_result = process_single_dataset(rest_file, f"rest_{sample_size}",
                                                         subdir_name, target_col, task_type, le,
                                                         sample_size, X_test, y_test_encoded)
                    _route_result(rest_result, "rest", subdir_name)

        _process_synth_group(synth_auc_files, "synth_auc", vanilla_auc_files)
        _process_synth_group(synth_tpr_files, "synth_tpr", vanilla_tpr_files)

    if not auc_results and not tpr_results and not results:
        raise ValueError("No valid results obtained from any dataset")

    summary_dir = os.path.join(base_dir, "utility_summary")
    os.makedirs(summary_dir, exist_ok=True)

    if auc_results:
        auc_df = pd.DataFrame(auc_results)
        auc_output_path = os.path.join(summary_dir, "utility_auc.csv")
        auc_df.to_csv(auc_output_path, index=False)
        print(f"AUC results saved to: {auc_output_path}")

    if tpr_results:
        tpr_df = pd.DataFrame(tpr_results)
        tpr_output_path = os.path.join(summary_dir, "utility_tpr.csv")
        tpr_df.to_csv(tpr_output_path, index=False)
        print(f"TPR results saved to: {tpr_output_path}")

    all_results = auc_results + tpr_results + results
    return pd.DataFrame(all_results) if all_results else pd.DataFrame()


# ---------------------- VISUALIZATION ----------------------
def create_visualization(base_dir):
    """
    Create bar plots for utility metrics from CSV files in utility_summary subdirectory.

    Args:
        base_dir (str): Base directory containing utility_summary subdirectory
    """
    utility_summary_dir = os.path.join(base_dir, 'utility_summary')
    visualization_dir = os.path.join(base_dir, 'visualization')
    os.makedirs(visualization_dir, exist_ok=True)

    auc_file = os.path.join(utility_summary_dir, 'utility_auc.csv')
    tpr_file = os.path.join(utility_summary_dir, 'utility_tpr.csv')

    for csv_file, output_name in [
        (auc_file, 'utility_auc_plot.png'),
        (tpr_file, 'utility_tpr_plot.png')
    ]:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found, skipping...")
            continue

        df = pd.read_csv(csv_file)
        df['data_type'] = df['data_type'].replace({'rest': 'REAL-DATA', 'synthetic': 'TLP', 'vanilla': 'VANILLA'})
        df = df.sort_values('n_train_samples')

        if 'rmse' in df.columns:
            metric_col = 'rmse'
            metric_label = 'RMSE'
        elif 'accuracy' in df.columns:
            metric_col = 'accuracy'
            metric_label = 'Accuracy'
        else:
            print(f"Warning: Neither 'rmse' nor 'accuracy' column found in {csv_file}, skipping...")
            continue

        train_samples = sorted(df['n_train_samples'].unique())
        data_types = ['REAL-DATA', 'VANILLA', 'TLP']

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(train_samples))
        width = 0.25
        colors = {'REAL-DATA': "#E70C0C", 'VANILLA': "#1721DD", 'TLP': "#DDDD27"}

        for i, data_type in enumerate(data_types):
            values = []
            for train_sample in train_samples:
                subset = df[(df['n_train_samples'] == train_sample) &
                           (df['data_type'] == data_type)]
                if not subset.empty:
                    values.append(subset[metric_col].iloc[0])
                else:
                    values.append(0)

            bar_positions = x + (i - 1) * width
            bars = ax.bar(bar_positions, values, width,
                         label=data_type, color=colors[data_type], alpha=0.8)

            for bar, value in zip(bars, values):
                if value > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Number of Training Samples')
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(train_samples)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(df[metric_col]) * 1.1)

        output_path = os.path.join(visualization_dir, output_name)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {output_name} to {visualization_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python advanced_utility.py <base_dir>")
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