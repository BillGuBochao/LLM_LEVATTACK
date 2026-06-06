import os
import gc
import csv
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score

# Fidelity functions from your module
from fidelity_eval import maximum_mean_discrepancy, wasserstein_distance
# Utility helpers from your module
from advanced_utility import encode_features, train_xgb_classification

# --- Configuration ---
ROOT_DIR = "sft_data/"
SUMMARY_PATH = os.path.join(ROOT_DIR, "fidelity_utility_summary.csv")
MAX_FIDELITY_SAMPLES = 200  # Crucial for MMD memory management
USE_GPU = True 

# --- Helper Functions ---

def _clean_dataframe(df):
    """Handles basic numeric cleanup to prevent OHE bloat."""
    for col in df.select_dtypes(include=["object"]).columns:
        cleaned = df[col].astype(str).str.strip().str.replace(r'[\$,]', '', regex=True)
        converted = pd.to_numeric(cleaned, errors='coerce')
        if converted.notna().mean() >= 0.5:
            df[col] = converted
    return df.dropna().reset_index(drop=True)

def get_fidelity_matrices(real_df, synth_df):
    """Aligns and one-hot encodes, ignoring case sensitivity in column names."""
    # 1. Normalize column names to lowercase for comparison
    real_df.columns = [c.lower().strip() for c in real_df.columns]
    synth_df.columns = [c.lower().strip() for c in synth_df.columns]
    
    # Find common columns
    common_cols = sorted(list(set(real_df.columns) & set(synth_df.columns)))
    
    if not common_cols:
        raise ValueError("No common columns found between real and synthetic data.")

    # Slice and clean
    r_df = _clean_dataframe(real_df[common_cols].copy())
    s_df = _clean_dataframe(synth_df[common_cols].copy())
    
    # Guard against empty sets after cleaning
    if r_df.empty or s_df.empty:
        raise ValueError("Dataframe empty after cleaning; check for NaN or formatting issues.")

    cat_cols = r_df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = r_df.select_dtypes(include=["number"]).columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    
    if cat_cols:
        real_cat = encoder.fit_transform(r_df[cat_cols].astype(str))
        synth_cat = encoder.transform(s_df[cat_cols].astype(str))
        X_real = np.hstack([r_df[num_cols].values, real_cat]).astype(np.float32)
        X_synth = np.hstack([s_df[num_cols].values, synth_cat]).astype(np.float32)
    else:
        X_real = r_df[num_cols].values.astype(np.float32)
        X_synth = s_df[num_cols].values.astype(np.float32)
    
    return X_real, X_synth
def run_classification_utility(train_df, test_df, target_col):
    """Train on Synthetic, Evaluate AUC and F1 on Real."""
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # Encode features using your advanced_utility helper
    X_train_enc, X_test_enc, _, _ = encode_features(X_train, X_test)
    
    # Label Encode Target
    le = LabelEncoder()
    # Fit on union to handle potential missing classes in synthetic data
    all_labels = pd.concat([y_train.astype(str), y_test.astype(str)])
    le.fit(all_labels)
    y_train_enc = le.transform(y_train.astype(str))
    y_test_enc = le.transform(y_test.astype(str))
    
    num_classes = len(le.classes_)
    
    # Train
    model = train_xgb_classification(X_train_enc, y_train_enc, use_gpu=USE_GPU, num_classes=num_classes)
    
    # Predict
    dtest = xgb.DMatrix(X_test_enc)
    y_prob = model.predict(dtest)
    
    # Metrics Logic
    if num_classes == 2:
        y_pred = (y_prob > 0.5).astype(int)
        auc = roc_auc_score(y_test_enc, y_prob)
    else:
        # Multiclass: reshape probs if necessary depending on xgb version/config
        y_prob_reshaped = y_prob.reshape(len(y_test_enc), num_classes)
        y_pred = np.argmax(y_prob_reshaped, axis=1)
        auc = roc_auc_score(y_test_enc, y_prob_reshaped, multi_class="ovr", average="weighted")
    
    f1 = f1_score(y_test_enc, y_pred, average="weighted")
    
    return {"auc": auc, "f1": f1}
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
def run_utility(train_df, test_df, target_col):
    """
    Unified utility: Detects task, trains model, returns task-specific metrics.
    """
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    # 1. Task Detection: Numeric with > 3 levels = Regression
    is_regression = False
    if pd.api.types.is_numeric_dtype(y_train) and y_train.nunique() > 3:
        is_regression = True

    # 2. Feature Encoding
    X_train_enc, X_test_enc, _, _ = encode_features(X_train, X_test)
    
    # Initialize results with None/null as requested
    results = {"task_type": "classification", "auc": None, "f1": None, "rmse": None}

    # Modern XGBoost GPU configuration
    xgb_config = {
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
        "random_state": 42
    }

    if is_regression:
        results["task_type"] = "regression"
        # Using XGBRegressor for continuous targets
        model = xgb.XGBRegressor(**xgb_config)
        model.fit(X_train_enc, y_train)
        
        preds = model.predict(X_test_enc)
        results["rmse"] = np.sqrt(mean_squared_error(y_test, preds))
    
    else:
        # Standard Classification Logic
        le = LabelEncoder()
        all_labels = pd.concat([y_train.astype(str), y_test.astype(str)])
        le.fit(all_labels)
        y_train_enc = le.transform(y_train.astype(str))
        y_test_enc = le.transform(y_test.astype(str))
        num_classes = len(le.classes_)

        # Note: If your train_xgb_classification helper is internal, 
        # ensure it uses tree_method="hist", device="cuda"
        model = train_xgb_classification(X_train_enc, y_train_enc, use_gpu=USE_GPU, num_classes=num_classes)
        
        dtest = xgb.DMatrix(X_test_enc)
        y_prob = model.predict(dtest)
        
        if num_classes == 2:
            y_pred = (y_prob > 0.5).astype(int)
            results["auc"] = roc_auc_score(y_test_enc, y_prob)
        else:
            y_prob_reshaped = y_prob.reshape(len(y_test_enc), num_classes)
            y_pred = np.argmax(y_prob_reshaped, axis=1)
            results["auc"] = roc_auc_score(y_test_enc, y_prob_reshaped, multi_class="ovr", average="weighted")
        
        results["f1"] = f1_score(y_test_enc, y_pred, average="weighted")
    
    return results
# --- Execution ---

fieldnames = ["experiment", "model_type", "file", "mmd", "wass", "task_type", "auc_roc", "f1_score", "rmse"]

with open(SUMMARY_PATH, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

for experiment in sorted(os.listdir(ROOT_DIR)):
    exp_path = os.path.join(ROOT_DIR, experiment)
    member_path = os.path.join(exp_path, "member.csv")
    non_member_path = os.path.join(exp_path, "non_member.csv")

    if not os.path.exists(member_path): continue
    
    train_df_full = pd.read_csv(member_path)
    test_df_full = pd.read_csv(non_member_path)

    # Normalize Real Data Columns
    train_df_full.columns = [c.lower().strip() for c in train_df_full.columns]
    test_df_full.columns = [c.lower().strip() for c in test_df_full.columns]

    # Now the target will definitely be lowercase
    target = test_df_full.columns[-1]

    synth_base = os.path.join(exp_path, "synth")
    if not os.path.exists(synth_base): continue

    for model_type in os.listdir(synth_base):
        model_dir = os.path.join(synth_base, model_type)
        if not os.path.isdir(model_dir): continue
        
        for synth_file in sorted([f for f in os.listdir(model_dir) if f.endswith(".csv")]):
            try:
                print(f"Processing: {experiment} | {model_type} | {synth_file}")
                synth_df = pd.read_csv(os.path.join(model_dir, synth_file))

                # Normalize Synthetic Data Columns
                synth_df.columns = [c.lower().strip() for c in synth_df.columns]

                # Ensure column order matches for utility training
                # This prevents "Feature names mismatch" in XGBoost
                synth_df = synth_df[test_df_full.columns]                
                # 1. Fidelity Calculation
                X_real, X_synth = get_fidelity_matrices(test_df_full, synth_df)
                n = min(len(X_real), len(X_synth), MAX_FIDELITY_SAMPLES)
                
                mmd_val = maximum_mean_discrepancy(X_real[:n], X_synth[:n])
                wass_val = wasserstein_distance(X_real[:n], X_synth[:n])
                
                # Clear memory before heavy XGBoost training
                del X_real, X_synth
                gc.collect()

                fieldnames = ["experiment", "model_type", "file", "mmd", "wass", "task_type", "auc_roc", "f1_score", "rmse"]
                # Inside your loop...
                util = run_utility(synth_df, test_df_full, target)

                with open(SUMMARY_PATH, 'a', newline='') as f:
                        csv.DictWriter(f, fieldnames=fieldnames).writerow({
                            "experiment": experiment,
                            "model_type": model_type,
                            "file": synth_file,
                            "mmd": mmd_val,
                            "wass": wass_val,
                            "task_type": util['task_type'],
                            "auc_roc": util['auc'],
                            "f1_score": util['f1'],
                            "rmse": util['rmse']
                        })
                                
            except Exception as e:
                print(f"  ! Error: {e}")
            finally:
                if 'synth_df' in locals(): del synth_df
                gc.collect()