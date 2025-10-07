import pandas as pd
import numpy as np
import gc
import torch
import os
import shutil
import sys
from typing import Literal, Tuple, Dict, Any, Optional, List
from lev_attack import lev_attack_evaluation
from fidelity_eval import maximum_mean_discrepancy, jensen_shannon_distance, wasserstein_distance
from realtabformer import REaLTabFormer


def compute_fidelity_metrics(member_df, synth_df):
    """
    Compute fidelity metrics between member and synthetic data.
    If the two datasets have different numbers of rows, the larger
    one is truncated to the first min(n_real, n_syn) rows.
    """
    # Basic shape sanity check
    if member_df.shape[1] != synth_df.shape[1]:
        raise ValueError(
            f"Column mismatch: member has {member_df.shape[1]} cols, "
            f"synth has {synth_df.shape[1]} cols."
        )

    # Convert to numpy
    X_real = member_df.to_numpy()
    X_syn  = synth_df.to_numpy()

    # Truncate to the same number of rows (take the first few from the larger set)
    n = min(len(X_real), len(X_syn))
    if len(X_real) != len(X_syn):
        print(f"Row mismatch detected: real={len(X_real)}, synth={len(X_syn)}. "
              f"Using first {n} rows from each for fidelity metrics.")
    X_real = X_real[:n]
    X_syn  = X_syn[:n]

    print("Computing fidelity metrics...")

    try:
        mmd_score   = maximum_mean_discrepancy(X_real, X_syn, kernel="rbf")
        js_distance = jensen_shannon_distance(X_real, X_syn)
        ws_distance = wasserstein_distance(X_real, X_syn)  # your Sinkhorn-based fn
        return {
            "mmd": mmd_score,
            "jensen_shannon": js_distance,
            "wasserstein": ws_distance,
        }
    except Exception as e:
        print(f"Error in fidelity computation: {e}")
        return {
            "mmd": 0.0,
            "jensen_shannon": 0.0,
            "wasserstein": 0.0
        }
    

def train_model(base_directory: str, num_train_length: int):
    """
    Train a synthetic data model and prepare member/non-member datasets.

    Args:
        base_directory: Directory containing the CSV file to train on
        num_train_length: Number of records to use for training (for both member and non-member)

    Returns:
        tuple: (member_df, non_member_df, experiment_id, rest_member_df)
    """

    # Find the CSV file in the base directory (exclude summary files)
    csv_files = [f for f in os.listdir(base_directory) if f.endswith('.csv') and 'summary' not in f.lower()]
    if not csv_files:
        raise ValueError(f"No dataset CSV files found in {base_directory} (excluding summary files)")

    # Assume we use the first CSV file found
    csv_path = os.path.join(base_directory, csv_files[0])
    full_df = pd.read_csv(csv_path)

    if len(full_df) < 2 * num_train_length:
        raise ValueError(f"Not enough records in CSV. Need {2 * num_train_length}, but only have {len(full_df)}")

    # Create member and non-member dataframes
    # Shuffle the data first to ensure random split (seed set in main.py)
    shuffled_df = full_df.sample(frac=1).reset_index(drop=True)

    member_df = shuffled_df.iloc[:num_train_length].copy()
    non_member_df = shuffled_df.iloc[num_train_length:2*num_train_length].copy()
    rest_member_df = shuffled_df.iloc[2*num_train_length:].copy()

    print(f"Created member_df with {len(member_df)} records")
    print(f"Created non_member_df with {len(non_member_df)} records")
    print(f"Created rest_member_df with {len(rest_member_df)} records")

    # Training phase (processor = False)
    print(f"Training on member data from: {csv_path}")
    rtf_model = REaLTabFormer(
        model_type="tabular",
        gradient_accumulation_steps=4,
        logging_steps=100,
        epochs=280
    )

    rtf_model.fit(member_df)
    experiment_id = rtf_model.experiment_id
    rtf_model.save("rtf_model/")

    print(f"Model saved with experiment_id: {experiment_id}")

    # Cleanup after training
    del rtf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return member_df, non_member_df, experiment_id, rest_member_df


def evaluate_tlp(
    member_df: pd.DataFrame,
    non_member_df: pd.DataFrame,
    rtf_model: REaLTabFormer,
    num_synthetic_record: int,
    eval_type: Literal["auc", "tpr"] = "auc",
    threshold: float = None,
    max_tendency: float = 20.0,
    tendency_step: float = 0.5,
    initial_tendency: float = 1.75
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame]]:
    """
    Unified function to evaluate a pre-trained model on different synthetic dataset sizes 
    with a dynamic tendency step.

    Args:
        member_df: DataFrame containing member data
        non_member_df: DataFrame containing non-member data
        rtf_model: Trained REaLTabFormer model
        num_synthetic_record: Number of synthetic records to generate
        eval_type: Type of evaluation - "auc" or "tpr"
        threshold: Privacy threshold (AUC threshold for "auc", TPR threshold for "tpr")
        max_tendency: Maximum tendency value to try
        tendency_step: Fallback step size for tendency increment
        initial_tendency: Starting tendency value

    Returns:
        tuple: (results_dict, synth_df) where synth_df is returned if threshold is met, 
               otherwise (results_dict, None)
    
    Dynamic step policy:
      For AUC (relative to threshold):
        - If (metric - threshold)/threshold >= 0.30 -> step = 3.0
        - If (metric - threshold)/threshold >= 0.20 -> step = 1.5
        - If (metric - threshold)/threshold >= 0.10 -> step = 1.25
        - If (metric - threshold)/threshold >= 0.05 -> step = 1.0
        - Else -> step = tendency_step (fallback)
      
      For TPR (relative to threshold):
        Same logic but using (metric - threshold) / max(threshold, eps)
    """
    # Set default thresholds if not provided
    if threshold is None:
        threshold = 0.55 if eval_type == "auc" else 0.125
    
    # Initialize results dictionary with eval-type specific fields
    metric_key = 'auc_roc' if eval_type == "auc" else 'tpr_at_fpr_0.1'
    final_metric_key = 'final_auc' if eval_type == "auc" else 'final_tpr'
    
    results = {
        'num_synthetic_record': num_synthetic_record,
        'tendency_results': [],
        'optimal_tendency': None,
        final_metric_key: None,
        'threshold_met': False,
        'fidelity_metrics': None
    }

    current_tendency = initial_tendency
    eps = 1e-8
    
    while current_tendency <= max_tendency:
        print(f"\nTesting tendency: {current_tendency} for {num_synthetic_record} samples")
        synth_df = rtf_model.sample(
            n_samples=num_synthetic_record,
            processor=True,
            tendency=current_tendency
        )

        # Evaluate privacy
        auc_roc, tpr_at_fpr_0_1 = lev_attack_evaluation(member_df, non_member_df, synth_df)
        
        # Select the metric based on eval_type
        metric_value = auc_roc if eval_type == "auc" else tpr_at_fpr_0_1
        metric_name = "AUC" if eval_type == "auc" else "TPR@FPR=0.1"
        
        print(f"Tendency: {current_tendency}, {metric_name}: {metric_value:.4f}")

        # Store result
        result_entry = {
            'tendency': current_tendency,
            metric_key: metric_value
        }
        results['tendency_results'].append(result_entry)

        # Check if threshold is met (different logic for AUC vs TPR)
        threshold_met = False
        if eval_type == "auc":
            # For AUC, we want metric < threshold
            threshold_met = metric_value < threshold
        else:  # tpr
            # For TPR, we want metric <= threshold
            threshold_met = metric_value <= threshold
        
        if threshold_met:
            results['optimal_tendency'] = current_tendency
            results[final_metric_key] = metric_value
            results['threshold_met'] = True

            try:
                fidelity_metrics = compute_fidelity_metrics(member_df, synth_df)
                results['fidelity_metrics'] = fidelity_metrics
                print(f"MMD: {fidelity_metrics['mmd']:.4f}")
                print(f"Jensen-Shannon: {fidelity_metrics['jensen_shannon']:.4f}")
                print(f"Wasserstein: {fidelity_metrics['wasserstein']:.4f}")
            except Exception as e:
                print(f"Error computing fidelity metrics: {e}")
                results['fidelity_metrics'] = None

            # Return the synth_df since threshold is met
            return results, synth_df

        # Calculate dynamic step size (metric is above threshold here)
        if eval_type == "auc":
            rel_margin = (metric_value - threshold) / threshold
        else:  # tpr
            rel_margin = (metric_value - threshold) / max(threshold, eps)
        
        # Determine step size based on relative margin
        if rel_margin >= 0.30:
            step = 3.0
        elif rel_margin >= 0.20:
            step = 1.5
        elif rel_margin >= 0.10:
            step = 1.25
        elif rel_margin >= 0.05:
            step = 1.0
        else:
            step = tendency_step  # fallback if close to threshold

        print(f"Using dynamic tendency step: {step} (relative margin: {rel_margin:.3f})")

        del synth_df
        gc.collect()

        current_tendency += step

    # Threshold not met within max_tendency
    if not results['threshold_met']:
        print(f"✗ Threshold not met within max_tendency={max_tendency}")
        if results['tendency_results']:
            # Pick the best result (lowest metric value)
            best_result = min(results['tendency_results'], key=lambda x: x[metric_key])
            results['optimal_tendency'] = best_result['tendency']
            results[final_metric_key] = best_result[metric_key]
        else:
            results['optimal_tendency'] = None
            results[final_metric_key] = None

    return results, None


def evaluate_vanilla(
    member_df: pd.DataFrame,
    non_member_df: pd.DataFrame,
    rtf_model: REaLTabFormer,
    synthetic_sizes: List[int]
) -> Tuple[List[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Evaluate vanilla REaLTabFormer model (no tendency parameter) for comparison.

    Args:
        member_df: DataFrame containing member data
        non_member_df: DataFrame containing non-member data
        rtf_model: Trained REaLTabFormer model
        synthetic_sizes: List of synthetic dataset sizes to evaluate

    Returns:
        tuple: (list of results for each synthetic size, vanilla_synth DataFrame or None)
    """
    vanilla_results = []
    vanilla_synth = None
    max_synthetic_size = max(synthetic_sizes)

    for num_synthetic_record in synthetic_sizes:
        print(f"\n{'='*50}")
        print(f"Evaluating VANILLA model for synthetic size: {num_synthetic_record}")
        print(f"{'='*50}")

        try:
            # Generate synthetic data without tendency parameter
            synth_df = rtf_model.sample(
                n_samples=num_synthetic_record,
                processor=False
                # No tendency parameter - this uses the default/vanilla generation
            )

            # Evaluate privacy (AUC and TPR)
            auc_roc, tpr_at_fpr_0_1 = lev_attack_evaluation(member_df, non_member_df, synth_df)
            print(f"Vanilla model - AUC: {auc_roc:.4f}, TPR@FPR=0.1: {tpr_at_fpr_0_1:.4f}")

            # Evaluate fidelity
            try:
                fidelity_metrics = compute_fidelity_metrics(member_df, synth_df)
                print(f"Vanilla model - MMD: {fidelity_metrics['mmd']:.4f}")
                print(f"Vanilla model - Jensen-Shannon: {fidelity_metrics['jensen_shannon']:.4f}")
                print(f"Vanilla model - Wasserstein: {fidelity_metrics['wasserstein']:.3f}")
            except Exception as e:
                print(f"Error computing fidelity metrics for vanilla model: {e}")
                fidelity_metrics = None

            result = {
                'synthetic_size': num_synthetic_record,
                'auc_roc': auc_roc,
                'tpr_at_fpr_0.1': tpr_at_fpr_0_1,
                'fidelity_metrics': fidelity_metrics
            }

            vanilla_results.append(result)

            # Keep the synthetic data if this is the largest synthetic size
            if num_synthetic_record == max_synthetic_size:
                vanilla_synth = synth_df.copy()
                print(f"Kept vanilla synthetic data for largest size: {max_synthetic_size}")
            else:
                # Cleanup
                del synth_df
                gc.collect()

        except Exception as e:
            print(f"ERROR evaluating vanilla model for size {num_synthetic_record}: {str(e)}")
            # Add a failed result
            vanilla_results.append({
                'synthetic_size': num_synthetic_record,
                'auc_roc': None,
                'tpr_at_fpr_0.1': None,
                'fidelity_metrics': None,
                'error': str(e)
            })
            continue

    return vanilla_results, vanilla_synth