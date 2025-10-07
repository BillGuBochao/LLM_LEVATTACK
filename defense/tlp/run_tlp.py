import pandas as pd
import os
import gc
import torch
import shutil
from typing import Literal, Tuple, List, Dict, Any, Optional
from realtabformer_generate import train_model, evaluate_tlp, evaluate_vanilla
from realtabformer import REaLTabFormer


def main(base_dir: str, eval_type: Literal["auc", "tpr"] = "auc"):
    """
    Main function that runs privacy experiments for either AUC or TPR evaluation.
    
    Args:
        base_dir: Base directory containing the dataset
        eval_type: Type of evaluation - "auc" for AUC-based or "tpr" for TPR-based
    
    Returns:
        List of result dictionaries for each synthetic size
    """
    # Configuration
    base_directory = base_dir
    train_size = 4000
    synthetic_sizes = [4000, 6000, 8000, 10000]
    max_tendency = 20.0
    tendency_step = 0.5
    
    # Set evaluation-specific parameters
    if eval_type == "auc":
        threshold = 0.55
        base_initial_tendency = 3.0
        dir_suffix = "auc"
    else:  # tpr
        threshold = 0.1 * 1.25
        base_initial_tendency = 2.5
        dir_suffix = "tpr"

    all_results = []

    # Train model once
    member_df, non_member_df, experiment_id, rest_member_df = train_model(base_directory, train_size)
    rtf_model = REaLTabFormer.load_from_dir(path=f"rtf_model/{experiment_id}")
    
    # Create train_test_synth directory and save dataframes
    train_test_dir = os.path.join(base_directory, f"train_test_synth_{dir_suffix}")
    os.makedirs(train_test_dir, exist_ok=True)
    member_df.to_csv(os.path.join(train_test_dir, "train.csv"), index=False)
    non_member_df.to_csv(os.path.join(train_test_dir, "test.csv"), index=False)
    rest_member_df.to_csv(os.path.join(train_test_dir, "rest.csv"), index=False)
    print(f"Saved member_df as train.csv, non_member_df as test.csv, and rest_member_df as rest.csv in {train_test_dir}")

    # Initialize for the first loop
    next_initial_tendency = base_initial_tendency

    for num_synthetic_record in synthetic_sizes:
        print(f"\n{'='*50}")
        print(f"Processing synthetic size: {num_synthetic_record}")
        print(f"Starting initial_tendency: {next_initial_tendency}")
        print(f"{'='*50}")

        try:
            # Use the unified evaluation function
            results, synth_df = evaluate_tlp(
                member_df=member_df,
                non_member_df=non_member_df,
                rtf_model=rtf_model,
                num_synthetic_record=num_synthetic_record,
                eval_type=eval_type,
                threshold=threshold,
                max_tendency=max_tendency,
                tendency_step=tendency_step,
                initial_tendency=next_initial_tendency
            )
            
            all_results.append(results)
            
            # Save synth_df if it was returned (threshold met)
            if synth_df is not None:
                synth_filename = os.path.join(train_test_dir, f"synth_{dir_suffix}_{num_synthetic_record}.csv")
                try:
                    synth_df.to_csv(synth_filename, index=False)
                    print(f"Saved synthetic data to {synth_filename}")
                except Exception as e:
                    print(f"Error saving synthetic data to {synth_filename}: {e}")
                del synth_df
                gc.collect()

            # Decide what to use as the "optimal" tendency for chaining
            opt = results.get('optimal_tendency')
            if opt is None:
                # threshold not met; use best (lowest metric) among tried points
                if results['tendency_results']:
                    metric_key = 'auc_roc' if eval_type == "auc" else 'tpr_at_fpr_0.1'
                    opt = min(results['tendency_results'], key=lambda x: x[metric_key])['tendency']
                else:
                    opt = next_initial_tendency  # very unlikely, but safe fallback

            # New initial_tendency for the next size: opt - 0.25, clamped to [0, max_tendency]
            next_initial_tendency = max(0.0, min(max_tendency, float(opt) - 0.25))
            print(f"Next initial_tendency set to: {next_initial_tendency}")

        except Exception as e:
            print(f"ERROR with synthetic size {num_synthetic_record}: {str(e)}")
            # Keep the same starting point for the next iteration if this one failed
            continue

    # Evaluate vanilla model for comparison
    print(f"\n{'='*60}")
    print("EVALUATING VANILLA MODEL (no tendency)")
    print(f"{'='*60}")
    vanilla_results, vanilla_synth = evaluate_vanilla(member_df, non_member_df, rtf_model, synthetic_sizes)

    # Save vanilla synthetic data if available
    if vanilla_synth is not None:
        max_size = max(synthetic_sizes)
        vanilla_filename = os.path.join(train_test_dir, f"vanilla_{dir_suffix}_{max_size}.csv")
        try:
            vanilla_synth.to_csv(vanilla_filename, index=False)
            print(f"Saved vanilla synthetic data to {vanilla_filename}")
        except Exception as e:
            print(f"Error saving vanilla synthetic data to {vanilla_filename}: {e}")
        del vanilla_synth
        gc.collect()
    
    # Save summary CSVs
    save_summary_csv(all_results, base_dir, eval_type)
    save_vanilla_summary_csv(vanilla_results, base_dir, eval_type)

    # Free up memory by deleting dataframes
    del member_df, non_member_df
    gc.collect()
    print("Deleted member_df and non_member_df to free memory")

    # Cleanup model and checkpoints
    del rtf_model
    model_path = f"rtf_model/{experiment_id}"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"Deleted model directory: {model_path}")
    checkpoint_path = "rtf_checkpoints"
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
        print(f"Deleted checkpoints directory: {checkpoint_path}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return all_results


def save_summary_csv(all_results: List[Dict[str, Any]], base_dir: str, eval_type: Literal["auc", "tpr"]):
    """
    Save summary table as CSV.
    
    Args:
        all_results: List of result dictionaries
        base_dir: Base directory to save the CSV
        eval_type: Type of evaluation - "auc" or "tpr"
    """
    print(f"\n\nSaving summary CSV for {base_dir} ({eval_type.upper()})...\n\n")
    
    # Collect data for CSV
    summary_data = []
    
    for result in all_results:
        row_data = {
            'synthetic_size': result['num_synthetic_record'],
            'threshold_met': result.get('threshold_met', False),
            'optimal_tendency': result.get('optimal_tendency'),
            'mmd_score': None,
            'jensen_shannon_score': None,
            'wasserstein_score': None
        }
        
        # Add the appropriate metric based on eval_type
        if eval_type == "auc":
            row_data['final_auc'] = result.get('final_auc')
        else:  # tpr
            row_data['final_tpr'] = result.get('final_tpr')
        
        # Add fidelity metrics if available
        if result.get('threshold_met') and result.get('fidelity_metrics'):
            fm = result['fidelity_metrics']
            row_data['mmd_score'] = fm.get('mmd')
            row_data['jensen_shannon_score'] = fm.get('jensen_shannon')
            row_data['wasserstein_score'] = fm.get('wasserstein')
        
        summary_data.append(row_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(summary_data)
    
    # Ensure base_dir exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Create summary directory
    summary_dir = os.path.join(base_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save CSV file in summary directory
    csv_filename = os.path.join(summary_dir, f"summary_{eval_type}_{os.path.basename(base_dir)}.csv")
    df.to_csv(csv_filename, index=False, float_format='%.6f')
    
    print(f"Saved summary CSV to {csv_filename}")
    print(f"Summary contains {len(df)} rows with columns: {list(df.columns)}")


def save_vanilla_summary_csv(vanilla_results: List[Dict[str, Any]], base_dir: str, eval_type: Literal["auc", "tpr"]):
    """
    Save vanilla model results as CSV for comparison.
    
    Args:
        vanilla_results: List of vanilla result dictionaries
        base_dir: Base directory to save the CSV
        eval_type: Type of evaluation - "auc" or "tpr"
    """
    print(f"\n\nSaving vanilla model summary CSV for {base_dir} ({eval_type.upper()})...\n\n")
    
    # Collect data for CSV
    summary_data = []
    
    for result in vanilla_results:
        row_data = {
            'synthetic_size': result['synthetic_size'],
            'mmd_score': None,
            'jensen_shannon_score': None,
            'wasserstein_score': None,
            'error': result.get('error')
        }
        
        # Add the appropriate metric based on eval_type
        if eval_type == "auc":
            row_data['auc_roc'] = result.get('auc_roc')
        else:  # tpr
            row_data['tpr_at_fpr_0.1'] = result.get('tpr_at_fpr_0.1')
        
        # Add fidelity metrics if available
        if result.get('fidelity_metrics'):
            fm = result['fidelity_metrics']
            row_data['mmd_score'] = fm.get('mmd')
            row_data['jensen_shannon_score'] = fm.get('jensen_shannon')
            row_data['wasserstein_score'] = fm.get('wasserstein')
        
        summary_data.append(row_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(summary_data)
    
    # Ensure base_dir exists
    os.makedirs(base_dir, exist_ok=True)
    
    # Create summary directory
    summary_dir = os.path.join(base_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Save CSV file in summary directory
    csv_filename = os.path.join(summary_dir, f"vanilla_summary_{eval_type}_{os.path.basename(base_dir)}.csv")
    df.to_csv(csv_filename, index=False, float_format='%.6f')
    
    print(f"Saved vanilla model summary CSV to {csv_filename}")
    print(f"Vanilla summary contains {len(df)} rows with columns: {list(df.columns)}")


if __name__ == "__main__":
    
    # Example usage - run both AUC and TPR evaluations
    # results_auc = main("experimentCASP", eval_type="auc")
    results_tpr = main("experimentCASP", eval_type="tpr")

    # Create list of (dataset_name, results, eval_type) tuples
    ALL_RESULTS = [
        # ("experimentCASP", results_auc, "auc"),
        ("experimentCASP", results_tpr, "tpr")
    ]

    for dataset_name, all_results, eval_type in ALL_RESULTS:
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY - {dataset_name.upper()} ({eval_type.upper()})")
        print(f"{'='*60}")
        
        for result in all_results:
            status = "✓" if result['threshold_met'] else "✗"
            fidelity_str = ""
            if result['threshold_met'] and result['fidelity_metrics']:
                f = result['fidelity_metrics']
                fidelity_str = f" | MMD: {f['mmd']:.5f} | JS: {f['jensen_shannon']:.5f} | WS: {f['wasserstein']:.3f}"

            metric_value = result.get('final_auc') if eval_type == "auc" else result.get('final_tpr')
            metric_label = "Final AUC" if eval_type == "auc" else "Final TPR"
            
            print(f"{status} Synthetic size: {result['num_synthetic_record']:5d} | "
                  f"Optimal tendency: {result['optimal_tendency']:4.1f} | "
                  f"{metric_label}: {metric_value:.4f}{fidelity_str}")

    print("\nAll experiments completed.")