import os
import pandas as pd
import numpy as np
import csv
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm
from synth_mia import evaluation

# Root directory containing all experiments
ROOT_DIR = "sft_data/"

# Optional: Only process specific model types (leave empty to process all)
model_include = []  # e.g., ["model1", "model2"] or [] for all

def row_to_str(row):
    """Converts a dataframe row to a consistent string format for Levenshtein distance."""
    return ','.join(
        f"value: {float(val)}" if isinstance(val, (int, float, np.number)) else f"value: {str(val)}"
        for val in row
    )

def compute_min_distances(test_rows, synth_rows):
    """Computes the minimum Levenshtein distance for each test row against all synth rows."""
    return [
        -min(Levenshtein.distance(test_row, synth_row) for synth_row in synth_rows)
        for test_row in tqdm(test_rows, desc="Computing distances", leave=False)
    ]

# Initialize the summary CSV file with headers
summary_path = os.path.join(ROOT_DIR, "mia_evaluation_summary.csv")
fieldnames = [
    "experiment", "model_type", "synth_file", 
    "lev_auc_roc", "lev_tpr_at_fpr_0", "lev_tpr_at_fpr_0.001", 
    "lev_tpr_at_fpr_0.01", "lev_tpr_at_fpr_0.1"
]

with open(summary_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

results_count = 0

# Loop through each experiment folder
for experiment_folder in sorted(os.listdir(ROOT_DIR)):
    experiment_path = os.path.join(ROOT_DIR, experiment_folder)
    
    if not os.path.isdir(experiment_path) or experiment_folder.startswith('.'):
        continue
    
    print(f"\nProcessing experiment: {experiment_folder}")
    
    member_path = os.path.join(experiment_path, "member.csv")
    non_member_path = os.path.join(experiment_path, "non_member.csv")
    synth_base_path = os.path.join(experiment_path, "synth")

    if not all(os.path.exists(p) for p in [member_path, non_member_path, synth_base_path]):
        print(f"  Skipping {experiment_folder}: Missing required files.")
        continue

    # Load and prepare test sets
    member_df = pd.read_csv(member_path)
    non_member_df = pd.read_csv(non_member_path)

    # CASE AGNOSTIC: Normalize column names to lowercase
    member_df.columns = [c.lower() for c in member_df.columns]
    non_member_df.columns = [c.lower() for c in non_member_df.columns]

    # Get common columns while maintaining ORDER based on member_df
    common_cols = [col for col in member_df.columns if col in non_member_df.columns]
    
    # Filter and limit rows
    member_df = member_df[common_cols].head(1000)
    non_member_df = non_member_df[common_cols].head(1000)

    member_rows = member_df.apply(row_to_str, axis=1).tolist()
    non_member_rows = non_member_df.apply(row_to_str, axis=1).tolist()
    test_rows = member_rows + non_member_rows
    true_labels = [1] * len(member_rows) + [0] * len(non_member_rows)

    # Process model types
    for model_type in sorted(os.listdir(synth_base_path)):
        model_path = os.path.join(synth_base_path, model_type)
    
        if not os.path.isdir(model_path) or (model_include and model_type not in model_include):
            continue
    
        print(f"  Processing model: {model_type}")
    
        # Process specific synthetic files
        for synth_file in sorted(os.listdir(model_path)):
            if synth_file not in ["1x_1.csv", "1x_2.csv", "1x_3.csv"]:
                continue
            
            print(f"    Evaluating: {synth_file}")
        
            try:
                synth_file_path = os.path.join(model_path, synth_file)
                synth_df = pd.read_csv(synth_file_path)
                
                # Normalize synthetic headers
                synth_df.columns = [c.lower() for c in synth_df.columns]
                
                # Reorder and filter synthetic data to match common_cols exactly
                # If columns are missing, reindex adds them as NaN to preserve string structure
                synth_df = synth_df.reindex(columns=common_cols)
                
                synth_rows = synth_df.apply(row_to_str, axis=1).tolist()
                min_distances = np.array(compute_min_distances(test_rows, synth_rows))
                
                # Run MIA evaluation
                AE = evaluation.AttackEvaluator(true_labels, min_distances)
                metrics = AE.roc_metrics(target_fprs=[0, 0.001, 0.01, 0.1])
                print(f"      Levenshtein AUC-ROC: {metrics['auc_roc']:.4f}")
                
                # Save individual scores
                scores_df = pd.DataFrame({'true_label': true_labels, 'score': min_distances})
                scores_path = os.path.join(experiment_path, f"lev_scores_2_{model_type}_{synth_file}")
                scores_df.to_csv(scores_path, index=False)
                
                result_row = {
                    "experiment": experiment_folder,
                    "model_type": model_type,
                    "synth_file": synth_file,
                    "lev_auc_roc": metrics["auc_roc"],
                    "lev_tpr_at_fpr_0": metrics["tpr_at_fpr_0"],
                    "lev_tpr_at_fpr_0.001": metrics["tpr_at_fpr_0.001"],
                    "lev_tpr_at_fpr_0.01": metrics["tpr_at_fpr_0.01"],
                    "lev_tpr_at_fpr_0.1": metrics["tpr_at_fpr_0.1"],
                }
                
            except Exception as e:
                print(f"      Failed to process {synth_file}: {str(e)}")
                result_row = {col: None for col in fieldnames}
                result_row.update({"experiment": experiment_folder, "model_type": model_type, "synth_file": synth_file})

            # Append results to CSV immediately
            with open(summary_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(result_row)
            
            results_count += 1

print(f"\nMIA results saved to {summary_path}")
print(f"Processed {results_count} synthetic datasets")