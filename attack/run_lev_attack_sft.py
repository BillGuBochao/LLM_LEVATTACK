import os
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from tqdm import tqdm
from synth_mia import evaluation
from synth_mia.attackers import dcr 
from rapidfuzz.distance import Levenshtein

# Root directory containing all experiments
ROOT_DIR = "experiments/"


def row_to_str(row):
    return ','.join(
        f"value: {float(val)}" if isinstance(val, (int, float, np.number)) else f"value: {str(val)}"
        for val in row
    )

def compute_min_distances(test_rows, synth_rows):
    return [
        -min(Levenshtein.distance(test_row, synth_row) for synth_row in synth_rows)
        for test_row in tqdm(test_rows, desc="Computing distances")
    ]

# Initialize the summary CSV file with headers
summary_path = os.path.join(ROOT_DIR, "mia_evaluation_summary.csv")
with open(summary_path, 'w', newline='') as csvfile:
    import csv
    fieldnames = [
        "experiment", "dataset", "model_type", "synth_file", 
        "lev_auc_roc", "lev_tpr_at_fpr_0", "lev_tpr_at_fpr_0.001", 
        "lev_tpr_at_fpr_0.01", "lev_tpr_at_fpr_0.1"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

results_count = 0

# Loop through each experiment folder in the root directory
for experiment_folder in sorted(os.listdir(ROOT_DIR)):
    experiment_path = os.path.join(ROOT_DIR, experiment_folder)
    
    # Skip if it's not a directory
    if not os.path.isdir(experiment_path):
        continue
    
    print(f"\nProcessing experiment: {experiment_folder}")
    
    # Loop through each dataset folder within the experiment
    for dataset_folder in sorted(os.listdir(experiment_path)):
        dataset_path = os.path.join(experiment_path, dataset_folder)
        
        # Skip if it's not a directory
        if not os.path.isdir(dataset_path):
            continue
        
        print(f"  Processing dataset: {dataset_folder}")
        
        member_path = os.path.join(dataset_path, "member.csv")
        non_member_path = os.path.join(dataset_path, "non_member.csv")
        synth_base_path = os.path.join(dataset_path, "synth")

        if not os.path.exists(member_path) or not os.path.exists(non_member_path) or not os.path.exists(synth_base_path):
            print(f"    Skipping {dataset_folder} due to missing files.")
            continue

        # Load an# Prepare test set - using only numeric fields
        member_df = pd.read_csv(member_path)
        non_member_df = pd.read_csv(non_member_path)

        # Filter to keep only numeric columns
        numeric_cols_member = member_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols_non_member = non_member_df.select_dtypes(include=[np.number]).columns.tolist()

        # Get intersection of numeric columns to ensure consistency
        common_numeric_cols = list(set(numeric_cols_member) & set(numeric_cols_non_member))

        # Keep only numeric columns
        member_df = member_df[common_numeric_cols]
        non_member_df = non_member_df[common_numeric_cols]

        # Limit to specified number of rows
        member_df = member_df.head(1000)
        non_member_df = non_member_df.head(1000)

        member_rows = member_df.apply(row_to_str, axis=1).tolist()
        non_member_rows = non_member_df.apply(row_to_str, axis=1).tolist()
        test_rows = member_rows + non_member_rows
        true_labels = [1] * len(member_rows) + [0] * len(non_member_rows)

        # Loop through each model type folder in synth directory
        for model_type in sorted(os.listdir(synth_base_path)):
            model_path = os.path.join(synth_base_path, model_type)
        
            # Skip if it's not a directory
            if not os.path.isdir(model_path):
                continue
        
            print(f"    Processing model: {model_type}")
        
            # Process each synthetic file for this model type
            for synth_file in sorted(os.listdir(model_path)):
                # Only process CSV files that contain "1x"
                if not synth_file.endswith(".csv") or "1x" not in synth_file:
                    continue
                print(f"      Evaluating: {synth_file}")
            
                try:
                    synth_file_path = os.path.join(model_path, synth_file)
                    synth_df = pd.read_csv(synth_file_path)
                    
                    # Filter synthetic dataframe to same numeric columns
                    synth_df = synth_df[common_numeric_cols]
                    synth_rows = synth_df.apply(row_to_str, axis=1).tolist()

                    min_distances = np.array(compute_min_distances(test_rows, synth_rows))  # Higher ratio = more similar
                    
                    # Run MIA evaluation
                    AE = evaluation.AttackEvaluator(true_labels, min_distances)
                    metrics = AE.roc_metrics(target_fprs=[0, 0.001, 0.01, 0.1])
                    print(f"        Levenshtein AUC-ROC: {metrics['auc_roc']:.4f}")
                    
                    # Save individual scores if needed
                    scores_df = pd.DataFrame({
                        'true_label': true_labels,
                        'score': min_distances
                    })
                    scores_path = os.path.join(dataset_path, f"lev_scores_2_{model_type}_{synth_file}")
                    scores_df.to_csv(scores_path, index=False)
                    
                    # Optional: Run DCR attack as well
                    print('        Running Euclidean DCR attack:')
                    
                    # Write results immediately to CSV
                    result_row = {
                        "experiment": experiment_folder,
                        "dataset": dataset_folder,
                        "model_type": model_type,
                        "synth_file": synth_file,
                        "lev_auc_roc": metrics["auc_roc"],
                        "lev_tpr_at_fpr_0": metrics["tpr_at_fpr_0"],
                        "lev_tpr_at_fpr_0.001": metrics["tpr_at_fpr_0.001"],
                        "lev_tpr_at_fpr_0.01": metrics["tpr_at_fpr_0.01"],
                        "lev_tpr_at_fpr_0.1": metrics["tpr_at_fpr_0.1"],
                    }
                    
                    with open(summary_path, 'a', newline='') as csvfile:
                        import csv
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result_row)
                    
                    results_count += 1
                    
                except Exception as e:
                    print(f"        Failed to process {synth_file}: {str(e)}")
                    # Write a row with None values for failed processing
                    result_row = {
                        "experiment": experiment_folder,
                        "dataset": dataset_folder,
                        "model_type": model_type,
                        "synth_file": synth_file,
                        "lev_auc_roc": None,
                        "lev_tpr_at_fpr_0": None,
                        "lev_tpr_at_fpr_0.001": None,
                        "lev_tpr_at_fpr_0.01": None,
                        "lev_tpr_at_fpr_0.1": None,
                    }
                    
                    with open(summary_path, 'a', newline='') as csvfile:
                        import csv
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result_row)
                    
                    results_count += 1
                    
                except Exception as e:
                    print(f"        DCR attack failed: {str(e)}")
                    result_row = {
                        "experiment": experiment_folder,
                        "dataset": dataset_folder,
                        "model_type": model_type,
                        "synth_file": synth_file,
                        "lev_auc_roc": metrics["auc_roc"],
                        "lev_tpr_at_fpr_0": metrics["tpr_at_fpr_0"],
                        "lev_tpr_at_fpr_0.001": metrics["tpr_at_fpr_0.001"],
                        "lev_tpr_at_fpr_0.01": metrics["tpr_at_fpr_0.01"],
                        "lev_tpr_at_fpr_0.1": metrics["tpr_at_fpr_0.1"],
                    }
                    
                    with open(summary_path, 'a', newline='') as csvfile:
                        import csv
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerow(result_row)
                    
                    results_count += 1

print(f"\nMIA results saved to {summary_path}")
print(f"Processed {results_count} synthetic datasets")