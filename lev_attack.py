import os
import pandas as pd
import numpy as np
import Levenshtein
from tqdm import tqdm
from synth_mia import evaluation
from synth_mia.attackers import dcr 
import ipdb


def row_to_str(row):
    return ','.join(
        str(float(val)) if isinstance(val, (int, float, np.number)) else str(val)
        for val in row
    )

def compute_min_distances(test_rows, synth_rows):
    return [
        min(Levenshtein.distance(test_row, synth_row) for synth_row in synth_rows)
        for test_row in tqdm(test_rows, desc="Computing distances")
    ]

def lev_attack_evaluation(base_dir: str):

    result = []

    all_items = sorted(os.listdir(base_dir))
    var_folders = [f for f in all_items if os.path.isdir(os.path.join(base_dir, f))]
    
    for idx, var_folder in enumerate(var_folders):
        var_path = os.path.join(base_dir, var_folder)
        if not os.path.isdir(var_path):
            continue

        member_path = os.path.join(var_path, "member.csv")
        non_member_path = os.path.join(var_path, "non_member.csv")
        synth_path = os.path.join(var_path, "synth")

        if not os.path.exists(member_path) or not os.path.exists(non_member_path) or not os.path.exists(synth_path):
            print(f"Skipping {var_folder} due to missing files.")
            continue

        # Load and prepare test set
        member_df = pd.read_csv(member_path)
        non_member_df = pd.read_csv(non_member_path)
        member_rows = member_df.apply(row_to_str, axis=1).tolist()
        non_member_rows = non_member_df.apply(row_to_str, axis=1).tolist()

        test_rows = member_rows + non_member_rows
        true_labels = [1] * len(member_rows) + [0] * len(non_member_rows)

        
        for synth_file in sorted(os.listdir(synth_path)):
            if not synth_file.endswith(".csv"):
                continue

            synth_df = pd.read_csv(os.path.join(synth_path, synth_file))
            synth_rows = synth_df.apply(row_to_str, axis=1).tolist()
            

            print(f"\nEvaluating: {var_folder} \n")
            min_distances = -np.array(compute_min_distances(test_rows, synth_rows))
            print(min_distances)
            # Convert distances to scores (invert, because smaller = more likely to be member)
            
            # Run MIA evaluation
            AE = evaluation.AttackEvaluator(true_labels, min_distances)
            metrics = AE.roc_metrics(target_fprs=[0, 0.001, 0.01, 0.1])
            
            
            print('Euclidean DCR:')
                # Create instances of attacker

            attacker  = dcr()
            true_labels, scores = attacker.attack(member_df.values, non_member_df.values, synth_df.values)
            
            # Evaluate the attack
            eval_results = attacker.eval(true_labels, scores, metrics=['roc'])

            result.append({
            "var": var_folder,
            "synth_file": synth_file,
            "auc_roc_lev": metrics["auc_roc"],
            "auc_roc_euc": eval_results["auc_roc"]
            })


    # Save all everthing to one summary CSV
    
    summary_df = pd.DataFrame(result)
    summary_path = os.path.join(base_dir, "mia_evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n MIA results saved to {summary_path}")
