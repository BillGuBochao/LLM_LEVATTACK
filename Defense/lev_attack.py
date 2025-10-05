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

def lev_attack_evaluation(member_df: pd.DataFrame, non_member_df: pd.DataFrame, syn_df: pd.DataFrame):
    """
    Evaluate membership inference attack using Levenshtein distance.
    
    Args:
        member_df: DataFrame containing member data
        non_member_df: DataFrame containing non-member data  
        syn_df: DataFrame containing synthetic data
        
    Returns:
        tuple: (auc_roc, tpr_at_fpr_0.1)
    """
    
    # Prepare test set
    member_rows = member_df.apply(row_to_str, axis=1).tolist()
    non_member_rows = non_member_df.apply(row_to_str, axis=1).tolist()
    
    test_rows = member_rows + non_member_rows
    true_labels = [1] * len(member_rows) + [0] * len(non_member_rows)
    
    # Prepare synthetic data
    synth_rows = syn_df.apply(row_to_str, axis=1).tolist()
    
    print(f"\nEvaluating attack with {len(member_rows)} members, {len(non_member_rows)} non-members, {len(synth_rows)} synthetic records\n")
    
    # Compute minimum distances (negative for scoring)
    min_distances = -np.array(compute_min_distances(test_rows, synth_rows))
    print(min_distances)
    
    # Run MIA evaluation
    AE = evaluation.AttackEvaluator(true_labels, min_distances)
    metrics = AE.roc_metrics(target_fprs=[0, 0.001, 0.01, 0.1])

    return metrics['auc_roc'], metrics[f'tpr_at_fpr_0.1']


