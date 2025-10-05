#!/usr/bin/env python3
"""
Main pipeline script for privacy-preserving synthetic data generation and evaluation.

This script orchestrates the entire workflow from model training to visualization,
supporting both AUC-based and TPR-based privacy evaluation approaches.
"""

import argparse
import os
import sys
import traceback
from typing import Literal
import random
import numpy as np

# Import the necessary modules
import vis_auc
import vis_tpr
from compare_visualization import compare_experiment_results_auc, compare_experiment_results_tpr


def set_random_seed(seed: int = 4231):
    """
    Set random seed for reproducible results across all random number generators.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Make PyTorch deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(base_dir: str, eval_type: Literal["auc", "tpr"]):
    """
    Run the complete privacy evaluation pipeline.
    
    This function orchestrates:
    1. Model training and synthetic data generation with tendency optimization
    2. Vanilla model evaluation for baseline comparison
    3. CSV export of summary results
    4. Visualization generation for comparison analysis
    
    Args:
        base_dir: Directory containing the dataset CSV file and where outputs will be saved
        eval_type: Type of privacy evaluation - either "auc" or "tpr"
        
    Returns:
        None
    """
    
    # Set random seed for reproducible results
    set_random_seed(42)
    
    print(f"{'='*80}")
    print(f"STARTING PRIVACY EVALUATION PIPELINE")
    print(f"Dataset directory: {base_dir}")
    print(f"Evaluation type: {eval_type.upper()}")
    print(f"{'='*80}")
    
    # Validate input arguments
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory '{base_dir}' does not exist!")
        return
    
    if eval_type not in ["auc", "tpr"]:
        print(f"ERROR: eval_type must be 'auc' or 'tpr', got '{eval_type}'")
        return
    
    # Check if dataset CSV exists (exclude summary files)
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv') and 'summary' not in f.lower()]
    if not csv_files:
        print(f"ERROR: No dataset CSV files found in {base_dir} (excluding summary files)")
        return
    
    print(f"Found dataset files: {csv_files}")
    
    try:
        # Step 1: Run the appropriate evaluation pipeline
        print(f"\\n{'-'*60}")
        print(f"STEP 1: Running {eval_type.upper()} evaluation pipeline")
        print(f"{'-'*60}")
        
        if eval_type == "auc":
            # Run AUC-based evaluation
            print("Executing AUC-based privacy evaluation...")
            all_results = vis_auc.main(base_dir)
        
        elif eval_type == "tpr":
            # Run TPR-based evaluation
            print("Executing TPR-based privacy evaluation...")
            all_results = vis_tpr.main(base_dir)
        
        # Check if evaluation was successful
        if not all_results:
            print("WARNING: No results returned from evaluation. Check for errors above.")
            return
        
        print(f"✓ {eval_type.upper()} evaluation completed successfully!")
        print(f"✓ Generated {len(all_results)} result entries")

        # Step 2: Generate comparison visualization
        print(f"\\n{'-'*60}")
        print(f"STEP 2: Generating comparison visualization")
        print(f"{'-'*60}")
        
        if eval_type == "auc":
            print("Creating AUC comparison plot...")
            compare_experiment_results_auc(base_dir, show_plot=False, save_plot=True)
            
        elif eval_type == "tpr":
            print("Creating TPR comparison plot...")
            compare_experiment_results_tpr(base_dir, show_plot=False, save_plot=True)
        
        print("✓ Comparison visualization completed!")
        
        # Step 3: Summary of generated files
        print(f"\\n{'-'*60}")
        print(f"STEP 3: Pipeline completion summary")
        print(f"{'-'*60}")
        
        dataset_name = os.path.basename(base_dir)
        
        # List expected output files
        expected_files = []
        if eval_type == "auc":
            expected_files = [
                f"summary_auc_{dataset_name}.csv",
                f"vanilla_summary_auc_{dataset_name}.csv", 
                f"comparison_auc_plot_{dataset_name}.png"
            ]
        elif eval_type == "tpr":
            expected_files = [
                f"summary_tpr_{dataset_name}.csv",
                f"vanilla_summary_tpr_{dataset_name}.csv",
                f"comparison_tpr_plot_{dataset_name}.png"
            ]
        
        print("Generated output files:")
        for filename in expected_files:
            filepath = os.path.join(base_dir, filename)
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"  ✓ {filename} ({file_size:,} bytes)")
            else:
                print(f"  ✗ {filename} (missing)")
        
        print(f"\\n{'='*80}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"All outputs saved to: {os.path.abspath(base_dir)}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\\n{'='*80}")
        print(f"PIPELINE FAILED!")
        print(f"{'='*80}")
        print(f"Error: {str(e)}")
        print(f"\\nFull traceback:")
        traceback.print_exc()
        print(f"\\nPlease check the error details above and try again.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run privacy-preserving synthetic data generation and evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py experimentCASP auc     # Run AUC-based evaluation
  python main.py experimentCASP tpr     # Run TPR-based evaluation
  python main.py experiment7 auc        # Run AUC evaluation on experiment7 dataset
        
Output files:
  For AUC evaluation:
    - summary_auc_{dataset}.csv: Tendency-based results
    - vanilla_summary_auc_{dataset}.csv: Vanilla baseline results  
    - comparison_auc_plot_{dataset}.png: Visual comparison
    
  For TPR evaluation:
    - summary_tpr_{dataset}.csv: Tendency-based results
    - vanilla_summary_tpr_{dataset}.csv: Vanilla baseline results
    - comparison_tpr_plot_{dataset}.png: Visual comparison
        """
    )
    
    parser.add_argument(
        "base_dir",
        type=str,
        help="Directory containing the dataset CSV file (and where outputs will be saved)"
    )
    
    parser.add_argument(
        "type", 
        type=str,
        choices=["auc", "tpr"],
        help="Type of privacy evaluation: 'auc' for AUC-based or 'tpr' for TPR-based"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the main pipeline
    main(args.base_dir, args.type)