import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def create_comparison_plot(summary_auc_path: str, vanilla_summary_auc_path: str, 
                          output_path: str = None, dataset_name: str = None):
    """
    Create comparison visualization between tendency-based and vanilla REaLTabFormer results.
    
    Args:
        summary_auc_path: Path to the tendency-based summary AUC CSV file
        vanilla_summary_auc_path: Path to the vanilla summary AUC CSV file
        output_path: Path to save the plot (optional, will show plot if None)
        dataset_name: Name of dataset for plot title (optional, inferred from path if None)
    
    Returns:
        None
    """
    
    # Read the CSV files
    try:
        tendency_df = pd.read_csv(summary_auc_path)
        vanilla_df = pd.read_csv(vanilla_summary_auc_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        return
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return
    
    # Infer dataset name from path if not provided
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.dirname(summary_auc_path))
    
    # Sort by synthetic_size for consistent plotting
    tendency_df = tendency_df.sort_values('synthetic_size')
    vanilla_df = vanilla_df.sort_values('synthetic_size')
    
    # Get synthetic sizes (assuming both datasets have same sizes)
    synthetic_sizes = tendency_df['synthetic_size'].values
    
    # Extract metrics
    tendency_auc = tendency_df['final_auc'].values
    vanilla_auc = vanilla_df['auc_roc'].values

    tendency_mmd = tendency_df['mmd_score'].values
    vanilla_mmd = vanilla_df['mmd_score'].values

    # Create the figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Bar width and positions
    bar_width = 0.35
    x_pos = np.arange(len(synthetic_sizes))
    
    # Colors
    tendency_color = 'skyblue'
    vanilla_color = 'lightcoral'
    
    # Subplot 1: AUC comparison
    ax1 = axes[0]
    bars1_vanilla = ax1.bar(x_pos - bar_width/2, vanilla_auc, bar_width,
                           label='VANILLA', color=vanilla_color, alpha=0.8)
    bars1_tendency = ax1.bar(x_pos + bar_width/2, tendency_auc, bar_width,
                            label='TLP', color=tendency_color, alpha=0.8)

    ax1.set_xlabel('Synthetic Dataset Size', fontweight='bold')
    ax1.set_ylabel('LevAtt AUC-ROC', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(synthetic_sizes)
    ax1.set_ylim(bottom=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, ax, format_str='.3f'):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:{format_str}}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1_vanilla, ax1)
    add_value_labels(bars1_tendency, ax1)

    # Subplot 2: MMD comparison
    ax2 = axes[1]
    bars2_vanilla = ax2.bar(x_pos - bar_width/2, vanilla_mmd, bar_width,
                           label='VANILLA', color=vanilla_color, alpha=0.8)
    bars2_tendency = ax2.bar(x_pos + bar_width/2, tendency_mmd, bar_width,
                            label='TLP', color=tendency_color, alpha=0.8)

    ax2.set_xlabel('Synthetic Dataset Size', fontweight='bold')
    ax2.set_ylabel('MMD Score', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(synthetic_sizes)
    # Set y-axis upper limit to make bars relatively shorter
    max_mmd = max(np.max(vanilla_mmd), np.max(tendency_mmd))
    ax2.set_ylim(0, max_mmd * 2.3)  # Scale up the limit to make bars appear shorter
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    add_value_labels(bars2_vanilla, ax2, '.4f')
    add_value_labels(bars2_tendency, ax2, '.4f')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_path}")
    else:
        plt.show()


def create_comparison_plot_tpr(summary_tpr_path: str, vanilla_summary_tpr_path: str, 
                              output_path: str = None, dataset_name: str = None):
    """
    Create comparison visualization between tendency-based and vanilla REaLTabFormer results for TPR metrics.
    
    Args:
        summary_tpr_path: Path to the tendency-based summary TPR CSV file
        vanilla_summary_tpr_path: Path to the vanilla summary TPR CSV file
        output_path: Path to save the plot (optional, will show plot if None)
        dataset_name: Name of dataset for plot title (optional, inferred from path if None)
    
    Returns:
        None
    """
    
    # Read the CSV files
    try:
        tendency_df = pd.read_csv(summary_tpr_path)
        vanilla_df = pd.read_csv(vanilla_summary_tpr_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        return
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return
    
    # Infer dataset name from path if not provided
    if dataset_name is None:
        dataset_name = os.path.basename(os.path.dirname(summary_tpr_path))
    
    # Sort by synthetic_size for consistent plotting
    tendency_df = tendency_df.sort_values('synthetic_size')
    vanilla_df = vanilla_df.sort_values('synthetic_size')
    
    # Get synthetic sizes (assuming both datasets have same sizes)
    synthetic_sizes = tendency_df['synthetic_size'].values
    
    # Extract metrics
    tendency_tpr = tendency_df['final_tpr'].values
    vanilla_tpr = vanilla_df['tpr_at_fpr_0.1'].values

    tendency_mmd = tendency_df['mmd_score'].values
    vanilla_mmd = vanilla_df['mmd_score'].values

    # Create the figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Bar width and positions
    bar_width = 0.35
    x_pos = np.arange(len(synthetic_sizes))
    
    # Colors
    tendency_color = 'skyblue'
    vanilla_color = 'lightcoral'
    
    # Subplot 1: TPR comparison
    ax1 = axes[0]
    bars1_vanilla = ax1.bar(x_pos - bar_width/2, vanilla_tpr, bar_width,
                           label='VANILLA', color=vanilla_color, alpha=0.8)
    bars1_tendency = ax1.bar(x_pos + bar_width/2, tendency_tpr, bar_width,
                            label='TLP', color=tendency_color, alpha=0.8)

    ax1.set_xlabel('Synthetic Dataset Size', fontweight='bold')
    ax1.set_ylabel('TPR@FPR=0.1', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(synthetic_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def add_value_labels(bars, ax, format_str='.3f'):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:{format_str}}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1_tendency, ax1)
    add_value_labels(bars1_vanilla, ax1)

    # Subplot 2: MMD comparison
    ax2 = axes[1]
    bars2_vanilla = ax2.bar(x_pos - bar_width/2, vanilla_mmd, bar_width,
                           label='VANILLA', color=vanilla_color, alpha=0.8)
    bars2_tendency = ax2.bar(x_pos + bar_width/2, tendency_mmd, bar_width,
                            label='TLP', color=tendency_color, alpha=0.8)

    ax2.set_xlabel('Synthetic Dataset Size', fontweight='bold')
    ax2.set_ylabel('MMD Score', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(synthetic_sizes)
    # Set y-axis upper limit to make bars relatively shorter
    max_mmd = max(np.max(vanilla_mmd), np.max(tendency_mmd))
    ax2.set_ylim(0, max_mmd * 2.3)  # Scale up the limit to make bars appear shorter
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    add_value_labels(bars2_vanilla, ax2, '.4f')
    add_value_labels(bars2_tendency, ax2, '.4f')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save or display the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"TPR comparison plot saved to: {output_path}")
    else:
        plt.show()


def compare_experiment_results_auc(experiment_dir: str, show_plot: bool = True, save_plot: bool = True):
    """
    Convenience function to create AUC comparison plot for an experiment directory.
    
    Args:
        experiment_dir: Directory containing the summary CSV files
        show_plot: Whether to display the plot
        save_plot: Whether to save the plot to file
    
    Returns:
        None
    """
    
    # Construct file paths
    dataset_name = os.path.basename(experiment_dir)
    summary_dir = os.path.join(experiment_dir, "summary")
    summary_auc_path = os.path.join(summary_dir, f"summary_auc_{dataset_name}.csv")
    vanilla_summary_auc_path = os.path.join(summary_dir, f"vanilla_summary_auc_{dataset_name}.csv")
    
    # Check if files exist
    if not os.path.exists(summary_auc_path):
        print(f"Error: {summary_auc_path} does not exist")
        return
    if not os.path.exists(vanilla_summary_auc_path):
        print(f"Error: {vanilla_summary_auc_path} does not exist")
        return
    
    # Determine output path
    output_path = None
    if save_plot:
        visualization_dir = os.path.join(experiment_dir, "visualization")
        os.makedirs(visualization_dir, exist_ok=True)
        output_path = os.path.join(visualization_dir, f"comparison_auc_plot_{dataset_name}.png")
    
    # Create the plot
    create_comparison_plot(summary_auc_path, vanilla_summary_auc_path, output_path, dataset_name)
    
    if show_plot and not save_plot:
        plt.show()


def compare_experiment_results_tpr(experiment_dir: str, show_plot: bool = True, save_plot: bool = True):
    """
    Convenience function to create TPR comparison plot for an experiment directory.
    
    Args:
        experiment_dir: Directory containing the summary CSV files
        show_plot: Whether to display the plot
        save_plot: Whether to save the plot to file
    
    Returns:
        None
    """
    
    # Construct file paths
    dataset_name = os.path.basename(experiment_dir)
    summary_dir = os.path.join(experiment_dir, "summary")
    summary_tpr_path = os.path.join(summary_dir, f"summary_tpr_{dataset_name}.csv")
    vanilla_summary_tpr_path = os.path.join(summary_dir, f"vanilla_summary_tpr_{dataset_name}.csv")
    
    # Check if files exist
    if not os.path.exists(summary_tpr_path):
        print(f"Error: {summary_tpr_path} does not exist")
        return
    if not os.path.exists(vanilla_summary_tpr_path):
        print(f"Error: {vanilla_summary_tpr_path} does not exist")
        return
    
    # Determine output path
    output_path = None
    if save_plot:
        visualization_dir = os.path.join(experiment_dir, "visualization")
        os.makedirs(visualization_dir, exist_ok=True)
        output_path = os.path.join(visualization_dir, f"comparison_tpr_plot_{dataset_name}.png")
    
    # Create the plot
    create_comparison_plot_tpr(summary_tpr_path, vanilla_summary_tpr_path, output_path, dataset_name)
    
    if show_plot and not save_plot:
        plt.show()


if __name__ == "__main__":
    # Example usage - replace with your experiment directory
    experiment_directories = [
        "experimentCASP",
        # "experiment7",
        # "experimentCALIFORNIA"
    ]
    
    for exp_dir in experiment_directories:
        if os.path.exists(exp_dir):
            print(f"\nCreating AUC comparison plot for {exp_dir}...")
            compare_experiment_results_auc(exp_dir, show_plot=False, save_plot=True)
            print(f"Creating TPR comparison plot for {exp_dir}...")
            compare_experiment_results_tpr(exp_dir, show_plot=False, save_plot=True)
        else:
            print(f"Directory {exp_dir} does not exist, skipping...")
    
    print("\nComparison visualization completed!")