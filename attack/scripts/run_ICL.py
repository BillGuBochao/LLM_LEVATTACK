import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from synth_mia.attackers import (
    gen_lra, dcr, dpi, logan, dcr_diff, domias,
    mc, density_estimate, local_neighborhood, classifier, levDCR
)
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm
from synth_mia.utils import tabular_preprocess
from datetime import datetime
import re
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

def count_continuous_columns(df):
    """Count the number of continuous (numeric) columns in the dataframe."""
    continuous_count = 0
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's truly continuous (not just integers that could be categorical)
            unique_vals = df[col].nunique()
            total_vals = len(df[col].dropna())
            # Consider it continuous if it has many unique values relative to total
            if unique_vals > min(20, total_vals * 0.1):
                continuous_count += 1
            # Also check if it contains floating point numbers
            elif df[col].dtype in ['float64', 'float32']:
                continuous_count += 1
    return continuous_count

def count_digits_in_row(row):
    """Count total digits in a row by converting to string and counting digit characters."""
    row_str = ''.join(str(val) for val in row if pd.notna(val))
    return len(re.findall(r'\d', row_str))

def compute_avg_digits_per_row(df):
    """Compute the average number of digits per row in the dataframe."""
    digit_counts = df.apply(count_digits_in_row, axis=1)
    return digit_counts.mean()

def filter_numeric_columns(df1, df2, df3):
    """
    Filter three dataframes to keep only common numeric columns.
    Returns filtered dataframes and the list of common numeric columns.
    """
    # Get numeric columns from each dataframe
    numeric_cols_1 = df1.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_2 = df2.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_3 = df3.select_dtypes(include=[np.number]).columns.tolist()

    # Find intersection of all numeric columns
    common_numeric_cols = list(set(numeric_cols_1) & set(numeric_cols_2) & set(numeric_cols_3))

    # Filter dataframes to keep only common numeric columns
    df1_filtered = df1[common_numeric_cols]
    df2_filtered = df2[common_numeric_cols]
    df3_filtered = df3[common_numeric_cols]

    print(f"[DEBUG] Found {len(common_numeric_cols)} common numeric columns: {common_numeric_cols}")

    return df1_filtered, df2_filtered, df3_filtered, common_numeric_cols

def get_attackers():
    """Instantiate membership-inference attackers with default hyperparameters."""
    return [
        dcr(),
        mc(),
        density_estimate(hyper_parameters={"method": "kde"}),
        levDCR()
    ]

def process_dataset_genmia(dataset_name: str, generator_name: str, global_results: list):
    """
    For each CSV in LTM_data/LTM_real_data/{dataset_name}/train/:
      - split 50/50 into mem vs ref (seed=42)
      - load non_mem as the single CSV under LTM_data/LTM_real_data/{dataset_name}/test/
      - load matching synthetic CSV
      - run membership-inference attacks directly on the data
      - compute ROC AUC
      - add results to global_results list
    """
    # Paths
    train_folder = os.path.join("LTM_data", "LTM_real_data", dataset_name, "train")
    test_folder = os.path.join("LTM_data", "LTM_real_data", dataset_name, "test")
    synth_folder = os.path.join(
        "LTM_data", "LTM_synthetic_data",
        f"LTM_{generator_name}_synthetic_data",
        f"synth_{dataset_name}"
    )

    # Check folders
    if not os.path.isdir(train_folder):
        print(f"[WARN] Train folder not found: {train_folder}")
        return
    if not os.path.isdir(test_folder):
        print(f"[WARN] Test folder not found: {test_folder}")
        return
    if not os.path.isdir(synth_folder):
        print(f"[WARN] Synthetic folder not found: {synth_folder}")
        return

    # Identify the one test CSV
    test_files = [f for f in os.listdir(test_folder) if f.lower().endswith('.csv')]
    if len(test_files) != 1:
        print(f"[WARN] Expected exactly one test CSV in {test_folder}, found {len(test_files)}")
        return

    non_mem_path = os.path.join(test_folder, test_files[0])
    attackers = get_attackers()

    # Loop over each train CSV
    for train_fname in sorted(os.listdir(train_folder)):
        if not train_fname.lower().endswith('.csv'):
            continue

        base = os.path.splitext(train_fname)[0]
        real_path = os.path.join(train_folder, train_fname)
        synth_fname = f"{base}_{generator_name}_default_1.csv"
        synth_path = os.path.join(synth_folder, synth_fname)

        if not os.path.isfile(synth_path):
            print(f"[WARN] Missing synthetic for {base}: {synth_path}")
            continue

        try:
            # 1) Load real data
            df = pd.read_csv(real_path, on_bad_lines='skip')

            # 2) Load non_mem data
            non_mem_df = pd.read_csv(non_mem_path, on_bad_lines='skip')

            # 3) Load synthetic data
            synth_df = pd.read_csv(synth_path, on_bad_lines='skip')

            # 4) Filter to keep only common numeric columns
            df_numeric, non_mem_df_numeric, synth_df_numeric, common_cols = filter_numeric_columns(
                df, non_mem_df, synth_df
            )

            if len(common_cols) == 0:
                print(f"[WARN] No common numeric columns found for {base}, skipping...")
                continue

            # Use filtered dataframes for string conversion
            member_rows = df_numeric.apply(row_to_str, axis=1).tolist()
            non_member_rows = non_mem_df_numeric.apply(row_to_str, axis=1).tolist()

            test_rows = member_rows + non_member_rows
            true_labels = [1] * len(member_rows) + [0] * len(non_member_rows)

            synth_rows = synth_df_numeric.apply(row_to_str, axis=1).tolist()
            min_distances = np.array(compute_min_distances(test_rows, synth_rows))

            # 5) Compute additional statistics on original dataframe (before filtering)
            continuous_cols_count = count_continuous_columns(df)
            avg_digits_per_row = compute_avg_digits_per_row(df)

            # 6) Preprocess data using numeric-filtered dataframes
            mem, non_mem, synth, transformer = tabular_preprocess(
                df_numeric, non_mem_df_numeric, synth_df_numeric, fit_target='synth', categorical_encoding='ordinal'
            )

            print(f"[DEBUG] {base} shapes – mem: {mem.shape}, non_mem: {non_mem.shape}, synth: {synth.shape}")
            print(f"[DEBUG] {base} stats – continuous cols: {continuous_cols_count}, avg digits/row: {avg_digits_per_row:.2f}")
            print(f"[DEBUG] {base} using {len(common_cols)} numeric columns")

            # 7) Run attacks and compute ROC AUC
            for attacker in attackers:
                try:
                    if attacker.name == "levDCR":
                        df_array = df_numeric.apply(row_to_str, axis=1).values.reshape(-1, 1)
                        non_mem_array = non_mem_df_numeric.apply(row_to_str, axis=1).values.reshape(-1, 1)
                        synth_array = synth_df_numeric.apply(row_to_str, axis=1).values.reshape(-1, 1)
                        scores, true_labels = attacker.attack(df_array, non_mem_array, synth_array)
                    else:
                        scores, true_labels = attacker.attack(mem, non_mem, synth)

                    # Evaluate the attack
                    eval_results = attacker.eval(true_labels, scores, metrics=['roc'])
                    print(f"[INFO] {attacker.name} on {base}: {eval_results}")

                    # Store results with metadata and debug shape info
                    result_row = {
                        'dataset': dataset_name,
                        'generator': generator_name,
                        'base': base,
                        'attacker': attacker.name,
                        'mem_shape_rows': mem.shape[0],
                        'mem_shape_cols': mem.shape[1],
                        'non_mem_shape_rows': non_mem.shape[0],
                        'non_mem_shape_cols': non_mem.shape[1],
                        'synth_shape_rows': synth.shape[0],
                        'synth_shape_cols': synth.shape[1],
                        'continuous_columns': continuous_cols_count,
                        'numeric_columns_used': len(common_cols),
                        'avg_digits_per_row': round(avg_digits_per_row, 4),
                        'status': 'success'
                    }
                    # Add all metrics from eval_results
                    result_row.update(eval_results)
                    global_results.append(result_row)

                except Exception as e:
                    print(f"[ERROR] {attacker.name} on {base}: {e}")
                    # Store error result with shape info
                    error_row = {
                        'dataset': dataset_name,
                        'generator': generator_name,
                        'base': base,
                        'attacker': attacker.name,
                        'mem_shape_rows': mem.shape[0] if 'mem' in locals() else 0,
                        'mem_shape_cols': mem.shape[1] if 'mem' in locals() else 0,
                        'non_mem_shape_rows': non_mem.shape[0] if 'non_mem' in locals() else 0,
                        'non_mem_shape_cols': non_mem.shape[1] if 'non_mem' in locals() else 0,
                        'synth_shape_rows': synth.shape[0] if 'synth' in locals() else 0,
                        'synth_shape_cols': synth.shape[1] if 'synth' in locals() else 0,
                        'continuous_columns': continuous_cols_count if 'continuous_cols_count' in locals() else 0,
                        'numeric_columns_used': len(common_cols) if 'common_cols' in locals() else 0,
                        'avg_digits_per_row': round(avg_digits_per_row, 4) if 'avg_digits_per_row' in locals() else 0,
                        'status': 'error',
                        'error': str(e)
                    }
                    global_results.append(error_row)

        except Exception as e:
            print(f"[ERROR] Processing {base}: {e}")
            # Store dataset-level error
            error_row = {
                'dataset': dataset_name,
                'generator': generator_name,
                'base': base,
                'attacker': 'N/A',
                'status': 'dataset_error',
                'error': str(e)
            }
            global_results.append(error_row)
            continue

def get_all_datasets():
    """Get all dataset names from the real data directory."""
    real_data_root = os.path.join("data/LTM_data", "data/LTM_real_data")
    if not os.path.isdir(real_data_root):
        print(f"[ERROR] Real data directory not found: {real_data_root}")
        return []

    datasets = []
    for item in os.listdir(real_data_root):
        item_path = os.path.join(real_data_root, item)
        if os.path.isdir(item_path):
            datasets.append(item)

    return sorted(datasets)

def main():
    # Get all available datasets
    datasets = get_all_datasets()
    generators = ["gpt-4o-mini-2024-07-18", "tabpfn","gpt-4o-mini-2024-07-18"]#,]

    print(f"[INFO] Found {len(datasets)} datasets: {datasets}")
    print(f"[INFO] Processing generators: {generators}")

    # Initialize global results storage
    global_results = []

    # Process each combination
    for dataset_name in datasets:
        for generator_name in generators:
            print(f"\n[INFO] Processing dataset: {dataset_name}, generator: {generator_name}")
            try:
                process_dataset_genmia(dataset_name, generator_name, global_results)
            except Exception as e:
                print(f"[ERROR] Failed to process {dataset_name} with {generator_name}: {e}")
                # Store top-level error
                error_row = {
                    'dataset': dataset_name,
                    'generator': generator_name,
                    'base': 'N/A',
                    'attacker': 'N/A',
                    'status': 'processing_error',
                    'error': str(e)
                }
                global_results.append(error_row)

    print(f"\n[INFO] Processing complete! Found {len(global_results)} results.")

    # Save results to CSV
    if global_results:
        # Create results DataFrame
        results_df = pd.DataFrame(global_results)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"mia_analysis_results_numeric_only_{timestamp}.csv"

        # Save to CSV
        results_df.to_csv(output_filename, index=False)
        print(f"[INFO] Results saved to: {output_filename}")

        # Print summary statistics
        print(f"\n[SUMMARY] Total results: {len(results_df)}")
        if 'status' in results_df.columns:
            status_counts = results_df['status'].value_counts()
            print("[SUMMARY] Status breakdown:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")

        if 'attacker' in results_df.columns:
            attacker_counts = results_df['attacker'].value_counts()
            print("[SUMMARY] Attacker breakdown:")
            for attacker, count in attacker_counts.items():
                print(f"  {attacker}: {count}")

        # Show successful ROC AUC results if available
        successful_results = results_df[results_df['status'] == 'success']
        if not successful_results.empty and 'roc' in successful_results.columns:
            print(f"\n[SUMMARY] ROC AUC Statistics (successful runs only):")
            print(f"  Mean ROC AUC: {successful_results['roc'].mean():.4f}")
            print(f"  Std ROC AUC: {successful_results['roc'].std():.4f}")
            print(f"  Min ROC AUC: {successful_results['roc'].min():.4f}")
            print(f"  Max ROC AUC: {successful_results['roc'].max():.4f}")

        # Show new statistics if available
        if not successful_results.empty:
            if 'continuous_columns' in successful_results.columns:
                print(f"\n[SUMMARY] Continuous Columns Statistics:")
                print(f"  Mean continuous columns: {successful_results['continuous_columns'].mean():.2f}")
                print(f"  Std continuous columns: {successful_results['continuous_columns'].std():.2f}")
                print(f"  Min continuous columns: {successful_results['continuous_columns'].min()}")
                print(f"  Max continuous columns: {successful_results['continuous_columns'].max()}")

            if 'numeric_columns_used' in successful_results.columns:
                print(f"\n[SUMMARY] Numeric Columns Used Statistics:")
                print(f"  Mean numeric columns used: {successful_results['numeric_columns_used'].mean():.2f}")
                print(f"  Std numeric columns used: {successful_results['numeric_columns_used'].std():.2f}")
                print(f"  Min numeric columns used: {successful_results['numeric_columns_used'].min()}")
                print(f"  Max numeric columns used: {successful_results['numeric_columns_used'].max()}")

            if 'avg_digits_per_row' in successful_results.columns:
                print(f"\n[SUMMARY] Average Digits per Row Statistics:")
                print(f"  Mean avg digits/row: {successful_results['avg_digits_per_row'].mean():.4f}")
                print(f"  Std avg digits/row: {successful_results['avg_digits_per_row'].std():.4f}")
                print(f"  Min avg digits/row: {successful_results['avg_digits_per_row'].min():.4f}")
                print(f"  Max avg digits/row: {successful_results['avg_digits_per_row'].max():.4f}")
    else:
        print("[WARN] No results to save.")

if __name__ == "__main__":
    main()
