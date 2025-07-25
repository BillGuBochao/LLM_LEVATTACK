import os
import glob
import pandas as pd
import numpy as np
from sklearn import metrics
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List


def maximum_mean_discrepancy(X_real, X_syn, kernel="rbf"):
    """
    Compute empirical maximum mean discrepancy between two one-hot encoded arrays.
    The lower the result, the more evidence that distributions are the same.
    
    Args:
        X_real: numpy array, ground truth one-hot encoded data (n_samples_real, n_features)
        X_syn: numpy array, synthetic one-hot encoded data (n_samples_syn, n_features)
        kernel: str, "rbf", "linear" or "polynomial"
    
    Returns:
        float: MMD score (0 = distributions are the same, 1 = totally different)
    """
    # Flatten arrays if needed
    X_real_flat = X_real.reshape(len(X_real), -1)
    X_syn_flat = X_syn.reshape(len(X_syn), -1)
   
    if kernel == "linear":
        # MMD using linear kernel (i.e., k(x,y) = <x,y>)
        delta = X_real_flat.mean(axis=0) - X_syn_flat.mean(axis=0)
        score = np.dot(delta, delta.T)
       
    elif kernel == "rbf":
        # MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))
        gamma = 1.0
        XX = metrics.pairwise.rbf_kernel(X_real_flat, X_real_flat, gamma)
        YY = metrics.pairwise.rbf_kernel(X_syn_flat, X_syn_flat, gamma)
        XY = metrics.pairwise.rbf_kernel(X_real_flat, X_syn_flat, gamma)
        score = XX.mean() + YY.mean() - 2 * XY.mean()
       
    elif kernel == "polynomial":
        # MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)
        degree = 2
        gamma = 1
        coef0 = 0
        XX = metrics.pairwise.polynomial_kernel(X_real_flat, X_real_flat, degree, gamma, coef0)
        YY = metrics.pairwise.polynomial_kernel(X_syn_flat, X_syn_flat, degree, gamma, coef0)
        XY = metrics.pairwise.polynomial_kernel(X_real_flat, X_syn_flat, degree, gamma, coef0)
        score = XX.mean() + YY.mean() - 2 * XY.mean()
       
    else:
        raise ValueError(f"Unsupported kernel {kernel}")
    
    return float(score)


def find_synthetic_file(directory: str, specific_file: str) -> Optional[str]:
    """
    Find a specific synthetic data file in the directory.
    
    Args:
        directory: Path to search in
        specific_file: Specific synthetic file name to look for
    
    Returns:
        Path to synthetic file or None if not found
    """
    dir_path = Path(directory)
    
    # Search recursively for the specific file
    matching_files = list(dir_path.rglob(specific_file))
    if matching_files:
        return str(matching_files[0])
    
    return None


def get_all_synthetic_files(directory: str) -> List[str]:
    """
    Get all synthetic files listed in mia_evaluation_summary.csv.
    
    Args:
        directory: Path to search in
    
    Returns:
        List of paths to synthetic files
    
    Raises:
        FileNotFoundError: If mia_evaluation_summary.csv is not found
        ValueError: If required columns are missing
    """
    dir_path = Path(directory)
    synthetic_files = []
    
    # Look for mia_evaluation_summary.csv
    summary_files = list(dir_path.rglob('mia_evaluation_summary.csv'))
    if not summary_files:
        raise FileNotFoundError(f"Could not find mia_evaluation_summary.csv in {directory}")
    
    summary_df = pd.read_csv(summary_files[0])
    
    if 'synth_file' not in summary_df.columns:
        raise ValueError("mia_evaluation_summary.csv must contain 'synth_file' column")
    
    # Get unique synthetic file names
    unique_synth_files = summary_df['synth_file'].unique()
    
    # Find each file
    for synth_file in unique_synth_files:
        file_path = find_synthetic_file(directory, synth_file)
        if file_path:
            synthetic_files.append(file_path)
        else:
            print(f"Warning: Could not find {synth_file}")
    
    if not synthetic_files:
        raise ValueError("No synthetic files found from mia_evaluation_summary.csv")
        
    return synthetic_files


def find_member_file(directory: str) -> Optional[str]:
    """
    Find member data file in the directory or any subdirectory.
    Searches recursively for member.csv.
    
    Args:
        directory: Path to search in
    
    Returns:
        Path to member file or None if not found
    """
    dir_path = Path(directory)
    
    # First try exact match 'member.csv' recursively
    member_files = list(dir_path.rglob('member.csv'))
    if member_files:
        return str(member_files[0])
    
    # Try case-insensitive search recursively
    for file in dir_path.rglob('*.csv'):
        if file.name.lower() == 'member.csv':
            return str(file)
    
    # If still not found, look for any file with 'member' in the name
    # but not 'non_member'
    for file in dir_path.rglob('*.csv'):
        filename_lower = file.name.lower()
        if 'member' in filename_lower and 'non_member' not in filename_lower:
            return str(file)
    
    return None


def prepare_data_for_mmd(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataframes for MMD calculation by aligning columns and converting to numpy arrays.
    
    Args:
        real_df: Real data dataframe
        syn_df: Synthetic data dataframe
    
    Returns:
        Tuple of (real_array, synthetic_array) ready for MMD
    """
    # Ensure both dataframes have the same columns
    common_cols = sorted(set(real_df.columns) & set(syn_df.columns))
    
    if len(common_cols) == 0:
        raise ValueError("No common columns found between real and synthetic data")
    
    # Align columns
    real_aligned = real_df[common_cols]
    syn_aligned = syn_df[common_cols]
    
    # Convert to numpy arrays
    real_array = real_aligned.values
    syn_array = syn_aligned.values
    
    return real_array, syn_array


def one_hot_encode_data(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    One-hot encode categorical columns in both dataframes.
    
    Args:
        real_df: Real data dataframe
        syn_df: Synthetic data dataframe
    
    Returns:
        Tuple of (real_encoded, synthetic_encoded) arrays
    """
    from sklearn.preprocessing import OneHotEncoder
    
    # Combine data to ensure consistent encoding
    combined_df = pd.concat([real_df, syn_df], keys=['real', 'syn'])
    
    # Identify categorical columns
    categorical_cols = combined_df.select_dtypes(include=['object', 'category']).columns
    numerical_cols = combined_df.select_dtypes(include=['number']).columns
    
    if len(categorical_cols) > 0:
        # One-hot encode categorical columns
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Fit on combined data
        encoder.fit(combined_df[categorical_cols])
        
        # Transform separately
        real_categorical = encoder.transform(real_df[categorical_cols])
        syn_categorical = encoder.transform(syn_df[categorical_cols])
        
        # Combine with numerical columns
        if len(numerical_cols) > 0:
            real_encoded = np.hstack([real_df[numerical_cols].values, real_categorical])
            syn_encoded = np.hstack([syn_df[numerical_cols].values, syn_categorical])
        else:
            real_encoded = real_categorical
            syn_encoded = syn_categorical
    else:
        # Only numerical columns
        real_encoded = real_df[numerical_cols].values
        syn_encoded = syn_df[numerical_cols].values
    
    return real_encoded, syn_encoded



def update_mia_summary_with_mmd(
   directory: str,
   kernel: str = "rbf",
   encode_categorical: bool = True,
   verbose: bool = True,
   mmd_column_name: str = None
) -> pd.DataFrame:
   """
   Update mia_evaluation_summary.csv with MMD scores for each synthetic file.
   
   Args:
       directory: Directory containing the data files
       kernel: Kernel to use for MMD ("linear", "rbf", "polynomial")
       encode_categorical: Whether to one-hot encode categorical variables
       verbose: Whether to print progress information
       mmd_column_name: Name for the MMD column (default: "mmd_{kernel}")
   
   Returns:
       Updated DataFrame
   """
   dir_path = Path(directory)
   
   # Find mia_evaluation_summary.csv
   summary_files = list(dir_path.rglob('mia_evaluation_summary.csv'))
   if not summary_files:
       raise FileNotFoundError("Could not find mia_evaluation_summary.csv")
   
   summary_path = summary_files[0]
   summary_df = pd.read_csv(summary_path)
   
   if verbose:
       print(f"Found summary file: {summary_path}")
       print(f"Processing {len(summary_df)} rows...")
   
   # Find member file once
   member_file = find_member_file(directory)
   if not member_file:
       raise FileNotFoundError("Could not find member.csv file")
   
   # Load member data once
   real_df = pd.read_csv(member_file)
   
   # Set column name for MMD scores
   if mmd_column_name is None:
       mmd_column_name = f"mmd_{kernel}"
   
   # Initialize MMD column
   summary_df[mmd_column_name] = np.nan
   
   # Process each unique synthetic file
   unique_synth_files = summary_df['synth_file'].unique()
   
   for synth_file_name in unique_synth_files:
       if verbose:
           print(f"\nProcessing: {synth_file_name}")
       
       try:
           # Find the synthetic file
           syn_file_path = find_synthetic_file(directory, synth_file_name)
           if not syn_file_path:
               print(f"Warning: Could not find {synth_file_name}")
               continue
           
           # Load synthetic data
           syn_df = pd.read_csv(syn_file_path)
           
           # Find common columns
           common_cols = sorted(set(real_df.columns) & set(syn_df.columns))
           real_aligned = real_df[common_cols]
           syn_aligned = syn_df[common_cols]
           
           # Encode if needed
           if encode_categorical:
               real_array, syn_array = one_hot_encode_data(real_aligned, syn_aligned)
           else:
               real_array, syn_array = prepare_data_for_mmd(real_aligned, syn_aligned)
           
           # Compute MMD
           score = maximum_mean_discrepancy(real_array, syn_array, kernel=kernel)
           
           # Update all rows with this synthetic file
           mask = summary_df['synth_file'] == synth_file_name
           summary_df.loc[mask, mmd_column_name] = score
           
           
               
       except Exception as e:
           print(f"Error processing {synth_file_name}: {e}")
           # Leave as NaN for error cases
   
   # Save the updated summary
   summary_df.to_csv(summary_path, index=False)

   
   return summary_df


def add_all_kernel_scores_to_summary(
   directory: str,
   encode_categorical: bool = True,
   verbose: bool = True
) -> pd.DataFrame:
   """
   Add MMD scores for all three kernels to mia_evaluation_summary.csv.
   
   Args:
       directory: Directory containing the data files
       encode_categorical: Whether to one-hot encode categorical variables
       verbose: Whether to print progress information
   
   Returns:
       Updated DataFrame with all kernel scores
   """
   kernels = ['linear', 'rbf', 'polynomial']
   
   # Process each kernel
   for kernel in kernels:
       if verbose:
           print(f"\n{'='*60}")
           print(f"Computing MMD with {kernel} kernel")
           print('='*60)
       
       summary_df = update_mia_summary_with_mmd(
           directory=directory,
           kernel=kernel,
           encode_categorical=encode_categorical,
           verbose=verbose,
       )
   
   return summary_df