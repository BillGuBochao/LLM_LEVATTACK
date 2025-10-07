import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import numpy as np
from synthcity.plugins import Plugins
from synthcity.utils.serialization import save_to_file, load_from_file


def prepare_dataset(file_path: str, output_dir: str, digits=5):
    """
    Prepare a single dataset with 80/20 member/non-member split
    """
    df = pd.read_csv(file_path)
    
    # Get the filename without extension for folder naming
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Use all available data
    df_subset = df.reset_index(drop=True)
    
    # 80/20 split for member/non-member
    member_df, non_member_df = train_test_split(
        df_subset, 
        train_size=0.8, 
        random_state=42
    )
    
    # Create directory for this dataset
    dataset_dir = os.path.join(output_dir, file_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Save member and non-member files
    member_df.to_csv(os.path.join(dataset_dir, "member.csv"), index=False)
    non_member_df.to_csv(os.path.join(dataset_dir, "non_member.csv"), index=False)
    
    print(f"Processed {file_name}: {len(member_df)} members, {len(non_member_df)} non-members")
    print(f"Saved to {dataset_dir}")
    
    return dataset_dir


def train_and_generate_synthetic_data(base_dir: str, multipliers: list = [1, 5, 10]):
    """
    Train both CTGAN, TVAE, GREAT models on member data and generate synthetic samples
    """
    member_path = os.path.join(base_dir, "member.csv")
    if not os.path.exists(member_path):
        raise FileNotFoundError(f"member.csv not found in {base_dir}")
    
    df = pd.read_csv(member_path)
    dataset_name = os.path.basename(base_dir)
    
    # Train CTGAN
    print(f"Training CTGAN on {dataset_name} ({len(df)} samples)")
    try:
        # Initialize CTGAN plugin
        ctgan_model = Plugins().get("ctgan")
        
        # Fit the model
        ctgan_model.fit(df)
        
        # Create directory for CTGAN synthetic data
        ctgan_synth_dir = os.path.join(base_dir, "synth", "ctgan")
        os.makedirs(ctgan_synth_dir, exist_ok=True)
        
        # Save the trained model
        
        # Generate synthetic data with different multipliers
        for m in multipliers:
            sample_size = int(len(df) * m)
            synthetic_df = ctgan_model.generate(count=sample_size)
            synthetic_df.dataframe().to_csv(os.path.join(ctgan_synth_dir, f"{m}x.csv"), index=False)
            print(f"Generated CTGAN {m}x synthetic samples ({sample_size} rows) for {dataset_name}")
            
    except Exception as e:
        print(f"Error training CTGAN for {dataset_name}: {str(e)}")
    
    # Train TVAE
    print(f"Training TVAE on {dataset_name} ({len(df)} samples)")
    try:
        # Initialize TVAE plugin
        tvae_model = Plugins().get("tvae")
        
        # Fit the model
        tvae_model.fit(df)
        
        # Create directory for TVAE synthetic data
        tvae_synth_dir = os.path.join(base_dir, "synth", "tvae")
        os.makedirs(tvae_synth_dir, exist_ok=True)
        
        # Save the trained model
        
        # Generate synthetic data with different multipliers
        for m in multipliers:
            sample_size = int(len(df) * m)
            synthetic_df = tvae_model.generate(count=sample_size)
            synthetic_df.dataframe().to_csv(os.path.join(tvae_synth_dir, f"{m}x.csv"), index=False)
            print(f"Generated TVAE {m}x synthetic samples ({sample_size} rows) for {dataset_name}")
            
    except Exception as e:
        print(f"Error training TVAE for {dataset_name}: {str(e)}")
    # Train CTGAN
    print(f"Training GREAT on {dataset_name} ({len(df)} samples)")
    try:
        # Initialize CTGAN plugin
        great_model = Plugins().get("great")
        
        # Fit the model
        great_model.fit(df)
        
        # Create directory for CTGAN synthetic data
        great_synth_dir = os.path.join(base_dir, "synth", "great")
        os.makedirs(great_synth_dir, exist_ok=True)
        
        # Save the trained model
        
        # Generate synthetic data with different multipliers
        for m in multipliers:
            sample_size = int(len(df) * m)
            synthetic_df = great_model.generate(count=sample_size)
            synthetic_df.dataframe().to_csv(os.path.join(great_synth_dir, f"{m}x.csv"), index=False)
            print(f"Generated GREAT {m}x synthetic samples ({sample_size} rows) for {dataset_name}")
            
    except Exception as e:
        print(f"Error training GREAT for {dataset_name}: {str(e)}")
    
    return True


def process_directory(data_dir: str, output_root: str, file_pattern="*.csv"):
    """
    Process all CSV files in a directory
    """
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    processed_datasets = []
    
    # Process each file
    for file_path in csv_files:
        try:
            dataset_dir = prepare_dataset(file_path, output_root)
            processed_datasets.append(dataset_dir)
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            continue
    
    # Train models and generate synthetic data for each processed dataset
    print("\n" + "="*50)
    print("Starting model training and synthetic data generation...")
    print("="*50)
    
    successful_trainings = 0
    for dataset_dir in processed_datasets:
        dataset_name = os.path.basename(dataset_dir)
        print(f"\n--- Processing {dataset_name} ---")
        
        try:
            success = train_and_generate_synthetic_data(dataset_dir)
            if success:
                successful_trainings += 1
                print(f"[SUCCESS] Successfully completed {dataset_name}")
            else:
                print(f"[FAILED] Failed to complete {dataset_name}")
        except Exception as e:
            print(f"[ERROR] Error with {dataset_name}: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successfully processed {successful_trainings}/{len(processed_datasets)} datasets")
    print(f"Results saved to: {output_root}")


if __name__ == "__main__":
    # Configuration
    DATA_DIRECTORY = "data/"  # Directory containing your CSV files
    OUTPUT_ROOT = "experiments/"
    FILE_PATTERN = "*.csv"  # Pattern to match files (e.g., "*.csv", "data_*.csv")
    
    # Create output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Process all files in the directory
    process_directory(DATA_DIRECTORY, OUTPUT_ROOT, FILE_PATTERN)