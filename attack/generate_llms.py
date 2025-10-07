import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from datasets import load_dataset
import pandas as pd
import numpy as np
from huggingface_hub import login
import os
from sklearn.model_selection import train_test_split
import realtabformer_non_gpt2.src.single_table_realtabformer_generalized as rtf_generalized


def prepare_dataset(file_path: str, output_dir: str):
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

def get_model_short_name(model_name: str) -> str:
    """
    Convert full model name to short directory-friendly name
    """
    model_mapping = {
        "meta-llama/Llama-3.2-1B": "llama-3.2-1b",
        "meta-llama/Llama-3.2-3B": "llama-3.2-3b",
        "unsloth/Qwen2.5-3B": "qwen2.5-3b",
        "mistralai/Mistral-7B-v0.1": "mistral-7b-v0.1",
        "openai-community/gpt2": "gpt2"
    }
    return model_mapping.get(model_name, model_name.replace("/", "_").lower())

def train_and_generate_synthetic_data(base_dir: str, llm_name: str, multipliers: list = [1, 5, 10]):
    """
    Train REaLTabFormer model on member data and generate synthetic samples
    """
    member_path = os.path.join(base_dir, "member.csv")
    if not os.path.exists(member_path):
        raise FileNotFoundError(f"member.csv not found in {base_dir}")
    
    df = pd.read_csv(member_path)
    dataset_name = os.path.basename(base_dir)
    model_short_name = get_model_short_name(llm_name)

    # Train REaLTabFormer
    print(f"Training REaLTabFormer with {llm_name} on {dataset_name} ({len(df)} samples)")
    try:
        rtf_model = rtf_generalized.REaLTabFormer(
            model_type = "tabular",
            llm_name = llm_name,
            gradient_accumulation_steps = 4,
            logging_steps = 1000,
            epochs = 200, 
            batch_size = 8, 
            train_size = 0.8
        )
        
        rtf_model.fit(df)
        
        rtf_synth_dir = os.path.join(base_dir, "synth", model_short_name)
        os.makedirs(rtf_synth_dir, exist_ok=True)
        
        for m in multipliers:
            sample_size = int(len(df) * m)
            synthetic_df = rtf_model.sample(n_samples=sample_size)
            synthetic_df.to_csv(os.path.join(rtf_synth_dir, f"{m}x.csv"), index=False)
            print(f"Generated {m}x synthetic samples ({sample_size} rows) for {dataset_name} with {llm_name}")
            
    except Exception as e:
        print(f"Error training REaLTabFormer for {dataset_name} with {llm_name}: {str(e)}")
        return False
       
    return True

def process_directory(data_dir: str, output_root: str, llm_names: list, file_pattern="*.csv"):
    """
    Process all CSV files in a directory for multiple LLM models
    """
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, file_pattern))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    print(f"\nWill train with {len(llm_names)} models:")
    for llm in llm_names:
        print(f"  - {llm}")
    
    # Process each LLM model
    for llm_name in llm_names:
        model_short_name = get_model_short_name(llm_name)
        model_output_root = os.path.join(output_root, f"experiment_{model_short_name}")
        os.makedirs(model_output_root, exist_ok=True)
        
        print("\n" + "="*70)
        print(f"PROCESSING MODEL: {llm_name}")
        print(f"Output directory: {model_output_root}")
        print("="*70)
        
        processed_datasets = []
        
        # Process each file for this model
        for file_path in csv_files:
            try:
                dataset_dir = prepare_dataset(file_path, model_output_root)
                processed_datasets.append(dataset_dir)
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
                continue
        
        # Train model and generate synthetic data for each processed dataset
        print("\n" + "-"*70)
        print(f"Starting training and synthetic data generation for {llm_name}...")
        print("-"*70)
        
        successful_trainings = 0
        for dataset_dir in processed_datasets:
            dataset_name = os.path.basename(dataset_dir)
            print(f"\n--- Processing {dataset_name} with {llm_name} ---")
            
            try:
                success = train_and_generate_synthetic_data(dataset_dir, llm_name)
                if success:
                    successful_trainings += 1
                    print(f"[SUCCESS] Successfully completed {dataset_name} with {llm_name}")
                else:
                    print(f"[FAILED] Failed to complete {dataset_name} with {llm_name}")
            except Exception as e:
                print(f"[ERROR] Error with {dataset_name} and {llm_name}: {str(e)}")
        
        print(f"\n{'-'*70}")
        print(f"Model {llm_name} complete!")
        print(f"Successfully processed {successful_trainings}/{len(processed_datasets)} datasets")
        print(f"Results saved to: {model_output_root}")
        print("-"*70)
    
    print(f"\n{'='*70}")
    print(f"ALL PROCESSING COMPLETE!")
    print(f"Results saved to: {output_root}")
    print("="*70)

if __name__ == "__main__":
    # Configuration
    DATA_DIRECTORY = "data/"  # Directory containing your CSV files
    OUTPUT_ROOT = "experiments/"  # Base output directory
    FILE_PATTERN = "*.csv"  # Pattern to match files
    
    # List of LLM models to train
    LLM_NAMES = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B",
        "unsloth/Qwen2.5-3B",
        "mistralai/Mistral-7B-v0.1",
        "openai-community/gpt2"
    ]
    
    # Create output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Process all files in the directory for all models
    process_directory(DATA_DIRECTORY, OUTPUT_ROOT, LLM_NAMES, FILE_PATTERN)