import os
import pandas as pd
from sklearn.model_selection import train_test_split
from realtabformer import REaLTabFormer

def prepare_casp_data(casp_path: str, output_dir: str, n_rows=5000, digits=5):
    df = pd.read_csv(casp_path)

    if len(df) < n_rows * 2:
        raise ValueError(f"CASP.csv must contain at least {n_rows * 2} rows.")

    # Use global seed set in main.py for reproducibility
    df_subset = df.sample(n=n_rows * 2).reset_index(drop=True)
    member_df, non_member_df = train_test_split(df_subset, train_size=n_rows)

    casp_dir = os.path.join(output_dir, "casp")
    os.makedirs(casp_dir, exist_ok=True)

    member_df.to_csv(os.path.join(casp_dir, "member.csv"), index=False)
    non_member_df.to_csv(os.path.join(casp_dir, "non_member.csv"), index=False)
    print(f"Saved member and non_member CSVs to {casp_dir}")

    return casp_dir

def train_and_generate_synthetic_data(base_dir: str, multipliers: list = [0.5, 1, 2, 3]):
    member_path = os.path.join(base_dir, "member.csv")
    if not os.path.exists(member_path):
        raise FileNotFoundError(f"member.csv not found in {base_dir}")

    df = pd.read_csv(member_path)
    print(f"Training REaLTabFormer on {member_path}")

    rtf_model = REaLTabFormer(
        model_type="tabular",
        gradient_accumulation_steps=4,
        logging_steps=100
    )
    rtf_model.fit(df)

    synth_dir = os.path.join(base_dir, "synth")
    os.makedirs(synth_dir, exist_ok=True)

    for m in multipliers:
        sample_size = int(len(df) * m)
        synthetic_df = rtf_model.sample(n_samples=sample_size)
        synthetic_df.to_csv(os.path.join(synth_dir, f"{m}x.csv"), index=False)
        print(f"Generated and saved {m}x synthetic samples.")

if __name__ == "__main__":
    CASP_PATH = "CASP.csv"
    OUTPUT_ROOT = "exp_5000_rows_5_digits"

    casp_dir = prepare_casp_data(CASP_PATH, OUTPUT_ROOT)
    train_and_generate_synthetic_data(casp_dir)

    print("✅ CASP dataset processing and synthetic generation complete.")
