import os
import pandas as pd
import ipdb
import sys
sys.path.insert(0, '/home/infamous/realtab/patt_att_logit/REaLTabFormer/src')
from realtabformer import REaLTabFormer


def train_and_generate_synthetic_data(base_dir: str, multipliers: list = [0.5, 1, 2, 3,10,100], 
                                        processor: bool = False, tendency: float = 2.0, 
                                        experiment_id: str =""):
   
    if processor:
        directory = f"rtf_model/{experiment_id}"
        assert os.path.exists(directory), f"Directory {directory} does not exist"


    for root, dirs, files in os.walk(base_dir):
       
        if "member.csv" in files:
            member_path = os.path.join(root, "member.csv")
            df = pd.read_csv(member_path)

            if not processor: # The model has not be trained

                print(f"Training on: {member_path}")
                rtf_model = REaLTabFormer(
                    model_type="tabular",
                    gradient_accumulation_steps=4,
                    logging_steps=100,
                    epochs = 280
                    )
                
                rtf_model.fit(df)
                experiment_id = rtf_model.experiment_id
                rtf_model.save("rtf_model/")
                # Prepare synth folder
            else:
                rtf_model = REaLTabFormer.load_from_dir(path=f"rtf_model/{experiment_id}")

            synth_folder = os.path.join(root, "synth")
            os.makedirs(synth_folder, exist_ok=True)

            for m in multipliers:
                sample_size = int(len(df) * m)

                if not processor: # This is the vanilla sampling
                    synthetic_df = rtf_model.sample(n_samples=sample_size, processor = False)
                    synth_path = os.path.join(synth_folder, f"synth_{m}x.csv")
                else: # This is the sampling with logit processor
                    synthetic_df = rtf_model.sample(n_samples=sample_size, processor = True, 
                                                    tendency = tendency)
                    synth_path = os.path.join(synth_folder, f"synth_{m}x_processed_{tendency}.csv")
                synthetic_df.to_csv(synth_path, index=False)
                print(f"Saved: {synth_path}")

            if not processor:
                return experiment_id
            else:
                return ""
            
        