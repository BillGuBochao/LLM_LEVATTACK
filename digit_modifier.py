from csv_digit_modifier import CSVDigitModifier
import numpy as np
import pandas as pd
import os
import ipdb 


def creating_modified_csv(base_dir: str, method = "magnitude", min_probability = [0.05],
                        max_probability = [0.1]):
    
    prob_len = len(min_probability)

    for root, dirs, files in os.walk(base_dir):
        # correctly finding all syth folders which contain 
        # all the synthetic data of different length
        if 'synth' in dirs:
            synth_folder = os.path.join(root, "synth")
        else:
            continue 
            
        for synth_file in sorted(os.listdir(synth_folder)):
            
            
            if '_modified_' in synth_file:
                    continue

            synth_df = pd.read_csv(os.path.join(synth_folder, synth_file))
            
            for i in range(prob_len):
                min_p = min_probability[i]
                max_p = max_probability[i]
                myModifier = CSVDigitModifier(min_probability= min_p, 
                                    max_probability = max_p, 
                                    method = method)
                modified_df = myModifier.process_csv(synth_df)
                base_name, extension = os.path.splitext(synth_file) # -> ('synth_0.1x', '.csv')
                new_file_name = f"{base_name}_modified_{min_p}_{max_p}{extension}"
                save_path = os.path.join(synth_folder, new_file_name)
                modified_df.to_csv(save_path, index= False)

        

# creating_modified_csv(base_dir = 'experiment')



