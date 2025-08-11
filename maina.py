
import os
from lev_attack import lev_attack_evaluation
from realtabformer_generate import train_and_generate_synthetic_data
from digit_modifier import creating_modified_csv
from fidelity_eval import (
    add_all_kernel_scores_to_summary,
    update_mia_summary_with_JS,
    update_mia_summary_with_WS,
)
import numpy as np
import pandas as pd



def testing_pipeline(base_dir: str, syn_len = [0.5, 1.0],
                        method = "magnitude", 
                        min_probability = [0.09],
                        max_probability = [0.15],
                        tendency = [2]):
    
    '''
    Args: 
    base_dir (str): The directory name where we stored all the member data, nonmemeber data
                    and synthetic data
    
    syn_len (list): The proportion of synthetic record compared to member data set.

    method (str): The method used in digit_modifier, available options are 'magnitude', 'logarithmic',
                    and 'linear'
    
    min_probability (list of float): The min_probability used in digit_modifier

    max_probability (list of float): The max_probability used in digit_modifier

    '''

    
    experiment_id =train_and_generate_synthetic_data(base_dir= base_dir, 
                                                     multipliers = syn_len, 
                                                     processor= False)
    
    
    creating_modified_csv(base_dir = base_dir, method = method, 
                        min_probability = min_probability,
                        max_probability = max_probability)
    

    for t in tendency:
        _= train_and_generate_synthetic_data(base_dir= base_dir, multipliers = syn_len, 
                                                     processor= True, tendency= t, 
                                                     experiment_id= experiment_id)

    lev_attack_evaluation(base_dir = base_dir)

    add_all_kernel_scores_to_summary(directory= base_dir)
    update_mia_summary_with_JS(directory= base_dir)
    update_mia_summary_with_WS(directory = base_dir) 




if __name__ == "__main__":
    BASE_DIR = "experiment7"
    testing_pipeline(BASE_DIR, syn_len = [0.5,1,3,5], min_probability = [0.05, 0.1], 
                    max_probability = [0.1,0.15], tendency= [1.5,2.0,3.0])
    print()
    print('Done')
    
