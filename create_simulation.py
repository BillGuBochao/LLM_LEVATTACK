import os
import numpy as np
import pandas as pd

def generate_normal_csv(rows: int, cols: int, mean: float, std: float, decimals: int, filepath: str):
    """Generate CSV with normal distribution data using custom mean and standard deviation."""
    data = np.random.normal(loc=mean, scale=std, size=(rows, cols))
    data_rounded = np.round(data, decimals=decimals)
    df = pd.DataFrame(data_rounded, columns=[f"x{i+1}" for i in range(cols)])
    df.to_csv(filepath, index=False)

def create_experiment_directory(base_dir: str, row_count: int, mean: float, std: float, decimals: int, var_list: list):
    """Create experiment directory structure with CSV files."""
    os.makedirs(base_dir, exist_ok=True)

    for var_count in var_list:
        subfolder = os.path.join(base_dir, f"var{var_count}")
        os.makedirs(subfolder, exist_ok=True)

        # Generate member.csv
        member_path = os.path.join(subfolder, "member.csv")
        generate_normal_csv(rows=row_count, cols=var_count, mean=mean, std=std, decimals=decimals, filepath=member_path)

        # Generate non_member.csv
        non_member_path = os.path.join(subfolder, "non_member.csv")
        generate_normal_csv(rows=row_count, cols=var_count, mean=mean, std=std, decimals=decimals, filepath=non_member_path)

def generate_depedent_gauss(base_dir: str, rows: int, cols: int, mean: float, corr: float,
                                std: float, decimals: int):
    os.makedirs(base_dir, exist_ok=True)
    var_folder = os.path.join(base_dir, f"var{cols}")
    os.makedirs(var_folder, exist_ok=True)
    filepath = [os.path.join(var_folder, "member.csv"), os.path.join(var_folder, "non_member.csv")]

    mean = np.ones(cols) * mean
    var = std ** 2
    cov_matrix = np.full((cols, cols), corr * var)  
    np.fill_diagonal(cov_matrix, var) 

    for f in filepath:
        data = np.random.multivariate_normal(mean, cov_matrix, rows)
        data_rounded = np.round(data, decimals=decimals)
        df = pd.DataFrame(data_rounded, columns=[f"x{i+1}" for i in range(cols)])
        df.to_csv(f, index=False)
    

    
    




if __name__ == "__main__":
    
    #====================================================================================================
    # Experiment0: 

    # base_dir = "experiment0"
    # row_count = 500
    # decimals = 0 
    # mean = 10000
    # std = 10000
    # var_list = [4]


    # create_experiment_directory(base_dir = base_dir, row_count = row_count, mean = mean, 
    #                             std = std, decimals = decimals, var_list = var_list)

     #====================================================================================================
    # Experiment0.5: 

    # base_dir = "experiment0.5"
    # row_count = 500
    # decimals = 0 
    # mean = 500
    # std = 150
    # var_list = [6]


    # create_experiment_directory(base_dir = base_dir, row_count = row_count, mean = mean, 
    #                             std = std, decimals = decimals, var_list = var_list)
    
    #====================================================================================================
    # Experiment1: 

    # base_dir = "experiment1"
    # row_count = 3500
    # decimals = 0 
    # mean = 10000
    # std = 10000
    # var_list = [4]


    # create_experiment_directory(base_dir = base_dir, row_count = row_count, mean = mean, 
    #                             std = std, decimals = decimals, var_list = var_list)

    #====================================================================================================
    # Experiment2: 

    # base_dir = "experiment2"
    # rows = 1000
    # cols = 4
    # mean = 10000
    # corr = 0.2
    # std = 10000
    # decimals = 0

    # generate_depedent_gauss(base_dir = base_dir, rows = rows, cols = cols, mean = mean, corr = corr,
    #                             std = std, decimals = decimals)
    #====================================================================================================
    # Experiment2.5: 

    # base_dir = "experiment2.5"
    # rows = 1000
    # cols = 8
    # mean = 50
    # corr = 0.2
    # std = 15
    # decimals = 0

    # generate_depedent_gauss(base_dir = base_dir, rows = rows, cols = cols, mean = mean, corr = corr,
    #                             std = std, decimals = decimals)
    
    #====================================================================================================
    # Experiment3: 

    # base_dir = "experiment3"
    # rows = 1000
    # cols = 4
    # mean = 10000
    # corr = 0.6
    # std = 10000
    # decimals = 0

    # generate_depedent_gauss(base_dir = base_dir, rows = rows, cols = cols, mean = mean, corr = corr,
    #                             std = std, decimals = decimals)

#====================================================================================================
    # Experiment3.5: 

    # base_dir = "experiment3.5"
    # rows = 1000
    # cols = 8
    # mean = 50
    # corr = 0.6
    # std = 15
    # decimals = 0

    # generate_depedent_gauss(base_dir = base_dir, rows = rows, cols = cols, mean = mean, corr = corr,
    #                             std = std, decimals = decimals)


#====================================================================================================
    # Experiment4: (The one where attack doesn't work)

    # base_dir = "experiment4"
    # row_count = 3500
    # decimals = 0 
    # mean = 1000
    # std = 100
    # var_list = [2]


    # create_experiment_directory(base_dir = base_dir, row_count = row_count, mean = mean, 
    #                             std = std, decimals = decimals, var_list = var_list)
    
#====================================================================================================
    # Experiment5: 
    
    # base_dir = "experiment5"
    # row_count = 1000
    # decimals = 0
    # mean = 1000
    # std = 100
    # var_list = [20]
    # create_experiment_directory(base_dir=base_dir, row_count=row_count, mean=mean,
    #                             std=std, decimals=decimals, var_list=var_list)



#====================================================================================================
    # Experiment6: 
    
    # base_dir = "experiment6"
    # row_count = 1000
    # decimals = 1
    # mean = 300
    # std = 100
    # var_list = [20]
    # create_experiment_directory(base_dir=base_dir, row_count=row_count, mean=mean,
    #                             std=std, decimals=decimals, var_list=var_list)
    
    #====================================================================================================
    # Experiment7:

    base_dir = "experiment7"
    row_count = 1000
    decimals = 1
    mean = 300
    std = 5
    var_list = [20]
    create_experiment_directory(base_dir=base_dir, row_count=row_count, mean=mean,
                                std=std, decimals=decimals, var_list=var_list)
