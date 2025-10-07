from digitModifier.digitModifier import digitModifier
import pandas as pd
import os
import argparse

def create_modified_csv(base_dir: str, method="magnitude", min_probability=[0.05],
                        max_probability=[0.1]):
   
    prob_len = len(min_probability)
    for root, dirs, files in os.walk(base_dir):
        # correctly finding all synth folders which contain
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
                myModifier = digitModifier(min_probability=min_p,
                                    max_probability=max_p,
                                    method=method)
                modified_df = myModifier.process_csv(synth_df)
                base_name, extension = os.path.splitext(synth_file)
                new_file_name = f"{base_name}_modified_{min_p}_{max_p}{extension}"
                save_path = os.path.join(synth_folder, new_file_name)
                modified_df.to_csv(save_path, index=False)
                print(f"Created: {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Create modified CSV files using digit modification techniques.'
    )
    
    parser.add_argument(
        'base_dir',
        type=str,
        help='Base directory containing synth folders with CSV files'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='magnitude',
        help='Modification method to use (default: magnitude)'
    )
    
    parser.add_argument(
        '--min-probability',
        type=float,
        nargs='+',
        default=[0.05],
        help='Minimum probability value(s) (default: 0.05)'
    )
    
    parser.add_argument(
        '--max-probability',
        type=float,
        nargs='+',
        default=[0.1],
        help='Maximum probability value(s) (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    # Validate that min and max probability lists have the same length
    if len(args.min_probability) != len(args.max_probability):
        parser.error("--min-probability and --max-probability must have the same number of values")
    
    # Validate that base directory exists
    if not os.path.exists(args.base_dir):
        parser.error(f"Base directory '{args.base_dir}' does not exist")
    
    create_modified_csv(
        base_dir=args.base_dir,
        method=args.method,
        min_probability=args.min_probability,
        max_probability=args.max_probability
    )
    
    print("Processing complete!")

if __name__ == '__main__':
    main()