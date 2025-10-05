import pandas as pd
import numpy as np
import random
import math
import argparse
from pathlib import Path
from tqdm import tqdm

class CSVDigitModifier:
    def __init__(self, min_probability=0.01, max_probability=0.1, method='magnitude', seed=None):
        """
        Initialize the CSV digit modifier.
        
        Args:
            min_probability (float): Minimum probability for digit modification (0-1)
            max_probability (float): Maximum probability for digit modification (0-1)
            method (str): Method for calculating probability ('magnitude', 'logarithmic', 'linear')
            seed (int): Random seed for reproducible results
        """
        self.min_prob = min_probability
        self.max_prob = max_probability
        self.method = method
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def calculate_probability(self, value, all_values):
        """
        Calculate the probability of modifying digits based on the value and method.
        
        Args:
            value (float): The number to calculate probability for
            all_values (list): All numeric values in the dataset
            
        Returns:
            float: Probability between min_prob and max_prob
        """
        abs_value = abs(value)
        
        if self.method == 'magnitude':
            max_magnitude = max(abs(v) for v in all_values if not pd.isna(v))
            if max_magnitude == 0:
                normalized = 0
            else:
                normalized = abs_value / max_magnitude
                
        elif self.method == 'logarithmic':
            log_values = [math.log(abs(v) + 1) for v in all_values if not pd.isna(v)]
            max_log = max(log_values)
            if max_log == 0:
                normalized = 0
            else:
                normalized = math.log(abs_value + 1) / max_log
                
        elif self.method == 'linear':
            sorted_values = sorted([abs(v) for v in all_values if not pd.isna(v)])
            if len(sorted_values) <= 1:
                normalized = 0.5
            else:
                try:
                    rank = sorted_values.index(abs_value)
                    normalized = rank / (len(sorted_values) - 1)
                except ValueError:
                    # Handle case where exact value isn't found (floating point precision)
                    # Find closest value
                    closest_idx = min(range(len(sorted_values)), 
                                    key=lambda i: abs(sorted_values[i] - abs_value))
                    normalized = closest_idx / (len(sorted_values) - 1)
        else:
            normalized = 0.5
        
        # Scale to the specified probability range
        return self.min_prob + (normalized * (self.max_prob - self.min_prob))
    
    def modify_digits(self, number, probability):
        """
        Modify each digit of a number with the given probability.
        
        Args:
            number (float): The number to modify
            probability (float): Probability of modifying each digit
            
        Returns:
            float: Modified number
        """
        if pd.isna(number) or not isinstance(number, (int, float)):
            return number
        
        # Handle negative numbers
        is_negative = number < 0
        abs_number = abs(number)
        
        # Convert to string to work with digits
        number_str = str(abs_number)
        
        # Handle scientific notation
        if 'e' in number_str.lower():
            # For scientific notation, convert to decimal first
            abs_number = float(number_str)
            number_str = f"{abs_number:.10f}".rstrip('0').rstrip('.')
        
        modified_digits = []
        
        for char in number_str:
            if char.isdigit():
                digit = int(char)
                # Apply probability to modify this digit
                if random.random() < probability:
                    # Add 1 to the digit, wrapping 9 to 0
                    modified_digit = (digit + 1) % 10
                    modified_digits.append(str(modified_digit))
                else:
                    modified_digits.append(char)
            else:
                # Keep decimal points and other characters as-is
                modified_digits.append(char)
        
        # Convert back to number
        try:
            modified_number = float(''.join(modified_digits))
            return -modified_number if is_negative else modified_number
        except ValueError:
            # If conversion fails, return original number
            return number
    
    def process_csv(self, df, output_file=None):
        """
        Process a CSV file and modify digits according to the probability rules.
        
        Args:
            df: an input df that should be 
            output_file (str): Path to output CSV file (optional)
            
        Returns:
            pd.DataFrame: Modified dataframe
        """
        # Read the CSV file
        # try:
        #     df = pd.read_csv(input_file)
        # except Exception as e:
        #     raise Exception(f"Error reading CSV file: {e}")
        
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numeric columns: {numeric_columns}")
        
        if not numeric_columns:
            print("No numeric columns found in the CSV!")
            return df
        
        # Get all numeric values for probability calculation
        all_numeric_values = []
        for col in numeric_columns:
            values = df[col].dropna().tolist()
            all_numeric_values.extend(values)
        
        if not all_numeric_values:
            print("No numeric values found!")
            return df
        
        print(f"Processing {len(all_numeric_values)} numeric values...")
        print(f"Value range: {min(all_numeric_values):.2f} to {max(all_numeric_values):.2f}")
        
        # Create a copy of the dataframe for modification
        modified_df = df.copy()
        
        # Track modifications for reporting
        modifications = []
        
        # Process each numeric column
        for col in tqdm(numeric_columns, desc="Processing numeric columns"):
            for idx, value in tqdm(enumerate(df[col]), 
                           total=len(df[col]), 
                           desc=f"Processing rows in {col}", 
                           leave=False): 
                if pd.notna(value):
                    probability = self.calculate_probability(value, all_numeric_values)
                    modified_value = self.modify_digits(value, probability)
                    
                    if modified_value != value:
                        modifications.append({
                            'row': idx,
                            'column': col,
                            'original': value,
                            'modified': modified_value,
                            'probability': probability
                        })
                    
                    modified_df.iloc[idx, modified_df.columns.get_loc(col)] = modified_value
        
        # Report modifications
        print(f"\nModifications made: {len(modifications)}")
        if modifications:
            print("\nFirst 10 modifications:")
            for i, mod in enumerate(modifications[:10]):
                print(f"  Row {mod['row']}, {mod['column']}: {mod['original']} → {mod['modified']} (p={mod['probability']:.3f})")
        
        # Save to output file if specified
        if output_file:
            try:
                modified_df.to_csv(output_file, index=False)
                print(f"\nModified CSV saved to: {output_file}")
            except Exception as e:
                print(f"Error saving CSV: {e}")
        
        return modified_df

def main():
    parser = argparse.ArgumentParser(description='Modify CSV digits with probability based on number magnitude')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('-o', '--output', help='Output CSV file path')
    parser.add_argument('--min-prob', type=float, default=0.1, help='Minimum probability (0-1, default: 0.1)')
    parser.add_argument('--max-prob', type=float, default=0.9, help='Maximum probability (0-1, default: 0.9)')
    parser.add_argument('--method', choices=['magnitude', 'logarithmic', 'linear'], 
                       default='magnitude', help='Probability calculation method')
    parser.add_argument('--seed', type=int, help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Validate probability range
    if not (0 <= args.min_prob <= 1) or not (0 <= args.max_prob <= 1):
        print("Error: Probabilities must be between 0 and 1")
        return
    
    if args.min_prob >= args.max_prob:
        print("Error: min-prob must be less than max-prob")
        return
    
    # Set default output filename if not provided
    if not args.output:
        input_path = Path(args.input_file)
        args.output = str(input_path.parent / f"{input_path.stem}_modified{input_path.suffix}")
    
    # Create modifier and process CSV
    modifier = CSVDigitModifier(
        min_probability=args.min_prob,
        max_probability=args.max_prob,
        method=args.method,
        seed=args.seed
    )
    
    try:
        print(f"Processing: {args.input_file}")
        print(f"Method: {args.method}")
        print(f"Probability range: {args.min_prob:.1%} to {args.max_prob:.1%}")
        if args.seed:
            print(f"Random seed: {args.seed}")
        print("-" * 50)
        
        modified_df = modifier.process_csv(args.input_file, args.output)
        
        print("\nProcessing complete!")
        
    except Exception as e:
        print(f"Error: {e}")

# Example usage as a module
if __name__ == "__main__":
    # If run as script, use command line arguments
    main()
else:
    # Example of how to use as a module
    print("CSV Digit Modifier loaded as module.")
    print("Example usage:")
    print("  modifier = CSVDigitModifier(min_probability=0.2, max_probability=0.8)")
    print("  df = modifier.process_csv('input.csv', 'output.csv')")