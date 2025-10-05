# Usage Guide for Privacy Evaluation Pipeline

This guide explains how to use `main.py` and `utility_measure.py` for privacy-preserving synthetic data generation and evaluation.

## Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Using main.py](#using-mainpy)
- [Using utility_measure.py](#using-utility_measurepy)
- [Understanding the Output](#understanding-the-output)
- [Examples](#examples)

---

## Overview

This pipeline evaluates privacy-preserving synthetic data generation using two complementary approaches:

1. **main.py**: Orchestrates the complete privacy evaluation pipeline (model training, synthetic data generation, evaluation, and visualization)
2. **utility_measure.py**: Evaluates the utility of synthetic data by training XGBoost models and comparing performance against real data

---

## Prerequisites

### Required Directory Structure
Your dataset directory must contain:
- A CSV file with your dataset (e.g., `data.csv`)
- A `config.json` file specifying the target column and task type

### config.json Format
```json
{
  "target_column": "your_target_column_name",
  "task_type": "regression"  // or "classification"
}
```

### Dependencies
Ensure you have the required packages installed:
```bash
pip install xgboost pandas numpy scikit-learn matplotlib tqdm
```

---

## Using main.py

### Purpose
`main.py` is the main entry point for running the complete privacy evaluation pipeline. It:
1. Trains models and generates synthetic data with tendency optimization
2. Evaluates vanilla baseline models for comparison
3. Exports summary results to CSV files
4. Generates comparison visualizations

### Command Line Usage

```bash
python main.py <base_dir> <eval_type>
```

**Parameters:**
- `base_dir`: Directory containing your dataset CSV file and where outputs will be saved
- `eval_type`: Type of privacy evaluation (`auc` or `tpr`)

### Evaluation Types

#### AUC-based Evaluation
Evaluates privacy using Area Under the ROC Curve (AUC) metrics:
```bash
python main.py experimentCASP auc
```

**Output files:**
- `summary_auc_{dataset}.csv`: Tendency-based results
- `vanilla_summary_auc_{dataset}.csv`: Vanilla baseline results
- `comparison_auc_plot_{dataset}.png`: Visual comparison plot

#### TPR-based Evaluation
Evaluates privacy using True Positive Rate at fixed False Positive Rate:
```bash
python main.py experimentCASP tpr
```

**Output files:**
- `summary_tpr_{dataset}.csv`: Tendency-based results
- `vanilla_summary_tpr_{dataset}.csv`: Vanilla baseline results
- `comparison_tpr_plot_{dataset}.png`: Visual comparison plot

### Pipeline Steps

When you run `main.py`, it executes the following steps:

1. **Validation**: Checks that the directory exists and contains required CSV files
2. **Evaluation**: Runs either AUC or TPR-based privacy evaluation
3. **Visualization**: Generates comparison plots
4. **Summary**: Reports generated output files and their sizes

### Example Output
```
================================================================================
STARTING PRIVACY EVALUATION PIPELINE
Dataset directory: experimentCASP
Evaluation type: AUC
================================================================================
Found dataset files: ['data.csv']

------------------------------------------------------------
STEP 1: Running AUC evaluation pipeline
------------------------------------------------------------
Executing AUC-based privacy evaluation...
✓ AUC evaluation completed successfully!
✓ Generated 150 result entries

------------------------------------------------------------
STEP 2: Generating comparison visualization
------------------------------------------------------------
Creating AUC comparison plot...
✓ Comparison visualization completed!

------------------------------------------------------------
STEP 3: Pipeline completion summary
------------------------------------------------------------
Generated output files:
  ✓ summary_auc_experimentCASP.csv (12,345 bytes)
  ✓ vanilla_summary_auc_experimentCASP.csv (8,901 bytes)
  ✓ comparison_auc_plot_experimentCASP.png (234,567 bytes)

================================================================================
PIPELINE COMPLETED SUCCESSFULLY!
All outputs saved to: /path/to/experimentCASP
================================================================================
```

---

## Using utility_measure.py

### Purpose
`utility_measure.py` evaluates the utility of synthetic data by:
1. Training XGBoost models on synthetic data
2. Testing on real test data
3. Comparing performance with vanilla and real data baselines
4. Generating utility visualizations

### Command Line Usage

```bash
python utility_measure.py <base_dir>
```

**Parameter:**
- `base_dir`: Base directory containing subdirectories with training/test data and `config.json`

### Required Directory Structure

The base directory must contain:
```
base_dir/
├── config.json
└── train_test_synth*/          # One or more subdirectories
    ├── test.csv                # Test data (required)
    ├── synth_auc_*.csv         # Synthetic data for AUC evaluation
    ├── synth_tpr_*.csv         # Synthetic data for TPR evaluation
    ├── vanilla_auc_*.csv       # Vanilla baseline for AUC
    ├── vanilla_tpr_*.csv       # Vanilla baseline for TPR
    └── rest.csv                # Remaining real data
```

### Training Process

For each synthetic data file (e.g., `synth_auc_500.csv`), the script:
1. Extracts the sample size from the filename (e.g., 500)
2. Trains an XGBoost model on the synthetic data
3. Evaluates on the test set
4. Trains comparable models on:
   - Vanilla data (same sample size)
   - Real data from `rest.csv` (same sample size)
5. Compares metrics across all three approaches

### XGBoost Training Configuration

**Regression Parameters:**
```python
{
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 3.0,
    "tree_method": "hist"
}
```

**Classification Parameters:**
```python
{
    "objective": "binary:logistic",  # or "multi:softprob" for multiclass
    "eval_metric": "logloss",
    "eta": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 3.0,
    "tree_method": "hist"
}
```

**Training settings:**
- Number of boosting rounds: 2000
- Early stopping rounds: 200
- Validation split: 20%
- GPU support: Enabled if available

### Output Files

The script creates a `utility_summary/` directory with:
- `utility_auc.csv`: Utility metrics for AUC-based synthetic data
- `utility_tpr.csv`: Utility metrics for TPR-based synthetic data

### Metrics Reported

**For Regression Tasks:**
- `rmse`: Root Mean Squared Error
- `r2_score`: R² Score

**For Classification Tasks:**
- `accuracy`: Classification accuracy
- `f1_score`: Weighted F1 score

**Additional Information:**
- `subdirectory`: Source subdirectory name
- `dataset`: Dataset identifier
- `n_train_samples`: Number of training samples
- `n_test_samples`: Number of test samples
- `n_features`: Number of features used
- `data_type`: Type of data (`synthetic`, `vanilla`, or `rest`)

### Example Output
```
Processing subdirectories: 100%|████████| 5/5 [02:30<00:00, 30.15s/it]

Processing synth_auc file with 500 samples: synth_auc_500.csv
Training XGBoost: 100%|████████| 2000/2000 [00:45<00:00, valid-rmse: 0.12345]
Best iteration: 1543

Processing corresponding vanilla file: vanilla_auc_500.csv
Training XGBoost: 100%|████████| 2000/2000 [00:42<00:00, valid-rmse: 0.13456]
Best iteration: 1621

Processing rest.csv with 500 samples
Training XGBoost: 100%|████████| 2000/2000 [00:44<00:00, valid-rmse: 0.11234]
Best iteration: 1489

AUC results saved to: /path/to/base_dir/utility_summary/utility_auc.csv
TPR results saved to: /path/to/base_dir/utility_summary/utility_tpr.csv

Utility measurement completed successfully!
Results shape: (45, 9)
```

### Creating Visualizations

After running `utility_measure.py`, you can generate visualization plots:

```python
from utility_measure import create_visualization

create_visualization("path/to/base_dir")
```

This creates bar plots comparing utility metrics across different data types and sample sizes in a `visualization/` subdirectory.

---

## Understanding the Output

### CSV Summary Files

The CSV files contain detailed metrics for each experiment:

| Column | Description |
|--------|-------------|
| subdirectory | Subdirectory where the data came from |
| dataset | Dataset name (e.g., synth_auc_500, vanilla_auc_500) |
| n_train_samples | Number of samples used for training |
| n_test_samples | Number of samples in test set |
| n_features | Number of features used |
| data_type | Type: `synthetic`, `vanilla`, or `rest` |
| rmse / accuracy | Primary metric (depends on task type) |
| r2_score / f1_score | Secondary metric (depends on task type) |

### Visualization Plots

The generated PNG files show:
- **X-axis**: Number of training samples
- **Y-axis**: Performance metric (RMSE or Accuracy)
- **Colors**:
  - 🔴 Red: REAL-DATA
  - 🔵 Blue: VANILLA
  - 🟡 Yellow: TLP (Tendency-based synthetic data)

---

## Examples

### Example 1: Complete AUC Evaluation

```bash
# Run the complete AUC-based pipeline
python main.py experiment7 auc

# This will:
# 1. Train models on experiment7 data
# 2. Generate synthetic data with tendency optimization
# 3. Evaluate privacy using AUC metrics
# 4. Create comparison visualizations
# 5. Save all results to experiment7/
```

### Example 2: TPR-Based Evaluation

```bash
# Run TPR-based privacy evaluation
python main.py experimentCASP tpr

# Output: summary_tpr_experimentCASP.csv, vanilla_summary_tpr_experimentCASP.csv, comparison_tpr_plot_experimentCASP.png
```

### Example 3: Utility Measurement

```bash
# Measure utility of synthetic data
python utility_measure.py experiment7

# This will:
# 1. Find all train_test_synth* subdirectories
# 2. Train XGBoost models on synthetic, vanilla, and real data
# 3. Compare performance metrics
# 4. Save results to experiment7/utility_summary/
```

### Example 4: Complete Workflow

```bash
# Step 1: Run privacy evaluation
python main.py experiment7 auc

# Step 2: Measure utility
python utility_measure.py experiment7

# Step 3: Create utility visualizations (in Python)
python -c "from utility_measure import create_visualization; create_visualization('experiment7')"
```

---

## Troubleshooting

### Common Issues

**Issue**: `ERROR: Directory 'experiment7' does not exist!`
- **Solution**: Ensure the directory path is correct and exists

**Issue**: `No dataset CSV files found`
- **Solution**: Make sure your directory contains at least one CSV file (excluding summary files)

**Issue**: `Configuration file not found: config.json`
- **Solution**: Create a `config.json` file with `target_column` and `task_type` fields

**Issue**: `Warning: test.csv not found in subdirectory`
- **Solution**: Ensure each `train_test_synth*` subdirectory contains a `test.csv` file

**Issue**: CUDA out of memory errors
- **Solution**: Set `use_gpu=False` in the training functions or reduce batch sizes

---

## Advanced Usage

### Customizing XGBoost Parameters

You can modify the training parameters in `utility_measure.py`:

```python
# Edit _base_params() function
def _base_params(use_gpu: bool):
    p = {
        "eta": 0.05,              # Learning rate
        "max_depth": 8,           # Maximum tree depth
        "subsample": 0.8,         # Row subsampling
        "colsample_bytree": 0.8,  # Column subsampling
        "lambda": 3.0,            # L2 regularization
        "tree_method": "hist",
    }
    return p
```

### Setting Random Seeds

Both scripts use random seeds for reproducibility. In `main.py`:

```python
set_random_seed(42)  # Default seed
```

You can modify this to use different seeds if needed.

---

## Additional Resources

- **REaLTabFormer Documentation**: See `REaLTabFormer/` submodule
- **Membership Inference Attacks**: See `lev_attack.py` for attack implementation
- **Fidelity Metrics**: See `fidelity_eval.py` for statistical distance metrics
- **Project Overview**: See `CLAUDE.md` for architecture details

---

## Support

For issues, bugs, or questions:
1. Check the error messages in the console output
2. Verify your directory structure matches the requirements
3. Ensure `config.json` is properly formatted
4. Review the generated CSV files for diagnostic information
