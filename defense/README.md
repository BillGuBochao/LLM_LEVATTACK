# Defense Methods - Usage Guide

This guide explains how to replicate the defense experiments from Section 5 of the paper, including the **Digit Modifier (DM)** and **Tendency Logit Processor (TLP)** methods.

## Overview

This section implements two main defense mechanisms against digit memorization attacks:

- **Digit Modifier (DM)**: A post-processing technique that modifies generated data to reduce memorization patterns
- **Tendency Logit Processor (TLP)**: A sampling-time defense that modifies the LLM generation process to prevent digit memorization. In this codebase we focus on the RealTabFormer Implementation.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Scripts Overview](#scripts-overview)
- [Setup](#setup)
- [Quick Start Examples](#quick-start-examples)


## Prerequisites

### Environment
Ensure you have activated the conda environment:
```bash
conda activate realtab
```

### Required Directory Structure

Each dataset directory must contain:
```
dataset_name/
├── data.csv           # Your dataset
└── config.json        # Configuration file
```

### config.json Format

```json
{
  "target_column": "your_target_column_name",
  "task_type": "regression"  // Options: "regression" or "classification"
}
```

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `create_simulation_data.py` | Generates synthetic Gaussian data for controlled experiments |
| `main.py` | Orchestrates the complete TLP pipeline (training, generation, evaluation) |
| `run_utility_eval.py` | Evaluates utility of synthetic data using XGBoost models |
| `run_digitModifier.py` | Applies the DM post-processing method to generated data |

## Setup

### Step 1: Prepare Your Data

**Option A: Use Gaussian Simulation Data**
```bash
python create_simulation_data.py
```
This creates a `gauss_experiment/` directory with synthetic Gaussian data.

**Option B: Use Your Own Dataset**
1. Create a directory for your dataset (e.g., `casp_experiment/`)
2. Place your `data.csv` inside
3. Create a `config.json` with target column and task type

## Quick Start Examples

### Example 1: Complete Workflow with Gaussian Data

```bash
# Step 1: Generate Gaussian data
python create_simulation_data.py

# Step 2: Run TLP defense and privacy evaluation
python main.py gauss_experiment auc

# Step 3: Evaluate utility of generated data
python run_utility_eval.py gauss_experiment

# Step 4 (Optional): Apply Digit Modifier post-processing
python run_digitModifier.py gauss_experiment
```

### Example 2: Complete Workflow with Real Data (CASP)

```bash
# Step 1: Run TLP defense and privacy evaluation
python main.py casp_experiment auc

# Step 2: Evaluate utility of generated data
python run_utility_eval.py casp_experiment

# Step 3 (Optional): Apply Digit Modifier post-processing
python run_digitModifier.py casp_experiment
```

### Example 3: Applying Only Digit Modifier to Existing Data

If you already have generated synthetic data:
```bash
python run_digitModifier.py path/to/your/experiment
```

