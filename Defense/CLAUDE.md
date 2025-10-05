# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a privacy and fidelity evaluation framework for tabular synthetic data generation, specifically focused on REaLTabFormer (Realistic Relational and Tabular Data using Transformers). The codebase implements membership inference attacks (MIA) and fidelity metrics to evaluate the privacy-utility tradeoff of synthetic tabular data.

## Environment Setup

The project uses conda environments with specific dependencies:
```bash
# Create environment from main environment.yml
conda env create -f environment.yml
conda activate realtab

# For REaLTabFormer development (optional)
cd REaLTabFormer
conda env create -f environment.yml
conda activate realtabformer-env
```

## Core Architecture

### Main Components

1. **Synthetic Data Generation** (`realtabformer_generate.py`)
   - Trains REaLTabFormer models using the submodule at `REaLTabFormer/src`
   - Implements adaptive synthetic data generation with tendency parameters
   - Handles model training, synthetic data generation, and evaluation orchestration

2. **Privacy Evaluation** (`lev_attack.py`)
   - Implements Levenshtein distance-based membership inference attacks
   - Uses the `synth_mia` package for attack evaluation and ROC metrics
   - Returns AUC-ROC and TPR@FPR=0.1 metrics

3. **Fidelity Evaluation** (`fidelity_eval.py`)
   - Computes statistical distance metrics: Maximum Mean Discrepancy (MMD), Jensen-Shannon distance, Wasserstein distance
   - Uses `geomloss` library for Sinkhorn-based optimal transport distances

4. **Visualization Scripts**
   - `vis_auc.py`: Creates AUC-based privacy-utility visualization plots
   - `vis_tpr.py`: Creates TPR-based privacy-utility visualization plots
   - Both scripts process experimental results and generate PNG visualizations

### Key Directories

- `REaLTabFormer/`: Git submodule containing the REaLTabFormer implementation
- `synth_mia/`: Custom membership inference attack evaluation package
- `experiment*`: Directories containing experimental data and results
- `rtf_model/`: Directory where trained REaLTabFormer models are saved

### Experimental Workflow

1. **Data Generation**: `create_simulation.py` generates synthetic datasets with normal distributions
2. **Model Training**: Train REaLTabFormer on member data
3. **Synthetic Generation**: Generate synthetic data with varying parameters
4. **Privacy Evaluation**: Run Levenshtein-based MIA using `lev_attack_evaluation()`
5. **Fidelity Evaluation**: Compute statistical distances using `compute_fidelity_metrics()`
6. **Visualization**: Generate privacy-utility tradeoff plots

## Development Commands

### Running Experiments
```bash
# Main AUC-based experiment
python vis_auc.py

# TPR-based experiment  
python vis_tpr.py

# Create simulation data
python create_simulation.py
```

### Testing and Quality Assurance
The REaLTabFormer submodule includes pre-commit hooks for code quality:
```bash
cd REaLTabFormer
pre-commit install
pre-commit run --all-files

# Run specific checks
python -m pytest tests/
python -m bandit -ll src/
python -m flake8 src/
```

### Model Management
Trained models are saved in `rtf_model/{experiment_id}/` and can be loaded using:
```python
from realtabformer import REaLTabFormer
rtf_model = REaLTabFormer.load_from_dir(path=f"rtf_model/{experiment_id}")
```

## Key Dependencies

- **REaLTabFormer**: Transformer-based tabular data generation (from submodule)
- **synth_mia**: Custom package for membership inference attack evaluation
- **geomloss**: Sinkhorn algorithm for optimal transport distances
- **Levenshtein**: String distance computation for privacy attacks
- **scikit-learn**: Standard ML utilities and preprocessing
- **torch**: Deep learning framework (PyTorch)

## Important Path Configuration

The codebase adds the REaLTabFormer source to Python path:
```python
sys.path.insert(0, '/home/infamous/realtab/patt_att_visualization/REaLTabFormer/src')
```

This allows importing `realtabformer` directly without package installation.