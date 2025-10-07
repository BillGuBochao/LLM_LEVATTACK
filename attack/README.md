# Attack Methods - Usage Guide

This guide explains how to replicate the attack experiments from Section 4 of the paper, demonstrating digit memorization vulnerabilities in LLM-based tabular data generation.

## Overview

This section implements the digit memorization attack framework to evaluate privacy leakage in synthetic tabular data. The pipeline generates synthetic data using multiple models and evaluates their vulnerability to Levenshtein distance-based membership inference attacks.

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
Also assume you have LLMs either loaded on local or have an active Hugging Face API login.
## Scripts Overview

| Script | Purpose |
|--------|---------|
| `generate_llms.py` | Generates synthetic data using various LLM-based models |
| `generate_synthcity.py` | Generates synthetic data using CTGAN, TVAE, and GREAT |
| `run_lev_attack_sft.py` | Executes Levenshtein distance-based attack on all synthetic datasets |

## Quick Start Examples

### Example 1: Complete Attack Pipeline with LLM Models

```bash
# Step 1: Generate synthetic data using LLMs, automatically creates an experiments directory
python generate_llms.py 

# Step 2: Run Levenshtein attack on generated data
python run_lev_attack_sft.py 
```

### Example 2: Complete Attack Pipeline with GAN/VAE Models

```bash
# Step 1: Generate synthetic data using CTGAN, TVAE, and GREAT
python generate_synthcity.py 

# Step 2: Run Levenshtein attack on generated data
python run_lev_attack_sft.py 
```

### Example 3: Full Evaluation (All Models)

```bash
# Step 1: Generate data with all LLM-based models
python generate_llms.py 

# Step 2: Generate data with GAN/VAE models
python generate_synthcity.py 

# Step 3: Run comprehensive attack evaluation
python run_lev_attack_sft.py 
```

### Example 4: Attack Existing Synthetic Data

If you already have synthetic data generated:
```bash
# Simply run the attack on your directory
python run_lev_attack_sft.py 
```