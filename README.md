# When Tables Leak: Attacking Digit Memorization in LLM-Based Tabular Data Generation

Official code repository for the paper: **When Tables Leak: Attacking Digit Memorization in LLM-Based Tabular Data Generation**

## Overview

This repository contains the implementation and experimental code for investigating privacy vulnerabilities in LLM-based tabular data generation systems. Our work demonstrates how adversaries can exploit digit memorization patterns to extract sensitive information from synthetically generated tabular data.

## Table of Contents

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Reproducing Experiments](#reproducing-experiments)
- [Citation](#citation)

## Installation

### Prerequisites

- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.8+
- CUDA-compatible GPU (recommended for faster execution)

### Environment Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate realtab
```
## Repository Structure

```
.
├── attack/              # Section 4: Attack experiments
│   ├── README.md       # Detailed instructions for attack experiments
│   └── ...
├── defense/            # Section 5: Defense experiments
│   ├── README.md       # Detailed instructions for defense experiments
│   └── ...
├── environment.yml     # Conda environment specification
└── README.md          # This file
```

## Getting Started

This repository is organized into two main sections:

### 1. **Attack Experiments** (`attack/`)
Contains code for replicating the attack methodologies on SFT'd LLM-based tabular generators described in Section 4 of the paper. 

See [`attack/README.md`](attack/README.md) for detailed instructions.

### 2. **Defense Experiments** (`defense/`)
Contains code for replicating the defense mechanisms described in Section 5 of the paper. 

See [`defense/README.md`](defense/README.md) for detailed instructions.

## Reproducing Experiments

To reproduce the complete experimental pipeline:

1. **Set up the environment** (see [Installation](#installation))

2. **Run attack experiments**:
```bash
cd attack/
# Follow instructions in attack/README.md
```

3. **Run defense experiments**:
```bash
cd defense/
# Follow instructions in defense/README.md
```

Each subdirectory contains its own README with specific commands and configurations needed to replicate the corresponding section of the paper.


## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{yourlastname2024tables,
  title={When Tables Leak: Attacking Digit Memorization in LLM-Based Tabular Data Generation},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
