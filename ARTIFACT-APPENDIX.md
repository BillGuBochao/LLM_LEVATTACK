
# Artifact Appendix (Required for all badges)

Paper title: **When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation**

Requested Badge(s):
  - [X] **Available**
  - [ ] **Functional**
  - [ ] **Reproduced**


## Description 
This the artifact for the paper "When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation." The authors are: Joshua Ward, Bochau Gu, Chi-Hua Wang, Guang Cheng and the paper is featured in PoPets 2026.3. This contains the code, intermidary data, results, and tables/ figures for the paper.

### Security/Privacy Issues and Ethical Concerns 
There are no immediate security or privacy concerns for a user. The membership inference attacks and defenses are only deployed on public benchmarks and do not require making a local system vulnerable. 

## Environment

The artifact can be accessed at https://github.com/BillGuBochao/LLM_LEVATTACK where there is a conda environment and a full instructions that can be used to run the artifact.

### Accessibility

All data and code can be found at: https://github.com/BillGuBochao/LLM_LEVATTACK

### Main Results and Claims

#### Main Result 1: LLM Attacks

There are 8 figures and 10 tables in this paper. The main results are Tables 1 and 2, which show that ICL and SFT models are suspectible to the main attack in the paper, LevAtt.

#### Main Result 2: LLM Defenses 

The main results for the defense is Figure 8, which show that SFT models with the TLP defense are able to defeat LevAtt with very little fidelity loss.

### Experiments

Experiment replication details can be found in the ReadMes of both the attack and defense subdirectories of the repository. The expected results are a series of csv files and jpgs that contain leakage and fidelity results. These experiments will take quite some to execute. We estimate that this will be ~100 hours of GPU time on a modern gpu that has the size to fit mistral 8b and SFT it. For running the attacks, these should take several hours on a standard CPU. This code base is honestly quite difficult to work with as it was developed at several different times, over several different people, on several different servers and therefore we just apply for an available badge. We recommend running analyze.ipynb to verify the attack table and figure results of the paper. 


## Notes on Reusability (Encouraged for all badges)
The main value of this codebase are LevAtt and TLP. LevAtt can be found in attack/synth_mia/attackers/lev_attack.py and TLP can be found in LLM_LEVATTACK/defense/tlp/realtabformer_generate.py.

