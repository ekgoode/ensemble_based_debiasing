# **Building Competent Models: Ensembles for Unlearning Single Feature Artifacts**

This repository contains the implementation of a Mixed Capacity Ensemble (MCE) framework for mitigating dataset artifacts in natural language inference (NLI) tasks. The project focuses on identifying dataset artifacts, training robust models, and evaluating their performance using datasets like SNLI.

---

## **Table of Contents**
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Pipeline Overview](#pipeline-overview)
- [Reproducing Results](#reproducing-results)
- [Key Results](#key-results)
- [Acknowledgments](#acknowledgments)

---

## **Introduction**

Large NLI datasets often contain spurious correlations, or "artifacts," which bias model predictions. This repository implements:
1. Analysis of dataset artifacts.
2. Training of a robust **ELECTRA-small** model.
3. Development of a **Mixed Capacity Ensemble (MCE)** that mitigates the influence of dataset artifacts.

The project demonstrates how to:
- Identify artifact-heavy features.
- Build a Mixed Capacity Ensemble (MCE) using both a high-capacity and low-capacity model.
- Evaluate and compare model performance on in-domain and challenge datasets.

---

## **Project Structure**

```
project_root/
│
├── data/                   # Dataset files
├── scrap/                  # Deprecated code. Mainly for archival purposes. 
├── src/                    # Source code
│   ├── generate_artifact_statistics.py  # Generate artifact stats (e.g., Figure 1)
│   ├── train_electra.py                  # Train ELECTRA-small model
│   ├── train_mce.py                      # Train Mixed Capacity Ensemble
│   ├── evaluate_electra.py               # Evaluate ELECTRA-small model (e.g. Table 1, 2, 3)
│   └── evaluate_mce.py                   # Evaluate MCE model (e.g. Table 3, 4, 5)
├── models/                 # Trained model checkpoints (ELECTRA-small, MCE)
├── run.py                  # End-to-end script to run results
├── requirements.txt        # Dependencies
└──  README.md               # Project overview (this file)
```

---

## **Dependencies**

The following dependencies are required to run the code:
- Python >= 3.8
- PyTorch >= 1.9
- Transformers
- Datasets
- scikit-learn
- tqdm
- matplotlib
- numpy
- pandas

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

---

## **Pipeline Overview**

This project consists of three main steps:

1. **Artifact Analysis**:
   - Analyze and visualize dataset artifacts (e.g., high-frequency words) using `src/generate_artifact_statistics.py`.
   - Output: Figure 1, showing the frequency of top artifact-heavy words.

2. **Model Training**:
   - Train two models:
     - **ELECTRA-small**: A transformer-based model fine-tuned on SNLI.
     - **Mixed Capacity Ensemble (MCE)**: Combines a high-capacity transformer model with a low-capacity Bag-of-Words model.

3. **Model Evaluation**:
   - Evaluate trained models on SNLI test data and HANS.
   - Output: Results included in tables 1-5.

---

## **Reproducing Results**

To reproduce the results from the paper, run the following commands:

### Step 1: Generate Artifact Statistics
```bash
python src/generate_artifact_statistics.py
```
This saves **Figure 1** as `artifact_statistics_snli.png` in the current directory.

### Step 2: Train Models
```bash
python src/train_electra.py
python src/train_mce.py

```
This trains and saves the **ELECTRA-small** and **MCE** models in the `models/` directory.

### Step 3: Evaluate Models
```bash
python src/evaluate_electra.py
python src/train_mce.py

```
This evaluates the trained models and prints accuracy and other metrics.

### Full Pipeline
Run the entire pipeline end-to-end:
```bash
python run.py
```

