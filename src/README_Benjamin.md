# Alzheimer's Disease Biomarker Analysis Pipeline

This repository contains a complete pipeline for the preprocessing, analysis, and modeling of Alzheimer's disease biomarker data. The workflow is designed for reproducibility, and is intended for use in a Python virtual environment (venv) on Visual Studio Code.

## Project Structure

```
.                           # Current directory (you are here)
├── README.md              # This file
├── data/                  # Data directory
│   ├── merged_data.csv
│   └── preprocessed_data.csv
├── Study_files/          # Raw study data
│   └── (raw study data files)
└── src/                  # Source code directory
    ├── merge_files.py
    ├── EDA.py
    ├── Preprocess.py
    ├── KNN_test.py
    ├── selectkbest_test.py
    ├── ModelsFinal.py
    ├── plot_confusion_matrices.py
    ├── Generic_report_figures.py
    ├── generate_figures_and_tables.py
    ├── requirements.txt
    └── Outputs/          # Output directory
        └── (output files)
```

## Setup Instructions

1. Navigate to the src directory and open it in Visual Studio Code:
   cd src

2. Create and activate a virtual environment (if not already done):
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux

3. Install dependencies:
   pip install -r requirements.txt

## Data Folders

- **../Study_files/**: Contains all raw input CSVs from the ADNI study (e.g., `ADNIMERGE_14Oct2024.csv`, `BLENNOWPLASMATAU_14Oct2024.csv`, etc.).
- **../data/**: Used for intermediate and processed datasets (`merged_data.csv`, `preprocessed_data.csv`).
- **Outputs/**: All output figures, tables, and metrics are saved here.

## Pipeline Overview

Run the following scripts in order:

### 1. merge_files.py
- Input: Files in ../Study_files/
- Output: ../data/merged_data.csv
- Purpose: Merges multiple raw study files into a single, analysis-ready CSV. Handles column selection, renaming, and merging on subject/timepoint keys.
   python merge_files.py

### 2. EDA.py
- Input: ../data/merged_data.csv
- Output: Figures and tables in Outputs/
- Purpose: Performs exploratory data analysis, including missingness analysis, outlier detection, and feature correlation with diagnosis. Outputs publication-quality figures.
   python EDA.py

### 3. Preprocess.py
- Input: ../data/merged_data.csv
- Output: ../data/preprocessed_data.csv, summary tables, and boxplots in Outputs/
- Purpose: Cleans and preprocesses the merged data, including string-to-numeric conversion, feature engineering, categorical encoding, and outlier treatment.
   python Preprocess.py

### 4. KNN_test.py
- Input: ../data/preprocessed_data.csv
- Output: Sensitivity analysis figures and metrics in Outputs/
- Purpose: Evaluates the effect of different K values in KNN imputation on downstream model performance (CatBoost, Random Forest, Neural Network, and ensemble).
   python KNN_test.py

### 5. selectkbest_test.py
- Input: ../data/preprocessed_data.csv
- Output: Feature selection analysis figures and metrics in Outputs/
- Purpose: Assesses the impact of selecting different numbers of top features (SelectKBest) on model performance.
   python selectkbest_test.py

### 6. ModelsFinal.py
- Input: ../data/preprocessed_data.csv
- Output: Final model metrics, feature importances, confusion matrices, and summary tables in Outputs/
- Purpose: Trains and evaluates the final models using the optimal preprocessing and feature selection settings. Outputs all results for reporting.
   python ModelsFinal.py

## Outputs

All key outputs (figures, tables, metrics) are saved in `Outputs/`. These include:

- Missing data and outlier figures
- Boxplots before/after outlier treatment
- Correlation bar charts
- Feature importance plots
- Per-class and overall metrics tables
- Confusion matrices

## Utility Scripts

- `Generic_report_figures.py`: Generates example confusion matrix and overfitting/underfitting figures (not part of main pipeline).
- `generate_figures_and_tables.py`: Aggregates and formats results for reporting (optional).
- `plot_confusion_matrices.py`: Generates a 2x2 grid of confusion matrices for all models (CatBoost, Random Forest, Neural Network, and Voting Ensemble). Creates publication-quality figures with proper formatting, annotations, and styling. Outputs to `Outputs/all_confusion_matrices.png`.

## Reproducibility

- All random seeds are fixed for reproducibility.
- The full environment is specified in `requirements.txt`

## Notes

- All scripts assume the current working directory is the `src` directory.
- All paths are relative to the `src` directory.
- For any issues, ensure your virtual environment is activated and all dependencies are installed. 