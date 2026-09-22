# Assignment 1: From Raw Data to a Reliable Classifier

**Course:** Data Science Tools and Techniques
**Dataset:** Breast Cancer Wisconsin (Diagnostic) — loaded directly from `sklearn.datasets`
**Target convention:** `1 = malignant` (positive class), `0 = benign`
**Reproducibility:** `random_state = 42` used everywhere a random state is required.

## Overview

This notebook walks through a full, small-scale machine learning workflow on the
Breast Cancer Wisconsin (Diagnostic) dataset: loading and inspecting the raw data,
exploring it visually and statistically, cleaning/preprocessing it, selecting
features, building a `sklearn` pipeline, and evaluating a Logistic Regression
classifier.

## Requirements

- Python 3
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn

Install with:
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

## How to Run

1. Open `Assignment1_24l-8000.ipynb` in Jupyter Notebook / JupyterLab / VS Code.
2. Run all cells top to bottom (`Restart & Run All`). No external data files are
   needed — the dataset is loaded directly from `sklearn.datasets.load_breast_cancer()`.

## Notebook Structure

- **Part A — Dataset Loading and Initial Inspection**
  Loads the dataset, flips the target so `1 = malignant`, checks shape, dtypes,
  unique value counts, memory usage, missing values, duplicates, infinite values,
  constant features, and summary statistics (mean, std, range, skew).

- **Part B — Exploratory Data Analysis**
  - B.1 Class balance (benign vs. malignant) via counts, percentages, and a bar chart.
  - B.2 Distribution plots (histograms + KDE) for selected mean/error/worst features.
  - B.3 Benign vs. malignant KDE comparisons for four key predictors.
  - B.4 Feature-pair scatter plots by class.
  - B.5 Full 30-feature correlation heatmap and the top 10 most correlated feature pairs.

- **Part C — Data Quality and Preprocessing**
  Records baseline missing/duplicate counts, injects ~5% synthetic missingness into
  four features to simulate real-world messiness, justifies median vs. mean
  imputation per feature (skew/outlier-based), detects outliers with both the IQR
  and z-score methods, and discusses feature-scaling considerations.

- **Part D — Feature Analysis and Selection (15 marks)**
  Identifies strongly correlated feature pairs (|r| ≥ 0.90), ranks features by
  point-biserial correlation, ANOVA F-statistic, and mutual information, and selects
  a compact 8-feature subset for modeling.

- **Part E — Train-Test Preparation and Pipeline Construction**
  Performs an 80/20 stratified train-test split and builds a `Pipeline` combining
  median imputation, `StandardScaler`, and `LogisticRegression`.

- **Part F — Basic Classification and Evaluation (20 marks)**
  Evaluates the fitted pipeline with accuracy, precision, recall, F1-score, and
  ROC-AUC; plots a confusion matrix; discusses which error type (false negative vs.
  false positive) matters more in this medical context; compares train vs. test
  performance to check for overfitting/underfitting; and compares Logistic
  Regression performance with vs. without feature scaling.

## Key Design Choices

- **Target flipping:** `sklearn`'s raw target (0 = malignant, 1 = benign) is
  inverted so that `1` consistently represents the positive/malignant class.
- **Median imputation:** chosen over mean imputation for the masked features because
  they are skewed and/or contain outliers, making the median more robust.
- **Feature selection:** an 8-feature subset (e.g. `worst concave points`,
  `worst perimeter`, `mean concave points`, `worst radius`, `worst area`, etc.) is
  chosen based on correlation with the target, ANOVA F-scores, and mutual information.
- **Scaling:** `StandardScaler` is included in the pipeline since Logistic
  Regression is scale-sensitive; the notebook also tests the pipeline without
  scaling for comparison.

## Output

Running the notebook produces printed statistics, several inline plots (class
distribution, feature distributions, KDE comparisons, scatter plots, correlation
heatmap, confusion matrix), and final classification metrics (accuracy, precision,
recall, F1, ROC-AUC) for the trained Logistic Regression pipeline.
