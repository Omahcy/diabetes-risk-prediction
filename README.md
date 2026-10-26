# diabetes-risk-prediction
Comparative machine learning benchmarks and SHAP explainability for diabetes risk prediction across CDC BRFSS cohorts.
# Comparative Machine Learning for Diabetes Risk Prediction Across CDC BRFSS Cohorts

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: CDC BRFSS](https://img.shields.io/badge/Dataset-CDC%20BRFSS-green.svg)](https://www.cdc.gov/brfss/)

This repository implements a comparative biostatistical and machine learning evaluation for population-level diabetes risk prediction using survey cohorts from the **Centers for Disease Control and Prevention (CDC) Behavioral Risk Factor Surveillance System (BRFSS)**. 

The project benchmarks tree-based ensembles (XGBoost, LightGBM, Random Forest, CatBoost) against linear baselines, enforces strict cross-validation isolation to prevent synthetic data leakage, and provides clinical interpretability via **SHAP (Shapley Additive Explanations)** and demographic parity audits.

---

## 1. Epidemiological Cohorts & Study Architecture

The empirical benchmarks evaluate population-level chronic disease surveillance across standardized BRFSS cohort releases:

| Cohort / Study Reference | Survey Release Year | Cohort Sample Size ($N$) | Predictors | Target Framework | Baseline Imbalance Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Liu et al. (2024)** | 2015 | 253,680 records | 21 features (7 continuous, 14 categorical) | Binary: 0 (Non-diabetic) vs. 1 (Prediabetes/Diabetes) | ~1:6.25 (Class 1 prevalence $\approx 16\%$) |
| **Nabila et al. (2026)** | 2024 | 454,774 records (363,819 train / 90,955 test) | 21 features (post-feature selection) | Multiclass (3-class): Non-diabetic, Pre-diabetic, Diabetic | Highly Imbalanced (Pre-diabetic minority) |
| **Experimental Subset** | Benchmark split | 70,692 stratified records | 21 clinical & socioeconomic features | Binary (0 / 1) | Controlled validation subset |

---

## 2. Class Imbalance Protocol & Leakage Prevention

In population health datasets, non-diabetic controls heavily outnumber diabetic cases. Standard models defaulting to global accuracy suffer from the **Accuracy Paradox**—achieving over 83% nominal accuracy while failing to identify high-risk individuals.

To address this without compromising statistical validity:
1. **Fold Isolation (Zero-Leakage Guarantee):** Resampling via **SMOTE / SMOTE-N** is applied **strictly within the training folds** of each cross-validation split using `imblearn.pipeline.Pipeline`. Validation and test partitions remain completely un-resampled, preserving true population prevalence.
2. **Cost-Sensitive Weighting:** Incorporates inverse class-frequency penalization (`class_weight="balanced"`) to penalize minority false negatives without synthetic point interpolation.
3. **Hyperparameter Optimization:** Utilizes **Optuna** (Tree-structured Parzen Estimator) targeting **Macro $F_1$-score** rather than raw accuracy across 5-fold cross-validation, counteracting minority-class suppression.

---

## 3. Comparative Benchmark Results

### Binary Classification Performance (80/20 Stratified Split)

| Model Architecture | ROC-AUC | Accuracy | Precision (Class 1) | Recall (Class 1) | Positive-Class $F_1$ | Macro $F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.83** | **83.0%** | **0.85** | **0.87** | **0.86** | **0.85** |
| **XGBoost** | **0.83** | **83.0%** | **0.84** | **0.86** | **0.85** | **0.85** |
| **Logistic Regression (Baseline)** | 0.81 | 82.3% | 0.82 | 0.84 | 0.83 | 0.82 |
| **Random Forest** | 0.80 | 82.6% | 0.83 | 0.85 | 0.84 | 0.83 |

* **Finding:** Gradient-boosted ensembles (**LightGBM** and **XGBoost**) demonstrated the highest discriminative power (**AUC: 0.83**), outperforming linear baselines by capturing complex non-linear interactions across BMI, physical health days, and age cohorts.

---

## 4. Model Explainability & Clinical Risk Attribution (SHAP)

Global and local decision boundaries were audited using **TreeSHAP** to verify alignment with established clinical pathophysiology:

1. **Primary Positive Risk Drivers:** 
   * **`GenHlth` (Self-reported General Health Rating):** Strongest non-linear predictor.
   * **`BMI` (Body Mass Index):** Elevated BMI sharply shifts SHAP values positively toward diabetes categorization.
   * **`HighBP` (Hypertension Status) & `Age`:** Substantial additive contribution in older demographics.
2. **Protective Indicators:** 
   * Higher **`Income`** and regular **`PhysActivity`** demonstrated negative attribution weights, decreasing predicted probability of disease onset.
3. **Algorithmic Fairness Audit:** Evaluated across demographic subgroups (e.g. sex/gender). Equalized Odds Difference and Demographic Parity metrics showed minimal variance, ensuring that model predictions do not induce disparities in clinical screening.

---

## 5. Repository Structure

```text
├── notebooks/
│   └── diabetes_risk_prediction.ipynb   # Interactive analysis & visualization
├── src/
│   └── diabetes_risk_prediction.py      # Production training and evaluation pipeline
├── requirements.txt                     # Python dependencies
└── README.md                            # Comprehensive biostatistical report
