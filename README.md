# Diabetes Risk Prediction on CDC BRFSS 2015

Machine learning pipeline that compares models and class-imbalance strategies for predicting diabetes risk from survey-based health indicators, with a fairness audit across sex and age group.

## Dataset
- **Source:** [Diabetes Health Indicators Dataset (Kaggle, Alex Teboul)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), a cleaned version of the CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey.
- **File used:** `diabetes_binary_health_indicators_BRFSS2015.csv`
- **Size:** 253,680 respondents, 21 features plus the target.
- **Target:** `Diabetes_binary` (1 = diabetic or pre-diabetic, 13.9% of respondents).
- The data file is not included in this repo. Download it from the link above.

## Method
1. **Feature engineering:** BMI x general health, age x physical health days, chronic-condition count, and a weighted comorbidity risk score.
2. **Split:** stratified 80/20 train/test split (50,736 test rows, same 13.9% prevalence).
3. **Leakage prevention:** the scaler and all resampling methods are fitted on the training data only. The test set is never resampled.
4. **Imbalance strategies:** raw (no rebalancing), SMOTE, random under-sampling, and class weighting.
5. **Models:** Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM (selected model and strategy pairs, see the notebook).
6. **Evaluation:** ROC-AUC, accuracy, balanced accuracy, precision, recall and F1 for the positive (diabetic) class on the held-out test set.
7. **Explainability:** LightGBM feature importance.
8. **Fairness audit:** demographic parity difference and equalized odds difference across sex and age group (under 60 vs 60 and over).

## Key results (test set, positive-class metrics)
| Strategy | Model | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Class weighted | LightGBM | 0.827 | 0.793 | 0.306 | 0.442 |
| Class weighted | Logistic Regression | 0.820 | 0.763 | 0.309 | 0.440 |
| Under-sampling | Gradient Boosting | 0.826 | 0.796 | 0.304 | 0.440 |
| SMOTE | Random Forest | 0.820 | 0.560 | 0.389 | 0.459 |
| Raw (no rebalancing) | LightGBM | 0.827 | 0.152 | 0.564 | 0.240 |

**Takeaways**
- All models reach a similar ROC-AUC (about 0.82-0.83). Gradient boosting gives only a small gain over Logistic Regression.
- Without rebalancing, models miss about 85% of diabetic cases (recall 0.15) even though accuracy looks high (0.87).
- Class weighting and under-sampling raise recall to about 0.79 at the cost of precision, which suits a screening use case where missing a case is costlier than a false alarm.

## Limitations
- BRFSS is a self-reported survey, so labels and features contain reporting error. The data is from 2015.
- About 9.5% of rows (roughly 24,000) are exact duplicates and were not removed, which can slightly inflate test scores.
- Single train/test split and fixed hyperparameters (no tuning or cross-validation).
- The dataset has no race column, so the fairness audit covers sex and age group only.
- This is a learning and portfolio project, not a clinical tool.

## How to run
1. Open `diabetes_ml_pipeline_real_BRFSS.ipynb` in Google Colab.
2. Run all cells. When prompted, upload the CSV from the Kaggle link above.
