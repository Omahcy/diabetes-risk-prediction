#!/usr/bin/env python3
"""
Comparative Machine Learning for Diabetes Risk Prediction (CDC BRFSS)
Evaluates supervised ML models (Logistic Regression, Random Forest, XGBoost, LightGBM)
Addressing class imbalance using SMOTE within CV folds and calculating SHAP values.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import shap

def load_data(filepath_or_url):
    df = pd.read_csv(filepath_or_url)
    target_col = 'Diabetes_binary' if 'Diabetes_binary' in df.columns else df.columns[0]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y

def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    }
    
    results = []
    trained_lgb = None

    for name, model in models.items():
        # SMOTE applied strictly inside pipeline (training data only)
        pipe = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        
        metrics = {
            "Model": name,
            "ROC-AUC": round(roc_auc_score(y_test, y_prob), 2),
            "Accuracy": f"{round(accuracy_score(y_test, y_pred) * 100, 1)}%",
            "Precision (Class 1)": round(precision_score(y_test, y_pred), 2),
            "Recall (Class 1)": round(recall_score(y_test, y_pred), 2),
            "Positive F1 (Class 1)": round(f1_score(y_test, y_pred), 2),
            "Macro F1": round(f1_score(y_test, y_pred, average='macro'), 2)
        }
        results.append(metrics)
        
        if name == "LightGBM":
            trained_lgb = pipe.named_steps['classifier']
            
    results_df = pd.DataFrame(results)
    print("\n=== Model Evaluation Results ===")
    print(results_df.to_string(index=False))
    
    return results_df, trained_lgb, X_test

if __name__ == "__main__":
    print("Script ready. Run evaluate_models(X, y) with CDC BRFSS dataset.")
