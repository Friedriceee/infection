import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================
# 1. 数据加载
# =========================
def load_and_prepare_data():
    df = pd.read_excel("/Users/wangqinyang.5/Desktop/Infection/original.xlsx")
    y = df["outcome"].astype(int)

    num_cols = [
        'C_G', 'C_WBC', 'C_RBC', 'C_P', 'C_N',
        'GCS', 'tem', 'age',
        'B_G', 'B_CRP', 'B_WBC', 'B_N', 'B_Lym', 'B_PCT', 'B_AC', 'B_RBC'
    ]
    cat_cols = ['transparency', 'sex', 'tube', 'site', 'other_inf']

    X = df[num_cols + cat_cols].copy()
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols, scale_numeric=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_transformer = Pipeline(num_steps)
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])

    return preprocessor


# =========================
# 2. 统一评估：threshold 固定 0.5
# =========================
def evaluate_fixed_threshold(y_true, y_prob, model_name, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "threshold": threshold,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_true": np.array(y_true)
    }

    print(f"===== {model_name} =====")
    print(f"Threshold: {threshold:.2f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1: {metrics['f1']:.3f}")
    print(f"AUC: {metrics['auc']:.3f}")
    print()

    return metrics


# =========================
# 3. sklearn 模型
# =========================
def train_logistic(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=3000,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_fixed_threshold(y_test, y_prob, "Logistic", threshold=0.5)


def train_random_forest(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_fixed_threshold(y_test, y_prob, "Random Forest", threshold=0.5)


def train_xgboost(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="logloss",
            scale_pos_weight=pos_weight * 1.2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_fixed_threshold(y_test, y_prob, "XGBoost", threshold=0.5)


def train_gbdt(X_train, X_test, y_train, y_test, num_cols, cat_cols):
    preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            max_features="sqrt",
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    return evaluate_fixed_threshold(y_test, y_prob, "GBDT", threshold=0.5)



# =========================
# 5. 主函数
# =========================
def main():
    X, y, num_cols, cat_cols = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = {}
    results["logistic"] = train_logistic(X_train, X_test, y_train, y_test, num_cols, cat_cols)
    results["random_forest"] = train_random_forest(X_train, X_test, y_train, y_test, num_cols, cat_cols)
    results["xgboost"] = train_xgboost(X_train, X_test, y_train, y_test, num_cols, cat_cols)
    results["gbdt"] = train_gbdt(X_train, X_test, y_train, y_test, num_cols, cat_cols)
    
    with open("baseline_results.pkl", "wb") as f:
        pickle.dump(results, f)

    return results


if __name__ == "__main__":
    results = main()