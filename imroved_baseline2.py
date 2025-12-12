"""
改进的基线模型（精简输出版）
目标：训练各模型，并输出最优阈值下的性能指标
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')


def load_and_prepare_data():
    """加载和准备数据"""
    df = pd.read_csv("转化后_编码数据_最终版本.csv")

    # 目标列
    y = df["outcome"]

    # 特征列
    num_cols = ['C_G', 'C_WBC', 'C_RBC', 'C_P', 'C_N',
                'transparency', 'GCS', 'tem',
                'B_G', 'B_CRP', 'B_WBC', 'B_N', 'B_Lym', 'B_PCT', 'B_AC']
    cat_cols = ['sex', 'tube', 'site', 'other_inf']

    X = df[num_cols + cat_cols]

    # 缺失值填充
    X = X.fillna(X.median())

    return X, y


def evaluate_with_threshold_optimization(model, X_test, y_test, model_name):
    """通过阈值搜索选择最优阈值，并输出结果"""
    y_prob = model.predict_proba(X_test)[:, 1]

    best_threshold = 0.5
    best_f1 = -1.0

    # 在 0.1~0.9 之间搜索最优阈值（按 F1 最大）
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # 使用最优阈值重新计算各项指标
    y_pred = (y_prob >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"===== {model_name} =====")
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "threshold": best_threshold
    }


def train_optimized_decision_tree(X_train, X_test, y_train, y_test):
    """训练决策树（带简单调参）"""
    dt = DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42
    )
    dt.fit(X_train, y_train)
    return evaluate_with_threshold_optimization(dt, X_test, y_test, "Decision Tree")


def train_optimized_random_forest(X_train, X_test, y_train, y_test):
    """训练随机森林"""
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return evaluate_with_threshold_optimization(rf, X_test, y_test, "Random Forest")


def train_optimized_xgboost(X_train, X_test, y_train, y_test):
    """训练XGBoost"""
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb = XGBClassifier(
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
    )
    xgb.fit(X_train, y_train)
    return evaluate_with_threshold_optimization(xgb, X_test, y_test, "XGBoost")


def train_optimized_gbdt(X_train, X_test, y_train, y_test):
    """训练GBDT"""
    gbdt = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gbdt", GradientBoostingClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            max_features='sqrt',
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42
        ))
    ])
    gbdt.fit(X_train, y_train)
    return evaluate_with_threshold_optimization(gbdt, X_test, y_test, "GBDT")


def create_ensemble_model(X_train, X_test, y_train, y_test):
    """简单集成模型（DT + RF + XGB，软投票）"""
    dt = DecisionTreeClassifier(
        max_depth=12, min_samples_split=8, min_samples_leaf=4,
        class_weight='balanced', random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=15, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        scale_pos_weight=2.5, random_state=42, n_jobs=-1
    )

    dt.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    dt_prob = dt.predict_proba(X_test)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    ensemble_prob = 0.2 * dt_prob + 0.4 * rf_prob + 0.4 * xgb_prob

    # 以 F1 最大为准，搜索最佳阈值
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (ensemble_prob >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_pred = (ensemble_prob >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, ensemble_prob)

    print("===== Ensemble Model =====")
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"AUC: {auc:.3f}")
    print()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "threshold": best_threshold
    }


def main():
    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 依次训练并打印结果
    dt_metrics = train_optimized_decision_tree(X_train, X_test, y_train, y_test)
    rf_metrics = train_optimized_random_forest(X_train, X_test, y_train, y_test)
    xgb_metrics = train_optimized_xgboost(X_train, X_test, y_train, y_test)
    gbdt_metrics = train_optimized_gbdt(X_train, X_test, y_train, y_test)
    ensemble_metrics = create_ensemble_model(X_train, X_test, y_train, y_test)

    return {
        "decision_tree": dt_metrics,
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "gbdt": gbdt_metrics,
        "ensemble": ensemble_metrics
    }


if __name__ == "__main__":
    results = main()
