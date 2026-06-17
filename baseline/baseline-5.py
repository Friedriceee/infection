import os
import json
import pickle
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier  # 新增
from sklearn.metrics import matthews_corrcoef
warnings.filterwarnings("ignore")

# =========================
# 0. 全局设置
# =========================
OUTPUT_DIR = "baselineresults"
SEED = 42
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


set_seed(SEED)


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

    return ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])


# =========================
# 2. 模型定义
# =========================
def get_model_pipeline(model_type, num_cols, cat_cols, y_train_fold=None):
    if model_type == "logistic":
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",
            class_weight="balanced",
            max_iter=3000,
            random_state=SEED
        )


    elif model_type == "decision_tree":  # 新增决策树分支
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        clf = DecisionTreeClassifier(
            max_depth=4,              # 限制深度防止过拟合
            min_samples_split=10,
            class_weight="balanced",  # 考虑到临床数据往往不平衡
            random_state=SEED
        )

    elif model_type == "random_forest":
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            max_features="sqrt",
            random_state=SEED,
            n_jobs=-1
        )

    elif model_type == "xgboost":
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        pos_weight = (y_train_fold == 0).sum() / max((y_train_fold == 1).sum(), 1)
        clf = XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=pos_weight * 1.2,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1
        )

    elif model_type == "gbdt":
        preprocessor = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
        clf = GradientBoostingClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.8,
            max_features="sqrt",
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=SEED
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", clf)
    ])


# =========================
# 3. 五折交叉验证
# =========================
def run_cv_experiment(X, y, model_type, num_cols, cat_cols, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    cv_metrics = {
        "accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [], "mcc":[]
    }

    fold_results = []

    print(f"--- Running {n_splits}-Fold CV for: {model_type} ---")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        model = get_model_pipeline(model_type, num_cols, cat_cols, y_train_fold)
        model.fit(X_train_fold, y_train_fold)

        y_prob = model.predict_proba(X_val_fold)[:, 1]
        y_pred = (y_prob >= THRESHOLD).astype(int)

        fold_res = {
            "model": model_type,
            "fold": fold,
            "accuracy": accuracy_score(y_val_fold, y_pred),
            "precision": precision_score(y_val_fold, y_pred, zero_division=0),
            "recall": recall_score(y_val_fold, y_pred),
            "f1": f1_score(y_val_fold, y_pred, zero_division=0),
            "auc": roc_auc_score(y_val_fold, y_prob),
            "mcc": matthews_corrcoef(y_val_fold, y_pred),
        }

        for k in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
            cv_metrics[k].append(fold_res[k])

        fold_results.append(fold_res)

        print(f"Fold {fold}: AUC={fold_res['auc']:.3f}, F1={fold_res['f1']:.3f}")

        pd.DataFrame({
            "y_true": np.array(y_val_fold),
            "y_prob": y_prob,
            "y_pred": y_pred
        }).to_csv(
            os.path.join(OUTPUT_DIR, f"{model_type}_fold{fold}_predictions.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    fold_df = pd.DataFrame(fold_results)
    fold_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{model_type}_5fold_details.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary = {}
    for k, v in cv_metrics.items():
        summary[f"{k}_mean"] = np.mean(v)
        summary[f"{k}_std"] = np.std(v)

    print(f"Final Mean AUC: {summary['auc_mean']:.3f} ± {summary['auc_std']:.3f}\n")
    return summary


# =========================
# 4. 主函数
# =========================
def main():
    X, y, num_cols, cat_cols = load_and_prepare_data()

    model_list = ["logistic", "decision_tree", "random_forest", "xgboost", "gbdt"]
    all_results = {}

    for m_name in model_list:
        res = run_cv_experiment(X, y, m_name, num_cols, cat_cols, n_splits=5)
        all_results[m_name] = res

    results_df = pd.DataFrame(all_results).T

    print("===== Baseline 5-Fold Summary =====")
    print(results_df)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "baseline_5fold_summary.csv"),
        encoding="utf-8-sig"
    )

    with open(os.path.join(OUTPUT_DIR, "baseline_5fold_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    with open(os.path.join(OUTPUT_DIR, "baseline_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": SEED,
                "threshold": THRESHOLD,
                "models": model_list
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    return all_results


if __name__ == "__main__":
    main()