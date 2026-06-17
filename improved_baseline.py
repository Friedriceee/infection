"""
基线模型 —— 5折交叉验证版
与 AMFormer 完全对齐：
  - 相同特征集（含log变换 + 4个衍生特征）
  - 相同 StratifiedKFold(5, seed=42)
  - 相同 SMOTE（仅在训练折内做）
  - 相同 StandardScaler（fit on train, transform val）
  - AUC 作为主指标；同时用固定阈值0.5输出其余指标
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ── 配置（与 AMFormer 保持一致）──────────────────
DATA_PATH      = "original.xlsx"
OUTPUT_DIR     = Path("baseline_5fold_results")
N_SPLITS       = 5
SEED           = 42
FIXED_THRESHOLD = 0.5   # 与 AMFormer 一致


# ── 特征工程（与 AMFormer build_feature_matrix 完全一致）──
def build_feature_matrix(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # log 变换
    log_cols = ["C_WBC", "C_RBC", "C_P", "B_CRP", "B_WBC", "B_PCT", "B_AC", "B_RBC"]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    # 衍生特征
    eps = 1e-6
    if "C_G" in df.columns and "B_G" in df.columns:
        df["ratio_C_G_B_G"] = df["C_G"] / (df["B_G"] + eps)
    if "C_N" in df.columns and "B_N" in df.columns:
        df["diff_C_N_B_N"] = df["C_N"] - df["B_N"]
    if all(c in df.columns for c in ["C_WBC", "B_WBC", "C_RBC", "B_RBC"]):
        df["corrected_WBC"] = df["C_WBC"] - (df["B_WBC"] * df["C_RBC"] / (df["B_RBC"] + eps))
    if all(c in df.columns for c in ["B_WBC", "B_RBC", "C_WBC", "C_RBC"]):
        df["ratio_WBC_RBC_diff"] = (
            df["B_WBC"] / (df["B_RBC"] + eps) - df["C_WBC"] / (df["C_RBC"] + eps)
        )

    base_features = [
        "age", "C_G", "C_WBC", "C_RBC", "C_P", "C_N", "transparency",
        "GCS", "tem", "B_G", "B_CRP", "B_WBC", "B_N", "B_Lym",
        "B_PCT", "B_AC", "B_RBC", "sex", "tube", "site", "other_inf",
    ]
    new_features = ["ratio_C_G_B_G", "diff_C_N_B_N", "corrected_WBC", "ratio_WBC_RBC_diff"]
    feature_cols = base_features + [f for f in new_features if f in df.columns]

    X = df[feature_cols].values.astype(np.float32)
    y = df["outcome"].values
    return X, y, feature_cols


# ── 单折评估 ─────────────────────────────────────
def evaluate_fold(y_true, y_prob, threshold=FIXED_THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":       float(roc_auc_score(y_true, y_prob)),
        "threshold": threshold,
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


# ── 5折汇总 ──────────────────────────────────────
def summarize(fold_results: list[dict]) -> dict:
    keys = ["accuracy", "precision", "recall", "f1", "auc"]
    return {k: {"mean": float(np.mean([r[k] for r in fold_results])),
                "std":  float(np.std ([r[k] for r in fold_results]))}
            for k in keys}


# ── 单个模型的5折CV ──────────────────────────────
def run_cv(name: str, make_model_fn, X: np.ndarray, y: np.ndarray,
           output_dir: Path) -> list[dict]:

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_results = []

    print(f"\n{'='*20} {name} {'='*20}")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # StandardScaler（fit on train only）
        scaler = StandardScaler()
        X_tr  = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # SMOTE（仅在训练折内）
        sm = SMOTE(random_state=SEED)
        X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

        # 训练
        model = make_model_fn(y_tr_res)
        model.fit(X_tr_res, y_tr_res)

        # 验证
        y_prob = model.predict_proba(X_val)[:, 1]
        res = evaluate_fold(y_val, y_prob)
        fold_results.append(res)

        print(f"  Fold {fold} | AUC={res['auc']:.4f}  "
              f"Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  "
              f"Recall={res['recall']:.4f}")

    summary = summarize(fold_results)
    print(f"  >> Mean AUC : {summary['auc']['mean']:.4f} ± {summary['auc']['std']:.4f}")
    print(f"  >> Mean F1  : {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")

    # 保存
    tag = name.lower().replace(" ", "_")
    with open(output_dir / f"{tag}_fold_results.json", "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2)
    with open(output_dir / f"{tag}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return fold_results


# ── 各模型的工厂函数 ─────────────────────────────
def make_dt(_y=None):
    return DecisionTreeClassifier(
        max_depth=12, min_samples_split=8, min_samples_leaf=4,
        class_weight='balanced', random_state=SEED,
    )

def make_rf(_y=None):
    return RandomForestClassifier(
        n_estimators=400, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, class_weight='balanced',
        max_features='sqrt', random_state=SEED, n_jobs=-1,
    )

def make_xgb(y_tr):
    neg = (y_tr == 0).sum(); pos = (y_tr == 1).sum()
    return XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=neg / pos * 1.2,
        eval_metric="logloss", random_state=SEED,
        n_jobs=-1, tree_method='hist',
    )

def make_gbdt(_y=None):
    return GradientBoostingClassifier(
        n_estimators=600, learning_rate=0.02, max_depth=5,
        subsample=0.8, max_features='sqrt',
        min_samples_split=8, min_samples_leaf=4, random_state=SEED,
    )

# ── 汇总对比表 ───────────────────────────────────
def print_comparison_table(all_summaries: dict):
    print("\n" + "=" * 70)
    print("5折交叉验证汇总对比表（固定阈值 0.50）")
    print("=" * 70)
    print(f"{'模型':<22} {'AUC':>14} {'F1':>14} {'Recall':>14} {'Precision':>14}")
    print("-" * 70)
    for name, s in all_summaries.items():
        print(f"{name:<22} "
              f"{s['auc']['mean']:.4f}±{s['auc']['std']:.4f}  "
              f"{s['f1']['mean']:.4f}±{s['f1']['std']:.4f}  "
              f"{s['recall']['mean']:.4f}±{s['recall']['std']:.4f}  "
              f"{s['precision']['mean']:.4f}±{s['precision']['std']:.4f}")
    print("=" * 70)
    print("注：AMFormer 结果从 41706results/summary_metrics.json 手动填入对比")


# ── 主流程 ───────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("加载数据并构建特征矩阵（与 AMFormer 完全一致）...")
    df = pd.read_excel(DATA_PATH)
    X, y, feature_cols = build_feature_matrix(df)
    print(f"样本量: {len(y)}，特征数: {len(feature_cols)}，阳性: {y.sum()}，阴性: {(y==0).sum()}")

    all_summaries = {}

    for name, fn in [
        ("Decision Tree",    make_dt),
        ("Random Forest",    make_rf),
        ("XGBoost",          make_xgb),
        ("GBDT",             make_gbdt),
    ]:
        fold_res = run_cv(name, fn, X, y, OUTPUT_DIR)
        all_summaries[name] = summarize(fold_res)


    # 保存完整对比
    with open(OUTPUT_DIR / "all_models_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    # 输出对比表
    print_comparison_table(all_summaries)

    # 输出 CSV 方便复制进论文
    rows = []
    for model_name, s in all_summaries.items():
        rows.append({
            "Model": model_name,
            "AUC_mean": round(s["auc"]["mean"], 4),
            "AUC_std":  round(s["auc"]["std"],  4),
            "F1_mean":  round(s["f1"]["mean"],  4),
            "F1_std":   round(s["f1"]["std"],   4),
            "Recall_mean":    round(s["recall"]["mean"],    4),
            "Precision_mean": round(s["precision"]["mean"], 4),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison_table.csv",
                               index=False, encoding="utf-8-sig")
    print(f"\n结果已保存至 {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()