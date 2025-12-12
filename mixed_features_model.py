import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ================== 数据加载 ==================

def load_original_data():
    df = pd.read_excel("original.xlsx")
    
    num_cols = [
        'C_G', 'C_WBC', 'C_RBC', 'C_P', 'C_N',
        'transparency', 'GCS', 'tem',
        'B_G', 'B_CRP', 'B_WBC', 'B_N',
        'B_Lym', 'B_PCT', 'B_AC', 'B_RBC'
    ]
    
    cat_cols = ['sex', 'tube', 'site', 'other_inf']
    
    return df, num_cols, cat_cols


def load_encoded_data():
    df_encoded = pd.read_csv("转化后_编码数据_最终版本.csv")
    encoded_cols = [col for col in df_encoded.columns if col not in ['ID', 'outcome']]
    return df_encoded, encoded_cols


# ================== 阈值搜索函数 ==================

def find_best_threshold(y_true, y_prob, metric="f1"):
    thresholds = np.linspace(0.1, 0.9, 81)

    best_t = 0.5
    best_score = 0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "recall":
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError("metric 只能为 'f1' 或 'recall'")

        if score > best_score:
            best_score = score
            best_t = t

    return best_t, best_score


# ================== 构建混合特征数据集 ==================

def create_mixed_features_dataset():
    df_original, num_cols, cat_cols = load_original_data()
    df_encoded, encoded_cols = load_encoded_data()
    
    mixed_df = df_original.copy()
    
    # ---------- 加入分级编码特征 ----------
    for col in encoded_cols:
        mixed_df[f"{col}_encoded"] = df_encoded[col].values
    
    # ---------- 工程特征 ----------
    mixed_df['csf_glucose_ratio'] = mixed_df['C_G'] / (mixed_df['B_G'] + 1e-3)
    mixed_df['low_glucose_flag'] = (mixed_df['csf_glucose_ratio'] < 0.4).astype(int)

    mixed_df['inflam_index'] = (
        np.log1p(mixed_df['B_CRP']) + np.log1p(mixed_df['B_WBC'])
    ) / 2.0

    mixed_df['csf_cell_protein_index'] = (
        np.log1p(mixed_df['C_WBC']) + np.log1p(mixed_df['C_P'])
    )

    mixed_df['immune_balance'] = mixed_df['B_Lym'] / (
        mixed_df['B_Lym'] + mixed_df['B_N'] + 1e-3
    )

    original_features = num_cols + cat_cols
    encoded_features = [f"{col}_encoded" for col in encoded_cols]
    engineered_features = [
        'csf_glucose_ratio',
        'low_glucose_flag',
        'inflam_index',
        'csf_cell_protein_index',
        'immune_balance'
    ]

    all_features = original_features + encoded_features + engineered_features
    
    X_mixed = mixed_df[all_features]
    y_mixed = mixed_df['outcome']

    # ============ 保存到 CSV ============
    save_df = mixed_df[all_features + ['outcome']]
    save_df.to_csv("混合特征数据集.csv", index=False, encoding="utf-8-sig")
    
   

    return X_mixed, y_mixed, all_features


# ================== 评估（自动最佳阈值） ==================

def evaluate_with_threshold(model, X_test, y_test, name, metric="f1"):
    y_prob = model.predict_proba(X_test)[:, 1]
    
    best_t, best_score = find_best_threshold(y_test, y_prob, metric=metric)
    
    y_pred = (y_prob >= best_t).astype(int)
    
    return {
        "model": name,
        "best_threshold": round(best_t, 3),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob)
    }


# ================== 训练 + 自动阈值 ==================

def train_and_eval_mixed_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results = []
    
    # 1. 决策树
    dt = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt.fit(X_train, y_train)
    results.append(evaluate_with_threshold(dt, X_test, y_test, "DecisionTree"))
    
    # 2. 随机森林
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)
    results.append(evaluate_with_threshold(rf, X_test, y_test, "RandomForest"))
    
    # 3. XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate_with_threshold(xgb, X_test, y_test, "XGBoost"))
    
    # 4. GBDT
    gbdt = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("gbdt", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ))
    ])
    gbdt.fit(X_train, y_train)
    results.append(evaluate_with_threshold(gbdt, X_test, y_test, "GBDT"))

    # 打印结果
    print("\n混合特征模型（自动搜索最佳阈值）结果：")
    print("{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12}".format(
        "Model", "Acc", "Prec", "Recall", "F1", "AUC", "Best_T"
    ))
    
    for r in results:
        print("{:<12} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>12}".format(
            r["model"], r["accuracy"], r["precision"], r["recall"],
            r["f1"], r["auc"], r["best_threshold"]
        ))
    
    return results


# ================== main ==================

def main():
    X_mixed, y_mixed, feature_names = create_mixed_features_dataset()
    results = train_and_eval_mixed_models(X_mixed, y_mixed)
    return results


if __name__ == "__main__":
    main()
