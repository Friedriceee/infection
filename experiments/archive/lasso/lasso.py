import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


# ── 数据加载 & 预处理（不变） ──────────────────────────────────────────

def load_data():
    df = pd.read_excel("original.xlsx")
    y = df["outcome"].astype(int)
    num_cols = [
        'C_G', 'C_WBC', 'C_RBC', 'C_P', 'C_N',
        'GCS', 'tem', 'age',
        'B_G', 'B_CRP', 'B_WBC', 'B_N', 'B_Lym', 'B_PCT', 'B_AC', 'B_RBC',
    ]
    cat_cols = ['transparency', 'sex', 'tube', 'site', 'other_inf']
    X = df[num_cols + cat_cols].copy()
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ]), cat_cols)
    ])


def get_feature_names(preprocessor, num_cols, cat_cols):
    names = list(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    names.extend(ohe.get_feature_names_out(cat_cols).tolist())
    return names


# ── 稳定性筛选核心 ────────────────────────────────────────────────────

def stability_selection(
    X_train_arr, y_train_arr, feature_names,
    C=0.05,              # ← 固定一个中等偏强的正则，不让 CV 反向选弱正则
    n_bootstrap=200,     # 重采样次数
    subsample_ratio=0.7, # 每次用 70% 样本
    threshold=0.5,       # 被选中概率 > 60% 才算稳定
    random_state=42
):
    """
    对每个 bootstrap 子样本跑 L1 Logistic，
    统计每个特征被选中（系数非零）的频率。
    频率超过 threshold 的特征视为稳定特征。
    """
    rng = np.random.RandomState(random_state)
    n_samples = X_train_arr.shape[0]
    n_features = X_train_arr.shape[1]
    selection_counts = np.zeros(n_features)

    for i in range(n_bootstrap):
        # 分层 bootstrap（保持正负样本比例）
        idx = resample(
            np.arange(n_samples),
            n_samples=int(n_samples * subsample_ratio),
            stratify=y_train_arr,
            random_state=rng.randint(0, 99999)
        )
        X_sub = X_train_arr[idx]
        y_sub = y_train_arr[idx]

        model = LogisticRegression(
            penalty="l1", solver="saga", C=C,
            class_weight="balanced", max_iter=5000,
            random_state=rng.randint(0, 99999)
        )
        model.fit(X_sub, y_sub)
        selected = (np.abs(model.coef_.ravel()) > 1e-6)
        selection_counts += selected.astype(int)

    selection_prob = selection_counts / n_bootstrap

    prob_df = pd.DataFrame({
        "feature": feature_names,
        "selection_prob": selection_prob
    }).sort_values("selection_prob", ascending=False)

    stable_features = prob_df[prob_df["selection_prob"] >= threshold]["feature"].tolist()

    print(f"\n稳定性筛选参数：C={C}, bootstrap={n_bootstrap}次, 子样本比={subsample_ratio}, 阈值={threshold}")
    print(f"\n{'特征':<30} {'入选概率':>10}")
    print("-" * 42)
    for _, row in prob_df.iterrows():
        flag = " ✓" if row["selection_prob"] >= threshold else ""
        print(f"{row['feature']:<30} {row['selection_prob']:>9.1%}{flag}")

    print(f"\n稳定入选：{len(stable_features)} / {len(feature_names)} 个特征")
    print(stable_features)

    return stable_features, prob_df


# ── 主流程 ────────────────────────────────────────────────────────────

def lasso_select_features(X, y, num_cols, cat_cols,
                          test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 预处理只在训练集 fit
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr  = preprocessor.transform(X_test)
    feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

    # 稳定性筛选
    # 👇 如果还是筛不掉，把 threshold 调高到 0.7 或把 C 调小到 0.01
    selected_features, prob_df = stability_selection(
        X_train_arr, y_train.values,
        feature_names,
        C=0.05,
        n_bootstrap=200,
        subsample_ratio=0.7,
        threshold=0.5
    )

    # 转 DataFrame
    X_train_df = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train.index)
    X_test_df  = pd.DataFrame(X_test_arr,  columns=feature_names, index=X_test.index)

    X_train_selected = X_train_df[selected_features].copy()
    X_test_selected  = X_test_df[selected_features].copy()

        # 保存到 lasso 文件夹
    output_dir = "lasso"
    os.makedirs(output_dir, exist_ok=True)

    prob_df.to_csv(os.path.join(output_dir, "stability_selection_prob.csv"),
                   index=False, encoding="utf-8-sig")

    X_train_df.to_csv(os.path.join(output_dir, "X_train_full.csv"),
                      index=False, encoding="utf-8-sig")
    X_test_df.to_csv(os.path.join(output_dir, "X_test_full.csv"),
                     index=False, encoding="utf-8-sig")

    X_train_selected.to_csv(os.path.join(output_dir, "X_train_lasso_selected.csv"),
                            index=False, encoding="utf-8-sig")
    X_test_selected.to_csv(os.path.join(output_dir, "X_test_lasso_selected.csv"),
                           index=False, encoding="utf-8-sig")

    y_train.to_csv(os.path.join(output_dir, "y_train.csv"),
                   index=False, encoding="utf-8-sig")
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"),
                  index=False, encoding="utf-8-sig")

    return {
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "selected_features": selected_features,
        "prob_df": prob_df,
        "X_train_full": X_train_df,
        "X_test_full": X_test_df,
        "X_train_selected": X_train_selected,
        "X_test_selected": X_test_selected,
        "y_train": y_train,
        "y_test": y_test
    }


if __name__ == "__main__":
    X, y, num_cols, cat_cols = load_data()
   
    result = lasso_select_features(X, y, num_cols, cat_cols)