import os
import json
import copy
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================================================
# 0. 全局设置
# =========================================================
OUTPUT_DIR = "ft_all_configs_5fold_optimized"
SEED = 42
FIXED_THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# 1. 搜索配置
# =========================================================
CONFIGS = [
    {
        "name": "ft_01",
        "d_token": 16,
        "n_heads": 2,
        "n_layers": 1,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 80,
        "patience": 12,
    },
    {
        "name": "ft_02",
        "d_token": 16,
        "n_heads": 2,
        "n_layers": 2,
        "dropout": 0.2,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 80,
        "patience": 12,
    },
    {
        "name": "ft_03",
        "d_token": 32,
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 80,
        "patience": 12,
    },
    {
        "name": "ft_04",
        "d_token": 32,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.2,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 80,
        "patience": 12,
    },
    {
        "name": "ft_05",
        "d_token": 32,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 128,
        "max_epochs": 80,
        "patience": 12,
    },
    {
        "name": "ft_06",
        "d_token": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.3,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 80,
        "patience": 12,
    },
    {
        "name": "ft_stable",
        "d_token": 16,
        "n_heads": 2,
        "n_layers": 1,
        "dropout": 0.3,
        "lr": 5e-4,
        "weight_decay": 1e-3,
        "batch_size": 64,
        "max_epochs": 80,
        "patience": 12,
    },
]


# =========================================================
# 2. 数据
# =========================================================
def load_and_prepare_data():
    df = pd.read_excel("/Users/wangqinyang.5/Desktop/Infection/original.xlsx")
    y = df["outcome"].astype(int)

    num_cols = [
        "C_G", "C_WBC", "C_RBC", "C_P", "C_N",
        "GCS", "tem", "age",
        "B_G", "B_CRP", "B_WBC", "B_N", "B_Lym",
        "B_PCT", "B_AC", "B_RBC"
    ]

    cat_cols = [
        "transparency", "sex", "tube", "site", "other_inf"
    ]

    X = df[num_cols + cat_cols].copy()
    return X, y, num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    return ColumnTransformer([
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ])


def to_dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


# =========================================================
# 3. Dataset
# =========================================================
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# =========================================================
# 4. 模型
# =========================================================
class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=32, n_heads=4, n_layers=1, dropout=0.2):
        super().__init__()

        self.feature_embed = nn.Parameter(torch.randn(n_features, d_token) * 0.02)
        self.value_proj = nn.Linear(1, d_token)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)
        value_tokens = self.value_proj(x)
        tokens = value_tokens + self.feature_embed.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        logits = self.cls_head(pooled).squeeze(-1)
        return logits


# =========================================================
# 5. 评估与阈值搜索
# =========================================================
def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def find_best_threshold_by_mcc(y_true, y_prob, low=0.40, high=0.60, step=0.01):
    thresholds = np.arange(low, high + 1e-9, step)

    best_t = 0.5
    best_mcc = -1

    for t in thresholds:
        mcc = evaluate_metrics(y_true, y_prob, threshold=t)["mcc"]

        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t

    return best_t, best_mcc


def get_probs(model, loader, device):
    model.eval()
    probs = []

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            batch_probs = torch.sigmoid(logits).cpu().numpy()
            probs.extend(batch_probs)

    return np.asarray(probs)


# =========================================================
# 6. 单折训练
# =========================================================
def run_one_fold(config, X_fold_train, y_fold_train, X_fold_test, y_fold_test,
                 num_cols, cat_cols, fold_id, device):

    print("\n" + "=" * 90)
    print(f"Config={config['name']} | Fold={fold_id}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_fold_train,
        y_fold_train,
        test_size=0.2,
        stratify=y_fold_train,
        random_state=SEED
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_train_arr = to_dense_float32(preprocessor.fit_transform(X_train))
    X_val_arr = to_dense_float32(preprocessor.transform(X_val))
    X_test_arr = to_dense_float32(preprocessor.transform(X_fold_test))

    train_dataset = TabularDataset(X_train_arr, y_train.values)
    val_dataset = TabularDataset(X_val_arr, y_val.values)
    test_dataset = TabularDataset(X_test_arr, y_fold_test.values)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    model = FTTransformer(
        n_features=X_train_arr.shape[1],
        d_token=config["d_token"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"]
    ).to(device)

    pos_count = max((y_train == 1).sum(), 1)
    neg_count = max((y_train == 0).sum(), 1)

    # 关键优化：降低阳性权重强度，避免过度预测阳性
    pos_weight_value = np.sqrt(neg_count / pos_count)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    best_state = None
    best_val_score = -1
    best_val_auc = -1
    best_epoch = -1
    patience_counter = 0

    history = []

    for epoch in range(config["max_epochs"]):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        val_probs = get_probs(model, val_loader, device)

        val_metrics_05 = evaluate_metrics(
            y_true=y_val.values,
            y_prob=val_probs,
            threshold=FIXED_THRESHOLD
        )

        val_auc = val_metrics_05["auc"]
        val_mcc = val_metrics_05["mcc"]

        # 早停综合目标：AUC为主，MCC为辅
        val_score = val_auc + 0.10 * val_mcc

        history.append({
            "config": config["name"],
            "fold": fold_id,
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_auc": val_auc,
            "val_mcc": val_mcc,
            "val_score": val_score,
        })

        print(
            f"[{config['name']}][Fold {fold_id}] "
            f"Epoch {epoch+1:02d}/{config['max_epochs']} | "
            f"loss={avg_train_loss:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"val_mcc={val_mcc:.4f} | "
            f"score={val_score:.4f}"
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

            if patience_counter >= config["patience"]:
                print(f"[{config['name']}][Fold {fold_id}] Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 验证集上搜索轻微优化阈值
    final_val_probs = get_probs(model, val_loader, device)
    best_threshold, best_val_mcc_threshold = find_best_threshold_by_mcc(
        y_true=y_val.values,
        y_prob=final_val_probs,
        low=0.40,
        high=0.60,
        step=0.01
    )

    # 测试折预测
    test_probs = get_probs(model, test_loader, device)

    metrics_05 = evaluate_metrics(
        y_true=y_fold_test.values,
        y_prob=test_probs,
        threshold=FIXED_THRESHOLD
    )

    metrics_opt = evaluate_metrics(
        y_true=y_fold_test.values,
        y_prob=test_probs,
        threshold=best_threshold
    )

    fold_result = {
        "model": config["name"],
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "best_val_score": best_val_score,
        "best_threshold": best_threshold,
        "best_val_mcc_threshold": best_val_mcc_threshold,

        "test_accuracy": metrics_05["accuracy"],
        "test_precision": metrics_05["precision"],
        "test_recall": metrics_05["recall"],
        "test_f1": metrics_05["f1"],
        "test_auc": metrics_05["auc"],
        "test_mcc": metrics_05["mcc"],

        "test_accuracy_opt": metrics_opt["accuracy"],
        "test_precision_opt": metrics_opt["precision"],
        "test_recall_opt": metrics_opt["recall"],
        "test_f1_opt": metrics_opt["f1"],
        "test_auc_opt": metrics_opt["auc"],
        "test_mcc_opt": metrics_opt["mcc"],
    }

    pd.DataFrame(history).to_csv(
        os.path.join(OUTPUT_DIR, f"{config['name']}_fold{fold_id}_history.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame({
        "y_true": y_fold_test.values,
        "y_prob": test_probs,
        "y_pred_05": (test_probs >= FIXED_THRESHOLD).astype(int),
        "y_pred_opt": (test_probs >= best_threshold).astype(int),
        "best_threshold": best_threshold
    })

    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{config['name']}_fold{fold_id}_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "fold": fold_id,
            "input_dim": X_train_arr.shape[1],
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
            "best_val_score": best_val_score,
            "best_threshold": best_threshold,
        },
        os.path.join(OUTPUT_DIR, f"{config['name']}_fold{fold_id}_model.pth")
    )

    print(
        f"[{config['name']}][Fold {fold_id}] TEST@0.5 | "
        f"AUC={metrics_05['auc']:.4f} | "
        f"F1={metrics_05['f1']:.4f} | "
        f"MCC={metrics_05['mcc']:.4f}"
    )

    print(
        f"[{config['name']}][Fold {fold_id}] TEST@OPT(t={best_threshold:.2f}) | "
        f"AUC={metrics_opt['auc']:.4f} | "
        f"F1={metrics_opt['f1']:.4f} | "
        f"MCC={metrics_opt['mcc']:.4f}"
    )

    return fold_result


# =========================================================
# 7. 主程序
# =========================================================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] output_dir = {OUTPUT_DIR}")

    X, y, num_cols, cat_cols = load_and_prepare_data()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED
    )

    all_results = []

    for config in CONFIGS:
        print("\n" + "#" * 100)
        print(f"Running 5-fold CV for {config['name']}")
        print(config)

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_fold_train = X.iloc[train_idx].copy()
            y_fold_train = y.iloc[train_idx].copy()
            X_fold_test = X.iloc[test_idx].copy()
            y_fold_test = y.iloc[test_idx].copy()

            fold_result = run_one_fold(
                config=config,
                X_fold_train=X_fold_train,
                y_fold_train=y_fold_train,
                X_fold_test=X_fold_test,
                y_fold_test=y_fold_test,
                num_cols=num_cols,
                cat_cols=cat_cols,
                fold_id=fold_id,
                device=device
            )

            all_results.append(fold_result)

    fold_df = pd.DataFrame(all_results)

    fold_df.to_csv(
        os.path.join(OUTPUT_DIR, "ft_all_configs_5fold_details.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    metric_cols = [
        "best_epoch",
        "best_val_auc",
        "best_val_score",
        "best_threshold",

        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_auc",
        "test_mcc",

        "test_accuracy_opt",
        "test_precision_opt",
        "test_recall_opt",
        "test_f1_opt",
        "test_auc_opt",
        "test_mcc_opt",
    ]

    summary = fold_df.groupby("model")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns.to_flat_index()]
    summary = summary.reset_index()

    summary = summary.sort_values(
        by=["test_auc_mean", "test_f1_mean", "test_mcc_mean"],
        ascending=False
    )

    summary.to_csv(
        os.path.join(OUTPUT_DIR, "ft_all_configs_5fold_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    best_auc_result = summary.iloc[0].to_dict()

    with open(os.path.join(OUTPUT_DIR, "best_result_by_auc.json"), "w", encoding="utf-8") as f:
        json.dump(best_auc_result, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, "configs.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIGS, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 100)
    print("Summary:")
    print(summary.to_string(index=False))
    print(f"\n[INFO] Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()