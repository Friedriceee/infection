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
    f1_score, roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import matthews_corrcoef

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# =========================================================
# 0. 全局设置
# =========================================================
OUTPUT_DIR = "ft_compare_5fold"
THRESHOLD = 0.5
SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# 1. 只比较这两个 config
# =========================================================
CONFIGS = [
    {
        "name": "ft_03",
        "d_token": 32,
        "n_heads": 4,
        "n_layers": 1,
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 128,
        "max_epochs": 50,
        "patience": 8,
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
        "max_epochs": 50,
        "patience": 8,
    },
]


# =========================================================
# 2. 数据
# =========================================================
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
# 4. FT-Transformer
# =========================================================
class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=32, n_heads=4, n_layers=2, dropout=0.2):
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

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
# 5. 评估
# =========================================================
def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

def find_best_threshold_limited(y_true, y_prob):
    """
    在0.4~0.6之间找最优阈值（保证公平性）
    优化目标：MCC
    """
    thresholds = np.arange(0.4, 0.61, 0.01)

    best_t = 0.5
    best_mcc = -1

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)

        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t

    return best_t
# =========================================================
# 6. 单折训练
# =========================================================
def run_one_fold(config, X_fold_train, y_fold_train, X_fold_test, y_fold_test,
                 num_cols, cat_cols, fold_id, device):

    # 从当前 fold_train 再切出 validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_fold_train, y_fold_train,
        test_size=0.2,
        stratify=y_fold_train,
        random_state=SEED
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_val_arr = preprocessor.transform(X_val)
    X_test_arr = preprocessor.transform(X_fold_test)

    if hasattr(X_train_arr, "toarray"):
        X_train_arr = X_train_arr.toarray()
    if hasattr(X_val_arr, "toarray"):
        X_val_arr = X_val_arr.toarray()
    if hasattr(X_test_arr, "toarray"):
        X_test_arr = X_test_arr.toarray()

    X_train_arr = np.asarray(X_train_arr, dtype=np.float32)
    X_val_arr = np.asarray(X_val_arr, dtype=np.float32)
    X_test_arr = np.asarray(X_test_arr, dtype=np.float32)

    train_dataset = TabularDataset(X_train_arr, y_train.values)
    val_dataset = TabularDataset(X_val_arr, y_val.values)
    test_dataset = TabularDataset(X_test_arr, y_fold_test.values)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = FTTransformer(
        n_features=X_train_arr.shape[1],
        d_token=config["d_token"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"]
    ).to(device)

    pos_count = max((y_train == 1).sum(), 1)
    neg_count = max((y_train == 0).sum(), 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    best_state = None
    best_val_auc = -1
    best_epoch = -1
    patience_counter = 0

    for epoch in range(config["max_epochs"]):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)

        val_auc = roc_auc_score(y_val, np.array(val_probs))

        val_probs = np.array(val_probs)

        best_threshold = find_best_threshold_limited(
            y_val.values,
            val_probs
        )
        print(
            f"[{config['name']}][Fold {fold_id}] Epoch {epoch+1}/{config['max_epochs']} "
            f"| val_auc={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"[{config['name']}][Fold {fold_id}] Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()

    test_probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)

    test_probs = np.array(test_probs)
    metrics = evaluate_metrics(
    y_fold_test,
    test_probs,
    threshold=best_threshold
)

    fold_result = {
        "model": config["name"],
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_auc": metrics["auc"],
        "test_mcc": metrics["mcc"],
    }

    pred_df = pd.DataFrame({
        "y_true": np.array(y_fold_test),
        "y_prob": test_probs,
        "y_pred": (test_probs >= best_threshold).astype(int)
    })

    pred_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{config['name']}_fold{fold_id}_predictions.csv"
        ),
        index=False,
        encoding="utf-8-sig"
)

    return fold_result





# =========================================================
# 7. 主程序
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    X, y, num_cols, cat_cols = load_and_prepare_data()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    all_fold_results = []

    for config in CONFIGS:
        print("\n" + "=" * 100)
        print(f"Running 5-fold CV for {config['name']}")
        print(config)

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_fold_train = X.iloc[train_idx].copy()
            y_fold_train = y.iloc[train_idx].copy()
            X_fold_test = X.iloc[test_idx].copy()
            y_fold_test = y.iloc[test_idx].copy()

            fold_result = run_one_fold(
                config,
                X_fold_train, y_fold_train,
                X_fold_test, y_fold_test,
                num_cols, cat_cols,
                fold_id, device
            )
            all_fold_results.append(fold_result)

    fold_df = pd.DataFrame(all_fold_results)
    fold_df.to_csv(
        os.path.join(OUTPUT_DIR, "ft_03_ft_05_5fold_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary_df = fold_df.groupby("model").agg({
        "best_epoch": ["mean", "std"],
        "best_val_auc": ["mean", "std"],
        "test_accuracy": ["mean", "std"],
        "test_precision": ["mean", "std"],
        "test_recall": ["mean", "std"],
        "test_f1": ["mean", "std"],
        "test_auc": ["mean", "std"],
        "test_mcc": ["mean", "std"],
    })

    summary_df.columns = [
        f"{a}_{b}" for a, b in summary_df.columns.to_flat_index()
    ]
    summary_df = summary_df.reset_index()

    summary_df.to_csv(
        os.path.join(OUTPUT_DIR, "ft_03_ft_05_5fold_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 100)
    print("5-fold summary:")
    print(summary_df.to_string(index=False))

    with open(os.path.join(OUTPUT_DIR, "configs.json"), "w", encoding="utf-8") as f:
        json.dump(CONFIGS, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()