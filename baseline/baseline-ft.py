import os
import json
import copy
import random
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
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
OUTPUT_DIR = "ft_tune"
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
# 1. 配置列表
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
        "max_epochs": 50,
        "patience": 8,
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
        "max_epochs": 50,
        "patience": 8,
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
        "max_epochs": 50,
        "patience": 8,
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
    {
        "name": "ft_06",
        "d_token": 64,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.3,
        "lr": 2e-4,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "max_epochs": 50,
        "patience": 8,
    },
]


# =========================================================
# 2. 数据读取与预处理
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
# 3. 数据集类
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
# 4. FT-Transformer 模型
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
        # x: [B, F]
        x = x.unsqueeze(-1)                 # [B, F, 1]
        value_tokens = self.value_proj(x)  # [B, F, d]
        tokens = value_tokens + self.feature_embed.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        logits = self.cls_head(pooled).squeeze(-1)
        return logits


# =========================================================
# 5. 评估函数
# =========================================================
def evaluate_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_val_fold, y_pred),
    }


# =========================================================
# 6. 单组 config 训练
# =========================================================
def train_one_config(X_train_arr, X_val_arr, X_test_arr, y_train, y_val, y_test, config, device):
    print("\n" + "=" * 90)
    print(f"Running config: {config['name']}")
    print(config)

    train_dataset = TabularDataset(X_train_arr, y_train.values)
    val_dataset = TabularDataset(X_val_arr, y_val.values)
    test_dataset = TabularDataset(X_test_arr, y_test.values)

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
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    best_state = None
    best_val_auc = -1.0
    best_epoch = -1
    patience_counter = 0

    history = []

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

        avg_train_loss = epoch_loss / max(len(train_loader), 1)

        # validation
        model.eval()
        val_probs = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)

        val_probs = np.array(val_probs)
        val_auc = roc_auc_score(y_val, val_probs)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_auc": val_auc
        })

        print(
            f"[{config['name']}] Epoch {epoch+1:02d}/{config['max_epochs']} "
            f"| train_loss={avg_train_loss:.4f} | val_auc={val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"[{config['name']}] Early stopping at epoch {epoch+1}")
                break

    # 恢复 best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # test evaluation
    model.eval()
    test_probs = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)

    test_probs = np.array(test_probs)
    metrics = evaluate_metrics(y_test, test_probs, threshold=THRESHOLD)

    result = {
        **config,
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_auc": metrics["auc"],
        "test_mcc": metrics["mcc"],
    }

    print(f"[{config['name']}] best_epoch={best_epoch}, best_val_auc={best_val_auc:.4f}")
    print(
        f"[{config['name']}] TEST | "
        f"ACC={metrics['accuracy']:.4f} | "
        f"PREC={metrics['precision']:.4f} | "
        f"REC={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"AUC={metrics['auc']:.4f}"
        f"mcc={metrics['mcc']:.4f}"
    )

    # 保存本组历史和预测
    history_df = pd.DataFrame(history)
    history_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{config['name']}_history.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame({
        "y_true": np.array(y_test),
        "y_prob": test_probs,
        "y_pred": (test_probs >= THRESHOLD).astype(int)
    })
    pred_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{config['name']}_test_predictions.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "input_dim": X_train_arr.shape[1],
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
        },
        os.path.join(OUTPUT_DIR, f"{config['name']}_model.pth")
    )

    return result


# =========================================================
# 7. 主程序
# =========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")
    print(f"[INFO] output_dir = {OUTPUT_DIR}")
    print(f"[INFO] threshold = {THRESHOLD}")

    X, y, num_cols, cat_cols = load_and_prepare_data()

    # 第一次划分：train_val / test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=SEED
    )

    # 第二次划分：train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.2,
        stratify=y_train_val,
        random_state=SEED
    )

    print(f"[INFO] train size = {len(X_train)}")
    print(f"[INFO] val size   = {len(X_val)}")
    print(f"[INFO] test size  = {len(X_test)}")

    # 预处理器只在 train 上 fit
    preprocessor = build_preprocessor(num_cols, cat_cols)

    X_train_arr = preprocessor.fit_transform(X_train)
    X_val_arr = preprocessor.transform(X_val)
    X_test_arr = preprocessor.transform(X_test)

    # 稀疏转 dense
    if hasattr(X_train_arr, "toarray"):
        X_train_arr = X_train_arr.toarray()
    if hasattr(X_val_arr, "toarray"):
        X_val_arr = X_val_arr.toarray()
    if hasattr(X_test_arr, "toarray"):
        X_test_arr = X_test_arr.toarray()

    X_train_arr = np.asarray(X_train_arr, dtype=np.float32)
    X_val_arr = np.asarray(X_val_arr, dtype=np.float32)
    X_test_arr = np.asarray(X_test_arr, dtype=np.float32)

    print(f"[INFO] input_dim = {X_train_arr.shape[1]}")

    # 保存 split 信息
    split_info = {
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "input_dim": int(X_train_arr.shape[1]),
        "threshold": THRESHOLD,
        "seed": SEED,
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    results = []

    for config in CONFIGS:
        result = train_one_config(
            X_train_arr, X_val_arr, X_test_arr,
            y_train, y_val, y_test,
            config, device
        )
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("test_auc", ascending=False)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "ft_tuning_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 90)
    print("Top tuning results:")
    print(results_df.to_string(index=False))

    # 保存 best config
    best_result = results_df.iloc[0].to_dict()
    with open(os.path.join(OUTPUT_DIR, "best_result.json"), "w", encoding="utf-8") as f:
        json.dump(best_result, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Saved all results to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()