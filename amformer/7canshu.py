import os
import json
import pickle
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"


# =============================================================
# 工具函数
# =============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


# =============================================================
# 配置
# =============================================================
@dataclass
class Config:
    data_path: str = "/Users/wangqinyang.5/Desktop/Infection/original.xlsx"
    output_dir: str = "7canshu"
    label_col: str = "outcome"

    batch_size: int = 32
    epochs: int = 120
    lr: float = 3e-4
    min_lr: float = 1e-5
    patience: int = 25
    fixed_threshold: float = 0.5

    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    top_k: int | None = None
    dropout: float = 0.20
    ff_mult: int = 4

    seed: int = 42
    n_splits: int = 5
    pos_weight_scale: float = 1.0
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 是否在搜索结束后，用最佳参数再完整跑一遍并保存每折文件
    rerun_best_after_search: bool = True


CFG = Config()

# =============================================================
# 网格搜索空间
# =============================================================
GRID_SEARCH_SPACE = {
    "top_k": [8],
    "pos_weight_scale": [0.9],
    "embed_dim": [64],
}


# =============================================================
# 数据集
# =============================================================
class CSFDataset(Dataset):
    def __init__(self, x_num, x_cat, labels=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.x_num[idx], self.x_cat[idx]
        return self.x_num[idx], self.x_cat[idx], self.labels[idx]


# =============================================================
# 特征工程
# =============================================================
def build_feature_dataframe(df: pd.DataFrame, cfg: Config):
    df = df.copy()

    cat_cols = [ "other_inf"]

    base_num_cols = [
        "age", "C_G", "C_WBC", "C_N",
        "GCS", "B_PCT", 
    ]

    cat_cols = [c for c in cat_cols if c in df.columns]
    base_num_cols = [c for c in base_num_cols if c in df.columns]

    for c in base_num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    log_cols = ["C_WBC", "C_RBC", "C_P", "B_CRP", "B_WBC", "B_PCT", "B_AC", "B_RBC"]
    for col in log_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = np.log1p(df[col].clip(lower=0))

    for c in base_num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    eps = 1e-6
    new_num_cols = []

    if "C_G" in df.columns and "B_G" in df.columns:
        df["ratio_C_G_B_G"] = df["C_G"] / (df["B_G"] + eps)
        new_num_cols.append("ratio_C_G_B_G")

    if "C_N" in df.columns and "B_N" in df.columns:
        df["diff_C_N_B_N"] = df["C_N"] - df["B_N"]
        new_num_cols.append("diff_C_N_B_N")

    if all(c in df.columns for c in ["C_WBC", "B_WBC", "C_RBC", "B_RBC"]):
        df["corrected_WBC"] = df["C_WBC"] - df["B_WBC"] * df["C_RBC"] / (df["B_RBC"] + eps)
        new_num_cols.append("corrected_WBC")

    if all(c in df.columns for c in ["B_WBC", "B_RBC", "C_WBC", "C_RBC"]):
        df["ratio_WBC_RBC_diff"] = (
            df["B_WBC"] / (df["B_RBC"] + eps) - df["C_WBC"] / (df["C_RBC"] + eps)
        )
        new_num_cols.append("ratio_WBC_RBC_diff")

    num_cols = base_num_cols + new_num_cols

    for c in new_num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown"})

    if cfg.label_col not in df.columns:
        raise ValueError(f"找不到标签列: {cfg.label_col}")

    y = pd.to_numeric(df[cfg.label_col], errors="coerce").values.astype(int)
    return df, num_cols, cat_cols, y


# =============================================================
# 类别编码器
# =============================================================
class CategoryEncoder:
    def __init__(self):
        self.maps = {}

    def fit(self, df: pd.DataFrame, cat_cols):
        self.maps = {}
        for c in cat_cols:
            values = df[c].astype(str).fillna("Unknown").tolist()
            uniq = sorted(list(set(values)))
            self.maps[c] = {v: i + 1 for i, v in enumerate(uniq)}  # 0 保留给未知
        return self

    def transform(self, df: pd.DataFrame, cat_cols):
        out = []
        for c in cat_cols:
            mapper = self.maps[c]
            vals = df[c].astype(str).fillna("Unknown").tolist()
            out.append([mapper.get(v, 0) for v in vals])
        if len(out) == 0:
            return np.zeros((len(df), 0), dtype=np.int64)
        return np.array(out, dtype=np.int64).T

    def get_cardinalities(self, cat_cols):
        return [max(self.maps[c].values(), default=0) + 1 for c in cat_cols]


# =============================================================
# 模型模块
# =============================================================
class TopKSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, top_k=None, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.top_k = top_k

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, L, _ = x.size()

        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if self.top_k is not None and self.top_k < L:
            effective_k = min(self.top_k, L)
            topk_vals, _ = torch.topk(scores, effective_k, dim=-1)
            threshold = topk_vals[..., -1:].expand_as(scores)
            scores = scores.masked_fill(scores < threshold, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        return self.w_o(context), attn_weights


class GatedArithmeticBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 3, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)

        add_feat = h + h_mean
        mul_feat = h * h_mean
        sub_feat = h - h_mean

        combined = torch.cat([add_feat, mul_feat, sub_feat], dim=-1)
        out = self.dropout(self.proj(combined))
        gate = self.gate(h_mean)

        return self.norm(h + gate * out)


class NumericFeatureEmbedding(nn.Module):
    def __init__(self, n_num_features: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_num_features)
        ])

    def forward(self, x_num: torch.Tensor):
        if x_num.size(1) == 0:
            return None
        tokens = [self.embeddings[i](x_num[:, i:i+1]) for i in range(len(self.embeddings))]
        return torch.stack(tokens, dim=1)


class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, cat_cardinalities, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities
        ])

    def forward(self, x_cat: torch.Tensor):
        if x_cat.size(1) == 0:
            return None
        tokens = [self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))]
        return torch.stack(tokens, dim=1)


class ImprovedAMFormerV2(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        top_k: int | None = None,
        dropout: float = 0.2,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)
        self.total_tokens = n_num_features + len(cat_cardinalities)
        self.embed_dim = embed_dim

        self.num_embedding = NumericFeatureEmbedding(n_num_features, embed_dim)
        self.cat_embedding = CategoricalFeatureEmbedding(cat_cardinalities, embed_dim)

        self.pos_encoding = nn.Parameter(
            torch.randn(1, max(1, self.total_tokens), embed_dim) * 0.01
        )

        self.attn_layers = nn.ModuleList([
            TopKSparseAttention(embed_dim, n_heads, top_k=top_k, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * ff_mult, embed_dim),
                nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])

        self.arith_blocks = nn.ModuleList([
            GatedArithmeticBlock(embed_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm1_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, return_attention: bool = False):
        num_tokens = self.num_embedding(x_num)
        cat_tokens = self.cat_embedding(x_cat)

        if num_tokens is not None and cat_tokens is not None:
            h = torch.cat([num_tokens, cat_tokens], dim=1)
        elif num_tokens is not None:
            h = num_tokens
        elif cat_tokens is not None:
            h = cat_tokens
        else:
            raise ValueError("数值特征和类别特征不能同时为空。")

        h = h + self.pos_encoding[:, :h.size(1), :]
        attention_maps = []

        for attn, ffn, arith, norm1, norm2 in zip(
            self.attn_layers, self.ffn_layers, self.arith_blocks,
            self.norm1_layers, self.norm2_layers
        ):
            attn_out, attn_w = attn(h)
            attention_maps.append(attn_w.detach().cpu().numpy())
            h = norm1(h + attn_out)
            h = norm2(h + ffn(h))
            h = arith(h)

        out_mean = h.mean(dim=1)
        out_max, _ = h.max(dim=1)
        out_feat = torch.cat([out_mean, out_max], dim=-1)

        logits = self.classifier(out_feat).squeeze(-1)
        if return_attention:
            return logits, attention_maps
        return logits


# =============================================================
# 训练器
# =============================================================
class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu", pos_weight: torch.Tensor | None = None):
        self.model = model.to(device)
        self.device = device
        self.pos_weight = pos_weight
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []

    def train_epoch(self, train_loader, optimizer, criterion, grad_clip=1.0):
        self.model.train()
        total_loss = 0.0

        for batch_num, batch_cat, batch_labels in tqdm(train_loader, desc="Training", leave=False):
            batch_num = batch_num.to(self.device)
            batch_cat = batch_cat.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(batch_num, batch_cat)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch_num, batch_cat, batch_labels in val_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_num, batch_cat)
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        val_auc = safe_auc(all_labels, all_probs)
        return total_loss / max(len(val_loader), 1), all_probs, all_labels, val_auc

    def train(self, train_loader, val_loader, save_path, epochs=100, lr=1e-3, patience=20,
              weight_decay=1e-2, min_lr=1e-5, grad_clip=1.0):
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=6, min_lr=min_lr
        )

        best_val_auc = -1.0
        patience_counter = 0

        torch.save(self.model.state_dict(), save_path)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion, grad_clip=grad_clip)
            val_loss, _, _, val_auc = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)

            scheduler.step(val_auc)

            if val_auc > best_val_auc + 1e-5:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))

    def evaluate(self, val_loader, threshold: float = 0.5) -> dict:
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch_num, batch_cat, batch_labels in val_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                logits = self.model(batch_num, batch_cat)

                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        probs = np.array(all_probs)
        labels = np.array(all_labels)
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(labels, preds)

        return {
            "predictions": probs.tolist(),
            "labels": labels.tolist(),
            "threshold": threshold,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc": safe_auc(labels, probs),
            "confusion_matrix": cm.tolist(),
        }


# =============================================================
# 汇总函数
# =============================================================
def summarize_results(fold_results, threshold):
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
    summary["threshold"] = threshold
    return summary


# =============================================================
# 单组参数跑 5 折
# =============================================================
def run_single_experiment(cfg: Config, df, num_cols, cat_cols, y, exp_output_dir=None, save_fold_files=False):
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_results = []
    fold_summary_rows = []

    if exp_output_dir is not None:
        exp_output_dir = Path(exp_output_dir)
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        scaler = StandardScaler()
        x_train_num = scaler.fit_transform(df_train[num_cols].values.astype(np.float32)) if len(num_cols) > 0 else np.zeros((len(df_train), 0), dtype=np.float32)
        x_val_num = scaler.transform(df_val[num_cols].values.astype(np.float32)) if len(num_cols) > 0 else np.zeros((len(df_val), 0), dtype=np.float32)

        cat_encoder = CategoryEncoder().fit(df_train, cat_cols)
        x_train_cat = cat_encoder.transform(df_train, cat_cols)
        x_val_cat = cat_encoder.transform(df_val, cat_cols)
        cat_cardinalities = cat_encoder.get_cardinalities(cat_cols)

        train_loader = DataLoader(
            CSFDataset(x_train_num, x_train_cat, y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            CSFDataset(x_val_num, x_val_cat, y_val),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers
        )

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        pw_value = (neg_count / max(pos_count, 1)) * cfg.pos_weight_scale
        pos_weight = torch.tensor(pw_value, dtype=torch.float32).to(cfg.device)

        model = ImprovedAMFormerV2(
            n_num_features=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            top_k=cfg.top_k,
            dropout=cfg.dropout,
            ff_mult=cfg.ff_mult,
        )

        trainer = ModelTrainer(model, device=cfg.device, pos_weight=pos_weight)

        if exp_output_dir is not None:
            model_path = exp_output_dir / f"best_model_fold{fold}.pth"
        else:
            model_path = Path(f"temp_best_model_fold{fold}.pth")

        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            save_path=str(model_path),
            epochs=cfg.epochs,
            lr=cfg.lr,
            patience=cfg.patience,
            weight_decay=cfg.weight_decay,
            min_lr=cfg.min_lr,
            grad_clip=cfg.grad_clip,
        )

        results = trainer.evaluate(val_loader, threshold=cfg.fixed_threshold)
        fold_results.append(results)

        fold_summary = {
            "fold": fold,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "auc": results["auc"],
            "threshold": cfg.fixed_threshold,
            "tn": results["confusion_matrix"][0][0],
            "fp": results["confusion_matrix"][0][1],
            "fn": results["confusion_matrix"][1][0],
            "tp": results["confusion_matrix"][1][1],
        }
        fold_summary_rows.append(fold_summary)

        if save_fold_files and exp_output_dir is not None:
            with open(exp_output_dir / f"fold_{fold}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(fold_summary, f, ensure_ascii=False, indent=2)

            pd.DataFrame({
                "y_true": results["labels"],
                "y_prob": results["predictions"],
                "threshold": results["threshold"],
            }).to_csv(exp_output_dir / f"fold_{fold}_predictions.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame({
                "epoch": np.arange(len(trainer.train_losses)),
                "train_loss": trainer.train_losses,
                "val_loss": trainer.val_losses,
                "val_auc": trainer.val_aucs,
            }).to_csv(exp_output_dir / f"fold_{fold}_learning_curve.csv", index=False, encoding="utf-8-sig")

            with open(exp_output_dir / f"fold_{fold}_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

            with open(exp_output_dir / f"fold_{fold}_cat_encoder.pkl", "wb") as f:
                pickle.dump(cat_encoder, f)

    summary = summarize_results(fold_results, cfg.fixed_threshold)
    return summary, fold_summary_rows


# =============================================================
# 自动网格搜索
# =============================================================
def grid_search(cfg: Config, df, num_cols, cat_cols, y):
    search_dir = Path(cfg.output_dir)
    search_dir.mkdir(parents=True, exist_ok=True)

    keys = list(GRID_SEARCH_SPACE.keys())
    values = list(GRID_SEARCH_SPACE.values())
    all_combinations = list(product(*values))

    print(f"总共需要搜索 {len(all_combinations)} 组参数。\n")

    search_results = []

    for i, combo in enumerate(all_combinations, start=1):
        params = dict(zip(keys, combo))
        exp_cfg = deepcopy(cfg)

        for k, v in params.items():
            setattr(exp_cfg, k, v)

        exp_name = f"exp_{i:02d}_topk_{params['top_k']}_pw_{params['pos_weight_scale']}_emb_{params['embed_dim']}"
        exp_dir = search_dir / exp_name

        print("=" * 80)
        print(f"[{i}/{len(all_combinations)}] 开始参数组合: {params}")
        print("=" * 80)

        summary, _ = run_single_experiment(
            cfg=exp_cfg,
            df=df,
            num_cols=num_cols,
            cat_cols=cat_cols,
            y=y,
            exp_output_dir=None,      # 搜索阶段不保存每折大文件，节省空间
            save_fold_files=False
        )

        row = {
            "exp_name": exp_name,
            "top_k": params["top_k"],
            "pos_weight_scale": params["pos_weight_scale"],
            "embed_dim": params["embed_dim"],
            "auc_mean": summary["auc"]["mean"],
            "auc_std": summary["auc"]["std"],
            "f1_mean": summary["f1"]["mean"],
            "recall_mean": summary["recall"]["mean"],
            "precision_mean": summary["precision"]["mean"],
            "accuracy_mean": summary["accuracy"]["mean"],
        }
        search_results.append(row)

        print(
            f"完成: {params} | "
            f"AUC={row['auc_mean']:.4f} ± {row['auc_std']:.4f} | "
            f"F1={row['f1_mean']:.4f} | "
            f"Recall={row['recall_mean']:.4f} | "
            f"Precision={row['precision_mean']:.4f}"
        )
        print()

        # 实时保存
        pd.DataFrame(search_results).sort_values(
            by=["auc_mean", "auc_std"], ascending=[False, True]
        ).to_csv(search_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")

        with open(search_dir / "grid_search_results.json", "w", encoding="utf-8") as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)

    results_df = pd.DataFrame(search_results).sort_values(
        by=["auc_mean", "auc_std"], ascending=[False, True]
    ).reset_index(drop=True)

    results_df.to_csv(search_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    with open(search_dir / "grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2)

    best_row = results_df.iloc[0].to_dict()
    print("\n" + "#" * 80)
    print("网格搜索完成，最佳参数为：")
    print(best_row)
    print("#" * 80 + "\n")

    return best_row, results_df


# =============================================================
# 主函数
# =============================================================
def main():
    set_seed(CFG.seed)

    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    raw_df = pd.read_excel(CFG.data_path)
    df, num_cols, cat_cols, y = build_feature_dataframe(raw_df, CFG)

    print(f"样本数: {len(df)}")
    print(f"阳性样本数: {int(y.sum())}")
    print(f"数值特征数: {len(num_cols)}")
    print(f"类别特征数: {len(cat_cols)}")
    print("数值特征:", num_cols)
    print("类别特征:", cat_cols)
    print(f"使用设备: {CFG.device}")

    # 1) 网格搜索
    best_row, results_df = grid_search(CFG, df, num_cols, cat_cols, y)

    # 2) 保存最佳参数
    best_cfg = deepcopy(CFG)
    best_cfg.top_k = None if pd.isna(best_row["top_k"]) else best_row["top_k"]
    best_cfg.pos_weight_scale = float(best_row["pos_weight_scale"])
    best_cfg.embed_dim = int(best_row["embed_dim"])

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({
            "top_k": best_cfg.top_k,
            "pos_weight_scale": best_cfg.pos_weight_scale,
            "embed_dim": best_cfg.embed_dim,
            "auc_mean": float(best_row["auc_mean"]),
            "auc_std": float(best_row["auc_std"]),
        }, f, ensure_ascii=False, indent=2)

    # 3) 用最佳参数重新完整跑一遍，并保存详细文件
    if best_cfg.rerun_best_after_search:
        print("\n开始用最佳参数重新完整跑 5 折并保存详细结果...\n")
        best_run_dir = output_dir / "best_run_detailed"

        summary, fold_summary_rows = run_single_experiment(
            cfg=best_cfg,
            df=df,
            num_cols=num_cols,
            cat_cols=cat_cols,
            y=y,
            exp_output_dir=best_run_dir,
            save_fold_files=True
        )

        pd.DataFrame(fold_summary_rows).to_csv(
            best_run_dir / "all_folds_metrics.csv",
            index=False,
            encoding="utf-8-sig"
        )

        with open(best_run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        with open(best_run_dir / "best_run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(best_cfg), f, ensure_ascii=False, indent=2)

        print("=" * 50)
        print("最佳参数完整 5 折结果")
        print("=" * 50)
        for key, stat in summary.items():
            if key == "threshold":
                continue
            print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    print(f"\n全部结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()