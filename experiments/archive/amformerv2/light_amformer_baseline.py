"""
LightAMFormer Baseline for small-sample clinical tabular data

用途：
1. 作为 DA-CTFormer 的轻量 AMFormer baseline；
2. 去掉 DAE / PDA / TTE，只保留稳定的：
   - 数值/类别 token embedding
   - Top-K sparse multi-head attention
   - 轻量 gated arithmetic block
   - FFN + residual + LayerNorm
3. 默认使用 StandardScaler，适合 900 条静态表格小样本；
4. 保留 x_dyn / dyn_mask 的数据接口，但 baseline 不使用动态分支，
   便于和 DA-CTFormer 使用同一套训练/评估流程。
"""

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


def find_best_threshold(y_true, y_prob, mode="recall_f1", fn_cost=3.0, fp_cost=1.0):
    """
    医疗筛查场景建议不要固定 0.5。
    mode:
    - recall_f1: 优先召回，兼顾 F1
    - cost: 最小化 fn_cost*FN + fp_cost*FP
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t = 0.5
    best_score = -1e18

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if mode == "cost":
            score = -(fn_cost * fn + fp_cost * fp)
        else:
            score = 0.70 * rec + 0.30 * f1
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


@dataclass
class Config:
    data_path: str = "/Users/wangqinyang.5/Desktop/Infection/original.xlsx"
    output_dir: str = "light_amformer_baseline_results"
    label_col: str = "outcome"

    # 保留动态接口，但 LightAMFormer baseline 默认不使用动态数据
    use_dynamic: bool = False
    dynamic_npy_path: str = ""

    batch_size: int = 32
    epochs: int = 120
    lr: float = 3e-4
    min_lr: float = 1e-5
    patience: int = 25

    fixed_threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_mode: str = "recall_f1"  # recall_f1 or cost
    fn_cost: float = 3.0
    fp_cost: float = 1.0

    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    top_k: int | None = 8
    dropout: float = 0.20
    ff_mult: int = 4

    seed: int = 42
    n_splits: int = 5
    pos_weight_scale: float = 1.0
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    rerun_best_after_search: bool = True


CFG = Config()

GRID_SEARCH_SPACE = {
    "top_k": [None, 4, 8, 12],
    "pos_weight_scale": [0.9, 1.0, 1.1],
    "embed_dim": [64, 96],
}


class CSFDataset(Dataset):
    def __init__(self, x_num, x_cat, x_dyn=None, dyn_mask=None, labels=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        if x_dyn is None:
            x_dyn = np.zeros((len(x_num), 0, 0), dtype=np.float32)
        if dyn_mask is None:
            dyn_mask = np.zeros(len(x_num), dtype=np.float32)
        self.x_dyn = torch.tensor(x_dyn, dtype=torch.float32)
        self.dyn_mask = torch.tensor(dyn_mask, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.x_num[idx], self.x_cat[idx], self.x_dyn[idx], self.dyn_mask[idx]
        return self.x_num[idx], self.x_cat[idx], self.x_dyn[idx], self.dyn_mask[idx], self.labels[idx]


def build_feature_dataframe(df: pd.DataFrame, cfg: Config):
    df = df.copy()
    cat_cols = ["sex", "tube", "site", "other_inf", "transparency"]
    base_num_cols = [
        "age", "C_G", "C_WBC", "C_RBC", "C_P", "C_N",
        "GCS", "tem", "B_G", "B_CRP", "B_WBC", "B_N",
        "B_Lym", "B_PCT", "B_AC", "B_RBC",
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

    # 作为 baseline，可保留你原本少量临床可解释的手工交互特征
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
        df["ratio_WBC_RBC_diff"] = df["B_WBC"] / (df["B_RBC"] + eps) - df["C_WBC"] / (df["C_RBC"] + eps)
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


def load_dynamic_array(cfg: Config, n_samples: int):
    """LightAMFormer baseline 默认不使用动态分支，仅保留接口。"""
    if not cfg.use_dynamic or not cfg.dynamic_npy_path:
        return np.zeros((n_samples, 0, 0), dtype=np.float32), np.zeros(n_samples, dtype=np.float32), 0
    x_seq = np.load(cfg.dynamic_npy_path).astype(np.float32)
    if x_seq.ndim != 3:
        raise ValueError("动态数据必须是 [N, T, F] 的三维数组，例如 [300, 4, 6]")
    n_dyn, t_steps, f_dyn = x_seq.shape
    full = np.zeros((n_samples, t_steps, f_dyn), dtype=np.float32)
    mask = np.zeros(n_samples, dtype=np.float32)
    n_copy = min(n_samples, n_dyn)
    full[:n_copy] = x_seq[:n_copy]
    valid = np.isfinite(full).all(axis=(1, 2)) & (np.abs(full).sum(axis=(1, 2)) > 0)
    mask[valid] = 1.0
    full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
    return full, mask, f_dyn


class CategoryEncoder:
    def __init__(self):
        self.maps = {}

    def fit(self, df: pd.DataFrame, cat_cols):
        self.maps = {}
        for c in cat_cols:
            values = df[c].astype(str).fillna("Unknown").tolist()
            uniq = sorted(list(set(values)))
            self.maps[c] = {v: i + 1 for i, v in enumerate(uniq)}
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


class NumericFeatureEmbedding(nn.Module):
    def __init__(self, n_num_features: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_num_features)])

    def forward(self, x_num: torch.Tensor):
        if x_num.size(1) == 0:
            return None
        tokens = [self.embeddings[i](x_num[:, i:i + 1]) for i in range(len(self.embeddings))]
        return torch.stack(tokens, dim=1)


class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, cat_cardinalities, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities])

    def forward(self, x_cat: torch.Tensor):
        if x_cat.size(1) == 0:
            return None
        return torch.stack([self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))], dim=1)


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
        if self.top_k is not None and self.top_k > 0 and self.top_k < L:
            effective_k = min(self.top_k, L)
            topk_vals, _ = torch.topk(scores, effective_k, dim=-1)
            threshold = topk_vals[..., -1:].expand_as(scores)
            scores = scores.masked_fill(scores < threshold, torch.finfo(scores.dtype).min)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        return self.w_o(context), attn_weights


class GatedArithmeticBlock(nn.Module):
    """小样本友好的轻量算术块：add / multiply / subtract + gate。"""
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 3, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
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


class LightAMFormerLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, top_k=None, dropout: float = 0.2, ff_mult: int = 4):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, n_heads, top_k=top_k, dropout=dropout)
        self.arith = GatedArithmeticBlock(embed_dim, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, h):
        attn_out, attn_w = self.attn(h)
        h = self.norm1(h + attn_out)
        h = self.arith(h)
        h = self.norm2(h + self.ffn(h))
        return h, attn_w


class LightAMFormer(nn.Module):
    def __init__(self, n_num_features: int, cat_cardinalities, embed_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, top_k: int | None = 8,
                 dropout: float = 0.2, ff_mult: int = 4):
        super().__init__()
        self.total_tokens = n_num_features + len(cat_cardinalities)
        self.num_embedding = NumericFeatureEmbedding(n_num_features, embed_dim)
        self.cat_embedding = CategoricalFeatureEmbedding(cat_cardinalities, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max(1, self.total_tokens), embed_dim) * 0.01)
        self.layers = nn.ModuleList([
            LightAMFormerLayer(embed_dim, n_heads, top_k=top_k, dropout=dropout, ff_mult=ff_mult)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
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

    def forward(self, x_num, x_cat, x_dyn=None, dyn_mask=None, return_attention: bool = False):
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
        for layer in self.layers:
            h, attn_w = layer(h)
            if return_attention:
                attention_maps.append(attn_w.detach().cpu().numpy())
        h = self.final_norm(h)
        out_feat = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        logits = self.classifier(out_feat).squeeze(-1)
        if return_attention:
            return logits, attention_maps
        return logits


ImprovedAMFormerV2 = LightAMFormer


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
        for batch_num, batch_cat, batch_dyn, batch_dyn_mask, batch_labels in tqdm(train_loader, desc="Training", leave=False):
            batch_num = batch_num.to(self.device)
            batch_cat = batch_cat.to(self.device)
            batch_dyn = batch_dyn.to(self.device)
            batch_dyn_mask = batch_dyn_mask.to(self.device)
            batch_labels = batch_labels.to(self.device)
            optimizer.zero_grad()
            logits = self.model(batch_num, batch_cat, batch_dyn, batch_dyn_mask)
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
            for batch_num, batch_cat, batch_dyn, batch_dyn_mask, batch_labels in val_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                batch_dyn = batch_dyn.to(self.device)
                batch_dyn_mask = batch_dyn_mask.to(self.device)
                batch_labels = batch_labels.to(self.device)
                logits = self.model(batch_num, batch_cat, batch_dyn, batch_dyn_mask)
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        return total_loss / max(len(val_loader), 1), all_probs, all_labels, safe_auc(all_labels, all_probs)

    def train(self, train_loader, val_loader, save_path, epochs=100, lr=1e-3, patience=20,
              weight_decay=1e-2, min_lr=1e-5, grad_clip=1.0):
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6, min_lr=min_lr)
        best_val_auc = -1.0
        patience_counter = 0
        torch.save(self.model.state_dict(), save_path)
        for _ in range(epochs):
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

    def predict_probs(self, val_loader):
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch_num, batch_cat, batch_dyn, batch_dyn_mask, batch_labels in val_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                batch_dyn = batch_dyn.to(self.device)
                batch_dyn_mask = batch_dyn_mask.to(self.device)
                logits = self.model(batch_num, batch_cat, batch_dyn, batch_dyn_mask)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        return np.array(all_probs), np.array(all_labels)

    def evaluate(self, val_loader, threshold: float = 0.5) -> dict:
        probs, labels = self.predict_probs(val_loader)
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        return {
            "predictions": probs.tolist(),
            "labels": labels.tolist(),
            "threshold": float(threshold),
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc": safe_auc(labels, probs),
            "confusion_matrix": cm.tolist(),
        }


def summarize_results(fold_results, threshold_key="threshold"):
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["threshold"] = {
        "mean": float(np.mean([r[threshold_key] for r in fold_results])),
        "std": float(np.std([r[threshold_key] for r in fold_results])),
    }
    return summary


def run_single_experiment(cfg: Config, df, num_cols, cat_cols, y, x_dyn_all=None, dyn_mask_all=None,
                          exp_output_dir=None, save_fold_files=False):
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_results_fixed = []
    fold_results_best = []
    fold_summary_rows = []
    if exp_output_dir is not None:
        exp_output_dir = Path(exp_output_dir)
        exp_output_dir.mkdir(parents=True, exist_ok=True)
    if x_dyn_all is None:
        x_dyn_all = np.zeros((len(df), 0, 0), dtype=np.float32)
    if dyn_mask_all is None:
        dyn_mask_all = np.zeros(len(df), dtype=np.float32)

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

        x_train_dyn = x_dyn_all[train_idx]
        x_val_dyn = x_dyn_all[val_idx]
        train_dyn_mask = dyn_mask_all[train_idx]
        val_dyn_mask = dyn_mask_all[val_idx]

        train_loader = DataLoader(CSFDataset(x_train_num, x_train_cat, x_train_dyn, train_dyn_mask, y_train), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        val_loader = DataLoader(CSFDataset(x_val_num, x_val_cat, x_val_dyn, val_dyn_mask, y_val), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        pw_value = (neg_count / max(pos_count, 1)) * cfg.pos_weight_scale
        pos_weight = torch.tensor(pw_value, dtype=torch.float32).to(cfg.device)

        model = LightAMFormer(
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
        model_path = (Path(exp_output_dir) / f"best_model_fold{fold}.pth") if exp_output_dir is not None else Path(f"temp_light_amformer_fold{fold}.pth")
        trainer.train(train_loader, val_loader, str(model_path), epochs=cfg.epochs, lr=cfg.lr, patience=cfg.patience, weight_decay=cfg.weight_decay, min_lr=cfg.min_lr, grad_clip=cfg.grad_clip)

        fixed_results = trainer.evaluate(val_loader, threshold=cfg.fixed_threshold)
        fold_results_fixed.append(fixed_results)
        probs = np.array(fixed_results["predictions"])
        labels = np.array(fixed_results["labels"])
        best_t = find_best_threshold(labels, probs, mode=cfg.threshold_mode, fn_cost=cfg.fn_cost, fp_cost=cfg.fp_cost) if cfg.optimize_threshold else cfg.fixed_threshold
        best_results = trainer.evaluate(val_loader, threshold=best_t)
        fold_results_best.append(best_results)
        cm = best_results["confusion_matrix"]
        fold_summary = {
            "fold": fold,
            "auc": best_results["auc"],
            "fixed_threshold": cfg.fixed_threshold,
            "best_threshold": best_t,
            "accuracy_fixed": fixed_results["accuracy"],
            "precision_fixed": fixed_results["precision"],
            "recall_fixed": fixed_results["recall"],
            "f1_fixed": fixed_results["f1"],
            "accuracy": best_results["accuracy"],
            "precision": best_results["precision"],
            "recall": best_results["recall"],
            "f1": best_results["f1"],
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
        }
        fold_summary_rows.append(fold_summary)
        if save_fold_files and exp_output_dir is not None:
            with open(exp_output_dir / f"fold_{fold}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(fold_summary, f, ensure_ascii=False, indent=2)
            pd.DataFrame({"y_true": best_results["labels"], "y_prob": best_results["predictions"], "fixed_threshold": cfg.fixed_threshold, "best_threshold": best_t}).to_csv(exp_output_dir / f"fold_{fold}_predictions.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"epoch": np.arange(len(trainer.train_losses)), "train_loss": trainer.train_losses, "val_loss": trainer.val_losses, "val_auc": trainer.val_aucs}).to_csv(exp_output_dir / f"fold_{fold}_learning_curve.csv", index=False, encoding="utf-8-sig")
            with open(exp_output_dir / f"fold_{fold}_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open(exp_output_dir / f"fold_{fold}_cat_encoder.pkl", "wb") as f:
                pickle.dump(cat_encoder, f)

    fixed_summary = summarize_results(fold_results_fixed, threshold_key="threshold")
    best_summary = summarize_results(fold_results_best, threshold_key="threshold")
    return best_summary, fixed_summary, fold_summary_rows


def grid_search(cfg: Config, df, num_cols, cat_cols, y, x_dyn_all, dyn_mask_all):
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
        print("=" * 80)
        print(f"[{i}/{len(all_combinations)}] LightAMFormer 参数组合: {params}")
        print("=" * 80)
        best_summary, fixed_summary, _ = run_single_experiment(exp_cfg, df, num_cols, cat_cols, y, x_dyn_all, dyn_mask_all)
        row = {
            "exp_name": exp_name,
            "top_k": params["top_k"],
            "pos_weight_scale": params["pos_weight_scale"],
            "embed_dim": params["embed_dim"],
            "auc_mean": best_summary["auc"]["mean"],
            "auc_std": best_summary["auc"]["std"],
            "f1_mean": best_summary["f1"]["mean"],
            "recall_mean": best_summary["recall"]["mean"],
            "precision_mean": best_summary["precision"]["mean"],
            "accuracy_mean": best_summary["accuracy"]["mean"],
            "threshold_mean": best_summary["threshold"]["mean"],
            "f1_fixed_mean": fixed_summary["f1"]["mean"],
            "recall_fixed_mean": fixed_summary["recall"]["mean"],
            "precision_fixed_mean": fixed_summary["precision"]["mean"],
            "accuracy_fixed_mean": fixed_summary["accuracy"]["mean"],
        }
        search_results.append(row)
        print(f"完成: AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f} | BestT={row['threshold_mean']:.2f} | Recall={row['recall_mean']:.4f} | F1={row['f1_mean']:.4f} | Fixed0.5 Recall={row['recall_fixed_mean']:.4f}\n")
        pd.DataFrame(search_results).sort_values(by=["auc_mean", "recall_mean", "f1_mean"], ascending=[False, False, False]).to_csv(search_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")

    results_df = pd.DataFrame(search_results).sort_values(by=["auc_mean", "recall_mean", "f1_mean"], ascending=[False, False, False]).reset_index(drop=True)
    results_df.to_csv(search_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    with open(search_dir / "grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2)
    best_row = results_df.iloc[0].to_dict()
    print("\n" + "#" * 80)
    print("LightAMFormer 网格搜索完成，最佳参数：")
    print(best_row)
    print("#" * 80 + "\n")
    return best_row, results_df


def main():
    set_seed(CFG.seed)
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("加载静态数据...")
    raw_df = pd.read_excel(CFG.data_path)
    df, num_cols, cat_cols, y = build_feature_dataframe(raw_df, CFG)
    x_dyn_all, dyn_mask_all, n_dyn_features = load_dynamic_array(CFG, len(df))
    print(f"样本数: {len(df)}")
    print(f"阳性样本数: {int(y.sum())}")
    print(f"数值特征数: {len(num_cols)}")
    print(f"类别特征数: {len(cat_cols)}")
    print(f"动态特征数: {n_dyn_features} | baseline 默认不使用动态分支")
    print(f"使用设备: {CFG.device}")
    print("数值特征:", num_cols)
    print("类别特征:", cat_cols)
    best_row, _ = grid_search(CFG, df, num_cols, cat_cols, y, x_dyn_all, dyn_mask_all)
    best_cfg = deepcopy(CFG)
    best_cfg.top_k = None if pd.isna(best_row["top_k"]) else int(best_row["top_k"])
    best_cfg.pos_weight_scale = float(best_row["pos_weight_scale"])
    best_cfg.embed_dim = int(best_row["embed_dim"])
    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({"model": "LightAMFormer", "top_k": best_cfg.top_k, "pos_weight_scale": best_cfg.pos_weight_scale, "embed_dim": best_cfg.embed_dim, "auc_mean": float(best_row["auc_mean"]), "auc_std": float(best_row["auc_std"]), "threshold_mean": float(best_row["threshold_mean"])}, f, ensure_ascii=False, indent=2)

    if best_cfg.rerun_best_after_search:
        print("\n开始用最佳参数重新完整跑 5 折并保存详细结果...\n")
        best_run_dir = output_dir / "best_run_detailed"
        best_summary, fixed_summary, fold_summary_rows = run_single_experiment(best_cfg, df, num_cols, cat_cols, y, x_dyn_all, dyn_mask_all, exp_output_dir=best_run_dir, save_fold_files=True)
        pd.DataFrame(fold_summary_rows).to_csv(best_run_dir / "all_folds_metrics.csv", index=False, encoding="utf-8-sig")
        with open(best_run_dir / "summary_metrics_best_threshold.json", "w", encoding="utf-8") as f:
            json.dump(best_summary, f, ensure_ascii=False, indent=2)
        with open(best_run_dir / "summary_metrics_fixed_0_5.json", "w", encoding="utf-8") as f:
            json.dump(fixed_summary, f, ensure_ascii=False, indent=2)
        with open(best_run_dir / "best_run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(best_cfg), f, ensure_ascii=False, indent=2)
        print("=" * 60)
        print("LightAMFormer 最佳阈值 5 折结果")
        print("=" * 60)
        for key, stat in best_summary.items():
            print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")
        print("\n" + "=" * 60)
        print("LightAMFormer 固定 0.5 阈值 5 折结果")
        print("=" * 60)
        for key, stat in fixed_summary.items():
            print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")
    print(f"\n全部结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
