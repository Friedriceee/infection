"""
PGA-AMFormer: Prior-guided Arithmetic Attention AMFormer (完整修复版)
=====================================================================
修复点：
1. run_single_experiment 使用 uuid 隔离每次实验的临时 .pth 文件，
   避免不同 n_layers/embed_dim 的模型互相覆盖导致 load_state_dict 报错。
2. PriorGuidedArithmeticBlock.forward() 移除误混入的 Attention 代码。
3. PriorGuidedAttention.forward() 加入 B 归一化。
4. 分组学习率：先验参数 lr*5，base 参数正常 lr。
5. main() rerun 阶段同步所有搜索参数。
"""

import os
import json
import math
import random
import uuid
import warnings
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================
# 工具
# =============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return matthews_corrcoef(y_true, y_pred)


def find_best_threshold(y_true, y_prob, mode: str = "recall_f1"):
    best_t, best_score = 0.5, -1e9
    for t in np.arange(0.02, 0.981, 0.02):
        pred = (y_prob >= t).astype(int)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        mcc = safe_mcc(y_true, pred)
        if mode == "mcc":
            score = mcc
        elif mode == "f1":
            score = f1
        else:
            score = 0.70 * rec + 0.30 * f1
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score


# =============================================================
# 配置
# =============================================================
@dataclass
class Config:
    data_path: str = "/Users/wangqinyang.5/Desktop/Infection/original.xlsx"
    output_dir: str = "pga2_results"
    label_col: str = "outcome"

    prior_topk: int = 3
    prior_min_abs: float = 1e-6

    batch_size: int = 32
    epochs: int = 120
    lr: float = 3e-4
    min_lr: float = 1e-5
    patience: int = 25
    fixed_threshold: float = 0.5
    threshold_mode: str = "recall_f1"

    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
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

    use_prior_attn: bool = True
    use_prior_arith: bool = True
    learnable_B: bool = True

    lambda_raw_init: float = 0.0
    lambda_max: float = 2.0

    rho_raw_init: float = 0.0
    rho_max: float = 1.0

    prior_l2: float = 1e-4
    prior_mask: bool = True
    prior_type: str = "clinical"


CFG = Config()

GRID_SEARCH_SPACE = {
    "pos_weight_scale": [1.0],
    "embed_dim": [96],
    "lambda_raw_init": [-0.5, 0.0, 0.5],
    "prior_topk": [2, 3, 5],
    "n_layers": [2, 3],
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
def build_feature_dataframe(df, cfg):
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
            df[col] = np.log1p(df[col].clip(lower=0))

    for c in base_num_cols:
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
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown"})

    y = pd.to_numeric(df[cfg.label_col], errors="coerce").values.astype(int)
    return df, num_cols, cat_cols, y


class CategoryEncoder:
    def __init__(self):
        self.maps = {}

    def fit(self, df, cat_cols):
        for c in cat_cols:
            vals = sorted(set(df[c].astype(str).fillna("Unknown").tolist()))
            self.maps[c] = {v: i + 1 for i, v in enumerate(vals)}
        return self

    def transform(self, df, cat_cols):
        out = []
        for c in cat_cols:
            vals = df[c].astype(str).fillna("Unknown").tolist()
            out.append([self.maps[c].get(v, 0) for v in vals])
        return np.array(out, dtype=np.int64).T if out else np.zeros((len(df), 0), dtype=np.int64)

    def get_cardinalities(self, cat_cols):
        return [max(self.maps[c].values(), default=0) + 1 for c in cat_cols]


# =============================================================
# 临床先验矩阵 B
# =============================================================
def build_clinical_prior_matrix(
    num_cols: List[str], cat_cols: List[str],
    prior_type: str = "clinical", seed: int = 42
) -> np.ndarray:
    all_cols = num_cols + cat_cols
    F_dim = len(all_cols)
    B = np.zeros((F_dim, F_dim), dtype=np.float32)
    col_idx = {c: i for i, c in enumerate(all_cols)}

    def set_prior(f1, f2, value):
        if f1 in col_idx and f2 in col_idx:
            i, j = col_idx[f1], col_idx[f2]
            B[i, j] = value
            B[j, i] = value

    csf_features = ["C_G", "C_WBC", "C_RBC", "C_P", "C_N"]
    for i, f1 in enumerate(csf_features):
        for f2 in csf_features[i + 1:]:
            set_prior(f1, f2, 0.5)

    set_prior("C_G", "C_WBC", 0.8)
    set_prior("C_G", "C_N", 0.8)
    set_prior("C_WBC", "C_N", 0.8)
    set_prior("C_P", "C_G", 0.7)
    set_prior("C_P", "C_WBC", 0.6)

    set_prior("C_G", "B_G", 0.9)
    set_prior("C_WBC", "B_WBC", 0.6)
    set_prior("C_RBC", "B_RBC", 0.7)
    set_prior("C_N", "B_N", 0.5)

    set_prior("B_CRP", "B_PCT", 0.7)
    set_prior("B_WBC", "B_N", 0.5)
    set_prior("B_WBC", "B_Lym", 0.4)
    set_prior("B_N", "B_Lym", 0.3)

    set_prior("GCS", "C_G", 0.4)
    set_prior("GCS", "C_WBC", 0.3)
    set_prior("tem", "B_CRP", 0.4)
    set_prior("tem", "B_PCT", 0.4)
    set_prior("tem", "C_WBC", 0.3)

    set_prior("tube", "C_WBC", 0.5)
    set_prior("tube", "C_G", 0.4)
    set_prior("tube", "C_P", 0.3)
    set_prior("other_inf", "B_CRP", 0.4)
    set_prior("other_inf", "B_PCT", 0.4)
    set_prior("other_inf", "C_WBC", 0.3)

    set_prior("ratio_C_G_B_G", "C_G", 0.6)
    set_prior("ratio_C_G_B_G", "B_G", 0.6)
    set_prior("diff_C_N_B_N", "C_N", 0.5)
    set_prior("diff_C_N_B_N", "B_N", 0.5)
    set_prior("corrected_WBC", "C_WBC", 0.5)
    set_prior("corrected_WBC", "C_RBC", 0.4)
    set_prior("ratio_WBC_RBC_diff", "C_WBC", 0.4)
    set_prior("ratio_WBC_RBC_diff", "C_RBC", 0.4)
    set_prior("ratio_WBC_RBC_diff", "B_WBC", 0.4)
    set_prior("ratio_WBC_RBC_diff", "B_RBC", 0.4)

    set_prior("transparency", "C_WBC", 0.5)
    set_prior("transparency", "C_P", 0.3)
    set_prior("transparency", "C_RBC", 0.4)

    if prior_type == "clinical":
        return B

    rng = np.random.default_rng(seed)
    if prior_type == "random":
        mask = (B != 0)
        vals = B[mask]
        B_rand = np.zeros_like(B)
        upper = np.triu_indices(F_dim, k=1)
        n_edges = int(np.sum(mask[upper]))
        if n_edges > 0:
            chosen = rng.choice(len(upper[0]), size=n_edges, replace=False)
            values = (
                rng.choice(vals[vals > 0], size=n_edges, replace=True)
                if np.any(vals > 0)
                else rng.uniform(0.3, 0.9, size=n_edges)
            )
            for idx, val in zip(chosen, values):
                i, j = upper[0][idx], upper[1][idx]
                B_rand[i, j] = B_rand[j, i] = float(val)
        return B_rand

    if prior_type == "shuffled":
        B_shuf = np.zeros_like(B)
        upper = np.triu_indices(F_dim, k=1)
        vals = B[upper]
        rng.shuffle(vals)
        B_shuf[upper] = vals
        B_shuf = B_shuf + B_shuf.T
        return B_shuf.astype(np.float32)

    raise ValueError(f"Unknown prior_type: {prior_type}")


# =============================================================
# 模型模块
# =============================================================
class NumericFeatureEmbedding(nn.Module):
    def __init__(self, n_num, embed_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_num)])

    def forward(self, x):
        if x.size(1) == 0:
            return None
        return torch.stack(
            [self.embeddings[i](x[:, i:i + 1]) for i in range(len(self.embeddings))], dim=1
        )


class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, cardinalities, embed_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cardinalities])

    def forward(self, x):
        if x.size(1) == 0:
            return None
        return torch.stack(
            [self.embeddings[i](x[:, i]) for i in range(len(self.embeddings))], dim=1
        )


# ─────────────────────────────────────────────────
# [创新 1] Prior-guided Multi-head Attention
# ─────────────────────────────────────────────────
class PriorGuidedAttention(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        n_tokens,
        B_init: Optional[np.ndarray] = None,
        use_prior: bool = True,
        learnable_B: bool = True,
        lambda_raw_init: float = 0.0,
        lambda_max: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_prior = use_prior
        self.lambda_max = float(lambda_max)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_prior and B_init is not None:
            self.lambda_raw = nn.Parameter(torch.tensor(float(lambda_raw_init)))
            self.register_buffer(
                "B_init",
                torch.tensor(B_init, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            if learnable_B:
                self.delta_B = nn.Parameter(torch.zeros(1, 1, n_tokens, n_tokens))
            else:
                self.delta_B = None
        else:
            self.lambda_raw = None
            self.delta_B = None
            self.register_buffer(
                "B_init",
                torch.zeros(1, 1, n_tokens, n_tokens, dtype=torch.float32)
            )

    def current_lambda(self):
        if self.lambda_raw is None:
            return torch.tensor(0.0, device=self.w_q.weight.device)
        return self.lambda_max * torch.sigmoid(self.lambda_raw)

    def forward(self, x):
        batch_size, L, _ = x.size()
        q = self.w_q(x).view(batch_size, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if self.use_prior and self.lambda_raw is not None:
            lam = self.current_lambda()
            B_prior = self.B_init[:, :, :L, :L].to(device=x.device, dtype=x.dtype)
            if self.delta_B is not None:
                B_prior = B_prior + self.delta_B[:, :, :L, :L].to(device=x.device, dtype=x.dtype)
            # B 归一化：让先验和 QK scores 在同一量级
            B_norm = (B_prior - B_prior.mean()) / (B_prior.std() + 1e-6)
            scores = scores + lam * B_norm

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, L, -1)
        return self.w_o(context), attn_weights


# ─────────────────────────────────────────────────
# [创新 2] Prior-guided Arithmetic Block
# ─────────────────────────────────────────────────
class PriorGuidedArithmeticBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_tokens,
        B_init: Optional[np.ndarray] = None,
        use_prior: bool = True,
        learnable_B: bool = True,
        dropout: float = 0.1,
        prior_topk: int = 3,
        prior_min_abs: float = 1e-6,
        rho_raw_init: float = 0.0,
        rho_max: float = 1.0,
    ):
        super().__init__()
        self.use_prior = use_prior
        self.embed_dim = embed_dim
        self.prior_topk = prior_topk
        self.prior_min_abs = prior_min_abs
        self.rho_max = float(rho_max)

        self.proj = nn.Linear(embed_dim * 3, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        if use_prior and B_init is not None:
            self.register_buffer(
                "B_arith_init",
                torch.tensor(B_init, dtype=torch.float32)
            )
            if learnable_B:
                self.delta_B_arith = nn.Parameter(torch.zeros(n_tokens, n_tokens))
            else:
                self.delta_B_arith = None
            self.temperature = nn.Parameter(torch.tensor(1.0))
            self.rho_raw = nn.Parameter(torch.tensor(float(rho_raw_init)))
        else:
            self.register_buffer(
                "B_arith_init",
                torch.zeros(n_tokens, n_tokens, dtype=torch.float32)
            )
            self.delta_B_arith = None
            self.temperature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
            self.rho_raw = None

    def current_rho(self):
        if self.rho_raw is None:
            return torch.tensor(0.0, device=self.proj.weight.device)
        return self.rho_max * torch.sigmoid(self.rho_raw)

    def _get_prior_neighbors(self, h):
        L = h.size(1)
        B_mat = self.B_arith_init[:L, :L].to(device=h.device, dtype=h.dtype)
        if self.delta_B_arith is not None:
            B_mat = B_mat + self.delta_B_arith[:L, :L].to(device=h.device, dtype=h.dtype)

        eye = torch.eye(L, device=h.device, dtype=torch.bool)
        prior_mask = (B_mat.abs() > self.prior_min_abs) & (~eye)

        if self.prior_topk is None or self.prior_topk <= 0:
            topk_mask = prior_mask
        else:
            k = min(int(self.prior_topk), max(L - 1, 1))
            masked_scores = B_mat.masked_fill(~prior_mask, -1e9)
            top_vals, top_idx = torch.topk(masked_scores, k=k, dim=-1)
            valid_top = top_vals > -1e8
            topk_mask = torch.zeros_like(prior_mask)
            topk_mask.scatter_(dim=1, index=top_idx, src=valid_top)

        row_has_prior = topk_mask.any(dim=-1, keepdim=True)
        final_scores = B_mat.masked_fill(~topk_mask, -1e9)
        temp = torch.clamp(self.temperature, min=0.2, max=5.0)
        prior_weights = F.softmax(final_scores / temp, dim=-1)
        mean_weights = torch.ones_like(prior_weights) / L
        weights = torch.where(row_has_prior, prior_weights, mean_weights)
        return torch.matmul(weights.unsqueeze(0), h)

    def forward(self, h):
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)

        if self.use_prior and self.rho_raw is not None:
            h_prior = self._get_prior_neighbors(h)
            rho = self.current_rho()
            h_context = (1 - rho) * h_mean + rho * h_prior
        else:
            h_context = h_mean

        add_feat = h + h_context
        mul_feat = h * h_context
        sub_feat = h - h_context

        combined = torch.cat([add_feat, mul_feat, sub_feat], dim=-1)
        out = self.dropout(self.proj(combined))
        g = self.gate(h_context)
        return self.norm(h + g * out)


# ─────────────────────────────────────────────────
# Encoder Layer
# ─────────────────────────────────────────────────
class PGAEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        n_heads,
        n_tokens,
        ff_mult=4,
        dropout=0.1,
        B_init=None,
        use_prior_attn=True,
        use_prior_arith=True,
        learnable_B=True,
        lambda_raw_init=0.0,
        lambda_max=2.0,
        prior_mask=True,
        prior_topk: int = 3,
        prior_min_abs: float = 1e-6,
        rho_raw_init: float = 0.0,
        rho_max: float = 1.0,
    ):
        super().__init__()
        self.attn = PriorGuidedAttention(
            embed_dim, n_heads, n_tokens,
            B_init=B_init, use_prior=use_prior_attn,
            learnable_B=learnable_B,
            lambda_raw_init=lambda_raw_init,
            lambda_max=lambda_max,
            dropout=dropout,
        )
        self.arith = PriorGuidedArithmeticBlock(
            embed_dim, n_tokens,
            B_init=B_init,
            use_prior=use_prior_arith,
            learnable_B=learnable_B,
            dropout=dropout,
            prior_topk=prior_topk,
            prior_min_abs=prior_min_abs,
            rho_raw_init=rho_raw_init,
            rho_max=rho_max,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_mult, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, attn_w = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = self.arith(x)
        return x, attn_w


# ─────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────
class PGAAMFormer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        num_cols: List[str],
        cat_cols: List[str],
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        ff_mult: int = 4,
        use_prior_attn: bool = True,
        use_prior_arith: bool = True,
        learnable_B: bool = True,
        lambda_raw_init: float = 0.0,
        lambda_max: float = 2.0,
        rho_raw_init: float = 0.0,
        prior_mask: bool = True,
        prior_type: str = "clinical",
        seed: int = 42,
        prior_topk: int = 3,
        prior_min_abs: float = 1e-6,
        rho_max: float = 1.0,
    ):
        super().__init__()
        self.total_tokens = n_num_features + len(cat_cardinalities)

        self.num_embedding = NumericFeatureEmbedding(n_num_features, embed_dim)
        self.cat_embedding = CategoricalFeatureEmbedding(cat_cardinalities, embed_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max(1, self.total_tokens), embed_dim) * 0.01
        )

        B_init = build_clinical_prior_matrix(num_cols, cat_cols, prior_type=prior_type, seed=seed)
        self._B_init_np = B_init

        self.layers = nn.ModuleList([
            PGAEncoderLayer(
                embed_dim, n_heads, self.total_tokens,
                ff_mult=ff_mult, dropout=dropout,
                B_init=B_init,
                use_prior_attn=use_prior_attn,
                use_prior_arith=use_prior_arith,
                learnable_B=learnable_B,
                lambda_raw_init=lambda_raw_init,
                lambda_max=lambda_max,
                prior_topk=prior_topk,
                prior_min_abs=prior_min_abs,
                rho_raw_init=rho_raw_init,
                rho_max=rho_max,
            )
            for _ in range(n_layers)
        ])

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

    def forward(self, x_num, x_cat, return_attention=False):
        num_tok = self.num_embedding(x_num)
        cat_tok = self.cat_embedding(x_cat)

        if num_tok is not None and cat_tok is not None:
            h = torch.cat([num_tok, cat_tok], dim=1)
        elif num_tok is not None:
            h = num_tok
        else:
            h = cat_tok

        h = h + self.pos_encoding[:, :h.size(1), :]

        attn_maps = []
        for layer in self.layers:
            h, attn_w = layer(h)
            if return_attention:
                attn_maps.append(attn_w.detach().cpu().numpy())

        out_mean = h.mean(dim=1)
        out_max, _ = h.max(dim=1)
        feat = torch.cat([out_mean, out_max], dim=-1)
        logits = self.classifier(feat).squeeze(-1)

        if return_attention:
            return logits, attn_maps
        return logits

    def get_lambda(self) -> float:
        vals = []
        for layer in self.layers:
            if hasattr(layer.attn, "lambda_raw") and layer.attn.lambda_raw is not None:
                vals.append(layer.attn.current_lambda().detach().cpu().item())
        return float(np.mean(vals)) if vals else 0.0

    def get_rho(self) -> float:
        vals = []
        for layer in self.layers:
            if hasattr(layer.arith, "rho_raw") and layer.arith.rho_raw is not None:
                vals.append(layer.arith.current_rho().detach().cpu().item())
        return float(np.mean(vals)) if vals else 0.0

    def get_effective_B(self, layer_idx=0) -> np.ndarray:
        layer = self.layers[layer_idx]
        B = layer.attn.B_init.squeeze().detach().cpu().numpy()
        if layer.attn.delta_B is not None:
            B = B + layer.attn.delta_B.squeeze().detach().cpu().numpy()
        return B

    def get_arith_weights(self, layer_idx=0) -> np.ndarray:
        layer = self.layers[layer_idx]
        if not hasattr(layer, "arith"):
            return None
        arith = layer.arith
        if not hasattr(arith, "B_arith_init"):
            return None

        B = arith.B_arith_init.detach().cpu()
        if arith.delta_B_arith is not None:
            B = B + arith.delta_B_arith.detach().cpu()

        L = B.shape[0]
        eye = torch.eye(L, dtype=torch.bool)
        prior_min_abs = getattr(arith, "prior_min_abs", 1e-6)
        prior_mask = (B.abs() > prior_min_abs) & (~eye)
        prior_topk = getattr(arith, "prior_topk", 0)

        if prior_topk is None or prior_topk <= 0:
            topk_mask = prior_mask
        else:
            k = min(int(prior_topk), max(L - 1, 1))
            masked_scores = B.masked_fill(~prior_mask, -1e9)
            top_vals, top_idx = torch.topk(masked_scores, k=k, dim=-1)
            valid_top = top_vals > -1e8
            topk_mask = torch.zeros_like(prior_mask)
            topk_mask.scatter_(dim=1, index=top_idx, src=valid_top)

        row_has_prior = topk_mask.any(dim=-1, keepdim=True)
        final_scores = B.masked_fill(~topk_mask, -1e9)
        temp = float(torch.clamp(
            arith.temperature.detach().cpu() if hasattr(arith, "temperature") else torch.tensor(1.0),
            min=0.2, max=5.0
        ))
        prior_weights = torch.softmax(final_scores / temp, dim=-1)
        mean_weights = torch.ones_like(prior_weights) / L
        weights = torch.where(row_has_prior, prior_weights, mean_weights)
        return weights.numpy()


# =============================================================
# 训练器
# =============================================================
class ModelTrainer:
    def __init__(self, model, device="cpu", pos_weight=None, prior_l2: float = 0.0):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.prior_l2 = float(prior_l2)
        self.train_losses, self.val_losses, self.val_aucs = [], [], []

    def prior_regularization(self):
        if self.prior_l2 <= 0:
            return torch.tensor(0.0, device=self.device)
        reg = torch.tensor(0.0, device=self.device)
        for m in self.model.modules():
            if hasattr(m, "delta_B") and m.delta_B is not None:
                reg = reg + (m.delta_B ** 2).mean()
            if hasattr(m, "delta_B_arith") and m.delta_B_arith is not None:
                reg = reg + (m.delta_B_arith ** 2).mean()
        return reg

    def train_epoch(self, loader, optimizer, grad_clip=1.0):
        self.model.train()
        total_loss = 0.0
        for x_num, x_cat, y in tqdm(loader, desc="Training", leave=False):
            x_num = x_num.to(self.device)
            x_cat = x_cat.to(self.device)
            y = y.to(self.device)
            optimizer.zero_grad()
            logits = self.model(x_num, x_cat)
            loss = self.criterion(logits, y)
            if self.prior_l2 > 0:
                loss = loss + self.prior_l2 * self.prior_regularization()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_num, x_cat, y in loader:
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                y = y.to(self.device)
                logits = self.model(x_num, x_cat)
                loss = self.criterion(logits, y)
                if self.prior_l2 > 0:
                    loss = loss + self.prior_l2 * self.prior_regularization()
                total_loss += loss.item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        probs = np.array(all_probs)
        labels = np.array(all_labels)
        return total_loss / max(len(loader), 1), probs, labels, safe_auc(labels, probs)

    def train(self, train_loader, val_loader, save_path, epochs=100, lr=1e-3, patience=20,
              weight_decay=1e-2, min_lr=1e-5, grad_clip=1.0):
        # 分组学习率：先验参数用 5x lr
        prior_params, base_params = [], []
        for name, p in self.model.named_parameters():
            if any(k in name for k in ["lambda_raw", "rho_raw", "delta_B"]):
                prior_params.append(p)
            else:
                base_params.append(p)

        optimizer = torch.optim.AdamW([
            {"params": base_params, "lr": lr, "weight_decay": weight_decay},
            {"params": prior_params, "lr": lr * 5, "weight_decay": 0.0},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=6, min_lr=min_lr
        )
        best_auc, patience_counter = -1.0, 0
        torch.save(self.model.state_dict(), save_path)

        for _epoch in range(epochs):
            tr_loss = self.train_epoch(train_loader, optimizer, grad_clip)
            va_loss, _, _, va_auc = self.validate(val_loader)
            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)
            self.val_aucs.append(va_auc)
            scheduler.step(va_auc)
            if va_auc > best_auc + 1e-5:
                best_auc = va_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))

    def evaluate(self, loader, threshold=0.5):
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for x_num, x_cat, y in loader:
                x_num = x_num.to(self.device)
                x_cat = x_cat.to(self.device)
                logits = self.model(x_num, x_cat)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(y.numpy())
        probs = np.array(all_probs)
        labels = np.array(all_labels)
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(labels, preds)
        return {
            "predictions": probs.tolist(),
            "labels": labels.tolist(),
            "threshold": float(threshold),
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc": safe_auc(labels, probs),
            "mcc": safe_mcc(labels, preds),
            "confusion_matrix": cm.tolist(),
        }


# =============================================================
# 实验运行
# =============================================================
def run_single_experiment(cfg, df, num_cols, cat_cols, y,
                          exp_output_dir=None, save_fold_files=False):
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_results, fold_best_results, fold_summary_rows = [], [], []

    if exp_output_dir:
        exp_output_dir = Path(exp_output_dir)
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    # ✅ 关键修复：每次实验生成唯一 run_id，避免不同结构的模型文件互相污染
    tmp_run_id = str(uuid.uuid4())[:8]

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        x_tr = (scaler.fit_transform(df_train[num_cols].values.astype(np.float32))
                if num_cols else np.zeros((len(df_train), 0), dtype=np.float32))
        x_va = (scaler.transform(df_val[num_cols].values.astype(np.float32))
                if num_cols else np.zeros((len(df_val), 0), dtype=np.float32))

        cat_enc = CategoryEncoder().fit(df_train, cat_cols)
        cat_tr = cat_enc.transform(df_train, cat_cols)
        cat_va = cat_enc.transform(df_val, cat_cols)
        cat_cards = cat_enc.get_cardinalities(cat_cols)

        train_loader = DataLoader(
            CSFDataset(x_tr, cat_tr, y_train),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            CSFDataset(x_va, cat_va, y_val),
            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        pos = int(y_train.sum())
        neg = len(y_train) - pos
        pw = torch.tensor(
            (neg / max(pos, 1)) * cfg.pos_weight_scale, dtype=torch.float32
        ).to(cfg.device)

        model = PGAAMFormer(
            n_num_features=len(num_cols),
            cat_cardinalities=cat_cards,
            num_cols=num_cols,
            cat_cols=cat_cols,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            ff_mult=cfg.ff_mult,
            use_prior_attn=cfg.use_prior_attn,
            use_prior_arith=cfg.use_prior_arith,
            learnable_B=cfg.learnable_B,
            lambda_raw_init=cfg.lambda_raw_init,
            lambda_max=cfg.lambda_max,
            rho_raw_init=cfg.rho_raw_init,
            prior_mask=cfg.prior_mask,
            prior_type=cfg.prior_type,
            seed=cfg.seed + fold,
            prior_topk=cfg.prior_topk,
            prior_min_abs=cfg.prior_min_abs,
            rho_max=cfg.rho_max,
        )

        trainer = ModelTrainer(model, cfg.device, pw, prior_l2=cfg.prior_l2)

        # ✅ 使用 uuid 隔离临时文件，彻底避免跨实验文件复用
        if exp_output_dir:
            model_path = exp_output_dir / f"best_fold{fold}.pth"
        else:
            model_path = Path(f"temp_pga_{tmp_run_id}_fold{fold}.pth")

        trainer.train(
            train_loader, val_loader, str(model_path),
            epochs=cfg.epochs, lr=cfg.lr, patience=cfg.patience,
            weight_decay=cfg.weight_decay, min_lr=cfg.min_lr, grad_clip=cfg.grad_clip,
        )

        fixed_results = trainer.evaluate(val_loader, cfg.fixed_threshold)
        best_t, _ = find_best_threshold(
            np.array(fixed_results["labels"]),
            np.array(fixed_results["predictions"]),
            mode=cfg.threshold_mode
        )
        best_results = trainer.evaluate(val_loader, best_t)

        fold_results.append(fixed_results)
        fold_best_results.append(best_results)

        lambda_val = model.get_lambda()
        rho_val = model.get_rho()

        row = {
            "fold": fold,
            **{f"fixed_{k}": fixed_results[k]
               for k in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]},
            **{f"best_{k}": best_results[k]
               for k in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]},
            "best_threshold": best_t,
            "lambda": lambda_val,
            "rho": rho_val,
            "tn": fixed_results["confusion_matrix"][0][0],
            "fp": fixed_results["confusion_matrix"][0][1],
            "fn": fixed_results["confusion_matrix"][1][0],
            "tp": fixed_results["confusion_matrix"][1][1],
        }
        fold_summary_rows.append(row)

        if save_fold_files and exp_output_dir:
            with open(exp_output_dir / f"fold_{fold}_metrics.json", "w") as f:
                json.dump(row, f, indent=2)
            pd.DataFrame({
                "y_true": fixed_results["labels"],
                "y_prob": fixed_results["predictions"],
            }).to_csv(exp_output_dir / f"fold_{fold}_predictions.csv", index=False)
            pd.DataFrame({
                "epoch": np.arange(len(trainer.train_losses)),
                "train_loss": trainer.train_losses,
                "val_loss": trainer.val_losses,
                "val_auc": trainer.val_aucs,
            }).to_csv(exp_output_dir / f"fold_{fold}_learning_curve.csv", index=False)
            np.save(exp_output_dir / f"fold_{fold}_effective_B.npy", model.get_effective_B(0))
            arith_w = model.get_arith_weights(0)
            if arith_w is not None:
                np.save(exp_output_dir / f"fold_{fold}_arith_weights.npy", arith_w)

        # ✅ 训练完立即清理临时文件，避免堆积
        if not exp_output_dir and model_path.exists():
            model_path.unlink()

    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        best_vals = [r[key] for r in fold_best_results]
        summary[f"best_{key}"] = {"mean": float(np.mean(best_vals)), "std": float(np.std(best_vals))}
    summary["lambda_mean"] = float(np.mean([r["lambda"] for r in fold_summary_rows]))
    summary["rho_mean"] = float(np.mean([r["rho"] for r in fold_summary_rows]))
    summary["best_threshold_mean"] = float(np.mean([r["best_threshold"] for r in fold_summary_rows]))
    summary["fixed_threshold"] = cfg.fixed_threshold
    return summary, fold_summary_rows


# =============================================================
# 网格搜索
# =============================================================
def grid_search(cfg, df, num_cols, cat_cols, y):
    search_dir = Path(cfg.output_dir)
    search_dir.mkdir(parents=True, exist_ok=True)
    keys = list(GRID_SEARCH_SPACE.keys())
    all_combos = list(product(*GRID_SEARCH_SPACE.values()))
    print(f"搜索 {len(all_combos)} 组参数\n")
    results = []

    for i, combo in enumerate(all_combos, 1):
        params = dict(zip(keys, combo))
        exp_cfg = deepcopy(cfg)
        for k, v in params.items():
            setattr(exp_cfg, k, v)
        print(f"[{i}/{len(all_combos)}] {params}")
        summary, _ = run_single_experiment(exp_cfg, df, num_cols, cat_cols, y)
        row = {
            "exp": i,
            **params,
            "auc_mean": summary["auc"]["mean"],
            "auc_std": summary["auc"]["std"],
            "f1_mean": summary["f1"]["mean"],
            "mcc_mean": summary["mcc"]["mean"],
            "best_f1_mean": summary["best_f1"]["mean"],
            "best_mcc_mean": summary["best_mcc"]["mean"],
            "lambda": summary["lambda_mean"],
            "rho": summary["rho_mean"],
        }
        results.append(row)
        print(
            f"  AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f} | "
            f"F1={row['f1_mean']:.4f} | MCC={row['mcc_mean']:.4f} | "
            f"λ={row['lambda']:.4f} | ρ={row['rho']:.4f}"
        )
        pd.DataFrame(results).sort_values("auc_mean", ascending=False).to_csv(
            search_dir / "grid_search.csv", index=False
        )

    best = pd.DataFrame(results).sort_values("auc_mean", ascending=False).iloc[0].to_dict()
    print(f"\n最佳: {best}")
    return best


# =============================================================
# 主函数
# =============================================================
def main():
    set_seed(CFG.seed)
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_excel(CFG.data_path)
    df, num_cols, cat_cols, y = build_feature_dataframe(raw_df, CFG)

    print(f"样本: {len(df)}, 阳性: {int(y.sum())}")
    print(f"数值特征: {len(num_cols)}, 类别: {len(cat_cols)}, 总 tokens: {len(num_cols) + len(cat_cols)}")
    print(
        f"PGA 开关: attn={CFG.use_prior_attn}, arith={CFG.use_prior_arith}, "
        f"learnable_B={CFG.learnable_B}, prior_type={CFG.prior_type}, prior_mask={CFG.prior_mask}"
    )

    best = grid_search(CFG, df, num_cols, cat_cols, y)

    # ✅ 同步所有搜索过的参数
    best_cfg = deepcopy(CFG)
    best_cfg.pos_weight_scale = float(best["pos_weight_scale"])
    best_cfg.embed_dim = int(best["embed_dim"])
    best_cfg.lambda_raw_init = float(best["lambda_raw_init"])
    best_cfg.prior_topk = int(best["prior_topk"])
    best_cfg.n_layers = int(best["n_layers"])

    # ✅ 先保存搜索结果（含所有参数）
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(
            {k: best[k] for k in [
                "pos_weight_scale", "embed_dim",
                "lambda_raw_init", "prior_topk", "n_layers",
                "auc_mean", "auc_std", "f1_mean", "mcc_mean", "lambda", "rho"
            ]},
            f, indent=2
        )

    if best_cfg.rerun_best_after_search:
        print("\n用最佳参数跑 5 折...\n")
        best_dir = output_dir / "best_run"
        summary, rows = run_single_experiment(
            best_cfg, df, num_cols, cat_cols, y,
            exp_output_dir=best_dir, save_fold_files=True
        )
        pd.DataFrame(rows).to_csv(best_dir / "all_folds.csv", index=False)
        with open(best_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("=" * 50)
        print("PGA-AMFormer 最终结果")
        print("=" * 50)
        for k, v in summary.items():
            if isinstance(v, dict):
                print(f"{k:>14}: {v['mean']:.4f} ± {v['std']:.4f}")
            elif k == "lambda_mean":
                print(f"{'lambda':>14}: {v:.4f}")
            elif k == "rho_mean":
                print(f"{'rho':>14}: {v:.4f}")
        print(f"\n结果保存至: {output_dir.resolve()}")


if __name__ == "__main__":
    main()