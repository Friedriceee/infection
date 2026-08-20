"""
RPI-Tab v2: Reliability-Gated Residual Prior Injection for Tabular Deep Learning
================================================================================

This script implements a self-contained RPI-Tab v2 framework for small-sample
binary tabular prediction.

Core idea:
    z_base = f_theta(x)
    delta_z_k = g_k(p_k(x)) for weak priors k in {group, prototype, arithmetic}
    z_final = z_base + lambda(x) * sum_k r_k(x) * delta_z_k

where:
    - r_k(x) is an instance-wise reliability distribution over priors;
    - lambda(x) is an instance-wise total residual strength;
    - priors are residual corrections, not hard constraints.

Experiments included:
    E0_NoPrior
    E1_ConcatAllPriors
    E3_RPIv1_Group
    E6_RPIv1_AllPriors
    E7_RPIv2_ReliabilityGated_All
    E8_RPIv2_ReliabilityGated_GroupArith

Usage:
    python rpi_tab_v2_framework.py --dataset cns --data_path original.xlsx --target outcome
    python rpi_tab_v2_framework.py --dataset csv --data_path heart.csv --target target
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
)

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")


# =============================================================================
# Utils
# =============================================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))


def safe_mcc(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return float(matthews_corrcoef(y_true, y_pred))


def find_best_threshold(y_true, y_prob, mode="recall_f1", fn_cost=3.0, fp_cost=1.0) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    best_t, best_score = 0.5, -1e18
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        if mode == "cost":
            score = -(fn_cost * fn + fp_cost * fp)
        elif mode == "youden":
            spec = tn / max(tn + fp, 1)
            score = rec + spec - 1
        else:
            score = 0.70 * rec + 0.30 * f1
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def compute_metrics(y_true, y_prob, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": safe_auc(y_true, y_prob),
        "mcc": safe_mcc(y_true, pred),
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }


def summarize(rows: List[Dict[str, Any]], prefix="") -> Dict[str, float]:
    out = {}
    for k in ["accuracy", "precision", "recall", "f1", "auc", "mcc", "threshold"]:
        vals = [float(r[k]) for r in rows]
        out[f"{prefix}{k}_mean"] = float(np.mean(vals))
        out[f"{prefix}{k}_std"] = float(np.std(vals))
    return out


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    dataset: str = "cns"
    data_path: str = "original.xlsx"
    target: str = "outcome"
    output_dir: str = "rpi_tab_v2_results"

    seed: int = 42
    n_splits: int = 5
    batch_size: int = 32
    epochs: int = 120
    patience: int = 25
    lr: float = 3e-4
    min_lr: float = 1e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0

    backbone: str = "light_amformer"  # currently light_amformer / mlp
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    top_k: Optional[int] = None
    dropout: float = 0.20
    ff_mult: int = 4
    pos_weight_scale: float = 1.1

    fixed_threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_mode: str = "recall_f1"
    fn_cost: float = 3.0
    fp_cost: float = 1.0

    n_prototypes: int = 4
    proto_temperature: float = 1.0
    sample_fraction: float = 1.0

    injection: str = "none"  # none / concat / residual / residual_gated
    prior_types: Tuple[str, ...] = ()
    residual_init: float = -2.0
    residual_l2: float = 0.01
    residual_lr_mult: float = 2.0
    scale_lr_mult: float = 5.0

    # RPI-v2
    prior_strength_init: float = -2.0
    prior_gate_entropy_lambda: float = 0.0
    prior_utility_lambda: float = 0.0
    prior_utility_temperature: float = 1.0
    gate_arithmetic: bool = True
    gate_l1_lambda: float = 0.0


# =============================================================================
# Feature recipes
# =============================================================================

class CategoryEncoder:
    def __init__(self):
        self.maps: Dict[str, Dict[str, int]] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str]):
        self.maps = {}
        for c in cat_cols:
            vals = df[c].astype(str).fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown"}).tolist()
            uniq = sorted(set(vals))
            self.maps[c] = {v: i + 1 for i, v in enumerate(uniq)}
        return self

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
        if not cat_cols:
            return np.zeros((len(df), 0), dtype=np.int64)
        arr = []
        for c in cat_cols:
            mapper = self.maps[c]
            vals = df[c].astype(str).fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown"}).tolist()
            arr.append([mapper.get(v, 0) for v in vals])
        return np.array(arr, dtype=np.int64).T

    def cardinalities(self, cat_cols: List[str]) -> List[int]:
        return [max(self.maps[c].values(), default=0) + 1 for c in cat_cols]


def load_raw_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p)


def add_numeric_col(df: pd.DataFrame, name: str, value) -> str:
    df[name] = value
    df[name] = pd.to_numeric(df[name], errors="coerce")
    df[name] = df[name].replace([np.inf, -np.inf], np.nan)
    med = df[name].median()
    if pd.isna(med):
        med = 0.0
    df[name] = df[name].fillna(med)
    return name


def cns_feature_recipe(df: pd.DataFrame, target: str):
    df = df.copy()
    cat_cols = [c for c in ["sex", "tube", "site", "other_inf", "transparency"] if c in df.columns]
    base_num_cols = [c for c in [
        "age", "C_G", "C_WBC", "C_RBC", "C_P", "C_N",
        "GCS", "tem", "B_G", "B_CRP", "B_WBC", "B_N",
        "B_Lym", "B_PCT", "B_AC", "B_RBC",
    ] if c in df.columns]

    for c in base_num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    log_cols = ["C_WBC", "C_RBC", "C_P", "B_CRP", "B_WBC", "B_PCT", "B_AC", "B_RBC"]
    for c in log_cols:
        if c in df.columns:
            df[c] = np.log1p(df[c].clip(lower=0))
    for c in base_num_cols:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        med = df[c].median()
        df[c] = df[c].fillna(0.0 if pd.isna(med) else med)

    eps = 1e-6
    base_interactions = []
    if "C_G" in df.columns and "B_G" in df.columns:
        base_interactions.append(add_numeric_col(df, "ratio_C_G_B_G", df["C_G"] / (df["B_G"] + eps)))
    if "C_N" in df.columns and "B_N" in df.columns:
        base_interactions.append(add_numeric_col(df, "diff_C_N_B_N", df["C_N"] - df["B_N"]))
    if all(c in df.columns for c in ["C_WBC", "B_WBC", "C_RBC", "B_RBC"]):
        base_interactions.append(add_numeric_col(df, "corrected_WBC", df["C_WBC"] - df["B_WBC"] * df["C_RBC"] / (df["B_RBC"] + eps)))
    if all(c in df.columns for c in ["B_WBC", "B_RBC", "C_WBC", "C_RBC"]):
        base_interactions.append(add_numeric_col(df, "ratio_WBC_RBC_diff", df["B_WBC"] / (df["B_RBC"] + eps) - df["C_WBC"] / (df["C_RBC"] + eps)))

    cai_cols = []
    def add_cai(name, val):
        if name not in df.columns:
            cai_cols.append(add_numeric_col(df, name, val))

    if "C_G" in df.columns and "B_G" in df.columns:
        add_cai("cai_ratio_C_G_B_G", df["C_G"] / (df["B_G"] + eps))
        add_cai("cai_diff_C_G_B_G", df["C_G"] - df["B_G"])
    if "C_WBC" in df.columns and "B_WBC" in df.columns:
        add_cai("cai_ratio_C_WBC_B_WBC", df["C_WBC"] / (df["B_WBC"] + eps))
        add_cai("cai_diff_C_WBC_B_WBC", df["C_WBC"] - df["B_WBC"])
    if "C_N" in df.columns and "B_N" in df.columns:
        add_cai("cai_ratio_C_N_B_N", df["C_N"] / (df["B_N"] + eps))
        add_cai("cai_diff_C_N_B_N", df["C_N"] - df["B_N"])
    if "C_RBC" in df.columns and "B_RBC" in df.columns:
        add_cai("cai_ratio_C_RBC_B_RBC", df["C_RBC"] / (df["B_RBC"] + eps))
    if "C_P" in df.columns and "C_G" in df.columns:
        add_cai("cai_ratio_C_P_C_G", df["C_P"] / (df["C_G"] + eps))
    if "C_N" in df.columns and "C_G" in df.columns:
        add_cai("cai_ratio_C_N_C_G", df["C_N"] / (df["C_G"] + eps))
    if "C_WBC" in df.columns and "C_G" in df.columns:
        add_cai("cai_ratio_C_WBC_C_G", df["C_WBC"] / (df["C_G"] + eps))
    if "C_WBC" in df.columns and "C_RBC" in df.columns:
        add_cai("cai_ratio_C_WBC_C_RBC", df["C_WBC"] / (df["C_RBC"] + eps))
    if "corrected_WBC" in df.columns:
        add_cai("cai_corrected_WBC_copy", df["corrected_WBC"])
    if "C_WBC" in df.columns and "C_N" in df.columns:
        add_cai("cai_joint_C_WBC_C_N", df["C_WBC"] * df["C_N"])
    if "C_P" in df.columns and "C_N" in df.columns:
        add_cai("cai_joint_C_P_C_N", df["C_P"] * df["C_N"])
    if "B_CRP" in df.columns and "B_PCT" in df.columns:
        add_cai("cai_joint_B_CRP_B_PCT", df["B_CRP"] * df["B_PCT"])

    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown"})
    num_cols = base_num_cols + base_interactions
    return df, num_cols, cat_cols, base_interactions, cai_cols


def generic_feature_recipe(df: pd.DataFrame, target: str):
    df = df.copy()
    num_cols, cat_cols = [], []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = df[c].median()
        df[c] = df[c].fillna(0.0 if pd.isna(med) else med)
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("Unknown").replace({"nan": "Unknown", "None": "Unknown"})
    return df, num_cols, cat_cols, [], []


def build_dataframe(cfg: Config):
    raw = load_raw_dataframe(cfg.data_path)
    if cfg.target not in raw.columns:
        raise ValueError(f"Target {cfg.target} not in columns")
    if cfg.dataset.lower() == "cns":
        df, num_cols, cat_cols, base_interactions, cai_cols = cns_feature_recipe(raw, cfg.target)
    else:
        df, num_cols, cat_cols, base_interactions, cai_cols = generic_feature_recipe(raw, cfg.target)
    y = pd.to_numeric(df[cfg.target], errors="coerce").values.astype(int)
    return df, num_cols, cat_cols, base_interactions, cai_cols, y

# =============================================================================
# Priors and datasets
# =============================================================================

def make_group_prior_from_scaled(x_num: np.ndarray, x_cat: np.ndarray, num_cols: List[str], cat_cols: List[str]) -> np.ndarray:
    idx = {c: i for i, c in enumerate(num_cols)}
    groups = {
        "csf": [idx[c] for c in ["C_G", "C_WBC", "C_RBC", "C_P", "C_N"] if c in idx],
        "blood": [idx[c] for c in ["B_G", "B_CRP", "B_WBC", "B_N", "B_Lym", "B_PCT", "B_AC", "B_RBC"] if c in idx],
        "clinical": [idx[c] for c in ["age", "GCS", "tem"] if c in idx],
        "interaction": [i for i, c in enumerate(num_cols) if c not in ["C_G", "C_WBC", "C_RBC", "C_P", "C_N", "B_G", "B_CRP", "B_WBC", "B_N", "B_Lym", "B_PCT", "B_AC", "B_RBC", "age", "GCS", "tem"]],
    }
    feats = []
    for _, ids in groups.items():
        if ids:
            g = x_num[:, ids]
            feats.extend([g.mean(axis=1, keepdims=True), g.max(axis=1, keepdims=True), g.min(axis=1, keepdims=True), g.std(axis=1, keepdims=True)])
        else:
            feats.append(np.zeros((len(x_num), 4), dtype=np.float32))
    if x_cat.shape[1] > 0:
        feats.append((x_cat > 0).astype(np.float32).mean(axis=1, keepdims=True))
    return np.concatenate(feats, axis=1).astype(np.float32)


def fit_transform_prototypes(x_train: np.ndarray, x_val: np.ndarray, cfg: Config):
    if x_train.shape[1] == 0:
        return np.zeros((len(x_train), 0), dtype=np.float32), np.zeros((len(x_val), 0), dtype=np.float32), None
    n_proto = min(cfg.n_prototypes, max(2, len(x_train) // 10))
    n_proto = min(n_proto, len(x_train))
    km = KMeans(n_clusters=n_proto, random_state=cfg.seed, n_init=10)
    km.fit(x_train)
    def soft_assign(x):
        dist = ((x[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(axis=-1)
        logits = -dist / max(cfg.proto_temperature, 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        q = np.exp(logits)
        q = q / (q.sum(axis=1, keepdims=True) + 1e-8)
        return q.astype(np.float32)
    return soft_assign(x_train), soft_assign(x_val), km


def make_generic_arithmetic_prior(x_num: np.ndarray, top_n: int = 6) -> np.ndarray:
    if x_num.shape[1] == 0:
        return np.zeros((len(x_num), 0), dtype=np.float32)
    m = min(top_n, x_num.shape[1])
    x = x_num[:, :m]
    feats = []
    eps = 1e-3
    for i in range(m):
        for j in range(i + 1, m):
            feats.append((x[:, i] * x[:, j])[:, None])
            feats.append((x[:, i] - x[:, j])[:, None])
            feats.append((x[:, i] / (np.abs(x[:, j]) + eps))[:, None])
    if not feats:
        return np.zeros((len(x_num), 0), dtype=np.float32)
    return np.clip(np.concatenate(feats, axis=1), -10, 10).astype(np.float32)


class RPITabDataset(Dataset):
    def __init__(self, x_num, x_cat, y, group_prior, proto_prior, arith_prior):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.group_prior = torch.tensor(group_prior, dtype=torch.float32)
        self.proto_prior = torch.tensor(proto_prior, dtype=torch.float32)
        self.arith_prior = torch.tensor(arith_prior, dtype=torch.float32)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        return self.x_num[idx], self.x_cat[idx], self.group_prior[idx], self.proto_prior[idx], self.arith_prior[idx], self.y[idx]


# =============================================================================
# Backbones
# =============================================================================

class NumericFeatureEmbedding(nn.Module):
    def __init__(self, n_num_features: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_num_features)])

    def forward(self, x_num):
        if x_num.size(1) == 0:
            return None
        return torch.stack([emb(x_num[:, i:i + 1]) for i, emb in enumerate(self.embeddings)], dim=1)


class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, cat_cardinalities: List[int], embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cardinalities])

    def forward(self, x_cat):
        if x_cat.size(1) == 0:
            return None
        return torch.stack([self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))], dim=1)


class TopKSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, top_k: Optional[int] = None, dropout: float = 0.1):
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

    def forward(self, x):
        B, L, D = x.shape
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.top_k is not None and self.top_k > 0 and self.top_k < L:
            vals, _ = torch.topk(scores, k=self.top_k, dim=-1)
            thr = vals[..., -1:].expand_as(scores)
            scores = scores.masked_fill(scores < thr, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.matmul(self.dropout(attn), v).transpose(1, 2).contiguous().view(B, L, D)
        return self.w_o(out), attn


class GatedArithmeticBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 3, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)
        combined = torch.cat([h + h_mean, h * h_mean, h - h_mean], dim=-1)
        out = self.dropout(self.proj(combined))
        return self.norm(h + self.gate(h_mean) * out)


class LightAMFormerLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, top_k, dropout, ff_mult):
        super().__init__()
        self.attn = TopKSparseAttention(embed_dim, n_heads, top_k, dropout)
        self.arith = GatedArithmeticBlock(embed_dim, dropout)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * ff_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * ff_mult, embed_dim), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, h):
        attn_out, _ = self.attn(h)
        h = self.norm1(h + attn_out)
        h = self.arith(h)
        return self.norm2(h + self.ffn(h))


class LightAMFormerBackbone(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, embed_dim, n_heads, n_layers, top_k, dropout, ff_mult):
        super().__init__()
        self.num_embedding = NumericFeatureEmbedding(n_num_features, embed_dim)
        self.cat_embedding = CategoricalFeatureEmbedding(cat_cardinalities, embed_dim)
        self.total_tokens = n_num_features + len(cat_cardinalities)
        self.pos = nn.Parameter(torch.randn(1, max(1, self.total_tokens), embed_dim) * 0.01)
        self.layers = nn.ModuleList([LightAMFormerLayer(embed_dim, n_heads, top_k, dropout, ff_mult) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.out_dim = embed_dim * 2

    def forward_tokens(self, x_num, x_cat):
        num_tok = self.num_embedding(x_num)
        cat_tok = self.cat_embedding(x_cat)
        if num_tok is not None and cat_tok is not None:
            h = torch.cat([num_tok, cat_tok], dim=1)
        elif num_tok is not None:
            h = num_tok
        elif cat_tok is not None:
            h = cat_tok
        else:
            raise ValueError("No features")
        h = h + self.pos[:, :h.size(1), :]
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)

    def forward(self, x_num, x_cat):
        h = self.forward_tokens(x_num, x_cat)
        feat = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return feat


class MLPBackbone(nn.Module):
    def __init__(self, n_num_features, cat_cardinalities, embed_dim, dropout):
        super().__init__()
        self.cat_embedding = CategoricalFeatureEmbedding(cat_cardinalities, embed_dim)
        in_dim = n_num_features + len(cat_cardinalities) * embed_dim
        hidden = max(64, embed_dim * 2)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out_dim = hidden

    def forward(self, x_num, x_cat):
        parts = [x_num]
        cat_tok = self.cat_embedding(x_cat)
        if cat_tok is not None:
            parts.append(cat_tok.flatten(1))
        return self.net(torch.cat(parts, dim=1))

# =============================================================================
# RPI-v1/v2 model
# =============================================================================

class ResidualPriorHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = None if in_dim <= 0 else nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        if self.net is None or x.size(1) == 0:
            return torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)
        return self.net(x).squeeze(-1)


class ArithmeticPriorHead(nn.Module):
    def __init__(self, arith_dim: int, state_dim: int, hidden_dim: int, dropout: float, use_gate: bool = True):
        super().__init__()
        self.use_gate = use_gate and arith_dim > 0 and state_dim > 0
        self.last_gate = None
        if arith_dim <= 0:
            self.encoder = None
            return
        if self.use_gate:
            self.gate = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, arith_dim), nn.Sigmoid())
        self.encoder = nn.Sequential(
            nn.Linear(arith_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x_arith, x_state):
        if self.encoder is None or x_arith.size(1) == 0:
            self.last_gate = None
            return torch.zeros((x_arith.size(0),), device=x_arith.device, dtype=x_arith.dtype)
        if self.use_gate:
            raw = self.gate(x_state)
            self.last_gate = raw.detach()
            x_arith = x_arith * (0.75 + 0.5 * raw)
        return self.encoder(x_arith).squeeze(-1)


class ReliabilityGate(nn.Module):
    def __init__(self, in_dim: int, n_priors: int, hidden_dim: int, dropout: float, strength_init: float = -2.0):
        super().__init__()
        self.gate_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, n_priors))
        self.strength_net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.strength_bias = nn.Parameter(torch.tensor(float(strength_init)))

    def forward(self, feat):
        logits = self.gate_net(feat)
        reliability = F.softmax(logits, dim=-1)
        strength = torch.sigmoid(self.strength_net(feat).squeeze(-1) + self.strength_bias)
        return reliability, strength, logits


class RPITabModel(nn.Module):
    def __init__(self, cfg: Config, n_num_features: int, cat_cardinalities: List[int], group_dim: int, proto_dim: int, arith_dim: int):
        super().__init__()
        self.cfg = cfg
        self.injection = cfg.injection
        self.prior_types = set(cfg.prior_types)

        concat_dim = 0
        if cfg.injection == "concat":
            concat_dim += group_dim if "group" in self.prior_types else 0
            concat_dim += proto_dim if "prototype" in self.prior_types else 0
            concat_dim += arith_dim if "arithmetic" in self.prior_types else 0

        if cfg.backbone == "mlp":
            self.backbone = MLPBackbone(n_num_features + concat_dim, cat_cardinalities, cfg.embed_dim, cfg.dropout)
        else:
            self.backbone = LightAMFormerBackbone(n_num_features + concat_dim, cat_cardinalities, cfg.embed_dim, cfg.n_heads, cfg.n_layers, cfg.top_k, cfg.dropout, cfg.ff_mult)

        self.classifier = nn.Sequential(nn.Linear(self.backbone.out_dim, cfg.embed_dim), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.embed_dim, cfg.embed_dim // 2), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.embed_dim // 2, 1))

        hidden = max(32, cfg.embed_dim)
        use_heads = cfg.injection in {"residual", "residual_gated"}
        self.group_head = ResidualPriorHead(group_dim, hidden, cfg.dropout) if use_heads and "group" in self.prior_types else None
        self.proto_head = ResidualPriorHead(proto_dim, hidden, cfg.dropout) if use_heads and "prototype" in self.prior_types else None
        self.arith_head = ArithmeticPriorHead(arith_dim, n_num_features, hidden, cfg.dropout, cfg.gate_arithmetic) if use_heads and "arithmetic" in self.prior_types else None

        if cfg.injection == "residual":
            self.group_scale = nn.Parameter(torch.tensor(float(cfg.residual_init))) if self.group_head is not None else None
            self.proto_scale = nn.Parameter(torch.tensor(float(cfg.residual_init))) if self.proto_head is not None else None
            self.arith_scale = nn.Parameter(torch.tensor(float(cfg.residual_init))) if self.arith_head is not None else None
        else:
            self.group_scale = self.proto_scale = self.arith_scale = None

        self.active_prior_names = []
        if cfg.injection == "residual_gated":
            if self.group_head is not None: self.active_prior_names.append("group")
            if self.proto_head is not None: self.active_prior_names.append("prototype")
            if self.arith_head is not None: self.active_prior_names.append("arithmetic")
            if not self.active_prior_names:
                raise ValueError("residual_gated needs at least one prior")
            self.reliability_gate = ReliabilityGate(self.backbone.out_dim, len(self.active_prior_names), hidden, cfg.dropout, cfg.prior_strength_init)
        else:
            self.reliability_gate = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _prior_logits(self, x_num, gp, pp, ap):
        d = {}
        gate_l1 = torch.tensor(0.0, device=x_num.device)
        if self.group_head is not None: d["group"] = self.group_head(gp)
        if self.proto_head is not None: d["prototype"] = self.proto_head(pp)
        if self.arith_head is not None:
            d["arithmetic"] = self.arith_head(ap, x_num)
            if self.arith_head.last_gate is not None:
                gate_l1 = self.arith_head.last_gate.mean()
        return d, gate_l1

    def forward(self, x_num, x_cat, gp, pp, ap, return_parts=False):
        x_in = x_num
        if self.injection == "concat":
            ps = []
            if "group" in self.prior_types and gp.size(1) > 0: ps.append(gp)
            if "prototype" in self.prior_types and pp.size(1) > 0: ps.append(pp)
            if "arithmetic" in self.prior_types and ap.size(1) > 0: ps.append(ap)
            if ps: x_in = torch.cat([x_num, *ps], dim=1)
        feat = self.backbone(x_in, x_cat)
        z_base = self.classifier(feat).squeeze(-1)
        z_final = z_base
        parts = {"logit": z_final, "z_base": z_base, "z_group": None, "z_proto": None, "z_arith": None,
                 "group_weight": None, "proto_weight": None, "arith_weight": None, "prior_gate": None,
                 "prior_strength": None, "prior_logits": None, "gate_l1": torch.tensor(0.0, device=x_num.device)}
        if self.injection == "residual":
            prior_logits, gate_l1 = self._prior_logits(x_num, gp, pp, ap)
            parts["gate_l1"] = gate_l1
            if "group" in prior_logits:
                w = torch.sigmoid(self.group_scale); z_final = z_final + w * prior_logits["group"]; parts["z_group"] = prior_logits["group"]; parts["group_weight"] = w
            if "prototype" in prior_logits:
                w = torch.sigmoid(self.proto_scale); z_final = z_final + w * prior_logits["prototype"]; parts["z_proto"] = prior_logits["prototype"]; parts["proto_weight"] = w
            if "arithmetic" in prior_logits:
                w = torch.sigmoid(self.arith_scale); z_final = z_final + w * prior_logits["arithmetic"]; parts["z_arith"] = prior_logits["arithmetic"]; parts["arith_weight"] = w
        elif self.injection == "residual_gated":
            prior_logits_dict, gate_l1 = self._prior_logits(x_num, gp, pp, ap)
            parts["gate_l1"] = gate_l1
            ordered = []
            for name in self.active_prior_names:
                z = prior_logits_dict[name]; ordered.append(z)
                if name == "group": parts["z_group"] = z
                if name == "prototype": parts["z_proto"] = z
                if name == "arithmetic": parts["z_arith"] = z
            prior_logits = torch.stack(ordered, dim=1)
            rel, strength, _ = self.reliability_gate(feat)
            residual = torch.sum(rel * prior_logits, dim=1)
            z_final = z_final + strength * residual
            parts["prior_gate"] = rel; parts["prior_strength"] = strength; parts["prior_logits"] = prior_logits
            for j, name in enumerate(self.active_prior_names):
                if name == "group": parts["group_weight"] = rel[:, j].mean()
                if name == "prototype": parts["proto_weight"] = rel[:, j].mean()
                if name == "arithmetic": parts["arith_weight"] = rel[:, j].mean()
        parts["logit"] = z_final
        return parts if return_parts else z_final


# =============================================================================
# Trainer and fold preparation
# =============================================================================

class Trainer:
    def __init__(self, model: RPITabModel, cfg: Config, pos_weight=None):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = cfg.device
        self.pos_weight = pos_weight
        self.train_losses, self.val_losses, self.val_aucs = [], [], []

    def unpack(self, batch):
        x_num, x_cat, gp, pp, ap, y = batch
        return x_num.to(self.device), x_cat.to(self.device), gp.to(self.device), pp.to(self.device), ap.to(self.device), y.to(self.device)

    def make_optimizer(self):
        residual_keys = ["group_head", "proto_head", "arith_head", "reliability_gate"]
        scale_keys = ["group_scale", "proto_scale", "arith_scale", "strength_bias"]
        base, residual, scale = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if any(k in name for k in scale_keys): scale.append(p)
            elif any(k in name for k in residual_keys): residual.append(p)
            else: base.append(p)
        groups = [{"params": base, "lr": self.cfg.lr}]
        if residual: groups.append({"params": residual, "lr": self.cfg.lr * self.cfg.residual_lr_mult})
        if scale: groups.append({"params": scale, "lr": self.cfg.lr * self.cfg.scale_lr_mult})
        return torch.optim.AdamW(groups, weight_decay=self.cfg.weight_decay)

    def step_loss(self, parts, y, criterion):
        logits = parts["logit"]
        loss = criterion(logits, y)
        if self.cfg.injection in {"residual", "residual_gated"} and self.cfg.residual_l2 > 0:
            loss = loss + self.cfg.residual_l2 * torch.mean((logits - parts["z_base"].detach()) ** 2)
        if self.cfg.gate_l1_lambda > 0 and parts.get("gate_l1") is not None:
            loss = loss + self.cfg.gate_l1_lambda * parts["gate_l1"]
        if self.cfg.injection == "residual_gated" and parts.get("prior_gate") is not None:
            if self.cfg.prior_gate_entropy_lambda > 0:
                g = parts["prior_gate"].clamp_min(1e-8)
                entropy = -(g * torch.log(g)).sum(dim=1).mean()
                loss = loss + self.cfg.prior_gate_entropy_lambda * entropy
            if self.cfg.prior_utility_lambda > 0 and parts.get("prior_logits") is not None:
                with torch.no_grad():
                    y_expand = y.view(-1, 1).expand_as(parts["prior_logits"])
                    base_loss = F.binary_cross_entropy_with_logits(parts["z_base"].detach(), y, reduction="none").view(-1, 1)
                    cand = parts["z_base"].detach().view(-1, 1) + parts["prior_logits"].detach()
                    cand_loss = F.binary_cross_entropy_with_logits(cand, y_expand, reduction="none")
                    utility = base_loss - cand_loss
                    target_gate = F.softmax(utility / max(self.cfg.prior_utility_temperature, 1e-6), dim=1)
                pred_gate = parts["prior_gate"].clamp_min(1e-8)
                loss = loss + self.cfg.prior_utility_lambda * F.kl_div(torch.log(pred_gate), target_gate, reduction="batchmean")
        return loss

    def train_epoch(self, loader, opt, criterion):
        self.model.train(); total = 0.0
        for batch in loader:
            x_num, x_cat, gp, pp, ap, y = self.unpack(batch)
            opt.zero_grad()
            parts = self.model(x_num, x_cat, gp, pp, ap, return_parts=True)
            loss = self.step_loss(parts, y, criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            opt.step(); total += float(loss.item())
        return total / max(len(loader), 1)

    def validate(self, loader, criterion):
        self.model.eval(); total=0.0; probs=[]; labels=[]
        with torch.no_grad():
            for batch in loader:
                x_num, x_cat, gp, pp, ap, y = self.unpack(batch)
                logits = self.model(x_num, x_cat, gp, pp, ap)
                total += float(criterion(logits, y).item())
                probs.extend(torch.sigmoid(logits).cpu().numpy()); labels.extend(y.cpu().numpy())
        probs=np.array(probs); labels=np.array(labels)
        return total / max(len(loader), 1), probs, labels, safe_auc(labels, probs)

    def train(self, train_loader, val_loader, save_path):
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        opt = self.make_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6, min_lr=self.cfg.min_lr)
        best_auc, patience = -1.0, 0
        torch.save(self.model.state_dict(), save_path)
        for _ in range(self.cfg.epochs):
            tr = self.train_epoch(train_loader, opt, criterion)
            va_loss, _, _, va_auc = self.validate(val_loader, criterion)
            self.train_losses.append(tr); self.val_losses.append(va_loss); self.val_aucs.append(va_auc)
            scheduler.step(va_auc)
            if va_auc > best_auc + 1e-5:
                best_auc = va_auc; patience = 0; torch.save(self.model.state_dict(), save_path)
            else:
                patience += 1
            if patience >= self.cfg.patience: break
        self.model.load_state_dict(torch.load(save_path, map_location=self.device))

    def predict(self, loader):
        self.model.eval(); probs=[]; labels=[]; weights={"group":[], "proto":[], "arith":[], "strength":[]}
        with torch.no_grad():
            for batch in loader:
                x_num, x_cat, gp, pp, ap, y = self.unpack(batch)
                parts = self.model(x_num, x_cat, gp, pp, ap, return_parts=True)
                probs.extend(torch.sigmoid(parts["logit"]).cpu().numpy()); labels.extend(y.cpu().numpy())
                for key, part_key in [("group","group_weight"),("proto","proto_weight"),("arith","arith_weight")]:
                    if parts.get(part_key) is not None:
                        weights[key].append(float(parts[part_key].detach().cpu().mean().item()))
                if parts.get("prior_strength") is not None:
                    weights["strength"].append(float(parts["prior_strength"].detach().cpu().mean().item()))
        summary = {f"{k}_weight_mean": float(np.mean(v)) if v else np.nan for k,v in weights.items()}
        return np.array(probs), np.array(labels), summary

def prepare_fold_arrays(cfg: Config, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], cai_cols: List[str], train_idx, val_idx):
    df_train = df.iloc[train_idx].copy(); df_val = df.iloc[val_idx].copy()
    scaler = StandardScaler()
    if num_cols:
        x_train_num = scaler.fit_transform(df_train[num_cols].values.astype(np.float32))
        x_val_num = scaler.transform(df_val[num_cols].values.astype(np.float32))
    else:
        x_train_num = np.zeros((len(df_train), 0), dtype=np.float32); x_val_num=np.zeros((len(df_val),0),dtype=np.float32)
    cat_encoder = CategoryEncoder().fit(df_train, cat_cols)
    x_train_cat = cat_encoder.transform(df_train, cat_cols); x_val_cat = cat_encoder.transform(df_val, cat_cols)

    if cai_cols:
        a_scaler = StandardScaler()
        a_train = a_scaler.fit_transform(df_train[cai_cols].values.astype(np.float32))
        a_val = a_scaler.transform(df_val[cai_cols].values.astype(np.float32))
    else:
        a_train = make_generic_arithmetic_prior(x_train_num)
        a_val = make_generic_arithmetic_prior(x_val_num)
        if a_train.shape[1] > 0:
            a_scaler = StandardScaler()
            a_train = a_scaler.fit_transform(a_train); a_val = a_scaler.transform(a_val)
    g_train = make_group_prior_from_scaled(x_train_num, x_train_cat, num_cols, cat_cols)
    g_val = make_group_prior_from_scaled(x_val_num, x_val_cat, num_cols, cat_cols)
    p_train, p_val, km = fit_transform_prototypes(x_train_num, x_val_num, cfg)
    return {
        "x_train_num": x_train_num.astype(np.float32), "x_val_num": x_val_num.astype(np.float32),
        "x_train_cat": x_train_cat.astype(np.int64), "x_val_cat": x_val_cat.astype(np.int64),
        "group_train": g_train.astype(np.float32), "group_val": g_val.astype(np.float32),
        "proto_train": p_train.astype(np.float32), "proto_val": p_val.astype(np.float32),
        "arith_train": a_train.astype(np.float32), "arith_val": a_val.astype(np.float32),
        "cat_cardinalities": cat_encoder.cardinalities(cat_cols), "scaler": scaler, "cat_encoder": cat_encoder, "kmeans": km,
    }


def default_experiments():
    return [
        {"name":"E0_NoPrior", "description":"Backbone without explicit prior injection.", "injection":"none", "prior_types":()},
        {"name":"E1_ConcatAllPriors", "description":"Concatenate group/prototype/arithmetic priors to numeric input.", "injection":"concat", "prior_types":("group","prototype","arithmetic")},
        {"name":"E3_RPIv1_Group", "description":"RPI-v1: residual injection with group prior only.", "injection":"residual", "prior_types":("group",)},
        {"name":"E5_RPIv1_Arithmetic", "description":"RPI-v1: residual injection with arithmetic prior only.", "injection":"residual", "prior_types":("arithmetic",)},
        {"name":"E6_RPIv1_AllPriors", "description":"RPI-v1: global residual weights for group+prototype+arithmetic priors.", "injection":"residual", "prior_types":("group","prototype","arithmetic")},
        {"name":"E7_RPIv2_ReliabilityGated_All", "description":"RPI-v2: instance-wise reliability-gated residual injection with all priors.", "injection":"residual_gated", "prior_types":("group","prototype","arithmetic")},
        {"name":"E8_RPIv2_ReliabilityGated_GroupArith", "description":"RPI-v2 without prototype: reliability-gated group+arithmetic priors.", "injection":"residual_gated", "prior_types":("group","arithmetic")},
    ]


def run_single_experiment(cfg: Config, exp: Dict[str, Any], df, num_cols, cat_cols, cai_cols, y, out_dir: Path):
    exp_cfg = deepcopy(cfg); exp_cfg.injection = exp["injection"]; exp_cfg.prior_types = tuple(exp["prior_types"])
    skf = StratifiedKFold(n_splits=exp_cfg.n_splits, shuffle=True, random_state=exp_cfg.seed)
    best_rows=[]; fixed_rows=[]; fold_rows=[]; weight_rows=[]
    exp_dir = out_dir / exp["name"]; exp_dir.mkdir(parents=True, exist_ok=True)
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        set_seed(exp_cfg.seed + fold * 97)
        if exp_cfg.sample_fraction < 1.0:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=exp_cfg.sample_fraction, random_state=exp_cfg.seed + fold)
            rel_train, _ = next(splitter.split(np.zeros(len(train_idx)), y[train_idx]))
            train_idx = train_idx[rel_train]
        y_train = y[train_idx]; y_val = y[val_idx]
        arr = prepare_fold_arrays(exp_cfg, df, num_cols, cat_cols, cai_cols, train_idx, val_idx)
        train_ds = RPITabDataset(arr["x_train_num"], arr["x_train_cat"], y_train, arr["group_train"], arr["proto_train"], arr["arith_train"])
        val_ds = RPITabDataset(arr["x_val_num"], arr["x_val_cat"], y_val, arr["group_val"], arr["proto_val"], arr["arith_val"])
        train_loader = DataLoader(train_ds, batch_size=exp_cfg.batch_size, shuffle=True, num_workers=exp_cfg.num_workers)
        val_loader = DataLoader(val_ds, batch_size=exp_cfg.batch_size, shuffle=False, num_workers=exp_cfg.num_workers)
        pos = int(y_train.sum()); neg = int(len(y_train) - pos)
        pos_weight = torch.tensor((neg / max(pos,1)) * exp_cfg.pos_weight_scale, dtype=torch.float32, device=exp_cfg.device)
        model = RPITabModel(exp_cfg, arr["x_train_num"].shape[1], arr["cat_cardinalities"], arr["group_train"].shape[1], arr["proto_train"].shape[1], arr["arith_train"].shape[1])
        trainer = Trainer(model, exp_cfg, pos_weight)
        trainer.train(train_loader, val_loader, str(exp_dir / f"best_model_fold{fold}.pth"))
        probs, labels, weights = trainer.predict(val_loader)
        best_t = find_best_threshold(labels, probs, exp_cfg.threshold_mode, exp_cfg.fn_cost, exp_cfg.fp_cost) if exp_cfg.optimize_threshold else exp_cfg.fixed_threshold
        best = compute_metrics(labels, probs, best_t); fixed = compute_metrics(labels, probs, exp_cfg.fixed_threshold)
        best_rows.append(best); fixed_rows.append(fixed); weight_rows.append(weights)
        row = {"fold": fold, **{f"best_{k}":v for k,v in best.items() if k not in ["tn","fp","fn","tp"]}, **{f"fixed_{k}":v for k,v in fixed.items() if k not in ["tn","fp","fn","tp"]}, **weights}
        fold_rows.append(row)
        pd.DataFrame({"y_true": labels, "y_prob": probs, "best_threshold": best_t}).to_csv(exp_dir / f"fold_{fold}_predictions.csv", index=False)
        print(f"[{exp['name']}] fold {fold}: AUC={best['auc']:.4f}, F1={best['f1']:.4f}, MCC={best['mcc']:.4f}, T={best_t:.2f}")
    best_summary = summarize(best_rows, ""); fixed_summary = summarize(fixed_rows, "fixed_")
    wsum = {}
    for k in ["group_weight_mean", "proto_weight_mean", "arith_weight_mean", "strength_weight_mean"]:
        vals = [w.get(k, np.nan) for w in weight_rows]
        vals = [v for v in vals if not np.isnan(v)]
        wsum[k] = float(np.mean(vals)) if vals else np.nan
    summary = {"experiment": exp["name"], "description": exp["description"], "backbone": cfg.backbone, "injection": exp_cfg.injection, "prior_types":"+".join(exp_cfg.prior_types) if exp_cfg.prior_types else "none", **best_summary, **fixed_summary, **wsum}
    pd.DataFrame(fold_rows).to_csv(exp_dir / "fold_metrics.csv", index=False)
    with open(exp_dir / "summary.json", "w", encoding="utf-8") as f: json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cns", choices=["cns","csv"])
    p.add_argument("--data_path", type=str, default="original.xlsx")
    p.add_argument("--target", type=str, default="outcome")
    p.add_argument("--output_dir", type=str, default="rpi_tab_v2_results")
    p.add_argument("--backbone", type=str, default="light_amformer", choices=["light_amformer","mlp"])
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample_fraction", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--residual_l2", type=float, default=0.01)
    p.add_argument("--prior_gate_entropy_lambda", type=float, default=0.0)
    p.add_argument("--prior_utility_lambda", type=float, default=0.0)
    args = p.parse_args()
    return Config(dataset=args.dataset, data_path=args.data_path, target=args.target, output_dir=args.output_dir, backbone=args.backbone, epochs=30 if args.quick else args.epochs, batch_size=args.batch_size, seed=args.seed, sample_fraction=args.sample_fraction, device=args.device, residual_l2=args.residual_l2, prior_gate_entropy_lambda=args.prior_gate_entropy_lambda, prior_utility_lambda=args.prior_utility_lambda)


def main():
    cfg = parse_args(); set_seed(cfg.seed)
    df, num_cols, cat_cols, base_interactions, cai_cols, y = build_dataframe(cfg)
    out_dir = Path(cfg.output_dir) / f"{cfg.dataset}_{cfg.backbone}_frac{cfg.sample_fraction}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("="*80)
    print("RPI-Tab v2: Reliability-Gated Residual Prior Injection")
    print("="*80)
    print(f"Dataset: {cfg.dataset}")
    print(f"Data path: {cfg.data_path}")
    print(f"Target: {cfg.target}")
    print(f"Samples: {len(df)} | Positives: {int(y.sum())} | Negatives: {int(len(y)-y.sum())}")
    print(f"Backbone: {cfg.backbone}")
    print(f"Numeric cols: {len(num_cols)} | Categorical cols: {len(cat_cols)} | CAI prior cols: {len(cai_cols)}")
    print(f"Device: {cfg.device}")
    print(f"Output: {out_dir.resolve()}")
    print("="*80)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f: json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    with open(out_dir / "features.json", "w", encoding="utf-8") as f: json.dump({"num_cols":num_cols,"cat_cols":cat_cols,"base_interactions":base_interactions,"cai_cols":cai_cols}, f, ensure_ascii=False, indent=2)
    summaries=[]
    for exp in default_experiments():
        print("\n" + "#"*80)
        print(f"Running {exp['name']}: {exp['description']}")
        print("#"*80)
        summary = run_single_experiment(cfg, exp, df, num_cols, cat_cols, cai_cols, y, out_dir)
        summaries.append(summary)
        pd.DataFrame(summaries).sort_values(["auc_mean","mcc_mean","f1_mean"], ascending=[False,False,False]).to_csv(out_dir / "summary.csv", index=False)
        with open(out_dir / "all_results.json", "w", encoding="utf-8") as f: json.dump(summaries, f, ensure_ascii=False, indent=2)
    final = pd.DataFrame(summaries).sort_values(["auc_mean","mcc_mean","f1_mean"], ascending=[False,False,False])
    final.to_csv(out_dir / "summary.csv", index=False)
    print("\n" + "="*80)
    print("Final summary")
    print("="*80)
    cols = ["experiment","backbone","injection","prior_types","auc_mean","auc_std","recall_mean","f1_mean","mcc_mean","fixed_mcc_mean","strength_weight_mean"]
    print(final[[c for c in cols if c in final.columns]].to_string(index=False))
    print(f"\nSaved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
