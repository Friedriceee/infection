"""
LightAMFormer 三阶段固定实验版
============================================================
目标：
1) 不做网格搜索，直接使用你已经跑出的 LightAMFormer 最佳参数；
2) 先复现稳定静态主干；
3) 再测试 DAE 静态增强；
4) 最后在静态模型稳定后，以“两阶段训练”的方式加入动态残差分支。

三个实验：
- exp1_light_static_best:
    LightAMFormer 原型，最佳参数 top_k=None, pos_weight_scale=1.1, embed_dim=64。

- exp2_light_dae_static:
    在 LightAMFormer 上只替换数值嵌入为 Deviation-Aware Embedding，仍然只用静态数据。
    用于判断 DAE 是否真的提升静态 AUC。

- exp3_light_dynamic_residual:
    以 LightAMFormer 为锚点的动态残差模型。
    Phase 1: 先训练静态 LightAMFormer 主干；
    Phase 2: 冻结静态主干，只训练动态 residual head；
    final_logit = static_logit + sigmoid(scale) * has_dyn * dynamic_logit。
    这样动态数据只做小幅修正，不会破坏静态模型。

注意：
- AUC 不受 threshold 影响；threshold 只影响 Recall/F1/Accuracy。
- 输出会同时保存 fixed threshold=0.5 和 best threshold 两套结果。
"""

import os
import json
import pickle
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Tuple, Dict

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
    matthews_corrcoef,
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_prob)


def safe_mcc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return matthews_corrcoef(y_true, y_pred)


def find_best_threshold(y_true, y_prob, mode="recall_f1", fn_cost=3.0, fp_cost=1.0):
    """
    医疗筛查场景建议不要固定 0.5。
    mode:
    - recall_f1: 优先召回，兼顾 F1
    - cost: 最小化 fn_cost*FN + fp_cost*FP
    - f1: 单纯最大化 F1
    - youden: 最大化 sensitivity + specificity - 1
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
        pre = precision_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        specificity = tn / max(tn + fp, 1)

        if mode == "cost":
            score = -(fn_cost * fn + fp_cost * fp)
        elif mode == "f1":
            score = f1
        elif mode == "youden":
            score = rec + specificity - 1
        else:
            # 你之前的结果 recall 很重要，所以默认仍然偏向召回
            score = 0.70 * rec + 0.30 * f1

        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t


def evaluate_from_probs(labels, probs, threshold: float):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
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
        "mcc": safe_mcc(labels, preds),
        "confusion_matrix": cm.tolist(),
    }


# =============================================================
# 配置：直接使用 LightAMFormer 已知最佳参数
# =============================================================
@dataclass
class Config:
    data_path: str = "original.xlsx"
    output_dir: str = "three_experiments_light_anchor_results"
    label_col: str = "outcome"

    # 动态数据：可选。exp1/exp2 不使用；exp3 使用。
    # 支持宽表 dynamic_curves.csv: patient_id, C_G_t0, C_G_t1, ...
    dynamic_data_path: str = "dynamic_curves.csv"
    static_id_col: str = "ID"
    dyn_id_col: str = "patient_id"
    dyn_cols: Tuple[str, ...] = ("C_G", "C_WBC", "C_RBC", "C_P", "C_N")
    n_time_points: int = 4

    # 训练参数
    batch_size: int = 32
    epochs: int = 120
    lr: float = 3e-4
    min_lr: float = 1e-5
    patience: int = 25
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    # 动态残差第二阶段训练参数
    dynamic_epochs: int = 80
    dynamic_lr: float = 1e-4
    dynamic_patience: int = 18
    dynamic_weight_decay: float = 1e-3

    # 阈值
    fixed_threshold: float = 0.5
    optimize_threshold: bool = True
    threshold_mode: str = "recall_f1"
    fn_cost: float = 3.0
    fp_cost: float = 1.0

    # LightAMFormer 最佳参数：直接上，不搜索
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    top_k: Optional[int] = None
    dropout: float = 0.20
    ff_mult: int = 4
    pos_weight_scale: float = 1.1

    seed: int = 42
    n_splits: int = 5
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()

EXPERIMENTS = [
    {
        "name": "exp1_light_static_best",
        "description": "LightAMFormer 原型；固定最佳参数；只用静态数据。",
        "use_dae": False,
        "use_dynamic_residual": False,
    },
    {
        "name": "exp2_light_dae_static",
        "description": "LightAMFormer + DAE；只用静态数据；验证 DAE 是否提升静态 AUC。",
        "use_dae": True,
        "use_dynamic_residual": False,
    },
    {
        "name": "exp3_light_dynamic_residual",
        "description": "LightAMFormer 锚点 + 动态残差；两阶段训练；动态只做小幅风险修正。",
        "use_dae": False,
        "use_dynamic_residual": True,
    },
]


# =============================================================
# 数据集
# =============================================================
class CSFDataset(Dataset):
    def __init__(self, x_num, x_cat, x_dyn=None, dyn_mask=None, labels=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        if x_dyn is None:
            x_dyn = np.full((len(x_num), 0, 0), np.nan, dtype=np.float32)
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


# =============================================================
# 特征工程
# =============================================================
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


class RobustDeviationStats:
    def __init__(self):
        self.mu = None
        self.sigma = None

    def fit(self, x: np.ndarray, eps: float = 1e-6):
        self.mu = np.median(x, axis=0)
        q75 = np.percentile(x, 75, axis=0)
        q25 = np.percentile(x, 25, axis=0)
        self.sigma = (q75 - q25) / 1.3489795 + eps
        self.sigma = np.where(self.sigma < eps, 1.0, self.sigma)
        return self

    def to_tensors(self, device):
        return (
            torch.tensor(self.mu, dtype=torch.float32, device=device),
            torch.tensor(self.sigma, dtype=torch.float32, device=device),
        )


# =============================================================
# 动态数据加载：支持 sparse wide csv
# =============================================================
def load_sparse_dynamic_data(static_df: pd.DataFrame, cfg: Config):
    N = len(static_df)
    F_dyn = len(cfg.dyn_cols)
    T = cfg.n_time_points
    dynamic_array = np.full((N, F_dyn, T), np.nan, dtype=np.float32)
    dyn_mask = np.zeros(N, dtype=np.float32)

    path = Path(cfg.dynamic_data_path)
    if not cfg.dynamic_data_path or not path.exists():
        print(f"  ⚠️ 未找到动态数据文件: {cfg.dynamic_data_path}，exp3 会自动退化为静态模型。")
        return dynamic_array, dyn_mask, 0

    if cfg.static_id_col not in static_df.columns:
        print(f"  ⚠️ 静态数据缺少 ID 列 {cfg.static_id_col}，无法匹配动态数据。")
        return dynamic_array, dyn_mask, 0

    dyn_df = pd.read_csv(path)
    if cfg.dyn_id_col not in dyn_df.columns:
        print(f"  ⚠️ 动态数据缺少 ID 列 {cfg.dyn_id_col}，无法匹配动态数据。")
        return dynamic_array, dyn_mask, 0

    id_to_idx = {str(pid): i for i, pid in enumerate(static_df[cfg.static_id_col].astype(str).values)}
    matched = 0
    missing_ids = 0

    for _, row in dyn_df.iterrows():
        pid = str(row[cfg.dyn_id_col])
        if pid not in id_to_idx:
            missing_ids += 1
            continue
        idx = id_to_idx[pid]
        any_valid = False
        for fi, feat in enumerate(cfg.dyn_cols):
            for ti in range(T):
                col = f"{feat}_t{ti}"
                if col in dyn_df.columns:
                    val = pd.to_numeric(row[col], errors="coerce")
                    if pd.notna(val):
                        dynamic_array[idx, fi, ti] = float(val)
                        any_valid = True
        if any_valid:
            dyn_mask[idx] = 1.0
            matched += 1

    print(f"  动态数据匹配: {matched}/{len(dyn_df)} 行；静态中找不到 ID: {missing_ids}")
    print(f"  动态覆盖率: {int(dyn_mask.sum())}/{N} ({100 * dyn_mask.sum() / max(N, 1):.1f}%)")
    if dyn_mask.sum() > 0:
        miss_rate = np.isnan(dynamic_array[dyn_mask.astype(bool)]).mean(axis=(0, 1))
        print("  各时间点缺失率: " + ", ".join([f"T{i}={miss_rate[i]:.1%}" for i in range(T)]))
    return dynamic_array, dyn_mask, F_dyn


def preprocess_dynamic_by_fold(
    dyn_train: np.ndarray,
    dyn_val: np.ndarray,
    dyn_cols: List[str],
    num_cols: List[str],
    scaler: StandardScaler,
):
    """动态变量与静态变量做同样 log1p 和 StandardScaler 参数对齐。NaN 保留给动态模块处理。"""
    if dyn_train.shape[1] == 0:
        return dyn_train, dyn_val

    log_cols = {"C_WBC", "C_RBC", "C_P", "B_CRP", "B_WBC", "B_PCT", "B_AC", "B_RBC"}
    dyn_train = dyn_train.copy()
    dyn_val = dyn_val.copy()

    for fi, feat in enumerate(dyn_cols):
        if feat not in num_cols:
            continue

        if feat in log_cols:
            valid = ~np.isnan(dyn_train[:, fi, :]) & (dyn_train[:, fi, :] >= 0)
            dyn_train[:, fi, :] = np.where(valid, np.log1p(dyn_train[:, fi, :]), dyn_train[:, fi, :])
            valid_v = ~np.isnan(dyn_val[:, fi, :]) & (dyn_val[:, fi, :] >= 0)
            dyn_val[:, fi, :] = np.where(valid_v, np.log1p(dyn_val[:, fi, :]), dyn_val[:, fi, :])

        static_idx = num_cols.index(feat)
        m = scaler.mean_[static_idx]
        s = scaler.scale_[static_idx] + 1e-6
        dyn_train[:, fi, :] = (dyn_train[:, fi, :] - m) / s
        dyn_val[:, fi, :] = (dyn_val[:, fi, :] - m) / s

    return dyn_train, dyn_val


# =============================================================
# 模型模块
# =============================================================
class NumericFeatureEmbedding(nn.Module):
    def __init__(self, n_num_features: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_num_features)])

    def forward(self, x_num: torch.Tensor):
        if x_num.size(1) == 0:
            return None
        tokens = [self.embeddings[i](x_num[:, i:i + 1]) for i in range(len(self.embeddings))]
        return torch.stack(tokens, dim=1)


class DeviationAwareNumericEmbedding(nn.Module):
    """静态 DAE：只替换数值 embedding，不改变 AMFormer 主干。"""
    def __init__(self, n_num_features: int, embed_dim: int):
        super().__init__()
        self.n = n_num_features
        self.w_raw = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_num_features)])
        self.w_dev = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_num_features)])
        self.alpha = nn.Parameter(torch.ones(n_num_features) * 0.5)
        self.register_buffer("mu", torch.zeros(n_num_features))
        self.register_buffer("sigma", torch.ones(n_num_features))

    def set_stats(self, mu: torch.Tensor, sigma: torch.Tensor):
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)

    def forward(self, x_num: torch.Tensor):
        if x_num.size(1) == 0:
            return None
        raw_tokens = [self.w_raw[i](x_num[:, i:i + 1]) for i in range(self.n)]
        x_dev = (x_num - self.mu) / self.sigma
        dev_tokens = [self.w_dev[i](x_dev[:, i:i + 1]) for i in range(self.n)]
        raw = torch.stack(raw_tokens, dim=1)
        dev = torch.stack(dev_tokens, dim=1)
        a = torch.sigmoid(self.alpha).view(1, self.n, 1)
        return (1.0 - a) * raw + a * dev


class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, cat_cardinalities, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities])

    def forward(self, x_cat: torch.Tensor):
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
    def __init__(self, embed_dim: int, n_heads: int, top_k: Optional[int] = None,
                 dropout: float = 0.2, ff_mult: int = 4):
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


class DynamicTrendEncoder(nn.Module):
    """短时序趋势编码器：输入 [B, F, T]，输出 [B, D]。NaN 表示缺失。"""
    def __init__(self, n_dyn_features: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_dyn_features = n_dyn_features
        self.trend_mlp = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)
        self.norm = nn.LayerNorm(embed_dim)
        self.last_attn = None

    def extract_stats(self, x_dyn: torch.Tensor):
        B, Fd, T = x_dyn.shape
        valid = (~torch.isnan(x_dyn)).float()
        n_valid = valid.sum(dim=-1, keepdim=True)
        x_safe = torch.nan_to_num(x_dyn, nan=0.0)

        mean = (x_safe * valid).sum(dim=-1, keepdim=True) / (n_valid + 1e-8)

        x_max_in = torch.where(valid > 0, x_safe, torch.full_like(x_safe, -1e9))
        x_min_in = torch.where(valid > 0, x_safe, torch.full_like(x_safe, 1e9))
        x_max = x_max_in.max(dim=-1, keepdim=True).values
        x_min = x_min_in.min(dim=-1, keepdim=True).values
        all_missing = n_valid < 1
        x_max = torch.where(all_missing, torch.zeros_like(x_max), x_max)
        x_min = torch.where(all_missing, torch.zeros_like(x_min), x_min)

        t = torch.linspace(0, 1, T, device=x_dyn.device).view(1, 1, T).expand(B, Fd, T)
        t_mean = (t * valid).sum(dim=-1, keepdim=True) / (n_valid + 1e-8)
        t_c = (t - t_mean) * valid
        x_c = (x_safe - mean) * valid
        slope = (t_c * x_c).sum(dim=-1, keepdim=True) / (t_c.pow(2).sum(dim=-1, keepdim=True) + 1e-8)

        var = ((x_safe - mean).pow(2) * valid).sum(dim=-1, keepdim=True) / (n_valid + 1e-8)
        volatility = torch.sqrt(var + 1e-8)
        volatility = torch.where(n_valid < 2, torch.zeros_like(volatility), volatility)

        return torch.cat([mean, x_max, x_min, slope, volatility], dim=-1)

    def forward(self, x_dyn: torch.Tensor):
        stats = self.extract_stats(x_dyn)             # [B, F, 5]
        tokens = self.trend_mlp(stats)                # [B, F, D]
        B, Fd, D = tokens.shape
        q = self.query.unsqueeze(0).expand(B, -1, -1) # [B, 1, D]
        scores = torch.matmul(q, tokens.transpose(-2, -1)) / (D ** 0.5)
        attn = F.softmax(scores, dim=-1)
        self.last_attn = attn.detach()
        h_dyn = torch.matmul(attn, tokens).squeeze(1)
        return self.norm(h_dyn)


class DynamicResidualHead(nn.Module):
    """动态残差头：只输出修正 logit，不替代静态 logit。"""
    def __init__(self, n_dyn_features: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = DynamicTrendEncoder(n_dyn_features, embed_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, h_static: torch.Tensor, x_dyn: torch.Tensor, dyn_mask: torch.Tensor):
        h_dyn = self.encoder(x_dyn)
        z = torch.cat([h_static.detach(), h_dyn], dim=-1)
        dyn_logit = self.head(z).squeeze(-1)
        return dyn_logit * dyn_mask


class LightAMFormer(nn.Module):
    def __init__(self, n_num_features: int, cat_cardinalities, embed_dim: int = 64,
                 n_heads: int = 4, n_layers: int = 2, top_k: Optional[int] = None,
                 dropout: float = 0.2, ff_mult: int = 4, use_dae: bool = False,
                 use_dynamic_residual: bool = False, n_dyn_features: int = 0):
        super().__init__()
        self.total_tokens = n_num_features + len(cat_cardinalities)
        self.use_dae = use_dae
        self.use_dynamic_residual = use_dynamic_residual and n_dyn_features > 0
        self.dynamic_enabled = self.use_dynamic_residual

        if use_dae:
            self.num_embedding = DeviationAwareNumericEmbedding(n_num_features, embed_dim)
        else:
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

        if self.use_dynamic_residual:
            self.dynamic_head = DynamicResidualHead(n_dyn_features, embed_dim, dropout=dropout)
            # 初始化为很小的动态影响，避免破坏静态模型。
            self.dynamic_scale = nn.Parameter(torch.tensor(-4.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def set_deviation_stats(self, mu: torch.Tensor, sigma: torch.Tensor):
        if self.use_dae and hasattr(self.num_embedding, "set_stats"):
            self.num_embedding.set_stats(mu, sigma)

    def encode_static(self, x_num, x_cat, return_attention: bool = False):
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
        out_mean = h.mean(dim=1)
        out_max = h.max(dim=1).values
        out_feat = torch.cat([out_mean, out_max], dim=-1)
        return out_feat, out_mean, attention_maps

    def forward(self, x_num, x_cat, x_dyn=None, dyn_mask=None, return_attention: bool = False):
        out_feat, out_mean, attention_maps = self.encode_static(x_num, x_cat, return_attention=return_attention)
        static_logit = self.classifier(out_feat).squeeze(-1)

        logits = static_logit
        if self.use_dynamic_residual and self.dynamic_enabled and x_dyn is not None and dyn_mask is not None:
            dyn_logit = self.dynamic_head(out_mean, x_dyn, dyn_mask)
            scale = torch.sigmoid(self.dynamic_scale)
            logits = static_logit + scale * dyn_logit

        if return_attention:
            return logits, attention_maps
        return logits

    def freeze_static_train_dynamic_only(self):
        for p in self.parameters():
            p.requires_grad = False
        if self.use_dynamic_residual:
            for p in self.dynamic_head.parameters():
                p.requires_grad = True
            self.dynamic_scale.requires_grad = True
            self.dynamic_enabled = True

    def disable_dynamic(self):
        self.dynamic_enabled = False

    def enable_dynamic(self):
        if self.use_dynamic_residual:
            self.dynamic_enabled = True


# =============================================================
# 训练器
# =============================================================
class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu", pos_weight: Optional[torch.Tensor] = None):
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
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=grad_clip)
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
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=6, min_lr=min_lr
        )
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


# =============================================================
# 单实验 5 折
# =============================================================
def summarize_results(fold_results, threshold_key="threshold"):
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["threshold"] = {
        "mean": float(np.mean([r[threshold_key] for r in fold_results])),
        "std": float(np.std([r[threshold_key] for r in fold_results])),
    }
    return summary


def run_single_experiment(cfg: Config, exp: Dict, df, num_cols, cat_cols, y,
                          x_dyn_all=None, dyn_mask_all=None, exp_output_dir=None):
    use_dae = exp["use_dae"]
    use_dynamic_residual = exp["use_dynamic_residual"]

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_results_fixed = []
    fold_results_best = []
    fold_summary_rows = []

    exp_output_dir = Path(exp_output_dir)
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    if x_dyn_all is None:
        x_dyn_all = np.full((len(df), 0, 0), np.nan, dtype=np.float32)
    if dyn_mask_all is None:
        dyn_mask_all = np.zeros(len(df), dtype=np.float32)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\n[{exp['name']}] Fold {fold}/{cfg.n_splits}")
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

        x_train_dyn = x_dyn_all[train_idx].copy()
        x_val_dyn = x_dyn_all[val_idx].copy()
        train_dyn_mask = dyn_mask_all[train_idx].copy()
        val_dyn_mask = dyn_mask_all[val_idx].copy()

        if x_train_dyn.shape[1] > 0:
            x_train_dyn, x_val_dyn = preprocess_dynamic_by_fold(
                x_train_dyn, x_val_dyn, list(cfg.dyn_cols), num_cols, scaler
            )

        train_loader = DataLoader(
            CSFDataset(x_train_num, x_train_cat, x_train_dyn, train_dyn_mask, y_train),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            CSFDataset(x_val_num, x_val_cat, x_val_dyn, val_dyn_mask, y_val),
            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        pw_value = (neg_count / max(pos_count, 1)) * cfg.pos_weight_scale
        pos_weight = torch.tensor(pw_value, dtype=torch.float32).to(cfg.device)

        n_dyn_features = x_train_dyn.shape[1] if (use_dynamic_residual and x_train_dyn.shape[1] > 0) else 0
        model = LightAMFormer(
            n_num_features=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            top_k=cfg.top_k,
            dropout=cfg.dropout,
            ff_mult=cfg.ff_mult,
            use_dae=use_dae,
            use_dynamic_residual=use_dynamic_residual,
            n_dyn_features=n_dyn_features,
        )

        if use_dae:
            dev_stats = RobustDeviationStats().fit(x_train_num)
            mu_t, sigma_t = dev_stats.to_tensors(cfg.device)
            model.set_deviation_stats(mu_t, sigma_t)
        else:
            dev_stats = None

        trainer = ModelTrainer(model, device=cfg.device, pos_weight=pos_weight)

        # Phase 1: 静态训练。动态残差实验也先禁用动态，保证静态主干稳定。
        static_path = exp_output_dir / f"fold{fold}_phase1_static_best.pth"
        if use_dynamic_residual:
            trainer.model.disable_dynamic()
            print("  Phase 1: 训练静态 LightAMFormer 主干，不启用动态残差。")
        else:
            print("  训练静态模型。")

        trainer.train(
            train_loader, val_loader, str(static_path),
            epochs=cfg.epochs, lr=cfg.lr, patience=cfg.patience,
            weight_decay=cfg.weight_decay, min_lr=cfg.min_lr, grad_clip=cfg.grad_clip
        )

        # Phase 2: 动态残差头训练。冻结静态主干，只训练 dynamic_head + dynamic_scale。
        if use_dynamic_residual and n_dyn_features > 0 and train_dyn_mask.sum() > 0:
            trainer.model.load_state_dict(torch.load(static_path, map_location=cfg.device))
            trainer.model.freeze_static_train_dynamic_only()
            dyn_path = exp_output_dir / f"fold{fold}_phase2_dynamic_residual_best.pth"
            print(f"  Phase 2: 冻结静态主干，只训练动态残差头。训练动态覆盖率={train_dyn_mask.mean()*100:.1f}%, 验证动态覆盖率={val_dyn_mask.mean()*100:.1f}%")
            trainer.train(
                train_loader, val_loader, str(dyn_path),
                epochs=cfg.dynamic_epochs, lr=cfg.dynamic_lr, patience=cfg.dynamic_patience,
                weight_decay=cfg.dynamic_weight_decay, min_lr=cfg.min_lr, grad_clip=cfg.grad_clip
            )
        elif use_dynamic_residual:
            print("  ⚠️ 没有可用动态数据，本 fold 保持静态模型。")
            trainer.model.enable_dynamic()

        probs, labels = trainer.predict_probs(val_loader)
        fixed_results = evaluate_from_probs(labels, probs, cfg.fixed_threshold)
        fold_results_fixed.append(fixed_results)

        best_t = find_best_threshold(labels, probs, mode=cfg.threshold_mode, fn_cost=cfg.fn_cost, fp_cost=cfg.fp_cost) if cfg.optimize_threshold else cfg.fixed_threshold
        best_results = evaluate_from_probs(labels, probs, best_t)
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
            "mcc_fixed": fixed_results["mcc"],
            "accuracy": best_results["accuracy"],
            "precision": best_results["precision"],
            "recall": best_results["recall"],
            "f1": best_results["f1"],
            "mcc": best_results["mcc"],
            "tn": cm[0][0], "fp": cm[0][1], "fn": cm[1][0], "tp": cm[1][1],
            "dynamic_train_coverage": float(train_dyn_mask.mean()),
            "dynamic_val_coverage": float(val_dyn_mask.mean()),
        }
        if use_dynamic_residual and hasattr(trainer.model, "dynamic_scale"):
            fold_summary["dynamic_scale_sigmoid"] = float(torch.sigmoid(trainer.model.dynamic_scale.detach()).cpu().item())

        fold_summary_rows.append(fold_summary)
        print(f"  Fold {fold}: AUC={fold_summary['auc']:.4f}, BestT={best_t:.2f}, Recall={fold_summary['recall']:.4f}, F1={fold_summary['f1']:.4f}, MCC={fold_summary['mcc']:.4f}")

        with open(exp_output_dir / f"fold_{fold}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_summary, f, ensure_ascii=False, indent=2)
        pd.DataFrame({
            "y_true": labels,
            "y_prob": probs,
            "fixed_threshold": cfg.fixed_threshold,
            "best_threshold": best_t,
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
        if dev_stats is not None:
            with open(exp_output_dir / f"fold_{fold}_dev_stats.pkl", "wb") as f:
                pickle.dump(dev_stats, f)

    fixed_summary = summarize_results(fold_results_fixed, threshold_key="threshold")
    best_summary = summarize_results(fold_results_best, threshold_key="threshold")

    pd.DataFrame(fold_summary_rows).to_csv(exp_output_dir / "all_folds_metrics.csv", index=False, encoding="utf-8-sig")
    with open(exp_output_dir / "summary_metrics_best_threshold.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)
    with open(exp_output_dir / "summary_metrics_fixed_0_5.json", "w", encoding="utf-8") as f:
        json.dump(fixed_summary, f, ensure_ascii=False, indent=2)
    with open(exp_output_dir / "experiment_config.json", "w", encoding="utf-8") as f:
        json.dump({"cfg": asdict(cfg), "experiment": exp}, f, ensure_ascii=False, indent=2)

    return best_summary, fixed_summary, fold_summary_rows


# =============================================================
# 主函数
# =============================================================
def main():
    set_seed(CFG.seed)
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载静态数据...")
    raw_df = pd.read_excel(CFG.data_path)
    df, num_cols, cat_cols, y = build_feature_dataframe(raw_df, CFG)

    print("加载动态数据，只有 exp3 会使用...")
    x_dyn_all, dyn_mask_all, n_dyn_features = load_sparse_dynamic_data(raw_df, CFG)

    print("\n" + "=" * 80)
    print("固定最佳参数，不做网格搜索")
    print("=" * 80)
    print(f"样本数: {len(df)}")
    print(f"阳性样本数: {int(y.sum())}")
    print(f"数值特征数: {len(num_cols)}")
    print(f"类别特征数: {len(cat_cols)}")
    print(f"动态特征数: {n_dyn_features}; 动态覆盖率: {dyn_mask_all.mean()*100:.1f}%")
    print(f"设备: {CFG.device}")
    print(f"固定参数: top_k={CFG.top_k}, pos_weight_scale={CFG.pos_weight_scale}, embed_dim={CFG.embed_dim}, dropout={CFG.dropout}, lr={CFG.lr}")
    print("数值特征:", num_cols)
    print("类别特征:", cat_cols)

    all_rows = []
    for exp in EXPERIMENTS:
        print("\n" + "#" * 80)
        print(f"开始实验: {exp['name']}")
        print(exp["description"])
        print("#" * 80)
        exp_dir = output_dir / exp["name"]
        best_summary, fixed_summary, _ = run_single_experiment(
            CFG, exp, df, num_cols, cat_cols, y,
            x_dyn_all=x_dyn_all,
            dyn_mask_all=dyn_mask_all,
            exp_output_dir=exp_dir,
        )

        row = {
            "experiment": exp["name"],
            "description": exp["description"],
            "auc_mean": best_summary["auc"]["mean"],
            "auc_std": best_summary["auc"]["std"],
            "recall_mean": best_summary["recall"]["mean"],
            "precision_mean": best_summary["precision"]["mean"],
            "f1_mean": best_summary["f1"]["mean"],
            "mcc_mean": best_summary["mcc"]["mean"],
            "threshold_mean": best_summary["threshold"]["mean"],
            "fixed_auc_mean": fixed_summary["auc"]["mean"],
            "fixed_recall_mean": fixed_summary["recall"]["mean"],
            "fixed_f1_mean": fixed_summary["f1"]["mean"],
            "fixed_mcc_mean": fixed_summary["mcc"]["mean"],
        }
        all_rows.append(row)

        print("\n" + "=" * 60)
        print(f"{exp['name']} 最佳阈值结果")
        print("=" * 60)
        for key, stat in best_summary.items():
            print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")
        print("\n" + "=" * 60)
        print(f"{exp['name']} 固定 0.5 阈值结果")
        print("=" * 60)
        for key, stat in fixed_summary.items():
            print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    results_df = pd.DataFrame(all_rows).sort_values(["auc_mean", "recall_mean", "f1_mean"], ascending=[False, False, False])
    results_df.to_csv(output_dir / "three_experiments_summary.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "three_experiments_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print("\n" + "#" * 80)
    print("三个固定实验完成，汇总如下：")
    print("#" * 80)
    print(results_df[["experiment", "auc_mean", "auc_std", "recall_mean", "f1_mean", "mcc_mean", "threshold_mean"]].to_string(index=False))
    print(f"\n全部结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
