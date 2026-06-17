"""
DA-CTFormer v2: Deviation-Aware Clinical Tabular Transformer with Trend Encoding
================================================================================
基于 AMFormer 改造，针对临床表格数据三个特性：
  1. 参考范围依赖（Deviation-Aware Embedding, DAE）
  2. 成对偏差算术交互（Pairwise Deviation Arithmetic, PDA）
  3. 短时序趋势（Trend Token Encoder, TTE）

v2 修复：
  [Fix 1] PDA 现在覆盖全部 token（数值+类别）。类别 token 没有 mu/sigma，
          直接用其原始嵌入作为"伪偏差"参与配对——这样不会丢失类别信息。
  [Fix 2] DAE buffers 始终注册（即使 use_dae=False），保证消融加载兼容。
  [Fix 3] Mixup 改为在输入空间（x_num + x_dyn 联动），代码精简清晰。
  [Fix 4] DAE 的 alpha 改为可学习向量，初始化为偏向 dev (logit=1.0)，
          让强信号特征自然偏向偏差表示。
  [Fix 5] TTE 改用 attention pooling，让模型自学"哪个特征的趋势最重要"，
          且这个 attention 权重可作为论文可解释性卖点。

核心消融开关：
  cfg.use_dae    = True/False
  cfg.use_pda    = True/False
  cfg.use_tte    = True/False
  cfg.use_focal  = True/False
  cfg.use_mixup  = True/False

论文叙事（与 AMFormer 的差异，三处独立可消融）：
  AMFormer  : raw token + add ICG + mul ICG (log space)
  DA-CTFormer: dae token + add ICG + PDA pair-wise (deviation space) + TTE
"""

import os
import json
import math
import pickle
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Optional, List, Dict

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
    matthews_corrcoef,
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


def safe_mcc(y_true, y_pred):
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return 0.0
    return matthews_corrcoef(y_true, y_pred)


# =============================================================
# 配置
# =============================================================
@dataclass
class Config:
    data_path: str = "/Users/wangqinyang.5/Desktop/Infection/original.xlsx"
    dynamic_data_path: Optional[str] = None
    output_dir: str = "best_dact"
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
    top_k: int = 8
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

    # ── 消融开关 ──
    use_dae: bool = True
    use_pda: bool = True
    use_tte: bool = True
    use_focal: bool = True
    use_mixup: bool = True

    mixup_alpha: float = 0.2
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    n_time_points: int = 4

    # PDA 相关
    top_k_pairs: int = 24   # PDA 中保留的成对算术对数


CFG = Config()

GRID_SEARCH_SPACE = {
    "top_k": [8, 12, 16],
    "pos_weight_scale": [0.9, 1.0, 1.1],
    "embed_dim": [64, 96],
}


# =============================================================
# 数据集
# =============================================================
class CSFDataset(Dataset):
    """
    支持静态 + 可选动态时序。
    无动态数据时，x_dynamic 用零填充，has_dyn=0 让 TTE 门控自动关闭。
    """
    def __init__(self, x_num, x_cat, labels=None, x_dynamic=None, n_time_points=4):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)

        # 统一规整动态数据形状: [N, F_num, T]
        if x_dynamic is None:
            # 全部样本无动态数据：用零张量
            self.x_dynamic = torch.zeros(len(x_num), x_num.shape[1], n_time_points, dtype=torch.float32)
            self.has_dyn = torch.zeros(len(x_num), dtype=torch.float32)
        else:
            self.x_dynamic = torch.tensor(x_dynamic, dtype=torch.float32)
            # 每个样本是否有有效动态数据：判断 std > 0（全零视为缺失）
            self.has_dyn = (self.x_dynamic.std(dim=(1, 2)) > 1e-6).float()

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.x_num[idx], self.x_cat[idx], self.x_dynamic[idx], self.has_dyn[idx]
        return self.x_num[idx], self.x_cat[idx], self.x_dynamic[idx], self.has_dyn[idx], self.labels[idx]


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


# =============================================================
# 偏差统计估计器（RobustScaler 风格）
# =============================================================
class RobustDeviationStats:
    def __init__(self):
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, eps: float = 1e-6):
        self.mu = np.median(x, axis=0)
        q75 = np.percentile(x, 75, axis=0)
        q25 = np.percentile(x, 25, axis=0)
        self.sigma = (q75 - q25) / 1.3489795 + eps
        # 防止零尺度
        self.sigma = np.where(self.sigma < eps, 1.0, self.sigma)
        return self

    def to_tensors(self, device):
        return (torch.tensor(self.mu, dtype=torch.float32, device=device),
                torch.tensor(self.sigma, dtype=torch.float32, device=device))


# =============================================================
# [Fix 4] 模块 1：Deviation-Aware Embedding (DAE)
# =============================================================
class DeviationAwareEmbedding(nn.Module):
    """
    每个数值特征产出: emb = (1-σ(α_i)) * W_raw_i·x_i + σ(α_i) * W_dev_i·z_i
    其中 z_i = (x_i - μ_i) / σ_i 为训练集 RobustScaler 风格的标准化偏差。

    [Fix 2] mu/sigma buffer 无论 use_dae 如何始终注册，保证 state_dict 兼容。
    [Fix 4] alpha 初始化为 1.0（即 σ(1)≈0.73, 偏向偏差表示），让强参考范围
            依赖的特征自然占优，弱依赖的特征学到压低 alpha。

    参数
    -----
    use_dae : 是否启用偏差分支。False 时 alpha 固定为 -∞（σ(-∞)=0），相当于
              纯原始嵌入 —— 这就是原始 AMFormer 的 NumericEmbedding。
    """
    def __init__(self, n_num_features: int, embed_dim: int, use_dae: bool = True):
        super().__init__()
        self.n = n_num_features
        self.embed_dim = embed_dim
        self.use_dae = use_dae

        # raw 嵌入分支（必有）
        self.w_raw = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_num_features)])

        # dev 嵌入分支（结构上始终构造，权重大小可被 alpha=−∞ 关闭）
        self.w_dev = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(n_num_features)])

        # 可学习融合权重，初始化为 1.0（默认偏向 dev）
        if use_dae:
            self.alpha = nn.Parameter(torch.ones(n_num_features))
        else:
            # 固定为大负数，sigmoid 输出≈0，dev 分支不起作用
            self.register_buffer("alpha", torch.full((n_num_features,), -10.0))

        # [Fix 2] 始终注册 mu/sigma buffer
        self.register_buffer("mu", torch.zeros(n_num_features))
        self.register_buffer("sigma", torch.ones(n_num_features))

    def set_stats(self, mu: torch.Tensor, sigma: torch.Tensor):
        """训练折拟合后调用，注入参考统计量。"""
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)

    def forward(self, x_num: torch.Tensor) -> Optional[torch.Tensor]:
        """
        x_num: [B, F]   返回: [B, F, embed_dim]
        """
        if x_num.size(1) == 0:
            return None

        # 向量化实现（避免 Python for-loop 慢）
        # raw: 把所有 W_raw 堆成 (F, 1, D) -> 用 einsum
        raw_w = torch.stack([m.weight for m in self.w_raw], dim=0)  # [F, D, 1]
        raw_b = torch.stack([m.bias   for m in self.w_raw], dim=0)  # [F, D]
        # x_num: [B, F] -> [B, F, 1]
        emb_raw = torch.einsum("bfi,fdi->bfd", x_num.unsqueeze(-1), raw_w) + raw_b.unsqueeze(0)

        dev_w = torch.stack([m.weight for m in self.w_dev], dim=0)  # [F, D, 1]
        dev_b = torch.stack([m.bias   for m in self.w_dev], dim=0)
        x_dev = (x_num - self.mu) / self.sigma                       # [B, F]
        emb_dev = torch.einsum("bfi,fdi->bfd", x_dev.unsqueeze(-1), dev_w) + dev_b.unsqueeze(0)

        a = torch.sigmoid(self.alpha).view(1, self.n, 1)             # [1, F, 1]
        return (1 - a) * emb_raw + a * emb_dev

    def get_deviation_tokens(self, x_num: torch.Tensor) -> Optional[torch.Tensor]:
        """单独返回偏差嵌入，供 PDA 使用。"""
        if x_num.size(1) == 0:
            return None
        dev_w = torch.stack([m.weight for m in self.w_dev], dim=0)
        dev_b = torch.stack([m.bias   for m in self.w_dev], dim=0)
        x_dev = (x_num - self.mu) / self.sigma
        return torch.einsum("bfi,fdi->bfd", x_dev.unsqueeze(-1), dev_w) + dev_b.unsqueeze(0)

    def get_alpha_weights(self) -> torch.Tensor:
        """暴露 σ(α) 给可解释性可视化。返回 [F]，每个特征对偏差信号的依赖度。"""
        return torch.sigmoid(self.alpha).detach()


# =============================================================
# [Fix 1] 模块 2：Pairwise Deviation Arithmetic (PDA)
# =============================================================
class PairwiseDeviationArithmetic(nn.Module):
    """
    在 Top-K 选出的特征对 (i, j) 上做成对算术：
        z_ij = Proj([ r_i ⊙ r_j ; r_i / (|r_j| + ε) ; r_i − r_j ])

    [Fix 1] PDA 现在作用于全部 token（数值 + 类别），不仅限于数值。类别 token
            没有显式偏差含义，但其嵌入向量本身可参与与数值偏差 token 的交互
            （比如"高 GCS × 低 CSF 葡萄糖"的乘性交互）。

    [创新点] 同时输出三种算子的：
        (a) concat 投影 路径（信息保真）
        (b) softmax 算子门控加权和路径（可解释）
      两路相加，充分利用算术信号。
    """
    def __init__(self, embed_dim: int, n_prompts: int, top_k_pairs: int,
                 dropout: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_prompts = n_prompts
        self.top_k_pairs = top_k_pairs
        self.eps = eps

        # 配对打分头（小型）
        self.pair_score = nn.Linear(2 * embed_dim, 1)

        # 三算子各自的轻量投影头
        self.proj_concat = nn.Linear(3 * embed_dim, embed_dim)

        # 算子门控：让模型为每个 pair 自学 (mul, div, diff) 的权重
        self.op_gate = nn.Linear(2 * embed_dim, 3)

        # k pair tokens → n_prompts tokens 的注意力池化（自适应聚合）
        self.agg_query = nn.Parameter(torch.randn(n_prompts, embed_dim) * 0.02)
        self.agg_proj = nn.Linear(embed_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, D]   输入的全部 token（数值+类别，数值 token 已经是偏差表示）
        返回:    [B, n_prompts, D]
        """
        B, N, D = tokens.shape
        if N < 2:
            # 极端情况：广播平均
            return tokens.mean(dim=1, keepdim=True).expand(B, self.n_prompts, D)

        # 1) 计算所有 (i, j) pair 的得分
        ti = tokens.unsqueeze(2).expand(B, N, N, D)   # [B, N, N, D]
        tj = tokens.unsqueeze(1).expand(B, N, N, D)
        pair_input = torch.cat([ti, tj], dim=-1)       # [B, N, N, 2D]
        scores = self.pair_score(pair_input).squeeze(-1)  # [B, N, N]

        # 排除对角线
        eye_mask = torch.eye(N, device=tokens.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(eye_mask, float("-inf"))

        # 2) Top-K 配对（在展平后的全 N*N 上选）
        K = min(self.top_k_pairs, N * (N - 1))
        scores_flat = scores.view(B, -1)
        topk = torch.topk(scores_flat, k=K, dim=-1)
        topk_idx = topk.indices                        # [B, K]
        # softmax 后的权重（用于后续聚合时的注意力加权）
        topk_w = F.softmax(topk.values, dim=-1)        # [B, K]

        row_idx = topk_idx // N
        col_idx = topk_idx % N
        b_idx = torch.arange(B, device=tokens.device).unsqueeze(1).expand(B, K)

        # 3) 取出选中的 (i, j) pair 的 token 表示
        sel_ri = tokens[b_idx, row_idx]                # [B, K, D]
        sel_rj = tokens[b_idx, col_idx]                # [B, K, D]

        # 4) 三种算术运算（element-wise）
        prod  = sel_ri * sel_rj
        ratio = sel_ri / (sel_rj.abs() + self.eps)
        diff  = sel_ri - sel_rj

        # 5a) concat 投影路径
        cat_feats = torch.cat([prod, ratio, diff], dim=-1)
        path_concat = F.gelu(self.proj_concat(cat_feats))           # [B, K, D]

        # 5b) 算子门控路径（可解释）
        gate_input = torch.cat([sel_ri, sel_rj], dim=-1)            # [B, K, 2D]
        op_w = F.softmax(self.op_gate(gate_input), dim=-1)          # [B, K, 3]
        path_gated = (op_w[..., 0:1] * prod
                    + op_w[..., 1:2] * ratio
                    + op_w[..., 2:3] * diff)                        # [B, K, D]

        # 两路相加
        pair_tokens = self.dropout(path_concat + path_gated)        # [B, K, D]

        # 6) 注意力池化：n_prompts 个 query 对 K 个 pair tokens 做 cross-attention
        Q = self.agg_query.unsqueeze(0).expand(B, -1, -1)           # [B, n_prompts, D]
        attn_logits = torch.matmul(Q, pair_tokens.transpose(-2, -1)) / math.sqrt(D)
        # 应用 pair 选择权重作为先验
        attn_logits = attn_logits + topk_w.unsqueeze(1).log()       # 加 log(prior)
        attn = F.softmax(attn_logits, dim=-1)                       # [B, n_prompts, K]
        aggregated = torch.matmul(attn, pair_tokens)                # [B, n_prompts, D]

        return self.norm(self.agg_proj(aggregated))


# =============================================================
# [Fix 5] 模块 3：Trend Token Encoder (TTE)
# =============================================================
class TrendTokenEncoder(nn.Module):
    """
    输入: x_dyn [B, F, T]
    对每个特征提取 5 个趋势统计量 → MLP → trend tokens
    [Fix 5] 用 attention pooling 而非 mean pooling，让模型自学
            "哪个特征的趋势对预测最重要"。这一权重作为论文可解释性卖点。

    最终通过门控融合到静态表征。
    """
    def __init__(self, n_num_features: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n = n_num_features
        self.embed_dim = embed_dim

        # 5 统计量 → embed_dim
        self.trend_mlp = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # [Fix 5] attention pooling: 用一个 query 对 F 个 trend tokens 做加权聚合
        self.pool_query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # 门控网络
        self.gate = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        # 缓存最近一次 forward 的特征级 attention 权重（可解释性用）
        self._last_attn = None

    def extract_trend_stats(self, x_dyn: torch.Tensor) -> torch.Tensor:
        """x_dyn: [B, F, T]  →  [B, F, 5]"""
        B, n_feat, T = x_dyn.shape
        t_mean = x_dyn.mean(dim=-1, keepdim=True)
        t_max  = x_dyn.max(dim=-1).values.unsqueeze(-1)
        t_min  = x_dyn.min(dim=-1).values.unsqueeze(-1)

        # 线性斜率
        t_pts = torch.linspace(0, 1, T, device=x_dyn.device)
        t_c   = t_pts - t_pts.mean()
        x_c   = x_dyn - x_dyn.mean(dim=-1, keepdim=True)
        slope = (x_c * t_c).sum(dim=-1, keepdim=True) / (t_c.pow(2).sum() + 1e-8)

        # 波动率（相邻差的标准差），T=1 时退化为 0
        if T > 1:
            diffs = x_dyn[..., 1:] - x_dyn[..., :-1]
            vol   = diffs.std(dim=-1, unbiased=False, keepdim=True)
        else:
            vol = torch.zeros_like(t_mean)

        return torch.cat([t_mean, t_max, t_min, slope, vol], dim=-1)

    def forward(self, h_stat: torch.Tensor, x_dyn: torch.Tensor,
                has_dyn: torch.Tensor) -> torch.Tensor:
        """
        h_stat  : [B, embed_dim]  静态分支全局表征
        x_dyn   : [B, F, T]
        has_dyn : [B]            0/1 mask
        返回    : [B, embed_dim]
        """
        stats = self.extract_trend_stats(x_dyn)               # [B, F, 5]
        trend_tokens = self.trend_mlp(stats)                  # [B, F, D]

        # [Fix 5] Attention pooling
        B, F_n, D = trend_tokens.shape
        q = self.pool_query.unsqueeze(0).expand(B, -1, -1)    # [B, 1, D]
        scores = torch.matmul(q, trend_tokens.transpose(-2, -1)) / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)                      # [B, 1, F]
        self._last_attn = attn.detach()
        h_dyn = torch.matmul(attn, trend_tokens).squeeze(1)   # [B, D]
        h_dyn = self.norm(h_dyn)

        # 门控融合
        g = torch.sigmoid(self.gate(h_stat))                  # [B, D]
        g = g * has_dyn.unsqueeze(-1)                         # 无动态数据 → g=0

        return (1 - g) * h_stat + g * h_dyn

    def get_feature_attention(self) -> Optional[torch.Tensor]:
        """暴露最近一次 forward 的特征级 attention，论文可视化用。"""
        return self._last_attn


# =============================================================
# 损失：FocalLoss
# =============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# =============================================================
# [Fix 3] Mixup（输入空间，统一处理静态+动态）
# =============================================================
def mixup_inputs(x_num, x_cat, x_dyn, has_dyn, y, alpha: float = 0.2):
    """
    在输入空间做 Mixup：
      x_num, x_dyn: 数值/动态张量做线性插值
      x_cat: 类别整数 不做插值（保持原样，实际只在数值/动态上 mix）
      y: 标签按 lam 软插值

    返回 mix 后的输入和 (y_a, y_b, lam) 用于损失计算
    """
    if alpha <= 0:
        return x_num, x_cat, x_dyn, has_dyn, y, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x_num.size(0), device=x_num.device)

    x_num_mix = lam * x_num + (1 - lam) * x_num[idx]
    x_dyn_mix = lam * x_dyn + (1 - lam) * x_dyn[idx]
    has_dyn_mix = torch.maximum(has_dyn, has_dyn[idx])  # 任一有动态就视为有
    y_a, y_b = y, y[idx]

    return x_num_mix, x_cat, x_dyn_mix, has_dyn_mix, y_a, y_b, lam


# =============================================================
# 类别嵌入
# =============================================================
class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, cat_cardinalities, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cardinalities])

    def forward(self, x_cat: torch.Tensor):
        if x_cat.size(1) == 0:
            return None
        return torch.stack([self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))], dim=1)


# =============================================================
# 加法 ICG（保留 AMFormer 原版）
# =============================================================
class PromptInteractionCandidateGenerator(nn.Module):
    def __init__(self, embed_dim, n_heads, n_prompts, top_k=8, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_prompts = n_prompts
        self.top_k = top_k
        self.prompt = nn.Parameter(torch.randn(n_prompts, embed_dim) * 0.02)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q = self.prompt.unsqueeze(0).expand(B, -1, -1)
        k = self.w_k(x); v = self.w_v(x)
        q = q.view(B, self.n_prompts, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if self.top_k is not None and 0 < self.top_k < N:
            topk_vals, _ = torch.topk(scores, k=self.top_k, dim=-1)
            threshold = topk_vals[..., -1:].expand_as(scores)
            scores = scores.masked_fill(scores < threshold, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, self.n_prompts, D)
        return self.w_o(out), attn


# =============================================================
# DA-CT 算术块（加法 ICG + PDA 或乘法 ICG）
# =============================================================
class DACTArithmeticBlock(nn.Module):
    """
    use_pda=True : 加法 ICG  +  PDA（创新点）
    use_pda=False: 加法 ICG  +  乘法 ICG（原始 AMFormer）

    [说明] 加法 ICG 始终在用，因此与原始 AMFormer 的差异点为：
           "乘法分支换成 PDA"，并不是"算术分支整体重写"。
           论文中需要明确这一点以避免审稿人质疑创新度。
    """
    def __init__(self, embed_dim, n_heads, n_prompts, top_k=8, dropout=0.1,
                 eps=1e-6, use_pda=True, top_k_pairs=24):
        super().__init__()
        self.use_pda = use_pda
        self.n_prompts = n_prompts
        self.eps = eps

        self.add_icg = PromptInteractionCandidateGenerator(embed_dim, n_heads, n_prompts, top_k, dropout)

        if use_pda:
            self.pda = PairwiseDeviationArithmetic(embed_dim, n_prompts, top_k_pairs, dropout, eps)
        else:
            self.mul_icg = PromptInteractionCandidateGenerator(embed_dim, n_heads, n_prompts, top_k, dropout)

        self.candidate_fusion = nn.Linear(2 * n_prompts, n_prompts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        [Fix 1] PDA 现在直接作用于 x（全部 token），不再需要 dev_tokens 参数。
                因为如果 use_dae=True，x 中数值 token 已经是偏差表示。
        """
        o_add, attn_add = self.add_icg(x)

        if self.use_pda:
            o_mul = self.pda(x)
            attn_mul = None
        else:
            x_log = torch.log(F.relu(x) + self.eps)
            o_mul_log, attn_mul = self.mul_icg(x_log)
            o_mul = torch.exp(torch.clamp(o_mul_log, min=-10.0, max=10.0))

        candidates = torch.cat([o_add, o_mul], dim=1)         # [B, 2Np, D]
        out = self.candidate_fusion(candidates.transpose(1, 2)).transpose(1, 2)
        return self.dropout(out), {"add": attn_add, "mul": attn_mul}


class DACTLayer(nn.Module):
    def __init__(self, embed_dim, n_heads, n_prompts, top_k=8, dropout=0.1,
                 ff_mult=4, use_pda=True, top_k_pairs=24):
        super().__init__()
        self.arithmetic = DACTArithmeticBlock(
            embed_dim, n_heads, n_prompts, top_k, dropout,
            use_pda=use_pda, top_k_pairs=top_k_pairs
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
        arith_out, attn = self.arithmetic(x)
        x = self.norm1(x + arith_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn


# =============================================================
# DA-CTFormer 主模型
# =============================================================
class DACTFormer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: List[int],
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        top_k: int = 8,
        dropout: float = 0.2,
        ff_mult: int = 4,
        use_dae: bool = True,
        use_pda: bool = True,
        use_tte: bool = True,
        n_time_points: int = 4,
        top_k_pairs: int = 24,
    ):
        super().__init__()
        self.total_tokens = n_num_features + len(cat_cardinalities)
        if self.total_tokens <= 0:
            raise ValueError("数值特征和类别特征不能同时为空。")

        self.n_prompts = self.total_tokens
        self.use_dae = use_dae
        self.use_pda = use_pda
        self.use_tte = use_tte
        self.n_num_features = n_num_features

        self.num_embedding = DeviationAwareEmbedding(n_num_features, embed_dim, use_dae=use_dae)
        self.cat_embedding = CategoricalFeatureEmbedding(cat_cardinalities, embed_dim)

        self.layers = nn.ModuleList([
            DACTLayer(
                embed_dim, n_heads, self.n_prompts,
                top_k=top_k, dropout=dropout, ff_mult=ff_mult,
                use_pda=use_pda, top_k_pairs=top_k_pairs
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        if use_tte:
            self.tte = TrendTokenEncoder(n_num_features, embed_dim, dropout=dropout)

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

    def forward(self, x_num, x_cat, x_dyn=None, has_dyn=None, return_attention=False):
        num_tok = self.num_embedding(x_num)
        cat_tok = self.cat_embedding(x_cat)

        if num_tok is not None and cat_tok is not None:
            h = torch.cat([num_tok, cat_tok], dim=1)
        elif num_tok is not None:
            h = num_tok
        else:
            h = cat_tok

        attn_maps = []
        for layer in self.layers:
            h, attn = layer(h)
            if return_attention:
                attn_maps.append({
                    "add": attn["add"].detach().cpu().numpy() if attn["add"] is not None else None,
                    "mul": attn["mul"].detach().cpu().numpy() if attn["mul"] is not None else None,
                })

        h = self.final_norm(h)
        out_mean = h.mean(dim=1)
        out_max, _ = h.max(dim=1)

        # TTE 融合（只对 mean 部分做融合，max 部分保留静态信号）
        if self.use_tte and x_dyn is not None and has_dyn is not None:
            out_mean = self.tte(out_mean, x_dyn, has_dyn)

        feat = torch.cat([out_mean, out_max], dim=-1)
        logits = self.classifier(feat).squeeze(-1)

        if return_attention:
            return logits, attn_maps
        return logits


# =============================================================
# 训练器
# =============================================================
class ModelTrainer:
    def __init__(self, model, device="cpu", pos_weight=None,
                 use_focal=True, use_mixup=True,
                 focal_alpha=0.25, focal_gamma=2.0, mixup_alpha=0.2):
        self.model = model.to(device)
        self.device = device
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.train_losses, self.val_losses, self.val_aucs = [], [], []

        if use_focal:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    @staticmethod
    def _unpack(batch):
        if len(batch) == 5:
            return batch
        else:
            return batch + (None,)

    def train_epoch(self, train_loader, optimizer, grad_clip=1.0):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            x_num, x_cat, x_dyn, has_dyn, y = self._unpack(batch)
            x_num   = x_num.to(self.device)
            x_cat   = x_cat.to(self.device)
            x_dyn   = x_dyn.to(self.device)
            has_dyn = has_dyn.to(self.device)
            y       = y.to(self.device)

            optimizer.zero_grad()

            # [Fix 3] Mixup 在输入空间统一处理
            if self.use_mixup and self.model.training:
                x_num_m, x_cat_m, x_dyn_m, has_dyn_m, y_a, y_b, lam = mixup_inputs(
                    x_num, x_cat, x_dyn, has_dyn, y, self.mixup_alpha
                )
                logits = self.model(x_num_m, x_cat_m, x_dyn_m, has_dyn_m)
                loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
            else:
                logits = self.model(x_num, x_cat, x_dyn, has_dyn)
                loss = self.criterion(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                x_num, x_cat, x_dyn, has_dyn, y = self._unpack(batch)
                x_num   = x_num.to(self.device)
                x_cat   = x_cat.to(self.device)
                x_dyn   = x_dyn.to(self.device)
                has_dyn = has_dyn.to(self.device)
                y       = y.to(self.device)

                logits = self.model(x_num, x_cat, x_dyn, has_dyn)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        probs  = np.array(all_probs)
        labels = np.array(all_labels)
        return total_loss / max(len(val_loader), 1), probs, labels, safe_auc(labels, probs)

    def train(self, train_loader, val_loader, save_path,
              epochs=100, lr=1e-3, patience=20, weight_decay=1e-2,
              min_lr=1e-5, grad_clip=1.0):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=6, min_lr=min_lr
        )

        best_val_auc, patience_counter = -1.0, 0
        torch.save(self.model.state_dict(), save_path)

        for epoch in range(epochs):
            tr_loss = self.train_epoch(train_loader, optimizer, grad_clip)
            va_loss, _, _, va_auc = self.validate(val_loader)

            self.train_losses.append(tr_loss)
            self.val_losses.append(va_loss)
            self.val_aucs.append(va_auc)
            scheduler.step(va_auc)

            if va_auc > best_val_auc + 1e-5:
                best_val_auc = va_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))

    def evaluate(self, val_loader, threshold=0.5):
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                x_num, x_cat, x_dyn, has_dyn, y = self._unpack(batch)
                x_num   = x_num.to(self.device)
                x_cat   = x_cat.to(self.device)
                x_dyn   = x_dyn.to(self.device)
                has_dyn = has_dyn.to(self.device)

                logits = self.model(x_num, x_cat, x_dyn, has_dyn)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(y.numpy())

        probs  = np.array(all_probs)
        labels = np.array(all_labels)
        preds  = (probs >= threshold).astype(int)
        cm = confusion_matrix(labels, preds)

        return {
            "predictions": probs.tolist(),
            "labels": labels.tolist(),
            "threshold": threshold,
            "accuracy":  accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "f1":        f1_score(labels, preds, zero_division=0),
            "auc":       safe_auc(labels, probs),
            "mcc":       safe_mcc(labels, preds),
            "confusion_matrix": cm.tolist(),
        }


# =============================================================
# 单组实验
# =============================================================
def run_single_experiment(cfg: Config, df, num_cols, cat_cols, y,
                          exp_output_dir=None, save_fold_files=False,
                          dynamic_array=None):
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_results, fold_summary_rows = [], []

    if exp_output_dir is not None:
        exp_output_dir = Path(exp_output_dir)
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        df_train = df.iloc[train_idx].copy()
        df_val   = df.iloc[val_idx].copy()
        y_train  = y[train_idx]
        y_val    = y[val_idx]

        # 数值标准化
        scaler = StandardScaler()
        if len(num_cols) > 0:
            x_train_num = scaler.fit_transform(df_train[num_cols].values.astype(np.float32))
            x_val_num   = scaler.transform(df_val[num_cols].values.astype(np.float32))
        else:
            x_train_num = np.zeros((len(df_train), 0), dtype=np.float32)
            x_val_num   = np.zeros((len(df_val),   0), dtype=np.float32)

        # 类别编码
        cat_encoder = CategoryEncoder().fit(df_train, cat_cols)
        x_train_cat = cat_encoder.transform(df_train, cat_cols)
        x_val_cat   = cat_encoder.transform(df_val,   cat_cols)
        cat_cardinalities = cat_encoder.get_cardinalities(cat_cols)

        # 偏差统计（在标准化后空间拟合）
        dev_stats = None
        if cfg.use_dae and len(num_cols) > 0:
            dev_stats = RobustDeviationStats().fit(x_train_num)

        # 动态数据切片（同样用 StandardScaler 标准化每个特征）
        if dynamic_array is not None:
            dyn_train = dynamic_array[train_idx].copy()
            dyn_val   = dynamic_array[val_idx].copy()
            # 用静态 scaler 的 mean/std 同时标准化动态数据（保持尺度一致）
            for fi in range(len(num_cols)):
                m = scaler.mean_[fi]
                s = scaler.scale_[fi]
                dyn_train[:, fi, :] = (dyn_train[:, fi, :] - m) / (s + 1e-6)
                dyn_val[:,   fi, :] = (dyn_val[:,   fi, :] - m) / (s + 1e-6)
        else:
            dyn_train = None
            dyn_val   = None

        train_loader = DataLoader(
            CSFDataset(x_train_num, x_train_cat, y_train, x_dynamic=dyn_train, n_time_points=cfg.n_time_points),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            CSFDataset(x_val_num, x_val_cat, y_val, x_dynamic=dyn_val, n_time_points=cfg.n_time_points),
            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        # 类别不平衡权重
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        pw_value = (neg / max(pos, 1)) * cfg.pos_weight_scale
        pos_weight = torch.tensor(pw_value, dtype=torch.float32).to(cfg.device)

        # 构建模型
        model = DACTFormer(
            n_num_features=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            embed_dim=cfg.embed_dim,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            top_k=cfg.top_k,
            dropout=cfg.dropout,
            ff_mult=cfg.ff_mult,
            use_dae=cfg.use_dae,
            use_pda=cfg.use_pda,
            use_tte=cfg.use_tte and (dynamic_array is not None),
            n_time_points=cfg.n_time_points,
            top_k_pairs=cfg.top_k_pairs,
        )

        # 注入偏差统计
        if cfg.use_dae and dev_stats is not None:
            mu_t, sigma_t = dev_stats.to_tensors(cfg.device)
            model.num_embedding.set_stats(mu_t, sigma_t)

        trainer = ModelTrainer(
            model, device=cfg.device, pos_weight=pos_weight,
            use_focal=cfg.use_focal, use_mixup=cfg.use_mixup,
            focal_alpha=cfg.focal_alpha, focal_gamma=cfg.focal_gamma,
            mixup_alpha=cfg.mixup_alpha,
        )

        model_path = (exp_output_dir / f"best_model_fold{fold}.pth") if exp_output_dir else Path(f"temp_best_model_fold{fold}.pth")

        trainer.train(
            train_loader=train_loader, val_loader=val_loader,
            save_path=str(model_path),
            epochs=cfg.epochs, lr=cfg.lr, patience=cfg.patience,
            weight_decay=cfg.weight_decay, min_lr=cfg.min_lr, grad_clip=cfg.grad_clip,
        )

        results = trainer.evaluate(val_loader, threshold=cfg.fixed_threshold)
        fold_results.append(results)

        fold_summary_rows.append({
            "fold": fold,
            **{k: results[k] for k in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]},
            "threshold": cfg.fixed_threshold,
            "tn": results["confusion_matrix"][0][0],
            "fp": results["confusion_matrix"][0][1],
            "fn": results["confusion_matrix"][1][0],
            "tp": results["confusion_matrix"][1][1],
        })

        if save_fold_files and exp_output_dir is not None:
            with open(exp_output_dir / f"fold_{fold}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(fold_summary_rows[-1], f, ensure_ascii=False, indent=2)

            pd.DataFrame({
                "y_true": results["labels"],
                "y_prob": results["predictions"],
                "threshold": results["threshold"],
            }).to_csv(exp_output_dir / f"fold_{fold}_predictions.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame({
                "epoch":      np.arange(len(trainer.train_losses)),
                "train_loss": trainer.train_losses,
                "val_loss":   trainer.val_losses,
                "val_auc":    trainer.val_aucs,
            }).to_csv(exp_output_dir / f"fold_{fold}_learning_curve.csv", index=False, encoding="utf-8-sig")

            with open(exp_output_dir / f"fold_{fold}_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open(exp_output_dir / f"fold_{fold}_cat_encoder.pkl", "wb") as f:
                pickle.dump(cat_encoder, f)
            if dev_stats is not None:
                with open(exp_output_dir / f"fold_{fold}_dev_stats.pkl", "wb") as f:
                    pickle.dump(dev_stats, f)

    # 汇总
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["threshold"] = cfg.fixed_threshold

    return summary, fold_summary_rows


# =============================================================
# 网格搜索
# =============================================================
def grid_search(cfg: Config, df, num_cols, cat_cols, y, dynamic_array=None):
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
        print(f"[{i}/{len(all_combinations)}] {params}")
        print("=" * 80)

        summary, _ = run_single_experiment(
            cfg=exp_cfg, df=df, num_cols=num_cols, cat_cols=cat_cols, y=y,
            dynamic_array=dynamic_array, exp_output_dir=None, save_fold_files=False,
        )

        row = {
            "exp_name": exp_name,
            **params,
            "auc_mean":  summary["auc"]["mean"],
            "auc_std":   summary["auc"]["std"],
            "f1_mean":   summary["f1"]["mean"],
            "recall_mean": summary["recall"]["mean"],
            "precision_mean": summary["precision"]["mean"],
            "mcc_mean":  summary["mcc"]["mean"],
        }
        search_results.append(row)
        print(f"AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f} | F1={row['f1_mean']:.4f} | Recall={row['recall_mean']:.4f} | MCC={row['mcc_mean']:.4f}")

        pd.DataFrame(search_results).sort_values(
            ["auc_mean", "auc_std"], ascending=[False, True]
        ).to_csv(search_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")

    results_df = pd.DataFrame(search_results).sort_values(
        ["auc_mean", "auc_std"], ascending=[False, True]
    ).reset_index(drop=True)
    best_row = results_df.iloc[0].to_dict()

    print("\n最佳参数：", best_row)
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

    dynamic_array = None
    if CFG.dynamic_data_path is not None and CFG.use_tte:
        print("加载动态数据...")
        dyn_df = pd.read_csv(CFG.dynamic_data_path)
        T = CFG.n_time_points
        F_n = len(num_cols)
        dynamic_array = np.zeros((len(df), F_n, T), dtype=np.float32)
        for fi, feat in enumerate(num_cols):
            for ti in range(T):
                col = f"{feat}_t{ti}"
                if col in dyn_df.columns:
                    dynamic_array[:, fi, ti] = pd.to_numeric(dyn_df[col], errors="coerce").fillna(0).values

    print(f"样本数: {len(df)}, 阳性: {int(y.sum())}")
    print(f"数值特征: {len(num_cols)}, 类别特征: {len(cat_cols)}")
    print(f"消融开关: DAE={CFG.use_dae}, PDA={CFG.use_pda}, TTE={CFG.use_tte}, "
          f"Focal={CFG.use_focal}, Mixup={CFG.use_mixup}")
    print(f"设备: {CFG.device}")

    best_row, _ = grid_search(CFG, df, num_cols, cat_cols, y, dynamic_array)

    best_cfg = deepcopy(CFG)
    best_cfg.top_k            = int(best_row["top_k"])
    best_cfg.pos_weight_scale = float(best_row["pos_weight_scale"])
    best_cfg.embed_dim        = int(best_row["embed_dim"])

    with open(output_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump({k: best_row[k] for k in ["top_k", "pos_weight_scale", "embed_dim",
                                            "auc_mean", "auc_std", "f1_mean", "mcc_mean"]},
                  f, ensure_ascii=False, indent=2)

    if best_cfg.rerun_best_after_search:
        print("\n用最佳参数完整跑 5 折...\n")
        best_run_dir = output_dir / "best_run_detailed"

        summary, fold_summary_rows = run_single_experiment(
            cfg=best_cfg, df=df, num_cols=num_cols, cat_cols=cat_cols, y=y,
            dynamic_array=dynamic_array,
            exp_output_dir=best_run_dir, save_fold_files=True,
        )

        pd.DataFrame(fold_summary_rows).to_csv(
            best_run_dir / "all_folds_metrics.csv", index=False, encoding="utf-8-sig"
        )
        with open(best_run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(best_run_dir / "best_run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(best_cfg), f, ensure_ascii=False, indent=2)

        print("=" * 50)
        print("DA-CTFormer 最终结果")
        print("=" * 50)
        for key, stat in summary.items():
            if key == "threshold":
                continue
            print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    print(f"\n全部结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()