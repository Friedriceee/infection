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
    output_dir: str = "amformer_v3_results"
    label_col: str = "outcome"

    batch_size: int = 32
    epochs: int = 120
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_epochs: int = 10         # Warmup 预热轮数
    patience: int = 25
    fixed_threshold: float = 0.5

    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    top_k: int | None = 5           # 25个特征建议 top_k=5，比之前更严格
    dropout: float = 0.20
    ff_mult: int = 4

    # Focal Loss 参数
    focal_alpha: float = 0.25       # 负样本权重（正样本 = 1-alpha = 0.75）
    focal_gamma: float = 2.0        # 难样本聚焦强度

    seed: int = 42
    n_splits: int = 5
    pos_weight_scale: float = 1.0
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    rerun_best_after_search: bool = True


CFG = Config()

# =============================================================
# 网格搜索空间（搜真正有影响的参数）
# =============================================================
GRID_SEARCH_SPACE = {
    "dropout":      [0.10, 0.20, 0.30],
    "n_layers":     [1, 2, 3],
    "focal_gamma":  [1.0, 2.0, 3.0],
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

    cat_cols = ["sex", "tube", "site", "other_inf", "transparency"]
    base_num_cols = [
        "age", "C_G", "C_WBC", "C_RBC", "C_P", "C_N",
        "GCS", "tem", "B_G", "B_CRP", "B_WBC", "B_N",
        "B_Lym", "B_PCT", "B_AC", "B_RBC",
    ]

    cat_cols      = [c for c in cat_cols      if c in df.columns]
    base_num_cols = [c for c in base_num_cols if c in df.columns]

    for c in base_num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    log_cols = ["C_WBC", "C_RBC", "C_P", "B_CRP", "B_WBC", "B_PCT", "B_AC", "B_RBC"]
    for col in log_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
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

    def fit(self, df, cat_cols):
        self.maps = {}
        for c in cat_cols:
            values = df[c].astype(str).fillna("Unknown").tolist()
            uniq = sorted(set(values))
            self.maps[c] = {v: i + 1 for i, v in enumerate(uniq)}  # 0 保留给未知
        return self

    def transform(self, df, cat_cols):
        out = []
        for c in cat_cols:
            mapper = self.maps[c]
            vals = df[c].astype(str).fillna("Unknown").tolist()
            out.append([mapper.get(v, 0) for v in vals])
        if not out:
            return np.zeros((len(df), 0), dtype=np.int64)
        return np.array(out, dtype=np.int64).T

    def get_cardinalities(self, cat_cols):
        return [max(self.maps[c].values(), default=0) + 1 for c in cat_cols]


# =============================================================
# 优化1：Focal Loss（难样本聚焦）
# =============================================================
class FocalLoss(nn.Module):
    """
    alpha：正样本基础权重（0.25 表示正样本权重 0.75，负样本 0.25，
           适合正样本稀少的场景；如果正负比已用 pos_weight 处理，alpha 设 0.5 即对称）。
    gamma：聚焦强度，gamma=0 退化为普通 BCE，gamma=2 是常用值。
    pos_weight：与 BCEWithLogitsLoss 的 pos_weight 语义相同，叠加在 focal 之上。
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss    = alpha_t * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# =============================================================
# 模型模块
# =============================================================
class TopKSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, top_k=None, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.top_k   = top_k

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
            topk_vals, _ = torch.topk(scores, min(self.top_k, L), dim=-1)
            threshold = topk_vals[..., -1:].expand_as(scores)
            scores = scores.masked_fill(scores < threshold, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        return self.w_o(context), attn_weights


class GatedArithmeticBlock(nn.Module):
    """
    三路算术交互（加/乘/减）+ 门控输出。
    gate 根据全局均值动态决定交互结果的保留比例，
    避免 Arithmetic Block 在训练初期破坏梯度。
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 3, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)

        add_feat = h + h_mean
        mul_feat = h * h_mean
        sub_feat = h - h_mean

        combined = torch.cat([add_feat, mul_feat, sub_feat], dim=-1)
        out  = self.dropout(self.proj(combined))
        gate = self.gate(h_mean)           # 门控：由全局均值决定保留比例
        return self.norm(h + gate * out)


# 优化2：注意力加权聚合（替代硬 Mean/Max）
class GlobalAttentionPooling(nn.Module):
    """
    对应图中 Weighted Sum。
    让模型自动学习哪些 token（特征）对最终诊断贡献更大，
    比 Mean/Max 更有表达力，且权重可解释（可视化哪些特征被重点关注）。
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, n_feat, embed_dim)
        attn_w  = F.softmax(self.attn(h), dim=1)   # (batch, n_feat, 1)
        pooled  = torch.sum(h * attn_w, dim=1)      # (batch, embed_dim)
        return pooled


class NumericFeatureEmbedding(nn.Module):
    def __init__(self, n_num: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_num)
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
            nn.Embedding(c, embed_dim) for c in cat_cardinalities
        ])

    def forward(self, x_cat: torch.Tensor):
        if x_cat.size(1) == 0:
            return None
        tokens = [self.embeddings[i](x_cat[:, i]) for i in range(len(self.embeddings))]
        return torch.stack(tokens, dim=1)


# =============================================================
# 主模型：AMFormer V3
# =============================================================
class AMFormerV3(nn.Module):
    """
    架构流向：
    1. 输入层：数值型 PerFeature Linear，类别型 Embedding。
    2. 交互层：Top-K 稀疏注意力 + GatedArithmeticBlock（门控三路交互）。
    3. 聚合层：GlobalAttentionPooling（Weighted Sum）+ LinearShortcut（原始特征直连）。
    4. 输出层：拼接后 MLP 分类。
    """
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        top_k: int | None = 5,
        dropout: float = 0.2,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        total_raw_dim = n_num_features + len(cat_cardinalities)

        # ── Embedding 层 ──
        self.num_embedding = NumericFeatureEmbedding(n_num_features, embed_dim)
        self.cat_embedding  = CategoricalFeatureEmbedding(cat_cardinalities, embed_dim)

        total_tokens = n_num_features + len(cat_cardinalities)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max(1, total_tokens), embed_dim) * 0.01
        )

        # ── Transformer 层 ──
        self.attn_layers  = nn.ModuleList([
            TopKSparseAttention(embed_dim, n_heads, top_k=top_k, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ffn_layers   = nn.ModuleList([
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

        # ── 优化2：注意力聚合（Weighted Sum）──
        self.global_attn_pool = GlobalAttentionPooling(embed_dim)

        # ── 优化3：线性跳连（原始数值特征直接到分类器）──
        # 只对数值特征做跳连（类别特征是整数，直接线性映射无意义）
        self.shortcut = nn.Sequential(
            nn.Linear(n_num_features, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        # ── 分类器：Weighted Pool + Max Pool + Shortcut 三路拼接 ──
        # 输入维度 = embed_dim (attn_pool) + embed_dim (max) + embed_dim (shortcut) = 3 * embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor,
                return_attention: bool = False):
        # 1. Embedding
        num_tokens = self.num_embedding(x_num)
        cat_tokens  = self.cat_embedding(x_cat)

        if num_tokens is not None and cat_tokens is not None:
            h = torch.cat([num_tokens, cat_tokens], dim=1)
        elif num_tokens is not None:
            h = num_tokens
        else:
            h = cat_tokens

        h = h + self.pos_encoding[:, :h.size(1), :]

        attention_maps = []

        # 2. 逐层：Sparse Attention → FFN → GatedArithmetic
        for attn, ffn, arith, norm1, norm2 in zip(
            self.attn_layers, self.ffn_layers, self.arith_blocks,
            self.norm1_layers, self.norm2_layers
        ):
            attn_out, attn_w = attn(h)
            attention_maps.append(attn_w.detach().cpu().numpy())
            h = norm1(h + attn_out)
            h = norm2(h + ffn(h))
            h = arith(h)

        # 3. 聚合：注意力加权池化 + Max 池化 + 原始特征跳连
        pool_attn  = self.global_attn_pool(h)          # (batch, embed_dim)
        pool_max, _ = h.max(dim=1)                      # (batch, embed_dim)
        shortcut   = self.shortcut(x_num)               # (batch, embed_dim)

        out_feat = torch.cat([pool_attn, pool_max, shortcut], dim=-1)  # (batch, 3*embed_dim)

        logits = self.classifier(out_feat).squeeze(-1)

        if return_attention:
            return logits, attention_maps
        return logits


# =============================================================
# Warmup + CosineAnnealing 调度器
# =============================================================
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    前 warmup_epochs 轮线性升温，之后余弦衰减到 min_lr。
    小样本下 Warmup 防止 ArithmeticBlock 参数在训练开始时被大梯度冲飞。
    """
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-5, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        lrs   = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                lr = base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
            else:
                progress = (epoch - self.warmup_epochs) / max(
                    self.total_epochs - self.warmup_epochs, 1
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress)
                )
            lrs.append(lr)
        return lrs


# =============================================================
# 训练器
# =============================================================
class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu",
                 pos_weight: torch.Tensor | None = None,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0):
        self.model       = model.to(device)
        self.device      = device
        self.pos_weight  = pos_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.val_aucs:     list[float] = []

    def train_epoch(self, train_loader, optimizer, criterion, grad_clip=1.0):
        self.model.train()
        total_loss = 0.0
        for batch_num, batch_cat, batch_labels in tqdm(train_loader, desc="Training", leave=False):
            batch_num    = batch_num.to(self.device)
            batch_cat    = batch_cat.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(batch_num, batch_cat)
            loss   = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch_num, batch_cat, batch_labels in val_loader:
                batch_num    = batch_num.to(self.device)
                batch_cat    = batch_cat.to(self.device)
                batch_labels = batch_labels.to(self.device)
                logits = self.model(batch_num, batch_cat)
                total_loss += criterion(logits, batch_labels).item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        all_probs  = np.array(all_probs)
        all_labels = np.array(all_labels)
        return total_loss / max(len(val_loader), 1), all_probs, all_labels, safe_auc(all_labels, all_probs)

    def train(self, train_loader, val_loader, save_path,
              epochs=120, lr=3e-4, patience=25, weight_decay=1e-2,
              min_lr=1e-5, grad_clip=1.0, warmup_epochs=10):

        # ── 优化1：Focal Loss ──
        criterion = FocalLoss(
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            pos_weight=self.pos_weight,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # ── Warmup + CosineAnnealing ──
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=epochs,
            min_lr=min_lr,
        )

        best_val_auc    = -1.0
        patience_counter = 0
        torch.save(self.model.state_dict(), save_path)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion, grad_clip)
            val_loss, _, _, val_auc = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            scheduler.step()

            if val_auc > best_val_auc + 1e-5:
                best_val_auc     = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  Epoch {epoch:03d}: Train={train_loss:.4f}  "
                      f"Val={val_loss:.4f}  AUC={val_auc:.4f}  LR={current_lr:.2e}")

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}  (best AUC={best_val_auc:.4f})")
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))

    def evaluate(self, val_loader, threshold: float = 0.5) -> dict:
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch_num, batch_cat, batch_labels in val_loader:
                logits = self.model(batch_num.to(self.device), batch_cat.to(self.device))
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        probs  = np.array(all_probs)
        labels = np.array(all_labels)
        preds  = (probs >= threshold).astype(int)
        cm     = confusion_matrix(labels, preds)

        return {
            "predictions":    probs.tolist(),
            "labels":         labels.tolist(),
            "threshold":      threshold,
            "accuracy":       accuracy_score(labels, preds),
            "precision":      precision_score(labels, preds, zero_division=0),
            "recall":         recall_score(labels, preds, zero_division=0),
            "f1":             f1_score(labels, preds, zero_division=0),
            "auc":            safe_auc(labels, probs),
            "confusion_matrix": cm.tolist(),
        }


# =============================================================
# 汇总
# =============================================================
def summarize_results(fold_results, threshold):
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["threshold"] = threshold
    return summary


# =============================================================
# 单组实验（5折）
# =============================================================
def run_single_experiment(cfg: Config, df, num_cols, cat_cols, y,
                          exp_output_dir=None, save_fold_files=False):
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    fold_results, fold_summary_rows = [], []

    if exp_output_dir is not None:
        exp_output_dir = Path(exp_output_dir)
        exp_output_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\n{'='*20} Fold {fold}/{cfg.n_splits} {'='*20}")
        df_train, df_val = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()
        y_train, y_val   = y[train_idx], y[val_idx]

        # 数值特征标准化
        scaler = StandardScaler()
        x_train_num = scaler.fit_transform(
            df_train[num_cols].values.astype(np.float32)
        ) if num_cols else np.zeros((len(df_train), 0), dtype=np.float32)
        x_val_num = scaler.transform(
            df_val[num_cols].values.astype(np.float32)
        ) if num_cols else np.zeros((len(df_val), 0), dtype=np.float32)

        # 类别特征编码
        cat_encoder     = CategoryEncoder().fit(df_train, cat_cols)
        x_train_cat     = cat_encoder.transform(df_train, cat_cols)
        x_val_cat       = cat_encoder.transform(df_val, cat_cols)
        cat_cardinalities = cat_encoder.get_cardinalities(cat_cols)

        train_loader = DataLoader(
            CSFDataset(x_train_num, x_train_cat, y_train),
            batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
        val_loader = DataLoader(
            CSFDataset(x_val_num, x_val_cat, y_val),
            batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
        )

        # pos_weight 基于原始训练集分布
        pos_count  = int(y_train.sum())
        neg_count  = int(len(y_train) - pos_count)
        pw_value   = (neg_count / max(pos_count, 1)) * cfg.pos_weight_scale
        pos_weight = torch.tensor(pw_value, dtype=torch.float32).to(cfg.device)
        print(f"  pos/neg = {pos_count}/{neg_count}  pos_weight = {pw_value:.2f}")

        model = AMFormerV3(
            n_num_features    = len(num_cols),
            cat_cardinalities = cat_cardinalities,
            embed_dim  = cfg.embed_dim,
            n_heads    = cfg.n_heads,
            n_layers   = cfg.n_layers,
            top_k      = cfg.top_k,
            dropout    = cfg.dropout,
            ff_mult    = cfg.ff_mult,
        )

        trainer = ModelTrainer(
            model,
            device       = cfg.device,
            pos_weight   = pos_weight,
            focal_alpha  = cfg.focal_alpha,
            focal_gamma  = cfg.focal_gamma,
        )

        model_path = (
            exp_output_dir / f"best_model_fold{fold}.pth"
            if exp_output_dir else Path(f"temp_model_fold{fold}.pth")
        )

        trainer.train(
            train_loader  = train_loader,
            val_loader    = val_loader,
            save_path     = str(model_path),
            epochs        = cfg.epochs,
            lr            = cfg.lr,
            patience      = cfg.patience,
            weight_decay  = cfg.weight_decay,
            min_lr        = cfg.min_lr,
            grad_clip     = cfg.grad_clip,
            warmup_epochs = cfg.warmup_epochs,
        )

        results = trainer.evaluate(val_loader, threshold=cfg.fixed_threshold)
        fold_results.append(results)

        fold_summary = {
            "fold":      fold,
            "accuracy":  results["accuracy"],
            "precision": results["precision"],
            "recall":    results["recall"],
            "f1":        results["f1"],
            "auc":       results["auc"],
            "threshold": cfg.fixed_threshold,
            "tn": results["confusion_matrix"][0][0],
            "fp": results["confusion_matrix"][0][1],
            "fn": results["confusion_matrix"][1][0],
            "tp": results["confusion_matrix"][1][1],
        }
        fold_summary_rows.append(fold_summary)
        print(f"  [Fold {fold}] AUC={results['auc']:.4f}  F1={results['f1']:.4f}  "
              f"Recall={results['recall']:.4f}  Precision={results['precision']:.4f}")
        print(f"  混淆矩阵: {np.array(results['confusion_matrix'])}")

        if save_fold_files and exp_output_dir:
            with open(exp_output_dir / f"fold_{fold}_metrics.json", "w", encoding="utf-8") as f:
                json.dump(fold_summary, f, ensure_ascii=False, indent=2)
            pd.DataFrame({
                "y_true":    results["labels"],
                "y_prob":    results["predictions"],
                "threshold": results["threshold"],
            }).to_csv(exp_output_dir / f"fold_{fold}_predictions.csv",
                      index=False, encoding="utf-8-sig")
            pd.DataFrame({
                "epoch":      np.arange(len(trainer.train_losses)),
                "train_loss": trainer.train_losses,
                "val_loss":   trainer.val_losses,
                "val_auc":    trainer.val_aucs,
            }).to_csv(exp_output_dir / f"fold_{fold}_learning_curve.csv",
                      index=False, encoding="utf-8-sig")
            with open(exp_output_dir / f"fold_{fold}_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            with open(exp_output_dir / f"fold_{fold}_cat_encoder.pkl", "wb") as f:
                pickle.dump(cat_encoder, f)

    summary = summarize_results(fold_results, cfg.fixed_threshold)
    return summary, fold_summary_rows


# =============================================================
# 网格搜索
# =============================================================
def grid_search(cfg: Config, df, num_cols, cat_cols, y):
    search_dir = Path(cfg.output_dir)
    search_dir.mkdir(parents=True, exist_ok=True)

    keys   = list(GRID_SEARCH_SPACE.keys())
    combos = list(product(*GRID_SEARCH_SPACE.values()))
    print(f"共 {len(combos)} 组参数需要搜索。\n")

    search_results = []

    for i, combo in enumerate(combos, start=1):
        params  = dict(zip(keys, combo))
        exp_cfg = deepcopy(cfg)
        for k, v in params.items():
            setattr(exp_cfg, k, v)

        exp_name = (
            f"exp_{i:02d}_drop_{params['dropout']}_"
            f"layers_{params['n_layers']}_gamma_{params['focal_gamma']}"
        )

        print("=" * 70)
        print(f"[{i}/{len(combos)}] 参数: {params}")
        print("=" * 70)

        summary, _ = run_single_experiment(
            cfg=exp_cfg, df=df, num_cols=num_cols, cat_cols=cat_cols, y=y,
            exp_output_dir=None, save_fold_files=False
        )

        row = {
            "exp_name":       exp_name,
            **params,
            "auc_mean":       summary["auc"]["mean"],
            "auc_std":        summary["auc"]["std"],
            "f1_mean":        summary["f1"]["mean"],
            "recall_mean":    summary["recall"]["mean"],
            "precision_mean": summary["precision"]["mean"],
            "accuracy_mean":  summary["accuracy"]["mean"],
        }
        search_results.append(row)

        print(f"  → AUC={row['auc_mean']:.4f} ± {row['auc_std']:.4f}  "
              f"F1={row['f1_mean']:.4f}  Recall={row['recall_mean']:.4f}\n")

        pd.DataFrame(search_results).sort_values(
            ["auc_mean", "auc_std"], ascending=[False, True]
        ).to_csv(search_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")

    results_df = pd.DataFrame(search_results).sort_values(
        ["auc_mean", "auc_std"], ascending=[False, True]
    ).reset_index(drop=True)

    results_df.to_csv(search_dir / "grid_search_results.csv", index=False, encoding="utf-8-sig")
    best_row = results_df.iloc[0].to_dict()

    print("\n" + "#" * 70)
    print("网格搜索完成，最佳参数：")
    for k in params.keys():
        print(f"  {k}: {best_row[k]}")
    print(f"  AUC: {best_row['auc_mean']:.4f} ± {best_row['auc_std']:.4f}")
    print("#" * 70 + "\n")

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

    print(f"样本数: {len(df)}  阳性: {int(y.sum())}")
    print(f"数值特征数: {len(num_cols)}  类别特征数: {len(cat_cols)}")
    print(f"使用设备: {CFG.device}\n")

    # 1. 网格搜索
    best_row, _ = grid_search(CFG, df, num_cols, cat_cols, y)

    # 2. 用最佳参数完整跑一遍
    if CFG.rerun_best_after_search:
        best_cfg = deepcopy(CFG)
        for k in GRID_SEARCH_SPACE.keys():
            v = best_row[k]
            if k == "n_layers":
                v = int(v)
            setattr(best_cfg, k, float(v) if isinstance(v, (int, float)) else v)
        best_cfg.n_layers = int(best_row["n_layers"])

        print("用最佳参数完整跑 5 折...\n")
        best_run_dir = output_dir / "best_run_detailed"

        summary, fold_rows = run_single_experiment(
            cfg=best_cfg, df=df, num_cols=num_cols, cat_cols=cat_cols, y=y,
            exp_output_dir=best_run_dir, save_fold_files=True
        )

        pd.DataFrame(fold_rows).to_csv(
            best_run_dir / "all_folds_metrics.csv", index=False, encoding="utf-8-sig"
        )
        with open(best_run_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(best_run_dir / "best_run_config.json", "w", encoding="utf-8") as f:
            json.dump(asdict(best_cfg), f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 50)
        print("最佳参数 5-Fold 汇总")
        print("=" * 50)
        for key, stat in summary.items():
            if key == "threshold":
                continue
            print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    print(f"\n结果保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()