"""
ImprovedAMFormerV7
三项方法论创新（非参数调优）：

① Masked Feature Reconstruction (MFR)
   训练时随机 mask 15-20% 特征 → 重建头预测原始值 → 辅助 MSE 损失
   强迫编码器理解特征语义，小样本正则化效果显著

② Feature Group Cross-Attention
   将特征显式分为 CSF组 / Blood组 / 其他
   先组内自注意力 → 再跨组交叉注意力
   架构与临床诊断逻辑对齐，消除无关特征间虚假关联

③ Prototype Network Classification Head
   每类学 K 个原型向量，基于距离分类而非线性层
   377 样本下比 Linear Head 更样本高效，且可解释
"""

import os
import json
import pickle
import random
import warnings
from dataclasses import dataclass, asdict
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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"


# ─────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2: return 0.5
    return roc_auc_score(y_true, y_prob)

def find_best_threshold_mcc(y_true, y_prob):
    best_thr, best_mcc = 0.5, -1.0
    for thr in np.arange(0.10, 0.91, 0.01):
        mcc = matthews_corrcoef(y_true, (y_prob >= thr).astype(int))
        if mcc > best_mcc: best_mcc, best_thr = mcc, float(thr)
    return best_thr, best_mcc


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
@dataclass
class Config:
    data_path:   str   = "/Users/wangqinyang.5/Desktop/Infection/original.xlsx"
    output_dir:  str   = "amformerv7_results"
    label_col:   str   = "outcome"

    batch_size:  int   = 32
    epochs:      int   = 120
    lr:          float = 3e-4
    min_lr_ratio:float = 0.05
    patience:    int   = 25

    embed_dim:   int   = 64
    n_heads:     int   = 4
    top_k:       int   = 16
    dropout:     float = 0.20

    label_smoothing:   float = 0.05
    pos_weight_scale:  float = 0.8

    # ① MFR
    mfr_mask_ratio: float = 0.15   # 每步随机 mask 比例
    mfr_lambda:     float = 0.4    # 辅助损失权重

    # ③ Prototype head
    n_prototypes:   int   = 4      # 每类原型数量（正/负各 K 个）
    proto_temp:     float = 0.1    # 距离 softmax 温度

    warmup_epochs: int   = 8
    weight_decay:  float = 1e-2
    grad_clip:     float = 1.0
    seed:          int   = 42
    n_splits:      int   = 5
    num_workers:   int   = 0
    device:        str   = "cuda" if torch.cuda.is_available() else "cpu"

    # ② 特征分组（列名）—— 与数据对齐后自动过滤
    csf_cols:   tuple = ("C_G", "C_WBC", "C_RBC", "C_P", "C_N")
    blood_cols: tuple = ("B_G", "B_CRP", "B_WBC", "B_N", "B_Lym", "B_PCT", "B_AC", "B_RBC")


CFG = Config()


# ─────────────────────────────────────────────
# 损失
# ─────────────────────────────────────────────
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing, self.pos_weight = smoothing, pos_weight

    def forward(self, logits, targets):
        st = targets.float() * (1 - self.smoothing) + (1 - targets.float()) * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, st, pos_weight=self.pos_weight)


# ─────────────────────────────────────────────
# 数据集
# ─────────────────────────────────────────────
class CSFDatasetFull(Dataset):
    def __init__(self, x_num, x_cat, x_raw, labels=None):
        self.x_num  = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat  = torch.tensor(x_cat, dtype=torch.long)
        self.x_raw  = torch.tensor(x_raw, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.x_num)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.x_num[idx], self.x_cat[idx], self.x_raw[idx]
        return self.x_num[idx], self.x_cat[idx], self.x_raw[idx], self.labels[idx]


# ─────────────────────────────────────────────
# 特征工程
# ─────────────────────────────────────────────
def build_feature_dataframe(df, cfg):
    df = df.copy()
    cat_cols      = [c for c in ["sex","tube","site","other_inf","transparency"] if c in df.columns]
    base_num_cols = [c for c in ["age","C_G","C_WBC","C_RBC","C_P","C_N",
                                  "GCS","tem","B_G","B_CRP","B_WBC","B_N",
                                  "B_Lym","B_PCT","B_AC","B_RBC"] if c in df.columns]

    for c in base_num_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["C_WBC","C_RBC","C_P","B_CRP","B_WBC","B_PCT","B_AC","B_RBC"]:
        if c in df.columns: df[c] = np.log1p(pd.to_numeric(df[c], errors="coerce").clip(lower=0))
    for c in base_num_cols:
        if c in df.columns: df[c] = df[c].fillna(df[c].median())

    eps, new_num_cols = 1e-6, []
    pairs = [("C_G","B_G","ratio_C_G_B_G","/"),("C_N","B_N","diff_C_N_B_N","-")]
    for a,b,name,op in pairs:
        if a in df.columns and b in df.columns:
            df[name] = df[a]/(df[b]+eps) if op=="/" else df[a]-df[b]
            new_num_cols.append(name)
    if all(c in df.columns for c in ["C_WBC","B_WBC","C_RBC","B_RBC"]):
        df["corrected_WBC"] = df["C_WBC"] - df["B_WBC"]*df["C_RBC"]/(df["B_RBC"]+eps)
        new_num_cols.append("corrected_WBC")
        df["ratio_WBC_RBC_diff"] = df["B_WBC"]/(df["B_RBC"]+eps) - df["C_WBC"]/(df["C_RBC"]+eps)
        new_num_cols.append("ratio_WBC_RBC_diff")

    num_cols = base_num_cols + new_num_cols
    for c in new_num_cols:
        df[c] = df[c].replace([np.inf,-np.inf], np.nan).fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].astype(str).fillna("Unknown").replace({"nan":"Unknown","None":"Unknown"})

    y = pd.to_numeric(df[cfg.label_col], errors="coerce").values.astype(int)
    return df, num_cols, cat_cols, y


def get_group_indices(num_cols, cfg):
    """② 返回 CSF 组 / Blood 组 / Other 组 的下标列表"""
    csf_idx   = [i for i,c in enumerate(num_cols) if c in cfg.csf_cols]
    blood_idx = [i for i,c in enumerate(num_cols) if c in cfg.blood_cols]
    other_idx = [i for i,c in enumerate(num_cols)
                 if i not in csf_idx and i not in blood_idx]
    return csf_idx, blood_idx, other_idx


class CategoryEncoder:
    def __init__(self): self.maps = {}
    def fit(self, df, cat_cols):
        for c in cat_cols:
            uniq = sorted(set(df[c].astype(str).fillna("Unknown")))
            self.maps[c] = {v:i+1 for i,v in enumerate(uniq)}
        return self
    def transform(self, df, cat_cols):
        if not cat_cols: return np.zeros((len(df),0), dtype=np.int64)
        return np.array([[self.maps[c].get(v,0) for v in df[c].astype(str).fillna("Unknown")]
                         for c in cat_cols], dtype=np.int64).T
    def get_cardinalities(self, cat_cols):
        return [max(self.maps[c].values(),default=0)+1 for c in cat_cols]


# ─────────────────────────────────────────────
# 模型基础模块
# ─────────────────────────────────────────────
class TopKSparseAttention(nn.Module):
    def __init__(self, d, n_heads, top_k=None, dropout=0.1):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dk, self.top_k = n_heads, d//n_heads, top_k
        self.wq = nn.Linear(d,d,bias=False); self.wk = nn.Linear(d,d,bias=False)
        self.wv = nn.Linear(d,d,bias=False); self.wo = nn.Linear(d,d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in, k_in=None, v_in=None):
        """支持 self-attn（k_in=None）和 cross-attn（传入 k_in, v_in）"""
        k_in = q_in if k_in is None else k_in
        v_in = k_in if v_in is None else v_in
        B, Lq, _ = q_in.size(); Lk = k_in.size(1)
        q = self.wq(q_in).view(B,Lq,self.h,self.dk).transpose(1,2)
        k = self.wk(k_in).view(B,Lk,self.h,self.dk).transpose(1,2)
        v = self.wv(v_in).view(B,Lk,self.h,self.dk).transpose(1,2)
        s = torch.matmul(q, k.transpose(-2,-1)) / (self.dk**0.5)
        if self.top_k and self.top_k < Lk:
            tv,_ = torch.topk(s, min(self.top_k,Lk), dim=-1)
            s = s.masked_fill(s < tv[...,-1:], float("-inf"))
        attn = torch.nan_to_num(F.softmax(s, dim=-1), nan=0.0)
        attn = self.drop(attn)
        ctx  = torch.matmul(attn, v).transpose(1,2).contiguous().view(B,Lq,-1)
        return self.wo(ctx), attn


class GatedArithmeticBlock(nn.Module):
    def __init__(self, d, dropout=0.1):
        super().__init__()
        self.wa = nn.Parameter(torch.ones(1))
        self.wm = nn.Parameter(torch.ones(1))
        self.ws = nn.Parameter(torch.ones(1))
        self.gate = nn.Sequential(nn.Linear(d*3, d), nn.Sigmoid())
        self.proj = nn.Linear(d*3, d)
        self.norm = nn.LayerNorm(d); self.drop = nn.Dropout(dropout)

    def forward(self, h):
        hm  = h.mean(1, keepdim=True).expand_as(h)
        cat = torch.tanh(torch.cat([(h+hm)*self.wa,(h*hm)*self.wm,(h-hm)*self.ws], dim=-1))
        return self.norm(h + self.drop(self.proj(cat)) * self.gate(cat))


class NonLinearNumericEmbedding(nn.Module):
    def __init__(self, n, d, dropout=0.1):
        super().__init__()
        self.p1   = nn.ModuleList([nn.Linear(1,d) for _ in range(n)])
        self.p2   = nn.ModuleList([nn.Linear(d,d) for _ in range(n)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(n)])
        self.act  = nn.GELU(); self.drop = nn.Dropout(dropout)

    def forward(self, x):
        if x.size(1) == 0: return None
        return torch.stack([
            n(self.drop(p2(self.act(p1(x[:,i:i+1])))))
            for i,(p1,p2,n) in enumerate(zip(self.p1,self.p2,self.norm))
        ], dim=1)


class CatEmbedding(nn.Module):
    def __init__(self, cards, d):
        super().__init__()
        self.emb  = nn.ModuleList([nn.Embedding(c,d) for c in cards])
        self.norm = nn.ModuleList([nn.LayerNorm(d)   for _ in cards])
    def forward(self, x):
        if x.size(1) == 0: return None
        return torch.stack([n(e(x[:,i])) for i,(e,n) in enumerate(zip(self.emb,self.norm))], dim=1)


# ─────────────────────────────────────────────
# ② Feature Group Cross-Attention Layer
# ─────────────────────────────────────────────
class GroupCrossAttentionLayer(nn.Module):
    """
    将 token 序列按预设的 CSF / Blood / Other 下标分组。
    流程：
      1. 各组独立做 intra-group self-attention（TopK）
      2. CSF 组 ← 以 Blood 组为 KV 做 cross-attention（双向）
      3. Blood 组 ← 以 CSF 组为 KV 做 cross-attention
      4. 重新拼回原位，保持 token 顺序
    """
    def __init__(self, embed_dim, n_heads, top_k, dropout, n_num):
        super().__init__()
        d = embed_dim
        # intra-group self-attention（3组，按需）
        self.self_csf   = TopKSparseAttention(d, n_heads, top_k, dropout)
        self.self_blood = TopKSparseAttention(d, n_heads, top_k, dropout)
        self.self_other = TopKSparseAttention(d, n_heads, top_k, dropout)
        # CSF ← Blood 交叉注意力
        self.cross_csf_from_blood  = TopKSparseAttention(d, n_heads, top_k, dropout)
        # Blood ← CSF 交叉注意力
        self.cross_blood_from_csf  = TopKSparseAttention(d, n_heads, top_k, dropout)

        self.norm_csf   = nn.LayerNorm(d)
        self.norm_blood = nn.LayerNorm(d)
        self.norm_other = nn.LayerNorm(d)
        self.n_num = n_num   # 数值特征数（类别特征排在后面）

    def _pick(self, h, idx):
        if not idx: return None
        return h[:, idx, :]

    def _put(self, h, idx, val):
        if not idx: return h
        h = h.clone()
        h[:, idx, :] = val
        return h

    def forward(self, h, csf_idx, blood_idx, other_idx):
        # ① 各组 intra self-attn
        def self_attn_group(attn_mod, norm_mod, tokens, h, idx):
            if tokens is None or len(idx) == 0: return h
            out, _ = attn_mod(tokens)
            return self._put(h, idx, norm_mod(tokens + out))

        h_csf   = self._pick(h, csf_idx)
        h_blood = self._pick(h, blood_idx)
        h_other = self._pick(h, other_idx)

        h = self_attn_group(self.self_csf,   self.norm_csf,   h_csf,   h, csf_idx)
        h = self_attn_group(self.self_blood, self.norm_blood, h_blood, h, blood_idx)
        h = self_attn_group(self.self_other, self.norm_other, h_other, h, other_idx)

        # ② 跨组 cross-attn（需要两组都存在）
        h_csf_new   = self._pick(h, csf_idx)
        h_blood_new = self._pick(h, blood_idx)

        if h_csf_new is not None and h_blood_new is not None:
            # CSF ← Blood: Q=CSF, KV=Blood
            cross_csf, _ = self.cross_csf_from_blood(h_csf_new, h_blood_new, h_blood_new)
            h = self._put(h, csf_idx, self.norm_csf(h_csf_new + cross_csf))

            # Blood ← CSF: Q=Blood, KV=CSF（更新后）
            h_csf_updated = self._pick(h, csf_idx)
            cross_blood, _ = self.cross_blood_from_csf(h_blood_new, h_csf_updated, h_csf_updated)
            h = self._put(h, blood_idx, self.norm_blood(h_blood_new + cross_blood))

        return h


# ─────────────────────────────────────────────
# ③ Prototype Classification Head
# ─────────────────────────────────────────────
class PrototypeHead(nn.Module):
    """
    每类学习 K 个原型向量。
    分类 logit = log P(y=1|x)，基于特征向量到各原型的负 L2 距离 softmax。

    正类得分 = log Σ_k exp(-||x - proto_pos_k||² / τ)
    负类得分 = log Σ_k exp(-||x - proto_neg_k||² / τ)
    logit    = 正类得分 - 负类得分
    """
    def __init__(self, in_dim, n_prototypes=4, temperature=0.1):
        super().__init__()
        self.K    = n_prototypes
        self.temp = temperature
        # 正/负类各 K 个原型，xavier 初始化
        self.proto_pos = nn.Parameter(torch.empty(n_prototypes, in_dim))
        self.proto_neg = nn.Parameter(torch.empty(n_prototypes, in_dim))
        nn.init.xavier_uniform_(self.proto_pos)
        nn.init.xavier_uniform_(self.proto_neg)

    def forward(self, x):
        # x: (B, D)
        # dist²(x, proto): (B, K)
        d_pos = torch.cdist(x.unsqueeze(0), self.proto_pos.unsqueeze(0)).squeeze(0) ** 2  # (B,K)
        d_neg = torch.cdist(x.unsqueeze(0), self.proto_neg.unsqueeze(0)).squeeze(0) ** 2  # (B,K)

        score_pos = torch.logsumexp(-d_pos / self.temp, dim=-1)   # (B,)
        score_neg = torch.logsumexp(-d_neg / self.temp, dim=-1)   # (B,)

        return score_pos - score_neg   # logit


# ─────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────
class ImprovedAMFormerV7(nn.Module):
    """
    ① MFR: forward 时可传入 mask_ratio 触发掩码重建分支
    ② GroupCrossAttentionLayer: CSF ↔ Blood 跨组交叉注意力
    ③ PrototypeHead: 基于距离的原型分类
    """
    def __init__(
        self, n_num, cat_cards, raw_dim,
        csf_idx, blood_idx, other_idx,
        embed_dim=64, n_heads=4, top_k=16,
        dropout=0.20, ff_mult=4,
        n_prototypes=4, proto_temp=0.1,
        mfr_mask_ratio=0.15,
    ):
        super().__init__()
        d = embed_dim
        self.n_num    = n_num
        self.csf_idx  = csf_idx
        self.blood_idx= blood_idx
        self.other_idx= other_idx
        self.mfr_mask_ratio = mfr_mask_ratio

        # 嵌入层
        self.num_emb = NonLinearNumericEmbedding(n_num, d, dropout) if n_num > 0 else None
        self.cat_emb = CatEmbedding(cat_cards, d) if cat_cards else None
        total_tok    = n_num + len(cat_cards)
        self.pos     = nn.Parameter(torch.randn(1, max(1, total_tok), d) * 0.01)

        # ② 两层 GroupCrossAttentionLayer
        self.group_layers = nn.ModuleList([
            GroupCrossAttentionLayer(d, n_heads, top_k, dropout, n_num)
            for _ in range(2)
        ])

        # FFN + 门控算术（每层后）
        self.ffns  = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d*ff_mult), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d*ff_mult, d), nn.Dropout(dropout)
            ) for _ in range(2)
        ])
        self.ariths = nn.ModuleList([GatedArithmeticBlock(d, dropout) for _ in range(2)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])

        # Global pooling
        self.pool_attn = nn.Sequential(nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1))
        self.shortcut  = nn.Sequential(nn.Linear(raw_dim, d), nn.LayerNorm(d), nn.Dropout(dropout))

        # ③ Prototype head
        proto_in = d * 4   # attn_pool + mean + max + shortcut
        self.proto_head = PrototypeHead(proto_in, n_prototypes, proto_temp)

        # ① MFR 重建头（预测被 mask 的数值特征原始值）
        if n_num > 0:
            self.recon_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _pool(self, h):
        w = F.softmax(self.pool_attn(h), dim=1)
        return (h * w).sum(dim=1)

    def forward(self, x_num, x_cat, x_raw,
                mask_ratio: float = 0.0,
                return_attention: bool = False):
        """
        mask_ratio > 0 时启动 MFR 分支，返回 (logit, recon_loss)
        mask_ratio = 0 时返回 logit（推理模式）
        """
        B = x_num.size(0)

        # ① MFR：在嵌入前对原始数值特征施加随机 mask
        x_num_orig = x_num.clone()
        mfr_mask   = None
        if mask_ratio > 0 and self.n_num > 0:
            # 生成 (B, n_num) 的 bool mask，True = 被遮蔽
            mfr_mask = torch.rand(B, self.n_num, device=x_num.device) < mask_ratio
            x_num_masked = x_num.clone()
            x_num_masked[mfr_mask] = 0.0   # 遮蔽置 0
        else:
            x_num_masked = x_num

        # 嵌入
        nt = self.num_emb(x_num_masked) if self.num_emb else None
        ct = self.cat_emb(x_cat)        if self.cat_emb else None
        h  = torch.cat([t for t in [nt, ct] if t is not None], dim=1)
        h  = h + self.pos[:, :h.size(1)]

        # ② Group Cross-Attention × 2 + FFN + Arith
        for group_layer, ffn, arith, fnorm in zip(
            self.group_layers, self.ffns, self.ariths, self.ffn_norms
        ):
            h = group_layer(h, self.csf_idx, self.blood_idx, self.other_idx)
            h = fnorm(h + ffn(h))
            h = arith(h)

        # Pooling
        pooled  = self._pool(h)
        out_mean = h.mean(1)
        out_max  = h.max(1).values
        sc       = self.shortcut(x_raw)
        feat     = torch.cat([pooled, out_mean, out_max, sc], dim=-1)   # (B, 4D)

        # ③ Prototype Head
        logits = self.proto_head(feat)   # (B,)

        # ① MFR 重建损失
        if mask_ratio > 0 and mfr_mask is not None and self.n_num > 0:
            # 用被 mask 的 token 的最终隐向量预测原始值
            num_h = h[:, :self.n_num, :]          # (B, n_num, D)
            recon = self.recon_head(num_h).squeeze(-1)  # (B, n_num)
            # 只在 mask=True 的位置计算重建 MSE
            if mfr_mask.any():
                recon_loss = F.mse_loss(recon[mfr_mask], x_num_orig[mfr_mask])
            else:
                recon_loss = torch.tensor(0.0, device=x_num.device)
            return logits, recon_loss

        if return_attention:
            return logits, None

        return logits


# ─────────────────────────────────────────────
# 学习率调度
# ─────────────────────────────────────────────
def build_scheduler(opt, warmup, total, min_r=0.05):
    def fn(ep):
        if ep < warmup: return (ep+1)/max(warmup,1)
        p = (ep-warmup)/max(total-warmup,1)
        return min_r + (1-min_r)*0.5*(1+np.cos(np.pi*p))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


# ─────────────────────────────────────────────
# 训练器
# ─────────────────────────────────────────────
class ModelTrainer:
    def __init__(self, model, device="cpu", pos_weight=None, cfg=None):
        self.model = model.to(device)
        self.device, self.pw, self.cfg = device, pos_weight, cfg
        self.train_losses, self.val_losses = [], []
        self.val_aucs,     self.val_mccs   = [], []

    def train_epoch(self, loader, optimizer, criterion):
        self.model.train()
        total = 0.0
        for batch in tqdm(loader, desc="Train", leave=False):
            bn,bc,br,bl = [x.to(self.device) for x in batch]
            # ① 训练时启用 MFR mask
            out = self.model(bn, bc, br, mask_ratio=self.cfg.mfr_mask_ratio)
            logits, recon_loss = out
            cls_loss  = criterion(logits, bl)
            loss      = cls_loss + self.cfg.mfr_lambda * recon_loss
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            optimizer.step()
            total += loss.item()
        return total / max(len(loader), 1)

    @torch.no_grad()
    def _collect(self, loader, criterion=None):
        self.model.eval()
        probs, labels, lsum = [], [], 0.0
        for batch in loader:
            bn,bc,br,bl = [x.to(self.device) for x in batch]
            # 推理：不 mask
            logits = self.model(bn, bc, br, mask_ratio=0.0)
            if isinstance(logits, tuple): logits = logits[0]
            if criterion: lsum += criterion(logits, bl).item()
            probs.extend(torch.sigmoid(logits).cpu().numpy())
            labels.extend(bl.cpu().numpy())
        return np.array(probs), np.array(labels), lsum/max(len(loader),1)

    def train(self, tr_loader, va_loader, save_path):
        criterion = LabelSmoothingBCE(self.cfg.label_smoothing, self.pw)
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        scheduler = build_scheduler(optimizer, self.cfg.warmup_epochs,
                                    self.cfg.epochs, self.cfg.min_lr_ratio)
        best_auc, wait = -1.0, 0
        torch.save(self.model.state_dict(), save_path)

        for ep in range(self.cfg.epochs):
            tl = self.train_epoch(tr_loader, optimizer, criterion)
            vp, vl, vls = self._collect(va_loader, criterion)
            vauc = safe_auc(vl, vp)
            _, vmcc = find_best_threshold_mcc(vl, vp)
            self.train_losses.append(tl); self.val_losses.append(vls)
            self.val_aucs.append(vauc);   self.val_mccs.append(vmcc)
            scheduler.step()

            if vauc > best_auc + 1e-5:
                best_auc, wait = vauc, 0
                torch.save(self.model.state_dict(), save_path)
            else:
                wait += 1

            if ep % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  Ep{ep:03d} | TrL={tl:.4f} | VaL={vls:.4f} | AUC={vauc:.4f} | MCC={vmcc:.4f} | lr={lr:.2e}")

            if wait >= self.cfg.patience:
                print(f"  Early stop @{ep}, bestAUC={best_auc:.4f}"); break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        print(f"  训练完成，BestValAUC={best_auc:.4f}")

    def evaluate(self, loader) -> dict:
        probs, labels, _ = self._collect(loader)
        thr, _ = find_best_threshold_mcc(labels, probs)
        preds  = (probs >= thr).astype(int)
        cm     = confusion_matrix(labels, preds)
        return {
            "predictions": probs.tolist(), "labels": labels.tolist(),
            "threshold":   thr,
            "accuracy":    accuracy_score(labels, preds),
            "precision":   precision_score(labels, preds, zero_division=0),
            "recall":      recall_score(labels, preds, zero_division=0),
            "f1":          f1_score(labels, preds, zero_division=0),
            "auc":         safe_auc(labels, probs),
            "mcc":         matthews_corrcoef(labels, preds),
            "confusion_matrix": cm.tolist(),
        }


# ─────────────────────────────────────────────
# 辅助
# ─────────────────────────────────────────────
def summarize(fold_results, tag=""):
    print(f"\n{'='*50}\n5-Fold 汇总 【{tag}】\n{'='*50}")
    s = {}
    for k in ["accuracy","precision","recall","f1","auc","mcc"]:
        v = [r[k] for r in fold_results]
        s[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        print(f"  {k:>10}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    return s

def save_preds(res, path):
    pd.DataFrame({"y_true": res["labels"], "y_prob": res["predictions"],
                  "threshold": res["threshold"]}).to_csv(path, index=False, encoding="utf-8-sig")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────
def main():
    set_seed(CFG.seed)
    out = Path(CFG.output_dir); out.mkdir(parents=True, exist_ok=True)

    print("加载数据 ...")
    df, num_cols, cat_cols, y = build_feature_dataframe(pd.read_excel(CFG.data_path), CFG)
    print(f"样本={len(df)} | 阳性={int(y.sum())} | 数值特征={len(num_cols)} | 类别特征={len(cat_cols)}")

    # ② 计算特征分组下标
    csf_idx, blood_idx, other_idx = get_group_indices(num_cols, CFG)
    print(f"特征分组 | CSF={csf_idx} | Blood={blood_idx} | Other={other_idx}")

    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    fold_results, rows = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(df, y), 1):
        print(f"\n{'='*22} Fold {fold}/{CFG.n_splits} {'='*22}")
        df_tr, df_va = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
        y_tr, y_va   = y[tr_idx], y[va_idx]

        scaler   = StandardScaler()
        x_tr_num = scaler.fit_transform(df_tr[num_cols].values.astype(np.float32)) if num_cols else np.zeros((len(df_tr),0),dtype=np.float32)
        x_va_num = scaler.transform(df_va[num_cols].values.astype(np.float32))     if num_cols else np.zeros((len(df_va),0),dtype=np.float32)

        enc       = CategoryEncoder().fit(df_tr, cat_cols)
        x_tr_cat  = enc.transform(df_tr, cat_cols)
        x_va_cat  = enc.transform(df_va, cat_cols)
        cat_cards = enc.get_cardinalities(cat_cols)

        x_tr_raw  = np.concatenate([x_tr_num, x_tr_cat.astype(np.float32)], axis=1)
        x_va_raw  = np.concatenate([x_va_num, x_va_cat.astype(np.float32)], axis=1)

        tr_ld = DataLoader(CSFDatasetFull(x_tr_num, x_tr_cat, x_tr_raw, y_tr),
                           batch_size=CFG.batch_size, shuffle=True,  num_workers=CFG.num_workers)
        va_ld = DataLoader(CSFDatasetFull(x_va_num, x_va_cat, x_va_raw, y_va),
                           batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

        pos_cnt = int(y_tr.sum()); neg_cnt = len(y_tr) - pos_cnt
        pw = torch.tensor(neg_cnt/max(pos_cnt,1)*CFG.pos_weight_scale, dtype=torch.float32).to(CFG.device)
        print(f"  pos/neg={pos_cnt}/{neg_cnt}  pos_weight={pw.item():.3f}")

        model = ImprovedAMFormerV7(
            n_num=len(num_cols), cat_cards=cat_cards, raw_dim=x_tr_raw.shape[1],
            csf_idx=csf_idx, blood_idx=blood_idx, other_idx=other_idx,
            embed_dim=CFG.embed_dim, n_heads=CFG.n_heads, top_k=CFG.top_k,
            dropout=CFG.dropout,
            n_prototypes=CFG.n_prototypes, proto_temp=CFG.proto_temp,
            mfr_mask_ratio=CFG.mfr_mask_ratio,
        )

        trainer = ModelTrainer(model, CFG.device, pw, CFG)
        trainer.train(tr_ld, va_ld, str(out/f"best_v7_fold{fold}.pth"))

        res = trainer.evaluate(va_ld)
        fold_results.append(res)

        row = {"fold": fold, **{k: res[k] for k in ["accuracy","precision","recall","f1","auc","mcc","threshold"]},
               "tn": res["confusion_matrix"][0][0], "fp": res["confusion_matrix"][0][1],
               "fn": res["confusion_matrix"][1][0], "tp": res["confusion_matrix"][1][1]}
        rows.append(row)

        print(f"  [V7] AUC={res['auc']:.4f} | MCC={res['mcc']:.4f} | F1={res['f1']:.4f} | "
              f"Prec={res['precision']:.4f} | Rec={res['recall']:.4f} | thr={res['threshold']:.2f}")
        print(f"  CM: {np.array(res['confusion_matrix'])}")

        save_preds(res, out/f"fold{fold}_pred.csv")
        pd.DataFrame({
            "epoch": np.arange(len(trainer.train_losses)),
            "train_loss": trainer.train_losses, "val_loss": trainer.val_losses,
            "val_auc": trainer.val_aucs,        "val_mcc": trainer.val_mccs,
        }).to_csv(out/f"fold{fold}_curve.csv", index=False, encoding="utf-8-sig")

        with open(out/f"fold{fold}_scaler.pkl",  "wb") as f: pickle.dump(scaler, f)
        with open(out/f"fold{fold}_encoder.pkl", "wb") as f: pickle.dump(enc, f)
        with open(out/f"fold{fold}_metrics.json","w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

    summary = summarize(fold_results, tag="AMFormerV7")
    pd.DataFrame(rows).to_csv(out/"all_folds.csv", index=False, encoding="utf-8-sig")
    with open(out/"summary.json","w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out/"config.json","w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {out.resolve()}")


if __name__ == "__main__":
    main()