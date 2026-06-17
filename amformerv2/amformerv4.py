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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,   # ① 新增 MCC
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

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
    output_dir: str = "amformerv5_results"
    label_col: str = "outcome"

    batch_size: int = 32
    epochs: int = 120
    lr: float = 3e-4
    min_lr_ratio: float = 0.05
    patience: int = 25
    fixed_threshold: float = 0.5

    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    top_k: int | None = 16
    dropout: float = 0.20
    ff_mult: int = 4

    # ⑤ Label Smoothing 替换 Focal Loss
    label_smoothing: float = 0.05      # 0→0.05, 1→0.95
    pos_weight_scale: float = 1.0      # 仍保留 pos_weight 以应对不平衡

    warmup_epochs: int = 8
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    seed: int = 42
    n_splits: int = 5

    use_smote: bool = False

    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# =============================================================
# ① 带 Label Smoothing 的 BCE 损失
# =============================================================
class LabelSmoothingBCE(nn.Module):
    """
    将硬标签 0/1 平滑为 ε/(2) 和 1-ε/(2)，
    防止模型过度自信，提升 Precision 和 AUC 稳定性。
    """
    def __init__(self, smoothing: float = 0.05, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # 平滑：0 → ε/2，1 → 1 - ε/2
        smooth_targets = targets * (1.0 - self.smoothing) + (1.0 - targets) * self.smoothing
        loss = F.binary_cross_entropy_with_logits(
            logits, smooth_targets,
            pos_weight=self.pos_weight,
            reduction="mean"
        )
        return loss


# =============================================================
# 数据集
# =============================================================
class CSFDatasetFull(Dataset):
    def __init__(self, x_num, x_cat, x_raw, labels=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.x_raw = torch.tensor(x_raw, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.labels is None:
            return self.x_num[idx], self.x_cat[idx], self.x_raw[idx]
        return self.x_num[idx], self.x_cat[idx], self.x_raw[idx], self.labels[idx]


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


# =============================================================
# ② Gated Arithmetic Block（门控交互，过滤噪声）
# =============================================================
class GatedArithmeticBlock(nn.Module):
    """
    借鉴 SAINT 动态加权思想：门控（Sigmoid）决定当前特征吸收多少
    算术交互信息，自动过滤小样本下的虚假关联噪声。

    相比原始 ArithmeticBlock，新增：
    - gate_proj: 输出 [0,1] 门控系数，按特征位维度独立控制
    - 三路算术特征（add / mul / sub）先被门控加权，再残差融合
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.weight_add = nn.Parameter(torch.ones(1))
        self.weight_mul = nn.Parameter(torch.ones(1))
        self.weight_sub = nn.Parameter(torch.ones(1))

        # 从拼接后的三路特征生成与 embed_dim 等宽的门控向量
        self.gate_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()          # 每个 embed_dim 维度独立门控
        )

        self.proj = nn.Linear(embed_dim * 3, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, L, D)
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)

        add_feat = (h + h_mean) * self.weight_add
        mul_feat = (h * h_mean) * self.weight_mul
        sub_feat = (h - h_mean) * self.weight_sub

        combined = torch.cat([add_feat, mul_feat, sub_feat], dim=-1)  # (B, L, 3D)
        combined = torch.tanh(combined)

        # 门控：决定每个 embed_dim 维度的信息通量
        gate = self.gate_proj(combined)             # (B, L, D)  ∈ [0,1]
        out = self.dropout(self.proj(combined))     # (B, L, D)
        out = out * gate                            # 逐维度门控调制

        return self.norm(h + out)


# =============================================================
# ③ 非线性数值嵌入（Linear → GELU → Linear）
# =============================================================
class NonLinearNumericEmbedding(nn.Module):
    """
    原版：Linear(1 → D)
    改进：Linear(1 → D) → GELU → Linear(D → D)

    双层投影 + 非线性激活，更好拟合医疗指标的长尾/偏态分布，
    与 SAINT 高性能 Embedding 设计对齐。
    """
    def __init__(self, n_num_features: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj1 = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_num_features)
        ])
        self.proj2 = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_num_features)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_num_features)
        ])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_num: torch.Tensor):
        if x_num.size(1) == 0:
            return None
        tokens = []
        for i, (p1, p2, norm) in enumerate(zip(self.proj1, self.proj2, self.norms)):
            out = self.act(p1(x_num[:, i:i+1]))   # Linear → GELU
            out = self.dropout(p2(out))            # Linear
            out = norm(out)
            tokens.append(out)
        return torch.stack(tokens, dim=1)


class CategoricalFeatureEmbedding(nn.Module):
    def __init__(self, cat_cardinalities, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in cat_cardinalities
        ])

    def forward(self, x_cat: torch.Tensor):
        if x_cat.size(1) == 0:
            return None
        tokens = []
        for i, (emb, norm) in enumerate(zip(self.embeddings, self.norms)):
            out = norm(emb(x_cat[:, i]))
            tokens.append(out)
        return torch.stack(tokens, dim=1)


class GlobalAttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, h):
        attn_weights = F.softmax(self.attn(h), dim=1)
        pooled = torch.sum(h * attn_weights, dim=1)
        return pooled, attn_weights


# =============================================================
# 主模型：ImprovedAMFormerV5
# =============================================================
class ImprovedAMFormerV5(nn.Module):
    """
    改进点汇总：
    ③ NonLinearNumericEmbedding（双层 Linear + GELU）
    ② GatedArithmeticBlock（Sigmoid 门控，过滤交互噪声）
    ① MCC 在 evaluate 中计算（见 ModelTrainer）
    ⑤ LabelSmoothingBCE（见损失函数，训练器中传入）
    """
    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities,
        raw_input_dim: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        top_k: int | None = 16,
        dropout: float = 0.2,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)
        self.total_tokens = n_num_features + len(cat_cardinalities)
        self.embed_dim = embed_dim

        # ③ 非线性数值嵌入
        self.num_embedding = NonLinearNumericEmbedding(n_num_features, embed_dim, dropout=dropout)
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

        # ② 门控算术块替换原始 ArithmeticBlock
        self.arith_blocks = nn.ModuleList([
            GatedArithmeticBlock(embed_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm1_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

        self.global_pool = GlobalAttentionPooling(embed_dim)

        self.shortcut = nn.Sequential(
            nn.Linear(raw_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

        classifier_in_dim = embed_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
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

    def forward(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor,
        x_raw: torch.Tensor,
        return_attention: bool = False
    ):
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
            h = arith(h)   # ② 门控算术块

        pooled_attn, pool_weights = self.global_pool(h)
        out_mean = h.mean(dim=1)
        out_max, _ = h.max(dim=1)
        shortcut_feat = self.shortcut(x_raw)

        out_feat = torch.cat([pooled_attn, out_mean, out_max, shortcut_feat], dim=-1)
        logits = self.classifier(out_feat).squeeze(-1)

        if return_attention:
            return logits, {
                "token_attention": attention_maps,
                "pool_attention": pool_weights.detach().cpu().numpy()
            }
        return logits


# =============================================================
# 学习率：Warmup + Cosine
# =============================================================
def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05):
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================
# 训练器
# =============================================================
class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        pos_weight: torch.Tensor | None = None,
        cfg: Config | None = None
    ):
        self.model = model.to(device)
        self.device = device
        self.pos_weight = pos_weight
        self.cfg = cfg

        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
        self.val_mccs = []   # ① 新增 MCC 记录

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0.0

        for batch_num, batch_cat, batch_raw, batch_labels in tqdm(train_loader, desc="Training", leave=False):
            batch_num = batch_num.to(self.device)
            batch_cat = batch_cat.to(self.device)
            batch_raw = batch_raw.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(batch_num, batch_cat, batch_raw)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.grad_clip)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch_num, batch_cat, batch_raw, batch_labels in val_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                batch_raw = batch_raw.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_num, batch_cat, batch_raw)
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        val_auc = safe_auc(all_labels, all_probs)

        # ① 验证时同步计算 MCC
        preds = (all_probs >= self.cfg.fixed_threshold).astype(int)
        val_mcc = matthews_corrcoef(all_labels, preds)

        return total_loss / max(len(val_loader), 1), all_probs, all_labels, val_auc, val_mcc

    def train(self, train_loader, val_loader, save_path):
        # ⑤ Label Smoothing BCE 替换 Focal Loss
        criterion = LabelSmoothingBCE(
            smoothing=self.cfg.label_smoothing,
            pos_weight=self.pos_weight
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay
        )

        scheduler = build_warmup_cosine_scheduler(
            optimizer=optimizer,
            warmup_epochs=self.cfg.warmup_epochs,
            total_epochs=self.cfg.epochs,
            min_lr_ratio=self.cfg.min_lr_ratio
        )

        best_val_auc = -1.0
        patience_counter = 0
        torch.save(self.model.state_dict(), save_path)

        print("开始训练 ImprovedAMFormerV5 ...")
        for epoch in range(self.cfg.epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_probs, val_labels, val_auc, val_mcc = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)
            self.val_mccs.append(val_mcc)   # ① 记录 MCC

            scheduler.step()

            if val_auc > best_val_auc + 1e-5:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch:03d} | "
                    f"TrainLoss={train_loss:.4f} | "
                    f"ValLoss={val_loss:.4f} | "
                    f"ValAUC={val_auc:.4f} | "
                    f"ValMCC={val_mcc:.4f} | "   # ① 打印 MCC
                    f"LR={current_lr:.6f}"
                )

            if patience_counter >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch}, best ValAUC={best_val_auc:.4f}")
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        print(f"训练完成！Best ValAUC = {best_val_auc:.4f}")

    def evaluate(self, val_loader, threshold: float = 0.5) -> dict:
        """
        ① 新增 MCC 指标，综合考量混淆矩阵四象限，
           对类别不平衡数据更健壮。
        """
        self.model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch_num, batch_cat, batch_raw, batch_labels in val_loader:
                batch_num = batch_num.to(self.device)
                batch_cat = batch_cat.to(self.device)
                batch_raw = batch_raw.to(self.device)

                logits = self.model(batch_num, batch_cat, batch_raw)
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
            "accuracy":  accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "f1":        f1_score(labels, preds, zero_division=0),
            "auc":       safe_auc(labels, probs),
            "mcc":       matthews_corrcoef(labels, preds),   # ① 新增 MCC
            "confusion_matrix": cm.tolist(),
        }


# =============================================================
# 辅助函数
# =============================================================
def save_attention_to_csv(attn_maps, feature_names, output_path, layer_idx=0):
    token_attention = attn_maps["token_attention"]
    if layer_idx >= len(token_attention):
        return
    att_mean = np.mean(token_attention[layer_idx], axis=(0, 1))
    pd.DataFrame(att_mean, columns=feature_names, index=feature_names).to_csv(
        output_path, encoding="utf-8-sig"
    )


def save_pool_attention_to_csv(pool_attn, feature_names, output_path):
    att_mean = np.mean(pool_attn.squeeze(-1), axis=0)
    pd.DataFrame({
        "feature": feature_names,
        "pool_attention": att_mean
    }).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_fold_predictions(results, output_path):
    pd.DataFrame({
        "y_true":    results["labels"],
        "y_prob":    results["predictions"],
        "threshold": results["threshold"],
    }).to_csv(output_path, index=False, encoding="utf-8-sig")


def summarize_results(fold_results, threshold):
    """① MCC 纳入汇总统计"""
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
        }
    summary["threshold"] = threshold
    return summary


# =============================================================
# 主流程
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
    print(f"Label Smoothing = {CFG.label_smoothing}")

    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

    fold_results = []
    fold_summary_rows = []
    first_fold_saved = False

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y), start=1):
        print(f"\n{'=' * 20} Fold {fold}/{CFG.n_splits} {'=' * 20}")

        df_train = df.iloc[train_idx].copy()
        df_val   = df.iloc[val_idx].copy()
        y_train  = y[train_idx]
        y_val    = y[val_idx]

        scaler = StandardScaler()
        x_train_num = scaler.fit_transform(df_train[num_cols].values.astype(np.float32)) if len(num_cols) > 0 else np.zeros((len(df_train), 0), dtype=np.float32)
        x_val_num   = scaler.transform(df_val[num_cols].values.astype(np.float32))       if len(num_cols) > 0 else np.zeros((len(df_val),   0), dtype=np.float32)

        cat_encoder    = CategoryEncoder().fit(df_train, cat_cols)
        x_train_cat    = cat_encoder.transform(df_train, cat_cols)
        x_val_cat      = cat_encoder.transform(df_val,   cat_cols)
        cat_cardinalities = cat_encoder.get_cardinalities(cat_cols)

        x_train_raw = np.concatenate([x_train_num, x_train_cat.astype(np.float32)], axis=1)
        x_val_raw   = np.concatenate([x_val_num,   x_val_cat.astype(np.float32)],   axis=1)

        train_loader = DataLoader(
            CSFDatasetFull(x_train_num, x_train_cat, x_train_raw, y_train),
            batch_size=CFG.batch_size, shuffle=True,  num_workers=CFG.num_workers
        )
        val_loader = DataLoader(
            CSFDatasetFull(x_val_num, x_val_cat, x_val_raw, y_val),
            batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers
        )

        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        pw_value  = (neg_count / max(pos_count, 1)) * CFG.pos_weight_scale
        pos_weight = torch.tensor(pw_value, dtype=torch.float32).to(CFG.device)
        print(f"train pos/neg = {pos_count}/{neg_count}, pos_weight = {pw_value:.4f}")

        model = ImprovedAMFormerV5(
            n_num_features=len(num_cols),
            cat_cardinalities=cat_cardinalities,
            raw_input_dim=x_train_raw.shape[1],
            embed_dim=CFG.embed_dim,
            n_heads=CFG.n_heads,
            n_layers=CFG.n_layers,
            top_k=CFG.top_k,
            dropout=CFG.dropout,
            ff_mult=CFG.ff_mult,
        )

        trainer = ModelTrainer(model, device=CFG.device, pos_weight=pos_weight, cfg=CFG)
        model_path = output_dir / f"best_amformerv5_fold{fold}.pth"

        trainer.train(train_loader, val_loader, save_path=str(model_path))

        results = trainer.evaluate(val_loader, threshold=CFG.fixed_threshold)
        fold_results.append(results)

        fold_summary = {
            "fold":      fold,
            "accuracy":  results["accuracy"],
            "precision": results["precision"],
            "recall":    results["recall"],
            "f1":        results["f1"],
            "auc":       results["auc"],
            "mcc":       results["mcc"],   # ① MCC 纳入汇总
            "threshold": CFG.fixed_threshold,
            "tn": results["confusion_matrix"][0][0],
            "fp": results["confusion_matrix"][0][1],
            "fn": results["confusion_matrix"][1][0],
            "tp": results["confusion_matrix"][1][1],
        }
        fold_summary_rows.append(fold_summary)

        with open(output_dir / f"fold_{fold}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(fold_summary, f, ensure_ascii=False, indent=2)

        save_fold_predictions(results, output_dir / f"fold_{fold}_predictions.csv")

        # ① 学习曲线同步保存 MCC
        pd.DataFrame({
            "epoch":      np.arange(len(trainer.train_losses)),
            "train_loss": trainer.train_losses,
            "val_loss":   trainer.val_losses,
            "val_auc":    trainer.val_aucs,
            "val_mcc":    trainer.val_mccs,
        }).to_csv(output_dir / f"fold_{fold}_learning_curve.csv", index=False, encoding="utf-8-sig")

        with open(output_dir / f"fold_{fold}_scaler.pkl",      "wb") as f: pickle.dump(scaler, f)
        with open(output_dir / f"fold_{fold}_cat_encoder.pkl", "wb") as f: pickle.dump(cat_encoder, f)

        print(
            f"[Fold {fold}] "
            f"AUC={results['auc']:.4f} | "
            f"MCC={results['mcc']:.4f} | "   # ① 打印 MCC
            f"F1={results['f1']:.4f} | "
            f"Acc={results['accuracy']:.4f} | "
            f"Prec={results['precision']:.4f} | "
            f"Recall={results['recall']:.4f}"
        )
        print("混淆矩阵:")
        print(np.array(results["confusion_matrix"]))

        # Attention 可视化（第一折）
        if not first_fold_saved:
            feature_names = num_cols + cat_cols
            sample_num = torch.tensor(x_val_num[0], dtype=torch.float32).unsqueeze(0).to(CFG.device)
            sample_cat = torch.tensor(x_val_cat[0], dtype=torch.long).unsqueeze(0).to(CFG.device)
            sample_raw = torch.tensor(x_val_raw[0], dtype=torch.float32).unsqueeze(0).to(CFG.device)

            with torch.no_grad():
                _, attn_maps = model(sample_num, sample_cat, sample_raw, return_attention=True)

            save_attention_to_csv(
                attn_maps, feature_names,
                str(output_dir / "attention_layer0_fold1.csv"), layer_idx=0
            )
            save_pool_attention_to_csv(
                attn_maps["pool_attention"], feature_names,
                str(output_dir / "pool_attention_fold1.csv")
            )
            first_fold_saved = True

    summary = summarize_results(fold_results, CFG.fixed_threshold)

    pd.DataFrame(fold_summary_rows).to_csv(
        output_dir / "all_folds_metrics.csv", index=False, encoding="utf-8-sig"
    )

    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(CFG), f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("5-Fold 汇总结果（ImprovedAMFormerV5）")
    print("=" * 50)
    for key, stat in summary.items():
        if key == "threshold":
            continue
        print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    print(f"\n结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()