import os
import json
import pickle
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
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

REPO_ROOT = Path(__file__).resolve().parents[2]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    data_path: str = str(REPO_ROOT / "data" / "legacy" / "original.xlsx")
    output_dir: str = str(REPO_ROOT / "results" / "metrics" / "amformer_sparse")
    label_col: str = "outcome"
    batch_size: int = 32
    epochs: int = 100
    lr: float = 3e-4      # 更稳定的学习率
    patience: int = 25    # 给模型更多收敛时间
    fixed_threshold: float = 0.5
    embed_dim: int = 64   # 小样本适当缩小，防过拟合
    n_heads: int = 4
    n_layers: int = 2     # 层数降低，加快收敛
    top_k: int = 8           # Top-K 稀疏注意力
    dropout: float = 0.15
    seed: int = 42
    n_splits: int = 5
    pos_weight_scale: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# =============================================================
# Dataset
# =============================================================
class CSFDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray | None = None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = None if labels is None else torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.features[idx]
        return self.features[idx], self.labels[idx]


# =============================================================
# 改进1：Top-K 稀疏多头注意力
# =============================================================
class TopKSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, top_k: int = 8, dropout: float = 0.1):
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
        # x: (batch, n_feat, d_model)
        B, L, _ = x.size()

        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # ── Top-K 稀疏化：每行只保留最高 K 个，其余 -inf ──
        effective_k = min(self.top_k, L)
        topk_vals, _ = torch.topk(scores, effective_k, dim=-1)
        threshold = topk_vals[..., -1:].expand_as(scores)
        scores = scores.masked_fill(scores < threshold, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        # 当整行都是 -inf 时 softmax 返回 nan，用 nan_to_num 兜底
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        return self.w_o(context), attn_weights


# =============================================================
# 改进2：Arithmetic Block（显式四则运算交互）
# =============================================================
class ArithmeticBlock(nn.Module):
    """
    对每个 token 与全局均值做四则运算，建模显式特征交互。
    对应图中 Arithmetic Block 的核心思想。
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(embed_dim * 3, embed_dim)  # 3路：去掉除法
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (batch, n_feat, embed_dim)
        h_mean = h.mean(dim=1, keepdim=True).expand_as(h)

        add_feat = h + h_mean          # 协同增强
        mul_feat = h * h_mean          # 乘性交互
        sub_feat = h - h_mean          # 差值
        # 除法在 LayerNorm 后 h_mean≈0 → 数值爆炸，已移除

        combined = torch.cat([add_feat, mul_feat, sub_feat], dim=-1)
        out = self.dropout(self.proj(combined))
        return self.norm(h + out)                    # 残差连接


# =============================================================
# 改进3：每特征独立 Embedding
# =============================================================
class PerFeatureEmbedding(nn.Module):
    """
    每个特征有独立的线性映射权重，而非共享同一个 nn.Linear(1, embed_dim)。
    让模型对不同特征学到差异化的表示。
    """
    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(n_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        tokens = [
            self.embeddings[i](x[:, i:i+1])   # (batch, embed_dim)
            for i in range(len(self.embeddings))
        ]
        return torch.stack(tokens, dim=1)      # (batch, n_features, embed_dim)


# =============================================================
# 主模型：ImprovedAMFormer
# =============================================================
class ImprovedAMFormer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,  # 小样本适当缩小，防过拟合,
        n_heads: int = 4,
        n_layers: int = 2,     # 层数降低，加快收敛,
        top_k: int = 8,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # ── 改进3：每特征独立 embedding ──
        self.feature_embedding = PerFeatureEmbedding(input_dim, embed_dim)

        # 位置编码（可选，保留原版）
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.01)

        # ── Transformer 层：Top-K 稀疏注意力 + FFN + Arithmetic Block ──
        self.attn_layers  = nn.ModuleList([
            TopKSparseAttention(embed_dim, n_heads, top_k=top_k, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ffn_layers   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])
        self.arith_blocks = nn.ModuleList([
            ArithmeticBlock(embed_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm1_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.norm2_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])

        # ── 改进2：Mean + Max 双路聚合 → 拼接 ──
        # 输出维度 embed_dim * 2
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

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (batch, input_dim)

        # 1. 每特征独立 embedding
        # 注意：数据已经过 StandardScaler，不能再做 relu+log（会清零所有负值）
        h = self.feature_embedding(x)                # (batch, n_feat, embed_dim)
        h = h + self.pos_encoding[:, :self.input_dim, :]

        attention_maps = []

        # 2. 多层：Sparse Attention → FFN → Arithmetic Block
        for attn, ffn, arith, norm1, norm2 in zip(
            self.attn_layers, self.ffn_layers, self.arith_blocks,
            self.norm1_layers, self.norm2_layers
        ):
            # Sparse Attention + 残差
            attn_out, attn_w = attn(h)
            attention_maps.append(attn_w.detach().cpu().numpy())
            h = norm1(h + attn_out)

            # FFN + 残差
            h = norm2(h + ffn(h))

            # Arithmetic Block（四则运算交互）
            h = arith(h)

        # 3. 改进2：Mean + Max 双路聚合
        out_mean = h.mean(dim=1)                     # (batch, embed_dim)
        out_max, _ = h.max(dim=1)                    # (batch, embed_dim)
        out_feat = torch.cat([out_mean, out_max], dim=-1)  # (batch, embed_dim*2)

        logits = self.classifier(out_feat).squeeze(-1)

        if return_attention:
            return logits, attention_maps
        return logits


# =============================================================
# 训练器
# =============================================================
class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu",
                 pos_weight: torch.Tensor | None = None):
        self.model = model.to(device)
        self.device = device
        self.pos_weight = pos_weight
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train_epoch(self, train_loader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_features, batch_labels in tqdm(train_loader, desc="Training", leave=False):
            batch_features = batch_features.to(self.device)
            batch_labels   = batch_labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(batch_features), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels   = batch_labels.to(self.device)
                logits = self.model(batch_features)
                total_loss += criterion(logits, batch_labels).item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        return total_loss / len(val_loader), np.array(all_probs), np.array(all_labels)

    def train(self, train_loader, val_loader, save_path,
              epochs=100, lr=1e-3, patience=15) -> None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01  # 小样本适当增大 weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        best_val_loss = float("inf")
        patience_counter = 0
        torch.save(self.model.state_dict(), save_path)

        print("开始训练改进版 AMFormer...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_probs, val_labels = self.validate(val_loader, criterion)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                val_auc = roc_auc_score(val_labels, val_probs)
                print(f"Epoch {epoch:03d}: Train={train_loss:.4f}  "
                      f"Val={val_loss:.4f}  AUC={val_auc:.4f}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        print("训练完成！")

    def evaluate(self, val_loader, threshold: float = 0.5) -> dict:
        """推理并计算全套指标，修复原版 results 未定义的 bug"""
        self.model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                logits = self.model(batch_features.to(self.device))
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_labels.numpy())

        probs  = np.array(all_probs)
        labels = np.array(all_labels)
        preds  = (probs >= threshold).astype(int)
        cm     = confusion_matrix(labels, preds)

        return {
            "predictions": probs.tolist(),
            "labels":      labels.tolist(),
            "threshold":   threshold,
            "accuracy":    accuracy_score(labels, preds),
            "precision":   precision_score(labels, preds, zero_division=0),
            "recall":      recall_score(labels, preds, zero_division=0),
            "f1":          f1_score(labels, preds, zero_division=0),
            "auc":         roc_auc_score(labels, probs),
            "confusion_matrix": cm.tolist(),
        }


# =============================================================
# 特征工程（与原版保持一致）
# =============================================================
def build_feature_matrix(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    log_cols = ["C_WBC", "C_RBC", "C_P", "B_CRP", "B_WBC", "B_PCT", "B_AC", "B_RBC"]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    eps = 1e-6
    if "C_G" in df.columns and "B_G" in df.columns:
        df["ratio_C_G_B_G"] = df["C_G"] / (df["B_G"] + eps)
    if "C_N" in df.columns and "B_N" in df.columns:
        df["diff_C_N_B_N"] = df["C_N"] - df["B_N"]
    if all(c in df.columns for c in ["C_WBC", "B_WBC", "C_RBC", "B_RBC"]):
        df["corrected_WBC"] = df["C_WBC"] - df["B_WBC"] * df["C_RBC"] / (df["B_RBC"] + eps)
    if all(c in df.columns for c in ["B_WBC", "B_RBC", "C_WBC", "C_RBC"]):
        df["ratio_WBC_RBC_diff"] = (
            df["B_WBC"] / (df["B_RBC"] + eps) - df["C_WBC"] / (df["C_RBC"] + eps)
        )

    base_features = [
        "age", "C_G", "C_WBC", "C_RBC", "C_P", "C_N",
        "transparency", "GCS", "tem", "B_G", "B_CRP",
        "B_WBC", "B_N", "B_Lym", "B_PCT", "B_AC", "B_RBC",
        "sex", "tube", "site", "other_inf",
    ]
    new_features = ["ratio_C_G_B_G", "diff_C_N_B_N", "corrected_WBC", "ratio_WBC_RBC_diff"]
    feature_cols = base_features + [f for f in new_features if f in df.columns]

    return df[feature_cols].values, df[CFG.label_col].values, feature_cols


# =============================================================
# 辅助函数
# =============================================================
def save_attention_to_csv(attn_maps, feature_names, output_path, layer_idx=0):
    if layer_idx >= len(attn_maps):
        return
    att_mean = np.mean(attn_maps[layer_idx], axis=(0, 1))
    pd.DataFrame(att_mean, columns=feature_names, index=feature_names).to_csv(
        output_path, encoding="utf-8-sig"
    )


def save_fold_predictions(results, output_path):
    pd.DataFrame({
        "y_true":    results["labels"],
        "y_prob":    results["predictions"],
        "threshold": results["threshold"],
    }).to_csv(output_path, index=False, encoding="utf-8-sig")


def summarize_results(fold_results):
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["threshold"] = CFG.fixed_threshold
    return summary


# =============================================================
# 主流程
# =============================================================
def main() -> None:
    set_seed(CFG.seed)
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    df = pd.read_excel(CFG.data_path)
    X, y, feature_cols = build_feature_matrix(df)
    print(f"特征数: {len(feature_cols)}  样本数: {len(y)}  阳性: {y.sum()}")

    device = CFG.device
    print(f"使用设备: {device}")

    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    fold_results, fold_summary_rows = [], []
    first_fold_saved = False

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'='*20} Fold {fold}/{CFG.n_splits} {'='*20}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val,   y_val   = X[val_idx],   y[val_idx]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val   = scaler.transform(X_val)

        sm = SMOTE(random_state=CFG.seed)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        print(f"  SMOTE后: {X_train_sm.shape[0]}条, 阳性{y_train_sm.sum()}个")

        train_loader = DataLoader(
            CSFDataset(X_train_sm, y_train_sm),
            batch_size=CFG.batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            CSFDataset(X_val, y_val),
            batch_size=CFG.batch_size, shuffle=False, num_workers=0
        )

        # 用原始训练集（SMOTE前）的分布计算 pos_weight
        # SMOTE 平衡了训练数据，但 val 集仍是原始分布，需要 pos_weight 校准决策边界
        orig_pos = y_train.sum()
        orig_neg = len(y_train) - orig_pos
        pw_value = float(orig_neg / orig_pos) if orig_pos > 0 else 1.0
        pos_weight = torch.tensor(pw_value, dtype=torch.float32).to(device)
        print(f"  原始 pos/neg = {orig_pos}/{orig_neg}, pos_weight = {pw_value:.2f}")

        model = ImprovedAMFormer(
            input_dim=len(feature_cols),
            embed_dim=CFG.embed_dim,
            n_heads=CFG.n_heads,
            n_layers=CFG.n_layers,
            top_k=CFG.top_k,
            dropout=CFG.dropout,
        )

        trainer   = ModelTrainer(model, device=device, pos_weight=pos_weight)
        model_path = output_dir / f"best_amformer_fold{fold}.pth"

        trainer.train(
            train_loader, val_loader,
            save_path=str(model_path),
            epochs=CFG.epochs,
            lr=CFG.lr,
            patience=CFG.patience,
        )

        # ── 修复原版 bug：evaluate() 返回 results ──
        results = trainer.evaluate(val_loader, threshold=CFG.fixed_threshold)
        fold_results.append(results)

        fold_summary = {
            "fold":      fold,
            "accuracy":  results["accuracy"],
            "precision": results["precision"],
            "recall":    results["recall"],
            "f1":        results["f1"],
            "auc":       results["auc"],
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
        pd.DataFrame({
            "epoch":      np.arange(len(trainer.train_losses)),
            "train_loss": trainer.train_losses,
            "val_loss":   trainer.val_losses,
        }).to_csv(output_dir / f"fold_{fold}_loss_curve.csv", index=False, encoding="utf-8-sig")
        with open(output_dir / f"fold_{fold}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print(
            f"[Fold {fold}] AUC={results['auc']:.4f}  "
            f"F1={results['f1']:.4f}  "
            f"Acc={results['accuracy']:.4f}  "
            f"Prec={results['precision']:.4f}  "
            f"Recall={results['recall']:.4f}"
        )
        print("混淆矩阵:", np.array(results["confusion_matrix"]))

        if not first_fold_saved:
            sample_x = torch.tensor(X_val[0], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, attn_maps = model(sample_x, return_attention=True)
            save_attention_to_csv(
                attn_maps, feature_cols,
                str(output_dir / "attention_layer0_fold1.csv"), layer_idx=0
            )
            first_fold_saved = True

    summary = summarize_results(fold_results)
    pd.DataFrame(fold_summary_rows).to_csv(
        output_dir / "all_folds_metrics.csv", index=False, encoding="utf-8-sig"
    )
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("5-Fold 汇总结果")
    print("=" * 50)
    for key, stat in summary.items():
        if key == "threshold":
            continue
        print(f"{key:>10}: {stat['mean']:.4f} ± {stat['std']:.4f}")
    print(f"\n结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
