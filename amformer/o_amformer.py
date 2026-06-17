"""
AMFormer 优化版
目标：AUC 0.92，Recall 0.79
核心改动（共6处，每处都有标注）：
  [1] Focal Loss 替换 BCE → 专注难分样本，提升AUC
  [2] 早停改为监控 val_AUC（不是 val_loss）→ 保存真正最优的模型
  [3] 阈值 0.50 → 0.35 → 直接提升 Recall
  [4] pos_weight_scale 2.0（SMOTE后仍施压）→ 进一步偏向阳性
  [5] Warmup + CosineAnnealing 替换 ReduceLROnPlateau → 更稳定收敛
  [6] 模型加深：n_layers 3→4，embed_dim 128→192 → 表达力更强
"""

import os, json, pickle, random, warnings
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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
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


@dataclass
class Config:
    data_path:        str   = "original.xlsx"
    output_dir:       str   = "amformer_optimized_results"
    label_col:        str   = "outcome"
    batch_size:       int   = 32
    epochs:           int   = 150
    lr:               float = 8e-4
    warmup_epochs:    int   = 10           # [5]
    patience:         int   = 25
    fixed_threshold:  float = 0.35         # [3]
    embed_dim:        int   = 192          # [6]
    n_heads:          int   = 6            # 192/6=32
    n_layers:         int   = 4            # [6]
    dropout:          float = 0.15
    seed:             int   = 42
    n_splits:         int   = 5
    pos_weight_scale: float = 2.0          # [4]
    focal_gamma:      float = 2.0          # [1]
    focal_alpha:      float = 0.75         # [1]
    device:           str   = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# ── [1] Focal Loss ─────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce   = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ── 数据集 ─────────────────────────────────────────────────────────
class CSFDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = None if labels is None else torch.tensor(labels, dtype=torch.float32)

    def __len__(self): return len(self.features)

    def __getitem__(self, idx):
        if self.labels is None: return self.features[idx]
        return self.features[idx], self.labels[idx]


# ── 模型 ───────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k     = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, S, D = x.size()
        res = x
        q = self.w_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        w = self.dropout(F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5), dim=-1))
        ctx = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, S, D)
        return self.layer_norm(self.dropout(self.w_o(ctx)) + res), w


class FeatureInteractionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fw  = nn.Parameter(torch.ones(input_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(0.2),
        )

    def forward(self, x): return self.net(x * self.fw)


class ImprovedAMFormer(nn.Module):
    def __init__(self, input_dim, embed_dim=192, n_heads=6, n_layers=4, dropout=0.15):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_embedding     = nn.Sequential(
            nn.Linear(1, embed_dim), nn.GELU(), nn.Dropout(dropout))
        self.feature_interaction = FeatureInteractionLayer(input_dim, embed_dim // 2)
        self.pos_encoding        = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)
        self.transformer_layers  = nn.ModuleList(
            [MultiHeadAttention(embed_dim, n_heads, dropout) for _ in range(n_layers)])
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier  = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),             nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        B, amaps = x.size(0), []
        inter = self.feature_interaction(x)
        emb   = self.input_embedding(x.unsqueeze(-1))
        emb   = emb + self.pos_encoding[:, :self.input_dim].expand(B, -1, -1)
        for attn, ff, ln in zip(self.transformer_layers, self.feed_forward_layers, self.layer_norms):
            ao, w = attn(emb)
            amaps.append(w.detach().cpu().numpy())
            emb = ln(ff(ao) + ao)
        logits = self.classifier(
            torch.cat([self.global_pool(emb.transpose(1,2)).squeeze(-1), inter], dim=1)
        ).squeeze(-1)
        return (logits, amaps) if return_attention else logits


# ── [5] Warmup + Cosine LR ─────────────────────────────────────────
def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        p = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Trainer ────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self, model, device="cpu", criterion=None):
        self.model     = model.to(device)
        self.device    = device
        self.criterion = criterion
        self.train_losses, self.val_aucs = [], []

    def _train_epoch(self, loader, optimizer):
        self.model.train()
        total = 0.0
        for X, y in tqdm(loader, desc="train", leave=False):
            X, y = X.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            self.criterion(self.model(X), y).backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total += self.criterion(self.model(X), y).item()
        return total / len(loader)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        probs, labels = [], []
        for X, y in loader:
            probs.extend(torch.sigmoid(self.model(X.to(self.device))).cpu().numpy())
            labels.extend(y.numpy())
        return np.array(probs), np.array(labels)

    # [2] 监控 val_AUC 而非 val_loss
    def train(self, train_loader, val_loader, save_path,
              epochs=150, lr=8e-4, patience=25, warmup_epochs=10):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler  = get_scheduler(optimizer, warmup_epochs, epochs)
        best_auc, patience_ctr = 0.0, 0
        torch.save(self.model.state_dict(), save_path)

        for epoch in range(epochs):
            loss          = self._train_epoch(train_loader, optimizer)
            probs, labels = self._validate(val_loader)
            auc           = roc_auc_score(labels, probs)
            scheduler.step()
            self.train_losses.append(loss)
            self.val_aucs.append(auc)

            if auc > best_auc:                              # ← [2]
                best_auc, patience_ctr = auc, 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_ctr += 1

            if epoch % 5 == 0:
                print(f"  Ep {epoch:03d} | loss={loss:.4f} | AUC={auc:.4f} "
                      f"| best={best_auc:.4f} | lr={scheduler.get_last_lr()[0]:.1e}")
            if patience_ctr >= patience:
                print(f"  Early stop ep={epoch}  best_AUC={best_auc:.4f}")
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        return best_auc


# ── 评估 ───────────────────────────────────────────────────────────
def evaluate_model(model, loader, device, threshold=0.35):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            probs.extend(torch.sigmoid(model(X.to(device))).cpu().numpy())
            labels.extend(y.numpy())
    probs, labels = np.array(probs), np.array(labels)
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(labels, preds)
    return {
        "accuracy":         float(accuracy_score(labels, preds)),
        "precision":        float(precision_score(labels, preds, zero_division=0)),
        "recall":           float(recall_score(labels, preds, zero_division=0)),
        "f1":               float(f1_score(labels, preds, zero_division=0)),
        "auc":              float(roc_auc_score(labels, probs)),
        "threshold":        threshold,
        "confusion_matrix": cm.tolist(),
        "labels":           labels.tolist(),
        "predictions":      probs.tolist(),
    }


# ── 特征工程 ───────────────────────────────────────────────────────
def build_feature_matrix(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num] = df[num].fillna(df[num].median())
    for col in ["C_WBC","C_RBC","C_P","B_CRP","B_WBC","B_PCT","B_AC","B_RBC"]:
        if col in df.columns: df[col] = np.log1p(df[col].clip(lower=0))
    eps = 1e-6
    if "C_G" in df.columns and "B_G" in df.columns:
        df["ratio_C_G_B_G"]  = df["C_G"] / (df["B_G"] + eps)
    if "C_N" in df.columns and "B_N" in df.columns:
        df["diff_C_N_B_N"]   = df["C_N"] - df["B_N"]
    if all(c in df.columns for c in ["C_WBC","B_WBC","C_RBC","B_RBC"]):
        df["corrected_WBC"]  = df["C_WBC"] - df["B_WBC"]*df["C_RBC"]/(df["B_RBC"]+eps)
    if all(c in df.columns for c in ["B_WBC","B_RBC","C_WBC","C_RBC"]):
        df["ratio_WBC_RBC_diff"] = (df["B_WBC"]/(df["B_RBC"]+eps)
                                   - df["C_WBC"]/(df["C_RBC"]+eps))
    base = ["age","C_G","C_WBC","C_RBC","C_P","C_N","transparency","GCS","tem",
            "B_G","B_CRP","B_WBC","B_N","B_Lym","B_PCT","B_AC","B_RBC",
            "sex","tube","site","other_inf"]
    new  = ["ratio_C_G_B_G","diff_C_N_B_N","corrected_WBC","ratio_WBC_RBC_diff"]
    cols = base + [f for f in new if f in df.columns]
    return df[cols].values.astype(np.float32), df["outcome"].values, cols


def summarize(fold_results):
    keys = ["accuracy","precision","recall","f1","auc"]
    return {k: {"mean": float(np.mean([r[k] for r in fold_results])),
                "std":  float(np.std ([r[k] for r in fold_results]))} for k in keys}


def save_attention_csv(amaps, feature_names, path, layer_idx=0):
    if layer_idx >= len(amaps): return
    att = np.mean(amaps[layer_idx], axis=(0, 1))
    pd.DataFrame(att, columns=feature_names, index=feature_names).to_csv(
        path, encoding="utf-8-sig")


# ── 主流程 ─────────────────────────────────────────────────────────
def main():
    set_seed(CFG.seed)
    out = Path(CFG.output_dir); out.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    df = pd.read_excel(CFG.data_path)
    X, y, feature_cols = build_feature_matrix(df)
    print(f"样本:{len(y)}  特征:{len(feature_cols)}  阳:{y.sum()}  阴:{(y==0).sum()}")
    print(f"改动: Focal(α={CFG.focal_alpha},γ={CFG.focal_gamma}) | "
          f"阈值={CFG.fixed_threshold} | pos_scale={CFG.pos_weight_scale} | "
          f"dim={CFG.embed_dim} | layers={CFG.n_layers}")

    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
    fold_results, fold_rows, first_saved = [], [], False

    for fold, (tr_i, val_i) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'='*22} Fold {fold}/{CFG.n_splits} {'='*22}")
        X_tr, y_tr   = X[tr_i], y[tr_i]
        X_val, y_val = X[val_i], y[val_i]

        sc    = StandardScaler()
        X_tr  = sc.fit_transform(X_tr)
        X_val = sc.transform(X_val)

        X_tr_s, y_tr_s = SMOTE(random_state=CFG.seed).fit_resample(X_tr, y_tr)
        print(f"  SMOTE后:{len(X_tr_s)}条  阳:{y_tr_s.sum()}")

        tr_ld  = DataLoader(CSFDataset(X_tr_s, y_tr_s),
                            batch_size=CFG.batch_size, shuffle=True,  num_workers=0)
        val_ld = DataLoader(CSFDataset(X_val, y_val),
                            batch_size=CFG.batch_size, shuffle=False, num_workers=0)

        pos = y_tr_s.sum(); neg = len(y_tr_s) - pos
        pw  = torch.tensor((neg/pos)*CFG.pos_weight_scale,
                            dtype=torch.float32).to(CFG.device)
        criterion = FocalLoss(CFG.focal_alpha, CFG.focal_gamma, pw)   # [1]

        model   = ImprovedAMFormer(len(feature_cols), CFG.embed_dim,
                                   CFG.n_heads, CFG.n_layers, CFG.dropout)
        trainer = ModelTrainer(model, CFG.device, criterion)
        best_auc = trainer.train(tr_ld, val_ld, str(out/f"best_fold{fold}.pth"),
                                 CFG.epochs, CFG.lr, CFG.patience, CFG.warmup_epochs)

        res = evaluate_model(model, val_ld, CFG.device, CFG.fixed_threshold)
        fold_results.append(res)
        row = {"fold": fold, **{k: res[k] for k in
               ["accuracy","precision","recall","f1","auc","threshold"]},
               "tn": res["confusion_matrix"][0][0], "fp": res["confusion_matrix"][0][1],
               "fn": res["confusion_matrix"][1][0], "tp": res["confusion_matrix"][1][1]}
        fold_rows.append(row)

        print(f"[Fold {fold}] AUC={res['auc']:.4f}  Recall={res['recall']:.4f}  "
              f"Prec={res['precision']:.4f}  F1={res['f1']:.4f}")
        print(f"  混淆矩阵: {np.array(res['confusion_matrix'])}")

        if not first_saved:
            sx = torch.tensor(X_val[0], dtype=torch.float32).unsqueeze(0).to(CFG.device)
            with torch.no_grad(): _, am = model(sx, return_attention=True)
            save_attention_csv(am, feature_cols, str(out/"attention_fold1.csv"))
            first_saved = True

        with open(out/f"fold_{fold}_scaler.pkl",  "wb") as f: pickle.dump(sc, f)
        with open(out/f"fold_{fold}_metrics.json","w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)

    summary = summarize(fold_results)
    pd.DataFrame(fold_rows).to_csv(out/"all_folds.csv", index=False, encoding="utf-8-sig")
    with open(out/"summary.json","w",encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*55)
    print(f"优化AMFormer 5-Fold 汇总（阈值={CFG.fixed_threshold}）")
    print("="*55)
    for k, v in summary.items():
        tag = ""
        if k == "auc"    and v["mean"] >= 0.92: tag = "  ✅ 达标"
        if k == "recall" and v["mean"] >= 0.79: tag = "  ✅ 达标"
        if k == "auc"    and v["mean"] <  0.92: tag = f"  ❌ 差{0.92-v['mean']:.4f}"
        if k == "recall" and v["mean"] <  0.79: tag = f"  ❌ 差{0.79-v['mean']:.4f}"
        print(f"  {k:12s}: {v['mean']:.4f} ± {v['std']:.4f}{tag}")
    print(f"\n结果保存至: {out.resolve()}")


if __name__ == "__main__":
    main()