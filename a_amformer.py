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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    data_path: str = "original.xlsx"
    output_dir: str = "41706results"
    label_col: str = "outcome"
    batch_size: int = 32
    epochs: int = 150
    lr: float = 8e-4
    warmup_epochs: int = 10
    patience: int = 25
    fixed_threshold: float = 0.50
    embed_dim: int = 192
    n_heads: int = 6
    n_layers: int = 4
    dropout: float = 0.15
    seed: int = 42
    n_splits: int = 5
    pos_weight_scale: float = 2.0
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────
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


# ── Model ─────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.size()
        residual = x
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.dropout(self.w_o(context))
        return self.layer_norm(output + residual), attn_weights


class FeatureInteractionLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        self.interaction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.interaction_net(x * self.feature_weights)


class ImprovedAMFormer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 192,
                 n_heads: int = 6, n_layers: int = 4, dropout: float = 0.15):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.input_embedding = nn.Sequential(
            nn.Linear(1, embed_dim), nn.ReLU(), nn.Dropout(dropout))
        self.feature_interaction = FeatureInteractionLayer(input_dim, embed_dim // 2)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, embed_dim) * 0.02)
        self.transformer_layers = nn.ModuleList(
            [MultiHeadAttention(embed_dim, n_heads, dropout) for _ in range(n_layers)])
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Dropout(dropout),
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
        batch_size = x.size(0)
        attention_maps = []
        interaction_features = self.feature_interaction(x)
        embedded = self.input_embedding(x.unsqueeze(-1))
        embedded = embedded + self.pos_encoding[:, :self.input_dim].expand(batch_size, -1, -1)
        for attn_layer, ff_layer, ln in zip(
                self.transformer_layers, self.feed_forward_layers, self.layer_norms):
            attn_output, attn_weights = attn_layer(embedded)
            attention_maps.append(attn_weights.detach().cpu().numpy())
            ff_output = ff_layer(attn_output)
            embedded = ln(ff_output + attn_output)
        pooled = self.global_pool(embedded.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(torch.cat([pooled, interaction_features], dim=1)).squeeze(-1)
        if return_attention:
            return logits, attention_maps
        return logits


# ── LR Scheduler: Warmup + Cosine ────────────────────────────────────────────
def make_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── ModelTrainer ──────────────────────────────────────────────────────────────
class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu",
                 pos_weight: torch.Tensor | None = None):
        self.model = model.to(device)
        self.device = device
        self.pos_weight = pos_weight
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _make_criterion(self):
        return FocalLoss(
            alpha=CFG.focal_alpha,
            gamma=CFG.focal_gamma,
            pos_weight=self.pos_weight,
        )

    def train_epoch(self, train_loader: DataLoader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_features, batch_labels in tqdm(train_loader, desc="Training", leave=False):
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(batch_features), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader, criterion):
        self.model.eval()
        total_loss, all_probs, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                logits = self.model(batch_features)
                total_loss += criterion(logits, batch_labels).item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        return total_loss / len(val_loader), np.array(all_probs), np.array(all_labels)

    def train(self, train_loader, val_loader, save_path,
              epochs=150, lr=8e-4, patience=25, warmup_epochs=10) -> None:
        criterion = self._make_criterion()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = make_scheduler(optimizer, warmup_epochs, epochs)

        best_val_auc = 0.0
        patience_counter = 0
        torch.save(self.model.state_dict(), save_path)

        print("开始训练 AMFormer...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_probs, val_labels = self.validate(val_loader, criterion)
            val_auc = roc_auc_score(val_labels, val_probs)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                print(f"Epoch {epoch:03d}: loss={train_loss:.4f} "
                      f"val_auc={val_auc:.4f} best={best_val_auc:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (best AUC={best_val_auc:.4f})")
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        print("训练完成！")


# ── DistillationTrainer ───────────────────────────────────────────────────────
class DistillationTrainer(ModelTrainer):
    def __init__(self, model, device="cpu", pos_weight=None,
                 alpha=0.8, temperature=1.5, save_path="best_distilled.pth"):
        super().__init__(model, device, pos_weight)
        self.alpha = alpha
        self.temperature = temperature
        self.save_path = save_path

    def train_epoch_distill(self, train_loader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        # 解包4列：features, hard_labels, soft_labels, confidence_weights
        for batch_features, batch_hard_labels, batch_soft_labels, batch_weights in tqdm(
                train_loader, desc="Distill", leave=False):
            batch_features = batch_features.to(self.device)
            batch_hard_labels = batch_hard_labels.to(self.device)
            batch_soft_labels = batch_soft_labels.to(self.device)
            batch_weights = batch_weights.to(self.device)

            optimizer.zero_grad()
            logits = self.model(batch_features)

            hard_loss = criterion(logits, batch_hard_labels)
            student_probs = torch.sigmoid(logits / self.temperature)
            # 置信度加权：教师越确定的样本软标签权重越高
            soft_loss_per = F.binary_cross_entropy(
                student_probs, batch_soft_labels, reduction="none")
            soft_loss = (soft_loss_per * batch_weights).mean()

            loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def train_distill(self, train_loader, val_loader,
                      epochs=150, lr=8e-4, patience=25) -> None:
        criterion = self._make_criterion()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = make_scheduler(optimizer, CFG.warmup_epochs, epochs)

        best_val_auc = 0.0
        patience_counter = 0
        torch.save(self.model.state_dict(), self.save_path)

        print(f"  蒸馏训练 (alpha={self.alpha}, T={self.temperature})...")
        for epoch in range(epochs):
            train_loss = self.train_epoch_distill(train_loader, optimizer, criterion)
            _, val_probs, val_labels = self.validate(val_loader, criterion)
            val_auc = roc_auc_score(val_labels, val_probs)
            scheduler.step()

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:03d}: train_loss={train_loss:.4f} "
                      f"val_auc={val_auc:.4f} best={best_val_auc:.4f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch} (best AUC={best_val_auc:.4f})")
                break

        self.model.load_state_dict(torch.load(self.save_path, map_location=self.device))


# ── XGBoost Teacher ───────────────────────────────────────────────────────────
def train_xgb_teacher(X_train: np.ndarray, y_train: np.ndarray):
    from xgboost import XGBClassifier
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    teacher = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        scale_pos_weight=neg / pos, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=2.0, min_child_weight=3,
        eval_metric='auc', random_state=42, n_jobs=1, tree_method='hist'
    )
    teacher.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, teacher.predict_proba(X_train)[:, 1])
    print(f"  XGBoost教师训练AUC: {train_auc:.4f}")
    return teacher


def generate_soft_labels(teacher, X_train: np.ndarray,
                          temperature: float = 1.5):
    """
    返回 (soft_labels, confidence_weights)
    - soft_labels: 温度缩放后的软概率
    - confidence_weights: 教师越确定权重越高，不确定样本权重接近0
    """
    probs = teacher.predict_proba(X_train)[:, 1]
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    logits = np.log(probs / (1 - probs))
    soft = 1 / (1 + np.exp(-logits / temperature))
    confidence = np.abs(probs - 0.5) * 2   # 0=最不确定, 1=最确定
    weights = confidence ** 2              # 平方加速衰减
    return soft.astype(np.float32), weights.astype(np.float32)


def make_distill_loader(X: np.ndarray, y_hard: np.ndarray,
                         y_soft: np.ndarray, weights: np.ndarray,
                         batch_size: int = 32) -> DataLoader:
    from torch.utils.data import TensorDataset
    ds = TensorDataset(
        torch.tensor(X,       dtype=torch.float32),
        torch.tensor(y_hard,  dtype=torch.float32),
        torch.tensor(y_soft,  dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_model(model: nn.Module, data_loader: DataLoader,
                   device: str, threshold: float = 0.50):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            logits = model(batch_features.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(batch_labels.numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    binary_preds = (all_probs >= threshold).astype(int)
    cm = confusion_matrix(all_labels, binary_preds)
    return {
        "accuracy":        float(accuracy_score(all_labels, binary_preds)),
        "precision":       float(precision_score(all_labels, binary_preds, zero_division=0)),
        "recall":          float(recall_score(all_labels, binary_preds, zero_division=0)),
        "f1":              float(f1_score(all_labels, binary_preds, zero_division=0)),
        "auc":             float(roc_auc_score(all_labels, all_probs)),
        "threshold":       float(threshold),
        "confusion_matrix": cm.tolist(),
        "labels":          all_labels.tolist(),
        "predictions":     all_probs.tolist(),
    }


def save_attention_to_csv(attn_maps, feature_names, output_path, layer_idx=0):
    if layer_idx >= len(attn_maps):
        return
    att_mean = np.mean(attn_maps[layer_idx], axis=(0, 1))
    pd.DataFrame(att_mean, columns=feature_names,
                 index=feature_names).to_csv(output_path, encoding="utf-8-sig")


def build_feature_matrix(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for col in ["C_WBC", "C_RBC", "C_P", "B_CRP", "B_WBC", "B_PCT", "B_AC", "B_RBC"]:
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
        df["ratio_WBC_RBC_diff"] = (df["B_WBC"] / (df["B_RBC"] + eps)
                                    - df["C_WBC"] / (df["C_RBC"] + eps))
    base = ["age", "C_G", "C_WBC", "C_RBC", "C_P", "C_N", "transparency", "GCS", "tem",
            "B_G", "B_CRP", "B_WBC", "B_N", "B_Lym", "B_PCT", "B_AC", "B_RBC",
            "sex", "tube", "site", "other_inf"]
    new = ["ratio_C_G_B_G", "diff_C_N_B_N", "corrected_WBC", "ratio_WBC_RBC_diff"]
    feature_cols = base + [f for f in new if f in df.columns]
    return df[feature_cols].values, df[CFG.label_col].values, feature_cols


def save_fold_predictions(results: dict, output_path: str) -> None:
    pd.DataFrame({
        "y_true": results["labels"],
        "y_prob": results["predictions"],
        "threshold": results["threshold"],
    }).to_csv(output_path, index=False, encoding="utf-8-sig")


def summarize_results(fold_results: list[dict]) -> dict:
    summary = {}
    for key in ["accuracy", "precision", "recall", "f1", "auc"]:
        vals = [r[key] for r in fold_results]
        summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    summary["threshold"] = CFG.fixed_threshold
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    set_seed(CFG.seed)
    output_dir = Path(CFG.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("加载数据...")
    df = pd.read_excel(CFG.data_path)
    X, y, feature_cols = build_feature_matrix(df)

    device = CFG.device
    print(f"使用设备: {device}")
    skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

    fold_results: list[dict] = []
    fold_summary_rows = []
    distill_fold_results: list[dict] = []
    first_fold_saved = False

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'=' * 20} Fold {fold} / {CFG.n_splits} {'=' * 20}")
        X_train_orig, y_train_orig = X[train_idx], y[train_idx]
        X_val_orig, y_val_orig = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_orig = scaler.fit_transform(X_train_orig)
        X_val_orig = scaler.transform(X_val_orig)

        sm = SMOTE(random_state=42)
        X_train_smote, y_train_smote = sm.fit_resample(X_train_orig, y_train_orig)
        print(f"  SMOTE后训练集: {X_train_smote.shape[0]}条, 阳性{y_train_smote.sum()}个")

        train_dataset = CSFDataset(X_train_smote, y_train_smote)
        val_dataset = CSFDataset(X_val_orig, y_val_orig)
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size,
                                  shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size,
                                shuffle=False, num_workers=0)

        pos_count = y_train_smote.sum()
        neg_count = len(y_train_smote) - pos_count
        pos_weight_value = (neg_count / pos_count) * CFG.pos_weight_scale
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)

        # ── 基础 AMFormer ──
        model = ImprovedAMFormer(
            input_dim=len(feature_cols), embed_dim=CFG.embed_dim,
            n_heads=CFG.n_heads, n_layers=CFG.n_layers, dropout=CFG.dropout)
        trainer = ModelTrainer(model, device=device, pos_weight=pos_weight)
        model_path = output_dir / f"best_amformer_fold{fold}.pth"
        trainer.train(train_loader, val_loader, save_path=str(model_path),
                      epochs=CFG.epochs, lr=CFG.lr,
                      patience=CFG.patience, warmup_epochs=CFG.warmup_epochs)

        results = evaluate_model(model, val_loader, device=device,
                                  threshold=CFG.fixed_threshold)
        fold_results.append(results)

        fold_summary = {
            "fold": fold,
            "accuracy": results["accuracy"], "precision": results["precision"],
            "recall": results["recall"], "f1": results["f1"], "auc": results["auc"],
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
        pd.DataFrame({"epoch": np.arange(len(trainer.train_losses)),
                      "train_loss": trainer.train_losses,
                      "val_loss": trainer.val_losses}).to_csv(
            output_dir / f"fold_{fold}_loss_curve.csv", index=False, encoding="utf-8-sig")
        with open(output_dir / f"fold_{fold}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print(f"[Fold {fold}] 基线AMFormer | "
              f"Acc={results['accuracy']:.4f} Prec={results['precision']:.4f} "
              f"Recall={results['recall']:.4f} F1={results['f1']:.4f} "
              f"AUC={results['auc']:.4f}")
        print("混淆矩阵:")
        print(np.array(results["confusion_matrix"]))

        if not first_fold_saved:
            sample_x = torch.tensor(X_val_orig[0], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, attn_maps = model(sample_x, return_attention=True)
            save_attention_to_csv(attn_maps, feature_cols,
                                  str(output_dir / "attention_layer0_fold1.csv"))
            first_fold_saved = True

        # ── 蒸馏实验 ──
        print(f"\n[Fold {fold}] 开始xgb→AMFormer知识蒸馏...")

        # 教师在 SMOTE 数据上训练（与软标签数据一致，避免错位）
        xgb_teacher = train_xgb_teacher(X_train_smote, y_train_smote)
        xgb_val_auc = roc_auc_score(y_val_orig,
                                     xgb_teacher.predict_proba(X_val_orig)[:, 1])
        print(f"xgb教师验证AUC: {xgb_val_auc:.4f}")

        # 生成软标签 + 置信度权重（tuple，必须解包）
        soft_labels, distill_weights = generate_soft_labels(
            xgb_teacher, X_train_smote, temperature=1.5)

        distill_loader = make_distill_loader(
            X_train_smote, y_train_smote, soft_labels, distill_weights, CFG.batch_size)

        val_dataset_d = CSFDataset(X_val_orig, y_val_orig)
        val_loader_d = DataLoader(val_dataset_d, batch_size=CFG.batch_size,
                                  shuffle=False, num_workers=0)

        model_d = ImprovedAMFormer(
            input_dim=len(feature_cols), embed_dim=CFG.embed_dim,
            n_heads=CFG.n_heads, n_layers=CFG.n_layers, dropout=CFG.dropout)
        save_path_d = str(output_dir / f"best_distilled_fold{fold}.pth")
        d_trainer = DistillationTrainer(
            model_d, device=device, pos_weight=pos_weight,
            alpha=0.8, temperature=1.5, save_path=save_path_d)
        d_trainer.train_distill(distill_loader, val_loader_d,
                                 epochs=CFG.epochs, lr=CFG.lr, patience=CFG.patience)

        results_d = evaluate_model(model_d, val_loader_d, device=device,
                                    threshold=CFG.fixed_threshold)
        print(f"[Fold {fold}] 蒸馏AMFormer | "
              f"Acc={results_d['accuracy']:.4f} Prec={results_d['precision']:.4f} "
              f"Recall={results_d['recall']:.4f} F1={results_d['f1']:.4f} "
              f"AUC={results_d['auc']:.4f}")

        distill_summary = {
            "fold": fold, "xgb_teacher_val_auc": float(xgb_val_auc),
            "accuracy": results_d["accuracy"], "precision": results_d["precision"],
            "recall": results_d["recall"], "f1": results_d["f1"], "auc": results_d["auc"],
            "threshold": CFG.fixed_threshold,
            "tn": results_d["confusion_matrix"][0][0],
            "fp": results_d["confusion_matrix"][0][1],
            "fn": results_d["confusion_matrix"][1][0],
            "tp": results_d["confusion_matrix"][1][1],
        }
        with open(output_dir / f"fold_{fold}_distill_metrics.json", "w") as f:
            json.dump(distill_summary, f, indent=2)
        distill_fold_results.append(results_d)

        del model_d, d_trainer, xgb_teacher, distill_loader, val_loader_d
        del model, trainer, train_loader, val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 汇总 ──
    summary = summarize_results(fold_results)
    pd.DataFrame(fold_summary_rows).to_csv(
        output_dir / "all_folds_metrics.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"基础AMFormer 5-Fold 汇总（阈值={CFG.fixed_threshold}）")
    print("=" * 50)
    for key, stat in summary.items():
        if key == "threshold":
            continue
        print(f"{key}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    distill_summary_out = summarize_results(distill_fold_results)
    with open(output_dir / "distill_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(distill_summary_out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print(f"蒸馏AMFormer 5-Fold 汇总（阈值={CFG.fixed_threshold}）")
    print("=" * 50)
    for key, stat in distill_summary_out.items():
        if key == "threshold":
            continue
        print(f"{key}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    print(f"\n结果已保存到: {output_dir.resolve()}")


if __name__ == "__main__":
    main()