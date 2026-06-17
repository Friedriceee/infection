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
    epochs: int = 100
    lr: float = 1e-3
    patience: int = 15
    fixed_threshold: float = 0.5
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.15
    seed: int = 42
    n_splits: int = 5
    pos_weight_scale: float = 2.0
    focal_gamma:      float = 2.0
    focal_alpha:      float = 0.75
    warmup_epochs:    int   = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()


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
        probs   = torch.sigmoid(logits)
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
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
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output, attn_weights


class FeatureInteractionLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        self.interaction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weighted_features = x * self.feature_weights
        return self.interaction_net(weighted_features)


class ImprovedAMFormer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.input_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.feature_interaction = FeatureInteractionLayer(input_dim, embed_dim // 2)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, embed_dim))

        self.transformer_layers = nn.ModuleList(
            [MultiHeadAttention(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.feed_forward_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_layers)])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size = x.size(0)
        attention_maps: list[np.ndarray] = []

        interaction_features = self.feature_interaction(x)
        x_tokens = x.unsqueeze(-1)
        embedded = self.input_embedding(x_tokens)
        embedded = embedded + self.pos_encoding[:, : self.input_dim, :].expand(batch_size, -1, -1)

        for attn_layer, ff_layer, ln in zip(
            self.transformer_layers, self.feed_forward_layers, self.layer_norms
        ):
            attn_output, attn_weights = attn_layer(embedded)
            attention_maps.append(attn_weights.detach().cpu().numpy())
            ff_output = ff_layer(attn_output)
            embedded = ln(ff_output + attn_output)

        pooled = self.global_pool(embedded.transpose(1, 2)).squeeze(-1)
        combined_features = torch.cat([pooled, interaction_features], dim=1)
        logits = self.classifier(combined_features).squeeze(-1)

        if return_attention:
            return logits, attention_maps
        return logits


class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cpu", pos_weight: torch.Tensor | None = None):
        self.model = model.to(device)
        self.device = device
        self.pos_weight = pos_weight
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def train_epoch(self, train_loader: DataLoader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        for batch_features, batch_labels in tqdm(train_loader, desc="Training", leave=False):
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader, criterion):
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        return total_loss / len(val_loader), np.array(all_probs), np.array(all_labels)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_path: str,
        epochs: int = 100,
        lr: float = 1e-3,
        patience: int = 15,
    ) -> None:
        criterion = (
            nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            if self.pos_weight is not None
            else nn.BCEWithLogitsLoss()
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        torch.save(self.model.state_dict(), save_path)

        print("开始训练基础 AMFormer 模型...")
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_probs, val_labels = self.validate(val_loader, criterion)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                val_auc = roc_auc_score(val_labels, val_probs)
                print(
                    f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}"
                )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        print("训练完成！")


#蒸馏

class DistillationTrainer(ModelTrainer):
    def __init__(self, model, device="cpu", pos_weight=None,
                 alpha=0.8, temperature=2.0, save_path="best_distilled.pth"):
        super().__init__(model, device, pos_weight)
        self.alpha = alpha
        self.temperature = temperature
        self.save_path = save_path

    def train_epoch_distill(self, train_loader, optimizer, criterion) -> float:
        self.model.train()
        total_loss = 0.0
        # Bug6 fix: 解包4个值（含置信度权重）
        for batch_features, batch_hard_labels, batch_soft_labels, batch_weights in tqdm(
            train_loader, desc="Distill", leave=False
        ):
            batch_features    = batch_features.to(self.device)
            batch_hard_labels = batch_hard_labels.to(self.device)
            batch_soft_labels = batch_soft_labels.to(self.device)
            batch_weights     = batch_weights.to(self.device)

            optimizer.zero_grad()
            logits = self.model(batch_features)

            hard_loss     = criterion(logits, batch_hard_labels)
            student_probs = torch.sigmoid(logits / self.temperature)
            # Bug6 fix: 置信度加权软标签 loss，不确定的样本软标签贡献接近0
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
                      epochs=100, lr=1e-3, patience=15) -> None:
        # Bug5 fix: 使用 FocalLoss 与基础模型一致
        criterion = FocalLoss(
            alpha=CFG.focal_alpha,
            gamma=CFG.focal_gamma,
            pos_weight=self.pos_weight,
        ) if self.pos_weight is not None else FocalLoss(
            alpha=CFG.focal_alpha,
            gamma=CFG.focal_gamma,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4)
        # Bug4 fix: 用 warmup+cosine 调度，监控 val_AUC
        def lr_lambda(epoch):
            warmup = CFG.warmup_epochs
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = (epoch - warmup) / max(epochs - warmup, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val_auc     = 0.0          # Bug4 fix: 监控 AUC 而非 loss
        patience_counter = 0
        torch.save(self.model.state_dict(), self.save_path)

        print(f"  蒸馏训练 (alpha={self.alpha}, T={self.temperature})...")
        for epoch in range(epochs):
            train_loss = self.train_epoch_distill(train_loader, optimizer, criterion)
            val_loss, val_probs, val_labels = self.validate(val_loader, criterion)
            val_auc = roc_auc_score(val_labels, val_probs)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step()

            # Bug4 fix: 保存 AUC 最高的 checkpoint
            if val_auc > best_val_auc:
                best_val_auc     = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:03d}: Train={train_loss:.4f} "
                      f"Val_AUC={val_auc:.4f} Best={best_val_auc:.4f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch} (best AUC={best_val_auc:.4f})")
                break

        self.model.load_state_dict(
            torch.load(self.save_path, map_location=self.device))

def train_xgb_teacher(X_train: np.ndarray, y_train: np.ndarray):
    from xgboost import XGBClassifier
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    teacher = XGBClassifier(
        n_estimators=300,
        max_depth=4,          # 4→3，限制树深度
        learning_rate=0.03,   # 0.1→0.05，更慢收敛
        scale_pos_weight=neg / pos,
        subsample=0.8,        # 新增：行采样
        colsample_bytree=0.8, # 新增：列采样
        reg_alpha=0.1,        # 新增：L1正则
        reg_lambda=2.0,       # 新增：L2正则
        min_child_weight=3,   # 新增：叶节点最小样本数
        eval_metric='auc',
        random_state=42,
        n_jobs=1,
        tree_method='hist'
    )
    teacher.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, teacher.predict_proba(X_train)[:, 1])
    print(f"  XGBoost教师训练AUC: {train_auc:.4f}")
    return teacher

def generate_soft_labels(teacher, X_train: np.ndarray,
                          temperature: float = 1.5):
    """
    返回 (soft_labels, confidence_weights)
    soft_labels: 温度缩放后的软概率
    confidence_weights: 教师越确定权重越高，不确定样本权重接近0
    """
    probs = teacher.predict_proba(X_train)[:, 1]
    eps   = 1e-6
    probs = np.clip(probs, eps, 1 - eps)
    logits = np.log(probs / (1 - probs))
    soft   = 1 / (1 + np.exp(-logits / temperature))
    # 置信度权重：概率离0.5越远越确定，权重越高
    confidence = np.abs(probs - 0.5) * 2   # 0~1
    weights    = confidence ** 2            # 平方加速衰减
    return soft.astype(np.float32), weights.astype(np.float32)


def make_distill_loader(X: np.ndarray, y_hard: np.ndarray,
                         y_soft: np.ndarray, weights: np.ndarray,
                         batch_size: int = 32) -> DataLoader:
    # Bug1+Bug6 fix: 加入 weights 作为第4列，与 train_epoch_distill 对应
    from torch.utils.data import TensorDataset
    ds = TensorDataset(
        torch.tensor(X,       dtype=torch.float32),
        torch.tensor(y_hard,  dtype=torch.float32),
        torch.tensor(y_soft,  dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

def evaluate_model(model: nn.Module, data_loader: DataLoader, device: str, threshold: float = 0.50):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            logits = model(batch_features)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    binary_preds = (all_probs >= threshold).astype(int)

    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, binary_preds)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "labels": all_labels.tolist(),
        "predictions": all_probs.tolist(),
    }


def save_attention_to_csv(attn_maps, feature_names, output_path: str, layer_idx: int = 0) -> None:
    if layer_idx >= len(attn_maps):
        return
    att = attn_maps[layer_idx]
    att_mean = np.mean(att, axis=(0, 1))
    df = pd.DataFrame(att_mean, columns=feature_names, index=feature_names)
    df.to_csv(output_path, encoding="utf-8-sig")


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    num_cols_for_impute = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols_for_impute] = df[num_cols_for_impute].fillna(df[num_cols_for_impute].median())

    log_cols = ["C_WBC", "C_RBC", "C_P", "B_CRP", "B_WBC", "B_PCT", "B_AC", "B_RBC"]
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    epsilon = 1e-6
    if "C_G" in df.columns and "B_G" in df.columns:
        df["ratio_C_G_B_G"] = df["C_G"] / (df["B_G"] + epsilon)
    if "C_N" in df.columns and "B_N" in df.columns:
        df["diff_C_N_B_N"] = df["C_N"] - df["B_N"]
    if all(c in df.columns for c in ["C_WBC", "B_WBC", "C_RBC", "B_RBC"]):
        df["corrected_WBC"] = df["C_WBC"] - (
            df["B_WBC"] * df["C_RBC"] / (df["B_RBC"] + epsilon)
        )
    if all(c in df.columns for c in ["B_WBC", "B_RBC", "C_WBC", "C_RBC"]):
        df["ratio_WBC_RBC_diff"] = (
            df["B_WBC"] / (df["B_RBC"] + epsilon)
            - df["C_WBC"] / (df["C_RBC"] + epsilon)
        )

    base_features = [
        "age",
        "C_G",
        "C_WBC",
        "C_RBC",
        "C_P",
        "C_N",
        "transparency",
        "GCS",
        "tem",
        "B_G",
        "B_CRP",
        "B_WBC",
        "B_N",
        "B_Lym",
        "B_PCT",
        "B_AC",
        "B_RBC",
        "sex",
        "tube",
        "site",
        "other_inf",
    ]
    new_features = ["ratio_C_G_B_G", "diff_C_N_B_N", "corrected_WBC", "ratio_WBC_RBC_diff"]
    feature_cols = base_features + [f for f in new_features if f in df.columns]

    X = df[feature_cols].values
    y = df[CFG.label_col].values
    return X, y, feature_cols




def save_fold_predictions(results: dict, output_path: str) -> None:
    pred_df = pd.DataFrame({
        "y_true": results["labels"],
        "y_prob": results["predictions"],
        "threshold": results["threshold"],
    })
    pred_df.to_csv(output_path, index=False, encoding="utf-8-sig")


def summarize_results(fold_results: list[dict]) -> dict:
    metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
    summary = {}
    for key in metric_keys:
        values = [r[key] for r in fold_results]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    summary["threshold"] = CFG.fixed_threshold
    return summary


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
    first_fold_saved = False

    distill_fold_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'=' * 20} Fold {fold} / {CFG.n_splits} {'=' * 20}")
        X_train_orig, y_train_orig = X[train_idx], y[train_idx]
        X_val_orig, y_val_orig = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_orig = scaler.fit_transform(X_train_orig)
        X_val_orig = scaler.transform(X_val_orig)

        #smote
        
        sm = SMOTE(random_state=42)
        X_train_smote, y_train_smote = sm.fit_resample(X_train_orig, y_train_orig)
        print(f"  SMOTE后训练集: {X_train_smote.shape[0]}条, 阳性{y_train_smote.sum()}个")

        train_dataset = CSFDataset(X_train_smote, y_train_smote)
        val_dataset = CSFDataset(X_val_orig, y_val_orig)
        train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

        pos_count = y_train_smote.sum()
        neg_count = len(y_train_smote) - pos_count
        pos_weight_value = (neg_count / pos_count) * CFG.pos_weight_scale if pos_count > 0 else 1.0
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)

        model = ImprovedAMFormer(
            input_dim=len(feature_cols),
            embed_dim=CFG.embed_dim,
            n_heads=CFG.n_heads,
            n_layers=CFG.n_layers,
            dropout=CFG.dropout,
        )
        trainer = ModelTrainer(model, device=device, pos_weight=pos_weight)
        model_path = output_dir / f"best_amformer_fold{fold}.pth"
        trainer.train(
            train_loader,
            val_loader,
            save_path=str(model_path),
            epochs=CFG.epochs,
            lr=CFG.lr,
            patience=CFG.patience,
        )

        results = evaluate_model(model, val_loader, device=device, threshold=CFG.fixed_threshold)
        fold_results.append(results)

        fold_summary = {
            "fold": fold,
            "accuracy": results["accuracy"],
            "precision": results["precision"],
            "recall": results["recall"],
            "f1": results["f1"],
            "auc": results["auc"],
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
            "epoch": np.arange(len(trainer.train_losses)),
            "train_loss": trainer.train_losses,
            "val_loss": trainer.val_losses,
        }).to_csv(output_dir / f"fold_{fold}_loss_curve.csv", index=False, encoding="utf-8-sig")
        with open(output_dir / f"fold_{fold}_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

        print(
            f"[Fold {fold}] 基线AMFormer | "
            f"Acc={results['accuracy']:.4f} "
            f"Prec={results['precision']:.4f} "
            f"Recall={results['recall']:.4f} "
            f"F1={results['f1']:.4f} "
            f"AUC={results['auc']:.4f}"
        )
        print("混淆矩阵:")
        print(np.array(results["confusion_matrix"]))

        if not first_fold_saved:
            sample_x = torch.tensor(X_val_orig[0], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, attn_maps = model(sample_x, return_attention=True)
            save_attention_to_csv(
                attn_maps,
                feature_cols,
                str(output_dir / "attention_layer0_fold1.csv"),
                layer_idx=0,
            )
            first_fold_saved = True

      


#new
        # ── 蒸馏实验 ──────────────────────────
        print(f"\n[Fold {fold}] 开始xgb→AMFormer知识蒸馏...")

        # Bug2 fix: 教师在 SMOTE 数据上训练，和软标签数据一致
        xgb_teacher = train_xgb_teacher(X_train_smote, y_train_smote)
        xgb_val_auc = roc_auc_score(
            y_val_orig,
            xgb_teacher.predict_proba(X_val_orig)[:, 1]
        )
        print(f"xgb教师验证AUC: {xgb_val_auc:.4f}")

        # Bug1 fix: 正确解包 (soft_labels, weights) 元组，温度降到1.5更有区分度
        soft_labels, distill_weights = generate_soft_labels(
            xgb_teacher, X_train_smote, temperature=1.5)

        # Bug6 fix: weights 传入 loader
        distill_loader  = make_distill_loader(
             X_train_smote, y_train_smote, soft_labels, distill_weights, CFG.batch_size)



        # 3. 构建蒸馏DataLoader（重新拿train/val loader）
       
        val_dataset_d   = CSFDataset(X_val_orig,   y_val_orig)
        val_loader_d    = DataLoader(
            val_dataset_d, batch_size=CFG.batch_size,
            shuffle=False, num_workers=0)
        
        # 4. 训练蒸馏AMFormer
        model_d = ImprovedAMFormer(
            input_dim=len(feature_cols),
            embed_dim=CFG.embed_dim,
            n_heads=CFG.n_heads,
            n_layers=CFG.n_layers,
            dropout=CFG.dropout,
        )
        save_path_d = str(output_dir / f"best_distilled_fold{fold}.pth")
        d_trainer = DistillationTrainer(
            model_d, device=device, pos_weight=pos_weight,
            alpha=0.8, temperature=2.0, save_path=save_path_d
        )
        d_trainer.train_distill(
            distill_loader, val_loader_d,
            epochs=CFG.epochs, lr=CFG.lr, patience=CFG.patience
        )

        # 5. 评估蒸馏模型（固定阈值，和基线保持一致）
        results_d = evaluate_model(
            model_d, val_loader_d, device=device,
            threshold=CFG.fixed_threshold
        )

        print(
            f"[Fold {fold}] 蒸馏AMFormer | "
            f"Acc={results_d['accuracy']:.4f} "
            f"Prec={results_d['precision']:.4f} "
            f"Recall={results_d['recall']:.4f} "
            f"F1={results_d['f1']:.4f} "
            f"AUC={results_d['auc']:.4f}"
        )

        # 6. 保存蒸馏结果
        distill_summary = {
            "fold": fold,
            "xgb_teacher_val_auc": float(xgb_val_auc),
            "accuracy":  results_d["accuracy"],
            "precision": results_d["precision"],
            "recall":    results_d["recall"],
            "f1":        results_d["f1"],
            "auc":       results_d["auc"],
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

    summary = summarize_results(fold_results)
    pd.DataFrame(fold_summary_rows).to_csv(output_dir / "all_folds_metrics.csv", index=False, encoding="utf-8-sig")
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("固定阈值 0.50 的 5-Fold 汇总结果")
    print("=" * 50)
    for key, stat in summary.items():
        if key == "threshold":
            continue
        print(f"{key}: {stat['mean']:.4f} ± {stat['std']:.4f}")

    print(f"\n结果已保存到文件夹: {output_dir.resolve()}")

    distill_summary_out = summarize_results(distill_fold_results)
    with open(output_dir / "distill_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(distill_summary_out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("蒸馏AMFormer 5-Fold 汇总结果")
    print("=" * 50)
    for key, stat in distill_summary_out.items():
        if key == "threshold":
            continue
        print(f"{key}: {stat['mean']:.4f} ± {stat['std']:.4f}")


if __name__ == "__main__":
    main()