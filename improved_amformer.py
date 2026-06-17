"""
改进的AMFormer深度学习模型（医学场景优化版本）
- 使用 BCEWithLogitsLoss + pos_weight 提高对阳性样本的关注（减少漏诊）
- 在验证集上自动搜索最佳阈值（考虑 FN/FP 代价）
- 使用最佳阈值在测试集上评估
- 支持导出 Attention 权重，用于分析特征间关系
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ===========================
#  基础设置
# ===========================
torch.manual_seed(42)
np.random.seed(42)


# ===========================
#  数据集定义
# ===========================
class CSFDataset(Dataset):
    """脑脊液数据集类"""
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


# ===========================
#  模型组件：多头注意力
# ===========================
class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        返回:
          - output: (batch_size, seq_len, d_model)
          - attn_weights: (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.size()
        residual = x

        # 计算 Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(context)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + residual)

        return output, attn_weights


# ===========================
#  模型组件：特征交互层
# ===========================
class FeatureInteractionLayer(nn.Module):
    """特征交互层 - 实现特征间的交互学习"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 特征交互网络
        self.interaction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 可学习的特征重要性权重
        self.feature_weights = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        # x: (batch_size, input_dim)
        weighted_features = x * self.feature_weights  # 广播：每个特征一个权重
        interactions = self.interaction_net(weighted_features)
        return interactions


# ===========================
#  AMFormer 模型
# ===========================
class ImprovedAMFormer(nn.Module):
    """改进的AMFormer模型"""
    def __init__(self, input_dim, embed_dim=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # 用来存注意力权重（每次 forward 会刷新）
        self.attention_maps = []

        # 输入嵌入层（将原始特征映射到高维）
        # 改动：从 input_dim -> embed_dim 改为 1 -> embed_dim
        # 每个特征值视为一个独立的 token
        self.input_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 特征交互层（手工定义的特征组合）
        self.feature_interaction = FeatureInteractionLayer(input_dim, embed_dim // 2)

        # 位置编码（把每个特征当作一个 token）
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, embed_dim))

        # 多层 Transformer 编码器
        self.transformer_layers = nn.ModuleList([
            MultiHeadAttention(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # 前馈网络层
        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])

        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])

        # 全局池化和分类头
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 最后一层输出 logits（不加 Sigmoid）
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, return_attention=False):
        """
        x: (batch_size, input_dim)
        return_attention:
          - False: 返回 logits
          - True:  返回 (logits, attention_maps)
        """
        batch_size = x.size(0)

        # 清空上一轮的注意力记录
        self.attention_maps = []

        # 特征交互
        interaction_features = self.feature_interaction(x)  # (batch_size, embed_dim//2)

        # 输入嵌入
        # 改动：先把 x 变成 (B, L, 1)，再 embedding
        x_tokens = x.unsqueeze(-1)                  # (batch_size, input_dim, 1)
        embedded = self.input_embedding(x_tokens)   # (batch_size, input_dim, embed_dim)

        # 添加位置编码
        pos_enc = self.pos_encoding[:, :self.input_dim, :].expand(batch_size, -1, -1)
        embedded = embedded + pos_enc                # (batch_size, input_dim, embed_dim)

        # Transformer 层
        for attn_layer, ff_layer, ln in zip(
            self.transformer_layers, self.feed_forward_layers, self.layer_norms
        ):
            attn_output, attn_weights = attn_layer(embedded)
            # 保存该层的注意力矩阵，detach + cpu 方便后处理
            self.attention_maps.append(attn_weights.detach().cpu().numpy())

            ff_output = ff_layer(attn_output)
            embedded = ln(ff_output + attn_output)

        # 全局池化
        pooled = self.global_pool(embedded.transpose(1, 2)).squeeze(-1)  # (batch_size, embed_dim)

        # 拼接交互特征
        combined_features = torch.cat([pooled, interaction_features], dim=1)

        # 分类输出 logits
        logits = self.classifier(combined_features).squeeze(-1)  # (batch_size,)

        if return_attention:
            return logits, self.attention_maps
        return logits


# ===========================
#  训练器
# ===========================
class ModelTrainer:
    """模型训练器"""
    def __init__(self, model, device='cpu', pos_weight=None):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.pos_weight = pos_weight  # Tensor on device

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0

        for batch_features, batch_labels in tqdm(train_loader, desc="Training"):
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(batch_features)          # logits
            loss = criterion(logits, batch_labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)

                logits = self.model(batch_features)
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()

                probs = torch.sigmoid(logits)  # 转成概率
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        return total_loss / len(val_loader), np.array(all_probs), np.array(all_labels)

    def train(self, train_loader, val_loader, epochs=100, lr=1e-3, patience=15):
        # 使用 BCEWithLogitsLoss，并且对阳性样本加权
        if self.pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float('inf')
        patience_counter = 0

        print("开始训练AMFormer模型...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_probs, val_labels = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            # 记录最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_amformer_model.pth')
            else:
                patience_counter += 1

            if epoch % 5 == 0:
                val_auc = roc_auc_score(val_labels, val_probs)
                print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_amformer_model.pth'))
        print("训练完成！")


# ===========================
#  阈值搜索函数（FN 代价更高）
# ===========================
def find_best_threshold_constrained(labels, probs, min_recall=0.80):
    """约束优化：recall≥min_recall条件下precision最高的阈值"""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t = 0.5
    best_precision = 0.0

    for t in thresholds:
        binary = (probs >= t).astype(int)
        if binary.sum() == 0:
            continue
        tn, fp, fn, tp = confusion_matrix(labels, binary).ravel()
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if recall >= min_recall and precision > best_precision:
            best_precision = precision
            best_t = t

    # 如果没有任何阈值满足recall≥0.80，退而求其次取recall最高
    if best_precision == 0.0:
        print(f"  ⚠️  无法同时满足recall≥{min_recall}，取recall最高阈值")
        best_t = thresholds[0]

    return best_t


# ===========================
#  评估函数（在测试集上使用给定阈值）
# ===========================
def evaluate_model(model, test_loader, device='cpu', threshold=0.5):
    """评估模型性能（使用给定阈值）"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)
            logits = model(batch_features)
            probs = torch.sigmoid(logits)  # 转成概率

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 使用自定义阈值
    binary_preds = (all_probs >= threshold).astype(int)

    # 各种指标
    accuracy = accuracy_score(all_labels, binary_preds)
    precision = precision_score(all_labels, binary_preds)
    recall = recall_score(all_labels, binary_preds)
    f1 = f1_score(all_labels, binary_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, binary_preds)

    print(f"\n===== 使用阈值 {threshold:.3f} 的 AMFormer 模型评估（测试集）=====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"AUC:       {auc:.4f}")
    print("\n混淆矩阵 (TN, FP; FN, TP):")
    print(cm)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_probs,
        'labels': all_labels,
        'confusion_matrix': cm
    }


# ===========================
#  可选：画训练曲线
# ===========================
def plot_training_curves(trainer):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('amformer_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()



def visualize_attention(model, sample_x, feature_names, device='cpu', layer_idx=0):
    """
    可视化某一层的注意力热力图（多头平均）
    sample_x: numpy 向量 (input_dim,) 或 torch.Tensor(1, input_dim)
    feature_names: 特征名列表，长度 = input_dim
    """
    model.eval()

    if isinstance(sample_x, np.ndarray):
        sample_x = torch.tensor(sample_x, dtype=torch.float32)
    if sample_x.dim() == 1:
        sample_x = sample_x.unsqueeze(0)
    sample_x = sample_x.to(device)

    with torch.no_grad():
        logits, attn_maps = model(sample_x, return_attention=True)

    # attn_maps[layer_idx] shape: (batch, n_heads, seq_len, seq_len)
    att = attn_maps[layer_idx][0]          # 取 batch 中第一个样本 (n_heads, seq_len, seq_len)
    att_mean = att.mean(axis=0)           # 多头平均 (seq_len, seq_len)

    seq_len = att_mean.shape[0]
    if feature_names is not None and len(feature_names) >= seq_len:
        xticks = yticks = feature_names[:seq_len]
    else:
        xticks = yticks = [f"F{i}" for i in range(seq_len)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(att_mean, xticklabels=xticks, yticklabels=yticks,
                cmap="viridis", square=True)
    plt.title(f"Attention Map (Layer {layer_idx})")
    plt.xlabel("Key / 被关注的特征")
    plt.ylabel("Query / 发起关注的特征")
    plt.tight_layout()
    plt.savefig(f"attention_layer_{layer_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()


# ===========================
#  主函数
# ===========================
def main():
    import gc, json, time
    
    print("加载数据...")
    df = pd.read_excel("original.xlsx")

    num_cols_for_impute = df.select_dtypes(include=[np.number]).columns.tolist()
    df[num_cols_for_impute] = df[num_cols_for_impute].fillna(df[num_cols_for_impute].median())

    log_cols = ['C_WBC', 'C_RBC', 'C_P', 'B_CRP', 'B_WBC', 'B_PCT', 'B_AC', 'B_RBC']
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    epsilon = 1e-6
    if 'C_G' in df.columns and 'B_G' in df.columns:
        df['ratio_C_G_B_G'] = df['C_G'] / (df['B_G'] + epsilon)
    if 'C_N' in df.columns and 'B_N' in df.columns:
        df['diff_C_N_B_N'] = df['C_N'] - df['B_N']
    if all(c in df.columns for c in ['C_WBC', 'B_WBC', 'C_RBC', 'B_RBC']):
        df['corrected_WBC'] = df['C_WBC'] - (df['B_WBC'] * df['C_RBC'] / (df['B_RBC'] + epsilon))
    if all(c in df.columns for c in ['B_WBC', 'B_RBC', 'C_WBC', 'C_RBC']):
        df['ratio_WBC_RBC_diff'] = (df['B_WBC'] / (df['B_RBC'] + epsilon)) - (df['C_WBC'] / (df['C_RBC'] + epsilon))

    base_features = [
        'age', 'C_G', 'C_WBC', 'C_RBC', 'C_P', 'C_N',
        'transparency', 'GCS', 'tem',
        'B_G', 'B_CRP', 'B_WBC', 'B_N', 'B_Lym', 'B_PCT', 'B_AC', 'B_RBC',
        'sex', 'tube', 'site', 'other_inf'
    ]
    new_features = ['ratio_C_G_B_G', 'diff_C_N_B_N', 'corrected_WBC', 'ratio_WBC_RBC_diff']
    feature_cols = base_features + [f for f in new_features if f in df.columns]

    X = df[feature_cols].values
    y = df['outcome'].values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import TensorDataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results      = []   # 基础AMFormer结果
    fold_results_xgb  = []   # XGB蒸馏结果
    fold_results_ft   = []   # FT蒸馏结果

    final_model = None
    final_scaler = None
    final_threshold = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*20} Fold {fold+1} / 5 {'='*20}")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_val_fold   = scaler.transform(X_val_fold)

        train_dataset = CSFDataset(X_train_fold, y_train_fold)
        val_dataset   = CSFDataset(X_val_fold,   y_val_fold)
        train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=0)
        val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=0)

        pos_count = y_train_fold.sum()
        neg_count = len(y_train_fold) - pos_count
        pos_weight_value = (neg_count / pos_count) * 0.8 if pos_count > 0 else 1.0
        pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)

        # ── 1. 基础AMFormer ──────────────────────────────
        print(f"\n[Fold {fold+1}] 训练基础AMFormer...")
        model = ImprovedAMFormer(
            input_dim=len(feature_cols),
            embed_dim=128, n_heads=4, n_layers=2, dropout=0.2
        )
        trainer = ModelTrainer(model, device=device, pos_weight=pos_weight)
        trainer.train(train_loader, val_loader, epochs=100, lr=1e-3, patience=15)

        # 在验证集上找阈值
        model.eval()
        val_probs_all, val_labels_all = [], []
        with torch.no_grad():
            for feats, labels in val_loader:
                probs = torch.sigmoid(model(feats.to(device)))
                val_probs_all.extend(probs.cpu().numpy())
                val_labels_all.extend(labels.numpy())

        best_threshold = find_best_threshold(
            np.array(val_labels_all), np.array(val_probs_all),
            fn_cost=2.5, fp_cost=1.5
        )
        results = evaluate_model(model, val_loader, device=device, threshold=best_threshold)
        fold_results.append(results)
        print(f"[Fold {fold+1}] 基础AMFormer  AUC: {results['auc']:.4f}  F1: {results['f1']:.4f}")

        # 第一折保存可视化
        if fold == 0:
            plot_training_curves(trainer)
            sample_x = X_val_fold[0]
            _, attn_maps = model(
                torch.tensor(sample_x).float().unsqueeze(0).to(device),
                return_attention=True
            )
            save_attention_to_csv(attn_maps, feature_cols, layer_idx=0)
            visualize_attention(model, sample_x, feature_cols, device=device, layer_idx=0)
            final_model     = model
            final_scaler    = scaler
            final_threshold = best_threshold

        # ── 2. 蒸馏实验 ──────────────────────────────────
        print(f"\n[Fold {fold+1}] 开始蒸馏实验...")

        # 训练教师（传入fold编号，避免文件名冲突）
        xgb_teacher = train_xgboost_teacher(X_train_fold, y_train_fold)
        ft_teacher  = train_fttransformer_teacher(
            X_train_fold, y_train_fold,
            X_val_fold, y_val_fold,
            device=device, fold_id=fold          # ← 传fold_id
        )

        # 生成soft label
        soft_xgb = generate_soft_labels(xgb_teacher, X_train_fold, teacher_type='xgb', temperature=3.0)
        soft_ft  = generate_soft_labels(ft_teacher,  X_train_fold, teacher_type='ft',
                                         device=device, temperature=3.0)

        def make_distill_loader(X, y_hard, y_soft, batch_size=32):
            ds = TensorDataset(
                torch.FloatTensor(X),
                torch.FloatTensor(y_hard),
                torch.FloatTensor(y_soft)
            )
            return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

        distill_configs = [
            ('XGB', make_distill_loader(X_train_fold, y_train_fold, soft_xgb)),
            ('FT',  make_distill_loader(X_train_fold, y_train_fold, soft_ft)),
        ]

        for teacher_name, distill_loader in distill_configs:
            print(f"\n[Fold {fold+1}] 蒸馏教师: {teacher_name}")
            model_d = ImprovedAMFormer(
                input_dim=len(feature_cols),
                embed_dim=128, n_heads=4, n_layers=2, dropout=0.2
            )
            save_path_d = f'best_distilled_{teacher_name}_fold{fold}.pth'
            d_trainer = DistillationTrainer(
                model_d, device=device, pos_weight=pos_weight,
                alpha=0.75, temperature=2.0,
                save_path=save_path_d              # ← 传存储路径
            )
            d_trainer.train_with_distillation(
                distill_loader, val_loader, epochs=100, lr=1e-3, patience=15
            )

            val_probs_d, val_labels_d = [], []
            model_d.eval()
            with torch.no_grad():
                for feats, labels in val_loader:
                    p = torch.sigmoid(model_d(feats.to(device)))
                    val_probs_d.extend(p.cpu().numpy())
                    val_labels_d.extend(labels.numpy())

            thr_d = find_best_threshold_constrained(
                np.array(val_labels_d), np.array(val_probs_d),
                min_recall=0.80
            )
            res_d = evaluate_model(model_d, val_loader, device=device, threshold=thr_d)
            print(f"[Fold {fold+1}] AMFormer-{teacher_name}蒸馏  "
                  f"AUC: {res_d['auc']:.4f}  F1: {res_d['f1']:.4f}")

            if teacher_name == 'XGB':
                fold_results_xgb.append(res_d)
            else:
                fold_results_ft.append(res_d)

            del model_d, d_trainer
            gc.collect()

        # 每折结束清理 + 保存
        del model, trainer, xgb_teacher, ft_teacher
        gc.collect()

        with open(f"fold_{fold+1}_results.json", "w") as f:
            json.dump({
                'base_auc':  results['auc'],  'base_f1':  results['f1'],
                'xgb_auc':   fold_results_xgb[-1]['auc'] if fold_results_xgb else None,
                'ft_auc':    fold_results_ft[-1]['auc']  if fold_results_ft  else None,
            }, f, indent=2)
        print(f"Fold {fold+1} 结果已保存到 fold_{fold+1}_results.json")

    # ── 汇总 ──────────────────────────────────────────
    print("\n" + "="*50)
    print("5-Fold 汇总结果")
    print("="*50)
    scalar_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    for label, res_list in [
        ('AMFormer基线', fold_results),
        ('AMFormer-XGB蒸馏', fold_results_xgb),
        ('AMFormer-FT蒸馏',  fold_results_ft),
    ]:
        if not res_list:
            continue
        print(f"\n── {label} ──")
        for k in scalar_keys:
            vals = [r[k] for r in res_list]
            print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # 保存pkl
    amformer_results = {
        k: np.mean([r[k] for r in fold_results])
        for k in scalar_keys
    }
    amformer_results['fold_details'] = fold_results
    with open("amformer_results.pkl", "wb") as f:
        pickle.dump(amformer_results, f)
    print("\namformer_results.pkl 已保存")

    os._exit(0)
def save_attention_to_csv(attn_maps, feature_names, layer_idx=0, filename="attention_layer0.csv"):
    """
    保存指定层的平均注意力权重到CSV
    """
    if layer_idx >= len(attn_maps):
        print(f"Layer index {layer_idx} out of range.")
        return

    # attn_maps[layer_idx] 是 (batch_size, heads, seq, seq)
    # 我们通常关心"全局"模式，所以可以对 batch 和 heads 取平均
    att = attn_maps[layer_idx] # (batch, heads, seq, seq)
    att_mean = np.mean(att, axis=(0, 1)) # (seq, seq)

    df = pd.DataFrame(att_mean, columns=feature_names, index=feature_names)
    df.to_csv(filename, encoding="utf-8-sig")
    print(f"已保存到 {filename}")


#蒸馏！！！！！！
# ===========================
#  知识蒸馏：教师模型训练
# ===========================

def train_xgboost_teacher(X_train, y_train):
    """用GradientBoosting替代XGBoost，避免macOS多进程segfault"""
    from sklearn.ensemble import GradientBoostingClassifier
    
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    
    teacher = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )
    teacher.fit(X_train, y_train)
    return teacher


def find_best_threshold(labels, probs, fn_cost=5, fp_cost=1):
    """原来的函数保持不变，基础AMFormer用这个"""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_t = 0.5
    best_cost = 1e9
    best_stats = None

    for t in thresholds:
        binary = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, binary).ravel()
        cost = fn_cost * fn + fp_cost * fp
        if cost < best_cost:
            best_cost = cost
            best_t = t
            best_stats = (tn, fp, fn, tp)

    tn, fp, fn, tp = best_stats
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    print(f"\n最佳阈值: {best_t:.3f} | P={precision:.4f} R={recall:.4f}")
    return best_t

def train_fttransformer_teacher(X_train, y_train, X_val, y_val,
                                 device='cpu', fold_id=0):
    import torch.nn as nn

    class FTTransformer(nn.Module):
        def __init__(self, input_dim, d_token=64, n_blocks=2):
            super().__init__()
            self.tokenizer = nn.Linear(1, d_token)
            self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_token))

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=4,
                dim_feedforward=d_token * 2,   # 缩小FFN
                dropout=0.2,
                activation='gelu',
                batch_first=True,
                norm_first=False               # 去掉Pre-LN，更稳定
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=n_blocks
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.Linear(d_token, 1)
            )

        def forward(self, x):
            B = x.size(0)
            tokens = self.tokenizer(x.unsqueeze(-1))
            cls    = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
            out    = self.transformer(tokens)
            return self.head(out[:, 0, :]).squeeze(-1)

    model = FTTransformer(input_dim=X_train.shape[1]).to(device)

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    pos_w     = torch.tensor(neg / pos, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    X_tr = torch.FloatTensor(X_train).to(device)
    y_tr = torch.FloatTensor(y_train).to(device)
    X_v  = torch.FloatTensor(X_val).to(device)
    y_v  = torch.FloatTensor(y_val).to(device)

    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=0
    )

    best_val_loss    = float('inf')
    patience_counter = 0
    save_path        = f'best_ft_transformer_fold{fold_id}.pth'
    torch.save(model.state_dict(), save_path)

    print("  训练FT-Transformer教师...")
    for epoch in range(200):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
            val_auc  = roc_auc_score(
                y_v.cpu().numpy(),
                torch.sigmoid(model(X_v)).cpu().numpy()
            )
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:03d}: val_loss={val_loss:.4f} "
                  f"val_auc={val_auc:.4f}")

        if patience_counter >= 20:
            print(f"    Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        final_auc = roc_auc_score(
            y_v.cpu().numpy(),
            torch.sigmoid(model(X_v)).cpu().numpy()
        )
    print(f"  FT教师最终: val_loss={best_val_loss:.4f}, AUC={final_auc:.4f}")

    if best_val_loss > 0.6:
        print(f"  ⚠️  教师质量偏低，蒸馏效果可能受限")
    return model

def generate_soft_labels(teacher, X_train, teacher_type='xgb',
                          device='cpu', temperature=3.0):
    """
    用教师模型生成soft label
    temperature: 温度系数，越大软标签越平滑
    """
    if teacher_type == 'xgb':
        # XGBoost直接输出概率
        probs = teacher.predict_proba(X_train)[:, 1]  # (N,)
    else:
        # FT-Transformer输出logit，转概率
        teacher.eval()
        X_t = torch.FloatTensor(X_train).to(device)
        with torch.no_grad():
            logits = teacher(X_t)
            # 用温度缩放软化概率
            probs = torch.sigmoid(logits / temperature).cpu().numpy()

    # 用温度缩放（XGBoost也做一次，统一处理）
    if teacher_type == 'xgb':
        # 将概率转回logit空间做温度缩放
        eps = 1e-6
        probs = np.clip(probs, eps, 1 - eps)
        logits_np = np.log(probs / (1 - probs))
        probs = 1 / (1 + np.exp(-logits_np / temperature))

    return probs.astype(np.float32)


# ===========================
#  知识蒸馏：蒸馏损失
# ===========================
class DistillationTrainer(ModelTrainer):
    """
    继承ModelTrainer，增加蒸馏损失支持
    alpha: hard label权重
    (1-alpha): soft label权重
    """
    def __init__(self, model, device='cpu', pos_weight=None,
                 alpha=0.4, temperature=3.0, save_path='best_distilled_model.pth'):
        super().__init__(model, device, pos_weight)
        self.alpha = alpha
        self.temperature = temperature
        self.save_path = save_path    

    def train_epoch_distill(self, train_loader, optimizer, criterion):
        """带软标签的训练epoch"""
        self.model.train()
        total_loss = 0

        for batch_features, batch_hard_labels, batch_soft_labels in tqdm(
                train_loader, desc="Distill Training"):
            batch_features    = batch_features.to(self.device)
            batch_hard_labels = batch_hard_labels.to(self.device)
            batch_soft_labels = batch_soft_labels.to(self.device)

            optimizer.zero_grad()
            logits = self.model(batch_features)

            # Hard loss（原始标签）
            hard_loss = criterion(logits, batch_hard_labels)

            # Soft loss（教师软标签）
            # 用BCE让学生概率逼近教师概率
            student_probs = torch.sigmoid(logits)
            soft_loss = F.binary_cross_entropy(
                student_probs, batch_soft_labels, reduction='mean'
            )

            # 总损失（温度系数²是标准蒸馏的缩放因子）
            loss = (self.alpha * hard_loss +
                      (1 - self.alpha) * soft_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def train_with_distillation(self, train_loader, val_loader,
                                epochs=100, lr=1e-3, patience=15):
        if self.pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        patience_counter = 0
        torch.save(self.model.state_dict(), self.save_path)

        print(f"  开始蒸馏训练 (alpha={self.alpha}, T={self.temperature})...")

        for epoch in range(epochs):
            train_loss = self.train_epoch_distill(
                train_loader, optimizer, criterion)
            # 验证集用标准validate（不需要软标签）
            val_loss, _, _ = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)  # ← 用self.save_path存
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:03d}: Train={train_loss:.4f}, "
                      f"Val={val_loss:.4f}")

            if patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break
               
        self.model.load_state_dict(torch.load(self.save_path))



if __name__ == "__main__":
    model, results, scaler, best_threshold = main()
