"""
改进的AMFormer深度学习模型（医学场景优化版本）
- 使用 BCEWithLogitsLoss + pos_weight 提高对阳性样本的关注（减少漏诊）
- 在验证集上自动搜索最佳阈值（考虑 FN/FP 代价）
- 使用最佳阈值在测试集上评估
- 支持导出 Attention 权重，用于分析特征间关系
"""

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
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
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
        embedded_input = self.input_embedding(x)           # (batch_size, embed_dim)
        embedded_input = embedded_input.unsqueeze(1).expand(-1, self.input_dim, -1)
        # 添加位置编码
        pos_enc = self.pos_encoding[:, :self.input_dim, :].expand(batch_size, -1, -1)
        embedded = embedded_input + pos_enc                # (batch_size, input_dim, embed_dim)

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
def find_best_threshold(labels, probs, fn_cost=5, fp_cost=1):
    """
    在验证集上搜索最佳阈值：
    - labels: 真实标签 (0/1)
    - probs: 模型预测概率
    - fn_cost: 漏诊的代价（越大表示越不想漏）
    - fp_cost: 误诊的代价
    目标：最小化 fn_cost * FN + fp_cost * FP
    """
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

    print("\n===== 在验证集上搜索最佳阈值 =====")
    print(f"最佳阈值: {best_t:.3f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Precision={precision:.4f}, Recall={recall:.4f}")
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
    print("加载数据...")
    df = pd.read_excel("original.xlsx")

    # 简单缺失值处理
    df = df.fillna(0)

    print(f"数据形状: {df.shape}")
    print(f"标签分布:\n{df['outcome'].value_counts()}")

    # 特征列（与之前一致）
    feature_cols = [
        'C_G', 'C_WBC', 'C_RBC', 'C_P', 'C_N',
        'transparency', 'GCS', 'tem',
        'B_G', 'B_CRP', 'B_WBC', 'B_N', 'B_Lym', 'B_PCT', 'B_AC', 'B_RBC',
        'sex', 'tube', 'site', 'other_inf'
    ]

    X = df[feature_cols].values
    y = df['outcome'].values

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练 / 验证 / 测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")

    # DataLoader
    train_dataset = CSFDataset(X_train, y_train)
    val_dataset = CSFDataset(X_val, y_val)
    test_dataset = CSFDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 计算正样本权重（用于 BCEWithLogitsLoss）
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight_value = neg_count / pos_count
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32).to(device)
    print(f"正样本权重 pos_weight = {pos_weight_value:.4f}")

    # 创建模型
    model = ImprovedAMFormer(
        input_dim=len(feature_cols),
        embed_dim=128,
        n_heads=8,
        n_layers=4,
        dropout=0.1
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练模型
    trainer = ModelTrainer(model, device=device, pos_weight=pos_weight)
    trainer.train(train_loader, val_loader, epochs=100, lr=1e-3, patience=15)

    # 画训练曲线
    plot_training_curves(trainer)

    # 保存 attention 
    sample_x = X_test[0]
    logits, attn_maps = model(torch.tensor(sample_x).float().unsqueeze(0).to(device), return_attention=True)

    # 保存 CSV
    save_attention_to_csv(attn_maps, feature_cols, layer_idx=0, filename="attention_layer0.csv")

    # ===== 在验证集上得到概率，用来搜索最佳阈值 =====
    model.eval()
    val_probs_all = []
    val_labels_all = []
    with torch.no_grad():
        for feats, labels in val_loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = torch.sigmoid(logits)
            val_probs_all.extend(probs.cpu().numpy())
            val_labels_all.extend(labels.numpy())

    val_probs_all = np.array(val_probs_all)
    val_labels_all = np.array(val_labels_all)

    # 搜索一个对漏诊更敏感的最佳阈值
    best_threshold = find_best_threshold(
        labels=val_labels_all,
        probs=val_probs_all,
        fn_cost=5,   # 漏诊代价
        fp_cost=1    # 误诊代价
    )

    # ===== 在测试集上用最佳阈值评估 =====
    results = evaluate_model(model, test_loader, device=device, threshold=best_threshold)

    # 可视化 attention（取测试集第一个样本）
    sample_x = X_test[0]  # 标准化后的特征
    visualize_attention(model, sample_x, feature_cols, device=device, layer_idx=0)

    return model, results, scaler, best_threshold

def save_attention_to_csv(attn_maps, feature_names, layer_idx=0, filename="attention_layer0.csv"):
    att = attn_maps[layer_idx][0]   # 取第一条样本 (heads, seq, seq)
    att_mean = att.mean(axis=0)     # 多头平均 (seq, seq)

    df = pd.DataFrame(att_mean, columns=feature_names, index=feature_names)
    df.to_csv(filename, encoding="utf-8-sig")
    print(f"已保存到 {attention}")






if __name__ == "__main__":
    model, results, scaler, best_threshold = main()
