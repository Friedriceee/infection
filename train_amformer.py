import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ========== 1. 读取 & 自动识别特征 ==========

DATA_FILE = "转化后_编码数据_最终版本.csv"
LABEL_COL = "outcome"                         # 标签列名

df = pd.read_csv(DATA_FILE)

# 丢掉无标签的行
df = df.dropna(subset=[LABEL_COL])

# 标签转成 0/1
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce").astype(int)

# 只看数值型的列（非字符串）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 移除ID列
if 'ID' in numeric_cols:
    numeric_cols.remove('ID')

# 只使用编码后的特征
cat_cols = []
for c in ["sex", "tube", "site", "other_inf"]:
    if c in numeric_cols and c != LABEL_COL:
        cat_cols.append(c)

# 数值特征：只使用其他编码后的特征，排除类别特征和标签列
num_cols = [c for c in numeric_cols if c not in cat_cols and c != LABEL_COL]

print("数值特征列 num_cols:", num_cols)
print("类别特征列 cat_cols:", cat_cols)

# 构建数值特征矩阵
num_df = df[num_cols].copy()
# 缺失值补中位数
num_df = num_df.apply(lambda x: x.fillna(x.median()))
X_num = num_df.to_numpy(dtype=np.float32)

# 构建类别特征矩阵（已经是编码好的整数）
cat_df = df[cat_cols].copy()
cat_df = cat_df.fillna(0).astype(int)
X_cat = cat_df.to_numpy(dtype=np.int64)

# 标签
y = df[LABEL_COL].to_numpy(dtype=np.int64)

print("X_num 形状:", X_num.shape)
print("X_cat 形状:", X_cat.shape)
print("y 形状:", y.shape)

# 对每个类别特征，统计类别数（最大值+1）
num_categories = []
for col in cat_cols:
    n = int(cat_df[col].max()) + 1
    num_categories.append(max(n, 2))  # 至少 2 类
print("每个类别特征的类别数 num_categories:", num_categories)


# ========== 2. 划分训练 / 测试集 ==========

Xn_train, Xn_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
    X_num, X_cat, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)


# ========== 3. Dataset & DataLoader ==========

class CNSInfectionDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)  # BCELoss 用 float

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


train_ds = CNSInfectionDataset(Xn_train, Xc_train, y_train)
test_ds  = CNSInfectionDataset(Xn_test,  Xc_test,  y_test)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)


# ========== 4. 定义 AMFormer 模型 ==========

class NumericalEncoder(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_features, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x):
        return self.fc(x)  # (B, D)


class CategoricalEncoder(nn.Module):
    def __init__(self, num_categories, embed_dim):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(n, embed_dim) for n in num_categories]
        )

    def forward(self, x):
        # x: (B, C)
        outs = []
        for i, emb in enumerate(self.embs):
            outs.append(emb(x[:, i]))
        return torch.stack(outs, dim=1)  # (B, C, D)


class ArithmeticBlock(nn.Module):
    """简单的 MLP 残差块，用来做偏差归纳"""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.fc(x)


class InteractionBlock(nn.Module):
    """Top-k 注意力交互模块"""
    def __init__(self, dim, num_heads=4, top_k=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.top_k = top_k

    def forward(self, x):
        # x: (B, L, D)
        attn_out, weights = self.attn(x, x, x)  # weights: (B, L, L)
        mean_w = weights.mean(1)                # (B, L)
        k = min(self.top_k, mean_w.size(1))
        topk_idx = torch.topk(mean_w, k=k, dim=1).indices  # (B, k)
        batch_idx = torch.arange(x.size(0)).unsqueeze(-1).to(x.device)
        selected = attn_out[batch_idx, topk_idx]           # (B, k, D)
        return selected.mean(1)                            # (B, D)


class AMFormer(nn.Module):
    def __init__(self, num_num_features, num_categories, embed_dim=32):
        super().__init__()
        self.num_encoder = NumericalEncoder(num_num_features, embed_dim)
        self.cat_encoder = CategoricalEncoder(num_categories, embed_dim)
        self.arith = ArithmeticBlock(embed_dim)
        self.interact = InteractionBlock(embed_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, num_x, cat_x):
        num_feat = self.num_encoder(num_x)               # (B, D)
        cat_feat = self.cat_encoder(cat_x)               # (B, C, D)
        x = torch.cat([num_feat.unsqueeze(1), cat_feat], dim=1)  # (B, C+1, D)
        x = self.arith(x)
        x = self.interact(x)                             # (B, D)
        out = self.fc_out(x).squeeze(1)                  # (B,)
        return out


model = AMFormer(
    num_num_features=Xn_train.shape[1],
    num_categories=num_categories,
    embed_dim=32
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ========== 5. 训练循环 ==========

EPOCHS = 50

for epoch in range(1, EPOCHS + 1):
    # ---- train ----
    model.train()
    train_losses = []

    for batch_num, batch_cat, batch_y in train_loader:
        batch_num = batch_num.to(device)
        batch_cat = batch_cat.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_num, batch_cat)           # (B,)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # ---- eval ----
    model.eval()
    all_pred = []
    all_true = []

    with torch.no_grad():
        for batch_num, batch_cat, batch_y in test_loader:
            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)
            batch_y = batch_y.to(device)

            prob = model(batch_num, batch_cat)       # (B,)
            all_pred.append(prob.cpu().numpy())
            all_true.append(batch_y.cpu().numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true).astype(int)

    pred_label = (all_pred >= 0.5).astype(int)
    acc = accuracy_score(all_true, pred_label)
    try:
        auc = roc_auc_score(all_true, all_pred)
    except ValueError:
        auc = np.nan

    # 计算混淆矩阵和其他指标
    cm = confusion_matrix(all_true, pred_label)
    TN, FP, FN, TP = cm.ravel()
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0  # 特异度 (TNR)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0  # 敏感度 (TPR/召回率)

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d} | "
              f"train_loss={np.mean(train_losses):.4f} | "
              f"val_acc={acc:.3f} | val_auc={auc:.3f}")
        print(f"Confusion Matrix:")
        print(f"[[{TN:3d}, {FP:3d}]")
        print(f" [{FN:3d}, {TP:3d}]]")
        print(f"TNR(特异度)={specificity:.3f}, TPR(敏感度/召回率)={sensitivity:.3f}")
        print("-" * 50)
