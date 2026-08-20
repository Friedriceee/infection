import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")


# ===========================
# 基础设置
# ===========================
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================
# 模型组件：多头注意力
# ===========================
class MultiHeadAttention(nn.Module):
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
        batch_size, seq_len, d_model = x.size()
        residual = x

        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output, attn_weights


# ===========================
# 模型组件：特征交互层
# ===========================
class FeatureInteractionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.interaction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.feature_weights = nn.Parameter(torch.ones(input_dim))

    def forward(self, x):
        weighted_features = x * self.feature_weights
        interactions = self.interaction_net(weighted_features)
        return interactions


# ===========================
# Improved AMFormer
# ===========================
class ImprovedAMFormer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.attention_maps = []

        self.input_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.feature_interaction = FeatureInteractionLayer(input_dim, embed_dim // 2)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, embed_dim))

        self.transformer_layers = nn.ModuleList([
            MultiHeadAttention(embed_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.feed_forward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(n_layers)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

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
        batch_size = x.size(0)
        self.attention_maps = []

        interaction_features = self.feature_interaction(x)

        x_tokens = x.unsqueeze(-1)
        embedded = self.input_embedding(x_tokens)

        pos_enc = self.pos_encoding[:, :self.input_dim, :].expand(batch_size, -1, -1)
        embedded = embedded + pos_enc

        for attn_layer, ff_layer, ln in zip(
            self.transformer_layers, self.feed_forward_layers, self.layer_norms
        ):
            attn_output, attn_weights = attn_layer(embedded)
            self.attention_maps.append(attn_weights.detach().cpu().numpy())

            ff_output = ff_layer(attn_output)
            embedded = ln(ff_output + attn_output)

        pooled = self.global_pool(embedded.transpose(1, 2)).squeeze(-1)
        combined_features = torch.cat([pooled, interaction_features], dim=1)

        logits = self.classifier(combined_features).squeeze(-1)

        if return_attention:
            return logits, self.attention_maps
        return logits


# ===========================
# 特征工程（与训练代码保持一致）
# ===========================
def build_features(df):
    df = df.copy()
    df = df.fillna(0)

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
        'C_G', 'C_WBC', 'C_RBC', 'C_P', 'C_N',
        'transparency', 'GCS', 'tem',
        'B_G', 'B_CRP', 'B_WBC', 'B_N', 'B_Lym', 'B_PCT', 'B_AC', 'B_RBC',
        'sex', 'tube', 'site', 'other_inf'
    ]
    new_features = ['ratio_C_G_B_G', 'diff_C_N_B_N', 'corrected_WBC', 'ratio_WBC_RBC_diff']
    feature_cols = base_features + [f for f in new_features if f in df.columns]

    X = df[feature_cols].values.astype(np.float32)
    y = df['outcome'].values.astype(np.float32)

    return X, y, feature_cols


# ===========================
# 模型概率预测
# ===========================
def predict_proba(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


# ===========================
# Permutation Importance
# ===========================
def permutation_importance(model, X, y, feature_names, n_repeats=5, random_state=42):
    rng = np.random.RandomState(random_state)

    base_pred = predict_proba(model, X)
    base_auc = roc_auc_score(y, base_pred)
    print(f"Base ROC-AUC: {base_auc:.4f}")

    importances = []

    for i, feat in enumerate(feature_names):
        auc_drops = []

        for _ in range(n_repeats):
            X_perm = X.copy()
            shuffled_col = X_perm[:, i].copy()
            rng.shuffle(shuffled_col)
            X_perm[:, i] = shuffled_col

            perm_pred = predict_proba(model, X_perm)
            perm_auc = roc_auc_score(y, perm_pred)
            auc_drop = base_auc - perm_auc
            auc_drops.append(auc_drop)

        mean_drop = np.mean(auc_drops)
        std_drop = np.std(auc_drops)
        importances.append((feat, mean_drop, std_drop))

        print(f"{feat:<20} mean_auc_drop = {mean_drop:.6f} ± {std_drop:.6f}")

    importance_df = pd.DataFrame(importances, columns=["Feature", "Importance", "Std"])
    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)

    return base_auc, importance_df


# ===========================
# 画 Top10
# ===========================
def plot_top10(importance_df, save_path="amformer_top10.png"):
    top10 = importance_df.head(10).copy()
    top10 = top10.iloc[::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(top10["Feature"], top10["Importance"])
    plt.xlabel("Mean AUC Decrease")
    plt.title("Top 10 Feature Importance of AMFormer")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Top10图已保存：{save_path}")


# ===========================
# 主函数
# ===========================
def main():
    data_path = "original.xlsx"
    model_path = "best_amformer_model.pth"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据文件：{data_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重：{model_path}")

    print("加载数据...")
    df = pd.read_excel(data_path)

    X, y, feature_cols = build_features(df)
    print("特征列表：", feature_cols)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 注意：
    # 你原始训练代码里每一折都单独 fit StandardScaler，
    # 但没有把 scaler 保存到磁盘。
    # 所以这个独立脚本里只能对当前全体数据重新标准化。
    # 如果你之后想和“第一折最佳模型”完全严格对应，最好在训练时把 scaler 也保存下来。
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    print("加载模型...")
    model = ImprovedAMFormer(
        input_dim=len(feature_cols),
        embed_dim=128,
        n_heads=4,
        n_layers=2,
        dropout=0.2
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print("开始计算 permutation importance...")
    base_auc, importance_df = permutation_importance(
        model=model,
        X=X_scaled,
        y=y,
        feature_names=feature_cols,
        n_repeats=10,
        random_state=42
    )

    importance_df.to_csv("amformer_feature_importance_all.csv", index=False, encoding="utf-8-sig")
    print("全部特征重要性已保存：amformer_feature_importance_all.csv")

    print("\nTop 10 特征：")
    print(importance_df.head(10).to_string(index=False))

    plot_top10(importance_df, save_path="amformer_top10.png")

    print(f"\nBase ROC-AUC = {base_auc:.4f}")
    print("完成。")


if __name__ == "__main__":
    main()