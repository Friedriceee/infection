"""
多模态渐进式CNS感染预测模型
支持：① 仅静态输入  ② 仅动态输入  ③ 静态+动态联合输入
动态数据：4个时间点，每个时间点间隔约1天，允许缺失
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 1. 配置
# ─────────────────────────────────────────────
STATIC_COLS = ['C_G', 'C_WBC', 'C_RBC', 'C_P', 'C_N', 'transparency',
               'GCS', 'age', 'sex', 'tem', 'B_G', 'B_CRP', 'B_WBC',
               'B_N', 'B_Lym', 'B_PCT']  # 按你的original列名调整

DYNAMIC_COLS = ['WBC', 'C_RBC', 'C_N', 'transparency', 'C_G', 'C_P']  # 曲线中的6个CSF特征

N_TIMEPOINTS  = 4      # t1 t2 t3 t4
STATIC_DIM    = len(STATIC_COLS)
DYNAMIC_DIM   = len(DYNAMIC_COLS)
EMBED_DIM     = 64
LSTM_HIDDEN   = 64
FC_HIDDEN     = 128
DROPOUT       = 0.3
BATCH_SIZE    = 32
EPOCHS        = 80
LR            = 1e-3
N_FOLDS       = 5
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─────────────────────────────────────────────
# 2. 数据集
# ─────────────────────────────────────────────
class CNSDataset(Dataset):
    """
    每个样本包含：
      static_x  : (STATIC_DIM,)  无静态时全0
      dynamic_x : (N_TIMEPOINTS, DYNAMIC_DIM)  无动态时全0
      mask      : (N_TIMEPOINTS,)  有效时间点为1，缺失为0
      has_static: bool
      has_dynamic: bool
      label     : 0/1
    """
    def __init__(self, records):
        # records: list of dict
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            'static_x'   : torch.tensor(r['static_x'],   dtype=torch.float32),
            'dynamic_x'  : torch.tensor(r['dynamic_x'],  dtype=torch.float32),
            'mask'        : torch.tensor(r['mask'],        dtype=torch.float32),
            'has_static'  : torch.tensor(r['has_static'],  dtype=torch.float32),
            'has_dynamic' : torch.tensor(r['has_dynamic'], dtype=torch.float32),
            'label'       : torch.tensor(r['label'],       dtype=torch.float32),
        }


# ─────────────────────────────────────────────
# 3. 模型
# ─────────────────────────────────────────────
class MultiModalLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        # 静态分支：MLP → static_emb
        self.static_mlp = nn.Sequential(
            nn.Linear(STATIC_DIM, FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_HIDDEN, EMBED_DIM),
            nn.ReLU(),
        )

        # 动态分支：LSTM → dynamic_emb
        self.lstm = nn.LSTM(
            input_size=DYNAMIC_DIM,
            hidden_size=LSTM_HIDDEN,
            num_layers=2,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=False,
        )
        self.dynamic_proj = nn.Sequential(
            nn.Linear(LSTM_HIDDEN, EMBED_DIM),
            nn.ReLU(),
        )

        # 融合层：输入维度按实际拼接（最大 128d）
        self.fusion = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_HIDDEN, 1),
        )

    def forward(self, static_x, dynamic_x, mask, has_static, has_dynamic):
        """
        static_x  : (B, STATIC_DIM)
        dynamic_x : (B, T, DYNAMIC_DIM)
        mask      : (B, T)   有效时间步为1
        has_static: (B,)     该样本是否有静态特征
        has_dynamic:(B,)     该样本是否有动态特征
        """
        B = static_x.size(0)

        # ── 静态 embedding ──
        static_emb = self.static_mlp(static_x)             # (B, EMBED_DIM)
        # 没有静态数据的样本，embedding 置0
        static_emb = static_emb * has_static.unsqueeze(1)

        # ── 动态 embedding ──
        # mask 屏蔽无效时间步（填0，LSTM 按 packed 方式处理）
        masked_dynamic = dynamic_x * mask.unsqueeze(-1)    # (B, T, D)
        # 计算每个样本有效的时间步数（至少1，防止 pack_padded_sequence 出错）
        lengths = mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            masked_dynamic, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        lstm_out = hn[-1]                                   # (B, LSTM_HIDDEN)
        dynamic_emb = self.dynamic_proj(lstm_out)           # (B, EMBED_DIM)
        # 没有动态数据的样本，embedding 置0
        dynamic_emb = dynamic_emb * has_dynamic.unsqueeze(1)

        # ── 融合 ──
        fused = torch.cat([static_emb, dynamic_emb], dim=1)  # (B, EMBED_DIM*2)
        logit = self.fusion(fused).squeeze(1)                 # (B,)
        return logit


# ─────────────────────────────────────────────
# 4. 数据准备函数
# ─────────────────────────────────────────────
def load_and_merge(orig_path, yang_path, yin_path):
    """
    读取三个文件，按 ID 关联，返回统一 records list。
    每条 record 包含静态+动态（缺失则置0+mask=0）。
    """
    orig = pd.read_excel(orig_path)
    yang = pd.read_excel(yang_path)
    yin  = pd.read_excel(yin_path)

    for df in [orig, yang, yin]:
        df['ID'] = df['ID'].astype(str).str.strip()

    yang['label'] = 1
    yin['label']  = 0
    curve = pd.concat([yang, yin], ignore_index=True)

    # 静态特征归一化（fit 在训练集，这里先全局做 placeholder）
    static_scaler  = StandardScaler()
    dynamic_scaler = StandardScaler()

    records = []

    for _, row in curve.iterrows():
        rid   = row['ID']
        label = int(row['label'])

        # ── 动态特征（4时间点）──
        dynamic_x = np.zeros((N_TIMEPOINTS, DYNAMIC_DIM), dtype=np.float32)
        mask = np.zeros(N_TIMEPOINTS, dtype=np.float32)
        for t_idx, t_sfx in enumerate(['_1', '_2', '_3', '_4']):
            cols_t = [c + t_sfx for c in DYNAMIC_COLS]
            vals = pd.to_numeric(
                row[cols_t].astype(str).str.replace(',', '.', regex=False),
                errors='coerce'
            ).values
            if not np.all(np.isnan(vals)):
                # 有任意一个值就算这个时间点有效
                vals = np.nan_to_num(vals, nan=0.0)
                dynamic_x[t_idx] = vals
                mask[t_idx] = 1.0
        has_dynamic = float(mask.sum() > 0)

        # ── 静态特征 ──
        orig_row = orig[orig['ID'] == rid]
        if len(orig_row) > 0:
            s = orig_row.iloc[0][STATIC_COLS].values.astype(float)
            s = np.nan_to_num(s, nan=0.0)
            has_static = 1.0
        else:
            s = np.zeros(STATIC_DIM, dtype=np.float32)
            has_static = 0.0

        records.append({
            'static_x'   : s,
            'dynamic_x'  : dynamic_x,
            'mask'        : mask,
            'has_static'  : has_static,
            'has_dynamic' : has_dynamic,
            'label'       : label,
            'ID'          : rid,
        })

    # 静态数据里 original-only 的样本（时序数据没有的）
    curve_ids = set(curve['ID'].tolist())
    for _, row in orig.iterrows():
        rid = row['ID']
        if rid in curve_ids:
            continue
        # 只有静态，没有动态
        label = int(row.get('outcome', -1))  # 按你的标签列名调整
        if label not in [0, 1]:
            continue
        s = row[STATIC_COLS].values.astype(float)
        s = np.nan_to_num(s, nan=0.0)
        records.append({
            'static_x'   : s,
            'dynamic_x'  : np.zeros((N_TIMEPOINTS, DYNAMIC_DIM), dtype=np.float32),
            'mask'        : np.zeros(N_TIMEPOINTS, dtype=np.float32),
            'has_static'  : 1.0,
            'has_dynamic' : 0.0,
            'label'       : label,
            'ID'          : rid,
        })

    return records


def normalize_records(records, static_scaler=None, dynamic_scaler=None, fit=False):
    """归一化 static_x 和 dynamic_x（对有效时间步）"""
    static_matrix  = np.stack([r['static_x']  for r in records])
    dynamic_matrix = np.stack([r['dynamic_x'].reshape(-1, DYNAMIC_DIM)
                                for r in records]).reshape(-1, DYNAMIC_DIM)

    if fit:
        static_scaler  = StandardScaler().fit(static_matrix)
        dynamic_scaler = StandardScaler().fit(dynamic_matrix)

    static_norm  = static_scaler.transform(static_matrix)
    dynamic_norm = dynamic_scaler.transform(dynamic_matrix).reshape(-1, N_TIMEPOINTS, DYNAMIC_DIM)

    for i, r in enumerate(records):
        r['static_x']  = static_norm[i].astype(np.float32)
        r['dynamic_x'] = dynamic_norm[i].astype(np.float32)
        # 对 mask=0 的时间步重新置0（scaler 可能把 padding 变成非零）
        for t in range(N_TIMEPOINTS):
            if r['mask'][t] == 0:
                r['dynamic_x'][t] = 0.0

    return records, static_scaler, dynamic_scaler


# ─────────────────────────────────────────────
# 5. 训练 & 评估
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch in loader:
        logit = model(
            batch['static_x'].to(DEVICE),
            batch['dynamic_x'].to(DEVICE),
            batch['mask'].to(DEVICE),
            batch['has_static'].to(DEVICE),
            batch['has_dynamic'].to(DEVICE),
        )
        loss = criterion(logit, batch['label'].to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        logit = model(
            batch['static_x'].to(DEVICE),
            batch['dynamic_x'].to(DEVICE),
            batch['mask'].to(DEVICE),
            batch['has_static'].to(DEVICE),
            batch['has_dynamic'].to(DEVICE),
        )
        prob = torch.sigmoid(logit).cpu().numpy()
        all_probs.extend(prob.tolist())
        all_labels.extend(batch['label'].numpy().tolist())
    auc = roc_auc_score(all_labels, all_probs)
    return auc, all_probs, all_labels


# ─────────────────────────────────────────────
# 6. 5折交叉验证主流程
# ─────────────────────────────────────────────
def cross_validate(records):
    labels = np.array([r['label'] for r in records])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(records)), labels)):
        print(f"\n── Fold {fold+1}/{N_FOLDS} ──")

        train_recs = [records[i] for i in train_idx]
        val_recs   = [records[i] for i in val_idx]

        # 归一化（fit on train）
        train_recs, ss, ds = normalize_records(train_recs, fit=True)
        val_recs, _, _     = normalize_records(val_recs, static_scaler=ss,
                                               dynamic_scaler=ds, fit=False)

        # SMOTE（仅对训练集的静态特征做，时序单独处理）
        # 简化：对 static_x 做 SMOTE，dynamic 同比复制
        train_static  = np.stack([r['static_x'] for r in train_recs])
        train_labels  = np.array([r['label'] for r in train_recs])
        smote = SMOTE(random_state=42)
        static_res, labels_res = smote.fit_resample(train_static, train_labels)

        # 重建 records（SMOTE 新增样本用 has_dynamic=0）
        aug_recs = []
        n_orig = len(train_recs)
        for i, (sx, lbl) in enumerate(zip(static_res, labels_res)):
            if i < n_orig:
                r = dict(train_recs[i])
                r['static_x'] = sx
            else:
                r = {
                    'static_x'   : sx,
                    'dynamic_x'  : np.zeros((N_TIMEPOINTS, DYNAMIC_DIM), np.float32),
                    'mask'        : np.zeros(N_TIMEPOINTS, np.float32),
                    'has_static'  : 1.0,
                    'has_dynamic' : 0.0,
                    'label'       : int(lbl),
                    'ID'          : f'smote_{i}',
                }
            aug_recs.append(r)

        train_loader = DataLoader(CNSDataset(aug_recs),  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(CNSDataset(val_recs),  batch_size=BATCH_SIZE, shuffle=False)

        model     = MultiModalLSTM().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc = 0
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            auc, _, _ = evaluate(model, val_loader)
            scheduler.step()
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f'best_fold{fold+1}.pt')
            if epoch % 10 == 0:
                print(f"  Epoch {epoch:3d} | loss={loss:.4f} | val_auc={auc:.4f}")

        print(f"  Best AUC fold {fold+1}: {best_auc:.4f}")
        fold_aucs.append(best_auc)

    print(f"\n{'='*40}")
    print(f"5折 AUC: {[f'{a:.4f}' for a in fold_aucs]}")
    print(f"Mean ± Std: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    return fold_aucs


# ─────────────────────────────────────────────
# 7. 单样本推理（临床使用）
# ─────────────────────────────────────────────
def predict_single(model, static_features=None, dynamic_sequence=None,
                   static_scaler=None, dynamic_scaler=None):
    """
    static_features : dict 或 None   e.g. {'C_G': 3.2, 'C_WBC': 500, ...}
    dynamic_sequence: list of dict 或 None
                      e.g. [{'WBC':200,'C_RBC':1000,...},   # t1
                             {'WBC':150,'C_RBC':800, ...},  # t2
                             None,                          # t3 缺失
                             {'WBC':100,'C_RBC':600, ...}]  # t4

    返回 CNS感染概率 (0~1)
    """
    model.eval()

    # 静态
    if static_features is not None:
        sx = np.array([static_features.get(c, 0.0) for c in STATIC_COLS], dtype=np.float32)
        sx = np.nan_to_num(sx, nan=0.0)
        if static_scaler:
            sx = static_scaler.transform(sx.reshape(1, -1))[0]
        has_static = 1.0
    else:
        sx = np.zeros(STATIC_DIM, dtype=np.float32)
        has_static = 0.0

    # 动态
    dx   = np.zeros((N_TIMEPOINTS, DYNAMIC_DIM), dtype=np.float32)
    mask = np.zeros(N_TIMEPOINTS, dtype=np.float32)
    if dynamic_sequence is not None:
        for t_idx, t_data in enumerate(dynamic_sequence[:N_TIMEPOINTS]):
            if t_data is None:
                continue
            vals = np.array([t_data.get(c, 0.0) for c in DYNAMIC_COLS], dtype=np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            if dynamic_scaler:
                vals = dynamic_scaler.transform(vals.reshape(1, -1))[0]
            dx[t_idx] = vals
            mask[t_idx] = 1.0
    has_dynamic = float(mask.sum() > 0)

    with torch.no_grad():
        logit = model(
            torch.tensor(sx).unsqueeze(0).to(DEVICE),
            torch.tensor(dx).unsqueeze(0).to(DEVICE),
            torch.tensor(mask).unsqueeze(0).to(DEVICE),
            torch.tensor([has_static]).to(DEVICE),
            torch.tensor([has_dynamic]).to(DEVICE),
        )
        prob = torch.sigmoid(logit).item()

    mode = []
    if has_static:  mode.append("静态")
    if has_dynamic: mode.append(f"动态({int(mask.sum())}个时间点)")
    print(f"输入模式: {' + '.join(mode) if mode else '无输入'}")
    print(f"CNS感染概率: {prob:.4f}")
    return prob


# ─────────────────────────────────────────────
# 8. 消融实验（论文故事线）
# ─────────────────────────────────────────────
def ablation_study(records):
    """
    对比三种模式的 AUC，量化动态信息的增量价值
    """
    results = {}

    # 只保留同时有静态+动态的331条
    both_records = [r for r in records if r['has_static'] == 1 and r['has_dynamic'] == 1]
    print(f"同时有静态+动态的样本: {len(both_records)}")

    for mode in ['static_only', 'dynamic_only', 'both']:
        print(f"\n── 消融模式: {mode} ──")
        mode_recs = []
        for r in both_records:
            nr = dict(r)
            if mode == 'static_only':
                nr['has_dynamic'] = 0.0
                nr['dynamic_x']   = np.zeros_like(r['dynamic_x'])
                nr['mask']        = np.zeros_like(r['mask'])
            elif mode == 'dynamic_only':
                nr['has_static'] = 0.0
                nr['static_x']   = np.zeros_like(r['static_x'])
            mode_recs.append(nr)

        aucs = cross_validate(mode_recs)
        results[mode] = {'mean': np.mean(aucs), 'std': np.std(aucs)}

    print("\n=== 消融实验汇总 ===")
    for mode, v in results.items():
        print(f"  {mode:15s}: AUC = {v['mean']:.4f} ± {v['std']:.4f}")
    return results


# ─────────────────────────────────────────────
# 9. 入口
# ─────────────────────────────────────────────
if __name__ == '__main__':
    # ① 读取数据
    records = load_and_merge(
        orig_path='original.xlsx',
        yang_path='曲线阳.xlsx',
        yin_path='曲线阴.xlsx',
    )
    print(f"总样本量: {len(records)}")
    print(f"  - 阳性: {sum(r['label']==1 for r in records)}")
    print(f"  - 阴性: {sum(r['label']==0 for r in records)}")
    print(f"  - 有静态: {sum(r['has_static']==1 for r in records)}")
    print(f"  - 有动态: {sum(r['has_dynamic']==1 for r in records)}")
    print(f"  - 两者都有: {sum(r['has_static']==1 and r['has_dynamic']==1 for r in records)}")

    # ② 主实验：5折交叉验证
    fold_aucs = cross_validate(records)

    # ③ 消融实验（论文故事线）
    ablation_results = ablation_study(records)

    # ④ 单样本推理示例
    # model = MultiModalLSTM().to(DEVICE)
    # model.load_state_dict(torch.load('best_fold1.pt'))
    # prob = predict_single(
    #     model,
    #     static_features={'C_G': 3.2, 'C_WBC': 500, 'age': 45, ...},
    #     dynamic_sequence=[
    #         {'WBC': 200, 'C_RBC': 1000, 'C_N': 80, 'transparency': 2, 'C_G': 3.0, 'C_P': 1200},
    #         {'WBC': 150, 'C_RBC': 800,  'C_N': 75, 'transparency': 2, 'C_G': 2.8, 'C_P': 1100},
    #         None,   # 第3天缺失
    #         {'WBC': 100, 'C_RBC': 600,  'C_N': 70, 'transparency': 1, 'C_G': 2.5, 'C_P': 1000},
    #     ]
    # )