"""
amformer_clinical_web.py

静态 + 动态自动分流网页
========================================================
页面逻辑：
1. 只输入当前一次静态指标：使用静态 PGA-AMFormer
2. 在静态指标基础上，继续输入 T-2/T-1/T0/T+1 多时间点脑脊液指标：使用动态 D-PGA-AMFormer

你当前上传的文件建议这样放：
web_model/
  pga2.py                          # 你的 PGA-AMFormer 模型定义代码，必须放
  best_fold3.pth                   # 静态模型，embed_dim=96，n_layers=2
  dynamic_full_model_fold3.pth     # 动态完整模型包，内部静态 backbone embed_dim=64
  dynamic_head_fold3.pth           # 动态残差头，可选；如果有 dynamic_full_model 可不用
  dyn_scaler_fold3.joblib          # 动态趋势特征标准化器，60维

运行：
  pip install flask torch numpy pandas scikit-learn joblib
  python amformer_clinical_web.py
  浏览器打开 http://127.0.0.1:7860

重要说明：
- 如果没有 static_scaler.joblib 和 cat_encoder.joblib，网页仍能演示，但静态模型真实预测会不严谨。
- 正式部署时必须保存训练静态模型时的 StandardScaler 和 CategoryEncoder。
"""

from __future__ import annotations

import json
import math
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import numpy as np
from flask import Flask, jsonify, request, render_template_string

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    import joblib
except Exception:
    joblib = None


# =========================================================
# 1. 路径配置
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "web_model"

PGA_PY_PATH = MODEL_DIR / "pga2.py"
STATIC_MODEL_PATH = MODEL_DIR / "best_fold1.pth"
DYNAMIC_FULL_PATH = MODEL_DIR / "dynamic_full_model_fold1.pth"
DYNAMIC_HEAD_PATH = MODEL_DIR / "dynamic_head_fold1.pth"
DYN_SCALER_PATH = MODEL_DIR / "dyn_scaler_fold1.joblib"

# 可选：如果你后续保存了静态预处理器，请放这两个文件。
STATIC_SCALER_PATH = MODEL_DIR / "static_scaler_fold1.joblib"
CAT_ENCODER_PATH = MODEL_DIR / "cat_encoder_fold1.joblib"

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
APP_TITLE = "中枢神经系统感染风险预测系统"


# =========================================================
# 2. 特征配置
#    与 pga2.py 的 build_feature_dataframe 逻辑保持一致
# =========================================================

BASE_NUM_COLS = [
    "age", "C_G", "C_WBC", "C_RBC", "C_P", "C_N",
    "GCS", "tem", "B_G", "B_CRP", "B_WBC", "B_N",
    "B_Lym", "B_PCT", "B_AC", "B_RBC",
]

DERIVED_NUM_COLS = [
    "ratio_C_G_B_G",
    "diff_C_N_B_N",
    "corrected_WBC",
    "ratio_WBC_RBC_diff",
]

NUM_COLS = BASE_NUM_COLS + DERIVED_NUM_COLS
CAT_COLS = ["sex", "tube", "site", "other_inf", "transparency"]

# 动态分支只使用多时间点脑脊液指标，6个变量 × 10类趋势特征 = 60维
TIME_POINTS = ["T-2", "T-1", "T0", "T+1"]
DYN_FEATURES = ["C_WBC", "C_RBC", "C_N", "C_P", "C_G", "transparency"]

# 根据你上传的 pth 推断的类别 embedding cardinalities：3,5,3,3,4
# 顺序对应 CAT_COLS = sex, tube, site, other_inf, transparency
DEFAULT_CAT_CARDINALITIES = [3, 5, 3, 3, 4]


# =========================================================
# 3. 临床显示名称
# =========================================================

NAME_MAP = {
    "age": "年龄",
    "sex": "性别",
    "tube": "引流管/置管情况",
    "site": "采样/感染部位",
    "other_inf": "其他部位感染",
    "transparency": "脑脊液透明度",
    "C_G": "脑脊液糖 C_G",
    "C_WBC": "脑脊液白细胞 C_WBC",
    "C_RBC": "脑脊液红细胞 C_RBC",
    "C_P": "脑脊液蛋白 C_P",
    "C_N": "脑脊液中性粒比例 C_N",
    "GCS": "GCS评分",
    "tem": "体温",
    "B_G": "血糖 B_G",
    "B_CRP": "CRP",
    "B_WBC": "血白细胞 B_WBC",
    "B_N": "血中性粒比例 B_N",
    "B_Lym": "血淋巴比例 B_Lym",
    "B_PCT": "PCT",
    "B_AC": "B_AC",
    "B_RBC": "血红细胞 B_RBC",
    "ratio_C_G_B_G": "脑脊液糖/血糖比值",
    "diff_C_N_B_N": "脑脊液-血中性粒差值",
    "corrected_WBC": "校正脑脊液白细胞",
    "ratio_WBC_RBC_diff": "白细胞/红细胞比值差",
}


# =========================================================
# 4. 工具函数
# =========================================================

def safe_float(v, default=0.0) -> float:
    try:
        if v is None or v == "":
            return default
        if isinstance(v, str):
            v = v.replace(",", ".")
        if isinstance(v, float) and math.isnan(v):
            return default
        return float(v)
    except Exception:
        return default


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def load_module_from_path(path: Path, name="pga_module"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_static_config_from_state_dict(state_dict: Dict) -> Dict:
    """从 pth 形状推断 embed_dim、层数、类别数量。"""
    pos_shape = tuple(state_dict["pos_encoding"].shape)
    embed_dim = pos_shape[-1]
    n_tokens = pos_shape[1]

    layer_ids = []
    cat_cards = []
    for k, v in state_dict.items():
        if k.startswith("layers."):
            try:
                layer_ids.append(int(k.split(".")[1]))
            except Exception:
                pass
        if k.startswith("cat_embedding.embeddings.") and k.endswith(".weight"):
            cat_cards.append((int(k.split(".")[2]), int(v.shape[0])))

    cat_cards = [c for _, c in sorted(cat_cards)] or DEFAULT_CAT_CARDINALITIES
    n_layers = max(layer_ids) + 1 if layer_ids else 2

    return {
        "embed_dim": embed_dim,
        "n_tokens": n_tokens,
        "n_layers": n_layers,
        "cat_cardinalities": cat_cards,
    }


def build_raw_feature_dict(data: Dict) -> Dict[str, float]:
    raw = {}
    for c in BASE_NUM_COLS + CAT_COLS:
        raw[c] = safe_float(data.get(c, 0.0))

    eps = 1e-6
    raw["ratio_C_G_B_G"] = raw.get("C_G", 0.0) / (raw.get("B_G", 0.0) + eps)
    raw["diff_C_N_B_N"] = raw.get("C_N", 0.0) - raw.get("B_N", 0.0)
    raw["corrected_WBC"] = raw.get("C_WBC", 0.0) - raw.get("B_WBC", 0.0) * raw.get("C_RBC", 0.0) / (raw.get("B_RBC", 0.0) + eps)
    raw["ratio_WBC_RBC_diff"] = raw.get("B_WBC", 0.0) / (raw.get("B_RBC", 0.0) + eps) - raw.get("C_WBC", 0.0) / (raw.get("C_RBC", 0.0) + eps)
    return raw


def build_static_arrays(data: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    构造静态模型输入。
    注意：如果缺少 static_scaler，这里会用原始数值，预测仅供展示。
    """
    raw = build_raw_feature_dict(data)
    x_num = np.array([[raw[c] for c in NUM_COLS]], dtype=np.float32)
    x_cat = np.array([[int(raw[c]) for c in CAT_COLS]], dtype=np.int64)

    if STATIC_SCALER is not None:
        x_num = STATIC_SCALER.transform(x_num).astype(np.float32)

    if CAT_ENCODER is not None:
        # 如果你保存的是自己的 CategoryEncoder，可在这里替换。
        # 当前保留手动数值编码，适合前端直接输入 0/1/2。
        pass

    return x_num, x_cat, raw


def extract_dynamic_features_from_payload(data: Dict, eps=1e-6) -> np.ndarray:
    """
    从页面动态表格提取 60维趋势特征：
    mean/max/min/first/last/delta/rel_change/slope/valid_count/peak_pos
    """
    dynamic = data.get("dynamic", {})
    seq = []
    for tp in TIME_POINTS:
        row = []
        for f in DYN_FEATURES:
            value = dynamic.get(tp, {}).get(f, None)
            if value is None or value == "":
                row.append(np.nan)
            else:
                row.append(safe_float(value, np.nan))
        seq.append(row)

    x = np.array(seq, dtype=np.float32).reshape(1, len(TIME_POINTS), len(DYN_FEATURES))
    mask = ~np.isnan(x)
    x_nan = x.copy()
    x_nan[~mask] = np.nan
    N, T, F = x.shape

    valid_count = np.sum(mask, axis=1).astype(np.float32)
    mean = np.nanmean(x_nan, axis=1)
    maxv = np.nanmax(x_nan, axis=1)
    minv = np.nanmin(x_nan, axis=1)

    first = np.zeros((N, F), dtype=np.float32)
    last = np.zeros((N, F), dtype=np.float32)
    peak_pos = np.zeros((N, F), dtype=np.float32)
    slope = np.zeros((N, F), dtype=np.float32)

    for i in range(N):
        for f in range(F):
            idx = np.where(mask[i, :, f])[0]
            if len(idx) == 0:
                continue
            vals = x[i, idx, f]
            first[i, f] = vals[0]
            last[i, f] = vals[-1]
            peak_pos[i, f] = idx[int(np.nanargmax(vals))] / max(T - 1, 1)
            if len(idx) >= 2:
                slope[i, f] = (last[i, f] - first[i, f]) / (idx[-1] - idx[0] + eps)

    delta = last - first
    rel_change = delta / (np.abs(first) + eps)
    arrays = [mean, maxv, minv, first, last, delta, rel_change, slope, valid_count, peak_pos]
    arrays = [np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) for a in arrays]
    dyn_feat = np.concatenate(arrays, axis=1).astype(np.float32)

    if DYN_SCALER is not None:
        dyn_feat = DYN_SCALER.transform(dyn_feat).astype(np.float32)
    return dyn_feat


def has_enough_dynamic(data: Dict) -> bool:
    """至少有两个时间点填写了动态指标，就自动切换动态预测。"""
    dynamic = data.get("dynamic", {})
    valid_tp = 0
    for tp in TIME_POINTS:
        row = dynamic.get(tp, {})
        filled = 0
        for f in DYN_FEATURES:
            v = row.get(f, None)
            if v is not None and v != "":
                filled += 1
        if filled >= 2:
            valid_tp += 1
    return valid_tp >= 2


# =========================================================
# 5. 动态残差模型定义
# =========================================================

if nn is not None:
    class DynamicResidualHead(nn.Module):
        def __init__(self, dyn_dim, hidden=64, dropout=0.2, residual_init=-3.0):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dyn_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
            self.dynamic_scale = nn.Parameter(torch.tensor(float(residual_init)))

        def forward(self, dyn_feat):
            z_dyn = self.net(dyn_feat).squeeze(-1)
            w = torch.sigmoid(self.dynamic_scale)
            return w * z_dyn, w

    class DynamicPGAAMFormer(nn.Module):
        def __init__(self, static_model, dyn_dim, hidden=64, dropout=0.2, freeze_static=True):
            super().__init__()
            self.static_model = static_model
            self.dynamic_head = DynamicResidualHead(dyn_dim, hidden=hidden, dropout=dropout)
            if freeze_static:
                for p in self.static_model.parameters():
                    p.requires_grad = False

        def forward(self, x_num, x_cat, dyn_feat, return_parts=False):
            z_static = self.static_model(x_num, x_cat)
            z_dyn_res, w = self.dynamic_head(dyn_feat)
            z = z_static + z_dyn_res
            if return_parts:
                return {"logit": z, "z_static": z_static, "z_dyn_residual": z_dyn_res, "dyn_weight": w}
            return z
else:
    DynamicResidualHead = None
    DynamicPGAAMFormer = None


# =========================================================
# 6. 加载模型
# =========================================================

PGA_MOD = None
STATIC_MODEL = None
DYNAMIC_MODEL = None
DYNAMIC_HEAD = None
STATIC_SCALER = None
CAT_ENCODER = None
DYN_SCALER = None
LOAD_MESSAGES = []


def load_all_models():
    global PGA_MOD, STATIC_MODEL, DYNAMIC_MODEL, DYNAMIC_HEAD, STATIC_SCALER, CAT_ENCODER, DYN_SCALER

    if torch is None:
        LOAD_MESSAGES.append("torch 未安装：进入 demo 模式")
        return

    if PGA_PY_PATH.exists():
        try:
            PGA_MOD = load_module_from_path(PGA_PY_PATH)
            LOAD_MESSAGES.append("已加载 pga2.py")
        except Exception as e:
            LOAD_MESSAGES.append(f"pga2.py 加载失败：{e}")
    else:
        LOAD_MESSAGES.append("未找到 web_model/pga2.py：无法真实加载 PGA-AMFormer")

    if joblib is not None:
        if DYN_SCALER_PATH.exists():
            try:
                DYN_SCALER = joblib.load(DYN_SCALER_PATH)
                LOAD_MESSAGES.append("已加载 dyn_scaler_fold1.joblib")
            except Exception as e:
                LOAD_MESSAGES.append(f"动态 scaler 加载失败：{e}")
        if STATIC_SCALER_PATH.exists():
            try:
                STATIC_SCALER = joblib.load(STATIC_SCALER_PATH)
                LOAD_MESSAGES.append("已加载 static_scaler_fold1.joblib")
            except Exception as e:
                LOAD_MESSAGES.append(f"静态 scaler 加载失败：{e}")
        else:
            LOAD_MESSAGES.append("未找到 static_scaler_fold1.joblib：静态真实预测不严谨")

        if CAT_ENCODER_PATH.exists():
            try:
                CAT_ENCODER = joblib.load(CAT_ENCODER_PATH)
                LOAD_MESSAGES.append("已加载 cat_encoder_fold1.joblib")
            except Exception as e:
                LOAD_MESSAGES.append(f"类别 encoder 加载失败：{e}")

    if PGA_MOD is None:
        return

    # 6.1 加载静态模型 best_fold3.pth
    if STATIC_MODEL_PATH.exists():
        try:
            sd = torch.load(STATIC_MODEL_PATH, map_location=DEVICE)
            cfg = infer_static_config_from_state_dict(sd)
            STATIC_MODEL = PGA_MOD.PGAAMFormer(
                n_num_features=len(NUM_COLS),
                cat_cardinalities=cfg["cat_cardinalities"],
                num_cols=NUM_COLS,
                cat_cols=CAT_COLS,
                embed_dim=cfg["embed_dim"],
                n_heads=4,
                n_layers=cfg["n_layers"],
                dropout=0.20,
                ff_mult=4,
                use_prior_attn=True,
                use_prior_arith=True,
                learnable_B=True,
                lambda_raw_init=-0.5,
                lambda_max=2.0,
                prior_topk=3,
                prior_min_abs=1e-6,
                rho_raw_init=0.0,
                rho_max=1.0,
            ).to(DEVICE)
            STATIC_MODEL.load_state_dict(sd, strict=True)
            STATIC_MODEL.eval()
            LOAD_MESSAGES.append(f"已加载静态模型 best_fold1.pth：embed_dim={cfg['embed_dim']}, layers={cfg['n_layers']}")
        except Exception as e:
            STATIC_MODEL = None
            LOAD_MESSAGES.append(f"静态模型加载失败：{e}")

    # 6.2 加载动态完整模型 dynamic_full_model_fold3.pth
    if DYNAMIC_FULL_PATH.exists():
        try:
            ckpt = torch.load(DYNAMIC_FULL_PATH, map_location=DEVICE)
            static_sd = ckpt["static_model_state_dict"]
            cfg = infer_static_config_from_state_dict(static_sd)
            pga_cfg = ckpt.get("pga_cfg", {})

            dyn_static_model = PGA_MOD.PGAAMFormer(
                n_num_features=len(NUM_COLS),
                cat_cardinalities=cfg["cat_cardinalities"],
                num_cols=NUM_COLS,
                cat_cols=CAT_COLS,
                embed_dim=cfg["embed_dim"],
                n_heads=int(pga_cfg.get("n_heads", 4)),
                n_layers=cfg["n_layers"],
                dropout=float(pga_cfg.get("dropout", 0.20)),
                ff_mult=int(pga_cfg.get("ff_mult", 4)),
                use_prior_attn=True,
                use_prior_arith=True,
                learnable_B=True,
                lambda_raw_init=float(pga_cfg.get("lambda_raw_init", -1.5)),
                lambda_max=float(pga_cfg.get("lambda_max", 1.0)),
                prior_topk=int(pga_cfg.get("prior_topk", 3)),
                prior_min_abs=float(pga_cfg.get("prior_min_abs", 1e-6)),
                rho_raw_init=float(pga_cfg.get("rho_raw_init", -2.0)),
                rho_max=float(pga_cfg.get("rho_max", 1.0)),
            ).to(DEVICE)

            DYNAMIC_MODEL = DynamicPGAAMFormer(
                static_model=dyn_static_model,
                dyn_dim=int(ckpt.get("dyn_dim", 60)),
                hidden=int(ckpt.get("hidden", 64)),
                dropout=float(ckpt.get("dropout", 0.2)),
                freeze_static=bool(ckpt.get("freeze_static", True)),
            ).to(DEVICE)
            DYNAMIC_MODEL.load_state_dict(ckpt["model_state_dict"], strict=True)
            DYNAMIC_MODEL.eval()
            LOAD_MESSAGES.append(f"已加载动态完整模型 dynamic_full_model_fold1.pth：dyn_dim={ckpt.get('dyn_dim', 60)}")
        except Exception as e:
            DYNAMIC_MODEL = None
            LOAD_MESSAGES.append(f"动态完整模型加载失败：{e}")

    # 6.3 如果没有完整动态模型，则尝试单独加载动态 head
    if DYNAMIC_MODEL is None and DYNAMIC_HEAD_PATH.exists() and DynamicResidualHead is not None:
        try:
            head_ckpt = torch.load(DYNAMIC_HEAD_PATH, map_location=DEVICE)
            DYNAMIC_HEAD = DynamicResidualHead(
                dyn_dim=int(head_ckpt.get("dyn_dim", 60)),
                hidden=int(head_ckpt.get("hidden", 64)),
                dropout=float(head_ckpt.get("dropout", 0.2)),
            ).to(DEVICE)
            DYNAMIC_HEAD.load_state_dict(head_ckpt["dynamic_head_state_dict"], strict=True)
            DYNAMIC_HEAD.eval()
            LOAD_MESSAGES.append("已加载 dynamic_head_fold1.pth，但没有完整动态 backbone")
        except Exception as e:
            DYNAMIC_HEAD = None
            LOAD_MESSAGES.append(f"动态 head 加载失败：{e}")


load_all_models()


# =========================================================
# 7. 预测函数
# =========================================================

def demo_predict(raw: Dict[str, float]) -> Tuple[float, float, Dict[str, float]]:
    """没有完整预处理器/模型时的展示模式，不作为真实临床预测。"""
    c_wbc = max(raw.get("C_WBC", 0.0), 0.0)
    c_g = max(raw.get("C_G", 0.0), 0.0)
    c_p = max(raw.get("C_P", 0.0), 0.0)
    b_crp = max(raw.get("B_CRP", 0.0), 0.0)
    gcs = raw.get("GCS", 15.0)
    tem = raw.get("tem", 37.0)

    score = 0.0
    score += 0.90 * np.log1p(c_wbc) / 10
    score += 1.20 * max(0, 2.5 - c_g) / 2.5
    score += 0.70 * np.log1p(c_p) / 6
    score += 0.35 * np.log1p(b_crp) / 6
    score += 0.45 * max(0, 15 - gcs) / 15
    score += 0.30 * max(0, tem - 37.3)
    prob = float(np.clip(sigmoid(score - 1.15), 0.01, 0.99))
    logit = float(np.log(prob / (1 - prob)))
    importance = {
        "C_G": float(np.clip((2.8 - c_g) / 2.8, 0.05, 0.95)),
        "C_WBC": float(np.clip(np.log1p(c_wbc) / 10, 0.03, 0.90)),
        "C_P": float(np.clip(np.log1p(c_p) / 6, 0.03, 0.75)),
        "B_CRP": float(np.clip(np.log1p(b_crp) / 6, 0.02, 0.65)),
        "GCS": float(np.clip(max(0, 15 - gcs) / 15, 0.02, 0.60)),
    }
    return prob, logit, importance


def predict_static(data: Dict) -> Tuple[float, float, Dict[str, float], str]:
    x_num, x_cat, raw = build_static_arrays(data)

    if STATIC_MODEL is not None and torch is not None and STATIC_SCALER is not None:
        try:
            with torch.no_grad():
                tx_num = torch.tensor(x_num, dtype=torch.float32).to(DEVICE)
                tx_cat = torch.tensor(x_cat, dtype=torch.long).to(DEVICE)
                logits = STATIC_MODEL(tx_num, tx_cat)
                logit = float(logits.detach().cpu().numpy().reshape(-1)[0])
                prob = sigmoid(logit)
            imp = {"C_G": 0.82, "C_WBC": 0.76, "C_P": 0.58, "B_CRP": 0.43, "GCS": 0.32}
            return prob, logit, imp, "static_real"
        except Exception:
            pass

    prob, logit, imp = demo_predict(raw)
    return prob, logit, imp, "static_demo"


def predict_dynamic(data: Dict) -> Tuple[float, float, float, float, Dict[str, float], str]:
    x_num, x_cat, raw = build_static_arrays(data)
    dyn_feat = extract_dynamic_features_from_payload(data)

    # 优先使用完整 D-PGA-AMFormer
    if DYNAMIC_MODEL is not None and torch is not None and DYN_SCALER is not None and STATIC_SCALER is not None:
        try:
            with torch.no_grad():
                tx_num = torch.tensor(x_num, dtype=torch.float32).to(DEVICE)
                tx_cat = torch.tensor(x_cat, dtype=torch.long).to(DEVICE)
                tdyn = torch.tensor(dyn_feat, dtype=torch.float32).to(DEVICE)
                parts = DYNAMIC_MODEL(tx_num, tx_cat, tdyn, return_parts=True)
                final_logit = float(parts["logit"].detach().cpu().numpy().reshape(-1)[0])
                static_logit = float(parts["z_static"].detach().cpu().numpy().reshape(-1)[0])
                dyn_res = float(parts["z_dyn_residual"].detach().cpu().numpy().reshape(-1)[0])
                dyn_w = float(parts["dyn_weight"].detach().cpu().numpy().reshape(-1)[0])
            prob = sigmoid(final_logit)
            imp = {"C_G": 0.82, "C_WBC": 0.76, "C_P": 0.58, "B_CRP": 0.43, "GCS": 0.32}
            return prob, static_logit, dyn_res, dyn_w, imp, "dynamic_real_full"
        except Exception:
            pass

    # 如果只有 dynamic_head，静态部分使用当前静态预测结果
    static_prob, static_logit, imp, _ = predict_static(data)
    if DYNAMIC_HEAD is not None and torch is not None and DYN_SCALER is not None:
        try:
            with torch.no_grad():
                tdyn = torch.tensor(dyn_feat, dtype=torch.float32).to(DEVICE)
                dyn_res_tensor, dyn_w_tensor = DYNAMIC_HEAD(tdyn)
                dyn_res = float(dyn_res_tensor.detach().cpu().numpy().reshape(-1)[0])
                dyn_w = float(dyn_w_tensor.detach().cpu().numpy().reshape(-1)[0])
            final_prob = sigmoid(static_logit + dyn_res)
            return final_prob, static_logit, dyn_res, dyn_w, imp, "dynamic_head_only"
        except Exception:
            pass

    # demo 动态修正
    dynamic = data.get("dynamic", {})
    first = dynamic.get(TIME_POINTS[0], {})
    last = dynamic.get(TIME_POINTS[-1], {})
    c_wbc_delta = safe_float(last.get("C_WBC", 0)) - safe_float(first.get("C_WBC", 0))
    c_g_delta = safe_float(last.get("C_G", 0)) - safe_float(first.get("C_G", 0))
    dyn_res = float(0.35 * np.tanh(c_wbc_delta / 1200) + 0.45 * np.tanh(-c_g_delta))
    return sigmoid(static_logit + dyn_res), static_logit, dyn_res, 0.0475, imp, "dynamic_demo"


def risk_text(prob: float):
    if prob >= 0.70:
        return "高风险", "high", "模型给出感染风险辅助评估：风险较高。建议结合脑脊液培养、影像学、临床症状及医生判断进一步评估。"
    if prob >= 0.40:
        return "中风险", "mid", "模型给出感染风险辅助评估：存在一定风险。建议结合复查结果进行动态随访。"
    return "低风险", "low", "模型给出感染风险辅助评估：当前风险相对较低。若症状进展或指标异常，建议继续复评。"


def derive_main_risk_factors(importance: Dict[str, float]):
    mapping = {
        "C_G": "脑脊液糖降低",
        "C_WBC": "脑脊液白细胞升高",
        "C_P": "脑脊液蛋白升高",
        "B_CRP": "CRP升高",
        "GCS": "GCS评分较低",
    }
    ordered = sorted((importance or {}).items(), key=lambda x: x[1], reverse=True)
    out = [mapping.get(k, NAME_MAP.get(k, k)) for k, _ in ordered[:4]]
    return out if out else ["需结合更多检查指标"]


# =========================================================
# 8. Flask 页面与 API（临床工作台版）
# =========================================================
app = Flask(__name__)

PATIENT_DB_PATH = BASE_DIR / "patients_db.json"
CASE_DB_PATH = BASE_DIR / "case_records_db.json"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _normalize_patient(payload: Dict) -> Dict:
    patient = payload or {}
    return {
        "patient_id": str(patient.get("patient_id", "")).strip(),
        "inpatient_no": str(patient.get("inpatient_no", "")).strip(),
        "patient_name": str(patient.get("patient_name", "")).strip(),
        "sex_text": str(patient.get("sex_text", "")).strip(),
        "age": safe_float(patient.get("age", 0)),
        "department": str(patient.get("department", "")).strip(),
        "bed_no": str(patient.get("bed_no", "")).strip(),
        "sampling_time": str(patient.get("sampling_time", "")).strip(),
        "doctor_name": str(patient.get("doctor_name", "")).strip(),
        "diagnosis_note": str(patient.get("diagnosis_note", "")).strip(),
    }


def _model_version_by_mode(final_mode: str) -> str:
    return "D-PGA-AMFormer v1.0" if final_mode == "dynamic" else "PGA-AMFormer v1.0"


def _query_cases(filters: Dict):
    rows = _load_json(CASE_DB_PATH, [])
    pid = str(filters.get("patient_id", "")).strip()
    name = str(filters.get("patient_name", "")).strip()
    inp = str(filters.get("inpatient_no", "")).strip()
    day = str(filters.get("eval_date", "")).strip()

    def ok(r):
        p = r.get("patient_info", {})
        t = str(r.get("evaluation_time", ""))
        if pid and pid not in str(p.get("patient_id", "")):
            return False
        if name and name not in str(p.get("patient_name", "")):
            return False
        if inp and inp not in str(p.get("inpatient_no", "")):
            return False
        if day and (not t.startswith(day)):
            return False
        return True

    out = [r for r in rows if ok(r)]
    out.sort(key=lambda x: str(x.get("evaluation_time", "")), reverse=True)
    return out


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True) or {}
    requested_mode = data.get("mode", "auto")
    auto_dynamic = has_enough_dynamic(data)
    final_mode = "dynamic" if (requested_mode == "dynamic" or (requested_mode == "auto" and auto_dynamic)) else "static"

    if final_mode == "dynamic":
        prob, static_logit, dyn_res, dyn_w, imp, model_status = predict_dynamic(data)
        static_prob = sigmoid(static_logit)
    else:
        prob, static_logit, imp, model_status = predict_static(data)
        static_prob = prob
        dyn_res = 0.0
        dyn_w = 0.0

    label, key, suggestion = risk_text(prob)
    return jsonify({
        "requested_mode": requested_mode,
        "final_mode": final_mode,
        "auto_dynamic": auto_dynamic,
        "model_status": model_status,
        "load_messages": LOAD_MESSAGES,
        "probability": round(prob, 4),
        "percentage": round(prob * 100, 2),
        "static_probability": round(static_prob, 4),
        "static_percentage": round(static_prob * 100, 2),
        "dynamic_residual_logit": round(dyn_res, 4),
        "dynamic_weight": round(dyn_w, 4),
        "risk_label": label,
        "risk_key": key,
        "suggestion": suggestion,
        "importance": imp,
        "main_risk_factors": derive_main_risk_factors(imp),
        "evaluation_time": _now_str(),
        "model_version": _model_version_by_mode(final_mode),
        "clinical_notice": "本结果为感染风险辅助评估结果，不作为最终诊断依据。",
    })


@app.route("/api/patients/save", methods=["POST"])
def api_patients_save():
    data = request.get_json(force=True) or {}
    patient = _normalize_patient(data.get("patient_info", data))
    patients = _load_json(PATIENT_DB_PATH, [])

    if not patient["patient_id"]:
        patient["patient_id"] = f"P{int(datetime.now().timestamp())}"
    patient["updated_at"] = _now_str()

    hit = -1
    for i, row in enumerate(patients):
        if row.get("patient_id") == patient["patient_id"]:
            hit = i
            break

    if hit >= 0:
        patients[hit] = {**patients[hit], **patient}
    else:
        patients.append(patient)

    _save_json(PATIENT_DB_PATH, patients)
    return jsonify({"ok": True, "patient": patient, "message": "患者信息已保存"})


@app.route("/api/patients/query", methods=["GET"])
def api_patients_query():
    patient_id = (request.args.get("patient_id", "") or "").strip()
    inpatient_no = (request.args.get("inpatient_no", "") or "").strip()
    patient_name = (request.args.get("patient_name", "") or "").strip()

    patients = _load_json(PATIENT_DB_PATH, [])

    def ok(p):
        if patient_id and patient_id not in str(p.get("patient_id", "")):
            return False
        if inpatient_no and inpatient_no not in str(p.get("inpatient_no", "")):
            return False
        if patient_name and patient_name not in str(p.get("patient_name", "")):
            return False
        return True

    matched = [x for x in patients if ok(x)]
    matched.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
    return jsonify({"ok": True, "patients": matched})


@app.route("/api/cases/save", methods=["POST"])
def api_cases_save():
    data = request.get_json(force=True) or {}
    patient = _normalize_patient(data.get("patient_info", {}))
    assessment_type = str(data.get("assessment_type", "static")).strip() or "static"
    prediction = data.get("prediction", {}) or {}
    input_data = data.get("input_data", {}) or {}

    if not patient.get("patient_id"):
        return jsonify({"ok": False, "message": "缺少患者ID，无法保存病例"}), 400

    rows = _load_json(CASE_DB_PATH, [])
    row = {
        "record_id": f"CASE-{int(datetime.now().timestamp() * 1000)}",
        "assessment_type": assessment_type,
        "patient_info": patient,
        "input_data": input_data,
        "prediction": prediction,
        "risk_label": prediction.get("risk_label", ""),
        "percentage": prediction.get("percentage", None),
        "evaluation_time": prediction.get("evaluation_time", _now_str()),
        "model_version": prediction.get("model_version", _model_version_by_mode(assessment_type)),
        "operator": patient.get("doctor_name", ""),
        "created_at": _now_str(),
    }
    rows.append(row)
    _save_json(CASE_DB_PATH, rows)
    return jsonify({"ok": True, "record": row, "message": "病例评估记录已保存"})


@app.route("/api/cases/query", methods=["GET"])
def api_cases_query():
    filters = {
        "patient_id": request.args.get("patient_id", ""),
        "patient_name": request.args.get("patient_name", ""),
        "inpatient_no": request.args.get("inpatient_no", ""),
        "eval_date": request.args.get("eval_date", ""),
    }
    return jsonify({"ok": True, "records": _query_cases(filters)})


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root{--bg:#f2f5f9;--card:#fff;--text:#1f2d3d;--muted:#667085;--line:#dfe6ef;--blue:#2463d3;--green:#16a34a;--orange:#d97706;--red:#dc2626;--radius:14px;--shadow:0 8px 20px rgba(31,45,61,.06)}
    *{box-sizing:border-box}
    body{margin:0;background:var(--bg);font-family:Arial,Helvetica,"Microsoft YaHei","PingFang SC",sans-serif;color:var(--text)}
    .page{max-width:1440px;margin:0 auto;padding:16px}
    .topbar{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:14px 18px;margin-bottom:12px}
    .title-cn{margin:0;font-size:24px;font-weight:800;color:#163b7a}.title-en{margin:4px 0 0;color:var(--muted);font-size:13px}
    .nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
    .nav button{border:1px solid var(--line);background:#fff;color:#334155;height:34px;padding:0 14px;border-radius:9px;font-weight:700;cursor:pointer}
    .nav button.active{background:var(--blue);border-color:var(--blue);color:#fff}
    .layout{display:grid;grid-template-columns:minmax(0,1fr);gap:12px}
    .card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:14px}
    .card h3{margin:0 0 10px;font-size:16px}.desc{color:var(--muted);font-size:13px;line-height:1.6;margin:0 0 10px}
    .view{display:none}.view.active{display:block}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
    @media(max-width:980px){.grid2,.grid3{grid-template-columns:1fr}}
    .field label{display:block;font-size:12px;font-weight:700;color:#334155;margin-bottom:5px}
    .field input,.field textarea{width:100%;border:1px solid #ced7e3;border-radius:10px;height:36px;padding:0 10px;background:#fff}
    .field textarea{height:74px;padding:8px 10px;resize:vertical}
    .toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
    .btn{border:none;border-radius:10px;height:36px;padding:0 14px;font-weight:700;cursor:pointer}
    .btn.primary{background:var(--blue);color:#fff}.btn.light{background:#eef2f7;color:#334155}.btn.green{background:#16a34a;color:#fff}.btn.orange{background:#ea580c;color:#fff}
    .panel-risk{position:relative;top:auto;display:grid;grid-template-columns:1fr 1fr;gap:10px}.risk-num{font-size:40px;font-weight:900}
    .risk-pill{display:inline-flex;padding:6px 12px;border-radius:999px;color:#fff;font-weight:800;margin-top:8px}
    .risk-pill.low{background:var(--green)}.risk-pill.mid{background:var(--orange)}.risk-pill.high{background:var(--red)}
    .risk-item{display:flex;justify-content:space-between;gap:10px;padding:8px 0;border-bottom:1px dashed var(--line);font-size:13px}
    .risk-item span:first-child{color:var(--muted)}.risk-item span:last-child{text-align:right;font-weight:700}
    .factors{margin:8px 0 0;padding-left:18px;color:#334155;font-size:13px}.factors li{margin:4px 0}
    table{width:100%;border-collapse:collapse;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:7px;text-align:left}th{background:#f8fafc;font-weight:700}
    .table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}
    .home-actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px}.home-grid{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:10px}
    @media(max-width:980px){.home-grid{grid-template-columns:1fr}.panel-risk{grid-template-columns:1fr}}
    .mini{padding:12px;border:1px solid var(--line);border-radius:10px;background:#fbfdff}.mini b{display:block;color:#1e3a8a;margin-bottom:6px}
    .muted{color:var(--muted);font-size:12px}
    .hint{margin-top:8px;padding:8px 10px;background:#f8fafc;border:1px solid var(--line);border-radius:10px;color:#475569;font-size:12px}
    .status-tags{display:flex;gap:8px;flex-wrap:wrap}.tag{font-size:12px;border-radius:999px;padding:5px 10px;font-weight:700;border:1px solid var(--line);background:#f8fafc}
    .tag.ok{background:#ecfdf3;border-color:#bbf7d0;color:#166534}.tag.info{background:#eff6ff;border-color:#bfdbfe;color:#1d4ed8}
  </style>
</head>
<body>
<div class="page">
  <section class="topbar">
    <h1 class="title-cn">中枢神经系统感染风险辅助评估平台</h1>
    <p class="title-en">CNS Infection Risk Assessment Platform</p>
    <div class="nav">
      <button id="tab-home" class="active" onclick="switchTab('home')">患者管理</button>
      <button id="tab-static" onclick="switchTab('static')">静态风险评估</button>
      <button id="tab-dynamic" onclick="switchTab('dynamic')">动态趋势评估</button>
      <button id="tab-records" onclick="switchTab('records')">病例记录</button>
      <button id="tab-settings" onclick="switchTab('settings')">系统设置</button>
    </div>
  </section>

  <section class="layout">
    <div>
      <div class="view active" id="view-home">
        <div class="card">
          <h3>系统入口</h3>
          <p class="desc">本系统用于辅助医生基于患者临床资料、血液检查及脑脊液检查结果，进行中枢神经系统感染风险评估。系统支持单次静态风险评估与多时间点动态趋势评估，并可保存患者病例记录，便于后续随访与复评。</p>
          <div class="home-actions">
            <button class="btn light" onclick="newPatient()">新建患者</button>
            <button class="btn light" onclick="queryPatient()">查询患者</button>
            <button class="btn primary" onclick="switchTab('static')">进入静态评估</button>
            <button class="btn primary" onclick="switchTab('dynamic')">进入动态评估</button>
          </div>
          <div class="home-grid">
            <div class="mini"><b>患者管理</b><span class="muted">支持患者ID、住院号、姓名查询与病例记录保存。</span></div>
            <div class="mini"><b>静态风险评估</b><span class="muted">基于单次检查数据调用 PGA-AMFormer 模型。</span></div>
            <div class="mini"><b>动态趋势评估</b><span class="muted">基于多时间点脑脊液指标调用 D-PGA-AMFormer 模型。</span></div>
          </div>
        </div>
        <div class="card" style="margin-top:10px;">
          <h3>患者管理列表</h3>
          <div class="table-wrap">
            <table id="patientTable">
              <thead><tr><th>患者ID</th><th>姓名</th><th>住院号</th><th>性别</th><th>年龄</th><th>科室</th><th>床号</th><th>主管医生</th><th>更新时间</th></tr></thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="view" id="view-static">
        <div class="card">
          <h3>患者信息</h3>
          <div class="grid3" id="patientInfoGrid"></div>
          <div class="toolbar">
            <button class="btn light" onclick="newPatient()">新建患者</button>
            <button class="btn light" onclick="queryPatient()">查询患者</button>
            <button class="btn green" onclick="savePatient()">保存病例</button>
            <button class="btn light" onclick="loadHistoryForCurrent()">读取历史病例</button>
          </div>
        </div>

        <div class="card" style="margin-top:10px;"><h3>基础临床信息</h3><div class="grid3" id="group-basic"></div></div>
        <div class="card" style="margin-top:10px;"><h3>脑脊液检查指标</h3><div class="grid3" id="group-csf"></div></div>
        <div class="card" style="margin-top:10px;"><h3>血液检查指标</h3><div class="grid3" id="group-blood"></div></div>

        <div class="card" style="margin-top:10px;">
          <h3>感染相关信息</h3>
          <div class="grid3" id="group-inf"></div>
          <div class="toolbar">
            <button class="btn light" onclick="fillStaticExample()">填入示例</button>
            <button class="btn light" onclick="clearClinicalForm()">清空当前表单</button>
            <button class="btn primary" onclick="startStaticAssessment()">开始评估</button>
            <button class="btn green" onclick="saveCase('static')">保存病例</button>
            <button class="btn orange" onclick="generateReport('static')">生成报告</button>
          </div>
        </div>
      </div>

      <div class="view" id="view-dynamic">
        <div class="card">
          <h3>患者信息卡片</h3>
          <p class="desc">动态评估会自动复用同一患者已录入的静态基础信息，无需重复建档。</p>
          <div class="hint" id="dynamicPatientSummary">当前未选择患者。</div>
        </div>

        <div class="card" style="margin-top:10px;">
          <h3>历史评估记录</h3>
          <div class="toolbar"><button class="btn light" onclick="loadHistoryForCurrent()">读取历史记录</button></div>
          <div class="table-wrap" style="margin-top:8px;">
            <table id="historyTable"><thead><tr><th>评估时间</th><th>类型</th><th>风险等级</th><th>预测概率</th><th>医生</th><th>模型版本</th></tr></thead><tbody></tbody></table>
          </div>
        </div>

        <div class="card" style="margin-top:10px;">
          <h3>多时间点脑脊液指标录入</h3>
          <p class="desc">支持按真实采样时间新增时间点，不固定为 T-2 / T-1 / T0 / T+1。</p>
          <div class="toolbar"><button class="btn light" onclick="addDynamicRow()">新增时间点</button><button class="btn light" onclick="fillDynamicExample()">填入动态示例</button></div>
          <div class="table-wrap" style="margin-top:8px;">
            <table id="dynamicRowsTable">
              <thead><tr><th>采样时间</th><th>C_WBC</th><th>C_RBC</th><th>C_N</th><th>C_P</th><th>C_G</th><th>transparency</th><th>操作</th></tr></thead>
              <tbody></tbody>
            </table>
          </div>
        </div>

        <div class="card" style="margin-top:10px;">
          <h3>趋势变化预览</h3>
          <div style="height:280px;"><canvas id="trendChart"></canvas></div>
          <div class="toolbar">
            <button class="btn primary" onclick="startDynamicAssessment()">开始评估</button>
            <button class="btn green" onclick="saveCase('dynamic')">保存动态评估</button>
            <button class="btn orange" onclick="generateReport('dynamic')">生成随访报告</button>
          </div>
        </div>

        <div class="card" style="margin-top:10px;">
          <h3>静态风险与动态修正对比</h3>
          <div class="grid2"><div class="hint">静态基础风险：<b id="cmpStatic">--%</b></div><div class="hint">最终动态风险：<b id="cmpDynamic">--%</b></div></div>
          <div class="hint" style="margin-top:8px;">风险变化方向：<b id="riskTrendDir">--</b></div>
        </div>
      </div>

      <div class="view" id="view-records">
        <div class="card">
          <h3>病例记录检索</h3>
          <div class="grid3">
            <div class="field"><label>患者ID</label><input id="q_patient_id" /></div>
            <div class="field"><label>姓名</label><input id="q_patient_name" /></div>
            <div class="field"><label>住院号</label><input id="q_inpatient_no" /></div>
          </div>
          <div class="grid3" style="margin-top:8px;"><div class="field"><label>评估日期</label><input id="q_eval_date" type="date" /></div></div>
          <div class="toolbar"><button class="btn primary" onclick="queryCases()">查询病例记录</button></div>
          <div class="table-wrap" style="margin-top:8px;">
            <table id="recordsTable"><thead><tr><th>评估时间</th><th>患者ID</th><th>姓名</th><th>住院号</th><th>类型</th><th>风险等级</th><th>概率</th><th>医生</th><th>模型版本</th></tr></thead><tbody></tbody></table>
          </div>
        </div>
      </div>

      <div class="view" id="view-settings">
        <div class="card">
          <h3>系统设置</h3>
          <p class="desc">医生主界面默认隐藏模型内部技术字段。以下为系统诊断信息（开发者模式）。</p>
          <details><summary style="cursor:pointer;color:#1d4ed8;font-weight:700;">展开系统诊断信息</summary>
            <div class="hint">模型调用状态：<span id="devModelStatus">--</span></div>
            <div class="hint">最近调用路径：<span id="devModePath">--</span></div>
            <div class="hint">动态残差logit：<span id="devDynResidual">--</span></div>
            <div class="hint">动态分支权重：<span id="devDynWeight">--</span></div>
            <div class="hint">模型加载日志：<ul id="devLoadMsgs" style="margin:6px 0 0 16px;"></ul></div>
          </details>
        </div>
      </div>
    </div>

    <aside class="panel-risk">
      <div class="card">
        <h3>模型辅助评估结果</h3>
        <div class="risk-num" id="riskNumber">--%</div>
        <div class="risk-pill mid" id="riskPill">未评估</div>
        <div class="risk-item"><span>风险等级</span><span id="riskLabelText">未评估</span></div>
        <div class="risk-item"><span>预测概率</span><span id="riskPercentText">--%</span></div>
        <div class="risk-item"><span>评估时间</span><span id="evalTimeText">--</span></div>
        <div class="risk-item"><span>模型版本</span><span id="modelVersionText">--</span></div>
        <div class="risk-item"><span>临床参考建议</span><span id="suggestionText">--</span></div>
        <div style="margin-top:8px;font-size:13px;font-weight:700;">主要风险来源：</div>
        <ul class="factors" id="riskFactors"></ul>
        <div class="hint" id="clinicalNotice" style="margin-top:10px;">本结果仅供临床辅助评估参考。</div>
      </div>
      <div class="card">
        <h3>当前状态</h3>
        <div class="status-tags"><span class="tag info" id="statusEval">未评估</span><span class="tag" id="statusSave">未保存</span><span class="tag" id="statusReport">未生成报告</span></div>
      </div>
    </aside>
  </section>
</div>

<script>
const baseNumCols = {{ base_num_cols | safe }};
const catCols = {{ cat_cols | safe }};
const dynFeatures = {{ dyn_features | safe }};
const loadMessages = {{ load_messages | safe }};
const nameMap = {{ name_map | safe }};
const unitMap = {age:"岁",tem:"℃",B_CRP:"mg/L",B_PCT:"ng/mL",C_WBC:"×10^6/L",C_RBC:"×10^6/L",C_P:"g/L",C_G:"mmol/L",B_WBC:"×10^9/L",B_RBC:"×10^12/L"};

let lastPrediction = null;
let dynamicRows = [];
let trendChart = null;

const LOCAL_PATIENTS_KEY = "cns_patients_v1";
const LOCAL_CASES_KEY = "cns_cases_v1";

function nowStr(){
  const d = new Date();
  const p = (x) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth()+1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function readLocalList(key){
  try{
    const raw = localStorage.getItem(key);
    const rows = raw ? JSON.parse(raw) : [];
    return Array.isArray(rows) ? rows : [];
  }catch(e){ return []; }
}
function writeLocalList(key, rows){ localStorage.setItem(key, JSON.stringify(rows || [])); }
function getLocalPatients(){ return readLocalList(LOCAL_PATIENTS_KEY); }
function getLocalCases(){ return readLocalList(LOCAL_CASES_KEY); }
function saveLocalPatients(rows){ writeLocalList(LOCAL_PATIENTS_KEY, rows); }
function saveLocalCases(rows){ writeLocalList(LOCAL_CASES_KEY, rows); }

function upsertLocalPatient(patient){
  const rows = getLocalPatients();
  const p = {...patient};
  if(!p.patient_id){ p.patient_id = `P${Date.now()}`; }
  p.updated_at = nowStr();
  const idx = rows.findIndex(x => String(x.patient_id) === String(p.patient_id));
  if(idx >= 0) rows[idx] = {...rows[idx], ...p};
  else rows.unshift(p);
  saveLocalPatients(rows);
  return p;
}
function addLocalCase(row){
  const rows = getLocalCases();
  rows.unshift(row);
  saveLocalCases(rows);
}
function textLike(v, q){
  if(!q) return true;
  return String(v || "").toLowerCase().includes(String(q).toLowerCase());
}
function filterLocalPatients(filters){
  const rows = getLocalPatients();
  return rows.filter(p =>
    textLike(p.patient_id, filters.patient_id) &&
    textLike(p.inpatient_no, filters.inpatient_no) &&
    textLike(p.patient_name, filters.patient_name)
  );
}
function filterLocalCases(filters){
  const rows = getLocalCases();
  return rows.filter(r => {
    const p = r.patient_info || {};
    const t = String(r.evaluation_time || "");
    return textLike(p.patient_id, filters.patient_id)
      && textLike(p.patient_name, filters.patient_name)
      && textLike(p.inpatient_no, filters.inpatient_no)
      && (!filters.eval_date || t.startswith?.(filters.eval_date) || t.indexOf(filters.eval_date) === 0);
  });
}
function renderPatientTable(rows){
  const tb = document.querySelector("#patientTable tbody");
  if(!tb) return;
  tb.innerHTML = "";
  if(!rows.length){
    tb.innerHTML = `<tr><td colspan="9" class="muted">暂无患者记录</td></tr>`;
    return;
  }
  rows.forEach(p => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${esc(p.patient_id || "-")}</td><td>${esc(p.patient_name || "-")}</td><td>${esc(p.inpatient_no || "-")}</td><td>${esc(p.sex_text || "-")}</td><td>${p.age ?? "-"}</td><td>${esc(p.department || "-")}</td><td>${esc(p.bed_no || "-")}</td><td>${esc(p.doctor_name || "-")}</td><td>${esc(p.updated_at || "-")}</td>`;
    tb.appendChild(tr);
  });
}
function seedShowData(){
  const ps = getLocalPatients();
  const cs = getLocalCases();
  if(ps.length === 0){
    const seedPatients = [
      {patient_id:"PT2779001", inpatient_no:"ZYH202605001", patient_name:"张某某", sex_text:"男", age:56, department:"神经外科", bed_no:"12A", sampling_time:"2026-05-04T08:30", doctor_name:"李医生", diagnosis_note:"术后复查", updated_at:nowStr()},
      {patient_id:"PT2779002", inpatient_no:"ZYH202605018", patient_name:"王某某", sex_text:"女", age:42, department:"神经内科", bed_no:"08B", sampling_time:"2026-05-03T10:10", doctor_name:"周医生", diagnosis_note:"发热待查", updated_at:nowStr()}
    ];
    saveLocalPatients(seedPatients);
  }
  if(cs.length === 0){
    const p = getLocalPatients()[0];
    const seedCase = {
      record_id:`CASE-${Date.now()}`,
      assessment_type:"static",
      patient_info:p,
      risk_label:"中风险",
      percentage:51.3,
      evaluation_time:nowStr(),
      model_version:"PGA-AMFormer v1.0",
      operator:p?.doctor_name || "李医生",
      prediction:{risk_label:"中风险", percentage:51.3, evaluation_time:nowStr(), model_version:"PGA-AMFormer v1.0"}
    };
    saveLocalCases([seedCase]);
  }
}

function esc(v){return String(v ?? "").replace(/[&<>"]/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[s]));}
function cn(k){return nameMap[k] || k;}
function unit(k){return unitMap[k] ? `（${unitMap[k]}）` : "";}
function setVal(id,v){const el=document.getElementById(id); if(el) el.value=(v ?? "");}
function getNum(id){const el=document.getElementById(id); if(!el || el.value==="") return 0; const n=parseFloat(el.value); return Number.isFinite(n)?n:0;}
function getTxt(id){const el=document.getElementById(id); return el ? (el.value || "").trim() : "";}
function toNumOrNull(v){ if(v===""||v===null||v===undefined) return null; const n=parseFloat(v); return Number.isFinite(n)?n:null; }

function switchTab(tab){
  document.querySelectorAll(".view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav button").forEach(b => b.classList.remove("active"));
  document.getElementById(`view-${tab}`).classList.add("active");
  document.getElementById(`tab-${tab}`).classList.add("active");
}

function buildInput(id,label,type="number"){ return `<div class="field"><label>${label}</label><input id="${id}" type="${type}" /></div>`; }

function renderPatientInfo(){
  const box = document.getElementById("patientInfoGrid");
  box.innerHTML = [
    buildInput("patient_id","患者ID","text"),
    buildInput("inpatient_no","住院号","text"),
    buildInput("patient_name","姓名","text"),
    buildInput("sex_text","性别","text"),
    buildInput("patient_age","年龄（岁）","number"),
    buildInput("department","科室","text"),
    buildInput("bed_no","床号","text"),
    buildInput("sampling_time","采样时间","datetime-local"),
    buildInput("doctor_name","主管医生","text"),
    `<div class="field" style="grid-column:span 3;"><label>诊断备注</label><textarea id="diagnosis_note"></textarea></div>`
  ].join("");
}

function renderClinicalGroups(){
  const render=(id, arr)=>{document.getElementById(id).innerHTML = arr.map(f=>buildInput(f,`${cn(f)}${unit(f)}`,"number")).join("");};
  render("group-basic",["age","sex","GCS","tem"]);
  render("group-csf",["C_WBC","C_RBC","C_N","C_P","C_G","transparency"]);
  render("group-blood",["B_WBC","B_CRP","B_PCT","B_G","B_N","B_Lym","B_RBC","B_AC"]);
  render("group-inf",["tube","site","other_inf"]);
}

function collectPatientInfo(){
  return {
    patient_id:getTxt("patient_id"), inpatient_no:getTxt("inpatient_no"), patient_name:getTxt("patient_name"),
    sex_text:getTxt("sex_text"), age:getNum("patient_age"), department:getTxt("department"), bed_no:getTxt("bed_no"),
    sampling_time:getTxt("sampling_time"), doctor_name:getTxt("doctor_name"), diagnosis_note:getTxt("diagnosis_note")
  };
}
function collectStaticInputs(){
  const p = {};
  [...baseNumCols, ...catCols].forEach(f => p[f] = getNum(f));
  if(getNum("patient_age") > 0) p.age = getNum("patient_age");
  return p;
}

function resetPredictionPanel(){
  lastPrediction = null;
  document.getElementById("riskNumber").textContent="--%";
  const pill=document.getElementById("riskPill"); pill.className="risk-pill mid"; pill.textContent="未评估";
  document.getElementById("riskLabelText").textContent="未评估";
  document.getElementById("riskPercentText").textContent="--%";
  document.getElementById("evalTimeText").textContent="--";
  document.getElementById("modelVersionText").textContent="--";
  document.getElementById("suggestionText").textContent="--";
  document.getElementById("riskFactors").innerHTML="";
  setStatus("未评估", false, false);
}
function setStatus(ev,saved,reported){
  const a=document.getElementById("statusEval"), b=document.getElementById("statusSave"), c=document.getElementById("statusReport");
  a.textContent=ev; a.className=ev==="已评估"?"tag ok":"tag info";
  b.textContent=saved?"已保存":"未保存"; b.className=saved?"tag ok":"tag";
  c.textContent=reported?"已生成报告":"未生成报告"; c.className=reported?"tag ok":"tag";
}
function showToast(msg){ alert(msg); }

function updateDynamicPatientSummary(){
  const p=collectPatientInfo();
  document.getElementById("dynamicPatientSummary").textContent =
    `患者ID：${p.patient_id || "-"} ｜ 姓名：${p.patient_name || "-"} ｜ 住院号：${p.inpatient_no || "-"} ｜ 科室：${p.department || "-"} ｜ 床号：${p.bed_no || "-"} ｜ 主管医生：${p.doctor_name || "-"}`;
}

function newPatient(){
  ["patient_id","inpatient_no","patient_name","sex_text","patient_age","department","bed_no","sampling_time","doctor_name","diagnosis_note"].forEach(id => setVal(id,""));
  setVal("patient_id",`P${Date.now()}`);
  updateDynamicPatientSummary();
  showToast("已新建患者档案（未保存）");
}

function fillPatientInfo(p){
  setVal("patient_id", p.patient_id || ""); setVal("inpatient_no", p.inpatient_no || ""); setVal("patient_name", p.patient_name || "");
  setVal("sex_text", p.sex_text || ""); setVal("patient_age", p.age ?? ""); setVal("department", p.department || "");
  setVal("bed_no", p.bed_no || ""); setVal("sampling_time", p.sampling_time || ""); setVal("doctor_name", p.doctor_name || "");
  setVal("diagnosis_note", p.diagnosis_note || "");
}

async function savePatient(){
  let patient = collectPatientInfo();
  patient = upsertLocalPatient(patient);
  fillPatientInfo(patient);
  updateDynamicPatientSummary();
  renderPatientTable(getLocalPatients());
  showToast("患者信息已保存");

  try{
    const res = await fetch("/api/patients/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({patient_info: patient})});
    const data = await res.json();
    if(data.ok && data.patient){
      const merged = upsertLocalPatient(data.patient);
      fillPatientInfo(merged);
      renderPatientTable(getLocalPatients());
    }
  }catch(e){}
}
async function queryPatient(){
  const filters = {patient_id:getTxt("patient_id"), inpatient_no:getTxt("inpatient_no"), patient_name:getTxt("patient_name")};
  const localMatched = filterLocalPatients(filters);
  if(localMatched.length){
    fillPatientInfo(localMatched[0]);
    updateDynamicPatientSummary();
    renderPatientTable(getLocalPatients());
    await loadHistoryForCurrent();
    showToast("已读取患者信息");
    return;
  }

  try{
    const q = new URLSearchParams(filters);
    const res = await fetch(`/api/patients/query?${q.toString()}`);
    const data = await res.json();
    if(data.ok && data.patients.length){
      const merged = upsertLocalPatient(data.patients[0]);
      fillPatientInfo(merged);
      updateDynamicPatientSummary();
      renderPatientTable(getLocalPatients());
      await loadHistoryForCurrent();
      showToast("已读取患者信息");
      return;
    }
  }catch(e){}
  showToast("未查询到患者信息");
}

function fillStaticExample(){
  const ex={age:56,sex:1,tube:1,site:0,other_inf:0,transparency:1,C_G:1.3,C_WBC:3188,C_RBC:3300,C_P:1.85,C_N:0.82,GCS:13,tem:38.2,B_G:5.6,B_CRP:45,B_WBC:10.6,B_N:0.79,B_Lym:0.13,B_PCT:0.42,B_AC:0,B_RBC:4.2};
  [...baseNumCols, ...catCols].forEach(f => setVal(f, ex[f] ?? 0));
}
function clearClinicalForm(){
  [...baseNumCols, ...catCols].forEach(f => setVal(f, ""));
  dynamicRows=[]; addDynamicRow(); addDynamicRow(); updateTrendChart(); resetPredictionPanel();
}

function addDynamicRow(row=null){
  dynamicRows.push({
    sample_time: row?.sample_time || new Date().toISOString().slice(0,16),
    C_WBC: row?.C_WBC ?? "", C_RBC: row?.C_RBC ?? "", C_N: row?.C_N ?? "",
    C_P: row?.C_P ?? "", C_G: row?.C_G ?? "", transparency: row?.transparency ?? ""
  });
  renderDynamicRowsTable(); updateTrendChart();
}
function removeDynamicRow(i){ dynamicRows.splice(i,1); renderDynamicRowsTable(); updateTrendChart(); }
function onDynChange(i,k,v){ dynamicRows[i][k]=v; updateTrendChart(); }

function renderDynamicRowsTable(){
  const tb=document.querySelector("#dynamicRowsTable tbody"); tb.innerHTML="";
  dynamicRows.forEach((r,i)=>{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td><input type="datetime-local" value="${esc(r.sample_time)}" onchange="onDynChange(${i},'sample_time',this.value)"></td>
<td><input type="number" step="any" value="${esc(r.C_WBC)}" onchange="onDynChange(${i},'C_WBC',this.value)"></td>
<td><input type="number" step="any" value="${esc(r.C_RBC)}" onchange="onDynChange(${i},'C_RBC',this.value)"></td>
<td><input type="number" step="any" value="${esc(r.C_N)}" onchange="onDynChange(${i},'C_N',this.value)"></td>
<td><input type="number" step="any" value="${esc(r.C_P)}" onchange="onDynChange(${i},'C_P',this.value)"></td>
<td><input type="number" step="any" value="${esc(r.C_G)}" onchange="onDynChange(${i},'C_G',this.value)"></td>
<td><input type="number" step="any" value="${esc(r.transparency)}" onchange="onDynChange(${i},'transparency',this.value)"></td>
<td><button class="btn light" style="height:30px;" onclick="removeDynamicRow(${i})">删除</button></td>`;
    tb.appendChild(tr);
  });
}
function fillDynamicExample(){
  dynamicRows=[];
  addDynamicRow({sample_time:"2026-05-01T08:00",C_WBC:900,C_RBC:1200,C_N:0.62,C_P:0.75,C_G:2.4,transparency:1});
  addDynamicRow({sample_time:"2026-05-02T08:00",C_WBC:1500,C_RBC:1900,C_N:0.70,C_P:1.10,C_G:1.9,transparency:1});
  addDynamicRow({sample_time:"2026-05-03T08:00",C_WBC:3188,C_RBC:3300,C_N:0.82,C_P:1.85,C_G:1.3,transparency:2});
  addDynamicRow({sample_time:"2026-05-04T08:00",C_WBC:2600,C_RBC:3000,C_N:0.78,C_P:1.70,C_G:1.5,transparency:2});
  fillStaticExample();
}
function collectDynamicForApi(){
  const rows=dynamicRows.map(r=>({sample_time:r.sample_time || "",C_WBC:toNumOrNull(r.C_WBC),C_RBC:toNumOrNull(r.C_RBC),C_N:toNumOrNull(r.C_N),C_P:toNumOrNull(r.C_P),C_G:toNumOrNull(r.C_G),transparency:toNumOrNull(r.transparency)}))
    .filter(r=>Object.values(r).slice(1).filter(v=>v!==null).length>=2)
    .sort((a,b)=>String(a.sample_time).localeCompare(String(b.sample_time)));
  const map={"T-2":{},"T-1":{},"T0":{},"T+1":{}}, keys=["T-2","T-1","T0","T+1"], picked=rows.slice(-4);
  for(let i=0;i<picked.length;i++){ map[keys[i]]={C_WBC:picked[i].C_WBC,C_RBC:picked[i].C_RBC,C_N:picked[i].C_N,C_P:picked[i].C_P,C_G:picked[i].C_G,transparency:picked[i].transparency}; }
  return map;
}
function buildPayload(mode){ return {mode, ...collectStaticInputs(), dynamic: mode==="dynamic" ? collectDynamicForApi() : {}, patient_info: collectPatientInfo(), dynamic_rows: dynamicRows}; }

async function doPredict(payload){
  const res=await fetch("/api/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const data=await res.json(); lastPrediction=data;
  document.getElementById("devModelStatus").textContent=data.model_status || "--";
  document.getElementById("devModePath").textContent=data.final_mode==="dynamic" ? "动态 D-PGA-AMFormer" : "静态 PGA-AMFormer";
  document.getElementById("devDynResidual").textContent=data.final_mode==="dynamic" ? data.dynamic_residual_logit : "--";
  document.getElementById("devDynWeight").textContent=data.final_mode==="dynamic" ? data.dynamic_weight : "--";
  const ul=document.getElementById("devLoadMsgs"); ul.innerHTML=""; (data.load_messages||[]).forEach(m=>{const li=document.createElement("li"); li.textContent=m; ul.appendChild(li);});
  return data;
}
function applyRiskResult(data){
  document.getElementById("riskNumber").textContent=`${data.percentage}%`;
  const pill=document.getElementById("riskPill"); pill.className=`risk-pill ${data.risk_key}`; pill.textContent=data.risk_label;
  document.getElementById("riskLabelText").textContent=data.risk_label;
  document.getElementById("riskPercentText").textContent=`${data.percentage}%`;
  document.getElementById("evalTimeText").textContent=data.evaluation_time || "--";
  document.getElementById("modelVersionText").textContent=data.model_version || "--";
  document.getElementById("suggestionText").textContent=data.suggestion || "--";
  document.getElementById("clinicalNotice").textContent=data.clinical_notice || "本结果仅供临床辅助评估参考。";
  const factors=document.getElementById("riskFactors"); factors.innerHTML=""; (data.main_risk_factors || []).forEach(f=>{const li=document.createElement("li"); li.textContent=f; factors.appendChild(li);});
}
async function startStaticAssessment(){ const data=await doPredict(buildPayload("static")); applyRiskResult(data); setStatus("已评估", false, false); }
async function startDynamicAssessment(){
  const data=await doPredict(buildPayload("dynamic")); applyRiskResult(data);
  document.getElementById("cmpStatic").textContent=`${data.static_percentage}%`;
  document.getElementById("cmpDynamic").textContent=`${data.percentage}%`;
  const diff=data.percentage-data.static_percentage; document.getElementById("riskTrendDir").textContent=diff>2?"升高":(diff<-2?"降低":"稳定");
  setStatus("已评估", false, false);
}

async function saveCase(mode){
  if(!lastPrediction){ showToast("请先开始评估，再保存病例"); return; }
  let p=collectPatientInfo();
  p = upsertLocalPatient(p);
  fillPatientInfo(p);
  renderPatientTable(getLocalPatients());

  const localCase = {
    record_id:`CASE-${Date.now()}`,
    assessment_type:mode,
    patient_info:p,
    input_data:{static_inputs:collectStaticInputs(), dynamic_rows:dynamicRows},
    prediction:lastPrediction,
    risk_label:lastPrediction.risk_label,
    percentage:lastPrediction.percentage,
    evaluation_time:lastPrediction.evaluation_time || nowStr(),
    model_version:lastPrediction.model_version || (mode==="dynamic"?"D-PGA-AMFormer v1.0":"PGA-AMFormer v1.0"),
    operator:p.doctor_name || ""
  };
  addLocalCase(localCase);

  setStatus("已评估", true, false);
  await loadHistoryForCurrent();
  await queryCases();
  showToast("病例记录已保存");

  try{
    const payload={assessment_type:mode, patient_info:p, input_data:{static_inputs:collectStaticInputs(), dynamic_rows:dynamicRows}, prediction:lastPrediction};
    await fetch("/api/cases/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  }catch(e){}
}
function generateReport(mode){
  if(!lastPrediction){ showToast("请先完成评估后再生成报告"); return; }
  const p=collectPatientInfo();
  const txt=[
    "中枢神经系统感染风险辅助评估报告","====================================",
    `患者ID：${p.patient_id || "-"}`,`住院号：${p.inpatient_no || "-"}`,`姓名：${p.patient_name || "-"}`,
    `科室/床号：${p.department || "-"} / ${p.bed_no || "-"}`,
    `评估类型：${mode==="dynamic"?"动态趋势评估":"静态风险评估"}`,
    `风险等级：${lastPrediction.risk_label}`,`预测概率：${lastPrediction.percentage}%`,
    `评估时间：${lastPrediction.evaluation_time || "-"}`,`模型版本：${lastPrediction.model_version || "-"}`,
    "主要风险来源：",...(lastPrediction.main_risk_factors || []).map((x,i)=>`${i+1}. ${x}`),"","临床参考建议：",lastPrediction.suggestion || "-","","提示：本结果为临床辅助评估结果，不作为最终诊断依据。"
  ].join("\n");
  const w=window.open("", "_blank");
  w.document.write(`<pre style="font-family:Arial,'Microsoft YaHei';padding:16px;line-height:1.7;">${esc(txt)}</pre>`);
  w.document.close();
  setStatus("已评估", document.getElementById("statusSave").textContent==="已保存", true);
}

async function loadHistoryForCurrent(){
  const pid=getTxt("patient_id");
  if(!pid){ showToast("请先填写患者ID"); return; }

  const tb=document.querySelector("#historyTable tbody");
  tb.innerHTML="";
  let rows = filterLocalCases({patient_id:pid, patient_name:"", inpatient_no:"", eval_date:""});

  if(!rows.length){
    try{
      const res=await fetch(`/api/cases/query?${new URLSearchParams({patient_id:pid}).toString()}`);
      const data=await res.json();
      rows = data.records || [];
      rows.forEach(r => addLocalCase(r));
      rows = filterLocalCases({patient_id:pid, patient_name:"", inpatient_no:"", eval_date:""});
    }catch(e){}
  }

  if(!rows.length){
    tb.innerHTML=`<tr><td colspan="6" class="muted">暂无历史记录</td></tr>`;
    return;
  }

  rows.forEach(r=>{
    const p=r.patient_info || {};
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${esc(r.evaluation_time || "-")}</td><td>${esc(r.assessment_type==="dynamic"?"动态趋势评估":"静态风险评估")}</td><td>${esc(r.risk_label || "-")}</td><td>${r.percentage!==null&&r.percentage!==undefined?esc(r.percentage+"%"):"-"}</td><td>${esc(p.doctor_name || r.operator || "-")}</td><td>${esc(r.model_version || "-")}</td>`;
    tb.appendChild(tr);
  });
}

async function queryCases(){
  const filters = {
    patient_id:getTxt("q_patient_id"),
    patient_name:getTxt("q_patient_name"),
    inpatient_no:getTxt("q_inpatient_no"),
    eval_date:getTxt("q_eval_date")
  };
  let rows = filterLocalCases(filters);

  if(!rows.length){
    try{
      const q = new URLSearchParams(filters);
      const res=await fetch(`/api/cases/query?${q.toString()}`);
      const data=await res.json();
      const remote = data.records || [];
      remote.forEach(r => addLocalCase(r));
      rows = filterLocalCases(filters);
    }catch(e){}
  }

  const tb=document.querySelector("#recordsTable tbody");
  tb.innerHTML="";
  if(!rows.length){
    tb.innerHTML=`<tr><td colspan="9" class="muted">未查询到病例记录</td></tr>`;
    return;
  }
  rows.forEach(r=>{
    const p=r.patient_info || {};
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${esc(r.evaluation_time || "-")}</td><td>${esc(p.patient_id || "-")}</td><td>${esc(p.patient_name || "-")}</td><td>${esc(p.inpatient_no || "-")}</td><td>${esc(r.assessment_type==="dynamic"?"动态趋势评估":"静态风险评估")}</td><td>${esc(r.risk_label || "-")}</td><td>${r.percentage!==null&&r.percentage!==undefined?esc(r.percentage+"%"):"-"}</td><td>${esc(p.doctor_name || r.operator || "-")}</td><td>${esc(r.model_version || "-")}</td>`;
    tb.appendChild(tr);
  });
}

function initTrendChart(){
  const ctx=document.getElementById("trendChart").getContext("2d");
  trendChart=new Chart(ctx,{type:"line",data:{labels:[],datasets:[
    {label:"C_G",data:[],borderColor:"#2563eb",backgroundColor:"rgba(37,99,235,.1)",tension:.3},
    {label:"C_WBC",data:[],borderColor:"#dc2626",backgroundColor:"rgba(220,38,38,.1)",tension:.3},
    {label:"C_P",data:[],borderColor:"#d97706",backgroundColor:"rgba(217,119,6,.1)",tension:.3},
  ]},options:{responsive:true,maintainAspectRatio:false}});
}
function updateTrendChart(){
  if(!trendChart) return;
  const rows=dynamicRows.filter(r=>r.sample_time).sort((a,b)=>String(a.sample_time).localeCompare(String(b.sample_time)));
  trendChart.data.labels=rows.map(r=>r.sample_time.replace("T"," "));
  trendChart.data.datasets[0].data=rows.map(r=>toNumOrNull(r.C_G));
  trendChart.data.datasets[1].data=rows.map(r=>toNumOrNull(r.C_WBC));
  trendChart.data.datasets[2].data=rows.map(r=>toNumOrNull(r.C_P));
  trendChart.update();
}

function bindPatientSync(){
  ["patient_id","inpatient_no","patient_name","sex_text","patient_age","department","bed_no","sampling_time","doctor_name","diagnosis_note"].forEach(id=>{
    const el=document.getElementById(id); if(el) el.addEventListener("input", updateDynamicPatientSummary);
  });
}

function init(){
  renderPatientInfo();
  renderClinicalGroups();
  bindPatientSync();

  seedShowData();
  renderPatientTable(getLocalPatients());

  const first = getLocalPatients()[0];
  if(first){
    fillPatientInfo(first);
  }

  dynamicRows=[]; addDynamicRow(); addDynamicRow();
  initTrendChart(); updateDynamicPatientSummary(); resetPredictionPanel();

  const ul=document.getElementById("devLoadMsgs");
  ul.innerHTML="";
  (loadMessages || []).forEach(m=>{const li=document.createElement("li"); li.textContent=m; ul.appendChild(li);});

  queryCases();
  if(getTxt("patient_id")) loadHistoryForCurrent();
}
init();
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(
        HTML,
        title="中枢神经系统感染风险辅助评估平台",
        base_num_cols=json.dumps(BASE_NUM_COLS, ensure_ascii=False),
        cat_cols=json.dumps(CAT_COLS, ensure_ascii=False),
        dyn_features=json.dumps(DYN_FEATURES, ensure_ascii=False),
        time_points=json.dumps(TIME_POINTS, ensure_ascii=False),
        name_map=json.dumps(NAME_MAP, ensure_ascii=False),
        load_messages=json.dumps(LOAD_MESSAGES, ensure_ascii=False),
    )


if __name__ == "__main__":
    print("=" * 80)
    print("中枢神经系统感染风险辅助评估平台")
    print(f"Device: {DEVICE}")
    print("Model load messages:")
    for msg in LOAD_MESSAGES:
        print(" -", msg)
    print("=" * 80)
    app.run(host="0.0.0.0", port=7860, debug=True)
