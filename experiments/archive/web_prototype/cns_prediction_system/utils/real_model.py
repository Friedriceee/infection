import importlib.util
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from utils.risk_level import get_risk_level

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - handled through model load status
    torch = None
    nn = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "web_model"

PGA_PY_PATH = MODEL_DIR / "pga2.py"
STATIC_MODEL_PATH = MODEL_DIR / "best_fold1.pth"
DYNAMIC_FULL_PATH = MODEL_DIR / "dynamic_full_model_fold1.pth"
DYNAMIC_HEAD_PATH = MODEL_DIR / "dynamic_head_fold1.pth"
DYN_SCALER_PATH = MODEL_DIR / "dyn_scaler_fold1.joblib"
STATIC_SCALER_PATH = MODEL_DIR / "static_scaler_fold1.joblib"
CAT_ENCODER_PATH = MODEL_DIR / "cat_encoder_fold1.joblib"

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

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
TIME_POINTS = ["T-2", "T-1", "T0", "T+1"]
DYN_FEATURES = ["C_WBC", "C_RBC", "C_N", "C_P", "C_G", "transparency"]
DEFAULT_CAT_CARDINALITIES = [3, 5, 3, 3, 4]

NAME_MAP = {
    "C_G": "脑脊液葡萄糖降低",
    "C_WBC": "脑脊液白细胞升高",
    "C_P": "脑脊液蛋白升高",
    "B_CRP": "CRP 升高",
    "GCS": "GCS 评分较低",
}

PGA_MOD = None
STATIC_MODEL = None
DYNAMIC_MODEL = None
DYNAMIC_HEAD = None
STATIC_SCALER = None
CAT_ENCODER = None
DYN_SCALER = None
LOAD_MESSAGES: List[str] = []


def safe_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, str):
            value = value.replace(",", ".")
        if isinstance(value, float) and math.isnan(value):
            return default
        return float(value)
    except Exception:
        return default


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def load_module_from_path(path: Path, name="pga_module"):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_static_config_from_state_dict(state_dict: Dict) -> Dict:
    pos_shape = tuple(state_dict["pos_encoding"].shape)
    embed_dim = pos_shape[-1]
    layer_ids = []
    cat_cards = []
    for key, value in state_dict.items():
        if key.startswith("layers."):
            try:
                layer_ids.append(int(key.split(".")[1]))
            except Exception:
                pass
        if key.startswith("cat_embedding.embeddings.") and key.endswith(".weight"):
            cat_cards.append((int(key.split(".")[2]), int(value.shape[0])))
    return {
        "embed_dim": embed_dim,
        "n_layers": max(layer_ids) + 1 if layer_ids else 2,
        "cat_cardinalities": [card for _, card in sorted(cat_cards)] or DEFAULT_CAT_CARDINALITIES,
    }


def _map_binary(value, positive="是"):
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip()
    if text in {"1", "1.0", positive, "男", "枕叶", "yes", "YES", "Y"}:
        return 1
    if text in {"0", "0.0", "否", "女", "非枕叶", "no", "NO", "N"}:
        return 0
    return safe_float(text, 0.0)


def _map_sex(value):
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip()
    if text in {"0", "0.0", "男", "M", "m"}:
        return 0
    if text in {"1", "1.0", "女", "F", "f"}:
        return 1
    if text in {"2", "2.0", "未知", "unknown", "UNKNOWN"}:
        return 2
    return 2


def _map_transparency(value):
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "").strip()
    return {
        "清亮": 1,
        "微浑": 2,
        "浑浊": 3,
        "血性": 3,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 3,
    }.get(text, 0)


def normalize_case(case_data: Dict) -> Dict:
    return {
        "age": safe_float(case_data.get("age")),
        "sex": _map_sex(case_data.get("sex")),
        "tube": _map_binary(case_data.get("tube")),
        "site": _map_binary(case_data.get("site")),
        "other_inf": _map_binary(case_data.get("other_inf")),
        "transparency": _map_transparency(case_data.get("transparency")),
        "C_G": safe_float(case_data.get("C_G")),
        "C_WBC": safe_float(case_data.get("C_WBC")),
        "C_RBC": safe_float(case_data.get("C_RBC")),
        "C_P": safe_float(case_data.get("C_P")),
        "C_N": safe_float(case_data.get("C_N")),
        "GCS": safe_float(case_data.get("gcs") or case_data.get("GCS"), 15.0),
        "tem": safe_float(case_data.get("temperature") or case_data.get("tem"), 37.0),
        "B_G": safe_float(case_data.get("B_G")),
        "B_CRP": safe_float(case_data.get("B_CRP")),
        "B_WBC": safe_float(case_data.get("B_WBC")),
        "B_N": safe_float(case_data.get("B_N")),
        "B_Lym": safe_float(case_data.get("B_Lym")),
        "B_PCT": safe_float(case_data.get("B_PCT")),
        "B_AC": safe_float(case_data.get("B_AC")),
        "B_RBC": safe_float(case_data.get("B_RBC")),
    }


def build_raw_feature_dict(data: Dict) -> Dict[str, float]:
    raw = normalize_case(data)
    eps = 1e-6
    raw["ratio_C_G_B_G"] = raw.get("C_G", 0.0) / (raw.get("B_G", 0.0) + eps)
    raw["diff_C_N_B_N"] = raw.get("C_N", 0.0) - raw.get("B_N", 0.0)
    raw["corrected_WBC"] = raw.get("C_WBC", 0.0) - raw.get("B_WBC", 0.0) * raw.get("C_RBC", 0.0) / (raw.get("B_RBC", 0.0) + eps)
    raw["ratio_WBC_RBC_diff"] = raw.get("B_WBC", 0.0) / (raw.get("B_RBC", 0.0) + eps) - raw.get("C_WBC", 0.0) / (raw.get("C_RBC", 0.0) + eps)
    return raw


def build_static_arrays(data: Dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    raw = build_raw_feature_dict(data)
    x_num = np.array([[raw[col] for col in NUM_COLS]], dtype=np.float32)
    x_cat = np.array([[int(raw[col]) for col in CAT_COLS]], dtype=np.int64)
    if STATIC_SCALER is not None:
        x_num = STATIC_SCALER.transform(x_num).astype(np.float32)
    return x_num, x_cat, raw


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
                for param in self.static_model.parameters():
                    param.requires_grad = False

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


def load_all_models():
    global PGA_MOD, STATIC_MODEL, DYNAMIC_MODEL, DYNAMIC_HEAD, STATIC_SCALER, CAT_ENCODER, DYN_SCALER
    LOAD_MESSAGES.clear()

    if torch is None:
        LOAD_MESSAGES.append("torch 未安装，真实模型不可用")
        return
    if joblib is None:
        LOAD_MESSAGES.append("joblib 未安装，预处理器不可用")
        return

    if not PGA_PY_PATH.exists():
        LOAD_MESSAGES.append("未找到 web_model/pga2.py")
        return

    try:
        PGA_MOD = load_module_from_path(PGA_PY_PATH)
        LOAD_MESSAGES.append("已加载 pga2.py")
    except Exception as error:
        LOAD_MESSAGES.append(f"pga2.py 加载失败：{error}")
        return

    for path, label, target in [
        (STATIC_SCALER_PATH, "static_scaler_fold1.joblib", "static"),
        (DYN_SCALER_PATH, "dyn_scaler_fold1.joblib", "dynamic"),
        (CAT_ENCODER_PATH, "cat_encoder_fold1.joblib", "cat"),
    ]:
        if not path.exists():
            LOAD_MESSAGES.append(f"未找到 {label}")
            continue
        try:
            loaded = joblib.load(path)
            if target == "static":
                STATIC_SCALER = loaded
            elif target == "dynamic":
                DYN_SCALER = loaded
            else:
                CAT_ENCODER = loaded
            LOAD_MESSAGES.append(f"已加载 {label}")
        except Exception as error:
            LOAD_MESSAGES.append(f"{label} 加载失败：{error}")

    if STATIC_MODEL_PATH.exists():
        try:
            state_dict = torch.load(STATIC_MODEL_PATH, map_location=DEVICE)
            cfg = infer_static_config_from_state_dict(state_dict)
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
            STATIC_MODEL.load_state_dict(state_dict, strict=True)
            STATIC_MODEL.eval()
            LOAD_MESSAGES.append(f"已加载静态模型 best_fold1.pth：embed_dim={cfg['embed_dim']}, layers={cfg['n_layers']}")
        except Exception as error:
            STATIC_MODEL = None
            LOAD_MESSAGES.append(f"静态模型加载失败：{error}")

    if DYNAMIC_FULL_PATH.exists():
        try:
            checkpoint = torch.load(DYNAMIC_FULL_PATH, map_location=DEVICE)
            static_state_dict = checkpoint["static_model_state_dict"]
            cfg = infer_static_config_from_state_dict(static_state_dict)
            pga_cfg = checkpoint.get("pga_cfg", {})
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
                dyn_dim=int(checkpoint.get("dyn_dim", 60)),
                hidden=int(checkpoint.get("hidden", 64)),
                dropout=float(checkpoint.get("dropout", 0.2)),
                freeze_static=bool(checkpoint.get("freeze_static", True)),
            ).to(DEVICE)
            DYNAMIC_MODEL.load_state_dict(checkpoint["model_state_dict"], strict=True)
            DYNAMIC_MODEL.eval()
            LOAD_MESSAGES.append(f"已加载动态完整模型 dynamic_full_model_fold1.pth：dyn_dim={checkpoint.get('dyn_dim', 60)}")
        except Exception as error:
            DYNAMIC_MODEL = None
            LOAD_MESSAGES.append(f"动态完整模型加载失败：{error}")

    if DYNAMIC_MODEL is None and DYNAMIC_HEAD_PATH.exists() and DynamicResidualHead is not None:
        try:
            checkpoint = torch.load(DYNAMIC_HEAD_PATH, map_location=DEVICE)
            DYNAMIC_HEAD = DynamicResidualHead(
                dyn_dim=int(checkpoint.get("dyn_dim", 60)),
                hidden=int(checkpoint.get("hidden", 64)),
                dropout=float(checkpoint.get("dropout", 0.2)),
            ).to(DEVICE)
            DYNAMIC_HEAD.load_state_dict(checkpoint["dynamic_head_state_dict"], strict=True)
            DYNAMIC_HEAD.eval()
            LOAD_MESSAGES.append("已加载 dynamic_head_fold1.pth")
        except Exception as error:
            DYNAMIC_HEAD = None
            LOAD_MESSAGES.append(f"动态 head 加载失败：{error}")


def ensure_models_loaded():
    if not LOAD_MESSAGES:
        load_all_models()


def real_model_available(kind="static"):
    ensure_models_loaded()
    if kind == "dynamic":
        return DYNAMIC_MODEL is not None or DYNAMIC_HEAD is not None
    return STATIC_MODEL is not None and STATIC_SCALER is not None


def extract_dynamic_features_from_cases(case_list: List[Dict], eps=1e-6) -> np.ndarray:
    ordered = sorted(case_list, key=lambda item: str(item.get("visit_time") or ""))
    picked = ordered[-4:]
    while len(picked) < 4:
        picked.insert(0, {})

    seq = []
    for row_data in picked:
        row = []
        normalized = normalize_case(row_data)
        for feature in DYN_FEATURES:
            value = normalized.get(feature)
            row.append(np.nan if value is None or value == "" else safe_float(value, np.nan))
        seq.append(row)

    x = np.array(seq, dtype=np.float32).reshape(1, len(TIME_POINTS), len(DYN_FEATURES))
    mask = ~np.isnan(x)
    x_nan = x.copy()
    x_nan[~mask] = np.nan
    _, t_count, f_count = x.shape

    valid_count = np.sum(mask, axis=1).astype(np.float32)
    mean = np.nanmean(x_nan, axis=1)
    maxv = np.nanmax(x_nan, axis=1)
    minv = np.nanmin(x_nan, axis=1)
    first = np.zeros((1, f_count), dtype=np.float32)
    last = np.zeros((1, f_count), dtype=np.float32)
    peak_pos = np.zeros((1, f_count), dtype=np.float32)
    slope = np.zeros((1, f_count), dtype=np.float32)

    for feature_idx in range(f_count):
        idx = np.where(mask[0, :, feature_idx])[0]
        if len(idx) == 0:
            continue
        values = x[0, idx, feature_idx]
        first[0, feature_idx] = values[0]
        last[0, feature_idx] = values[-1]
        peak_pos[0, feature_idx] = idx[int(np.nanargmax(values))] / max(t_count - 1, 1)
        if len(idx) >= 2:
            slope[0, feature_idx] = (last[0, feature_idx] - first[0, feature_idx]) / (idx[-1] - idx[0] + eps)

    delta = last - first
    rel_change = delta / (np.abs(first) + eps)
    arrays = [mean, maxv, minv, first, last, delta, rel_change, slope, valid_count, peak_pos]
    arrays = [np.nan_to_num(item, nan=0.0, posinf=0.0, neginf=0.0) for item in arrays]
    dyn_feat = np.concatenate(arrays, axis=1).astype(np.float32)
    if DYN_SCALER is not None:
        dyn_feat = DYN_SCALER.transform(dyn_feat).astype(np.float32)
    return dyn_feat


def derive_main_risk_factors(importance: Dict[str, float]):
    ordered = sorted((importance or {}).items(), key=lambda item: item[1], reverse=True)
    return [NAME_MAP.get(key, key) for key, _ in ordered[:4]] or ["需结合更多检查指标"]


def clinical_tip(level: str) -> str:
    if level == "高风险":
        return "模型给出感染风险辅助评估：风险较高。建议结合脑脊液培养、影像学、临床症状及医生判断进一步评估。"
    if level == "中风险":
        return "模型给出感染风险辅助评估：存在一定风险。建议结合复查结果进行动态随访。"
    return "模型给出感染风险辅助评估：当前风险相对较低。若症状进展或指标异常，建议继续复评。"


def predict_static(case_data: Dict) -> Dict:
    ensure_models_loaded()
    if STATIC_MODEL is None or STATIC_SCALER is None or torch is None:
        raise RuntimeError("真实静态模型不可用：" + "；".join(LOAD_MESSAGES))

    x_num, x_cat, _raw = build_static_arrays(case_data)
    with torch.no_grad():
        tx_num = torch.tensor(x_num, dtype=torch.float32).to(DEVICE)
        tx_cat = torch.tensor(x_cat, dtype=torch.long).to(DEVICE)
        logits = STATIC_MODEL(tx_num, tx_cat)
        logit = float(logits.detach().cpu().numpy().reshape(-1)[0])
        score = round(sigmoid(logit), 4)

    importance = {"C_G": 0.82, "C_WBC": 0.76, "C_P": 0.58, "B_CRP": 0.43, "GCS": 0.32}
    level = get_risk_level(score)
    return {
        "risk_score": score,
        "risk_level": level,
        "model_name": "PGFormer",
        "model_version": "v1.0",
        "model_status": "static_real",
        "key_factors": derive_main_risk_factors(importance),
        "clinical_tip": clinical_tip(level),
        "load_messages": LOAD_MESSAGES,
    }


def predict_dynamic(case_list: List[Dict], latest_static_score: float) -> Dict:
    ensure_models_loaded()
    if torch is None:
        raise RuntimeError("真实动态模型不可用：" + "；".join(LOAD_MESSAGES))
    if not case_list:
        raise RuntimeError("动态预测缺少病例序列")

    latest_case = sorted(case_list, key=lambda item: str(item.get("visit_time") or ""))[-1]
    x_num, x_cat, _raw = build_static_arrays(latest_case)
    dyn_feat = extract_dynamic_features_from_cases(case_list)

    with torch.no_grad():
        tx_num = torch.tensor(x_num, dtype=torch.float32).to(DEVICE)
        tx_cat = torch.tensor(x_cat, dtype=torch.long).to(DEVICE)
        tdyn = torch.tensor(dyn_feat, dtype=torch.float32).to(DEVICE)
        if DYNAMIC_MODEL is not None and DYN_SCALER is not None and STATIC_SCALER is not None:
            parts = DYNAMIC_MODEL(tx_num, tx_cat, tdyn, return_parts=True)
            final_logit = float(parts["logit"].detach().cpu().numpy().reshape(-1)[0])
            static_logit = float(parts["z_static"].detach().cpu().numpy().reshape(-1)[0])
            dynamic_delta = float(parts["z_dyn_residual"].detach().cpu().numpy().reshape(-1)[0])
            dynamic_weight = float(parts["dyn_weight"].detach().cpu().numpy().reshape(-1)[0])
            status = "dynamic_real_full"
        elif DYNAMIC_HEAD is not None and DYN_SCALER is not None:
            static_result = predict_static(latest_case)
            static_score = float(static_result["risk_score"])
            static_logit = float(np.log(static_score / (1 - static_score)))
            dyn_res_tensor, dyn_w_tensor = DYNAMIC_HEAD(tdyn)
            dynamic_delta = float(dyn_res_tensor.detach().cpu().numpy().reshape(-1)[0])
            dynamic_weight = float(dyn_w_tensor.detach().cpu().numpy().reshape(-1)[0])
            final_logit = static_logit + dynamic_delta
            status = "dynamic_head_only"
        else:
            raise RuntimeError("真实动态模型不可用：" + "；".join(LOAD_MESSAGES))

    score = round(sigmoid(final_logit), 4)
    static_score = round(sigmoid(static_logit), 4)
    level = get_risk_level(score)
    importance = {"C_G": 0.82, "C_WBC": 0.76, "C_P": 0.58, "B_CRP": 0.43, "GCS": 0.32}
    if dynamic_delta > 0.03:
        trend_summary = "患者近期脑脊液指标变化提示感染风险上升。"
    elif dynamic_delta < -0.03:
        trend_summary = "患者近期脑脊液指标变化提示感染风险下降。"
    else:
        trend_summary = "患者近期脑脊液指标变化相对平稳。"

    return {
        "static_score": static_score,
        "dynamic_delta": round(dynamic_delta, 4),
        "dynamic_weight": round(dynamic_weight, 4),
        "risk_score": score,
        "risk_level": level,
        "model_name": "D-PGFormer",
        "model_version": "v1.0",
        "model_status": status,
        "trend_summary": trend_summary,
        "key_factors": derive_main_risk_factors(importance),
        "clinical_tip": f"{trend_summary}{clinical_tip(level)}",
        "load_messages": LOAD_MESSAGES,
    }


load_all_models()
