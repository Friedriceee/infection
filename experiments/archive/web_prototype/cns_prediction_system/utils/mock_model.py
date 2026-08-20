from utils.risk_level import get_risk_level


def _num(data, key, default=0.0):
    value = data.get(key)
    try:
        return float(value) if value is not None and value != "" else default
    except (TypeError, ValueError):
        return default


def _clip(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def predict_static(case_data: dict) -> dict:
    c_wbc = _num(case_data, "C_WBC")
    c_p = _num(case_data, "C_P")
    c_g = _num(case_data, "C_G")
    c_n = _num(case_data, "C_N")
    crp = _num(case_data, "B_CRP")
    pct = _num(case_data, "B_PCT")
    temp = _num(case_data, "temperature")
    gcs = _num(case_data, "gcs", 15)

    score = 0.12
    factors = []

    if c_wbc >= 100:
        score += 0.34
        factors.append("脑脊液白细胞显著升高")
    elif c_wbc >= 10:
        score += 0.20
        factors.append("脑脊液白细胞升高")

    if c_p >= 1.0:
        score += 0.22
        factors.append("脑脊液蛋白显著升高")
    elif c_p >= 0.45:
        score += 0.14
        factors.append("脑脊液蛋白升高")

    if c_g > 0 and c_g < 2.2:
        score += 0.17
        factors.append("脑脊液葡萄糖降低")

    if c_n >= 70:
        score += 0.08
        factors.append("脑脊液中性粒细胞比例升高")
    if crp >= 10:
        score += 0.08
        factors.append("CRP 升高")
    if pct >= 0.5:
        score += 0.08
        factors.append("PCT 升高")
    if temp >= 38:
        score += 0.06
        factors.append("体温升高")
    if gcs < 13:
        score += 0.05
        factors.append("GCS 评分下降")
    if case_data.get("transparency") and case_data.get("transparency") != "清亮":
        score += 0.04
        factors.append("脑脊液透明度异常")

    score = round(_clip(score, 0.02, 0.98), 2)
    level = get_risk_level(score)

    if not factors:
        factors = ["当前录入指标未见明显高危特征"]

    tip_map = {
        "低风险": "当前感染风险较低，建议结合临床表现继续常规观察和复查。",
        "中风险": "当前感染风险中等，建议结合症状、影像学和病原学检查进行综合评估。",
        "高风险": "当前感染风险较高，建议结合患者症状、脑脊液检查、血液炎症指标及病原学结果进行重点评估。",
    }

    return {
        "risk_score": score,
        "risk_level": level,
        "model_name": "PGFormer",
        "model_version": "v1.0",
        "key_factors": factors,
        "clinical_tip": tip_map[level],
    }


def predict_dynamic(case_list: list, latest_static_score: float) -> dict:
    ordered = sorted(case_list, key=lambda item: item.get("visit_time") or "")
    first = ordered[0] if ordered else {}
    latest = ordered[-1] if ordered else {}

    delta = 0.0
    factors = []

    def trend(key):
        return _num(latest, key) - _num(first, key)

    if trend("C_WBC") > 20:
        delta += 0.07
        factors.append("脑脊液白细胞持续升高")
    elif trend("C_WBC") < -20:
        delta -= 0.04
        factors.append("脑脊液白细胞下降")

    if trend("C_P") > 0.2:
        delta += 0.05
        factors.append("脑脊液蛋白升高")
    elif trend("C_P") < -0.2:
        delta -= 0.03
        factors.append("脑脊液蛋白下降")

    if trend("C_G") < -0.3:
        delta += 0.06
        factors.append("脑脊液葡萄糖下降")
    elif trend("C_G") > 0.3:
        delta -= 0.03
        factors.append("脑脊液葡萄糖回升")

    if trend("C_N") > 10:
        delta += 0.04
        factors.append("脑脊液中性粒细胞比例上升")
    elif trend("C_N") < -10:
        delta -= 0.03
        factors.append("脑脊液中性粒细胞比例下降")

    delta = round(_clip(delta, -0.15, 0.18), 2)
    risk_score = round(_clip(float(latest_static_score or 0) + delta, 0.02, 0.98), 2)
    level = get_risk_level(risk_score)

    if delta > 0.03:
        trend_summary = "患者近期脑脊液指标变化提示感染风险上升。"
    elif delta < -0.03:
        trend_summary = "患者近期脑脊液指标变化提示感染风险下降。"
    else:
        trend_summary = "患者近期脑脊液指标变化相对平稳。"

    if not factors:
        factors = ["多时间点指标变化暂未提示明显风险波动"]

    return {
        "static_score": round(float(latest_static_score or 0), 2),
        "dynamic_delta": delta,
        "risk_score": risk_score,
        "risk_level": level,
        "model_name": "D-PGFormer",
        "model_version": "v1.0",
        "trend_summary": trend_summary,
        "key_factors": factors,
        "clinical_tip": f"{trend_summary}建议结合临床表现和病原学检查结果进行重点评估。",
    }
