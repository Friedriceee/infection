def get_risk_level(score: float) -> str:
    if score < 0.3:
        return "低风险"
    if score < 0.7:
        return "中风险"
    return "高风险"


def risk_badge_class(level: str) -> str:
    return {
        "低风险": "risk-low",
        "中风险": "risk-medium",
        "高风险": "risk-high",
    }.get(level or "", "risk-empty")

