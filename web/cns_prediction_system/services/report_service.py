from datetime import datetime

from models import case_model, prediction_model
from services.prediction_service import parse_result


def build_patient_chart_data(cases, predictions):
    labels = [f"T{index + 1}" for index, _row in enumerate(cases)]
    static_by_case = {}
    dynamic_by_case = {}
    for item in reversed(predictions):
        if item["case_id"]:
            if item["prediction_type"] == "static":
                static_by_case[item["case_id"]] = item["risk_score"]
            if item["prediction_type"] == "dynamic":
                dynamic_by_case[item["case_id"]] = item["risk_score"]

    return {
        "risk": {
            "labels": labels,
            "static": [static_by_case.get(row["id"]) for row in cases],
            "dynamic": [dynamic_by_case.get(row["id"]) for row in cases],
        },
        "indicators": {
            "labels": labels,
            "C_WBC": [row["C_WBC"] for row in cases],
            "C_P": [row["C_P"] for row in cases],
            "C_G": [row["C_G"] for row in cases],
            "C_N": [row["C_N"] for row in cases],
        },
    }


def build_report(patient):
    cases = case_model.list_cases_by_patient(patient["id"])
    predictions = prediction_model.list_predictions_by_patient(patient["id"])
    latest_static = prediction_model.latest_prediction(patient["id"], "static")
    latest_dynamic = prediction_model.latest_prediction(patient["id"], "dynamic")
    latest_prediction = latest_dynamic or latest_static
    latest_result = parse_result(latest_prediction)
    return {
        "cases": cases,
        "predictions": predictions,
        "latest_static": latest_static,
        "latest_dynamic": latest_dynamic,
        "latest_result": latest_result,
        "chart_data": build_patient_chart_data(cases, predictions),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

