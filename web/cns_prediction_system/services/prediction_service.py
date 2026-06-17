import json

from models import case_model, model_version_model, patient_model, prediction_model
from services.case_service import row_to_dict
from utils.real_model import predict_dynamic, predict_static


def _json(data):
    return json.dumps(data, ensure_ascii=False)


def _apply_enabled_model_version(result, model_type):
    enabled = next(
        (
            row
            for row in model_version_model.list_model_versions()
            if row["model_type"] == model_type and row["status"] == "enabled"
        ),
        None,
    )
    if enabled:
        result = dict(result)
        result["model_name"] = enabled["model_name"]
        result["model_version"] = enabled["model_version"]
    return result


def run_static_prediction(case_id, created_by=None):
    case_row = case_model.get_case(case_id)
    case_data = row_to_dict(case_row)
    result = _apply_enabled_model_version(predict_static(case_data), "static")
    prediction_id = prediction_model.create_prediction(
        {
            "patient_id": case_row["patient_id"],
            "case_id": case_id,
            "prediction_type": "static",
            "model_name": result["model_name"],
            "model_version": result["model_version"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "static_score": result["risk_score"],
            "dynamic_score": None,
            "dynamic_delta": None,
            "input_snapshot": _json(case_data),
            "result_json": _json(result),
            "created_by": created_by,
        }
    )
    return prediction_id, result


def run_dynamic_prediction(patient_id, latest_case_id=None, latest_static_score=None, created_by=None):
    case_rows = case_model.list_cases_by_patient(patient_id)
    if len(case_rows) < 2:
        return None, None

    case_list = [row_to_dict(row) for row in case_rows]
    if latest_static_score is None:
        latest_static = prediction_model.latest_prediction(patient_id, "static")
        latest_static_score = latest_static["risk_score"] if latest_static else 0

    result = _apply_enabled_model_version(predict_dynamic(case_list, latest_static_score), "dynamic")
    prediction_id = prediction_model.create_prediction(
        {
            "patient_id": patient_id,
            "case_id": latest_case_id or case_rows[-1]["id"],
            "prediction_type": "dynamic",
            "model_name": result["model_name"],
            "model_version": result["model_version"],
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "static_score": result["static_score"],
            "dynamic_score": result["risk_score"],
            "dynamic_delta": result["dynamic_delta"],
            "input_snapshot": _json(case_list),
            "result_json": _json(result),
            "created_by": created_by,
        }
    )
    return prediction_id, result


def run_auto_prediction_after_case(case_id, created_by=None):
    case_row = case_model.get_case(case_id)
    static_id, static_result = run_static_prediction(case_id, created_by)
    dynamic_id, dynamic_result = run_dynamic_prediction(
        case_row["patient_id"],
        latest_case_id=case_id,
        latest_static_score=static_result["risk_score"],
        created_by=created_by,
    )

    final_result = dynamic_result or static_result
    patient_model.update_patient_current_risk(
        case_row["patient_id"],
        final_result["risk_score"],
        final_result["risk_level"],
    )
    return {
        "static_prediction_id": static_id,
        "dynamic_prediction_id": dynamic_id,
        "final_result": final_result,
    }


def rerun_case_prediction(case_id, created_by=None):
    return run_auto_prediction_after_case(case_id, created_by)


def refresh_patient_current_risk(patient_id):
    latest = prediction_model.latest_prediction(patient_id)
    if latest:
        patient_model.update_patient_current_risk(
            patient_id,
            latest["risk_score"],
            latest["risk_level"],
        )
    else:
        patient_model.update_patient_current_risk(patient_id, None, None)


def parse_result(prediction):
    if not prediction or not prediction["result_json"]:
        return {}
    try:
        return json.loads(prediction["result_json"])
    except json.JSONDecodeError:
        return {}
