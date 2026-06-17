from models import case_model, patient_model
from services import prediction_service


def import_patient_records(records, created_by=None):
    summary = {
        "patients_created": 0,
        "patients_reused": 0,
        "cases_created": 0,
        "cases_skipped": 0,
        "static_predictions": 0,
        "dynamic_predictions": 0,
        "errors": [],
    }

    for record in records:
        patient_data = record["patient"]
        patient = patient_model.get_patient_by_no_any_status(patient_data["patient_no"])
        if patient:
            patient_id = patient["id"]
            if patient["is_deleted"]:
                patient_model.restore_patient(patient_id, patient_data)
            summary["patients_reused"] += 1
        else:
            patient_id = patient_model.create_patient(patient_data)
            summary["patients_created"] += 1

        for case_data in record["cases"]:
            if not case_data.get("visit_time"):
                summary["cases_skipped"] += 1
                continue
            if case_model.case_exists(patient_id, case_data["visit_time"]):
                summary["cases_skipped"] += 1
                continue
            payload = dict(case_data)
            payload["patient_id"] = patient_id
            payload["doctor_id"] = created_by
            case_id = case_model.create_case(payload)
            summary["cases_created"] += 1
            prediction_result = prediction_service.run_auto_prediction_after_case(case_id, created_by)
            if prediction_result.get("static_prediction_id"):
                summary["static_predictions"] += 1
            if prediction_result.get("dynamic_prediction_id"):
                summary["dynamic_predictions"] += 1

    return summary
