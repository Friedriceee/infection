from models import case_model
from utils.validators import to_float


TEXT_FIELDS = {"tube", "site", "other_inf", "transparency", "remark", "visit_time"}


def payload_from_form(form, patient_id, doctor_id):
    data = {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
    }
    for field in case_model.CASE_FIELDS:
        if field in ("patient_id", "doctor_id"):
            continue
        value = (form.get(field) or "").strip()
        if field in case_model.NUMERIC_FIELDS:
            data[field] = to_float(value)
        else:
            data[field] = value
    return data


def row_to_dict(row):
    return {key: row[key] for key in row.keys()}


def create_case_from_form(form, patient_id, doctor_id):
    data = payload_from_form(form, patient_id, doctor_id)
    return case_model.create_case(data)


def update_case_from_form(case_id, form, patient_id, doctor_id):
    data = payload_from_form(form, patient_id, doctor_id)
    case_model.update_case(case_id, data)

