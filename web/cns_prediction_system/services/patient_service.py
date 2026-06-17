from models import patient_model


def build_dashboard_stats(today):
    return {
        "patient_count": patient_model.count_patients(),
        "today_patient_count": patient_model.count_today_patients(today),
        "high_risk_count": patient_model.count_high_risk_patients(),
    }


def get_patient_overview(patient_id):
    return patient_model.get_patient(patient_id)
