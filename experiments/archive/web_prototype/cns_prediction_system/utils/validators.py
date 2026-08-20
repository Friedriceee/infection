def validate_patient_payload(data, existing_patient=None):
    errors = []
    patient_no = (data.get("patient_no") or "").strip()
    name = (data.get("name") or "").strip()
    sex = (data.get("sex") or "").strip()
    age_raw = (data.get("age") or "").strip()

    if not patient_no:
        errors.append("患者编号不能为空")
    if not name:
        errors.append("姓名不能为空")
    if not sex:
        errors.append("性别不能为空")
    try:
        age = int(age_raw)
        if age < 0 or age > 120:
            errors.append("年龄必须在 0-120 之间")
    except ValueError:
        errors.append("年龄必须为数字")

    if existing_patient:
        errors.append("患者编号已存在")

    return errors


def to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

