from models.db import get_db, now_text


def create_patient(data):
    db = get_db()
    now = now_text()
    cur = db.execute(
        """
        INSERT INTO patients
        (patient_no, name, sex, age, department, inpatient_no, outpatient_no, admission_time,
         remark, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("patient_no"),
            data.get("name"),
            data.get("sex"),
            data.get("age"),
            data.get("department"),
            data.get("inpatient_no"),
            data.get("outpatient_no"),
            data.get("admission_time"),
            data.get("remark"),
            now,
            now,
        ),
    )
    db.commit()
    return cur.lastrowid


def get_patient(patient_id):
    return get_db().execute(
        "SELECT * FROM patients WHERE id = ? AND is_deleted = 0",
        (patient_id,),
    ).fetchone()


def get_patient_by_no(patient_no):
    return get_db().execute(
        "SELECT * FROM patients WHERE patient_no = ? AND is_deleted = 0",
        (patient_no,),
    ).fetchone()


def get_patient_by_no_any_status(patient_no):
    return get_db().execute(
        "SELECT * FROM patients WHERE patient_no = ?",
        (patient_no,),
    ).fetchone()


def restore_patient(patient_id, data=None):
    db = get_db()
    data = data or {}
    db.execute(
        """
        UPDATE patients
        SET name = COALESCE(NULLIF(?, ''), name),
            sex = COALESCE(NULLIF(?, ''), sex),
            age = COALESCE(?, age),
            department = COALESCE(NULLIF(?, ''), department),
            inpatient_no = COALESCE(NULLIF(?, ''), inpatient_no),
            outpatient_no = COALESCE(NULLIF(?, ''), outpatient_no),
            admission_time = COALESCE(NULLIF(?, ''), admission_time),
            remark = COALESCE(NULLIF(?, ''), remark),
            is_deleted = 0,
            updated_at = ?
        WHERE id = ?
        """,
        (
            data.get("name"),
            data.get("sex"),
            data.get("age"),
            data.get("department"),
            data.get("inpatient_no"),
            data.get("outpatient_no"),
            data.get("admission_time"),
            data.get("remark"),
            now_text(),
            patient_id,
        ),
    )
    db.commit()


def list_patients(keyword="", risk_level=""):
    params = []
    where = ["p.is_deleted = 0"]
    if keyword:
        where.append("(p.patient_no LIKE ? OR p.name LIKE ?)")
        params.extend([f"%{keyword}%", f"%{keyword}%"])
    if risk_level:
        where.append("p.current_risk_level = ?")
        params.append(risk_level)

    sql = f"""
        SELECT p.*,
               COUNT(c.id) AS case_count,
               MAX(c.visit_time) AS latest_visit_time
        FROM patients p
        LEFT JOIN cases c ON c.patient_id = p.id AND c.is_deleted = 0
        WHERE {' AND '.join(where)}
        GROUP BY p.id
        ORDER BY p.updated_at DESC, p.id DESC
    """
    return get_db().execute(sql, params).fetchall()


def update_patient_current_risk(patient_id, risk_score, risk_level):
    db = get_db()
    db.execute(
        """
        UPDATE patients
        SET current_risk_score = ?, current_risk_level = ?, updated_at = ?
        WHERE id = ?
        """,
        (risk_score, risk_level, now_text(), patient_id),
    )
    db.commit()


def soft_delete_patient(patient_id):
    db = get_db()
    db.execute(
        """
        UPDATE patients
        SET is_deleted = 1, updated_at = ?
        WHERE id = ?
        """,
        (now_text(), patient_id),
    )
    db.execute(
        """
        UPDATE cases
        SET is_deleted = 1, updated_at = ?
        WHERE patient_id = ?
        """,
        (now_text(), patient_id),
    )
    db.execute(
        """
        UPDATE predictions
        SET status = 'invalid'
        WHERE patient_id = ?
        """,
        (patient_id,),
    )
    db.commit()


def count_patients():
    return get_db().execute(
        "SELECT COUNT(*) AS count FROM patients WHERE is_deleted = 0"
    ).fetchone()["count"]


def count_today_patients(today_prefix):
    return get_db().execute(
        """
        SELECT COUNT(*) AS count FROM patients
        WHERE is_deleted = 0 AND created_at LIKE ?
        """,
        (f"{today_prefix}%",),
    ).fetchone()["count"]


def count_high_risk_patients():
    return get_db().execute(
        "SELECT COUNT(*) AS count FROM patients WHERE is_deleted = 0 AND current_risk_level = '高风险'"
    ).fetchone()["count"]


def list_high_risk_patients(limit=8):
    return get_db().execute(
        """
        SELECT * FROM patients
        WHERE is_deleted = 0 AND current_risk_level = '高风险'
        ORDER BY current_risk_score DESC, updated_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
