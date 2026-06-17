from models.db import get_db, now_text


CASE_FIELDS = [
    "patient_id", "visit_time", "doctor_id", "temperature", "gcs", "tube", "site",
    "other_inf", "transparency", "B_G", "B_WBC", "B_N", "B_Lym", "B_CRP", "B_PCT",
    "B_AC", "B_RBC", "C_G", "C_WBC", "C_RBC", "C_P", "C_N", "remark",
]


NUMERIC_FIELDS = {
    "temperature", "gcs", "B_G", "B_WBC", "B_N", "B_Lym", "B_CRP", "B_PCT",
    "B_AC", "B_RBC", "C_G", "C_WBC", "C_RBC", "C_P", "C_N",
}


def create_case(data):
    db = get_db()
    now = now_text()
    fields = CASE_FIELDS + ["created_at", "updated_at"]
    values = [data.get(field) for field in CASE_FIELDS] + [now, now]
    placeholders = ", ".join(["?"] * len(fields))
    cur = db.execute(
        f"INSERT INTO cases ({', '.join(fields)}) VALUES ({placeholders})",
        values,
    )
    db.commit()
    return cur.lastrowid


def update_case(case_id, data):
    db = get_db()
    assignments = ", ".join([f"{field} = ?" for field in CASE_FIELDS if field != "patient_id"])
    values = [data.get(field) for field in CASE_FIELDS if field != "patient_id"]
    values.extend([now_text(), case_id])
    db.execute(
        f"UPDATE cases SET {assignments}, updated_at = ? WHERE id = ?",
        values,
    )
    db.commit()


def soft_delete_case(case_id):
    db = get_db()
    db.execute(
        """
        UPDATE cases
        SET is_deleted = 1, updated_at = ?
        WHERE id = ?
        """,
        (now_text(), case_id),
    )
    db.commit()


def get_case(case_id):
    return get_db().execute(
        """
        SELECT c.*, p.patient_no, p.name AS patient_name, p.sex, p.age, p.department AS patient_department
        FROM cases c
        JOIN patients p ON p.id = c.patient_id
        WHERE c.id = ? AND c.is_deleted = 0
        """,
        (case_id,),
    ).fetchone()


def list_cases_by_patient(patient_id):
    return get_db().execute(
        """
        SELECT * FROM cases
        WHERE patient_id = ? AND is_deleted = 0
        ORDER BY visit_time ASC, id ASC
        """,
        (patient_id,),
    ).fetchall()


def case_exists(patient_id, visit_time):
    return get_db().execute(
        """
        SELECT id FROM cases
        WHERE patient_id = ? AND visit_time = ? AND is_deleted = 0
        LIMIT 1
        """,
        (patient_id, visit_time),
    ).fetchone()


def count_cases():
    return get_db().execute(
        "SELECT COUNT(*) AS count FROM cases WHERE is_deleted = 0"
    ).fetchone()["count"]


def count_today_cases(today_prefix):
    return get_db().execute(
        """
        SELECT COUNT(*) AS count FROM cases
        WHERE is_deleted = 0 AND created_at LIKE ?
        """,
        (f"{today_prefix}%",),
    ).fetchone()["count"]
