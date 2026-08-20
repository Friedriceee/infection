from models.db import get_db, now_text


def create_prediction(data):
    db = get_db()
    cur = db.execute(
        """
        INSERT INTO predictions
        (patient_id, case_id, prediction_type, model_name, model_version, risk_score, risk_level,
         static_score, dynamic_score, dynamic_delta, input_snapshot, result_json, status,
         created_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("patient_id"),
            data.get("case_id"),
            data.get("prediction_type"),
            data.get("model_name"),
            data.get("model_version"),
            data.get("risk_score"),
            data.get("risk_level"),
            data.get("static_score"),
            data.get("dynamic_score"),
            data.get("dynamic_delta"),
            data.get("input_snapshot"),
            data.get("result_json"),
            data.get("status", "valid"),
            data.get("created_by"),
            now_text(),
        ),
    )
    db.commit()
    return cur.lastrowid


def get_prediction(prediction_id):
    return get_db().execute(
        """
        SELECT pr.*, p.patient_no, p.name AS patient_name, p.sex, p.age, p.department AS patient_department
        FROM predictions pr
        JOIN patients p ON p.id = pr.patient_id
        WHERE pr.id = ? AND pr.status = 'valid'
        """,
        (prediction_id,),
    ).fetchone()


def list_predictions_by_patient(patient_id):
    return get_db().execute(
        """
        SELECT * FROM predictions
        WHERE patient_id = ? AND status = 'valid'
        ORDER BY created_at DESC, id DESC
        """,
        (patient_id,),
    ).fetchall()


def list_predictions_for_case(case_id, prediction_type=None):
    params = [case_id]
    where = ["case_id = ?", "status = 'valid'"]
    if prediction_type:
        where.append("prediction_type = ?")
        params.append(prediction_type)
    return get_db().execute(
        f"SELECT * FROM predictions WHERE {' AND '.join(where)} ORDER BY created_at DESC, id DESC",
        params,
    ).fetchall()


def latest_prediction(patient_id, prediction_type=None):
    params = [patient_id]
    where = ["patient_id = ?", "status = 'valid'"]
    if prediction_type:
        where.append("prediction_type = ?")
        params.append(prediction_type)
    return get_db().execute(
        f"SELECT * FROM predictions WHERE {' AND '.join(where)} ORDER BY created_at DESC, id DESC LIMIT 1",
        params,
    ).fetchone()


def invalidate_by_case(case_id):
    db = get_db()
    db.execute(
        """
        UPDATE predictions
        SET status = 'invalid'
        WHERE case_id = ?
        """,
        (case_id,),
    )
    db.commit()


def invalidate_by_patient(patient_id):
    db = get_db()
    db.execute(
        """
        UPDATE predictions
        SET status = 'invalid'
        WHERE patient_id = ?
        """,
        (patient_id,),
    )
    db.commit()


def count_by_type(prediction_type):
    return get_db().execute(
        """
        SELECT COUNT(*) AS count FROM predictions
        WHERE prediction_type = ? AND status = 'valid'
        """,
        (prediction_type,),
    ).fetchone()["count"]


def list_recent_predictions(limit=8):
    return get_db().execute(
        """
        SELECT pr.*, p.patient_no, p.name AS patient_name
        FROM predictions pr
        JOIN patients p ON p.id = pr.patient_id
        WHERE pr.status = 'valid'
        ORDER BY pr.created_at DESC, pr.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
