from models.db import get_db, now_text


def list_model_versions():
    return get_db().execute(
        """
        SELECT * FROM model_versions
        ORDER BY model_type ASC, updated_at DESC, id DESC
        """
    ).fetchall()


def get_model_version(version_id):
    return get_db().execute(
        "SELECT * FROM model_versions WHERE id = ?",
        (version_id,),
    ).fetchone()


def create_model_version(data):
    db = get_db()
    now = now_text()
    cur = db.execute(
        """
        INSERT INTO model_versions
        (model_name, model_version, model_type, description, input_features, status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            data.get("model_name"),
            data.get("model_version"),
            data.get("model_type"),
            data.get("description"),
            data.get("input_features"),
            data.get("status", "disabled"),
            now,
            now,
        ),
    )
    db.commit()
    return cur.lastrowid


def set_enabled(version_id):
    db = get_db()
    row = get_model_version(version_id)
    if not row:
        return False
    now = now_text()
    db.execute(
        """
        UPDATE model_versions
        SET status = 'disabled', updated_at = ?
        WHERE model_type = ?
        """,
        (now, row["model_type"]),
    )
    db.execute(
        """
        UPDATE model_versions
        SET status = 'enabled', updated_at = ?
        WHERE id = ?
        """,
        (now, version_id),
    )
    db.commit()
    return True


def delete_model_version(version_id):
    db = get_db()
    db.execute("DELETE FROM model_versions WHERE id = ?", (version_id,))
    db.commit()
