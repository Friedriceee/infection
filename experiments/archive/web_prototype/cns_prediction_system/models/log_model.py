from models.db import get_db, now_text


def create_log(user_id, action, target_type=None, target_id=None, description=None, ip_address=None):
    db = get_db()
    db.execute(
        """
        INSERT INTO operation_logs
        (user_id, action, target_type, target_id, description, ip_address, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (user_id, action, target_type, target_id, description, ip_address, now_text()),
    )
    db.commit()


def list_logs(limit=200):
    return get_db().execute(
        """
        SELECT l.*, u.username, u.real_name
        FROM operation_logs l
        LEFT JOIN users u ON u.id = l.user_id
        ORDER BY l.created_at DESC, l.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()

