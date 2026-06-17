from models.db import get_db, now_text
from werkzeug.security import generate_password_hash


def get_user_by_username(username):
    return get_db().execute(
        "SELECT * FROM users WHERE username = ? AND is_active = 1",
        (username,),
    ).fetchone()


def get_user_by_id(user_id):
    return get_db().execute(
        "SELECT * FROM users WHERE id = ? AND is_active = 1",
        (user_id,),
    ).fetchone()


def update_last_login(user_id):
    db = get_db()
    db.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (now_text(), user_id))
    db.commit()


def list_users():
    return get_db().execute(
        """
        SELECT id, username, role, real_name, department, created_at, last_login_at, is_active
        FROM users
        ORDER BY created_at DESC, id DESC
        """
    ).fetchall()


def create_user(data):
    db = get_db()
    cur = db.execute(
        """
        INSERT INTO users
        (username, password_hash, role, real_name, department, created_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, 1)
        """,
        (
            data.get("username"),
            generate_password_hash(data.get("password")),
            data.get("role"),
            data.get("real_name"),
            data.get("department"),
            now_text(),
        ),
    )
    db.commit()
    return cur.lastrowid
