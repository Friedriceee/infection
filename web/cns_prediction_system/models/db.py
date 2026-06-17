import sqlite3
from datetime import datetime

from flask import current_app, g
from werkzeug.security import generate_password_hash


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(current_app.config["DATABASE"])
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(_error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(current_app.config["DATABASE"])
    db.row_factory = sqlite3.Row
    cur = db.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            real_name TEXT,
            department TEXT,
            created_at TEXT,
            last_login_at TEXT,
            is_active INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_no TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            sex TEXT NOT NULL,
            age INTEGER NOT NULL,
            department TEXT,
            inpatient_no TEXT,
            outpatient_no TEXT,
            admission_time TEXT,
            remark TEXT,
            current_risk_score REAL,
            current_risk_level TEXT,
            created_at TEXT,
            updated_at TEXT,
            is_deleted INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            visit_time TEXT NOT NULL,
            doctor_id INTEGER,
            temperature REAL,
            gcs REAL,
            tube TEXT,
            site TEXT,
            other_inf TEXT,
            transparency TEXT,
            B_G REAL,
            B_WBC REAL,
            B_N REAL,
            B_Lym REAL,
            B_CRP REAL,
            B_PCT REAL,
            B_AC REAL,
            B_RBC REAL,
            C_G REAL,
            C_WBC REAL,
            C_RBC REAL,
            C_P REAL,
            C_N REAL,
            remark TEXT,
            created_at TEXT,
            updated_at TEXT,
            is_deleted INTEGER DEFAULT 0,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            case_id INTEGER,
            prediction_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_level TEXT NOT NULL,
            static_score REAL,
            dynamic_score REAL,
            dynamic_delta REAL,
            input_snapshot TEXT,
            result_json TEXT,
            status TEXT DEFAULT 'valid',
            created_by INTEGER,
            created_at TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id),
            FOREIGN KEY(case_id) REFERENCES cases(id)
        );

        CREATE TABLE IF NOT EXISTS operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            target_type TEXT,
            target_id INTEGER,
            description TEXT,
            ip_address TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            model_version TEXT NOT NULL,
            model_type TEXT NOT NULL,
            description TEXT,
            input_features TEXT,
            status TEXT DEFAULT 'enabled',
            created_at TEXT,
            updated_at TEXT
        );
        """
    )

    created_at = now_text()
    default_users = [
        ("doctor", "123456", "doctor", "默认医生", "神经内科"),
        ("admin", "123456", "admin", "系统管理员", "信息科"),
    ]
    for username, password, role, real_name, department in default_users:
        cur.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cur.fetchone() is None:
            cur.execute(
                """
                INSERT OR IGNORE INTO users (username, password_hash, role, real_name, department, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (username, generate_password_hash(password), role, real_name, department, created_at),
            )

    cur.execute("UPDATE model_versions SET model_name = 'PGFormer' WHERE model_name = 'PGA-AMFormer'")
    cur.execute("UPDATE model_versions SET model_name = 'D-PGFormer' WHERE model_name = 'D-PGA-AMFormer'")
    cur.execute("UPDATE predictions SET model_name = 'PGFormer' WHERE model_name = 'PGA-AMFormer'")
    cur.execute("UPDATE predictions SET model_name = 'D-PGFormer' WHERE model_name = 'D-PGA-AMFormer'")

    default_versions = [
        ("PGFormer", "v1.0", "static", "静态感染风险预测模型", "病例单时间点临床指标、血液指标、脑脊液指标"),
        ("D-PGFormer", "v1.0", "dynamic", "多时间点动态趋势残差融合预测模型", "同一患者多时间点脑脊液趋势指标与最新静态风险"),
    ]
    for model_name, version, model_type, desc, features in default_versions:
        cur.execute(
            "SELECT id FROM model_versions WHERE model_name = ? AND model_version = ? AND model_type = ?",
            (model_name, version, model_type),
        )
        if cur.fetchone() is None:
            cur.execute(
                """
                INSERT OR IGNORE INTO model_versions
                (model_name, model_version, model_type, description, input_features, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'enabled', ?, ?)
                """,
                (model_name, version, model_type, desc, features, created_at, created_at),
            )

    db.commit()
    db.close()
