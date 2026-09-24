import sqlite3
import os

DB_NAME = "cerberus.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            confidence REAL,
            image_path TEXT NOT NULL,
            student_track_id INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration safety: if cerberus.db already exists from before this
    # column existed, CREATE TABLE IF NOT EXISTS above is a no-op and the
    # old schema sticks around. Check for the column and add it if missing,
    # so this doesn't crash on an existing dev/demo database.
    cursor.execute("PRAGMA table_info(violations)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "student_track_id" not in existing_columns:
        cursor.execute("ALTER TABLE violations ADD COLUMN student_track_id INTEGER")

    conn.commit()
    conn.close()

def log_violation(session_id, v_type, confidence, img_path, student_track_id=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO violations (session_id, violation_type, confidence, image_path, student_track_id)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, v_type, confidence, img_path, student_track_id))
    conn.commit()
    conn.close()

def get_recent_violations(session_id, limit=20):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM violations 
        WHERE session_id = ? 
        ORDER BY id DESC 
        LIMIT ?
    """, (session_id, limit))
    records = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return records