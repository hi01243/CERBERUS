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
            camera_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migration safety: if cerberus.db already exists from before either
    # of these columns existed, CREATE TABLE IF NOT EXISTS is a no-op and
    # the old schema sticks around. Add any missing columns so this
    # doesn't crash against an existing dev/demo database.
    cursor.execute("PRAGMA table_info(violations)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if "student_track_id" not in existing_columns:
        cursor.execute("ALTER TABLE violations ADD COLUMN student_track_id INTEGER")
    if "camera_id" not in existing_columns:
        cursor.execute("ALTER TABLE violations ADD COLUMN camera_id TEXT")

    conn.commit()
    conn.close()

def log_violation(session_id, v_type, confidence, img_path, student_track_id=None, camera_id=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO violations (session_id, violation_type, confidence, image_path, student_track_id, camera_id)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, v_type, confidence, img_path, student_track_id, camera_id))
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

def clear_violations(session_id):
    """Deletes all DB rows for a session. Does NOT touch files on disk --
    the Flask /reset_session route handles deleting the evidence images
    separately, since that's a filesystem operation, not a DB one."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM violations WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()