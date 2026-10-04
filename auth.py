"""Username accounts, persistent invite codes, and optional TOTP MFA."""
import base64
import hashlib
import hmac
import os
import re
import sqlite3
import secrets
import struct
import time
from contextlib import closing

from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_DB = os.path.join(BASE_DIR, "users.db")

MIN_PASSWORD_LEN = 8
USERNAME_RE = re.compile(r'^[A-Za-z0-9_.-]{3,32}$')

def _q(sql, args=(), fetch=False):
    with closing(sqlite3.connect(AUTH_DB, timeout=10)) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()] if fetch else None
        con.commit()
        return rows if fetch else cur.lastrowid



def init_auth_db():
    _q("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            email_verified INTEGER NOT NULL DEFAULT 0,
            totp_secret TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    try:
        _q("ALTER TABLE users ADD COLUMN email TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        _q("ALTER TABLE users ADD COLUMN totp_secret TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        _q("ALTER TABLE users ADD COLUMN email_verified INTEGER NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    _q("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email COLLATE NOCASE)")
    _init_session_owners()


def generate_totp_secret():
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def verify_totp(secret, code, now=None):
    code = str(code or "").strip()
    if not secret or not re.fullmatch(r"\d{6}", code):
        return False
    try:
        padded_secret = secret + "=" * (-len(secret) % 8)
        key = base64.b32decode(padded_secret, casefold=True)
    except (ValueError, base64.binascii.Error):
        return False
    counter = int((time.time() if now is None else now) // 30)
    for offset in (-1, 0, 1):
        message = struct.pack(">Q", counter + offset)
        digest = hmac.new(key, message, hashlib.sha1).digest()
        index = digest[-1] & 0x0F
        value = (struct.unpack(">I", digest[index:index + 4])[0] & 0x7FFFFFFF) % 1_000_000
        if hmac.compare_digest(f"{value:06d}", code):
            return True
    return False


def get_totp_secret(user_id):
    rows = _q("SELECT totp_secret FROM users WHERE id=?", (user_id,), fetch=True)
    return rows[0]["totp_secret"] if rows else None


def save_totp_secret(user_id, secret):
    _q("UPDATE users SET totp_secret=? WHERE id=?", (secret, user_id))


def create_user(username, password, invite_code=None):
    """Returns (user_id, error). error is None on success.
    Sign-up is open: `invite_code` is accepted for backward compatibility
    and ignored.  Access is controlled by the account itself (session
    ownership), not by a shared code."""
    username = str(username or "").strip()
    if not USERNAME_RE.fullmatch(username):
        return None, "Username must be 3-32 characters: letters, numbers, _ . -"
    if len(password or "") < MIN_PASSWORD_LEN:
        return None, f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if _q("SELECT id FROM users WHERE username=?", (username,), fetch=True):
        return None, "That username is already taken."

    pw_hash = generate_password_hash(password)
    try:
        uid = _q("INSERT INTO users (username, password_hash) VALUES (?,?)",
                 (username, pw_hash))
    except sqlite3.IntegrityError:
        return None, "That username is already taken."
    return uid, None


def verify_user(username, password):
    """Returns the user row on success, else None."""
    rows = _q("SELECT * FROM users WHERE username=?", ((username or "").strip(),), fetch=True)
    if not rows:
        generate_password_hash(password or "")   # keep timing similar to a real lookup
        return None
    user = rows[0]
    return user if check_password_hash(user["password_hash"], password or "") else None


def get_user_by_id(user_id):
    rows = _q("SELECT * FROM users WHERE id=?", (user_id,), fetch=True)
    return rows[0] if rows else None


def user_count():
    rows = _q("SELECT COUNT(*) AS c FROM users", fetch=True)
    return rows[0]["c"] if rows else 0


# ---------------------------------------------------------------------------
# Session ownership: every session belongs to the account that created it.
# ---------------------------------------------------------------------------
def _init_session_owners():
    _q("""CREATE TABLE IF NOT EXISTS session_owners (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    _q("CREATE INDEX IF NOT EXISTS idx_session_owners_user ON session_owners(user_id)")


def first_user_id():
    rows = _q("SELECT MIN(id) AS m FROM users", fetch=True)
    return rows[0]["m"] if rows else None


def first_user_id_excluding(user_id):
    rows = _q("SELECT MIN(id) AS m FROM users WHERE id != ?", (user_id,), fetch=True)
    return rows[0]["m"] if rows else None


def delete_user(user_id, replacement_session_id=None, replacement_owner_id=None):
    with closing(sqlite3.connect(AUTH_DB, timeout=10)) as con:
        con.execute("DELETE FROM session_owners WHERE user_id=?", (user_id,))
        if replacement_session_id and replacement_owner_id is not None:
            con.execute(
                "INSERT OR REPLACE INTO session_owners (session_id, user_id) VALUES (?, ?)",
                (replacement_session_id, replacement_owner_id),
            )
        cur = con.execute("DELETE FROM users WHERE id=?", (user_id,))
        if cur.rowcount != 1:
            con.rollback()
            return False
        con.commit()
        return True


def get_session_owner(session_id):
    rows = _q("SELECT user_id FROM session_owners WHERE session_id=?", (session_id,), fetch=True)
    return rows[0]["user_id"] if rows else None


def set_session_owner(session_id, user_id):
    """Records the owner of a session that has none yet.  Never takes a session
    away from its current owner.  Returns True if `user_id` owns it afterwards."""
    if user_id is None:
        return False
    _q("INSERT OR IGNORE INTO session_owners (session_id, user_id) VALUES (?, ?)",
       (session_id, user_id))
    return get_session_owner(session_id) == user_id


def remove_session_owner(session_id):
    _q("DELETE FROM session_owners WHERE session_id=?", (session_id,))


def list_owned_sessions(user_id):
    if user_id is None:
        return []
    rows = _q("SELECT session_id FROM session_owners WHERE user_id=? ORDER BY created_at DESC",
              (user_id,), fetch=True)
    return [r["session_id"] for r in rows]
