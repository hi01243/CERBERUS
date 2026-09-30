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


def _load_invite_code():
    configured = os.environ.get("CERBERUS_INVITE_CODE", "").strip()
    if configured:
        return configured, True

    _q("CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    rows = _q("SELECT value FROM app_settings WHERE key='invite_code'", fetch=True)
    if rows:
        return rows[0]["value"], False

    generated = secrets.token_urlsafe(6)
    _q("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('invite_code', ?)", (generated,))
    rows = _q("SELECT value FROM app_settings WHERE key='invite_code'", fetch=True)
    return rows[0]["value"], False


INVITE_CODE, INVITE_CODE_CONFIGURED = _load_invite_code()
if not INVITE_CODE_CONFIGURED:
    print("=" * 64)
    print(f"  CERBERUS sign-up invite code: {INVITE_CODE}")
    print("  This code is stored in users.db and remains stable across restarts.")
    print("  Share this with proctors who need an account.")
    print("  (set CERBERUS_INVITE_CODE to override it)")
    print("=" * 64)


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


def create_user(username, password, invite_code):
    """Returns (user_id, error). error is None on success."""
    username = str(username or "").strip()
    invite_code = str(invite_code or "").strip()

    supplied_code = invite_code.casefold().encode("utf-8")
    expected_code = INVITE_CODE.strip().casefold().encode("utf-8")
    if not hmac.compare_digest(supplied_code, expected_code):
        return None, "Invalid invite code."
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