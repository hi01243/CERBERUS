"""
auth.py -- per-user accounts for CERBERUS, replacing the single shared
password from Fix 8.

- Passwords are hashed with werkzeug's salted PBKDF2-SHA256 (via
  generate_password_hash / check_password_hash) -- never stored or
  compared in plaintext. werkzeug ships with Flask, so no new dependency.
- Sign-up still requires an invite code, printed to the console at
  startup (like the old shared password was), so moving to named
  accounts doesn't quietly become "anyone on the network can create
  their own login". Share the code with proctors once; each of them
  then owns a password only they know.
- verify_user() hashes a dummy password on a lookup miss so a login
  attempt against a nonexistent username takes about as long as one
  against a real username -- a basic defense against username
  enumeration via response timing.
"""
import os
import re
import sqlite3
import secrets
from contextlib import closing

from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTH_DB = os.path.join(BASE_DIR, "users.db")

MIN_PASSWORD_LEN = 8
USERNAME_RE = re.compile(r'^[A-Za-z0-9_.-]{3,32}$')

INVITE_CODE = os.environ.get("CERBERUS_INVITE_CODE")
if not INVITE_CODE:
    INVITE_CODE = secrets.token_urlsafe(6)
    print("=" * 64)
    print(f"  CERBERUS sign-up invite code for this run: {INVITE_CODE}")
    print("  Share this with proctors who need an account.")
    print("  (set CERBERUS_INVITE_CODE to fix it across restarts)")
    print("=" * 64)


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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")


def create_user(username, password, invite_code):
    """Returns (user_id, error). error is None on success."""
    username = (username or "").strip()

    if invite_code != INVITE_CODE:
        return None, "Invalid invite code."
    if not USERNAME_RE.match(username):
        return None, "Username must be 3-32 characters: letters, numbers, _ . -"
    if len(password or "") < MIN_PASSWORD_LEN:
        return None, f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if _q("SELECT id FROM users WHERE username=?", (username,), fetch=True):
        return None, "That username is already taken."

    pw_hash = generate_password_hash(password)
    uid = _q("INSERT INTO users (username, password_hash) VALUES (?,?)", (username, pw_hash))
    return uid, None


def verify_user(username, password):
    """Returns the user row on success, else None."""
    rows = _q("SELECT * FROM users WHERE username=?", ((username or "").strip(),), fetch=True)
    if not rows:
        generate_password_hash(password or "")   # keep timing similar to a real lookup
        return None
    user = rows[0]
    return user if check_password_hash(user["password_hash"], password or "") else None


def user_count():
    rows = _q("SELECT COUNT(*) AS c FROM users", fetch=True)
    return rows[0]["c"] if rows else 0