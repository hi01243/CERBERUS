"""Supabase connection for CERBERUS (accounts, incident records, flag images).

Cloud mode is ON when a Supabase project URL and public key are configured,
either through environment variables

    CERBERUS_SUPABASE_URL=https://<project>.supabase.co
    CERBERUS_SUPABASE_KEY=<anon or publishable key>

or a cloud_config.json file (see cloud_config.example.json) placed in the
data folder or next to app.py. Without them CERBERUS keeps its local,
offline accounts exactly as before.

Only ever use the PUBLIC key (legacy "anon" key, or "sb_publishable_...").
The app ships to other people's PCs, so anything in it can be read; the
service_role / secret key would give full access to every account's data.
Row Level Security (supabase/setup.sql) is what keeps accounts apart.
"""
import base64
import json
import os
import sys
import threading
import uuid

BASE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("CERBERUS_DATA_DIR")
CONFIG_NAME = "cloud_config.json"
EVIDENCE_BUCKET = "evidence"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def _read_config_file():
    for folder in (DATA_DIR, BASE_DIR):
        if not folder:
            continue
        path = os.path.join(folder, CONFIG_NAME)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f), path
            except (OSError, ValueError) as e:
                print(f"[cloud] could not read {path}: {e}")
    return {}, None


def _looks_secret(key):
    """True for keys that must never ship inside the app."""
    if key.startswith("sb_secret_"):
        return True
    parts = key.split(".")
    if len(parts) == 3:   # legacy JWT key: check its role claim
        try:
            payload = parts[1] + "=" * (-len(parts[1]) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload))
            return claims.get("role") == "service_role"
        except (ValueError, json.JSONDecodeError):
            return False
    return False


def _load():
    if os.environ.get("CERBERUS_CLOUD", "").strip().lower() in ("0", "off", "false", "no"):
        print("[cloud] CERBERUS_CLOUD=0: using local accounts")
        return None, None, None
    file_cfg, path = _read_config_file()
    url = (os.environ.get("CERBERUS_SUPABASE_URL") or file_cfg.get("supabase_url") or "").strip().rstrip("/")
    # The dashboard sometimes shows the REST endpoint; the client wants the base URL.
    for suffix in ("/rest/v1", "/auth/v1", "/storage/v1"):
        if url.endswith(suffix):
            url = url[: -len(suffix)]
    key = (os.environ.get("CERBERUS_SUPABASE_KEY") or file_cfg.get("supabase_key") or "").strip()
    if not url and not key:
        return None, None, None
    if not url or not key:
        return None, None, "Cloud is half-configured: both the Supabase URL and key are needed."
    if not url.startswith("https://"):
        return None, None, "The Supabase URL must start with https://"
    if _looks_secret(key):
        return None, None, ("Refusing to use a Supabase SECRET / service_role key. "
                            "Use the anon or publishable key instead.")
    if path:
        print(f"[cloud] using settings from {path}")
    return url, key, None


SUPABASE_URL, SUPABASE_KEY, CONFIG_ERROR = _load()
if CONFIG_ERROR:
    print(f"[cloud] {CONFIG_ERROR} Falling back to local accounts.")


def enabled():
    return bool(SUPABASE_URL and SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Device id: tells this PC's uploads apart from other PCs of the same account
# (recordings exist only on the PC that made them).
# ---------------------------------------------------------------------------
def _device_id():
    folder = DATA_DIR or BASE_DIR
    path = os.path.join(folder, "device_id.txt")
    try:
        with open(path, encoding="utf-8") as f:
            value = f.read().strip()
            if value:
                return value
    except OSError:
        pass
    value = uuid.uuid4().hex
    try:
        os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(value)
    except OSError as e:
        print(f"[cloud] could not save device id ({e}); uploads use a temporary one")
    return value


DEVICE_ID = _device_id()


# ---------------------------------------------------------------------------
# Supabase clients: one per signed-in account, kept in memory only.
# Nothing about the login is written to disk; restarting the app signs
# everyone out, as in local mode.
# ---------------------------------------------------------------------------
_clients = {}
_clients_lock = threading.Lock()


class CloudUnavailable(Exception):
    """Supabase could not be reached (no internet, project paused, ...)."""


def new_client():
    from supabase import create_client
    from supabase.lib.client_options import SyncClientOptions
    return create_client(SUPABASE_URL, SUPABASE_KEY,
                         SyncClientOptions(auto_refresh_token=True, persist_session=True))


def remember_client(user_id, client):
    with _clients_lock:
        _clients[user_id] = client


def client_for(user_id):
    with _clients_lock:
        return _clients.get(user_id)


def forget_client(user_id, sign_out=True):
    with _clients_lock:
        client = _clients.pop(user_id, None)
    if client is not None and sign_out:
        try:
            client.auth.sign_out()
        except Exception as e:
            print(f"[cloud] sign-out did not reach Supabase: {e}")


def signed_in_clients():
    with _clients_lock:
        return dict(_clients)


def friendly_error(exc):
    """Short message for the sign-in pages."""
    import httpx
    try:
        from supabase_auth.errors import AuthRetryableError
    except ImportError:
        AuthRetryableError = CloudUnavailable
    message = str(getattr(exc, "message", "") or exc)
    lowered = message.lower()
    if (isinstance(exc, (httpx.TransportError, AuthRetryableError, CloudUnavailable))
            or "getaddrinfo" in lowered or "connection" in lowered or "timed out" in lowered):
        return ("Can't reach the CERBERUS cloud. Check your internet connection. "
                "If it keeps failing, the Supabase project may be paused.")
    if "invalid login credentials" in lowered:
        return "Incorrect email or password."
    if "email not confirmed" in lowered:
        return "Confirm your email first: enter the code that was emailed to you."
    if "error sending" in lowered or "smtp" in lowered or "not authorized" in lowered:
        return ("Supabase couldn't send the email. Check the SMTP settings in Supabase "
                "(Authentication > Emails), or turn off 'Confirm email' while testing.")
    if "rate limit" in lowered or "too many" in lowered:
        return "Too many attempts. Wait a minute and try again."
    if "expired" in lowered or "invalid" in lowered and "otp" in lowered:
        return "That code is invalid or has expired."
    return message or "Something went wrong talking to the cloud."
