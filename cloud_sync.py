"""Background upload of incident records and flag images to Supabase.

Local-first: incidents are always written to the local incidents.db first
(recording.py), so detection never waits for the network. This worker then
copies them to the cloud every few seconds, for every account currently
signed in on this PC, and retries automatically after a network failure.

What is uploaded: the incident row (type, times, confidence, dismissed, ...)
and its thumbnail JPEG. Recordings are never uploaded.

Only sessions owned by a cloud account are uploaded. Sessions that belong
to old local accounts (integer ids from users.db) never match a cloud user
id, so they stay on this PC.
"""
import os
import sqlite3
import threading
import time
from contextlib import closing

import cloud
from auth import list_owned_sessions, session_display_name
from config import VIOLATION_FOLDER
from recording import INCIDENT_DB, apply_cloud_dismiss, utc_now_iso

SYNC_INTERVAL_SEC = 8
BATCH_SIZE = 50
ERROR_LOG_EVERY_SEC = 60
REMOTE_CACHE_SEC = 20          # how long the list of other-PC sessions is reused
IMAGE_CACHE_MAX = 300          # flag images from other PCs kept in memory

_stop = threading.Event()
_thread = None
_status = {"last_ok": None, "last_error": None, "uploaded": 0}
_last_error_log = 0.0
# Columns added to supabase/setup.sql after the first version. None = not
# known yet, False = this project doesn't have it (re-run setup.sql), so the
# app keeps working without it.
_optional = {"session_name": None, "dismissed_at": None}
_dismiss_watermark = {}        # uid -> newest dismissed_at already pulled
_remote_cache = {}             # uid -> (time, {session_id: info})
_image_cache = {}              # object name -> bytes
_cache_lock = threading.Lock()


def _missing_column(exc):
    """Name of the optional column a Supabase error complains about, or None."""
    text = str(getattr(exc, "message", "") or exc).lower()
    if "column" not in text and "schema" not in text:
        return None
    for name in _optional:
        if name in text:
            return name
    return None


def _mark_missing(name):
    if _optional.get(name) is not False:
        _optional[name] = False
        print(f"[cloud] the incidents table has no {name} column yet; "
              "re-run supabase/setup.sql in Supabase to enable it")


def _norm_iso(value):
    """Any ISO time -> one fixed UTC format, so times compare as text."""
    if not value:
        return None
    from datetime import datetime, timezone
    try:
        t = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def _db():
    con = sqlite3.connect(INCIDENT_DB, timeout=10)
    con.row_factory = sqlite3.Row
    return closing(con)


def _object_name(uid, thumb):
    rel = (thumb or "").replace("\\", "/")
    if rel.startswith("violations/"):
        rel = rel[len("violations/"):]
    if not rel or ".." in rel.split("/"):
        return None, None
    return f"{uid}/{cloud.DEVICE_ID}/{rel}", os.path.join(VIOLATION_FOLDER, *rel.split("/"))


def _iso_utc(sqlite_ts):
    # SQLite CURRENT_TIMESTAMP is UTC "YYYY-MM-DD HH:MM:SS".
    return sqlite_ts.replace(" ", "T") + "Z" if sqlite_ts else None


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def _push(uid, client):
    sessions = list_owned_sessions(uid)
    if not sessions:
        return 0
    marks = ",".join("?" * len(sessions))
    with _db() as con:
        rows = [dict(r) for r in con.execute(
            f"SELECT * FROM incidents WHERE session_id IN ({marks}) AND rev > synced_rev "
            f"ORDER BY id LIMIT {BATCH_SIZE}", sessions)]
    if not rows:
        return 0

    bucket = client.storage.from_(cloud.EVIDENCE_BUCKET)
    records, thumb_confs = [], {}
    for r in rows:
        name, path = _object_name(uid, r.get("thumb"))
        conf = r.get("confidence") or 0.0
        uploaded_conf = r.get("thumb_synced_conf")
        if name and os.path.isfile(path) and (uploaded_conf is None or conf > uploaded_conf):
            with open(path, "rb") as f:
                bucket.upload(name, f.read(),
                              {"content-type": "image/jpeg", "upsert": "true"})
            thumb_confs[r["id"]] = conf
        records.append({
            "owner": uid,
            "device_id": cloud.DEVICE_ID,
            "local_id": r["id"],
            "session_id": r["session_id"],
            "camera_id": r.get("camera_id"),
            "track_id": r.get("track_id"),
            "violation_type": r.get("violation_type") or "UNKNOWN",
            "confidence": r.get("confidence"),
            "start_offset": r.get("start_offset"),
            "end_offset": r.get("end_offset"),
            "recording": r.get("recording"),
            "thumb_path": name,
            "dismissed": bool(r.get("dismissed")),
            "created_at": _iso_utc(r.get("created_at")),
        })
        if _optional["session_name"] is not False:
            records[-1]["session_name"] = session_display_name(r["session_id"])
        if _optional["dismissed_at"] is not False:
            records[-1]["dismissed_at"] = r.get("dismissed_at")

    _upsert(client, records)

    with _db() as con:
        for r in rows:
            # Mark the revision we uploaded; a change made meanwhile has a
            # higher rev and is picked up on the next pass.
            if r["id"] in thumb_confs:
                con.execute("UPDATE incidents SET synced_rev=?, thumb_synced_conf=? WHERE id=?",
                            (r["rev"], thumb_confs[r["id"]], r["id"]))
            else:
                con.execute("UPDATE incidents SET synced_rev=? WHERE id=?", (r["rev"], r["id"]))
        con.commit()
    return len(rows)


def _upsert(client, records):
    for _ in range(len(_optional) + 1):
        try:
            client.table("incidents").upsert(records, on_conflict="owner,device_id,local_id").execute()
            for name in _optional:
                if any(name in r for r in records):
                    _optional[name] = True
            return
        except Exception as e:
            name = _missing_column(e)
            if not name:
                raise
            _mark_missing(name)
            for r in records:
                r.pop(name, None)
    raise RuntimeError("upload failed after removing optional columns")


# ---------------------------------------------------------------------------
# Dismiss / restore from any PC
# ---------------------------------------------------------------------------
NEEDS_SETUP = "Re-run supabase/setup.sql in Supabase to dismiss flags from other PCs."


def _pull_dismissals(uid, client):
    """Take dismiss/restore changes other PCs made to flags recorded HERE."""
    if _optional["dismissed_at"] is False:
        return 0
    query = (client.table("incidents").select("local_id,session_id,dismissed,dismissed_at")
             .eq("device_id", cloud.DEVICE_ID).not_.is_("dismissed_at", "null"))
    mark = _dismiss_watermark.get(uid)
    if mark:
        query = query.gt("dismissed_at", mark)
    try:
        rows = query.order("dismissed_at").range(0, 999).execute().data or []
    except Exception as e:
        if _missing_column(e) == "dismissed_at":
            _mark_missing("dismissed_at")
            return 0
        raise
    owned = set(list_owned_sessions(uid))
    changed = 0
    for row in rows:
        stamp = _norm_iso(row.get("dismissed_at"))
        if not stamp:
            continue
        if row.get("session_id") in owned and apply_cloud_dismiss(
                row["local_id"], bool(row.get("dismissed")), stamp):
            changed += 1
            print(f"[cloud] flag #{row['local_id']} was "
                  f"{'dismissed' if row.get('dismissed') else 'restored'} on another PC")
        if mark is None or stamp > mark:
            mark = stamp
    if mark:
        _dismiss_watermark[uid] = mark
    return changed


def set_remote_dismissed(uid, cloud_id, dismissed):
    """Dismiss or restore a flag that was recorded on another PC. That PC
    picks the change up on its next sync. Returns an error message or None."""
    client = cloud.client_for(uid) if cloud.enabled() and uid else None
    if client is None:
        return "Not signed in to the cloud."
    if _optional["dismissed_at"] is False:
        return NEEDS_SETUP
    try:
        result = (client.table("incidents")
                  .update({"dismissed": bool(dismissed), "dismissed_at": utc_now_iso()})
                  .eq("id", cloud_id).neq("device_id", cloud.DEVICE_ID).execute())
    except Exception as e:
        if _missing_column(e) == "dismissed_at":
            _mark_missing("dismissed_at")
            return NEEDS_SETUP
        return cloud.friendly_error(e)
    if not result.data:
        return "That flag was not found."
    with _cache_lock:
        _remote_cache.pop(uid, None)
    return None


# ---------------------------------------------------------------------------
# Reading this account's flags from OTHER PCs (review on any device).
# Their videos stay on the PC that recorded them, so only the flag record
# and image are shown. They are read-only here.
# ---------------------------------------------------------------------------
def _select_remote(client, columns, **filters):
    cols = columns + ("" if _optional["session_name"] is False else ",session_name")
    query = client.table("incidents").select(cols).neq("device_id", cloud.DEVICE_ID)
    for key, value in filters.items():
        query = query.eq(key, value)
    try:
        return query.range(0, 4999).execute().data or []
    except Exception as e:
        if _missing_column(e) != "session_name" or _optional["session_name"] is False:
            raise
        _mark_missing("session_name")
        return _select_remote(client, columns, **filters)


def remote_sessions(uid):
    """{session_id: {"name", "flag_count", "updated_at"}} recorded on other PCs."""
    client = cloud.client_for(uid) if cloud.enabled() and uid else None
    if client is None:
        return {}
    with _cache_lock:
        hit = _remote_cache.get(uid)
        if hit and time.time() - hit[0] < REMOTE_CACHE_SEC:
            return hit[1]
    try:
        rows = _select_remote(client, "session_id,created_at")
    except Exception as e:
        _log_error(uid, e)
        return hit[1] if hit else {}
    out = {}
    for row in rows:
        sid = row.get("session_id")
        if not sid:
            continue
        info = out.setdefault(sid, {"name": row.get("session_name") or sid,
                                    "flag_count": 0, "updated_at": None})
        info["flag_count"] += 1
        stamp = _local_ts(row.get("created_at"))
        if stamp and (info["updated_at"] is None or stamp > info["updated_at"]):
            info["updated_at"] = stamp
    with _cache_lock:
        _remote_cache[uid] = (time.time(), out)
    return out


def is_remote_session(uid, session_id):
    return session_id in remote_sessions(uid)


def remote_session_name(uid, session_id):
    info = remote_sessions(uid).get(session_id)
    return (info or {}).get("name") or session_id


def _local_ts(iso):
    """Cloud ISO time -> the 'YYYY-MM-DD HH:MM:SS' UTC format local rows use."""
    if not iso:
        return None
    from datetime import datetime, timezone
    try:
        value = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.strftime("%Y-%m-%d %H:%M:%S")


def remote_incidents(uid, session_id):
    """Incident rows (review-page shape) for one session, from other PCs."""
    client = cloud.client_for(uid) if cloud.enabled() and uid else None
    if client is None:
        return []
    try:
        rows = _select_remote(
            client, "id,device_id,camera_id,track_id,violation_type,confidence,"
                    "start_offset,end_offset,thumb_path,dismissed,created_at",
            session_id=session_id)
    except Exception as e:
        _log_error(uid, e)
        return []
    return [{
        "id": f"r{row['id']}",
        "remote": True,
        "device_id": row.get("device_id"),
        "session_id": session_id,
        "camera_id": row.get("camera_id"),
        "track_id": row.get("track_id"),
        "violation_type": row.get("violation_type") or "UNKNOWN",
        "confidence": row.get("confidence") or 0,
        "start_offset": row.get("start_offset") or 0,
        "end_offset": row.get("end_offset") or 0,
        "recording": None,
        "video_url": None,
        "thumb": None,
        "thumb_path": row.get("thumb_path"),
        "dismissed": 1 if row.get("dismissed") else 0,
        "created_at": _local_ts(row.get("created_at")),
    } for row in rows]


def fetch_evidence(uid, object_name):
    """Bytes of one of this account's flag images from the cloud, or None."""
    client = cloud.client_for(uid) if cloud.enabled() and uid else None
    if client is None or not object_name.startswith(f"{uid}/") or ".." in object_name:
        return None
    with _cache_lock:
        if object_name in _image_cache:
            return _image_cache[object_name]
    try:
        data = client.storage.from_(cloud.EVIDENCE_BUCKET).download(object_name)
    except Exception as e:
        print(f"[cloud] could not download {object_name}: {e}")
        return None
    with _cache_lock:
        if len(_image_cache) >= IMAGE_CACHE_MAX:
            _image_cache.pop(next(iter(_image_cache)))
        _image_cache[object_name] = data
    return data


# ---------------------------------------------------------------------------
# Deletes
# ---------------------------------------------------------------------------
def queue_session_delete(session_id, owner):
    """Called when a session is deleted on this PC: remove its cloud copy
    (rows + images from this device) as soon as the owner is signed in."""
    if not cloud.enabled() or not isinstance(owner, str):
        return
    with _db() as con:
        con.execute("INSERT INTO cloud_deletes (owner, session_id) VALUES (?, ?)", (owner, session_id))
        con.commit()


def _remove_objects(client, names):
    names = [n for n in names if n]
    bucket = client.storage.from_(cloud.EVIDENCE_BUCKET)
    for i in range(0, len(names), 100):
        bucket.remove(names[i:i + 100])


def _process_deletes(uid, client):
    with _db() as con:
        jobs = [dict(r) for r in con.execute(
            "SELECT id, session_id FROM cloud_deletes WHERE owner=? ORDER BY id", (uid,))]
    for job in jobs:
        table = client.table("incidents")
        found = (table.select("thumb_path").eq("device_id", cloud.DEVICE_ID)
                 .eq("session_id", job["session_id"]).execute().data or [])
        _remove_objects(client, [row.get("thumb_path") for row in found])
        (client.table("incidents").delete().eq("device_id", cloud.DEVICE_ID)
         .eq("session_id", job["session_id"]).execute())
        with _db() as con:
            con.execute("DELETE FROM cloud_deletes WHERE id=?", (job["id"],))
            con.commit()
        print(f"[cloud] removed session '{job['session_id']}' from the cloud")


def delete_all_cloud_data(client, uid):
    """Remove every flag image the account has (all devices). Rows are
    removed by delete_my_account() through ON DELETE CASCADE."""
    names, start, page = [], 0, 1000
    while True:
        data = (client.table("incidents").select("thumb_path")
                .range(start, start + page - 1).execute().data or [])
        names += [row.get("thumb_path") for row in data]
        if len(data) < page:
            break
        start += page
    _remove_objects(client, names)
    with _db() as con:
        con.execute("DELETE FROM cloud_deletes WHERE owner=?", (uid,))
        con.commit()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _log_error(uid, exc):
    global _last_error_log
    _status["last_error"] = f"{type(exc).__name__}: {exc}"
    now = time.time()
    if now - _last_error_log >= ERROR_LOG_EVERY_SEC:
        _last_error_log = now
        print(f"[cloud] sync paused for {uid}: {_status['last_error']} (will retry)")


def _run():
    while not _stop.is_set():
        for uid, client in cloud.signed_in_clients().items():
            try:
                _process_deletes(uid, client)
                _pull_dismissals(uid, client)
                while True:   # drain the backlog in batches
                    n = _push(uid, client)
                    _status["uploaded"] += n
                    if n < BATCH_SIZE or _stop.is_set():
                        break
                _status["last_ok"] = time.time()
                _status["last_error"] = None
            except Exception as e:
                _log_error(uid, e)
        _stop.wait(SYNC_INTERVAL_SEC)


def start():
    global _thread
    if not cloud.enabled() or (_thread and _thread.is_alive()):
        return
    _thread = threading.Thread(target=_run, name="cloud-sync", daemon=True)
    _thread.start()
    print(f"[cloud] sync on (device {cloud.DEVICE_ID[:8]}) -> {cloud.SUPABASE_URL}")


def stop():
    _stop.set()


def pending_count(uid):
    sessions = list_owned_sessions(uid)
    if not sessions:
        return 0
    marks = ",".join("?" * len(sessions))
    with _db() as con:
        return con.execute(
            f"SELECT COUNT(*) FROM incidents WHERE session_id IN ({marks}) AND rev > synced_rev",
            sessions).fetchone()[0]


def status(uid):
    return {
        "enabled": cloud.enabled(),
        "pending": pending_count(uid) if uid else 0,
        "last_ok": _status["last_ok"],
        "last_error": _status["last_error"],
    }
