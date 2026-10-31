"""Session helpers, file summaries, incident verification, and ownership guards."""
import os
import re
import shutil
import secrets
from datetime import datetime
from flask import request, session, abort, redirect, url_for, jsonify

from config import VIOLATION_FOLDER, LIVE_ENDPOINTS, SESSION_QUERY_ENDPOINTS
from recording import (
    RECORDINGS_FOLDER, list_incident_sessions, list_incidents,
    clear_incidents, SessionRecorder, IncidentTracker
)
from database import clear_violations
from auth import (
    get_session_owner, set_session_owner, remove_session_owner,
    list_owned_sessions, first_user_id
)
import state
from camera_manager import _finish_recording, _clear_tracking_state


def _valid_session_id(session_id):
    return bool(session_id and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', session_id))


def _session_recordings(session_id):
    if not _valid_session_id(session_id):
        return []
    session_dir = os.path.join(RECORDINGS_FOLDER, session_id)
    recordings = []
    if not os.path.isdir(session_dir):
        return recordings

    for root, _dirs, files in os.walk(session_dir):
        for name in files:
            if not name.lower().endswith('.mp4') or name.lower().endswith('.tmp.mp4'):
                continue
            path = os.path.join(root, name)
            relative_path = os.path.relpath(path, RECORDINGS_FOLDER).replace(os.sep, '/')
            stat = os.stat(path)
            recordings.append({
                'name': name,
                'path': relative_path,
                'url': f'/recordings/{relative_path}',
                'size': stat.st_size,
                'modified': stat.st_mtime,
            })
    return sorted(recordings, key=lambda item: item['modified'], reverse=True)


def _session_summaries():
    summaries = {}
    for row in list_incident_sessions():
        session_id = row.get('session_id')
        if _valid_session_id(session_id):
            summaries[session_id] = {
                'session_id': session_id,
                'flag_count': row['flag_count'],
                'updated_at': row['updated_at'],
            }

    if os.path.isdir(RECORDINGS_FOLDER):
        for name in os.listdir(RECORDINGS_FOLDER):
            if _valid_session_id(name) and os.path.isdir(os.path.join(RECORDINGS_FOLDER, name)):
                summaries.setdefault(name, {'session_id': name, 'flag_count': 0, 'updated_at': None})

    summaries.setdefault(state.CURRENT_SESSION, {
        'session_id': state.CURRENT_SESSION, 'flag_count': 0, 'updated_at': None,
    })
    owned = set(list_owned_sessions(session.get('user_id')))
    summaries = {sid: s for sid, s in summaries.items() if sid in owned}
    for summary in summaries.values():
        summary['recordings'] = _session_recordings(summary['session_id'])
        summary['recording_count'] = len(summary['recordings'])
        if summary['recordings']:
            summary['updated_at'] = max(
                summary['updated_at'] or '',
                datetime.fromtimestamp(summary['recordings'][0]['modified']).isoformat(timespec='seconds'),
            )
    current = summaries.get(state.CURRENT_SESSION)
    if current and not current['recording_count'] and not current['flag_count']:
        summaries.pop(state.CURRENT_SESSION)
    return sorted(summaries.values(), key=lambda item: item['updated_at'] or '', reverse=True)


def _delete_session_data(session_id):
    deleting_current = session_id == state.CURRENT_SESSION
    if deleting_current:
        for entry in list(state.camera_registry.values()):
            _finish_recording(entry)

    clear_violations(session_id)
    clear_incidents(session_id)
    shutil.rmtree(os.path.join(RECORDINGS_FOLDER, session_id), ignore_errors=True)
    shutil.rmtree(os.path.join(VIOLATION_FOLDER, session_id), ignore_errors=True)
    remove_session_owner(session_id)

    if deleting_current:
        for cam_id, entry in list(state.camera_registry.items()):
            _clear_tracking_state(entry)
        while True:
            replacement = "SESSION_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
            if replacement != session_id and not os.path.exists(os.path.join(RECORDINGS_FOLDER, replacement)):
                break
        state.CURRENT_SESSION = replacement
        set_session_owner(replacement, session.get('user_id'))
        os.makedirs(os.path.join(VIOLATION_FOLDER, state.CURRENT_SESSION), exist_ok=True)
        for cam_id, entry in list(state.camera_registry.items()):
            entry["recorder"] = SessionRecorder(state.CURRENT_SESSION, cam_id)
            entry["incidents"] = IncidentTracker()
    return state.CURRENT_SESSION


def _owns_current():
    uid = session.get("user_id")
    return uid is not None and get_session_owner(state.CURRENT_SESSION) == uid


def _all_known_session_ids():
    """Every session that has data on disk or in incident log, plus current."""
    ids = {state.CURRENT_SESSION}
    try:
        for row in list_incident_sessions():
            ids.add(row.get("session_id"))
    except Exception as e:
        print(f"[sessions] could not list incident sessions: {e}")
    for folder in (RECORDINGS_FOLDER, VIOLATION_FOLDER):
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if os.path.isdir(os.path.join(folder, name)):
                    ids.add(name)
    return {i for i in ids if _valid_session_id(i)}


def _assign_unowned_sessions(user_id):
    """Gives every existing session that has no owner to user_id."""
    if user_id is None:
        return 0
    count = 0
    for sid in sorted(_all_known_session_ids()):
        if get_session_owner(sid) is None and set_session_owner(sid, user_id):
            count += 1
    if count:
        print(f"[sessions] gave {count} existing session(s) with no owner to user #{user_id}")
    return count


def _user_owns_incident(inc_id, uid):
    if uid is None:
        return False
    for sid in list_owned_sessions(uid):
        try:
            if any(r.get("id") == inc_id for r in list_incidents(sid)):
                return True
        except Exception as e:
            print(f"[sessions] incident lookup failed for {sid}: {e}")
    return False


def _guard_static():
    """Evidence images live under /static/violations/<session>/..."""
    filename = (request.view_args or {}).get("filename", "")
    if not filename.startswith("violations/"):
        return None
    uid = session.get("user_id")
    if uid is None:
        abort(404)
    parts = filename.split("/")
    if len(parts) == 2:
        return None
    sid = parts[1]
    if not _valid_session_id(sid) or get_session_owner(sid) != uid:
        abort(404)
    return None


def _guard_session_access():
    ep = request.endpoint
    uid = session.get("user_id")

    if ep in LIVE_ENDPOINTS:
        if get_session_owner(state.CURRENT_SESSION) != uid:
            if ep == "monitor":
                return redirect(url_for("index"))
            return jsonify({"status": "error", "message":
                            "The live session belongs to another account. "
                            "Create your own session first."}), 403
        return None

    if ep == "dismiss_incident_route":
        if not _user_owns_incident((request.view_args or {}).get("inc_id"), uid):
            abort(404)
        return None

    sid = None
    must_be_valid = False
    if ep in SESSION_QUERY_ENDPOINTS:
        sid = request.args.get("session_id", state.CURRENT_SESSION)
    elif ep == "delete_session_route":
        sid = (request.view_args or {}).get("session_id")
    elif ep == "serve_recording":
        sid = ((request.view_args or {}).get("filepath") or "").split("/")[0]
        must_be_valid = True
    elif ep == "serve_evidence":
        sid = ((request.view_args or {}).get("filepath") or "").split("/")[0]
        must_be_valid = True

    if sid is not None:
        if not _valid_session_id(sid):
            if must_be_valid:
                abort(404)
        elif get_session_owner(sid) != uid:
            abort(404)
    return None

