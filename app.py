"""Cerberus - AI-powered exam proctoring web application."""
import os
import re
import secrets
import atexit
import cv2
import numpy as np
from flask import (
    Flask, render_template, Response, jsonify, request,
    send_from_directory, session, redirect, url_for
)

from config import (
    DEMO_MODE, _CUDA, _GPU_NAME, USE_HALF, BUILD_ID,
    VIOLATION_FOLDER, DETECTION_ENABLED, RESET_CONFIRM_PHRASE
)
import state
from database import init_db, log_violation, get_recent_violations, clear_violations
from recording import (
    init_incident_db, cleanup_orphaned_recordings, RECORDINGS_FOLDER,
    list_incidents, dismiss_incident, SessionRecorder, IncidentTracker
)
from auth import (
    init_auth_db, get_totp_secret, get_session_owner, set_session_owner, first_user_id
)
from camera_manager import (
    BrowserCapture, add_camera, activate_camera, pause_camera,
    remove_camera, scan_local_cameras, generate_frames, _finish_recording
)
from session_manager import (
    _valid_session_id, _session_recordings, _session_summaries,
    _delete_session_data, _owns_current, _assign_unowned_sessions
)
from auth_routes import init_auth_routes

print(f"[cerberus] app.py build {BUILD_ID}")
_where = ("GPU: " + _GPU_NAME) if _CUDA else "CPU (torch.cuda.is_available() is False)"
print(f"[cerberus] inference on {_where}{', half precision' if USE_HALF else ''}")

app = Flask(__name__)
app.secret_key = os.environ.get("CERBERUS_SECRET_KEY") or secrets.token_hex(32)

init_db()
init_incident_db()
init_auth_db()
cleanup_orphaned_recordings()

_assign_unowned_sessions(first_user_id())

# Register auth and security routes / middleware
init_auth_routes(app)


# ==========================================
# CORE WEB VIEWS
# ==========================================

@app.route('/')
def index():
    return render_template(
        'sessions.html',
        current_session=state.CURRENT_SESSION if _owns_current() else None,
        monitoring_active=bool(state.monitoring_active and _owns_current()),
        mfa_enabled=bool(get_totp_secret(session.get('user_id')))
    )


@app.route('/monitor')
def monitor():
    if DEMO_MODE:
        return redirect(url_for('index'))
    return render_template('index.html')


@app.route('/review')
def review():
    session_id = request.args.get('session_id', state.CURRENT_SESSION)
    if not _valid_session_id(session_id):
        return redirect(url_for('index'))
    return render_template('review.html', session_id=session_id)


# ==========================================
# SESSION & INCIDENT MANAGEMENT ROUTES
# ==========================================

@app.route('/sessions')
def sessions_route():
    return jsonify(_session_summaries())


@app.route('/session_files')
def session_files_route():
    session_id = request.args.get('session_id', '')
    if not _valid_session_id(session_id):
        return jsonify({'message': 'Invalid session.'}), 400
    return jsonify(_session_recordings(session_id))


@app.route('/session', methods=['GET'])
def get_session_route():
    return jsonify({"session_id": state.CURRENT_SESSION})


@app.route('/session/new', methods=['POST'])
def new_session_route():
    if state.monitoring_active:
        return jsonify({
            "status":  "error",
            "message": ("Stop monitoring before starting a new session." if _owns_current()
                        else "Another account is currently monitoring. Try again when they have paused."),
        }), 400

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "Session name is required."}), 400

    safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', name)[:64]
    if not safe_name:
        return jsonify({"status": "error", "message": "Session name has no usable characters."}), 400

    uid = session.get('user_id')
    existing_owner = get_session_owner(safe_name)
    if existing_owner is not None and existing_owner != uid:
        return jsonify({"status": "error",
                        "message": "That session name is already in use. Choose another."}), 409
    set_session_owner(safe_name, uid)

    for _entry in list(state.camera_registry.values()):
        _finish_recording(_entry)

    state.CURRENT_SESSION = safe_name
    os.makedirs(os.path.join(VIOLATION_FOLDER, state.CURRENT_SESSION), exist_ok=True)

    for cam_id, entry in list(state.camera_registry.items()):
        entry["recorder"]  = SessionRecorder(state.CURRENT_SESSION, cam_id)
        entry["incidents"] = IncidentTracker()

    return jsonify({"status": "ok", "session_id": state.CURRENT_SESSION})


@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session_route(session_id):
    if not _valid_session_id(session_id):
        return jsonify({"message": "Invalid session."}), 400
    if state.monitoring_active and session_id == state.CURRENT_SESSION:
        return jsonify({"message": "Pause monitoring before deleting the active session."}), 400
    data = request.get_json(silent=True) or {}
    if data.get("confirm") != session_id:
        return jsonify({"message": "Type the exact session name to confirm deletion."}), 400

    print(f"[session] {session.get('username', 'unknown')} deleted session '{session_id}'")
    replacement = _delete_session_data(session_id)
    return jsonify({"status": "deleted", "session_id": session_id,
                    "current_session": replacement})


@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Deletes current session after phrase confirmation."""
    if state.monitoring_active:
        return jsonify({
            "status":  "error",
            "message": "Stop monitoring before resetting.",
        }), 400

    data = request.get_json(silent=True) or {}
    typed = str(data.get("confirm", "")).strip().lower()
    valid_phrases = {
        RESET_CONFIRM_PHRASE.lower(),
        "delete all data",
    }
    if typed not in valid_phrases:
        return jsonify({
            "status":  "error",
            "message": f'Type "{RESET_CONFIRM_PHRASE}" to confirm this cannot be undone.',
        }), 400

    deleted_session = state.CURRENT_SESSION
    print(f"[session] {session.get('username', 'unknown')} reset session '{deleted_session}'")
    replacement = _delete_session_data(deleted_session)
    return jsonify({"status": "reset", "session_id": deleted_session,
                    "current_session": replacement})


@app.route('/incidents')
def incidents_route():
    session_id = request.args.get('session_id', state.CURRENT_SESSION)
    if not _valid_session_id(session_id):
        return jsonify({'message': 'Invalid session.'}), 400
    rows = list_incidents(session_id)
    for r in rows:
        r['video_url'] = f"/recordings/{r['recording']}" if r['recording'] else None
        thumb_path = r.get('thumb') or ''
        if thumb_path.startswith('violations/'):
            thumb_path = thumb_path[len('violations/'):]
        r['thumb_url'] = url_for('serve_evidence', filepath=thumb_path) if thumb_path else None
    return jsonify(rows)


@app.route('/incidents/<int:inc_id>/dismiss', methods=['POST'])
def dismiss_incident_route(inc_id):
    dismiss_incident(inc_id, True)
    print(f"[incidents] {session.get('username', 'unknown')} dismissed incident #{inc_id}")
    return jsonify({"status": "dismissed", "id": inc_id})


@app.route('/recordings/<path:filepath>')
def serve_recording(filepath):
    return send_from_directory(RECORDINGS_FOLDER, filepath, conditional=True)


@app.route('/evidence/<path:filepath>')
def serve_evidence(filepath):
    return send_from_directory(VIOLATION_FOLDER, filepath, conditional=True)


# ==========================================
# CAMERA & MONITORING CONTROLS
# ==========================================

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    if camera_id not in state.camera_registry:
        return f"Unknown camera_id: {camera_id}", 404
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/cameras', methods=['GET'])
def list_cameras():
    result = []
    for cam_id, entry in list(state.camera_registry.items()):
        result.append({
            "id":     cam_id,
            "label":  entry["label"],
            "type":   entry["type"],
            "source": str(entry["source"]),
            "status": entry["status"],
        })
    return jsonify(result)


@app.route('/cameras/scan', methods=['POST'])
def scan_cameras_route():
    found = scan_local_cameras()
    return jsonify({"found_indices": found})


@app.route('/cameras/add', methods=['POST'])
def add_camera_route():
    data       = request.get_json(force=True, silent=True) or {}
    raw_source = str(data.get("source", "")).strip()
    label      = (data.get("label") or "").strip() or None

    if not raw_source:
        return jsonify({"status": "error", "message": "Source is required"}), 400

    if raw_source.isdigit():
        source   = int(raw_source)
        cam_type = "local"
    elif os.path.isfile(raw_source):
        source   = raw_source
        cam_type = "file"
    elif (raw_source.startswith("http://") or raw_source.startswith("https://")
          or raw_source.startswith("rtsp://")):
        source   = raw_source
        cam_type = "ip"
    else:
        return jsonify({
            "status":  "error",
            "message": (
                "Source must be a device index (0, 1, ...), "
                "a video file path (e.g. C:\\footage.mp4 or /home/user/clip.mp4), "
                "or a URL starting with http://, https://, or rtsp://"
            ),
        }), 400

    cam_id = add_camera(source, cam_type, label)
    return jsonify({"status": "added", "camera_id": cam_id})


@app.route('/cameras/browser/add', methods=['POST'])
def add_browser_camera_route():
    data = request.get_json(silent=True) or {}
    label = (data.get("label") or "").strip() or "This Device"
    cam_id = add_camera("browser", "browser", label)
    return jsonify({"status": "added", "camera_id": cam_id})


@app.route('/cameras/<camera_id>/frame', methods=['POST'])
def browser_camera_frame_route(camera_id):
    entry = state.camera_registry.get(camera_id)
    if entry is None or entry["type"] != "browser":
        return jsonify({"status": "error", "message": "Browser camera not found."}), 404
    if entry["status"] != "active" or not isinstance(entry.get("cap"), BrowserCapture):
        return jsonify({"status": "error", "message": "Browser camera is not active."}), 409
    if request.mimetype != "image/jpeg":
        return jsonify({"status": "error", "message": "Expected an image/jpeg frame."}), 415
    max_frame_bytes = 2 * 1024 * 1024
    if request.content_length is not None and request.content_length > max_frame_bytes:
        return jsonify({"status": "error", "message": "Camera frame is too large."}), 413

    payload = request.stream.read(max_frame_bytes + 1)
    if len(payload) > max_frame_bytes:
        return jsonify({"status": "error", "message": "Camera frame is too large."}), 413
    if not payload:
        return jsonify({"status": "error", "message": "Camera frame is empty."}), 400

    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return jsonify({"status": "error", "message": "Could not decode camera frame."}), 400
    height, width = frame.shape[:2]
    if width > 1280 or height > 720:
        scale = min(1280 / width, 720 / height)
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))

    cap = entry.get("cap")
    if not isinstance(cap, BrowserCapture) or not cap.submit(frame):
        return jsonify({"status": "error", "message": "Browser camera is no longer active."}), 409
    return jsonify({"status": "received"})


@app.route('/cameras/<camera_id>/activate', methods=['POST'])
def activate_camera_route(camera_id):
    ok, msg = activate_camera(camera_id)
    if not ok:
        return jsonify({"status": "error", "message": msg}), 400
    return jsonify({"status": "activated", "camera_id": camera_id})


@app.route('/cameras/<camera_id>/pause', methods=['POST'])
def pause_camera_route(camera_id):
    ok, msg = pause_camera(camera_id)
    if not ok:
        return jsonify({"status": "error", "message": msg}), 400
    return jsonify({"status": "paused", "camera_id": camera_id})


@app.route('/cameras/<camera_id>/remove', methods=['POST'])
def remove_camera_route(camera_id):
    ok, msg = remove_camera(camera_id)
    if not ok:
        return jsonify({"status": "error", "message": msg}), 400
    return jsonify({"status": "removed", "camera_id": camera_id})


@app.route('/get_violations')
def get_violations():
    records = get_recent_violations(state.CURRENT_SESSION, limit=50)
    return jsonify(records)


@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    state.monitoring_active = True
    print(f"[session] monitoring started, rules: {DETECTION_ENABLED}")
    return jsonify({"status": "started"})


@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    state.monitoring_active = False
    return jsonify({"status": "stopped"})


@app.route('/recalibrate', methods=['POST'])
def recalibrate_route():
    for entry in list(state.camera_registry.values()):
        entry["calibration_start"].clear()
        entry["gaze_timers"].clear()
        entry["turn_history"].clear()
        entry["gaze_lost"].clear()
    print(f"[session] {session.get('username', 'unknown')} requested recalibration")
    return jsonify({"status": "recalibrating"})


@app.route('/settings', methods=['GET', 'POST'])
def settings_route():
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        for k in list(DETECTION_ENABLED):
            if k in data:
                DETECTION_ENABLED[k] = bool(data[k])
        print(f"[settings] {session.get('username', 'unknown')} set rules: {DETECTION_ENABLED}")
    return jsonify(DETECTION_ENABLED)


@app.route('/system_stats')
def system_stats():
    try:
        records = get_recent_violations(state.CURRENT_SESSION, limit=100)
    except Exception as e:
        print(f"[DB ERROR in system_stats]: {e}")
        records = []

    phone_count = reach_count = gaze_count = 0
    for r in records:
        v_type = str(r.get('violation_type', '')).upper()
        if "OBJECT" in v_type or "PHONE" in v_type:
            phone_count += 1
        elif "REACH" in v_type:
            reach_count += 1
        elif "GAZE" in v_type:
            gaze_count += 1

    return jsonify({
        "monitoring":       state.monitoring_active,
        "total_violations": len(records),
        "phone":            phone_count,
        "reach":            reach_count,
        "gaze":             gaze_count,
        "rules":            dict(DETECTION_ENABLED),
        "build":            BUILD_ID,
        "session_id":       state.CURRENT_SESSION,
    })


def _shutdown():
    """Finalize open recordings on exit."""
    for entry in list(state.camera_registry.values()):
        entry["stop_event"].set()
    for entry in list(state.camera_registry.values()):
        t = entry.get("thread")
        if t and t.is_alive():
            t.join(timeout=3.0)
        try:
            _finish_recording(entry)
        except Exception as e:
            print(f"[shutdown] could not finalize recording: {e}")


atexit.register(_shutdown)

if __name__ == '__main__':
    app.run(debug=False, threaded=True,
            host=os.environ.get("CERBERUS_HOST", "127.0.0.1"), port=5000)
