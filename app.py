from flask import (Flask, render_template, Response, jsonify, request,
                    send_from_directory, session, redirect, url_for)
import cv2
from ultralytics import YOLO
import os
import re
import time
import shutil
import secrets
import threading
from datetime import datetime
from collections import deque
import math
import numpy as np

# Database helper functions from database.py
from database import init_db, log_violation, get_recent_violations, clear_violations
from recording import (SessionRecorder, IncidentTracker, init_incident_db,
                       list_incidents, clear_incidents, dismiss_incident,
                       cleanup_orphaned_recordings, RECORDINGS_FOLDER)

app = Flask(__name__)

# Initialize SQLite database schema
init_db()
init_incident_db()

# Fix 7: finalize any recordings a previous crash / power loss left half-written,
# so they're playable in /review instead of permanently stuck at "processing".
cleanup_orphaned_recordings()

VIOLATION_FOLDER = os.path.join('static', 'violations')
os.makedirs(VIOLATION_FOLDER, exist_ok=True)

CURRENT_SESSION = "SESSION_ROOM_A1"

# --- SYSTEM STATE & THRESHOLDS ---
monitoring_active = False   # session-wide arm/disarm (the ARM/PAUSE buttons)

# NOTE: all thresholds below are hand-calibrated, not learned. Before demo
# day, run a real test session with people seated at actual exam spacing
# and adjust these against what you observe -- don't trust them untested.
HEAD_TURN_RATIO_THRESHOLD = 0.6   # see the long comment at the calculation
                                    # site -- this is a horizontal nose-
                                    # displacement ratio, not a raw angle
GAZE_DURATION_THRESHOLD = 3.0
SPINE_TILT_THRESHOLD_DEG = 25.0
PROXIMITY_MULTIPLIER = 1.8
INTERACTION_DURATION_THRESHOLD = 3.0
ATTRIBUTION_RADIUS_MULTIPLIER = 4.0
MIN_KEYPOINT_CONF = 0.5
SMOOTHING_WINDOW = 5

# --- ADAPTIVE "NORMAL POSITION" BASELINE ---
# These control the per-student gaze baseline: what counts as "forward"
# for THIS student, which can slowly drift if they hold a new, moderate
# position steadily (e.g. shifted their chair). Also hand-calibrated --
# validate against a real test session before demo day.
CALIBRATION_DURATION = 2.0          # seconds to sample a student's initial "neutral" pose
BASELINE_ADAPT_RATE = 0.02          # how fast the baseline creeps toward a newly-held position
REPOSITION_STABILITY_WINDOW = 10.0  # student must hold a new position this long before baseline shifts
REPOSITION_JUMP_LIMIT = HEAD_TURN_RATIO_THRESHOLD * 0.8
REPOSITION_SETTLE_BAND = HEAD_TURN_RATIO_THRESHOLD * 0.3

# --- SMARTWATCH / WALL-CLOCK FILTER (Fix 1) ---
# YOLO class 74 ("clock") fires on both smartwatches AND wall clocks.
# Two guards are applied before a clock detection is counted as a violation:
#
#   A. Static guard: track each clock bbox centre across frames.  If it has
#      not moved more than STATIC_CLOCK_MOVE_PX pixels in STATIC_CLOCK_SEC
#      seconds it is almost certainly a wall clock -> suppress.  Suppressed
#      detections are drawn as a dim grey box so the operator can confirm
#      the filter is working.
#
#   B. Wrist guard: if at least one student is tracked by pose, a clock
#      detection must be within WRIST_PROXIMITY_PX pixels of a visible wrist
#      keypoint.  If no wrists are visible (hands under the desk) we stay
#      conservative and do NOT suppress.
#
# Cell-phone detections (class 67) bypass both guards entirely.
STATIC_CLOCK_SEC       = 8.0   # suppress after this many seconds without moving
STATIC_CLOCK_RADIUS_PX = 60    # detections within this radius = same object
STATIC_CLOCK_MOVE_PX   = 35    # must move at least this far to reset the static timer
WRIST_PROXIMITY_PX     = 150   # max distance (px) from any wrist keypoint to count as worn

# --- CAMERA-BUMP DETECTION (Fix 5) ---
# If the camera gets knocked, every student's head-turn baseline is now
# measuring the wrong thing, and everyone false-flags for "looking away" at
# once. If a large fraction of currently-visible, already-calibrated
# students all read as turned in the SAME frame, treat it as a probable
# camera disturbance rather than a coordinated multi-student violation:
# drop their calibration so the existing CALIBRATING flow silently
# reseeds each baseline, instead of raising a wall of false alerts.
CAMERA_BUMP_RATIO        = 0.6   # fraction of visible students turned at once
CAMERA_BUMP_MIN_STUDENTS = 3     # don't fire this on 1-2 people (e.g. solo testing)
CAMERA_BUMP_COOLDOWN_SEC = 5.0   # don't re-trigger every frame while still settling

# --- TRACK-AGE GATE (Fix 6) ---
# A track that has only existed for a moment is more likely to be a
# passing proctor, a brief occlusion-driven ID swap, or a tracking glitch
# than a genuine student behavior. Object attribution and interaction
# checks ignore tracks younger than this. (Gaze detection is left as-is:
# it already requires CALIBRATION_DURATION + GAZE_DURATION_THRESHOLD of
# continuous tracking before it can fire, which is a stronger gate than
# this on its own.)
TRACK_MIN_AGE_SEC = 1.0

# ==========================================
# DYNAMIC CAMERA REGISTRY
# ==========================================
# Architecture (post background-thread refactor, Fix 3):
#   activate_camera()  -> opens cap, loads models, starts _detection_loop()
#                         as a daemon thread for this camera.
#   _detection_loop()  -> reads frames, runs YOLO, records, logs incidents.
#                         Stores latest annotated JPEG in entry["latest_frame"].
#   generate_frames()  -> trivial MJPEG streamer: just yields latest_frame.
#                         No YOLO, no recording.  Multiple browser tabs can
#                         connect simultaneously without fighting over frames
#                         or scrambling track IDs.
#   pause_camera()     -> sets stop_event, joins thread, releases cap.
#   remove_camera()    -> same, then drops entry from registry.
#
# registry_lock guards dict structure changes (add/remove entries).
# Each entry has its own lock for the brief cap.read() critical section.
camera_registry = {}
registry_lock = threading.Lock()
_next_camera_num = 1

# COCO class 67 = cell phone, 74 = clock (smartwatch proxy)
PROHIBITED_CLASSES = {67: "CELL PHONE", 74: "SMARTWATCH"}


def _generate_camera_id():
    global _next_camera_num
    with registry_lock:
        cam_id = f"cam{_next_camera_num}"
        _next_camera_num += 1
    return cam_id


def _finish_recording(entry):
    """Closes the open recording and flushes open incidents. Safe to call repeatedly."""
    entry["incidents"].finish()
    entry["recorder"].close()


def add_camera(source, cam_type, label=None):
    """Registers a camera as INACTIVE.  Does not open the device or load
    models -- that happens on activate_camera(), so registering many cameras
    stays cheap.

    cam_type: "local" (device-index int) | "ip" (URL str) | "file" (path str)
    """
    cam_id = _generate_camera_id()
    entry = {
        "source":   source,
        "type":     cam_type,          # "local" | "ip" | "file"
        "label":    label or cam_id,
        "status":   "inactive",        # "inactive" | "active" | "paused"
        "cap":      None,
        "pose_model":   None,
        "object_model": None,
        "lock":     threading.Lock(),
        "gaze_timers":        {},
        "turn_history":       {},
        "interaction_timers": {},
        "recorder":  SessionRecorder(CURRENT_SESSION, cam_id),
        "incidents": IncidentTracker(),
        # --- adaptive baseline state, per track_id ---
        "baseline_ratio":    {},
        "calibration_start": {},
        "stable_since":      {},
        "stable_anchor":     {},
        # --- background detection thread (Fix 3) ---
        "latest_frame": None,           # bytes: latest annotated JPEG for the HTTP stream
        "stop_event":   threading.Event(),
        "thread":       None,
        # --- smartwatch static-position tracker (Fix 1) ---
        "static_objects": {},           # {int_id: {center, static_since, last_seen}}
        # --- camera-bump detection (Fix 5) ---
        "last_bump_time": 0.0,
        # --- track-age gate (Fix 6) ---
        "track_first_seen": {},         # track_id -> time first seen
    }
    with registry_lock:
        camera_registry[cam_id] = entry
    return cam_id


def activate_camera(cam_id):
    """Opens the capture device (if not already open), loads models (if not
    cached), then starts the per-camera detection thread.

    Paused cameras keep their models cached, so resuming skips the slow
    model-load step and only reopens the device + relaunches the thread.
    """
    entry = camera_registry.get(cam_id)
    if entry is None:
        return False, "Camera not found"

    # Guard: do not spawn a second thread if one is already running.
    if entry.get("thread") and entry["thread"].is_alive():
        return True, "already active"

    with entry["lock"]:
        if entry["cap"] is None:
            cap = cv2.VideoCapture(entry["source"])
            if not cap.isOpened():
                return False, f"Could not open camera source: {entry['source']}"
            # Minimise the internal buffer for live cameras so the feed stays
            # current.  Video files are sequential -- leave their buffer alone.
            if entry["type"] != "file":
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            entry["cap"] = cap

        if entry["pose_model"] is None:
            entry["pose_model"] = YOLO('yolo11n-pose.pt')
        if entry["object_model"] is None:
            entry["object_model"] = YOLO('yolo11n.pt')

        entry["stop_event"].clear()
        t = threading.Thread(target=_detection_loop, args=(cam_id,),
                             daemon=True, name=f"detect-{cam_id}")
        entry["thread"] = t
        entry["status"] = "active"
        t.start()

    return True, "activated"


def pause_camera(cam_id):
    """Signals the detection thread to stop, waits for it to exit (recording
    is finalised in the thread's finally-block), then releases the capture
    device.  Models stay cached in memory so a later resume is fast.
    """
    entry = camera_registry.get(cam_id)
    if entry is None:
        return False, "Camera not found"

    # Signal, then join with a generous timeout for any in-flight YOLO call.
    entry["stop_event"].set()
    t = entry.get("thread")
    if t and t.is_alive():
        t.join(timeout=5.0)
    entry["thread"] = None

    with entry["lock"]:
        if entry["cap"] is not None:
            entry["cap"].release()
            entry["cap"] = None
        entry["status"] = "paused"
        entry["latest_frame"] = None
        entry["gaze_timers"].clear()
        entry["turn_history"].clear()
        entry["interaction_timers"].clear()
        entry["baseline_ratio"].clear()
        entry["calibration_start"].clear()
        entry["stable_since"].clear()
        entry["stable_anchor"].clear()
        entry["static_objects"].clear()
        entry["track_first_seen"].clear()

    return True, "paused"


def remove_camera(cam_id):
    """Stops the detection thread, releases the device, finalises the
    recording, and drops the camera from the registry.
    """
    entry = camera_registry.get(cam_id)
    if entry is None:
        return False, "Camera not found"

    entry["stop_event"].set()
    t = entry.get("thread")
    if t and t.is_alive():
        t.join(timeout=5.0)

    with entry["lock"]:
        if entry["cap"] is not None:
            entry["cap"].release()

    _finish_recording(entry)   # thread finally-block also calls this; second call is a no-op

    with registry_lock:
        camera_registry.pop(cam_id, None)

    return True, "removed"


def scan_local_cameras(max_index=4):
    """Probes local device indices for available cameras not already
    registered.  Blocking (a few seconds) -- triggered by an explicit
    'Scan' button, not polled automatically.
    """
    registered_local_sources = {
        entry["source"] for entry in camera_registry.values()
        if entry["type"] == "local"
    }
    found = []
    for i in range(max_index):
        if i in registered_local_sources:
            continue
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


# --- Pre-register one default local camera so the dashboard is not empty on
# first load. It does NOT auto-start -- click "Use" on it. ---
add_camera(0, "local", label="cam1")


def calculate_angle(p1, p2):
    """Calculates the tilt angle of a 2D vector relative to the vertical line."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = math.atan2(abs(dx), abs(dy))
    return math.degrees(rads)


def make_status_frame(label, message, sub_message=""):
    """Placeholder frame shown for an inactive, paused, or disconnected camera."""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, f"'{label}'", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    cv2.putText(placeholder, message, (40, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if sub_message:
        cv2.putText(placeholder, sub_message, (40, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)
    return placeholder


# ==========================================
# SMARTWATCH STATIC-POSITION TRACKER (Fix 1, Guard A)
# ==========================================
def _check_static_clocks(entry, centers, now):
    """Update the per-camera tracker for clock bbox centres and return the
    set of indices (into `centers`) that are suppressed because the object
    has sat in the same spot for >= STATIC_CLOCK_SEC (almost certainly a
    wall clock).

    State lives in entry['static_objects']:
        { int_id: {"center": (x,y), "static_since": float, "last_seen": float} }
    The special key '__next' holds the running ID counter.
    """
    tracker  = entry["static_objects"]
    NEXT_KEY = "__next"
    next_id  = tracker.get(NEXT_KEY, 0)

    # Expire clusters not seen recently.
    stale = [k for k in list(tracker)
             if k != NEXT_KEY and now - tracker[k]["last_seen"] > 3.0]
    for k in stale:
        del tracker[k]

    suppressed = set()
    for idx, center in enumerate(centers):
        best_id, best_dist = None, float("inf")
        for tid, td in tracker.items():
            if tid == NEXT_KEY:
                continue
            d = math.hypot(center[0] - td["center"][0], center[1] - td["center"][1])
            if d < STATIC_CLOCK_RADIUS_PX and d < best_dist:
                best_id, best_dist = tid, d

        if best_id is not None:
            td = tracker[best_id]
            td["last_seen"] = now
            if best_dist > STATIC_CLOCK_MOVE_PX:
                # Object shifted -- reset the static timer.
                td["static_since"] = now
                td["center"]       = center
            elif now - td["static_since"] >= STATIC_CLOCK_SEC:
                suppressed.add(idx)
        else:
            tracker[next_id] = {"center": center, "static_since": now, "last_seen": now}
            next_id += 1
            tracker[NEXT_KEY] = next_id

    return suppressed


# ==========================================
# BACKGROUND DETECTION THREAD (Fix 3)
# ==========================================
def _detection_loop(camera_id):
    """Per-camera daemon thread.

    Owns ALL frame reading, YOLO inference, HUD drawing, session recording,
    and incident logging.  Stores the latest annotated JPEG in
    entry['latest_frame'] so generate_frames() can serve it to any number
    of browser tabs with zero YOLO work and zero interference between them.

    Lifecycle:
        activate_camera() -> clears stop_event, starts this thread
        pause_camera()    -> sets stop_event -> thread exits loop
                             -> finally-block finalises recording
        remove_camera()   -> same, then drops entry from registry

    Video-file cameras (cam_type == "file") loop back to frame 0 on EOF for
    continuous demo replay / threshold tuning without needing a live camera.

    Live cameras (local/ip) drain their internal buffer each cycle -- this
    is the Fix 4 live-feed lag fix.
    """
    entry = camera_registry.get(camera_id)
    if entry is None:
        return

    stop_event = entry["stop_event"]

    try:
        while not stop_event.is_set():

            # ==========================================
            # 0. READ FRAME
            # ==========================================
            with entry["lock"]:
                cap = entry["cap"]
                if cap is None:
                    time.sleep(0.05)
                    continue

                # Fix 4 -- live cameras: drain buffered frames so we always
                # process the NEWEST one.  Video files must not skip frames.
                if entry["type"] != "file":
                    for _ in range(4):
                        if stop_event.is_set():
                            break
                        cap.grab()

                success, frame = cap.read()

            if not success:
                if entry["type"] == "file":
                    # End of file -- loop back to the start for demo replay.
                    with entry["lock"]:
                        cap = entry["cap"]
                        if cap is not None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    continue
                else:
                    # Live camera disconnected -- attempt reconnect.
                    with entry["lock"]:
                        if entry["cap"] is not None:
                            entry["cap"].release()
                        new_cap = cv2.VideoCapture(entry["source"])
                        new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        entry["cap"] = new_cap
                    placeholder = make_status_frame(
                        entry["label"], "DISCONNECTED", "Attempting to reconnect...")
                    ret, buf = cv2.imencode('.jpg', placeholder)
                    if ret:
                        entry["latest_frame"] = buf.tobytes()
                    time.sleep(1.0)
                    continue

            h, w, _ = frame.shape
            current_time = time.time()

            gaze_timers      = entry["gaze_timers"]
            turn_history     = entry["turn_history"]
            pair_timers      = entry["interaction_timers"]
            baseline_ratio   = entry["baseline_ratio"]
            calib_start      = entry["calibration_start"]
            stable_since     = entry["stable_since"]
            stable_anchor    = entry["stable_anchor"]
            track_first_seen = entry["track_first_seen"]

            cv2.putText(frame, f"CAM: {entry['label']}", (20, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if not monitoring_active:
                gaze_timers.clear()
                turn_history.clear()
                pair_timers.clear()
                baseline_ratio.clear()
                calib_start.clear()
                stable_since.clear()
                stable_anchor.clear()
                track_first_seen.clear()
                _finish_recording(entry)
                cv2.putText(frame, "MONITORING PAUSED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                ret, buf = cv2.imencode('.jpg', frame)
                if ret:
                    entry["latest_frame"] = buf.tobytes()
                time.sleep(0.05)
                continue

            object_violation      = False
            interaction_violation = False
            gaze_trigger          = False
            violation_reason      = ""
            violation_confidence  = 0.0
            violating_track_id    = None

            # ==========================================
            # 1. PROHIBITED OBJECT DETECTION (YOLO)
            # ==========================================
            # Collect detections only; drawing is deferred to section 2.5
            # so the smartwatch filter can annotate suppressed detections
            # differently from live ones.
            object_detections = []
            try:
                obj_results = entry["object_model"](frame, stream=True, verbose=False)
                for r in obj_results:
                    for box in r.boxes:
                        cls_id     = int(box.cls[0])
                        confidence = float(box.conf[0])
                        if cls_id in PROHIBITED_CLASSES and confidence > 0.45:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            object_detections.append({
                                "name":       PROHIBITED_CLASSES[cls_id],
                                "confidence": confidence,
                                "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
                                "bbox":       (x1, y1, x2, y2),
                            })
            except Exception as e:
                print(f"[{camera_id}] Object detection error: {e}")

            # ==========================================
            # 2. SKELETAL POSE ESTIMATION + TRACKING
            # ==========================================
            students        = []
            active_ids      = []
            wrist_positions = []   # (x, y) of every visible wrist -- for clock guard B

            try:
                pose_results = entry["pose_model"].track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    stream=True,
                    verbose=False
                )

                for r in pose_results:
                    if r.keypoints is None or len(r.keypoints) == 0:
                        continue

                    kpts  = r.keypoints.xy.cpu().numpy()
                    confs = (r.keypoints.conf.cpu().numpy()
                             if r.keypoints.conf is not None else None)

                    if r.boxes is not None and r.boxes.id is not None:
                        track_ids = r.boxes.id.int().cpu().tolist()
                    else:
                        track_ids = [None] * len(kpts)

                    for idx, person in enumerate(kpts):
                        track_id = track_ids[idx] if idx < len(track_ids) else None
                        if track_id is None:
                            continue

                        nose                   = person[0]
                        l_shoulder, r_shoulder = person[5],  person[6]
                        l_hip,      r_hip      = person[11], person[12]
                        l_wrist,    r_wrist    = person[9],  person[10]

                        if confs is not None:
                            c_nose   = confs[idx][0]
                            c_lsh    = confs[idx][5]
                            c_rsh    = confs[idx][6]
                            c_lhip   = confs[idx][11]
                            c_rhip   = confs[idx][12]
                            c_lwrist = confs[idx][9]
                            c_rwrist = confs[idx][10]
                            if c_lsh < MIN_KEYPOINT_CONF or c_rsh < MIN_KEYPOINT_CONF:
                                continue
                        else:
                            c_nose = c_lhip = c_rhip = 1.0
                            c_lwrist = c_rwrist = 1.0

                        nose_confident = c_nose  >= MIN_KEYPOINT_CONF
                        hips_confident = (c_lhip >= MIN_KEYPOINT_CONF and
                                          c_rhip >= MIN_KEYPOINT_CONF)

                        # Fix 6: remember the first time this track_id was seen,
                        # so downstream checks can require a minimum track age
                        # before trusting it (filters passing proctors / brief
                        # ID-swap glitches).
                        if track_id not in track_first_seen:
                            track_first_seen[track_id] = current_time

                        # Collect visible wrists for clock guard B.
                        if c_lwrist >= MIN_KEYPOINT_CONF and l_wrist[0] > 0:
                            wrist_positions.append((int(l_wrist[0]), int(l_wrist[1])))
                        if c_rwrist >= MIN_KEYPOINT_CONF and r_wrist[0] > 0:
                            wrist_positions.append((int(r_wrist[0]), int(r_wrist[1])))

                        shoulder_center = (int((l_shoulder[0] + r_shoulder[0]) / 2),
                                           int((l_shoulder[1] + r_shoulder[1]) / 2))
                        shoulder_width  = max(20.0, abs(r_shoulder[0] - l_shoulder[0]))

                        spine_tilt = 0.0
                        if hips_confident and l_hip[0] > 0 and r_hip[0] > 0:
                            hip_center = ((l_hip[0] + r_hip[0]) / 2,
                                          (l_hip[1] + r_hip[1]) / 2)
                            spine_tilt = calculate_angle(hip_center, shoulder_center)

                        students.append({
                            "track_id":        track_id,
                            "shoulder_center": shoulder_center,
                            "shoulder_width":  shoulder_width,
                            "spine_tilt":      spine_tilt,
                            "nose":            nose,
                            "person_turn":     "FORWARD",   # filled in below
                        })

                        cv2.circle(frame, (int(l_shoulder[0]), int(l_shoulder[1])), 4, (0, 255, 255), -1)
                        cv2.circle(frame, (int(r_shoulder[0]), int(r_shoulder[1])), 4, (0, 255, 255), -1)
                        cv2.line(frame,
                                 (int(l_shoulder[0]), int(l_shoulder[1])),
                                 (int(r_shoulder[0]), int(r_shoulder[1])),
                                 (0, 255, 255), 2)
                        cv2.putText(frame, f"ID {track_id}",
                                    (shoulder_center[0] - 20, shoulder_center[1] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                        # --- HEAD YAW: HORIZONTAL-DISPLACEMENT RATIO, WITH
                        #     ADAPTIVE PER-STUDENT "NORMAL POSITION" BASELINE ---
                        # Measures the sideways nose offset from the shoulder-
                        # line midpoint, in a coordinate frame rotated to
                        # account for the student's seat/torso angle (ignores
                        # pitch, so looking down at a paper isn't conflated
                        # with turning sideways). That raw ratio is then
                        # compared against THIS student's own learned
                        # "forward" baseline rather than a universal zero,
                        # so students seated off-centre aren't penalised.
                        person_turn    = "FORWARD"
                        smoothed_ratio = 0.0

                        if (nose_confident and nose[0] > 0
                                and l_shoulder[0] > 0 and r_shoulder[0] > 0):
                            d_x_sh = r_shoulder[0] - l_shoulder[0]
                            d_y_sh = r_shoulder[1] - l_shoulder[1]
                            shoulder_angle_rad = math.atan2(d_y_sh, d_x_sh)

                            mid_sh_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
                            mid_sh_y = (l_shoulder[1] + r_shoulder[1]) / 2.0

                            d_x_head = nose[0] - mid_sh_x
                            d_y_head = nose[1] - mid_sh_y

                            cos_t = math.cos(-shoulder_angle_rad)
                            sin_t = math.sin(-shoulder_angle_rad)
                            sideways_offset = d_x_head * cos_t - d_y_head * sin_t
                            raw_ratio       = sideways_offset / (shoulder_width / 2.0)

                            if track_id not in calib_start:
                                calib_start[track_id]    = current_time
                                baseline_ratio[track_id] = raw_ratio
                                stable_since[track_id]   = current_time
                                stable_anchor[track_id]  = raw_ratio

                            if current_time - calib_start[track_id] < CALIBRATION_DURATION:
                                # Still in the initial calibration window -- snap
                                # the baseline and don't flag anything yet.
                                baseline_ratio[track_id] = raw_ratio
                                person_turn = "CALIBRATING"
                            else:
                                baseline       = baseline_ratio[track_id]
                                adjusted_ratio = raw_ratio - baseline

                                if track_id not in turn_history:
                                    turn_history[track_id] = deque(maxlen=SMOOTHING_WINDOW)
                                turn_history[track_id].append(adjusted_ratio)
                                smoothed_ratio = (sum(turn_history[track_id])
                                                  / len(turn_history[track_id]))

                                if smoothed_ratio < -HEAD_TURN_RATIO_THRESHOLD:
                                    person_turn = "LOOKING LEFT"
                                elif smoothed_ratio > HEAD_TURN_RATIO_THRESHOLD:
                                    person_turn = "LOOKING RIGHT"

                                # --- Slow repositioning: let baseline drift toward
                                # a new pose only if it is a moderate, steadily-held
                                # shift and the student is NOT currently mid-violation.
                                currently_flagged = track_id in gaze_timers

                                if currently_flagged or abs(adjusted_ratio) > REPOSITION_JUMP_LIMIT:
                                    stable_since[track_id]  = current_time
                                    stable_anchor[track_id] = raw_ratio
                                else:
                                    if abs(raw_ratio - stable_anchor[track_id]) > REPOSITION_SETTLE_BAND:
                                        stable_since[track_id]  = current_time
                                        stable_anchor[track_id] = raw_ratio
                                    else:
                                        held = current_time - stable_since[track_id]
                                        if held >= REPOSITION_STABILITY_WINDOW:
                                            baseline_ratio[track_id] += BASELINE_ADAPT_RATE * (
                                                raw_ratio - baseline_ratio[track_id]
                                            )

                        students[-1]["person_turn"] = person_turn
                        active_ids.append(track_id)

                        if person_turn not in ("FORWARD", "CALIBRATING"):
                            if track_id not in gaze_timers:
                                gaze_timers[track_id] = current_time
                            elapsed = current_time - gaze_timers[track_id]

                            if elapsed >= GAZE_DURATION_THRESHOLD:
                                gaze_trigger = True
                                if not violation_reason:
                                    violation_reason     = "PROLONGED_SIDE_GAZE"
                                    violation_confidence = 0.95
                                    violating_track_id   = track_id
                                cv2.putText(frame, person_turn,
                                            (shoulder_center[0] - 50,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                time_rem = GAZE_DURATION_THRESHOLD - elapsed
                                cv2.putText(frame, f"{person_turn} ({time_rem:.1f}s)",
                                            (shoulder_center[0] - 60,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        else:
                            gaze_timers.pop(track_id, None)
                            turn_history.pop(track_id, None)
                            if person_turn == "CALIBRATING":
                                cv2.putText(frame, "CALIBRATING...",
                                            (shoulder_center[0] - 55,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            except Exception as e:
                print(f"[{camera_id}] Pose estimation error: {e}")
                students   = []
                active_ids = []

            entry["gaze_timers"]       = {k: v for k, v in gaze_timers.items()    if k in active_ids}
            entry["turn_history"]      = {k: v for k, v in turn_history.items()   if k in active_ids}
            entry["baseline_ratio"]    = {k: v for k, v in baseline_ratio.items() if k in active_ids}
            entry["calibration_start"] = {k: v for k, v in calib_start.items()    if k in active_ids}
            entry["stable_since"]      = {k: v for k, v in stable_since.items()   if k in active_ids}
            entry["stable_anchor"]     = {k: v for k, v in stable_anchor.items()  if k in active_ids}
            entry["track_first_seen"]  = {k: v for k, v in track_first_seen.items() if k in active_ids}
            track_first_seen = entry["track_first_seen"]

            established_ids = {tid for tid, seen in track_first_seen.items()
                               if current_time - seen >= TRACK_MIN_AGE_SEC}

            # ==========================================
            # 2.6 CAMERA-BUMP DETECTION (Fix 5)
            # ==========================================
            # If most already-calibrated, visible students all read as
            # turned in this same frame, it's more likely the camera moved
            # than that a majority of the class simultaneously cheated.
            # Drop their calibration so each baseline silently reseeds via
            # the existing CALIBRATING flow, instead of raising a wall of
            # false gaze alerts while everyone "settles" against a baseline
            # that no longer matches the camera's new angle.
            scored = [s for s in students if s["person_turn"] != "CALIBRATING"]
            if len(scored) >= CAMERA_BUMP_MIN_STUDENTS:
                turned = sum(1 for s in scored if s["person_turn"] != "FORWARD")
                if (turned / len(scored) >= CAMERA_BUMP_RATIO
                        and current_time - entry["last_bump_time"] >= CAMERA_BUMP_COOLDOWN_SEC):
                    print(f"[{camera_id}] Possible camera disturbance: {turned}/{len(scored)} "
                          f"students turned at once. Recalibrating all baselines.")
                    entry["last_bump_time"] = current_time
                    entry["calibration_start"].clear()
                    entry["gaze_timers"].clear()
                    entry["turn_history"].clear()

            # ==========================================
            # 2.5. SMARTWATCH FILTER (Fix 1)
            # ==========================================
            # Isolate clock detections, run both guards, then reassemble.
            clock_dets    = [(i, d) for i, d in enumerate(object_detections)
                             if d["name"] == "SMARTWATCH"]
            clock_centers = [d["center"] for _, d in clock_dets]
            suppressed    = _check_static_clocks(entry, clock_centers, current_time)

            accepted_detections = []
            for i, det in enumerate(object_detections):
                x1, y1, x2, y2 = det["bbox"]

                if det["name"] != "SMARTWATCH":
                    # Cell phones bypass both guards -- draw immediately.
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"CELL PHONE ({det['confidence']*100:.0f}%)",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    accepted_detections.append(det)
                    continue

                # Find this detection's position within clock_dets.
                ck_idx = next(j for j, (oi, _) in enumerate(clock_dets) if oi == i)

                # Guard A: static position (wall clock).
                if ck_idx in suppressed:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
                    cv2.putText(frame, "STATIC CLOCK (suppressed)",
                                (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)
                    continue

                # Guard B: wrist proximity (only when students are visible).
                if students and wrist_positions:
                    near_wrist = any(
                        math.hypot(det["center"][0] - wx,
                                   det["center"][1] - wy) < WRIST_PROXIMITY_PX
                        for wx, wy in wrist_positions
                    )
                    if not near_wrist:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
                        cv2.putText(frame, "CLOCK: no wrist nearby (suppressed)",
                                    (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)
                        continue

                # Passed both guards -- treat as a real smartwatch alert.
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"SMARTWATCH ({det['confidence']*100:.0f}%)",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                accepted_detections.append(det)

            object_detections = accepted_detections

            # ==========================================
            # 3. ATTRIBUTE OBJECT DETECTIONS TO A STUDENT
            # ==========================================
            # Best-effort nearest-shoulder match, NOT a verified possession
            # check -- a phone on a neighbouring desk close to another student
            # can still be misattributed. Known limitation; state it if asked.
            # Fix 6: only attribute to a track that's been established for
            # TRACK_MIN_AGE_SEC, so a brand-new (possibly-swapped) track ID
            # can't immediately absorb a violation.
            if object_detections:
                object_violation = True
                best_det = max(object_detections, key=lambda d: d["confidence"])
                violation_reason     = f"OBJECT_{best_det['name'].replace(' ', '_')}"
                violation_confidence = best_det["confidence"]

                nearest_student = None
                nearest_dist    = float("inf")
                for s in students:
                    if s["track_id"] not in established_ids:
                        continue
                    d = math.hypot(s["shoulder_center"][0] - best_det["center"][0],
                                   s["shoulder_center"][1] - best_det["center"][1])
                    radius = s["shoulder_width"] * ATTRIBUTION_RADIUS_MULTIPLIER
                    if d < radius and d < nearest_dist:
                        nearest_dist    = d
                        nearest_student = s
                violating_track_id = nearest_student["track_id"] if nearest_student else None

            # ==========================================
            # 4. INTERACTION & DESK LEANING DETECTION (DURATION-GATED)
            # ==========================================
            # Fix 6: only established tracks participate, so a proctor
            # walking through the frame (or a momentary ID swap) can't pair
            # up with a real student and start an interaction timer.
            interaction_candidates = [s for s in students if s["track_id"] in established_ids]
            active_pairs = []
            num_students = len(interaction_candidates)
            for i in range(num_students):
                for j in range(i + 1, num_students):
                    s1, s2 = interaction_candidates[i], interaction_candidates[j]

                    dist           = math.hypot(
                        s1["shoulder_center"][0] - s2["shoulder_center"][0],
                        s1["shoulder_center"][1] - s2["shoulder_center"][1])
                    avg_shoulder_w  = (s1["shoulder_width"] + s2["shoulder_width"]) / 2.0
                    proximity_limit = avg_shoulder_w * PROXIMITY_MULTIPLIER
                    excessive_lean  = (s1["spine_tilt"] > SPINE_TILT_THRESHOLD_DEG or
                                       s2["spine_tilt"] > SPINE_TILT_THRESHOLD_DEG)
                    is_close = (dist < proximity_limit or
                                (dist < proximity_limit * 1.3 and excessive_lean))

                    pair_key = frozenset({s1["track_id"], s2["track_id"]})

                    if is_close:
                        active_pairs.append(pair_key)
                        if pair_key not in pair_timers:
                            pair_timers[pair_key] = current_time
                        elapsed = current_time - pair_timers[pair_key]

                        if elapsed >= INTERACTION_DURATION_THRESHOLD:
                            interaction_violation = True
                            if not violation_reason:
                                violation_reason     = "STUDENT_INTERACTION"
                                violation_confidence = 0.90
                                violating_track_id   = s1["track_id"]

                            cv2.line(frame, s1["shoulder_center"], s2["shoulder_center"],
                                     (0, 0, 255), 3)
                            mid_pt = (
                                (s1["shoulder_center"][0] + s2["shoulder_center"][0]) // 2,
                                (s1["shoulder_center"][1] + s2["shoulder_center"][1]) // 2,
                            )
                            cv2.putText(frame, "INTERACTION",
                                        (mid_pt[0] - 45, mid_pt[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        pair_timers.pop(pair_key, None)

            entry["interaction_timers"] = {k: v for k, v in pair_timers.items()
                                           if k in active_pairs}

            # ==========================================
            # 5. HUD / DASHBOARD STATUS OVERLAY
            # ==========================================
            status_color = (0, 255, 0)
            status_text  = "STATUS: CLEAR"

            if object_violation:
                status_text  = f"ALERT: {violation_reason}!"
                status_color = (0, 0, 255)
            elif interaction_violation:
                status_text  = "ALERT: STUDENTS INTERACTING!"
                status_color = (0, 0, 255)
            elif gaze_trigger:
                status_text  = "ALERT: PROLONGED SIDE GAZE!"
                status_color = (0, 0, 255)
            elif len(entry["gaze_timers"]) > 0:
                status_text  = "WARNING: SIDE GAZE TRACKED"
                status_color = (0, 255, 255)

            cv2.putText(frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # ==========================================
            # 6. SESSION RECORDING + INCIDENT LOGGING
            # ==========================================
            entry["recorder"].write(frame)

            is_violating = object_violation or interaction_violation or gaze_trigger

            if is_violating:
                inc, is_new = entry["incidents"].observe(
                    entry["recorder"], frame, violation_reason, violating_track_id,
                    violation_confidence, CURRENT_SESSION, camera_id,
                    thumb_dir=os.path.join(VIOLATION_FOLDER, CURRENT_SESSION, camera_id),
                    thumb_web_prefix=f"violations/{CURRENT_SESSION}/{camera_id}")
                if is_new:
                    log_violation(CURRENT_SESSION, violation_reason, violation_confidence,
                                  inc["thumb_web"], student_track_id=violating_track_id,
                                  camera_id=camera_id)
                    print(f"[{camera_id}] NEW INCIDENT {violation_reason} "
                          f"(track_id={violating_track_id}, t={inc['start']:.1f}s)")

            # Store the annotated frame for the HTTP streamer.
            ret, buf = cv2.imencode('.jpg', frame)
            if ret:
                entry["latest_frame"] = buf.tobytes()

    finally:
        # Always finalise the recording when this thread exits -- whether from
        # a normal stop_event signal, an unhandled exception, or app shutdown.
        _finish_recording(entry)


# ==========================================
# HTTP FRAME STREAMING (Fix 3)
# ==========================================
def generate_frames(camera_id):
    """Serves the latest annotated JPEG produced by _detection_loop.

    No YOLO work happens here.  Multiple browser tabs can connect
    simultaneously without interfering with detection or with each other --
    they all read the same entry['latest_frame'] byte string.
    """
    while True:
        entry = camera_registry.get(camera_id)
        if entry is None:
            break   # camera was removed; end the generator gracefully

        if entry["status"] != "active":
            msg = "PAUSED" if entry["status"] == "paused" else "NOT IN USE"
            placeholder = make_status_frame(
                entry["label"], msg, "Click 'Use' on the dashboard")
            ret, buf = cv2.imencode('.jpg', placeholder)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.5)
            continue

        frame_bytes = entry.get("latest_frame")
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)   # ~30 fps cap regardless of detection speed


# ==========================================
# ACCESS CONTROL (Fix 8)
# ==========================================
# The app used to bind to 0.0.0.0 with no auth -- anyone on the classroom
# Wi-Fi could open the dashboard, watch every camera, or hit Reset Session.
# A single shared password is enough for a class demo; it isn't meant to
# replace per-user accounts. Set CERBERUS_PASSWORD yourself before running
# for a real exam -- otherwise a random one is generated and printed to the
# console each time the app starts (so it changes on every restart).
APP_PASSWORD = os.environ.get("CERBERUS_PASSWORD")
if not APP_PASSWORD:
    APP_PASSWORD = secrets.token_urlsafe(9)
    print("=" * 64)
    print(f"  CERBERUS password for this run: {APP_PASSWORD}")
    print("  (set the CERBERUS_PASSWORD environment variable to fix it)")
    print("=" * 64)
app.secret_key = secrets.token_hex(32)   # session cookies invalidate on restart -- fine for this use

PUBLIC_ENDPOINTS = {"login"}


@app.before_request
def _require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return
    if not session.get("authed"):
        return redirect(url_for("login"))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == APP_PASSWORD:
            session['authed'] = True
            return redirect(url_for('index'))
        error = "Incorrect password."
    return render_template('login.html', error=error)


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('authed', None)
    return redirect(url_for('login'))


# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    if camera_id not in camera_registry:
        return f"Unknown camera_id: {camera_id}", 404
    return Response(generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ==========================================
# CAMERA MANAGEMENT ROUTES
# ==========================================

@app.route('/cameras', methods=['GET'])
def list_cameras():
    result = []
    for cam_id, entry in list(camera_registry.items()):
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

    # --- Determine source type ---

    if raw_source.isdigit():
        # Local webcam device index.
        source   = int(raw_source)
        cam_type = "local"

    elif os.path.isfile(raw_source):
        # Fix 2: video file path.  OpenCV reads files the same way it reads
        # live cameras, so the detection loop needs no further changes.
        # On EOF the loop rewinds to frame 0 for continuous demo replay.
        source   = raw_source
        cam_type = "file"

    elif (raw_source.startswith("http://") or raw_source.startswith("https://")
          or raw_source.startswith("rtsp://")):
        # IP camera / RTSP stream URL.
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


# ==========================================
# SESSION / MONITORING ROUTES
# ==========================================

@app.route('/get_violations')
def get_violations():
    records = get_recent_violations(CURRENT_SESSION, limit=50)
    return jsonify(records)


@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global monitoring_active
    monitoring_active = True
    return jsonify({"status": "started"})


@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    return jsonify({"status": "stopped"})


# Fix 9: Reset Session permanently deletes every recording, thumbnail, and DB
# row for the session. A single mouse-click confirm() is too easy to fire by
# accident, so the frontend must now prompt the operator to TYPE a fixed
# phrase, and the backend independently checks it matches -- a stray click
# with no typed confirmation is always rejected.
RESET_CONFIRM_PHRASE = "DELETE ALL DATA"


@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Deletes all logged violations, incidents, recordings, and evidence
    photos for the current session, and clears in-memory tracking state for
    every registered camera. Requires monitoring to be paused first, and
    requires the exact confirmation phrase in the request body."""
    global monitoring_active

    if monitoring_active:
        return jsonify({
            "status":  "error",
            "message": "Stop monitoring before resetting.",
        }), 400

    data = request.get_json(silent=True) or {}
    if data.get("confirm") != RESET_CONFIRM_PHRASE:
        return jsonify({
            "status":  "error",
            "message": f'Type "{RESET_CONFIRM_PHRASE}" to confirm this cannot be undone.',
        }), 400

    for _entry in list(camera_registry.values()):
        _finish_recording(_entry)

    clear_violations(CURRENT_SESSION)
    clear_incidents(CURRENT_SESSION)
    shutil.rmtree(os.path.join(RECORDINGS_FOLDER, CURRENT_SESSION), ignore_errors=True)

    session_dir = os.path.join(VIOLATION_FOLDER, CURRENT_SESSION)
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir)

    for cam_id, entry in list(camera_registry.items()):
        os.makedirs(os.path.join(session_dir, cam_id), exist_ok=True)
        entry["gaze_timers"].clear()
        entry["turn_history"].clear()
        entry["interaction_timers"].clear()
        entry["baseline_ratio"].clear()
        entry["calibration_start"].clear()
        entry["stable_since"].clear()
        entry["stable_anchor"].clear()
        entry["static_objects"].clear()
        entry["track_first_seen"].clear()

    return jsonify({"status": "reset", "session_id": CURRENT_SESSION})


# ==========================================
# PER-EXAM SESSIONS (Fix 10)
# ==========================================
# CURRENT_SESSION used to be a hardcoded constant, so two exams run the same
# day landed in the same recordings/DB rows. This lets the operator name a
# new session between exams; each camera gets a fresh recorder + incident
# tracker bound to the new name, so nothing from the old session leaks in.

@app.route('/session', methods=['GET'])
def get_session_route():
    return jsonify({"session_id": CURRENT_SESSION})


@app.route('/session/new', methods=['POST'])
def new_session_route():
    global CURRENT_SESSION

    if monitoring_active:
        return jsonify({
            "status":  "error",
            "message": "Stop monitoring before starting a new session.",
        }), 400

    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"status": "error", "message": "Session name is required."}), 400

    safe_name = re.sub(r'[^A-Za-z0-9_-]', '_', name)[:64]
    if not safe_name:
        return jsonify({"status": "error", "message": "Session name has no usable characters."}), 400

    for _entry in list(camera_registry.values()):
        _finish_recording(_entry)

    CURRENT_SESSION = safe_name
    os.makedirs(os.path.join(VIOLATION_FOLDER, CURRENT_SESSION), exist_ok=True)

    for cam_id, entry in list(camera_registry.items()):
        entry["recorder"]  = SessionRecorder(CURRENT_SESSION, cam_id)
        entry["incidents"] = IncidentTracker()

    return jsonify({"status": "ok", "session_id": CURRENT_SESSION})


@app.route('/review')
def review():
    return render_template('review.html')


@app.route('/incidents')
def incidents_route():
    rows = list_incidents(CURRENT_SESSION)
    for r in rows:
        r['video_url'] = f"/recordings/{r['recording']}" if r['recording'] else None
        r['thumb_url'] = f"/static/{r['thumb']}"         if r['thumb']      else None
    return jsonify(rows)


@app.route('/incidents/<int:inc_id>/dismiss', methods=['POST'])
def dismiss_incident_route(inc_id):
    """Fix 6: lets a reviewer mark a flag as a false positive (a tracking
    glitch, a passing proctor, a misfire) without deleting the underlying
    evidence -- the incident stays in the record, just tagged dismissed."""
    dismiss_incident(inc_id, True)
    return jsonify({"status": "dismissed", "id": inc_id})


@app.route('/recordings/<path:filepath>')
def serve_recording(filepath):
    # conditional=True enables HTTP Range requests, which <video> needs for seeking.
    return send_from_directory(RECORDINGS_FOLDER, filepath, conditional=True)


@app.route('/system_stats')
def system_stats():
    try:
        records = get_recent_violations(CURRENT_SESSION, limit=100)
    except Exception as e:
        print(f"[DB ERROR in system_stats]: {e}")
        records = []

    phone_count = interaction_count = gaze_count = 0
    for r in records:
        v_type = str(r.get('violation_type', '')).upper()
        if "OBJECT" in v_type or "PHONE" in v_type or "WATCH" in v_type:
            phone_count += 1
        elif "INTERACTION" in v_type:
            interaction_count += 1
        elif "GAZE" in v_type:
            gaze_count += 1

    return jsonify({
        "monitoring":       monitoring_active,
        "total_violations": len(records),
        "phone":            phone_count,
        "interaction":      interaction_count,
        "gaze":             gaze_count,
        "session_id":       CURRENT_SESSION,
    })


if __name__ == '__main__':
    # debug=False: Flask's reloader can double-import this module, which would
    # double-open cameras and expose raw stack traces in-browser -- not what
    # you want in front of a panel.
    # threaded=True: /video_feed/<id> is a long-lived streaming connection per
    # camera; without threading other requests queue behind it.
    # host: Fix 8 -- default to localhost only. Set CERBERUS_HOST=0.0.0.0 to
    # allow other devices on the network to reach the dashboard (needed if
    # you view it from a phone/second laptop), but then make sure
    # CERBERUS_PASSWORD is set to something real first.
    app.run(debug=False, threaded=True,
             host=os.environ.get("CERBERUS_HOST", "127.0.0.1"), port=5000)