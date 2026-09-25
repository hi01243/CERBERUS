from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLO
import os
import time
import shutil
import threading
from datetime import datetime
from collections import deque
import math
import numpy as np

# Database helper functions from database.py
from database import init_db, log_violation, get_recent_violations, clear_violations

app = Flask(__name__)

# Initialize SQLite database schema
init_db()

VIOLATION_FOLDER = os.path.join('static', 'violations')
os.makedirs(VIOLATION_FOLDER, exist_ok=True)

CURRENT_SESSION = "SESSION_ROOM_A1"

# --- SYSTEM STATE & THRESHOLDS ---
monitoring_active = False   # session-wide arm/disarm (the ARM/PAUSE buttons)
CAPTURE_COOLDOWN = 3.0      # Seconds between evidence photos, per camera

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

# ==========================================
# DYNAMIC CAMERA REGISTRY
# ==========================================
# Replaces the old hardcoded CAMERA_SOURCES dict. Cameras are now added,
# activated ("Use"), paused, and removed at runtime through the routes
# below and the dashboard UI -- no code edits or restarts needed.
#
# Each camera is registered but INACTIVE until "Use" is clicked: its
# VideoCapture and YOLO models are only created at that point. This means
# even the default local camera pre-registered at startup does NOT stream
# automatically -- click "Use" on it once the app is running.
#
# registry_lock guards adding/removing entries (dict structure changes).
# Each entry also has its own lock guarding its cap/model swap + read, so
# controlling one camera never blocks another camera's frame loop.
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


def add_camera(source, cam_type, label=None):
    """Registers a camera as INACTIVE. Does not open the device or load
    models -- that happens on activate_camera(), so registering (or
    scanning) many cameras stays cheap."""
    cam_id = _generate_camera_id()
    entry = {
        "source": source,
        "type": cam_type,          # "local" (device index) or "ip" (URL)
        "label": label or cam_id,
        "status": "inactive",      # "inactive" | "active" | "paused"
        "cap": None,
        "pose_model": None,
        "object_model": None,
        "lock": threading.Lock(),
        "gaze_timers": {},
        "turn_history": {},
        "interaction_timers": {},
        "last_capture_time": 0,
    }
    with registry_lock:
        camera_registry[cam_id] = entry
    return cam_id


def activate_camera(cam_id):
    """Opens the capture device (if not already open) and loads models (if
    not already loaded), then marks the camera active. Used for both first
    activation AND resuming from paused -- paused cameras keep their
    cached models, so resuming skips the (slow) model-loading step and
    only has to reopen the capture device."""
    entry = camera_registry.get(cam_id)
    if entry is None:
        return False, "Camera not found"

    with entry["lock"]:
        if entry["cap"] is None:
            cap = cv2.VideoCapture(entry["source"])
            if not cap.isOpened():
                return False, f"Could not open camera source: {entry['source']}"
            entry["cap"] = cap

        if entry["pose_model"] is None:
            entry["pose_model"] = YOLO('yolo11n-pose.pt')
        if entry["object_model"] is None:
            entry["object_model"] = YOLO('yolo11n.pt')

        entry["status"] = "active"

    return True, "activated"


def pause_camera(cam_id):
    """Fully releases the capture device (frees the physical camera for
    other use) but keeps loaded models cached in memory, so a later
    resume is fast. Also clears this camera's tracking state -- resuming
    starts clean rather than reviving stale gaze/interaction timers."""
    entry = camera_registry.get(cam_id)
    if entry is None:
        return False, "Camera not found"

    with entry["lock"]:
        if entry["cap"] is not None:
            entry["cap"].release()
            entry["cap"] = None
        entry["status"] = "paused"
        entry["gaze_timers"].clear()
        entry["turn_history"].clear()
        entry["interaction_timers"].clear()

    return True, "paused"


def remove_camera(cam_id):
    """Fully deletes a camera: releases its device and drops it from the
    registry. Any browser tab still displaying its stream will see the
    generator end gracefully (see generate_frames)."""
    entry = camera_registry.get(cam_id)
    if entry is None:
        return False, "Camera not found"

    with entry["lock"]:
        if entry["cap"] is not None:
            entry["cap"].release()

    with registry_lock:
        camera_registry.pop(cam_id, None)

    return True, "removed"


def scan_local_cameras(max_index=4):
    """Probes local device indices for available cameras not already
    registered. Blocking, can take a few seconds -- meant to be triggered
    by an explicit 'Scan' button click, not polled automatically. Indices
    already registered are skipped so we never try to open a device our
    own registry already has a handle on."""
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


# --- Pre-register one default local camera so the dashboard isn't empty
# on first load. It does NOT auto-start -- click "Use" on it. ---
add_camera(0, "local", label="cam1")


def calculate_angle(p1, p2):
    """Calculates the tilt angle of a 2D vector relative to the vertical line."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = math.atan2(abs(dx), abs(dy))
    return math.degrees(rads)


def make_status_frame(label, message, sub_message=""):
    """Placeholder frame for a camera that's inactive, paused, or
    disconnected -- shown instead of hitting an empty/broken video_feed."""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, f"'{label}'", (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
    cv2.putText(placeholder, message, (40, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    if sub_message:
        cv2.putText(placeholder, sub_message, (40, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2)
    return placeholder


def generate_frames(camera_id):
    while True:
        entry = camera_registry.get(camera_id)
        if entry is None:
            # Camera was removed while a browser tab still had this
            # stream open -- end the generator gracefully rather than
            # erroring on a missing registry entry.
            break

        if entry["status"] != "active":
            msg = "PAUSED" if entry["status"] == "paused" else "NOT IN USE"
            placeholder = make_status_frame(entry["label"], msg, "Click 'Use' on the dashboard")
            ret, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.5)  # don't spin a tight loop on an idle camera
            continue

        with entry["lock"]:
            cap = entry["cap"]
            if cap is None:
                # Status says active but cap is missing -- can happen for
                # a brief instant during a concurrent pause. Skip this
                # cycle rather than crash.
                success, frame = False, None
            else:
                success, frame = cap.read()

        if not success:
            # Basic reconnect: release and reopen. Not production-grade
            # (no backoff cap, no max-attempt circuit breaker) but enough
            # to survive a brief USB/WiFi hiccup during a live demo
            # instead of permanently killing the feed.
            with entry["lock"]:
                if entry["cap"] is not None:
                    entry["cap"].release()
                entry["cap"] = cv2.VideoCapture(entry["source"])

            placeholder = make_status_frame(entry["label"], "DISCONNECTED", "Attempting to reconnect...")
            ret, buffer = cv2.imencode('.jpg', placeholder)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(1.0)
            continue

        h, w, _ = frame.shape
        current_time = time.time()

        gaze_timers = entry["gaze_timers"]
        turn_history = entry["turn_history"]
        pair_timers = entry["interaction_timers"]

        cv2.putText(frame, f"CAM: {entry['label']}", (20, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if not monitoring_active:
            gaze_timers.clear()
            turn_history.clear()
            pair_timers.clear()
            cv2.putText(frame, "MONITORING PAUSED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        object_violation = False
        interaction_violation = False
        gaze_trigger = False
        violation_reason = ""
        violation_confidence = 0.0
        violating_track_id = None

        # ==========================================
        # 1. PROHIBITED OBJECT DETECTION (YOLO)
        # ==========================================
        object_detections = []
        try:
            obj_results = entry["object_model"](frame, stream=True, verbose=False)
            for r in obj_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if cls_id in PROHIBITED_CLASSES and confidence > 0.45:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        detected_name = PROHIBITED_CLASSES[cls_id]
                        object_detections.append({
                            "name": detected_name,
                            "confidence": confidence,
                            "center": center,
                        })
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"{detected_name} ({confidence*100:.0f}%)",
                                    (x1, max(y1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        except Exception as e:
            print(f"[{camera_id}] Object detection error: {e}")

        # ==========================================
        # 2. SKELETAL POSE ESTIMATION + TRACKING
        # ==========================================
        students = []
        active_ids = []

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

                kpts = r.keypoints.xy.cpu().numpy()
                confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

                if r.boxes is not None and r.boxes.id is not None:
                    track_ids = r.boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(kpts)

                for idx, person in enumerate(kpts):
                    track_id = track_ids[idx] if idx < len(track_ids) else None
                    if track_id is None:
                        continue

                    nose = person[0]
                    l_shoulder, r_shoulder = person[5], person[6]
                    l_hip, r_hip = person[11], person[12]

                    if confs is not None:
                        c_nose = confs[idx][0]
                        c_lsh = confs[idx][5]
                        c_rsh = confs[idx][6]
                        c_lhip = confs[idx][11]
                        c_rhip = confs[idx][12]
                        if c_lsh < MIN_KEYPOINT_CONF or c_rsh < MIN_KEYPOINT_CONF:
                            continue
                    else:
                        c_nose, c_lhip, c_rhip = 1.0, 1.0, 1.0

                    nose_confident = c_nose >= MIN_KEYPOINT_CONF
                    hips_confident = c_lhip >= MIN_KEYPOINT_CONF and c_rhip >= MIN_KEYPOINT_CONF

                    shoulder_center = (int((l_shoulder[0] + r_shoulder[0]) / 2),
                                       int((l_shoulder[1] + r_shoulder[1]) / 2))
                    shoulder_width = max(20.0, abs(r_shoulder[0] - l_shoulder[0]))

                    spine_tilt = 0.0
                    if hips_confident and l_hip[0] > 0 and r_hip[0] > 0:
                        hip_center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
                        spine_tilt = calculate_angle(hip_center, shoulder_center)

                    students.append({
                        "track_id": track_id,
                        "shoulder_center": shoulder_center,
                        "shoulder_width": shoulder_width,
                        "spine_tilt": spine_tilt,
                        "nose": nose
                    })

                    cv2.circle(frame, (int(l_shoulder[0]), int(l_shoulder[1])), 4, (0, 255, 255), -1)
                    cv2.circle(frame, (int(r_shoulder[0]), int(r_shoulder[1])), 4, (0, 255, 255), -1)
                    cv2.line(frame, (int(l_shoulder[0]), int(l_shoulder[1])),
                                    (int(r_shoulder[0]), int(r_shoulder[1])), (0, 255, 255), 2)
                    cv2.putText(frame, f"ID {track_id}",
                                (shoulder_center[0] - 20, shoulder_center[1] + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                    # --- HEAD YAW: HORIZONTAL-DISPLACEMENT RATIO ---
                    # Measures only the sideways nose offset from the
                    # shoulder-line midpoint, in a coordinate frame rotated
                    # to account for the student's seat/torso angle. This
                    # deliberately ignores vertical (pitch) movement, so
                    # looking down at an exam paper doesn't get conflated
                    # with turning sideways.
                    person_turn = "FORWARD"

                    if nose_confident and nose[0] > 0 and l_shoulder[0] > 0 and r_shoulder[0] > 0:
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
                        sideways_ratio = sideways_offset / (shoulder_width / 2.0)

                        if track_id not in turn_history:
                            turn_history[track_id] = deque(maxlen=SMOOTHING_WINDOW)
                        turn_history[track_id].append(sideways_ratio)
                        smoothed_ratio = sum(turn_history[track_id]) / len(turn_history[track_id])

                        if smoothed_ratio < -HEAD_TURN_RATIO_THRESHOLD:
                            person_turn = "LOOKING LEFT"
                        elif smoothed_ratio > HEAD_TURN_RATIO_THRESHOLD:
                            person_turn = "LOOKING RIGHT"

                    active_ids.append(track_id)

                    if person_turn != "FORWARD":
                        if track_id not in gaze_timers:
                            gaze_timers[track_id] = current_time
                        elapsed = current_time - gaze_timers[track_id]

                        if elapsed >= GAZE_DURATION_THRESHOLD:
                            gaze_trigger = True
                            if not violation_reason:
                                violation_reason = "PROLONGED_SIDE_GAZE"
                                violation_confidence = 0.95
                                violating_track_id = track_id
                            cv2.putText(frame, person_turn,
                                        (shoulder_center[0] - 50, max(20, shoulder_center[1] - 40)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            time_rem = GAZE_DURATION_THRESHOLD - elapsed
                            cv2.putText(frame, f"{person_turn} ({time_rem:.1f}s)",
                                        (shoulder_center[0] - 60, max(20, shoulder_center[1] - 40)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    else:
                        gaze_timers.pop(track_id, None)
                        turn_history.pop(track_id, None)

        except Exception as e:
            print(f"[{camera_id}] Pose estimation error: {e}")
            students = []
            active_ids = []

        entry["gaze_timers"] = {k: v for k, v in gaze_timers.items() if k in active_ids}
        entry["turn_history"] = {k: v for k, v in turn_history.items() if k in active_ids}

        # ==========================================
        # 3. ATTRIBUTE OBJECT DETECTIONS TO A STUDENT
        # ==========================================
        # Best-effort nearest-shoulder match, NOT a verified possession
        # check -- a phone on a neighboring desk close to another student
        # can still get misattributed. State this as a known limitation
        # if asked in defense, don't present it as certain.
        if object_detections:
            object_violation = True
            best_det = max(object_detections, key=lambda d: d["confidence"])
            violation_reason = f"OBJECT_{best_det['name'].replace(' ', '_')}"
            violation_confidence = best_det["confidence"]

            nearest_student = None
            nearest_dist = float("inf")
            for s in students:
                d = math.hypot(s["shoulder_center"][0] - best_det["center"][0],
                                s["shoulder_center"][1] - best_det["center"][1])
                radius = s["shoulder_width"] * ATTRIBUTION_RADIUS_MULTIPLIER
                if d < radius and d < nearest_dist:
                    nearest_dist = d
                    nearest_student = s
            violating_track_id = nearest_student["track_id"] if nearest_student else None

        # ==========================================
        # 4. INTERACTION & DESK LEANING DETECTION (DURATION-GATED)
        # ==========================================
        active_pairs = []
        num_students = len(students)
        for i in range(num_students):
            for j in range(i + 1, num_students):
                s1, s2 = students[i], students[j]

                dist = math.hypot(s1["shoulder_center"][0] - s2["shoulder_center"][0],
                                  s1["shoulder_center"][1] - s2["shoulder_center"][1])
                avg_shoulder_w = (s1["shoulder_width"] + s2["shoulder_width"]) / 2.0
                proximity_limit = avg_shoulder_w * PROXIMITY_MULTIPLIER
                excessive_lean = (s1["spine_tilt"] > SPINE_TILT_THRESHOLD_DEG) or \
                                  (s2["spine_tilt"] > SPINE_TILT_THRESHOLD_DEG)
                is_close = dist < proximity_limit or (dist < proximity_limit * 1.3 and excessive_lean)

                pair_key = frozenset({s1["track_id"], s2["track_id"]})

                if is_close:
                    active_pairs.append(pair_key)
                    if pair_key not in pair_timers:
                        pair_timers[pair_key] = current_time
                    elapsed = current_time - pair_timers[pair_key]

                    if elapsed >= INTERACTION_DURATION_THRESHOLD:
                        interaction_violation = True
                        if not violation_reason:
                            violation_reason = "STUDENT_INTERACTION"
                            violation_confidence = 0.90
                            violating_track_id = s1["track_id"]

                        cv2.line(frame, s1["shoulder_center"], s2["shoulder_center"], (0, 0, 255), 3)
                        mid_pt = ((s1["shoulder_center"][0] + s2["shoulder_center"][0]) // 2,
                                  (s1["shoulder_center"][1] + s2["shoulder_center"][1]) // 2)
                        cv2.putText(frame, "INTERACTION", (mid_pt[0] - 45, mid_pt[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    pair_timers.pop(pair_key, None)

        entry["interaction_timers"] = {k: v for k, v in pair_timers.items() if k in active_pairs}

        # ==========================================
        # 5. HUD / DASHBOARD STATUS OVERLAY
        # ==========================================
        status_color = (0, 255, 0)
        status_text = "STATUS: CLEAR"

        if object_violation:
            status_text = f"ALERT: {violation_reason}!"
            status_color = (0, 0, 255)
        elif interaction_violation:
            status_text = "ALERT: STUDENTS INTERACTING!"
            status_color = (0, 0, 255)
        elif gaze_trigger:
            status_text = "ALERT: PROLONGED SIDE GAZE!"
            status_color = (0, 0, 255)
        elif len(entry["gaze_timers"]) > 0:
            status_text = "WARNING: SIDE GAZE TRACKED"
            status_color = (0, 255, 255)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # ==========================================
        # 6. AUTOMATIC EVIDENCE CAPTURE & DB LOGGING
        # ==========================================
        is_violating = object_violation or interaction_violation or gaze_trigger

        if is_violating and (current_time - entry["last_capture_time"] > CAPTURE_COOLDOWN):
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{violation_reason}_{timestamp_str}.jpg"

            session_dir = os.path.join(VIOLATION_FOLDER, CURRENT_SESSION, camera_id)
            os.makedirs(session_dir, exist_ok=True)

            disk_path = os.path.join(session_dir, filename)
            web_path = f"violations/{CURRENT_SESSION}/{camera_id}/{filename}"

            cv2.imwrite(disk_path, frame)
            log_violation(CURRENT_SESSION, violation_reason, violation_confidence,
                          web_path, student_track_id=violating_track_id, camera_id=camera_id)
            print(f"[{camera_id}] CAPTURED {web_path} (track_id={violating_track_id})")

            entry["last_capture_time"] = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    if camera_id not in camera_registry:
        return f"Unknown camera_id: {camera_id}", 404
    return Response(generate_frames(camera_id), mimetype='multipart/x-mixed-replace; boundary=frame')


# ==========================================
# CAMERA MANAGEMENT ROUTES
# ==========================================

@app.route('/cameras', methods=['GET'])
def list_cameras():
    result = []
    for cam_id, entry in list(camera_registry.items()):
        result.append({
            "id": cam_id,
            "label": entry["label"],
            "type": entry["type"],
            "source": entry["source"],
            "status": entry["status"],
        })
    return jsonify(result)


@app.route('/cameras/scan', methods=['POST'])
def scan_cameras_route():
    found = scan_local_cameras()
    return jsonify({"found_indices": found})


@app.route('/cameras/add', methods=['POST'])
def add_camera_route():
    data = request.get_json(force=True, silent=True) or {}
    raw_source = str(data.get("source", "")).strip()
    label = (data.get("label") or "").strip() or None

    if not raw_source:
        return jsonify({"status": "error", "message": "Source is required"}), 400

    if raw_source.isdigit():
        source = int(raw_source)
        cam_type = "local"
    else:
        if not (raw_source.startswith("http://") or raw_source.startswith("https://")
                or raw_source.startswith("rtsp://")):
            return jsonify({
                "status": "error",
                "message": "IP camera source must be a URL starting with http://, https://, or rtsp://"
            }), 400
        source = raw_source
        cam_type = "ip"

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


@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Deletes all logged violations and evidence photos for the current
    session, and clears in-memory tracking state for every registered
    camera. Requires monitoring to be paused first."""
    global monitoring_active

    if monitoring_active:
        return jsonify({
            "status": "error",
            "message": "Stop monitoring before resetting."
        }), 400

    clear_violations(CURRENT_SESSION)

    session_dir = os.path.join(VIOLATION_FOLDER, CURRENT_SESSION)
    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir)

    for cam_id, entry in list(camera_registry.items()):
        os.makedirs(os.path.join(session_dir, cam_id), exist_ok=True)
        entry["gaze_timers"].clear()
        entry["turn_history"].clear()
        entry["interaction_timers"].clear()
        entry["last_capture_time"] = 0

    return jsonify({"status": "reset", "session_id": CURRENT_SESSION})


@app.route('/system_stats')
def system_stats():
    try:
        records = get_recent_violations(CURRENT_SESSION, limit=100)
    except Exception as e:
        print(f"[DB ERROR in system_stats]: {e}")
        records = []

    phone_count = 0
    interaction_count = 0
    gaze_count = 0

    for r in records:
        v_type = str(r.get('violation_type', '')).upper()
        if "OBJECT" in v_type or "PHONE" in v_type or "WATCH" in v_type:
            phone_count += 1
        elif "INTERACTION" in v_type:
            interaction_count += 1
        elif "GAZE" in v_type:
            gaze_count += 1

    return jsonify({
        "monitoring": monitoring_active,
        "total_violations": len(records),
        "phone": phone_count,
        "interaction": interaction_count,
        "gaze": gaze_count
    })


if __name__ == '__main__':
    # debug=False: Flask's debugger/reloader can double-import this module,
    # which can double-open camera devices, and it shows a raw stack trace
    # in-browser on any unhandled exception -- not something you want live
    # in front of a panel.
    # threaded=True: /video_feed/<id> is a long-lived streaming connection
    # per camera; without threading, other requests (start/stop/reset,
    # camera management, and additional camera streams) can queue behind it.
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)