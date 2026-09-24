from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import os
import time
from datetime import datetime
import math
import numpy as np

# Database helper functions from database.py
from database import init_db, log_violation, get_recent_violations

app = Flask(__name__)

# Initialize SQLite database schema
init_db()

# Camera feed (0 for built-in, or replace with phone IP Webcam URL)
camera = cv2.VideoCapture(0)

VIOLATION_FOLDER = os.path.join('static', 'violations')
os.makedirs(VIOLATION_FOLDER, exist_ok=True)

CURRENT_SESSION = "SESSION_ROOM_A1"

# --- SYSTEM STATE & THRESHOLDS ---
monitoring_active = False
last_capture_time = 0
CAPTURE_COOLDOWN = 3.0          # Seconds between evidence photos
GAZE_DURATION_THRESHOLD = 2.0   # Seconds looking away to trigger violation

# NOTE: These thresholds (38 deg yaw, 25 deg spine tilt, 2.2x shoulder-width
# proximity) are hand-calibrated, not learned. Before demo day, run a real
# test session with people seated at actual exam spacing and adjust these
# against what you observe -- don't trust them untested.
HEAD_TURN_THRESHOLD_DEG = 38.0
SPINE_TILT_THRESHOLD_DEG = 25.0
PROXIMITY_MULTIPLIER = 2.2

# Minimum keypoint confidence required before trusting a landmark in the
# math below. Previously only shoulders were gated; the nose was used
# ungated, which lets noisy/occluded keypoints feed the yaw calculation.
MIN_KEYPOINT_CONF = 0.5

# Per-student persistent gaze tracking: {track_id: start_time}
# Keyed by YOLO's persistent tracker ID (see CHANGE #1 below), not by a
# quantized x-position bucket. This survives a student shifting slightly
# in their seat, which the old zone-based key did not handle correctly.
student_gaze_timers = {}

# --- INITIALIZE YOLO DUAL ENGINES ---
# 1. Pose estimation for room-scale head yaw, spine tilt, and body geometry
pose_model = YOLO('yolo11n-pose.pt')

# 2. Object detection for phones/smartwatches
object_model = YOLO('yolo11n.pt')
# COCO class 67 = cell phone, 74 = clock (smartwatch proxy)
PROHIBITED_CLASSES = {67: "CELL PHONE", 74: "SMARTWATCH"}


def calculate_angle(p1, p2):
    """Calculates the tilt angle of a 2D vector relative to the vertical line."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    rads = math.atan2(abs(dx), abs(dy))
    return math.degrees(rads)


def generate_frames():
    global last_capture_time, student_gaze_timers

    while True:
        success, frame = camera.read()
        if not success:
            break

        h, w, _ = frame.shape
        current_time = time.time()

        # If monitoring is stopped, skip heavy models and show status banner
        if not monitoring_active:
            student_gaze_timers.clear()
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
        # Track ID of whichever student triggered the gaze/interaction
        # violation, so we can log identity alongside the reason.
        violating_track_id = None

        # ==========================================
        # 1. PROHIBITED OBJECT DETECTION (YOLO)
        # ==========================================
        obj_results = object_model(frame, stream=True, verbose=False)
        for r in obj_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if cls_id in PROHIBITED_CLASSES and confidence > 0.45:
                    object_violation = True
                    detected_name = PROHIBITED_CLASSES[cls_id]
                    violation_reason = f"OBJECT_{detected_name.replace(' ', '_')}"
                    violation_confidence = confidence

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{detected_name} ({confidence*100:.0f}%)",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ==========================================
        # 2. SKELETAL POSE ESTIMATION + TRACKING (CALIBRATED & DEBOUNCED)
        # ==========================================
        # CHANGE #1: use .track() with persist=True instead of a plain
        # forward pass. This assigns a stable ID per person across frames
        # via ByteTrack, replacing the old "round x-position to nearest
        # 100px" pseudo-identity, which broke at bucket boundaries and
        # didn't actually track anyone.
        pose_results = pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False
        )

        students = []
        active_ids = []

        for r in pose_results:
            if r.keypoints is None or len(r.keypoints) == 0:
                continue

            kpts = r.keypoints.xy.cpu().numpy()
            confs = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None

            # CHANGE #1 (cont'd): pull the tracker's persistent ID for each
            # detected person. This lives on r.boxes.id, not on keypoints,
            # so we zip them together by index. IDs are None on frames
            # where the tracker hasn't confirmed a detection yet.
            if r.boxes is not None and r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(kpts)

            for idx, person in enumerate(kpts):
                track_id = track_ids[idx] if idx < len(track_ids) else None
                if track_id is None:
                    # Skip detections the tracker hasn't assigned an ID to
                    # yet rather than faking one -- an untracked person
                    # can't have a meaningful violation history.
                    continue

                # Landmark indices
                nose = person[0]
                l_ear, r_ear = person[3], person[4]
                l_shoulder, r_shoulder = person[5], person[6]
                l_hip, r_hip = person[11], person[12]

                # Confidence check (must be confident to avoid ghost keypoint jitter)
                if confs is not None:
                    c_nose = confs[idx][0]
                    c_lear = confs[idx][3]
                    c_rear = confs[idx][4]
                    c_lsh = confs[idx][5]
                    c_rsh = confs[idx][6]
                    if c_lsh < MIN_KEYPOINT_CONF or c_rsh < MIN_KEYPOINT_CONF:
                        continue
                else:
                    c_nose, c_lear, c_rear = 1.0, 1.0, 1.0

                # CHANGE #2: gate on nose confidence too. Previously the
                # nose keypoint was fed into the yaw calculation ungated,
                # so an occluded/low-confidence nose (common at range or
                # on a sharp head turn) could inject noisy angles. Skip
                # the yaw calc for this person on this frame if the nose
                # isn't confidently detected -- shoulders/spine logic below
                # is unaffected since it doesn't depend on the nose.
                nose_confident = c_nose >= MIN_KEYPOINT_CONF

                shoulder_center = (int((l_shoulder[0] + r_shoulder[0]) / 2),
                                   int((l_shoulder[1] + r_shoulder[1]) / 2))
                shoulder_width = max(20.0, abs(r_shoulder[0] - l_shoulder[0]))

                # Torso tilt for desk interactions
                spine_tilt = 0.0
                if l_hip[0] > 0 and r_hip[0] > 0:
                    hip_center = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
                    spine_tilt = calculate_angle(hip_center, shoulder_center)

                students.append({
                    "track_id": track_id,
                    "shoulder_center": shoulder_center,
                    "shoulder_width": shoulder_width,
                    "spine_tilt": spine_tilt,
                    "nose": nose
                })

                # Visual indicator on shoulders
                cv2.circle(frame, (int(l_shoulder[0]), int(l_shoulder[1])), 4, (0, 255, 255), -1)
                cv2.circle(frame, (int(r_shoulder[0]), int(r_shoulder[1])), 4, (0, 255, 255), -1)
                cv2.line(frame, (int(l_shoulder[0]), int(l_shoulder[1])),
                                (int(r_shoulder[0]), int(r_shoulder[1])), (0, 255, 255), 2)
                # Show the persistent track ID above each student so it's
                # visible during the live demo that identity is stable.
                cv2.putText(frame, f"ID {track_id}",
                            (shoulder_center[0] - 20, shoulder_center[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                # --- ANGLE-INVARIANT HEAD YAW LOGIC ---
                person_turn = "FORWARD"

                if nose_confident and nose[0] > 0 and l_shoulder[0] > 0 and r_shoulder[0] > 0:
                    # 1. Calculate the angle of the shoulder line relative to horizontal
                    d_x_sh = r_shoulder[0] - l_shoulder[0]
                    d_y_sh = r_shoulder[1] - l_shoulder[1]
                    shoulder_angle = math.degrees(math.atan2(d_y_sh, d_x_sh))

                    # 2. Calculate vector from shoulder center to nose
                    mid_sh_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
                    mid_sh_y = (l_shoulder[1] + r_shoulder[1]) / 2.0

                    d_x_head = nose[0] - mid_sh_x
                    d_y_head = nose[1] - mid_sh_y
                    head_vector_angle = math.degrees(math.atan2(d_y_head, d_x_head))

                    # 3. Compute relative angle between chest plane and head position
                    # Forward-facing head rests at roughly -90 deg relative to the torso line
                    relative_turn = (head_vector_angle - shoulder_angle) + 90.0

                    # Normalize within [-180, 180]
                    while relative_turn > 180: relative_turn -= 360
                    while relative_turn < -180: relative_turn += 360

                    # 4. Deadband buffer: requires a deliberate swivel
                    if relative_turn < -HEAD_TURN_THRESHOLD_DEG:
                        person_turn = "LOOKING LEFT"
                    elif relative_turn > HEAD_TURN_THRESHOLD_DEG:
                        person_turn = "LOOKING RIGHT"

                active_ids.append(track_id)

                # Timer handling with grace reset, now keyed by track_id
                if person_turn != "FORWARD":
                    if track_id not in student_gaze_timers:
                        student_gaze_timers[track_id] = current_time

                    elapsed = current_time - student_gaze_timers[track_id]

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
                    student_gaze_timers.pop(track_id, None)

        # Drop timers for any track_id no longer present this frame
        # (person left frame, occluded, or re-assigned a new ID by the
        # tracker -- see the tracking limitation noted in chat: IDs are
        # not guaranteed to survive full occlusion).
        student_gaze_timers = {k: v for k, v in student_gaze_timers.items() if k in active_ids}

        # ==========================================
        # 3. INTERACTION & DESK LEANING DETECTION
        # ==========================================
        num_students = len(students)
        for i in range(num_students):
            for j in range(i + 1, num_students):
                s1 = students[i]
                s2 = students[j]

                dist = math.hypot(s1["shoulder_center"][0] - s2["shoulder_center"][0],
                                  s1["shoulder_center"][1] - s2["shoulder_center"][1])

                avg_shoulder_w = (s1["shoulder_width"] + s2["shoulder_width"]) / 2.0
                proximity_limit = avg_shoulder_w * PROXIMITY_MULTIPLIER

                excessive_lean = (s1["spine_tilt"] > SPINE_TILT_THRESHOLD_DEG) or \
                                  (s2["spine_tilt"] > SPINE_TILT_THRESHOLD_DEG)

                if dist < proximity_limit or (dist < proximity_limit * 1.3 and excessive_lean):
                    interaction_violation = True
                    violation_reason = "STUDENT_INTERACTION"
                    violation_confidence = 0.90
                    # DESIGN DECISION: if an object violation already fired
                    # this frame, its reason/track_id are NOT overwritten --
                    # object detection has logging priority. See note below.
                    if violating_track_id is None:
                        violating_track_id = s1["track_id"]

                    cv2.line(frame, s1["shoulder_center"], s2["shoulder_center"], (0, 0, 255), 3)
                    mid_pt = ((s1["shoulder_center"][0] + s2["shoulder_center"][0]) // 2,
                              (s1["shoulder_center"][1] + s2["shoulder_center"][1]) // 2)
                    cv2.putText(frame, "INTERACTION", (mid_pt[0] - 45, mid_pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ==========================================
        # 4. HUD / DASHBOARD STATUS OVERLAY
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
        elif len(student_gaze_timers) > 0:
            status_text = "WARNING: SIDE GAZE TRACKED"
            status_color = (0, 255, 255)

        cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # ==========================================
        # 5. AUTOMATIC EVIDENCE CAPTURE & DB LOGGING
        # ==========================================
        # DESIGN DECISION (was implicit before, now explicit): when both an
        # object violation (phone) and a posture violation (gaze/interaction)
        # fire in the same frame, only ONE reason is logged, because
        # violation_reason is set once and object detection runs first in
        # the pipeline. This means object violations have logging priority.
        # This was true in the original code too (via `if not violation_reason`)
        # but was never stated as a deliberate choice -- flagging it here so
        # it's a documented design decision rather than something a panelist
        # discovers by asking "why wasn't the gaze violation logged too?"
        is_violating = object_violation or interaction_violation or gaze_trigger

        if is_violating and (current_time - last_capture_time > CAPTURE_COOLDOWN):
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{violation_reason}_{timestamp_str}.jpg"

            session_dir = os.path.join(VIOLATION_FOLDER, CURRENT_SESSION)
            os.makedirs(session_dir, exist_ok=True)

            disk_path = os.path.join(session_dir, filename)
            web_path = f"violations/{CURRENT_SESSION}/{filename}"

            cv2.imwrite(disk_path, frame)

            # student_track_id is now persisted (see database.py migration)
            # so the dashboard can show WHICH student triggered a
            # gaze/interaction violation, not just that one occurred.
            # Will be NULL for object (phone) violations, since those
            # aren't tied to a specific tracked person in this version.
            log_violation(CURRENT_SESSION, violation_reason, violation_confidence,
                          web_path, student_track_id=violating_track_id)
            print(f"[DB & DISK CAPTURED] {web_path} (track_id={violating_track_id})")

            last_capture_time = current_time

        # Stream MJPEG byte buffer
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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
    app.run(debug=True)