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
from auth import init_auth_db, create_user, verify_user

BUILD_ID = "2026-09-30-r13-wrist-gate"
print(f"[cerberus] app.py build {BUILD_ID}")

app = Flask(__name__)

# Initialize SQLite database schema
init_db()
init_incident_db()
init_auth_db()

# Finalize any recordings a previous crash / power loss left half-written,
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
HEAD_TURN_RATIO_THRESHOLD = 0.45  # horizontal nose-displacement ratio, not a raw angle
GAZE_DURATION_THRESHOLD = 3.0
SPINE_TILT_THRESHOLD_DEG = 25.0
PROXIMITY_MULTIPLIER = 1.8
INTERACTION_DURATION_THRESHOLD = 3.0
ATTRIBUTION_RADIUS_MULTIPLIER = 4.0
MIN_KEYPOINT_CONF = 0.5
SMOOTHING_WINDOW = 5

# --- DETECTION RULE SWITCHES (change live via POST /settings) ---
# Per-exam rules.  A regular watch or an allowed calculator looks like a
# "smartwatch" / "cell phone" to YOLO, so turn those off when the exam allows
# them.  Interaction (shoulder proximity) is OFF by default: it has no
# ground-truth class in the test dataset and fires on normal seating.
DETECTION_ENABLED = {
    "phone":       True,
    "watch":       False,
    "reach":       True,
    "gaze":        True,
}

# --- WRIST-CROP PHONE PASS ---
# Second, zoomed-in phone check on a crop around each wrist (see
# _merge_wrist_phones).  Governed by DETECTION_ENABLED["phone"] as well.
WRIST_CROP_ENABLED      = True
WRIST_CROP_SCALE        = 1.6    # crop width = this x the student's shoulder width
WRIST_CROP_MIN_HALF_PX  = 32     # never crop smaller than 64 px across
WRIST_CROP_IMGSZ        = 320    # crops are upscaled to this before inference
WRIST_CROP_CONF         = 0.30   # lower than the full-frame 0.45; persistence compensates
WRIST_CROP_INTERVAL_SEC = 0.33   # run ~3x per second, not every frame
WRIST_CROP_MAX_CROPS    = 40     # cap per pass (large rooms: raise on a GPU)
WRIST_CROP_MIN_HITS     = 3      # hits needed ...
WRIST_CROP_WINDOW_SEC   = 2.0    # ... inside this window
WRIST_CROP_HOLD_SEC     = 0.8    # keep a confirmed phone alive between passes
WRIST_CROP_DEDUP_PX     = 50     # same phone as a full-frame detection if this close

# --- NEIGHBOR-AWARE SIDE GAZE ---
# A left/right head turn only counts as "looking at another student" if some
# other student sits on that side, in roughly the same row.  Distances are in
# units of the turning student's shoulder width (image left = "LOOKING LEFT").
GAZE_REQUIRE_NEIGHBOR = False   # r8: off by default; turn on only after testing on real footage
GAZE_NEIGHBOR_MIN_DX  = 0.6   # neighbor must be at least this far to the side
GAZE_NEIGHBOR_MAX_DX  = 6.0   # ...and no farther than this
GAZE_NEIGHBOR_ROW_MULT = 2.0  # max vertical offset between shoulder centers

# --- POSE MODEL INFERENCE SETTINGS ---
# A higher NMS IoU / confidence cuts down on the model emitting two
# overlapping skeletons for one person (the root cause of "ghost" students).
POSE_IOU  = 0.5
POSE_CONF = 0.4

# --- DUPLICATE-SKELETON (GHOST TRACK) FILTER ---
# Two skeletons whose shoulder centers sit closer than this fraction of
# their shoulder width cannot be two different seated people -- it's one
# person detected twice.  The older / higher-confidence one is kept.
# Without this, a ghost paired with its real twin reads as "shoulder
# distance ~0" and trips STUDENT_INTERACTION and REACHING_NEIGHBOR_ZONE.
DUPLICATE_DIST_MULTIPLIER = 0.8

# --- ADAPTIVE "NORMAL POSITION" BASELINE ---
CALIBRATION_DURATION = 2.0          # seconds to sample a student's initial "neutral" pose
BASELINE_ADAPT_RATE = 0.02          # how fast the baseline creeps toward a newly-held position
REPOSITION_STABILITY_WINDOW = 10.0  # student must hold a new position this long before baseline shifts
REPOSITION_JUMP_LIMIT = HEAD_TURN_RATIO_THRESHOLD * 0.8
REPOSITION_SETTLE_BAND = HEAD_TURN_RATIO_THRESHOLD * 0.3

# --- SMARTWATCH / WALL-CLOCK FILTER ---
# YOLO class 74 ("clock") fires on both smartwatches AND wall clocks.
#   A. Static guard: a clock that hasn't moved in STATIC_CLOCK_SEC is a wall clock.
#   B. Wrist guard: a clock must be near a visible wrist to count as worn.
# Cell-phone detections (class 67) bypass both guards.
STATIC_CLOCK_SEC       = 8.0
STATIC_CLOCK_RADIUS_PX = 60
STATIC_CLOCK_MOVE_PX   = 35
WRIST_PROXIMITY_PX     = 150
WRIST_CROP_MAX_WRIST_DIST  = 0.6   # phone centre must be within this x shoulder width of the wrist
WRIST_CROP_STATIC_SEC      = 6.0   # a wrist-crop 'phone' this still ...
WRIST_CROP_STATIC_MOVE_PX  = 8     # ... (moves less than this) is furniture
HELD_PHONE_WRIST_MULT      = 0.6   # full-frame static filter: a hand this close keeps the phone flagged
STATIC_PHONE_SEC       = 10.0   # a full-frame 'phone' that never moves this long is furniture,
                                # unless a visible wrist is within WRIST_PROXIMITY_PX of it

# --- CAMERA-BUMP DETECTION ---
CAMERA_BUMP_RATIO        = 0.6
CAMERA_BUMP_MIN_STUDENTS = 3
CAMERA_BUMP_COOLDOWN_SEC = 5.0

# --- TRACK-AGE GATE ---
TRACK_MIN_AGE_SEC = 1.0

# --- NOTE-PASSING / REACHING DETECTION ---
# A hand-off is detected two ways (either one counts):
#   A. ZONE:      a wrist enters a circle around the neighbor's shoulder
#                 center (REACH_ZONE_RADIUS_MULTIPLIER x their shoulder width).
#   B. EXTENSION: the arm is stretched out (wrist farther than
#                 REACH_EXTENSION_MULTIPLIER x own shoulder width from own
#                 shoulders) AND the wrist has travelled at least
#                 REACH_PROGRESS_MIN of the way toward the neighbor.  This
#                 is what catches a paper being passed across an aisle,
#                 where the hand never gets near the neighbor's chest.
# B only applies to students in roughly the same row, so a person merely
# stretching in front of / behind another isn't paired with them.
REACH_ZONE_RADIUS_MULTIPLIER    = 1.5
REACH_EXTENSION_MULTIPLIER      = 1.6
REACH_PROGRESS_MIN              = 0.35
REACH_SAME_ROW_MULTIPLIER       = 1.0    # max vertical offset between shoulder centers, x avg shoulder width
REACH_MAX_PAIR_DIST_MULTIPLIER  = 8.0    # generous: the extension test does the real filtering
REACH_DURATION_THRESHOLD        = 1.0
REACH_GRACE_SEC                 = 0.4    # tolerate a dropped wrist keypoint for this long
REACH_WRIST_CONF                = 0.3    # wrists are blurry mid-motion; accept lower confidence
# Shoulder-width ratio between the two students must fall in this band, or
# they are at different depths (different rows) and are never paired.
REACH_DEPTH_RATIO_MIN = 0.75
REACH_DEPTH_RATIO_MAX = 1.33
# Only students who are really side by side (shoulder centers at least this many
# average shoulder widths apart horizontally) can hand something to each other.
# Someone sitting directly behind another student has almost no horizontal gap.
REACH_MIN_SIDE_GAP_MULTIPLIER = 1.2

# --- HEAD-POSE ROBUSTNESS ---
# Looking DOWN at a paper shifts the nose sideways relative to the shoulder
# line and used to read as "looking left/right".  We now also track how far
# the nose sits above the shoulder line (normalised by shoulder width); if it
# has dropped by more than PITCH_DOWN_DELTA versus this student's baseline,
# a sideways reading is classified LOOKING DOWN and ignored.
PITCH_DOWN_DELTA = 0.30
# A very large sideways reading is a real turn no matter what the pitch says.
# Twisting / leaning tilts the shoulder line and inflates the pitch number, so
# without this override a genuine turn (yaw ~ -1.3 in testing) was being
# excused as "looking down".  Only MODERATE readings can be excused.
HEAD_TURN_STRONG_RATIO = 0.7   # was 1.0; tune on the 'looking at another student' clips
# A gaze timer survives a dip below threshold this long (a blink, one noisy
# frame) instead of restarting from zero.
GAZE_GRACE_SEC = 0.8
# If a student stays "flagged" this long, it is a held posture (leaning over
# to write, shifted chair) rather than a glance: the incident was already
# logged at GAZE_DURATION_THRESHOLD, so re-baseline to stop the endless alert.
GAZE_REBASELINE_SEC = 8.0
# Per-track state is kept this long after a track disappears, so a single
# missed frame doesn't wipe a student's baseline and restart calibration.
TRACK_LOST_GRACE_SEC = 1.5
# Draws yaw / pitch numbers by each student so thresholds can be tuned by eye.
DEBUG_OVERLAY = True

# ==========================================
# DYNAMIC CAMERA REGISTRY
# ==========================================
# activate_camera()  -> opens cap, loads models, starts _detection_loop() thread.
# _detection_loop()  -> reads frames, runs YOLO, records, logs incidents.
# generate_frames()  -> trivial MJPEG streamer yielding entry["latest_frame"].
# pause_camera()     -> stop_event, join, release cap.
# remove_camera()    -> same, then drops entry from registry.
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
    models -- that happens on activate_camera().

    cam_type: "local" (device-index int) | "ip" (URL str) | "file" (path str)
    """
    cam_id = _generate_camera_id()
    entry = {
        "source":   source,
        "type":     cam_type,
        "label":    label or cam_id,
        "status":   "inactive",        # "inactive" | "active" | "paused"
        "cap":      None,
        "pose_model":   None,
        "object_model": None,
        "lock":     threading.Lock(),
        "gaze_timers":        {},
        "turn_history":       {},
        "interaction_timers": {},
        "reach_timers":       {},
        "recorder":  SessionRecorder(CURRENT_SESSION, cam_id),
        "incidents": IncidentTracker(),
        # --- adaptive baseline state, per track_id ---
        "baseline_ratio":    {},
        "calibration_start": {},
        "stable_since":      {},
        "stable_anchor":     {},
        # --- background detection thread ---
        "latest_frame": None,
        "stop_event":   threading.Event(),
        "thread":       None,
        # --- smartwatch static-position tracker ---
        "static_objects": {},
        "static_phones":  {},
        "static_wc":      {},
        # --- camera-bump detection ---
        "last_bump_time": 0.0,
        # --- track-age gate ---
        "track_first_seen": {},
        "track_last_seen":  {},
        # --- head-pose baseline extras ---
        "calib_samples":   {},
        "baseline_height": {},
        "pitch_history":   {},
        "gaze_lost":       {},
    }
    with registry_lock:
        camera_registry[cam_id] = entry
    return cam_id


def activate_camera(cam_id):
    """Opens the capture device (if not already open), loads models (if not
    cached), then starts the per-camera detection thread."""
    entry = camera_registry.get(cam_id)
    if entry is None:
        return False, "Camera not found"

    if entry.get("thread") and entry["thread"].is_alive():
        return True, "already active"

    with entry["lock"]:
        if entry["cap"] is None:
            cap = cv2.VideoCapture(entry["source"])
            if not cap.isOpened():
                return False, f"Could not open camera source: {entry['source']}"
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


PER_TRACK_KEYS = ("gaze_timers", "turn_history", "baseline_ratio", "calibration_start",
                  "stable_since", "stable_anchor", "track_first_seen", "track_last_seen",
                  "calib_samples", "baseline_height", "pitch_history", "gaze_lost")


def _clear_tracking_state(entry):
    """Wipes every per-track dict on a camera entry."""
    for key in PER_TRACK_KEYS + ("interaction_timers", "reach_timers", "static_objects", "static_phones", "static_wc"):
        entry[key].clear()


def pause_camera(cam_id):
    """Stops the detection thread, releases the capture device.  Models stay
    cached so a later resume is fast."""
    entry = camera_registry.get(cam_id)
    if entry is None:
        return False, "Camera not found"

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
        _clear_tracking_state(entry)

    return True, "paused"


def remove_camera(cam_id):
    """Stops the thread, releases the device, finalises the recording, and
    drops the camera from the registry."""
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

    _finish_recording(entry)

    with registry_lock:
        camera_registry.pop(cam_id, None)

    return True, "removed"


def scan_local_cameras(max_index=4):
    """Probes local device indices for available cameras not already registered."""
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


# Pre-register one default local camera so the dashboard is not empty.
add_camera(0, "local", label="cam1")


def _neighbor_in_direction(turn, track_id, candidates, kept_idx):
    """True if another (non-ghost) student sits on the side this student is
    turned toward, in roughly the same row.  `turn` is 'LOOKING LEFT' or
    'LOOKING RIGHT' (image-left / image-right, matching the yaw sign)."""
    if not GAZE_REQUIRE_NEIGHBOR or turn not in ("LOOKING LEFT", "LOOKING RIGHT"):
        return True
    me = next((c for c in candidates
               if c["track_id"] == track_id and c["idx"] in kept_idx), None)
    if me is None:
        return True
    direction = -1 if turn == "LOOKING LEFT" else 1
    for c in candidates:
        if c["idx"] not in kept_idx or c["track_id"] == track_id:
            continue
        dx = (c["center"][0] - me["center"][0]) * direction
        dy = abs(c["center"][1] - me["center"][1])
        if (GAZE_NEIGHBOR_MIN_DX * me["width"] <= dx <= GAZE_NEIGHBOR_MAX_DX * me["width"]
                and dy <= GAZE_NEIGHBOR_ROW_MULT * me["width"]):
            return True
    return False


def _wrist_crop_sample(entry, camera_id, frame, students, now):
    """One zoomed-in phone pass over every established student's wrists."""
    h, w = frame.shape[:2]
    crops, meta = [], []
    for s in students:
        tid = s["track_id"]
        if now - entry["track_first_seen"].get(tid, now) < TRACK_MIN_AGE_SEC:
            continue
        half = int(max(WRIST_CROP_MIN_HALF_PX, s["shoulder_width"] * WRIST_CROP_SCALE / 2.0))
        for wrist in (s["l_wrist"], s["r_wrist"]):
            if wrist is None:
                continue
            x1, y1 = max(0, wrist[0] - half), max(0, wrist[1] - half)
            x2, y2 = min(w, wrist[0] + half), min(h, wrist[1] + half)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crops.append(frame[y1:y2, x1:x2].copy())
            meta.append((tid, x1, y1, wrist, s["shoulder_width"]))
            if len(crops) >= WRIST_CROP_MAX_CROPS:
                break
        if len(crops) >= WRIST_CROP_MAX_CROPS:
            break
    if not crops:
        return

    results = entry["object_model"](crops, imgsz=WRIST_CROP_IMGSZ,
                                    conf=WRIST_CROP_CONF, classes=[67], verbose=False)

    best = {}   # track_id -> strongest phone detection this pass (full-frame coords)
    for (tid, ox, oy, wpt, sw), r in zip(meta, results):
        for box in r.boxes:
            conf = float(box.conf[0])
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            ccx, ccy = ox + (bx1 + bx2) // 2, oy + (by1 + by2) // 2
            if math.hypot(ccx - wpt[0], ccy - wpt[1]) > WRIST_CROP_MAX_WRIST_DIST * sw:
                continue   # too far from the hand to be held in it (furniture at the crop edge)
            if tid not in best or conf > best[tid]["confidence"]:
                best[tid] = {
                    "name":       "CELL PHONE",
                    "confidence": conf,
                    "center":     (ox + (bx1 + bx2) // 2, oy + (by1 + by2) // 2),
                    "bbox":       (ox + bx1, oy + by1, ox + bx2, oy + by2),
                }

    hits      = entry.setdefault("_wc_hits", {})
    confirmed = entry.setdefault("_wc_confirmed", {})

    for tid, det in best.items():
        recent = [t for t in hits.get(tid, []) if now - t <= WRIST_CROP_WINDOW_SEC]
        recent.append(now)
        hits[tid] = recent
        if len(recent) >= WRIST_CROP_MIN_HITS:
            if tid not in confirmed:
                print(f"[{camera_id}] ID {tid}: phone in hand (wrist crop, "
                      f"{det['confidence']*100:.0f}%)")
            confirmed[tid] = {"det": det, "until": now + WRIST_CROP_HOLD_SEC}

    for tid in list(hits):
        if tid not in best:
            hits[tid] = [t for t in hits[tid] if now - t <= WRIST_CROP_WINDOW_SEC]
            if not hits[tid]:
                del hits[tid]


def _merge_wrist_phones(entry, camera_id, clean_frame, students, now, object_detections):
    """Adds confirmed wrist-crop phones to `object_detections` (in place).
    Never raises: a failure here must not stop the detection loop."""
    if (not WRIST_CROP_ENABLED or not DETECTION_ENABLED["phone"]
            or entry.get("object_model") is None):
        return
    try:
        confirmed = entry.setdefault("_wc_confirmed", {})
        if now - entry.get("_wc_last", 0.0) >= WRIST_CROP_INTERVAL_SEC:
            entry["_wc_last"] = now
            _wrist_crop_sample(entry, camera_id, clean_frame, students, now)

        for tid in [t for t, c in confirmed.items() if c["until"] < now]:
            del confirmed[tid]

        conf_items = list(confirmed.items())
        wc_static = _check_static_clocks(
            entry, [c["det"]["center"] for _, c in conf_items], now,
            store="static_wc", static_sec=WRIST_CROP_STATIC_SEC,
            move_px=WRIST_CROP_STATIC_MOVE_PX)
        for k, (tid, c) in enumerate(conf_items):
            if k in wc_static:
                continue   # has not moved: furniture, not a phone in a hand
            det = c["det"]
            dup = next((d for d in object_detections
                        if d["name"] == "CELL PHONE"
                        and math.hypot(d["center"][0] - det["center"][0],
                                       d["center"][1] - det["center"][1]) < WRIST_CROP_DEDUP_PX),
                       None)
            if dup is not None:
                dup.setdefault("track_id", tid)
                dup["confidence"] = max(dup["confidence"], det["confidence"])
            else:
                object_detections.append(dict(det, track_id=tid))
    except Exception as e:
        print(f"[{camera_id}] Wrist-crop phone error: {e}")


def _phone_in_hand(det, students):
    """True if any student's wrist is within HELD_PHONE_WRIST_MULT x their
    shoulder width of the detection's centre."""
    cx, cy = det["center"]
    for s in students:
        lim = s["shoulder_width"] * HELD_PHONE_WRIST_MULT
        for wr in (s["l_wrist"], s["r_wrist"]):
            if wr is not None and math.hypot(cx - wr[0], cy - wr[1]) < lim:
                return True
    return False


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
# DUPLICATE-SKELETON FILTER
# ==========================================
def _select_unique_people(candidates, track_first_seen, current_time):
    """Given pre-pass candidates (dicts with idx, track_id, center, width,
    conf), drop ghost skeletons: any candidate whose shoulder center falls
    within DUPLICATE_DIST_MULTIPLIER x shoulder width of an already-kept one.

    Priority: older track first (stable ID), then higher keypoint confidence.
    Returns the set of kept `idx` values.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (track_first_seen.get(c["track_id"], current_time), -c["conf"])
    )
    kept = []
    for c in ordered:
        is_dup = False
        for k in kept:
            d = math.hypot(c["center"][0] - k["center"][0],
                           c["center"][1] - k["center"][1])
            limit = min(c["width"], k["width"]) * DUPLICATE_DIST_MULTIPLIER
            if d < limit:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
    return {c["idx"] for c in kept}


# ==========================================
# NOTE-PASSING / REACH TEST
# ==========================================
def _find_reaching_wrist(reacher, target, same_row):
    """Returns the wrist (x, y) of `reacher` that is reaching toward `target`,
    or None.  See the REACH_* constants for the two tests."""
    rx, ry = reacher["shoulder_center"]
    tx, ty = target["shoulder_center"]
    sw     = reacher["shoulder_width"]
    gap_x  = tx - rx

    for wrist in (reacher["l_wrist"], reacher["r_wrist"]):
        if wrist is None:
            continue

        # A: wrist inside the neighbor's personal zone.
        if math.hypot(wrist[0] - tx, wrist[1] - ty) < target["shoulder_width"] * REACH_ZONE_RADIUS_MULTIPLIER:
            return wrist

        # B: arm stretched out toward the neighbor (same row only).
        if not same_row or abs(gap_x) < 1:
            continue
        extension = math.hypot(wrist[0] - rx, wrist[1] - ry) / sw
        if extension < REACH_EXTENSION_MULTIPLIER:
            continue
        if abs(wrist[1] - ry) > 2.5 * sw:
            continue
        progress = (wrist[0] - rx) / gap_x
        if progress >= REACH_PROGRESS_MIN:
            return wrist
    return None


# ==========================================
# SMARTWATCH STATIC-POSITION TRACKER (Guard A)
# ==========================================
def _check_static_clocks(entry, centers, now, store="static_objects",
                        static_sec=STATIC_CLOCK_SEC,
                        move_px=STATIC_CLOCK_MOVE_PX):
    """Update the per-camera tracker for clock bbox centres and return the
    set of indices (into `centers`) suppressed because the object has sat in
    the same spot for >= STATIC_CLOCK_SEC (almost certainly a wall clock)."""
    tracker  = entry.setdefault(store, {})
    NEXT_KEY = "__next"
    next_id  = tracker.get(NEXT_KEY, 0)

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
            if best_dist > move_px:
                td["static_since"] = now
                td["center"]       = center
            elif now - td["static_since"] >= static_sec:
                suppressed.add(idx)
        else:
            tracker[next_id] = {"center": center, "static_since": now, "last_seen": now}
            next_id += 1
            tracker[NEXT_KEY] = next_id

    return suppressed


# ==========================================
# BACKGROUND DETECTION THREAD
# ==========================================
def _detection_loop(camera_id):
    """Per-camera daemon thread.  Owns ALL frame reading, YOLO inference, HUD
    drawing, session recording, and incident logging.  Stores the latest
    annotated JPEG in entry['latest_frame'] for generate_frames().

    Video-file cameras loop back to frame 0 on EOF for demo replay.
    Live cameras drain their internal buffer each cycle to avoid lag.
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

                if entry["type"] != "file":
                    for _ in range(4):
                        if stop_event.is_set():
                            break
                        cap.grab()

                success, frame = cap.read()

            if not success:
                if entry["type"] == "file":
                    with entry["lock"]:
                        cap = entry["cap"]
                        if cap is not None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    continue
                else:
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
            clean_frame = frame.copy()   # un-annotated copy for the wrist-crop pass
            current_time = time.time()

            gaze_timers      = entry["gaze_timers"]
            turn_history     = entry["turn_history"]
            pair_timers      = entry["interaction_timers"]
            reach_timers     = entry["reach_timers"]
            baseline_ratio   = entry["baseline_ratio"]
            calib_start      = entry["calibration_start"]
            stable_since     = entry["stable_since"]
            stable_anchor    = entry["stable_anchor"]
            track_first_seen = entry["track_first_seen"]
            track_last_seen  = entry["track_last_seen"]
            calib_samples    = entry["calib_samples"]
            baseline_height  = entry["baseline_height"]
            pitch_history    = entry["pitch_history"]

            cv2.putText(frame, f"CAM: {entry['label']}", (20, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if not monitoring_active:
                _clear_tracking_state(entry)
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
            reach_violation       = False
            gaze_trigger          = False
            violation_reason      = ""
            violation_confidence  = 0.0
            violating_track_id    = None

            # ==========================================
            # 1. PROHIBITED OBJECT DETECTION (YOLO)
            # ==========================================
            object_detections = []
            try:
                obj_results = entry["object_model"](frame, stream=True, verbose=False)
                for r in obj_results:
                    for box in r.boxes:
                        cls_id     = int(box.cls[0])
                        confidence = float(box.conf[0])
                        if (cls_id in PROHIBITED_CLASSES and confidence > 0.45
                                and DETECTION_ENABLED["phone" if cls_id == 67 else "watch"]):
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
            wrist_positions = []

            try:
                pose_results = entry["pose_model"].track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    stream=True,
                    verbose=False,
                    iou=POSE_IOU,
                    conf=POSE_CONF,
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

                    # --- PRE-PASS: register first-seen times and collect
                    # valid candidates so ghost skeletons can be dropped
                    # BEFORE they touch any timer, baseline, or violation. ---
                    candidates = []
                    for idx, person in enumerate(kpts):
                        track_id = track_ids[idx] if idx < len(track_ids) else None
                        if track_id is None:
                            continue

                        if confs is not None:
                            if (confs[idx][5] < MIN_KEYPOINT_CONF or
                                    confs[idx][6] < MIN_KEYPOINT_CONF):
                                continue
                            pair_conf = float(confs[idx][5] + confs[idx][6]) / 2.0
                        else:
                            pair_conf = 1.0

                        l_sh, r_sh = person[5], person[6]
                        candidates.append({
                            "idx":      idx,
                            "track_id": track_id,
                            "center":   ((l_sh[0] + r_sh[0]) / 2.0,
                                         (l_sh[1] + r_sh[1]) / 2.0),
                            "width":    max(20.0, abs(r_sh[0] - l_sh[0])),
                            "conf":     pair_conf,
                        })
                        if track_id not in track_first_seen:
                            track_first_seen[track_id] = current_time

                    kept_idx = _select_unique_people(candidates, track_first_seen,
                                                     current_time)

                    for idx, person in enumerate(kpts):
                        if idx not in kept_idx:
                            continue
                        track_id = track_ids[idx]

                        nose                   = person[0]
                        l_shoulder, r_shoulder = person[5],  person[6]
                        l_hip,      r_hip      = person[11], person[12]
                        l_wrist,    r_wrist    = person[9],  person[10]

                        if confs is not None:
                            c_nose   = confs[idx][0]
                            c_lhip   = confs[idx][11]
                            c_rhip   = confs[idx][12]
                            c_lwrist = confs[idx][9]
                            c_rwrist = confs[idx][10]
                        else:
                            c_nose = c_lhip = c_rhip = 1.0
                            c_lwrist = c_rwrist = 1.0

                        nose_confident = c_nose  >= MIN_KEYPOINT_CONF
                        hips_confident = (c_lhip >= MIN_KEYPOINT_CONF and
                                          c_rhip >= MIN_KEYPOINT_CONF)

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

                        l_wrist_pt = ((int(l_wrist[0]), int(l_wrist[1]))
                                      if (c_lwrist >= REACH_WRIST_CONF and l_wrist[0] > 0) else None)
                        r_wrist_pt = ((int(r_wrist[0]), int(r_wrist[1]))
                                      if (c_rwrist >= REACH_WRIST_CONF and r_wrist[0] > 0) else None)

                        students.append({
                            "track_id":        track_id,
                            "shoulder_center": shoulder_center,
                            "shoulder_width":  shoulder_width,
                            "spine_tilt":      spine_tilt,
                            "nose":            nose,
                            "l_wrist":         l_wrist_pt,
                            "r_wrist":         r_wrist_pt,
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
                        # Baseline = MEDIAN over the calibration window (not one
                        # noisy frame).  A pitch term (nose height above the
                        # shoulder line) separates "looking down at the paper"
                        # from "turning sideways".
                        person_turn    = "FORWARD"
                        smoothed_ratio = 0.0
                        adjusted_ratio = 0.0
                        down_delta     = 0.0

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
                            vertical_offset = d_x_head * sin_t + d_y_head * cos_t   # + = below shoulder line
                            raw_ratio       = sideways_offset / (shoulder_width / 2.0)
                            raw_height      = -vertical_offset / shoulder_width      # + = nose above shoulder line

                            if track_id not in calib_start:
                                calib_start[track_id]    = current_time
                                calib_samples[track_id]  = []
                                stable_since[track_id]   = current_time
                                stable_anchor[track_id]  = raw_ratio

                            if current_time - calib_start[track_id] < CALIBRATION_DURATION:
                                samples = calib_samples.setdefault(track_id, [])
                                samples.append((raw_ratio, raw_height))
                                baseline_ratio[track_id]  = float(np.median([x[0] for x in samples]))
                                baseline_height[track_id] = float(np.median([x[1] for x in samples]))
                                stable_anchor[track_id]   = baseline_ratio[track_id]
                                person_turn = "CALIBRATING"
                            else:
                                adjusted_ratio = raw_ratio - baseline_ratio[track_id]

                                if track_id not in turn_history:
                                    turn_history[track_id] = deque(maxlen=SMOOTHING_WINDOW)
                                turn_history[track_id].append(adjusted_ratio)
                                smoothed_ratio = (sum(turn_history[track_id])
                                                  / len(turn_history[track_id]))

                                if track_id not in pitch_history:
                                    pitch_history[track_id] = deque(maxlen=SMOOTHING_WINDOW)
                                pitch_history[track_id].append(raw_height)
                                smoothed_height = (sum(pitch_history[track_id])
                                                   / len(pitch_history[track_id]))
                                down_delta   = baseline_height[track_id] - smoothed_height
                                looking_down = down_delta > PITCH_DOWN_DELTA

                                if smoothed_ratio < -HEAD_TURN_RATIO_THRESHOLD:
                                    person_turn = "LOOKING LEFT"
                                elif smoothed_ratio > HEAD_TURN_RATIO_THRESHOLD:
                                    person_turn = "LOOKING RIGHT"

                                if (person_turn != "FORWARD" and looking_down
                                        and abs(smoothed_ratio) < HEAD_TURN_STRONG_RATIO):
                                    person_turn = "LOOKING DOWN"   # reading / writing, not a violation

                                # Held posture: flagged for ages => new normal.
                                if (track_id in gaze_timers and
                                        current_time - gaze_timers[track_id] >= GAZE_REBASELINE_SEC):
                                    baseline_ratio[track_id]  = raw_ratio
                                    baseline_height[track_id] = raw_height
                                    turn_history[track_id].clear()
                                    pitch_history[track_id].clear()
                                    gaze_timers.pop(track_id, None)
                                    stable_since[track_id]    = current_time
                                    stable_anchor[track_id]   = raw_ratio
                                    person_turn = "FORWARD"
                                    print(f"[{camera_id}] ID {track_id}: held posture for "
                                          f"{GAZE_REBASELINE_SEC:.0f}s, re-baselined.")
                                else:
                                    # Slow repositioning: baseline drifts toward a
                                    # steadily-held moderate shift.
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
                                                    raw_ratio - baseline_ratio[track_id])
                                                baseline_height[track_id] += BASELINE_ADAPT_RATE * (
                                                    raw_height - baseline_height[track_id])

                        if DEBUG_OVERLAY and person_turn != "CALIBRATING":
                            cv2.putText(frame, f"yaw {smoothed_ratio:+.2f} dn {down_delta:+.2f}",
                                        (shoulder_center[0] - 45, shoulder_center[1] + 42),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        students[-1]["person_turn"] = person_turn
                        last_turn = entry.setdefault("last_turn", {})
                        if last_turn.get(track_id) != person_turn:
                            print(f"[{camera_id}] ID {track_id}: {last_turn.get(track_id, '-')} -> "
                                  f"{person_turn}  yaw={smoothed_ratio:+.2f} dn={down_delta:+.2f}")
                            last_turn[track_id] = person_turn
                        active_ids.append(track_id)

                        gaze_candidate = (DETECTION_ENABLED["gaze"]
                                          and person_turn not in ("FORWARD", "CALIBRATING", "LOOKING DOWN")
                                          and _neighbor_in_direction(person_turn, track_id,
                                                                     candidates, kept_idx))
                        if (DETECTION_ENABLED["gaze"] and not gaze_candidate
                                and person_turn in ("LOOKING LEFT", "LOOKING RIGHT")):
                            cv2.putText(frame, "turn ignored: no neighbor",
                                        (shoulder_center[0] - 70, max(20, shoulder_center[1] - 40)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                        if gaze_candidate:
                            entry["gaze_lost"].pop(track_id, None)
                            if track_id not in gaze_timers:
                                gaze_timers[track_id] = current_time
                            elapsed = current_time - gaze_timers[track_id]

                            if elapsed >= GAZE_DURATION_THRESHOLD:
                                if not entry.get("_gaze_logged", {}).get(track_id):
                                    entry.setdefault("_gaze_logged", {})[track_id] = True
                                    print(f"[{camera_id}] ID {track_id}: GAZE ALERT ({person_turn}, {elapsed:.1f}s)")
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
                            in_grace = False
                            if track_id in gaze_timers and person_turn != "CALIBRATING":
                                entry["gaze_lost"].setdefault(track_id, current_time)
                                in_grace = (current_time - entry["gaze_lost"][track_id]) < GAZE_GRACE_SEC
                            if not in_grace:
                                gaze_timers.pop(track_id, None)
                                turn_history.pop(track_id, None)
                                entry["gaze_lost"].pop(track_id, None)
                                entry.get("_gaze_logged", {}).pop(track_id, None)
                            if person_turn == "LOOKING DOWN":
                                cv2.putText(frame, "looking down (ok)",
                                            (shoulder_center[0] - 55,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                            if person_turn == "CALIBRATING":
                                cv2.putText(frame, "CALIBRATING...",
                                            (shoulder_center[0] - 55,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            except Exception as e:
                print(f"[{camera_id}] Pose estimation error: {e}")
                students   = []
                active_ids = []

            # Prune per-track state only after a track has been gone for
            # TRACK_LOST_GRACE_SEC, so one missed frame doesn't wipe a
            # student's baseline and force a re-calibration.
            for tid in active_ids:
                track_last_seen[tid] = current_time
            for tid in list(track_first_seen):
                track_last_seen.setdefault(tid, current_time)
            expired = [tid for tid, ts in track_last_seen.items()
                       if current_time - ts > TRACK_LOST_GRACE_SEC]
            for tid in expired:
                for key in PER_TRACK_KEYS:
                    entry[key].pop(tid, None)

            # 2.4 WRIST-CROP PHONE PASS: small phones held in hands
            _merge_wrist_phones(entry, camera_id, clean_frame, students,
                                current_time, object_detections)

            established_ids = {tid for tid, seen in track_first_seen.items()
                               if current_time - seen >= TRACK_MIN_AGE_SEC}

            # ==========================================
            # 2.6 CAMERA-BUMP DETECTION
            # ==========================================
            scored = [s for s in students if s["person_turn"] != "CALIBRATING"]
            if len(scored) >= CAMERA_BUMP_MIN_STUDENTS:
                turned = sum(1 for s in scored
                             if s["person_turn"] in ("LOOKING LEFT", "LOOKING RIGHT"))
                if (turned / len(scored) >= CAMERA_BUMP_RATIO
                        and current_time - entry["last_bump_time"] >= CAMERA_BUMP_COOLDOWN_SEC):
                    print(f"[{camera_id}] Possible camera disturbance: {turned}/{len(scored)} "
                          f"students turned at once. Recalibrating all baselines.")
                    entry["last_bump_time"] = current_time
                    entry["calibration_start"].clear()
                    entry["gaze_timers"].clear()
                    entry["turn_history"].clear()

            # ==========================================
            # 2.5. SMARTWATCH FILTER
            # ==========================================
            clock_dets    = [(i, d) for i, d in enumerate(object_detections)
                             if d["name"] == "SMARTWATCH"]
            clock_centers = [d["center"] for _, d in clock_dets]
            suppressed    = _check_static_clocks(entry, clock_centers, current_time)

            # Stationary "phones" (seat-number plates etc.).  Wrist-crop detections carry
            # a track_id and are exempt.
            phone_dets = [(i, d) for i, d in enumerate(object_detections)
                          if d["name"] == "CELL PHONE" and d.get("track_id") is None]
            phone_sup  = _check_static_clocks(entry, [d["center"] for _, d in phone_dets],
                                               current_time, store="static_phones",
                                               static_sec=STATIC_PHONE_SEC)
            static_phone_idx = {phone_dets[k][0] for k in phone_sup}

            accepted_detections = []
            for i, det in enumerate(object_detections):
                x1, y1, x2, y2 = det["bbox"]

                if det["name"] != "SMARTWATCH":
                    if i in static_phone_idx and not _phone_in_hand(det, students):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
                        cv2.putText(frame, "STATIC OBJECT (suppressed)", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)
                        continue
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"CELL PHONE ({det['confidence']*100:.0f}%)",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    accepted_detections.append(det)
                    continue

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
            # check.  Known limitation; state it if asked.
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
                if best_det.get("track_id") is not None:   # found in a wrist crop
                    violating_track_id = best_det["track_id"]

            # ==========================================
            # 4. INTERACTION & DESK LEANING / REACH DETECTION
            # ==========================================
            interaction_candidates = [s for s in students if s["track_id"] in established_ids]
            active_pairs = []
            active_reach_pairs = []
            num_students = len(interaction_candidates)
            for i in range(num_students):
                for j in range(i + 1, num_students):
                    s1, s2 = interaction_candidates[i], interaction_candidates[j]

                    dist           = math.hypot(
                        s1["shoulder_center"][0] - s2["shoulder_center"][0],
                        s1["shoulder_center"][1] - s2["shoulder_center"][1])
                    avg_shoulder_w  = (s1["shoulder_width"] + s2["shoulder_width"]) / 2.0

                    # Safety net: if a ghost skeleton still slipped past the
                    # pre-pass filter, two "students" occupying the same
                    # body are one person, not a pair.
                    if dist < min(s1["shoulder_width"], s2["shoulder_width"]) * DUPLICATE_DIST_MULTIPLIER:
                        continue

                    # (student-interaction / proximity detection removed: gaze + reach cover it)

                    # --- Note-passing / reach check, both directions. ---
                    if dist > avg_shoulder_w * REACH_MAX_PAIR_DIST_MULTIPLIER:
                        continue
                    same_row = abs(s1["shoulder_center"][1] - s2["shoulder_center"][1]) \
                               < avg_shoulder_w * REACH_SAME_ROW_MULTIPLIER

                    depth_ratio = s1["shoulder_width"] / max(1.0, s2["shoulder_width"])
                    side_gap = abs(s1["shoulder_center"][0] - s2["shoulder_center"][0])
                    if (not same_row
                            or not (REACH_DEPTH_RATIO_MIN <= depth_ratio <= REACH_DEPTH_RATIO_MAX)
                            or side_gap < avg_shoulder_w * REACH_MIN_SIDE_GAP_MULTIPLIER):
                        continue   # different row / depth: 2D geometry can't tell a hand-off from overlap
                    for reacher, target in ((s1, s2), (s2, s1)):
                        reach_key   = (reacher["track_id"], target["track_id"])
                        reach_wrist = _find_reaching_wrist(reacher, target, same_row)

                        if reach_wrist is not None:
                            active_reach_pairs.append(reach_key)
                            if reach_key not in reach_timers:
                                reach_timers[reach_key] = {"start": current_time, "last": current_time}
                            reach_timers[reach_key]["last"] = current_time
                            r_elapsed = current_time - reach_timers[reach_key]["start"]

                            if r_elapsed >= REACH_DURATION_THRESHOLD and DETECTION_ENABLED["reach"]:
                                reach_violation = True
                                if not violation_reason:
                                    violation_reason     = "REACHING_NEIGHBOR_ZONE"
                                    violation_confidence = 0.85
                                    violating_track_id   = reacher["track_id"]

                                cv2.line(frame, reach_wrist, target["shoulder_center"],
                                         (0, 140, 255), 2)
                                cv2.putText(frame, "REACH",
                                            (reach_wrist[0] - 20, max(20, reach_wrist[1] - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            entry["interaction_timers"] = {k: v for k, v in pair_timers.items()
                                           if k in active_pairs}
            # Keep a reach timer alive through brief keypoint dropouts.
            for k in [k for k, v in reach_timers.items()
                      if k not in active_reach_pairs
                      and current_time - v["last"] > REACH_GRACE_SEC]:
                reach_timers.pop(k, None)

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
            elif reach_violation:
                status_text  = "ALERT: SUSPECTED NOTE PASSING!"
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

            is_violating = object_violation or interaction_violation or reach_violation or gaze_trigger

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

            ret, buf = cv2.imencode('.jpg', frame)
            if ret:
                entry["latest_frame"] = buf.tobytes()

    finally:
        _finish_recording(entry)


# ==========================================
# HTTP FRAME STREAMING
# ==========================================
def generate_frames(camera_id):
    """Serves the latest annotated JPEG produced by _detection_loop.  No YOLO
    work happens here, so multiple browser tabs don't interfere."""
    while True:
        entry = camera_registry.get(camera_id)
        if entry is None:
            break

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
# ACCESS CONTROL
# ==========================================
# Per-user accounts (see auth.py).  A persistent CERBERUS_SECRET_KEY is
# optional -- without one, a new random key is generated each run and
# everyone is signed out on restart.
app.secret_key = os.environ.get("CERBERUS_SECRET_KEY") or secrets.token_hex(32)

PUBLIC_ENDPOINTS = {"login", "signup"}


@app.before_request
def _require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return
    if request.endpoint == "static":
        return
    if not session.get("user_id"):
        return redirect(url_for("login"))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        user = verify_user(request.form.get('username', ''), request.form.get('password', ''))
        if user:
            session['user_id']  = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
        error = "Incorrect username or password."
    return render_template('login.html', error=error)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        uid, error = create_user(
            request.form.get('username', ''),
            request.form.get('password', ''),
            request.form.get('invite_code', ''),
        )
        if uid:
            user = verify_user(request.form.get('username', ''), request.form.get('password', ''))
            session['user_id']  = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
    return render_template('signup.html', error=error)


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
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

    if raw_source.isdigit():
        source   = int(raw_source)
        cam_type = "local"

    elif os.path.isfile(raw_source):
        # Video file path; loops on EOF for continuous demo replay.
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
    print(f"[session] monitoring started, rules: {DETECTION_ENABLED}")
    return jsonify({"status": "started"})


@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    return jsonify({"status": "stopped"})


# Reset Session permanently deletes every recording, thumbnail, and DB row
# for the session, so the operator must TYPE a fixed phrase and the backend
# independently checks it.
RESET_CONFIRM_PHRASE = "DELETE ALL DATA"


@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Deletes all logged violations, incidents, recordings, and evidence
    photos for the current session, and clears in-memory tracking state for
    every registered camera.  Requires monitoring to be paused first, and
    the exact confirmation phrase in the request body."""
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

    print(f"[session] {session.get('username', 'unknown')} reset session '{CURRENT_SESSION}'")

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
        _clear_tracking_state(entry)

    return jsonify({"status": "reset", "session_id": CURRENT_SESSION})


# ==========================================
# PER-EXAM SESSIONS
# ==========================================

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
    """Lets a reviewer mark a flag as a false positive without deleting the
    underlying evidence -- the incident stays in the record, tagged dismissed."""
    dismiss_incident(inc_id, True)
    print(f"[incidents] {session.get('username', 'unknown')} dismissed incident #{inc_id}")
    return jsonify({"status": "dismissed", "id": inc_id})


@app.route('/recordings/<path:filepath>')
def serve_recording(filepath):
    # conditional=True enables HTTP Range requests, which <video> needs for seeking.
    return send_from_directory(RECORDINGS_FOLDER, filepath, conditional=True)


@app.route('/settings', methods=['GET', 'POST'])
def settings_route():
    """Read or change which detection rules are active.  Changes are logged
    with the username so a reviewer can see why something was not flagged."""
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
        records = get_recent_violations(CURRENT_SESSION, limit=100)
    except Exception as e:
        print(f"[DB ERROR in system_stats]: {e}")
        records = []

    phone_count = interaction_count = reach_count = gaze_count = 0
    for r in records:
        v_type = str(r.get('violation_type', '')).upper()
        if "OBJECT" in v_type or "PHONE" in v_type or "WATCH" in v_type:
            phone_count += 1
        elif "REACH" in v_type:
            reach_count += 1
        elif "INTERACTION" in v_type:
            interaction_count += 1
        elif "GAZE" in v_type:
            gaze_count += 1

    return jsonify({
        "monitoring":       monitoring_active,
        "total_violations": len(records),
        "phone":            phone_count,
        "interaction":      interaction_count,
        "reach":            reach_count,
        "gaze":             gaze_count,
        "rules":            dict(DETECTION_ENABLED),
        "session_id":       CURRENT_SESSION,
    })


if __name__ == '__main__':
    # debug=False: the reloader can double-import this module and double-open
    # cameras.  threaded=True: each /video_feed is a long-lived connection.
    # Default host is localhost only; set CERBERUS_HOST=0.0.0.0 to allow
    # other devices on the network to reach the dashboard.
    app.run(debug=False, threaded=True,
            host=os.environ.get("CERBERUS_HOST", "127.0.0.1"), port=5000)