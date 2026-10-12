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
try:
    import torch
    _CUDA = bool(torch.cuda.is_available())
    _GPU_NAME = torch.cuda.get_device_name(0) if _CUDA else ""
except Exception:
    _CUDA, _GPU_NAME = False, ""
DEVICE = 0 if _CUDA else "cpu"   # NVIDIA GPU when CUDA works, otherwise CPU
USE_HALF = _CUDA                 # fp16 only makes sense on a CUDA GPU


from database import init_db, log_violation, get_recent_violations, clear_violations
import phone_verify as _pv   # r47: Stage 2 phone verification
from recording import (SessionRecorder, IncidentTracker, init_incident_db,
                       list_incidents, list_incident_sessions, clear_incidents, dismiss_incident,
                       cleanup_orphaned_recordings, RECORDINGS_FOLDER)
from auth import (create_user, get_user_by_id, init_auth_db, verify_user,
                  generate_totp_secret, get_totp_secret, save_totp_secret, verify_totp,
                  delete_user)

BUILD_ID = "2026-10-11-r47-phone-verify"
print(f"[cerberus] app.py build {BUILD_ID}")
_where = ("GPU: " + _GPU_NAME) if _CUDA else "CPU (torch.cuda.is_available() is False)"
print(f"[cerberus] inference on {_where}{', half precision' if USE_HALF else ''}")

app = Flask(__name__)

init_db()
init_incident_db()
init_auth_db()
cleanup_orphaned_recordings()  # finalise any half-written recordings from a previous crash

VIOLATION_FOLDER = os.path.join('static', 'violations')
os.makedirs(VIOLATION_FOLDER, exist_ok=True)

CURRENT_SESSION = "SESSION_ROOM_A1"

monitoring_active = False  # session-wide arm/disarm toggle
HEAD_TURN_RATIO_THRESHOLD = 0.55  # horizontal nose-displacement ratio, not a raw angle
HEAD_TURN_EXIT_RATIO = 0.40  # once turned, stay turned until |yaw| falls below this
GAZE_DURATION_THRESHOLD = 1.5
ATTRIBUTION_RADIUS_MULTIPLIER = 4.0
MIN_KEYPOINT_CONF = 0.5
SHOULDER_MIN_CONF = 0.3   # shoulders only: lets partly hidden back-row students be tracked
SMOOTHING_WINDOW = 5
BASELINE_MAX_ABS = 0.45   # a 'neutral' pose further off-axis than this was a turned head

# Detection rules — toggled live via POST /settings.
DETECTION_ENABLED = {
    "phone":       True,
    "reach":       True,
    "gaze":        True,
}

# --- WRIST-CROP PHONE PASS ---
WRIST_CROP_ENABLED      = True
WRIST_CROP_SCALE        = 1.6    # crop width = this x the student's shoulder width
WRIST_CROP_MIN_HALF_PX  = 32     # never crop smaller than 64 px across
WRIST_CROP_IMGSZ        = 640    # crops are upscaled to this before inference
WRIST_CROP_CONF         = 0.80  # lower than the full-frame 0.45; persistence compensates
WRIST_CROP_INTERVAL_SEC = 0.33   # run ~3x per second, not every frame
WRIST_CROP_MAX_CROPS    = 40     # cap per pass (large rooms: raise on a GPU)
WRIST_CROP_MIN_HITS     = 3      # hits needed ...
WRIST_CROP_WINDOW_SEC   = 2.0    # ... inside this window
WRIST_CROP_HOLD_SEC     = 0.8    # keep a confirmed phone alive between passes
WRIST_CROP_DEDUP_PX     = 50     # same phone as a full-frame detection if this close
WRIST_CROP_MODEL        = "best.pt"  # bigger model for the zoomed-in pass only; "yolo11n.pt" = old/faster
PHONE_MODEL = WRIST_CROP_MODEL   # full-frame pass uses the same trained phone model -> only 2 models load (set "yolo11n.pt" to go back)
WRIST_CROP_CLASSES      = [0]   # class 0 = cell phone in the trained model

# --- PHONE FRAME SKIP (r46) ---
PHONE_FULL_EVERY_N = 2   # full-frame phone pass runs on every Nth frame (1 = every frame)

# --- STAGE-2 PHONE VERIFIER (r47, see phone_verify.py) ---
# "off"    = old pen/badge/torso filters only (r46 behaviour)
# "shadow" = old filters still decide; the verifier only draws its verdict next to each candidate
# "on"     = the verifier decides; the old pen/badge/torso filters are skipped for phones
PHONE_VERIFY = "off"
PHONE_VERIFY_DUMP = True             # True: save candidate crops for calibration (python phone_verify.py --probe)
PHONE_VERIFY_DUMP_DIR = "phone_candidates"
PHONE_FULL_CONF = 0.80        # full-frame phone pass: confidence needed (was 0.45)
# --- HAND-RISE CONDITION (r48) ---
# A phone only counts if the student's hand was recently down low or out of sight
# (pocket / lap / bag), i.e. it "came from somewhere". Set False to switch off.
PHONE_NEEDS_HAND_RISE = True
HAND_LOW_SW           = 1.0    # wrist this far below the shoulder line (x shoulder width) = "low"
HAND_LOW_MIN_SEC      = 0.3    # must stay low / out of sight this long to count
HAND_RISE_MEMORY_SEC  = 10.0   # a phone may appear this long after the hand was low
HAND_RISE_KEEP_SEC    = 2.0    # an accepted phone stays allowed while it keeps being detected
PHONE_FULL_NEEDS_HAND = True  # full-frame phone only counts if a hand is near it

# --- WRIST-CROP REST SKIP (r46) ---
WRIST_CROP_REST_SKIP        = True   # skip a crop if the last check found nothing and the hand has not moved
WRIST_CROP_REST_MOVE_SW     = 0.25   # "has not moved" = less than this x shoulder width
WRIST_CROP_REST_RECHECK_SEC = 2.0    # ...but look again at least this often

# --- HANDS-TOGETHER PHONE CROP + WEAK-EVIDENCE PATH (r34) ---
WRIST_HANDS_CROP         = True   # extra zoomed crop centred between both hands
WRIST_HANDS_MAX_GAP_SW   = 1.3    # only when the wrists are this close (x shoulder width)
WRIST_HANDS_MIN_HALF_SW  = 0.60   # hands-crop half-size is at least this x shoulder width
WRIST_WEAK_CONF          = 0.99   # faint detections count only in the hands-at-chest posture
WRIST_WEAK_MIN_HITS      = 4
WRIST_WEAK_WINDOW_SEC    = 3.0
WRIST_WEAK_MAX_GAP_SW    = 0.9    # wrists no further apart than this ...
WRIST_WEAK_MIN_DROP_SW   = 0.15   # ... and both between this far below the shoulder line
WRIST_WEAK_MAX_DROP_SW   = 1.6    #     and this far (above the desk/lap)

# --- PEN FILTER (phones) ---
PHONE_MAX_ASPECT   = 3.5    # box longer than 3x its width is a pen, not a phone
PHONE_MIN_SHORT_SW = 0.08   # box short side must be >= this x shoulder width
PHONE_TINY_SW        = 0.18   # long side below this x shoulder width: always rejected
PHONE_SMALL_SW       = 0.32   # long side below this: needs high confidence
PHONE_SMALL_MIN_CONF = 0.60
PHONE_MAX_LONG_SW   = 0.80   # long side above this x shoulder width: too big to be a phone

# --- ID-BADGE / LANYARD FILTER (phones) ---
# A badge hangs on the chest midline. Any 'phone' centred in that strip is
# ignored unless a wrist is almost touching it. All values are x shoulder width.
BADGE_FILTER_ENABLED = True
BADGE_ZONE_HALF_W    = 0.60   # strip half-width around the shoulder-centre x
BADGE_ZONE_TOP       = -0.10   # strip starts this far below the shoulder line
BADGE_ZONE_BOTTOM    = 1.70   # ...and ends this far below it (above the lap)
BADGE_HAND_HOLD_MULT = 0.30   # wrist this close to the box centre = held phone
BADGE_TWO_HAND_MULT  = 0.50   # BOTH wrists this close to the box = phone in two hands
# Appearance test: an ID card is mostly white / low-saturation pixels.
BADGE_WHITE_V    = 165    # HSV value above this ...
BADGE_WHITE_S    = 70     # ... and saturation below this = "white" pixel
BADGE_WHITE_FRAC = 0.28   # box with at least this share of white pixels = ID card
                          # (lower it if cards still get through; raise it if
                          #  white-cased phones in the chest area get ignored)

# --- NEIGHBOR-AWARE SIDE GAZE ---
GAZE_REQUIRE_NEIGHBOR = True
GAZE_NEIGHBOR_MIN_DX  = 0.6   # neighbor must be at least this far to the side
GAZE_NEIGHBOR_MAX_DX  = 5.0   # ...and no farther than this
GAZE_NEIGHBOR_ROW_MULT = 2.0  # max vertical offset between shoulder centers
GAZE_NEIGHBOR_ROW_SLOPE = 0.30   # extra row tolerance per shoulder-width of distance (diagonal cameras)
GAZE_NEIGHBOR_MEMORY_SEC = 8.0   # a neighbour seen on that side this recently still counts

# --- POSE MODEL INFERENCE SETTINGS ---
POSE_IOU  = 0.5
POSE_CONF = 0.25
POSE_IMGSZ = 960   # try 960 if back-row students are small and FPS allows

# --- DUPLICATE-SKELETON (GHOST TRACK) FILTER ---
DUPLICATE_DIST_MULTIPLIER = 0.8

# --- ADAPTIVE "NORMAL POSITION" BASELINE ---
CALIBRATION_DURATION = 2.0          # seconds to sample a student's initial "neutral" pose
BASELINE_ADAPT_RATE = 0.02          # how fast the baseline creeps toward a newly-held position
REPOSITION_STABILITY_WINDOW = 10.0  # student must hold a new position this long before baseline shifts
REPOSITION_JUMP_LIMIT = HEAD_TURN_RATIO_THRESHOLD * 0.8
REPOSITION_SETTLE_BAND = HEAD_TURN_RATIO_THRESHOLD * 0.3

# --- STATIC-OBJECT FILTER (phones) ---
# Suppresses stationary detections that are seat plates or furniture, not phones.
STATIC_OBJECT_RADIUS_PX   = 60
STATIC_OBJECT_MOVE_PX     = 35
WRIST_CROP_MAX_WRIST_DIST = 1.2   # phone centre must be within this x shoulder width of the wrist
WRIST_CROP_STATIC_SEC     = 60.0   # a wrist-crop 'phone' this still ...
WRIST_CROP_STATIC_MOVE_PX = 8     # ... (moves less than this) is furniture
HELD_PHONE_WRIST_MULT     = 0.6   # full-frame static filter: a hand this close keeps the phone flagged
STATIC_PHONE_SEC          = 30.0  # a full-frame 'phone' that never moves this long is furniture,
                                  # unless a hand is holding it

# --- CAMERA-BUMP DETECTION ---
CAMERA_BUMP_RATIO        = 0.8
CAMERA_BUMP_MIN_STUDENTS = 6
CAMERA_BUMP_COOLDOWN_SEC = 5.0

# --- TRACK-AGE GATE ---
TRACK_MIN_AGE_SEC = 1.0

# --- NOTE-PASSING / REACHING DETECTION ---
# Two triggers (either counts): A) wrist enters a circle around the neighbour's
# shoulder; B) arm stretched out toward a side-by-side neighbour (same-row only).
REACH_ZONE_RADIUS_MULTIPLIER    = 1.5
REACH_ZONE_MIN_EXTENSION = 1.0   # wrist must be this many shoulder-widths from its own shoulders
REACH_EXTENSION_MULTIPLIER      = 1.6
REACH_PROGRESS_MIN              = 0.35
REACH_SAME_ROW_MULTIPLIER       = 1.0    # max vertical offset between shoulder centers, x avg shoulder width
REACH_MAX_PAIR_DIST_MULTIPLIER  = 8.0    # generous: the extension test does the real filtering
REACH_DURATION_THRESHOLD        = 1.0
REACH_GRACE_SEC                 = 0.4    # tolerate a dropped wrist keypoint for this long
REACH_WRIST_CONF                = 0.3    # accept lower conf; wrists are blurry mid-motion
REACH_DEPTH_RATIO_MIN = 0.75  # shoulder-width ratio band — outside this means different rows
REACH_DEPTH_RATIO_MAX = 1.33
REACH_MIN_SIDE_GAP_MULTIPLIER = 1.2  # students must be side-by-side, not front-to-back
# Mutual hand-off: a wrist of EACH student out in the middle band of the gap.
REACH_MID_MIN_PROGRESS  = 0.30   # fraction of the way from own shoulders to the neighbour's
REACH_MID_MAX_PROGRESS  = 0.70
REACH_MID_MIN_EXTENSION = 0.80   # wrist at least this many shoulder-widths from own shoulders
REACH_MID_MAX_DY        = 2.00   # ...and no more than this far above/below the shoulder line

# --- HEAD-POSE ROBUSTNESS ---
# Pitch term: nose dropping below baseline => LOOKING DOWN (paper), not a side gaze.
PITCH_DOWN_DELTA = 0.30
# Strong yaw overrides the pitch excuse — only moderate readings can be dismissed.
HEAD_TURN_STRONG_RATIO = 0.7

# --- r38: leaky-bucket gaze timer ---
GAZE_DRAIN_RATE = 0.5          # seconds of countdown lost per second spent NOT gazing

# --- r38: hands-together phone posture (fallback when the detector sees nothing) ---
HANDS_POSTURE_ENABLED  = False
HANDS_POSTURE_SEC      = 3.0   # hold this long before flagging
HANDS_POSTURE_GRACE_SEC = 1.5  # brief dropouts don't reset the hold
HANDS_POSTURE_MAX_GAP_SW = 0.75   # wrists no further apart than this x shoulder width
HANDS_POSTURE_MIN_DROP_SW = 0.15  # both wrists this far below the shoulder line ...
HANDS_POSTURE_MAX_DROP_SW = 0.75  # ... but no lower (writing on the desk is lower than this)


# Motion test: an ID card stays put relative to the shoulders; a phone follows the hand.
BADGE_LOCK_WINDOW_SEC = 3.0    # history kept per student
BADGE_LOCK_MIN_SEC    = 1.2    # need this much history before deciding ("CHECKING" until then)
BADGE_LOCK_STD        = 0.05   # offset-from-shoulders std below this (x shoulder width) = steady
BADGE_LOCK_MIN_REL    = 0.10   # ... and offset-from-wrist std must exceed this ...
BADGE_LOCK_RATIO      = 3.0    # ... and be this many times the shoulder-offset std

# --- r37: looking-down must not cancel a running side-gaze ---
GAZE_DOWN_HYSTERESIS  = 2.5    # while a gaze timer runs, head-drop needed for 'down' is this x PITCH_DOWN_DELTA

# --- r37: wrist-crop phone pass ---
WRIST_CROP_FLOOR_CONF = 0.35   # inference floor (the real thresholds are applied afterwards)
WRIST_CROP_AUGMENT    = False   # test-time augmentation (flip/scales); auto-disables if unsupported

# If flagged this long it's a held posture (leaning to write); re-baseline silently.
GAZE_REBASELINE_SEC = 30.0
TRACK_LOST_GRACE_SEC = 1.5  # keep per-track state briefly after a track vanishes
DEBUG_OVERLAY = True        # draw yaw/pitch numbers on frame for threshold tuning
VIDEO_CLOCK = True   # video files: timers follow video time, not wall-clock time
RECORD_RAW = False   # True: record the clean frame (no overlays) so a run can be replayed as a 'file' camera; /review then shows no boxes

# ==========================================
# DYNAMIC CAMERA REGISTRY
# ==========================================
# --- LIVE-CAMERA SPEED (r46) ---
THREADED_CAPTURE    = True   # live cameras: a background thread always holds the newest frame
SMOOTH_DISPLAY      = True   # live cameras: video runs at camera speed, boxes lag a little behind
OVERLAY_MAX_AGE_SEC = 3.0    # stop drawing old boxes if detection stalls longer than this

camera_registry = {}
registry_lock = threading.Lock()
_next_camera_num = 1

# Class 0 of the trained phone model (best.pt) is "cell phone".
PROHIBITED_CLASSES = {0: "CELL PHONE"}


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


class ThreadedCapture:
    """Stand-in for cv2.VideoCapture on LIVE cameras.

    A background thread keeps reading, so the camera never queues old frames and
    read() always returns the NEWEST frame.  latest() lets the display thread show
    video at camera speed while the models are still busy."""

    def __init__(self, source):
        self._cap = cv2.VideoCapture(source)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0        # frames received so far
        self._taken = 0      # last frame number handed out by read()
        self._failed = False
        self._stop = threading.Event()
        self._thread = None
        if self._cap.isOpened():
            self._thread = threading.Thread(target=self._run, daemon=True, name="grabber")
            self._thread.start()

    def _run(self):
        misses = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok and frame is not None:
                misses = 0
                with self._lock:
                    self._frame = frame
                    self._seq += 1
                    self._failed = False
            else:
                misses += 1
                if misses >= 3:
                    with self._lock:
                        self._failed = True
                time.sleep(0.05)

    def isOpened(self):
        return self._cap.isOpened()

    def grab(self):
        return True          # the background thread already drains the camera

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, value):
        return self._cap.set(prop, value)

    def read(self, timeout=3.0):
        """(True, newest frame not handed out yet) or (False, None) if the camera went quiet."""
        end = time.time() + timeout
        while not self._stop.is_set():
            with self._lock:
                if self._frame is not None and self._seq != self._taken:
                    self._taken = self._seq
                    return True, self._frame.copy()
                if self._failed:
                    return False, None
            if time.time() > end:
                return False, None
            time.sleep(0.004)
        return False, None

    def latest(self):
        """(frame number, newest frame) for display.  Copy it before drawing on it."""
        with self._lock:
            return self._seq, self._frame

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._thread is None or not self._thread.is_alive():
            self._cap.release()


def _open_capture(source, cam_type):
    """Live cameras get the background reader; video files stay a plain VideoCapture
    so every frame is processed (nothing is skipped while testing on files)."""
    if cam_type != "file" and THREADED_CAPTURE:
        return ThreadedCapture(source)
    cap = cv2.VideoCapture(source)
    if cam_type != "file":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _display_loop(camera_id, gen):
    """Smooth live view: the newest camera frame with the boxes/text of the latest
    detection pasted on top.  The detection loop only hands over the pixels it drew
    (entry['_overlay']); this thread does the rest at camera speed."""
    entry = camera_registry.get(camera_id)
    if entry is None:
        return
    last_seq = -1
    while not entry["stop_event"].is_set() and entry.get("_display_gen") == gen:
        cap = entry.get("cap")
        ov = entry.get("_overlay")
        if ov is None or not isinstance(cap, ThreadedCapture):
            time.sleep(0.03)     # detection loop publishes its own frames (paused / reconnecting)
            continue
        seq, raw = cap.latest()
        if raw is None or seq == last_seq:
            time.sleep(0.004)
            continue
        last_seq = seq
        try:
            out = raw.copy()
            if time.time() - ov["t"] <= OVERLAY_MAX_AGE_SEC and ov["frame"].shape == out.shape:
                out[ov["mask"]] = ov["frame"][ov["mask"]]
            ok, buf = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                entry["latest_frame"] = buf.tobytes()
        except Exception as e:
            print(f"[{camera_id}] display error: {e}")
            time.sleep(0.1)
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
        "reach_timers":       {},
        "recorder":  SessionRecorder(CURRENT_SESSION, cam_id),
        "incidents": IncidentTracker(),
        "baseline_ratio":    {},
        "calibration_start": {},
        "stable_since":      {},
        "stable_anchor":     {},
        "latest_frame": None,
        "stop_event":   threading.Event(),
        "thread":       None,
        "static_phones":  {},
        "static_wc":      {},
        "last_bump_time": 0.0,
        "track_first_seen": {},
        "track_last_seen":  {},
        "calib_samples":   {},
        "baseline_height": {},
        "pitch_history":   {},
        "gaze_lost":       {},
        "home_ratio":      {},
        "home_height":     {},
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
            cap = _open_capture(entry["source"], entry["type"])
            if not cap.isOpened():
                cap.release()
                return False, f"Could not open camera source: {entry['source']}"
            entry["cap"] = cap

        if entry["pose_model"] is None:
            entry["pose_model"] = YOLO('yolo11n-pose.pt')
        if entry["object_model"] is None:
            try:
                entry["object_model"] = YOLO(PHONE_MODEL)
            except Exception as e:
                return False, f"Could not load phone model {PHONE_MODEL}: {e}"
            if PHONE_MODEL == WRIST_CROP_MODEL:
                entry["wrist_model"] = entry["object_model"]   # same weights: one model, loaded once
            print(f"[cerberus] phone model {PHONE_MODEL}: classes {entry['object_model'].names}")

        entry["stop_event"].clear()
        entry["_smooth_display"] = bool(SMOOTH_DISPLAY and isinstance(entry["cap"], ThreadedCapture))
        if entry["_smooth_display"]:
            gen = entry["_display_gen"] = entry.get("_display_gen", 0) + 1
            threading.Thread(target=_display_loop, args=(cam_id, gen),
                             daemon=True, name=f"display-{cam_id}").start()
        t = threading.Thread(target=_detection_loop, args=(cam_id,),
                             daemon=True, name=f"detect-{cam_id}")
        entry["thread"] = t
        entry["status"] = "active"
        t.start()

    return True, "activated"


PER_TRACK_KEYS = ("gaze_timers", "turn_history", "baseline_ratio", "calibration_start",
                  "stable_since", "stable_anchor", "track_first_seen", "track_last_seen",
                  "calib_samples", "baseline_height", "pitch_history", "gaze_lost",
                  "home_ratio", "home_height")


def _clear_tracking_state(entry):
    """Wipes every per-track dict on a camera entry."""
    entry.pop("_overlay", None)
    entry.pop("_wc_rest", None)
    entry.pop("_obj_cache", None)
    for key in PER_TRACK_KEYS + ("reach_timers", "static_phones", "static_wc"):
        entry[key].clear()
    for key in ("_wc_hits", "_wc_weak_hits", "_badge_hist", "_wc_confirmed", "_wc_last", "_wc_debug", "last_turn", "_gaze_logged", "_nb_seen", "_nb_dbg", "_turn_hold", "_pv", "_hand_low", "_phone_ok"):
        entry.pop(key, None)


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


add_camera(0, "local", label="cam1")  # default camera so the dashboard is not empty


def _neighbor_in_direction(turn, track_id, candidates, kept_idx, entry=None, now=None,
                           image_dir=None):
    """True if another (non-ghost) student sits on the side this student is
    turned toward, in roughly the same row.  `turn` is 'LOOKING LEFT' or
    'LOOKING RIGHT' (image-left / image-right, matching the yaw sign).

    Distances use the larger of the two shoulder widths (perspective), row
    tolerance grows with distance (diagonal cameras), and a neighbour seen on
    that side recently still counts (a dropped pose must not reset a gaze)."""
    if not GAZE_REQUIRE_NEIGHBOR or turn not in ("LOOKING LEFT", "LOOKING RIGHT"):
        return True
    me = next((c for c in candidates
               if c["track_id"] == track_id and c["idx"] in kept_idx), None)
    if me is None:
        return True
    if image_dir is None:
        # Fallback only. The labels are the PERSON's left/right, which is the
        # opposite image side for a student facing the camera.
        image_dir = 1 if turn == "LOOKING LEFT" else -1
    direction = image_dir   # +1 = look for a neighbour on the image-right, -1 = image-left
    side = "L" if direction < 0 else "R"
    nearest = {"L": None, "R": None}   # per side: (signed dx, dy) in shoulder widths, for the overlay
    found = False
    for c in candidates:
        if c["idx"] not in kept_idx or c["track_id"] == track_id:
            continue
        scale = max(me["width"], c["width"])
        sdx = (c["center"][0] - me["center"][0]) / scale     # + = neighbour is image-right
        dy = abs(c["center"][1] - me["center"][1]) / scale
        dx = sdx * direction
        s = "R" if sdx > 0 else "L"
        if nearest[s] is None or abs(sdx) < abs(nearest[s][0]):
            nearest[s] = (sdx, dy)
        if (GAZE_NEIGHBOR_MIN_DX <= dx <= GAZE_NEIGHBOR_MAX_DX
                and dy <= GAZE_NEIGHBOR_ROW_MULT + GAZE_NEIGHBOR_ROW_SLOPE * dx):
            found = True
    if entry is not None and now is not None:
        seen = entry.setdefault("_nb_seen", {}).setdefault(track_id, {})
        if found:
            seen[side] = now
        elif now - seen.get(side, -1e9) <= GAZE_NEIGHBOR_MEMORY_SEC:
            found = True   # a neighbour was there a moment ago
    if entry is not None:
        dbg = entry.setdefault("_nb_dbg", {})
        if found:
            dbg.pop(track_id, None)
        else:
            def fmt(k):
                return "-" if nearest[k] is None else f"{nearest[k][0]:+.1f}/{nearest[k][1]:.1f}"
            dbg[track_id] = f"no neighbor {side} (L {fmt('L')} R {fmt('R')})"
    return found


def _wrist_model(entry):
    """Model used for the zoomed-in wrist-crop pass (loaded once per camera)."""
    m = entry.get("wrist_model")
    if m is None:
        try:
            m = YOLO(WRIST_CROP_MODEL)
            print(f"[cerberus] wrist-crop model: {WRIST_CROP_MODEL}")
        except Exception as e:
            print(f"[cerberus] could not load {WRIST_CROP_MODEL} ({e}); using the nano model")
            m = entry["object_model"]
        entry["wrist_model"] = m
    return m


def _run_wrist_model(entry, crops):
    """Runs the zoomed-in phone model, with test-time augmentation when possible."""
    model = _wrist_model(entry)
    kw = dict(imgsz=WRIST_CROP_IMGSZ, conf=WRIST_CROP_FLOOR_CONF,
              classes=WRIST_CROP_CLASSES, verbose=False,
              device=DEVICE, half=USE_HALF)
    if WRIST_CROP_AUGMENT and not entry.get("_wc_aug_off"):
        try:
            return model(crops, augment=True, **kw)
        except Exception as e:
            entry["_wc_aug_off"] = True
            print(f"[cerberus] augment not usable ({e}); continuing without it")
    return model(crops, **kw)


def _wrist_can_rest(entry, key, pos, sw, now):
    """True if this wrist was checked a moment ago, held nothing, and has barely moved.
    A hand that moves (or the periodic re-check) brings it straight back."""
    if not WRIST_CROP_REST_SKIP:
        return False
    st = entry.get("_wc_rest", {}).get(key)
    if st is None or not st["empty"] or now - st["t"] >= WRIST_CROP_REST_RECHECK_SEC:
        return False
    return math.hypot(pos[0] - st["pos"][0], pos[1] - st["pos"][1]) < WRIST_CROP_REST_MOVE_SW * sw
def _wrist_crop_sample(entry, camera_id, frame, students, now):
    """One zoomed-in phone pass over every established student's wrists."""
    h, w = frame.shape[:2]
    entry["_wc_rej"] = []
    crops, meta, rest_keys = [], [], []
    for s in students:
        tid = s["track_id"]
        if now - entry["track_first_seen"].get(tid, now) < TRACK_MIN_AGE_SEC:
            continue
        half = int(max(WRIST_CROP_MIN_HALF_PX, s["shoulder_width"] * WRIST_CROP_SCALE / 2.0))
        for side, wrist in (("L", s["l_wrist"]), ("R", s["r_wrist"])):
            if wrist is None:
                continue
            if _wrist_can_rest(entry, (tid, side), wrist, s["shoulder_width"], now):
                continue   # checked a moment ago, held nothing, has not moved
            x1, y1 = max(0, wrist[0] - half), max(0, wrist[1] - half)
            x2, y2 = min(w, wrist[0] + half), min(h, wrist[1] + half)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crops.append(frame[y1:y2, x1:x2].copy())
            meta.append((tid, x1, y1, wrist, s["shoulder_width"]))
            rest_keys.append(((tid, side), wrist, s["shoulder_width"]))
            if len(crops) >= WRIST_CROP_MAX_CROPS:
                break
        if (WRIST_HANDS_CROP and s["l_wrist"] is not None and s["r_wrist"] is not None
                and len(crops) < WRIST_CROP_MAX_CROPS):
            lw, rw = s["l_wrist"], s["r_wrist"]
            gap = math.hypot(lw[0] - rw[0], lw[1] - rw[1])
            if gap <= WRIST_HANDS_MAX_GAP_SW * s["shoulder_width"]:
                mx, my = (lw[0] + rw[0]) // 2, (lw[1] + rw[1]) // 2
                hh = int(max(WRIST_CROP_MIN_HALF_PX,
                             WRIST_HANDS_MIN_HALF_SW * s["shoulder_width"],
                             gap * 0.75))
                x1, y1 = max(0, mx - hh), max(0, my - hh)
                x2, y2 = min(w, mx + hh), min(h, my + hh)
                if (x2 - x1 >= 16 and y2 - y1 >= 16
                        and not _wrist_can_rest(entry, (tid, "H"), (mx, my),
                                               s["shoulder_width"], now)):
                    crops.append(frame[y1:y2, x1:x2].copy())
                    meta.append((tid, x1, y1, (mx, my), s["shoulder_width"]))
                    rest_keys.append(((tid, "H"), (mx, my), s["shoulder_width"]))
        if len(crops) >= WRIST_CROP_MAX_CROPS:
            break
    if not crops:
        entry["_wc_debug"] = []   # nothing checked this pass: no stale debug boxes
        return
    entry["_wc_ncrops"] = len(crops)

    results = _run_wrist_model(entry, crops)
    entry["_wc_debug"] = [
        (ox, oy, ox + c.shape[1], oy + c.shape[0], tid,
         max([(float(b.conf[0]), int(b.cls[0])) for b in r.boxes], default=(0.0, -1)))
        for (tid, ox, oy, _wpt, _sw), c, r in zip(meta, crops, results)
    ]

    # remember which wrists came back empty (see _wrist_can_rest)
    for (key, pos, _sw), r in zip(rest_keys, results):
        entry.setdefault("_wc_rest", {})[key] = {"pos": pos, "empty": len(r.boxes) == 0, "t": now}
    best = {}   # track_id -> strongest phone detection this pass (full-frame coords)
    pv_by_tid = {}   # r47: track_id -> (conf, Stage-2 verdict) of its strongest candidate
    _stu = {s["track_id"]: s for s in students}
    for (tid, ox, oy, wpt, sw), r in zip(meta, results):
        for box in r.boxes:
            conf = float(box.conf[0])
            weak = False
            if conf < WRIST_CROP_CONF:
                if conf >= WRIST_WEAK_CONF and _hands_at_chest(_stu.get(tid)):
                    weak = True    # faint, but both hands are up at chest height
                else:
                    continue   # below the real threshold
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            ccx, ccy = ox + (bx1 + bx2) // 2, oy + (by1 + by2) // 2
            full_box = (ox + bx1, oy + by1, ox + bx2, oy + by2)
            rej = entry.setdefault("_wc_rej", [])
            if math.hypot(ccx - wpt[0], ccy - wpt[1]) > WRIST_CROP_MAX_WRIST_DIST * sw:
                rej.append(full_box + ("FAR FROM HAND",))
                continue   # too far from the hand to be held in it (furniture at the crop edge)
            if PHONE_VERIFY != "off":
                _pvr = _pv.store_for(entry, PHONE_VERIFY_DUMP_DIR if PHONE_VERIFY_DUMP else None,
                                     camera_id).feed(frame, full_box, conf, now, owner=tid)
                if tid not in pv_by_tid or conf > pv_by_tid[tid][0]:
                    pv_by_tid[tid] = (conf, _pvr)
            if PHONE_VERIFY != "on" and _pen_like((bx1, by1, bx2, by2), sw, conf):
                rej.append(full_box + ("PEN-LIKE",))
                continue   # long thin box: a pen, not a phone
            if PHONE_VERIFY != "on" and _badge_like((ccx, ccy), students, frame, full_box):
                rej.append(full_box + ("ID BADGE",))
                continue   # ID badge on a lanyard, not a phone
            if tid in _stu and PHONE_VERIFY != "on":
                verdict = _torso_lock_verdict(entry, _stu[tid], (ccx, ccy), now)
                if verdict != "phone":
                    rej.append(full_box + (("ID BADGE (moves with body)" if verdict == "badge"
                                            else "CHECKING..."),))
                    continue
            if not _hand_rise_ok(entry, tid, now):
                rej.append(full_box + ("NO HAND RISE",))
                continue   # hand never came up from below: not a phone pulled out
            if tid not in best or conf > best[tid]["confidence"]:
                best[tid] = {
                    "name":       "CELL PHONE",
                    "confidence": conf,
                    "center":     (ox + (bx1 + bx2) // 2, oy + (by1 + by2) // 2),
                    "bbox":       (ox + bx1, oy + by1, ox + bx2, oy + by2),
                    "weak":       weak,
                }

    hits      = entry.setdefault("_wc_hits", {})
    confirmed = entry.setdefault("_wc_confirmed", {})

    for tid, det in best.items():
        weak = det.get("weak")
        store = entry.setdefault("_wc_weak_hits", {}) if weak else hits
        window = WRIST_WEAK_WINDOW_SEC if weak else WRIST_CROP_WINDOW_SEC
        need_hits = WRIST_WEAK_MIN_HITS if weak else WRIST_CROP_MIN_HITS
        recent = [t for t in store.get(tid, []) if now - t <= window]
        recent.append(now)
        store[tid] = recent
        if len(recent) >= need_hits and (PHONE_VERIFY != "on" or
                                         pv_by_tid.get(tid, (0, {}))[1].get("verdict") == "CONFIRMED"):
            if tid not in confirmed:
                print(f"[{camera_id}] ID {tid}: phone in hand (wrist crop, "
                      f"{det['confidence']*100:.0f}%)")
            confirmed[tid] = {"det": det, "until": now + WRIST_CROP_HOLD_SEC}
            entry.setdefault("_phone_ok", {})[tid] = now

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
            _t0 = time.perf_counter()
            _wrist_crop_sample(entry, camera_id, clean_frame, students, now)
            _n = entry["_wc_n"] = entry.get("_wc_n", 0) + 1
            entry["_wc_sum"] = entry.get("_wc_sum", 0.0) + (time.perf_counter() - _t0)
            if _n % 20 == 0:
                print(f"[{camera_id}] wrist pass: {entry['_wc_sum'] / _n * 1000:.0f} ms avg, "
                      f"{entry.get('_wc_ncrops', 0)} crops last pass")

        for tid in [t for t, c in confirmed.items() if c["until"] < now]:
            del confirmed[tid]

        conf_items = list(confirmed.items())
        wc_static = _check_static_objects(
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


def _owner_of_detection(det, students, established_ids):
    """track_id of the student a detection belongs to (or None)."""
    if det.get("track_id") is not None:
        return det["track_id"]
    best, best_d = None, float("inf")
    for s in students:
        if s["track_id"] not in established_ids:
            continue
        d = math.hypot(s["shoulder_center"][0] - det["center"][0],
                       s["shoulder_center"][1] - det["center"][1])
        if d < s["shoulder_width"] * ATTRIBUTION_RADIUS_MULTIPLIER and d < best_d:
            best, best_d = s["track_id"], d
    return best


def _local_sw(det, students, fallback):
    """Shoulder width of the student whose wrist is nearest the detection."""
    cx, cy = det["center"]
    best, best_d = fallback, float("inf")
    for s in students:
        for wr in (s["l_wrist"], s["r_wrist"]):
            if wr is None:
                continue
            d = math.hypot(cx - wr[0], cy - wr[1])
            if d < s["shoulder_width"] * HELD_PHONE_WRIST_MULT and d < best_d:
                best, best_d = s["shoulder_width"], d
    return best


def _pen_like(bbox, ref_sw, conf=None):
    """True for long, thin boxes (pens) or boxes far thinner than a phone."""
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    long_side, short_side = max(bw, bh), min(bw, bh)
    if long_side / short_side > PHONE_MAX_ASPECT:
        return True
    if ref_sw is not None:
        if short_side < PHONE_MIN_SHORT_SW * ref_sw:
            return True
        if long_side > PHONE_MAX_LONG_SW * ref_sw:
            return True   # person-sized box, not a phone
        if long_side < PHONE_TINY_SW * ref_sw:
            return True
        if (conf is not None and long_side < PHONE_SMALL_SW * ref_sw
                and conf < PHONE_SMALL_MIN_CONF):
            return True
    return False


def _white_card_frac(frame, bbox):
    """Share of near-white, low-saturation pixels inside bbox (ID cards are mostly
    white; a phone body or a dark screen is not)."""
    if frame is None or bbox is None:
        return 0.0
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(fw, int(x2)), min(fh, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > BADGE_WHITE_V) & (hsv[..., 1] < BADGE_WHITE_S)
    return float(white.mean())


def _lanyard_above(frame, bbox, s):
    """True if a thin strap runs from the top of `bbox` up toward the student's
    neck and is clearly darker than the shirt beside it (a lanyard)."""
    if frame is None or bbox is None or s is None:
        return False
    try:
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        sx, sy = s["shoulder_center"]
        sw = float(s["shoulder_width"])
        top = np.array([(x1 + x2) / 2.0, float(y1)])
        neck = np.array([float(sx), float(sy) - 0.1 * sw])
        seg = neck - top
        length = float(np.hypot(seg[0], seg[1]))
        if length < 0.25 * sw:
            return False
        perp = np.array([-seg[1], seg[0]]) / length * (0.14 * sw)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def px(p):
            x = int(min(max(p[0], 1), fw - 2))
            y = int(min(max(p[1], 1), fh - 2))
            return float(gray[y - 1:y + 2, x - 1:x + 2].mean())

        darker = 0
        for t in np.linspace(0.15, 0.85, 8):
            p = top + seg * t
            side = (px(p + perp) + px(p - perp)) / 2.0
            if side - px(p) >= 18:
                darker += 1
        return darker >= 6
    except Exception:
        return False


def _badge_like(center, students, frame=None, bbox=None):
    """True if `center` is inside some student's chest strip (where an ID badge
    on a lanyard hangs) and none of that student's wrists is right on it."""
    if not BADGE_FILTER_ENABLED:
        return False
    cx, cy = center
    card_like = _white_card_frac(frame, bbox) >= BADGE_WHITE_FRAC
    for s in students:
        scx, scy = s["shoulder_center"]
        sw = s["shoulder_width"]
        if abs(cx - scx) > BADGE_ZONE_HALF_W * sw:
            continue
        if not (scy + BADGE_ZONE_TOP * sw <= cy <= scy + BADGE_ZONE_BOTTOM * sw):
            continue
        if card_like:
            return True    # white ID-card look in the chest strip: a badge, even with a hand on it
        # r41 two-hand: both hands round it = phone, regardless of any lanyard
        _both = [wr for wr in (s["l_wrist"], s["r_wrist"])
                 if wr is not None
                 and math.hypot(cx - wr[0], cy - wr[1]) < BADGE_TWO_HAND_MULT * sw]
        if len(_both) == 2:
            return False
        strap = _lanyard_above(frame, bbox, s)
        hand_limit = BADGE_HAND_HOLD_MULT * sw
        for wr in (s["l_wrist"], s["r_wrist"]):
            if wr is not None and math.hypot(cx - wr[0], cy - wr[1]) < hand_limit:
                if strap:
                    return True    # lanyard strap leads to it: a badge under a resting hand
                return False   # hand is on it: treat as a real phone
        near_both = [wr for wr in (s["l_wrist"], s["r_wrist"])
                     if wr is not None
                     and math.hypot(cx - wr[0], cy - wr[1]) < BADGE_TWO_HAND_MULT * sw]
        if len(near_both) == 2:
            return False   # both hands wrapped round it: a phone, not a badge
        return True
    return False


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


def _hands_at_chest(s):
    """True if both wrists are close together at chest height (typical of a phone
    held in two hands) -- used to accept faint detections."""
    if s is None:
        return False
    l, r = s["l_wrist"], s["r_wrist"]
    if l is None or r is None:
        return False
    sw = s["shoulder_width"]
    sy = s["shoulder_center"][1]
    if math.hypot(l[0] - r[0], l[1] - r[1]) > WRIST_WEAK_MAX_GAP_SW * sw:
        return False
    return all(WRIST_WEAK_MIN_DROP_SW * sw <= w[1] - sy <= WRIST_WEAK_MAX_DROP_SW * sw
               for w in (l, r))


def _hands_posture(s):
    """Both wrists together and raised at chest height (a phone cupped in two
    hands). Lower than HANDS_POSTURE_MAX_DROP_SW means hands on the desk."""
    l, r = s["l_wrist"], s["r_wrist"]
    if l is None or r is None:
        return False
    sw = s["shoulder_width"]
    sy = s["shoulder_center"][1]
    if math.hypot(l[0] - r[0], l[1] - r[1]) > HANDS_POSTURE_MAX_GAP_SW * sw:
        return False
    return all(HANDS_POSTURE_MIN_DROP_SW * sw <= w[1] - sy <= HANDS_POSTURE_MAX_DROP_SW * sw
               for w in (l, r))


def _torso_lock_verdict(entry, s, center, now):
    """'phone' | 'badge' | 'pending' for a candidate at `center` owned by student `s`.

    Only candidates inside the student's chest strip are judged. A badge keeps a
    steady offset from the shoulders while the hand moves around it; a phone in a
    hand keeps a steady offset from the HAND."""
    sw = s["shoulder_width"]
    scx, scy = s["shoulder_center"]
    cx, cy = center
    if not (abs(cx - scx) <= BADGE_ZONE_HALF_W * sw
            and scy + BADGE_ZONE_TOP * sw <= cy <= scy + BADGE_ZONE_BOTTOM * sw):
        return "phone"
    wrists = [wr for wr in (s["l_wrist"], s["r_wrist"]) if wr is not None]
    if not wrists:
        return "phone"
    # r41 two-hand: both hands wrapped round the candidate = phone, never a badge
    if sum(1 for p in wrists
           if math.hypot(cx - p[0], cy - p[1]) < BADGE_TWO_HAND_MULT * sw) == 2:
        return "phone"
    wr = min(wrists, key=lambda p: math.hypot(cx - p[0], cy - p[1]))

    hist = entry.setdefault("_badge_hist", {}).setdefault(s["track_id"], [])
    if hist and now - hist[-1][0] > 2.5:
        hist.clear()                                   # candidate vanished: start over
    if not hist or now - hist[-1][0] >= 0.05:          # one sample per pass/frame
        hist.append((now,
                     ((cx - scx) / sw, (cy - scy) / sw),
                     ((cx - wr[0]) / sw, (cy - wr[1]) / sw)))
    hist[:] = [h for h in hist if h[0] >= now - BADGE_LOCK_WINDOW_SEC]

    if len(hist) < 4 or hist[-1][0] - hist[0][0] < BADGE_LOCK_MIN_SEC:
        return "pending"
    r = np.array([h[1] for h in hist])
    d = np.array([h[2] for h in hist])
    std_r = float(math.hypot(r[:, 0].std(), r[:, 1].std()))
    std_d = float(math.hypot(d[:, 0].std(), d[:, 1].std()))
    if std_r < BADGE_LOCK_STD and std_d > max(BADGE_LOCK_MIN_REL, BADGE_LOCK_RATIO * std_r):
        return "badge"
    return "phone"


def _update_hand_rise(entry, students, now):
    """r48: remember when each student's hand was last low or out of sight."""
    if not PHONE_NEEDS_HAND_RISE:
        return
    st = entry.setdefault("_hand_low", {})
    for s in students:
        sw = s["shoulder_width"]
        sy = s["shoulder_center"][1]
        low = any(w is None or (w[1] - sy) >= HAND_LOW_SW * sw
                  for w in (s["l_wrist"], s["r_wrist"]))
        d = st.setdefault(s["track_id"], {"since": None, "last": -1e9})
        if low:
            if d["since"] is None:
                d["since"] = now
            if now - d["since"] >= HAND_LOW_MIN_SEC:
                d["last"] = now
        else:
            d["since"] = None


def _hand_rise_ok(entry, tid, now):
    """r48: True if this student's phone is allowed to count right now."""
    if not PHONE_NEEDS_HAND_RISE:
        return True
    if now - entry.get("_phone_ok", {}).get(tid, -1e9) <= HAND_RISE_KEEP_SEC:
        return True
    d = entry.get("_hand_low", {}).get(tid)
    return d is not None and now - d["last"] <= HAND_RISE_MEMORY_SEC


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
def _mid_band_wrist(a, b):
    """A wrist of student `a` that is out in the middle band of the gap between
    a and b (and actually extended away from a's body), or None."""
    ax, ay = a["shoulder_center"]
    bx, _by = b["shoulder_center"]
    gap = bx - ax
    if abs(gap) < 1:
        return None
    sw = a["shoulder_width"]
    for wrist in (a["l_wrist"], a["r_wrist"]):
        if wrist is None:
            continue
        p = (wrist[0] - ax) / gap
        if (REACH_MID_MIN_PROGRESS <= p <= REACH_MID_MAX_PROGRESS
                and math.hypot(wrist[0] - ax, wrist[1] - ay) / sw >= REACH_MID_MIN_EXTENSION
                and abs(wrist[1] - ay) <= REACH_MID_MAX_DY * sw):
            return wrist
    return None


def _find_reaching_wrist(reacher, target, same_row):
    """Returns the wrist (x, y) of `reacher` that is reaching toward `target`,
    or None.  See the REACH_* constants for the two tests."""
    rx, ry = reacher["shoulder_center"]
    tx, ty = target["shoulder_center"]
    sw     = reacher["shoulder_width"]
    gap_x  = tx - rx

    # C: mutual hand-off -- both students have a hand out in the gap between them.
    if same_row and abs(gap_x) >= 1:
        mine = _mid_band_wrist(reacher, target)
        if mine is not None and _mid_band_wrist(target, reacher) is not None:
            return mine
    for wrist in (reacher["l_wrist"], reacher["r_wrist"]):
        if wrist is None:
            continue

        # A: wrist inside the neighbor's personal zone -- but only if the hand is
        # actually away from its owner (not just resting on its own desk).
        d_target = math.hypot(wrist[0] - tx, wrist[1] - ty)
        d_self   = math.hypot(wrist[0] - rx, wrist[1] - ry)
        if (d_target < target["shoulder_width"] * REACH_ZONE_RADIUS_MULTIPLIER
                and d_target < d_self
                and d_self >= sw * REACH_ZONE_MIN_EXTENSION):
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
# STATIC-OBJECT POSITION TRACKER
# ==========================================
def _check_static_objects(entry, centers, now, store="static_phones",
                          static_sec=STATIC_PHONE_SEC,
                          move_px=STATIC_OBJECT_MOVE_PX):
    """Update the per-camera tracker for detected object bbox centres and return the
    set of indices (into `centers`) suppressed because the object has sat in
    the same spot for >= static_sec (furniture, seat-number plates, etc.)."""
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
            if d < STATIC_OBJECT_RADIUS_PX and d < best_dist:
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
                        new_cap = _open_capture(entry["source"], entry["type"])
                        entry["cap"] = new_cap
                        entry["_overlay"] = None
                    placeholder = make_status_frame(
                        entry["label"], "DISCONNECTED", "Attempting to reconnect...")
                    ret, buf = cv2.imencode('.jpg', placeholder)
                    if ret:
                        entry["latest_frame"] = buf.tobytes()
                    time.sleep(1.0)
                    continue

            h, w, _ = frame.shape
            clean_frame = frame.copy()
            current_time = time.time()
            if entry["type"] == "file" and VIDEO_CLOCK:
                try:
                    _vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    _vt = cap.get(cv2.CAP_PROP_POS_FRAMES) / _vfps
                    if _vt + 1.0 < entry.get("_vt_last", 0.0):   # the clip looped
                        entry["_vt_off"] = (entry.get("_vt_off", 0.0)
                                            + entry.get("_vt_last", 0.0) + 1.0)
                    entry["_vt_last"] = _vt
                    current_time = 1000.0 + entry.get("_vt_off", 0.0) + _vt
                except Exception:
                    pass

            gaze_timers      = entry["gaze_timers"]
            turn_history     = entry["turn_history"]
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

            _dt = max(0.0, min(0.5, current_time - entry.get("_gz_prev", current_time)))
            entry["_gz_prev"] = current_time
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
            reach_violation       = False
            gaze_trigger          = False
            violation_reason      = ""
            violation_confidence  = 0.0
            violating_track_id    = None
            violations            = []   # every (reason, conf, track_id) this frame

            # ==========================================
            # 1. PROHIBITED OBJECT DETECTION (YOLO)
            # ==========================================
            object_detections = []
            entry["_obj_n"] = entry.get("_obj_n", 0) + 1
            _obj_cache = entry.get("_obj_cache")
            _run_full_pass = (_obj_cache is None or not DETECTION_ENABLED["phone"]
                              or entry["_obj_n"] % max(1, PHONE_FULL_EVERY_N) == 0)
            if not _run_full_pass:
                # skipped frame: reuse the last result (as copies, later steps edit these dicts)
                object_detections = [dict(d) for d in _obj_cache]
            try:
                obj_results = (entry["object_model"](frame, stream=True, verbose=False, classes=[0],
                                                     imgsz=640, device=DEVICE, half=USE_HALF)
                               if (DETECTION_ENABLED["phone"] and _run_full_pass) else [])
                for r in obj_results:
                    for box in r.boxes:
                        cls_id     = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bw, bh = x2 - x1, y2 - y1

                        # --- ANTI-HUMAN SANITY FILTER ---
                        if bh > h * 0.35 or bw > w * 0.35:
                            continue  # Ignore human torsos
                        if (bw * bh) > (w * h * 0.08):
                            continue

                        if (cls_id in PROHIBITED_CLASSES and confidence > PHONE_FULL_CONF
                                and DETECTION_ENABLED["phone"]):
                            object_detections.append({
                                "name":       PROHIBITED_CLASSES[cls_id],
                                "confidence": confidence,
                                "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
                                "bbox":       (x1, y1, x2, y2),
                            })
            except Exception as e:
                print(f"[{camera_id}] Object detection error: {e}")
            if _run_full_pass:
                entry["_obj_cache"] = [dict(d) for d in object_detections]

            # ==========================================
            # 2. SKELETAL POSE ESTIMATION + TRACKING
            # ==========================================
            students        = []
            active_ids      = []

            try:
                pose_results = entry["pose_model"].track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    stream=True,
                    verbose=False,
                    iou=POSE_IOU,
                    imgsz=POSE_IMGSZ,
                    conf=POSE_CONF,
                    device=DEVICE,
                    half=USE_HALF,
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

                    # Pre-pass: collect candidates and drop ghost skeletons before
                    # any timer/baseline/violation logic runs.
                    candidates = []
                    for idx, person in enumerate(kpts):
                        track_id = track_ids[idx] if idx < len(track_ids) else None
                        if track_id is None:
                            continue

                        if confs is not None:
                            if (confs[idx][5] < SHOULDER_MIN_CONF or
                                    confs[idx][6] < SHOULDER_MIN_CONF):
                                if DEBUG_OVERLAY and r.boxes is not None and idx < len(r.boxes):
                                    bx1, by1, bx2, by2 = map(int, r.boxes.xyxy[idx].tolist())
                                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (90, 90, 90), 1)
                                    cv2.putText(frame,
                                                f"sh {confs[idx][5]:.2f}/{confs[idx][6]:.2f}",
                                                (bx1, max(by1 - 4, 12)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
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
                        l_wrist,    r_wrist    = person[9],  person[10]

                        if confs is not None:
                            c_nose   = confs[idx][0]
                            c_lwrist = confs[idx][9]
                            c_rwrist = confs[idx][10]
                        else:
                            c_nose = 1.0
                            c_lwrist = c_rwrist = 1.0

                        nose_confident = c_nose  >= MIN_KEYPOINT_CONF

                        shoulder_center = (int((l_shoulder[0] + r_shoulder[0]) / 2),
                                           int((l_shoulder[1] + r_shoulder[1]) / 2))
                        shoulder_width  = max(20.0, abs(r_shoulder[0] - l_shoulder[0]))

                        l_wrist_pt = ((int(l_wrist[0]), int(l_wrist[1]))
                                      if (c_lwrist >= REACH_WRIST_CONF and l_wrist[0] > 0) else None)
                        r_wrist_pt = ((int(r_wrist[0]), int(r_wrist[1]))
                                      if (c_rwrist >= REACH_WRIST_CONF and r_wrist[0] > 0) else None)

                        students.append({
                            "track_id":        track_id,
                            "shoulder_center": shoulder_center,
                            "shoulder_width":  shoulder_width,
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

                        # Head yaw: nose displacement ratio vs adaptive per-student baseline.
                        # A pitch term (nose height) separates looking-down from turning sideways.
                        person_turn    = "FORWARD"
                        smoothed_ratio = 0.0
                        adjusted_ratio = 0.0
                        down_delta     = 0.0

                        sh_sign        = -1.0   # image-x direction of the l_shoulder->r_shoulder vector (-1 = front-facing)
                        if (nose_confident and nose[0] > 0
                                and l_shoulder[0] > 0 and r_shoulder[0] > 0):
                            d_x_sh = r_shoulder[0] - l_shoulder[0]
                            sh_sign = 1.0 if d_x_sh >= 0 else -1.0
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
                                entry["home_ratio"][track_id]  = baseline_ratio[track_id]
                                entry["home_height"][track_id] = baseline_height[track_id]
                                person_turn = "CALIBRATING"
                            else:
                                # Snap-back: a re-baselined (held) pose must not make the
                                # return to true neutral look like a turn the other way.
                                # Clamp: a baseline this far off-axis means calibration (or a held-posture
                                # re-baseline) caught a head that was already turned, which would invert
                                # the sign of every later reading.
                                baseline_ratio[track_id] = max(-BASELINE_MAX_ABS, min(BASELINE_MAX_ABS, baseline_ratio[track_id]))
                                entry["home_ratio"][track_id] = max(-BASELINE_MAX_ABS, min(BASELINE_MAX_ABS,
                                    entry["home_ratio"].get(track_id, baseline_ratio[track_id])))
                                home_r = entry["home_ratio"].get(track_id, baseline_ratio[track_id])
                                if (abs(baseline_ratio[track_id] - home_r) > REPOSITION_SETTLE_BAND
                                        and abs(raw_ratio - home_r)
                                            < abs(raw_ratio - baseline_ratio[track_id])):
                                    baseline_ratio[track_id]  = home_r
                                    baseline_height[track_id] = entry["home_height"].get(
                                        track_id, baseline_height[track_id])
                                    turn_history.pop(track_id, None)
                                    pitch_history.pop(track_id, None)
                                    stable_since[track_id]  = current_time
                                    stable_anchor[track_id] = raw_ratio
                                    print(f"[{camera_id}] ID {track_id}: back near home pose, "
                                          f"baseline restored.")
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
                                looking_down = down_delta > (PITCH_DOWN_DELTA * GAZE_DOWN_HYSTERESIS
                                                             if track_id in gaze_timers
                                                             else PITCH_DOWN_DELTA)

                                _prev = entry.get("last_turn", {}).get(track_id)
                                _l_thr = (HEAD_TURN_EXIT_RATIO if _prev == "LOOKING LEFT"
                                          else HEAD_TURN_RATIO_THRESHOLD)
                                _r_thr = (HEAD_TURN_EXIT_RATIO if _prev == "LOOKING RIGHT"
                                          else HEAD_TURN_RATIO_THRESHOLD)
                                if smoothed_ratio < -_l_thr:
                                    person_turn = "LOOKING LEFT"
                                elif smoothed_ratio > _r_thr:
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

                        # r39: keep a side turn alive through brief nose dropouts
                        _hold = entry.setdefault("_turn_hold", {})
                        if person_turn in ("LOOKING LEFT", "LOOKING RIGHT"):
                            _hold[track_id] = (person_turn, smoothed_ratio, current_time)
                        elif (not nose_confident and track_id in _hold
                              and current_time - _hold[track_id][2] <= 0.8):
                            person_turn, smoothed_ratio = _hold[track_id][0], _hold[track_id][1]

                        if DEBUG_OVERLAY and person_turn != "CALIBRATING":
                            cv2.putText(frame, f"yaw {smoothed_ratio:+.2f} dn {down_delta:+.2f} b {baseline_ratio.get(track_id, 0.0):+.2f}",
                                        (shoulder_center[0] - 45, shoulder_center[1] + 42),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        students[-1]["person_turn"] = person_turn
                        students[-1]["down_delta"] = down_delta
                        last_turn = entry.setdefault("last_turn", {})
                        if last_turn.get(track_id) != person_turn:
                            print(f"[{camera_id}] ID {track_id}: {last_turn.get(track_id, '-')} -> "
                                  f"{person_turn}  yaw={smoothed_ratio:+.2f} dn={down_delta:+.2f} base={baseline_ratio.get(track_id, 0.0):+.2f}")
                            last_turn[track_id] = person_turn
                        active_ids.append(track_id)

                        gaze_candidate = (DETECTION_ENABLED["gaze"]
                                          and person_turn not in ("FORWARD", "CALIBRATING", "LOOKING DOWN")
                                          and _neighbor_in_direction(person_turn, track_id, candidates, kept_idx,
                                                                     entry, current_time,
                                                                     (1 if smoothed_ratio > 0 else -1) * sh_sign))
                        if (DETECTION_ENABLED["gaze"] and not gaze_candidate
                                and person_turn in ("LOOKING LEFT", "LOOKING RIGHT")):
                            cv2.putText(frame, entry.get("_nb_dbg", {}).get(track_id, "turn ignored: no neighbor"),
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
                                violations.append(("PROLONGED_SIDE_GAZE", 0.95, track_id))
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
                            if person_turn == "CALIBRATING":
                                gaze_timers.pop(track_id, None)
                                turn_history.pop(track_id, None)
                                entry.get("_gaze_logged", {}).pop(track_id, None)
                            elif track_id in gaze_timers:
                                # Not gazing this frame (forward / looking down / lost neighbour /
                                # flicker): drain the countdown instead of resetting it.
                                gaze_timers[track_id] += _dt * (1.0 + GAZE_DRAIN_RATE)
                                if current_time - gaze_timers[track_id] <= 0.0:
                                    print(f"[{camera_id}] ID {track_id}: gaze timer dropped "
                                          f"(turn={person_turn}, yaw={smoothed_ratio:+.2f}, dn={down_delta:+.2f})")
                                    gaze_timers.pop(track_id, None)
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

            # Prune per-track state after TRACK_LOST_GRACE_SEC to survive brief dropouts.
            for tid in active_ids:
                track_last_seen[tid] = current_time
            for tid in list(track_first_seen):
                track_last_seen.setdefault(tid, current_time)
            expired = [tid for tid, ts in track_last_seen.items()
                       if current_time - ts > TRACK_LOST_GRACE_SEC]
            for tid in expired:
                for key in PER_TRACK_KEYS:
                    entry[key].pop(tid, None)
                for key in ("last_turn", "_gaze_logged", "_nb_seen", "_nb_dbg"):
                    entry.get(key, {}).pop(tid, None)

            _update_hand_rise(entry, students, current_time)
            if DEBUG_OVERLAY and PHONE_NEEDS_HAND_RISE:
                for _s in students:
                    _ok = _hand_rise_ok(entry, _s["track_id"], current_time)
                    cv2.putText(frame, "rise" if _ok else "no-rise",
                                (_s["shoulder_center"][0] - 45, _s["shoulder_center"][1] + 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 255, 0) if _ok else (150, 150, 150), 1)
            _merge_wrist_phones(entry, camera_id, clean_frame, students,
                                current_time, object_detections)
            if DEBUG_OVERLAY:
                for (dx1, dy1, dx2, dy2, dtid, dtop) in entry.get("_wc_debug", []):
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 0, 255), 1)
                    cv2.putText(frame, f"{dtop[0]:.2f}",
                        (dx1, max(dy1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            if DEBUG_OVERLAY:
                for (rx1, ry1, rx2, ry2, why) in entry.get("_wc_rej", []):
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 200, 255), 1)
                    cv2.putText(frame, why, (rx1, max(ry1 - 4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            if DEBUG_OVERLAY and PHONE_VERIFY != "off" and entry.get("_pv") is not None:
                for (px1, py1, px2, py2, plabel, pcolor) in entry["_pv"].overlay(current_time):
                    cv2.rectangle(frame, (px1, py1), (px2, py2), pcolor, 1)
                    cv2.putText(frame, plabel, (px1, min(h - 4, py2 + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, pcolor, 1)
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
            # 2.5. STATIC-PHONE FILTER
            # ==========================================
            # Wrist-crop detections carry a track_id and are exempt from suppression.
            phone_dets = [(i, d) for i, d in enumerate(object_detections)
                          if d["name"] == "CELL PHONE" and d.get("track_id") is None]
            phone_sup  = _check_static_objects(entry, [d["center"] for _, d in phone_dets],
                                               current_time, store="static_phones",
                                               static_sec=STATIC_PHONE_SEC)
            static_phone_idx = {phone_dets[k][0] for k in phone_sup}

            ref_sw = (float(np.median([s["shoulder_width"] for s in students]))
                      if students else None)
            accepted_detections = []
            for i, det in enumerate(object_detections):
                x1, y1, x2, y2 = det["bbox"]
                # wrist-crop detections (they carry a track_id) were already vetted
                if PHONE_VERIFY != "off" and det.get("track_id") is None:
                    _pvr = _pv.store_for(entry, PHONE_VERIFY_DUMP_DIR if PHONE_VERIFY_DUMP else None,
                                         camera_id).feed(clean_frame, det["bbox"],
                                                         det["confidence"], current_time)
                    if PHONE_VERIFY == "on" and _pvr["verdict"] != "CONFIRMED":
                        if DEBUG_OVERLAY:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                            cv2.putText(frame, "UNVERIFIED (" + _pvr["verdict"].lower() + ")",
                                        (x1, max(y1 - 8, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                        continue
                if PHONE_VERIFY != "on" and det.get("track_id") is None and _pen_like(
                        det["bbox"], _local_sw(det, students, ref_sw), det["confidence"]):
                    if DEBUG_OVERLAY:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                        cv2.putText(frame, "PEN-LIKE (ignored)", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                    continue
                if (PHONE_FULL_NEEDS_HAND and det.get("track_id") is None and not _phone_in_hand(det, students)):
                    if DEBUG_OVERLAY:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                        cv2.putText(frame, "NOT IN HAND (ignored)", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                    continue
                if PHONE_VERIFY != "on" and det.get("track_id") is None and _badge_like(det["center"], students, clean_frame, det["bbox"]):
                    if DEBUG_OVERLAY:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                        cv2.putText(frame, "ID BADGE (ignored)", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                    continue
                if PHONE_VERIFY != "on" and det.get("track_id") is None:
                    _own = _owner_of_detection(det, students, established_ids)
                    _os = next((s for s in students if s["track_id"] == _own), None)
                    if _os is not None:
                        _verdict = _torso_lock_verdict(entry, _os, det["center"], current_time)
                        if _verdict != "phone":
                            if DEBUG_OVERLAY:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                                cv2.putText(frame, ("ID BADGE (moves with body)" if _verdict == "badge"
                                                    else "CHECKING..."),
                                            (x1, max(y1 - 8, 12)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
                            continue
                if det.get("track_id") is None and PHONE_NEEDS_HAND_RISE:
                    _ro = _owner_of_detection(det, students, established_ids)
                    if _ro is not None:
                        if not _hand_rise_ok(entry, _ro, current_time):
                            if DEBUG_OVERLAY:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                                cv2.putText(frame, "NO HAND RISE (ignored)", (x1, max(y1 - 8, 12)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                            continue
                        entry.setdefault("_phone_ok", {})[_ro] = current_time
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

            object_detections = accepted_detections

            # ==========================================
            # 3. ATTRIBUTE OBJECT DETECTIONS TO A STUDENT
            # ==========================================
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
                if best_det.get("track_id") is not None:
                    violating_track_id = best_det["track_id"]
                for det in object_detections:
                    violations.append((f"OBJECT_{det['name'].replace(' ', '_')}",
                                       det["confidence"],
                                       _owner_of_detection(det, students, established_ids)))

            # ==========================================
            # 4. INTERACTION & DESK LEANING / REACH DETECTION
            # ==========================================
            if DEBUG_OVERLAY:
                for _s in students:
                    for _wr in (_s["l_wrist"], _s["r_wrist"]):
                        if _wr is not None:
                            cv2.circle(frame, _wr, 4, (0, 165, 255), -1)
            # r38: hands-together phone posture fallback
            if HANDS_POSTURE_ENABLED and DETECTION_ENABLED["phone"]:
                _hp = entry.setdefault("_hp", {})
                _hp_now = set()
                for _s in students:
                    _tid = _s["track_id"]
                    if _tid not in established_ids or _s["person_turn"] == "CALIBRATING":
                        continue
                    if not _hands_posture(_s):
                        continue
                    _hp_now.add(_tid)
                    _st = _hp.setdefault(_tid, {"start": current_time, "last": current_time})
                    _st["last"] = current_time
                    _held = current_time - _st["start"]
                    _pos = (_s["shoulder_center"][0] - 60, _s["shoulder_center"][1] + 60)
                    if _held >= HANDS_POSTURE_SEC:
                        object_violation = True
                        violations.append(("OBJECT_PHONE_POSTURE", 0.6, _tid))
                        if not violation_reason:
                            violation_reason     = "OBJECT_PHONE_POSTURE"
                            violation_confidence = 0.6
                            violating_track_id   = _tid
                        cv2.putText(frame, "PHONE? (hands at chest)", _pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"hands up ({HANDS_POSTURE_SEC - _held:.1f}s)", _pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                for _k in [k for k, v in _hp.items()
                           if k not in _hp_now and current_time - v["last"] > HANDS_POSTURE_GRACE_SEC]:
                    _hp.pop(_k, None)

            interaction_candidates = [s for s in students if s["track_id"] in established_ids]
            active_reach_pairs = []
            num_students = len(interaction_candidates)
            for i in range(num_students):
                for j in range(i + 1, num_students):
                    s1, s2 = interaction_candidates[i], interaction_candidates[j]

                    dist           = math.hypot(
                        s1["shoulder_center"][0] - s2["shoulder_center"][0],
                        s1["shoulder_center"][1] - s2["shoulder_center"][1])
                    avg_shoulder_w  = (s1["shoulder_width"] + s2["shoulder_width"]) / 2.0

                    if dist < min(s1["shoulder_width"], s2["shoulder_width"]) * DUPLICATE_DIST_MULTIPLIER:
                        continue  # ghost pair — same body detected twice

                    # Reach check, both directions.
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
                                violations.append(("REACHING_NEIGHBOR_ZONE", 0.85,
                                                   min(reacher["track_id"], target["track_id"])))
                                if not violation_reason:
                                    violation_reason     = "REACHING_NEIGHBOR_ZONE"
                                    violation_confidence = 0.85
                                    violating_track_id   = reacher["track_id"]

                                cv2.line(frame, reach_wrist, target["shoulder_center"],
                                         (0, 140, 255), 2)
                                cv2.putText(frame, "REACH",
                                            (reach_wrist[0] - 20, max(20, reach_wrist[1] - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

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
            entry["recorder"].write(clean_frame if RECORD_RAW else frame)

            is_violating = object_violation or reach_violation or gaze_trigger

            if is_violating and violations:
                unique = {}
                for v_reason, v_conf, v_tid in violations:
                    k = (v_tid, v_reason)
                    if k not in unique or v_conf > unique[k]:
                        unique[k] = v_conf
                for (v_tid, v_reason), v_conf in unique.items():
                    inc, is_new = entry["incidents"].observe(
                        entry["recorder"], frame, v_reason, v_tid,
                        v_conf, CURRENT_SESSION, camera_id,
                        thumb_dir=os.path.join(VIOLATION_FOLDER, CURRENT_SESSION, camera_id),
                        thumb_web_prefix=f"violations/{CURRENT_SESSION}/{camera_id}")
                    if is_new:
                        log_violation(CURRENT_SESSION, v_reason, v_conf,
                                      inc["thumb_web"], student_track_id=v_tid,
                                      camera_id=camera_id)
                        print(f"[{camera_id}] NEW INCIDENT {v_reason} "
                              f"(track_id={v_tid}, t={inc['start']:.1f}s)")

            if entry.get("_smooth_display"):
                # hand the pixels we drew to the display thread; it pastes them on live video
                entry["_overlay"] = {"mask": np.any(frame != clean_frame, axis=2),
                                     "frame": frame, "t": time.time()}
            else:
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

        time.sleep(0.033)  # ~30 fps


# ==========================================
# ACCESS CONTROL
# ==========================================
app.secret_key = os.environ.get("CERBERUS_SECRET_KEY") or secrets.token_hex(32)

PUBLIC_ENDPOINTS = {"login", "signup"}


@app.before_request
def _require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
        return
    if request.endpoint == "static":
        return _guard_static()
    if session.get("user_id") is not None and not get_user_by_id(session["user_id"]):
        session.clear()
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return _guard_session_access()


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        pending_user_id = session.get('pending_login_user_id')
        if pending_user_id and request.form.get('step') == 'verify':
            user = get_user_by_id(pending_user_id)
            if user and verify_totp(user.get('totp_secret'), request.form.get('code')):
                session.clear()
                session['user_id'] = user['id']
                session['username'] = user['username']
                return redirect(url_for('index'))
            error = 'Authenticator code is invalid or expired.'
            return render_template('login.html', error=error, verification_username=session.get('pending_login_username'))

        username = (request.form.get('username') or '').strip()
        user = verify_user(username, request.form.get('password', ''))
        if user:
            if user.get('totp_secret'):
                session.clear()
                session['pending_login_user_id'] = user['id']
                session['pending_login_username'] = user['username']
                return render_template('login.html', verification_username=user['username'])
            session.clear()
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
        else:
            error = 'Incorrect username or password.'
    return render_template('login.html', error=error)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    error = None
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password', '')
        user_id, error = create_user(username, password)
        if user_id:
            if user_count() == 1:
                _assign_unowned_sessions(user_id)   # first account inherits sessions that predate ownership
            session.clear()
            session['user_id'] = user_id
            session['username'] = username
            if request.form.get('enable_mfa') == '1':
                return redirect(url_for('mfa_setup'))
            return redirect(url_for('index'))
    return render_template('signup.html', error=error)


@app.route('/security/mfa', methods=['GET', 'POST'])
def mfa_setup():
    user_id = session.get('user_id')
    active_secret = get_totp_secret(user_id)
    error = None
    pending_secret = session.get('pending_totp_secret')

    if request.method == 'POST' and not active_secret:
        if pending_secret and verify_totp(pending_secret, request.form.get('code')):
            save_totp_secret(user_id, pending_secret)
            session.pop('pending_totp_secret', None)
            return redirect(url_for('index'))
        error = 'That authenticator code is invalid or expired.'

    if not active_secret and not pending_secret:
        pending_secret = generate_totp_secret()
        session['pending_totp_secret'] = pending_secret

    return render_template('mfa_setup.html', enabled=bool(active_secret),
                           secret=pending_secret, error=error)


@app.route('/security/mfa/status')
def mfa_status():
    return jsonify({'enabled': bool(get_totp_secret(session.get('user_id')))})


@app.route('/security/mfa/disable', methods=['POST'])
def disable_mfa():
    user_id = session.get('user_id')
    secret = get_totp_secret(user_id)
    data = request.get_json(silent=True) or {}
    if not secret or not verify_totp(secret, data.get('code')):
        return jsonify({'message': 'Authenticator code is invalid or expired.'}), 403
    save_totp_secret(user_id, None)
    return jsonify({'status': 'disabled'})


@app.route('/account/delete', methods=['GET', 'POST'])
def delete_account():
    confirmation_phrase = 'DELETE ACCOUNT'
    user_id = session.get('user_id')
    user = get_user_by_id(user_id)
    if not user:
        session.clear()
        return redirect(url_for('login'))

    csrf_token = session.get('account_delete_csrf_token')
    if not csrf_token:
        csrf_token = secrets.token_urlsafe(32)
        session['account_delete_csrf_token'] = csrf_token

    current_owned = get_session_owner(CURRENT_SESSION) == user_id
    active_owned_session = bool(monitoring_active and current_owned)
    mfa_secret = get_totp_secret(user_id)

    def render_page(error=None, status=200):
        return render_template(
            'delete_account.html', username=user['username'], csrf_token=csrf_token,
            active_owned_session=active_owned_session, mfa_enabled=bool(mfa_secret),
            confirmation_phrase=confirmation_phrase, error=error,
        ), status

    if request.method == 'GET':
        return render_page()

    submitted_token = request.form.get('csrf_token', '')
    if not secrets.compare_digest(csrf_token, submitted_token):
        return render_page('This request expired. Reload the page and try again.', 400)
    if request.form.get('confirm_username') != user['username'] or not verify_user(
            user['username'], request.form.get('password', '')):
        return render_page('The username confirmation or password is incorrect.', 400)
    if request.form.get('confirm_phrase') != confirmation_phrase:
        return render_page(f'Type "{confirmation_phrase}" exactly to confirm deletion.', 400)
    if mfa_secret and not verify_totp(mfa_secret, request.form.get('totp_code')):
        return render_page('The authenticator code is invalid or expired.', 400)
    if active_owned_session:
        return render_page('Stop monitoring before deleting this account.', 400)

    owned_sessions = set(list_owned_sessions(user_id))
    if current_owned:
        owned_sessions.add(CURRENT_SESSION)

    try:
        for session_id in owned_sessions:
            if _valid_session_id(session_id):
                _delete_session_data(session_id)

        replacement_owner = first_user_id_excluding(user_id)
        if not delete_user(
                user_id,
                replacement_session_id=CURRENT_SESSION if current_owned else None,
                replacement_owner_id=replacement_owner):
            raise RuntimeError('Account record was not deleted.')
    except Exception as error:
        print(f"[account] could not delete account #{user_id}: {error}")
        return render_page('Account deletion could not be completed. Some session data may already have been removed.', 500)

    print(f"[account] {user['username']} deleted their account and {len(owned_sessions)} owned session(s)")
    session.clear()
    return redirect(url_for('login'))


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('login'))




@app.route('/')
def index():
    return render_template('sessions.html',
                           current_session=CURRENT_SESSION if _owns_current() else None,
                           monitoring_active=bool(monitoring_active and _owns_current()),
                           mfa_enabled=bool(get_totp_secret(session.get('user_id'))))


@app.route('/monitor')
def monitor():
    return render_template('index.html')


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

    summaries.setdefault(CURRENT_SESSION, {
        'session_id': CURRENT_SESSION, 'flag_count': 0, 'updated_at': None,
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
    current = summaries.get(CURRENT_SESSION)
    if current and not current['recording_count'] and not current['flag_count']:
        summaries.pop(CURRENT_SESSION)
    return sorted(summaries.values(), key=lambda item: item['updated_at'] or '', reverse=True)


@app.route('/sessions')
def sessions_route():
    return jsonify(_session_summaries())


@app.route('/session_files')
def session_files_route():
    session_id = request.args.get('session_id', '')
    if not _valid_session_id(session_id):
        return jsonify({'message': 'Invalid session.'}), 400
    return jsonify(_session_recordings(session_id))


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


@app.route('/recalibrate', methods=['POST'])
def recalibrate_route():
    """Restart every student's neutral-pose calibration without cutting the
    recording (use when students change seats)."""
    for entry in list(camera_registry.values()):
        entry["calibration_start"].clear()
        entry["gaze_timers"].clear()
        entry["turn_history"].clear()
        entry["gaze_lost"].clear()
    print(f"[session] {session.get('username', 'unknown')} requested recalibration")
    return jsonify({"status": "recalibrating"})


RESET_CONFIRM_PHRASE = "DELETE ALL DATA"  # required phrase for /reset_session


def _delete_session_data(session_id):
    global CURRENT_SESSION

    deleting_current = session_id == CURRENT_SESSION
    if deleting_current:
        for entry in list(camera_registry.values()):
            _finish_recording(entry)

    clear_violations(session_id)
    clear_incidents(session_id)
    shutil.rmtree(os.path.join(RECORDINGS_FOLDER, session_id), ignore_errors=True)
    shutil.rmtree(os.path.join(VIOLATION_FOLDER, session_id), ignore_errors=True)
    remove_session_owner(session_id)

    if deleting_current:
        for cam_id, entry in list(camera_registry.items()):
            _clear_tracking_state(entry)
        while True:
            replacement = "SESSION_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
            if replacement != session_id and not os.path.exists(os.path.join(RECORDINGS_FOLDER, replacement)):
                break
        CURRENT_SESSION = replacement
        set_session_owner(replacement, session.get('user_id'))
        os.makedirs(os.path.join(VIOLATION_FOLDER, CURRENT_SESSION), exist_ok=True)
        for cam_id, entry in list(camera_registry.items()):
            entry["recorder"] = SessionRecorder(CURRENT_SESSION, cam_id)
            entry["incidents"] = IncidentTracker()
    return CURRENT_SESSION


@app.route('/sessions/<session_id>', methods=['DELETE'])
def delete_session_route(session_id):
    if not _valid_session_id(session_id):
        return jsonify({"message": "Invalid session."}), 400
    if monitoring_active and session_id == CURRENT_SESSION:
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
    """Deletes the current session after phrase confirmation."""
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

    deleted_session = CURRENT_SESSION
    print(f"[session] {session.get('username', 'unknown')} reset session '{deleted_session}'")
    replacement = _delete_session_data(deleted_session)
    return jsonify({"status": "reset", "session_id": deleted_session,
                    "current_session": replacement})


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
    session_id = request.args.get('session_id', CURRENT_SESSION)
    if not _valid_session_id(session_id):
        return redirect(url_for('index'))
    return render_template('review.html', session_id=session_id)


@app.route('/incidents')
def incidents_route():
    session_id = request.args.get('session_id', CURRENT_SESSION)
    if not _valid_session_id(session_id):
        return jsonify({'message': 'Invalid session.'}), 400
    rows = list_incidents(session_id)
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
        "monitoring":       monitoring_active,
        "total_violations": len(records),
        "phone":            phone_count,
        "reach":            reach_count,
        "gaze":             gaze_count,
        "rules":            dict(DETECTION_ENABLED),
        "build":            BUILD_ID,
        "session_id":       CURRENT_SESSION,
    })


# ==========================================
# SESSION OWNERSHIP
# ==========================================
from flask import abort
from auth import (user_count, first_user_id, first_user_id_excluding,
                  get_session_owner, set_session_owner, remove_session_owner,
                  list_owned_sessions)

LIVE_ENDPOINTS = {
    "monitor", "video_feed", "get_session_route", "settings_route", "system_stats",
    "get_violations", "start_monitoring", "stop_monitoring", "reset_session",
    "list_cameras", "scan_cameras_route", "add_camera_route",
    "activate_camera_route", "pause_camera_route", "remove_camera_route",
    "recalibrate_route",
}
SESSION_QUERY_ENDPOINTS = {"review", "incidents_route", "session_files_route"}


def _owns_current():
    uid = session.get("user_id")
    return uid is not None and get_session_owner(CURRENT_SESSION) == uid


def _all_known_session_ids():
    """Every session that has data on disk or in the incident log, plus the
    current one.  Used to give pre-existing sessions an owner."""
    ids = {CURRENT_SESSION}
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
    """Gives every existing session that has no owner to `user_id`."""
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
    """Evidence images live under /static/violations/<session>/...  They need a
    login and must belong to one of the user's own sessions."""
    filename = (request.view_args or {}).get("filename", "")
    if not filename.startswith("violations/"):
        return None                      # style.css, script.js, ...
    uid = session.get("user_id")
    if uid is None:
        abort(404)
    parts = filename.split("/")
    if len(parts) == 2:
        return None                      # old flat file, not tied to any session
    sid = parts[1]
    if not _valid_session_id(sid) or get_session_owner(sid) != uid:
        abort(404)
    return None


def _guard_session_access():
    ep = request.endpoint
    uid = session.get("user_id")

    if ep in LIVE_ENDPOINTS:
        if get_session_owner(CURRENT_SESSION) != uid:
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
        sid = request.args.get("session_id", CURRENT_SESSION)
    elif ep == "delete_session_route":
        sid = (request.view_args or {}).get("session_id")
    elif ep == "serve_recording":
        sid = ((request.view_args or {}).get("filepath") or "").split("/")[0]
        must_be_valid = True

    if sid is not None:
        if not _valid_session_id(sid):
            if must_be_valid:
                abort(404)
        elif get_session_owner(sid) != uid:
            abort(404)
    return None


_assign_unowned_sessions(first_user_id())  # pre-existing sessions inherit to first account


import atexit


def _shutdown():
    """Finalize open recordings on Ctrl+C / normal exit."""
    for entry in list(camera_registry.values()):
        entry["stop_event"].set()
    for entry in list(camera_registry.values()):
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
