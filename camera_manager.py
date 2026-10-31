"""Video capture classes and dynamic camera management for Cerberus."""
import threading
import time
import cv2
import numpy as np

from config import (
    THREADED_CAPTURE, SMOOTH_DISPLAY, OVERLAY_MAX_AGE_SEC,
    WRIST_CROP_MODEL, PHONE_MODEL, _load_yolo
)
import state
from recording import SessionRecorder, IncidentTracker


def _finish_recording(entry):
    """Closes the open recording and flushes open incidents. Safe to call repeatedly."""
    if "incidents" in entry:
        entry["incidents"].finish()
    if "recorder" in entry:
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
        return True

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, value):
        return self._cap.set(prop, value)

    def read(self, timeout=3.0):
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
        with self._lock:
            return self._seq, self._frame

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._thread is None or not self._thread.is_alive():
            self._cap.release()


class BrowserCapture:
    """Receives the newest JPEG-decoded frame uploaded by a browser camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._taken = 0
        self._closed = False

    def submit(self, frame):
        with self._lock:
            if self._closed:
                return False
            self._frame = frame
            self._seq += 1
        return True

    def isOpened(self):
        with self._lock:
            return not self._closed

    def grab(self):
        return self.isOpened()

    def get(self, _prop):
        return 0

    def set(self, _prop, _value):
        return True

    def read(self, timeout=3.0):
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._closed:
                    return False, None
                if self._frame is not None and self._seq != self._taken:
                    self._taken = self._seq
                    return True, self._frame.copy()
            time.sleep(0.004)
        return False, None

    def release(self):
        with self._lock:
            self._closed = True
            self._frame = None


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


def _open_capture(source, cam_type):
    """Live cameras get the background reader; video files stay a plain VideoCapture."""
    if cam_type == "browser":
        return BrowserCapture()
    if cam_type != "file" and THREADED_CAPTURE:
        return ThreadedCapture(source)
    cap = cv2.VideoCapture(source)
    if cam_type != "file":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _display_loop(camera_id, gen):
    """Smooth live view: newest camera frame with latest detection boxes pasted on top."""
    entry = state.camera_registry.get(camera_id)
    if entry is None:
        return
    last_seq = -1
    while not entry["stop_event"].is_set() and entry.get("_display_gen") == gen:
        cap = entry.get("cap")
        ov = entry.get("_overlay")
        if ov is None or not isinstance(cap, ThreadedCapture):
            time.sleep(0.03)
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
    """Registers a camera as INACTIVE."""
    cam_id = state.generate_camera_id()
    entry = {
        "source":   source,
        "type":     cam_type,
        "label":    label or cam_id,
        "status":   "inactive",
        "cap":      None,
        "pose_model":   None,
        "object_model": None,
        "lock":     threading.Lock(),
        "gaze_timers":        {},
        "turn_history":       {},
        "reach_timers":       {},
        "recorder":  SessionRecorder(state.CURRENT_SESSION, cam_id),
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
    with state.registry_lock:
        state.camera_registry[cam_id] = entry
    return cam_id


def activate_camera(cam_id):
    """Opens capture device, loads models, starts per-camera detection thread."""
    from detection import _detection_loop

    entry = state.camera_registry.get(cam_id)
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
            entry["pose_model"] = _load_yolo('yolo11n-pose.pt')
        if entry["object_model"] is None:
            try:
                entry["object_model"] = _load_yolo(PHONE_MODEL)
            except Exception as e:
                return False, f"Could not load phone model {PHONE_MODEL}: {e}"
            if PHONE_MODEL == WRIST_CROP_MODEL:
                entry["wrist_model"] = entry["object_model"]
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
    for key in ("_wc_hits", "_wc_weak_hits", "_badge_hist", "_wc_confirmed",
                "_wc_last", "_wc_debug", "last_turn", "_gaze_logged", "_nb_seen",
                "_nb_dbg", "_turn_hold", "_pv", "_hand_low", "_phone_ok"):
        entry.pop(key, None)


def pause_camera(cam_id):
    """Stops the detection thread, releases capture device. Models stay cached."""
    entry = state.camera_registry.get(cam_id)
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
    """Stops the thread, releases device, finalises recording, drops from registry."""
    entry = state.camera_registry.get(cam_id)
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

    with state.registry_lock:
        state.camera_registry.pop(cam_id, None)

    return True, "removed"


def scan_local_cameras(max_index=4):
    """Probes local device indices for available cameras not already registered."""
    registered_local_sources = {
        entry["source"] for entry in state.camera_registry.values()
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


def generate_frames(camera_id):
    """Serves latest annotated JPEG produced by detection loop."""
    while True:
        entry = state.camera_registry.get(camera_id)
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

        time.sleep(0.033)

