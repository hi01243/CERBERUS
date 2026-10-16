import os
import shutil
import sqlite3
import subprocess
import threading
import time
from contextlib import closing

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("CERBERUS_DATA_DIR")
if DATA_DIR:
    os.makedirs(DATA_DIR, exist_ok=True)
RECORDINGS_FOLDER = os.path.join(DATA_DIR, "recordings") if DATA_DIR else os.path.join(BASE_DIR, "recordings")
INCIDENT_DB = os.path.join(DATA_DIR, "incidents.db") if DATA_DIR else os.path.join(BASE_DIR, "incidents.db")

RECORD_FPS = 10.0             # fixed output fps (real time is preserved by duplicating frames)
RECORD_MAX_WIDTH = 960        # wider frames are downscaled to keep files small
MAX_CATCHUP_FRAMES = 30       # cap on duplicated frames after a stall

# a crash, power loss, or forced quit mid-exam otherwise leaves one
# giant recording file open (and unplayable) for the whole session. Rotating
# to a fresh file every RECORD_CHUNK_SEC caps the damage to at most one
# chunk, and means incidents automatically split cleanly across chunk
# boundaries (see IncidentTracker.observe: a recording-path change closes
# the open incident and starts a new one).
RECORD_CHUNK_SEC = 300.0      # 5 minutes per file

INCIDENT_GAP_SEC = 10.0       # same student + type within this gap = same incident
CONF_OVERWRITE_MARGIN = 0.10  # replace the thumbnail only if a clearly better detection arrives
DB_UPDATE_EVERY = 1.0         # throttle end_offset writes while an incident is ongoing
THUMB_WIDTH = 480


# ==========================================
# RECORDING
# ==========================================
class SessionRecorder:
    """One per camera. Opens a new file on the first write() after being
    closed, so each ARM..PAUSE run produces its own video."""

    def __init__(self, session_id, camera_id):
        self.session_id = session_id
        self.camera_id = camera_id
        self.dir = os.path.join(RECORDINGS_FOLDER, session_id, camera_id)
        self.lock = threading.Lock()
        self.writer = None
        self.filename = None
        self._tmp = None
        self._final = None
        self._codec = None
        self._size = None
        self._t0 = 0.0
        self._frames = 0
        self._failed = False

    @property
    def rel_path(self):
        """Path relative to RECORDINGS_FOLDER, or None if not recording."""
        if self.writer is None:
            return None
        return f"{self.session_id}/{self.camera_id}/{self.filename}"

    def offset(self):
        with self.lock:
            return self._frames / RECORD_FPS

    def _open(self, frame, now):
        h, w = frame.shape[:2]
        scale = min(1.0, RECORD_MAX_WIDTH / w)
        size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)   # H.264 needs even dimensions
        os.makedirs(self.dir, exist_ok=True)
        # Milliseconds in the name: PAUSE -> ARM inside one second no longer
        # reuses the .tmp.mp4 that the previous file is still being finalized from.
        base = "rec_" + time.strftime("%Y%m%d_%H%M%S") + f"_{int(now * 1000) % 1000:03d}"
        tmp = os.path.join(self.dir, base + ".tmp.mp4")
        for codec in (("mp4v",) if shutil.which("ffmpeg") else ("avc1", "mp4v")):
            wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*codec), RECORD_FPS, size)
            if wr.isOpened():
                break
            wr.release()
        else:
            print(f"[{self.camera_id}] Could not open a video writer; recording disabled for this run.")
            self._failed = True
            return False
        self.writer, self._codec, self._size = wr, codec, size
        self._tmp = tmp
        self._final = os.path.join(self.dir, base + ".mp4")
        self.filename = base + ".mp4"
        self._t0, self._frames = now, 0
        return True

    def _rotate_locked(self):
        """Finalizes the current file and clears writer state so the next
        write() call opens a fresh one. Caller must already hold self.lock."""
        self.writer.release()
        threading.Thread(target=_finalize, args=(self._tmp, self._final, self._codec),
                         daemon=True).start()
        self.writer = None

    def write(self, frame):
        now = time.time()
        with self.lock:
            if self._failed:
                return
            if self.writer is not None and (now - self._t0) >= RECORD_CHUNK_SEC:
                self._rotate_locked()
            if self.writer is None and not self._open(frame, now):
                return
            if (frame.shape[1], frame.shape[0]) != self._size:
                frame = cv2.resize(frame, self._size)
            behind = int((now - self._t0) * RECORD_FPS) + 1 - self._frames
            if behind > MAX_CATCHUP_FRAMES:
                self._t0 += (behind - MAX_CATCHUP_FRAMES) / RECORD_FPS
                behind = MAX_CATCHUP_FRAMES
            for _ in range(max(behind, 0)):
                self.writer.write(frame)
                self._frames += 1

    def close(self):
        """Finalizes the file. Safe to call repeatedly."""
        with self.lock:
            self._failed = False
            if self.writer is None:
                return
            self.writer.release()
            self.writer = None
            args = (self._tmp, self._final, self._codec)
        threading.Thread(target=_finalize, args=args, daemon=True).start()


def _low_priority_kwargs():
    """Run ffmpeg at below-normal priority so the re-encode doesn't starve YOLO."""
    if os.name == "nt":
        return {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}
    return {"preexec_fn": lambda: os.nice(10)}


def _finalize(tmp, final, codec):
    """Makes the file browser-playable. OpenCV's 'mp4v' output usually won't
    play in a browser, so re-encode to H.264 if ffmpeg is installed."""
    try:
        ffmpeg = shutil.which("ffmpeg")
        if codec != "avc1" and ffmpeg:
            r = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-i", tmp,
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
                 "-threads", "2",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", final],
                **_low_priority_kwargs())
            if r.returncode == 0:
                os.remove(tmp)
                return
        os.replace(tmp, final)
    except Exception as e:
        print(f"Recording finalize error: {e}")


def cleanup_orphaned_recordings():
    """on startup, finalize any *.tmp.mp4 left behind by a crash,
    forced quit, or power loss during a previous run -- otherwise that
    chunk stays permanently unplayable and its incidents point nowhere.
    Call once, before the app starts serving requests."""
    if not os.path.isdir(RECORDINGS_FOLDER):
        return
    recovered = 0
    for root, _dirs, files in os.walk(RECORDINGS_FOLDER):
        for name in files:
            if not name.endswith(".tmp.mp4"):
                continue
            tmp = os.path.join(root, name)
            final = tmp[: -len(".tmp.mp4")] + ".mp4"
            # Codec used before the crash is unknown; "mp4v" makes _finalize
            # always attempt the ffmpeg re-encode path when ffmpeg exists,
            # which is the safe assumption either way.
            _finalize(tmp, final, "mp4v")
            recovered += 1
    if recovered:
        print(f"[recording] Recovered {recovered} recording(s) left open by a previous crash.")


# ==========================================
# INCIDENT INDEX (own SQLite file, doesn't touch database.py)
# ==========================================
def _q(sql, args=(), fetch=False):
    with closing(sqlite3.connect(INCIDENT_DB, timeout=10)) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()] if fetch else None
        con.commit()
        return rows if fetch else cur.lastrowid


def init_incident_db():
    _q("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, camera_id TEXT, track_id INTEGER,
            violation_type TEXT, start_offset REAL, end_offset REAL,
            confidence REAL, recording TEXT, thumb TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    # lets a reviewer mark a flag as a false positive without deleting
    # the underlying evidence. Added via ALTER so existing incidents.db files
    # (from before this column existed) upgrade in place instead of breaking.
    try:
        _q("ALTER TABLE incidents ADD COLUMN dismissed INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass  # column already exists


def dismiss_incident(inc_id, dismissed=True):
    _q("UPDATE incidents SET dismissed=? WHERE id=?", (1 if dismissed else 0, inc_id))


def create_incident(session_id, camera_id, track_id, vtype, offset, conf, recording, thumb):
    return _q("INSERT INTO incidents (session_id, camera_id, track_id, violation_type, "
              "start_offset, end_offset, confidence, recording, thumb) VALUES (?,?,?,?,?,?,?,?,?)",
              (session_id, camera_id, track_id, vtype, offset, offset, conf, recording, thumb))


def update_incident(inc_id, end_offset, conf=None):
    if conf is None:
        _q("UPDATE incidents SET end_offset=? WHERE id=?", (end_offset, inc_id))
    else:
        _q("UPDATE incidents SET end_offset=?, confidence=? WHERE id=?", (end_offset, conf, inc_id))


def list_incidents(session_id):
    return _q("SELECT * FROM incidents WHERE session_id=? ORDER BY id DESC", (session_id,), fetch=True)


def list_incident_sessions():
    return _q("SELECT session_id, COUNT(*) AS flag_count, MAX(created_at) AS updated_at "
              "FROM incidents GROUP BY session_id", fetch=True)


def clear_incidents(session_id):
    _q("DELETE FROM incidents WHERE session_id=?", (session_id,))


def _save_thumb(frame, path):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (THUMB_WIDTH, max(1, int(h * THUMB_WIDTH / w))))
    cv2.imwrite(path, small, [cv2.IMWRITE_JPEG_QUALITY, 70])


class IncidentTracker:
    """One per camera. Groups repeated triggers into single incidents."""

    def __init__(self):
        self.lock = threading.Lock()
        self.open = {}   # (track_id, violation_type) -> incident state

    def observe(self, recorder, frame, reason, track_id, conf,
                session_id, camera_id, thumb_dir, thumb_web_prefix):
        """Returns (incident_state, is_new)."""
        now = time.time()
        rec_path = recorder.rel_path
        offset = recorder.offset() if rec_path else 0.0
        key = (track_id, reason)
        with self.lock:
            self._expire(now)
            inc = self.open.get(key)
            if inc and inc["recording"] != rec_path:
                update_incident(inc["id"], inc["end"])
                inc = None
            if inc:
                inc["last_seen"], inc["end"] = now, offset
                if conf > inc["conf"] + CONF_OVERWRITE_MARGIN:
                    inc["conf"] = conf
                    _save_thumb(frame, inc["thumb_disk"])
                    update_incident(inc["id"], offset, conf)
                    inc["last_db"] = now
                elif now - inc["last_db"] >= DB_UPDATE_EVERY:
                    update_incident(inc["id"], offset)
                    inc["last_db"] = now
                return inc, False

            name = f"{reason}_{time.strftime('%Y%m%d_%H%M%S')}_id{track_id}.jpg"
            os.makedirs(thumb_dir, exist_ok=True)
            thumb_disk = os.path.join(thumb_dir, name)
            _save_thumb(frame, thumb_disk)
            thumb_web = f"{thumb_web_prefix}/{name}"
            inc_id = create_incident(session_id, camera_id, track_id, reason,
                                     offset, conf, rec_path, thumb_web)
            inc = {"id": inc_id, "recording": rec_path, "start": offset, "end": offset,
                   "conf": conf, "last_seen": now, "last_db": now,
                   "thumb_disk": thumb_disk, "thumb_web": thumb_web}
            self.open[key] = inc
            return inc, True

    def _expire(self, now):
        for key, inc in list(self.open.items()):
            if now - inc["last_seen"] > INCIDENT_GAP_SEC:
                update_incident(inc["id"], inc["end"])
                del self.open[key]

    def finish(self):
        """Flush final end offsets (call when the recording closes)."""
        with self.lock:
            for inc in self.open.values():
                update_incident(inc["id"], inc["end"])
            self.open.clear()