# CERBERUS AI — Classroom Examination Monitoring System

A Flask + OpenCV + YOLO (Ultralytics) system that watches an exam room
through one or more cameras, flags possible violations (unauthorized
devices, prolonged side-gazing, student-to-student interaction), and logs
timestamped evidence photos to a live dashboard.

This README documents the system **as it currently stands**, including
the design decisions and known limitations that came out of iterating on
it — not just how to run it.

---

## 1. What it does

- **Multi-camera support**, added and controlled entirely from the
  dashboard at runtime — no code edits or restarts needed to add a
  camera.
- **Per-student pose tracking** (YOLO11-pose + ByteTrack) with persistent
  IDs, so violations are attributed to a specific student across frames,
  not just logged anonymously.
- **Prohibited object detection** (YOLO11, COCO classes: cell phone,
  clock-as-smartwatch-proxy), with a best-effort match to the nearest
  tracked student.
- **Side-gaze detection** using a horizontal nose-displacement ratio
  (not a raw head angle — see §4), so looking down at an exam paper
  doesn't get misread as turning away.
- **Student interaction / desk-leaning detection**, gated on *sustained*
  proximity over time, not instant proximity, to avoid flooding the
  evidence log with false positives from normal seating.
- **Live dashboard**: per-camera tiles, an audit log, a filterable
  evidence gallery with a lightbox, and live stats.
- **Session reset**: wipe all logged violations and evidence photos for
  the current session from the dashboard, no manual file deletion
  needed.

---

## 2. Requirements

```
Python 3.9+
flask
opencv-python
ultralytics
numpy
```

Install with:

```bash
pip install flask opencv-python ultralytics numpy
```

The first run will auto-download `yolo11n-pose.pt` and `yolo11n.pt`
(Ultralytics nano weights) if they aren't already present.

---

## 3. Running it

```bash
python app.py
```

Then open `http://localhost:5000` (or `http://<your-machine-ip>:5000`
from another device on the same network, since the app binds to
`0.0.0.0`).

**Important:** on first launch, no camera streams automatically.
`cam1` (local device index `0`) is pre-registered but **inactive** —
click **Use** on its tile in the dashboard to actually open it and start
processing. This is intentional (see §5), not a bug.

---

## 4. Detection logic — how and why

### 4.1 Side-gaze detection (horizontal-displacement ratio)

**This is not a simple angle threshold**, and that distinction matters
if you're asked about it in a defense.

An earlier version computed a single combined angle from the
nose-to-shoulder vector. That approach conflated **yaw** (turning the
head sideways — the actual thing we want to detect) with **pitch**
(tilting the head down to read the exam paper — completely normal, and
probably the single most common head position during an actual exam).
At the point where the nose crosses the shoulder line during a downward
tilt, that angle calculation becomes numerically unstable (`atan2`
discontinuity), so a student just reading their paper could trigger a
false "turned away" flag.

The current method instead:
1. Computes the shoulder line's angle (to stay invariant to how a
   student's seat/torso is angled relative to the camera).
2. Rotates the nose-to-shoulder-center offset into that torso-aligned
   coordinate frame.
3. Takes **only the horizontal component** of that rotated offset,
   normalized by half the shoulder width, giving a ratio: how far
   sideways has the nose moved, independent of how far up/down it moved.
4. Smooths this ratio over the last `SMOOTHING_WINDOW` frames (a plain
   rolling average — safe here because, unlike an angle, this ratio
   never wraps around ±180°).
5. Flags `LOOKING LEFT` / `LOOKING RIGHT` only if the smoothed ratio
   exceeds `HEAD_TURN_RATIO_THRESHOLD` for `GAZE_DURATION_THRESHOLD`
   continuous seconds.

Looking down at a desk barely moves this ratio. An actual sideways turn
moves it a lot.

### 4.2 Student interaction / desk-leaning

Two students are flagged only if they stay within a proximity threshold
(scaled by their shoulder width, to stay distance-invariant) for
`INTERACTION_DURATION_THRESHOLD` **continuous** seconds — not on the
first frame they happen to be close. Without this duration gate,
students seated at normal exam spacing would trigger a violation every
single capture cooldown, for the entire exam, just from sitting still.

### 4.3 Object → student attribution

A detected phone is matched to the **nearest tracked student** within a
radius scaled by that student's shoulder width. This is a **best-effort
proximity guess, not a verified possession check.** Two students seated
close together could result in a misattribution. State this plainly if
asked in a defense — don't oversell it as certain.

### 4.4 Per-camera model isolation

Each camera gets its **own** YOLO pose and object model instances. This
isn't just for parallelism — ByteTrack's tracking state lives on the
model/predictor object itself. Sharing one model across two camera
streams would let track IDs (and therefore student identities) cross-
contaminate between physically different cameras.

---

## 5. Camera management

Cameras are managed entirely from the dashboard, not by editing code:

| Action | What it does |
|---|---|
| **Scan for Local Cameras** | Probes local device indices (0–3) not already registered and lists any found. Can take a few seconds — it's opening/closing each device to test it. |
| **Add IP Camera** | Registers a camera from a URL (`http://`, `https://`, or `rtsp://`) — e.g. a phone running an IP Webcam app. |
| **Use** | Opens the camera device and loads its YOLO models (if not already loaded), then starts streaming/processing. |
| **Pause** | Fully releases the camera device (frees it for another app) and clears that camera's in-progress tracking state. Loaded models stay cached, so resuming is fast. |
| **Remove** | Releases the device and forgets the camera entirely. You'd need to Scan/Add it again to bring it back. |

A camera is registered but idle until you click **Use** — this is
deliberate, so you're not loading YOLO weights or holding a device open
for a camera you haven't decided to use yet.

**Performance note:** each active camera roughly multiplies total
inference load, since frames are processed sequentially per stream in
the same process. Two active cameras is a reasonable target for a
laptop demo — benchmark on your actual hardware before assuming more
will run smoothly live.

---

## 6. Tunable thresholds

All of the following are hand-calibrated defaults, **not learned
values**, at the top of `app.py`. Test against real seated footage at
your actual camera distance/angle before trusting them on demo day.

| Constant | Default | Meaning |
|---|---|---|
| `HEAD_TURN_RATIO_THRESHOLD` | `0.6` | Horizontal nose-displacement ratio that counts as "turned" (see §4.1) |
| `GAZE_DURATION_THRESHOLD` | `3.0`s | How long a turn must be sustained before it's flagged |
| `SPINE_TILT_THRESHOLD_DEG` | `25.0`° | Torso tilt considered "leaning" |
| `PROXIMITY_MULTIPLIER` | `1.8` | How close (× average shoulder width) counts as "close" between two students |
| `INTERACTION_DURATION_THRESHOLD` | `3.0`s | How long proximity must be sustained before it's flagged (see §4.2) |
| `ATTRIBUTION_RADIUS_MULTIPLIER` | `4.0` | How close (× shoulder width) an object must be to a student to be attributed to them |
| `MIN_KEYPOINT_CONF` | `0.5` | Minimum confidence before a keypoint (nose/shoulders/hips) is trusted |
| `SMOOTHING_WINDOW` | `5` frames | Rolling-average window for the head-turn ratio |
| `CAPTURE_COOLDOWN` | `3.0`s | Minimum time between evidence photos, per camera |

---

## 7. Project structure

```
.
├── app.py              # Flask app, camera registry, detection pipeline
├── database.py         # SQLite schema + query helpers
├── templates/
│   └── index.html      # Dashboard shell (camera grid is JS-rendered)
├── static/
│   ├── style.css
│   ├── script.js        # Dashboard logic: camera mgmt, polling, gallery
│   └── violations/
│       └── <session>/
│           └── <camera_id>/
│               └── *.jpg   # Evidence photos, organized per camera
└── cerberus.db          # SQLite database (auto-created on first run)
```

> Note: `app.py` uses `render_template`, which expects `index.html` in a
> `templates/` folder, and `url_for('static', ...)` for `style.css` /
> `script.js`, which expects a `static/` folder. Place the files
> accordingly if you're setting this up fresh.

---

## 8. Database schema

Table `violations` (SQLite, `cerberus.db`):

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `session_id` | TEXT | Currently a single hardcoded session (`SESSION_ROOM_A1`) |
| `violation_type` | TEXT | e.g. `OBJECT_CELL_PHONE`, `PROLONGED_SIDE_GAZE`, `STUDENT_INTERACTION` |
| `confidence` | REAL | |
| `image_path` | TEXT | Relative to `static/` |
| `student_track_id` | INTEGER, nullable | Attributed student, if any (see §4.3 for object violations) |
| `camera_id` | TEXT, nullable | Which camera caught it |
| `timestamp` | DATETIME | SQLite default is **UTC**, not local time — see §10 |

Migrations for `student_track_id` and `camera_id` run automatically on
startup (`init_db()`) if you're upgrading an existing `cerberus.db`.

---

## 9. API routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard |
| `/video_feed/<camera_id>` | GET | MJPEG stream for one camera |
| `/cameras` | GET | List all registered cameras + status |
| `/cameras/scan` | POST | Probe for unregistered local devices |
| `/cameras/add` | POST | `{source, label}` — register a new camera |
| `/cameras/<id>/activate` | POST | Open device + load models, mark active |
| `/cameras/<id>/pause` | POST | Release device, keep models cached |
| `/cameras/<id>/remove` | POST | Release device, forget camera |
| `/start_monitoring` / `/stop_monitoring` | POST | Session-wide arm/disarm |
| `/reset_session` | POST | Wipe all violations + evidence photos (requires monitoring stopped) |
| `/get_violations` | GET | Recent violation records |
| `/system_stats` | GET | Aggregate counts for the stats strip |

None of these routes are authenticated — fine for a controlled demo on a
closed network, a real gap if this were ever deployed for real. Worth
one line in a "future work" section of your report.

---

## 10. Known limitations (be upfront about these)

- **Object → student attribution is a proximity guess**, not a verified
  possession check (§4.3).
- **Track IDs are not guaranteed to survive full occlusion.** A student
  fully blocked by a neighbor and then re-appearing may get a new
  tracker ID, losing violation-history continuity for that person.
- **Timestamp mismatch:** evidence filenames use local time
  (`datetime.now()`), but the database's default timestamp is UTC. In
  the Philippines that's an 8-hour offset between a photo's filename and
  its logged DB timestamp.
- **No authentication** on any route, including camera management and
  session reset.
- **Small/occluded objects at range** (a phone mostly hidden under a
  desk, at the back of a wide shot) will have lower recall — this is an
  inherent limitation of the pretrained COCO object detector at
  classroom distances, not something tunable away.
- **All thresholds in §6 are hand-calibrated**, not learned from data.
  If your professor expects a trained/fine-tuned component as part of
  the AI contribution, that's a deliberate scope decision made earlier
  in this project's development, not an oversight — revisit it if that
  expectation is still live.

---

## 11. Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| Camera tile stuck on "Not in use" after clicking Use | Device may be held open by another app (Zoom, another browser tab, etc.) — close it and try again |
| Violations flooding in for people just sitting still | Re-check `PROXIMITY_MULTIPLIER` / `INTERACTION_DURATION_THRESHOLD` against your actual seating distance |
| Side-gaze still firing while reading | Re-check `HEAD_TURN_RATIO_THRESHOLD` and confirm the camera has a clear, front-on view of shoulders |
| Reset button refuses to run | Monitoring must be stopped (paused) first — this is intentional, not a bug |
| Second camera won't add | Confirm the IP camera URL is reachable from the machine running Flask (same network, correct port, `/video` path if using IP Webcam) |