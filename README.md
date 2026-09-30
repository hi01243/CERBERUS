# CERBERUS AI: Classroom Examination Monitoring System

A Flask app that watches classroom cameras during an exam, flags possible violations with computer vision, and records the session so a human can review each flag in context.

CERBERUS flags things for review. It does not decide that anyone cheated.

## What it detects

| Violation | How it works |
|---|---|
| Unauthorized device | YOLO object detection for cell phones and smartwatches (COCO classes `cell phone` and `clock`, used as a smartwatch proxy). A wall clock is filtered out if it's stationary or not near a wrist. Matched to the nearest established student. |
| Prolonged side gaze | YOLO-pose keypoints give a head-turn ratio (nose offset from the shoulder midpoint, corrected for seat angle). Each student has their own "forward" baseline, calibrated in the first 2 seconds and slowly adapted if they settle into a new position. A flag needs the turn held for 3 seconds. |
| Student interaction / desk leaning | Shoulder-to-shoulder proximity (scaled by shoulder width) or excessive spine tilt, held for 3 seconds. |
| Reaching / note passing | A wrist keypoint held inside a neighboring student's zone for 1 second — catches an outstretched arm even when the two torsos aren't close enough to trip the interaction check above. |

Students are tracked across frames with ByteTrack. A track has to exist for at least a second before it can be attributed a violation, so a passing proctor or a momentary tracking glitch can't trigger a false flag.

If a large fraction of visible, already-calibrated students all read as turned in the same frame, it's treated as a probable camera bump (not mass cheating): their calibration is dropped and silently reseeded instead of raising a wall of false alerts.

## How evidence works

- **Session recording:** each camera records its own video while monitoring is armed, auto-rotating to a new file every 5 minutes so a crash loses at most one chunk instead of the whole exam. Any file left open by a previous crash is recovered on startup.
- **Incidents, not screenshots:** repeated triggers for the same student and violation type within 10 seconds become one incident with a start time, an end time, and a confidence score, plus one thumbnail.
- **Review page (`/review`):** shows the video, timeline markers, and the incident list. Clicking an incident plays the recording from 5 seconds before the alert. Incidents can be **dismissed** (marked as reviewed false positives) without deleting the underlying evidence; filter the list to active or dismissed only.
- The dashboard's evidence popup has a "Watch recording" button that opens the same page at that incident.

## Accounts & access

- The app binds to `127.0.0.1` by default — set `CERBERUS_HOST=0.0.0.0` if you need it reachable from another device (e.g. viewing from a phone), and only do that on a network you trust.
- Every page requires signing in. Accounts are per-user, with passwords hashed (never stored in plaintext).
- Creating an account requires an invite code, printed to the console each time the app starts unless you set `CERBERUS_INVITE_CODE` yourself. Share that code with whoever needs an account; each person then sets their own username and password.
- Reset Session and incident dismissal are attributed to whoever is signed in (printed to the console as a lightweight audit trail).

## Sessions

- `CURRENT_SESSION` groups all recordings, incidents, and violations. Start a new one between exams (dashboard header, "New" button) so two exams the same day don't share data — each camera gets a fresh recorder and incident tracker. Requires monitoring to be paused first.
- **Reset Session** permanently deletes everything for the current session (recordings, incidents, evidence, DB rows). It requires monitoring to be paused and requires typing the exact confirmation phrase shown in the prompt — a stray click can't trigger it.

## Features

- Multi-camera support: local webcam (Scan), IP camera by URL, or a video file path (for offline demos or replaying the same footage while tuning thresholds). Each camera runs its own detection thread — closing the browser tab does not stop detection or recording, and multiple tabs can view the same feed without interfering with each other.
- ARM / PAUSE monitoring for the whole session.
- Live dashboard with counters, an audit log, and an evidence gallery with filters (devices, collab, reaching, gaze).

## Requirements

- Python 3.10+
- `flask`, `opencv-python`, `ultralytics`, `numpy`, `werkzeug` (installed automatically with Flask)
- **ffmpeg** on your PATH (strongly recommended). OpenCV records with a codec browsers usually can't play, and ffmpeg converts each recording to H.264 after it closes. Without it the review page may not play the videos.

```bash
pip install flask opencv-python ultralytics numpy
```

The YOLO weights (`yolo11n-pose.pt`, `yolo11n.pt`) are downloaded automatically by Ultralytics on first run.

## Run

```bash
python app.py
```

On startup, note the printed **invite code** (for creating accounts) — set `CERBERUS_PASSWORD`-style env vars ahead of time for a real exam rather than relying on the random one:

```bash
export CERBERUS_INVITE_CODE="something-only-your-proctors-know"
export CERBERUS_SECRET_KEY="a-long-random-string"   # optional: keeps logins across restarts
python app.py
```

Then open http://127.0.0.1:5000, sign up (with the invite code) or sign in, and:

1. Scan for a local camera, or add an IP camera URL / video file path.
2. Click **Use** on the camera tile.
3. Click **ARM MONITORING**.
4. When the exam ends, click **PAUSE SYSTEM** and wait a few seconds. Then open **Review recordings**.

Keep the dashboard tab open at least until a camera is activated — after that, detection and recording run in a background thread independent of the browser.

## Project structure

```
app.py                  Flask app, detection loop, camera registry, routes
auth.py                 Per-user accounts (hashed passwords, invite codes)
recording.py            Session recorder, incident tracker, crash recovery
database.py             SQLite logging for the violations table
templates/index.html    Live dashboard
templates/review.html   Incident review page
templates/login.html    Sign in
templates/signup.html   Create account (invite code required)
static/script.js        Dashboard logic
static/style.css        Dashboard styling
static/violations/      Incident thumbnails (generated)
recordings/             Session videos (generated)
incidents.db            Incident index (generated)
users.db                Accounts (generated)
```

## Configuration

Thresholds are at the top of `app.py`. They are hand-calibrated, not learned, so test them in a real session at real exam spacing before relying on them.

| Setting | Default | Effect |
|---|---|---|
| `HEAD_TURN_RATIO_THRESHOLD` | 0.6 | How far a head must turn to count as looking sideways. Lower is more sensitive. |
| `GAZE_DURATION_THRESHOLD` | 3.0 s | How long a turn must be held before it is flagged. |
| `PROXIMITY_MULTIPLIER` | 1.8 | How close two students must be to count as interacting. |
| `INTERACTION_DURATION_THRESHOLD` | 3.0 s | How long they must stay close. |
| `REACH_ZONE_RADIUS_MULTIPLIER` | 1.3 | How large a neighbor's "personal zone" is, for the reach check. |
| `REACH_DURATION_THRESHOLD` | 1.0 s | How long a wrist must stay in that zone. |
| `CALIBRATION_DURATION` | 2.0 s | How long the initial "forward" baseline is sampled. |
| `CAMERA_BUMP_RATIO` / `CAMERA_BUMP_MIN_STUDENTS` | 0.6 / 3 | Fraction of visible students turned at once, and minimum headcount, before it's treated as a camera bump rather than mass cheating. |
| `TRACK_MIN_AGE_SEC` | 1.0 s | How long a track must exist before it can be attributed a violation. |

Recording settings (`RECORD_FPS`, `RECORD_MAX_WIDTH`, `RECORD_CHUNK_SEC`, `INCIDENT_GAP_SEC`) are at the top of `recording.py`.

## Known limitations

- Detection is heuristic and **has only been tested solo**, never with a full class in frame. Interaction and reach thresholds in particular are a first guess, not validated at real seating density.
- Phone-to-student and reach attribution use nearest-shoulder matching, not verified possession — a phone or reach can be misattributed to the wrong student.
- Head-turn detection uses pose keypoints, not eye tracking. It can't tell what a student is looking at.
- Student track IDs are per-camera and don't persist across a pause/resume or link the same person across two cameras.
- Recordings are only playable once a chunk closes (on PAUSE, or every 5 minutes during a long session).
- No named-student identity (no face recognition) — a violation is tied to a track number, not a roster entry. See the project notes for why (pixel-distance constraints at classroom range) and what a seat-mapping approach would look like if named attribution is ever required.
- Built and tested as a college demo, not for production deployment.

## Privacy

Recordings contain video of students and their surroundings. Get consent before use, and delete recordings once the review period ends (Reset Session does this, with confirmation required). Do not commit `recordings/`, `incidents.db`, `static/violations/`, or `users.db` to git — the included `.gitignore` excludes them.