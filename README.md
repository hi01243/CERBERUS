# CERBERUS AI: Classroom Examination Monitoring System

A Flask app that watches classroom cameras during an exam, flags possible violations with computer vision, and records the session so a human can review each flag in context.

CERBERUS flags things for review. It does not decide that anyone cheated.

## What it detects

| Violation | How it works |
|---|---|
| Unauthorized device | YOLO object detection for cell phones and smartwatches (COCO classes `cell phone` and `clock`, used as a smartwatch proxy). Each detection is matched to the nearest student. |
| Prolonged side gaze | YOLO-pose keypoints give a head-turn ratio (nose offset from the shoulder midpoint, corrected for seat angle). Each student has their own "forward" baseline, calibrated in the first 2 seconds and slowly adapted if they settle into a new position. A flag needs the turn to be held for 3 seconds. |
| Student interaction / desk leaning | Shoulder-to-shoulder proximity (scaled by shoulder width) or excessive spine tilt, held for 3 seconds. |

Students are tracked across frames with ByteTrack, so each flag carries a student ID.

## How evidence works

- **Session recording:** each camera records one video per ARM to PAUSE run (10 fps, up to 960 px wide).
- **Incidents, not screenshots:** repeated triggers for the same student and violation type within 10 seconds become one incident with a start time, an end time, and a confidence score. Each incident keeps one small thumbnail.
- **Review page (`/review`):** shows the video, timeline markers, and the incident list. Clicking an incident plays the recording from 5 seconds before the alert. The evidence popup on the dashboard has a "Watch recording" button that opens the same page.

## Features

- Multi-camera support. Add local cameras (Scan) or IP cameras (URL) from the dashboard, then use, pause, or remove each one.
- ARM / PAUSE monitoring for the whole session.
- Live dashboard with counters, an audit log, and an evidence gallery with filters.
- Reset session, which deletes all logged violations, thumbnails, incidents, and recordings.

## Requirements

- Python 3.10+
- `flask`, `opencv-python`, `ultralytics`, `numpy`
- **ffmpeg** on your PATH (strongly recommended). OpenCV records with a codec browsers usually can't play, and ffmpeg converts each recording to H.264 after PAUSE. Without it the review page may not play the videos.

```bash
pip install flask opencv-python ultralytics numpy
```

The YOLO weights (`yolo11n-pose.pt`, `yolo11n.pt`) are downloaded automatically by Ultralytics on first run.

## Run

```bash
python app.py
```

Open http://127.0.0.1:5000, then:

1. Scan for a local camera, or add an IP camera URL (for example, the IP Webcam Android app at `http://<phone-ip>:8080/video`).
2. Click **Use** on the camera tile.
3. Click **ARM MONITORING**.
4. When the exam ends, click **PAUSE SYSTEM** and wait a few seconds. Then open **Review recordings**.

Keep the dashboard tab open while monitoring. Frames are only processed while a browser tab is streaming the camera, so closing the tab stops detection and recording.

## Project structure

```
app.py                  Flask app, detection loop, camera registry, routes
recording.py            Session recorder and incident tracker
database.py             SQLite logging for the violations table
templates/index.html    Live dashboard
templates/review.html   Incident review page
static/script.js        Dashboard logic
static/style.css        Dashboard styling
static/violations/      Incident thumbnails (generated)
recordings/             Session videos (generated)
incidents.db            Incident index (generated)
```

## Configuration

Thresholds are at the top of `app.py`. They are hand-calibrated, not learned, so test them in a real session at real exam spacing before relying on them.

| Setting | Default | Effect |
|---|---|---|
| `HEAD_TURN_RATIO_THRESHOLD` | 0.6 | How far a head must turn to count as looking sideways. Lower is more sensitive. |
| `GAZE_DURATION_THRESHOLD` | 3.0 s | How long a turn must be held before it is flagged. |
| `PROXIMITY_MULTIPLIER` | 1.8 | How close two students must be to count as interacting. |
| `INTERACTION_DURATION_THRESHOLD` | 3.0 s | How long they must stay close. |
| `CALIBRATION_DURATION` | 2.0 s | How long the initial "forward" baseline is sampled. |

Recording settings (`RECORD_FPS`, `RECORD_MAX_WIDTH`, `INCIDENT_GAP_SEC`) are at the top of `recording.py`.

## Known limitations

- Detection is heuristic. Glasses, seating angle, lighting, and normal habits can produce false flags, so every flag needs human review.
- Phone-to-student attribution uses the nearest shoulders. A phone on a neighboring desk can be attributed to the wrong student.
- Head-turn detection uses pose keypoints, not eye tracking. It can't tell what a student is looking at.
- Students in the back rows (seen from the front) have less reliable keypoints.
- Recordings are only playable after PAUSE, once they are finalized.
- Built and tested as a college demo, not for production deployment.

## Privacy

Recordings contain video of students and their surroundings. Get consent before use, limit who can open `/review`, and delete recordings once the review period ends (Reset Session does this). Do not commit `recordings/`, `incidents.db`, or `static/violations/` to git. The included `.gitignore` excludes them.