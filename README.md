# CERBERUS — Classroom Examination Monitoring System

CERBERUS is a computer-vision assistant for exam proctors. It watches one or more classroom cameras, flags **suspicious behavior** in real time, records the whole session, and lets a reviewer jump to each flagged moment afterwards.

> **CERBERUS flags; a human decides.** Every alert is a prompt for review, not proof of cheating. Read [Limitations](#limitations) before relying on any result.

---

## What it detects

| Detector | What it looks for | Default |
|---|---|---|
| **Phone** | Cell phone detected by YOLO (COCO class 67), attributed to the nearest student | On |
| **Smartwatch** | Clock-like object (COCO class 74) near a visible wrist; static wall clocks are suppressed | Off |
| **Prolonged side gaze** | Head turned left/right for 3 s or more, measured against each student's own calibrated "normal" position | On |
| **Reaching / note passing** | A wrist entering a neighbor's zone, or an arm stretched toward a side-by-side neighbor, for 1 s or more | On |

Each rule can be switched on or off during an exam from the **Detection Rules** row on the dashboard (see [Exam rules](#exam-rules)). Turn **Phones** off if the exam allows calculators, since YOLO cannot tell a calculator from a phone.

### How it works (short version)

1. **Pose + tracking:** `yolo11n-pose` with ByteTrack gives each student a persistent ID and body keypoints.
2. **Duplicate filter:** two skeletons on the same body are collapsed into one so a "ghost" student can't trigger alerts.
3. **Per-student baseline:** for the first 2 s a student is seen, their head position is sampled to set what "looking forward" means for them. The baseline adapts slowly if they settle into a new position, and re-baselines if a posture is held for a long time. This is what stops a student seated at the edge of the frame being flagged.
4. **Gaze:** the nose's sideways offset from the shoulder line, as a ratio of shoulder width. Moderate readings with the head tilted down are treated as looking at one's own paper and ignored; strong sideways turns are always flagged.
5. **Reach:** only students at a similar depth, in the same row, and genuinely side by side are checked for a hand-off. This is essential on angled cameras, where students in different rows overlap in the image.
6. **Objects:** `yolo11n` finds phones/clocks; clock detections must be near a wrist and not static to count as a watch.
7. **Incidents:** repeated triggers for the same person and reason are grouped into one incident with one thumbnail and a timestamp in the session recording.

---

## Requirements

- Python 3.10+
- A webcam, a phone running **IP Webcam**, an RTSP/HTTP stream, or a video file
- A GPU is strongly recommended for more than a couple of students (CPU works but is slow)
- `ffmpeg` on your PATH is recommended for recordings

```bash
pip install flask opencv-python ultralytics numpy
```

The YOLO weights (`yolo11n-pose.pt`, `yolo11n.pt`) are downloaded automatically by Ultralytics on first run.

## Run it

```bash
python app.py
```

Open <http://127.0.0.1:5000>, sign up (an invite code is required, see `auth.py`), and log in. The console prints the build ID on startup so you can confirm which version is running.

| Environment variable | Purpose |
|---|---|
| `CERBERUS_SECRET_KEY` | Keeps logins valid across restarts (otherwise a random key is generated each run) |
| `CERBERUS_HOST` | Set to `0.0.0.0` to reach the dashboard from other devices (default is localhost only) |

## Using it

1. **Add a camera** on the dashboard: scan for local cameras, or enter an IP camera URL (for IP Webcam use `http://<phone-ip>:8080/video`), an RTSP URL, or a video file path. Video files loop, which is handy for demos.
2. Click **Use** on the camera tile to start it. Cameras can be paused or removed at any time.
3. Set the **Detection Rules** for this exam.
4. Click **ARM MONITORING**. Students are calibrated during their first couple of seconds on camera, so ask them to sit normally.
5. Watch the live log and evidence gallery. **PAUSE SYSTEM** stops monitoring.
6. Afterwards open **REVIEW RECORDINGS**: a video player with a marker for each incident. Reviewers can dismiss false positives without deleting the evidence.
7. Use **New** next to the session name to start a separate session for each exam.
8. **RESET SESSION** permanently deletes that session's recordings, thumbnails and database rows. It requires monitoring to be paused and the phrase `DELETE ALL DATA` to be typed.

### Exam rules

`GET` / `POST /settings` reads and changes the rules; the dashboard checkboxes use it. Every change is printed to the console with the username, and the active rules are printed when monitoring starts, so there is a record of why something was or wasn't flagged.

```json
{ "phone": true, "watch": false, "reach": true, "gaze": true }
```

## Camera placement matters most

Most of the tuning problems seen in testing came from the camera, not the code.

- Mount the camera **above head height, at the front, looking back** at the class, so rows run roughly horizontally across the image.
- A low camera in a corner makes students overlap and makes row-based logic (reach) unreliable.
- Students should be at least ~60 px wide across the shoulders in the image for reliable pose and head-turn readings. Back rows of a large room may not be detected.
- Use the highest stream resolution your hardware can keep real-time.

## Tuning

All thresholds are **hand-calibrated, not learned**. They sit at the top of `app.py`. Test at real exam spacing before relying on them.

| Constant | Default | Effect |
|---|---|---|
| `HEAD_TURN_RATIO_THRESHOLD` | 0.45 | Sideways head offset that counts as a turn |
| `HEAD_TURN_STRONG_RATIO` | 0.7 | Turns above this are never excused as "looking down" |
| `PITCH_DOWN_DELTA` | 0.30 | Nose drop that marks "looking down at own paper" |
| `GAZE_DURATION_THRESHOLD` | 3.0 s | How long a turn must last before it is flagged |
| `GAZE_REQUIRE_NEIGHBOR` | False | If True, a turn only counts when another student sits on that side |
| `REACH_DURATION_THRESHOLD` | 1.0 s | How long a reach must last |
| `REACH_SAME_ROW_MULTIPLIER` | 1.0 | Max vertical offset between paired students |
| `REACH_DEPTH_RATIO_MIN/MAX` | 0.75 / 1.33 | Shoulder-width ratio band for "same depth" |
| `REACH_MIN_SIDE_GAP_MULTIPLIER` | 1.2 | Minimum horizontal gap so someone directly behind isn't paired |

The overlay shows `yaw` and `dn` (nose drop) beside each student, and the console logs every state change, which is how these numbers should be tuned.

## Project layout

```
app.py          Flask app, camera registry, detection loop, routes
database.py     SQLite violation log
recording.py    Session recording and incident grouping
auth.py         User accounts (invite code required to sign up)
templates/      index.html (dashboard), review.html, login/signup
static/         script.js, style.css, saved evidence thumbnails
recordings/     Session videos (generated)
```

Keep generated data (recordings, evidence images, `*.db`) out of version control.

## Design decisions

- **Student-interaction (shoulder proximity) detection was removed.** It fired on normal seating, could not be checked against the test data, and duplicated what side gaze and reach already cover.
- **Reach only checks students who are side by side.** From a camera that isn't overhead, a student sitting behind another looks close to them in the image, and 2D geometry cannot separate that from a real hand-off.
- **Evidence is a recording plus a thumbnail per incident**, not repeated screenshots, so a reviewer sees context.

## Testing

Test with clips of each behavior and, just as importantly, clips of **normal** behavior. Recommended:

- The public [Student Cheating Behavior dataset](https://www.kaggle.com/datasets/fairouzfarouq/student-cheating-behavior-dataset) (exchanging papers, looking at another student's paper, cheat sheet, phone, not cheating)
- Staged clips with classmates from your own camera and room

Record how often each detector fires per minute on normal-behavior clips, and which detector fires on each cheating clip:

| Clip type | Expected detector | Fired? | False alerts / min |
|---|---|---|---|
| Not cheating | none | | |
| Looking at another student | Gaze | | |
| Exchanging papers | Reach | | |
| Phone | Phone | | |
| Cheat sheet | none (not supported) | | |

## Limitations

- **Cheat sheets are not detected.** No model here recognizes paper notes, and one in a lap or under a desk is invisible to a room camera.
- **Phones are only detected when visible.** Small, tilted, hand-covered or hidden phones are often missed, especially in back rows. Calculators may be flagged as phones.
- **Smartwatch detection is weak.** YOLO cannot tell a smartwatch from a regular watch, and a wrist is tiny at classroom distance, so it is off by default.
- **Reach only covers side-by-side neighbors**, not notes passed forward or backward between rows.
- **Gaze measures head direction, not eye direction**, and only left/right; turning around to look behind is not covered.
- **Thresholds are hand-set** and were tuned on a small number of clips. Expect to re-tune for a new room or camera position.
- **False positives and false negatives both happen.** Treat alerts as leads for human review.
- **Privacy:** recording students requires consent and a retention policy appropriate to your institution. Use **Reset Session** or delete recordings once they are no longer needed.