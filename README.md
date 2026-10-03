<p align="center">
  <img src="static/logo-full.svg" alt="CERBERUS" width="300">
</p>

AI-assisted exam proctoring dashboard. CERBERUS watches one or more camera feeds, uses YOLO11 pose and object models to flag suspicious behaviour, records every session, and gives a reviewer a timeline to check each flag against the recording.

It flags three things:

| Rule | What it catches |
|------|-----------------|
| **Phone** | A cell phone in a student's hand or on the desk (also flags some calculators) |
| **Reach** | A hand-off between neighbouring students, such as a passed note |
| **Gaze** | A prolonged sideways head turn |

> **CERBERUS flags, it does not judge.** Every flag is a prompt for a human reviewer, who can dismiss false positives without deleting the evidence. Only use it where students have been told they are being recorded and your institution's policies allow it.

---

## Features

- **Live monitoring dashboard** with a camera grid, live counters, an audit log, and an evidence gallery you can filter by flag type
- **Multiple cameras**: local webcams, IP/RTSP streams (for example a phone), and video files that loop for demos. Cameras are added from the dashboard without a restart
- **Per-exam sessions**: each exam gets its own recordings and incidents, so two exams on the same day never mix
- **Per-account session ownership**: every session belongs to the account that created it. You only see, review, play back, dismiss flags on, or delete your own sessions, and evidence images are protected the same way
- **Review page**: play an incident from 5 seconds before the alert, jump around on an incident timeline, pick between recordings, filter active or dismissed flags, and dismiss false positives
- **Multi-camera grouping**: the same violation seen by several cameras within 5 seconds shows as one card with a camera picker (display only; nothing is merged or deleted)
- **Live rule toggles**: switch phone, reach, or gaze detection on or off per exam, for example when calculators are allowed
- **Accounts and MFA**: username and password sign-up, optional authenticator-app (TOTP) MFA, and MFA-gated session deletion
- **Crash recovery**: recordings left half-written by a crash or power loss are finalised on the next start so they stay playable

---

## Requirements

- Python 3.9+
- A webcam, IP camera, or video file to test with
- Packages: Flask (which brings Werkzeug, used for password hashing), OpenCV (`opencv-python`), Ultralytics, NumPy. MFA codes are generated with the standard library, so no extra TOTP package is needed
- The YOLO weights `yolo11n-pose.pt` and `yolo11n.pt`. Ultralytics downloads them on first run unless you place them next to `app.py`, so the first run needs internet access
- **Recommended:** `ffmpeg` on your PATH. Without it, some recordings may not play in the browser

## Installation

```bash
git clone <your-repo-url> cerberus
cd cerberus

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install flask opencv-python ultralytics numpy
```

## Running

```bash
python app.py
```

Then open <http://127.0.0.1:5000>. The console prints the build ID on startup. Create your first account at `/signup`.

### Configuration

| Environment variable | Purpose | Default |
|----------------------|---------|---------|
| `CERBERUS_SECRET_KEY` | Flask session key. Set it to stay signed in across restarts | Random key each run (everyone is signed out on restart) |
| `CERBERUS_HOST` | Interface to listen on. Use `0.0.0.0` to reach the dashboard from other devices | `127.0.0.1` |

The server listens on port 5000.

> **Sign-up is open.** Anyone who can reach the login page can create an account. The default `127.0.0.1` setting keeps that to your own machine. If you set `CERBERUS_HOST=0.0.0.0`, only do so on a network you trust.

---

## Using CERBERUS

1. **Sign up** at `/signup` with a username (3-32 characters: letters, numbers, `_ . -`) and a password (8+ characters). You can enable authenticator MFA at the same time, or later at `/security/mfa`.
2. On the **Sessions** page (`/`), either name a new session and click **Create session and start**, or open one of your existing sessions. A new account starts with an empty list.
3. On the **Monitor** page (`/monitor`):
   - Camera 0 is registered by default. Click **Scan for Local Cameras**, or add an IP camera URL (`http://`, `https://`, `rtsp://`) or a video file path.
   - Click **Use** on a camera tile to start it.
   - Choose which **Detection Rules** apply to this exam.
   - Click **ARM MONITORING** to start flagging, and **PAUSE SYSTEM** to stop. Recordings are finalised on pause.
4. Open **Review Recordings** (`/review`) to go through flags, watch the footage, and dismiss false positives. Use **Sessions** in the header to return to the home page.

### Accounts and session ownership

- A session belongs to the account that created it. Another account that asks for your session, recording, incident, or evidence image gets a 404, so it cannot even tell that the session exists.
- There is **one live session for the whole app**. Cameras, ARM/PAUSE, the live feed, and the rule toggles belong to whoever owns the live session. Another account must create its own session first, which is only possible while monitoring is paused, and doing so takes over the live controls.
- Session names are unique across accounts. Creating a name that another account already uses is refused.
- **Upgrading from an older build:** sessions that existed before ownership was added are given to the first account that was created. If no account exists yet, the first sign-up inherits them.

### Deleting data

Deletion is deliberately hard to do by accident:

- **Delete a session** (Sessions page): monitoring must be paused for the active session, you must have MFA enabled, and you must enter an authenticator code and type the exact session name.
- **Reset session** (Monitor page): monitoring must be paused, and you must type `DELETE ALL DATA` and enter an authenticator code. A fresh session, owned by you, is created afterwards.

Both remove the session's database rows, recordings, and evidence images, and cannot be undone.

> **Keep your MFA setup key somewhere safe.** Once MFA is on, sign-in needs an authenticator code, and there are no backup codes. If you lose the device, you are locked out of that account.

---

## How detection works

### Phone
- Every frame, YOLO11n looks for COCO class 67 (cell phone) at confidence above 0.45.
- A second **wrist-crop pass** runs about 3 times a second: it crops a zoomed window around each wrist and re-checks it, which catches small phones held in a hand. A phone is confirmed after 3 hits within 2 seconds, and it must be close to the wrist.
- A **static-object filter** drops "phones" that never move (seat-number plates, furniture): 10 seconds for full-frame detections unless a hand is near, and 6 seconds for wrist-crop detections.
- Calculators can look like phones to the model, so turn the Phone rule off when they are allowed.

### Gaze
- Head turn is estimated from how far the nose sits sideways from the shoulder line, relative to shoulder width.
- Each student gets a personal baseline from a 2-second calibration, and it adapts slowly if they settle into a new position.
- A turn must pass 0.45 and last 3 seconds to be flagged. Looking down at a paper is excused unless the turn is very strong.
- A held posture (flagged for 8 seconds) is treated as the new normal and re-baselined.
- If most students turn at once, CERBERUS assumes the camera was bumped and recalibrates everyone.
- Optional: `GAZE_REQUIRE_NEIGHBOR` only counts a turn if another student sits on that side. It is off by default.

### Reach (note passing)
A hand-off is detected when either test passes for 1 second:
- **Zone**: a wrist enters a circle around the neighbour's shoulders.
- **Extension**: the arm is stretched out and the wrist has travelled toward the neighbour, which catches a paper passed across an aisle.

Only students who are side by side, in the same row, and at a similar depth are paired, so someone reaching past a student in front or behind is not flagged.

### Tracking quality
Duplicate ("ghost") skeletons are filtered out before they touch any timer, and per-student state survives short tracking drops, so one missed frame does not restart calibration.

> All thresholds are hand-calibrated, not learned. Test with people seated at real exam spacing and tune the constants at the top of `app.py` before relying on it. Setting `DEBUG_OVERLAY = True` draws yaw and pitch numbers beside each student to help.

---

## Project structure

```
cerberus/
├── app.py            # Flask app, camera registry, detection loop, routes
├── database.py       # SQLite violation log (cerberus.db)
├── recording.py      # Session recording and incident tracking (incidents.db)
├── auth.py           # Accounts, session ownership, TOTP MFA (users.db)
├── recordings/       # Session videos, created at runtime
├── templates/
│   ├── sessions.html   # Session browser and landing page
│   ├── index.html      # Live monitor dashboard (/monitor)
│   ├── review.html     # Incident review (/review)
│   ├── login.html
│   ├── signup.html
│   └── mfa_setup.html
└── static/
    ├── script.js       # Monitor dashboard logic
    ├── style.css       # Monitor dashboard styles
    └── violations/     # Evidence thumbnails, per session and camera
```

## HTTP routes

All routes except `/login`, `/signup`, and public static files (such as `style.css`) require sign-in. "Own sessions only" means a request for someone else's session returns 404. "Live owner only" means the signed-in account must own the current live session, otherwise the request is refused (403 for API calls, a redirect to `/` for `/monitor`).

| Route | Method | Access | Purpose |
|-------|--------|--------|---------|
| `/` | GET | Signed in | Sessions page (your sessions only) |
| `/monitor` | GET | Live owner only | Live dashboard |
| `/review?session_id=` | GET | Own sessions only | Incident review |
| `/video_feed/<camera_id>` | GET | Live owner only | MJPEG stream of the annotated feed |
| `/cameras`, `/cameras/scan`, `/cameras/add` | GET / POST | Live owner only | List, scan for, and add cameras |
| `/cameras/<id>/activate`, `/pause`, `/remove` | POST | Live owner only | Control a camera |
| `/start_monitoring`, `/stop_monitoring` | POST | Live owner only | Arm or pause detection |
| `/settings` | GET / POST | Live owner only | Read or change which rules are on |
| `/session` | GET | Live owner only | Current session |
| `/session/new` | POST | Signed in | Start a new session (monitoring must be paused) |
| `/sessions`, `/session_files` | GET | Own sessions only | Session summaries and recordings |
| `/sessions/<id>` | DELETE | Own sessions only, MFA required | Delete a session |
| `/reset_session` | POST | Live owner only, MFA required | Reset the current session |
| `/incidents`, `/incidents/<id>/dismiss` | GET / POST | Own sessions only | List incidents, dismiss a false positive |
| `/get_violations`, `/system_stats` | GET | Live owner only | Recent violations and dashboard counters |
| `/recordings/<path>` | GET | Own sessions only | Serve a recording (supports seeking) |
| `/static/violations/<session>/...` | GET | Own sessions only | Evidence thumbnails |
| `/security/mfa`, `/security/mfa/status`, `/security/mfa/disable` | GET / POST | Signed in | MFA management |

## Known limitations

- Subtle phone use is still easy to miss, and a phone under a desk or in a lap may not be seen.
- Reach and gaze use 2D geometry from one camera angle, so overlapping students at different depths can confuse them.
- Matching a phone to a student is a best-effort nearest-shoulder guess, not verified possession.
- Calculators can be flagged as phones, so use the Phone rule toggle.
- Everything is tuned for seated students at normal exam spacing. Small objects in the back rows of a large room are only a few pixels wide and will be missed.
- There is only one live session at a time, shared by the whole app. Any account can take over the live controls by creating a session while monitoring is paused.
- Video lag on weaker machines. Run on a GPU or lower the number of cameras for large rooms.
