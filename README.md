<p align="center">
  <img src="static/logo-full.svg" alt="CERBERUS" width="300">
</p>

AI-assisted exam proctoring dashboard. CERBERUS watches one or more camera feeds, uses YOLO11 pose and object models to flag suspicious behaviour, records every session, and gives a reviewer a timeline to check each flag against the recording.

It flags three things:

| Rule | What it catches |
|------|-----------------|
| **Phone** | A cell phone in a student's hand or on the desk (also flags some calculators), plus a lower-confidence "hands held together at chest height" fallback |
| **Reach** | A hand-off between neighbouring students, such as a passed note |
| **Gaze** | A prolonged sideways head turn toward a neighbouring student |

Internally, flags are logged under these types:

| Flag type | Rule | Fixed confidence |
|-----------|------|------------------|
| `OBJECT_CELL_PHONE` | Phone | The detector's own score |
| `OBJECT_PHONE_POSTURE` | Phone (hands-at-chest fallback) | 0.60 |
| `REACHING_NEIGHBOR_ZONE` | Reach | 0.85 |
| `PROLONGED_SIDE_GAZE` | Gaze | 0.95 |

> **CERBERUS flags, it does not judge.** Every flag is a prompt for a human reviewer, who can dismiss false positives without deleting the evidence. Only use it where students have been told they are being recorded and your institution's policies allow it.

---

## Features

- **Live monitoring dashboard** with a camera grid, live counters, an audit log, and an evidence gallery you can filter by flag type
- **Multiple cameras**: local webcams, IP/RTSP streams (for example a phone), and video files that loop for demos. Cameras are added from the dashboard without a restart, and each one can be started, paused, or removed on its own. A scan option probes the first few local device indices for webcams that are not registered yet
- **Per-exam sessions**: each exam gets its own recordings and incidents, so two exams on the same day never mix
- **Per-account session ownership**: every session belongs to the account that created it, and other accounts cannot see it
- **Recording in 5-minute files**: each camera saves to `recordings/<session>/<camera>/` at 10 fps and at most 960 px wide, and evidence images go to `static/violations/<session>/<camera>/`. A crash costs at most one file
- **Incident grouping**: repeats of the same flag on the same student within 10 seconds are one incident, and students flagged at the same moment each get their own
- **Review page**: play an incident from 5 seconds before the alert, jump around on an incident timeline, pick between recordings, filter active or dismissed flags, and dismiss false positives
- **Multi-camera grouping**: the same violation seen by several cameras within 5 seconds shows as one card with a camera picker (display only; nothing is merged or deleted)
- **Live rule toggles**: switch phone, reach, or gaze detection on or off per exam, for example when calculators are allowed. Turning Phone off also turns off the wrist-crop check and the hands-at-chest fallback. Changes are written to the console log with the username
- **Recalibrate button**: restarts every student's neutral-pose calibration without cutting the recording, for when students change seats
- **Accounts and MFA**: username and password sign-up, optional authenticator-app (TOTP) MFA that can be switched off again with a current code, and MFA-gated account deletion
- **Crash recovery**: recordings left half-written by a crash or power loss are finalised on the next start (a file cut off mid-write may still not play)

---

## Requirements

- Python 3.9+
- A webcam, IP camera, or video file to test with
- The YOLO weights `yolo11n-pose.pt`, `yolo11n.pt`, and `yolo11m.pt`. Ultralytics downloads them on first run unless you place them next to `app.py`, so the first run needs internet access. `yolo11m.pt` (about 40 MB) powers the zoomed-in phone check; if it cannot be loaded, CERBERUS falls back to the nano model and prints a message
- **Recommended:** `ffmpeg` on your PATH. Without it, some recordings may not play in the browser
- **Strongly recommended:** an NVIDIA GPU with a CUDA build of PyTorch. CERBERUS uses it automatically when PyTorch can see it (see [Performance](#performance))

## Installation

```bash
git clone <your-repo-url> cerberus
cd cerberus

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install flask opencv-python ultralytics numpy lapx
```

To use an NVIDIA GPU, install the CUDA build of PyTorch from <https://pytorch.org> before running. Check it with `python -c "import torch; print(torch.cuda.is_available())"`, which should print `True`.

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

The port is fixed at 5000.

> **Sign-up is open.** Anyone who can reach the login page can create an account. The default `127.0.0.1` setting keeps that to your own machine. If you set `CERBERUS_HOST=0.0.0.0`, only do so on a network you trust.

### Switches at the top of `app.py`

All detection thresholds are plain constants at the top of `app.py`. The ones you are most likely to touch:

| Constant | What it does |
|----------|--------------|
| `HEAD_TURN_RATIO_THRESHOLD` | How far the head must turn before it counts as a side turn |
| `GAZE_DURATION_THRESHOLD` | Seconds a side turn must last before it is flagged |
| `GAZE_DRAIN_RATE` | How fast a running gaze timer counts down while the student is not turned |
| `GAZE_REQUIRE_NEIGHBOR` | Only flag a side turn if another student sits on that side |
| `POSE_IMGSZ`, `POSE_CONF` | Pose model input size (default 960) and confidence cutoff |
| `WRIST_CROP_ENABLED`, `WRIST_CROP_MODEL`, `WRIST_CROP_AUGMENT` | The zoomed-in phone check and the model it uses |
| `WRIST_CROP_MAX_CROPS` | Cap on crops per pass (default 40); raise it for large rooms on a GPU |
| `BADGE_FILTER_ENABLED`, `BADGE_WHITE_FRAC` | ID-badge filter and how white a box must be to count as a card |
| `HANDS_POSTURE_ENABLED`, `HANDS_POSTURE_SEC` | The hands-at-chest phone fallback and how long the pose must be held |
| `BASELINE_MAX_ABS` | Largest neutral head offset a baseline may have (see Gaze) |
| `DEBUG_OVERLAY` | Draws tuning numbers and rejected candidates on the video (on by default; these are also recorded unless `RECORD_RAW` is on) |
| `RECORD_RAW` | `True` records the clean frame without overlays, so a run can be replayed later as a video-file camera. `/review` then shows no boxes |
| `VIDEO_CLOCK` | For video-file cameras, timers follow video time instead of wall-clock time, so a clip behaves the same however slowly it plays |

---

## Recommended setup

These figures follow from how the detection geometry works. They are starting points, not guarantees, so confirm them in a test session.

- **Camera:** 2-2.5 m high at the front of the room, facing the students and tilted slightly down, so every face and both shoulders are visible and rows separate in the image.
- **Range:** keep the farthest row within about 4-5 m of a camera. Beyond that, shoulders are too small in the frame for reliable head-turn and wrist tracking. Cover larger rooms with several cameras, each watching a block of roughly 3 rows by 4 seats.
- **Seating:** about 1 m between students sideways and about 1.2 m between rows. Tighter spacing makes a hand-off harder to tell apart from an ordinary arm position.
- **Start-up:** press ARM only when everyone is seated and facing forward, and wait for the on-screen "CALIBRATING" labels to clear before the exam begins.

---

## Using CERBERUS

1. **Sign up** at `/signup` with a username (3-32 characters: letters, numbers, `_ . -`) and a password (8+ characters). You can enable authenticator MFA at the same time, or later at `/security/mfa`.
2. On the **Sessions** page (`/`), name a new session and start it, or open one of your existing sessions.
3. On the **Monitor** page (`/monitor`):
   - Camera 0 is registered by default. Add others by entering a device number, an IP/RTSP URL, or a video file path (the file must exist on the machine running CERBERUS).
   - Click **Use** on a camera tile to start it. You can pause or remove a camera at any time.
   - Choose which **Detection Rules** apply to this exam.
   - Click **ARM MONITORING** to start flagging, and **PAUSE SYSTEM** to stop. Recordings are finalised on pause.
   - Click **RECALIBRATE** if students change seats.
4. Open **Review Recordings** (`/review`) to go through flags, watch the footage, and dismiss false positives. Dismissing cannot be undone.

### Testing with video files

A video-file camera loops its clip and processes every frame, so it plays only as fast as your machine can run the detectors. A slow machine plays the clip in slow motion, but because timers follow video time (`VIDEO_CLOCK`), the flags it produces are the same as in real time.

### Accounts and session ownership

- Another account that asks for your session, recording, incident, or evidence image gets a 404.
- There is one live session for the whole app. If the live session belongs to another account, the live controls refuse your requests (403) and `/monitor` sends you back to the Sessions page until you create your own session.
- Another account can take over the live controls by creating its own session, which is only possible while monitoring is paused.
- Session names are unique across accounts.
- When upgrading from a build without ownership, existing sessions go to the first account created.
- Turning MFA off (at `/security/mfa`) requires a current authenticator code.

### Deleting data

Deletion is deliberately hard to do by accident:

- **Delete a session** (Sessions page): monitoring must be paused if it is the active session, and you must type the exact session name.
- **Reset session** (Monitor page): monitoring must be paused, and you must type `DELETE ALL DATA`. A fresh session, owned by you, is created afterwards.
- **Delete an account**: stop monitoring if the live session belongs to you, enter your password, type your exact username and `DELETE ACCOUNT`, and provide an authenticator code if MFA is enabled. This permanently removes all sessions and evidence owned by that account.

Deleting or resetting a session removes its database rows, recordings, and evidence images. Deleting an account removes all of its owned sessions and evidence. These actions cannot be undone.

> **Keep your MFA setup key somewhere safe.** Once MFA is on, sign-in needs an authenticator code, and there are no backup codes. If you lose the device, you are locked out of that account.

---

## How detection works

Students are tracked with the pose model and ByteTrack. A student needs both shoulders visible (confidence 0.3 or more) to be tracked at all, and a track must be at least 1 second old before it can be blamed for a phone or a reach. Per-student state survives tracking drops of up to 1.5 seconds. Duplicate ("ghost") skeletons are filtered out.

### Phone
- **Full-frame check:** every frame, YOLO11n looks for COCO class 67 (cell phone) at confidence above 0.45.
- **Wrist-crop check:** about 3 times a second, CERBERUS crops a zoomed window around each wrist, and an extra one centred between both hands when they are close together, and re-checks them with the larger `yolo11m` model with test-time augmentation (switched off automatically if unsupported). This catches small or dark phones held in a hand. A cell phone counts at 0.25 confidence or more, and the COCO class "remote" (a held phone is often labelled that way) counts at 0.35 or more. A phone is confirmed after 3 hits within 2 seconds, stays alive for 0.8 seconds between passes, and must be close to the wrist.
- **Faint detections:** when both hands are held together at chest height, weaker detections (down to 0.08) also count, but need 4 hits within 3 seconds.
- **Hands-at-chest fallback:** if both wrists stay together at chest height for 3 seconds, the student is flagged as `OBJECT_PHONE_POSTURE` at a fixed 0.6 confidence, even when no phone box was found. A dropout of up to 1.5 seconds does not reset the hold. Hands resting on the desk are lower than this and do not count.
- **Static-object filter:** drops "phones" that never move (seat-number plates, furniture): 30 seconds for full-frame detections unless a hand is near, and 60 seconds for wrist-crop detections.
- **Size filter:** ignores boxes that are long and thin (more than 3.5 times longer than wide, typically a pen) or small compared with the student's shoulder width. Boxes with a long side under about a fifth of a shoulder width are always dropped, and boxes under about a third of a shoulder width need at least 0.60 confidence.
- **ID-badge filter:** a card hanging on a lanyard sits on the chest midline, so a "phone" in that strip is ignored when any of these is true:
  - it is mostly white (an ID card), even if a hand is on it;
  - a dark strap runs from it up to the neck (a lanyard), even if a hand is resting on it;
  - it keeps a steady position relative to the shoulders while the hands move around it.

  A non-white object held in a hand or in both hands is still flagged. A candidate in the chest area shows "CHECKING..." for about 1.2 seconds while the motion test collects data. The size and badge filters apply to both the full-frame and the wrist-crop detections.

### Gaze
- Head turn is estimated from how far the nose sits sideways from the shoulder line, relative to shoulder width.
- Each student gets a personal baseline from a 2-second calibration, and it adapts slowly if they settle into a new position for about 10 seconds. A baseline further off-axis than 0.45 is clamped, because it means the head was already turned during calibration.
- A turn must pass 0.55 and last 1.5 seconds to be flagged. Brief interruptions (a blink, a noisy frame) do not reset the timer; it drains at half speed instead. A lost nose keypoint keeps the previous side turn alive for up to 0.8 seconds.
- Looking down at a paper is excused unless the turn is very strong (above 0.7). While a gaze timer is already running, the head must drop considerably more before it counts as looking down, so a quick glance down does not cancel a real side gaze.
- **Neighbour check:** a side turn is only flagged if another student sits on the side the head is turned toward, between 0.6 and 5 shoulder widths away and in roughly the same row (the row tolerance widens with distance for diagonal cameras). Looking at a wall, window, or empty seat is ignored, and with `DEBUG_OVERLAY` on, "turn ignored: no neighbor" appears beside the student. A neighbour seen on that side in the last 8 seconds still counts, so a dropped pose does not reset a gaze.
- A held posture (flagged for 30 seconds) is treated as the new normal and re-baselined. When the student returns close to their original position, the original baseline is restored, so turning back to their seat is not flagged as a turn the other way.
- If at least 80% of the students (and at least 6 are visible) turn at once, CERBERUS assumes the camera was bumped and recalibrates everyone (at most once every 5 seconds).

### Reach (note passing)
A hand-off is detected when any of three tests passes for 1 second:
- **Zone**: a wrist enters a circle around the neighbour's shoulders, while the hand is held away from its own owner and is closer to the neighbour than to them. A hand resting on its own desk does not count.
- **Extension**: the arm is stretched out and the wrist has travelled toward the neighbour, which catches a paper passed across an aisle.
- **Mutual hand-off**: both students have a hand out in the middle of the gap between them.

Only students who are side by side, in the same row, and at a similar depth are paired, so someone reaching past a student in front or behind is not flagged. A dropped wrist keypoint is tolerated for 0.4 seconds. One hand-off between two students is logged as a single incident.

> All thresholds are hand-calibrated, not learned. Test with people seated at real exam spacing and tune the constants at the top of `app.py` before relying on it. Setting `DEBUG_OVERLAY = True` draws yaw and pitch numbers beside each student, the wrist keypoints, the wrist-crop boxes with their scores (magenta), candidates that were rejected and why (orange: pen-like, ID badge, far from hand, still checking), and a grey box with the shoulder confidences around anyone the model found but could not track.

---

## Performance

- The heaviest part is the wrist-crop phone check. It runs the medium model with test-time augmentation on every wrist crop (up to 40 per pass) about 3 times a second, so its cost grows with the number of students.
- On a CPU or integrated GPU, expect the live feed and video-file playback to be slow, especially in a full classroom. On an NVIDIA GPU with a CUDA build of PyTorch, the same check takes a fraction of the time.
- Each active camera runs its own detection thread and model copies, so several cameras on one machine share the same compute.
- Lowering `POSE_IMGSZ` from the default 960 to 640 gives more frame rate but weakens detection of small back-row students. Changing `WRIST_CROP_MODEL`, `WRIST_CROP_IMGSZ`, or `WRIST_CROP_AUGMENT` also trades accuracy for speed on faint, dark, or two-handed phones, so re-test with your own clips after any of these changes.

---

## Known limitations

- Subtle phone use is still easy to miss, and a phone under a desk or in a lap may not be seen.
- Smartwatches and cheat sheets are not detected.
- Gaze only covers a student looking at a neighbour to the side. Looking at the student in front or behind is not flagged.
- Reach and gaze use 2D geometry from one camera angle, so overlapping students at different depths can confuse them.
- Matching a phone to a student is a best-effort nearest-shoulder guess, not verified possession.
- Calculators can be flagged as phones, so use the Phone rule toggle.
- The ID-badge filter can ignore a white-cased phone in the chest area, even when it is held, and the hands-at-chest fallback can flag someone who simply keeps their hands together at chest height for 3 seconds.
- Small objects in the back rows of a large room are only a few pixels wide and will be missed.
- Only one live session runs at a time, shared by the whole app.
- Video lag on weaker machines. Run on a GPU or use fewer cameras for large rooms.