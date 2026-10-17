<p align="center">
  <img src="static/logo-full.svg" alt="CERBERUS" width="300">
</p>

AI-assisted exam proctoring dashboard. CERBERUS watches one or more camera feeds, uses YOLO11 pose and a custom-trained phone detector to flag suspicious behaviour, records every session, and gives a reviewer a timeline to check each flag against the recording.

It flags three things:

| Rule | What it catches |
|------|-----------------|
| **Phone** | A cell phone in a student's hand or on the desk (also flags some calculators) |
| **Reach** | A hand-off between neighbouring students, such as a passed note |
| **Gaze** | A prolonged sideways head turn toward a neighbouring student |

Internally, flags are logged under these types:

| Flag type | Rule | Confidence |
|-----------|------|------------|
| `OBJECT_CELL_PHONE` | Phone | The detector's own score |
| `REACHING_NEIGHBOR_ZONE` | Reach | Fixed 0.85 |
| `PROLONGED_SIDE_GAZE` | Gaze | Fixed 0.95 |

> **CERBERUS flags, it does not judge.** Every flag is a prompt for a human reviewer, who can dismiss false positives without deleting the evidence. Only use it where students have been told they are being recorded and your institution's policies allow it.

---

## Features

- **Live monitoring dashboard** with a camera grid, live counters, an audit log, and an evidence gallery you can filter by flag type
- **Multiple cameras**: local webcams, IP/RTSP streams (for example a phone), and video files that loop for demos. Cameras are added from the dashboard without a restart, and each one can be started, paused, or removed on its own. A scan option probes the first few local device indices for webcams that are not registered yet
- **Smooth live video**: live cameras are read by a background thread, so the picture runs at camera speed and always shows the newest frame. Detection boxes may lag slightly behind the video on slower machines, and are hidden if detection stalls for more than 3 seconds
- **Per-exam sessions**: each exam gets its own recordings and incidents, so two exams on the same day never mix
- **Per-account session ownership**: every session belongs to the account that created it, and other accounts cannot see it
- **Recording in 5-minute files**: each camera saves to `recordings/<session>/<camera>/` at 10 fps and at most 960 px wide, and evidence images go to `static/violations/<session>/<camera>/`. A crash costs at most one file
- **Incident grouping**: repeats of the same flag on the same student within 10 seconds are one incident, and students flagged at the same moment each get their own
- **Review page**: play an incident from 5 seconds before the alert, jump around on an incident timeline, pick between recordings, filter active or dismissed flags, and dismiss false positives
- **Multi-camera grouping**: the same violation seen by several cameras within 5 seconds shows as one card with a camera picker (display only; nothing is merged or deleted)
- **Live rule toggles**: switch phone, reach, or gaze detection on or off per exam, for example when calculators are allowed. Turning Phone off also turns off the wrist-crop and desk-crop checks. Changes are written to the console log with the username
- **Recalibrate button**: restarts every student's neutral-pose calibration without cutting the recording, for when students change seats
- **Accounts and MFA**: username and password sign-up, optional authenticator-app (TOTP) MFA, one-time recovery codes for password/account recovery, and MFA-gated account deletion
- **Crash recovery**: recordings left half-written by a crash or power loss are finalised on the next start (a file cut off mid-write may still not play)

---

## Requirements

- Python 3.9+
- A webcam, IP camera, or video file to test with
- **The phone model `best_tuned.pt`**, placed next to `app.py`. This is the custom-trained detector used for every phone check (class 0 = cell phone). CERBERUS cannot start a camera without it
- The pose weights `yolo11n-pose.pt`. Ultralytics downloads them on first run unless you place them next to `app.py`, so the first run needs internet access
- **Recommended:** `ffmpeg` on your PATH. Without it, some recordings may not play in the browser
- **A computer with a good NVIDIA GPU and a CUDA build of PyTorch is needed to run CERBERUS smoothly.** CERBERUS uses the GPU automatically when PyTorch can see it. It will run on a CPU, but slowly (see [Hardware](#hardware-a-good-gpu-is-needed-for-smooth-use) and [Performance](#performance))

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

<<<<<<< HEAD
Then open <http://127.0.0.1:5000>. The console prints the build ID on startup. Create your first account at `/signup`. On a phone, `localhost` means the phone itself; use it only when the app server is running on that phone. If the server is on a computer, open it from the phone over HTTPS (for example, through an HTTPS tunnel) to allow browser camera access.

### Deploying on Render (free demo)

The included [`render.yaml`](./render.yaml) deploys a **free demo**. To enable camera processing, use the full `requirements-render.txt` build command and set `CERBERUS_DEMO_MODE` to `0`. The monitor page then offers **Use this device's camera**, which asks the browser for camera permission and sends small frames to the server. Open the Render HTTPS site on the phone whose camera you want to use, and keep that monitor page open.

1. Push the project to a GitHub repository that Render can access. Make sure the repository includes `render.yaml`, `requirements-render-free.txt`, and the app files.
2. Sign in to Render, click **New + → Blueprint**, connect GitHub if prompted, and select your repository.
3. Review the Blueprint. It should show a web service named `cerberus-demo` with the **Free** plan. Click **Apply** to create it.
4. Open the service's **Events** or **Logs** page and wait for the build and deploy to complete. On first build, installing OpenCV may take a few minutes.
5. Open the service's `https://...onrender.com` URL. Create a test account at `/signup` and save the recovery codes if you want to test recovery.
6. Expect the first page load after inactivity to take about a minute while Render wakes the service. Free services can sleep after 15 minutes without traffic.

**Important free-tier limits:** Render's free filesystem is ephemeral. The app's users, password hashes, recovery codes, and other database records are in local SQLite files under `/tmp/cerberus-data`; they are lost when the service sleeps, restarts, or redeploys. Recordings and evidence are also temporary. Don't use real account details or depend on this instance to retain data. The phone camera works only through browser permission and HTTPS; local-camera scanning still looks for cameras attached to the Render server. Frame processing and model startup may be very slow or exceed free-instance memory. Do not use this deployment for a real exam.

For current free-service behavior, see Render's [Free services documentation](https://render.com/docs/free). To enable live detection and durable records later, switch to a paid instance with persistent storage and use the full `requirements-render.txt` dependencies.
=======
Then open <http://127.0.0.1:5000>. The console prints the build ID on startup, and whether inference runs on the GPU or the CPU. Create your first account at `/signup`.
>>>>>>> dc99ada45b1ce22b77b88dae9f03033bf52c7ae0

### Configuration

| Environment variable | Purpose | Default |
|----------------------|---------|---------|
| `CERBERUS_SECRET_KEY` | Flask session key. Set it to stay signed in across restarts | Random key each run (everyone is signed out on restart) |
| `CERBERUS_HOST` | Interface to listen on. Use `0.0.0.0` to reach the dashboard from other devices | `127.0.0.1` |
| `CERBERUS_DATA_DIR` | Directory for SQLite databases, recordings, and evidence images | Project-specific locations |

The built-in development server listens on port 5000.

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
| `PHONE_MODEL`, `WRIST_CROP_MODEL` | Phone model weights (default `best_tuned.pt`, shared by all phone passes) |
| `PHONE_FULL_CONF`, `PHONE_FULL_EVERY_N` | Full-frame phone confidence (default 0.80) and how often that pass runs (default every 2nd frame) |
| `PHONE_FULL_NEEDS_HAND` | Full-frame phone only counts if a wrist is near it |
| `WRIST_CROP_ENABLED`, `WRIST_CROP_CONF`, `WRIST_CROP_MAX_CROPS` | The zoomed-in wrist phone check, its confidence (default 0.78), and the cap on crops per pass (default 40; raise it for large rooms on a GPU) |
| `DESK_CROP_ENABLED`, `DESK_CROP_CONF` | The zoomed-in desk/lap phone check for students looking down |
| `PHONE_NEEDS_HAND_RISE` | Require the hand to have been low or out of sight before a phone counts (default off) |
| `BADGE_FILTER_ENABLED`, `BADGE_WHITE_FRAC` | ID-badge filter and how white a box must be to count as a card |
| `BASELINE_MAX_ABS` | Largest neutral head offset a baseline may have (see Gaze) |
| `THREADED_CAPTURE`, `SMOOTH_DISPLAY` | Background frame reader and camera-speed live video for live cameras |
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

1. **Sign up** at `/signup` with a username (3-32 characters: letters, numbers, `_ . -`) and a password (8+ characters). Save the one-time recovery codes shown after signup. You can enable authenticator MFA at the same time, or later at `/security/mfa`.
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

### Testing with a phone camera through ngrok

Run CERBERUS on the computer that has the detection models installed, then expose its local port with an HTTPS ngrok URL. Open that HTTPS URL on the phone, sign in, go to **Monitor**, click **Use this device's camera**, then click **Use** on its camera tile and allow browser camera access. Start monitoring to run detection. The tile shows the processed server stream with the overlays; the phone sends video frames only (no audio). Keep the phone's monitor page open during the test. A plain `http://localhost:5000` works for camera permission only when CERBERUS is running on that same phone; on the phone, `localhost` does not refer to your computer. Use this for testing only and make sure everyone being recorded has agreed.

### Accounts and session ownership

- Another account that asks for your session, recording, incident, or evidence image gets a 404.
- There is one live session for the whole app. If the live session belongs to another account, the live controls refuse your requests (403) and `/monitor` sends you back to the Sessions page until you create your own session.
- Another account can take over the live controls by creating its own session, which is only possible while monitoring is paused.
- Session names are unique across accounts.
- When upgrading from a build without ownership, existing sessions go to the first account created.
- Turning MFA off (at `/security/mfa`) requires a current authenticator code.
- MFA is standard TOTP (6 digits, 30-second steps, SHA-1), so any authenticator app works. The previous and next code are also accepted to allow for clock drift. It is built into CERBERUS and needs no extra package.

### Deleting data

Deletion is deliberately hard to do by accident:

- **Delete a session** (Sessions page): monitoring must be paused if it is the active session, and you must type the exact session name.
- **Reset session** (Monitor page): monitoring must be paused, and you must type `DELETE ALL DATA`. A fresh session, owned by you, is created afterwards.
- **Delete an account**: stop monitoring if the live session belongs to you, enter your password, type your exact username and `DELETE ACCOUNT`, and provide an authenticator code if MFA is enabled. This permanently removes all sessions and evidence owned by that account.

Deleting or resetting a session removes its database rows, recordings, and evidence images. Deleting an account removes all of its owned sessions and evidence. These actions cannot be undone.

> **Save your recovery codes somewhere private.** Each code can be used once to reset your password or sign in when MFA is enabled. You can generate a new set while signed in; doing so invalidates the previous set.

---

## How detection works

### Models used

| Model | Used for | Settings |
|-------|----------|----------|
| `yolo11n-pose.pt` (YOLO11 nano pose) | Finds each student's body keypoints (nose, shoulders, wrists) for the Gaze and Reach rules and to locate wrists and desks for the phone checks. Run with ByteTrack (`bytetrack.yaml`) to give each student a stable ID | Input 960 px (`POSE_IMGSZ`), confidence 0.25, IoU 0.5 |
| `best_tuned.pt` (custom-trained phone detector) | All phone checks: the full-frame pass, the wrist crops, and the desk crops. Class 0 is "cell phone" and is the only class requested | Full frame: 640 px, counts above 0.80. Wrist crops: 640 px, count at 0.78 or more. Desk crops: counts at 0.50 or more |

The pose model is a pretrained Ultralytics YOLO11 model trained on COCO. The phone model is a custom-trained model, and its accuracy depends on how close your classroom, camera, and phones are to its training data. Everything after the model output (head turn, hand-off, badge and pen filters, timers) is hand-written geometry and rules. Because the full-frame and crop passes share the same weights, each active camera loads two models (pose and phone), and each camera loads its own copies. ByteTrack needs the `lapx` package, which is in the install line.

### Hardware: a good GPU is needed for smooth use

Every frame goes through the pose model, the phone model runs on every second frame, and the wrist and desk crops re-check the phone model about 3 times a second. That is a lot of work for a CPU. **Use a computer with a good NVIDIA GPU and a CUDA build of PyTorch.**

- **With a suitable GPU**, the live feed and flags can keep up with real time.
- **Without one** (CPU or integrated graphics), expect a low frame rate, and timing-based rules such as gaze and reach react late because they see fewer frames. The live video itself stays smooth on live cameras because it is shown separately from detection, but boxes lag behind it. Video-file cameras play in slow motion, though the flags stay the same because of `VIDEO_CLOCK`.
- The load grows with the number of students (more crops) and the number of active cameras (each runs its own detection thread and model copies on the same GPU). A full room or several cameras needs a stronger GPU.
- Check that PyTorch sees your GPU with `python -c "import torch; print(torch.cuda.is_available())"`, which should print `True`.

If your machine is too slow, see [Performance](#performance) for the settings that trade accuracy for speed.

### Pipeline

Students are tracked with the pose model and ByteTrack. A student needs both shoulders visible (confidence 0.3 or more) to be tracked at all, and a track must be at least 1 second old before it can be blamed for a phone or a reach. Per-student state survives tracking drops of up to 1.5 seconds. Duplicate ("ghost") skeletons are filtered out.

### Phone

There are three detection passes, followed by a set of filters.

**Passes**

- **Full-frame check:** the phone model scans the whole frame on every 2nd frame (`PHONE_FULL_EVERY_N`; the previous result is reused in between). A detection must score above 0.80, and boxes as large as a person (over 35% of the frame in width or height) are ignored. A full-frame phone only counts if a student's wrist is close to it (within 0.6 shoulder widths).
- **Wrist-crop check:** about 3 times a second, CERBERUS crops a zoomed window around each wrist, plus an extra one centred between both hands when they are close together, and re-checks them with the phone model. A phone is confirmed after 2 hits at 0.78 confidence or more within 2 seconds, stays alive for 0.8 seconds between passes, and must be within 1.2 shoulder widths of the wrist. To save time, a wrist whose last check found nothing and which has not moved is skipped (it is re-checked at least every 2 seconds).
- **Desk-crop check:** for students whose head is down (nose close to the shoulder line, or not visible), CERBERUS also crops the desk and lap area below the shoulders and checks it with the phone model. This is aimed at phones held low that the wrist crops miss. A desk detection must score 0.50 or more, be close to a wrist, be large enough to be a phone, and contain enough dark pixels (white paper and bare skin are rejected). It is confirmed after 2 hits within 2 seconds.

**Filters**

- **Static-object filter:** drops "phones" that never move (seat-number plates, furniture): 30 seconds for full-frame detections unless a hand is near, and 60 seconds for wrist-crop detections.
- **Size filter:** ignores boxes that are long and thin (more than 3.5 times longer than wide, typically a pen), too small, or too big compared with the student's shoulder width. Boxes with a long side under about a fifth of a shoulder width are always dropped, and boxes under about a third of a shoulder width need at least 0.60 confidence.
- **ID-badge filter:** a card hanging on a lanyard sits on the chest midline, so a "phone" in that strip is ignored when any of these is true:
  - it is mostly white (an ID card), even if a hand is on it;
  - a dark strap runs from it up to the neck (a lanyard), even if a hand is resting on it;
  - it keeps a steady position relative to the shoulders while the hands move around it.

  A non-white object held in both hands is still flagged. A candidate in the chest area shows "CHECKING..." for about 1.2 seconds while the motion test collects data. These filters apply to the full-frame, wrist-crop, and desk-crop detections.

**Optional extra (off by default)**

- **Hand-rise condition (`PHONE_NEEDS_HAND_RISE`):** a phone only counts if that student's hand was recently low or out of sight (pocket, lap, bag), i.e. the phone "came from somewhere".

### Gaze
- Head turn is estimated from how far the nose sits sideways from the shoulder line, relative to shoulder width.
- Each student gets a personal baseline from a 2-second calibration, and it adapts slowly if they settle into a new position for about 10 seconds. A baseline further off-axis than 0.45 is clamped, because it means the head was already turned during calibration.
- A turn must pass 0.55 to start, and once turned it stays turned until it falls below 0.40. It must last 1.5 seconds to be flagged. Brief interruptions (a blink, a noisy frame) do not reset the timer; it drains at half speed instead. A lost nose keypoint keeps the previous side turn alive for up to 0.8 seconds.
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

> All thresholds are hand-calibrated, not learned. Test with people seated at real exam spacing and tune the constants at the top of `app.py` before relying on it. Setting `DEBUG_OVERLAY = True` draws yaw and pitch numbers beside each student, the wrist keypoints, the wrist-crop boxes with their scores (magenta), candidates that were rejected and why (orange: pen-like, ID badge, far from hand, still checking, and "DESK: ..." reasons for desk crops; grey: ignored detections such as "NOT IN HAND" or "STATIC OBJECT"), and a grey box with the shoulder confidences around anyone the model found but could not track.

---

## Performance

- The heaviest parts are the wrist-crop and desk-crop phone checks. They run about 3 times a second, so their cost grows with the number of students (up to 40 wrist crops and 20 desk crops per pass).
- Several things already reduce the load: the full-frame phone pass runs on every 2nd frame, wrist crops are skipped for a hand that has not moved and held nothing, and desk crops only run for students who are looking down.
- On a CPU or integrated GPU, expect slow detection and slow video-file playback, especially in a full classroom. On an NVIDIA GPU with a CUDA build of PyTorch, the same checks take a fraction of the time.
- Each active camera runs its own detection thread and model copies, so several cameras on one machine share the same compute.
- Lowering `POSE_IMGSZ` from the default 960 to 640 gives more frame rate but weakens detection of small back-row students. Raising `PHONE_FULL_EVERY_N`, lowering `WRIST_CROP_MAX_CROPS`, or turning off `DESK_CROP_ENABLED` also trades accuracy for speed, so re-test with your own clips after any of these changes.

---

## Known limitations

- Subtle phone use is still easy to miss. The desk-crop check helps with phones held low, but a phone under a desk or in a lap may still not be seen.
- Gaze only covers a student looking at a neighbour to the side. Looking at the student in front or behind is not flagged.
- Reach and gaze use 2D geometry from one camera angle, so overlapping students at different depths can confuse them.
- Matching a phone to a student is a best-effort nearest-shoulder guess, not verified possession.
- Calculators can be flagged as phones, so use the Phone rule toggle.
- The ID-badge filter can ignore a white-cased phone in the chest area, even when it is held.
- The phone model only knows what it was trained on. Phone types, desks, lighting, or camera angles very different from its training data can cause misses or false flags.
- Small objects in the back rows of a large room are only a few pixels wide and will be missed.
- Detection lags on weaker machines (boxes trail the live video). Run on a GPU or use fewer cameras for large rooms.