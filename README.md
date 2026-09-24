```markdown
# CERBERUS: Real-Time Vision-Based Examination Proctoring System

CERBERUS (**C**lassroom **E**xamination **R**isk & **B**ehavioral **E**valuation via **R**eal-time **V**ision **S**ystems) is an automated proctoring pipeline engineered for physical examination environments[cite: 1]. It utilizes dual-engine computer vision to monitor behavioral anomalies—such as prolonged side gazing, unauthorized collaboration, and prohibited devices—while recording audit-logged evidence to a local SQLite database[cite: 1].

---

## System Architecture

```text
       [ Physical Surveillance Feed ] (Phone IP Camera / RTSP / USB)
                                    │
                                    ▼
                     [ OpenCV Ingestion Engine ]
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
   [ YOLO11 Object Detection ]                [ YOLO11-Pose Estimation ]
   (Blacklisted Items: Phones, Watches)       (17 Skeletal Joint Vectors)
              │                                           │
              └─────────────────────┬─────────────────────┘
                                    │
                        [ Behavioral Telemetry ]
                 • Angle-Invariant Head Yaw Analysis
                 • Perspective-Scaled Desk Interaction
                 • Per-Student Temporal Debounce Timers
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                           ▼
      [ SQLite Database ]                         [ Flask Web Server ]
      (cerberus.db Session Logs)                  (Command Center UI)
              │                                           │
              ▼                                           ▼
   [ Static Proof Snapshots ]                 [ Live Stream & Analytics ]

```

---

## Core Features & Rules

* **Angle-Invariant Head Turn (Gaze Tracking):** Evaluates rotational displacement between the head vector ($K_0$ nose) and the chest/shoulder vector ($K_5, K_6$ shoulders) rather than relying on flat 2D camera coordinates, avoiding false triggers from off-center camera angles.
* **Temporal Gaze Debouncing:** Quick natural glances do not trigger violations. The system tracks student positions and uses a countdown timer ($2.0\text{s}$ threshold) to log only persistent, deliberate turns.


* **Dynamic Student Interaction & Leaning:** Flags collaboration by tracking proximity between shoulder centers normalized against anatomical shoulder width, alongside torso tilt angles (neck-to-hip vectors) to detect leaning across desks.


* **Prohibited Object Spotting:** Uses YOLO object detection with confidence thresholds to identify cell phones and smartwatch proxies in the testing zone.


* **Audit-Trail Logging:** Captures evidence snapshots throttled by a 3.0-second cooldown. Every infraction is written to `cerberus.db` alongside timestamps, confidence scores, and session folders.



---

## Project Structure

```text
cerberus/
├── app.py                  # Core Flask server & computer vision processing loop
├── database.py             # SQLite persistence layer & schema management
├── requirements.txt        # Runtime dependencies
├── yolo11n.pt              # YOLO object detection weights
├── yolo11n-pose.pt         # YOLO skeletal pose estimation weights
├── static/
│   ├── style.css           # Dark-themed dashboard styling
│   ├── script.js           # Telemetry polling, lightbox, and event listeners
│   └── violations/         # Structured proof image storage
│       └── <SESSION_ID>/   # Partitioned session snapshots
└── templates/
    └── index.html          # Surveillance HUD and evidence gallery

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/cerberus.git](https://github.com/your-username/cerberus.git)
cd cerberus

```

### 2. Configure Virtual Environment

```bash
python -m venv .venv

# Windows (Command Prompt / PowerShell)
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install flask opencv-python ultralytics numpy

```

*(Note: On first execution, the Ultralytics framework will automatically download `yolo11n.pt` and `yolo11n-pose.pt` weights to the root directory).*

---

## Configuration

Camera and threshold settings are centralized in `app.py`:

```python
# Video Source: Use 0 for built-in camera, or replace with phone stream URL:
camera = cv2.VideoCapture(0)
# Example: camera = cv2.VideoCapture("[http://192.168.1.15:8080/video](http://192.168.1.15:8080/video)")

CURRENT_SESSION = "SESSION_ROOM_A1"  # Target examination partition
CAPTURE_COOLDOWN = 3.0               # Minimum seconds between snapshots
GAZE_DURATION_THRESHOLD = 2.0        # Seconds of sustained turn before flagging

```

---

## Usage

### 1. Launch the Server

```bash
python app.py

```

### 2. Open the Dashboard

Navigate in your browser to:

```text
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

```

### 3. Controls

* Click **ARM MONITORING** to begin inference, skeletal vector tracking, and violation logging.
* Click **PAUSE SYSTEM** to halt model inference and conserve compute resources.
* Review evidence in real time through the **Audit Log** and inspect captured snapshots via the interactive **Evidence Repository** gallery.

```

```