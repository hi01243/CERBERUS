# Classroom Examination Monitoring System

An AI-powered classroom examination monitoring system that uses computer vision and deep learning to detect potential examination violations in real time.

## Features

### Real-Time Monitoring

* Live camera feed monitoring
* Start and stop monitoring controls
* Real-time status dashboard

### Prohibited Object Detection

Detects:

* Cell Phones
* Smartwatches

### Student Behavior Analysis

Detects:

* Prolonged side gazes
* Student-to-student interaction
* Excessive leaning toward nearby students

### Evidence Collection

* Automatic screenshot capture of violations
* Timestamped evidence storage
* Violation logging to SQLite database

### Tracking System

* Persistent student tracking using ByteTrack
* Unique student IDs during monitoring
* Violation association with tracked individuals

---

## Technologies Used

### Backend

* Python
* Flask
* SQLite

### Computer Vision & AI

* OpenCV
* Ultralytics YOLO
* YOLO11 Pose Estimation
* ByteTrack
* NumPy

---

## System Architecture

### Detection Pipeline

1. Camera captures classroom video.
2. YOLO Object Detection identifies prohibited devices.
3. YOLO Pose Estimation tracks student posture and head orientation.
4. Violation rules evaluate:

   * Side gaze duration
   * Student interaction
   * Prohibited objects
5. Evidence image is captured.
6. Violation is logged into the database.
7. Dashboard updates in real time.

---

## Detected Violations

### Object Violations

| Violation  | Description                            |
| ---------- | -------------------------------------- |
| Cell Phone | Phone detected during examination      |
| Smartwatch | Smartwatch detected during examination |

### Behavior Violations

| Violation           | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| Prolonged Side Gaze | Student continuously looks left or right beyond threshold      |
| Student Interaction | Students are detected interacting or leaning toward each other |

---

## Project Structure

```text
project/
│
├── app.py
├── database.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── violations/
│
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/exam-monitoring-system.git
cd exam-monitoring-system
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Required Models

Download or place the following models in the project directory:

```text
yolo11n.pt
yolo11n-pose.pt
```

---

## Run the System

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## API Routes

### Home Page

```http
GET /
```

### Video Feed

```http
GET /video_feed
```

### Start Monitoring

```http
POST /start_monitoring
```

### Stop Monitoring

```http
POST /stop_monitoring
```

### Retrieve Violations

```http
GET /get_violations
```

### System Statistics

```http
GET /system_stats
```

---

## Current Limitations

* Requires clear classroom visibility.
* Accuracy depends on camera placement and resolution.
* Smartwatch detection uses the COCO "clock" class as a proxy.
* Student IDs may change after long occlusions.
* Thresholds require calibration for each classroom setup.

---

## Future Improvements

* Multi-camera support
* Student seat mapping
* Face recognition integration
* Exam attendance monitoring
* Advanced cheating behavior analysis
* Cloud-based monitoring dashboard

---

## Author/s

BS Computer Science Student
Technological University of the Philippines – Manila

---

## Disclaimer

This project is intended for educational and academic research purposes. It serves as a proof-of-concept classroom examination monitoring system and should be validated extensively before deployment in real examination environments.
