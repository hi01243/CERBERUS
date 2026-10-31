"""Configuration constants, thresholds, and runtime options for Cerberus."""
import os

DEMO_MODE = os.environ.get("CERBERUS_DEMO_MODE") == "1"
if DEMO_MODE:
    _CUDA, _GPU_NAME = False, ""
else:
    try:
        import torch
        _CUDA = bool(torch.cuda.is_available())
        _GPU_NAME = torch.cuda.get_device_name(0) if _CUDA else ""
    except Exception:
        _CUDA, _GPU_NAME = False, ""
DEVICE = 0 if _CUDA else "cpu"   # NVIDIA GPU when CUDA works, otherwise CPU
USE_HALF = False                 # fp16 only makes sense on a CUDA GPU


def _load_yolo(model_path):
    if DEMO_MODE:
        raise RuntimeError("Camera detection is disabled in Render free demo mode.")
    from ultralytics import YOLO
    return YOLO(model_path)


BUILD_ID = "2026-10-11-r47-phone-verify"
DATA_DIR = os.environ.get("CERBERUS_DATA_DIR")
if DATA_DIR:
    os.makedirs(DATA_DIR, exist_ok=True)
VIOLATION_FOLDER = (os.path.join(DATA_DIR, 'violations') if DATA_DIR
                    else os.path.join('static', 'violations'))
os.makedirs(VIOLATION_FOLDER, exist_ok=True)

# Head & gaze thresholds
HEAD_TURN_RATIO_THRESHOLD = 0.55  # horizontal nose-displacement ratio, not a raw angle
HEAD_TURN_EXIT_RATIO = 0.40  # once turned, stay turned until |yaw| falls below this
GAZE_DURATION_THRESHOLD = 1.5
ATTRIBUTION_RADIUS_MULTIPLIER = 4.0
MIN_KEYPOINT_CONF = 0.5
SHOULDER_MIN_CONF = 0.3   # shoulders only: lets partly hidden back-row students be tracked
SMOOTHING_WINDOW = 5
BASELINE_MAX_ABS = 0.45   # a 'neutral' pose further off-axis than this was a turned head

# Detection rules — toggled live via POST /settings.
DETECTION_ENABLED = {
    "phone":       True,
    "reach":       True,
    "gaze":        True,
}

# --- WRIST-CROP PHONE PASS ---
WRIST_CROP_ENABLED      = True
WRIST_CROP_SCALE        = 1.6    # crop width = this x the student's shoulder width
WRIST_CROP_MIN_HALF_PX  = 32     # never crop smaller than 64 px across
WRIST_CROP_IMGSZ        = 640    # crops are upscaled to this before inference
WRIST_CROP_CONF         = 0.78  # lower than the full-frame 0.45; persistence compensates
WRIST_CROP_INTERVAL_SEC = 0.33   # run ~3x per second, not every frame
WRIST_CROP_MAX_CROPS    = 40     # cap per pass (large rooms: raise on a GPU)
WRIST_CROP_MIN_HITS     = 2      # hits needed ...
WRIST_CROP_WINDOW_SEC   = 2.0    # ... inside this window
WRIST_CROP_HOLD_SEC     = 0.8    # keep a confirmed phone alive between passes
WRIST_CROP_DEDUP_PX     = 50     # same phone as a full-frame detection if this close
WRIST_CROP_MODEL        = "best_tuned.pt"  # bigger model for the zoomed-in pass only; "yolo11n.pt" = old/faster
PHONE_MODEL = WRIST_CROP_MODEL   # full-frame pass uses the same trained phone model -> only 2 models load
WRIST_CROP_CLASSES      = [0]   # class 0 = cell phone in the trained model

# --- DESK-CROP PHONE PASS (r49) ---
DESK_CROP_ENABLED        = True    # False = switch the whole feature off
DESK_REQUIRE_LOOKDOWN    = True    # only check students whose head is down
DESK_LOOKDOWN_MAX_RISE_SW = 0.5    # nose less than this x shoulder width above the shoulder line = head down
DESK_HALF_W_SW           = 1.1     # crop half-width, x shoulder width, around the shoulder centre
DESK_TOP_SW              = 0.8     # crop starts this far BELOW the shoulder line (x shoulder width)
DESK_BOTTOM_SW           = 2.2     # ... and ends this far below it
DESK_MAX_CROPS           = 20
DESK_IMGSZ               = 640
DESK_FLOOR_CONF          = 0.30    # model inference floor for desk crops
DESK_CROP_CONF           = 0.50    # a desk detection must reach this to count as a hit
DESK_MIN_HITS            = 2       # hits needed ...
DESK_WINDOW_SEC          = 2.0     # ... inside this window
DESK_HOLD_SEC            = 0.8     # keep a confirmed phone alive between passes
DESK_MAX_MEAN_GRAY       = 185     # brighter than this = white paper, not a phone
DESK_DARK_PIXEL          = 75      # gray value below this = "dark" pixel
DESK_MIN_DARK_FRAC       = 0.16    # a phone box needs at least this share of dark pixels
DESK_MIN_SHORT_SW        = 0.18    # desk phone box: short side at least this x shoulder width
DESK_MIN_LONG_SW         = 0.40    # ... and long side at least this x shoulder width
DESK_MAX_HAND_SW         = 0.7     # desk phone centre must be within this x shoulder width of a wrist

# --- PHONE FRAME SKIP (r46) ---
PHONE_FULL_EVERY_N = 2   # full-frame phone pass runs on every Nth frame (1 = every frame)

# --- STAGE-2 PHONE VERIFIER (r47, see phone_verify.py) ---
PHONE_VERIFY = "off"
PHONE_VERIFY_DUMP = True             # True: save candidate crops for calibration
PHONE_VERIFY_DUMP_DIR = "phone_candidates"
PHONE_FULL_CONF = 0.80         # full-frame phone pass: confidence needed

# --- HAND-RISE CONDITION (r48) ---
PHONE_NEEDS_HAND_RISE = False
HAND_LOW_SW           = 1.0    # wrist this far below the shoulder line (x shoulder width) = "low"
HAND_LOW_MIN_SEC      = 0.3    # must stay low / out of sight this long to count
HAND_RISE_MEMORY_SEC  = 10.0   # a phone may appear this long after the hand was low
HAND_RISE_KEEP_SEC    = 2.0    # an accepted phone stays allowed while it keeps being detected
PHONE_FULL_NEEDS_HAND = True  # full-frame phone only counts if a hand is near it

# --- WRIST-CROP REST SKIP (r46) ---
WRIST_CROP_REST_SKIP        = True   # skip a crop if the last check found nothing and the hand has not moved
WRIST_CROP_REST_MOVE_SW     = 0.25   # "has not moved" = less than this x shoulder width
WRIST_CROP_REST_RECHECK_SEC = 2.0    # ...but look again at least this often

# --- HANDS-TOGETHER PHONE CROP + WEAK-EVIDENCE PATH (r34) ---
WRIST_HANDS_CROP         = True   # extra zoomed crop centred between both hands
WRIST_HANDS_MAX_GAP_SW   = 1.3    # only when the wrists are this close (x shoulder width)
WRIST_HANDS_MIN_HALF_SW  = 0.60   # hands-crop half-size is at least this x shoulder width
WRIST_WEAK_CONF          = 0.99   # faint detections count only in the hands-at-chest posture
WRIST_WEAK_MIN_HITS      = 4
WRIST_WEAK_WINDOW_SEC    = 3.0
WRIST_WEAK_MAX_GAP_SW    = 0.9    # wrists no further apart than this ...
WRIST_WEAK_MIN_DROP_SW   = 0.15   # ... and both between this far below the shoulder line
WRIST_WEAK_MAX_DROP_SW   = 1.6    #     and this far (above the desk/lap)

# --- PEN FILTER (phones) ---
PHONE_MAX_ASPECT   = 3.5    # box longer than 3x its width is a pen, not a phone
PHONE_MIN_SHORT_SW = 0.08   # box short side must be >= this x shoulder width
PHONE_TINY_SW        = 0.18   # long side below this x shoulder width: always rejected
PHONE_SMALL_SW       = 0.32   # long side below this: needs high confidence
PHONE_SMALL_MIN_CONF = 0.60
PHONE_MAX_LONG_SW   = 0.80   # long side above this x shoulder width: too big to be a phone

# --- ID-BADGE / LANYARD FILTER (phones) ---
BADGE_FILTER_ENABLED = True
BADGE_ZONE_HALF_W    = 0.60   # strip half-width around the shoulder-centre x
BADGE_ZONE_TOP       = -0.10   # strip starts this far below the shoulder line
BADGE_ZONE_BOTTOM    = 1.70   # ...and ends this far below it (above the lap)
BADGE_HAND_HOLD_MULT = 0.30   # wrist this close to the box centre = held phone
BADGE_TWO_HAND_MULT  = 0.50   # BOTH wrists this close to the box = phone in two hands
BADGE_WHITE_V    = 165    # HSV value above this ...
BADGE_WHITE_S    = 70     # ... and saturation below this = "white" pixel
BADGE_WHITE_FRAC = 0.28   # box with at least this share of white pixels = ID card

# --- NEIGHBOR-AWARE SIDE GAZE ---
GAZE_REQUIRE_NEIGHBOR = True
GAZE_NEIGHBOR_MIN_DX  = 0.6   # neighbor must be at least this far to the side
GAZE_NEIGHBOR_MAX_DX  = 5.0   # ...and no farther than this
GAZE_NEIGHBOR_ROW_MULT = 2.0  # max vertical offset between shoulder centers
GAZE_NEIGHBOR_ROW_SLOPE = 0.30   # extra row tolerance per shoulder-width of distance
GAZE_NEIGHBOR_MEMORY_SEC = 8.0   # a neighbour seen on that side this recently still counts

# --- POSE MODEL INFERENCE SETTINGS ---
POSE_IOU  = 0.5
POSE_CONF = 0.25
POSE_IMGSZ = 960

# --- DUPLICATE-SKELETON (GHOST TRACK) FILTER ---
DUPLICATE_DIST_MULTIPLIER = 0.8

# --- ADAPTIVE "NORMAL POSITION" BASELINE ---
CALIBRATION_DURATION = 2.0          # seconds to sample a student's initial "neutral" pose
BASELINE_ADAPT_RATE = 0.02          # how fast the baseline creeps toward a newly-held position
REPOSITION_STABILITY_WINDOW = 10.0  # student must hold a new position this long before baseline shifts
REPOSITION_JUMP_LIMIT = HEAD_TURN_RATIO_THRESHOLD * 0.8
REPOSITION_SETTLE_BAND = HEAD_TURN_RATIO_THRESHOLD * 0.3

# --- STATIC-OBJECT FILTER (phones) ---
STATIC_OBJECT_RADIUS_PX   = 60
STATIC_OBJECT_MOVE_PX     = 35
WRIST_CROP_MAX_WRIST_DIST = 1.2   # phone centre must be within this x shoulder width of the wrist
WRIST_CROP_STATIC_SEC     = 60.0   # a wrist-crop 'phone' this still ...
WRIST_CROP_STATIC_MOVE_PX = 8     # ... (moves less than this) is furniture
HELD_PHONE_WRIST_MULT     = 0.6   # full-frame static filter: a hand this close keeps the phone flagged
STATIC_PHONE_SEC          = 30.0  # a full-frame 'phone' that never moves this long is furniture

# --- CAMERA-BUMP DETECTION ---
CAMERA_BUMP_RATIO        = 0.8
CAMERA_BUMP_MIN_STUDENTS = 6
CAMERA_BUMP_COOLDOWN_SEC = 5.0

# --- TRACK-AGE GATE ---
TRACK_MIN_AGE_SEC = 1.0

# --- NOTE-PASSING / REACHING DETECTION ---
REACH_ZONE_RADIUS_MULTIPLIER    = 1.5
REACH_ZONE_MIN_EXTENSION = 1.0   # wrist must be this many shoulder-widths from its own shoulders
REACH_EXTENSION_MULTIPLIER      = 1.6
REACH_PROGRESS_MIN              = 0.35
REACH_SAME_ROW_MULTIPLIER       = 1.0    # max vertical offset between shoulder centers, x avg shoulder width
REACH_MAX_PAIR_DIST_MULTIPLIER  = 8.0    # generous: the extension test does the real filtering
REACH_DURATION_THRESHOLD        = 1.0
REACH_GRACE_SEC                 = 0.4    # tolerate a dropped wrist keypoint for this long
REACH_WRIST_CONF                = 0.3    # accept lower conf; wrists are blurry mid-motion
REACH_DEPTH_RATIO_MIN = 0.75  # shoulder-width ratio band — outside this means different rows
REACH_DEPTH_RATIO_MAX = 1.33
REACH_MIN_SIDE_GAP_MULTIPLIER = 1.2  # students must be side-by-side, not front-to-back
REACH_MID_MIN_PROGRESS  = 0.30   # fraction of the way from own shoulders to the neighbour's
REACH_MID_MAX_PROGRESS  = 0.70
REACH_MID_MIN_EXTENSION = 0.80   # wrist at least this many shoulder-widths from own shoulders
REACH_MID_MAX_DY        = 2.00   # ...and no more than this far above/below the shoulder line

# --- HEAD-POSE ROBUSTNESS ---
PITCH_DOWN_DELTA = 0.30
HEAD_TURN_STRONG_RATIO = 0.7

# --- GAZE LEAKY-BUCKET & HYSTERESIS ---
GAZE_DRAIN_RATE = 0.5          # seconds of countdown lost per second spent NOT gazing
GAZE_DOWN_HYSTERESIS  = 2.5    # while a gaze timer runs, head-drop needed for 'down' is this x PITCH_DOWN_DELTA
GAZE_REBASELINE_SEC = 30.0

# --- HANDS-TOGETHER POSTURE ---
HANDS_POSTURE_ENABLED  = False
HANDS_POSTURE_SEC      = 3.0
HANDS_POSTURE_GRACE_SEC = 1.5
HANDS_POSTURE_MAX_GAP_SW = 0.75
HANDS_POSTURE_MIN_DROP_SW = 0.15
HANDS_POSTURE_MAX_DROP_SW = 0.75

# --- BADGE LOCK MOTION FILTER ---
BADGE_LOCK_WINDOW_SEC = 3.0    # history kept per student
BADGE_LOCK_MIN_SEC    = 1.2    # need this much history before deciding ("CHECKING" until then)
BADGE_LOCK_STD        = 0.05   # offset-from-shoulders std below this (x shoulder width) = steady
BADGE_LOCK_MIN_REL    = 0.10   # ... and offset-from-wrist std must exceed this ...
BADGE_LOCK_RATIO      = 3.0    # ... and be this many times the shoulder-offset std

# --- WRIST CROP TUNING ---
WRIST_CROP_FLOOR_CONF = 0.50
WRIST_CROP_AUGMENT    = False

TRACK_LOST_GRACE_SEC = 1.5
DEBUG_OVERLAY = True
VIDEO_CLOCK = True
RECORD_RAW = False

# --- LIVE-CAMERA CAPTURE & DISPLAY ---
THREADED_CAPTURE    = True
SMOOTH_DISPLAY      = True
OVERLAY_MAX_AGE_SEC = 3.0

PROHIBITED_CLASSES = {0: "CELL PHONE"}
RESET_CONFIRM_PHRASE = "DELETE ALL DATA"

# Endpoint categories for access control
PUBLIC_ENDPOINTS = {"login", "signup", "recover_account"}

LIVE_ENDPOINTS = {
    "monitor", "video_feed", "get_session_route", "settings_route", "system_stats",
    "get_violations", "start_monitoring", "stop_monitoring", "reset_session",
    "list_cameras", "scan_cameras_route", "add_camera_route",
    "add_browser_camera_route", "browser_camera_frame_route",
    "activate_camera_route", "pause_camera_route", "remove_camera_route",
    "recalibrate_route",
}

SESSION_QUERY_ENDPOINTS = {"review", "incidents_route", "session_files_route"}

