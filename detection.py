"""Core detection loop and real-time proctoring inference thread."""
import os
import time
import math
from collections import deque
import cv2
import numpy as np

import state
import phone_verify as _pv
from database import log_violation
from camera_manager import (
    _open_capture, make_status_frame, _finish_recording,
    _clear_tracking_state, PER_TRACK_KEYS
)
from heuristics import (
    _select_unique_people, _neighbor_in_direction, _update_hand_rise,
    _hand_rise_ok, _merge_wrist_phones, _check_static_objects,
    _pen_like, _local_sw, _phone_in_hand, _badge_like,
    _owner_of_detection, _torso_lock_verdict, _hands_posture,
    _find_reaching_wrist
)
from config import (
    DEVICE, USE_HALF, DETECTION_ENABLED, PROHIBITED_CLASSES,
    POSE_IOU, POSE_CONF, POSE_IMGSZ, SHOULDER_MIN_CONF, MIN_KEYPOINT_CONF,
    REACH_WRIST_CONF, CALIBRATION_DURATION, BASELINE_MAX_ABS,
    REPOSITION_SETTLE_BAND, SMOOTHING_WINDOW, PITCH_DOWN_DELTA,
    GAZE_DOWN_HYSTERESIS, HEAD_TURN_EXIT_RATIO, HEAD_TURN_RATIO_THRESHOLD,
    HEAD_TURN_STRONG_RATIO, GAZE_REBASELINE_SEC, REPOSITION_JUMP_LIMIT,
    REPOSITION_STABILITY_WINDOW, BASELINE_ADAPT_RATE, GAZE_DURATION_THRESHOLD,
    GAZE_DRAIN_RATE, TRACK_LOST_GRACE_SEC, TRACK_MIN_AGE_SEC,
    PHONE_NEEDS_HAND_RISE, PHONE_FULL_EVERY_N, PHONE_FULL_CONF,
    PHONE_FULL_NEEDS_HAND, PHONE_VERIFY, PHONE_VERIFY_DUMP, PHONE_VERIFY_DUMP_DIR,
    CAMERA_BUMP_MIN_STUDENTS, CAMERA_BUMP_RATIO, CAMERA_BUMP_COOLDOWN_SEC,
    STATIC_PHONE_SEC, ATTRIBUTION_RADIUS_MULTIPLIER, HANDS_POSTURE_ENABLED,
    HANDS_POSTURE_SEC, HANDS_POSTURE_GRACE_SEC, DUPLICATE_DIST_MULTIPLIER,
    REACH_MAX_PAIR_DIST_MULTIPLIER, REACH_SAME_ROW_MULTIPLIER,
    REACH_DEPTH_RATIO_MIN, REACH_DEPTH_RATIO_MAX, REACH_MIN_SIDE_GAP_MULTIPLIER,
    REACH_DURATION_THRESHOLD, REACH_GRACE_SEC, RECORD_RAW, VIOLATION_FOLDER,
    DEBUG_OVERLAY, VIDEO_CLOCK
)


def _detection_loop(camera_id):
    """Per-camera daemon thread. Owns frame reading, YOLO inference, HUD
    drawing, session recording, and incident logging."""
    entry = state.camera_registry.get(camera_id)
    if entry is None:
        return

    stop_event = entry["stop_event"]

    try:
        while not stop_event.is_set():

            # ==========================================
            # 0. READ FRAME
            # ==========================================
            with entry["lock"]:
                cap = entry["cap"]
                if cap is None:
                    time.sleep(0.05)
                    continue

                if entry["type"] != "file":
                    for _ in range(4):
                        if stop_event.is_set():
                            break
                        cap.grab()

                success, frame = cap.read()

            if not success:
                if entry["type"] == "file":
                    with entry["lock"]:
                        cap = entry["cap"]
                        if cap is not None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.1)
                    continue
                elif entry["type"] == "browser":
                    time.sleep(0.03)
                    continue
                else:
                    with entry["lock"]:
                        if entry["cap"] is not None:
                            entry["cap"].release()
                        new_cap = _open_capture(entry["source"], entry["type"])
                        entry["cap"] = new_cap
                        entry["_overlay"] = None
                    placeholder = make_status_frame(
                        entry["label"], "DISCONNECTED", "Attempting to reconnect...")
                    ret, buf = cv2.imencode('.jpg', placeholder)
                    if ret:
                        entry["latest_frame"] = buf.tobytes()
                    time.sleep(1.0)
                    continue

            h, w, _ = frame.shape
            clean_frame = frame.copy()
            current_time = time.time()
            if entry["type"] == "file" and VIDEO_CLOCK:
                try:
                    _vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    _vt = cap.get(cv2.CAP_PROP_POS_FRAMES) / _vfps
                    if _vt + 1.0 < entry.get("_vt_last", 0.0):
                        entry["_vt_off"] = (entry.get("_vt_off", 0.0)
                                            + entry.get("_vt_last", 0.0) + 1.0)
                    entry["_vt_last"] = _vt
                    current_time = 1000.0 + entry.get("_vt_off", 0.0) + _vt
                except Exception:
                    pass

            gaze_timers      = entry["gaze_timers"]
            turn_history     = entry["turn_history"]
            reach_timers     = entry["reach_timers"]
            baseline_ratio   = entry["baseline_ratio"]
            calib_start      = entry["calibration_start"]
            stable_since     = entry["stable_since"]
            stable_anchor    = entry["stable_anchor"]
            track_first_seen = entry["track_first_seen"]
            track_last_seen  = entry["track_last_seen"]
            calib_samples    = entry["calib_samples"]
            baseline_height  = entry["baseline_height"]
            pitch_history    = entry["pitch_history"]

            _dt = max(0.0, min(0.5, current_time - entry.get("_gz_prev", current_time)))
            entry["_gz_prev"] = current_time
            cv2.putText(frame, f"CAM: {entry['label']}", (20, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if not state.monitoring_active:
                _clear_tracking_state(entry)
                _finish_recording(entry)
                cv2.putText(frame, "MONITORING PAUSED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                ret, buf = cv2.imencode('.jpg', frame)
                if ret:
                    entry["latest_frame"] = buf.tobytes()
                time.sleep(0.05)
                continue

            object_violation      = False
            reach_violation       = False
            gaze_trigger          = False
            violation_reason      = ""
            violation_confidence  = 0.0
            violating_track_id    = None
            violations            = []

            # ==========================================
            # 1. PROHIBITED OBJECT DETECTION (YOLO)
            # ==========================================
            object_detections = []
            entry["_obj_n"] = entry.get("_obj_n", 0) + 1
            _obj_cache = entry.get("_obj_cache")
            _run_full_pass = (_obj_cache is None or not DETECTION_ENABLED["phone"]
                              or entry["_obj_n"] % max(1, PHONE_FULL_EVERY_N) == 0)
            if not _run_full_pass:
                object_detections = [dict(d) for d in _obj_cache]
            try:
                obj_results = (entry["object_model"](frame, stream=True, verbose=False, classes=[0],
                                                     imgsz=640, device=DEVICE, half=USE_HALF)
                               if (DETECTION_ENABLED["phone"] and _run_full_pass) else [])
                for r in obj_results:
                    for box in r.boxes:
                        cls_id     = int(box.cls[0])
                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bw, bh = x2 - x1, y2 - y1

                        if bh > h * 0.35 or bw > w * 0.35:
                            continue
                        if (bw * bh) > (w * h * 0.08):
                            continue

                        if (cls_id in PROHIBITED_CLASSES and confidence > PHONE_FULL_CONF
                                and DETECTION_ENABLED["phone"]):
                            object_detections.append({
                                "name":       PROHIBITED_CLASSES[cls_id],
                                "confidence": confidence,
                                "center":     ((x1 + x2) // 2, (y1 + y2) // 2),
                                "bbox":       (x1, y1, x2, y2),
                            })
            except Exception as e:
                print(f"[{camera_id}] Object detection error: {e}")
            if _run_full_pass:
                entry["_obj_cache"] = [dict(d) for d in object_detections]

            # ==========================================
            # 2. SKELETAL POSE ESTIMATION + TRACKING
            # ==========================================
            students        = []
            active_ids      = []

            try:
                pose_results = entry["pose_model"].track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    stream=True,
                    verbose=False,
                    iou=POSE_IOU,
                    imgsz=POSE_IMGSZ,
                    conf=POSE_CONF,
                    device=DEVICE,
                    half=USE_HALF,
                )

                for r in pose_results:
                    if r.keypoints is None or len(r.keypoints) == 0:
                        continue

                    kpts  = r.keypoints.xy.cpu().numpy()
                    confs = (r.keypoints.conf.cpu().numpy()
                             if r.keypoints.conf is not None else None)

                    if r.boxes is not None and r.boxes.id is not None:
                        track_ids = r.boxes.id.int().cpu().tolist()
                    else:
                        track_ids = [None] * len(kpts)

                    candidates = []
                    for idx, person in enumerate(kpts):
                        track_id = track_ids[idx] if idx < len(track_ids) else None
                        if track_id is None:
                            continue

                        if confs is not None:
                            if (confs[idx][5] < SHOULDER_MIN_CONF or
                                    confs[idx][6] < SHOULDER_MIN_CONF):
                                if DEBUG_OVERLAY and r.boxes is not None and idx < len(r.boxes):
                                    bx1, by1, bx2, by2 = map(int, r.boxes.xyxy[idx].tolist())
                                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (90, 90, 90), 1)
                                    cv2.putText(frame,
                                                f"sh {confs[idx][5]:.2f}/{confs[idx][6]:.2f}",
                                                (bx1, max(by1 - 4, 12)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
                                continue
                            pair_conf = float(confs[idx][5] + confs[idx][6]) / 2.0
                        else:
                            pair_conf = 1.0

                        l_sh, r_sh = person[5], person[6]
                        candidates.append({
                            "idx":      idx,
                            "track_id": track_id,
                            "center":   ((l_sh[0] + r_sh[0]) / 2.0,
                                         (l_sh[1] + r_sh[1]) / 2.0),
                            "width":    max(20.0, abs(r_sh[0] - l_sh[0])),
                            "conf":     pair_conf,
                        })
                        if track_id not in track_first_seen:
                            track_first_seen[track_id] = current_time

                    kept_idx = _select_unique_people(candidates, track_first_seen,
                                                     current_time)

                    for idx, person in enumerate(kpts):
                        if idx not in kept_idx:
                            continue
                        track_id = track_ids[idx]

                        nose                   = person[0]
                        l_shoulder, r_shoulder = person[5],  person[6]
                        l_wrist,    r_wrist    = person[9],  person[10]

                        if confs is not None:
                            c_nose   = confs[idx][0]
                            c_lwrist = confs[idx][9]
                            c_rwrist = confs[idx][10]
                        else:
                            c_nose = 1.0
                            c_lwrist = c_rwrist = 1.0

                        nose_confident = c_nose  >= MIN_KEYPOINT_CONF

                        shoulder_center = (int((l_shoulder[0] + r_shoulder[0]) / 2),
                                           int((l_shoulder[1] + r_shoulder[1]) / 2))
                        shoulder_width  = max(20.0, abs(r_shoulder[0] - l_shoulder[0]))

                        l_wrist_pt = ((int(l_wrist[0]), int(l_wrist[1]))
                                      if (c_lwrist >= REACH_WRIST_CONF and l_wrist[0] > 0) else None)
                        r_wrist_pt = ((int(r_wrist[0]), int(r_wrist[1]))
                                      if (c_rwrist >= REACH_WRIST_CONF and r_wrist[0] > 0) else None)

                        students.append({
                            "track_id":        track_id,
                            "shoulder_center": shoulder_center,
                            "shoulder_width":  shoulder_width,
                            "nose":            nose,
                            "l_wrist":         l_wrist_pt,
                            "r_wrist":         r_wrist_pt,
                            "person_turn":     "FORWARD",
                        })

                        cv2.circle(frame, (int(l_shoulder[0]), int(l_shoulder[1])), 4, (0, 255, 255), -1)
                        cv2.circle(frame, (int(r_shoulder[0]), int(r_shoulder[1])), 4, (0, 255, 255), -1)
                        cv2.line(frame,
                                 (int(l_shoulder[0]), int(l_shoulder[1])),
                                 (int(r_shoulder[0]), int(r_shoulder[1])),
                                 (0, 255, 255), 2)
                        cv2.putText(frame, f"ID {track_id}",
                                    (shoulder_center[0] - 20, shoulder_center[1] + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                        person_turn    = "FORWARD"
                        smoothed_ratio = 0.0
                        adjusted_ratio = 0.0
                        down_delta     = 0.0

                        sh_sign        = -1.0
                        if (nose_confident and nose[0] > 0
                                and l_shoulder[0] > 0 and r_shoulder[0] > 0):
                            d_x_sh = r_shoulder[0] - l_shoulder[0]
                            sh_sign = 1.0 if d_x_sh >= 0 else -1.0
                            d_y_sh = r_shoulder[1] - l_shoulder[1]
                            shoulder_angle_rad = math.atan2(d_y_sh, d_x_sh)

                            mid_sh_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
                            mid_sh_y = (l_shoulder[1] + r_shoulder[1]) / 2.0

                            d_x_head = nose[0] - mid_sh_x
                            d_y_head = nose[1] - mid_sh_y

                            cos_t = math.cos(-shoulder_angle_rad)
                            sin_t = math.sin(-shoulder_angle_rad)
                            sideways_offset = d_x_head * cos_t - d_y_head * sin_t
                            vertical_offset = d_x_head * sin_t + d_y_head * cos_t
                            raw_ratio       = sideways_offset / (shoulder_width / 2.0)
                            raw_height      = -vertical_offset / shoulder_width

                            if track_id not in calib_start:
                                calib_start[track_id]    = current_time
                                calib_samples[track_id]  = []
                                stable_since[track_id]   = current_time
                                stable_anchor[track_id]  = raw_ratio

                            if current_time - calib_start[track_id] < CALIBRATION_DURATION:
                                samples = calib_samples.setdefault(track_id, [])
                                samples.append((raw_ratio, raw_height))
                                baseline_ratio[track_id]  = float(np.median([x[0] for x in samples]))
                                baseline_height[track_id] = float(np.median([x[1] for x in samples]))
                                stable_anchor[track_id]   = baseline_ratio[track_id]
                                entry["home_ratio"][track_id]  = baseline_ratio[track_id]
                                entry["home_height"][track_id] = baseline_height[track_id]
                                person_turn = "CALIBRATING"
                            else:
                                baseline_ratio[track_id] = max(-BASELINE_MAX_ABS, min(BASELINE_MAX_ABS, baseline_ratio[track_id]))
                                entry["home_ratio"][track_id] = max(-BASELINE_MAX_ABS, min(BASELINE_MAX_ABS,
                                    entry["home_ratio"].get(track_id, baseline_ratio[track_id])))
                                home_r = entry["home_ratio"].get(track_id, baseline_ratio[track_id])
                                if (abs(baseline_ratio[track_id] - home_r) > REPOSITION_SETTLE_BAND
                                        and abs(raw_ratio - home_r)
                                            < abs(raw_ratio - baseline_ratio[track_id])):
                                    baseline_ratio[track_id]  = home_r
                                    baseline_height[track_id] = entry["home_height"].get(
                                        track_id, baseline_height[track_id])
                                    turn_history.pop(track_id, None)
                                    pitch_history.pop(track_id, None)
                                    stable_since[track_id]  = current_time
                                    stable_anchor[track_id] = raw_ratio
                                    print(f"[{camera_id}] ID {track_id}: back near home pose, "
                                          f"baseline restored.")
                                adjusted_ratio = raw_ratio - baseline_ratio[track_id]

                                if track_id not in turn_history:
                                    turn_history[track_id] = deque(maxlen=SMOOTHING_WINDOW)
                                turn_history[track_id].append(adjusted_ratio)
                                smoothed_ratio = (sum(turn_history[track_id])
                                                  / len(turn_history[track_id]))

                                if track_id not in pitch_history:
                                    pitch_history[track_id] = deque(maxlen=SMOOTHING_WINDOW)
                                pitch_history[track_id].append(raw_height)
                                smoothed_height = (sum(pitch_history[track_id])
                                                   / len(pitch_history[track_id]))
                                down_delta   = baseline_height[track_id] - smoothed_height
                                looking_down = down_delta > (PITCH_DOWN_DELTA * GAZE_DOWN_HYSTERESIS
                                                             if track_id in gaze_timers
                                                             else PITCH_DOWN_DELTA)

                                _prev = entry.get("last_turn", {}).get(track_id)
                                _l_thr = (HEAD_TURN_EXIT_RATIO if _prev == "LOOKING LEFT"
                                          else HEAD_TURN_RATIO_THRESHOLD)
                                _r_thr = (HEAD_TURN_EXIT_RATIO if _prev == "LOOKING RIGHT"
                                          else HEAD_TURN_RATIO_THRESHOLD)
                                if smoothed_ratio < -_l_thr:
                                    person_turn = "LOOKING LEFT"
                                elif smoothed_ratio > _r_thr:
                                    person_turn = "LOOKING RIGHT"

                                if (person_turn != "FORWARD" and looking_down
                                        and abs(smoothed_ratio) < HEAD_TURN_STRONG_RATIO):
                                    person_turn = "LOOKING DOWN"

                                if (track_id in gaze_timers and
                                        current_time - gaze_timers[track_id] >= GAZE_REBASELINE_SEC):
                                    baseline_ratio[track_id]  = raw_ratio
                                    baseline_height[track_id] = raw_height
                                    turn_history[track_id].clear()
                                    pitch_history[track_id].clear()
                                    gaze_timers.pop(track_id, None)
                                    stable_since[track_id]    = current_time
                                    stable_anchor[track_id]   = raw_ratio
                                    person_turn = "FORWARD"
                                    print(f"[{camera_id}] ID {track_id}: held posture for "
                                          f"{GAZE_REBASELINE_SEC:.0f}s, re-baselined.")
                                else:
                                    currently_flagged = track_id in gaze_timers
                                    if currently_flagged or abs(adjusted_ratio) > REPOSITION_JUMP_LIMIT:
                                        stable_since[track_id]  = current_time
                                        stable_anchor[track_id] = raw_ratio
                                    else:
                                        if abs(raw_ratio - stable_anchor[track_id]) > REPOSITION_SETTLE_BAND:
                                            stable_since[track_id]  = current_time
                                            stable_anchor[track_id] = raw_ratio
                                        else:
                                            held = current_time - stable_since[track_id]
                                            if held >= REPOSITION_STABILITY_WINDOW:
                                                baseline_ratio[track_id] += BASELINE_ADAPT_RATE * (
                                                    raw_ratio - baseline_ratio[track_id])
                                                baseline_height[track_id] += BASELINE_ADAPT_RATE * (
                                                    raw_height - baseline_height[track_id])

                        _hold = entry.setdefault("_turn_hold", {})
                        if person_turn in ("LOOKING LEFT", "LOOKING RIGHT"):
                            _hold[track_id] = (person_turn, smoothed_ratio, current_time)
                        elif (not nose_confident and track_id in _hold
                              and current_time - _hold[track_id][2] <= 0.8):
                            person_turn, smoothed_ratio = _hold[track_id][0], _hold[track_id][1]

                        if DEBUG_OVERLAY and person_turn != "CALIBRATING":
                            cv2.putText(frame, f"yaw {smoothed_ratio:+.2f} dn {down_delta:+.2f} b {baseline_ratio.get(track_id, 0.0):+.2f}",
                                        (shoulder_center[0] - 45, shoulder_center[1] + 42),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        students[-1]["person_turn"] = person_turn
                        students[-1]["down_delta"] = down_delta
                        last_turn = entry.setdefault("last_turn", {})
                        if last_turn.get(track_id) != person_turn:
                            print(f"[{camera_id}] ID {track_id}: {last_turn.get(track_id, '-')} -> "
                                  f"{person_turn}  yaw={smoothed_ratio:+.2f} dn={down_delta:+.2f} base={baseline_ratio.get(track_id, 0.0):+.2f}")
                            last_turn[track_id] = person_turn
                        active_ids.append(track_id)

                        gaze_candidate = (DETECTION_ENABLED["gaze"]
                                          and person_turn not in ("FORWARD", "CALIBRATING", "LOOKING DOWN")
                                          and _neighbor_in_direction(person_turn, track_id, candidates, kept_idx,
                                                                     entry, current_time,
                                                                     (1 if smoothed_ratio > 0 else -1) * sh_sign))
                        if (DETECTION_ENABLED["gaze"] and not gaze_candidate
                                and person_turn in ("LOOKING LEFT", "LOOKING RIGHT")):
                            cv2.putText(frame, entry.get("_nb_dbg", {}).get(track_id, "turn ignored: no neighbor"),
                                        (shoulder_center[0] - 70, max(20, shoulder_center[1] - 40)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                        if gaze_candidate:
                            entry["gaze_lost"].pop(track_id, None)
                            if track_id not in gaze_timers:
                                gaze_timers[track_id] = current_time
                            elapsed = current_time - gaze_timers[track_id]

                            if elapsed >= GAZE_DURATION_THRESHOLD:
                                if not entry.get("_gaze_logged", {}).get(track_id):
                                    entry.setdefault("_gaze_logged", {})[track_id] = True
                                    print(f"[{camera_id}] ID {track_id}: GAZE ALERT ({person_turn}, {elapsed:.1f}s)")
                                gaze_trigger = True
                                violations.append(("PROLONGED_SIDE_GAZE", 0.95, track_id))
                                if not violation_reason:
                                    violation_reason     = "PROLONGED_SIDE_GAZE"
                                    violation_confidence = 0.95
                                    violating_track_id   = track_id
                                cv2.putText(frame, person_turn,
                                            (shoulder_center[0] - 50,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                time_rem = GAZE_DURATION_THRESHOLD - elapsed
                                cv2.putText(frame, f"{person_turn} ({time_rem:.1f}s)",
                                            (shoulder_center[0] - 60,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        else:
                            if person_turn == "CALIBRATING":
                                gaze_timers.pop(track_id, None)
                                turn_history.pop(track_id, None)
                                entry.get("_gaze_logged", {}).pop(track_id, None)
                            elif track_id in gaze_timers:
                                gaze_timers[track_id] += _dt * (1.0 + GAZE_DRAIN_RATE)
                                if current_time - gaze_timers[track_id] <= 0.0:
                                    print(f"[{camera_id}] ID {track_id}: gaze timer dropped "
                                          f"(turn={person_turn}, yaw={smoothed_ratio:+.2f}, dn={down_delta:+.2f})")
                                    gaze_timers.pop(track_id, None)
                                    entry.get("_gaze_logged", {}).pop(track_id, None)
                            if person_turn == "LOOKING DOWN":
                                cv2.putText(frame, "looking down (ok)",
                                            (shoulder_center[0] - 55,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
                            if person_turn == "CALIBRATING":
                                cv2.putText(frame, "CALIBRATING...",
                                            (shoulder_center[0] - 55,
                                             max(20, shoulder_center[1] - 40)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

            except Exception as e:
                print(f"[{camera_id}] Pose estimation error: {e}")
                students   = []
                active_ids = []

            for tid in active_ids:
                track_last_seen[tid] = current_time
            for tid in list(track_first_seen):
                track_last_seen.setdefault(tid, current_time)
            expired = [tid for tid, ts in track_last_seen.items()
                       if current_time - ts > TRACK_LOST_GRACE_SEC]
            for tid in expired:
                for key in PER_TRACK_KEYS:
                    entry[key].pop(tid, None)
                for key in ("last_turn", "_gaze_logged", "_nb_seen", "_nb_dbg"):
                    entry.get(key, {}).pop(tid, None)

            _update_hand_rise(entry, students, current_time)
            if DEBUG_OVERLAY and PHONE_NEEDS_HAND_RISE:
                for _s in students:
                    _ok = _hand_rise_ok(entry, _s["track_id"], current_time)
                    cv2.putText(frame, "rise" if _ok else "no-rise",
                                (_s["shoulder_center"][0] - 45, _s["shoulder_center"][1] + 56),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (0, 255, 0) if _ok else (150, 150, 150), 1)
            _merge_wrist_phones(entry, camera_id, clean_frame, students,
                                current_time, object_detections)
            if DEBUG_OVERLAY:
                for (dx1, dy1, dx2, dy2, dtid, dtop) in entry.get("_wc_debug", []):
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 0, 255), 1)
                    cv2.putText(frame, f"{dtop[0]:.2f}",
                        (dx1, max(dy1 - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            if DEBUG_OVERLAY:
                for (rx1, ry1, rx2, ry2, why) in entry.get("_wc_rej", []):
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 200, 255), 1)
                    cv2.putText(frame, why, (rx1, max(ry1 - 4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            if DEBUG_OVERLAY and PHONE_VERIFY != "off" and entry.get("_pv") is not None:
                for (px1, py1, px2, py2, plabel, pcolor) in entry["_pv"].overlay(current_time):
                    cv2.rectangle(frame, (px1, py1), (px2, py2), pcolor, 1)
                    cv2.putText(frame, plabel, (px1, min(h - 4, py2 + 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, pcolor, 1)
            established_ids = {tid for tid, seen in track_first_seen.items()
                               if current_time - seen >= TRACK_MIN_AGE_SEC}

            # ==========================================
            # 2.6 CAMERA-BUMP DETECTION
            # ==========================================
            scored = [s for s in students if s["person_turn"] != "CALIBRATING"]
            if len(scored) >= CAMERA_BUMP_MIN_STUDENTS:
                turned = sum(1 for s in scored
                             if s["person_turn"] in ("LOOKING LEFT", "LOOKING RIGHT"))
                if (turned / len(scored) >= CAMERA_BUMP_RATIO
                        and current_time - entry["last_bump_time"] >= CAMERA_BUMP_COOLDOWN_SEC):
                    print(f"[{camera_id}] Possible camera disturbance: {turned}/{len(scored)} "
                          f"students turned at once. Recalibrating all baselines.")
                    entry["last_bump_time"] = current_time
                    entry["calibration_start"].clear()
                    entry["gaze_timers"].clear()
                    entry["turn_history"].clear()

            # ==========================================
            # 2.5. STATIC-PHONE FILTER
            # ==========================================
            phone_dets = [(i, d) for i, d in enumerate(object_detections)
                          if d["name"] == "CELL PHONE" and d.get("track_id") is None]
            phone_sup  = _check_static_objects(entry, [d["center"] for _, d in phone_dets],
                                               current_time, store="static_phones",
                                               static_sec=STATIC_PHONE_SEC)
            static_phone_idx = {phone_dets[k][0] for k in phone_sup}

            ref_sw = (float(np.median([s["shoulder_width"] for s in students]))
                      if students else None)
            accepted_detections = []
            for i, det in enumerate(object_detections):
                x1, y1, x2, y2 = det["bbox"]
                if PHONE_VERIFY != "off" and det.get("track_id") is None:
                    _pvr = _pv.store_for(entry, PHONE_VERIFY_DUMP_DIR if PHONE_VERIFY_DUMP else None,
                                         camera_id).feed(clean_frame, det["bbox"],
                                                         det["confidence"], current_time)
                    if PHONE_VERIFY == "on" and _pvr["verdict"] != "CONFIRMED":
                        if DEBUG_OVERLAY:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                            cv2.putText(frame, "UNVERIFIED (" + _pvr["verdict"].lower() + ")",
                                        (x1, max(y1 - 8, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                        continue
                if PHONE_VERIFY != "on" and det.get("track_id") is None and _pen_like(
                        det["bbox"], _local_sw(det, students, ref_sw), det["confidence"]):
                    if DEBUG_OVERLAY:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                        cv2.putText(frame, "PEN-LIKE (ignored)", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                    continue
                if (PHONE_FULL_NEEDS_HAND and det.get("track_id") is None and not _phone_in_hand(det, students)):
                    if DEBUG_OVERLAY:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                        cv2.putText(frame, "NOT IN HAND (ignored)", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                    continue
                if PHONE_VERIFY != "on" and det.get("track_id") is None and _badge_like(det["center"], students, clean_frame, det["bbox"]):
                    if DEBUG_OVERLAY:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                        cv2.putText(frame, "ID BADGE (ignored)", (x1, max(y1 - 8, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                    continue
                if PHONE_VERIFY != "on" and det.get("track_id") is None:
                    _own = _owner_of_detection(det, students, established_ids)
                    _os = next((s for s in students if s["track_id"] == _own), None)
                    if _os is not None:
                        _verdict = _torso_lock_verdict(entry, _os, det["center"], current_time)
                        if _verdict != "phone":
                            if DEBUG_OVERLAY:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                                cv2.putText(frame, ("ID BADGE (moves with body)" if _verdict == "badge"
                                                    else "CHECKING..."),
                                            (x1, max(y1 - 8, 12)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)
                            continue
                if det.get("track_id") is None and PHONE_NEEDS_HAND_RISE:
                    _ro = _owner_of_detection(det, students, established_ids)
                    if _ro is not None:
                        if not _hand_rise_ok(entry, _ro, current_time):
                            if DEBUG_OVERLAY:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)
                                cv2.putText(frame, "NO HAND RISE (ignored)", (x1, max(y1 - 8, 12)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)
                            continue
                        entry.setdefault("_phone_ok", {})[_ro] = current_time
                if i in static_phone_idx and not _phone_in_hand(det, students):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)
                    cv2.putText(frame, "STATIC OBJECT (suppressed)", (x1, max(y1 - 8, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (70, 70, 70), 1)
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"CELL PHONE ({det['confidence']*100:.0f}%)",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                accepted_detections.append(det)

            object_detections = accepted_detections

            # ==========================================
            # 3. ATTRIBUTE OBJECT DETECTIONS TO A STUDENT
            # ==========================================
            if object_detections:
                object_violation = True
                best_det = max(object_detections, key=lambda d: d["confidence"])
                violation_reason     = f"OBJECT_{best_det['name'].replace(' ', '_')}"
                violation_confidence = best_det["confidence"]

                nearest_student = None
                nearest_dist    = float("inf")
                for s in students:
                    if s["track_id"] not in established_ids:
                        continue
                    d = math.hypot(s["shoulder_center"][0] - best_det["center"][0],
                                   s["shoulder_center"][1] - best_det["center"][1])
                    radius = s["shoulder_width"] * ATTRIBUTION_RADIUS_MULTIPLIER
                    if d < radius and d < nearest_dist:
                        nearest_dist    = d
                        nearest_student = s
                violating_track_id = nearest_student["track_id"] if nearest_student else None
                if best_det.get("track_id") is not None:
                    violating_track_id = best_det["track_id"]
                for det in object_detections:
                    violations.append((f"OBJECT_{det['name'].replace(' ', '_')}",
                                       det["confidence"],
                                       _owner_of_detection(det, students, established_ids)))

            # ==========================================
            # 4. INTERACTION & REACH DETECTION
            # ==========================================
            if DEBUG_OVERLAY:
                for _s in students:
                    for _wr in (_s["l_wrist"], _s["r_wrist"]):
                        if _wr is not None:
                            cv2.circle(frame, _wr, 4, (0, 165, 255), -1)

            if HANDS_POSTURE_ENABLED and DETECTION_ENABLED["phone"]:
                _hp = entry.setdefault("_hp", {})
                _hp_now = set()
                for _s in students:
                    _tid = _s["track_id"]
                    if _tid not in established_ids or _s["person_turn"] == "CALIBRATING":
                        continue
                    if not _hands_posture(_s):
                        continue
                    _hp_now.add(_tid)
                    _st = _hp.setdefault(_tid, {"start": current_time, "last": current_time})
                    _st["last"] = current_time
                    _held = current_time - _st["start"]
                    _pos = (_s["shoulder_center"][0] - 60, _s["shoulder_center"][1] + 60)
                    if _held >= HANDS_POSTURE_SEC:
                        object_violation = True
                        violations.append(("OBJECT_PHONE_POSTURE", 0.6, _tid))
                        if not violation_reason:
                            violation_reason     = "OBJECT_PHONE_POSTURE"
                            violation_confidence = 0.6
                            violating_track_id   = _tid
                        cv2.putText(frame, "PHONE? (hands at chest)", _pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"hands up ({HANDS_POSTURE_SEC - _held:.1f}s)", _pos,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                for _k in [k for k, v in _hp.items()
                           if k not in _hp_now and current_time - v["last"] > HANDS_POSTURE_GRACE_SEC]:
                    _hp.pop(_k, None)

            interaction_candidates = [s for s in students if s["track_id"] in established_ids]
            active_reach_pairs = []
            num_students = len(interaction_candidates)
            for i in range(num_students):
                for j in range(i + 1, num_students):
                    s1, s2 = interaction_candidates[i], interaction_candidates[j]

                    dist           = math.hypot(
                        s1["shoulder_center"][0] - s2["shoulder_center"][0],
                        s1["shoulder_center"][1] - s2["shoulder_center"][1])
                    avg_shoulder_w  = (s1["shoulder_width"] + s2["shoulder_width"]) / 2.0

                    if dist < min(s1["shoulder_width"], s2["shoulder_width"]) * DUPLICATE_DIST_MULTIPLIER:
                        continue

                    if dist > avg_shoulder_w * REACH_MAX_PAIR_DIST_MULTIPLIER:
                        continue
                    same_row = abs(s1["shoulder_center"][1] - s2["shoulder_center"][1]) \
                               < avg_shoulder_w * REACH_SAME_ROW_MULTIPLIER

                    depth_ratio = s1["shoulder_width"] / max(1.0, s2["shoulder_width"])
                    side_gap = abs(s1["shoulder_center"][0] - s2["shoulder_center"][0])
                    if (not same_row
                            or not (REACH_DEPTH_RATIO_MIN <= depth_ratio <= REACH_DEPTH_RATIO_MAX)
                            or side_gap < avg_shoulder_w * REACH_MIN_SIDE_GAP_MULTIPLIER):
                        continue
                    for reacher, target in ((s1, s2), (s2, s1)):
                        reach_key   = (reacher["track_id"], target["track_id"])
                        reach_wrist = _find_reaching_wrist(reacher, target, same_row)

                        if reach_wrist is not None:
                            active_reach_pairs.append(reach_key)
                            if reach_key not in reach_timers:
                                reach_timers[reach_key] = {"start": current_time, "last": current_time}
                            reach_timers[reach_key]["last"] = current_time
                            r_elapsed = current_time - reach_timers[reach_key]["start"]

                            if r_elapsed >= REACH_DURATION_THRESHOLD and DETECTION_ENABLED["reach"]:
                                reach_violation = True
                                violations.append(("REACHING_NEIGHBOR_ZONE", 0.85,
                                                   min(reacher["track_id"], target["track_id"])))
                                if not violation_reason:
                                    violation_reason     = "REACHING_NEIGHBOR_ZONE"
                                    violation_confidence = 0.85
                                    violating_track_id   = reacher["track_id"]

                                cv2.line(frame, reach_wrist, target["shoulder_center"],
                                         (0, 140, 255), 2)
                                cv2.putText(frame, "REACH",
                                            (reach_wrist[0] - 20, max(20, reach_wrist[1] - 10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            for k in [k for k, v in reach_timers.items()
                      if k not in active_reach_pairs
                      and current_time - v["last"] > REACH_GRACE_SEC]:
                reach_timers.pop(k, None)

            # ==========================================
            # 5. HUD / DASHBOARD STATUS OVERLAY
            # ==========================================
            status_color = (0, 255, 0)
            status_text  = "STATUS: CLEAR"

            if object_violation:
                status_text  = f"ALERT: {violation_reason}!"
                status_color = (0, 0, 255)
            elif reach_violation:
                status_text  = "ALERT: SUSPECTED NOTE PASSING!"
                status_color = (0, 0, 255)
            elif gaze_trigger:
                status_text  = "ALERT: PROLONGED SIDE GAZE!"
                status_color = (0, 0, 255)
            elif len(entry["gaze_timers"]) > 0:
                status_text  = "WARNING: SIDE GAZE TRACKED"
                status_color = (0, 255, 255)

            cv2.putText(frame, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            # ==========================================
            # 6. SESSION RECORDING + INCIDENT LOGGING
            # ==========================================
            entry["recorder"].write(clean_frame if RECORD_RAW else frame)

            is_violating = object_violation or reach_violation or gaze_trigger

            if is_violating and violations:
                unique = {}
                for v_reason, v_conf, v_tid in violations:
                    k = (v_tid, v_reason)
                    if k not in unique or v_conf > unique[k]:
                        unique[k] = v_conf
                for (v_tid, v_reason), v_conf in unique.items():
                    inc, is_new = entry["incidents"].observe(
                        entry["recorder"], frame, v_reason, v_tid,
                        v_conf, state.CURRENT_SESSION, camera_id,
                        thumb_dir=os.path.join(VIOLATION_FOLDER, state.CURRENT_SESSION, camera_id),
                        thumb_web_prefix=f"violations/{state.CURRENT_SESSION}/{camera_id}")
                    if is_new:
                        log_violation(state.CURRENT_SESSION, v_reason, v_conf,
                                      inc["thumb_web"], student_track_id=v_tid,
                                      camera_id=camera_id)
                        print(f"[{camera_id}] NEW INCIDENT {v_reason} "
                              f"(track_id={v_tid}, t={inc['start']:.1f}s)")

            if entry.get("_smooth_display"):
                entry["_overlay"] = {"mask": np.any(frame != clean_frame, axis=2),
                                     "frame": frame, "t": time.time()}
            else:
                ret, buf = cv2.imencode('.jpg', frame)
                if ret:
                    entry["latest_frame"] = buf.tobytes()

    finally:
        _finish_recording(entry)

