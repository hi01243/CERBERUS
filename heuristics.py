"""Computer vision heuristics, filters, and crop analysis for Cerberus."""
import math
import time
import cv2
import numpy as np

import phone_verify as _pv
from config import (
    DEVICE, USE_HALF, _load_yolo, DETECTION_ENABLED,
    WRIST_CROP_ENABLED, WRIST_CROP_SCALE, WRIST_CROP_MIN_HALF_PX,
    WRIST_CROP_IMGSZ, WRIST_CROP_CONF, WRIST_CROP_INTERVAL_SEC,
    WRIST_CROP_MAX_CROPS, WRIST_CROP_MIN_HITS, WRIST_CROP_WINDOW_SEC,
    WRIST_CROP_HOLD_SEC, WRIST_CROP_DEDUP_PX, WRIST_CROP_MODEL,
    WRIST_CROP_CLASSES, WRIST_CROP_FLOOR_CONF, WRIST_CROP_AUGMENT,
    WRIST_CROP_REST_SKIP, WRIST_CROP_REST_MOVE_SW, WRIST_CROP_REST_RECHECK_SEC,
    WRIST_CROP_MAX_WRIST_DIST, WRIST_CROP_STATIC_SEC, WRIST_CROP_STATIC_MOVE_PX,
    WRIST_HANDS_CROP, WRIST_HANDS_MAX_GAP_SW, WRIST_HANDS_MIN_HALF_SW,
    WRIST_WEAK_CONF, WRIST_WEAK_MIN_HITS, WRIST_WEAK_WINDOW_SEC,
    WRIST_WEAK_MAX_GAP_SW, WRIST_WEAK_MIN_DROP_SW, WRIST_WEAK_MAX_DROP_SW,
    DESK_CROP_ENABLED, DESK_REQUIRE_LOOKDOWN, DESK_LOOKDOWN_MAX_RISE_SW,
    DESK_HALF_W_SW, DESK_TOP_SW, DESK_BOTTOM_SW, DESK_MAX_CROPS,
    DESK_IMGSZ, DESK_FLOOR_CONF, DESK_CROP_CONF, DESK_MIN_HITS,
    DESK_WINDOW_SEC, DESK_HOLD_SEC, DESK_MAX_MEAN_GRAY, DESK_DARK_PIXEL,
    DESK_MIN_DARK_FRAC, DESK_MIN_SHORT_SW, DESK_MIN_LONG_SW, DESK_MAX_HAND_SW,
    PHONE_VERIFY, PHONE_VERIFY_DUMP, PHONE_VERIFY_DUMP_DIR,
    PHONE_MAX_ASPECT, PHONE_MIN_SHORT_SW, PHONE_TINY_SW, PHONE_SMALL_SW,
    PHONE_SMALL_MIN_CONF, PHONE_MAX_LONG_SW,
    PHONE_NEEDS_HAND_RISE, HAND_LOW_SW, HAND_LOW_MIN_SEC,
    HAND_RISE_MEMORY_SEC, HAND_RISE_KEEP_SEC,
    BADGE_FILTER_ENABLED, BADGE_ZONE_HALF_W, BADGE_ZONE_TOP, BADGE_ZONE_BOTTOM,
    BADGE_HAND_HOLD_MULT, BADGE_TWO_HAND_MULT, BADGE_WHITE_V, BADGE_WHITE_S,
    BADGE_WHITE_FRAC, BADGE_LOCK_WINDOW_SEC, BADGE_LOCK_MIN_SEC,
    BADGE_LOCK_STD, BADGE_LOCK_MIN_REL, BADGE_LOCK_RATIO,
    GAZE_REQUIRE_NEIGHBOR, GAZE_NEIGHBOR_MIN_DX, GAZE_NEIGHBOR_MAX_DX,
    GAZE_NEIGHBOR_ROW_MULT, GAZE_NEIGHBOR_ROW_SLOPE, GAZE_NEIGHBOR_MEMORY_SEC,
    STATIC_OBJECT_RADIUS_PX, STATIC_OBJECT_MOVE_PX, STATIC_PHONE_SEC,
    HELD_PHONE_WRIST_MULT, DUPLICATE_DIST_MULTIPLIER, TRACK_MIN_AGE_SEC,
    ATTRIBUTION_RADIUS_MULTIPLIER,
    REACH_ZONE_RADIUS_MULTIPLIER, REACH_ZONE_MIN_EXTENSION,
    REACH_EXTENSION_MULTIPLIER, REACH_PROGRESS_MIN,
    REACH_MID_MIN_PROGRESS, REACH_MID_MAX_PROGRESS, REACH_MID_MIN_EXTENSION,
    REACH_MID_MAX_DY,
    HANDS_POSTURE_MAX_GAP_SW, HANDS_POSTURE_MIN_DROP_SW, HANDS_POSTURE_MAX_DROP_SW
)


def _neighbor_in_direction(turn, track_id, candidates, kept_idx, entry=None, now=None,
                           image_dir=None):
    """True if another (non-ghost) student sits on the side this student is
    turned toward, in roughly the same row."""
    if not GAZE_REQUIRE_NEIGHBOR or turn not in ("LOOKING LEFT", "LOOKING RIGHT"):
        return True
    me = next((c for c in candidates
               if c["track_id"] == track_id and c["idx"] in kept_idx), None)
    if me is None:
        return True
    if image_dir is None:
        image_dir = 1 if turn == "LOOKING LEFT" else -1
    direction = image_dir   # +1 = look on the image-right, -1 = image-left
    side = "L" if direction < 0 else "R"
    nearest = {"L": None, "R": None}
    found = False
    for c in candidates:
        if c["idx"] not in kept_idx or c["track_id"] == track_id:
            continue
        scale = max(me["width"], c["width"])
        sdx = (c["center"][0] - me["center"][0]) / scale
        dy = abs(c["center"][1] - me["center"][1]) / scale
        dx = sdx * direction
        s = "R" if sdx > 0 else "L"
        if nearest[s] is None or abs(sdx) < abs(nearest[s][0]):
            nearest[s] = (sdx, dy)
        if (GAZE_NEIGHBOR_MIN_DX <= dx <= GAZE_NEIGHBOR_MAX_DX
                and dy <= GAZE_NEIGHBOR_ROW_MULT + GAZE_NEIGHBOR_ROW_SLOPE * dx):
            found = True
    if entry is not None and now is not None:
        seen = entry.setdefault("_nb_seen", {}).setdefault(track_id, {})
        if found:
            seen[side] = now
        elif now - seen.get(side, -1e9) <= GAZE_NEIGHBOR_MEMORY_SEC:
            found = True
    if entry is not None:
        dbg = entry.setdefault("_nb_dbg", {})
        if found:
            dbg.pop(track_id, None)
        else:
            def fmt(k):
                return "-" if nearest[k] is None else f"{nearest[k][0]:+.1f}/{nearest[k][1]:.1f}"
            dbg[track_id] = f"no neighbor {side} (L {fmt('L')} R {fmt('R')})"
    return found


def _wrist_model(entry):
    """Model used for the zoomed-in wrist-crop pass (loaded once per camera)."""
    m = entry.get("wrist_model")
    if m is None:
        try:
            m = _load_yolo(WRIST_CROP_MODEL)
            print(f"[cerberus] wrist-crop model: {WRIST_CROP_MODEL}")
        except Exception as e:
            print(f"[cerberus] could not load {WRIST_CROP_MODEL} ({e}); using the nano model")
            m = entry["object_model"]
        entry["wrist_model"] = m
    return m


def _run_wrist_model(entry, crops):
    """Runs the zoomed-in phone model, with test-time augmentation when possible."""
    model = _wrist_model(entry)
    kw = dict(imgsz=WRIST_CROP_IMGSZ, conf=WRIST_CROP_FLOOR_CONF,
              classes=WRIST_CROP_CLASSES, verbose=False,
              device=DEVICE, half=USE_HALF)
    global WRIST_CROP_AUGMENT
    if WRIST_CROP_AUGMENT:
        try:
            return model(crops, augment=True, **kw)
        except Exception as e:
            print(f"[cerberus] wrist-crop augment failed ({e}); disabling")
            WRIST_CROP_AUGMENT = False
    return model(crops, **kw)


def _wrist_can_rest(entry, key, pos, sw, now):
    """True if we checked this wrist recently, saw nothing, and the hand hasn't moved."""
    if not WRIST_CROP_REST_SKIP:
        return False
    st = entry.get("_wc_rest", {}).get(key)
    if st is None or not st["empty"] or now - st["t"] >= WRIST_CROP_REST_RECHECK_SEC:
        return False
    return math.hypot(pos[0] - st["pos"][0], pos[1] - st["pos"][1]) < WRIST_CROP_REST_MOVE_SW * sw


def _wrist_crop_sample(entry, camera_id, frame, students, now):
    """One zoomed-in phone pass over every established student's wrists."""
    h, w = frame.shape[:2]
    entry["_wc_rej"] = []
    crops, meta, rest_keys = [], [], []
    for s in students:
        tid = s["track_id"]
        if now - entry["track_first_seen"].get(tid, now) < TRACK_MIN_AGE_SEC:
            continue
        half = int(max(WRIST_CROP_MIN_HALF_PX, s["shoulder_width"] * WRIST_CROP_SCALE / 2.0))
        for side, wrist in (("L", s["l_wrist"]), ("R", s["r_wrist"])):
            if wrist is None:
                continue
            if _wrist_can_rest(entry, (tid, side), wrist, s["shoulder_width"], now):
                continue
            x1, y1 = max(0, wrist[0] - half), max(0, wrist[1] - half)
            x2, y2 = min(w, wrist[0] + half), min(h, wrist[1] + half)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            crops.append(frame[y1:y2, x1:x2].copy())
            meta.append((tid, x1, y1, wrist, s["shoulder_width"]))
            rest_keys.append(((tid, side), wrist, s["shoulder_width"]))
            if len(crops) >= WRIST_CROP_MAX_CROPS:
                break
        if (WRIST_HANDS_CROP and s["l_wrist"] is not None and s["r_wrist"] is not None
                and len(crops) < WRIST_CROP_MAX_CROPS):
            lw, rw = s["l_wrist"], s["r_wrist"]
            gap = math.hypot(lw[0] - rw[0], lw[1] - rw[1])
            if gap <= WRIST_HANDS_MAX_GAP_SW * s["shoulder_width"]:
                mx, my = (lw[0] + rw[0]) // 2, (lw[1] + rw[1]) // 2
                hh = int(max(WRIST_CROP_MIN_HALF_PX,
                             WRIST_HANDS_MIN_HALF_SW * s["shoulder_width"],
                             gap * 0.75))
                x1, y1 = max(0, mx - hh), max(0, my - hh)
                x2, y2 = min(w, mx + hh), min(h, my + hh)
                if (x2 - x1 >= 16 and y2 - y1 >= 16
                        and not _wrist_can_rest(entry, (tid, "H"), (mx, my),
                                               s["shoulder_width"], now)):
                    crops.append(frame[y1:y2, x1:x2].copy())
                    meta.append((tid, x1, y1, (mx, my), s["shoulder_width"]))
                    rest_keys.append(((tid, "H"), (mx, my), s["shoulder_width"]))
        if len(crops) >= WRIST_CROP_MAX_CROPS:
            break
    if not crops:
        entry["_wc_debug"] = []
        return
    entry["_wc_ncrops"] = len(crops)

    results = _run_wrist_model(entry, crops)
    entry["_wc_debug"] = [
        (ox, oy, ox + c.shape[1], oy + c.shape[0], tid,
         max([(float(b.conf[0]), int(b.cls[0])) for b in r.boxes], default=(0.0, -1)))
        for (tid, ox, oy, _wpt, _sw), c, r in zip(meta, crops, results)
    ]

    for (key, pos, _sw), r in zip(rest_keys, results):
        entry.setdefault("_wc_rest", {})[key] = {"pos": pos, "empty": len(r.boxes) == 0, "t": now}
    best = {}
    pv_by_tid = {}
    _stu = {s["track_id"]: s for s in students}
    for (tid, ox, oy, wpt, sw), r in zip(meta, results):
        for box in r.boxes:
            conf = float(box.conf[0])
            weak = False
            if conf < WRIST_CROP_CONF:
                if conf >= WRIST_WEAK_CONF and _hands_at_chest(_stu.get(tid)):
                    weak = True
                else:
                    continue
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            ccx, ccy = ox + (bx1 + bx2) // 2, oy + (by1 + by2) // 2
            full_box = (ox + bx1, oy + by1, ox + bx2, oy + by2)
            rej = entry.setdefault("_wc_rej", [])
            if math.hypot(ccx - wpt[0], ccy - wpt[1]) > WRIST_CROP_MAX_WRIST_DIST * sw:
                rej.append(full_box + ("FAR FROM HAND",))
                continue
            if PHONE_VERIFY != "off":
                _pvr = _pv.store_for(entry, PHONE_VERIFY_DUMP_DIR if PHONE_VERIFY_DUMP else None,
                                     camera_id).feed(frame, full_box, conf, now, owner=tid)
                if tid not in pv_by_tid or conf > pv_by_tid[tid][0]:
                    pv_by_tid[tid] = (conf, _pvr)
            if PHONE_VERIFY != "on" and _pen_like((bx1, by1, bx2, by2), sw, conf):
                rej.append(full_box + ("PEN-LIKE",))
                continue
            if PHONE_VERIFY != "on" and _badge_like((ccx, ccy), students, frame, full_box):
                rej.append(full_box + ("ID BADGE",))
                continue
            if tid in _stu and PHONE_VERIFY != "on":
                verdict = _torso_lock_verdict(entry, _stu[tid], (ccx, ccy), now)
                if verdict != "phone":
                    rej.append(full_box + (("ID BADGE (moves with body)" if verdict == "badge"
                                            else "CHECKING..."),))
                    continue
            if not _hand_rise_ok(entry, tid, now):
                rej.append(full_box + ("NO HAND RISE",))
                continue
            if tid not in best or conf > best[tid]["confidence"]:
                best[tid] = {
                    "name":       "CELL PHONE",
                    "confidence": conf,
                    "center":     (ox + (bx1 + bx2) // 2, oy + (by1 + by2) // 2),
                    "bbox":       (ox + bx1, oy + by1, ox + bx2, oy + by2),
                    "weak":       weak,
                }

    hits      = entry.setdefault("_wc_hits", {})
    confirmed = entry.setdefault("_wc_confirmed", {})

    for tid, det in best.items():
        weak = det.get("weak")
        store = entry.setdefault("_wc_weak_hits", {}) if weak else hits
        window = WRIST_WEAK_WINDOW_SEC if weak else WRIST_CROP_WINDOW_SEC
        need_hits = WRIST_WEAK_MIN_HITS if weak else WRIST_CROP_MIN_HITS
        recent = [t for t in store.get(tid, []) if now - t <= window]
        recent.append(now)
        store[tid] = recent
        if len(recent) >= need_hits and (PHONE_VERIFY != "on" or
                                         pv_by_tid.get(tid, (0, {}))[1].get("verdict") == "CONFIRMED"):
            if tid not in confirmed:
                print(f"[{camera_id}] ID {tid}: phone in hand (wrist crop, "
                      f"{det['confidence']*100:.0f}%)")
            confirmed[tid] = {"det": det, "until": now + WRIST_CROP_HOLD_SEC}
            entry.setdefault("_phone_ok", {})[tid] = now

    for tid in list(hits):
        if tid not in best:
            hits[tid] = [t for t in hits[tid] if now - t <= WRIST_CROP_WINDOW_SEC]
            if not hits[tid]:
                del hits[tid]


def _desk_looking_down(s):
    """r49: True if the student's head is down (nose close to the shoulder line)."""
    if not DESK_REQUIRE_LOOKDOWN:
        return True
    try:
        nose = s.get("nose")
        if nose is None or float(nose[0]) <= 0:
            return True
        rise = (s["shoulder_center"][1] - float(nose[1])) / float(s["shoulder_width"])
        return rise < DESK_LOOKDOWN_MAX_RISE_SW
    except Exception:
        return True


def _desk_patch_ok(patch):
    """r49: rejects white paper and bare skin: a phone box needs dark pixels."""
    if patch is None or patch.size == 0:
        return False
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) > DESK_MAX_MEAN_GRAY:
        return False
    return np.count_nonzero(gray < DESK_DARK_PIXEL) / float(gray.size) >= DESK_MIN_DARK_FRAC


def _desk_crop_sample(entry, camera_id, frame, students, now):
    """r49: one zoomed pass over the desk / lap area of every student who is looking down."""
    h, w = frame.shape[:2]
    crops, meta = [], []
    for s in students:
        tid = s["track_id"]
        if now - entry["track_first_seen"].get(tid, now) < TRACK_MIN_AGE_SEC:
            continue
        if s.get("person_turn") == "CALIBRATING" or not _desk_looking_down(s):
            continue
        sx, sy = s["shoulder_center"]
        sw = s["shoulder_width"]
        x1, x2 = max(0, int(sx - DESK_HALF_W_SW * sw)), min(w, int(sx + DESK_HALF_W_SW * sw))
        y1, y2 = max(0, int(sy + DESK_TOP_SW * sw)), min(h, int(sy + DESK_BOTTOM_SW * sw))
        if x2 - x1 < 24 or y2 - y1 < 24:
            continue
        crops.append(frame[y1:y2, x1:x2].copy())
        meta.append((tid, x1, y1, sw, s))
        if len(crops) >= DESK_MAX_CROPS:
            break
    entry["_desk_ncrops"] = len(crops)
    if not crops:
        return

    results = _wrist_model(entry)(crops, imgsz=DESK_IMGSZ, conf=DESK_FLOOR_CONF,
                                  classes=WRIST_CROP_CLASSES, verbose=False,
                                  device=DEVICE, half=USE_HALF)
    rej = entry.setdefault("_wc_rej", [])
    best = {}
    for (tid, ox, oy, sw, s), crop, r in zip(meta, crops, results):
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < DESK_CROP_CONF:
                continue
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            full_box = (ox + bx1, oy + by1, ox + bx2, oy + by2)
            ccx, ccy = (full_box[0] + full_box[2]) // 2, (full_box[1] + full_box[3]) // 2
            hands = [wr for wr in (s["l_wrist"], s["r_wrist"]) if wr is not None]
            _hd = min((math.hypot(ccx - wr[0], ccy - wr[1]) for wr in hands), default=1e9) / float(sw)
            if _hd > DESK_MAX_HAND_SW:
                rej.append(full_box + (f"DESK: FAR FROM HAND ({_hd:.1f} sw)",))
                continue
            _bw, _bh = bx2 - bx1, by2 - by1
            if min(_bw, _bh) < DESK_MIN_SHORT_SW * sw or max(_bw, _bh) < DESK_MIN_LONG_SW * sw:
                rej.append(full_box + (f"DESK: TOO SMALL ({max(_bw, _bh) / sw:.2f} x {min(_bw, _bh) / sw:.2f} sw)",))
                continue
            if _pen_like((bx1, by1, bx2, by2), sw, conf):
                rej.append(full_box + ("DESK: PEN-LIKE",))
                continue
            if _badge_like((ccx, ccy), students, frame, full_box):
                rej.append(full_box + ("DESK: ID BADGE",))
                continue
            _verdict = _torso_lock_verdict(entry, s, (ccx, ccy), now)
            if _verdict != "phone":
                rej.append(full_box + (("DESK: ID BADGE (moves with body)" if _verdict == "badge"
                                       else "DESK: CHECKING..."),))
                continue
            if not _desk_patch_ok(crop[by1:by2, bx1:bx2]):
                rej.append(full_box + ("DESK: NOT PHONE-LIKE",))
                continue
            if tid not in best or conf > best[tid]["confidence"]:
                best[tid] = {"name": "CELL PHONE", "confidence": conf,
                             "center": (ccx, ccy), "bbox": full_box, "weak": False, "hand_sw": _hd}

    hits = entry.setdefault("_desk_hits", {})
    confirmed = entry.setdefault("_wc_confirmed", {})
    for tid, det in best.items():
        recent = [t for t in hits.get(tid, []) if now - t <= DESK_WINDOW_SEC]
        recent.append(now)
        hits[tid] = recent
        if len(recent) >= DESK_MIN_HITS:
            if tid not in confirmed:
                print(f"[{camera_id}] ID {tid}: phone at desk (desk crop, "
                      f"{det['confidence']*100:.0f}%, hand {det.get('hand_sw', 0):.2f} sw)")
            confirmed[tid] = {"det": det, "until": now + DESK_HOLD_SEC}
    for tid in list(hits):
        if tid not in best:
            hits[tid] = [t for t in hits[tid] if now - t <= DESK_WINDOW_SEC]
            if not hits[tid]:
                del hits[tid]


def _merge_wrist_phones(entry, camera_id, clean_frame, students, now, object_detections):
    """Adds confirmed wrist-crop phones to `object_detections` (in place)."""
    if (not WRIST_CROP_ENABLED or not DETECTION_ENABLED["phone"]
            or entry.get("object_model") is None):
        return
    try:
        confirmed = entry.setdefault("_wc_confirmed", {})
        if now - entry.get("_wc_last", 0.0) >= WRIST_CROP_INTERVAL_SEC:
            entry["_wc_last"] = now
            _t0 = time.perf_counter()
            _wrist_crop_sample(entry, camera_id, clean_frame, students, now)
            if DESK_CROP_ENABLED:
                _desk_crop_sample(entry, camera_id, clean_frame, students, now)
            _n = entry["_wc_n"] = entry.get("_wc_n", 0) + 1
            entry["_wc_sum"] = entry.get("_wc_sum", 0.0) + (time.perf_counter() - _t0)
            if _n % 20 == 0:
                print(f"[{camera_id}] wrist pass: {entry['_wc_sum'] / _n * 1000:.0f} ms avg, "
                      f"{entry.get('_wc_ncrops', 0)} crops last pass")

        for tid in [t for t, c in confirmed.items() if c["until"] < now]:
            del confirmed[tid]

        conf_items = list(confirmed.items())
        wc_static = _check_static_objects(
            entry, [c["det"]["center"] for _, c in conf_items], now,
            store="static_wc", static_sec=WRIST_CROP_STATIC_SEC,
            move_px=WRIST_CROP_STATIC_MOVE_PX)
        for k, (tid, c) in enumerate(conf_items):
            if k in wc_static:
                continue
            det = c["det"]
            dup = next((d for d in object_detections
                        if d["name"] == "CELL PHONE"
                        and math.hypot(d["center"][0] - det["center"][0],
                                       d["center"][1] - det["center"][1]) < WRIST_CROP_DEDUP_PX),
                       None)
            if dup is not None:
                dup.setdefault("track_id", tid)
                dup["confidence"] = max(dup["confidence"], det["confidence"])
            else:
                object_detections.append(dict(det, track_id=tid))
    except Exception as e:
        print(f"[{camera_id}] Wrist-crop phone error: {e}")


def _owner_of_detection(det, students, established_ids):
    """track_id of the student a detection belongs to (or None)."""
    if det.get("track_id") is not None:
        return det["track_id"]
    best, best_d = None, float("inf")
    for s in students:
        if s["track_id"] not in established_ids:
            continue
        d = math.hypot(s["shoulder_center"][0] - det["center"][0],
                       s["shoulder_center"][1] - det["center"][1])
        if d < s["shoulder_width"] * ATTRIBUTION_RADIUS_MULTIPLIER and d < best_d:
            best, best_d = s["track_id"], d
    return best


def _local_sw(det, students, fallback):
    """Shoulder width of the student whose wrist is nearest the detection."""
    cx, cy = det["center"]
    best, best_d = fallback, float("inf")
    for s in students:
        for wr in (s["l_wrist"], s["r_wrist"]):
            if wr is None:
                continue
            d = math.hypot(cx - wr[0], cy - wr[1])
            if d < s["shoulder_width"] * HELD_PHONE_WRIST_MULT and d < best_d:
                best, best_d = s["shoulder_width"], d
    return best


def _pen_like(bbox, ref_sw, conf=None):
    """True for long, thin boxes (pens) or boxes far thinner than a phone."""
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    long_side, short_side = max(bw, bh), min(bw, bh)
    if long_side / short_side > PHONE_MAX_ASPECT:
        return True
    if ref_sw is not None:
        if short_side < PHONE_MIN_SHORT_SW * ref_sw:
            return True
        if long_side > PHONE_MAX_LONG_SW * ref_sw:
            return True
        if long_side < PHONE_TINY_SW * ref_sw:
            return True
        if (conf is not None and long_side < PHONE_SMALL_SW * ref_sw
                and conf < PHONE_SMALL_MIN_CONF):
            return True
    return False


def _white_card_frac(frame, bbox):
    """Share of near-white, low-saturation pixels inside bbox."""
    if frame is None or bbox is None:
        return 0.0
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(fw, int(x2)), min(fh, int(y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > BADGE_WHITE_V) & (hsv[..., 1] < BADGE_WHITE_S)
    return float(white.mean())


def _lanyard_above(frame, bbox, s):
    """True if a thin strap runs from the top of `bbox` up toward the neck."""
    if frame is None or bbox is None or s is None:
        return False
    try:
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        sx, sy = s["shoulder_center"]
        sw = float(s["shoulder_width"])
        top = np.array([(x1 + x2) / 2.0, float(y1)])
        neck = np.array([float(sx), float(sy) - 0.1 * sw])
        seg = neck - top
        length = float(np.hypot(seg[0], seg[1]))
        if length < 0.25 * sw:
            return False
        perp = np.array([-seg[1], seg[0]]) / length * (0.14 * sw)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def px(p):
            x = int(min(max(p[0], 1), fw - 2))
            y = int(min(max(p[1], 1), fh - 2))
            return float(gray[y - 1:y + 2, x - 1:x + 2].mean())

        darker = 0
        for t in np.linspace(0.15, 0.85, 8):
            p = top + seg * t
            side = (px(p + perp) + px(p - perp)) / 2.0
            if side - px(p) >= 18:
                darker += 1
        return darker >= 6
    except Exception:
        return False


def _badge_like(center, students, frame=None, bbox=None):
    """True if `center` is inside some student's chest strip without a hand on it."""
    if not BADGE_FILTER_ENABLED:
        return False
    cx, cy = center
    card_like = _white_card_frac(frame, bbox) >= BADGE_WHITE_FRAC
    for s in students:
        scx, scy = s["shoulder_center"]
        sw = s["shoulder_width"]
        if abs(cx - scx) > BADGE_ZONE_HALF_W * sw:
            continue
        if not (scy + BADGE_ZONE_TOP * sw <= cy <= scy + BADGE_ZONE_BOTTOM * sw):
            continue
        if card_like:
            return True
        _both = [wr for wr in (s["l_wrist"], s["r_wrist"])
                 if wr is not None
                 and math.hypot(cx - wr[0], cy - wr[1]) < BADGE_TWO_HAND_MULT * sw]
        if len(_both) == 2:
            return False
        strap = _lanyard_above(frame, bbox, s)
        hand_limit = BADGE_HAND_HOLD_MULT * sw
        for wr in (s["l_wrist"], s["r_wrist"]):
            if wr is not None and math.hypot(cx - wr[0], cy - wr[1]) < hand_limit:
                if strap:
                    return True
                return False
        near_both = [wr for wr in (s["l_wrist"], s["r_wrist"])
                     if wr is not None
                     and math.hypot(cx - wr[0], cy - wr[1]) < BADGE_TWO_HAND_MULT * sw]
        if len(near_both) == 2:
            return False
        return True
    return False


def _phone_in_hand(det, students):
    """True if any student's wrist is near the detection centre."""
    cx, cy = det["center"]
    for s in students:
        lim = s["shoulder_width"] * HELD_PHONE_WRIST_MULT
        for wr in (s["l_wrist"], s["r_wrist"]):
            if wr is not None and math.hypot(cx - wr[0], cy - wr[1]) < lim:
                return True
    return False


def _hands_at_chest(s):
    """True if both wrists are close together at chest height."""
    if s is None:
        return False
    l, r = s["l_wrist"], s["r_wrist"]
    if l is None or r is None:
        return False
    sw = s["shoulder_width"]
    sy = s["shoulder_center"][1]
    if math.hypot(l[0] - r[0], l[1] - r[1]) > WRIST_WEAK_MAX_GAP_SW * sw:
        return False
    return all(WRIST_WEAK_MIN_DROP_SW * sw <= w[1] - sy <= WRIST_WEAK_MAX_DROP_SW * sw
               for w in (l, r))


def _hands_posture(s):
    """Both wrists together and raised at chest height."""
    l, r = s["l_wrist"], s["r_wrist"]
    if l is None or r is None:
        return False
    sw = s["shoulder_width"]
    sy = s["shoulder_center"][1]
    if math.hypot(l[0] - r[0], l[1] - r[1]) > HANDS_POSTURE_MAX_GAP_SW * sw:
        return False
    return all(HANDS_POSTURE_MIN_DROP_SW * sw <= w[1] - sy <= HANDS_POSTURE_MAX_DROP_SW * sw
               for w in (l, r))


def _torso_lock_verdict(entry, s, center, now):
    """'phone' | 'badge' | 'pending' for a candidate inside chest strip."""
    sw = s["shoulder_width"]
    scx, scy = s["shoulder_center"]
    cx, cy = center
    if not (abs(cx - scx) <= BADGE_ZONE_HALF_W * sw
            and scy + BADGE_ZONE_TOP * sw <= cy <= scy + BADGE_ZONE_BOTTOM * sw):
        return "phone"
    wrists = [wr for wr in (s["l_wrist"], s["r_wrist"]) if wr is not None]
    if not wrists:
        return "phone"
    if sum(1 for p in wrists
           if math.hypot(cx - p[0], cy - p[1]) < BADGE_TWO_HAND_MULT * sw) == 2:
        return "phone"
    wr = min(wrists, key=lambda p: math.hypot(cx - p[0], cy - p[1]))

    hist = entry.setdefault("_badge_hist", {}).setdefault(s["track_id"], [])
    if hist and now - hist[-1][0] > 2.5:
        hist.clear()
    if not hist or now - hist[-1][0] >= 0.05:
        hist.append((now,
                     ((cx - scx) / sw, (cy - scy) / sw),
                     ((cx - wr[0]) / sw, (cy - wr[1]) / sw)))
    hist[:] = [h for h in hist if h[0] >= now - BADGE_LOCK_WINDOW_SEC]

    if len(hist) < 4 or hist[-1][0] - hist[0][0] < BADGE_LOCK_MIN_SEC:
        return "pending"
    r = np.array([h[1] for h in hist])
    d = np.array([h[2] for h in hist])
    std_r = float(math.hypot(r[:, 0].std(), r[:, 1].std()))
    std_d = float(math.hypot(d[:, 0].std(), d[:, 1].std()))
    if std_r < BADGE_LOCK_STD and std_d > max(BADGE_LOCK_MIN_REL, BADGE_LOCK_RATIO * std_r):
        return "badge"
    return "phone"


def _update_hand_rise(entry, students, now):
    """Remember when each student's hand was last low or out of sight."""
    if not PHONE_NEEDS_HAND_RISE:
        return
    st = entry.setdefault("_hand_low", {})
    for s in students:
        sw = s["shoulder_width"]
        sy = s["shoulder_center"][1]
        low = any(w is None or (w[1] - sy) >= HAND_LOW_SW * sw
                  for w in (s["l_wrist"], s["r_wrist"]))
        d = st.setdefault(s["track_id"], {"since": None, "last": -1e9})
        if low:
            if d["since"] is None:
                d["since"] = now
            if now - d["since"] >= HAND_LOW_MIN_SEC:
                d["last"] = now
        else:
            d["since"] = None


def _hand_rise_ok(entry, tid, now):
    """True if this student's phone is allowed to count right now."""
    if not PHONE_NEEDS_HAND_RISE:
        return True
    if now - entry.get("_phone_ok", {}).get(tid, -1e9) <= HAND_RISE_KEEP_SEC:
        return True
    d = entry.get("_hand_low", {}).get(tid)
    return d is not None and now - d["last"] <= HAND_RISE_MEMORY_SEC


def _select_unique_people(candidates, track_first_seen, current_time):
    """Drop ghost skeletons when two poses overlap closely."""
    ordered = sorted(
        candidates,
        key=lambda c: (track_first_seen.get(c["track_id"], current_time), -c["conf"])
    )
    kept = []
    for c in ordered:
        is_dup = False
        for k in kept:
            d = math.hypot(c["center"][0] - k["center"][0],
                           c["center"][1] - k["center"][1])
            limit = min(c["width"], k["width"]) * DUPLICATE_DIST_MULTIPLIER
            if d < limit:
                is_dup = True
                break
        if not is_dup:
            kept.append(c)
    return {c["idx"] for c in kept}


def _mid_band_wrist(a, b):
    """A wrist of student `a` out in the middle band of the gap toward `b`."""
    ax, ay = a["shoulder_center"]
    bx, _by = b["shoulder_center"]
    gap = bx - ax
    if abs(gap) < 1:
        return None
    sw = a["shoulder_width"]
    for wrist in (a["l_wrist"], a["r_wrist"]):
        if wrist is None:
            continue
        p = (wrist[0] - ax) / gap
        if (REACH_MID_MIN_PROGRESS <= p <= REACH_MID_MAX_PROGRESS
                and math.hypot(wrist[0] - ax, wrist[1] - ay) / sw >= REACH_MID_MIN_EXTENSION
                and abs(wrist[1] - ay) <= REACH_MID_MAX_DY * sw):
            return wrist
    return None


def _find_reaching_wrist(reacher, target, same_row):
    """Returns the wrist (x, y) of `reacher` reaching toward `target`, or None."""
    rx, ry = reacher["shoulder_center"]
    tx, ty = target["shoulder_center"]
    sw     = reacher["shoulder_width"]
    gap_x  = tx - rx

    if same_row and abs(gap_x) >= 1:
        mine = _mid_band_wrist(reacher, target)
        if mine is not None and _mid_band_wrist(target, reacher) is not None:
            return mine
    for wrist in (reacher["l_wrist"], reacher["r_wrist"]):
        if wrist is None:
            continue
        d_target = math.hypot(wrist[0] - tx, wrist[1] - ty)
        d_self   = math.hypot(wrist[0] - rx, wrist[1] - ry)
        if (d_target < target["shoulder_width"] * REACH_ZONE_RADIUS_MULTIPLIER
                and d_target < d_self
                and d_self >= sw * REACH_ZONE_MIN_EXTENSION):
            return wrist
        if not same_row or abs(gap_x) < 1:
            continue
        extension = math.hypot(wrist[0] - rx, wrist[1] - ry) / sw
        if extension < REACH_EXTENSION_MULTIPLIER:
            continue
        if abs(wrist[1] - ry) > 2.5 * sw:
            continue
        progress = (wrist[0] - rx) / gap_x
        if progress >= REACH_PROGRESS_MIN:
            return wrist
    return None


def _check_static_objects(entry, centers, now, store="static_phones",
                          static_sec=STATIC_PHONE_SEC,
                          move_px=STATIC_OBJECT_MOVE_PX):
    """Return indices into `centers` suppressed because the object has sat still."""
    tracker  = entry.setdefault(store, {})
    NEXT_KEY = "__next"
    next_id  = tracker.get(NEXT_KEY, 0)

    stale = [k for k in list(tracker)
             if k != NEXT_KEY and now - tracker[k]["last_seen"] > 3.0]
    for k in stale:
        del tracker[k]

    suppressed = set()
    for idx, center in enumerate(centers):
        best_id, best_dist = None, float("inf")
        for tid, td in tracker.items():
            if tid == NEXT_KEY:
                continue
            d = math.hypot(center[0] - td["center"][0], center[1] - td["center"][1])
            if d < STATIC_OBJECT_RADIUS_PX and d < best_dist:
                best_id, best_dist = tid, d

        if best_id is not None:
            td = tracker[best_id]
            td["last_seen"] = now
            if best_dist > move_px:
                td["static_since"] = now
                td["center"]       = center
            elif now - td["static_since"] >= static_sec:
                suppressed.add(idx)
        else:
            tracker[next_id] = {"center": center, "static_since": now, "last_seen": now}
            next_id += 1
            tracker[NEXT_KEY] = next_id

    return suppressed

