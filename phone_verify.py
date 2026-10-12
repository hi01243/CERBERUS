"""
phone_verify.py -- Stage 2 phone verification for CERBERUS (r47)

Stage 1 (app.py + best.pt) only finds CANDIDATES.  This file looks at the pixels of
each candidate over about a second and decides whether it behaves like a phone.
No neural network: only OpenCV + numpy.  It never imports app.py.

Verdicts
  CONFIRMED  positive evidence of a phone (screen activity + shape/brightness)
  UNKNOWN    not enough evidence either way (still collecting, too small, screen off ...)
  REJECTED   clearly the wrong shape (pen, square-ish notepad ...)

What it measures (all on the clean camera frame, never on the annotated one)
  1. screen activity : while the object is held still, does the inside of it keep
                       changing in a way the hand motion does not explain?
                       (a lit screen shows scrolling / video / UI; paper and badges do not)
  2. shape           : contour -> rotated rectangle -> aspect ratio + how rectangular.
                       Uses the pixels, not the YOLO box, so a tilted pen cannot hide.
  3. brightness      : inside of the object vs the area around it (weak evidence).

ALL numbers below are PLACEHOLDERS.  They have not been measured on a real classroom
camera.  Collect real crops (PHONE_VERIFY_DUMP = True in app.py), sort them into
phone/ and other/ folders, then run:

    python phone_verify.py --probe phone_candidates_sorted

to see what the numbers really look like, and edit CFG from that.
"""
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

CFG = {
    # --- candidate bookkeeping ---
    "ASSOC_MULT":       1.5,    # same candidate if the centre moved less than this x box long side (+ pad)
    "ASSOC_PAD_PX":     20,
    "STALE_SEC":        3.0,    # forget a candidate not seen this long
    "MAX_CANDIDATES":   64,
    "SAMPLE_EVERY_SEC": 0.2,    # take at most one sample this often per candidate
    "MAX_SAMPLES":      10,
    "MIN_SAMPLES":      4,      # need this many samples ...
    "MIN_SPAN_SEC":     0.9,    # ... spread over at least this long before judging
    "MIN_BOX_PX":       8,      # ignore boxes smaller than this
    "PATCH_MARGIN":     0.15,   # extra background around the box, x its long side
    # --- shape ---
    "GEOM_MIN_SHORT_PX": 3,     # object short side (px) needed to trust the shape ...
    "GEOM_MIN_LONG_PX":  24,    # ... and long side
    "ASPECT_MIN":       1.40,   # rotated-rectangle long/short outside [MIN, MAX] => REJECTED
    "ASPECT_MAX":       3.00,
    "RECT_MIN":         0.78,   # contour area / rotated-rectangle area (shape "pass" needs this)
    "GEOM_MIN_VALID":   3,      # samples with a usable shape before the shape is trusted
    # --- screen activity ---
    "STILL_MAX_MOVE":   0.25,   # "held still" = centre moved less than this x long side between samples
    "CHANGE_MIN_PAIRS": 2,      # still pairs needed to judge activity
    "CHANGE_SCREEN":    0.08,   # median relative change at/above this = active screen
    # --- brightness (weak) ---
    "EMIT_MIN":         0.25,   # inside P90 brightness minus surroundings median, as fraction of 255
    # --- scoring ---
    "W_CHANGE": 0.55, "W_SHAPE": 0.25, "W_EMIT": 0.20,
    "CONFIRM_SCORE":    0.70,
    "HOLD_SEC":         3.0,    # stay CONFIRMED this long after the last positive evaluation
    # --- dumping crops for calibration ---
    "DUMP_MAX":         300,
}

VERDICT_COLORS = {              # BGR, for the on-screen debug boxes
    "CONFIRMED": (0, 0, 255),
    "UNKNOWN":   (0, 165, 255),
    "REJECTED":  (120, 120, 120),
    "COLLECTING": (255, 255, 0),
}

_HANN = None


def _hann():
    global _HANN
    if _HANN is None:
        _HANN = cv2.createHanningWindow((32, 32), cv2.CV_32F)
    return _HANN


# ----------------------------------------------------------------------
# per-sample measurements
# ----------------------------------------------------------------------
def _geometry(big_bgr):
    """Rotated-rectangle shape of the main object in a patch that has background
    around it.  Returns dict(aspect, rect, short_px, valid) or None.
    The object = everything that differs from the background colour measured on
    the patch border (so dark body + bright screen are both 'object')."""
    h, w = big_bgr.shape[:2]
    if h < 8 or w < 8:
        return None
    img = cv2.GaussianBlur(big_bgr, (3, 3), 0).astype(np.int16)
    b = max(1, int(min(h, w) * 0.06))
    ring = np.concatenate([img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
                           img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    dist = np.abs(img - bg).max(axis=2).astype(np.uint8)
    t, _ = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m = cv2.threshold(dist, max(20.0, t), 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 0.05 * h * w:
        return None
    (_cx, _cy), (rw, rh), _ang = cv2.minAreaRect(c)
    long_s, short_s = max(rw, rh), min(rw, rh)
    if short_s < 1.0:
        return None
    return {
        "aspect":   float(long_s / short_s),
        "rect":     float(area / (rw * rh)),
        "short_px": float(short_s),
        "valid":    bool(short_s >= CFG["GEOM_MIN_SHORT_PX"] and long_s >= CFG["GEOM_MIN_LONG_PX"]
                     and area <= 0.92 * h * w),
    }


def analyse_patch(big_bgr, box, t, cx, cy, size):
    """big_bgr: crop of the frame = detection box + margin.
    box: (x, y, w, h) of the detection box INSIDE big_bgr.  Returns a sample dict."""
    gx, gy, bw, bh = box
    gray_big = cv2.cvtColor(big_bgr, cv2.COLOR_BGR2GRAY)
    gray_box = gray_big[gy:gy + bh, gx:gx + bw]
    if gray_box.size == 0:
        return None

    patch = cv2.resize(gray_box.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
    patch /= max(float(patch.mean()), 20.0)     # divide out exposure: we compare RELATIVE change

    mx, my = int(bw * 0.15), int(bh * 0.15)
    inner = gray_box[my:bh - my, mx:bw - mx]
    if inner.size == 0:
        inner = gray_box
    ring_mask = np.ones(gray_big.shape, dtype=bool)
    ring_mask[gy:gy + bh, gx:gx + bw] = False
    ring = gray_big[ring_mask]
    ring_med = float(np.median(ring)) if ring.size >= 20 else float(np.median(gray_box))
    lum_delta = (float(np.percentile(inner, 90)) - ring_med) / 255.0

    return {
        "t": float(t), "cx": float(cx), "cy": float(cy), "size": float(max(bw, bh)),
        "patch": patch, "res": None,
        "lum_delta": lum_delta,
        "geom": _geometry(big_bgr),
        "short_px": float(min(bw, bh)),
    }


def make_sample(frame, bbox, now, keep_bgr=False):
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < CFG["MIN_BOX_PX"] or bh < CFG["MIN_BOX_PX"]:
        return None
    m = int(round(CFG["PATCH_MARGIN"] * max(bw, bh)))
    X1, Y1, X2, Y2 = max(0, x1 - m), max(0, y1 - m), min(fw, x2 + m), min(fh, y2 + m)
    big = frame[Y1:Y2, X1:X2]
    box = (x1 - X1, y1 - Y1, bw, bh)
    s = analyse_patch(big, box, now, (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(bw, bh))
    if s is not None and keep_bgr:
        s["bgr"] = big.copy()
        s["box"] = box
    return s


def _residual(a, b):
    """Relative change between two exposure-normalised patches after removing the
    shift between them (so a hand that only moved the object does not count).
    The smaller of 'as is' and 'shifted back' is used: phase correlation can
    return nonsense on flat patches, and a still object should not need a big shift."""
    inner = (slice(6, 26), slice(6, 26))
    plain = float(np.abs(a - b)[inner].mean())
    # phaseCorrelate edits its inputs (applies the window in place): give it copies
    (dx, dy), _resp = cv2.phaseCorrelate(a.copy(), b.copy(), _hann())
    if abs(dx) > 3.0 or abs(dy) > 3.0:
        return plain
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    b2 = cv2.warpAffine(b, M, (32, 32), borderMode=cv2.BORDER_REPLICATE)
    return min(plain, float(np.abs(a - b2)[inner].mean()))


def _add_sample(cand, s):
    S = cand["samples"]
    if S:
        p = S[-1]
        lim = CFG["STILL_MAX_MOVE"] * max(p["size"], s["size"])
        ratio = s["size"] / max(1.0, p["size"])
        if math.hypot(s["cx"] - p["cx"], s["cy"] - p["cy"]) <= lim and 0.75 <= ratio <= 1.33:
            s["res"] = _residual(p["patch"], s["patch"])
    S.append(s)


# ----------------------------------------------------------------------
# the decision
# ----------------------------------------------------------------------
def _med(v):
    return float(np.median(v)) if v else None


def features(cand):
    S = list(cand["samples"])
    geoms = [s["geom"] for s in S if s["geom"] and s["geom"]["valid"]]
    res = [s["res"] for s in S if s["res"] is not None]
    return {
        "n": len(S),
        "span": (S[-1]["t"] - S[0]["t"]) if S else 0.0,
        "short_px": _med([s["short_px"] for s in S]),
        "aspect": _med([g["aspect"] for g in geoms]) if len(geoms) >= CFG["GEOM_MIN_VALID"] else None,
        "rect": _med([g["rect"] for g in geoms]) if len(geoms) >= CFG["GEOM_MIN_VALID"] else None,
        "change": _med(res) if len(res) >= CFG["CHANGE_MIN_PAIRS"] else None,
        "lum": _med([s["lum_delta"] for s in S]),
    }


def _fmt(v, f):
    return "-" if v is None else f.format(v)


def verify(cand, now):
    """Returns dict(verdict, score, why, label)."""
    f = features(cand)
    held = now <= cand.get("hold_until", -1e9)
    why = (f"chg {_fmt(f['change'], '{:.3f}')} asp {_fmt(f['aspect'], '{:.2f}')} "
           f"rect {_fmt(f['rect'], '{:.2f}')} px {_fmt(f['short_px'], '{:.0f}')} "
           f"lum {_fmt(f['lum'], '{:+.2f}')}")
    short = (f"chg {_fmt(f['change'], '{:.2f}')} a {_fmt(f['aspect'], '{:.1f}')} "
             f"px {_fmt(f['short_px'], '{:.0f}')}")

    if f["n"] < CFG["MIN_SAMPLES"] or f["span"] < CFG["MIN_SPAN_SEC"]:
        return {"verdict": "UNKNOWN", "score": 0.0, "why": "collecting " + why,
                "label": "collecting", "phase": "COLLECTING"}

    if f["aspect"] is not None and not (CFG["ASPECT_MIN"] <= f["aspect"] <= CFG["ASPECT_MAX"]):
        cand["hold_until"] = -1e9
        return {"verdict": "REJECTED", "score": 0.0, "why": "shape " + why,
                "label": "REJECTED " + short, "phase": "REJECTED"}

    score = 0.0
    if f["change"] is not None and f["change"] >= CFG["CHANGE_SCREEN"]:
        score += CFG["W_CHANGE"]
    shape_pass = (f["aspect"] is not None and f["rect"] is not None
                  and f["rect"] >= CFG["RECT_MIN"])
    if shape_pass:
        score += CFG["W_SHAPE"]
    if f["lum"] is not None and f["lum"] >= CFG["EMIT_MIN"]:
        score += CFG["W_EMIT"]

    if score >= CFG["CONFIRM_SCORE"]:
        cand["hold_until"] = now + CFG["HOLD_SEC"]
        verdict = "CONFIRMED"
    elif held:
        verdict = "CONFIRMED"
        why = "hold " + why
    else:
        verdict = "UNKNOWN"
    return {"verdict": verdict, "score": round(score, 2), "why": why,
            "label": f"{verdict} {score:.2f} {short}", "phase": verdict}


# ----------------------------------------------------------------------
# candidate store (one per camera)
# ----------------------------------------------------------------------
class CandidateStore:
    def __init__(self, dump_dir=None, tag="cam"):
        self.cands = {}
        self.next_id = 1
        self.dump_dir = dump_dir
        self.tag = tag
        self.dumped = 0

    def _prune(self, now):
        for k in [k for k, c in self.cands.items() if now - c["last_seen"] > CFG["STALE_SEC"]]:
            c = self.cands.pop(k)
            # a candidate seen only briefly never reaches the "mature" dump below: save it now
            if (self.dump_dir and not c["dumped"] and len(c["samples"]) >= 2
                    and self.dumped < CFG["DUMP_MAX"]):
                self._dump(c)
        if len(self.cands) > CFG["MAX_CANDIDATES"]:
            oldest = sorted(self.cands.values(), key=lambda c: c["last_seen"])
            for c in oldest[:len(self.cands) - CFG["MAX_CANDIDATES"]]:
                self.cands.pop(c["id"], None)

    def _associate(self, cx, cy, size):
        best, best_d = None, float("inf")
        for c in self.cands.values():
            lim = CFG["ASSOC_MULT"] * max(size, c["size"]) + CFG["ASSOC_PAD_PX"]
            d = math.hypot(cx - c["center"][0], cy - c["center"][1])
            if d <= lim and d < best_d:
                best, best_d = c, d
        return best

    def feed(self, frame, bbox, conf, now, owner=None):
        """Show the store one Stage-1 detection (box in full-frame pixels).
        Returns the current verdict dict for that candidate (+ 'id').
        Never raises: a verifier problem must not stop the detection loop."""
        try:
            return self._feed(frame, bbox, conf, now, owner)
        except Exception as e:      # pragma: no cover - safety net
            print(f"[{self.tag}] phone_verify error: {e}")
            return {"verdict": "UNKNOWN", "score": 0.0, "why": f"error {e}",
                    "label": "error", "phase": "UNKNOWN", "id": -1}

    def _feed(self, frame, bbox, conf, now, owner):
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx, cy, size = (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1)
        self._prune(now)
        cand = self._associate(cx, cy, size)
        if cand is None:
            cand = {"id": self.next_id, "samples": deque(maxlen=CFG["MAX_SAMPLES"]),
                    "last_sample": -1e9, "hold_until": -1e9, "last_verdict": None,
                    "dumped": False, "result": None}
            self.cands[cand["id"]] = cand
            self.next_id += 1
        cand.update(last_seen=now, center=(cx, cy), size=size, bbox=(x1, y1, x2, y2),
                    conf=conf, owner=owner)

        if now - cand["last_sample"] >= CFG["SAMPLE_EVERY_SEC"]:
            s = make_sample(frame, bbox, now, keep_bgr=self.dump_dir is not None)
            if s is not None:
                _add_sample(cand, s)
                cand["last_sample"] = now

        res = verify(cand, now)
        cand["result"] = res
        old = cand["last_verdict"]
        if res["verdict"] != old:
            if res["verdict"] in ("CONFIRMED", "REJECTED") or old == "CONFIRMED":
                print(f"[{self.tag}] PV c{cand['id']}: {old or '-'} -> {res['verdict']}  ({res['why']})")
            cand["last_verdict"] = res["verdict"]
        if (self.dump_dir and not cand["dumped"]
                and res["phase"] != "COLLECTING" and self.dumped < CFG["DUMP_MAX"]):
            self._dump(cand)
        out = dict(res)
        out["id"] = cand["id"]
        return out

    def overlay(self, now):
        """(x1, y1, x2, y2, label, BGR colour) for every candidate seen in the last 0.6 s."""
        items = []
        for c in self.cands.values():
            if now - c["last_seen"] > 0.6 or not c.get("result"):
                continue
            r = c["result"]
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            items.append((x1, y1, x2, y2, f"PV{c['id']} {r['label']}",
                          VERDICT_COLORS.get(r["phase"], (200, 200, 200))))
        return items

    def _dump(self, cand):
        cand["dumped"] = True
        S = [s for s in cand["samples"] if "bgr" in s]
        if not S:
            return
        folder = os.path.join(self.dump_dir, f"{self.tag}_c{cand['id']}_{int(time.time())}")
        os.makedirs(folder, exist_ok=True)
        meta = []
        for i, s in enumerate(S):
            cv2.imwrite(os.path.join(folder, f"s{i:02d}.png"), s["bgr"])
            meta.append({"file": f"s{i:02d}.png", "t": s["t"], "cx": s["cx"], "cy": s["cy"],
                         "size": s["size"], "box": list(s["box"])})
        with open(os.path.join(folder, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        self.dumped += 1
        print(f"[{self.tag}] PV c{cand['id']}: saved {len(S)} crops to {folder}")


def store_for(entry, dump_dir, tag):
    """The camera's store, created on first use (entry is the camera registry dict)."""
    st = entry.get("_pv")
    if st is None:
        st = entry["_pv"] = CandidateStore(dump_dir, tag)
    return st


# ----------------------------------------------------------------------
# calibration helper:  python phone_verify.py --probe <folder>
# <folder>/phone/<candidate dirs>   and   <folder>/other/<candidate dirs>
# (candidate dirs are the ones the app saved in phone_candidates/)
# ----------------------------------------------------------------------
def _load_candidate(folder):
    with open(os.path.join(folder, "meta.json")) as fh:
        meta = json.load(fh)
    cand = {"samples": deque(maxlen=CFG["MAX_SAMPLES"]), "hold_until": -1e9}
    for m in meta:
        img = cv2.imread(os.path.join(folder, m["file"]))
        if img is None:
            continue
        s = analyse_patch(img, tuple(m["box"]), m["t"], m["cx"], m["cy"], m["size"])
        if s is not None:
            _add_sample(cand, s)
    return cand


def probe(root):
    groups = {}
    for g in sorted(os.listdir(root)):
        gp = os.path.join(root, g)
        if os.path.isdir(gp):
            groups[g] = [os.path.join(gp, d) for d in sorted(os.listdir(gp))
                         if os.path.isfile(os.path.join(gp, d, "meta.json"))]
    if not groups:
        print("no sub-folders found (expected phone/ and other/)")
        return
    for g, dirs in groups.items():
        print(f"\n=== {g}: {len(dirs)} candidates ===")
        counts = {}
        rows = []
        for d in dirs:
            c = _load_candidate(d)
            if len(c["samples"]) == 0:
                continue
            now = c["samples"][-1]["t"]
            r = verify(c, now)
            f = features(c)
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
            rows.append(f)
            print(f"  {os.path.basename(d):32s} {r['verdict']:9s} {r['why']}")
        print("  verdicts:", counts)
        for k in ("change", "aspect", "rect", "short_px", "lum"):
            v = [r[k] for r in rows if r[k] is not None]
            if v:
                print(f"  {k:9s} min {min(v):7.3f}  median {float(np.median(v)):7.3f}  max {max(v):7.3f}   (n={len(v)})")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        probe(sys.argv[2])
    else:
        print(__doc__)