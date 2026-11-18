"""Badge + QR scanner app, served as a local web page.

    python web_ui.py                     # live D435i, open http://localhost:8000
    python web_ui.py --replay DIR        # replay a folder of photos instead of the camera
    python web_ui.py --host 0.0.0.0      # also reachable from other devices on the network

Everything runs locally (Python standard library + OpenCV); no internet needed.
"""

import argparse
import json
import os
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# OpenCV would spread small per-frame jobs (resize, colour conversion, warps) over every core, where the
# hand-off costs more than the work; a few threads are as fast.
cv2.setNumThreads(int(os.environ.get("OPENCV_THREADS", "4")))

from badge_detector import BadgeDetector
from badge_scanner import load_badge_db
from badge_tracker import BadgeTracker
from auto_tune import AutoTuner, CameraIO
from test_ui import LiveSource, ReplaySource, WebcamSource, process
try:
    from qr_soft import SoftWorker          # model-based fallback decoder for small/blurry QR codes
except Exception as _e:                     # (needs scipy + reedsolo)
    SoftWorker = None
    print("deep QR decoder disabled:", _e)
try:
    import qr_depth                          # depth mask + burst super-resolution for range (numpy/cv2 only)
    import qr_burst                          # joint multi-frame decoder for still, far badges
except Exception as _e:
    qr_depth = None
    print("depth QR helpers disabled:", _e)

from person_tracker import PersonTracker
try:
    from name_reader import NameService      # reads the printed name off the badge (offline OCR)
except Exception as _e:                      # (needs rapidocr + onnxruntime)
    NameService = None
    print("badge name reading disabled:", _e)
from free_qr import FreeQRScanner
from object_qr import ObjectQR

HERE = os.path.dirname(os.path.abspath(__file__))
YELLOW, GREEN, GREY, CYAN, MAGENTA = (0, 220, 255), (0, 210, 0), (160, 160, 160), (255, 200, 0), (255, 80, 255)
WHITE = (255, 255, 255)
OBJ_MODE = "OBJECT -> QR"
STATE_COLORS = {"ENGAGED": (255, 160, 0), "APPROACHING": (255, 220, 120), "PRESENT": (200, 200, 200),
                "FAR": (120, 120, 120), "IDENTIFIED": (0, 210, 0), "LEAVING": (0, 140, 0)}


def _badge_at(badges, where):
    """The badge quad containing `where` (QR corners 4x2 or a box x1,y1,x2,y2), else None."""
    if where is None or not badges:
        return None
    w = np.asarray(where, np.float32).reshape(-1)
    cx, cy = ((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) if w.size == 4 else \
        tuple(np.asarray(where, np.float32).reshape(-1, 2).mean(axis=0))
    for q, *_ in badges:
        if q[:, 0].min() <= cx <= q[:, 0].max() and q[:, 1].min() <= cy <= q[:, 1].max():
            return q
    return None


def draw(color, res):
    view = color.copy()
    s = max(1, view.shape[1] // 960)
    for p in res.get("people", []):
        x1, y1, x2, y2 = p["box"]
        col = STATE_COLORS.get(p["state"], GREY)
        cv2.rectangle(view, (x1, y1), (x2, y2), col, 2 * s)
        lab = f"P{p['id']} {p['state']}" + (f" {p['dist']:.1f}m" if p["dist"] else "")
        if p.get("name"):
            lab += f" {p['name']}"
        if p["identity"]:
            lab += f" {p['identity']}" + (" (back)" if p["returning"] else "")
        cv2.putText(view, lab, (x1 + 6, max(y1 + 26 * s, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, col, 2 * s, cv2.LINE_AA)
    for quad, why in res["rejected"]:
        cv2.polylines(view, [quad.astype(np.int32)], True, GREY, 2 * s)
        x, y = quad[0].astype(int)
        cv2.putText(view, why, (x, max(y - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, GREY, s, cv2.LINE_AA)
    names = res.get("names") or [None] * len(res["badges"])
    for (quad, dist, lmm, smm), name in zip(res["badges"], names):
        cv2.polylines(view, [quad.astype(np.int32)], True, YELLOW, 3 * s)
        x, y = quad[0].astype(int)
        lab = (name if name else "BADGE") + (f" {dist:.2f}m" if dist else "")
        cv2.putText(view, lab, (x, max(y - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s,
                    WHITE if name else YELLOW, 2 * s, cv2.LINE_AA)
    if res["qr"]:
        # show the code whenever it is known: exact QR corners when this frame's fast read found them,
        # otherwise (code remembered by the tracker, or read by the soft / multi-frame decoder) the badge
        # the code belongs to
        text, pts, _ = res["qr"]
        owner = _badge_at(res["badges"], pts if pts is not None else res.get("qr_box"))
        target = pts if pts is not None else owner
        if target is not None:
            cv2.polylines(view, [np.asarray(target).astype(np.int32)], True, GREEN, 3 * s)
            if pts is not None and owner is not None:
                cv2.polylines(view, [owner.astype(np.int32)], True, GREEN, 2 * s)
            bx = owner if owner is not None else np.asarray(target)
            x, y = int(bx[:, 0].min()), int(bx[:, 1].max())
            lab = "QR " + (text if len(text) <= 32 else text[:30] + "..")
            cv2.putText(view, lab, (x, min(y + 30 * s, view.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8 * s, GREEN, 2 * s, cv2.LINE_AA)
    for o in res.get("objects", []):                       # 'Other objects' mode
        x1, y1, x2, y2 = (int(v) for v in o["box"])
        cv2.rectangle(view, (x1, y1), (x2, y2), MAGENTA, 3 * s)
        lab = f"OBJECT ({o['label']}?)" + (f" {o['dist']:.2f}m" if o["dist"] else "")
        cv2.putText(view, lab, (x1 + 4, max(y1 - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, MAGENTA, 2 * s, cv2.LINE_AA)
    for text, quad, dist, _ in res.get("codes", []):
        cv2.polylines(view, [quad.astype(np.int32)], True, GREEN, 3 * s)
        x, y = quad.min(axis=0).astype(int)
        lab = "QR " + (text if len(text) <= 32 else text[:30] + "..")
        cv2.putText(view, lab, (x, int(quad[:, 1].max()) + 28 * s), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, GREEN, 2 * s, cv2.LINE_AA)
    for text, quad, dist in res.get("other_qr", []):      # QR codes that are not on a badge
        cv2.polylines(view, [quad.astype(np.int32)], True, CYAN, 3 * s)
        x, y = quad.min(axis=0).astype(int)
        lab = "QR " + (text if len(text) <= 32 else text[:30] + "..") + (f" {dist:.2f}m" if dist else "")
        cv2.putText(view, lab, (x, max(y - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, CYAN, 2 * s, cv2.LINE_AA)
    return view


def jpeg(img, width, q=80):
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


class Engine:
    def __init__(self, src, db, qr_pattern=None, autotune=True, soft_qr=True, mains_hz=0, people=True,
                 any_qr=True, remember_codes=True, names=True):
        self.src, self.db = src, db
        self.qr_pattern = re.compile(qr_pattern) if qr_pattern else None
        self.lock = threading.Lock()
        self.mode, self.max_dist, self.paused = "BADGE -> QR", 1.2, False
        self.jpg = {"color": None, "depth": None, "crop": None}
        self.jpg_id = {}                                      # per stream: bumped when a new image is ready
        self.viewers = {"color": 0, "depth": 0, "crop": 0}
        self.timing = {}                                     # stage -> ms (EMA), for the performance readout   # active stream viewers, for on-demand encoding
        self.frame_id = 0
        self.color = self.depth = None
        self.res = None
        self.stats = dict(frames=0, badge=0, qr=0, qr_raw=0, qr_soft=0, qr_burst=0, qr_multi=0, fps=0.0, ms=0.0)
        self.soft = SoftWorker() if (soft_qr and SoftWorker is not None) else None
        # still-badge fallback: joint decoding of 8 frames, in the soft decoder's process pool
        self.multi = qr_burst.BurstCollector(self.soft) if (self.soft is not None and qr_depth is not None) else None
        self._soft_buf = []
        self.soft_after_s = 1.0              # zxing must have failed this long before the soft decoder runs
        # people: who is here / approaching / at the robot; their badge and identity
        self.person_det = BadgeDetector.load(os.path.join(HERE, "models", "person.onnx"), conf=0.25, classes=(0,)) \
            if people else None
        self.people = PersonTracker() if self.person_det is not None else None
        # any QR code in the frame (boxes, signs, phones): shown + logged, never a person's identity
        self.free_qr = FreeQRScanner(hold_s=0.8) if any_qr else None
        self._badge_codes = {}
        # printed name on the badge (OCR in a background thread; GPU if available)
        self.names = None
        if names and NameService is not None:
            try:
                self.names = NameService()
                print(f"badge names: OCR on {self.names.reader.device}")
            except Exception as e:
                print("badge name reading disabled:", e)
        self.objects = None                                 # 'Other objects' mode, loaded on first use
        self.obj_tracker = BadgeTracker(hold_s=1.5, confirm=2, persist=1)   # votes, no badge-code rules                              # badge code -> last time read on a badge
        self.log = deque(maxlen=40)
        self.last_logged = {}
        self.error = None
        self.det_lock = threading.Lock()
        self.detector = None
        self.reload_model()
        validator = (lambda t: bool(self.qr_pattern.fullmatch(t))) if self.qr_pattern else None
        # remember mode (default): a badge's code keeps showing while that badge stays in view;
        # --live-codes shows a code only up to 1 s after it was really read
        self.tracker = BadgeTracker(hold_s=2.0, confirm=2, persist=2, validator=validator,
                                    live_s=None if remember_codes else 1.0)
        # camera auto-tune: exposure/gain/sharpness follow the badge (live camera only)
        self.tuner = AutoTuner(CameraIO(src.color_sensor), mains_hz=mains_hz) if hasattr(src, "color_sensor") else None
        if self.tuner is not None and autotune:
            self.tuner.enable()
        if hasattr(src, "on_restart") and self.tuner is not None:
            # camera stream restarted after a stall: point auto-tune at the new sensor, re-apply it
            def _after_restart(sensor):
                self.tuner.cam.s = sensor
                if self.tuner.enabled:
                    self.tuner.enable()
                if self.tuner.sharp_cur is not None:     # the reloaded settings reset sharpness: re-apply the
                    try:                                  # value auto-tune is crediting its read rate to
                        self.tuner.cam.set("sharpness", self.tuner.sharp_cur)
                    except Exception:
                        pass
            src.on_restart = _after_restart

    def reload_model(self):
        try:
            det = BadgeDetector.load()
        except Exception as e:
            det, self.error = None, f"model load failed: {e}"
            self._error_t = float("inf")                 # permanent: never auto-cleared
        with self.det_lock:
            self.detector = det
        return det is not None

    def _tm(self, name, t):
        """Per-stage timing (EMA, ms) for the performance readout; returns now."""
        now = time.perf_counter()
        ms = (now - t) * 1000
        old = self.timing.get(name)
        self.timing[name] = ms if old is None else 0.9 * old + 0.1 * ms
        return now

    def _multi_inject(self, res):
        """A finished multi-frame read counts as this frame's read for the badge still in its window
        (it is one vote; the badge tracker still needs 2 agreeing reads before trusting it)."""
        if self.multi is None or res["qr"] is not None or not res["badges"]:
            return
        for text, win, _ in self.multi.poll():
            for q, d, _, _ in res["badges"]:
                cx, cy = (q[:, 0].min() + q[:, 0].max()) / 2, (q[:, 1].min() + q[:, 1].max()) / 2
                if win[0] <= cx <= win[2] and win[1] <= cy <= win[3]:
                    res["qr"], res["qr_multi"] = (text, q.astype(np.float32), d), True
                    self.stats["qr_multi"] += 1
                    return

    def _multi_feed(self, res, color):
        """Feed the priority badge to the burst collector (it starts only on a still, unread badge >0.55 m)."""
        if self.multi is None:
            return
        b = res["badges"][0] if res["badges"] else None
        box = dist = gray = None
        if b is not None:
            q = b[0]
            box, dist = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()), b[1]
            if dist is not None and dist >= self.multi.min_dist:
                gray = color                          # the collector converts only its window to grey
        m = getattr(self.src, "meter", None)
        mv = m.get("badge") if m is not None else None
        still = mv["still"] if mv else m is None          # no motion meter (replay): treat as still
        if gray is None:
            self.multi.step(np.zeros((1, 1), np.uint8), None, None, False, bool(res["qr"]))
        else:
            self.multi.step(gray, box, dist, still, bool(res["qr"]))

    def _soft_step(self, res, depth=None):
        """Hand a failed badge crop to the slow decoder; inject its answer when it arrives.
        When the badge is still, first try a real-time burst-super-resolution read of the recent crops."""
        # best-frame selection: keep the last ~1 s of failed crops with a sharpness score and give
        # the (slow) decoder the sharpest one each time a worker is free, not just the newest
        now = time.time()
        if res["qr"] is None and res.get("qr_try"):
            crop, box, dist = res["qr_try"]
            g = crop[:, :, 1] if crop.ndim == 3 else crop
            sharp = float(cv2.Laplacian(g, cv2.CV_32F).var()) if g.size else 0.0
            glare = float((g >= 250).mean()) if g.size else 0.0
            score = sharp * max(0.1, 1 - 10 * glare)          # a glare spot on the code costs more than blur
            self._soft_buf.append((score, now, crop, box, g, dist))
        self._soft_buf = [c for c in self._soft_buf if now - c[1] < 1.0]
        # --- burst super-resolution: when the badge is STILL, fuse the recent crops and read the fused
        # image in real time (sub-pixel jitter between frames recovers detail; ~sqrt(N) less noise).
        # (off by default: it runs in the camera loop; the background multi-frame decoder does this job
        # without slowing it. self.realtime_burst = True turns it on.)
        if (getattr(self, "realtime_burst", False) and qr_depth is not None and res["qr"] is None
                and res.get("badges") and len(self._soft_buf) >= 4):
            mv = self.src.meter.get("badge") if getattr(self.src, "meter", None) is not None else None
            still = mv["still"] if mv else True               # no meter (replay): assume still
            if still and now - getattr(self, "_last_burst", 0) > 0.4:
                self._last_burst = now
                grays = [c[4] for c in self._soft_buf[-10:]]
                box = self._soft_buf[-1][3]
                dist = self._soft_buf[-1][5]
                try:
                    fused = qr_depth.burst_fuse(grays, 2)
                    if depth is not None and dist:            # mask background on the fused crop
                        pr = qr_depth.prep_fused(fused, depth, self.src.intr, box, dist)
                        fused = pr if pr is not None else fused
                    from qr_reader import read_qr_in_badge
                    text, pts = read_qr_in_badge(fused, 120)
                    import os as _os
                    if _os.environ.get("BURST_DEBUG"):
                        print("BURST n=%d fused=%s text=%s" % (len(grays), getattr(fused, "shape", None), text), flush=True)
                except Exception as _e:
                    import os as _os
                    if _os.environ.get("BURST_DEBUG"):
                        import traceback; traceback.print_exc()
                    text = None
                if text:
                    quad = None
                    for q, d, _, _ in res["badges"]:
                        bx1, by1, bx2, by2 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
                        bcx = (box[0] + box[2]) / 2
                        if bx1 <= bcx <= bx2:
                            quad, dist = q.astype(np.float32), d
                            break
                    res["qr"], res["qr_burst"] = (text, quad, dist), True
                    self.stats["qr_burst"] += 1
                    return
        # the soft decoder only takes over once zxing CAN'T read this badge any more: no fast read for
        # `soft_after_s` and the badge in view at least that long (zxing gets the first second at full
        # effort). Close up, where zxing reads most frames, the soft decoder stays idle.
        zx_fail_s = now - max(getattr(self, "_fast_ok_t", 0.0), getattr(self, "_badge_since", now))
        if self._soft_buf and not self.soft.busy and zx_fail_s >= self.soft_after_s:
            best = max(self._soft_buf, key=lambda c: c[0])
            if self.soft.submit(best[2], best[3]):
                self._soft_buf.remove(best)
        if res["qr"] is not None:
            self.soft.poll()                                # drop: this frame was read anyway
            return
        if not res["badges"]:
            return                                          # keep results for the next frame
        pending = self.soft.poll()
        for k, (text, (x1, y1, x2, y2), _) in enumerate(pending):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            for quad, dist, _, _ in res["badges"]:        # attach to the badge that is still there
                bx1, by1 = quad[:, 0].min(), quad[:, 1].min()
                bx2, by2 = quad[:, 0].max(), quad[:, 1].max()
                if bx1 - 0.25 * (bx2 - bx1) <= cx <= bx2 + 0.25 * (bx2 - bx1) and \
                        by1 - 0.25 * (by2 - by1) <= cy <= by2 + 0.25 * (by2 - by1):
                    res["qr"], res["qr_soft"] = (text, quad.astype(np.float32), dist), True
                    self.stats["qr_soft"] += 1
                    self.soft.requeue(pending[:k] + pending[k + 1:])   # one vote per frame; rest later
                    return
        self.soft.requeue(pending)                          # no badge matched this frame: keep (they expire)

    # ---- processing loop
    def run(self):
        t_prev = time.time()
        while True:
            try:
                if not self.paused or self.color is None:
                    tr = time.perf_counter()
                    color, depth = self.src.read()
                    tt = self._tm("camera wait", tr)
                    t0 = time.time()
                    mode = self.mode                     # once per frame (the web UI may change it mid-frame)
                    order = None
                    if mode == OBJ_MODE:
                        res = self._object_frame(color, depth)
                    elif self.people is not None:
                        # run the person detector every OTHER frame: the tracker predicts between frames
                        # (2.5 s hold), so ids/positions persist and this halves the detector cost
                        self._pframe = getattr(self, "_pframe", 0) + 1
                        if self._pframe % 2 == 0:
                            self.people.update(self.person_det.detect(color), depth, self.src.intr)
                        order = self.people.badge_priority          # read the visitor at the robot first
                        tt = self._tm("people", tt)
                    if mode != OBJ_MODE:
                        # adaptive fast-QR effort: full retries when a badge has just appeared or is being read;
                        # once it has gone unread for 1 s (too far / blurry for the fast reader), full retries
                        # only every 4th frame and one quick try otherwise - the background decoders take over
                        nowq = time.time()
                        self._qrf = getattr(self, "_qrf", 0) + 1
                        full = (nowq - getattr(self, "_fast_ok_t", 0) < 1.0 or nowq - getattr(self, "_badge_since", nowq) < 1.5
                                or self._qrf % 4 == 0)
                        # idle mode: nobody in view and no badge for 1.5 s -> the badge detector (~1 core at
                        # 30 fps) runs every 3rd frame only; a person or badge brings it back to every frame
                        idle = (self.people is not None and mode != "QR ONLY" and self.detector is not None
                                and not self.people.visible(nowq)
                                and nowq - getattr(self, "_last_badge_t", 0.0) > 1.5)
                        if idle and self._qrf % 3:
                            res = {"badges": [], "rejected": [], "qr": None, "crop": None, "skipped": True}
                        else:
                            with self.det_lock:
                                res = process(color, depth, self.src.intr, mode, self.max_dist, self.detector, order,
                                              qr_budget_ms=None if full else 6)
                            if res["badges"]:
                                if not getattr(self, "_had_badge", False):
                                    self._badge_since = nowq
                                self._had_badge, self._last_badge_t = True, nowq
                            else:
                                self._had_badge = False
                        self.stats["idle"] = idle
                        if res["qr"]:
                            self._fast_ok_t = nowq
                        tt = self._tm("badge detect + fast QR", tt)
                    if self.soft is not None and mode != OBJ_MODE:
                        self._soft_step(res, depth)
                        self._multi_inject(res)
                        tt = self._tm("background decoders", tt)
                    raw_read = bool(res["qr"])
                    if mode == OBJ_MODE:
                        res = self.obj_tracker.update(res)
                    else:
                        res = self.tracker.update(res)       # keep a badge's QR while it stays in view
                        self._multi_feed(res, color)
                        if self.names is not None:
                            fresh = res["qr"] is not None and not (res.get("qr_held") or res.get("qr_soft")
                                                                     or res.get("qr_multi") or res.get("qr_burst"))
                            self.names.step(color, res["badges"], res["qr"][1] if fresh else None)
                            res["names"] = [self.names.name_for(b[0]) for b in res["badges"]]
                    if self.people is not None and mode != OBJ_MODE:
                        self.people.attach_badges([(q, d) for q, d, _, _ in res["badges"]])
                        if res["qr"]:
                            q = res["qr"][1]
                            qb = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()) if q is not None else \
                                res.get("qr_box")                # the badge the (remembered) code belongs to
                            if qb is not None:
                                self.people.identify(res["qr"][0], qb, res["qr"][2])
                        for (q, d, _, _), name in zip(res["badges"], res.get("names", [])):
                            if name:
                                self.people.set_name(name, (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()), d)
                        now_p = time.time()
                        res["people"] = [p.to_dict(now_p) for p in sorted(self.people.visible(now_p), key=self.people.priority)]
                    tt = self._tm("trackers", tt)
                    if self.free_qr is not None and mode != OBJ_MODE:   # object mode scans everything itself
                        res["other_qr"] = self._other_qr(color, depth, res)
                        tt = self._tm("other QR", tt)
                    # the badge (or object) being served: metered by auto-tune, watched at 90 fps for motion
                    near = (res["badges"][0] if self.people is not None else
                            min(res["badges"], key=lambda b: b[1] if b[1] else 9.0)) if res["badges"] else None
                    b = near[0] if near else None
                    box = (b[:, 0].min(), b[:, 1].min(), b[:, 0].max(), b[:, 1].max()) if b is not None else None
                    box_dist = near[1] if near else None
                    meter = None
                    if mode == OBJ_MODE:
                        # meter the code itself if one is seen, else the object (with a code, else biggest);
                        # keep its BRIGHTEST part under clipping - objects are often dark (product boxes)
                        if res.get("codes"):
                            c = min(res["codes"], key=lambda c: c[2] or 9.0)
                            q = c[1]
                            box, meter, box_dist = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()), (90, 200), c[2]
                        elif res.get("objects"):
                            o = next((o for o in res["objects"] if o["qr"]), res["objects"][0])
                            box, meter, box_dist = o["box"], (99, 225), o["dist"]
                    speed = None
                    if getattr(self.src, "meter", None) is not None:
                        self.src.set_motion_target("badge", box, box_dist)
                        top = (res.get("people") or [None])[0]
                        self.src.set_motion_target("person", top["box"] if top else None, top["dist"] if top else None)
                        mv = self.src.meter.get("badge")
                        speed = mv["speed"] if mv else None
                    if self.tuner is not None and self.tuner.enabled:
                        try:
                            self.tuner.step(color[:, :, 1], box, raw_read, meter=meter, speed=speed)   # green ~ brightness
                        except Exception as e:
                            self.error, self._error_t = f"auto-tune: {e}", time.time()
                    tt = self._tm("motion + auto-tune", tt)
                    ms = (time.time() - t0) * 1000
                    now = time.time()
                    with self.lock:
                        self.color, self.depth, self.res = color, depth, res
                        st = self.stats
                        st["frames"] += 1
                        st["badge"] += bool(res["badges"])
                        st["qr"] += bool(res["qr"])
                        st["qr_raw"] += raw_read
                        st["ms"] = 0.9 * st["ms"] + 0.1 * ms if st["ms"] else ms
                        fps = 1 / max(now - t_prev, 1e-3)
                        st["fps"] = 0.9 * st["fps"] + 0.1 * fps if st["fps"] else fps
                        t_prev = now
                        if res["qr"]:
                            self._log_qr(res["qr"], now)
                    # encode only the streams a browser is actually watching (color is the main view).
                    # the depth colormap + crop/depth JPEGs are skipped when nobody is viewing them.
                    want = self.viewers
                    # the main view: drawn + encoded only while a browser watches it (headless robot: skipped)
                    frames = {"color": jpeg(draw(color, res), 1280) if want.get("color", 0) > 0 else None}
                    # the two small previews (depth, QR reader input) refresh at ~10 fps: plenty for a thumbnail
                    thumbs = time.time() - getattr(self, "_thumb_t", 0.0) >= 0.1
                    if thumbs:
                        self._thumb_t = time.time()
                    if thumbs and want.get("depth", 0) > 0 and depth is not None:
                        # shrink to the 480-px preview FIRST, then colour (much cheaper than colouring the full frame)
                        dsm = cv2.resize(depth, (480, max(1, depth.shape[0] * 480 // depth.shape[1])),
                                         interpolation=cv2.INTER_NEAREST)
                        dv = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(dsm, 0, 3000), alpha=255 / 3000),
                                               cv2.COLORMAP_JET)
                        dv[dsm == 0] = 0
                        frames["depth"] = jpeg(dv, 480, 70)
                    if thumbs and want.get("crop", 0) > 0:
                        crop = res["crop"]
                        if crop is None:
                            crop = np.full((360, 640, 3), 24, np.uint8)
                            msg = "QR-only mode: whole frame is scanned" if mode == "QR ONLY" else \
                                "no object found" if mode == OBJ_MODE else "no badge found"
                            cv2.putText(crop, msg, (30, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 140), 1, cv2.LINE_AA)
                        frames["crop"] = jpeg(crop, 640)
                    self._tm("draw + encode", tt)
                    with self.lock:
                        self.jpg = {**self.jpg, **frames}     # thumbnails not redrawn this frame keep the last one
                        for k in frames:
                            self.jpg_id[k] = self.jpg_id.get(k, 0) + 1
                        self.frame_id += 1
                else:
                    time.sleep(0.05)

            except Exception as e:  # keep the UI alive and show the error
                self.error = f"{type(e).__name__}: {e}"
                self._error_t = time.time()
                import traceback
                traceback.print_exc()
                time.sleep(0.5)
            else:
                if self.error and time.time() - getattr(self, "_error_t", 0) > 10:
                    self.error = None                    # a passing problem: clear it after 10 s of good frames

    def _motion_state(self):
        m = getattr(self.src, "meter", None)
        if m is None:
            return None
        out = {"rate": round(m.rate)}
        for k in ("badge", "person"):
            g = m.get(k)
            out[k] = None if g is None else {"speed": round(g["speed"]), "vz": round(g["vz"], 2), "still": g["still"]}
        return out

    def _object_frame(self, color, depth):
        if self.objects is None:
            self.objects = ObjectQR()
        return self.objects.process(color, depth, self.src.intr, self.max_dist)

    def _other_qr(self, color, depth, res):
        """Codes found anywhere in the frame that are not the badge reads: [(text, quad, dist)]."""
        self.free_qr.submit(color[:, :, 1])
        now = time.time()
        if res["qr"]:
            self._badge_codes[res["qr"][0]] = now
        self._badge_codes = {t: v for t, v in self._badge_codes.items() if now - v < 10}
        out = []
        for text, quad in self.free_qr.current():
            c = quad.mean(axis=0)
            if text in self._badge_codes or any(cv2.pointPolygonTest(b[0].astype(np.float32), (float(c[0]), float(c[1])), False) >= 0
                                         for b in res["badges"]):
                continue                                    # on a badge: the badge path owns it
            dist = None
            if depth is not None:
                x0, y0 = np.clip(quad.min(axis=0).astype(int), 0, None)
                x1, y1 = quad.max(axis=0).astype(int)
                v = depth[y0:y1 + 1, x0:x1 + 1]
                v = v[v > 0]
                dist = float(np.median(v)) / 1000 if v.size else None
            out.append((text, quad, dist))
            self._log_qr((text, None, dist), time.time(), mode="other QR")
        return out

    def _log_qr(self, qr, now, mode=None):
        text, _, dist = qr
        if now - self.last_logged.get(text, 0) > 3:
            info = self.db.get(text)
            name = info.get("name") if isinstance(info, dict) else info
            self.log.appendleft({"t": time.strftime("%H:%M:%S"), "mode": mode or self.mode, "qr": text,
                                 "who": name or ("UNKNOWN" if self.db else None),
                                 "dist": round(dist, 2) if dist else None})
        self.last_logged[text] = now

    def state(self):
        # slow parts (auto-tune status) outside the lock the camera loop needs
        tune = self.tuner.status() if self.tuner else None
        with self.lock:
            res = self.res
            qr = None
            if res and res["qr"]:
                t, _, dist = res["qr"]
                info = self.db.get(t)
                name = info.get("name") if isinstance(info, dict) else info
                qr = {"text": t, "who": name, "known": info is not None, "dist": round(dist, 2) if dist else None,
                      "held": bool(res.get("qr_held")), "soft": bool(res.get("qr_soft")), "burst": bool(res.get("qr_burst")),
                      "multi": bool(res.get("qr_multi"))}
            badges = []
            if res:
                names = res.get("names") or [None] * len(res["badges"])
                for (_, dist, lmm, smm), nm in zip(res["badges"], names):
                    badges.append({"dist": round(dist, 2) if dist else None, "name": nm,
                                   "size": f"{lmm:.0f}x{smm:.0f} mm" if lmm else None})
            return {
                "mode": self.mode, "max_dist": self.max_dist, "paused": self.paused,
                "badges": badges, "rejected": len(res["rejected"]) if res else 0, "qr": qr,
                "stats": {k: round(v, 1) if isinstance(v, float) else v for k, v in self.stats.items()},
                "deep_qr": dict(self.soft.stats, workers=self.soft.workers) if self.soft else None,
                "multi_qr": dict(self.multi.stats) if self.multi else None,
                "timing": {k: round(v, 1) for k, v in self.timing.items()},
                "people": (res or {}).get("people", []) if self.people is not None else None,
                "objects": [{"label": o["label"], "dist": round(o["dist"], 2) if o["dist"] else None, "qr": o["qr"]}
                            for o in (res or {}).get("objects", [])],
                "codes": [{"text": t, "dist": round(d, 2) if d else None, "on": lab}
                          for t, _, d, lab in (res or {}).get("codes", [])],
                "other_qr": None if self.free_qr is None else
                [{"text": t, "dist": round(d, 2) if d else None} for t, _, d in (res or {}).get("other_qr", [])],
                "any_qr": None if self.free_qr is None else
                {"on": self.free_qr.enabled, "ms": round(self.free_qr.ms), "scans": self.free_qr.scans},
                "people_log": [{"t": time.strftime("%H:%M:%S", time.localtime(t)), "text": x}
                               for t, x in reversed(self.people.events[-8:])] if self.people is not None else None,
                "log": list(self.log)[:12], "db_size": len(self.db),
                "source": "replay" if isinstance(self.src, ReplaySource) else "D435i", "error": self.error,
                "detector": "trained model" if self.detector else "edge rules (no model yet)",
                "has_camera_controls": hasattr(self.src, "get_controls"),
                "autotune": tune,
                "motion": self._motion_state(),
            }

    def command(self, action, body):
        if action == "mode":
            m = body.get("mode")
            self.mode = m if m in ("QR ONLY", OBJ_MODE) else "BADGE -> QR"
        elif action == "max_dist":
            self.max_dist = min(max(float(body.get("value", 1.2)), 0.1), 3.0)
        elif action == "pause":
            self.paused = bool(body.get("on", not self.paused))
        elif action == "camera":
            if not hasattr(self.src, "set_control"):
                return {"ok": False, "error": "no live camera"}
            try:
                op = body.get("op", "set")
                if op == "save":
                    self.src.save_camera_settings()
                elif op == "reset":
                    self.src.reset_camera_settings()
                else:
                    self.src.set_control(str(body.get("name")), float(body.get("value")))
            except Exception as e:
                return {"ok": False, "error": str(e)}
            with self.lock:
                self.stats.update(frames=0, badge=0, qr=0, qr_raw=0)   # fresh read rates for the new settings
            return {"ok": True, "controls": self.src.get_controls()}
        elif action == "autotune":
            if self.tuner is None:
                return {"ok": False, "error": "no live camera"}
            self.tuner.enable() if body.get("on", not self.tuner.enabled) else self.tuner.disable()
            return {"ok": True, "controls": self.src.get_controls()}
        elif action == "any_qr":
            if self.free_qr is None:
                return {"ok": False, "error": "started with --no-any-qr"}
            self.free_qr.enabled = bool(body.get("on", not self.free_qr.enabled))
        elif action == "reload_model":
            return {"ok": self.reload_model()}
        elif action == "reset_stats":
            with self.lock:
                self.stats.update(frames=0, badge=0, qr=0, qr_raw=0)
        else:
            return {"ok": False, "error": "unknown action"}
        return {"ok": True}


def make_handler(engine):
    page = os.path.join(HERE, "web_ui.html")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype, cache=False):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                with open(page, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if path == "/api/camera":
                ctl = engine.src.get_controls() if hasattr(engine.src, "get_controls") else []
                return self._json({"controls": ctl})
            if path == "/api/state":
                return self._send(200, json.dumps(engine.state()).encode(), "application/json")
            if path.startswith("/stream/"):
                return self._stream(path[len("/stream/"):])
            self._send(404, b"not found", "text/plain")

        def _json(self, obj):
            self._send(200, json.dumps(obj).encode(), "application/json")

        def do_POST(self):
            if not self.path.startswith("/api/"):
                return self._send(404, b"", "text/plain")
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            out = engine.command(self.path[len("/api/"):], body)
            self._send(200, json.dumps(out).encode(), "application/json")

        def _stream(self, name):
            if name not in ("color", "depth", "crop"):
                return self._send(404, b"", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            with engine.lock:
                engine.viewers[name] = engine.viewers.get(name, 0) + 1   # mark this stream as watched
            try:
                while True:
                    with engine.lock:
                        fid, data = engine.jpg_id.get(name, 0), engine.jpg.get(name)
                    if fid != last and data:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
                        last = fid
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with engine.lock:
                    engine.viewers[name] = max(0, engine.viewers.get(name, 1) - 1)

    return H


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="folder of saved .jpg (+ _depth.png) instead of the camera")
    ap.add_argument("--webcam", type=int, default=None,
                    help="use a plain USB webcam at /dev/video<N> (no depth); try --webcam 0")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--badges", default="badges.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-autotune", action="store_true", help="start with camera auto-tune off")
    ap.add_argument("--mains-hz", type=int, default=0, choices=(0, 50, 60),
                    help="snap exposure to flicker-free values for this mains frequency (60 at CES; 0 = off)")
    ap.add_argument("--no-people", action="store_true", help="turn off person tracking")
    ap.add_argument("--live-codes", action="store_true",
                    help="show a badge's code only up to 1 s after a real read (default: remember it while in view)")
    ap.add_argument("--no-any-qr", action="store_true", help="only read QR codes on badges (not boxes, signs, ...)")
    ap.add_argument("--no-names", action="store_true", help="don't read the printed name on badges (OCR)")
    ap.add_argument("--no-soft-qr", action="store_true", help="turn off the slow deep QR decoder fallback")
    ap.add_argument("--qr-pattern", help="regex every valid badge code must match, e.g. '[A-Z0-9]{11}'")
    return ap


def start(args):
    """Open the camera (or replay), start processing and the HTTP server.
    Returns (server, source, url). If the port is taken, a free one is used."""
    if args.replay:
        src = ReplaySource(args.replay)
    elif getattr(args, 'webcam', None) is not None:
        src = WebcamSource(args.webcam, args.width, args.height)
    else:
        src = LiveSource(args.width, args.height)
    engine = Engine(src, load_badge_db(args.badges), getattr(args, 'qr_pattern', None),
                    autotune=not getattr(args, 'no_autotune', False),
                    soft_qr=not getattr(args, 'no_soft_qr', False), mains_hz=getattr(args, 'mains_hz', 0),
                    people=not getattr(args, 'no_people', False) and getattr(args, 'webcam', None) is None,
                    any_qr=not getattr(args, 'no_any_qr', False),
                    remember_codes=not getattr(args, 'live_codes', False),
                    names=not getattr(args, 'no_names', False))
    threading.Thread(target=engine.run, daemon=True).start()
    handler = make_handler(engine)
    try:
        srv = ThreadingHTTPServer((args.host, args.port), handler)
    except OSError:
        srv = ThreadingHTTPServer((args.host, 0), handler)
    srv.daemon_threads = True
    port = srv.server_address[1]
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{port}"
    return srv, src, url


def main():
    args = build_parser().parse_args()
    srv, src, url = start(args)
    print(f"Badge + QR scanner running: {url}   Ctrl-C to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        src.close()


if __name__ == "__main__":
    main()
