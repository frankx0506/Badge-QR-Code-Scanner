"""Desktop viewer: live view of badge identification -> QR scanning, step by step.

    python test_ui.py                 # live from the D435i
    python test_ui.py --replay DIR    # replay saved photos (DIR/*.jpg, optional *_depth.png)

Window layout
  left   : camera image. Yellow = badge found (step 1). Green = QR read (step 2).
           Grey = card-shaped outline rejected (reason shown).
  right  : depth view, "what the QR reader sees" (flattened badge), status panel
  bottom : scan log

Keys
  m      switch mode: BADGE -> QR  /  QR ONLY (whole frame)
  p      pause / resume
  q/Esc  quit
Trackbar "max dist cm": ignore badges farther than this.
"""

import argparse
import glob
import json
import math
import os
import time
from collections import deque

import cv2
import numpy as np

from badge_scanner import (BADGE_LONG_MM, BADGE_SHORT_MM, find_card_quads,
                           load_badge_db, warp_badge)
from qr_reader import read_qr, read_qr_in_badge
from badge_filter import BadgeFilter

_FILTER = BadgeFilter(min_conf=0.5)
QR_BUDGET_MS = 60      # per badge per frame; keeps the frame rate steady (raise on a fast PC, lower on a Pi)
CROP_MARGINS = (0.08, -0.05, 0.15)     # badge crop margins tried in turn (fraction of box size)

WIN = "Badge + QR test"
W, H = 1600, 900            # window canvas
LW, LH = 1120, 630          # left camera panel
RW = W - LW                 # right column width
FONT = cv2.FONT_HERSHEY_SIMPLEX
YELLOW, GREEN, GREY, RED, WHITE = (0, 220, 255), (0, 210, 0), (150, 150, 150), (0, 0, 255), (235, 235, 235)


# ---------------------------------------------------------------- sources
class Intr:
    def __init__(self, w, h, fx, fy, cx, cy):
        self.width, self.height, self.fx, self.fy, self.ppx, self.ppy = w, h, fx, fy, cx, cy


class LiveSource:
    """D435i colour (30 fps) + depth aligned to it. With fast_depth, depth and the left IR camera run at
    90 fps (global shutter) in a background thread that feeds a MotionMeter (motion90.py): real badge /
    person speed between colour frames. Falls back to 30 fps depth if the camera refuses."""

    def __init__(self, w, h, fast_depth=True):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        self.meter, prof = None, None
        self.recoveries, self.on_restart = 0, None       # on_restart(new_colour_sensor) after a stream restart
        if fast_depth:
            try:
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
                cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
                cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
                # small buffer: holding many frames can use up the camera's frame slots and stall it
                self._q = rs.frame_queue(8, True)
                prof = self.pipe.start(cfg, self._q)
                self._cfg = cfg
            except RuntimeError as e:
                print("90 fps depth/IR not available, using 30 fps depth:", e)
                prof = None
        if prof is None:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            prof = self.pipe.start(cfg)
            self._cfg = cfg
            fast_depth = False
        self.align = rs.align(rs.stream.color)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intr = Intr(i.width, i.height, i.fx, i.fy, i.ppx, i.ppy)
        self.scale_mm = prof.get_device().first_depth_sensor().get_depth_scale() * 1000
        self.color_sensor = prof.get_device().first_color_sensor()
        self.load_camera_settings()
        if fast_depth:
            import threading
            from motion90 import MotionMeter
            cs = prof.get_stream(rs.stream.color).as_video_stream_profile()
            dsp = prof.get_stream(rs.stream.depth).as_video_stream_profile()
            self._ci, self._di = cs.get_intrinsics(), dsp.get_intrinsics()
            self._c2d = cs.get_extrinsics_to(dsp)
            self.meter = MotionMeter(self._di.fx, self._ci.fx)
            self._cond = threading.Condition()
            import queue
            self._latest, self._latest_n, self._read_n, self._run = None, 0, 0, True
            self._to_align = queue.Queue(maxsize=2)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._athread = threading.Thread(target=self._align_loop, daemon=True)
            self._athread.start()

    # ---- 90 fps thread: every depth+IR frame -> motion meter; every new colour frame -> aligned pair
    def _loop(self):
        import queue
        last_c, last_d = None, None
        while self._run:
            try:
                fs = self._q.wait_for_frame(2000).as_frameset()
            except RuntimeError:
                continue
            try:
                d, ir = fs.get_depth_frame(), fs.get_infrared_frame(1)
                if d and ir and d.get_frame_number() != last_d:
                    last_d = d.get_frame_number()
                    self.meter.update(np.asanyarray(ir.get_data()), np.asanyarray(d.get_data()),
                                      d.get_timestamp() / 1000.0, self.scale_mm)
                c = fs.get_color_frame()
                if c and d and c.get_frame_number() != last_c:
                    last_c = c.get_frame_number()
                    fs.keep()                                   # hand to the align thread; never wait here
                    try:
                        self._to_align.put_nowait(fs)
                    except queue.Full:
                        try:
                            self._to_align.get_nowait()         # drop the older one, keep the newest
                        except queue.Empty:
                            pass
                        self._to_align.put_nowait(fs)
            except Exception as e:                      # keep the camera thread alive
                print("camera thread:", e)

    def _align_loop(self):
        import queue
        while self._run:
            try:
                fs = self._to_align.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                a = self.align.process(fs)
                cf, df = a.get_color_frame(), a.get_depth_frame()
                if cf and df:
                    pair = (np.asanyarray(cf.get_data()).copy(),
                            np.asanyarray(df.get_data()).astype(np.float32) * self.scale_mm)
                    with self._cond:
                        self._latest, self._latest_n = pair, self._latest_n + 1
                        self._cond.notify_all()
            except Exception as e:
                print("align thread:", e)

    def color_box_to_ir(self, box, dist_m):
        """A colour-image box at distance dist_m -> the same box in IR/depth pixels."""
        rs = self.rs
        out = []
        for u, v in ((box[0], box[1]), (box[2], box[3])):
            p = rs.rs2_deproject_pixel_to_point(self._ci, [float(u), float(v)], float(dist_m))
            p = rs.rs2_transform_point_to_point(self._c2d, p)
            out += rs.rs2_project_point_to_pixel(self._di, p)
        return out[0], out[1], out[2], out[3]

    def set_motion_target(self, name, box, dist_m):
        """Tell the 90 fps meter what to watch (colour-image box), or None to stop."""
        if self.meter is None:
            return
        if box is None or not dist_m:
            self.meter.set_target(name, None, None)
        else:
            self.meter.set_target(name, self.color_box_to_ir(box, dist_m), dist_m)

    # ---- color camera controls (exposure etc.) - less motion blur = more QR reads
    CONTROLS = ["enable_auto_exposure", "exposure", "gain", "auto_exposure_priority",
                "sharpness", "contrast", "brightness", "backlight_compensation"]
    SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_settings.json")

    def get_controls(self):
        out = []
        for name in self.CONTROLS:
            opt = getattr(self.rs.option, name, None)
            if opt is None or not self.color_sensor.supports(opt):
                continue
            r = self.color_sensor.get_option_range(opt)
            out.append({"name": name, "value": self.color_sensor.get_option(opt),
                        "min": r.min, "max": r.max, "step": r.step, "default": r.default,
                        "readonly": self.color_sensor.is_option_read_only(opt)})
        return out

    def set_control(self, name, value):
        if name not in self.CONTROLS:
            raise ValueError(f"unknown control {name}")
        opt = getattr(self.rs.option, name)
        if name == "exposure":   # manual exposure needs auto exposure off
            self.color_sensor.set_option(self.rs.option.enable_auto_exposure, 0)
        self.color_sensor.set_option(opt, float(value))

    def save_camera_settings(self):
        vals = {c["name"]: c["value"] for c in self.get_controls() if not c["readonly"]}
        with open(self.SETTINGS_FILE, "w") as f:
            json.dump(vals, f, indent=2)

    def reset_camera_settings(self):
        for c in self.get_controls():
            if not c["readonly"]:
                self.color_sensor.set_option(getattr(self.rs.option, c["name"]), c["default"])
        if os.path.exists(self.SETTINGS_FILE):
            os.remove(self.SETTINGS_FILE)

    def load_camera_settings(self):
        if not os.path.exists(self.SETTINGS_FILE):
            return
        with open(self.SETTINGS_FILE) as f:
            vals = json.load(f)
        # auto-exposure first, so a saved manual exposure isn't overridden
        for name in sorted(vals, key=lambda n: n != "enable_auto_exposure"):
            try:
                opt = getattr(self.rs.option, name)
                if name == "exposure" and vals.get("enable_auto_exposure", 0):
                    continue
                self.color_sensor.set_option(opt, float(vals[name]))
            except Exception as e:
                print("camera setting", name, "not applied:", e)

    def _recover(self):
        """The camera stopped sending frames (a known RealSense stall under heavy load): restart the
        stream; after repeated stalls also reset the camera hardware. Settings are re-applied."""
        rs = self.rs
        self.recoveries += 1
        print(f"camera stalled - restarting the stream (#{self.recoveries})", flush=True)
        try:
            self.pipe.stop()
        except Exception:
            pass
        if self.recoveries % 3 == 0:
            try:
                rs.context().query_devices()[0].hardware_reset()
                time.sleep(4)
            except Exception as e:
                print("camera hardware reset failed:", e, flush=True)
        if self.meter is not None:
            self._q = rs.frame_queue(8, True)            # the 90 fps thread picks up the new queue
            prof = self.pipe.start(self._cfg, self._q)
        else:
            prof = self.pipe.start(self._cfg)
        self.color_sensor = prof.get_device().first_color_sensor()
        self.load_camera_settings()
        if self.on_restart is not None:
            try:
                self.on_restart(self.color_sensor)
            except Exception as e:
                print("after camera restart:", e, flush=True)

    def read(self):
        if self.meter is not None:                          # 90 fps mode: newest aligned pair from the thread
            for attempt in range(3):
                with self._cond:
                    if self._cond.wait_for(lambda: self._latest_n != self._read_n, timeout=2.0):
                        self._read_n = self._latest_n
                        return self._latest
                self._recover()                             # no frame for 2 s: restart, then wait again
            raise RuntimeError("no colour frame from the camera (stream restarted 3x)")
        while True:
            try:
                fr = self.pipe.wait_for_frames(3000)
            except RuntimeError:
                self._recover()
                continue
            f = self.align.process(fr)
            cf, df = f.get_color_frame(), f.get_depth_frame()
            if cf and df:
                color = np.asanyarray(cf.get_data()).copy()
                depth = np.asanyarray(df.get_data()).astype(np.float32) * self.scale_mm
                return color, depth

    def close(self):
        if self.meter is not None:
            self._run = False
            self._thread.join(timeout=3)
            self._athread.join(timeout=3)
            while not self._to_align.empty():               # release kept frames before stopping the camera
                self._to_align.get_nowait()
            self._latest = None
        self.pipe.stop()


class WebcamSource:
    """Any plain USB/UVC webcam (e.g. Logitech Brio) via OpenCV. Colour only, NO depth, so the
    size filter and person tracking are skipped; QR reading / deep decoder / object mode still work.
    Intrinsics are estimated from the webcam's field of view (no factory calibration)."""

    def __init__(self, index=0, w=1920, h=1080, hfov_deg=70.0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open webcam {index}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        aw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or w
        ah = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or h
        f = (aw / 2) / math.tan(math.radians(hfov_deg / 2))
        self.intr = Intr(aw, ah, f, f, aw / 2, ah / 2)

    def read(self):
        ok, color = self.cap.read()
        if not ok or color is None:
            time.sleep(0.03)
            return np.zeros((self.intr.height, self.intr.width, 3), np.uint8), None
        return color, None                         # no depth

    def close(self):
        self.cap.release()


class ReplaySource:
    def __init__(self, folder):
        self.files = sorted(p for p in glob.glob(os.path.join(folder, "*.jpg")))
        if not self.files:
            raise SystemExit(f"no .jpg files in {folder}")
        self.i = 0
        img = cv2.imread(self.files[0])
        h, w = img.shape[:2]
        f = (w / 2) / math.tan(math.radians(69.4 / 2))   # D435 color HFOV ~69.4 deg
        self.intr = Intr(w, h, f, f, w / 2, h / 2)

    def read(self):
        path = self.files[self.i % len(self.files)]
        self.i += 1
        time.sleep(0.4)
        color = cv2.imread(path)
        dpath = path[:-4] + "_depth.png"
        depth = cv2.imread(dpath, cv2.IMREAD_UNCHANGED).astype(np.float32) if os.path.exists(dpath) else None
        return color, depth

    def close(self):
        pass


# ---------------------------------------------------------------- processing
def measure(quad, depth_mm, intr):
    """(distance_m, long_mm, short_mm) from aligned depth in mm, or None."""
    if depth_mm is None:
        return None
    H, W = depth_mm.shape[:2]
    q = quad.astype(np.int32)
    x1, y1 = max(int(q[:, 0].min()), 0), max(int(q[:, 1].min()), 0)
    x2, y2 = min(int(q[:, 0].max()) + 1, W), min(int(q[:, 1].max()) + 1, H)
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillConvexPoly(mask, q - [x1, y1], 255)
    v = depth_mm[y1:y2, x1:x2][mask > 0]
    v = v[v > 0]
    if v.size < 50:
        return None
    z = float(np.median(v))
    pts = np.array([[(x - intr.ppx) * z / intr.fx, (y - intr.ppy) * z / intr.fy] for x, y in quad])
    sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    a, b = (sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2
    return z / 1000, max(a, b), min(a, b)


def process_detector(color, depth, intr, max_dist, detector, out, order=None, qr_budget_ms=None):
    """Badge step with the trained model; QR read inside each detected box.
    order(box_xyxy, dist) -> sort key: which badges to read first (e.g. the visitor at the robot)."""
    H, W = color.shape[:2]
    _FILTER.max_dist = max_dist
    boxes = _FILTER.filter(detector.detect(color), depth, intr)   # real-size / distance / confidence check
    for b, why in _FILTER.last_rejects:
        if not why.startswith("low confidence"):
            out["rejected"].append((np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32), why))
    for x1, y1, x2, y2, score in boxes:
        quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
        m = measure(quad, depth, intr)
        dist, lmm, smm = m if m else (None, None, None)
        if dist is not None and dist > max_dist:
            out["rejected"].append((quad, f"too far {dist:.2f}m"))
            continue
        out["badges"].append((quad, dist, lmm, smm))
        out.setdefault("scores", []).append(score)
    # Try the QR only in the 2 most confident badge boxes, so stray boxes don't slow every frame.
    if order is not None and len(out["badges"]) > 1:
        out["badges"].sort(key=lambda b: order((b[0][0][0], b[0][0][1], b[0][2][0], b[0][2][1]), b[1]))
    for quad, dist, _, _ in out["badges"][:2]:
        x1, y1 = quad[0]
        x2, y2 = quad[2]
        # zxing is sensitive to how the badge is framed: if the normal crop fails, retry two
        # other framings within the same time budget
        t_start = time.perf_counter()
        text = pts = None
        budget = QR_BUDGET_MS if qr_budget_ms is None else qr_budget_ms
        margins = CROP_MARGINS if budget >= 20 else CROP_MARGINS[:1]     # quick mode: one plain attempt
        for k, m in enumerate(margins):
            left = budget - (time.perf_counter() - t_start) * 1000
            if k and left < 8:
                break
            mx, my = m * (x2 - x1), m * (y2 - y1)
            cx1, cy1 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
            cx2, cy2 = int(min(x2 + mx, W)), int(min(y2 + my, H))
            crop = color[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            if k == 0 and out["crop"] is None:
                out["crop"] = crop
            text, pts = read_qr_in_badge(crop, left)
            if text:
                break
        if not text and "qr_try" not in out:
            mx, my = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            cx1, cy1 = int(max(x1 - mx, 0)), int(max(y1 - my, 0))
            cx2, cy2 = int(min(x2 + mx, W)), int(min(y2 + my, H))
            if cx2 > cx1 and cy2 > cy1:          # copy: a view would keep the whole frame alive in the buffer
                out["qr_try"] = (color[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2), dist)   # for the slow decoder
        if text:
            out["crop"] = crop.copy()
            if pts is not None:
                cv2.polylines(out["crop"], [pts.astype(np.int32)], True, GREEN, 3)
                pts = pts + np.array([cx1, cy1], np.float32)
            out["qr"] = (text, pts, dist)
            break
    return out


def process(color, depth, intr, mode, max_dist, detector=None, order=None, qr_budget_ms=None):
    """Return a dict describing everything found in this frame."""
    out = {"badges": [], "rejected": [], "qr": None, "crop": None}
    if mode != "QR ONLY" and detector is not None:
        return process_detector(color, depth, intr, max_dist, detector, out, order, qr_budget_ms)
    if mode == "QR ONLY":
        text, pts = read_qr(color)
        if text:
            dist = None
            if depth is not None:
                cx, cy = np.clip(pts.mean(axis=0).astype(int), 0, [intr.width - 1, intr.height - 1])
                d = depth[cy, cx]
                dist = d / 1000 if d > 0 else None
            out["qr"] = (text, pts, dist)
        return out

    for quad in find_card_quads(color):
        m = measure(quad, depth, intr)
        if m is not None:
            dist, lmm, smm = m
            if dist > max_dist:
                out["rejected"].append((quad, f"too far {dist:.2f}m"))
                continue
            if not (BADGE_LONG_MM[0] <= lmm <= BADGE_LONG_MM[1] and BADGE_SHORT_MM[0] <= smm <= BADGE_SHORT_MM[1]):
                out["rejected"].append((quad, f"size {lmm:.0f}x{smm:.0f}mm"))
                continue
        else:
            dist, lmm, smm = None, None, None
        out["badges"].append((quad, dist, lmm, smm))

    # Read QR inside the badge(s), largest first.
    out["badges"].sort(key=lambda b: -cv2.contourArea(b[0]))
    for quad, dist, _, _ in out["badges"]:
        crop = warp_badge(color, quad, long_px=900)
        if out["crop"] is None:
            out["crop"] = crop
        text, pts = read_qr_in_badge(crop, QR_BUDGET_MS)
        if not text and "qr_try" not in out:
            bx, by, bw, bh = cv2.boundingRect(quad.astype(np.int32))
            out["qr_try"] = (crop, (bx, by, bx + bw, by + bh), dist)  # for the slow decoder (qr_soft)
        if text:
            out["crop"] = crop.copy()
            if pts is not None:
                cv2.polylines(out["crop"], [pts.astype(np.int32)], True, GREEN, 4)
            # map QR corners back to the full image for drawing
            tl, tr, br, bl = quad
            ch, cw = crop.shape[:2]
            M = cv2.getPerspectiveTransform(
                np.array([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]], np.float32), quad)
            full = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), M).reshape(4, 2) \
                if pts is not None else None
            out["qr"] = (text, full, dist)
            break
    return out


# ---------------------------------------------------------------- drawing
def fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s), int(ih * s)))
    panel = np.full((h, w, 3), 30, np.uint8)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    panel[y:y + r.shape[0], x:x + r.shape[1]] = r
    return panel


def text(img, s, org, color=WHITE, scale=0.6, th=1):
    (tw, tht), base = cv2.getTextSize(s, FONT, scale, th)
    x, y = int(org[0]), int(org[1])
    x0, y0 = max(x - 3, 0), max(y - tht - 4, 0)
    roi = img[y0:y + base + 3, x0:x + tw + 3]
    roi[:] = (roi * 0.35).astype(np.uint8)      # darkened backdrop for readability
    cv2.putText(img, s, (x, y), FONT, scale, color, th, cv2.LINE_AA)


def render(color, depth, res, state, db):
    canvas = np.full((H, W, 3), 22, np.uint8)
    view = color.copy()
    s = max(1, view.shape[1] // 960)

    for quad, why in res["rejected"]:
        cv2.polylines(view, [quad.astype(np.int32)], True, GREY, 2 * s)
        text(view, why, tuple(quad[0].astype(int) + [0, -8]), GREY, 0.6 * s, s)
    for quad, dist, lmm, smm in res["badges"]:
        cv2.polylines(view, [quad.astype(np.int32)], True, YELLOW, 3 * s)
        lab = "BADGE" + (f" {dist:.2f}m {lmm:.0f}x{smm:.0f}mm" if dist else "")
        text(view, lab, tuple(quad[0].astype(int) + [0, -10]), YELLOW, 0.7 * s, 2 * s)
    if res["qr"] and res["qr"][1] is not None:
        t, pts, _ = res["qr"]
        cv2.polylines(view, [pts.astype(np.int32)], True, GREEN, 3 * s)

    canvas[:LH, :LW] = fit(view, LW, LH)
    text(canvas, f"MODE: {state['mode']}", (12, 30), YELLOW if state["mode"] != "QR ONLY" else GREEN, 0.8, 2)

    # right column: depth
    dh = 260
    if depth is not None:
        dv = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth, 0, 3000), alpha=255 / 3000), cv2.COLORMAP_JET)
        dv[depth == 0] = 0
        canvas[:dh, LW:] = fit(dv, RW, dh)
    else:
        text(canvas, "no depth", (LW + 20, 130), GREY)
    text(canvas, "Depth (0-3 m)", (LW + 10, 22))

    # right column: what the QR reader sees
    ch = 300
    if res["crop"] is not None:
        canvas[dh:dh + ch, LW:] = fit(res["crop"], RW, ch)
    else:
        text(canvas, "(no badge crop)", (LW + 20, dh + 150), GREY)
    text(canvas, "QR reader input (flattened badge)" if state["mode"] != "QR ONLY" else "QR ONLY: whole frame",
         (LW + 10, dh + 22))

    # right column: status
    y = dh + ch + 30
    nb = len(res["badges"])
    step1 = ("FOUND" if nb else "none") if state["mode"] != "QR ONLY" else "skipped"
    text(canvas, f"1. Badge: {step1}" + (f" ({nb})" if nb > 1 else ""), (LW + 12, y),
         YELLOW if nb else GREY, 0.75, 2)
    if res["qr"]:
        t, _, dist = res["qr"]
        info = db.get(t)
        name = info.get("name") if isinstance(info, dict) else info
        text(canvas, f"2. QR: {t}", (LW + 12, y + 35), GREEN, 0.75, 2)
        text(canvas, f"   who: {name or ('UNKNOWN' if db else '-')}", (LW + 12, y + 65),
             GREEN if name else (RED if db else GREY), 0.65, 1)
        if dist:
            text(canvas, f"   distance: {dist:.2f} m", (LW + 12, y + 92), WHITE, 0.6)
    else:
        text(canvas, "2. QR: not read", (LW + 12, y + 35), GREY, 0.75, 2)
    text(canvas, f"FPS {state['fps']:.1f}   max dist {state['max_dist']:.2f} m", (LW + 12, H - 15), WHITE, 0.55)

    # bottom: stats + log
    text(canvas, f"frames {state['frames']}   badge found {state['n_badge']}   QR read {state['n_qr']}"
         f"{'   PAUSED' if state['paused'] else ''}", (12, LH + 28), WHITE, 0.6)
    text(canvas, "keys: m=mode  p=pause  q=quit", (12, LH + 55), GREY, 0.55)
    for i, line in enumerate(state["log"]):
        text(canvas, line, (12, LH + 90 + i * 24), GREEN, 0.55)
    return canvas


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", help="folder of saved .jpg (+ _depth.png) to replay instead of the camera")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--badges", default="badges.json")
    ap.add_argument("--snapshot", help="render one frame to this image file and exit (for testing)")
    args = ap.parse_args()

    db = load_badge_db(args.badges)
    src = ReplaySource(args.replay) if args.replay else LiveSource(args.width, args.height)
    state = dict(mode="BADGE -> QR", fps=0.0, max_dist=1.2, frames=0, n_badge=0, n_qr=0,
                 paused=False, log=deque(maxlen=8))
    last_logged, t_prev = {}, time.time()

    if not args.snapshot:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W, H)
        cv2.createTrackbar("max dist cm", WIN, 120, 300, lambda v: None)

    color = depth = None
    res = {"badges": [], "rejected": [], "qr": None, "crop": None}
    try:
        while True:
            if not state["paused"] or color is None:
                color, depth = src.read()
                if not args.snapshot:
                    state["max_dist"] = max(cv2.getTrackbarPos("max dist cm", WIN), 10) / 100
                res = process(color, depth, src.intr, state["mode"], state["max_dist"])
                state["frames"] += 1
                state["n_badge"] += bool(res["badges"])
                state["n_qr"] += bool(res["qr"])
                now = time.time()
                state["fps"] = 0.9 * state["fps"] + 0.1 / max(now - t_prev, 1e-3)
                t_prev = now
                if res["qr"]:
                    t, _, dist = res["qr"]
                    if now - last_logged.get(t, 0) > 3:
                        info = db.get(t)
                        who = f"  -> {info}" if info else ("  -> UNKNOWN" if db else "")
                        d = f"{dist:.2f} m" if dist else "? m"
                        line = f"[{time.strftime('%H:%M:%S')}] {state['mode']}  {d}  QR={t}{who}"
                        state["log"].appendleft(line)
                        print(line, flush=True)
                    last_logged[t] = now

            canvas = render(color, depth, res, state, db)
            if args.snapshot:
                cv2.imwrite(args.snapshot, canvas)
                break
            cv2.imshow(WIN, canvas)
            k = cv2.waitKey(1) & 0xFF
            now = time.time()
            if k in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == ord("m"):
                state["mode"] = "QR ONLY" if state["mode"] != "QR ONLY" else "BADGE -> QR"
            elif k == ord("p"):
                state["paused"] = not state["paused"]
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
