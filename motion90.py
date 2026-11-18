"""90 fps motion meter: how fast the badge / visitor is REALLY moving, between colour frames.

The D435i's IR camera is global-shutter and runs at 90 fps (depth too, in the same pixel grid), while
the colour camera gives 30. Between two colour frames this measures, for each target (badge, person):
  - sideways motion: phase correlation of the target's IR patch between consecutive IR frames
    (sub-pixel; the projector's dot pattern - which stays still in the image - is removed first with a
    median filter, so it doesn't pull the answer to zero)
  - toward / away motion: median depth of the target's pixels, frame to frame
and converts it to COLOUR-camera pixels per second, the unit auto-tune needs for its blur limit.

Uses: auto-tune's motion limit (instead of the jittery badge-box estimate), "badge is still now",
approach speed. Pure numpy/OpenCV: the camera code feeds it frames (see test_ui.LiveSource).

    m = MotionMeter(fx_ir, fx_color)
    m.set_target("badge", ir_box, dist_m)        # from the main loop, ~15-30 Hz
    m.update(ir_u8, depth_mm, t_s)               # from the camera thread, 90 Hz
    m.get("badge") -> {"speed": colour px/s, "vx", "vy", "vz": m/s, "still": bool, "age": s}
"""
import math
import threading
import time

import cv2
import numpy as np


class _Target:
    def __init__(self, box, dist, t):
        self.box, self.dist, self.set_t = [float(v) for v in box], dist, t
        self.prev_t = None
        self.vx = self.vy = self.vz = 0.0     # IR px/s, IR px/s, m/s (smoothed)
        self.peak = []                   # (t, colour px/s) for the last ~0.15 s
        self.z = dist
        self.n = 0


class MotionMeter:
    def __init__(self, fx_ir, fx_color, patch=64, still_px_s=60.0, still_mps=0.15, max_age_s=1.0):
        self.fx_ir, self.fx_color = fx_ir, fx_color
        self.scale = fx_color / fx_ir                       # IR px -> colour px (same angle)
        self.patch, self.still_px_s, self.still_mps, self.max_age_s = patch, still_px_s, still_mps, max_age_s
        self._t = {}
        self._lock = threading.Lock()
        self._win = cv2.createHanningWindow((patch, patch), cv2.CV_32F)
        self.rate = 0.0                                     # measured update rate (Hz)
        self._last_upd = None
        self._prev = None                                   # previous IR frame, dots removed

    # ---------------------------------------------------------------- main-loop side
    def set_target(self, name, ir_box, dist_m, now=None):
        """(Re)place a target from a fresh detection (box in IR pixel coordinates). None removes it."""
        now = time.time() if now is None else now
        with self._lock:
            if ir_box is None:
                self._t.pop(name, None)
                return
            t = self._t.get(name)
            if t is None or _iou(t.box, ir_box) < 0.2:      # a different object: start over
                self._t[name] = _Target(ir_box, dist_m, now)
            else:                                           # same object: re-anchor, keep the speed history
                t.box, t.dist, t.set_t = [float(v) for v in ir_box], dist_m, now

    def get(self, name, now=None):
        now = time.time() if now is None else now
        with self._lock:
            t = self._t.get(name)
            if t is None or t.n < 3 or now - t.set_t > self.max_age_s:
                return None
            sp = math.hypot(t.vx, t.vy) * self.scale
            peak = max([p for _, p in t.peak], default=sp)
            return {"speed": sp, "peak": peak, "vx": t.vx * self.scale, "vy": t.vy * self.scale,
                    "vz": t.vz, "still": sp < self.still_px_s and abs(t.vz) < self.still_mps,
                    "age": now - t.set_t, "z": t.z}

    # ---------------------------------------------------------------- camera-thread side
    def update(self, ir, depth_mm, ts, depth_scale=1.0):
        """One IR frame (uint8, H x W) + depth (same grid) at time ts (s). depth * depth_scale = mm, so the
        camera's raw z16 frame can be passed as is."""
        if self._last_upd is not None and ts > self._last_upd:
            self.rate = 0.9 * self.rate + 0.1 / (ts - self._last_upd) if self.rate else 1 / (ts - self._last_upd)
        self._last_upd = ts
        with self._lock:
            targets = list(self._t.values())
        H, W = ir.shape[:2]
        cur = ir                                            # raw; dots are filtered per patch below
        prev, self._prev = self._prev, ir.copy()
        for t in targets:
            x1, y1, x2, y2 = t.box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            side = max(24.0, min(max(x2 - x1, y2 - y1) * 1.2, 200.0))
            ax, ay = int(round(cx - side / 2)), int(round(cy - side / 2))
            bx, by = ax + int(side), ay + int(side)
            if ax < 0 or ay < 0 or bx > W or by > H:
                t.prev_t = None
                continue
            k = self.patch / side                                             # IR px -> patch px
            size = (self.patch, self.patch)
            # both frames cut at the SAME window, so window moves never look like motion
            dots = lambda a: cv2.medianBlur(cv2.resize(a, size, interpolation=cv2.INTER_AREA), 3)   # remove projector dots
            patch = dots(cur[ay:by, ax:bx]).astype(np.float32)
            # toward / away: median depth of the target-depth pixels in the box
            if depth_mm is not None:
                d = depth_mm[max(0, int(y1)):int(y2):3, max(0, int(x1)):int(x2):3]     # sparse sample is enough
                d = d[d > 0].astype(np.float32) * depth_scale
                d = d[np.abs(d - t.z * 1000) < 300] if t.z else d
                z = float(np.median(d)) / 1000 if d.size > 30 else None
            else:
                z = None
            if prev is not None and t.prev_t is not None and ts > t.prev_t:
                dt = ts - t.prev_t
                before = dots(prev[ay:by, ax:bx]).astype(np.float32)
                if np.std(patch) > 4:                                         # enough texture to measure
                    (dx, dy), resp = cv2.phaseCorrelate(before, patch, self._win)
                    if resp > 0.05 and abs(dx) < self.patch / 3 and abs(dy) < self.patch / 3:
                        vx, vy = dx / k / dt, dy / k / dt
                        a = 0.5
                        t.vx, t.vy = (1 - a) * t.vx + a * vx, (1 - a) * t.vy + a * vy
                        with self._lock:                                      # follow the target
                            t.box = [x1 + dx / k, y1 + dy / k, x2 + dx / k, y2 + dy / k]
                if z is not None and t.z:
                    if abs(z - t.z) > max(0.06, 3.0 * dt):   # > 3 m/s in one frame: a hand / occlusion, not motion
                        z = None
                    else:
                        t.vz = 0.7 * t.vz + 0.3 * (z - t.z) / dt
                t.n += 1
                sp = math.hypot(t.vx, t.vy) * self.scale
                t.peak = [p for p in t.peak if ts - p[0] < 0.15] + [(ts, sp)]
            if z is not None:
                t.z = z
            t.prev_t = ts


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0
