"""Camera auto-tune: the color camera adjusts itself for the BADGE, not the room.

Closed loop, run once per processed frame (acts every `period` frames so the
camera has time to apply the previous change):

 1. Brightness on the badge  - keep the white card bright but unclipped: the card level
      (85th percentile of the badge box) -> target ~190, corrected in the
      log domain with a dead band (+-6%) and a damped step, so it settles instead of pumping.
      Small shiny reflections are ignored (darkening for them blacks out the card); the card
      clipping white does darken it.
 2. Motion -> exposure cap   - badge speed (px/s) limits exposure so motion blur
      stays under ~1 px; the rest of the brightness comes from gain. If even max gain is not
      enough (dim room), brightness wins: the frame is never made more than `max_dark` (1.25x)
      darker than the light needs - a slightly soft badge reads, a black one doesn't.
      Box jumps (another badge, a bad detection) are not counted as motion.
      The cap drops at once and recovers gradually.
 1b. Dim room                - when exposure AND gain are both at maximum and the badge is still
      too dark, the camera's own brightness setting is raised (up to +64); when it gets too bright,
      that boost is taken back first, before exposure/gain are lowered (no pumping).
 3. QR feedback (learning)   - while a badge is in view, the camera's sharpness
      setting is chosen among a few values by measured QR read rate.

When the badge leaves (or detection drops a frame), the settings are HELD for 2.5 s;
only then does it meter the centre of the frame, slowly and a little darker.
Units for the D435/D435i color sensor: exposure in 100 us steps, gain 0-128.

    tuner = AutoTuner(CameraIO(color_sensor))
    tuner.step(gray_frame, badge_box_or_None, qr_read_this_frame)
    tuner.status()   # what it's doing, for the UI
"""

import json
import math
import os
import random
import time

import numpy as np


class CameraIO:
    """Thin wrapper over a pyrealsense2 color sensor (swap for a fake in tests)."""

    def __init__(self, sensor):
        import pyrealsense2 as rs
        self.rs, self.s = rs, sensor

    def get(self, name):
        return self.s.get_option(getattr(self.rs.option, name))

    def set(self, name, value):
        opt = getattr(self.rs.option, name)
        r = self.s.get_option_range(opt)
        self.s.set_option(opt, float(min(max(value, r.min), r.max)))

    def range(self, name):
        r = self.s.get_option_range(getattr(self.rs.option, name))
        return r.min, r.max


class AutoTuner:
    def __init__(self, cam, period=4, target=190, scene_target=165, blur_px=1.0,
                 min_exp=10, max_exp=330, min_gain=16, max_gain=128,
                 sharpness_choices=(20, 35, 50, 70), seed=0,
                 deadband=0.06, k_badge=0.4, k_scene=0.15, max_step=0.35, hold_s=2.5, mains_hz=0,
                 max_dark=1.25):
        self.cam, self.period = cam, period
        self.target, self.scene_target, self.blur_px = target, scene_target, blur_px
        self.deadband, self.k_badge, self.k_scene, self.max_step, self.hold_s = deadband, k_badge, k_scene, max_step, hold_s
        self.last_badge_t, self.glare_on, self.cap = -1e9, False, max_exp
        self.max_dark = max_dark
        self.boost, self.max_boost, self.boost_step = 0, 64, 8
        # lights flicker at 2x mains (120 Hz at CES / US, 100 Hz in 50 Hz countries); exposures that
        # are whole multiples of the flicker period cancel it. 0 = off.
        self.flicker_unit = (1e4 / (2 * mains_hz)) if mains_hz else 0
        self.min_exp, self.max_exp, self.min_gain, self.max_gain = min_exp, max_exp, min_gain, max_gain
        self.sharp_choices = list(sharpness_choices)
        self.rng = random.Random(seed)
        self.enabled = False
        self.n = 0
        self.path, self.speed = [], 0.0
        self.badge_frames = 0
        self.qr_hist = []                            # recent (badge present, qr read) for the feedback loop
        self.sharp_stats = {v: [0, 0] for v in self.sharp_choices}   # value -> [reads, frames]
        self.sharp_cur, self.sharp_since = None, 0
        self.last = {}
        self.state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autotune_state.json")
        self._load()

    # ---------------------------------------------------------------- remember what was learned
    def _load(self):
        try:
            with open(self.state_file) as f:
                saved = json.load(f).get("sharpness_stats", {})
            for k, (r, n) in saved.items():
                if int(k) in self.sharp_stats:
                    # keep the ranking but let new evidence move it quickly
                    self.sharp_stats[int(k)] = [r * 50 / max(n, 1), 50] if n else [0, 0]
        except (OSError, ValueError):
            pass

    def save(self):
        try:
            with open(self.state_file, "w") as f:
                json.dump({"sharpness_stats": self.sharp_stats, "saved": time.time()}, f)
        except OSError:
            pass

    # ---------------------------------------------------------------- on / off
    def _set_boost(self, v):
        try:
            self.cam.set("brightness", v)
            self.boost = v
        except Exception:
            self.max_boost = 0                       # camera has no brightness option

    def enable(self):
        self.cam.set("enable_auto_exposure", 0)
        self._set_boost(0)
        try:
            self.cam.set("auto_exposure_priority", 0)    # never trade frame rate for long exposures
        except Exception:
            pass
        self.enabled = True

    def disable(self):
        self.save()
        self.enabled = False
        self._set_boost(0)
        self.cam.set("enable_auto_exposure", 1)

    # ---------------------------------------------------------------- one frame
    def step(self, gray, badge=None, qr_read=False, now=None, meter=None, speed=None):
        """meter=(percentile, target) overrides how the box is metered. Default (85, 190) suits a white
        badge card. For a dark object (product box) with a QR code use (99, 225): keep its brightest part
        (the code's white background) just under clipping, instead of pushing the dark box to 190.
        speed: the badge's real speed in colour px/s measured at 90 fps (motion90); when given it replaces
        the estimate from the badge box's movement (which jitters and jumps)."""
        if not self.enabled:
            return
        now = time.time() if now is None else now
        self.n += 1
        H, W = gray.shape[:2]

        # motion of the badge (px/s), smoothed
        if badge is not None:
            c = ((badge[0] + badge[2]) / 2, (badge[1] + badge[3]) / 2)
            bw = max(badge[2] - badge[0], 1.0)
            if self.path and math.dist(c, self.path[-1][1]) > 0.5 * bw * max(1.0, 15 * (now - self.path[-1][0])):
                self.path = []                       # box jumped (other badge / bad box): not motion
            self.path.append((now, c))
            self.path = [p for p in self.path if now - p[0] <= 0.2][-8:]
            if len(self.path) >= 3 and now > self.path[0][0]:
                # speed over ~0.2 s: box jitter of a few px averages out, real motion doesn't
                v = math.dist(c, self.path[0][1]) / (now - self.path[0][0])
                v = max(0.0, v - 30)                 # dead band
                self.speed = 0.6 * self.speed + 0.4 * v
            if speed is not None:                    # measured at 90 fps: use it (small dead band only)
                self.speed = max(0.0, speed - 10)
                self.speed_src = "90fps"
            else:
                self.speed_src = "box"
            self.badge_frames += 1
        else:
            self.path = []
            self.speed *= 0.8
            self.badge_frames = 0

        self.qr_hist.append((badge is not None, bool(qr_read)))
        self.qr_hist = self.qr_hist[-60:]
        if badge is not None and self.sharp_cur is not None:
            st = self.sharp_stats[self.sharp_cur]
            st[0] += bool(qr_read)
            st[1] += 1

        if self.n % 900 == 0:              # ~30 s at 30 fps
            self.save()
        if badge is not None:
            self.last_badge_t = now
        if self.n % self.period:
            return

        # ---- what to meter: the badge; right after it leaves, HOLD (no flicker when detection
        # drops a frame or the person turns); only after hold_s meter the scene centre, gently
        if badge is not None:
            x1, y1, x2, y2 = (int(v) for v in badge[:4])
            roi = gray[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            mode, target, k = "badge", self.target, self.k_badge
            pct = 85
            if meter is not None:
                pct, target = meter
                mode = "object"
        elif now - self.last_badge_t < self.hold_s:
            self.last.update(metering="hold")
            self._sharp_idle(badge)
            return
        else:
            roi = gray[H // 5:4 * H // 5, W // 5:4 * W // 5]
            mode, target, k = "centre", self.scene_target, self.k_scene
        if roi.size < 100:
            return
        roi = roi[::2, ::2] if roi.size > 200000 else roi
        glare = float((roi >= 250).mean())
        # the card's white: 85th percentile of the badge box. The white card is most of the box,
        # so this follows the card (and sees it when it clips), while a small shiny reflection on
        # the sleeve (< ~15% of the box) does not pull it - darkening for a reflection would black
        # out the card without removing the reflection.
        level = float(np.percentile(roi, pct if mode in ("badge", "object") else 85))
        if glare > 0.03:
            self.glare_on = True
        elif glare < 0.01:
            self.glare_on = False

        # ---- brightness error in the log domain (pixel values ~ exposure^(1/2.2)), with a dead band
        err = math.log(target / max(level, 1.0))
        if level >= 250:
            err = min(err, -0.7)                     # clipped: true level unknown, assume far too bright
        exp_now, gain_now = self.cam.get("exposure"), self.cam.get("gain")
        product = exp_now * gain_now
        maxed = gain_now >= self.max_gain - 1 and exp_now >= 0.9 * self.max_exp
        if err > self.deadband and maxed and self.boost < self.max_boost:
            self._set_boost(min(self.max_boost, self.boost + self.boost_step))   # dim room: lift
        elif -0.3 < err < -self.deadband and self.boost > 0:
            self._set_boost(max(0, self.boost - self.boost_step))                # undo lift first
        elif abs(err) > self.deadband:
            if err < 0 and self.boost > 0:
                self._set_boost(0)                   # lights came on: drop the lift at once
            step = max(-self.max_step, min(self.max_step, k * 2.2 * err))
            product *= math.exp(step)

        # ---- motion -> exposure cap (drops at once, recovers slowly so brightness doesn't pump)
        cap = (self.blur_px / self.speed) * 1e4 if self.speed > 1 else self.max_exp
        cap = min(max(cap, self.min_exp), self.max_exp)
        # recovers faster when the speed is MEASURED (90 fps): a still badge is really still
        self.cap = cap if cap < self.cap else min(cap, self.cap * (1.4 if getattr(self, "speed_src", "") == "90fps" else 1.15))
        cap = self.cap

        # ---- split: exposure first (least noise) up to the motion cap, then gain;
        # if even max gain is not enough, go past the cap (up to 2x): a slightly soft badge
        # reads better than a dark one
        exp_new = min(max(product / self.min_gain, self.min_exp), cap)
        gain_new = product / exp_new
        if gain_new > self.max_gain:
            # not enough light to honour the motion cap at max gain: brightness wins - go past the
            # cap so the frame is at most max_dark darker than the light needs
            e_full = product / self.max_gain
            exp_new = min(self.max_exp, max(cap, e_full / self.max_dark))
            gain_new = product / exp_new
        gain_new = min(max(gain_new, self.min_gain), self.max_gain)
        if self.flicker_unit and exp_new >= self.flicker_unit:
            k = max(1, round(exp_new / self.flicker_unit))       # snap to a flicker-free exposure,
            snapped = k * self.flicker_unit                      # keep brightness with gain
            gain_new = min(max(gain_new * exp_new / snapped, self.min_gain), self.max_gain)
            exp_new = snapped
        if abs(exp_new - exp_now) / max(exp_now, 1) > 0.04:
            self.cam.set("exposure", round(exp_new))
        if abs(gain_new - gain_now) >= 2:
            self.cam.set("gain", round(gain_new))

        self._sharp_idle(badge)
        # ---- QR feedback (only while a badge is in view): pick the best camera sharpness
        recent = [q for b_, q in self.qr_hist[-30:] if b_]
        rate = sum(recent) / len(recent) if recent else None
        if self.badge_frames >= 15 and rate is not None:
            self._tune_sharpness()

        self.last = dict(bright=round(level), glare=round(100 * glare, 1), speed=round(self.speed),
                         exposure_ms=round(exp_new / 10, 1), gain=round(gain_new), cap_ms=round(cap / 10, 1),
                         qr_rate=None if rate is None else round(100 * rate), sharpness=self.sharp_cur,
                         speed_src=getattr(self, "speed_src", "box"),
                         metering=mode, target=round(target), boost=self.boost)

    def _sharp_idle(self, badge):
        """No badge for a while: go back to the best sharpness found so far (not the last one tried)."""
        if badge is None and self.n - self.sharp_since > 30:
            tried = {v: r / f for v, (r, f) in self.sharp_stats.items() if f >= 10}
            if tried:
                best_v = max(tried, key=tried.get)
                if best_v != self.sharp_cur:
                    try:
                        self.cam.set("sharpness", best_v)
                        self.sharp_cur, self.sharp_since = best_v, self.n
                    except Exception:
                        pass

    def _tune_sharpness(self, dwell=30, explore=0.15):
        """Pick the camera sharpness with the best QR read rate (epsilon-greedy, ~1 s per try)."""
        if self.sharp_cur is not None and self.n - self.sharp_since < dwell:
            return
        untried = [v for v, (_, f) in self.sharp_stats.items() if f < dwell]
        if untried:
            choice = untried[0]
        elif self.rng.random() < explore:
            choice = self.rng.choice(self.sharp_choices)
        else:
            choice = max(self.sharp_stats, key=lambda v: self.sharp_stats[v][0] / max(self.sharp_stats[v][1], 1))
        if choice != self.sharp_cur:
            try:
                self.cam.set("sharpness", choice)
            except Exception:
                return
        self.sharp_cur, self.sharp_since = choice, self.n

    def status(self):
        if not self.enabled:
            return {"enabled": False}
        best = {v: round(100 * r / f) for v, (r, f) in self.sharp_stats.items() if f}
        return dict(enabled=True, **self.last, sharpness_read_rates=best)
