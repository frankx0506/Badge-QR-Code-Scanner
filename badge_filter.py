"""Reject boxes that can't be a badge, using the depth camera's real-world size.

The detector only sees pixels; the D435i also knows how far away each box is, so
we can compute its physical size. Conference badges (with ribbon) are about
100-140 mm wide and 150-230 mm tall; a laptop, arm or person is far bigger.

    filt = BadgeFilter()
    boxes = filt.filter(detector.detect(img), depth_mm, intrinsics)

Limits are deliberately generous (other badge types, partly hidden badges);
boxes touching the image edge (cut-off badges) only need to be small enough.
"""

import numpy as np


class BadgeFilter:
    def __init__(self, min_conf=0.5, max_dist=1.5,
                 width_mm=(50, 200), height_mm=(30, 300), max_aspect=3.5):
        self.min_conf, self.max_dist = min_conf, max_dist
        self.width_mm, self.height_mm, self.max_aspect = width_mm, height_mm, max_aspect
        self.last_rejects = []

    def measure(self, box, depth, intr):
        """(distance_m, width_mm, height_mm) from the central part of the box, or None."""
        if depth is None:
            return None
        x1, y1, x2, y2 = (int(v) for v in box[:4])
        qx, qy = (x2 - x1) // 4, (y2 - y1) // 4           # central half: the badge face, not the background
        v = depth[max(0, y1 + qy):max(0, y2 - qy), max(0, x1 + qx):max(0, x2 - qx)]
        v = v[v > 0]
        if v.size < 30:
            return None
        z = float(np.median(v))
        return z / 1000, (x2 - x1) * z / intr.fx, (y2 - y1) * z / intr.fy

    def check(self, box, depth, intr):
        """None if the box is plausible, else the reason it isn't."""
        if box[4] < self.min_conf:
            return f"low confidence {box[4]:.2f}"
        m = self.measure(box, depth, intr)
        if m is None:
            return None                                       # no depth: trust the model
        dist, w, h = m
        if dist > self.max_dist:
            return f"too far {dist:.2f} m"
        W, H = intr.width, intr.height
        at_edge = box[0] < 4 or box[1] < 4 or box[2] > W - 5 or box[3] > H - 5
        if w > self.width_mm[1] or h > self.height_mm[1]:
            return f"too big {w:.0f}x{h:.0f} mm"
        if not at_edge and (w < self.width_mm[0] or h < self.height_mm[0]):
            return f"too small {w:.0f}x{h:.0f} mm"
        if max(w, h) / max(min(w, h), 1) > self.max_aspect:
            return f"wrong shape {w:.0f}x{h:.0f} mm"
        return None

    def filter(self, boxes, depth, intr):
        keep, self.last_rejects = [], []
        for b in boxes:
            why = self.check(b, depth, intr)
            if why is None:
                keep.append(b)
            else:
                self.last_rejects.append((b, why))
        return keep
