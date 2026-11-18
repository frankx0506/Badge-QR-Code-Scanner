"""Multi-frame QR reading: remember a badge's QR code while the badge stays in view.

A small QR code may only decode in some frames (blur, angle, noise). The QR text
read in one frame is attached to that badge's box; in later frames, a badge box
overlapping it (same badge, moved a little) keeps the result for `hold_s`
seconds even if that frame's decode failed. Optionally `confirm` reads of the
same text are required before a result is reported (extra safety).

    tracker = BadgeTracker()
    res = process(...)                 # one frame
    res = tracker.update(res)          # res["qr"] may now be a held result; res["qr_held"] tells
"""

import time

import numpy as np


def _box(quad):
    q = np.asarray(quad)
    return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class BadgeTracker:
    def __init__(self, hold_s=2.0, min_iou=0.3, confirm=2, persist=2, validator=None, live_s=None, gap_s=0.6):
        """confirm:   a QR text must be read this many times on the same badge before it's reported
                      (a damaged code can very rarely decode to a wrong text; it won't repeat)
           persist:   a badge box must appear in this many consecutive frames before it's shown
           validator: optional function(text) -> bool, e.g. a format check for the event's codes
           live_s:    if set, a remembered code is only reported up to live_s seconds after its last real
                      read (live mode: what you see is what is being read now); None = remember while in view"""
        self.hold_s, self.min_iou, self.confirm = hold_s, min_iou, confirm
        self.persist, self.validator, self.live_s = persist, validator, live_s
        # gap_s: a badge out of sight longer than this is treated as possibly a different badge when a box
        # shows up in its place again: its remembered code is dropped (someone else stepped into the spot)
        self.gap_s = gap_s
        self.tracks = []          # dicts: box, text, last_read, count
        self.badge_hist = []      # [(box, consecutive_frames, last_time)]

    def _persist(self, res, now):
        """Hide badge boxes until they've been seen in `persist` consecutive frames."""
        new_hist, shown, pending = [], [], []
        for b in res["badges"]:
            box = _box(b[0])
            prev = max(((_iou(h[0], box), h) for h in self.badge_hist if now - h[2] <= 0.5), default=(0, None),
                       key=lambda x: x[0])
            n = prev[1][1] + 1 if prev[0] >= self.min_iou else 1
            new_hist.append((box, n, now))
            (shown if n >= self.persist else pending).append(b)
        self.badge_hist = new_hist
        res["badges"], res["badges_pending"] = shown, pending

    def update(self, res, now=None):
        now = time.time() if now is None else now
        self.tracks = [t for t in self.tracks if now - t["last_seen"] <= self.hold_s]
        if self.persist > 1:
            self._persist(res, now)
        if res["qr"] and self.validator is not None and not self.validator(res["qr"][0]):
            res["qr_rejected"], res["qr"] = res["qr"][0], None      # not a valid code for this event
        boxes = [_box(b[0]) for b in res["badges"] + res.get("badges_pending", [])]

        # match every badge box of this frame to an existing track
        for t in self.tracks:
            best = max(((_iou(t["box"], b), i) for i, b in enumerate(boxes)), default=(0, -1))
            if best[0] >= self.min_iou:
                if now - t["last_seen"] > self.gap_s and t.get("missed", 0) >= 3:
                    t.update(votes={}, text=None, count=0)
                t["box"], t["last_seen"], t["missed"] = boxes[best[1]], now, 0
            else:
                t["missed"] = t.get("missed", 0) + 1     # frames missed (a slow frame rate is no gap)

        res["qr_held"], res["qr_box"] = False, None
        if res["qr"]:
            text, pts, dist = res["qr"]
            box = _box(pts) if pts is not None else None
            # attach the read to the badge box that contains it
            owner = None
            if box is not None:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                owner = next((b for b in boxes if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]), None)
            owner = owner or box or (boxes[0] if len(boxes) == 1 else None)   # unsure which badge: don't guess
            if owner is None:            # nothing to attach the read to: report it, don't track
                return res
            track = next((t for t in self.tracks if _iou(t["box"], owner) >= self.min_iou), None)
            if track is None:
                track = {"box": owner, "votes": {}}
                self.tracks.append(track)
            # vote: the badge's code is the text read most often; a one-off misread can't win. Counts are
            # capped (confirm + 1) and ties go to the latest read, so when a different badge takes this
            # spot its code wins after a few real reads
            v = track["votes"].get(text, (0, 0.0))
            track["votes"][text] = (min(v[0] + 1, self.confirm + 1), now)
            track.update(box=owner, last_read=now, last_seen=now, dist=dist)
            best, (n, _) = max(track["votes"].items(), key=lambda kv: kv[1])
            track["text"], track["count"] = best, n
            res["qr_box"] = owner                            # the badge this code belongs to
            if n < self.confirm:
                res["qr"] = None                             # not confirmed yet
            elif best != text:                               # this frame misread: report the confirmed code
                res["qr"], res["qr_held"] = (best, None, dist), True
            return res

        # no read this frame: report a recent read of a badge that's still in view
        live = [t for t in self.tracks if t.get("text") and t["count"] >= self.confirm and t["last_seen"] == now
                and (self.live_s is None or now - t["last_read"] <= self.live_s)]
        if live:
            t = max(live, key=lambda t: t["last_read"])
            res["qr"] = (t["text"], None, t.get("dist"))
            res["qr_held"], res["qr_box"] = True, t["box"]
        return res
