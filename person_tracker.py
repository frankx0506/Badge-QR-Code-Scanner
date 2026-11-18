"""Person tracking for the badge scanner: who is here, who is coming to the robot, whose badge is whose.

Design:
  * Track PEOPLE (not badges) with a small SORT-style tracker in 3-D: each person's position is the
    depth under the middle of their box, deprojected to metres. Matching uses predicted 3-D distance
    (gate ~0.6 m) plus box overlap, solved as an assignment (Hungarian), ByteTrack-style: confident
    detections first, weak ones only extend existing tracks. Lost tracks are kept for hold_s seconds.
  * The badge is an attribute of the person: a badge belongs to the person whose box contains it at
    the same depth (+-0.35 m); two badges can't go to one person (assignment again).
  * The QR payload is the person's identity (re-identification key). An identity seen within
    regreet_s seconds is marked "returning", so the robot doesn't greet the same visitor twice.
  * Approach states from the depth trajectory:
        FAR          > far_m away
        PRESENT      in range, not coming closer
        APPROACHING  moving toward the robot (radial speed < -0.2 m/s)
        ENGAGED      within engage_m and nearly still for 1 s (standing at the robot)
        IDENTIFIED   badge read and confirmed
        LEAVING      identified/engaged person moving away
  * priority() orders people for scanning: unidentified first, ENGAGED > APPROACHING > PRESENT > FAR,
    nearer first. The app reads QR codes, meters exposure and feeds the slow decoder in this order.
"""

import math
import time

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:                                   # tiny greedy fallback if scipy is missing
    linear_sum_assignment = None

STATE_RANK = {"ENGAGED": 0, "APPROACHING": 1, "PRESENT": 2, "FAR": 3, "LEAVING": 4, "IDENTIFIED": 5}


def _assign(cost, max_cost):
    """Min-cost matching; pairs with cost >= max_cost are dropped."""
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(np.minimum(cost, max_cost + 1))
        return [(i, j) for i, j in zip(r, c) if cost[i, j] < max_cost]
    pairs, used_r, used_c = [], set(), set()
    for k in np.argsort(cost, axis=None):
        i, j = divmod(int(k), cost.shape[1])
        if cost[i, j] >= max_cost:
            break
        if i not in used_r and j not in used_c:
            pairs.append((i, j))
            used_r.add(i)
            used_c.add(j)
    return pairs


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


def person_xyz(box, depth_mm, intr, all_candidates=False):
    """3-D position (m, camera frame) of a person in this box.

    Distance = the main depth cluster in the box: the surface with the most pixels (10 cm bins), and of
    two similar-sized surfaces the nearer one (the person stands in front of the wall). A fixed patch in
    the box centre is not used: when someone is close and fills the frame, it lands on the wall behind
    their head (people at 0.8 m were placed at 7-8 m)."""
    if depth_mm is None:
        return [] if all_candidates else None
    x1, y1, x2, y2 = box[:4]
    H, W = depth_mm.shape[:2]
    h = y2 - y1
    step = max(2, int(min(x2 - x1, h) // 60))                      # ~60 x 60 samples whatever the size
    patch = depth_mm[max(0, int(y1 + 0.1 * h)):min(H, int(y2 - 0.1 * h)):step, max(0, int(x1)):min(W, int(x2)):step]
    ok = (patch > 150) & (patch < 9000)
    vals = patch[ok] / 1000.0
    if vals.size < 30:
        return [] if all_candidates else None
    # pixels near the box's vertical centre line count more: that's where the person's own body is
    # (a held badge / box, an arm, or a person passing in front cover only part of it)
    pw = patch.shape[1]
    wx = 1.0 - np.abs(np.arange(pw) - (pw - 1) / 2) / (pw / 2 + 1e-6)
    wts = np.broadcast_to(0.15 + wx, patch.shape)[ok]
    hist, edges = np.histogram(vals, bins=np.arange(0.15, 9.05, 0.1), weights=wts)
    hist = np.convolve(hist, [1, 2, 1], mode="same")
    # candidate surfaces, best first: the biggest (of two near-equal ones, the nearer), then the others
    peaks = [k for k in range(len(hist)) if hist[k] > 0 and hist[k] >= hist[max(0, k - 1)] and
             hist[k] >= hist[min(len(hist) - 1, k + 1)] and hist[k] >= 0.25 * hist.max()]
    top = max(hist[k] for k in peaks)
    peaks.sort(key=lambda k: (0 if hist[k] >= 0.85 * top else 1, k if hist[k] >= 0.85 * top else -hist[k]))
    out = []
    for k in peaks[:3]:
        c = (edges[k] + edges[k + 1]) / 2
        if any(abs(c - o[2]) < 0.3 for o in out):
            continue
        sel = vals[np.abs(vals - c) < 0.2]
        z = float(np.median(sel)) if sel.size else float(c)
        ys, xs = np.nonzero((patch > 0) & (np.abs(patch / 1000.0 - z) < 0.2))
        u = x1 + (xs.mean() * step if xs.size else (x2 - x1) / 2)
        v = max(0, int(y1 + 0.1 * h)) + (ys.mean() * step if ys.size else 0.4 * h)
        out.append(np.array([(u - intr.ppx) * z / intr.fx, (v - intr.ppy) * z / intr.fy, z]))
    if not all_candidates:
        return out[0]
    return out


def plausible_person(box, xyz, intr, min_m=0.3, thin_m=0.1, max_h_m=2.3, max_w_m=1.6):
    """Is this box, at its measured distance, the real size of a (part of a) person?
    Rejects hands / arms / objects (a hand on a mouse is ~5 x 10 cm) and boxes whose depth is the wall
    behind them (a box 566 px tall at 7.7 m would be a 3 m tall person). No depth -> can't judge -> True."""
    if xyz is None or intr is None:
        return True
    z = float(xyz[2])
    w_m = (box[2] - box[0]) * z / intr.fx
    h_m = (box[3] - box[1]) * z / intr.fy
    return max(w_m, h_m) >= min_m and min(w_m, h_m) >= thin_m and h_m <= max_h_m and w_m <= max_w_m


class Person:
    _next = 1

    def __init__(self, box, conf, xyz, now):
        self.id = Person._next
        Person._next += 1
        self.box, self.conf, self.xyz = tuple(box[:4]), conf, xyz
        self.vel = np.zeros(3)
        self.v_radial = 0.0
        self.first_seen = self.last_seen = now
        self.hits = 1
        self.state = "PRESENT"
        self.still_since = None
        self.identity = None
        self.identified_at = None
        self.name = None                             # printed name read off the badge (OCR)
        self.returning = False                       # identity was seen recently (don't re-greet)
        self.badge = None                            # (quad, dist) of this person's badge, this frame
        self.badge_seen = None

    @property
    def dist(self):
        return float(np.linalg.norm(self.xyz)) if self.xyz is not None else None

    def predict(self, now):
        if self.xyz is None:
            return None
        return self.xyz + self.vel * min(now - self.last_seen, 0.5)

    def to_dict(self, now):
        return {"id": self.id, "state": self.state, "dist": round(self.dist, 2) if self.dist else None,
                "speed": round(float(-self.v_radial), 2), "identity": self.identity, "name": self.name, "returning": self.returning,
                "has_badge": self.badge is not None, "age_s": round(now - self.first_seen, 1),
                "box": [round(float(v)) for v in self.box]}


class PersonTracker:
    def __init__(self, hold_s=2.5, gate_m=0.6, high_conf=0.5, min_hits=2, far_m=2.5, engage_m=1.2,
                 approach_mps=0.2, still_mps=0.3, engage_after_s=1.0, regreet_s=300.0):
        self.hold_s, self.gate_m, self.high_conf, self.min_hits = hold_s, gate_m, high_conf, min_hits
        self.far_m, self.engage_m, self.approach_mps = far_m, engage_m, approach_mps
        self.still_mps, self.engage_after_s, self.regreet_s = still_mps, engage_after_s, regreet_s
        self.tracks = []
        self.rejected = []                            # last frame's detections that failed the size check
        self.seen = {}                                # identity -> last time confirmed (re-greet memory)
        self.names = {}                               # identity (QR code) -> printed name
        self.events = []                              # (time, text) for the UI log

    # ------------------------------------------------------------------ people
    def update(self, detections, depth_mm, intr, now=None):
        """detections: [(x1, y1, x2, y2, conf)] from the person detector (this frame)."""
        now = time.time() if now is None else now
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.hold_s]
        dets, self.rejected = [], []
        for d in detections:
            # the first depth surface in the box that gives a person-sized box (a badge / box held up close
            # in front of the body is skipped); none -> not a person (a hand on a mouse, an arm)
            cands = person_xyz(d, depth_mm, intr, all_candidates=True) if depth_mm is not None else []
            xyz = next((c for c in cands if plausible_person(d, c, intr)), None)
            if cands and xyz is None:
                self.rejected.append(d)
            else:
                dets.append((d, xyz))
        high = [k for k, (d, _) in enumerate(dets) if d[4] >= self.high_conf]
        low = [k for k, (d, _) in enumerate(dets) if d[4] < self.high_conf]
        unmatched_tracks = list(range(len(self.tracks)))
        matched = {}
        for group in (high, low):                     # ByteTrack: confident first, weak ones second
            if not group or not unmatched_tracks:
                continue
            cost = np.full((len(unmatched_tracks), len(group)), 9.0)
            for a, ti in enumerate(unmatched_tracks):
                t = self.tracks[ti]
                pred = t.predict(now)
                for b, di in enumerate(group):
                    d, xyz = dets[di]
                    iou = _iou(t.box, d)
                    if pred is not None and xyz is not None:
                        dist3 = float(np.linalg.norm(pred - xyz))
                        if dist3 < self.gate_m:
                            cost[a, b] = dist3 / self.gate_m + 0.5 * (1 - iou)
                    elif iou > 0.2:
                        cost[a, b] = 1.5 * (1 - iou)
            for a, b in _assign(cost, 2.0):
                matched[unmatched_tracks[a]] = group[b]
            unmatched_tracks = [ti for ti in unmatched_tracks if ti not in matched]
        for ti, di in matched.items():
            self._refresh(self.tracks[ti], *dets[di], now)
        used = set(matched.values())
        for di in high:                               # new people only from confident detections
            if di not in used:
                d, xyz = dets[di]
                self.tracks.append(Person(d, d[4], xyz, now))
        for t in self.tracks:
            t.badge = None if now - (t.badge_seen or -1e9) > 0.5 else t.badge
            self._state(t, now)
        return self.visible(now)

    def _refresh(self, t, d, xyz, now):
        dt = max(now - t.last_seen, 1e-3)
        if xyz is not None and t.xyz is not None and dt < 1.0:
            v = (xyz - t.xyz) / dt
            t.vel = 0.6 * t.vel + 0.4 * v
            r_new, r_old = float(np.linalg.norm(xyz)), float(np.linalg.norm(t.xyz))
            t.v_radial = 0.6 * t.v_radial + 0.4 * (r_new - r_old) / dt
        if xyz is not None:
            t.xyz = xyz
        t.box, t.conf, t.last_seen = tuple(d[:4]), d[4], now
        t.hits += 1

    def _state(self, t, now):
        r = t.dist
        speed = float(np.linalg.norm(t.vel))
        if speed < self.still_mps:
            t.still_since = t.still_since or now
        else:
            t.still_since = None
        prev = t.state
        if t.identity and t.v_radial > self.approach_mps and (r is None or r > self.engage_m):
            t.state = "LEAVING"
        elif t.identity:
            t.state = "IDENTIFIED"
        elif r is not None and r > self.far_m:
            t.state = "FAR"
        elif r is not None and r <= self.engage_m and t.still_since and now - t.still_since >= self.engage_after_s:
            t.state = "ENGAGED"
        elif t.v_radial < -self.approach_mps:
            t.state = "APPROACHING"
        elif prev == "ENGAGED" and r is not None and r <= self.engage_m * 1.15:
            t.state = "ENGAGED"                       # hysteresis: small shifts don't drop engagement
        else:
            t.state = "PRESENT"

    def visible(self, now):
        """Confirmed tracks seen in the last frame or so (tentative ones need min_hits)."""
        return [t for t in self.tracks if t.hits >= self.min_hits and now - t.last_seen <= 0.5]

    # ------------------------------------------------------------------ badges and identities
    def owner(self, box, dist=None, now=None):
        """The visible person a badge box (x1, y1, x2, y2) belongs to, or None."""
        now = time.time() if now is None else now
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best, best_cost = None, 9.0
        for t in self.visible(now):
            x1, y1, x2, y2 = t.box
            mx = 0.05 * (x2 - x1)
            if not (x1 - mx <= cx <= x2 + mx and y1 <= cy <= y2):
                continue
            dz = abs(dist - t.xyz[2]) if (dist is not None and t.xyz is not None) else 0.0
            if dz > 0.35:
                continue
            cost = dz + abs(cx - (x1 + x2) / 2) / max(x2 - x1, 1)
            if cost < best_cost:
                best, best_cost = t, cost
        return best

    def attach_badges(self, badges, now=None):
        """badges: [(quad (4x2), dist)]. One badge per person, one person per badge."""
        now = time.time() if now is None else now
        people = self.visible(now)
        if not badges or not people:
            return
        cost = np.full((len(badges), len(people)), 9.0)
        for i, (quad, dist) in enumerate(badges):
            q = np.asarray(quad)
            box = (q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max())
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            for j, t in enumerate(people):
                x1, y1, x2, y2 = t.box
                mx = 0.05 * (x2 - x1)
                if not (x1 - mx <= cx <= x2 + mx and y1 <= cy <= y2):
                    continue
                dz = abs(dist - t.xyz[2]) if (dist is not None and t.xyz is not None) else 0.0
                if dz <= 0.35:
                    cost[i, j] = dz + abs(cx - (x1 + x2) / 2) / max(x2 - x1, 1)
        for i, j in _assign(cost, 9.0):
            people[j].badge, people[j].badge_seen = badges[i], now

    def identify(self, text, qr_box, dist=None, now=None):
        """A confirmed QR read at qr_box: give the identity to its owner. Returns the person or None."""
        now = time.time() if now is None else now
        t = self.owner(qr_box, dist, now)
        if t is None:
            return None
        if t.identity != text:
            # the same identity can't be two people: take it from any other (older) track
            for o in self.tracks:
                if o is not t and o.identity == text:
                    o.identity, o.name = None, None
            last = self.seen.get(text)
            t.returning = last is not None and now - last < self.regreet_s and (t.identity is None)
            t.identity, t.identified_at = text, now
            t.name = self.names.get(text)             # name already read for this code (memory), else none yet
            self.events.append((now, f"person {t.id}: {text}" + (" (returning)" if t.returning else "")))
            self.events = self.events[-20:]
        self.seen[text] = now
        return t

    def set_name(self, name, badge_box, dist=None, now=None):
        """A confirmed badge name at badge_box: give it to the badge's owner (and remember it by code)."""
        t = self.owner(badge_box, dist, now)
        if t is None:
            return None
        if t.name != name:
            t.name = name
            self.events.append((time.time() if now is None else now, f"person {t.id}: name {name}"))
            self.events = self.events[-20:]
        if t.identity:
            self.names[t.identity] = name
        return t

    def priority(self, t):
        """Sort key: who to scan first."""
        return (t.identity is not None, STATE_RANK.get(t.state, 9), t.dist if t.dist is not None else 9.0)

    def badge_priority(self, box, dist=None, now=None):
        """Sort key for a badge box via its owner (badges with no owner go after owned ones)."""
        t = self.owner(box, dist, now)
        return self.priority(t) if t is not None else (False, 8, dist if dist is not None else 9.0)
