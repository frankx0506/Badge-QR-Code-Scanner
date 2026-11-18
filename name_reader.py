"""Reads the attendee's printed name off a badge crop (offline OCR: PP-OCR via RapidOCR on onnxruntime).

    nr = NameReader()                      # models from models/ocr/, GPU if available
    name, conf, lines = nr.read(badge_bgr, qr_quad=None)

How the name is picked from all text on the badge:
  * OCR every text line (detector + recogniser).
  * Keep "name-like" lines: mostly letters, no digits, confident, not an event / badge-type / job-title
    word (ROBOBUSINESS, CES, EXPO ONLY, ATTENDEE, CEO, ...), and above the QR code when it is known.
  * The name is the tallest such line plus the line right under it when that line is nearly as tall
    (first name / last name on two lines). Company and title lines are printed smaller.

Names are text, so they read much further than the QR code (letters ~5-8 mm tall vs 0.86 mm QR
modules), so a name often reads where the QR code no longer does.
"""
import os
import re
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "ocr")

# words that are printed on badges but are never part of a person's name
STOP = {
    # events / organisers
    "CES", "CEA", "ROBOBUSINESS", "ROBO", "BUSINESS", "EXPO", "SUMMIT", "CONFERENCE", "WORLD",
    "SHOW", "LAS", "VEGAS", "CTA", "CONSUMER", "TECHNOLOGY", "ASSOCIATION", "INNOVATION", "AWARDS",
    # badge types / access
    "ONLY", "ATTENDEE", "EXHIBITOR", "EXHIBIT", "MEDIA", "PRESS", "SPEAKER", "STAFF", "VIP", "BADGE",
    "ACCESS", "PASS", "VISITOR", "GUEST", "ALL", "FULL", "DAY", "HALL", "BOOTH", "INDUSTRY", "BUYER",
    "DELEGATE", "SPONSOR", "PARTNER", "ANALYST", "STUDENT", "CONTRACTOR", "SECURITY", "VOLUNTEER",
    "ORGANIZER", "ADMIT", "ONE", "REGISTRATION", "MEMBER", "PREMIUM", "PLATINUM", "GOLD", "SILVER",
    "SUITES", "SUITE", "EVENT", "NOT", "TRANSFERABLE", "VALID", "PHOTO", "ID", "REQUIRED",
    # job titles
    "CEO", "CTO", "CFO", "COO", "CMO", "CIO", "VP", "SVP", "EVP", "PRESIDENT", "FOUNDER", "COFOUNDER",
    "CO-FOUNDER", "DIRECTOR", "MANAGER", "ENGINEER", "ADVISOR", "ADVISER", "OWNER",
    "CONSULTANT", "SALES", "MARKETING", "HEAD", "LEAD", "SENIOR", "PRINCIPAL", "CHIEF", "OFFICER",
    "EXECUTIVE", "SCIENTIST", "RESEARCHER", "PROFESSOR", "DEVELOPER", "DESIGNER", "ARCHITECT",
    "SPECIALIST", "COORDINATOR", "ASSOCIATE", "INTERN", "EDITOR", "REPORTER", "WRITER",
    # company / place words
    "UNIVERSITY", "COLLEGE", "INSTITUTE", "INC", "LLC", "LTD", "CORP", "CORPORATION", "COMPANY", "CO",
    "GROUP", "TECHNOLOGIES", "TECHNOLOGY", "SYSTEMS", "SOLUTIONS", "LABS", "ROBOTICS", "GMBH",
    "USA", "UNITED", "STATES", "CHINA", "JAPAN", "KOREA", "CANADA", "GERMANY", "FRANCE", "KINGDOM",
}
# letters of any script (Latin, accented, CJK, Hangul, Cyrillic, ...), plus ' . - and spaces; no digits
_NAMECH = re.compile(r"^[^\W\d_](?:[^\W\d_]|['’.\- ])*$")


def _stop_word(w):
    """A badge word that is never a name. Also catches a logo mark read as a letter stuck to the word
    ('©CES' -> 'OCES', 'CCES')."""
    w = w.strip("'’.")
    return w in STOP or (len(w) >= 3 and w[0] in "OCGQ@©" and w[1:] in STOP)


def _clean(t):
    return re.sub(r"\s+", " ", t.strip().strip(".,:;|_-*•·")).strip()


def name_like(text, score, min_score=0.8):
    if "..." in text or "…" in text:                     # truncated / partly read text
        return False
    t = _clean(text)
    if score < min_score or len(t.replace(" ", "")) < 2 or not _NAMECH.match(t):
        return False
    words = re.split(r"[ \-]", t.upper())
    if any(_stop_word(w) for w in words if w):
        return False
    if "BUSINESS" in t.upper() or "ROBO" in t.upper():
        return False
    return len(words) <= 4


def pick_name(lines, qr_top=None):
    """lines: [(text, score, box 4x2)] in crop coords -> (name, conf) or (None, 0).

    The name is the TOP line among the big name-like lines (within 75% of the tallest): on badges the
    name comes first and the company / country under it, and a company can be printed as big as the
    name. A last name on the next line is added when it is close below, aligned, and not much smaller
    (a name already on one line with 2+ words only takes a line that is nearly as tall)."""
    cand = []
    for text, score, box in lines:
        box = np.asarray(box, np.float32).reshape(4, 2)
        h = float(np.linalg.norm(box[3] - box[0]) + np.linalg.norm(box[2] - box[1])) / 2
        cy, cx = float(box[:, 1].mean()), float(box[:, 0].mean())
        if qr_top is not None and cy > qr_top:
            continue
        if name_like(text, score):
            cand.append({"t": _clean(text), "s": float(score), "h": h, "cy": cy, "cx": cx,
                         "w": float(np.ptp(box[:, 0]))})
    if not cand:
        return None, 0.0
    hmax = max(c["h"] for c in cand)
    first = min((c for c in cand if c["h"] >= 0.75 * hmax), key=lambda c: c["cy"])
    picked = [first]
    below = sorted((c for c in cand if c["cy"] > first["cy"]), key=lambda c: c["cy"])
    if below:
        c = below[0]
        need = 0.9 if len(first["t"].split()) >= 2 else 0.6
        if (c["h"] >= need * first["h"] and c["h"] <= 1.4 * first["h"]
                and c["cy"] - first["cy"] <= 2.2 * max(first["h"], c["h"])
                and abs(c["cx"] - first["cx"]) <= 0.5 * max(first["w"], c["w"]) + first["h"]):
            picked.append(c)
    name = " ".join(c["t"] for c in picked).upper()
    return name, float(min(c["s"] for c in picked))


def _no_spin():
    """OCR sessions: don't busy-wait between runs (spinning threads would take CPU from the camera loop)."""
    try:
        from rapidocr.inference_engine.onnxruntime import main as m
        cls = next(v for v in vars(m).values() if isinstance(v, type) and hasattr(v, "_init_sess_opts"))
        if getattr(cls, "_no_spin", False):
            return
        orig = cls._init_sess_opts

        def init(cfg, _orig=orig):
            so = _orig(cfg)
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            return so
        cls._init_sess_opts, cls._no_spin = staticmethod(init), True
    except Exception as e:
        print("name OCR: could not turn off thread spinning:", e)


def _cuda_ok():
    """True only if onnxruntime can really run on an NVIDIA GPU (driver loaded, CUDA libs found)."""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers() or not os.path.exists("/dev/nvidia0"):
            return False
        try:
            ort.preload_dlls()          # CUDA/cuDNN from the pip nvidia-* packages
        except Exception:
            pass
        return True
    except Exception:
        return False


class NameReader:
    """OCR engine wrapper. read() is thread-safe (one call at a time)."""

    def __init__(self, size="small", gpu=True, threads=4, det_side=640, model_dir=MODEL_DIR):
        from rapidocr import RapidOCR
        _no_spin()
        cuda = gpu and _cuda_ok()
        params = {"Global.log_level": "error", "Global.use_cls": False,
                  "EngineConfig.onnxruntime.use_cuda": cuda,
                  # a few threads only: OCR runs beside the camera loop and the detectors
                  "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                  "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                  "Det.limit_side_len": det_side}
        det = os.path.join(model_dir, f"PP-OCRv6_det_{size}.onnx")
        rec = os.path.join(model_dir, f"PP-OCRv6_rec_{size}.onnx")
        if os.path.exists(det) and os.path.exists(rec):
            params.update({"Det.model_path": det, "Rec.model_path": rec})
        self.engine = RapidOCR(params=params)
        self.device = "GPU" if cuda else "CPU"
        self._lock = threading.Lock()
        self.ms = 0.0

    def ocr(self, bgr):
        with self._lock:
            t = time.perf_counter()
            r = self.engine(bgr)
            self.ms = (time.perf_counter() - t) * 1000
        if r is None or r.txts is None:
            return []
        return list(zip(r.txts, r.scores, r.boxes))

    def read(self, bgr, qr_quad=None):
        """bgr: badge crop. qr_quad: QR corners in crop coords (text below the QR is ignored)."""
        if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 24:
            return None, 0.0, []
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        lines = self.ocr(bgr)
        qr_top = None if qr_quad is None else float(np.min(np.asarray(qr_quad)[:, 1]))
        name, conf = pick_name(lines, qr_top)
        return name, conf, lines


class NameVoter:
    """Per-badge votes over several reads; a name is confirmed when it was read `need` times."""

    def __init__(self, need=2):
        self.need, self.votes = need, {}

    def add(self, name, conf):
        if name:
            v = self.votes.setdefault(name, [0, 0.0])
            v[0] += 1
            v[1] = max(v[1], conf)

    def best(self):
        if not self.votes:
            return None, 0
        name, (n, _) = max(self.votes.items(), key=lambda kv: (kv[1][0], kv[1][1]))
        return name, n

    def confirmed(self):
        name, n = self.best()
        return name if n >= self.need else None


def _bbox(quad):
    q = np.asarray(quad)
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class NameService:
    """Reads badge names in a background thread so the camera loop never waits for OCR.

    Every frame: svc.step(color, badges, qr_quad) with the frame's badges [(quad, dist, ...)] and the QR
    corners read THIS frame (or None), then svc.name_for(quad) -> confirmed name of that badge or None.

      * Each badge is followed by its box (IoU). A badge that was out of sight for more than `gap_s`
        is a new badge: its name state starts over (someone else may have stepped into the spot).
      * A name is shown after `need` agreeing reads in a row. A confirmed badge is re-read every
        `recheck_s`; a read that disagrees hides the name at once (never show a possibly wrong name)
        and the badge is re-read quickly: an agreeing read restores it, `need` agreeing reads of a new
        name replace it.
      * Unreadable badges back off (0.3 s, doubling to 3 s); badges beyond `max_dist` are skipped.
    """

    def __init__(self, reader=None, need=2, hold_s=2.0, recheck_s=3.0, gap_s=0.4, min_iou=0.3, margin=0.06,
                 max_dist=1.3, **kw):
        self.reader = reader or NameReader(**kw)
        self.need, self.hold_s, self.recheck_s, self.gap_s = need, hold_s, recheck_s, gap_s
        self.min_iou, self.margin, self.max_dist = min_iou, margin, max_dist
        self.tracks = []
        self._job, self._done, self._running = None, [], False
        self._lock, self._ev = threading.Lock(), threading.Event()
        self.reads = 0
        threading.Thread(target=self._loop, daemon=True).start()

    # ---- background OCR
    def _loop(self):
        while True:
            self._ev.wait()
            self._ev.clear()
            with self._lock:
                job, self._job = self._job, None
                self._running = job is not None
            if job is None:
                continue
            crop, track, qr = job
            try:
                name, conf, _ = self.reader.read(crop, qr)
            except Exception as e:             # never kill the thread
                print("name OCR:", e)
                name, conf = None, 0.0
            with self._lock:
                self._done.append((track, name, conf))
                self._running = False
                self.reads += 1

    @property
    def busy(self):
        with self._lock:
            return self._job is not None or self._running

    # ---- per frame (camera loop)
    def _new_track(self, box, dist, now):
        return {"box": box, "dist": dist, "last_seen": now, "last_try": 0.0, "name": None, "doubt": False,
                "cand": None, "streak": 0, "fails": 0, "missed": 0}

    @staticmethod
    def _reset(t):
        t.update(name=None, doubt=False, cand=None, streak=0, fails=0, last_try=0.0)

    def _track(self, box, now=None):
        best = max(((_iou(t["box"], box), t) for t in self.tracks), default=(0, None), key=lambda x: x[0])
        return best[1] if best[0] >= self.min_iou else None

    def _result(self, t, name):
        if not name:
            t["fails"] += 1
            return
        t["fails"] = 0
        if name == t["name"]:                          # confirmed again
            t["doubt"], t["cand"], t["streak"] = False, None, 0
            return
        if t["name"] is not None:
            t["doubt"] = True                          # disagrees: hide the name until settled
        if name == t["cand"]:
            t["streak"] += 1
        else:
            t["cand"], t["streak"] = name, 1
        if t["streak"] >= self.need:                   # `need` agreeing reads in a row
            t["name"], t["doubt"], t["cand"], t["streak"] = name, False, None, 0

    def step(self, color, badges, qr_quad=None, now=None):
        now = time.time() if now is None else now
        with self._lock:
            done, self._done = self._done, []
            idle = self._job is None and not self._running
        self.tracks = [t for t in self.tracks if now - t["last_seen"] <= self.hold_s]
        seen = []
        for b in badges:                                   # follow badges
            box = _bbox(b[0])
            t = self._track(box)
            if t is None:
                t = self._new_track(box, b[1], now)
                self.tracks.append(t)
            elif now - t["last_seen"] > self.gap_s and t["missed"] >= 3:
                self._reset(t)                             # gone for a while: maybe someone else now
            t.update(box=box, last_seen=now, dist=b[1], missed=0)
            seen.append(t)
        for t in self.tracks:                              # frames missed (a slow frame rate is no gap)
            if not any(t is u for u in seen):
                t["missed"] += 1
        for t, name, conf in done:                         # results -> the track they were read for
            if any(t is u for u in self.tracks):
                self._result(t, name)
        if not idle or not badges or color is None:
            return
        live = [t for t in self.tracks if t["last_seen"] == now
                and (t["dist"] is None or t["dist"] <= self.max_dist)]

        def wait(t):
            if t["name"] is not None and not t["doubt"]:
                return self.recheck_s
            return min(3.0, 0.3 * 2 ** min(t["fails"], 4))
        due = [t for t in live if now - t["last_try"] >= wait(t)]
        if not due:
            return
        # unnamed / doubtful first, then nearest
        t = min(due, key=lambda t: (t["name"] is not None and not t["doubt"], t["dist"] if t["dist"] else 9.0))
        x1, y1, x2, y2 = t["box"]
        H, W = color.shape[:2]
        mx, my = self.margin * (x2 - x1), self.margin * (y2 - y1)
        X1, Y1, X2, Y2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(W, x2 + mx)), int(min(H, y2 + my))
        if X2 - X1 < 24 or Y2 - Y1 < 24:
            return
        qr_local = None
        if qr_quad is not None:
            q = np.asarray(qr_quad, np.float32).reshape(-1, 2)
            c = q.mean(axis=0)
            qa = float(np.ptp(q[:, 0]) * np.ptp(q[:, 1]))
            # real QR corners only: a whole-badge quad (from the slow decoders) would hide every line
            if X1 <= c[0] <= X2 and Y1 <= c[1] <= Y2 and qa < 0.5 * (x2 - x1) * (y2 - y1):
                qr_local = q - [X1, Y1]
        t["last_try"] = now
        with self._lock:
            self._job = (color[Y1:Y2, X1:X2].copy(), t, qr_local)
        self._ev.set()

    def name_for(self, quad):
        t = self._track(_bbox(quad))
        return t["name"] if t is not None and not t["doubt"] else None
