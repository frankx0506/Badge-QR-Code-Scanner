"""Any-QR scanner: finds QR codes anywhere in the frame, not only on badges (product boxes, signs,
phones, flyers). Runs in a background thread on the newest frame, so the camera loop never waits.

Passes per frame (stops adding work once `budget_ms` is spent):
 1. zxing on the whole frame (all codes, ~3 ms on 1080p).
 2. zxing on the frame enlarged 2x and sharpened - small codes (~1.5-2.5 px per square).
 3. Codes that were LOCATED but not read (zxing errors, OpenCV finder) get the badge cascade
    (tight crop, enlarge, sharpen, threshold) on a crop around them.

These codes are shown and logged as "other QR"; they never become a person's identity
(only a QR on a detected badge does).

    sc = FreeQRScanner(); sc.submit(gray_frame); ... sc.current() -> [(text, quad), ...]
"""
import threading
import time

import cv2
import numpy as np
import zxingcpp

from qr_reader import _pos, read_qr_in_badge, _sharpen

_QR = zxingcpp.BarcodeFormat.QRCode | zxingcpp.BarcodeFormat.MicroQRCode
_cv = cv2.QRCodeDetector()


def _centre(q):
    return q.mean(axis=0)


def scan(gray, budget_ms=150):
    """All QR codes in a grey frame -> [(text, quad 4x2 float32)]."""
    t0 = time.perf_counter()
    left = lambda: budget_ms - (time.perf_counter() - t0) * 1000
    found, located = [], []

    def add(text, quad):
        c = _centre(quad)
        for t, q in found:
            if t == text and np.linalg.norm(_centre(q) - c) < 40:
                return
        found.append((text, quad.astype(np.float32)))

    # 1. whole frame
    for r in zxingcpp.read_barcodes(gray, formats=_QR, return_errors=True):
        (add(r.text, _pos(r.position)) if r.valid and r.text else located.append(_pos(r.position)))
    # 2. enlarged 2x + sharpened (small codes)
    if left() > 0:
        big = _sharpen(cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 0.8, 2.0)
        for r in zxingcpp.read_barcodes(big, formats=_QR, return_errors=True):
            q = _pos(r.position) / 2
            (add(r.text, q) if r.valid and r.text else located.append(q))
    # 3. located but unread: OpenCV finder too, then the badge cascade around each
    if left() > 0:
        try:
            ok, pts = _cv.detectMulti(gray)
            if ok and pts is not None:
                located += [p.reshape(4, 2).astype(np.float32) for p in pts]
        except cv2.error:
            pass
    H, W = gray.shape
    done = [_centre(q) for _, q in found]
    for q in located:
        if left() <= 0:
            break
        c = _centre(q)
        size = float(np.ptp(q, axis=0).max())
        if size < 12 or any(np.linalg.norm(c - d) < max(size, 20) for d in done):
            continue
        done.append(c)
        m = 0.6 * size
        x0, y0 = int(max(0, q[:, 0].min() - m)), int(max(0, q[:, 1].min() - m))
        x1, y1 = int(min(W, q[:, 0].max() + m)), int(min(H, q[:, 1].max() + m))
        text, p = read_qr_in_badge(gray[y0:y1, x0:x1], budget_ms=min(80, max(left(), 10)))
        if text:
            add(text, (p if p is not None else q - [x0, y0]) + np.array([x0, y0], np.float32))
    return found


class FreeQRScanner:
    """Background thread: always works on the newest submitted frame; results held ~1.5 s."""

    def __init__(self, budget_ms=150, hold_s=1.5, min_interval=0.5):
        # min_interval: scan at most this often (s), so the scanner doesn't compete with the camera loop
        # (shared interpreter lock); 2 scans/s is plenty for codes on boxes, phones, signs
        self.budget_ms, self.hold_s, self.min_interval = budget_ms, hold_s, min_interval
        self._frame, self._lock, self._ev = None, threading.Lock(), threading.Event()
        self._hits = {}                       # text -> (quad, last seen)
        self.ms, self.scans = 0.0, 0
        self.enabled = True
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, gray):
        if not self.enabled:
            return
        with self._lock:
            self._frame = gray
        self._ev.set()

    def _loop(self):
        while True:
            self._ev.wait()
            self._ev.clear()
            with self._lock:
                g, self._frame = self._frame, None
            if g is None:
                continue
            t = time.perf_counter()
            try:
                res = scan(g, self.budget_ms)
            except Exception as e:            # never kill the thread
                print("any-QR scan:", e)
                res = []
            now = time.time()
            with self._lock:
                for text, quad in res:
                    self._hits[text] = (quad, now)
                self.ms = 0.8 * self.ms + 0.2 * (time.perf_counter() - t) * 1000 if self.scans else \
                    (time.perf_counter() - t) * 1000
                self.scans += 1
            rest = self.min_interval - (time.perf_counter() - t)
            if rest > 0:
                time.sleep(rest)

    def current(self):
        now = time.time()
        with self._lock:
            self._hits = {k: v for k, v in self._hits.items() if now - v[1] < self.hold_s}
            return [(k, v[0]) for k, v in self._hits.items()]
