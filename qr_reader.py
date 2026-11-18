"""Robust QR reading for small / slightly blurry codes.

read_qr_in_badge(crop)  -> (text, corners)   QR inside a badge crop (main path)
read_qr(img)            -> (text, corners)   QR anywhere in a full frame (QR-only mode)

corners is a 4x2 float array in the input image's coordinates, or None.

The badge cascade tries progressively heavier image enhancements and stops at
the first read; every result still has to pass the QR's own error correction.
"""

import time

import cv2
import numpy as np
import zxingcpp

_cv = cv2.QRCodeDetector()
_cva = cv2.QRCodeDetectorAruco()
_QR = zxingcpp.BarcodeFormat.QRCode
_BOOL = zxingcpp.Binarizer.BoolCast


def _pos(p):
    return np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                     [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]], np.float32)


def _zxing(gray, **kw):
    for r in zxingcpp.read_barcodes(gray, formats=_QR, **kw):
        if r.text:
            return r.text, _pos(r.position)
    return None, None


def _sharpen(gray, amount=0.8, sigma=2.0):
    blur = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def _up(gray, f):
    return cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def _adaptive(gray):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5)


def locate_qr(gray):
    """Corners of a QR code even when it can't be decoded (zxing, then OpenCV)."""
    for r in zxingcpp.read_barcodes(gray, formats=_QR, return_errors=True):
        return _pos(r.position)
    for d in (_cv, _cva):
        ok, pts = d.detect(gray)
        if ok and pts is not None:
            return pts.reshape(4, 2).astype(np.float32)
    return None


def read_qr_in_badge(crop, budget_ms=None):
    """Decode the QR inside a badge crop with the enhancement cascade.
    budget_ms: optional time limit (e.g. on a Raspberry Pi); stops early when exceeded."""
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    over = lambda: budget_ms is not None and (time.perf_counter() - t0) * 1000 > budget_ms

    # 1-4: whole badge crop, as is / enlarged + sharpened
    text, pts = _zxing(gray, try_downscale=False)
    if text:
        return text, pts
    for f, amt, sig in ((1.5, 0.8, 2.0), (2, 1.5, 2.5), (3, 1.5, 2.5)):
        if over():
            return None, None
        text, pts = _zxing(_sharpen(_up(gray, f), amt, sig))
        if text:
            return text, pts / f

    # 5-8: locate the QR (even undecodable), crop tight, enlarge a lot, threshold / sharpen
    if over():
        return None, None
    loc = locate_qr(gray)
    if loc is None:
        return None, None
    x0, y0 = loc.min(0)
    x1, y1 = loc.max(0)
    m = 0.35 * max(x1 - x0, y1 - y0)
    h, w = gray.shape
    ox, oy = int(max(0, x0 - m)), int(max(0, y0 - m))
    tight = gray[oy:int(min(h, y1 + m)), ox:int(min(w, x1 + m))]
    if tight.size < 100:
        return None, None
    steps = ((3, "adaptive"), (4, "sharp"), (6, "sharp"), (4, "adaptive"))
    for f, kind in steps:
        if over():
            return None, None
        big = _up(tight, f)
        if kind == "adaptive":
            text, pts = _zxing(_adaptive(_sharpen(big)), binarizer=_BOOL)
        else:
            text, pts = _zxing(_sharpen(big, 1.5, 2.5))
        if text:
            return text, pts / f + np.array([ox, oy], np.float32)
    return None, None


def read_qr(img, upscale=3):
    """QR anywhere in a full frame (QR-only mode)."""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text, pts = _zxing(gray)
    if text:
        return text, pts

    ok, pts = _cv.detect(gray)
    if not ok or pts is None:
        return None, None
    text = _cv.decode(gray, pts)[0]
    if text:
        return text, pts.reshape(4, 2)

    # Close-up retry around the located QR, using the badge cascade on it.
    pts = pts.reshape(4, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    m = 1.5 * max(x1 - x0, y1 - y0)
    h, w = gray.shape
    x0, y0 = int(max(x0 - m, 0)), int(max(y0 - m, 0))
    x1, y1 = int(min(x1 + m, w)), int(min(y1 + m, h))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None, None
    t, p = read_qr_in_badge(crop)
    if t:
        return t, (p + np.array([x0, y0], np.float32)) if p is not None else None
    return None, None
