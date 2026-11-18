"""Trained badge detector (YOLO exported to ONNX), run with onnxruntime (or OpenCV's dnn if
onnxruntime is missing). No PyTorch needed at runtime.

onnxruntime runs with a few threads and no busy-waiting, so the detectors leave CPU for the rest
of the pipeline.

    det = BadgeDetector.load()            # None if models/badge.onnx is missing
    boxes = det.detect(bgr_image)         # [(x1, y1, x2, y2, score), ...] in pixels
"""

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "badge.onnx")
THREADS = int(os.environ.get("DETECTOR_THREADS", "4"))

try:
    import onnxruntime as _ort
except Exception:                                # fall back to OpenCV's dnn
    _ort = None
    cv2.setNumThreads(6)                         # a few threads: same speed, much less CPU


def _ort_session(path):
    so = _ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = THREADS, 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")   # don't busy-wait between frames
    return _ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class BadgeDetector:
    def __init__(self, path, imgsz=640, conf=0.35, iou=0.5, classes=None):
        self.path, self.classes = path, classes      # classes: keep only these class ids (e.g. (0,) = person)
        self.sess = _ort_session(path) if _ort is not None else None
        self.net = None if self.sess is not None else cv2.dnn.readNetFromONNX(path)
        meta = path[:-5] + ".json"
        if os.path.exists(meta):
            with open(meta) as f:
                imgsz = json.load(f).get("imgsz", imgsz)
        # imgsz may be a single int (square) or [H, W] (rectangular, letterbox-free for 16:9 frames)
        self.imgh, self.imgw = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
        self.conf, self.iou = conf, iou

    @classmethod
    def load(cls, path=DEFAULT_MODEL, **kw):
        return cls(path, **kw) if os.path.exists(path) else None

    def _letterbox(self, img):
        h, w = img.shape[:2]
        s = min(self.imgw / w, self.imgh / h)        # fit inside (imgw x imgh), keep aspect
        nw, nh = int(round(w * s)), int(round(h * s))
        canvas = np.full((self.imgh, self.imgw, 3), 114, np.uint8)
        px, py = (self.imgw - nw) // 2, (self.imgh - nh) // 2
        canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, s, px, py

    def detect(self, img, with_class=False):
        """Boxes (x1, y1, x2, y2, score), best first; with_class adds the class id as a 6th value."""
        lb, s, px, py = self._letterbox(img)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, size=(self.imgw, self.imgh), swapRB=True)
        if self.sess is not None:
            out = self.sess.run(None, {self.sess.get_inputs()[0].name: blob})[0]
        else:
            self.net.setInput(blob)
            out = self.net.forward()             # (1, 4 + classes, N) for YOLOv8/11
        out = out[0]
        H, W = img.shape[:2]
        if out.ndim == 2 and out.shape[1] == 6 and out.shape[0] <= 1000:
            # YOLO26-style end-to-end head (no NMS needed): rows of x1, y1, x2, y2, score, class
            keep = out[:, 4] >= self.conf
            if self.classes is not None:
                keep &= np.isin(out[:, 5].astype(int), self.classes)
            boxes = []
            for x1, y1, x2, y2, sc, c in out[keep]:
                x1, x2 = (float(x1) - px) / s, (float(x2) - px) / s
                y1, y2 = (float(y1) - py) / s, (float(y2) - py) / s
                b = (max(0.0, x1), max(0.0, y1), min(W - 1.0, x2), min(H - 1.0, y2), float(sc))
                boxes.append(b + (int(c),) if with_class else b)
            boxes.sort(key=lambda b: -b[4])
            return boxes
        if out.shape[0] > out.shape[1]:          # some exports are (N, 4 + classes)
            out = out.T
        cls_scores = out[4:]
        if self.classes is not None:
            cls_scores = cls_scores[list(self.classes)]
        scores = cls_scores.max(axis=0)
        cls_ids = np.array(self.classes if self.classes is not None else range(cls_scores.shape[0]))[cls_scores.argmax(axis=0)]
        keep = scores >= self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[:4, keep]
        scores, cls_ids = scores[keep], cls_ids[keep]
        x1 = (cx - bw / 2 - px) / s
        y1 = (cy - bh / 2 - py) / s
        ww, hh = bw / s, bh / s
        rects = np.stack([x1, y1, ww, hh], 1)
        idx = cv2.dnn.NMSBoxes(rects.tolist(), scores.tolist(), self.conf, self.iou)
        H, W = img.shape[:2]
        boxes = []
        for i in np.array(idx).reshape(-1):
            x, y, w_, h_ = (float(v) for v in rects[i])   # plain Python floats (JSON-safe)
            b = (max(0.0, x), max(0.0, y), min(W - 1.0, x + w_), min(H - 1.0, y + h_), float(scores[i]))
            boxes.append(b + (int(cls_ids[i]),) if with_class else b)
        boxes.sort(key=lambda b: -b[4])
        return boxes
