"""'Other objects' mode: read QR codes on anything - product boxes, phones, books, signs, laptops.

Per frame:
 1. The general object detector (YOLO26n, COCO's 80 everyday classes, the same file as the person
    model) finds objects; people and room-sized things (tables, beds...) are skipped.
 2. The whole frame is searched for QR codes (free_qr.scan: normal, enlarged, close-up retries).
 3. Each object without a code yet gets the badge cascade on a full-resolution crop of it (a code
    that is too small in the whole-frame search often reads from the object's close-up).
 4. Every code is attached to the object it sits on (or "no object"), with distance from depth.

Badge things (size filter, person identity) are NOT used in this mode.

    oq = ObjectQR(); res = oq.process(color, depth, intr, max_dist)
"""
import os
import time

import cv2
import numpy as np

from badge_detector import BadgeDetector
from free_qr import scan
from qr_reader import read_qr_in_badge

HERE = os.path.dirname(os.path.abspath(__file__))
COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote "
        "keyboard cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear "
        "hair_drier toothbrush").split()
COCO = [c.replace("_", " ") for c in COCO]
# not things a QR code is held up on (the room, vehicles, animals, furniture)
SKIP = {"person", "dining table", "bed", "couch", "chair", "toilet", "sink", "refrigerator", "oven", "bench",
        "car", "bus", "train", "truck", "airplane", "boat", "motorcycle", "bicycle", "potted plant",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}


def _depth_of(depth, x1, y1, x2, y2):
    if depth is None:
        return None
    v = depth[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
    v = v[v > 0]
    return float(np.median(v)) / 1000 if v.size > 20 else None


class ObjectQR:
    def __init__(self, model=os.path.join(HERE, "models", "person.onnx"), conf=0.3, max_objects=4,
                 scan_ms=80, crop_ms=45, total_ms=170):
        self.det = BadgeDetector.load(model, conf=conf)          # all 80 classes
        self.max_objects, self.scan_ms, self.crop_ms, self.total_ms = max_objects, scan_ms, crop_ms, total_ms

    def process(self, color, depth, intr, max_dist=1.5):
        t0 = time.perf_counter()
        H, W = color.shape[:2]
        out = {"badges": [], "rejected": [], "qr": None, "crop": None, "objects": [], "codes": []}
        objs = []
        if self.det is not None:
            for x1, y1, x2, y2, sc, c in self.det.detect(color, with_class=True):
                name = COCO[c] if 0 <= c < len(COCO) else f"class {c}"
                if name in SKIP or (x2 - x1) * (y2 - y1) > 0.6 * W * H:
                    continue
                dist = _depth_of(depth, x1, y1, x2, y2)
                if dist is not None and dist > max_dist:
                    continue
                objs.append({"box": (x1, y1, x2, y2), "label": name, "conf": sc, "dist": dist, "qr": None})
        objs.sort(key=lambda o: -(o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))
        objs = objs[:self.max_objects]

        gray = color[:, :, 1]
        codes = scan(gray, self.scan_ms)                          # [(text, quad)]

        def inside(o, q):
            cx, cy = q.mean(axis=0)
            x1, y1, x2, y2 = o["box"]
            return x1 <= cx <= x2 and y1 <= cy <= y2

        # close-up retry on each object that has no code yet
        for o in objs:
            if any(inside(o, q) for _, q in codes):
                continue
            if (time.perf_counter() - t0) * 1000 > self.total_ms:
                break
            x1, y1, x2, y2 = o["box"]
            m = 0.06 * max(x2 - x1, y2 - y1)
            ox, oy = int(max(0, x1 - m)), int(max(0, y1 - m))
            crop = gray[oy:int(min(H, y2 + m)), ox:int(min(W, x2 + m))]
            if crop.size < 400:
                continue
            text, p = read_qr_in_badge(crop, self.crop_ms)
            if text and p is not None:
                codes.append((text, p + np.array([ox, oy], np.float32)))

        for text, q in codes:
            owner = next((o for o in objs if inside(o, q) and o["qr"] is None), None)
            x0, y0 = q.min(axis=0)
            x1, y1 = q.max(axis=0)
            dist = _depth_of(depth, x0, y0, x1, y1) or (owner["dist"] if owner else None)
            if owner is not None:
                owner["qr"] = text
            out["codes"].append((text, q, dist, owner["label"] if owner else None))
        out["objects"] = objs
        if out["codes"]:
            # the main read: the nearest code (or the first, without depth)
            text, q, dist, _ = min(out["codes"], key=lambda c: c[2] if c[2] else 9.0)
            out["qr"] = (text, q, dist)
        # crop pane: the object with a code, else the biggest object
        show = next((o for o in objs if o["qr"]), objs[0] if objs else None)
        if show is not None:
            x1, y1, x2, y2 = (int(v) for v in show["box"])
            out["crop"] = color[y1:y2 + 1, x1:x2 + 1].copy()
        out["ms"] = (time.perf_counter() - t0) * 1000
        return out
