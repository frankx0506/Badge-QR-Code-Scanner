"""Badge scanner for Intel RealSense D435/D435i.

Pipeline per frame:
  1. Detect the badge: find card-shaped quadrilaterals in the color image,
     then use the aligned depth to measure their real-world size and distance
     and keep only ones that are badge-sized and within range.
  2. Scan the QR code: perspective-warp each badge to a flat image and decode
     the QR code inside it (only inside the badge, not the whole frame).

Run:  python badge_scanner.py            (live window, press q / Esc to quit)
      python badge_scanner.py --headless (console log only)
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from qr_reader import read_qr

# Badge physical size limits in millimetres (ID-1 card is 85.6 x 54).
# Widened to accept common lanyard badges too.
BADGE_LONG_MM = (60, 160)
BADGE_SHORT_MM = (40, 120)


def parse_args():
    p = argparse.ArgumentParser(description="D435 badge detector + QR scanner")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--qr-only", action="store_true",
                   help="skip badge detection; scan the whole frame for QR codes")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-dist", type=float, default=1.2,
                   help="ignore badges farther than this (metres)")
    p.add_argument("--no-size-check", action="store_true",
                   help="skip the physical-size check (accept any card-shaped quad)")
    p.add_argument("--badges", default="badges.json",
                   help="optional JSON file mapping QR text -> badge info")
    p.add_argument("--headless", action="store_true", help="no video window")
    p.add_argument("--repeat-after", type=float, default=3.0,
                   help="seconds before the same QR is logged again")
    return p.parse_args()


def load_badge_db(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def order_corners(pts):
    """Return corners as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def find_card_quads(color):
    """Stage 1a: candidate card-shaped quadrilaterals in the color image."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    min_area = 0.004 * w * h
    quads = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > 0.9 * w * h:
            continue
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quads.append(order_corners(approx))
    return quads


def measure_quad(quad, depth_frame, intr):
    """Return (distance_m, long_mm, short_mm) using aligned depth, or None."""
    mask = np.zeros((intr.height, intr.width), np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 255)
    depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    vals = depth[mask > 0]
    vals = vals[vals > 0]
    if vals.size < 50:
        return None
    dist = float(np.median(vals)) * depth_frame.get_units()

    pts3d = [np.array(rs.rs2_deproject_pixel_to_point(intr, [float(x), float(y)], dist))
             for x, y in quad]
    sides = [np.linalg.norm(pts3d[i] - pts3d[(i + 1) % 4]) * 1000 for i in range(4)]
    a = (sides[0] + sides[2]) / 2
    b = (sides[1] + sides[3]) / 2
    return dist, max(a, b), min(a, b)


def warp_badge(color, quad, long_px=600):
    """Flatten the badge so the QR code is square and upright-ish."""
    tl, tr, br, bl = quad
    wpx = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    hpx = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    scale = long_px / max(wpx, hpx, 1)
    W, H = int(wpx * scale), int(hpx * scale)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(color, M, (W, H))


def decode_qr(img):
    """Stage 2: decode a QR code inside the badge image."""
    return read_qr(img)[0]


def log_scan(text, dist, db, last_seen, repeat_after, extra=""):
    info = db.get(text)
    now = time.time()
    if now - last_seen.get(text, 0) > repeat_after:
        who = f" -> {info}" if info else (" -> UNKNOWN BADGE" if db else "")
        d = f"{dist:.2f} m" if dist else "? m"
        print(f"[{time.strftime('%H:%M:%S')}] {extra}at {d} QR={text!r}{who}", flush=True)
    last_seen[text] = now
    return info


def main():
    args = parse_args()
    db = load_badge_db(args.badges)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, args.fps)
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    last_seen = {}
    print("Scanning... (q/Esc to quit)" if not args.headless else "Scanning... (Ctrl-C to quit)")

    try:
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_f, depth_f = frames.get_color_frame(), frames.get_depth_frame()
            if not color_f or not depth_f:
                continue
            color = np.asanyarray(color_f.get_data())
            view = color.copy()

            if args.qr_only:
                text, pts = read_qr(color)
                if text:
                    cx, cy = pts.mean(axis=0).astype(int)
                    cx = min(max(cx, 0), intr.width - 1)
                    cy = min(max(cy, 0), intr.height - 1)
                    dist = depth_f.get_distance(int(cx), int(cy))
                    info = log_scan(text, dist, db, last_seen, args.repeat_after)
                    name = info.get("name") if isinstance(info, dict) else info
                    cv2.polylines(view, [pts.astype(np.int32)], True, (0, 200, 0), 3)
                    cv2.putText(view, f"{name or text} {dist:.2f}m",
                                (int(pts[0][0]), max(int(pts[0][1]) - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
                if not args.headless:
                    cv2.imshow("Badge scanner", cv2.resize(view, None, fx=0.6, fy=0.6))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                continue

            for quad in find_card_quads(color):
                m = measure_quad(quad, depth_f, intr)
                if m is None:
                    continue
                dist, long_mm, short_mm = m
                if dist > args.max_dist:
                    continue
                if not args.no_size_check and not (
                        BADGE_LONG_MM[0] <= long_mm <= BADGE_LONG_MM[1]
                        and BADGE_SHORT_MM[0] <= short_mm <= BADGE_SHORT_MM[1]):
                    continue

                # Badge found -> now scan its QR code.
                text = decode_qr(warp_badge(color, quad, long_px=900))
                ok = text is not None
                info = log_scan(text, dist, db, last_seen, args.repeat_after,
                                f"badge {long_mm:.0f}x{short_mm:.0f} mm ") if ok else None

                colour = (0, 200, 0) if ok else (0, 200, 255)
                cv2.polylines(view, [quad.astype(np.int32)], True, colour, 3)
                label = f"BADGE {dist:.2f}m"
                if ok:
                    name = info.get("name") if isinstance(info, dict) else info
                    label += f" | {name or text}"
                else:
                    label += " | no QR read"
                x, y = quad[0].astype(int)
                cv2.putText(view, label, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

            if not args.headless:
                cv2.imshow("Badge scanner", cv2.resize(view, None, fx=0.6, fy=0.6))
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
