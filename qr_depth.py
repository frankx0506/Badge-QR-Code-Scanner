"""Use the D435i depth to help read a QR code at range (toward ~1 m), three ways:

 1. depth mask   - the badge is a flat surface at a known distance; keep only pixels within a depth band
                   around it and blank the rest to white. The decoder then sees the code on a clean
                   quiet zone instead of a busy shirt/room, which is what its locate + threshold steps
                   fail on first when the code is small.
 2. frontalize   - fit a plane to the badge's depth and warp the crop to face-on, undoing the tilt of a
                   chest badge. A tilted code loses modules along the tilt axis; this recovers them.
 3. feasibility  - from the distance we know the code's size in pixels, so we can tell the caller whether
                   a frame is worth the heavy decoder at all (>= ~1 px/module) and at what scale to search.

Pure numpy/OpenCV. prep() returns a cleaned (optionally frontal) grey crop ready for qr_reader / qr_soft.

    out = prep(color, depth_mm, intr, box_xyxy, dist_m)
    #   out.gray        cleaned grey crop            (None if the badge couldn't be isolated)
    #   out.frontal     True if it was tilt-corrected
    #   out.px_per_mm   image scale on the badge     (for feasibility / locate scale)
"""
from dataclasses import dataclass

import cv2
import numpy as np

QR_VERSION1_MODULES = 21            # RoboBusiness / CES badge codes are v1-3; v1 = 21 modules across
BADGE_QR_MM = 25.0                  # printed QR side on these badges (~2.5 cm); only used for feasibility


def burst_fuse(crops, scale=2, max_frames=10):
    """Fuse several grey crops of the SAME still badge into one higher-resolution, lower-noise image.

    The badge jitters sub-pixel between frames (hand/arm micro-motion), so each frame samples the code
    on a slightly different grid. Aligning them to one reference and shift-and-adding on a 'scale'x grid
    recovers detail a single frame misses and cuts noise by ~sqrt(N). Only worth running when the 90 fps
    meter says the badge is still (otherwise the frames are a motion smear). Returns a uint8 image, or the
    sharpest single crop if alignment fails.
    """
    crops = [c if c.ndim == 2 else cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops if c is not None and c.size]
    if not crops:
        return None
    # common size = the sharpest crop's size; it is the reference
    sharp = [cv2.Laplacian(c, cv2.CV_32F).var() for c in crops]
    order = np.argsort(sharp)[::-1][:max_frames]
    ref = crops[order[0]]
    h, w = ref.shape
    up = (w * scale, h * scale)
    ref_up = cv2.resize(ref, up, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    acc = ref_up.copy()
    wsum = 1.0
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
    for i in order[1:]:
        c = crops[i]
        if c.shape != ref.shape:
            c = cv2.resize(c, (w, h))
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ref, c, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        except cv2.error:
            continue
        warp_up = warp.copy()
        warp_up[:, 2] *= scale                                    # translation scales to the up-grid
        cu = cv2.resize(c, up, interpolation=cv2.INTER_CUBIC).astype(np.float32)
        aligned = cv2.warpAffine(cu, warp_up, up, flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
        acc += aligned
        wsum += 1
    fused = acc / wsum
    if wsum >= 2:                                                 # mild unsharp to undo the averaging blur
        fused = cv2.addWeighted(fused, 1.6, cv2.GaussianBlur(fused, (0, 0), 1.0 * scale), -0.6, 0)
    return np.clip(fused, 0, 255).astype(np.uint8)


def prep_fused(fused, depth_mm, intr, box, dist, band_mm=55.0):
    """Depth-mask an already burst-fused crop: blank the background to white. The fused image is the
    badge box upscaled, so the box's depth is resized to match. Returns the masked fused crop, or None."""
    if depth_mm is None or not dist:
        return None
    x1, y1, x2, y2 = (int(v) for v in box)
    dcrop = depth_mm[max(0, y1):y2, max(0, x1):x2]
    if dcrop.size < 100:
        return None
    mask = _band_mask(dcrop, dist, band_mm)
    if mask is None:
        return None
    mask = cv2.resize(mask, (fused.shape[1], fused.shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(mask.astype(bool), fused, 255).astype(np.uint8)


@dataclass
class Prep:
    gray: np.ndarray | None
    frontal: bool
    px_per_mm: float
    mask_frac: float                # fraction of the crop kept as badge (diagnostic)
    note: str = ""


def _intr_crop(intr, ox, oy):
    """Camera matrix K shifted to a crop whose top-left is (ox, oy) in the full image."""
    return np.array([[intr.fx, 0, intr.ppx - ox], [0, intr.fy, intr.ppy - oy], [0, 0, 1]], np.float64)


def _band_mask(depth_mm, dist_m, band_mm=55.0):
    """Boolean mask of the flat badge surface: pixels within +-band of dist, largest blob, holes filled.
    White badges give depth holes, so holes inside the blob are filled back in."""
    z = dist_m * 1000.0
    band = np.abs(depth_mm - z) <= band_mm
    band &= depth_mm > 0
    if band.sum() < 50:
        return None
    band = cv2.morphologyEx(band.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(band, 8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (lab == big).astype(np.uint8)
    # fill depth holes (0) that sit inside the badge blob
    holes = (depth_mm == 0).astype(np.uint8)
    filled = cv2.morphologyEx(mask | (mask & 0), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    near = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    mask = ((filled | (holes & near)) > 0).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def _fit_plane(depth_mm, intr, ox, oy, mask):
    """Fit z = ax + by + c (camera-frame metres) to the masked badge pixels. Returns unit normal
    (pointing toward the camera, nz<0) and the badge centre in 3-D, or None."""
    ys, xs = np.nonzero(mask & (depth_mm > 0))
    if xs.size < 200:
        return None
    if xs.size > 4000:
        s = np.random.default_rng(0).choice(xs.size, 4000, replace=False)
        xs, ys = xs[s], ys[s]
    z = depth_mm[ys, xs] / 1000.0
    X = (xs + ox - intr.ppx) * z / intr.fx
    Y = (ys + oy - intr.ppy) * z / intr.fy
    A = np.c_[X, Y, np.ones_like(X)]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)          # z = a X + b Y + c
    a, b, _c = coef
    n = np.array([a, b, -1.0])                            # plane a X + b Y - Z + c = 0 -> normal (a,b,-1)
    n /= np.linalg.norm(n)                                # nz<0: points toward the camera
    centre = np.array([X.mean(), Y.mean(), z.mean()])
    resid = float(np.std(A @ coef - z))
    return (n, centre, resid) if resid < 0.02 else None   # reject if not planar to ~2 cm


def _rect_H(n, intr, centre_xy):
    """Plane-rectifying homography: rotate so the badge normal aligns with the optical axis (de-tilt).
    Rotation is about the ray through the badge centre. The badge is re-centred/re-scaled afterwards."""
    cx, cy = centre_xy
    K = np.array([[intr.fx, 0, cx], [0, intr.fy, cy], [0, 0, 1]], np.float64)
    axis = np.array([0.0, 0.0, 1.0])
    nn = n / np.linalg.norm(n)
    if nn[2] > 0:
        nn = -nn                                          # make it point toward the camera (nz<0)
    v = np.cross(nn, -axis)                               # rotate n -> -z (face the camera)
    s = np.linalg.norm(v)
    if s < 1e-6:
        return None
    c = float(np.dot(nn, -axis))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    H = K @ R @ np.linalg.inv(K)
    return H / H[2, 2]


def _recenter(H, mask, out_wh):
    """Compose H with a scale+translate so the warped badge mask fills the output, centred."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    corners = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()],
                        [xs.max(), ys.max()], [xs.min(), ys.max()]], np.float64)
    w = cv2.perspectiveTransform(corners[None], H)[0]
    wmin, wmax = w.min(0), w.max(0)
    size = max(wmax - wmin)
    if size < 1:
        return None
    ow, oh = out_wh
    s = 0.82 * min(ow, oh) / size
    cen = (wmin + wmax) / 2
    S = np.array([[s, 0, ow / 2 - s * cen[0]], [0, s, oh / 2 - s * cen[1]], [0, 0, 1]], np.float64)
    return S @ H


def prep(color, depth_mm, intr, box, dist, frontalize=True, pad=0.10):
    """Clean, depth-masked, optionally tilt-corrected grey crop of the badge for the QR decoders."""
    H, W = color.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    ox = int(max(0, x1 - pad * bw)); oy = int(max(0, y1 - pad * bh))
    ex = int(min(W, x2 + pad * bw)); ey = int(min(H, y2 + pad * bh))
    gray = cv2.cvtColor(color[oy:ey, ox:ex], cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color[oy:ey, ox:ex]
    px_per_mm = (intr.fx / (dist * 1000.0)) if dist else 0.0

    if depth_mm is None or not dist:
        return Prep(gray, False, px_per_mm, 1.0, "no depth")
    dcrop = depth_mm[oy:ey, ox:ex]
    mask = _band_mask(dcrop, dist)
    if mask is None:
        return Prep(gray, False, px_per_mm, 1.0, "no badge surface in depth")

    # 1. mask background to white (a quiet zone the decoder likes), keep the badge pixels
    m3 = mask.astype(bool)
    cleaned = np.where(m3, gray, 255).astype(np.uint8)
    frac = float(m3.mean())

    # 2. frontalize via the depth plane (skip tiny tilts / non-planar fits)
    frontal = False
    if frontalize:
        pl = _fit_plane(dcrop, intr, ox, oy, mask)
        if pl is not None:
            n, _c, _r = pl
            tilt = np.degrees(np.arccos(min(1.0, abs(n[2]))))
            ys_m, xs_m = np.nonzero(mask)
            centre_xy = (float(xs_m.mean()), float(ys_m.mean())) if xs_m.size else (cleaned.shape[1] / 2, cleaned.shape[0] / 2)
            if tilt > 12:                                 # worth correcting
                ch, cw = cleaned.shape[:2]
                Hm = _rect_H(n, intr, centre_xy)
                Hm = _recenter(Hm, mask, (cw, ch)) if Hm is not None else None
                if Hm is not None:
                    warped = cv2.warpPerspective(cleaned, Hm, (cw, ch), flags=cv2.INTER_CUBIC, borderValue=255)
                    mwarp = cv2.warpPerspective(mask * 255, Hm, (cw, ch)) > 127
                    if mwarp.sum() > 0.4 * mask.sum():    # badge stayed in frame
                        cleaned, frontal = warped, True
    return Prep(cleaned, frontal, px_per_mm, frac, "")
