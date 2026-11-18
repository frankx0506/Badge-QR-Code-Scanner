"""Joint multi-frame QR grid reconstruction ("QR-aware multi-frame super-resolution").

For a still badge seen in N frames, every pixel of every frame is a blurred mix of the same 21x21
modules. Instead of super-resolving pixels and then reading, this solves for the modules directly
from all frames at once:

  1. registration (optical flow, parametric): each frame is aligned to the sharpest one -
     phase correlation for the shift, then ECC with an affine model (the right flow model for a
     flat, rigid badge; dense per-pixel flow is noisier on a QR's texture);
  2. motion-compensated fusion of all frames on a 2x grid, used only to FIND the code reliably
     (a single 1.3-1.7 px/module frame is too weak to locate in);
  3. the module grid fitted once is transferred into every frame through its registration, so
     sub-pixel shifts between frames sample the modules at different phases (this is where the
     super-resolution comes from);
  4. one shared least-squares solve for all modules from every frame's pixels, each frame with
     its own illumination; grid geometry and blur are refined jointly over all frames;
  5. Reed-Solomon errors+erasures, padding check and clean re-render verification
     (qr_soft.decode_from_evidence) on the summed evidence.

    out = decode_burst(grey_crops)        # dict(text, frames, px_per_module, ...) or None
"""
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

import qr_soft
import qr_struct
from qr_soft import build_A, struct

_ECC_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)


# ------------------------------------------------------------------ 1. registration
def register(ref, img):
    """2x3 affine W mapping ref pixel coords -> img pixel coords, and a quality score (0..1)."""
    a = cv2.GaussianBlur(ref.astype(np.float32), (0, 0), 0.8)
    b = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    W = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    try:
        cc, W = cv2.findTransformECC(a, b, W, cv2.MOTION_AFFINE, _ECC_CRIT, None, 3)
    except cv2.error:
        cc = resp if resp > 0.2 else 0.0
    return W.astype(np.float64), float(cc)


def _compose(W, aff):
    """module -> ref pixels (aff), then ref -> frame pixels (W)."""
    A = W[:, :2] @ aff[:, :2]
    return np.hstack([A, (W[:, :2] @ aff[:, 2] + W[:, 2])[:, None]])


# ------------------------------------------------------------------ 2. motion-compensated fusion
def fuse(frames, Ws, ref_shape, scale=2):
    """All frames warped into the reference frame on a 'scale'x grid and averaged."""
    h, w = ref_shape
    acc = np.zeros((h * scale, w * scale), np.float64)
    cnt = np.zeros_like(acc)
    S = np.diag([scale, scale, 1.0])
    for f, W in zip(frames, Ws):
        # dst (fused) -> src (frame):  frame = W @ ref ; ref = (fused + 0.5)/scale - 0.5
        Wf = np.vstack([W, [0, 0, 1]])
        T = np.array([[1 / scale, 0, 0.5 / scale - 0.5], [0, 1 / scale, 0.5 / scale - 0.5], [0, 0, 1]])
        M = (Wf @ T)[:2]
        warped = cv2.warpAffine(f.astype(np.float32), M, (w * scale, h * scale),
                                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                                borderValue=-1)
        ok = warped >= 0
        acc[ok] += warped[ok]
        cnt[ok] += 1
    fused = acc / np.maximum(cnt, 1)
    fused[cnt == 0] = np.median(fused[cnt > 0]) if (cnt > 0).any() else 0
    del S
    return np.clip(fused, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ 4. joint solve over frames
def _pixels(gray, aff, n):
    corners = np.array([[-2.5, -2.5], [n + 2.5, -2.5], [n + 2.5, n + 2.5], [-2.5, n + 2.5]])
    poly = corners @ aff[:, :2].T + aff[:, 2]
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(poly).astype(np.int32), 1)
    vv, uu = np.nonzero(mask)
    return gray[vv, uu].astype(np.float64), uu.astype(np.float64), vv.astype(np.float64)


def _frame_terms(st, y, u, v, aff, sig, K):
    """Per-frame illumination from the known modules; returns (Au, yr, c) in lightness units."""
    A = build_A(st, aff, sig, u, v, K)
    kn = st.known
    pk = A[:, kn] @ st.light_known[kn]
    Au = A[:, ~kn]
    wu = np.asarray(Au.sum(1)).ravel()
    w = np.clip(1 - wu / 0.15, 0, 1) ** 2
    if w.sum() < 20:
        return None
    uc, vc = (u - u.mean()) / 50.0, (v - v.mean()) / 50.0
    X = np.stack([np.ones(len(y)), uc, vc, pk], 1) * w[:, None]
    b, bu, bv, c = np.linalg.lstsq(X, y * w, rcond=None)[0]
    if c < 12:
        return None
    yr = (y - (b + bu * uc + bv * vc)) / c - pk
    return Au.tocsr(), yr, float(c)


def joint_evidence(st, pix, Ws, aff, sig, K=2, noise=4.0, weights=None):
    """Shared module solve across frames. Returns (G, h, l, residual_per_frame, terms)."""
    nu = int((~st.known).sum())
    G = np.zeros((nu, nu))
    h = np.zeros(nu)
    terms = []
    for k, ((y, u, v), W) in enumerate(zip(pix, Ws)):
        t = _frame_terms(st, y, u, v, _compose(W, aff), sig, K)
        terms.append(t)
        if t is None:
            continue
        Au, yr, c = t
        wk = (c / noise) ** 2 * (1.0 if weights is None else weights[k])
        AuT = Au.T.tocsr()
        G += (AuT @ Au).toarray() * wk
        h += (AuT @ yr) * wk
    lam = 4.0
    l = np.linalg.solve(G + lam * np.eye(nu), h + lam * st.prior_u)
    rms = []
    for t in terms:
        if t is None:
            rms.append(np.inf)
            continue
        Au, yr, c = t
        rms.append(float(np.sqrt(np.mean((yr - Au @ l) ** 2)) * c))
    return G, h, l, np.array(rms), terms


def joint_fit(st, pix, Ws, aff0, sig0, max_nfev=6):
    """Refine the shared grid geometry + blur against all frames at once (shared modules)."""
    p0 = np.array([aff0[0, 0], aff0[0, 1], aff0[0, 2], aff0[1, 0], aff0[1, 1], aff0[1, 2],
                   np.log(sig0[0]), np.log(sig0[1])])
    n_res = sum(len(p[0]) for p in pix)

    def fun(p):
        aff = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]])
        sig = (np.exp(p[6]), np.exp(p[7]))
        if not (0.15 < sig[0] < 6 and 0.15 < sig[1] < 6):
            return np.full(n_res, 50.0)
        _, _, l, _, terms = joint_evidence(st, pix, Ws, aff, sig, K=1)
        out = []
        for (y, _, _), t in zip(pix, terms):
            if t is None:
                out.append(np.full(len(y), 50.0))
            else:
                Au, yr, c = t
                out.append((yr - Au @ l) * c / 4.0)
        return np.concatenate(out)

    s = np.hypot(p0[0], p0[3])
    xs = np.array([0.02 * s, 0.02 * s, 0.3 * s, 0.02 * s, 0.02 * s, 0.3 * s, 0.1, 0.1])
    try:
        p = least_squares(fun, p0, x_scale=xs, diff_step=1e-3, max_nfev=max_nfev, method="trf").x
    except Exception:
        p = p0
    return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]]), (np.exp(p[6]), np.exp(p[7]))


# ------------------------------------------------------------------ whole pipeline
def decode_burst(grays, versions=(1,), budget_s=10.0, max_cands=2, min_frames=3, return_info=False, refine=False,
                 fit_nfev=3, use_struct=False, evidence_only=None, ppm_range=None, geom_on="fused"):
    """Read one still badge from N grey crops of it (same crop window, consecutive frames).
    refine=True also re-fits geometry+blur jointly over all frames (slower; usually not needed,
    because the grid fitted on the fused image is already accurate)."""
    t0 = time.time()
    frames = [g if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) for g in grays]
    frames = [f for f in frames if f is not None and f.size]
    info = {"frames": len(frames)}
    if len(frames) < min_frames:
        return (None, info) if return_info else None
    h0, w0 = frames[0].shape
    frames = [f if f.shape == (h0, w0) else cv2.resize(f, (w0, h0)) for f in frames]
    sharp = [cv2.Laplacian(f, cv2.CV_32F).var() for f in frames]
    r = int(np.argmax(sharp))
    ref = frames[r]
    # 1. registration
    Ws, qual = [], []
    for k, f in enumerate(frames):
        if k == r:
            Ws.append(np.array([[1.0, 0, 0], [0, 1.0, 0]]))
            qual.append(1.0)
        else:
            W, q = register(ref, f)
            Ws.append(W)
            qual.append(q)
    keep = [k for k in range(len(frames)) if qual[k] > 0.5]
    info["registered"] = len(keep)
    if len(keep) < min_frames:
        return (None, info) if return_info else None
    frames = [frames[k] for k in keep]
    Ws = [Ws[k] for k in keep]
    # 2. fused 2x image: locate + first fit there (much cleaner than any single frame)
    fused = fuse(frames, Ws, ref.shape, 2)
    lo2, hi2 = (ppm_range[0] * 2, ppm_range[1] * 2) if ppm_range else (1.7, 1e9)
    cands = qr_soft.locate(fused, versions, top=4, ppm_range=(lo2, hi2) if ppm_range else None)
    # size prior: the finder's inner 3x3 core looks like a whole finder at 1/3 scale, so the locator
    # often ranks a decoy at ~1/3 of the true size first. Reject sizes the code can't have; with depth
    # the expected px/module is known (ppm_range), otherwise just drop sub-0.85 px/module grids.
    lo, hi = ppm_range if ppm_range else (0.85, 1e9)
    cands = [c for c in cands if lo <= np.hypot(c[2][0, 0], c[2][1, 0]) / 2 <= hi][:max_cands]
    info["candidates"] = len(cands)
    for score, ver, aff_f in cands:
        if time.time() - t0 > budget_s:
            break
        st = struct(ver)
        f = qr_soft.fit_geometry(fused, st, aff_f, max_nfev=fit_nfev)
        if f is None:
            continue
        a = f.aff.copy()
        a[:, 2] -= f.pad
        # fused (2x) coords -> reference frame coords: ref = (fused + 0.5)/2 - 0.5
        aff = np.vstack([a[0] / 2, a[1] / 2])
        aff[:, 2] += 0.25 - 0.5
        sig = (max(f.sig[0] / 2, 0.3), max(f.sig[1] / 2, 0.3))
        if time.time() - t0 > budget_s:
            break
        if geom_on == "ref":
            # full single-frame fit (incl. motion-streak model) on the sharpest frame, from this start
            fr_ = qr_soft.fit_geometry(ref, st, aff, sig0=sig, motion=True)
            if fr_ is not None:
                aff = fr_.aff.copy()
                aff[:, 2] -= fr_.pad
                sig = fr_.sig[:2]
        pix = [_pixels(fr, _compose(W, aff), st.n) for fr, W in zip(frames, Ws)]
        # 3-4. joint refinement + shared solve, one robust reweight of badly registered frames
        if refine:
            aff, sig = joint_fit(st, pix, Ws, aff, sig)
            pix = [_pixels(fr, _compose(W, aff), st.n) for fr, W in zip(frames, Ws)]
        G, hv, l, rms, _ = joint_evidence(st, pix, Ws, aff, sig)
        med = np.median(rms[np.isfinite(rms)]) if np.isfinite(rms).any() else 1.0
        wts = np.where(np.isfinite(rms), np.clip((1.5 * med / np.maximum(rms, 1e-6)) ** 2, 0, 1), 0)
        G, hv, l, rms, _ = joint_evidence(st, pix, Ws, aff, sig, weights=wts)
        if evidence_only is not None:                 # return the evidence only, no decode
            evidence_only.append((st, G, hv))
            continue
        # 5. Reed-Solomon errors+erasures + padding + re-render verify
        out = qr_soft.decode_from_evidence(st, G, hv, 4.0)
        if out is None and use_struct and st.version == 1 and time.time() - t0 < budget_s:
            # badge-format structure (43 known data bits) + OSD, behind its safety gates
            so = qr_struct.osd_decode(st, G, hv, 4.0)
            if so:
                out = {"text": so["text"], "path": "structure-OSD", "margin": so["margin"], "fit": so["fit"]}
        info.update(px_per_module=float(np.hypot(aff[0, 0], aff[1, 0])), sigma=tuple(round(float(s), 2) for s in sig),
                    loc_score=float(score), ms=round((time.time() - t0) * 1000))
        if out:
            out.update(frames=len(frames), px_per_module=info["px_per_module"])
            return (out, info) if return_info else out
    info["ms"] = round((time.time() - t0) * 1000)
    return (None, info) if return_info else None


# ------------------------------------------------------------------ in the app: still-badge fallback
MODULE_MM = 0.86          # badge QR module size in mm (used to predict px/module from distance)


def expected_ppm(dist_m, fx=1387.0, module_mm=MODULE_MM):
    """Expected px/module at this distance; the decoder searches 0.55x..1.7x of it."""
    p = fx * module_mm / 1000.0 / max(dist_m, 0.15)
    return (0.55 * p, 1.7 * p)


def _burst_job(frames, ppm_range):
    """Worker-process entry point (runs in the soft decoder's process pool)."""
    import cv2 as _cv2
    _cv2.setNumThreads(1)
    t = time.time()
    try:
        out = decode_burst(frames, versions=(1,), use_struct=True, geom_on="ref", ppm_range=ppm_range)
        return (out["text"] if out else None), (time.time() - t) * 1000, None
    except Exception as e:
        return None, (time.time() - t) * 1000, repr(e)


class BurstCollector:
    """Collects 8 frames of a STILL, unread, far-ish badge through one fixed window and decodes them
    jointly in a background process. Used only as a fallback: the fast reader and the single-frame
    soft decoder run as usual; this kicks in where they fail (still badges at longer range)."""

    def __init__(self, pool, n=8, min_dist=0.55, unread_s=1.0, cooldown_s=1.0, max_age_s=4.0):
        import threading
        self.pool, self.n, self.min_dist = pool, n, min_dist
        self.unread_s, self.cooldown_s, self.max_age_s = unread_s, cooldown_s, max_age_s
        self._lock = threading.Lock()
        self._win, self._frames, self._dist = None, [], None
        self._inflight, self._results = False, []
        self._last_read, self._last_done = time.time(), 0.0
        self.stats = {"bursts": 0, "read": 0, "last_ms": 0, "collecting": False}

    def step(self, gray, box, dist, still, read_now, now=None):
        """Call every frame with the full grey frame and the priority badge (or None)."""
        now = time.time() if now is None else now
        if read_now:
            self._last_read = now
        idle = (not read_now and box is not None and dist is not None and dist >= self.min_dist and still
                and now - self._last_read >= self.unread_s and now - self._last_done >= self.cooldown_s
                and not self._inflight)
        if self._win is None:
            if not idle:
                return
            x1, y1, x2, y2 = (int(v) for v in box)
            bw, bh = x2 - x1, y2 - y1
            H, W = gray.shape[:2]
            self._win = (max(0, int(x1 - 0.25 * bw)), max(0, int(y1 - 0.25 * bh)),
                         min(W, int(x2 + 0.25 * bw)), min(H, int(y2 + 0.25 * bh)))
            self._frames, self._dist = [], dist
        w = self._win
        if box is None or read_now or not (w[0] <= (box[0] + box[2]) / 2 <= w[2] and w[1] <= (box[1] + box[3]) / 2 <= w[3]):
            self._win, self._frames = None, []           # badge moved away / got read: drop this burst
            self.stats["collecting"] = False
            return
        crop = gray[w[1]:w[3], w[0]:w[2]]
        self._frames.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy())
        self.stats["collecting"] = True
        if len(self._frames) >= self.n:
            frames, win, ppm = self._frames, self._win, expected_ppm(self._dist)
            self._win, self._frames = None, []
            self.stats["collecting"] = False
            self._inflight = True
            self.stats["bursts"] += 1
            try:   # a SoftWorker shares its pool and counts this job as busy; a plain pool also works
                sub = getattr(self.pool, "submit_external", None) or self.pool.submit
                fut = sub(_burst_job, frames, ppm)
            except Exception as e:
                self._inflight = False
                print("qr_burst: submit failed:", e)
                return
            fut.add_done_callback(lambda f, w=win: self._done(f, w))

    def _done(self, fut, win):
        try:
            text, ms, err = fut.result()
        except Exception as e:
            text, ms, err = None, 0.0, repr(e)
        with self._lock:
            self._inflight = False
            self._last_done = time.time()
            self.stats["last_ms"] = round(ms)
            if err:
                print("qr_burst:", err)
            if text:
                self.stats["read"] += 1
                self._results = self._results[-3:] + [(text, win, time.time())]

    def poll(self):
        with self._lock:
            rs, self._results = self._results, []
        return [r for r in rs if time.time() - r[2] <= self.max_age_s]
