"""Model-based ("soft") QR decoder for codes too small or blurry for normal readers.

A normal reader samples each module (square) of the code and forces it to black or white.
Below ~2.5 px/module that fails: the grid is placed slightly wrong and blur mixes neighbours.
This decoder instead:

  1. finds the code by matching its fixed patterns (finders, timing, quiet zone) at many scales
     and the 4 orientations;
  2. fits the code's position (affine grid) and the blur by rendering the code through a camera
     model (box pixels + Gaussian blur) and comparing with the image, solving all unknown
     modules at the same time (variable projection);
  3. estimates every module's lightness by least squares on the whole patch (deconvolution),
     then flips bits that make the rendered code match the image better;
  4. picks the format (ECC level + mask) by maximum likelihood over the 32 valid ones;
  5. lets Reed-Solomon fix errors, telling it which codewords are least trusted (erasures);
  6. re-draws the corrected code cleanly and lets zxing-cpp parse it.

Several frames of the same badge can be decoded jointly (decode_frames): each frame gets its
own grid fit, and the per-module evidence of all frames is added before deciding the bits.

Safety: a read is accepted only if Reed-Solomon succeeds with spare capacity left, the padding
after the data is valid, and the result is a clean QR that zxing parses. The app still
requires 2 agreeing frames on top of this.

Supports QR versions 1-3 (single Reed-Solomon block: v1, v2, v3-L/M).
"""

import json
import multiprocessing as mp
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import cv2
import numpy as np
import reedsolo
import zxingcpp
from scipy import sparse
from scipy.optimize import least_squares
from scipy.special import ndtr

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT_CACHE = os.path.join(HERE, "models", "qr_formats.json")

# version -> (size, total codewords, {ecc: ecc codewords}) ; single-block cases only
VERSIONS = {1: (21, 26, {"L": 7, "M": 10, "Q": 13, "H": 17}),
            2: (25, 44, {"L": 10, "M": 16, "Q": 22, "H": 28}),
            3: (29, 70, {"L": 15, "M": 26})}
QM = 3          # modelled quiet-zone ring (modules); rings 1-2 known white, ring 3 free
K = 2           # sub-samples per pixel side
NM = 7          # samples along a motion streak

MASKS = [lambda i, j: (i + j) % 2 == 0, lambda i, j: i % 2 == 0, lambda i, j: j % 3 == 0,
         lambda i, j: (i + j) % 3 == 0, lambda i, j: (i // 2 + j // 3) % 2 == 0,
         lambda i, j: (i * j) % 2 + (i * j) % 3 == 0, lambda i, j: ((i * j) % 2 + (i * j) % 3) % 2 == 0,
         lambda i, j: ((i + j) % 2 + (i * j) % 3) % 2 == 0]


# ------------------------------------------------------------------ QR structure
def _function_mask(n):
    f = np.zeros((n, n), bool)
    f[:9, :9] = True
    f[:9, n - 8:] = True
    f[n - 8:, :9] = True
    f[6, :] = True
    f[:, 6] = True
    if n > 21:
        a = n - 7
        f[a - 2:a + 3, a - 2:a + 3] = True
    return f


def _format_mask(n):
    g = np.zeros((n, n), bool)
    g[8, :9] = True
    g[:9, 8] = True
    g[8, n - 8:] = True
    g[n - 8:, 8] = True
    g[6, 8] = g[8, 6] = False
    g[n - 8, 8] = False          # always-dark module
    return g


def _placement(n):
    f = _function_mask(n)
    order, right = [], n - 1
    while right >= 1:
        if right == 6:
            right = 5
        for vert in range(n):
            for k in range(2):
                x = right - k
                y = n - 1 - vert if ((right + 1) & 2) == 0 else vert
                if not f[y, x]:
                    order.append((y, x))
        right -= 2
    return np.array(order)


def _matrix(text, ec, version=None):
    kw = {"ec_level": ec}
    if version:
        kw["version"] = version
    b = zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.QRCode, **kw)
    im = np.array(b.to_image(scale=1, add_quiet_zones=False))
    return (im < 128).astype(np.uint8)          # 1 = dark


class Struct:
    """Fixed layout of one QR version."""

    def __init__(self, version):
        n, ncw, ecc = VERSIONS[version]
        self.version, self.n, self.ncw, self.ecc = version, n, ncw, ecc
        self.func = _function_mask(n)
        self.fmt = _format_mask(n)
        self.bit_rc = _placement(n)[:ncw * 8]
        ref = _matrix("A", "L", version)        # any code of this size: function patterns are fixed
        assert ref.shape[0] == n
        known_code = self.func & ~self.fmt
        known_code[n - 8, 8] = True
        Ng = n + 2 * QM
        self.Ng = Ng
        light = np.ones((Ng, Ng))
        light[QM:QM + n, QM:QM + n] = 1 - ref
        known = np.ones((Ng, Ng), bool)
        known[QM:QM + n, QM:QM + n] = known_code
        ring3 = np.zeros((Ng, Ng), bool)
        ring3[[0, -1], :] = True
        ring3[:, [0, -1]] = True
        known[ring3] = False                     # outer ring: free (nearby text allowed), prior white
        prior = np.full((Ng, Ng), 0.5)
        prior[ring3] = 1.0
        self.light_known = light.ravel()
        self.known = known.ravel()
        self.prior_u = prior.ravel()[~self.known]
        code_idx = np.arange(Ng * Ng).reshape(Ng, Ng)[QM:QM + n, QM:QM + n]
        self.code_idx = code_idx
        self.formats = _format_table(self)       # list of (ec, mask, dark bits at fmt positions)

    def codewords(self, dark, mask_id):
        i, j = self.bit_rc[:, 0], self.bit_rc[:, 1]
        bits = dark[i, j] ^ MASKS[mask_id](i, j).astype(np.uint8)
        return np.packbits(bits.reshape(self.ncw, 8), axis=1).ravel()

    def place(self, dark, cw, mask_id):
        out = dark.copy()
        bits = np.unpackbits(np.asarray(cw, np.uint8))
        i, j = self.bit_rc[:, 0], self.bit_rc[:, 1]
        out[i, j] = bits ^ MASKS[mask_id](i, j).astype(np.uint8)
        return out


def _syndromes_zero(cw, nsym):
    try:
        rs = reedsolo.RSCodec(nsym, nsize=len(cw), fcr=0, prim=0x11d, generator=2, c_exp=8)
        return rs.check(bytearray(cw))[0]
    except Exception:
        return False


def _format_table(st):
    """(ec, mask) -> the 15 format bits as they appear at st.fmt positions (cached on disk)."""
    cache = {}
    if os.path.exists(FORMAT_CACHE):
        cache = json.load(open(FORMAT_CACHE))
    key = str(st.n)
    if key not in cache:
        rng = np.random.default_rng(0)
        alnum = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        table = {}
        for ec in "LMQH":
            for _ in range(5000):
                txt = "".join(rng.choice(alnum, int(rng.integers(1, 8))))
                try:
                    m = _matrix(txt, ec, st.version)
                except Exception:
                    break
                nsym = VERSIONS[st.version][2].get(ec)
                if nsym is None:
                    break
                mk = next((k for k in range(8) if _syndromes_zero(list(st.codewords(m, k)), nsym)), None)
                if mk is not None:
                    table[f"{ec}{mk}"] = m[st.fmt].astype(int).tolist()
                if sum(k.startswith(ec) for k in table) == 8:
                    break
        cache[key] = table
        os.makedirs(os.path.dirname(FORMAT_CACHE), exist_ok=True)
        json.dump(cache, open(FORMAT_CACHE, "w"))
    return [(k[0], int(k[1]), np.array(v, np.uint8)) for k, v in cache[key].items()]


_STRUCTS = {}


def struct(version):
    if version not in _STRUCTS:
        _STRUCTS[version] = Struct(version)
    return _STRUCTS[version]


# ------------------------------------------------------------------ camera model
def build_A(st, aff, sig, pix_u, pix_v, K=K):
    """Sparse matrix: observed pixel <- module-grid lightness.

    aff: 2x3 affine, module coords (x right, y down, origin = code top-left) -> pixel coords.
    sig: (sx, sy) Gaussian blur sigma in pixels, or (sx, sy, mx, my) adding a straight motion
    streak of (mx, my) pixels (box blur along the motion). Pixels listed by (pix_u, pix_v).
    """
    M, t = aff[:, :2], aff[:, 2]
    Minv = np.linalg.inv(M)
    sub = (np.arange(K) + 0.5) / K
    su, sv = np.meshgrid(sub, sub)
    su, sv = su.ravel(), sv.ravel()
    ns = K * K
    if len(sig) == 4:                                  # motion streak: average NM shifted copies
        sh = np.linspace(-0.5, 0.5, NM)
        su = (su[:, None] + sh[None] * sig[2]).ravel()
        sv = (sv[:, None] + sh[None] * sig[3]).ravel()
        ns = K * K * NM
    u = (pix_u[:, None] + su[None]).ravel()
    v = (pix_v[:, None] + sv[None]).ravel()
    pid = np.repeat(np.arange(len(pix_u)), ns)
    xm = Minv[0, 0] * (u - t[0]) + Minv[0, 1] * (v - t[1]) + QM
    ym = Minv[1, 0] * (u - t[0]) + Minv[1, 1] * (v - t[1]) + QM
    # blur sigma in module units along module axes
    smx = max(sig[0] * np.hypot(Minv[0, 0], Minv[0, 1]), 0.05)
    smy = max(sig[1] * np.hypot(Minv[1, 0], Minv[1, 1]), 0.05)
    R = int(np.ceil(3 * max(smx, smy))) + 1
    Ng = st.Ng
    d = np.arange(-R, R + 1)
    jx = np.floor(xm).astype(int)[:, None] + d[None]               # (N, 2R+1) module columns
    iy = np.floor(ym).astype(int)[:, None] + d[None]
    cx = ndtr((xm[:, None] - np.floor(xm)[:, None] - np.arange(-R, R + 2)[None]) / smx)   # CDF at module edges
    cy = ndtr((ym[:, None] - np.floor(ym)[:, None] - np.arange(-R, R + 2)[None]) / smy)
    wx = cx[:, :-1] - cx[:, 1:]
    wy = cy[:, :-1] - cy[:, 1:]
    wx[(jx < 0) | (jx >= Ng)] = 0
    wy[(iy < 0) | (iy >= Ng)] = 0
    w = (wy[:, :, None] * wx[:, None, :]).reshape(len(xm), -1) / ns
    col = (np.clip(iy, 0, Ng - 1)[:, :, None] * Ng + np.clip(jx, 0, Ng - 1)[:, None, :]).reshape(len(xm), -1)
    keep = w > 1e-4
    return sparse.csr_matrix((w[keep], (np.broadcast_to(pid[:, None], w.shape)[keep], col[keep])),
                             shape=(len(pix_u), Ng * Ng))


class Fit:
    pass


def _solve(st, y, u, v, aff, sig, noise=4.0, K=K):
    """Variable projection: given geometry, solve illumination + all unknown modules."""
    A = build_A(st, aff, sig, u, v, K)
    kn = st.known
    pk = A[:, kn] @ st.light_known[kn]
    Au = A[:, ~kn]
    wu = np.asarray(Au.sum(1)).ravel()
    w = np.clip(1 - wu / 0.15, 0, 1) ** 2          # smooth weight: pixels explained by known modules
    if w.sum() < 20:
        return None
    uc, vc = (u - u.mean()) / 50.0, (v - v.mean()) / 50.0
    # y = b + bu*u + bv*v + c*(A l)   (illumination: offset + plane, single contrast)
    X = np.stack([np.ones(len(y)), uc, vc, pk], 1) * w[:, None]
    coef = np.linalg.lstsq(X, y * w, rcond=None)[0]
    b, bu, bv, c = coef
    if c < 12:
        return None
    base = b + bu * uc + bv * vc
    yr = (y - base) / c - pk
    AuT = Au.T.tocsr()
    G = (AuT @ Au).toarray()                       # sparse product: each pixel touches few modules
    h = AuT @ yr
    lam = (noise / c) ** 2 / 0.25
    l = np.linalg.solve(G + lam * np.eye(len(h)), h + lam * st.prior_u)
    r = yr - Au @ l
    f = Fit()
    f.A, f.G, f.h, f.l, f.c, f.b, f.lam, f.yr = A, G, h, l, c, base, lam, yr
    f.res = np.concatenate([r, np.sqrt(lam) * (l - st.prior_u)])
    return f


def _params_to_geom(p):
    aff = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]])
    sig = (np.exp(p[6]), np.exp(p[7]))
    if len(p) == 10:
        sig = sig + (p[8], p[9])
    return aff, sig


def _fit_once(st, y, uu, vv, p0, max_nfev, polish=2):
    nres = len(y) + int((~st.known).sum())

    def make(k):
        def fun(p):
            aff, sig = _params_to_geom(p)
            if not (0.15 < sig[0] < 6 and 0.15 < sig[1] < 6) or (len(sig) == 4 and np.hypot(sig[2], sig[3]) > 25):
                return np.full(nres, 50.0)
            f = _solve(st, y, uu, vv, aff, sig, K=k)
            if f is None:
                return np.full(nres, 50.0)
            return f.res * f.c / 4.0
        return fun

    s = np.hypot(p0[0], p0[3])
    xs = np.array([0.02 * s, 0.02 * s, 0.3 * s, 0.02 * s, 0.02 * s, 0.3 * s, 0.1, 0.1] + [0.5, 0.5] * (len(p0) == 10))
    try:
        # coarse search with 1 sample/pixel (4x faster), then polish with the full model
        p = least_squares(make(1), p0, x_scale=xs, diff_step=1e-3, max_nfev=max_nfev, method="trf").x
        if polish:
            p = least_squares(make(K), p, x_scale=xs, diff_step=1e-3, max_nfev=polish, method="trf").x
    except Exception:
        p = p0
    aff, sig = _params_to_geom(p)
    f = _solve(st, y, uu, vv, aff, sig)
    if f is None:
        return None
    f.aff, f.sig, f.u, f.v, f.y = aff, sig, uu, vv, y
    f.noise = float(np.std(f.res[:len(y)]) * f.c)
    f.cost = float(np.mean((f.res[:len(y)] * f.c) ** 2))
    return f


def fit_geometry(gray, st, aff0, sig0=(0.8, 0.8), max_nfev=5, motion=False):
    """Refine the grid (affine) + blur so the rendered code best matches the image.

    First with a Gaussian blur; if that blur comes out large, also try a straight motion
    streak (what a swaying/walking badge produces) and keep whichever model fits better.
    """
    n = st.n
    corners = np.array([[-2.5, -2.5], [n + 2.5, -2.5], [n + 2.5, n + 2.5], [-2.5, n + 2.5]])
    poly = corners @ aff0[:, :2].T + aff0[:, 2]
    x1, y1 = np.floor(poly.min(0)).astype(int)
    x2, y2 = np.ceil(poly.max(0)).astype(int)
    H, W = gray.shape
    pad = 0
    if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
        pad = 40
        gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        aff0 = aff0.copy()
        aff0[:, 2] += pad
        poly += pad
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), 1)
    vv, uu = np.nonzero(mask)
    y = gray[vv, uu].astype(np.float64)
    p0 = np.array([aff0[0, 0], aff0[0, 1], aff0[0, 2], aff0[1, 0], aff0[1, 1], aff0[1, 2],
                   np.log(sig0[0]), np.log(sig0[1])])
    f = _fit_once(st, y, uu, vv, p0, max_nfev)
    if f is None:
        return None
    fits = [f]
    if motion and max(f.sig[:2]) > 1.2:
        # motion along the module axis with the larger blur, length from box std = L / sqrt(12)
        ax = 0 if f.sig[0] >= f.sig[1] else 1
        direction = f.aff[:, ax] / np.linalg.norm(f.aff[:, ax])
        L = np.sqrt(12 * max(f.sig[ax] ** 2 - 0.6 ** 2, 0.5))
        p1 = np.concatenate([f.aff[0], f.aff[1], [np.log(0.6), np.log(0.6)], direction * L])
        f2 = _fit_once(st, y, uu, vv, p1, max_nfev)
        if f2 is not None:
            fits.append(f2)
    for g in fits:
        g.gray, g.pad = gray, pad
    fits.sort(key=lambda g: g.cost)
    best = fits[0]
    best.alternatives = fits[1:]
    return best


# ------------------------------------------------------------------ locating the code
_TPL = {}


def _pattern(st, q=2, R=8):
    """High-res image (R px/module) of the fixed finder patterns + quiet ring, and its mask."""
    key = (st.version, q, R)
    if key not in _TPL:
        n = st.n
        g = n + 2 * q
        light = np.full((g, g), 0.5, np.float32)
        known = np.zeros((g, g), np.float32)
        lk = st.light_known.reshape(st.Ng, st.Ng)[QM - q:QM + n + q, QM - q:QM + n + q]
        kk = st.known.reshape(st.Ng, st.Ng)[QM - q:QM + n + q, QM - q:QM + n + q]
        fin = np.zeros((n, n), bool)
        fin[:8, :8] = fin[:8, n - 8:] = fin[n - 8:, :8] = True     # finders + separators
        sel = np.ones((g, g), bool)
        sel[q:q + n, q:q + n] = fin
        use = kk & sel
        light[use] = lk[use]
        known[use] = 1
        _TPL[key] = (np.kron(light, np.ones((R, R), np.float32)), np.kron(known, np.ones((R, R), np.float32)), q, R)
    return _TPL[key]


def _template(st, S, angle_deg, blur):
    """Template at S px/module rotated by angle (deg, any value); returns img, mask, module->template affine."""
    big, bigm, q, R = _pattern(st)
    n = st.n
    th = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) * S
    # padded square corners in module coords -> rotated
    cm = np.array([[-q, -q], [n + q, -q], [n + q, n + q], [-q, n + q]], float)
    pc = cm @ rot.T
    off = -pc.min(0)
    size = np.ceil(pc.max(0) - pc.min(0)).astype(int) + 1
    T = np.hstack([rot, off[:, None]])                         # module -> template px
    # big image px (R per module, origin at module -q) -> module: m = p/R - q
    Bm = np.array([[1 / R, 0, -q], [0, 1 / R, -q]])
    Tb = np.hstack([T[:, :2] @ Bm[:, :2], (T[:, :2] @ Bm[:, 2] + T[:, 2])[:, None]])
    sig_big = max(0.5 * R / S, 0.3) if blur is None else np.hypot(0.5 * R / S, blur * R / S)
    src = cv2.GaussianBlur(big, (0, 0), sig_big)
    srcm = cv2.GaussianBlur(bigm, (0, 0), 0.5 * R / S)
    img = cv2.warpAffine(src, Tb, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR, borderValue=0.5)
    msk = cv2.warpAffine(srcm, Tb, (int(size[0]), int(size[1])), flags=cv2.INTER_LINEAR, borderValue=0)
    return img, (msk > 0.97).astype(np.float32), T


def _match(g, st, S, ang, blur, centre=None, radius=0, masked=True):
    """Best placement of the template; optionally only with the code centre within radius px of centre."""
    img, msk, T = _template(st, S, ang, blur)
    H, W = g.shape
    x0, y0 = 0, 0
    if centre is not None:
        tc = T @ np.array([st.n / 2, st.n / 2, 1])            # code centre inside the template
        x0, y0 = int(np.floor(centre[0] - tc[0] - radius)), int(np.floor(centre[1] - tc[1] - radius))
        x1, y1 = x0 + 2 * radius + 2 + img.shape[1], y0 + 2 * radius + 2 + img.shape[0]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        g = g[y0:y1, x0:x1]
    if img.shape[0] + 1 >= g.shape[0] or img.shape[1] + 1 >= g.shape[1] or msk.sum() < 20:
        return -1, None
    res = cv2.matchTemplate(g, img, cv2.TM_CCOEFF_NORMED, mask=msk) if masked else \
        cv2.matchTemplate(g, img, cv2.TM_CCOEFF_NORMED)
    res[~np.isfinite(res)] = -1
    _, mx, _, loc = cv2.minMaxLoc(res)
    aff = T.copy()
    aff[:, 2] += (loc[0] + x0, loc[1] + y0)
    return mx, aff


def _refine(g, ver, S0, a0, aff0, mx0=-1.0, rad_modules=3, angle_span=14):
    """Fine template search around a placement: angle, then scale, then angle again."""
    st = struct(ver)
    best = (mx0, aff0)
    n = st.n
    ctr = aff0 @ np.array([n / 2, n / 2, 1])
    rad = int(np.ceil(rad_modules * S0))
    S_best, a_best = S0, a0
    for stage in ("angle", "scale", "angle2"):
        if stage == "scale":
            trials = [(S, a_best) for S in S_best * np.array([0.9, 0.94, 0.97, 1.0, 1.03, 1.06, 1.1])]
        elif stage == "angle":
            trials = [(S_best, a0 + da) for da in range(-angle_span, angle_span + 1, 2)]
        else:
            trials = [(S_best, a_best + da) for da in (-1, 0, 1)]
        for S, a in trials:
            mx, aff = _match(g, st, S, a, 0.6, ctr, rad)
            if aff is not None and mx > best[0]:
                best, S_best, a_best = (mx, aff), S, a
    return best[0], ver, best[1]


def locate_near(gray, ver, aff, rad_modules=6):
    """Re-find a code near where it was on the previous frame (fast local search)."""
    aff = np.asarray(aff, float)
    S0 = float(np.hypot(aff[0, 0], aff[1, 0]))
    a0 = float(np.degrees(np.arctan2(aff[1, 0], aff[0, 0])))
    return _refine(gray.astype(np.float32), ver, S0, a0, aff, -1.0, rad_modules, angle_span=10)


def locate(gray, versions=(1, 2, 3), top=3, coarse_keep=6, qr_frac=(0.07, 0.9), ppm_range=None):
    """Candidate (score, version, affine) placements of a QR code, best first.

    ppm_range=(lo, hi): expected pixels per module (e.g. from the depth distance). Restricting the
    search to it removes the classic decoy - a finder's inner 3x3 core looks like a whole finder at
    1/3 scale - and makes the search faster.

    Coarse: masked template matching of the finder patterns (8% scale steps over the plausible QR sizes for this crop) x 12 angles (30 deg).
    Fine: around the best coarse hits, rotation +-14 deg (2 deg steps) and scale +-10%.
    """
    g = gray.astype(np.float32)
    coarse = []
    pyr = {}
    side = min(g.shape)                 # badge crop: the QR is qr_frac of the badge's short side
    for ver in versions:
        st = struct(ver)
        lo, hi = max(1.0, qr_frac[0] * side / st.n), min(6.0, qr_frac[1] * side / st.n)
        if ppm_range is not None:
            lo, hi = max(lo, ppm_range[0]), min(hi, ppm_range[1])
        if lo >= hi:
            continue
        for S in np.exp(np.arange(np.log(lo), np.log(hi) + 1e-9, 0.1)):
            f = min(1.0, 1.8 / S)                       # coarse search at <= 1.8 px/module (fast)
            fk = round(f, 2)
            if fk not in pyr:
                pyr[fk] = cv2.resize(g, None, fx=fk, fy=fk, interpolation=cv2.INTER_AREA) if fk < 1 else g
            gs = pyr[fk]
            for ang in range(0, 360, 30):
                mx, aff = _match(gs, st, S * fk, ang, 0.6, masked=False)
                if aff is not None:
                    aff = aff / fk
                    coarse.append((mx, ver, S, ang, aff))
    coarse.sort(key=lambda c: -c[0])
    picked, seen = [], []
    for c in coarse:
        n = struct(c[1]).n
        ctr = c[4] @ np.array([n / 2, n / 2, 1])
        if any(np.hypot(*(ctr - p)) < 4 * c[2] for p in seen):
            continue
        seen.append(ctr)
        picked.append(c)
        if len(picked) >= coarse_keep:
            break
    fine = [_refine(g, ver, S0, a0, aff0, mx0) for mx0, ver, S0, a0, aff0 in picked]
    fine.sort(key=lambda c: -c[0])
    return fine[:top]


# ------------------------------------------------------------------ decoding
def _bits_and_reliability(st, G, h, lam, l_init):
    """ICM bit-flip search on the (summed) quadratic likelihood; returns bits and flip costs."""
    lb = (l_init > 0.5).astype(float)
    pu = st.prior_u
    # energy E(l) = l'Gl - 2h'l (+ prior, ignored for bits)
    g = h - G @ lb
    dg = np.diag(G)
    for _ in range(30):
        d = 1 - 2 * lb
        dE = -2 * d * g + dg
        if dE.min() >= -1e-12:
            break
        for k in np.argsort(dE)[:60]:
            dk = 1 - 2 * lb[k]
            if -2 * dk * g[k] + dg[k] < 0:
                lb[k] += dk
                g -= G[:, k] * dk
    d = 1 - 2 * lb
    rel = -2 * d * g + dg
    return lb, rel


def _to_code(st, vec_u, fill):
    full = np.where(st.known, st.light_known, 0.0).astype(float)
    full[~st.known] = vec_u
    if fill is not None:
        full[st.known] = fill
    return full.reshape(st.Ng, st.Ng)[QM:QM + st.n, QM:QM + st.n]


def _valid_padding(data, version, ec):
    """After the payload, the encoder writes a terminator and alternating 0xEC 0x11 pad bytes."""
    bits = np.unpackbits(np.frombuffer(bytes(data), np.uint8))
    pos, mode_len = 0, {1: (10, 9, 8), 2: (10, 9, 8), 3: (10, 9, 8)}[version]
    while pos + 4 <= len(bits):
        mode = int("".join(map(str, bits[pos:pos + 4])), 2)
        pos += 4
        if mode == 0:
            break
        if mode not in (1, 2, 4):
            return False
        cb = {1: mode_len[0], 2: mode_len[1], 4: mode_len[2]}[mode]
        cnt = int("".join(map(str, bits[pos:pos + cb])), 2)
        pos += cb
        if mode == 1:
            pos += 10 * (cnt // 3) + (4 if cnt % 3 == 1 else 7 if cnt % 3 == 2 else 0)
        elif mode == 2:
            pos += 11 * (cnt // 2) + 6 * (cnt % 2)
        else:
            pos += 8 * cnt
        if pos > len(bits):
            return False
    byte_end = (pos + 7) // 8
    pads = bytes(data[byte_end:])
    return all(b == (0xEC if i % 2 == 0 else 0x11) for i, b in enumerate(pads))


def decode_from_evidence(st, G, h, lam, spare=1, max_formats=3):
    """Decide bits from (summed) normal equations and run format ML + RS with erasures."""
    l = np.linalg.solve(G + lam * np.eye(len(h)), h + lam * st.prior_u)
    lb, rel = _bits_and_reliability(st, G, h, lam, l)
    code_l = _to_code(st, l, None)
    code_b = _to_code(st, lb, None)
    code_r = _to_code(st, rel, 1e9)
    v = 1 - code_l[st.fmt]
    scored = sorted(((float(((bits - v) ** 2).sum()), ec, mk, bits) for ec, mk, bits in st.formats),
                    key=lambda x: x[0])
    dark = (code_b < 0.5).astype(np.uint8)
    for score, ec, mk, fbits in scored[:max_formats]:
        nsym = st.ecc[ec]
        cw = st.codewords(dark, mk)
        i, j = st.bit_rc[:, 0], st.bit_rc[:, 1]
        rel_cw = code_r[i, j].reshape(st.ncw, 8).min(1)
        order = np.argsort(rel_cw)
        rs = reedsolo.RSCodec(nsym, nsize=st.ncw, fcr=0, prim=0x11d, generator=2, c_exp=8)
        for s in range(0, nsym - spare + 1):
            er = sorted(order[:s].tolist())
            try:
                msg, full, errata = rs.decode(bytearray(cw.tobytes()), erase_pos=er or None)
            except reedsolo.ReedSolomonError:
                continue
            e = len(set(errata) - set(er))
            if 2 * e + s > nsym - spare:
                continue
            if not _valid_padding(msg, st.version, ec):
                continue
            clean = st.place(dark, full, mk)
            clean[st.fmt] = fbits
            fn = st.func & ~st.fmt
            ref = 1 - st.light_known.reshape(st.Ng, st.Ng)[QM:QM + st.n, QM:QM + st.n]
            clean[fn] = ref[fn].astype(np.uint8)
            clean[st.n - 8, 8] = 1
            img = np.kron(1 - clean, np.ones((8, 8), np.uint8)) * 255
            img = cv2.copyMakeBorder(img, 32, 32, 32, 32, cv2.BORDER_CONSTANT, value=255)
            for r in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.QRCode):
                if r.text:
                    return {"text": r.text, "ec": ec, "mask": mk, "errors": e, "erasures": s, "nsym": nsym,
                            "version": st.version}
    return None


def _prep(gray):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return gray


def decode(gray, budget_s=6.0, versions=(1, 2, 3), return_fit=False, ppm_range=None):
    """Single frame: locate, fit, decode. Returns dict(text, ...) or None."""
    gray = _prep(gray)
    cands = locate(gray, versions, ppm_range=ppm_range)
    t0 = time.time()
    for score, ver, aff in cands:
        if time.time() - t0 > budget_s:
            break
        st = struct(ver)
        best = fit_geometry(gray, st, aff)
        if best is None:
            continue
        for f in [best] + best.alternatives:
            out = decode_from_evidence(st, f.G, f.h, f.lam)
            if out:
                out.update(loc_score=score, sig=tuple(round(float(s), 2) for s in f.sig),
                           px_per_module=float(np.hypot(f.aff[0, 0], f.aff[1, 0])))
                return (out, f) if return_fit else out
    return (None, None) if return_fit else None


def decode_track(gray, hint=None, versions=(1, 2, 3), min_loc_score=0.7, ppm_range=None):
    """Decode one frame of a badge that may have been seen before.

    hint = (version, affine in crop coords) from the previous frame of the same badge: the
    code is refitted from there (no search, ~5x faster). Returns (result or None, new hint or
    None). The new hint is the refined position even when decoding failed, so the next frame
    starts closer; it is dropped (None) if the fit looks wrong, forcing a fresh search.
    """
    gray = _prep(gray)
    if hint is not None:
        ver, aff = hint
        st = struct(ver)
        score, _, aff = locate_near(gray, ver, aff)
        f = fit_geometry(gray, st, aff) if score >= 0.5 else None
        if f is not None:
            out = decode_from_evidence(st, f.G, f.h, f.lam)
            a = f.aff.copy()
            a[:, 2] -= f.pad
            s_old = np.hypot(aff[0][0], aff[1][0])
            s_new = np.hypot(a[0, 0], a[1, 0])
            ok = f.c > 20 and 0.8 < s_new / max(s_old, 1e-6) < 1.25 and score >= min_loc_score
            if out:
                out.update(px_per_module=float(s_new), path="tracked")
            return out, ((ver, a.tolist()) if ok else None)
    for score, ver, aff in locate(gray, versions, ppm_range=ppm_range)[:2]:
        st = struct(ver)
        f = fit_geometry(gray, st, aff)
        if f is None:
            continue
        a = f.aff.copy()
        a[:, 2] -= f.pad
        out = decode_from_evidence(st, f.G, f.h, f.lam)
        if out:
            out.update(px_per_module=float(np.hypot(a[0, 0], a[1, 0])), path="searched")
            return out, (ver, a.tolist())
        if score >= min_loc_score:
            return None, (ver, a.tolist())
    return None, None


def _init_worker():
    cv2.setNumThreads(1)


def _job(gray, hint, versions, ppm_range=None):
    """Worker-process entry point."""
    t = time.time()
    try:
        out, new_hint = decode_track(gray, hint, versions, ppm_range=ppm_range)
    except Exception as e:
        return None, None, (time.time() - t) * 1000, repr(e)
    return out, new_hint, (time.time() - t) * 1000, None


def decode_frames(grays, versions=(1, 2, 3), budget_s=6.0):
    """Several frames of the same badge: fit each, add their evidence, decode once.

    Frames are aligned by the code's own fitted grid (each frame's model is in module
    coordinates), so no image warping/averaging is involved.
    """
    grays = [_prep(g) for g in grays]
    t0 = time.time()
    # locate on each frame; keep the version that the best-located frame suggests
    fits = {}
    for g in grays:
        if time.time() - t0 > budget_s:
            break
        for score, ver, aff in locate(g, versions, top=2):
            st = struct(ver)
            f = fit_geometry(g, st, aff)
            if f is None:
                continue
            fits.setdefault(ver, []).append(f)
            break
    best = None
    for ver, fl in fits.items():
        st = struct(ver)
        G = sum(f.G * (f.c / max(f.noise, 1.0)) ** 2 for f in fl)
        h = sum(f.h * (f.c / max(f.noise, 1.0)) ** 2 for f in fl)
        out = decode_from_evidence(st, G, h, 4.0)
        if out:
            out["frames"] = len(fl)
            return out
    return best


# ------------------------------------------------------------------ background worker for the app
class SoftWorker:
    """Runs the model-based decoder off the camera loop, on several CPU cores at once.

    The app submits the crop of a badge whose QR the fast reader could not read. Up to
    `workers` crops are decoded in parallel (separate processes, so they really use separate
    cores). The code's position found on one frame is reused for the next frames of the same
    badge (rescaled with the badge box), so most attempts skip the slow search.
    A result is fed to the tracker as one vote (it still needs 2 agreeing reads).
    """

    def __init__(self, workers=None, versions=(1, 2, 3), min_side=60, max_age_s=2.5):
        cpus = os.cpu_count() or 2
        # 4-core Pi: 1 worker (the camera loop and detector need the other cores); laptop: up to 3
        n = workers or (1 if cpus <= 4 else min(3, cpus - 2))
        # one math thread per worker: these problems are tiny, and 28 threads per process
        # fighting over the cores made each decode 3-5x slower (the env is read by the children)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[k] = "1"
        self.pool = self._new_pool(n)
        self.workers, self.versions, self.min_side, self.max_age_s = n, versions, min_side, max_age_s
        self._lock = threading.Lock()
        self.inflight = 0
        self.external = 0             # jobs of other decoders sharing this pool (multi-frame)
        self.rebuilds = 0
        self._results = []
        self.hint = None              # (version, affine, crop_w, box, t) of the current badge
        self.stats = {"tried": 0, "read": 0, "tracked": 0, "last_ms": 0.0, "per_s": 0.0}
        self._times = []

    @staticmethod
    def _new_pool(n):
        return ProcessPoolExecutor(n, mp_context=mp.get_context("spawn"), initializer=_init_worker)

    def _submit(self, fn, *args):
        """pool.submit; if a worker process died (out of memory, killed) the pool is broken for good:
        build a new one and retry once instead of failing every later frame."""
        try:
            return self.pool.submit(fn, *args)
        except (BrokenProcessPool, RuntimeError) as e:
            print("qr_soft: decoder pool broken, restarting it:", e)
            try:
                self.pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self.pool = self._new_pool(self.workers)
            self.rebuilds += 1
            return self.pool.submit(fn, *args)

    def submit_external(self, fn, *args):
        """Run another decoder's job in this pool; counted in `busy` so the two don't queue behind each other."""
        with self._lock:
            self.external += 1
        try:
            fut = self._submit(fn, *args)
        except Exception:
            with self._lock:
                self.external -= 1
            raise
        fut.add_done_callback(self._external_done)
        return fut

    def _external_done(self, _):
        with self._lock:
            self.external -= 1

    @property
    def busy(self):
        return self.inflight + self.external >= self.workers

    def submit(self, gray, box, ppm_range=None):
        """gray: badge crop; box: (x1, y1, x2, y2) of the crop in the frame. False if all busy.
        ppm_range: expected px/module from the depth distance (removes 1/3-scale decoys)."""
        if gray is None or min(gray.shape[:2]) < self.min_side or self.busy:
            return False
        gray = _prep(gray).copy()
        hint = None
        with self._lock:
            h = self.hint
            if h is not None and time.time() - h[4] < 1.5 and _box_iou(h[3], box) > 0.2:
                scale = gray.shape[1] / max(h[2], 1)            # badge got nearer/further: rescale
                aff = np.asarray(h[1], float) * scale
                hint = (h[0], aff.tolist())
            self.inflight += 1
        try:
            fut = self._submit(_job, gray, hint, self.versions, ppm_range)
        except Exception as e:
            with self._lock:
                self.inflight -= 1
            print("qr_soft: submit failed:", e)
            return False
        fut.add_done_callback(lambda f, w=gray.shape[1], b=tuple(box): self._done(f, w, b))
        return True

    def _done(self, fut, crop_w, box):
        try:
            out, new_hint, ms, err = fut.result()
        except Exception as e:
            out, new_hint, ms, err = None, None, 0.0, repr(e)
        now = time.time()
        with self._lock:
            self.inflight -= 1
            self.stats["tried"] += 1
            self.stats["last_ms"] = round(ms)
            self._times = [t for t in self._times if now - t < 5] + [now]
            self.stats["per_s"] = round(len(self._times) / 5, 1)
            if err:
                print("qr_soft:", err)
            self.hint = (new_hint[0], new_hint[1], crop_w, box, now) if new_hint else None
            if out:
                self.stats["read"] += 1
                self.stats["tracked"] += out.get("path") == "tracked"
                self._results = self._results[-4:] + [(out["text"], box, now)]

    def poll(self):
        """Results (text, box, t) not older than max_age_s, newest last; each returned once."""
        with self._lock:
            rs, self._results = self._results, []
        return [r for r in rs if time.time() - r[2] <= self.max_age_s]

    def requeue(self, results):
        with self._lock:
            self._results = list(results) + self._results

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0
