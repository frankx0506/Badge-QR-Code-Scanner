"""Structure-constrained decoding for the badge codes: QR v1-Q, alphanumeric, exactly 11 chars A-Z0-9.

Without any list of valid codes, a lot of every badge code is still known in advance:
the mode + length header, the terminator, bit padding and the last 3 data codewords (pad bytes),
i.e. 43 of the 104 data bits; only 61 bits (the 11 characters) are free. All 208 codeword bits are a
GF(2)-linear function of those 61 bits, so decoding = finding the 61 bits whose codeword best
matches the soft evidence.

Ordered-statistics decoding (OSD, order 2): take the 61 most reliable independent bit positions,
trust them, and try all patterns of up to 2 flips (1892 candidates per mask). Keep only candidates
whose characters are all A-Z0-9, then re-score the best ones with the decoder's exact quadratic
(blur-aware) likelihood.

OSD ALWAYS returns some valid-looking code, so a result is only accepted through safety gates:
  - margin: best candidate must beat the runner-up (any mask) by a clear likelihood gap;
  - fit:   best candidate must fit nearly as well as the unconstrained best bit pattern;
  - self-consistency: re-encoding the text must reproduce the same mask the image shows;
  - the caller still requires agreement across independent frames/bursts before trusting it.

    out = osd_decode(st, G, h, lam)   # dict(text, margin, fit, mask, ...) or None
"""
import itertools

import numpy as np

import qr_soft

ALNUM45 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
NCHAR = 11

# ------------------------------------------------------------------ GF(256) RS(26,13) encoder
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i], _LOG[_x] = _x, _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _gmul(a, b):
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


_GEN = [1]
for _i in range(13):
    _ng = [0] * (len(_GEN) + 1)
    for _j, _g in enumerate(_GEN):
        _ng[_j] ^= _g
        _ng[_j + 1] ^= _gmul(_g, _EXP[_i])
    _GEN = _ng


def _rs_encode(data):
    msg = list(data) + [0] * 13
    for i in range(13):
        c = msg[i]
        if c:
            for j in range(1, 14):
                msg[i + j] ^= _gmul(_GEN[j], c)
    return list(data) + msg[13:]


def _b2bits(bs):
    return np.array([(b >> (7 - k)) & 1 for b in bs for k in range(8)], np.uint8)


def _bits2b(bits):
    return [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]


def payload_bits(s):
    b = [0, 0, 1, 0] + [int(c) for c in format(len(s), "09b")]
    for i in range(0, len(s) - 1, 2):
        b += [int(c) for c in format(45 * ALNUM45.index(s[i]) + ALNUM45.index(s[i + 1]), "011b")]
    b += [int(c) for c in format(ALNUM45.index(s[-1]), "06b")]
    b += [0, 0, 0, 0]
    b += [0] * ((8 - len(b) % 8) % 8)
    pads, k = [0xEC, 0x11], 0
    while len(b) < 104:
        b += [int(c) for c in format(pads[k % 2], "08b")]
        k += 1
    return np.array(b, np.uint8)


def encode_bits(s):
    """The 208 codeword bits (data then ECC, MSB first) of an 11-char code."""
    return _b2bits(_rs_encode(_bits2b(payload_bits(s))))


FREE = np.arange(13, 74)                       # the 61 bits that carry the 11 characters
_known = np.ones(104, bool)
_known[FREE] = False
_GFULL = np.zeros((104, 208), np.uint8)
for _i in range(104):
    _u = np.zeros(104, np.uint8)
    _u[_i] = 1
    _GFULL[_i] = _b2bits(_rs_encode(_bits2b(_u)))
_REF = payload_bits("0" * NCHAR)
C0 = ((_REF * _known) @ _GFULL % 2).astype(np.uint8)   # fixed part of every codeword
GF = _GFULL[FREE]                                        # 61 x 208

_pats = [np.zeros(61, np.uint8)]
for _i in range(61):
    _e = np.zeros(61, np.uint8)
    _e[_i] = 1
    _pats.append(_e)
for _i, _j in itertools.combinations(range(61), 2):
    _e = np.zeros(61, np.uint8)
    _e[_i] = _e[_j] = 1
    _pats.append(_e)
PATS = np.array(_pats)                                   # 1892 x 61 (order <= 2)


def valid_alnum(u61):
    w = 1 << np.arange(10, -1, -1)
    ok = np.ones(u61.shape[:-1], bool)
    for p in range(5):
        v = u61[..., 11 * p:11 * p + 11] @ w
        ok &= (v // 45 < 36) & (v % 45 < 36)
    v = u61[..., 55:61] @ (1 << np.arange(5, -1, -1))
    return ok & (v < 36)


def u_to_text(u):
    out = []
    for p in range(5):
        v = int("".join(map(str, u[11 * p:11 * p + 11])), 2)
        out += [ALNUM45[v // 45], ALNUM45[v % 45]]
    out.append(ALNUM45[int("".join(map(str, u[55:61])), 2)])
    return "".join(out)


def _systematic(G, order):
    G = G.copy()
    k = G.shape[0]
    T = np.eye(k, dtype=np.uint8)
    piv, r = [], 0
    for col in order:
        if r == k:
            break
        rows = np.nonzero(G[r:, col])[0]
        if rows.size == 0:
            continue
        p = r + rows[0]
        if p != r:
            G[[r, p]] = G[[p, r]]
            T[[r, p]] = T[[p, r]]
        others = np.nonzero(G[:, col])[0]
        others = others[others != r]
        G[others] ^= G[r]
        T[others] ^= T[r]
        piv.append(col)
        r += 1
    return G, np.array(piv), T


def _osd_candidates(hard, rel, keep=24):
    """OSD-2 on the structured subcode. Returns [(metric, u61, codeword208)] for valid A-Z0-9."""
    order = np.argsort(-rel)
    Gs, piv, T = _systematic(GF, order)
    if len(piv) < 61:
        return []
    z = hard ^ C0
    W = z[piv][None, :] ^ PATS
    U = (W @ T % 2).astype(np.uint8)
    C = (C0[None, :] ^ (W @ Gs % 2)).astype(np.uint8)
    metric = ((C != hard[None, :]) * rel[None, :]).sum(1)
    valid = valid_alnum(U)
    idx = np.nonzero(valid)[0]
    if idx.size == 0:
        return []
    idx = idx[np.argsort(metric[idx])][:keep]
    return [(float(metric[i]), U[i], C[i]) for i in idx]


def _energy(G, h, lvec):
    return float(lvec @ G @ lvec - 2 * h @ lvec)


def osd_decode(st, G, h, lam, masks=None, top_masks=3, min_margin=None, max_fit=None, return_all=False):
    """Structure-constrained decode from (summed) normal equations of qr_soft (v1 only)."""
    if st.version != 1:
        return None
    nu = len(h)
    l = np.linalg.solve(G + lam * np.eye(nu), h + lam * st.prior_u)
    lb, rel_mod = qr_soft._bits_and_reliability(st, G, h, lam, l)
    e_free = _energy(G, h, lb)                       # unconstrained best bit pattern
    code_b = qr_soft._to_code(st, lb, None)          # lightness 0/1, n x n
    code_r = qr_soft._to_code(st, rel_mod, 1e9)
    dark = (code_b < 0.5).astype(np.uint8)
    # which masks: format-bit likelihood for EC level Q, best first
    code_l = qr_soft._to_code(st, l, None)
    v = 1 - code_l[st.fmt]
    fmts = sorted(((float(((bits - v) ** 2).sum()), mk, bits) for ec, mk, bits in st.formats if ec == "Q"),
                  key=lambda t: t[0])
    if masks is not None:
        fmts = [f for f in fmts if f[1] in masks]
    i, j = st.bit_rc[:, 0], st.bit_rc[:, 1]
    unk_idx = np.nonzero(~st.known)[0]               # global indices of the unknown modules
    pos_in_u = {g: k for k, g in enumerate(unk_idx)}
    Ng, n = st.Ng, st.n
    cand = []
    for fscore, mk, fbits in fmts[:top_masks]:
        mbits = qr_soft.MASKS[mk](i, j).astype(np.uint8)
        hard = dark[i, j] ^ mbits                    # codeword bits under this mask
        rel = np.maximum(code_r[i, j], 1e-6)
        for metric, u, cw in _osd_candidates(hard, rel):
            # candidate module lightness on the unknown set: data modules from the codeword,
            # format modules from (Q, mask), everything else (quiet ring) from the solved l
            lc = lb.copy()                           # same base as e_free, so fit >= ~0
            dk = cw ^ mbits                          # dark modules at bit positions
            for (r_, c_), d in zip(st.bit_rc, dk):
                g = (r_ + qr_soft.QM) * Ng + (c_ + qr_soft.QM)
                if g in pos_in_u:
                    lc[pos_in_u[g]] = 1.0 - d
            fr, fc = np.nonzero(st.fmt)
            for (r_, c_), d in zip(zip(fr, fc), fbits):
                g = (r_ + qr_soft.QM) * Ng + (c_ + qr_soft.QM)
                if g in pos_in_u:
                    lc[pos_in_u[g]] = 1.0 - d
            cand.append((_energy(G, h, lc), u, mk, metric))
    if not cand:
        return None
    cand.sort(key=lambda c: c[0])
    best = cand[0]
    text = u_to_text(best[1])
    # runner-up = best candidate with a DIFFERENT text (any mask)
    second = next((c for c in cand[1:] if u_to_text(c[1]) != text), None)
    dg = np.diag(G)
    if not (dg > 0).any():
        return None                                  # no usable evidence
    scale = float(np.median(dg[dg > 0]))             # energy of one module flip, roughly
    margin = (second[0] - best[0]) / scale if second else np.inf
    fit = (best[0] - e_free) / scale                 # extra misfit vs the free optimum (>= ~0)
    # self-consistency: a standard encoder writing this text must pick the same mask
    try:
        ref = qr_soft._matrix(text, "Q", 1)
        enc_mask = next((k for _, k, fb in st.formats if _ == "Q" and np.array_equal(ref[st.fmt], fb)), None)
    except Exception:
        enc_mask = None
    info = {"text": text, "mask": best[2], "enc_mask": enc_mask, "margin": float(margin), "fit": float(fit),
            "osd_metric": best[3], "candidates": len(cand)}
    if return_all:
        return info
    mm = 6.0 if min_margin is None else min_margin
    mf = 8.0 if max_fit is None else max_fit
    if margin >= mm and fit <= mf and enc_mask == best[2]:
        info["accepted"] = True
        return info
    return None
