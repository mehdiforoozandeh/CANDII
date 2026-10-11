"""t125 h — the exact push-forward of a predicted NB count through a map h (design note Part 2).

No sampling. X' is a count, so the predicted P is the finite set {h(k)} over the NB support, each
value with weight NB(k). Its mean and CRPS are computed exactly over that set.

**NB parameterisation (matches t118 exactly).** The t118 ladder gives, per bin, `(loc, disp)`; the
NB has mean mu = `pred.mean(loc)` = exp(loc) clamped to [1e-4, 1e6] and size n = exp(disp) clipped
to [1e-3, 1e4] (`ladder.model.MU_MIN..N_MAX`, `ladder.score.spread_of("counts", disp)`). Its success
probability is p = n / (n + mu) by `candi.bench.distributional.p_from_mu` (clipped to
[1e-9, 1 - 1e-9]), and the law is scipy's `nbinom(n, p)`: pmf(k) = C(k + n - 1, k) p^n (1 - p)^k,
mean n (1 - p) / p = mu. This is the law `ladder.score` scores with `candi.metrics.nb_crps(n, p, y)`.
The caller passes mu and n already clamped as t118 does; this module does not clamp them again.

**The support and how its k-axis is bounded.** `nb_support` returns a dense `[n, K]` layout, each row
left-aligned and contiguous (column j holds k = start_i + j):
  1. one-sided: 0..k_i, k_i = `nbinom.ppf(1 - tail, n, p)` (dropped upper mass <= tail), used when
     its k_i + 1 points fit in k_max;
  2. else two-sided: lo_i..hi_i with lo_i = ppf(tail / 2), hi_i = ppf(1 - tail / 2) (total dropped
     mass <= tail), used when hi_i - lo_i + 1 <= k_max — this keeps large-mu bins (a peak bin with
     mu in the thousands) exact without listing the empty range below them;
  3. else the row is CAPPED: k_max + 1 points around the NB mode, clamped inside [lo_i, hi_i]. Its
     dropped mass exceeds tail; the caller counts such rows.
So an uncapped row has at most k_max points and a capped row exactly k_max + 1: K <= k_max + 1, and
when K == k_max + 1, `valid[:, -1]` is True exactly for the capped rows (`support_size > k_max` says
the same). Memory of the returned arrays is n x K x 17 bytes (int64 k, float64 w, bool valid); K is
the batch maximum, so the caller chunks rows (one capped row makes its chunk 4097 wide).

**CRPS of a discrete set** (`crps_discrete`): sum_i w_i |v_i - y| - 1/2 sum_ij w_i w_j |v_i - v_j|.
v is sorted per row (any h: monotone or not); the pair sum is 2 sum_i w_i v_i (C_<i - C_>i) with C
the cumulative weight below / above in sorted order, O(K log K) per row; ties cancel exactly.

**Own noise** (`crps_gauss_mixture`): h emits N(mu_k, sigma_k^2) on log(1 + P) at each k, so the
prediction is the mixture sum_k w_k N(mu_k, sigma_k^2). Its CRPS is closed form (Grimit et al.
2006): sum_k w_k A(y - mu_k, sigma_k^2) - 1/2 sum_jk w_j w_k A(mu_j - mu_k, sigma_j^2 + sigma_k^2),
A(m, s^2) = m (2 Phi(m / s) - 1) + 2 s phi(m / s). O(K^2) per row: rows are sorted by their
nonzero width and processed in chunks, each chunk trimmed to its own widest row, with at most
`chunk` rows and at most PAIR_ELEMS = 2^24 pair elements (so a wide chunk takes fewer rows).

Entries with w == 0 (padding) are ignored everywhere, even when their values are NaN or inf.
Weights are renormalised per row before use; a row with no positive weight gives NaN.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, ndtr
from scipy.stats import nbinom

from candi.bench.distributional import p_from_mu

TAIL = 1e-6
K_MAX = 4096
#: crps_gauss_mixture: at most this many (row, j, k) elements per chunk (~134 MB per temporary)
PAIR_ELEMS = 1 << 24
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
#: a ppf that is not finite (extreme n, p) is treated as this, so the row is capped
_HUGE = np.int64(1) << 60


# ---------------------------------------------------------------------------------------------
# NB support
# ---------------------------------------------------------------------------------------------


def _check(mu, n_size) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mu, dtype=np.float64).ravel()
    n_size = np.asarray(n_size, dtype=np.float64).ravel()
    if mu.shape != n_size.shape:
        raise ValueError(f"mu {mu.shape} and n_size {n_size.shape} differ")
    if not (np.all(np.isfinite(mu)) and np.all(mu >= 0.0)):
        raise ValueError("mu must be finite and >= 0")
    if not (np.all(np.isfinite(n_size)) and np.all(n_size > 0.0)):
        raise ValueError("n_size must be finite and > 0")
    return mu, n_size


def _ppf(q: float, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """int64 nbinom.ppf(q); non-finite -> _HUGE."""
    if n.size == 0:
        return np.zeros(0, np.int64)
    r = nbinom.ppf(q, n, p)
    out = np.full(r.shape, _HUGE, dtype=np.int64)
    ok = np.isfinite(r) & (r < float(_HUGE))
    out[ok] = r[ok].astype(np.int64)
    return out


def _upper(n, p, tail) -> np.ndarray:
    """smallest k with P(X > k) <= tail (scipy ppf, raised by one step while sf says it is short)."""
    hi = _ppf(1.0 - tail, n, p)
    for _ in range(4):
        fin = hi < _HUGE
        short = np.zeros(hi.shape, bool)
        short[fin] = nbinom.sf(hi[fin], n[fin], p[fin]) > tail
        if not short.any():
            break
        hi[short] += 1
    return hi


def _lower(n, p, half) -> np.ndarray:
    """largest lo with P(X < lo) <= half (scipy ppf, lowered while cdf says it is too high)."""
    lo = _ppf(half, n, p)
    lo[lo >= _HUGE] = 0
    for _ in range(4):
        pos = lo > 0
        high = np.zeros(lo.shape, bool)
        high[pos] = nbinom.cdf(lo[pos] - 1, n[pos], p[pos]) > half
        if not high.any():
            break
        lo[high] -= 1
    return lo


def _plan(mu, n_size, tail, k_max):
    """(start int64[n], size int64[n] needed before the cap, p float64[n])."""
    mu, n_size = _check(mu, n_size)
    p = p_from_mu(n_size, mu)
    start = np.zeros(mu.shape, np.int64)
    size = _upper(n_size, p, tail) + 1
    wide = size > k_max
    if wide.any():
        nw, pw = n_size[wide], p[wide]
        lo = _lower(nw, pw, tail / 2.0)
        hi = _upper(nw, pw, tail / 2.0)
        start[wide] = lo
        size[wide] = np.where(hi >= _HUGE, _HUGE, hi - lo + 1)
    return start, size, p, n_size


def support_size(mu, n_size, tail: float = TAIL, k_max: int = K_MAX) -> np.ndarray:
    """int64[n]: the number of support points each bin needs (rules 1-2 of the module docstring),
    before the cap. A bin is capped by `nb_support` exactly when this exceeds k_max."""
    return _plan(mu, n_size, tail, k_max)[1]


def nb_support(mu, n_size, tail: float = TAIL, k_max: int = K_MAX):
    """(k int64[n, K], w float64[n, K], valid bool[n, K]) — see the module docstring.

    `w` is the NB(n, p) pmf over each row's kept support, renormalised to sum 1; 0 outside `valid`.
    `k` outside `valid` continues the row's arithmetic run (start + j), so h may be evaluated on it
    harmlessly; its weight is 0.
    """
    k_max = int(k_max)
    if k_max < 1:
        raise ValueError("k_max must be >= 1")
    start, size, p, n_size = _plan(mu, n_size, tail, k_max)
    capped = size > k_max
    if capped.any():
        nc, pc = n_size[capped], p[capped]
        mode = np.where(nc > 1.0, np.floor((nc - 1.0) * (1.0 - pc) / pc), 0.0)
        mode = np.minimum(mode, float(_HUGE // 2)).astype(np.int64)
        lo = start[capped]
        hi_start = np.maximum(lo, lo + np.minimum(size[capped], _HUGE // 2) - (k_max + 1))
        start[capped] = np.clip(mode - k_max // 2, lo, hi_start)
        size[capped] = k_max + 1
    n = start.size
    K = int(size.max()) if n else 1
    j = np.arange(K, dtype=np.int64)
    k = start[:, None] + j[None, :]
    valid = j[None, :] < size[:, None]
    kf = k.astype(np.float64)
    nn = n_size[:, None]
    # log pmf up to a per-row constant (removed by the renormalisation)
    logw = gammaln(kf + nn) - gammaln(kf + 1.0) + kf * np.log1p(-p)[:, None]
    logw = np.where(valid, logw, -np.inf)
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(axis=1, keepdims=True)
    return k, w, valid


# ---------------------------------------------------------------------------------------------
# discrete pushed set
# ---------------------------------------------------------------------------------------------


def _clean(v, w):
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if v.shape != w.shape or v.ndim != 2:
        raise ValueError(f"v {v.shape} and w {w.shape} must be the same [n, K]")
    pos = w > 0.0
    v = np.where(pos, v, 0.0)
    w = np.where(pos, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = w / w.sum(axis=1, keepdims=True)
    return v, w


def mean_discrete(v, w) -> np.ndarray:
    """float64[n]: sum_i w_i v_i per row."""
    v, w = _clean(v, w)
    return (w * v).sum(axis=1)


def crps_discrete(v, w, y) -> np.ndarray:
    """float64[n]: exact CRPS of the weighted set {v_i, w_i} against y (sorted one-pass pair sum)."""
    v, w = _clean(v, w)
    y = np.asarray(y, dtype=np.float64).ravel()
    if y.shape[0] != v.shape[0]:
        raise ValueError(f"y has {y.shape[0]} rows, v has {v.shape[0]}")
    e1 = (w * np.abs(v - y[:, None])).sum(axis=1)
    order = np.argsort(v, axis=1, kind="stable")
    vs = np.take_along_axis(v, order, axis=1)
    ws = np.take_along_axis(w, order, axis=1)
    below = np.cumsum(ws, axis=1) - ws
    pair = 2.0 * (ws * vs * (2.0 * below + ws - 1.0)).sum(axis=1)
    return e1 - 0.5 * pair


# ---------------------------------------------------------------------------------------------
# Gaussian mixture (own noise, on log(1 + P))
# ---------------------------------------------------------------------------------------------


def _A(m: np.ndarray, s2: np.ndarray) -> np.ndarray:
    s = np.sqrt(s2)
    z = m / s
    return m * (2.0 * ndtr(z) - 1.0) + 2.0 * s * _INV_SQRT_2PI * np.exp(-0.5 * z * z)


def crps_gauss_mixture(mu, sigma, w, y, chunk: int = 4096) -> np.ndarray:
    """float64[n]: closed-form CRPS of sum_k w_k N(mu_k, sigma_k^2) against y (Grimit et al. 2006)."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if not (mu.shape == sigma.shape == w.shape) or mu.ndim != 2:
        raise ValueError(f"mu {mu.shape}, sigma {sigma.shape}, w {w.shape} must be the same [n, K]")
    y = np.asarray(y, dtype=np.float64).ravel()
    n, K = mu.shape
    if y.shape[0] != n:
        raise ValueError(f"y has {y.shape[0]} rows, mu has {n}")
    pos = w > 0.0
    if np.any(pos & ~(sigma > 0.0)):
        raise ValueError("sigma must be > 0 wherever w > 0")
    mu = np.where(pos, mu, 0.0)
    sigma = np.where(pos, sigma, 1.0)
    w = np.where(pos, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = w / w.sum(axis=1, keepdims=True)
    # each row's width = 1 + its last positive column; rows sorted by width, chunks trimmed
    width = np.where(pos.any(axis=1), K - np.argmax(pos[:, ::-1], axis=1), 1)
    order = np.argsort(width, kind="stable")
    out = np.empty(n, dtype=np.float64)
    chunk = max(1, int(chunk))
    s = 0
    while s < n:
        r = min(chunk, n - s)
        kc = int(width[order[s + r - 1]])
        if r * kc * kc > PAIR_ELEMS:
            r = max(1, PAIR_ELEMS // (kc * kc))
            kc = int(width[order[s + r - 1]])
        idx = order[s:s + r]
        m, sg, ww, yy = mu[idx, :kc], sigma[idx, :kc], w[idx, :kc], y[idx]
        s2 = sg * sg
        e1 = (ww * _A(yy[:, None] - m, s2)).sum(axis=1)
        pair = _A(m[:, :, None] - m[:, None, :], s2[:, :, None] + s2[:, None, :])
        e2 = np.einsum("rj,rjk,rk->r", ww, pair, ww)
        out[idx] = e1 - 0.5 * e2
        s += r
    return out


def mean_gauss_mixture_logp(mu, sigma, w) -> np.ndarray:
    """float64[n]: the P-scale mean of the own-noise prediction, sum_k w_k (exp(mu_k + sigma_k^2/2) - 1)."""
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    pos = w > 0.0
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        w = np.where(pos, w, 0.0)
        w = w / w.sum(axis=1, keepdims=True)
        term = np.where(pos, np.expm1(mu + 0.5 * sigma * sigma), 0.0)
    return (w * term).sum(axis=1)
