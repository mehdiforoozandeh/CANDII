"""Richardson-Lucy deconvolution of overlap counts to strand-summed read starts (t396 chunk C1).

numpy only. The span kernel `w` [17] has `w[REACH + k]` = average reads per start in bin j that
cover bin j + k (k from -REACH to REACH), as in tools/minicandi/read_start.py.

  mix_kernel(kernel, n_reads_strand) -> [17]   read-weighted mix of the two rows
  forward(s, w) -> c        c[i] = sum_k w[REACH + k] s[i - k]   (starts -> overlap counts)
  adjoint(x, w) -> y        the exact transpose of `forward`; y[j] = sum_k w[REACH + k] x[j + k]
                            on interior bins
  richardson_lucy(c, w, n_iter, s0=None, eps=1e-9) -> s >= 0
  match_size(s, w, mu_c, n_c) -> n_s   NB size of the starts by moment matching

Edge rule: `forward` reads bins outside the array as the edge bin (the rule of
`read_start.convert_nb`). `adjoint` is the transpose of that map, so near an edge it adds the
terms of the clipped source bins into the edge bin. With the exact transpose, each RL step keeps
sum(forward(s)) = sum(c) (for forward(s) > eps everywhere).
"""
from __future__ import annotations

import numpy as np

REACH = 8
KERNEL_WIDTH = 2 * REACH + 1
N_MIN, N_MAX = 1e-3, 1e4          # minicandi.contract (ladder.model); a test checks they agree
N_ITER_PRIMARY = 50
N_ITER_GRID = (10, 25, 50, 100, 200)


def _kernel(w) -> np.ndarray:
    w = np.asarray(w, np.float64)
    if w.shape != (KERNEL_WIDTH,):
        raise ValueError(f"kernel {w.shape}, want ({KERNEL_WIDTH},)")
    return w


def mix_kernel(kernel, n_reads_strand) -> np.ndarray:
    """float64 [17]: the rows of a [2, 17] kernel mixed by the read count of each strand."""
    k = np.asarray(kernel, np.float64)
    nr = np.asarray(n_reads_strand, np.float64)
    if k.shape != (2, KERNEL_WIDTH) or nr.shape != (2,):
        raise ValueError(f"kernel {k.shape} and reads {nr.shape}, want (2, {KERNEL_WIDTH}) and (2,)")
    tot = float(nr.sum())
    if tot <= 0:
        raise ValueError("n_reads_strand sums to 0: no reads to weight the kernel rows")
    return (nr[:, None] * k).sum(0) / tot


def forward(s, w) -> np.ndarray:
    """c [..., L]: c[i] = sum_k w[REACH + k] s[clip(i - k)]."""
    s = np.asarray(s, np.float64)
    w = _kernel(w)
    L = s.shape[-1]
    pos = np.arange(L)
    c = np.zeros_like(s)
    for k in range(-REACH, REACH + 1):
        wk = w[REACH + k]
        if wk != 0:
            c += wk * s[..., np.clip(pos - k, 0, L - 1)]
    return c


def adjoint(x, w) -> np.ndarray:
    """y [..., L], the transpose of `forward`: y[j] = sum_{i,k: clip(i - k) = j} w[REACH + k] x[i]."""
    x = np.asarray(x, np.float64)
    w = _kernel(w)
    L = x.shape[-1]
    pos = np.arange(L)
    xt = np.moveaxis(x, -1, 0)
    yt = np.zeros_like(xt)
    for k in range(-REACH, REACH + 1):
        wk = w[REACH + k]
        if wk != 0:
            np.add.at(yt, np.clip(pos - k, 0, L - 1), wk * xt)
    return np.moveaxis(yt, 0, -1)


def richardson_lucy(c, w, n_iter: int, s0=None, eps: float = 1e-9) -> np.ndarray:
    """s [..., L] >= 0 with forward(s, w) ~ c. Start c / w.sum(); step
    s <- s * adjoint(c / max(forward(s), eps)) / adjoint(1)."""
    c = np.asarray(c, np.float64)
    w = _kernel(w)
    if np.any(c < 0):
        raise ValueError("c has negative values; RL needs a non-negative mean")
    s = c / w.sum() if s0 is None else np.array(s0, np.float64, copy=True)
    if s.shape != c.shape:
        raise ValueError(f"s0 {s.shape} and c {c.shape} differ")
    norm = adjoint(np.ones(c.shape[-1]), w)
    norm = np.where(norm > 0, norm, 1.0)
    for _ in range(int(n_iter)):
        s = s * adjoint(c / np.maximum(forward(s, w), eps), w) / norm
    return np.maximum(s, 0.0)


def match_size(s, w, mu_c, n_c) -> np.ndarray:
    """n_s [..., L]. Per count bin nu[i] = n_c[i] * sum_k (w_k s[i-k])^2 / (sum_k w_k s[i-k])^2;
    then n_s[j] = sum_k w_k nu[j+k] / sum_k w_k (edge bins read as the edge value), clamped to
    N_MIN .. N_MAX. A bin with no start mass in reach gets nu = N_MAX. Assumes independent start
    counts, as convert_nb does. `mu_c` is checked for shape only; the formula uses forward(s)."""
    s = np.asarray(s, np.float64)
    w = _kernel(w)
    n_c = np.asarray(n_c, np.float64)
    if np.shape(mu_c) != s.shape or n_c.shape != s.shape:
        raise ValueError(f"s {s.shape}, mu_c {np.shape(mu_c)}, n_c {n_c.shape} must be equal")
    L = s.shape[-1]
    pos = np.arange(L)
    m = np.zeros_like(s)
    q = np.zeros_like(s)
    for k in range(-REACH, REACH + 1):
        t = w[REACH + k] * s[..., np.clip(pos - k, 0, L - 1)]
        m += t
        q += t * t
    nu = np.where(m > 0, n_c * q / np.where(m > 0, m * m, 1.0), N_MAX)
    out = np.zeros_like(s)
    for k in range(-REACH, REACH + 1):
        out += w[REACH + k] * nu[..., np.clip(pos + k, 0, L - 1)]
    return np.clip(out / w.sum(), N_MIN, N_MAX)


__all__ = ["KERNEL_WIDTH", "N_ITER_GRID", "N_ITER_PRIMARY", "N_MAX", "N_MIN", "REACH", "adjoint",
           "forward", "match_size", "mix_kernel", "richardson_lucy"]
