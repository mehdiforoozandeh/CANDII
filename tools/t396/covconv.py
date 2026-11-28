"""Start-to-count NB conversion that keeps the covariance between start counts (t396 chunk C3).

`read_start.convert_nb` (numpy port `t392/conv_sparsity.convert_nb_np`) moment-matches the overlap
count of bin i as a sum of independent thinned start counts. On observed data the start counts of
neighbouring bins share a rate, so their excess parts (the NB variance above Poisson, mu^2 / n)
are correlated and the independent sum understates the variance (t394 INVESTIGATION.md, t395
SMOOTHING.md). This module adds the correlation as an explicit term.

  MAXLAG = 16; rho float64 [2, 2, 2 * MAXLAG + 1]
      rho[s, s2, MAXLAG + l] = correlation of the excess parts of the start count of strand s at
      bin j and strand s2 at bin j + l; rho[s, s, MAXLAG] = 1.
  identity_rho() -> rho          1 on the same strand at lag 0, else 0 (= the independent sum)
  convert_nb_cov(mu_sub, n_sub, w, stranded, rho) -> (mu, n)   one array [L, 2]
  estimate_rho(y_sub, mu_sub, n_sub) -> (num, den)   additive parts over regions and targets
  finish_rho(num, den) -> rho
  scale_size(n, c) -> n / c;  fit_scale(records) -> c   (calibration_ratio = 1 on the records)

Conversion. c[k, s] = w[s, REACH + k] * mu_sub[i - k, s] is the mean contribution of the starts of
bin i - k, strand s, to bin i. mu = sum c. The excess variance is the quadratic form
    v = sum over (k, s), (k2, s2) of rho[s, s2, MAXLAG + lag] * c[k, s] c[k2, s2] / sqrt(n n2)
with lag the bin offset from source bin i - k to source bin i - k2, i.e. lag = k - k2. For a rho
with the symmetry rho[s, s2, l] = rho[s2, s, -l] (every rho of finish_rho has it) the quadratic
form is the same with lag = k2 - k. n = mu^2 / v; v <= 0 gives N_MAX; n is clamped to
N_MIN .. N_MAX; n_sub is floored at N_MIN; bins outside the array take the edge bin's values (each
k still counts as its own virtual bin, as in convert_nb_np). numpy only.
"""
from __future__ import annotations

import numpy as np

REACH = 8                       # read_start.REACH
KW = 2 * REACH + 1
N_MIN, N_MAX = 1e-3, 1e4        # contract / ladder.model clamps used by convert_nb(_np)
MAXLAG = 16
NLAG = 2 * MAXLAG + 1


def identity_rho() -> np.ndarray:
    rho = np.zeros((2, 2, NLAG))
    rho[0, 0, MAXLAG] = rho[1, 1, MAXLAG] = 1.0
    return rho


def _check_rho(rho) -> np.ndarray:
    rho = np.asarray(rho, np.float64)
    if rho.shape != (2, 2, NLAG):
        raise ValueError(f"rho {rho.shape}, want (2, 2, {NLAG})")
    return rho


def convert_nb_cov(mu_sub, n_sub, w, stranded: bool, rho):
    """(mu, n) [L] of overlap counts from read-start (mu_sub, n_sub) [L, 2]; see module docstring.
    `w` is [2, 17] or [17] (row 0 only; enough for an unstranded track)."""
    mu_sub = np.asarray(mu_sub, np.float64)
    n_sub = np.maximum(np.asarray(n_sub, np.float64), N_MIN)
    if mu_sub.shape != n_sub.shape or mu_sub.ndim != 2 or mu_sub.shape[1] != 2:
        raise ValueError(f"mu_sub {mu_sub.shape} and n_sub {n_sub.shape} must be equal [L, 2]")
    w = np.asarray(w, np.float64)
    if w.ndim == 1:
        w = w[None, :]
    if w.shape[-1] != KW or (stranded and w.shape[0] != 2):
        raise ValueError(f"kernel {w.shape}, want (2, {KW}) (or ({KW},) unstranded)")
    rho = _check_rho(rho)
    strands = (0, 1) if stranded else (0,)
    L = len(mu_sub)
    pos = np.arange(L)
    ks = np.arange(-REACH, REACH + 1)
    cols, sk = [], []                      # columns c / sqrt(n), one per (k, s)
    mu = np.zeros(L)
    for k in ks:
        src = np.clip(pos - k, 0, L - 1)
        for s in strands:
            c = w[s, REACH + k] * mu_sub[src, s]
            mu += c
            cols.append(c / np.sqrt(n_sub[src, s]))
            sk.append((s, k))
    Q = np.stack(cols, 1)                  # [L, P]
    s_ = np.array([a for a, _ in sk])
    k_ = np.array([b for _, b in sk])
    lag = k_[:, None] - k_[None, :]        # bin offset from source a to source b
    M = rho[s_[:, None], s_[None, :], MAXLAG + lag]
    v = np.einsum("ia,ab,ib->i", Q, M, Q)
    n = np.where(v > 0, mu * mu / np.where(v > 0, v, 1.0), N_MAX)
    return mu, np.clip(n, N_MIN, N_MAX)


def _regions(a) -> np.ndarray:
    a = np.asarray(a, np.float64)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[-1] != 2:
        raise ValueError(f"array {a.shape}, want [L, 2] or [R, L, 2] (R regions)")
    return a


def estimate_rho(y_sub, mu_sub, n_sub):
    """(num, den) float64 [2, 2, 33] for one array [L, 2] or regions [R, L, 2]; bin pairs never
    cross a region. num = sum r[i, s] r[i + l, s2] (r = y - mu); den = sum mu[i, s] mu[i + l, s2] /
    sqrt(n[i, s] n[i + l, s2]). Add the parts over regions and targets, then call finish_rho."""
    y, mu = _regions(y_sub), _regions(mu_sub)
    n = np.maximum(_regions(n_sub), N_MIN)
    if not (y.shape == mu.shape == n.shape):
        raise ValueError(f"shapes {y.shape}, {mu.shape}, {n.shape} differ")
    r = y - mu
    e = mu / np.sqrt(n)
    L = y.shape[1]
    num = np.zeros((2, 2, NLAG))
    den = np.zeros((2, 2, NLAG))
    for l in range(-MAXLAG, MAXLAG + 1):
        if abs(l) >= L:
            continue
        a = slice(max(0, -l), L - max(0, l))   # bins i with i + l inside the region
        b = slice(max(0, l), L - max(0, -l))
        num[:, :, MAXLAG + l] = np.einsum("ris,rit->st", r[:, a], r[:, b])
        den[:, :, MAXLAG + l] = np.einsum("ris,rit->st", e[:, a], e[:, b])
    return num, den


def finish_rho(num, den) -> np.ndarray:
    """rho = clip(num / den, -1, 1) (0 where den is 0); the lag-0 same-strand entries set to 1."""
    num, den = _check_rho(num), _check_rho(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    rho = np.clip(rho, -1.0, 1.0)
    rho[0, 0, MAXLAG] = rho[1, 1, MAXLAG] = 1.0
    return rho


# ------------------------------------------------------------------ equal recalibration (C9)
def scale_size(n, c):
    return np.asarray(n, np.float64) / c


def fit_scale(records, lo: float = 1e-3, hi: float = 1e3, iters: int = 60) -> float:
    """The scalar c with deviance.calibration_ratio = 1 when every record's n is replaced by n / c.

    A record is a dict with arrays "y", "mu", "n" of one target (one assay's records per call).
    D_model and D0 are those of deviance.target_terms; E_nb(c) = sum nb_expected_deviance(n / c,
    mu). The ratio falls as c grows (a smaller size expects more deviance); c is found by bisection
    on log c in [lo, hi] and returned clipped to that range when the root lies outside it."""
    import sys
    from pathlib import Path
    tools = str(Path(__file__).resolve().parents[1])
    if tools not in sys.path:
        sys.path.insert(0, tools)
    from onion_eval import deviance as dev

    if not records:
        return float("nan")
    pre = []
    for r in records:
        y = np.asarray(r["y"], np.float64).ravel()
        mu = np.maximum(np.asarray(r["mu"], np.float64).ravel(), dev.MU_MIN)
        n = np.asarray(r["n"], np.float64).ravel()
        D0 = dev.deviance(y, np.full_like(y, max(y.mean(), dev.MU_MIN)))
        pre.append((mu, n, D0, dev.deviance(y, mu)))
    m_model = np.mean([d / d0 for _, _, d0, d in pre])

    def log_ratio(log_c):
        c = np.exp(log_c)
        e = np.mean([dev.nb_expected_deviance(scale_size(n, c), mu).sum() / d0
                     for mu, n, d0, _ in pre])
        return np.log(m_model) - np.log(e)

    a, b = np.log(lo), np.log(hi)
    fa, fb = log_ratio(a), log_ratio(b)
    if fa <= 0:
        return float(lo)
    if fb >= 0:
        return float(hi)
    for _ in range(iters):
        m = 0.5 * (a + b)
        if log_ratio(m) > 0:
            a = m
        else:
            b = m
    return float(np.exp(0.5 * (a + b)))
