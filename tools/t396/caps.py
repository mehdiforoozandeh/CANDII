"""Closed-form cap laws for deduplication and SE/PE thinning, their fits and pair statistics (t396 C4).

The dedup-on track is a deterministic subset of the dedup-off reads. So the start-space law is
conditional on the observed dedup-off count ``c`` of a strand in a 25 bp bin: the ``c`` read starts
fall on ``K = 25 * units`` positions (one slot per position per dedup unit) with symmetric
Dirichlet(a) positional weights, and deduplication keeps one read per occupied slot. The expected
number of kept reads is the Dirichlet-multinomial expected number of distinct occupied slots
(``dm_distinct``). ``a = inf`` is uniform positions (``occupancy``). One shape ``a`` per assay.

The count-space comparator is the saturation law ``on = Kc (1 - exp(-off / Kc))``,
``Kc = kappa (L + 25) / 25`` (Max's sprints/07 notes/covariates.md section 5.2).

``start_cap`` is the unconditional Gamma-Poisson form of h58 (input: the expected count ``lam``);
it is a secondary row only.

Pair statistics are sparse joint tables ``{(source value, target value): number of bins}`` per
scope; every deviance, fit and floor is computed from them. Deviances are Poisson deviances from
``tools/onion_eval/deviance.py``. Within one gap closed, every deviance is on the same target at the
target's own depth. Raw deviances are never compared across spaces; gaps closed are.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable, Dict, Iterable, Sequence, Tuple, Union

import numpy as np
from scipy import optimize, special, stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onion_eval.deviance import unit_deviance  # noqa: E402

Hist = Dict[Tuple[float, float], int]

A_BOUNDS = (1e-3, 1e3)       # bounds of a fitted shape a or kappa
TOP_FRACTION = 0.01          # the top1pct scope: the 1% of bins with the largest source

__all__ = ["K_of", "dm_distinct", "occupancy", "thinned_dm_distinct", "start_cap",
           "count_saturation", "joint_hist", "merge_hists", "depth_ratio", "fit_shape",
           "fit_kappa", "deviance_of", "simulate_floor", "gap_closed"]


# ------------------------------------------------------------------ laws
def K_of(units) -> float:
    """Positions that can each keep one read per strand in a 25 bp bin of a view with ``units``
    dedup units (deduplication ran per unit, so a position holds one read per unit)."""
    return 25.0 * units


def occupancy(c, K):
    """Expected distinct positions when ``c`` reads fall uniformly on ``K`` positions."""
    c = np.asarray(c, np.float64)
    return K * (1.0 - np.exp(c * np.log1p(-1.0 / K))) if K > 1 else np.minimum(c, 1.0) * K


def dm_distinct(c, a, K):
    """Expected distinct positions when ``c`` reads fall on ``K`` positions with symmetric
    Dirichlet(a) weights: K (1 - P(one position empty)), the empty probability being the
    beta-binomial P(0) = G((K-1)a + c) G(Ka) / (G((K-1)a) G(Ka + c)). ``a = inf`` -> occupancy."""
    c = np.asarray(c, np.float64)
    if np.isinf(a):
        return occupancy(c, K)
    lg = special.gammaln
    log_p0 = lg((K - 1) * a + c) + lg(K * a) - lg((K - 1) * a) - lg(K * a + c)
    return K * (1.0 - np.exp(log_p0))


def thinned_dm_distinct(c, q, a, K):
    """Mean of ``dm_distinct(m, a, K)`` over m ~ Binomial(c, q) (one mate kept with probability q).
    ``c`` must hold non-negative integers."""
    c = np.asarray(c)
    ci = np.rint(c).astype(np.int64)
    if np.any(np.abs(c - ci) > 1e-9) or np.any(ci < 0):
        raise ValueError("thinned_dm_distinct needs non-negative integer counts c")
    if q == 1.0:
        return dm_distinct(ci, a, K)
    uniq, inv = np.unique(ci, return_inverse=True)
    out = np.empty(uniq.size, np.float64)
    for i, cu in enumerate(uniq):
        m = np.arange(cu + 1)
        out[i] = float(np.dot(stats.binom.pmf(m, cu, q), dm_distinct(m, a, K)))
    return out[inv].reshape(ci.shape)


def start_cap(lam, a, K):
    """Unconditional Gamma-Poisson cap of h58: K (1 - (1 + lam / (K a))^(-a)); ``lam`` is the
    expected number of read starts before deduplication. Secondary row only."""
    lam = np.asarray(lam, np.float64)
    if np.isinf(a):
        return K * (1.0 - np.exp(-lam / K))
    return K * (1.0 - np.exp(-a * np.log1p(lam / (K * a))))


def count_saturation(c, kappa, read_length, res=25):
    """Count-space saturation law: Kc (1 - exp(-c / Kc)), Kc = kappa (L + res) / res."""
    c = np.asarray(c, np.float64)
    Kc = kappa * (read_length + res) / res
    return Kc * (-np.expm1(-c / Kc))


# ------------------------------------------------------------------ joint tables
def _table(s: np.ndarray, t: np.ndarray) -> Hist:
    if s.size == 0:
        return {}
    pairs, n = np.unique(np.stack([s, t], 1), axis=0, return_counts=True)
    return {(_key(p[0]), _key(p[1])): int(k) for p, k in zip(pairs, n)}


def _key(v):
    f = float(v)
    return int(f) if f.is_integer() else f


def joint_hist(source, target, top_mask=None) -> Dict[str, Hist]:
    """Sparse joint counts ``{(source value, target value): bins}`` for the scopes ``genome`` and
    ``top1pct``. The top 1% is the ceil(1% of bins) with the largest ``source`` (ties broken by
    position), or ``top_mask`` when given; it is never chosen by the target."""
    s = np.asarray(source, np.float64).ravel()
    t = np.asarray(target, np.float64).ravel()
    if s.shape != t.shape:
        raise ValueError(f"source {s.shape} and target {t.shape} differ")
    if top_mask is None:
        n_top = int(math.ceil(TOP_FRACTION * s.size)) if s.size else 0
        top = np.zeros(s.size, bool)
        top[np.argsort(-s, kind="stable")[:n_top]] = True
    else:
        top = np.asarray(top_mask, bool).ravel()
    return {"genome": _table(s, t), "top1pct": _table(s[top], t[top])}


def merge_hists(hists: Iterable[Hist]) -> Hist:
    """Add joint tables of one scope (over chromosomes, strands or experiments)."""
    out: Hist = {}
    for h in hists:
        for k, n in h.items():
            out[k] = out.get(k, 0) + n
    return out


def _arrays(hist: Hist):
    if "genome" in hist or "top1pct" in hist:
        raise ValueError("pass one scope of joint_hist (hist['genome'] or hist['top1pct'])")
    if not hist:
        z = np.zeros(0)
        return z, z, z
    keys = np.array(list(hist.keys()), np.float64)
    return keys[:, 0], keys[:, 1], np.array(list(hist.values()), np.float64)


def depth_ratio(hist: Hist) -> float:
    """Total target over total source in a joint table (the depth-scaling factor when the table is
    the whole track)."""
    s, t, w = _arrays(hist)
    return float((t * w).sum() / (s * w).sum())


# ------------------------------------------------------------------ deviances and fits
def deviance_of(hist: Hist, law: Callable) -> float:
    """Poisson deviance of the target under ``law(source)``, summed over the bins of the table."""
    s, t, w = _arrays(hist)
    if s.size == 0:
        return 0.0
    return float((w * unit_deviance(t, law(s))).sum())


def _as_list(x, n):
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError("per-table parameters must match the number of tables")
        return list(x)
    return [x] * n


def _fit(dev_of_loga: Callable[[float], float]) -> float:
    lo, hi = np.log(A_BOUNDS[0]), np.log(A_BOUNDS[1])
    r = optimize.minimize_scalar(dev_of_loga, bounds=(lo, hi), method="bounded",
                                 options={"xatol": 1e-4})
    return float(np.exp(r.x))


def fit_shape(hist: Union[Hist, Sequence[Hist]], K, q=1.0) -> float:
    """One shape ``a`` minimizing the Poisson deviance of the target under
    ``thinned_dm_distinct(source, q, a, K)`` (``dm_distinct`` when q = 1); bounded 1-D search on
    log a in [1e-3, 1e3]. ``hist`` may be a list of tables (pairs of one assay) with ``K`` and ``q``
    scalars or lists of the same length; their deviances add."""
    hs = list(hist) if isinstance(hist, (list, tuple)) else [hist]
    Ks, qs = _as_list(K, len(hs)), _as_list(q, len(hs))

    def dev(la):
        a = float(np.exp(la))
        return sum(deviance_of(h, lambda s, k=k, qq=qq: thinned_dm_distinct(s, qq, a, k))
                   for h, k, qq in zip(hs, Ks, qs))
    return _fit(dev)


def fit_kappa(hist: Union[Hist, Sequence[Hist]], read_length) -> float:
    """One ``kappa`` minimizing the Poisson deviance of the target under
    ``count_saturation(source, kappa, read_length)``; bounded 1-D search on log kappa in
    [1e-3, 1e3]. Lists of tables and read lengths add as in ``fit_shape``."""
    hs = list(hist) if isinstance(hist, (list, tuple)) else [hist]
    Ls = _as_list(read_length, len(hs))

    def dev(lk):
        kappa = float(np.exp(lk))
        return sum(deviance_of(h, lambda s, L=L: count_saturation(s, kappa, L))
                   for h, L in zip(hs, Ls))
    return _fit(dev)


# ------------------------------------------------------------------ attainable floor
def _draw_distinct(m: np.ndarray, a: float, K: int, rng: np.random.Generator) -> np.ndarray:
    """Distinct occupied positions for each read number in ``m`` (one fresh Dirichlet per draw)."""
    if np.isinf(a):
        p = np.full((m.size, K), 1.0 / K)
    else:
        g = rng.gamma(a, 1.0, size=(m.size, K))
        z = g.sum(1, keepdims=True)
        bad = z[:, 0] <= 0                  # all gammas underflow at tiny a: one position wins
        if bad.any():
            g[bad] = 0.0
            g[bad, rng.integers(0, K, bad.sum())] = 1.0
            z = g.sum(1, keepdims=True)
        p = g / z
    return (rng.multinomial(m, p) > 0).sum(1)


def simulate_floor(hist: Hist, a, K, q=1.0, rng=None, n_rep=20,
                   max_bins_per_value=500) -> float:
    """Attainable floor of the law: the expected Poisson deviance, under the law's mean
    ``thinned_dm_distinct(source, q, a, K)``, of targets drawn from the Dirichlet-multinomial
    occupancy given the source counts (m ~ Binomial(source, q) reads on K positions). For each
    source value with w bins, ``n_rep * min(w, max_bins_per_value)`` draws estimate the per-bin
    expectation, which is then weighted by w. Same units as ``deviance_of`` on the same table."""
    rng = np.random.default_rng() if rng is None else rng
    Ki = int(round(K))
    if abs(Ki - K) > 1e-9:
        raise ValueError("simulate_floor needs an integer K")
    s, _, w = _arrays(hist)
    if s.size == 0:
        return 0.0
    us = np.unique(s)
    ws = np.array([w[s == u].sum() for u in us])
    mu = thinned_dm_distinct(us, q, a, Ki)
    total = 0.0
    for u, wu, mu_u in zip(us, ws, mu):
        n = int(n_rep * min(wu, max_bins_per_value))
        m = np.full(n, int(u), np.int64)
        if q != 1.0:
            m = rng.binomial(m, q)
        y = _draw_distinct(m, a, Ki, rng)
        total += wu * float(unit_deviance(y, np.full(n, mu_u)).mean())
    return total


def gap_closed(D_base, D_law, D_floor):
    """(D_base - D_law) / (D_base - D_floor): 1 when the law reaches its attainable floor, 0 when it
    is no better than the baseline (identity or depth scaling)."""
    return (D_base - D_law) / (D_base - D_floor)
