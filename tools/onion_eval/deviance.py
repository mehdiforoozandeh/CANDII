"""Deviance-based count metrics, scored beside the current headline set (task t371).

The current headline metrics stay as they are. This module adds three count-space metrics, one per
goal, all built on the Poisson unit deviance

    d(y, mu) = 2 * [ y * log(y / mu) - (y - mu) ]          (y * log(y / mu) = 0 when y = 0)

and on the normalized deviance D(mu) = sum d(y, mu) / sum d(y, mean(y)) of one track.

Why Poisson deviance (sprint 10 of Max's research repo, `notes/counts.md`, section 4.1 of its report):
it is a Bregman divergence, so for any count distribution with mean lam and a target y drawn
independently of the prediction, E[d(y, mu)] = E[d(y, lam)] + B(lam, mu): the noise floor and the
model's error add exactly. In Max's simulation it was the only unit tested whose floor-corrected
error did not drift with depth, and the Poisson plug-in floor came within 0.01 of the true floor.

Imputation  `dev_imputation_skill` = (D_pa - D_model) / (D_pa - D_floor): the share of the gap from
            the per-position training average (0) to the read-noise floor (1) that the model closes.
Correction  `dev_correction_gap_closed` = (D_depth - D_model) / (D_depth - D_floor) on the never-
            trained pairs: 0 = depth-only scaling, 1 = the read-noise floor of the target pr2.
Calibration `dev_calibration_ratio` R = sum d(y, mu) / sum E_{Y ~ NB(n, mu)} d(Y, mu): the observed
            deviance over the deviance the model's own NB expects. 1 = calibrated, > 1 overconfident
            (the spread is too narrow), < 1 underconfident. `dev_calibration_error` = |log2 R|.
            `nb_logscore_gain_vs_marginal` (mean NB log-likelihood per bin, model minus marginal NB,
            nats) is a proper score of the whole distribution, kept as a background check.

The read-noise floor is the Poisson plug-in floor: sum E_{Y ~ Poisson(lam)} d(Y, lam) with lam the
monotone (isotonic) fit of y on the model's mean, as Max's `cnt_lib.py::dev_scores`. Within one
experiment, read sampling given the rate is Poisson (Max's 52 replicate twins: replicate floor 0.572,
Poisson plug-in floor 0.576), so the floor is Poisson even though the model's predictive is NB.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import numpy as np
from scipy import stats

MU_MIN = 1e-3                  # = baselines.PRED_FLOOR, the floor on a predicted mean

__all__ = ["unit_deviance", "deviance", "poisson_expected_deviance", "nb_expected_deviance",
           "isotonic_fit", "target_terms", "pair_terms", "gap", "calibration_ratio"]


def unit_deviance(y, mu) -> np.ndarray:
    """Per-bin Poisson unit deviance d(y, mu), float64; mu is floored at MU_MIN."""
    y = np.asarray(y, np.float64)
    mu = np.maximum(np.asarray(mu, np.float64), MU_MIN)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return 2.0 * (t - (y - mu))


def deviance(y, mu) -> float:
    return float(unit_deviance(y, mu).sum())


# ------------------------------------------------------------------ expected deviances (tables)
_LGRID = np.linspace(np.log(MU_MIN), np.log(5000.0), 481)     # log mu grid
_NGRID = np.linspace(np.log(1e-2), np.log(1e4), 145)          # log n (NB size) grid
_KMAX = 400000


def _ylogy(k):
    k = np.asarray(k, np.float64)
    return np.where(k > 0, k * np.log(np.where(k > 0, k, 1.0)), 0.0)


@lru_cache(maxsize=1)
def _pois_table() -> np.ndarray:
    out = np.empty(_LGRID.size)
    for i, lm in enumerate(_LGRID):
        m = float(np.exp(lm))
        kmax = int(m + 12 * np.sqrt(m) + 40)
        k = np.arange(kmax + 1)
        p = stats.poisson.pmf(k, m)
        out[i] = 2.0 * ((p * _ylogy(k)).sum() - m * np.log(m))
    return out


def poisson_expected_deviance(lam) -> np.ndarray:
    """E[d(Y, lam)] for Y ~ Poisson(lam), per bin (table on log lam; 1 + 1/(6 lam) above the grid)."""
    lam = np.maximum(np.asarray(lam, np.float64), MU_MIN)
    ll = np.log(lam)
    v = np.interp(ll, _LGRID, _pois_table())
    hi = ll > _LGRID[-1]
    if hi.any():
        v[hi] = 1.0 + 1.0 / (6.0 * lam[hi])
    return v


@lru_cache(maxsize=1)
def _nb_table() -> np.ndarray:
    """E[d(Y, mu)] = 2 (E[Y log Y] - mu log mu) for Y ~ NB(n, mu), on the (log mu, log n) grid.
    A cell whose 1 - 1e-9 quantile is above _KMAX uses the delta approximation 1 + mu / n."""
    T = np.empty((_LGRID.size, _NGRID.size))
    k_all = np.arange(_KMAX + 1)
    yly = _ylogy(k_all)
    for j, ln in enumerate(_NGRID):
        n = float(np.exp(ln))
        for i, lm in enumerate(_LGRID):
            m = float(np.exp(lm))
            p = n / (n + m)
            kq = stats.nbinom.ppf(1 - 1e-9, n, p)
            if not np.isfinite(kq) or kq > _KMAX:
                T[i, j] = 1.0 + m / n
                continue
            k = k_all[: int(kq) + 1]
            pk = stats.nbinom.pmf(k, n, p)
            T[i, j] = 2.0 * ((pk * yly[: k.size]).sum() / pk.sum() - m * np.log(m))
    return T


def nb_expected_deviance(n, mu) -> np.ndarray:
    """E[d(Y, mu)] for Y ~ NB(size n, mean mu), per bin: bilinear interpolation on (log mu, log n).
    Off the grid: mu above it uses 1 + mu / n; n above it is treated as the largest n (near Poisson)."""
    mu = np.maximum(np.asarray(mu, np.float64), MU_MIN)
    n = np.asarray(n, np.float64)
    lm, ln = np.log(mu), np.log(np.clip(n, np.exp(_NGRID[0]), np.exp(_NGRID[-1])))
    T = _nb_table()
    fi = np.clip((lm - _LGRID[0]) / (_LGRID[1] - _LGRID[0]), 0, _LGRID.size - 1 - 1e-9)
    fj = np.clip((ln - _NGRID[0]) / (_NGRID[1] - _NGRID[0]), 0, _NGRID.size - 1 - 1e-9)
    i0, j0 = fi.astype(int), fj.astype(int)
    a, b = fi - i0, fj - j0
    v = ((1 - a) * (1 - b) * T[i0, j0] + a * (1 - b) * T[i0 + 1, j0]
         + (1 - a) * b * T[i0, j0 + 1] + a * b * T[i0 + 1, j0 + 1])
    hi = lm > _LGRID[-1]
    if hi.any():
        v[hi] = 1.0 + mu[hi] / np.maximum(n[hi], 1e-12)
    return v


# ------------------------------------------------------------------ isotonic plug-in rate
def _pav(v, w):
    """Weighted pool-adjacent-violators: the non-decreasing fit of v."""
    vals, wts, cnt = [], [], []
    for x, ww in zip(v, w):
        vals.append(x); wts.append(ww); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w2 = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w2
            wts[-2] = w2; cnt[-2] += cnt[-1]
            vals.pop(); wts.pop(); cnt.pop()
    return np.repeat(vals, cnt)


def isotonic_fit(mu, y, n_bins: int = 512) -> np.ndarray:
    """Monotone fit of y on mu, per bin: equal-count bins of mu, PAV on the bin means, linear
    interpolation between bin centres. Floored at MU_MIN."""
    mu = np.asarray(mu, np.float64)
    y = np.asarray(y, np.float64)
    order = np.argsort(mu, kind="stable")
    groups = np.array_split(order, min(n_bins, max(1, mu.size)))
    groups = [g for g in groups if g.size]
    xm = np.array([mu[g].mean() for g in groups])
    ym = np.array([y[g].mean() for g in groups])
    wt = np.array([g.size for g in groups], np.float64)
    fit = _pav(ym, wt)
    return np.maximum(np.interp(mu, xm, fit), MU_MIN)


# ------------------------------------------------------------------ per-track terms
def target_terms(y, mu, n=None, mu_ref=None, *, n_marg=None, mu_marg=None) -> Dict[str, float]:
    """Sufficient sums of one track (all deviances are sums over its bins).

    D0 deviance of the track's own mean; D_model of mu; D_scaled of c * mu with c = sum y / sum mu;
    D_iso of the monotone fit (the error a monotone recalibration leaves); floor the Poisson plug-in
    floor; D_ref of a reference prediction (position average or depth-only scaling); E_nb the
    deviance the model's NB expects; nll_model / nll_marg the NB negative log-likelihoods (sums)."""
    y = np.asarray(y, np.float64)
    mu = np.maximum(np.asarray(mu, np.float64), MU_MIN)
    lam = isotonic_fit(mu, y)
    c = y.sum() / mu.sum() if mu.sum() > 0 else 1.0
    out = {"bins": int(y.size),
           "D0": deviance(y, np.full_like(y, max(y.mean(), MU_MIN))),
           "D_model": deviance(y, mu),
           "D_scaled": deviance(y, c * mu),
           "D_iso": deviance(y, lam),
           "floor": float(poisson_expected_deviance(lam).sum())}
    if mu_ref is not None:
        out["D_ref"] = deviance(y, mu_ref)
    if n is not None:
        n = np.asarray(n, np.float64)
        out["E_nb"] = float(nb_expected_deviance(n, mu).sum())
        out["nll_model"] = float(-stats.nbinom.logpmf(y, n, n / (n + mu)).sum())
    if n_marg is not None and mu_marg is not None:
        nm = np.asarray(n_marg, np.float64)
        mm = np.maximum(np.asarray(mu_marg, np.float64), MU_MIN)
        out["nll_marg"] = float(-stats.nbinom.logpmf(y, nm, nm / (nm + mm)).sum())
    return out


def pair_terms(y, mu, mu_depth, mu_floorpred=None) -> Dict[str, float]:
    """A correction pair: target pr2 y, the model's mean mu, depth-only scaling's mean, and the
    pseudoreplicate floor predictor's mean (target pr1 -> pr2), as sums."""
    t = target_terms(y, mu, mu_ref=mu_depth)
    t["D_depth"] = t.pop("D_ref")
    if mu_floorpred is not None:
        t["D_floorpred"] = deviance(y, mu_floorpred)
    return t


# ------------------------------------------------------------------ aggregates over records
def _norm_mean(recs, key):
    return float(np.mean([r[key] / r["D0"] for r in recs]))


def gap(recs, base_key: str, model_key: str = "D_model", floor_key: str = "floor") -> float:
    """(mean D_base - mean D_model) / (mean D_base - mean D_floor), each D normalized by its track's
    D0 first; a ratio of target means. NaN when the denominator is not positive."""
    if not recs:
        return float("nan")
    b, m, f = (_norm_mean(recs, k) for k in (base_key, model_key, floor_key))
    return float((b - m) / (b - f)) if b - f > 0 else float("nan")


def calibration_ratio(recs) -> float:
    """sum over targets of D_model / D0 over the same sum of E_nb / D0 (a ratio of target means)."""
    if not recs:
        return float("nan")
    return float(_norm_mean(recs, "D_model") / _norm_mean(recs, "E_nb"))
