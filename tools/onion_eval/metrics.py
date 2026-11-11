"""The 7 headline metrics of the onion eval protocol, as pure numpy functions.

Source of the definitions: `cruxvault/INSTRUCTIONS/peeling_the_onion/onion eval protocol.md`
(section headline-metric-set) and `long-horizon decisions.md` (D12, D13), with the PI rulings of
2026-11-10 (metric 1 in two spaces; metric 4 floor = half against half; uncertainty term =
1 - PIT deviation(arm) / PIT deviation(marginal NB)).

Every metric takes arrays that are already in the chosen space. `to_space` is the only function that
applies a space (`counts_log1p` or `signal_arcsinh`); no metric hard-wires one.

| # | metric                                              | function                          |
|---|-----------------------------------------------------|-----------------------------------|
| 1 | recovered share of specific variance above noise    | shared_specific, noise_floor,     |
|   |                                                     | specific_variance_recovered       |
| 2 | r_spec                                              | r_spec                            |
| 3 | count CRPS skill against the per-position average   | crps_split, crps_skill            |
| 4 | gap closed against depth-only scaling               | crps_split, gap_closed            |
| 5 | depth-law error                                     | depth_law_error                   |
| 6 | non-randomized PIT deviation, pooled / worst stratum| pit_deviation                     |
| 7 | predicted over replicate variance                   | variance_ratio                    |
| - | D13 uncertainty term                                | uncertainty_gap_closed            |

An undefined metric never raises. A scalar result is then an `Undefined` (a float NaN that carries a
`.reason`), and a dict result carries a `"reason"` key. A defined dict result has `"reason": ""`.

Reused, not duplicated: `candi.metrics.nb_crps` and `calibration_pit_curve` (21-point grid),
`candi.bench.distributional.oracle_scale`, `marginal_nb`, `p_from_mu`, and
`tools/eic_decomp/errdecomp.py` (`haar`, `gram_by_scale`, `shared_specific`, `calibrate`).
`r_spec` is ported from Max's sprint 03 `scripts/metrics/metrics_lib.py::rspec`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from candi.bench.distributional import marginal_nb, oracle_scale, p_from_mu  # noqa: E402
from candi.metrics import PIT_GRID, calibration_pit_curve, nb_crps  # noqa: E402
from eic_decomp import errdecomp as _ed  # noqa: E402

__all__ = [
    "SPACES", "Undefined", "to_space",
    "shared_specific", "noise_floor", "specific_variance_recovered",
    "Level", "r_spec",
    "crps_split", "crps_skill", "gap_closed", "uncertainty_gap_closed",
    "depth_law_error", "pit_deviation", "variance_ratio", "marginal_nb",
]

SPACES = ("counts_log1p", "signal_arcsinh")
J = _ed.J                       # Haar levels (14: 50 bp .. 409.6 kb, plus the coarse remainder)


class Undefined(float):
    """A NaN that says why the metric is undefined. Behaves as float('nan') everywhere."""

    def __new__(cls, reason: str):
        obj = float.__new__(cls, float("nan"))
        obj.reason = reason
        return obj

    def __repr__(self):
        return f"Undefined({self.reason!r})"


def _f64(x):
    return np.asarray(x, dtype=np.float64)


# --------------------------------------------------------------------------------------------- space

def to_space(x, space: str, depth: Optional[float] = None) -> np.ndarray:
    """Put x into a metric space. The only place a space is applied.

    counts_log1p:   log(1 + depth * x). `depth` is the factor that brings x to the target's own depth
                    (None or 1.0: x is already at the target's depth, e.g. an NB mean decoded with the
                    target's own C').
    signal_arcsinh: asinh(x), x an assay pipeline signal track. Signal is already depth-normalized, so
                    `depth` must be None or 1.0.
    """
    x = _f64(x)
    if space == "counts_log1p":
        s = 1.0 if depth is None else float(depth)
        return np.log1p(s * x)
    if space == "signal_arcsinh":
        if depth not in (None, 1, 1.0):
            raise ValueError("signal_arcsinh takes no depth rescale")
        return np.arcsinh(x)
    raise ValueError(f"unknown space {space!r}; expected one of {SPACES}")


# --------------------------------------------------------------------------------------------- metric 1

def _pad(M, J_):
    L = M.shape[-1]
    Lp = -(-L // (1 << J_)) * (1 << J_)
    if Lp == L:
        return M
    out = np.zeros(M.shape[:-1] + (Lp,), dtype=M.dtype)
    out[..., :L] = M
    return out


def shared_specific(truth, pred, groups, noise=None, *, J: int = J, calibrated: bool = True) -> Dict:
    """Shared / specific split of truth variance and error, per target, in units of each target's Var(x).

    truth, pred: [T, M], T targets of ONE assay over the same M bins, already in the chosen space.
    groups: [T] labels (ENCODE term); a target's shared part is estimated only from targets of other
    groups (errdecomp.shared_specific). noise: [T] noise floor N per target in units of Var(x) (see
    `noise_floor`), or None.

    Per target: x is centred and divided by SD(x); the prediction is first calibrated to x (the best
    monotone map, errdecomp.calibrate; its level error is reported as `level`) when `calibrated`, then
    put on the same standardization. Gram matrices by Haar scale of the truth (V), prediction (P), the
    symmetrized cross term (C) and the error E = V - 2C + P are split into shared and specific parts
    (Max s06 aggregate2.py). Zero padding to a multiple of 2^J preserves every inner product.

    Returns a dict of [T] arrays: V_sh, V_sp, E_sh, E_sp, P_sh, P_sp, C_sh, C_sp, N, level; `by_level`
    (same names, [J+1, T]); and `reason`, a list of [T] strings ('' when defined).
    """
    X, Y = _f64(truth), _f64(pred)
    if X.ndim != 2 or X.shape != Y.shape:
        raise ValueError("truth and pred must both be [T, M]")
    T, M = X.shape
    groups = np.asarray(groups)
    reason = [""] * T
    Xs, Ys = np.zeros_like(X), np.zeros_like(Y)
    level = np.full(T, np.nan)
    ok = np.zeros(T, bool)
    for t in range(T):
        x, y = X[t], Y[t]
        mx, sx = x.mean(), x.std()
        if not sx > 0:
            reason[t] = "truth has zero variance"
            continue
        if calibrated:
            terms, yc = _ed.calibrate(x, y)
            level[t] = terms["offset"] + terms["scale"] + terms["nonlinear"]
        else:
            yc, level[t] = y, np.nan
        Xs[t], Ys[t] = (x - mx) / sx, (yc - mx) / sx
        ok[t] = True
    Xp, Yp = _pad(Xs, J), _pad(Ys, J)
    Gx = _ed.gram_by_scale(Xp, J) / M
    Gy = _ed.gram_by_scale(Yp, J) / M
    cx, cy = _ed.haar(Xp, J), _ed.haar(Yp, J)
    Gxy = np.stack([c1 @ c2.T for c1, c2 in zip(cx, cy)]) / M
    Gc = 0.5 * (Gxy + np.transpose(Gxy, (0, 2, 1)))
    Ge = Gx - 2 * Gc + Gy
    # targets with an undefined standardization take no part in any other target's shared estimate
    idx = np.flatnonzero(ok)
    out: Dict = {"by_level": {}}
    for nm, G in (("V", Gx), ("P", Gy), ("C", Gc), ("E", Ge)):
        sh = np.full((J + 1, T), np.nan)
        sp = np.full((J + 1, T), np.nan)
        if idx.size:
            s1, s2 = _ed.shared_specific(G[:, idx][:, :, idx], groups[idx])
            sh[:, idx], sp[:, idx] = s1, s2
        out["by_level"][f"{nm}_sh"], out["by_level"][f"{nm}_sp"] = sh, sp
        out[f"{nm}_sh"], out[f"{nm}_sp"] = sh.sum(0), sp.sum(0)
    for t in idx:
        if not np.isfinite(out["V_sh"][t]):
            reason[t] = "no target of another group in this assay"
    out["N"] = np.full(T, np.nan) if noise is None else _f64(noise).reshape(T)
    out["level"] = level
    out["reason"] = reason
    return out


def noise_floor(pr1, pr2, target, depth_ratio: float = 2.0, *, J: int = J) -> float:
    """Sampling-noise floor N of a target track, in units of Var(target).

    pr1, pr2: the target's two pseudoreplicate halves (same space as target). At each Haar level k,
    R_k = corr of the halves' level-k coefficients (clipped to [0, 1]); the noise share of a track at
    `depth_ratio` times the half depth (fragments) is 1 - k R_k / (1 + (k - 1) R_k) (Spearman-Brown).
    N = sum_k share_k * (energy share of the target at level k). Undefined if a vector is constant.
    """
    H = np.stack([_f64(pr1), _f64(pr2), _f64(target)])
    H = H - H.mean(1, keepdims=True)
    if not (H.std(1) > 0).all():
        return Undefined("a half or the target has zero variance")
    G = _ed.gram_by_scale(_pad(H, J), J)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = G[:, 0, 1] / np.sqrt(G[:, 0, 0] * G[:, 1, 1])
    R = np.clip(np.nan_to_num(R, nan=0.0), 0.0, 1.0)
    k = float(depth_ratio)
    share = 1.0 - k * R / (1.0 + (k - 1.0) * R)
    xs = G[:, 2, 2] / G[:, 2, 2].sum()
    return float(np.dot(share, xs))


def specific_variance_recovered(E_sp, V_sp, N) -> float:
    """Headline 1: 1 - (mean E_sp - mean N) / (mean V_sp - mean N), a ratio of means over targets.

    Arrays [T] (or scalars) in units of each target's Var(x). Targets with any non-finite input are
    dropped. 1 = all specific variance above the noise floor recovered; 0 = none (error as large as the
    specific variance). Undefined when no target is left or mean V_sp <= mean N.
    """
    E, V, Nn = (np.atleast_1d(_f64(v)) for v in (E_sp, V_sp, N))
    E, V, Nn = np.broadcast_arrays(E, V, Nn)
    m = np.isfinite(E) & np.isfinite(V) & np.isfinite(Nn)
    if not m.any():
        return Undefined("no target with finite E_sp, V_sp and N")
    e, v, n = E[m].mean(), V[m].mean(), Nn[m].mean()
    if not v - n > 0:
        return Undefined("no specific variance above the noise floor (mean V_sp <= mean N)")
    return float(1.0 - (e - n) / (v - n))


# --------------------------------------------------------------------------------------------- metric 2

_EPS = 2e-3   # residual SD relative to the SD of the vector below which it is "a function of a"


def _corr(x, y):
    x = _f64(x) - np.mean(x)
    y = _f64(y) - np.mean(y)
    d = np.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / d) if d > 0 else float("nan")


def _make_knots(a, n_knots, tail_knots, min_count):
    """Knots at quantiles of a, ties collapsed, intervals with fewer than min_count bins merged."""
    s = np.sort(a)
    n = s.size
    if n == 0 or s[0] == s[-1]:
        return s[:1]
    K = int(max(2, min(n_knots, n // max(min_count, 1))))
    Kt = int(min(tail_knots, K // 4)) if K >= 40 else 0
    if Kt:
        probs = np.concatenate([np.linspace(0.0, 0.99, K - Kt + 1), np.linspace(0.99, 1.0, Kt + 1)[1:]])
    else:
        probs = np.linspace(0.0, 1.0, K + 1)
    pos = probs * (n - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, n - 1)
    k = np.unique(s[lo] + (pos - lo) * (s[hi] - s[lo]))
    if k.size < 2:
        return k
    below = np.searchsorted(s, k, side="left")
    keep = [0]
    for j in range(1, k.size):
        if below[j] - below[keep[-1]] >= min_count:
            keep.append(j)
    if keep[-1] != k.size - 1:
        if len(keep) > 1:
            keep[-1] = k.size - 1
        else:
            keep.append(k.size - 1)
    return k[keep]


class Level:
    """Continuous piecewise-linear function of `a` (hat basis at quantile knots), least-squares fit.
    Ported from Max s03 metrics_lib.py::Level."""

    def __init__(self, a, n_knots=400, tail_knots=80, min_count=1000):
        a = _f64(a)
        self.n = a.size
        self.knots = _make_knots(a, n_knots, tail_knots, min_count)
        m = self.knots.size
        self.const = m < 2
        if self.const:
            return
        k = self.knots
        idx = np.clip(np.searchsorted(k, a, side="right") - 1, 0, m - 2)
        t = np.clip((a - k[idx]) / (k[idx + 1] - k[idx]), 0.0, 1.0)
        w0 = 1.0 - t
        d = np.bincount(idx, weights=w0 * w0, minlength=m) + np.bincount(idx + 1, weights=t * t, minlength=m)
        o = np.bincount(idx, weights=w0 * t, minlength=m)[: m - 1]
        A = np.diag(d) + np.diag(o, 1) + np.diag(o, -1)
        A[np.diag_indices(m)] += 1e-12 * d.mean()
        try:
            self._Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self._Ainv = np.linalg.pinv(A)
        self._idx, self._t = idx, t

    def fit(self, v):
        v = _f64(v)
        if self.const:
            return np.full(self.n, v.mean() if v.size else 0.0)
        m = self.knots.size
        idx, t = self._idx, self._t
        b = np.bincount(idx, weights=(1.0 - t) * v, minlength=m) + np.bincount(idx + 1, weights=t * v, minlength=m)
        c = self._Ainv @ b
        return (1.0 - t) * c[idx] + t * c[idx + 1]


def r_spec(y, p, a, *, mask=None, n_knots: int = 400, tail_knots: int = 80, min_count: int = 1000,
           guard: bool = True) -> float:
    """Headline 2: corr(y - g(a), p - h(a)); g, h the Level fits of truth y and prediction p on the
    training average a (fit and scored on the bins in `mask`).

    0.0 when p is a function of a (residual SD <= 0.2% of SD(p), the guard), so the average and any
    linear rescale of it score exactly 0. Undefined when the truth is a function of a.
    """
    y, p, a = _f64(y), _f64(p), _f64(a)
    if mask is not None:
        y, p, a = y[mask], p[mask], a[mask]
    lv = Level(a, n_knots, tail_knots, min_count)
    ry = y - lv.fit(y)
    rp = p - lv.fit(p)
    if not ry.std() > _EPS * y.std():
        return Undefined("truth is a function of the average")
    sp = p.std()
    share = rp.std() / sp if sp > 0 else 0.0
    if guard and share <= _EPS:
        return 0.0
    return _corr(ry, rp)


# --------------------------------------------------------------------------------------------- metrics 3, 4

def crps_split(n, mu, y, **oracle_kw) -> Dict[str, float]:
    """Mean NB CRPS of NB(n, mu) against counts y, with its oracle split.

    Returns {crps, oracle_scaled, scale_error}; crps = oracle_scaled + scale_error exactly.
    oracle_scaled is `candi.bench.distributional.oracle_scale`'s crps_oracle_scaled (one per-call
    multiplicative rescale of mu granted for free)."""
    n, mu, y = (_f64(v).ravel() for v in (n, mu, y))
    if y.size == 0:
        r = Undefined("no bins")
        return {"crps": r, "oracle_scaled": r, "scale_error": r, "reason": r.reason}
    crps = float(np.mean(nb_crps(n, p_from_mu(n, mu), y)))
    orc = float(oracle_scale(mu, n, y, **oracle_kw)["crps_oracle_scaled"])
    return {"crps": crps, "oracle_scaled": orc, "scale_error": crps - orc, "reason": ""}


def crps_skill(model: Dict, baseline: Dict) -> Dict[str, float]:
    """Headline 3: CRPS(baseline) - CRPS(model) for each of crps, oracle_scaled, scale_error
    (positive = the model is better). Inputs are `crps_split` dicts."""
    return {k: float(baseline[k] - model[k]) for k in ("crps", "oracle_scaled", "scale_error")}


def gap_closed(model, baseline, floor) -> float:
    """(baseline - model) / (baseline - floor) for a lower-is-better score; arrays are averaged first
    (a ratio of means). 1 = the model reaches the floor; 0 = no better than the baseline.
    Headline 4: CRPS on never-trained pairs, baseline = depth-only, floor = half against half.
    Undefined when no finite value is left or baseline <= floor."""
    m, b, f = (np.atleast_1d(_f64(v)) for v in (model, baseline, floor))
    m, b, f = np.broadcast_arrays(m, b, f)
    ok = np.isfinite(m) & np.isfinite(b) & np.isfinite(f)
    if not ok.any():
        return Undefined("no finite model, baseline and floor")
    mm, bb, ff = m[ok].mean(), b[ok].mean(), f[ok].mean()
    if not bb - ff > 0:
        return Undefined("baseline is not above the floor")
    return float((bb - mm) / (bb - ff))


def uncertainty_gap_closed(pit_arm, pit_marginal) -> float:
    """D13 uncertainty term: 1 - PIT deviation(arm) / PIT deviation(marginal NB), i.e. gap_closed
    with floor 0. Undefined when the marginal NB has zero PIT deviation."""
    return gap_closed(pit_arm, pit_marginal, 0.0)


# --------------------------------------------------------------------------------------------- metric 5

def depth_law_error(pred_total, source_total, log2_depth_ratio) -> float:
    """Headline 5: median over pairs of |log2(pred_total / source_total) - log2_depth_ratio|.

    pred_total: total predicted mean of the target; source_total: total counts of the source;
    log2_depth_ratio: log2(target depth / source depth). Pairs with a non-positive total are dropped.
    Undefined when no pair is left."""
    pt, st, r = (np.atleast_1d(_f64(v)) for v in (pred_total, source_total, log2_depth_ratio))
    pt, st, r = np.broadcast_arrays(pt, st, r)
    ok = (pt > 0) & (st > 0) & np.isfinite(r)
    if not ok.any():
        return Undefined("no pair with positive totals")
    return float(np.median(np.abs(np.log2(pt[ok] / st[ok]) - r[ok])))


# --------------------------------------------------------------------------------------------- metrics 6, 7

def _strata_masks(strata, size):
    if strata is None:
        return {}
    if isinstance(strata, dict):
        return {str(k): np.asarray(v, bool).ravel() for k, v in strata.items()}
    lab = np.asarray(strata).ravel()
    if lab.size != size:
        raise ValueError("strata must have one label per bin")
    return {str(u): lab == u for u in np.unique(lab)}


def _pit_dev(n, mu, y):
    g, fbar = calibration_pit_curve(n, p_from_mu(n, mu), y, PIT_GRID)
    return float(np.mean(np.abs(np.asarray(fbar) - np.asarray(g))))


def pit_deviation(n, mu, y, strata=None) -> Dict:
    """Headline 6: mean |F_bar(u) - u| over the 21-point grid (non-randomized PIT,
    `candi.metrics.calibration_pit_curve`), pooled over all bins and per stratum.

    strata: None, one label per bin, or {name: bool mask}. Returns {pooled, worst_stratum,
    worst_stratum_name, per_stratum, reason}; an empty stratum is skipped."""
    n, mu, y = (_f64(v).ravel() for v in (n, mu, y))
    if y.size == 0:
        r = Undefined("no bins")
        return {"pooled": r, "worst_stratum": r, "worst_stratum_name": "", "per_stratum": {}, "reason": r.reason}
    per = {k: _pit_dev(n[m], mu[m], y[m]) for k, m in _strata_masks(strata, y.size).items() if m.any()}
    worst = max(per, key=per.get) if per else ""
    return {"pooled": _pit_dev(n, mu, y),
            "worst_stratum": per[worst] if per else Undefined("no non-empty stratum"),
            "worst_stratum_name": worst, "per_stratum": per, "reason": ""}


def variance_ratio(n, mu, rep_a, rep_b, strata=None) -> Dict:
    """Headline 7: mean predicted NB variance (mu + mu^2 / n) over mean replicate variance
    ((rep_a - rep_b)^2 / 2, unbiased for one replicate's variance), per stratum and pooled.
    1 = predicted spread matches replicate spread; > 1 over-dispersed. The replicates and the
    prediction must be at the same depth.

    Returns {pooled, per_stratum, worst_stratum (largest |log2 ratio|), worst_stratum_name, reason}."""
    n, mu, a, b = (_f64(v).ravel() for v in (n, mu, rep_a, rep_b))
    pv = mu + mu * mu / n
    rv = 0.5 * (a - b) ** 2

    def _ratio(m):
        d = rv[m].mean() if m.any() else 0.0
        return float(pv[m].mean() / d) if d > 0 else Undefined("replicate variance is zero")

    allm = np.ones(a.size, bool)
    pooled = _ratio(allm)
    per = {k: _ratio(m) for k, m in _strata_masks(strata, a.size).items() if m.any()}
    fin = {k: v for k, v in per.items() if np.isfinite(v) and v > 0}
    worst = max(fin, key=lambda k: abs(np.log2(fin[k]))) if fin else ""
    return {"pooled": pooled, "per_stratum": per,
            "worst_stratum": fin[worst] if worst else Undefined("no stratum with a defined ratio"),
            "worst_stratum_name": worst, "reason": getattr(pooled, "reason", "")}
