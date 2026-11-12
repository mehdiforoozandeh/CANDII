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

from candi.bench.distributional import marginal_nb, nb_crps_mean, oracle_scale, p_from_mu  # noqa: E402
from candi.metrics import PIT_GRID, calibration_pit_curve, nb_crps  # noqa: E402
from eic_decomp import errdecomp as _ed  # noqa: E402

__all__ = [
    "SPACES", "Undefined", "to_space",
    "shared_specific", "noise_floor", "noise_floor_log1p_bias", "specific_variance_recovered",
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

MAX_RESCALE = 100.0


def to_space(x, space: str, depth: Optional[float] = None, *, rescale: Optional[float] = None) -> np.ndarray:
    """Put x into a metric space. The only place a space is applied.

    counts_log1p:   log(1 + rescale * x). `rescale` is a multiplicative FACTOR (target depth / depth of
                    x), never an absolute depth: it must lie in (0, 100]. None or 1.0: x is already at
                    the target's depth (e.g. an NB mean decoded with the target's own C'). A
                    pseudoreplicate half of an even split takes rescale=2.0, so the halves enter the
                    noise floor at the target's depth, not at their own half depth.
    signal_arcsinh: asinh(x), x an assay pipeline signal track. Signal is already depth-normalized, so
                    the factor must be None or 1.0.
    `depth` is the positional alias of `rescale`, kept for the pinned signature `to_space(x, space,
    depth)`; give one or the other. Raises on non-finite or negative x.
    """
    if depth is not None and rescale is not None:
        raise ValueError("give rescale or its alias depth, not both")
    f = rescale if rescale is not None else depth
    x = _f64(x)
    if not np.isfinite(x).all():
        raise ValueError("x holds non-finite values")
    if (x < 0).any():
        raise ValueError("x holds negative values")
    if f is not None and not (0.0 < float(f) <= MAX_RESCALE):
        raise ValueError(f"rescale {f!r} is outside (0, {MAX_RESCALE:g}]: it is a depth ratio, not a depth")
    if space == "counts_log1p":
        return np.log1p((1.0 if f is None else float(f)) * x)
    if space == "signal_arcsinh":
        if f not in (None, 1, 1.0):
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
    put on the same standardization. `level` (= offset + scale + nonlinear, errdecomp.calibrate) and its
    three terms are computed in both modes, so they are never NaN for a defined target. Gram matrices by Haar scale of the truth (V), prediction (P), the
    symmetrized cross term (C) and the error E = V - 2C + P are split into shared and specific parts
    (Max s06 aggregate2.py). Zero padding to a multiple of 2^J preserves every inner product.

    Returns a dict of [T] arrays: V_sh, V_sp, E_sh, E_sp, P_sh, P_sp, C_sh, C_sp, N, level, offset,
    scale, nonlinear; `by_level`
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
    lev_terms = {nm: np.full(T, np.nan) for nm in ("offset", "scale", "nonlinear")}
    ok = np.zeros(T, bool)
    for t in range(T):
        x, y = X[t], Y[t]
        mx, sx = x.mean(), x.std()
        if not sx > 0:
            reason[t] = "truth has zero variance"
            continue
        terms, yc = _ed.calibrate(x, y)     # level terms in both modes
        for nm in ("offset", "scale", "nonlinear"):
            lev_terms[nm][t] = terms[nm]
        level[t] = terms["offset"] + terms["scale"] + terms["nonlinear"]
        if not calibrated:
            yc = y
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
    out.update(lev_terms)
    out["reason"] = reason
    return out


def noise_floor(pr1, pr2, target, depth_ratio: float = 2.0, *, J: int = J, return_levels: bool = False,
                shares=None):
    """Sampling-noise floor N of a target track, in units of Var(target).

    pr1, pr2: the target's two pseudoreplicate halves, in the same space as target. In `counts_log1p`
    the halves are rescaled to the target's depth BEFORE log1p (`to_space(half, "counts_log1p",
    rescale=1 / share)`, 2.0 for an even split), not taken at their own half depth. At each Haar level
    k, R_k = corr of the halves' level-k coefficients (clipped to [0, 1]); the noise share of a track at
    `depth_ratio` times the half depth (fragments) is 1 - k R_k / (1 + (k - 1) R_k) (Spearman-Brown).
    shares = (f1, f2), the halves' real shares of the target's depth, replaces depth_ratio with the
    unequal-length form: half i carries 1 / f_i times the target's noise variance, so with s_k the
    target's signal-to-noise ratio, R_k^2 (s_k + 1/f1)(s_k + 1/f2) = s_k^2, and the noise share is
    1 / (1 + s_k). Equal shares f give exactly the Spearman-Brown form with k = 1 / f.
    N = sum_k share_k * (energy share of the target at level k). Undefined if a vector is constant.
    In counts_log1p at a few reads per bin the step understates the sampling share (about -4% at 3
    reads per bin in the total-variance audit); `noise_floor_log1p_bias` reports its simulated size.

    return_levels=True returns a dict {N, R, share, energy_share, n_coef} instead of the scalar;
    n_coef [J+1] is the number of coefficients per level that touch real (unpadded) bins, so a caller
    can see thin coarse levels whose R is poorly estimated.
    """
    H = np.stack([_f64(pr1), _f64(pr2), _f64(target)])
    L = H.shape[1]
    H = H - H.mean(1, keepdims=True)
    if not (H.std(1) > 0).all():
        u = Undefined("a half or the target has zero variance")
        return {"N": u, "reason": u.reason} if return_levels else u
    G = _ed.gram_by_scale(_pad(H, J), J)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = G[:, 0, 1] / np.sqrt(G[:, 0, 0] * G[:, 1, 1])
    R = np.clip(np.nan_to_num(R, nan=0.0), 0.0, 1.0)
    if shares is not None and float(shares[0]) != float(shares[1]):
        a, b = 1.0 / float(shares[0]), 1.0 / float(shares[1])
        R2 = R * R
        with np.errstate(invalid="ignore", divide="ignore"):
            snr = (R2 * (a + b) + np.sqrt(R2 * R2 * (a + b) ** 2 + 4.0 * (1.0 - R2) * R2 * a * b)) / (
                2.0 * (1.0 - R2))
        share = np.where(R2 >= 1.0, 0.0, 1.0 / (1.0 + snr))
    else:
        k = float(depth_ratio) if shares is None else 1.0 / float(shares[0])
        share = 1.0 - k * R / (1.0 + (k - 1.0) * R)
    xs = G[:, 2, 2] / G[:, 2, 2].sum()
    N = float(np.dot(share, xs))
    if not return_levels:
        return N
    n_coef = np.array([-(-L // (1 << j)) for j in range(1, J + 1)] + [-(-L // (1 << J))])
    return {"N": N, "R": R, "share": share, "energy_share": xs, "n_coef": n_coef, "reason": ""}


def _rate_npmle(counts, n_grid: int = 120, iters: int = 300):
    """Nonparametric maximum-likelihood rate distribution of a Poisson mixture (EM on a fixed grid of
    rates, 0 plus a geometric grid up to 1.5 x the largest count): (grid, weights)."""
    from scipy.special import gammaln, xlogy
    vals, cnt = np.unique(np.asarray(counts, np.int64), return_counts=True)
    grid = np.concatenate([[0.0], np.geomspace(0.01, max(float(vals.max()), 1.0) * 1.5, n_grid - 1)])
    lik = np.exp(xlogy(vals[:, None], grid[None, :]) - grid[None, :] - gammaln(vals[:, None] + 1.0))
    w = np.full(grid.size, 1.0 / grid.size)
    for _ in range(iters):
        p = lik * w
        p /= np.maximum(p.sum(1, keepdims=True), 1e-300)
        w = (cnt[:, None] * p).sum(0) / cnt.sum()
    return grid, w


def noise_floor_log1p_bias(counts, shares=(0.5, 0.5), *, n: int = 2 ** 14, J: int = J, seed: int = 0) -> float:
    """Simulated bias of `noise_floor` in counts_log1p at a track's count distribution (a diagnostic;
    the estimator is not corrected by it).

    The rate distribution of the track is fitted to its counts (`_rate_npmle`, a Poisson mixture, so
    it is set by the track's mean count and the shape of its counts), n rates are drawn from it, a
    Poisson track is drawn and split binomially into halves with the given depth shares, and N is
    computed as score.py does (halves rescaled by 1 / share before log1p). The planted sampling share is
    0.5 mean((log1p y - log1p y')^2) / Var(log1p y), y' a second Poisson draw of the same rates.
    Returns N - planted (negative: N understates the sampling share). On the 24 tracks of the
    total-variance audit (3 reads per bin) its mean was -0.011 for even halves and -0.014 for 0.3 / 0.7
    halves; the audit's true rates give -0.009 and -0.014."""
    y = _f64(counts).reshape(-1)
    y = y[np.isfinite(y)]
    if not (y.size and y.mean() > 0):
        return Undefined("mean count is not positive")
    grid, w = _rate_npmle(np.rint(y))
    rng = np.random.default_rng(seed)
    lam = rng.choice(grid, n, p=w / w.sum())
    y1 = rng.poisson(lam)
    f1, f2 = float(shares[0]), float(shares[1])
    f1, f2 = f1 / (f1 + f2), f2 / (f1 + f2)
    h1 = rng.binomial(y1, f1)
    t1, t2 = np.log1p(y1.astype(np.float64)), np.log1p(rng.poisson(lam).astype(np.float64))
    if not t1.var() > 0:
        return Undefined("the simulated track is constant")
    est = noise_floor(to_space(h1, "counts_log1p", rescale=1.0 / f1),
                      to_space(y1 - h1, "counts_log1p", rescale=1.0 / f2), t1, J=J, shares=(f1, f2))
    if isinstance(est, Undefined):
        return est
    return float(est - 0.5 * np.mean((t1 - t2) ** 2) / t1.var())


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

def _clean(arrays, counts, mask):
    """Validate per-bin inputs. arrays: dict name -> array (all broadcast to one shape and raveled);
    counts: names that are observed counts (raise on a negative finite value); mask: optional bool
    selection. Bins with any non-finite value are dropped. Returns (dict of 1-D arrays, n_dropped)."""
    names = list(arrays)
    vals = np.broadcast_arrays(*(_f64(arrays[k]) for k in names))
    vals = [v.ravel() for v in vals]
    for k, v in zip(names, vals):
        if k in counts and (v[np.isfinite(v)] < 0).any():
            raise ValueError(f"{k} holds negative counts")
    keep = np.ones(vals[0].size, bool) if mask is None else np.asarray(mask, bool).ravel().copy()
    if keep.size != vals[0].size:
        raise ValueError("mask must have one entry per bin")
    sel = keep.sum()
    for v in vals:
        keep &= np.isfinite(v)
    return {k: v[keep] for k, v in zip(names, vals)}, int(sel - keep.sum()), keep


def crps_split(n, mu, y, *, mask=None, crps_approx: Optional[int] = None, crps_seed: int = 0,
               **oracle_kw) -> Dict[str, float]:
    """Mean NB CRPS of NB(n, mu) against counts y, with its oracle split.

    Returns {crps, oracle_scaled, scale_error, n_dropped, reason}; crps = oracle_scaled + scale_error
    exactly. oracle_scaled is `candi.bench.distributional.oracle_scale`'s crps_oracle_scaled (one
    multiplicative rescale of mu granted for free). Both CRPS numbers use the same estimator:
    `crps_approx` / `crps_seed` go to both (closed form by default). Bins outside `mask` or with a
    non-finite n, mu or y are dropped (counted in n_dropped); negative y raises."""
    d, dropped, _ = _clean({"n": n, "mu": mu, "y": y}, ("y",), mask)
    if d["y"].size == 0:
        r = Undefined("no bins with finite n, mu and y")
        return {"crps": r, "oracle_scaled": r, "scale_error": r, "n_dropped": dropped, "reason": r.reason}
    kw = dict(crps_approx=crps_approx, crps_seed=crps_seed)
    crps = nb_crps_mean(d["n"], d["mu"], d["y"], **kw)
    orc = float(oracle_scale(d["mu"], d["n"], d["y"], **kw, **oracle_kw)["crps_oracle_scaled"])
    return {"crps": crps, "oracle_scaled": orc, "scale_error": crps - orc, "n_dropped": dropped, "reason": ""}


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


def pit_deviation(n, mu, y, strata=None, *, mask=None) -> Dict:
    """Headline 6: mean |F_bar(u) - u| over the 21-point grid (non-randomized PIT,
    `candi.metrics.calibration_pit_curve`), pooled over all bins and per stratum.

    strata: None, one label per bin, or {name: bool mask}. Bins outside `mask` or with a non-finite
    n, mu or y are dropped (n_dropped); negative y raises. Returns {pooled, worst_stratum,
    worst_stratum_name, per_stratum, n_dropped, reason}; an empty stratum is skipped."""
    d, dropped, keep = _clean({"n": n, "mu": mu, "y": y}, ("y",), mask)
    n, mu, y = d["n"], d["mu"], d["y"]
    if y.size == 0:
        r = Undefined("no bins with finite n, mu and y")
        return {"pooled": r, "worst_stratum": r, "worst_stratum_name": "", "per_stratum": {},
                "n_dropped": dropped, "reason": r.reason}
    per = {k: _pit_dev(n[m[keep]], mu[m[keep]], y[m[keep]])
           for k, m in _strata_masks(strata, keep.size).items() if m[keep].any()}
    worst = max(per, key=per.get) if per else ""
    return {"pooled": _pit_dev(n, mu, y),
            "worst_stratum": per[worst] if per else Undefined("no non-empty stratum"),
            "worst_stratum_name": worst, "per_stratum": per, "n_dropped": dropped, "reason": ""}


def _thin(x, p, rng):
    if not np.array_equal(x, np.round(x)):
        raise ValueError("thinning needs integer counts")
    return rng.binomial(x.astype(np.int64), p).astype(np.float64)


def variance_ratio(n, mu, rep_a, rep_b, strata=None, *, depth_ratio_b: float = 1.0, mask=None,
                   total_tol: float = 0.05, seed: int = 0) -> Dict:
    """Headline 7: mean predicted NB variance (mu + mu^2 / n) over mean replicate variance
    ((a - b)^2 / 2, unbiased for one replicate's variance when a and b are iid at one depth), per
    stratum and pooled. 1 = predicted spread matches replicate spread; > 1 over-dispersed.

    The prediction NB(n, mu) is for rep_a's depth. depth_ratio_b = depth(a) / depth(b). The deeper
    replicate is binomially thinned (seeded) to the shallower depth: if b is deeper, b is thinned by
    depth_ratio_b; if b is shallower, a is thinned by 1 / depth_ratio_b and mu with it (a thinned
    NB(n, mu) is NB(n, p mu)). After matching, sum(a) / sum(b) must lie within 1 +- total_tol, else
    the result is Undefined (the replicates are not at the depth that was declared). Bins outside
    `mask` or with a non-finite input are dropped (n_dropped); negative counts raise.

    Returns {pooled, per_stratum, worst_stratum (largest |log2 ratio|), worst_stratum_name,
    n_dropped, reason}."""
    d, dropped, keep = _clean({"n": n, "mu": mu, "a": rep_a, "b": rep_b}, ("a", "b"), mask)
    n, mu, a, b = d["n"], d["mu"], d["a"], d["b"]

    def _undef(reason):
        u = Undefined(reason)
        return {"pooled": u, "per_stratum": {}, "worst_stratum": u, "worst_stratum_name": "",
                "n_dropped": dropped, "reason": reason}

    if a.size == 0:
        return _undef("no bins with finite n, mu and replicates")
    r = float(depth_ratio_b)
    if not r > 0:
        raise ValueError("depth_ratio_b must be positive")
    rng = np.random.default_rng(seed)
    if r < 1.0:
        b = _thin(b, r, rng)
    elif r > 1.0:
        a = _thin(a, 1.0 / r, rng)
        mu = mu / r
    sa, sb = a.sum(), b.sum()
    if not (sa > 0 and sb > 0):
        return _undef("a replicate has zero total counts")
    if abs(sa / sb - 1.0) > total_tol:
        return _undef(f"replicate totals differ after depth matching: sum(a)/sum(b) = {sa / sb:.3f}")
    pv = mu + mu * mu / n
    rv = 0.5 * (a - b) ** 2

    def _ratio(m):
        dd = rv[m].mean() if m.any() else 0.0
        return float(pv[m].mean() / dd) if dd > 0 else Undefined("replicate variance is zero")

    pooled = _ratio(np.ones(a.size, bool))
    per = {k: _ratio(m[keep]) for k, m in _strata_masks(strata, keep.size).items() if m[keep].any()}
    fin = {k: v for k, v in per.items() if np.isfinite(v) and v > 0}
    worst = max(fin, key=lambda k: abs(np.log2(fin[k]))) if fin else ""
    return {"pooled": pooled, "per_stratum": per,
            "worst_stratum": fin[worst] if worst else Undefined("no stratum with a defined ratio"),
            "worst_stratum_name": worst, "n_dropped": dropped, "reason": getattr(pooled, "reason", "")}
