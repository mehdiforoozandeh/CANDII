"""Numeric metric primitives for candi (NB CRPS, quantiles, PIT calibration, correlations).

VENDORED VERBATIM of sandbox/diagnostics/dual_conditioning/metrics.py
  L27 (P_EPS), L29-131 (_nb_cdf, nb_crps, nb_quantile, r2, pearson, spearman, PIT_GRID,
  calibration_pit_curve, ece, _cos_dist), L341-353 (_steering_index).
The synthetic-transform-bound eval harness (make_eval_units onward) is NOT shipped; the real-data
measurement stack lives in `candi.bench` (`candi.eval`, which held it before, is deleted — D15).

The closed-form NB CRPS (`nb_crps`) is the risky numeric: derived as E|X-y| - 1/2 E|X-X'| with a
Pfaff-transformed hypergeometric Gini term, verified bit-close to an exact discrete sum and to
Monte-Carlo across the full parameter range (tests/test_metrics_primitives.py).

One departure from the vendored source (t56): `nb_crps` gained a large-dispersion branch — for
n > N_GINI_HYP2F1_MAX it scores the sd-standardized Poisson limit, because scipy's hyp2f1 is NaN
there. The n <= 1e4 path is bit-identical to the vendored original.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson, rankdata
from scipy.special import hyp2f1, ive

P_EPS = 1e-9

# scipy's hyp2f1(0.5, 1-n, 2, w) hard-fails to NaN once n exceeds ~1.04e4 — a function of the 1-n
# parameter alone, independent of mu, p and y (boundary mapped in t56). Below this the Pfaff-2F1
# Gini term is verified to <=2e-9 relative against the exact discrete sum; above it nb_crps scores
# the NB in its Poisson limit, standardized to the NB's own sd (see nb_crps docstring).
N_GINI_HYP2F1_MAX = 1e4

# NO fp32 FENCE ANYWHERE IN THIS FILE, and that is the answer rather than an omission. The AMP audit
# lists `metrics.py` among the things evaluation must never autocast; it already cannot be. Every
# primitive here takes and returns NUMPY arrays, and `torch.autocast` is a torch dispatcher feature —
# it rewrites the dtype of torch ops and reaches nothing else, so no setting of `--precision` can
# change a number computed below. The float64 convention these rely on
# (`bench.distributional.p_from_mu`, `nb_crps`) is a numpy fact for the same reason. A fence here
# would imply a hazard that does not exist and would send the next reader looking for one. The
# fence that DOES matter is at `bench.cli.main`, which is where the torch forwards live.


# ---------------------------------------------------------------------------
# Closed-form negative-binomial CRPS (mean-parameterization p = n/(n+mu))
# ---------------------------------------------------------------------------

def _nb_cdf(k, n, p):
    """NB CDF on 1-D arrays; F(k<0) := 0."""
    out = np.zeros(k.shape, dtype=float)
    pos = k >= 0
    if np.any(pos):
        out[pos] = nbinom.cdf(np.floor(k[pos]), n[pos], p[pos])
    return out


def nb_crps(n, p, y) -> np.ndarray:
    """CRPS of NB(size=n, prob=p) against integer observation y (element-wise, >= 0).

    CRPS = E|X-y| - 1/2 E|X-X'|, with
      E|X-y|   = (mu - y) + 2[y F(y-1; n,p) - mu F(y-2; n+1,p)],   mu = n(1-p)/p
      E|X-X'|  = (2 mu / p) 2F1(1/2, n+1; 2; z),  z = -4(1-p)/p^2   [Gini mean difference]
    The 2F1 is evaluated via the Pfaff transform 2F1(a,b;c;z)=(1-z)^-a 2F1(a,c-b;c; z/(z-1)) so the
    argument stays in (0,1) and the term is stable at the large means the power family reaches.

    For n > N_GINI_HYP2F1_MAX (scipy's hyp2f1 is NaN there — see the constant above) the NB is
    scored in its Poisson limit, standardized to the NB's own sd:
      CRPS_NB(n,p,y) ~ r * CRPS_Pois(mu, mu + (y-mu)/r),  r = sqrt(1 + mu/n) = sd_NB/sd_Pois,
    with E|X-y| = (mu-y') + 2[y' F(floor(y'); mu) - mu F(floor(y')-1; mu)] valid at real y', and
    E|X-X'| = 2 mu e^-2mu [I0(2mu) + I1(2mu)] via the exponentially scaled Bessel `ive`. The
    rescaling restores the variance the plain Poisson limit drops: worst relative error vs the
    exact discrete sum is ~7e-5 over n in [1e4, 1e7] x mu in [0.5, 5e3] (plain Poisson limit: 18%
    at the switch), and the jump across the switch itself is <= 7e-5 (t56).
    """
    n_a = np.asarray(n, dtype=float)
    p_a = np.clip(np.asarray(p, dtype=float), P_EPS, 1.0 - P_EPS)
    y_a = np.maximum(np.asarray(y, dtype=float), 0.0)
    shape = np.broadcast_shapes(n_a.shape, p_a.shape, y_a.shape)
    n = np.broadcast_to(n_a, shape).ravel()
    p = np.broadcast_to(p_a, shape).ravel()
    y = np.broadcast_to(y_a, shape).ravel()
    mu = n * (1.0 - p) / p
    out = np.empty(n.shape, dtype=float)
    big = n > N_GINI_HYP2F1_MAX
    if not np.all(big):
        s = ~big
        ns, ps, ys, ms = n[s], p[s], y[s], mu[s]
        Exy = (ms - ys) + 2.0 * (ys * _nb_cdf(ys - 1.0, ns, ps) - ms * _nb_cdf(ys - 2.0, ns + 1.0, ps))
        z = -4.0 * (1.0 - ps) / (ps * ps)
        w = z / (z - 1.0)
        gmd = (2.0 * ms / ps) * np.power(1.0 - z, -0.5) * hyp2f1(0.5, 1.0 - ns, 2.0, w)
        out[s] = Exy - 0.5 * gmd
    if np.any(big):
        lam = mu[big]
        r = np.sqrt(1.0 + lam / n[big])
        ys = lam + (y[big] - lam) / r
        yf = np.floor(ys)
        Exy = (lam - ys) + 2.0 * (ys * poisson.cdf(yf, lam) - lam * poisson.cdf(yf - 1.0, lam))
        gmd = 2.0 * lam * (ive(0, 2.0 * lam) + ive(1, 2.0 * lam))
        out[big] = r * (Exy - 0.5 * gmd)
    return np.maximum(out, 0.0).reshape(shape)


def nb_quantile(q: float, n, p) -> np.ndarray:
    """Upper quantile of NB(size=n, prob=p) — the predicted tail statistic."""
    p = np.clip(np.asarray(p, dtype=float), P_EPS, 1.0 - P_EPS)
    return nbinom.ppf(q, np.asarray(n, dtype=float), p)


# ---------------------------------------------------------------------------
# Sampled negative-binomial CRPS — the same score, estimated instead of derived
# ---------------------------------------------------------------------------
# `nb_crps` above is the definition and stays the default everywhere. This is an OPT-IN estimator of
# the same quantity, for the one situation the closed form cannot serve: a genome-wide panel, where
# the hypergeometric term costs ~2.6 h per track and 121 M bins x 45 tracks is ~117 CPU-hours per
# method. It samples the forecast instead of integrating it, so its cost is linear in k and it has
# no hypergeometric to lose precision in.
#
# WHY THE "FAIR" FORM AND NOT THE OBVIOUS ONE. With k draws the plug-in estimator divides the
# pairwise sum by k^2, which counts the k zero terms x_i - x_i and therefore UNDERSTATES E|X-X'| by
# a factor (k-1)/k. That is an O(1/k) BIAS, not noise: it does not average away over bins, so a
# macro mean over 124 M bins would inherit it whole (+0.5 E|X-X'|/k, ~ +0.02 at k = 50 on a track
# whose spread is 2). Dividing by k(k-1) — the number of DISTINCT ordered pairs — removes it, and
# both terms are then unbiased, so the estimator is unbiased for every k >= 2 (Ferro 2008's fair
# CRPS). Everything below follows from wanting that property and keeping it:
#   * NO clamp at zero. `nb_crps` ends in `np.maximum(..., 0.0)` because a closed form can only go
#     negative by rounding; an unbiased estimator goes slightly negative on purpose, on the bins
#     where the draws happened to spread wider than the truth sits, and clipping those away is the
#     one edit that would put the bias back.
#   * FLOAT64 sums, like the rest of this file.

#: Bins x samples held in memory at once. A CONSTANT and not a caller's knob: the chunking decides
#: how the RNG stream is cut up, so pinning it is what makes the result a pure function of
#: `(n, p, y, k, seed)` rather than of how much memory the machine that ran it happened to have.
CRPS_SAMPLE_CHUNK = 4_000_000

#: Ceiling on a drawn Gamma rate before it reaches `rng.poisson`, which raises above ~9.2e18. No
#: epigenomic count comes within eleven orders of magnitude of this; it exists so a pathological
#: (n, mu) kills one bin's tail rather than a 100-hour job.
_LAM_MAX = 1e15


def _fair_crps(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unbiased CRPS from `x` (m, k) draws against `y` (m,).

        CRPS = mean_i |x_i - y| - (1 / 2k(k-1)) sum_{i != j} |x_i - x_j|

    The second sum is evaluated from the SORTED draws in O(k log k) instead of O(k^2):
    `sum_{i<j} (x_(j) - x_(i)) = sum_i (2i - k + 1) x_(i)`, because the i-th order statistic is the
    larger element of i pairs and the smaller element of k-1-i. The ordered-pair sum is twice that,
    so the term subtracted is exactly `sum_i (2i - k + 1) x_(i) / k(k-1)`.
    """
    k = x.shape[1]
    absdev = np.abs(x - y[:, None]).mean(axis=1)
    xs = np.sort(x, axis=1)
    w = 2.0 * np.arange(k, dtype=np.float64) - (k - 1)
    return absdev - (xs * w).sum(axis=1) / (k * (k - 1))


def nb_crps_sampled(n, p, y, *, k: int = 100, seed: int = 0) -> np.ndarray:
    """Unbiased sample estimate of `nb_crps(n, p, y)` from `k` draws per bin.

    Same signature, same parameterization (`mu = n(1-p)/p`, i.e. `p = n/(n+mu)`), same shape out.
    Draws are gamma-Poisson — `lam ~ Gamma(shape=n, scale=(1-p)/p)`, `x ~ Poisson(lam)` — which is
    the NB by construction and puts no integrality condition on `n`.

    **It is finite where the closed form is not.** `nb_crps`'s Gini term is a 2F1 whose `b = 1 - n`,
    and that returns NaN once `n` is large (the `n1e4` floor in `t49` exists only to stay under it);
    the Poisson limit `n -> inf` is exactly the regime a count baseline with no over-dispersion to
    report lands in. Sampling has no such ceiling: at `n = 1e6` the Gamma is a near-point mass at
    `mu` and the draws are Poisson(mu), which is the right answer rather than a NaN.

    Deterministic: `(n, p, y, k, seed)` fixes every draw. `k >= 2` — one draw carries no information
    about `E|X - X'|`, and the fair denominator `k(k-1)` says so by being zero.
    """
    k = int(k)
    if k < 2:
        raise ValueError(f"nb_crps_sampled needs k >= 2 draws per bin to estimate E|X-X'|, got {k}")
    n_a = np.asarray(n, dtype=float)
    p_a = np.clip(np.asarray(p, dtype=float), P_EPS, 1.0 - P_EPS)
    y_a = np.maximum(np.asarray(y, dtype=float), 0.0)
    shape = np.broadcast_shapes(n_a.shape, p_a.shape, y_a.shape)
    nf = np.broadcast_to(n_a, shape).ravel()
    pf = np.broadcast_to(p_a, shape).ravel()
    yf = np.broadcast_to(y_a, shape).ravel()

    out = np.empty(nf.size, dtype=np.float64)
    rng = np.random.default_rng(seed)
    step = max(1, CRPS_SAMPLE_CHUNK // k)
    for lo in range(0, nf.size, step):
        hi = min(lo + step, nf.size)
        m = hi - lo
        g_shape = np.broadcast_to(nf[lo:hi, None], (m, k))
        g_scale = np.broadcast_to(((1.0 - pf[lo:hi]) / pf[lo:hi])[:, None], (m, k))
        lam = np.minimum(rng.gamma(g_shape, g_scale), _LAM_MAX)
        out[lo:hi] = _fair_crps(rng.poisson(lam).astype(np.float64), yf[lo:hi])
    return out.reshape(shape)


# ---------------------------------------------------------------------------
# Scalar metric primitives
# ---------------------------------------------------------------------------

def r2(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    ss_res = float(((pred - target) ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    return float("nan") if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


PIT_GRID = np.linspace(0.0, 1.0, 21)


def calibration_pit_curve(n, p, y, grid=PIT_GRID):
    """Non-randomized PIT reliability curve (Czado-Gneiting-Held 2009): F_bar(u) vs u.

    Deterministic and PROPER for discrete forecasts. (Interval-coverage ECE spuriously over-covers at
    low counts -- e.g. a perfectly calibrated NB at mu=1 scores 0.25 -- because discrete quantile
    intervals cannot hit the nominal level; most epigenomic positions are low-count, so that artifact
    would dominate.) The non-randomized PIT: for each obs y, F^i(u) linearly interpolates from F(y-1)
    to F(y); F_bar(u)=mean_i F^i(u) equals u iff calibrated. Returns (grid, F_bar)."""
    p = np.clip(np.asarray(p, dtype=float), P_EPS, 1.0 - P_EPS)
    n = np.asarray(n, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), 0.0)
    Fy = nbinom.cdf(y, n, p)
    Fym1 = np.where(y > 0, nbinom.cdf(y - 1.0, n, p), 0.0)
    denom = np.maximum(Fy - Fym1, 1e-12)
    fbar = [float(np.mean(np.clip((u - Fym1) / denom, 0.0, 1.0))) for u in grid]
    return list(np.asarray(grid, dtype=float)), fbar


def ece(n: np.ndarray, p: np.ndarray, y: np.ndarray, grid=PIT_GRID) -> float:
    """Calibration error = mean |F_bar(u) - u| over the PIT grid (proper discrete calibration)."""
    g, fbar = calibration_pit_curve(n, p, y, grid)
    return float(np.mean([abs(f - u) for f, u in zip(fbar, g)]))


def _cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return 1.0 - (an * bn).sum(-1)


def _steering_index(crps_mat: np.ndarray) -> float:
    """Relative CRPS reduction at matched h_y: mean_i (rowmean_i - C[i,i]) / rowmean_i.

    0 for an h_y-ignoring model (prediction independent of the metadata column j), -> 1 for perfect
    steering (matched prediction dominates). crps_mat[i,j] = CRPS(pred_j, target_i).
    """
    P = crps_mat.shape[0]
    diag = np.diag(crps_mat)
    rowmean = crps_mat.mean(axis=1)
    good = rowmean > 1e-9
    if not np.any(good):
        return float("nan")
    return float(np.mean((rowmean[good] - diag[good]) / rowmean[good]))
