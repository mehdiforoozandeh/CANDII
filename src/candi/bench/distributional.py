"""D-block — measures that read the whole predictive distribution, not just its centre.

This is the half of the suite the ENCODE Imputation Challenge could not have had: every submission
produced point estimates, so calibration and ranking quality were outside its scope
(`cruxvault/wiki/imputation-evaluation-measures.md`). CANDI emits a Negative Binomial per assay per
bin, and the Gaussian signal head emits `(µ, σ²)`, so both are scoreable here.

Primitives that already exist and are already verified — `nb_crps`, `calibration_pit_curve`, `ece` —
are imported from `candi.metrics` rather than re-implemented. Two copies of a closed-form CRPS
derivation is exactly how the two drift.

**The C-index is the one metric in the whole suite that does not read every position** (`EVAL_PLAN.md`
D3), because it is pairwise: chr21 is 1,868,399 bins, so exhaustive concordance is 1.7e12 pairs per
track. It samples pairs, and it always reports the standard error of that sampling beside the
estimate. A C-index quoted without its SE is not quotable.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import nbinom, norm

from candi.metrics import P_EPS, calibration_pit_curve, ece, nb_crps

__all__ = [
    "p_from_mu", "nb_crps_mean", "oracle_scale", "marginal_nb", "gauss_crps", "gauss_crps_mean",
    "pit_curve", "ece", "c_index_nb", "c_index_gauss", "coverage_nb", "coverage_gauss",
    "nb_suite", "gauss_suite",
]

pit_curve = calibration_pit_curve


def p_from_mu(n: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """NB success probability from the mean parameterisation: `p = n / (n + µ)`.

    float64 throughout, and that is not an accident of numpy defaults — see the AMP note at the top
    of `candi/metrics.py`. Nothing here is a torch op, so no autocast setting can reach it.
    """
    n = np.asarray(n, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    return np.clip(n / (n + np.maximum(mu, 1e-12)), P_EPS, 1.0 - P_EPS)


def nb_crps_mean(n: np.ndarray, mu: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(nb_crps(n, p_from_mu(n, mu), y)))


# ---------------------------------------------------------------------------
# the capability / calibration split
# ---------------------------------------------------------------------------

def oracle_scale(mu: np.ndarray, n: np.ndarray, target: np.ndarray, *,
                 fit_budget: int = 20_000, seed: int = 0) -> Dict[str, float]:
    """Oracle per-assay multiplicative rescale `c* = argmin_c CRPS(NB(n, µ·2^c), y)`.

    CRPS is location-dominated — a 4x error in µ costs +84%, a 4x error in dispersion +9% — so an arm
    that wins on level and an arm that wins on shape sum into an apparent Pareto front that is really
    one axis. This splits them. `crps_oracle_scaled` is the **capability** term: what the model gets
    right once its per-assay level is granted for free. `scale_error = crps − crps_oracle_scaled` is
    the **fixable** term, removable by one constant per assay.

    `fit_budget` subsamples the **grid search**, never the reported number: both CRPS values are
    evaluated on the full input at the selected oracle. That is why this does not violate the
    no-subsampling rule (D2) — it is a hyperparameter of an argmin, not a sample of a metric. It also
    makes `crps_oracle_scaled` an in-sample oracle, i.e. an upper bound on capability, which is why
    `scale_error` is allowed to go very slightly negative.

    Ported unchanged from `eval.py::_oracle_scale` so the cutover's equivalence report can show this
    key did not move.
    """
    N = len(mu)
    sel = (np.arange(N) if N <= fit_budget
           else np.random.default_rng(seed).choice(N, fit_budget, replace=False))
    mf, nf, tf = mu[sel], n[sel], target[sel]

    def _fit(c: float, k: float = 0.0) -> float:
        m, nv = mf * 2.0 ** c, nf * 2.0 ** k
        return float(np.mean(nb_crps(nv, p_from_mu(nv, m), tf)))

    c = float(min(np.arange(-6.0, 6.001, 0.25), key=_fit))
    c = float(min(np.arange(c - 0.25, c + 0.2501, 0.01), key=_fit))
    k = float(min(np.arange(-4.0, 4.001, 0.25), key=lambda kk: _fit(c, kk)))
    ms, nk = mu * 2.0 ** c, n * 2.0 ** k
    return dict(
        c_star=c, n_star_log2=k,
        crps_oracle_scaled=float(np.mean(nb_crps(n, p_from_mu(n, ms), target))),
        crps_oracle_scaled_and_n=float(np.mean(nb_crps(nk, p_from_mu(nk, ms), target))),
    )


def marginal_nb(target: np.ndarray) -> Dict[str, float]:
    """The CRPS-optimal **constant** NB forecast — the weakest bar a model must clear.

    Not the median-based baseline: on assays whose median count is 0 that degenerates to a point mass
    at 0, i.e. "predict nothing", which any model beats. The legacy median value is computed anyway
    and entered as a candidate, so the degeneracy stays visible rather than being asserted away.

    Because the forecast is constant its CRPS depends on the target only through the histogram, so the
    grid search over unique values is exact and cheap regardless of how many positions there are —
    which is what lets this stay a full-track number.

    Ported unchanged from `eval.py::_marginal_nb`.
    """
    target = np.asarray(target)
    if target.size < 2:
        return dict(marg_crps=float("nan"), marg_mu=float("nan"), marg_n=float("nan"),
                    marg_mu_legacy_median=float("nan"), marg_crps_legacy_median=float("nan"))
    vals, cnts = np.unique(target, return_counts=True)
    w = cnts / cnts.sum()
    mean = float(np.dot(w, vals))
    var = float(np.dot(w, (vals - mean) ** 2))
    mu0 = max(mean, 1e-3)
    n0 = max((mu0 * mu0) / max(var - mu0, 1e-3), 1e-3)          # NB var = mu + mu^2/n

    def _crps_const(mu_c: float, n_c: float) -> float:
        nv, mv = np.full_like(vals, n_c, dtype=float), np.full_like(vals, mu_c, dtype=float)
        return float(np.dot(w, nb_crps(nv, p_from_mu(nv, mv), vals)))

    med = float(np.median(target)) + 1e-6
    med_n = max((med * med) / max(var - med, 1e-6), 1e-6)
    crps_legacy = _crps_const(med, med_n)
    best = (crps_legacy, med, med_n)
    for c in np.arange(-8.0, 3.01, 0.25):
        mu_c = mu0 * 2.0 ** c
        for f in np.arange(-3.0, 3.01, 0.5):
            n_c = max(n0 * 2.0 ** f, 1e-6)
            v = _crps_const(mu_c, n_c)
            if v < best[0]:
                best = (v, mu_c, n_c)
    return dict(marg_crps=best[0], marg_mu=best[1], marg_n=best[2],
                marg_mu_legacy_median=med, marg_crps_legacy_median=crps_legacy)


# ---------------------------------------------------------------------------
# Gaussian CRPS — the pval arm's proper score
# ---------------------------------------------------------------------------

def gauss_crps(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of `N(µ, σ²)` against `y` (Gneiting & Raftery 2007):

        CRPS = σ · [ z(2Φ(z) − 1) + 2φ(z) − 1/√π ],   z = (y − µ)/σ

    As `σ → 0` this converges to `|y − µ|`, which is the degenerate-forecast limit and is asserted in
    the analytic tests. `σ` is floored rather than allowed to be zero: at exactly 0 the expression is
    `0 · ∞`, and the limit is the answer we want, not a nan.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    y = np.asarray(y, dtype=np.float64)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))


def gauss_crps_mean(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(gauss_crps(mu, sigma, y)))


# ---------------------------------------------------------------------------
# C-index — the one sampled metric (D3)
# ---------------------------------------------------------------------------

def _sample_pairs(y: np.ndarray, n_pairs: int, rng: np.random.Generator):
    """Draw `n_pairs` position pairs whose targets differ. Tied targets carry no ordering to recover."""
    N = len(y)
    i = rng.integers(0, N, size=n_pairs)
    j = rng.integers(0, N, size=n_pairs)
    keep = y[i] != y[j]
    return i[keep], j[keep]


def _concordance(prob_i_gt_j: np.ndarray, y_i: np.ndarray, y_j: np.ndarray,
                 n_pairs_drawn: int) -> Dict[str, float]:
    """Mean over pairs of `P(Ŷ_i > Ŷ_j)` oriented so that 1.0 means "ranked the right way".

    The estimate is a plain mean over independently drawn pairs, so its standard error is
    `std / sqrt(n)` with no bootstrap needed. Reporting it is mandatory: the whole reason this metric
    is allowed to sample is that its sampling error is stated.
    """
    if prob_i_gt_j.size == 0:
        return dict(c_index=float("nan"), c_index_se=float("nan"), c_index_n_pairs=0,
                    c_index_n_drawn=int(n_pairs_drawn))
    oriented = np.where(y_i > y_j, prob_i_gt_j, 1.0 - prob_i_gt_j)
    return dict(
        c_index=float(np.mean(oriented)),
        c_index_se=float(np.std(oriented, ddof=1) / np.sqrt(oriented.size))
        if oriented.size > 1 else float("nan"),
        c_index_n_pairs=int(oriented.size),
        c_index_n_drawn=int(n_pairs_drawn),
    )


def c_index_gauss(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, *,
                  n_pairs: int = 200_000, seed: int = 0) -> Dict[str, float]:
    """Distributional concordance for a Gaussian head. **Exact per pair**, sampled over pairs.

    For two independent normals, `P(X_i > X_j) = Φ((µ_i − µ_j) / sqrt(σ_i² + σ_j²))` in closed form,
    so the only Monte-Carlo error is the choice of pairs — never the probability itself.
    """
    rng = np.random.default_rng(seed)
    i, j = _sample_pairs(np.asarray(y), n_pairs, rng)
    mu, sigma = np.asarray(mu, dtype=np.float64), np.asarray(sigma, dtype=np.float64)
    denom = np.sqrt(np.maximum(sigma[i] ** 2 + sigma[j] ** 2, 1e-24))
    prob = norm.cdf((mu[i] - mu[j]) / denom)
    out = _concordance(prob, np.asarray(y)[i], np.asarray(y)[j], n_pairs)
    out["c_index_exact_per_pair"] = True
    return out


def c_index_nb(n: np.ndarray, mu: np.ndarray, y: np.ndarray, *, n_pairs: int = 200_000,
               draws: int = 1, seed: int = 0) -> Dict[str, float]:
    """Distributional concordance for the NB head, by sampling.

    **Why one draw per pair rather than the old 500.** The archive implementation
    (`CANDI/_utils.py::c_index_nbinom`) estimated `P(X_i > X_j)` to high precision with M=500 draws
    for each of 10,000 pairs. But the C-index is a *mean over pairs*: per-pair noise averages out,
    while per-pair bias does not, and a Bernoulli draw is already unbiased for the probability. The
    same compute therefore buys 500x more pairs, and pairs are the scarce thing — 10,000 pairs out of
    1.7e12 is a far larger source of error than the per-pair estimate ever was. `draws` is exposed for
    anyone who wants to check that claim rather than take it.

    Ties in the drawn counts score 0.5, which is what a discrete distribution honestly deserves: two
    equal draws express no preference.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    i, j = _sample_pairs(y, n_pairs, rng)
    if i.size == 0:
        return {**_concordance(np.empty(0), np.empty(0), np.empty(0), n_pairs),
                "c_index_exact_per_pair": False, "c_index_draws": int(draws)}
    n = np.asarray(n, dtype=np.float64)
    p = p_from_mu(n, np.asarray(mu, dtype=np.float64))
    acc = np.zeros(i.size, dtype=np.float64)
    for _ in range(max(1, draws)):
        xi = nbinom.rvs(n[i], p[i], random_state=rng)
        xj = nbinom.rvs(n[j], p[j], random_state=rng)
        acc += np.where(xi > xj, 1.0, np.where(xi == xj, 0.5, 0.0))
    prob = acc / max(1, draws)
    out = _concordance(prob, y[i], y[j], n_pairs)
    out["c_index_exact_per_pair"] = False
    out["c_index_draws"] = int(draws)
    return out


# ---------------------------------------------------------------------------
# interval coverage
# ---------------------------------------------------------------------------

def coverage_nb(n: np.ndarray, mu: np.ndarray, y: np.ndarray, level: float = 0.95) -> float:
    """Empirical coverage of the central `level` predictive interval.

    This is **not** the calibration metric — `ece` is, and `EVAL.md` records why interval coverage was
    rejected for that job: a perfectly calibrated NB at µ=1 scores 0.25 under interval-coverage ECE,
    because a discrete quantile interval cannot hit a nominal level. Coverage is still worth reporting
    as a descriptive number; it just must not be read as calibration error.
    """
    n = np.asarray(n, dtype=np.float64)
    p = p_from_mu(n, np.asarray(mu, dtype=np.float64))
    a = (1.0 - level) / 2.0
    lo = nbinom.ppf(a, n, p)
    hi = nbinom.ppf(1.0 - a, n, p)
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


def coverage_gauss(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, level: float = 0.95) -> float:
    a = (1.0 - level) / 2.0
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    lo = norm.ppf(a, loc=mu, scale=sigma)
    hi = norm.ppf(1.0 - a, loc=mu, scale=sigma)
    y = np.asarray(y)
    return float(np.mean((y >= lo) & (y <= hi)))


# ---------------------------------------------------------------------------
# the whole block
# ---------------------------------------------------------------------------

def nb_suite(n: np.ndarray, mu: np.ndarray, y: np.ndarray, *, seed: int = 0,
             n_pairs: int = 200_000, with_marginal: bool = True) -> Dict[str, float]:
    """Every D-block key for one NB-scored track. `crps` is never emitted without its split."""
    p = p_from_mu(n, mu)
    crps = float(np.mean(nb_crps(n, p, y)))
    orc = oracle_scale(np.asarray(mu, float), np.asarray(n, float), np.asarray(y), seed=seed)
    grid, fbar = calibration_pit_curve(n, p, y)
    out: Dict[str, float] = dict(
        crps=crps, **orc, scale_error=crps - orc["crps_oracle_scaled"],
        ece=ece(n, p, y), calib_grid=grid, calib_fbar=fbar,
        coverage_95=coverage_nb(n, mu, y),
        **c_index_nb(n, mu, y, n_pairs=n_pairs, seed=seed),
        n_points=int(len(y)),
    )
    if with_marginal:
        mar = marginal_nb(np.asarray(y))
        out.update(mar)
        out["beats_marginal"] = bool(np.isfinite(mar["marg_crps"]) and crps < mar["marg_crps"])
        out["beats_marginal_oracle_scaled"] = bool(
            np.isfinite(mar["marg_crps"]) and orc["crps_oracle_scaled"] < mar["marg_crps"])
    return out


def gauss_suite(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, *, seed: int = 0,
                n_pairs: int = 200_000) -> Dict[str, float]:
    """Every D-block key for one Gaussian-scored track.

    No PIT/ECE here: `calibration_pit_curve` is the *discrete* non-randomised PIT, which is the right
    instrument for counts and the wrong one for a continuous forecast. The continuous PIT is simply
    `F(y)`, and it is emitted as `pit_ks` — the Kolmogorov–Smirnov distance of `Φ((y−µ)/σ)` from
    uniform — rather than being forced into the discrete machinery.
    """
    from scipy.stats import kstest
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-12)
    y = np.asarray(y, dtype=np.float64)
    u = norm.cdf((y - mu) / sigma)
    return dict(
        crps=gauss_crps_mean(mu, sigma, y),
        pit_ks=float(kstest(u, "uniform").statistic),
        coverage_95=coverage_gauss(mu, sigma, y),
        **c_index_gauss(mu, sigma, y, n_pairs=n_pairs, seed=seed),
        n_points=int(len(y)),
    )
