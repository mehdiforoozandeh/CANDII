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

**The last block in this file is not a benchmark measure — it is the LOSS.** `nb_nll`,
`gaussian_nll` and `bernoulli_nll` are the three terms the training objective is built from, so
scoring them here is what makes `val_loss` and `test_loss` knowable at all: the training loop logs
`train/nll` every step and, until these existed, nothing else in the repo computed any NLL. They are
re-derived in pure numpy rather than imported, because `candi.bench` may not import `candi.train`
(`tests/test_bench_harness.py` pins it in a subprocess). The equivalence that would otherwise be an
assertion is a test instead — `tests/test_bench_analytic.py` imports both sides and compares them on
random tensors — so a drift in either copy fails a test rather than quietly producing a val loss
that is not the number the training loop would print.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.special import gammaln
from scipy.stats import nbinom, norm

from candi.metrics import P_EPS, calibration_pit_curve, ece, nb_crps

__all__ = [
    "p_from_mu", "nb_crps_mean", "oracle_scale", "marginal_nb", "gauss_crps", "gauss_crps_mean",
    "pit_curve", "ece", "c_index_nb", "c_index_gauss", "coverage_nb", "coverage_gauss",
    "nb_suite", "gauss_suite",
    "SIGNAL_TARGET_TRANSFORMS", "transform_signal_target", "invert_signal_prediction",
    "SIGNAL_EVAL_SPACE",
    "NLL_EPS", "GAUSSIAN_VAR_EPS", "BERNOULLI_PROB_EPS",
    "nb_nll", "gaussian_nll", "bernoulli_nll",
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
        # a non-finite CRPS on either side makes the comparison ABSENT (None), never a loss:
        # bool(nan < x) is False, which silently recorded unscoreable tracks as 0.0 (t56)
        out["beats_marginal"] = (
            bool(crps < mar["marg_crps"])
            if np.isfinite(crps) and np.isfinite(mar["marg_crps"]) else None)
        out["beats_marginal_oracle_scaled"] = (
            bool(orc["crps_oracle_scaled"] < mar["marg_crps"])
            if np.isfinite(orc["crps_oracle_scaled"]) and np.isfinite(mar["marg_crps"]) else None)
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


# ---------------------------------------------------------------------------
# the LOSS — the three NLLs the objective is made of, in pure numpy
# ---------------------------------------------------------------------------
# Every constant below is matched to the training loop's, not chosen here. `NLL_EPS` is
# `train.py::_elem_nb_nll`'s default; `GAUSSIAN_VAR_EPS` is `torch.nn.functional.gaussian_nll_loss`'s
# own default clamp; the Bernoulli one is this file's alone and is explained on the function.
#
# WHY THESE ARE MEANS AND THE TRAINING TERMS ARE ALSO MEANS, BUT NOT THE SAME MEAN. The training loop
# reduces over the positions its masks selected and reports `obs` and `imp` separately; a bench track
# is one (pair, assay) over every bin of every eval chromosome, and every one of those bins is an
# imputation target by construction. So the bench number is the `imp` term of the same formula,
# evaluated on the eval panel instead of on a training batch — which is exactly what a val/test loss
# is. The formula is identical; the population it is averaged over is the thing that differs, and
# that difference is the point.

#: `train.py::_elem_nb_nll(..., eps=1e-6)`. Clamps `1 - p` away from both ends and floors `n`.
NLL_EPS = 1e-6

#: `torch.nn.functional.gaussian_nll_loss(..., eps=1e-6)` — its own default, applied to `var` before
#: the log and the division. Not a number chosen here.
GAUSSIAN_VAR_EPS = 1e-6

#: THIS ONE HAS NO COUNTERPART IN `train.py`, and the reason is worth stating. The training loop
#: scores the peak head with `binary_cross_entropy_with_logits`, which fuses the sigmoid into the
#: loss and never forms `1 - p`, so it needs no clamp at all. `harness.stream_tracks` stores
#: `sigmoid(peak_logit)` — the probability, not the logit — so this copy has to take the log of a
#: number that may have already rounded to exactly 0.0 or 1.0 in float32, and a clamp is the only
#: way back. At float32's ulp near 1.0 this bites from about |logit| = 17.
BERNOULLI_PROB_EPS = 1e-12

#: `train.py::SIGNAL_TARGET_TRANSFORMS`, mirrored rather than imported. Kept a tuple in the same
#: order so a mismatch is a one-line diff.
SIGNAL_TARGET_TRANSFORMS = ("none", "arcsinh", "log1p")


def transform_signal_target(y: np.ndarray, mode: str = "none") -> np.ndarray:
    """Bend `-log10 p` into the space the Gaussian head was TRAINED to predict (D30).

    The numpy twin of `train.py::_apply_signal_target_transform`, and it belongs to the LOSS PATH
    ALONE. `gaussian_nll` is only the training loop's number if its target lives in the training
    loop's space: on a store the head is fit against `arcsinh(-log10 p)`, so the loss bends the
    TRUTH forward to meet the prediction and records which bend it used.

    THE BENCHMARK PATH GOES THE OTHER WAY, and that is the space contract. Every benchmark measure
    in this suite is quoted in `-log10 p` (`SIGNAL_EVAL_SPACE`), so the pval arm bends the
    PREDICTION back with `invert_signal_prediction` and leaves the truth alone. Truth-forward for
    the loss, prediction-back for the benchmark: the two directions are not interchangeable, because
    a mean under a nonlinear map is not the map of the mean.
    """
    if mode == "none":
        return np.asarray(y, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if mode == "arcsinh":
        return np.arcsinh(y)
    if mode == "log1p":
        return np.log1p(y)
    raise ValueError(
        f"signal_target_transform must be one of {SIGNAL_TARGET_TRANSFORMS}; got {mode!r}")


#: The space EVERY pval-arm benchmark number in this suite is quoted in — raw `-log10 p`, the units
#: the store reader hands back (D26) and the units the truth arrives in on both data paths. Written
#: onto the emitted arm as `pred_space` so a reader can tell a row scored under this contract from a
#: row scored before it existed, where an `arcsinh` prediction was compared to a raw truth.
SIGNAL_EVAL_SPACE = "-log10p"


def invert_signal_prediction(mu: np.ndarray, sigma: np.ndarray, mode: str = "none"):
    """Bend the Gaussian head's `(µ, σ)` BACK from its training space into `-log10 p`.

    The inverse direction of `transform_signal_target`, and the benchmark path's half of the space
    contract: the truth is already `-log10 p`, so the prediction is what has to move. `mode` is the
    run's own `signal_target_transform`; `"none"` is the identity and returns the caller's arrays
    UNTOUCHED, so the h5 path is bit-identical to what it computed before this function existed.

    THE PUSHFORWARD IS NOT A GAUSSIAN AND THIS RETURNS A GAUSSIAN ANYWAY. `sinh` of a normal
    variate is not normal — it is a Johnson S_U — so `(sinh µ, cosh µ · σ)` is the DELTA METHOD: a
    first-order Taylor expansion about the mean, exact only in the limit of small `σ`. Every
    downstream key that reads the distribution rather than its centre (`crps`, `pit_ks`,
    `coverage_95`, `c_index`) inherits that approximation.

    WHAT THAT COSTS, IN THE DIRECTION IT COSTS IT. `cosh` grows exponentially, so a bin the head is
    confident about at large `µ` comes back with a very wide `σ'` — the same relative uncertainty
    the store codec's own error analysis has (`store/layout.py`), but now inside a metric. Read a
    store-path `pit_ks` or `coverage_95` as "calibration of the delta-method Gaussian", not as
    calibration of the head, and prefer the point tier and `crps` when the two disagree. The h5 path
    carries none of this: `mode="none"` does nothing at all.

    The exception, stated once here and again in `harness.loss_block`: the LOSS tier does not use
    this. `gaussian_nll` mirrors the training objective, so it stays in the training space with the
    transform recorded beside it.

    Both arrays come back as float64 for every mode but `"none"`; `sinh` overflows to `inf` around
    `µ = 710`, well above anything an arcsinh-space TARGET can be — a uint16 store cannot even
    represent one past `store.layout.PVAL_UINT16_MAX / PVAL_SCALE`. A prediction is not clamped to
    that range, so an untrained head can still emit a `µ` that overflows, and an `inf` here is the
    head's own output rather than an artefact of this function.
    """
    if mode == "none":
        return mu, sigma
    if mode not in SIGNAL_TARGET_TRANSFORMS:
        raise ValueError(
            f"signal_target_transform must be one of {SIGNAL_TARGET_TRANSFORMS}; got {mode!r}")
    m = np.asarray(mu, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    if mode == "arcsinh":
        return np.sinh(m), np.cosh(m) * s
    # log1p: the inverse is expm1 and d/dµ expm1(µ) = exp(µ).
    return np.expm1(m), np.exp(m) * s


def nb_nll(n: np.ndarray, mu: np.ndarray, counts: np.ndarray, *, eps: float = NLL_EPS) -> float:
    """Mean per-bin NB negative log-likelihood — `train.py::_elem_nb_nll`, in numpy.

    Parameterised by `(n, µ)` because that is what the harness carries; `p` is derived with
    `p_from_mu`, the same `n / (n + µ)` every other key in this file uses. The training loop is
    handed the model's own `p` instead, so the pin in `tests/test_bench_analytic.py` feeds torch the
    `p` this function derives — the claim being tested is that the two formulas agree, not that two
    different parameterisations do.

    THE THREE GUARDS ARE THE TRAINING LOOP'S, VALUE FOR VALUE. `probs` is `1 - p` clamped to
    `[eps, 1-eps]`; `total` is `n` floored at `eps`; the target is floored at 0. `torch`'s
    `NegativeBinomial(total_count, probs=probs)` puts the MEAN in `probs`'s complement, which is why
    `probs` here is `1 - p` and not `p`.

    float64 throughout while the objective runs behind `fp32_fence`, so the two agree to float32's
    precision rather than bit-exactly — which is the right way round: this side is the more accurate
    one.
    """
    n = np.asarray(n, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    probs = np.clip(1.0 - p_from_mu(n, mu), eps, 1.0 - eps)
    total = np.maximum(n, eps)
    k = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    log_unnorm = total * np.log1p(-probs) + k * np.log(probs)
    log_norm = -gammaln(total + k) + gammaln(1.0 + k) + gammaln(total)
    return float(-np.mean(log_unnorm - log_norm))


def gaussian_nll(mu: np.ndarray, var: np.ndarray, truth: np.ndarray, *,
                 eps: float = GAUSSIAN_VAR_EPS) -> float:
    """Mean per-bin Gaussian NLL with `full=True` — `train.py::_elem_gaussian_nll`, in numpy.

    `full=True` is not a detail: it adds `0.5*log(2*pi)`, and without it the number is an
    unnormalised score rather than a log-likelihood comparable to anything. Production uses
    `full=True` (`torch.nn.GaussianNLLLoss(reduction="none", full=True)`), so this does.

    `var` — the VARIANCE, not the standard deviation. The harness stores `signal_sigma`, so its
    caller squares it back; `torch` clamps `var` at `eps` before both the log and the division and
    that clamp is reproduced here, which is also what keeps a zero-variance bin finite.

    `truth` must ALREADY be in the head's target space — see `transform_signal_target`. Passing raw
    `-log10 p` for a run trained with `arcsinh` gives a finite, plausible, wrong number.
    """
    mu = np.asarray(mu, dtype=np.float64)
    v = np.maximum(np.asarray(var, dtype=np.float64), eps)
    y = np.asarray(truth, dtype=np.float64)
    return float(np.mean(0.5 * (np.log(v) + (mu - y) ** 2 / v + np.log(2.0 * np.pi))))


def bernoulli_nll(prob: np.ndarray, labels: np.ndarray, *,
                  eps: float = BERNOULLI_PROB_EPS) -> float:
    """Mean per-bin BCE **from probabilities** — `train.py::_elem_peak_bce`, in numpy.

    THE INPUT IS A PROBABILITY, NOT A LOGIT, and that is forced by the harness: `stream_tracks`
    stores `sigmoid(peak_logit)`, and the logit is gone by the time a score is computed. So this
    cannot use the log-sum-exp identity the training loop relies on and must clamp `prob` into
    `[eps, 1-eps]` before taking logs — see `BERNOULLI_PROB_EPS` for what that costs and where.

    THE TARGET CHECK IS `train.py::_elem_peak_bce`'S, FOR ITS REASON. `y_peaks` holds MISSING = -1
    for an assay the biosample does not have, and BCE against -1 is not defined. A -1 arriving here
    means the availability rule that picks scoring targets failed, and a silent finite number is the
    worst possible response to that.

    ONLY CALL THIS WHEN THE PEAK HEAD EXISTS. `stream_tracks` fills `peak_score` from the NB MEAN
    when there is no peak head — an unbounded count, not a probability — and a BCE of that is
    meaningless before it is numerically broken. `harness.loss_block` gates on
    `TrackRecord.has_peak_head` for exactly this.
    """
    y = np.asarray(labels, dtype=np.float64)
    if y.size and bool(((y < 0.0) | (y > 1.0)).any()):
        raise ValueError(
            "peak BCE received a label outside [0, 1] — almost certainly the MISSING = -1 sentinel "
            "for an assay this biosample does not have. Only available assays are scoring targets; "
            "see `harness.EvalSource.targets`.")
    p = np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
