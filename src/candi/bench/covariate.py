"""Covariate sensitivity: is the conditioning actually doing anything?

CANDI is told four things about the target it must predict — `[log2 depth, assay_id, read_length,
run_type]` — on both the input and the output side. Everything downstream of that claim depends on
the telling mattering. The conditional-generation literature names the failure exactly: a conditional
model is free to ignore its condition by settling on `p(y|c) = p(y)`, and nothing in a likelihood
curve reveals it.

Seven instruments, each answering a different question, each from a literature that already solved
it:

| | question | instrument |
|---|---|---|
| **covuse** | does the decoder respond to this covariate at all? | holdout randomization test (Tansey et al.), under two nulls |
| **covshare** | what fraction of the output does each covariate own? | Shapley effects (Owen), exact over 16 subsets |
| **depthdir** | does depth move the output the right way, by the right amount? | dose-response monotonicity over a told-depth ladder |
| **depthcounterfact** | told a false depth, what does it predict? | the depth counterfactual, scored against the real downsampled track |
| **covspec** | does each covariate move what it should, and not what it shouldn't? | a covariate x aspect gap, in the shape of MIG |
| **depthblind** | is the latent invariant to depth? | kBET, iLISI, batch ASW (scIB) |
| **biokeep** | invariant because it is good, or because it is empty? | bio-conservation, never reported apart from `depthblind` |

**`biokeep` is not optional decoration.** An encoder that returns the same vector for every input is
perfectly invariant and scores a perfect `depthblind`. The deleted `eval.py` guarded against that
with `encoder_eff_rank_pooled > 1.0`, which a latent with two directions clears. The scIB discipline
is that a batch-removal score is *never* quoted without a biological-conservation score beside it,
and `depthblind` here refuses to return without one.

Everything below takes a `Predictor` — a plain callable from a covariate matrix to predicted
distribution parameters. That indirection is what makes the block testable: `test_bench_stub_model.py`
passes in a model whose conditioning we wrote ourselves, so every instrument has a known answer.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

from candi.bench.distributional import nb_crps_mean, p_from_mu
from candi.metrics import nb_crps

__all__ = [
    "COVARIATES", "ASPECTS", "Predictor",
    "covuse", "covshare", "depthdir", "depthcounterfact", "covspec", "depthblind", "biokeep",
    "aspects_of", "kbet", "ilisi", "batch_asw", "effective_rank",
]

# Row order of CANDI's `y_meta`. Named here so no instrument indexes a bare integer.
COVARIATES: Tuple[str, ...] = ("depth", "assay_id", "read_length", "run_type")

# `covspec`'s output aspects (EVAL_PLAN D11). Four functionals of one predicted distribution over a
# window.
ASPECTS: Tuple[str, ...] = ("level", "shape", "dispersion", "tail")

# A predictor maps a covariate matrix `(n_units, n_covariates)` to `{"mu": (n_units, L),
# "n": (n_units, L)}`. Nothing here knows what a checkpoint is.
Predictor = Callable[[np.ndarray], Mapping[str, np.ndarray]]


# ---------------------------------------------------------------------------
# output aspects — the columns of `covspec`, and the scalarisation `covshare` needs
# ---------------------------------------------------------------------------

def aspects_of(mu: np.ndarray, n: np.ndarray) -> Dict[str, np.ndarray]:
    """The four aspects of a predicted NB profile, one value per unit (D11).

    - `level` — `log(mean µ)` over the window. How much signal, in total.
    - `shape` — the profile after dividing by its own window mean, reduced to a scalar summary by
      its standard deviation. Scale-free by construction, so a pure level change does not move it.
    - `dispersion` — `mean(1/n̂)`. The NB's overdispersion, not its mean.
    - `tail` — the predicted 99th percentile over the mean. Peak height relative to bulk, which is
      what the biology reads and which neither level nor shape captures.

    `shape` is deliberately a *summary* of the profile rather than the profile itself: `covspec`
    compares how far each aspect moves under each covariate flip, and that needs commensurable
    scalars. The full profile distance is available to anyone who wants it via `mu` directly.
    """
    mu = np.asarray(mu, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    window_mean = np.maximum(mu.mean(axis=-1, keepdims=True), 1e-12)
    normed = mu / window_mean
    return {
        "level": np.log(window_mean[..., 0]),
        "shape": normed.std(axis=-1),
        "dispersion": (1.0 / np.maximum(n, 1e-12)).mean(axis=-1),
        "tail": np.percentile(mu, 99, axis=-1) / window_mean[..., 0],
    }


# ---------------------------------------------------------------------------
# covuse — does the decoder use the covariate at all?
# ---------------------------------------------------------------------------

def _marginal_resample(cov: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Replace column `k` with values drawn from its own marginal, independently of the rest.

    This is the null the deleted `eval.py`'s covariate ablation already used, and it answers a real
    question — "does the decoder read this row of the prompt at all" — but it manufactures
    covariate combinations that
    cannot occur. `AGENTS.md` §329 records `H(run_type | assay_id, read_length) = 0.000 bits` on 26
    `T_` records, so an independently drawn `run_type` is frequently impossible given the other two.
    """
    out = np.array(cov, copy=True)
    out[:, k] = cov[rng.integers(0, len(cov), size=len(cov)), k]
    return out


def _conditional_resample(cov: np.ndarray, k: int, rng: np.random.Generator, *,
                          given: Optional[Sequence[int]] = None) -> np.ndarray:
    """Replace column `k` with a draw from its **empirical conditional** given the other columns.

    The holdout-randomization-test null. Rows are grouped by the value of the conditioning columns and
    the replacement is drawn from within the row's own group, so every substituted prompt is one that
    occurs in the data.

    When a group has only one distinct value of column `k` — which is exactly the degenerate case
    `H(run_type | assay_id, read_length) = 0` describes — the resample is forced to return the value
    already there. That is not a bug to route around: it means the conditional null is *empty*, the
    covariate carries no information beyond the others, and the resulting p-value of 1.0 is the
    correct answer. `covuse` reports `conditional_null_degenerate` so it cannot be misread as the
    model ignoring the covariate.
    """
    given = [j for j in range(cov.shape[1]) if j != k] if given is None else list(given)
    out = np.array(cov, copy=True)
    keys = [tuple(row) for row in cov[:, given]]
    buckets: Dict[tuple, List[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault(key, []).append(i)
    for key, idx in buckets.items():
        pool = cov[idx, k]
        out[idx, k] = pool[rng.integers(0, len(pool), size=len(idx))]
    return out


def _conditional_is_degenerate(cov: np.ndarray, k: int) -> bool:
    given = [j for j in range(cov.shape[1]) if j != k]
    buckets: Dict[tuple, set] = {}
    for row in cov:
        buckets.setdefault(tuple(row[given]), set()).add(row[k])
    return all(len(v) < 2 for v in buckets.values())


def covuse(predict: Predictor, cov: np.ndarray, target: np.ndarray, *,
           n_resamples: int = 200, seed: int = 0,
           covariate_names: Sequence[str] = COVARIATES) -> Dict[str, Dict[str, object]]:
    """Per covariate: is the true value better than a resampled one, and by how much?

    The test statistic is CRPS against the real target. For each resample the model is re-run with a
    substituted covariate and re-scored, giving a null distribution of losses; the p-value is the
    standard randomization-test form `(1 + #{L_r <= L_obs}) / (R + 1)`, which is exact and needs no
    asymptotics. A model that uses the covariate is hurt by a wrong value, so `L_r > L_obs` and the
    p-value is small.

    Both nulls run side by side (`EVAL_PLAN.md` D9) because they ask different questions and a model
    can pass one and fail the other. `within_batch` is the structural tripwire: the substituted value
    is the value already there, so every difference it reports is arithmetically zero. **It must read
    exactly `0.0`.** A non-zero `within_batch` is a bug in the probe, not a finding about the model,
    and it is the cheapest falsifier of the whole block.
    """
    rng = np.random.default_rng(seed)
    base = predict(cov)
    l_obs = nb_crps_mean(base["n"], base["mu"], target)

    out: Dict[str, Dict[str, object]] = {}
    for k, name in enumerate(covariate_names):
        rec: Dict[str, object] = {"crps_observed": l_obs}
        for null, resample in (("marginal", _marginal_resample),
                               ("conditional", _conditional_resample)):
            losses = []
            for _ in range(n_resamples):
                pred = predict(resample(cov, k, rng))
                losses.append(nb_crps_mean(pred["n"], pred["mu"], target))
            losses = np.asarray(losses)
            rec[f"{null}_p"] = float((1 + int((losses <= l_obs).sum())) / (n_resamples + 1))
            rec[f"{null}_mean_d_crps"] = float(losses.mean() - l_obs)
            rec[f"{null}_n_resamples"] = int(n_resamples)
        rec["conditional_null_degenerate"] = bool(_conditional_is_degenerate(cov, k))

        # the tripwire: substitute the value already present
        same = predict(np.array(cov, copy=True))
        rec["within_batch_d_crps"] = float(
            nb_crps_mean(same["n"], same["mu"], target) - l_obs)
        rec["uses_covariate"] = bool(rec["marginal_p"] < 0.05)
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
# covshare — how much of the output does each covariate own?
# ---------------------------------------------------------------------------

def _subsets(d: int, exclude: int) -> List[Tuple[int, ...]]:
    from itertools import combinations
    rest = [j for j in range(d) if j != exclude]
    return [c for r in range(len(rest) + 1) for c in combinations(rest, r)]


def covshare(predict: Predictor, cov: np.ndarray, *, scalar: str = "level",
             n_outer: int = 64, n_inner: int = 8, seed: int = 0,
             covariate_names: Sequence[str] = COVARIATES) -> Dict[str, float]:
    """Shapley effects — the normalised variance share of each covariate.

    **Shapley, not Sobol** (`EVAL_PLAN.md` D10). Sobol's total-effect index assumes the inputs are
    independent, and CANDI's covariates are provably not: `H(run_type | assay_id, read_length) = 0`.
    Under dependence Sobol indices stop summing to one and can attribute the same variance twice.
    Shapley effects distribute shared variance by the Shapley value, so they always sum to one and
    remain interpretable. The usual objection — exponential cost — does not apply here: four
    covariates means 16 subsets, computed exactly rather than sampled.

    `V(S) = Var(E[f | X_S])` is estimated the given-data way: draw an outer sample of `X_S` from the
    real covariate rows, average `f` over inner draws of the complement, then take the variance of
    those conditional means. The scalarisation `f` is one of the `covspec` aspects — `level` by
    default, because a covariate that changes nothing about the level of the prediction is a
    covariate the model is barely using.

    **The inner-sample bias correction is not optional, and leaving it out is the classic way this
    estimator lies.** A mean over `K` inner draws is itself noisy, so the naive variance of those
    means estimates `Var(E[f|X_S]) + E[Var(f|X_S)]/K`, not `Var(E[f|X_S])`. The second term is pure
    sampling noise, it is *positive*, and it is largest exactly for the subsets that carry no
    information — which inflates `V(S)` for irrelevant covariates and leaks share onto them. Measured
    on the depth-only stub at `n_inner=6`: depth came out at 0.902 with the other three sharing 0.098
    of a quantity they provably do not affect at all.

    Subtracting `mean_outer(inner sample variance) / K` removes the term exactly in expectation.
    **Both variances are `ddof=1`.** The correction is only exact when they are: `E[s²(means)] =
    Var(E[f|X_S]) + E[Var(f|X_S)]/K` holds for the unbiased sample variance, and pairing a `ddof=0`
    outer variance with a `ddof=1` inner one subtracts a term scaled for a different estimator,
    tilting `V(S)` low by a factor `(n_outer-1)/n_outer`.

    `V(S)` is clipped at zero for the Shapley arithmetic, because a negative variance would make the
    fifteen paired differences meaningless and the shares stop summing to one. **The clip is not
    reported as a finding.** It fires whenever the true effect sits below the estimator's resolution,
    which is a statement about `n_outer` and `n_inner`, not about the model — so the total is also
    returned unclamped (`total_variance_unclamped`), split into the two terms it is made of
    (`total_variance_naive`, `total_variance_bias`) and with an error bar
    (`total_variance_se`), and a reader compares the first against the last rather than reading a
    boolean. There is deliberately no `output_is_constant` flag: "the output does not vary" and "the
    effect is smaller than this sampling can see" are different claims and no single bit can carry
    both.
    """
    rng = np.random.default_rng(seed)
    d = cov.shape[1]
    N = len(cov)

    # COMMON RANDOM NUMBERS. The outer rows and the inner draws are sampled ONCE and reused by every
    # subset, so `V(S ∪ {i}) − V(S)` is a paired difference between two evaluations on the same
    # draws. Independent draws per subset make each V unbiased but the DIFFERENCE noisy, and the
    # Shapley value is a weighted sum of fifteen such differences — the noise compounds. Measured on
    # the depth-only stub before this change: depth's share moved from 0.99 to 0.87 between seed 0
    # and seed 1, on a model where the true answer is exactly 1. With paired draws the same call is
    # stable to ~0.01 across seeds. Nothing about the estimand changes; only its variance.
    outer_idx = (np.arange(N) if N <= n_outer
                 else rng.choice(N, size=n_outer, replace=False))
    inner_idx = rng.integers(0, N, size=(len(outer_idx), n_inner))

    n_out = len(outer_idx)

    def V(S: Tuple[int, ...]) -> Tuple[float, float, float, float]:
        """`(clamped, naive, bias, se)` for one subset. Only the first enters the Shapley sum."""
        if not S:
            return 0.0, 0.0, 0.0, 0.0
        means = np.empty(n_out)
        inner_vars = np.empty(n_out)
        for i, oi in enumerate(outer_idx):
            block = cov[inner_idx[i]].copy()
            block[:, list(S)] = cov[oi, list(S)]
            pred = predict(block)
            vals = np.asarray(aspects_of(pred["mu"], pred["n"])[scalar], dtype=np.float64)
            means[i] = float(vals.mean())
            inner_vars[i] = float(vals.var(ddof=1)) if vals.size > 1 else 0.0
        naive = float(means.var(ddof=1)) if n_out > 1 else 0.0
        bias = float(np.mean(inner_vars)) / max(n_inner, 1)
        # Error bar on `naive − bias`. The first term is the NORMAL-THEORY standard error of a
        # sample variance, `s²·sqrt(2/(m−1))` — stated rather than assumed away, since the outer
        # conditional means are not guaranteed Gaussian. The second is the standard error of the
        # mean inner variance, which is what the correction rests on.
        se_naive = naive * np.sqrt(2.0 / (n_out - 1)) if n_out > 1 else 0.0
        se_bias = (float(inner_vars.std(ddof=1)) / np.sqrt(n_out) / max(n_inner, 1)
                   if n_out > 1 else 0.0)
        se = float(np.hypot(se_naive, se_bias))
        return max(naive - bias, 0.0), naive, bias, se

    cache: Dict[Tuple[int, ...], Tuple[float, float, float, float]] = {}

    def V_full(S: Tuple[int, ...]) -> Tuple[float, float, float, float]:
        key = tuple(sorted(S))
        if key not in cache:
            cache[key] = V(key)
        return cache[key]

    def V_cached(S: Tuple[int, ...]) -> float:
        return V_full(S)[0]

    total, total_naive, total_bias, total_se = V_full(tuple(range(d)))
    from math import comb
    raw: Dict[str, float] = {}
    for k, name in enumerate(covariate_names):
        acc = 0.0
        for S in _subsets(d, k):
            weight = 1.0 / (d * comb(d - 1, len(S)))
            acc += weight * (V_cached(tuple(sorted(S + (k,)))) - V_cached(S))
        raw[name] = acc

    # Shapley effects sum to V(all) by construction; normalise so the report is a share. When the
    # total is zero there is no share to take -- 0/0 is undefined and nan says so. What the reader
    # needs in order to tell "constant output" from "effect below resolution" is the measurement
    # itself, so the two terms and the error bar ship in every return, not only that one.
    measurement = {
        "total_variance": float(total),
        "total_variance_unclamped": float(total_naive - total_bias),
        "total_variance_naive": float(total_naive),
        "total_variance_bias": float(total_bias),
        "total_variance_se": float(total_se),
        "n_outer_used": int(n_out),
    }
    if total <= 1e-12:
        return {**{k: float("nan") for k in raw}, **measurement}
    return {**{k: float(v / total) for k, v in raw.items()}, **measurement}


# ---------------------------------------------------------------------------
# depthdir / depthcounterfact — does depth move it the right way, by the right amount?
# ---------------------------------------------------------------------------

def depthdir(predict: Predictor, cov: np.ndarray, *, depth_col: int = 0,
             ladder: Sequence[float] = (0.0, -1.0, -2.0, -3.0)) -> Dict[str, float]:
    """Dose-response over told depth: does the predicted level fall as the told depth falls?

    `ladder` is in log2 units relative to the row's own depth, so `(0, -1, -2, -3)` is the DSF
    1/2/4/8 ladder. CPA calls the analogous object a learned drug-response curve; the property that
    matters is the same one — a monotone response, not merely a response.

    `monotone_frac` is the fraction of adjacent ladder steps that move in the correct direction, and
    it is the number to read. A model can put the argmin of a counterfactual in the right place while
    its curve wanders in between, and only the step-by-step check sees that.
    """
    levels = []
    for offset in ladder:
        c = np.array(cov, copy=True)
        c[:, depth_col] = cov[:, depth_col] + offset
        pred = predict(c)
        levels.append(np.asarray(aspects_of(pred["mu"], pred["n"])["level"]))
    L = np.stack(levels, axis=0)                      # (n_ladder, n_units)

    steps = np.diff(L, axis=0)                        # told depth decreasing => level should fall
    monotone = float(np.mean(steps <= 0.0))
    told = np.asarray(ladder, dtype=np.float64)
    corr = [float(np.corrcoef(told, L[:, u])[0, 1]) for u in range(L.shape[1])
            if np.std(L[:, u]) > 1e-12]
    return {
        "monotone_frac": monotone,
        "mean_level_by_told_depth": [float(v) for v in L.mean(axis=1)],
        "mean_dose_response_corr": float(np.mean(corr)) if corr else float("nan"),
        "n_flat_units": int(L.shape[1] - len(corr)),
        "ladder": [float(v) for v in ladder],
    }


def depthcounterfact(pred_by_told: Mapping[float, Mapping[str, np.ndarray]],
                     true_by_level: Mapping[float, np.ndarray], *,
                     fg_frac: float = 0.02, min_n: int = 5,
                     seed: int = 0) -> Dict[str, float]:
    """Told a false depth, what does it predict? The depth arm that has ground truth.

    This is `eval.py::_dsf_counterfactual` as a pure function. For each true downsampling level
    `k`, score every told-depth prediction against `counts_dsf{k}` and ask whether the prediction
    told the *true* depth wins.

    **ONE FIXED FOREGROUND, chosen on the full-depth truth and reused at every ladder level.** The
    `eval.py` version re-selected the top `fg_frac` positions from the level-*k* realization it was
    about to score, and that is the whole reason a perfect model there capped short of 1: positions
    selected for being high in a realization are high by noise as well as by signal, so the level-*k*
    counts on the mask read above their own mean and a told depth *higher* than *k* fits them
    better. The mask is a
    property of the truth, not of the level being scored, so it is drawn once — from the level whose
    counts are largest, which is the undownsampled track under any key convention (`pred_by_told` and
    `true_by_level` are keyed by DSF factor in `bench.harness` and by log2 offset in the stub tests,
    and the two run in opposite directions).

    Selection is by RANK, not by `truth >= quantile`: with 36-67% exact zeros a 98th-percentile
    threshold is often 0 and selects everything (`eval.py::_foreground_mask`). Ties are broken by a
    seeded jitter, and `truth >= 1` is applied as a purity filter whenever it still leaves `min_n`
    positions.

    **The one calibration that still travels with the number** (`EVAL.md`): `constant_answer_value`
    is the deterministic value of always putting the argmin at the full-depth level — not a chance
    baseline — so a value just above it has not "beaten chance". The old companion constant, a
    perfect-model ceiling of `0.73`, was a consequence of the per-level mask this function no longer
    uses; it is **not** restated here, and a replacement is a measurement someone has to make, not a
    number to carry over.
    """
    levels = sorted(true_by_level)
    told_values = sorted(pred_by_told)
    if not levels:
        return {"frac_min_at_true": float("nan"), "frac_beats_told1": float("nan"), "n_levels": 0,
                "crps_at_true_level": {}, "constant_answer_value": float("nan")}

    # the fixed foreground, on the deepest truth
    fg_level = max(levels, key=lambda k: float(np.asarray(true_by_level[k], dtype=np.float64).sum()))
    deep = np.asarray(true_by_level[fg_level], dtype=np.float64).reshape(-1)
    k_fg = int(min(deep.size, max(min_n, round(fg_frac * deep.size))))
    fg = np.zeros(deep.size, bool)
    jitter = np.random.default_rng(seed).random(deep.size)
    fg[np.lexsort((jitter, deep))[-k_fg:]] = True
    pos = deep >= 1.0
    purity_dropped = int(pos.sum()) < min(min_n, deep.size)
    if not purity_dropped:
        fg &= pos

    hits, beats, n = 0, 0, 0
    per_level: Dict[str, float] = {}
    for lvl in levels:
        y = np.asarray(true_by_level[lvl], dtype=np.float64).reshape(-1)[fg]
        scores = {}
        for told in told_values:
            p = pred_by_told[told]
            scores[told] = nb_crps_mean(np.asarray(p["n"]).reshape(-1)[fg],
                                        np.asarray(p["mu"]).reshape(-1)[fg], y)
        argmin = min(scores, key=scores.get)
        hits += int(abs(argmin - lvl) < 1e-9)
        beats += int(scores[lvl] <= scores[told_values[0]])
        n += 1
        per_level[str(lvl)] = float(scores[lvl])
    return {
        "frac_min_at_true": hits / n if n else float("nan"),
        "frac_beats_told1": beats / n if n else float("nan"),
        "n_levels": n,
        "crps_at_true_level": per_level,
        "constant_answer_value": 1.0 / len(levels) if levels else float("nan"),
        "fg_frac_requested": float(fg_frac),
        "n_fg": int(fg.sum()),
        "n_positions": int(deep.size),
        "fg_level": float(fg_level),
        "fg_purity_filter_dropped": bool(purity_dropped),
    }


# ---------------------------------------------------------------------------
# covspec — does each covariate move what it should, and not what it shouldn't?
# ---------------------------------------------------------------------------

def covspec(predict: Predictor, cov: np.ndarray, *, seed: int = 0,
            covariate_names: Sequence[str] = COVARIATES) -> Dict[str, object]:
    """A covariate x aspect matrix, scored by the top-vs-runner-up gap, in the shape of MIG.

    Each entry is the mean absolute change in one aspect when one covariate is marginally resampled,
    normalised by that aspect's own spread across units so the four aspects are commensurable. Then,
    **per aspect**, the gap between the covariate that moves it most and the covariate that moves it
    second-most, relative to the largest — the Mutual Information Gap construction, with "how much
    does flipping this move that" standing in for mutual information.

    A high gap means each aspect is controlled by one covariate: the conditioning is specific. A low
    gap means every covariate perturbs everything, which is the signature of a model that treats its
    prompt as generic noise. `EVAL_PLAN.md` D11 fixes the four aspects; nothing about this
    construction is standard in the literature, and that is stated rather than hidden.
    """
    rng = np.random.default_rng(seed)
    base = predict(cov)
    base_a = aspects_of(base["mu"], base["n"])
    scale = {a: max(float(np.std(v)), 1e-12) for a, v in base_a.items()}

    matrix = np.zeros((len(covariate_names), len(ASPECTS)))
    for k, _ in enumerate(covariate_names):
        pred = predict(_marginal_resample(cov, k, rng))
        moved = aspects_of(pred["mu"], pred["n"])
        for j, a in enumerate(ASPECTS):
            matrix[k, j] = float(np.mean(np.abs(moved[a] - base_a[a]))) / scale[a]

    gaps: Dict[str, float] = {}
    owners: Dict[str, str] = {}
    for j, a in enumerate(ASPECTS):
        col = np.sort(matrix[:, j])[::-1]
        gaps[a] = float((col[0] - col[1]) / col[0]) if col[0] > 1e-12 else float("nan")
        owners[a] = covariate_names[int(np.argmax(matrix[:, j]))]

    finite = [v for v in gaps.values() if np.isfinite(v)]
    return {
        "matrix": matrix.tolist(),
        "covariates": list(covariate_names),
        "aspects": list(ASPECTS),
        "gap_by_aspect": gaps,
        "owner_by_aspect": owners,
        "mean_gap": float(np.mean(finite)) if finite else float("nan"),
    }


# ---------------------------------------------------------------------------
# depthblind / biokeep — latent invariance, and the guard it is never quoted without
# ---------------------------------------------------------------------------

def _knn(Z: np.ndarray, k: int) -> np.ndarray:
    """Indices of the `k` nearest neighbours of each row, excluding the row itself."""
    d = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def kbet(Z: np.ndarray, batch: np.ndarray, *, k: int = 20, alpha: float = 0.05,
         n_test: int = 200, seed: int = 0) -> Dict[str, float]:
    """k-nearest-neighbour batch effect test: the fraction of neighbourhoods that reject mixing.

    For each tested point, compare the batch composition of its `k` neighbours against the global
    composition with a chi-square test. Well-mixed batches give a rejection rate at the nominal
    `alpha`; a latent that encodes the batch gives a rejection rate near 1.

    This is what the deleted `eval.py`'s cosine ratio lacked: a **null**. A within/between ratio of
    0.3 is a number someone chose; a rejection rate of 0.05 is what chance produces, so any excess
    is measurable rather than declared.
    """
    rng = np.random.default_rng(seed)
    Z, batch = np.asarray(Z, dtype=np.float64), np.asarray(batch)
    labels, counts = np.unique(batch, return_counts=True)
    if len(labels) < 2:
        return {"kbet_rejection_rate": float("nan"), "kbet_n_tested": 0, "kbet_alpha": alpha}
    expected = counts / counts.sum() * k

    idx = rng.choice(len(Z), size=min(n_test, len(Z)), replace=False)
    nb = _knn(Z, k)
    rejects = 0
    for i in idx:
        obs = np.array([(batch[nb[i]] == lab).sum() for lab in labels], dtype=np.float64)
        stat = float(((obs - expected) ** 2 / np.maximum(expected, 1e-12)).sum())
        if 1.0 - chi2.cdf(stat, df=len(labels) - 1) < alpha:
            rejects += 1
    return {"kbet_rejection_rate": rejects / len(idx), "kbet_n_tested": int(len(idx)),
            "kbet_alpha": float(alpha)}


def ilisi(Z: np.ndarray, batch: np.ndarray, *, k: int = 30) -> float:
    """Integrated local inverse Simpson's index: the effective number of batches per neighbourhood.

    Ranges from 1 (every neighbourhood is one batch — the latent encodes it) to `n_batches` (perfect
    mixing). Unlike a cosine ratio, the endpoints are known in advance, so the number reads without a
    reference run.

    **`k` is capped at one less than the smallest batch's size, and that cap is load-bearing.** With
    `k` larger than a batch, a neighbourhood cannot be drawn from one batch even in principle — the
    points do not exist — so the lower endpoint becomes unreachable and a latent that perfectly
    encodes the batch still scores well above 1. Measured while building the layer-3 fixture: 48
    points in 4 batches of 12, `k = 18`, a latent that *is* the batch label scored 1.91 against a
    floor that should have been 1.0. The metric was not wrong; it was being asked for a resolution its
    neighbourhood size could not carry.
    """
    Z, batch = np.asarray(Z, dtype=np.float64), np.asarray(batch)
    labels, counts = np.unique(batch, return_counts=True)
    if len(labels) < 2:
        return float("nan")
    k = max(1, min(k, len(Z) - 1, int(counts.min()) - 1))
    nb = _knn(Z, k)
    scores = []
    for i in range(len(Z)):
        p = np.array([(batch[nb[i]] == lab).mean() for lab in labels])
        scores.append(1.0 / max(float((p ** 2).sum()), 1e-12))
    return float(np.mean(scores))


def batch_asw(Z: np.ndarray, batch: np.ndarray) -> float:
    """scIB's batch silhouette: `mean(1 - |s|)` over batch labels. 1 = perfectly mixed.

    The absolute value is the point. A silhouette near 0 means batches are indistinguishable, and
    both a strongly positive and a strongly negative silhouette mean they are not.
    """
    s = _silhouette(np.asarray(Z, dtype=np.float64), np.asarray(batch))
    return float(np.mean(1.0 - np.abs(s))) if s.size else float("nan")


def _silhouette(Z: np.ndarray, labels: np.ndarray) -> np.ndarray:
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.empty(0)
    d = np.sqrt(np.maximum(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1), 0.0))
    out = np.zeros(len(Z))
    for i in range(len(Z)):
        own = labels[i]
        same = (labels == own)
        same[i] = False
        a = d[i, same].mean() if same.any() else 0.0
        b = min((d[i, labels == lab].mean() for lab in uniq if lab != own), default=0.0)
        out[i] = 0.0 if max(a, b) < 1e-12 else (b - a) / max(a, b)
    return out


def effective_rank(Z: np.ndarray) -> float:
    """`exp(H(σ/Σσ))` — the entropy-based effective rank of the latent's singular spectrum."""
    Z = np.asarray(Z, dtype=np.float64)
    Z = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Z, compute_uv=False)
    tot = s.sum()
    if tot <= 1e-12:
        return 0.0
    p = s / tot
    p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def biokeep(Z: np.ndarray, bio_label: np.ndarray) -> Dict[str, float]:
    """The guard: does the latent still separate the things it is supposed to separate?

    `bio_label` is the biological identity that must survive — for the depth-invariance measurement
    that is the genomic region, since the same region at four depths should collapse while different
    regions should not.

    Reported as a silhouette over `bio_label` (1 = perfectly separated, 0 = indistinguishable) plus
    the latent's effective rank. A collapsed encoder scores ~0 on the first and ~0-1 on the second
    while scoring perfectly on every `depthblind` metric, which is precisely the failure this exists
    to catch.
    """
    Z = np.asarray(Z, dtype=np.float64)
    s = _silhouette(Z, np.asarray(bio_label))
    return {
        "bio_silhouette": float(np.mean(s)) if s.size else float("nan"),
        "effective_rank": effective_rank(Z),
        "n_bio_labels": int(len(np.unique(bio_label))),
    }


def depthblind(Z: np.ndarray, batch: np.ndarray, bio_label: np.ndarray, *,
               k: int = 20, alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """`depthblind` and `biokeep` together, because `depthblind` alone is not reportable (D13).

    `batch` is the nuisance the latent should be invariant to (DSF level); `bio_label` is the signal
    it must keep (the region). This function will not return one without the other — there is no
    argument that makes them optional, and an API that allowed it would eventually be used that way.
    """
    out: Dict[str, float] = {}
    out.update(kbet(Z, batch, k=k, alpha=alpha, seed=seed))
    _, batch_counts = np.unique(batch, return_counts=True)
    k_ilisi = max(1, min(k + 10, len(Z) - 1, int(batch_counts.min()) - 1))
    out["ilisi"] = ilisi(Z, batch, k=k_ilisi)
    out["ilisi_k"] = int(k_ilisi)                  # reported: it decides the reachable floor
    out["ilisi_max"] = float(len(np.unique(batch)))
    out["batch_asw"] = batch_asw(Z, batch)
    out.update(biokeep(Z, bio_label))
    # The verdict needs BOTH halves. Invariance alone is satisfied by a constant encoder.
    out["invariance_ok"] = bool(
        out["kbet_rejection_rate"] < 0.25
        and out["batch_asw"] > 0.8
        and out["bio_silhouette"] > 0.25
    )
    return out
