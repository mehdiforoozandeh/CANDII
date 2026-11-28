"""t396 C2: score one target in one of three spaces from arrays, and aggregate as the scorer does.

The three spaces (h57): overlap counts per 25 bp bin (`overlap_count`), strand-summed read starts
per 25 bp bin (`start_sum`), and 1 kb sums of overlap counts (`sum_1kb_count`, 40 bins per sum).

Every deviance term comes from `onion_eval.deviance` (`target_terms`, `gap`, `calibration_ratio`).
Next to the Poisson plug-in floor of `target_terms`, each target carries a half-vs-half floor: the
deviance of the pr2 half under the depth-scaled pr1 half (`D_halves`), and the model's deviance on
the same pr2 half (`D_model_pr2`). `ratio_to_halves` compares the two within one space only; the
share of bins at MU_MIN differs between spaces, so the ratio is not comparable across spaces.

The aggregate follows `onion_eval.score.deviance_block`: per assay a ratio of target means of
deviances normalized by D0, then the mean over the scorable assays (the eval set's `scorable`
filter, 9 assays for `evalset_p16.json`). An assay undefined in the scorable set makes the macro
undefined (NaN), as `score._macro`. numpy and scipy only (no torch), so it runs in numpy-only jobs.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

_TOOLS = Path(__file__).resolve().parents[1]
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_eval import deviance as DV  # noqa: E402

SPACES = ("overlap_count", "start_sum", "sum_1kb_count")
KB_BINS = 40                    # 40 bins x 25 bp = 1 kb
MU_MIN = DV.MU_MIN              # the floor on a predicted mean in every deviance term
# The clamps of `minicandi.losses.pool_nb` (`minicandi.contract` = tools/t118/ladder/model.py).
POOL_MU_MIN = 1e-4
POOL_N_MIN = 1e-3

METRICS = ("skill", "calibration_ratio", "r2", "ratio_to_halves")

__all__ = ["SPACES", "KB_BINS", "METRICS", "to_1kb", "pool_size_1kb", "score_target", "aggregate"]


# ------------------------------------------------------------------ the 1 kb space
def _blocks(x, k: int = KB_BINS) -> np.ndarray:
    x = np.asarray(x, np.float64)
    nb = x.shape[-1] // k
    return x[..., :nb * k].reshape(*x.shape[:-1], nb, k)


def to_1kb(x) -> np.ndarray:
    """[R, rb] -> [R, rb // 40]: the sum of 40 consecutive bins per region; the tail remainder is
    dropped. Works on any leading shape (the last axis is the bin axis)."""
    return _blocks(x).sum(-1)


def pool_size_1kb(mu, n) -> np.ndarray:
    """The moment-matched NB size of each 40-bin sum, the rule of `minicandi.losses.pool_nb`:
    n_1kb = (sum mu)^2 / sum(mu^2 / n), with mu and n clamped below first. Independence of bins
    given the rate is assumed, as in `pool_nb`."""
    mu = _blocks(np.maximum(np.asarray(mu, np.float64), POOL_MU_MIN))
    n = _blocks(np.maximum(np.asarray(n, np.float64), POOL_N_MIN))
    s = mu.sum(-1)
    return s * s / (mu * mu / n).sum(-1)


# ------------------------------------------------------------------ one target
def score_target(y, mu, n, mu_ref, y_pr1, y_pr2, shares) -> Dict[str, float]:
    """Every key of `deviance.target_terms(y, mu, n, mu_ref=mu_ref)` plus the half-vs-half terms:

    D_halves    deviance of y_pr2 under max(y_pr1 * f2 / f1, MU_MIN) (the half-vs-half floor);
    D0_pr2      deviance of y_pr2 under its own mean;
    D_model_pr2 deviance of y_pr2 under f2 * mu.

    `shares` = (f1, f2), the shares of the space's own halves in the full track. All arrays are in
    one space and share one shape (any shape; they are flattened, as one track)."""
    f1, f2 = (float(v) for v in shares)
    if not (f1 > 0 and f2 > 0):
        raise ValueError(f"shares {shares!r}: both must be positive")
    flat = lambda a: np.asarray(a, np.float64).ravel()     # target_terms takes 1-D tracks
    y, mu, n, mu_ref = flat(y), flat(mu), flat(n), flat(mu_ref)
    out = DV.target_terms(y, mu, n, mu_ref=mu_ref)
    y1, y2 = flat(y_pr1), flat(y_pr2)
    out["D_halves"] = DV.deviance(y2, np.maximum(y1 * (f2 / f1), MU_MIN))
    out["D0_pr2"] = DV.deviance(y2, np.full_like(y2, max(y2.mean(), MU_MIN)))
    out["D_model_pr2"] = DV.deviance(y2, f2 * mu)
    return out


# ------------------------------------------------------------------ aggregates
def _ratio_to_halves(recs) -> float:
    try:
        m = np.mean([r["D_model_pr2"] / r["D0_pr2"] for r in recs])
        h = np.mean([r["D_halves"] / r["D0_pr2"] for r in recs])
    except KeyError:
        return float("nan")
    return float(m / h) if h > 0 else float("nan")


def _assay_metrics(recs) -> Dict[str, float]:
    out = {"skill": DV.gap(recs, "D_ref") if all("D_ref" in r for r in recs) else float("nan"),
           "calibration_ratio": (DV.calibration_ratio(recs) if all("E_nb" in r for r in recs)
                                 else float("nan")),
           "r2": float(1.0 - np.mean([r["D_model"] / r["D0"] for r in recs])),
           "ratio_to_halves": _ratio_to_halves(recs)}
    out["n_targets"] = len(recs)
    return out


def _macro(per: Dict[str, Dict[str, float]], assays: Iterable[str]) -> Dict[str, float]:
    """Mean over `assays` per metric; NaN when one of them is undefined or none has a target."""
    sel = [per[a] for a in sorted(assays) if a in per]
    out = {}
    for k in METRICS:
        v = [float(p[k]) for p in sel]
        out[k] = float(np.mean(v)) if v and all(np.isfinite(v)) else float("nan")
    out["n_assays"] = len(sel)
    return out


def aggregate(records: List[dict], scorable) -> dict:
    """Per assay, then the macro over the assays in `scorable` (a dict or an iterable of names).

    Each record is a `score_target` dict plus the key `assay`. Returns
    {"per_assay": {assay: {skill, calibration_ratio, r2, ratio_to_halves, n_targets}},
     "scorable": {metric: macro over the scorable assays, n_assays},
     "all_assays": {metric: macro over every assay, n_assays},
     "missing_scorable": [scorable assays with no record]}.
    skill = deviance.gap(recs, "D_ref"); r2 = 1 - mean(D_model / D0);
    ratio_to_halves = mean(D_model_pr2 / D0_pr2) / mean(D_halves / D0_pr2)."""
    by = defaultdict(list)
    for r in records:
        by[r["assay"]].append(r)
    per = {a: _assay_metrics(rs) for a, rs in sorted(by.items())}
    sc = set(scorable)
    return {"per_assay": per,
            "scorable": _macro(per, [a for a in per if a in sc]),
            "all_assays": _macro(per, list(per)),
            "missing_scorable": sorted(sc - set(per))}
