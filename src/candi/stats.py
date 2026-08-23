"""Cluster-level inference — the statistics, with no model and no h5 anywhere near them.

Lifted VERBATIM out of `candi.eval`, which has since been deleted (`EVAL_PLAN.md` D15). These three
functions were never part of the measurement stack: they are a bootstrap, a sign test and a
target-clustered interval, and every one of their callers — `compare_arms.py`, `report_h74.py` —
wants them without wanting an evaluation harness. Leaving them inside `eval.py` would have made the
cutover delete a statistic that three live call sites depend on.

**This module moves no number.** The bodies are unchanged, including the two hard-won details
below, and the move is pinned by `tests/test_stats.py` against values recorded from `eval.py`
before it happened.

The two details, because both were bugs once and both would look like style to a future reader:

1. **The cluster bootstrap resamples TARGETS, not positions.** Effective replication is the
   `(biosample, imp_biosample, assay)` target, not the ~893k positions inside it. A position-level
   interval ran **~24x too narrow**, which is not a rounding problem — it is the difference between
   an effect and a coincidence.
2. **The sign test DROPS exact ties rather than scoring them as negatives.** Counting ties as
   negatives made a bit-exactly unresponsive arm — all twelve deltas exactly `0` — read `n_pos=0`
   and therefore `p=0.00049`. The most perfectly null arm printed the most significant-looking
   p-value on the scorecard.
"""
from __future__ import annotations

from math import comb
from typing import Dict, List

import numpy as np

__all__ = ["bootstrap_ci", "sign_test_p", "cluster_bootstrap_ci"]


def bootstrap_ci(delta: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """LEGACY / over-confident: paired bootstrap over per-POSITION `delta` (= crps_flip - crps_true).

    Kept only as a labelled comparison. Positions within a target are not independent draws -- the real
    unit of replication is the target (12 of them / 5 biosample pairs), so this CI is ~24x too narrow.
    Use `cluster_bootstrap_ci` for any inferential claim.
    """
    if delta.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), excludes_zero=False, n=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    boot = delta[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])
    return dict(mean=float(delta.mean()), lo=float(lo), hi=float(hi),
                excludes_zero=bool(lo > 0.0 or hi < 0.0), n=int(delta.size))


def sign_test_p(n_pos: int, n: int) -> float:
    """Exact two-sided binomial sign test at p=0.5 (small-p method). n<=12 here, so enumerate."""
    if n <= 0:
        return float("nan")
    pmf = [comb(n, k) * 0.5 ** n for k in range(n + 1)]
    return float(sum(p for p in pmf if p <= pmf[n_pos] + 1e-12))


def cluster_bootstrap_ci(per_target: List[Dict], *, value_key: str = "mean_delta",
                          n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                          exclude_flagged: bool = True) -> Dict:
    """PRIMARY inference: bootstrap over TARGETS, `n_fg`-weighted, with a target-level sign test.

    Effective replication is 12 `(biosample, imp_biosample, assay)` targets / 5 biosample pairs, not the
    ~893k positions `bootstrap_ci` resamples. Records are first collapsed to one `n_fg`-weighted number
    per target, then the TARGETS are resampled. The unweighted variant is dominated by a few blown-up
    records, so the weighting is not cosmetic.

    Also reports `sign` and `supports_direction = lo > 0` separately, because `excludes_zero` is
    sign-blind -- a significant effect in the WRONG direction would satisfy it.

    Records whose foreground fell back off the `target >= 1` purity filter are dropped by default
    (`exclude_flagged`): they look healthy in `n_fg` but were scored on background.
    """
    n_flagged = 0
    if exclude_flagged:
        n_flagged = sum(1 for r in per_target if r.get("purity_fallback_fired", False))
        per_target = [r for r in per_target if not r.get("purity_fallback_fired", False)]
    if not per_target:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), excludes_zero=False,
                    supports_direction=False, sign=0, n_clusters=0, n_records=0, n_pos=0,
                    sign_test_p=float("nan"), n_flagged_excluded=n_flagged)
    agg: Dict = {}
    for r in per_target:
        k = tuple(r["target"])
        v, w = float(r[value_key]), max(float(r.get("n_fg", 1) or 1), 1.0)
        s = agg.setdefault(k, [0.0, 0.0])
        s[0] += w * v
        s[1] += w
    keys = sorted(agg)
    vals = np.array([agg[k][0] / agg[k][1] for k in keys])
    wts = np.array([agg[k][1] for k in keys])
    C = len(keys)
    idx = np.random.default_rng(seed).integers(0, C, size=(n_boot, C))
    boot = (vals[idx] * wts[idx]).sum(axis=1) / wts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1.0 - alpha / 2])
    mean = float((vals * wts).sum() / wts.sum())
    # Sign test: DROP exact ties (standard practice), do not score them as negatives. Counting ties as
    # negatives made a bit-exactly unresponsive arm (all 12 deltas == 0) read n_pos=0 -> p=0.00049, i.e.
    # the most perfectly NULL arm printed the most significant-looking p-value in the scorecard.
    n_pos = int((vals > 0).sum())
    n_neg = int((vals < 0).sum())
    n_eff = n_pos + n_neg
    n_tied = C - n_eff
    return dict(mean=mean, lo=float(lo), hi=float(hi),
                excludes_zero=bool(lo > 0.0 or hi < 0.0),
                supports_direction=bool(lo > 0.0), sign=int(np.sign(mean)),
                n_clusters=C, n_records=len(per_target), n_flagged_excluded=n_flagged,
                n_pos=n_pos, n_neg=n_neg, n_tied=n_tied,
                frac_pos=n_pos / C, sign_test_p=sign_test_p(n_pos, n_eff),
                cluster_values=[float(v) for v in vals])
