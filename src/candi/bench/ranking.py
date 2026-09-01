"""The challenge's rank aggregation — inert unless a competitor score table is supplied (D4).

The ENCODE Imputation Challenge never ranked on a score. It ranked on a **rank**, through a specific
four-stage procedure, and reproducing that procedure is the only way a CANDI number can be placed
next to a published one:

1. Ten equally sized bootstraps are drawn from the pool of genomic positions, and all nine measures
   are computed for every team on every bootstrap of every experiment.
2. Within each (bootstrap, experiment) cell the scores are converted to **ranks across teams**, per
   measure, and those ranks are averaged over the nine measures.
3. Across experiments, a team's score for that bootstrap is `mean_e min(0.5, r_e)`, where `r_e` is
   its rank divided by the number of teams. **The cap bounds the penalty for a bad experiment**:
   everything at or below the median counts the same, so placing last on one experiment costs
   exactly as much as placing median. It therefore *helps* a team that is excellent on some
   experiments and terrible on others, and it can flip the winner — on an eight-team pair where an
   uncapped average puts a consistent team ahead, the cap puts the erratic one ahead instead.
4. Teams are ranked within each bootstrap, and the **second-best** bootstrap rank (the challenge's
   "90th percentile") decides the winner. This is an **optimistic** statistic, and it is worth being
   explicit about the direction: it discards the eight worst bootstraps, so a team needs only two
   good ones. It does not reward stability — a team that is best twice and mediocre eight times
   beats a team that is second every single time.

Neither of those is our design choice and neither is defended here; they are what produced the
published numbers, and `EVAL_PLAN.md` D16 says we reproduce the benchmark including the parts we
would not have chosen.

A team missing from a bootstrap is scored 0.5, the same as the cap: absent is treated as
below-median rather than as disqualifying.

**What this can and cannot tell you.** Archive `h71` refuted exact reproduction of the published
score table — the best of fifteen reconstructions landed at 3.514e-02 against a 1e-4 bar, because
recovering `gwcorr` and recovering `mse1obs` pull against each other along a frontier of slope −0.40.
Archive `h77` then showed the residual is **common across teams** rather than team-dependent, so it
largely cancels under ranking: rescoring all 25 teams through one identical path held 16 of 25 exact
published ranks, with no team moving more than two places.

So the ORDER is reproducible and the SCORES are not. Two numbers must therefore travel with any
placement produced here, and `aggregate_ranks` returns both rather than leaving them to a footnote:
a resolution limit of **~0.005 correlation units**, and the **5 of 24** adjacent team pairs that
invert on three or more of the ten chromosome subsets. A placement that separates two teams by less
than that is not a placement.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from candi.bench.eic import MEASURES, RANK_DIRECTION

__all__ = ["MISSING_SCORE", "RESOLUTION_LIMIT_CORR", "UNSEPARABLE_ADJACENT_PAIRS",
           "rank_within_cell", "aggregate_ranks", "bootstrap_positions"]

#: What a team absent from a bootstrap-experiment cell scores. Equal to the cap, by design.
MISSING_SCORE = 0.5

#: Archive h77's measured resolution limit, in correlation units. Quoted with every placement.
RESOLUTION_LIMIT_CORR = 0.005

#: Archive h77: adjacent pairs that invert on >= 3 of the 10 chromosome subsets, out of 24 pairs.
UNSEPARABLE_ADJACENT_PAIRS = 5


def bootstrap_positions(n: int, n_boot: int = 10, seed: int = 0) -> List[np.ndarray]:
    """Ten equally sized position bootstraps.

    The challenge drew "ten equally sized bootstraps ... from the pool of all genomic positions".
    Equally sized and drawn with replacement, so each is `n` indices; the seed is explicit because
    two runs on different bootstraps are not comparable and the JSON has to record which was used.
    """
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=n) for _ in range(n_boot)]


def rank_within_cell(scores: Mapping[str, Mapping[str, float]],
                     measures: Sequence[str] = MEASURES) -> Dict[str, float]:
    """Average rank across measures, for one (bootstrap, experiment) cell.

    `scores` is `{team: {measure: value}}`. Rank 1 is best, and "best" depends on the measure:
    `gwcorr` and `gwspear` are ASCENDING (bigger is better) while every MSE is DESCENDING. Getting
    that table wrong silently inverts a rank, which is why it lives in `eic.RANK_DIRECTION` beside
    the measures and is asserted against the reference's own copy.

    Ties take the average rank, so two teams that score identically cannot be separated by input
    order. A team missing a measure is ranked last for that measure rather than dropped, because
    dropping it would let a team improve its average by simply not reporting its worst number.
    """
    teams = list(scores)
    out: Dict[str, List[float]] = {t: [] for t in teams}
    for m in measures:
        vals = []
        for t in teams:
            v = scores[t].get(m, np.nan)
            vals.append(np.nan if v is None else float(v))
        arr = np.asarray(vals, dtype=float)
        # Orient so that SMALLER is always better, then rank ascending.
        oriented = -arr if RANK_DIRECTION.get(m, "DESCENDING") == "ASCENDING" else arr
        # Missing values sort last under either orientation.
        oriented = np.where(np.isfinite(oriented), oriented, np.inf)
        order = np.argsort(oriented, kind="mergesort")
        ranks = np.empty(len(teams), dtype=float)
        ranks[order] = np.arange(1, len(teams) + 1, dtype=float)
        # average ties
        for value in np.unique(oriented):
            sel = oriented == value
            if sel.sum() > 1:
                ranks[sel] = ranks[sel].mean()
        for t, r in zip(teams, ranks):
            out[t].append(float(r))
    return {t: float(np.mean(v)) for t, v in out.items()}


def aggregate_ranks(table: Mapping[int, Mapping[str, Mapping[str, Mapping[str, float]]]],
                    *, measures: Sequence[str] = MEASURES) -> Dict[str, object]:
    """The full four-stage aggregation.

    `table[bootstrap][experiment][team][measure] = score`.

    Stage 3's `min(0.5, r)` cap is the part most often dropped when this is reimplemented, and it
    changes the winner rather than merely the margins — see the module docstring for the direction it
    changes it in, which is the opposite of the one people assume.

    Stage 4 takes the **second-best** of ten bootstrap ranks, not the mean — the challenge's own
    "90th percentile score", and an optimistic one.
    """
    boots = sorted(table)
    teams = sorted({t for b in boots for e in table[b] for t in table[b][e]})

    per_boot_score: Dict[int, Dict[str, float]] = {}
    for b in boots:
        per_exp: Dict[str, Dict[str, float]] = {}
        for e, cell in table[b].items():
            per_exp[e] = rank_within_cell(cell, measures)
        n_teams = max((len(c) for c in table[b].values()), default=1)
        scores: Dict[str, float] = {}
        for t in teams:
            vals = []
            for e in per_exp:
                if t in per_exp[e]:
                    vals.append(min(0.5, per_exp[e][t] / n_teams))
                else:
                    vals.append(MISSING_SCORE)
            scores[t] = float(np.mean(vals)) if vals else MISSING_SCORE
        per_boot_score[b] = scores

    # rank teams within each bootstrap (smaller aggregate score is better)
    per_boot_rank: Dict[int, Dict[str, float]] = {}
    for b, scores in per_boot_score.items():
        ordered = sorted(teams, key=lambda t: scores[t])
        per_boot_rank[b] = {t: float(i + 1) for i, t in enumerate(ordered)}

    final: Dict[str, float] = {}
    for t in teams:
        ranks = sorted(per_boot_rank[b][t] for b in boots)
        # the 90th-percentile score: the second-best bootstrap rank
        final[t] = float(ranks[1]) if len(ranks) > 1 else float(ranks[0])

    order = sorted(teams, key=lambda t: (final[t], np.mean([per_boot_score[b][t] for b in boots])))
    return {
        "final_rank": {t: i + 1 for i, t in enumerate(order)},
        "second_best_bootstrap_rank": final,
        "mean_bootstrap_score": {t: float(np.mean([per_boot_score[b][t] for b in boots]))
                                 for t in teams},
        "per_bootstrap_score": {str(b): per_boot_score[b] for b in boots},
        "n_bootstraps": len(boots),
        "n_teams": len(teams),
        # These two are not commentary. A placement that separates two entries by less than the
        # resolution limit is not a placement, and the reader needs the number to check.
        "resolution_limit_corr_units": RESOLUTION_LIMIT_CORR,
        "unseparable_adjacent_pairs_in_reference": UNSEPARABLE_ADJACENT_PAIRS,
        "provenance": (
            "Order reproduces the published EIC ranking (archive h77: 16/25 exact ranks, max move 2). "
            "Absolute scores do NOT reproduce it (archive h71, refuted at 3.514e-02 against a 1e-4 "
            "bar). Quote the resolution limit with any placement."
        ),
    }
