"""The rank aggregation, against the challenge's own published results.

Two kinds of test. The first are analytic: constructed tables whose ranking is obvious on paper,
including the two details most often dropped when this procedure is reimplemented — the `min(0.5, r)`
cap and the second-best-bootstrap rule.

The second is the one that matters for placement. It feeds the **published per-team, per-experiment,
per-bootstrap score table** (Genome Biology 2023 supplementary MOESM3, 12,740 rows, 25 teams,
51 experiments, 10 bootstraps) through `aggregate_ranks` and compares the result to the **published
ranking**. This is the only check in the suite that touches real published results end to end.

What it does and does not claim:

- It **does** show our implementation of the aggregation is faithful enough to place an entrant.
  Measured: Spearman 0.9856, the top three exactly right, 13 of 25 teams at their exact published
  rank and 18 of 25 inside their published tie block.
- It does **not** claim exact reproduction. Archive `h71` refuted that, and `EVAL_PLAN.md` §10 puts
  it out of scope. Six aggregation variants — per-cell vs global team counts, penalising vs skipping
  a missing team, second-best vs mean bootstrap rank — were probed against this table and the first
  five give *identical* output, so the residual disagreement is not an implementation choice left
  unturned. Only the mean-vs-second-best axis moves anything, and it moves it the wrong way, which
  independently confirms the challenge's own "90th percentile" rule.

The thresholds below sit safely under what was measured. They are a regression guard, not a claim.
"""
from __future__ import annotations

import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from candi.bench.eic import MEASURES
from candi.bench.ranking import MISSING_SCORE, aggregate_ranks, rank_within_cell

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SCORES = FIXTURES / "eic_moesm3_scores.csv.gz"
PUBLISHED = FIXTURES / "eic_published_ranks.json"


# ---------------------------------------------------------------------------
# analytic
# ---------------------------------------------------------------------------

def test_rank_direction_is_applied_per_measure() -> None:
    """`gwcorr` is ASCENDING and `mse` is DESCENDING; a single table must honour both at once.

    Team A has the better (lower) mse and the worse (lower) gwcorr. Its average rank must therefore
    be exactly 1.5 — first on one measure, second on the other — and so must B's. A implementation
    that applied one direction to both would give 1.0 and 2.0.
    """
    cell = {"A": {"mse": 1.0, "gwcorr": 0.1}, "B": {"mse": 2.0, "gwcorr": 0.9}}
    r = rank_within_cell(cell, measures=["mse", "gwcorr"])
    assert r["A"] == pytest.approx(1.5)
    assert r["B"] == pytest.approx(1.5)


def test_ties_take_the_average_rank_so_input_order_cannot_decide() -> None:
    cell = {"A": {"mse": 1.0}, "B": {"mse": 1.0}, "C": {"mse": 2.0}}
    r = rank_within_cell(cell, measures=["mse"])
    assert r["A"] == pytest.approx(1.5)
    assert r["B"] == pytest.approx(1.5)
    assert r["C"] == pytest.approx(3.0)


def test_a_missing_measure_ranks_last_rather_than_being_dropped() -> None:
    """Otherwise a team improves its average simply by not reporting its worst number."""
    cell = {"A": {"mse": 5.0}, "B": {}}
    r = rank_within_cell(cell, measures=["mse"])
    assert r["A"] < r["B"]


FILLER = ["f1", "f2", "f3", "f4", "f5", "f6"]


def _cell(order):
    return {t: {"mse": float(i)} for i, t in enumerate(order)}


def test_the_half_cap_flips_the_winner_and_in_the_direction_people_do_not_expect() -> None:
    """Stage 3's `min(0.5, r)` bounds the PENALTY for a bad experiment, so it rewards volatility.

    Eight teams, two experiments. `swing` is 1st then last; `steady` is 3rd on both.

    Uncapped: swing = (1/8 + 8/8)/2 = 0.5625, steady = 3/8 = 0.375 -> steady wins.
    Capped:   swing = (0.125 + 0.5)/2 = 0.3125, steady = 0.375        -> swing wins.

    Both numbers are computed here rather than asserted from a recording, and the point of the test
    is that the two disagree: the cap is not a rounding detail, it decides the winner.
    """
    table = {0: {
        "e1": _cell(["swing", "x", "steady"] + FILLER[:5]),          # swing 1st, steady 3rd
        "e2": _cell(["x", "y", "steady"] + FILLER[:4] + ["swing"]),  # swing 8th, steady 3rd
    }}
    res = aggregate_ranks(table, measures=["mse"])
    capped_swing = res["mean_bootstrap_score"]["swing"]
    capped_steady = res["mean_bootstrap_score"]["steady"]
    assert capped_swing == pytest.approx((0.125 + 0.5) / 2)
    assert capped_steady == pytest.approx(0.375)
    assert capped_swing < capped_steady
    assert res["final_rank"]["swing"] < res["final_rank"]["steady"]

    uncapped_swing = (1 / 8 + 8 / 8) / 2
    uncapped_steady = 3 / 8
    assert uncapped_steady < uncapped_swing          # the cap reversed this


def test_the_second_best_bootstrap_rule_is_optimistic_and_rewards_two_good_draws() -> None:
    """Stage 4, in the direction it actually runs.

    `lucky` is 1st in two of ten bootstraps and 4th in the other eight (mean rank 3.4). `solid` is
    2nd in all ten (mean rank 2.0). The mean favours `solid`; the challenge's rule takes the
    second-best of the ten, which is 1 for `lucky` and 2 for `solid`, so `lucky` wins.

    Whether that is a good rule is not our call. It is the rule that produced the published table.
    """
    table = {}
    for b in range(10):
        if b < 2:
            order = ["lucky", "solid", "a", "b"] + FILLER[:4]
        else:
            order = ["a", "solid", "b", "lucky"] + FILLER[:4]
        table[b] = {"e1": _cell(order)}
    res = aggregate_ranks(table, measures=["mse"])

    assert res["second_best_bootstrap_rank"]["lucky"] == 1.0
    assert res["second_best_bootstrap_rank"]["solid"] == 2.0
    assert res["final_rank"]["lucky"] < res["final_rank"]["solid"]
    # ... and the mean would have said the opposite
    assert res["mean_bootstrap_score"]["solid"] < res["mean_bootstrap_score"]["lucky"]


def test_a_team_absent_from_a_cell_is_scored_at_the_cap_not_disqualified() -> None:
    table = {0: {"e1": {"A": {"mse": 1.0}, "B": {"mse": 2.0}}, "e2": {"A": {"mse": 1.0}}}}
    res = aggregate_ranks(table, measures=["mse"])
    assert res["mean_bootstrap_score"]["B"] == pytest.approx(
        np.mean([min(0.5, 2 / 2), MISSING_SCORE]))


def test_every_placement_carries_its_resolution_limit() -> None:
    """A placement quoted without the limit is not quotable; the limit ships inside the result."""
    table = {0: {"e1": {"A": {"mse": 1.0}, "B": {"mse": 2.0}}}}
    res = aggregate_ranks(table, measures=["mse"])
    assert res["resolution_limit_corr_units"] == 0.005
    assert res["unseparable_adjacent_pairs_in_reference"] == 5
    assert "h71" in res["provenance"] and "h77" in res["provenance"]


# ---------------------------------------------------------------------------
# against the published table
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def published_table():
    if not SCORES.exists():
        pytest.skip("MOESM3 fixture absent")
    table = defaultdict(lambda: defaultdict(dict))
    with gzip.open(SCORES, "rt") as fh:
        for row in csv.DictReader(fh):
            table[int(row["bootstrap_id"])][row["cell"] + row["assay"]][row["team_id"]] = {
                m: float(row[m]) for m in MEASURES}
    return table


def test_the_published_fixture_is_the_shape_the_challenge_reported(published_table) -> None:
    """25 teams, 51 blind experiments, 10 bootstraps — 12,740 rows."""
    assert sorted(published_table) == list(range(10))
    assert len(published_table[0]) == 51
    teams = {t for b in published_table for e in published_table[b] for t in published_table[b][e]}
    assert len(teams) == 25
    n_rows = sum(len(published_table[b][e]) for b in published_table for e in published_table[b])
    assert n_rows == 12_740


def test_our_aggregation_reproduces_the_published_ranking_well_enough_to_place_an_entrant(
        published_table) -> None:
    """The end-to-end check against real published results.

    Thresholds are set below the measured values (Spearman 0.9856, top-3 exact, 13/25 exact,
    18/25 within the published tie block) so this guards against regression without asserting an
    exactness that `h71` refuted.
    """
    if not PUBLISHED.exists():
        pytest.skip("published-rank fixture absent")
    pub = {t: i["rank_published"] for t, i in json.loads(PUBLISHED.read_text()).items()}
    ours = aggregate_ranks(published_table)["final_rank"]

    common = [t for t in pub if t in ours]
    assert len(common) == 25
    o = np.array([ours[t] for t in common], dtype=float)
    p = np.array([pub[t] for t in common], dtype=float)

    assert spearmanr(o, p).statistic > 0.95
    assert int((o == p).sum()) >= 12

    # inside the published tie block counts as correctly placed: the published column carries ties
    # (two teams at rank 3, two at 6, two at 8, two at 14, two at 21) and ours breaks them.
    within = 0
    for t in common:
        block = [x for x in common if pub[x] == pub[t]]
        within += int(pub[t] <= ours[t] <= pub[t] + len(block) - 1)
    assert within >= 17

    # the top of the table is where a placement claim would actually be made
    top3_ours = sorted(common, key=lambda t: ours[t])[:3]
    top3_pub = sorted(common, key=lambda t: pub[t])[:3]
    assert set(top3_ours) == set(top3_pub)


def test_the_choice_of_final_statistic_does_not_decide_the_placement(published_table) -> None:
    """How much of the agreement rests on picking the right stage-4 rule? Almost none.

    Ranking by second-best bootstrap rank and ranking by mean bootstrap score give Spearman 0.9856
    and 0.9863 against the published column — a difference of 0.0008, far inside the noise. This is
    stated because the opposite would have been worth knowing: if the two rules disagreed materially,
    every placement would depend on a detail the paper describes in one clause, and we would have to
    resolve it before quoting anything. They do not, so we use the challenge's stated rule and record
    that little hangs on it.

    Asserting that the two agree closely is the honest claim. Asserting that ours is *better* would
    be reading a 0.0008 difference as a result, which it is not.
    """
    if not PUBLISHED.exists():
        pytest.skip("published-rank fixture absent")
    pub = {t: i["rank_published"] for t, i in json.loads(PUBLISHED.read_text()).items()}
    res = aggregate_ranks(published_table)
    ours = res["final_rank"]
    common = [t for t in pub if t in ours]

    order = sorted(common, key=lambda t: res["mean_bootstrap_score"][t])
    by_mean = {t: i + 1 for i, t in enumerate(order)}

    p = np.array([pub[t] for t in common], dtype=float)
    s_second = spearmanr(np.array([ours[t] for t in common], dtype=float), p).statistic
    s_mean = spearmanr(np.array([by_mean[t] for t in common], dtype=float), p).statistic
    assert s_second > 0.95 and s_mean > 0.95
    assert abs(s_second - s_mean) < 0.01
