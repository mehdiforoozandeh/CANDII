"""`candi.stats` — the three statistics lifted out of `eval.py`, pinned so the move cost nothing.

`EVAL_PLAN.md` D15 deletes `eval.py`. These three were never part of the measurement stack — a
bootstrap, a sign test and a target-clustered interval — and three live call sites outside it
(`compare_arms.py` once, `report_h74.py` twice) want them without wanting an eval harness. So they
moved out ahead of the cutover, as a **no-op**: this file is the proof of the no-op, and the two
behaviours below are the ones that were bugs once.
"""
from __future__ import annotations

import numpy as np
import pytest

from candi.stats import bootstrap_ci, cluster_bootstrap_ci, sign_test_p


def _targets(n=37, seed=7):
    rng = np.random.default_rng(seed)
    rng.normal(0.02, 1.0, 5000)                      # burn the same draws eval.py's recording did
    return [{"target": (f"T_{i % 4}", f"V_{i % 4}", f"a{i % 3}"),
             "mean_delta": float(rng.normal(0.05, 0.3)),
             "n_fg": int(rng.integers(10, 500)),
             "purity_fallback_fired": bool(i % 11 == 0)} for i in range(n)]


# ---------------------------------------------------------------------------
# the move itself
# ---------------------------------------------------------------------------

def test_the_move_out_of_eval_py_changed_no_number() -> None:
    """Recorded from `candi.eval` BEFORE the bodies moved. Same seeds, same input, same output."""
    rng = np.random.default_rng(7)
    delta = rng.normal(0.02, 1.0, 5000)
    b = bootstrap_ci(delta, n_boot=200, seed=3)
    assert b["mean"] == pytest.approx(0.0008945523673244075, abs=1e-12)
    assert b["lo"] == pytest.approx(-0.025888384135406724, abs=1e-12)
    assert b["hi"] == pytest.approx(0.0307142204820142, abs=1e-12)
    assert b["n"] == 5000 and b["excludes_zero"] is False

    c = cluster_bootstrap_ci(_targets(), n_boot=300, seed=5)
    assert c["mean"] == pytest.approx(0.052778295280749093, abs=1e-12)
    assert c["lo"] == pytest.approx(-0.07658804938517991, abs=1e-12)
    assert c["hi"] == pytest.approx(0.20201833598205646, abs=1e-12)
    assert c["n_clusters"] == 12 and c["n_records"] == 33
    assert c["n_flagged_excluded"] == 4
    assert (c["n_pos"], c["n_neg"], c["n_tied"]) == (7, 5, 0)


# ---------------------------------------------------------------------------
# the two behaviours that were bugs
# ---------------------------------------------------------------------------

def test_a_perfectly_null_arm_is_not_the_most_significant_thing_on_the_scorecard() -> None:
    """Ties are DROPPED, not scored as negatives.

    Every delta exactly zero is what a bit-exactly unresponsive arm produces. Scoring those ties as
    negatives gave `n_pos = 0` out of 12 and therefore `p = 0.00049` — so the one arm that provably
    did nothing printed the smallest p-value in the table.
    """
    flat = [{"target": ("T_a", "V_a", f"assay{i}"), "mean_delta": 0.0, "n_fg": 100}
            for i in range(12)]
    out = cluster_bootstrap_ci(flat, n_boot=200, seed=0)
    assert out["n_tied"] == 12 and out["n_pos"] == 0 and out["n_neg"] == 0
    assert np.isnan(out["sign_test_p"]), "a sign test on zero informative pairs has no p-value"
    assert out["supports_direction"] is False
    assert out["mean"] == pytest.approx(0.0)


def test_clustering_over_targets_is_far_wider_than_over_positions() -> None:
    """The interval that matters is over TARGETS; the position-level one ran ~24x too narrow.

    Built so the two see the same signal: 8 targets whose true means differ, each backed by many
    correlated positions. The position bootstrap sees ~n_positions independent draws that are not
    independent at all, and its interval collapses.
    """
    rng = np.random.default_rng(0)
    per_target, positions = [], []
    for t in range(8):
        m = rng.normal(0.05, 0.25)                   # the target's own effect
        pos = m + rng.normal(0, 0.02, 4000)          # positions inside it barely vary
        positions.append(pos)
        per_target.append({"target": ("T_x", "V_x", f"a{t}"), "mean_delta": float(pos.mean()),
                           "n_fg": 4000})
    wide = cluster_bootstrap_ci(per_target, n_boot=2000, seed=0)
    narrow = bootstrap_ci(np.concatenate(positions), n_boot=2000, seed=0)
    assert (wide["hi"] - wide["lo"]) > 10 * (narrow["hi"] - narrow["lo"])


def test_the_weighting_is_not_cosmetic() -> None:
    """Records collapse to one `n_fg`-weighted number per target before the targets are resampled.

    Without the weighting a single tiny-support record moves the mean as much as a large one.
    """
    heavy = {"target": ("T", "V", "a"), "mean_delta": 1.0, "n_fg": 10_000}
    light = {"target": ("T", "V", "a"), "mean_delta": -1.0, "n_fg": 1}
    out = cluster_bootstrap_ci([heavy, light], n_boot=50, seed=0)
    assert out["n_clusters"] == 1, "both records are the SAME target and collapse into it"
    assert out["mean"] == pytest.approx((10_000 * 1.0 - 1.0) / 10_001)
    assert out["mean"] > 0.99


def test_the_flagged_records_are_dropped_by_default_and_counted() -> None:
    """A record whose foreground fell back off the purity filter looks healthy in `n_fg` but was
    scored on background, so it is excluded — and the exclusion is reported, never silent."""
    rows = [{"target": ("T", "V", f"a{i}"), "mean_delta": 1.0, "n_fg": 100,
             "purity_fallback_fired": i < 3} for i in range(6)]
    out = cluster_bootstrap_ci(rows, n_boot=50, seed=0)
    assert out["n_flagged_excluded"] == 3 and out["n_clusters"] == 3
    assert cluster_bootstrap_ci(rows, n_boot=50, seed=0, exclude_flagged=False)["n_clusters"] == 6


def test_an_empty_input_returns_nan_rather_than_raising() -> None:
    for out in (cluster_bootstrap_ci([]), bootstrap_ci(np.array([]))):
        assert np.isnan(out["mean"]) and out["excludes_zero"] is False


@pytest.mark.parametrize("n_pos,n,expected", [(6, 12, 1.0), (0, 12, 2 * 0.5 ** 12),
                                              (12, 12, 2 * 0.5 ** 12), (0, 0, None)])
def test_the_sign_test_is_the_exact_two_sided_binomial(n_pos, n, expected) -> None:
    got = sign_test_p(n_pos, n)
    if expected is None:
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)
