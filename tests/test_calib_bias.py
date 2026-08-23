"""t31 — the per-track bias table, and the one way it could lie.

Calibration (a) found the mid-training macro reads 0.152 too low at the shipped coverage. The
follow-up asks WHICH tracks carry that, which means differencing per-track scores measured at two
coverages. There is exactly one way for that difference to be fake, and it is the thing these tests
guard: if the two coverages do not score the SAME set of tracks, the gap between their means is
partly a change of track set rather than a change of coverage -- and that is itself a coverage
effect, so the measurement would be reporting its own artifact as its result.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.calib_bias import bias_table                               # noqa: E402


def _rec(bios, imp, assay, crps, n=100):
    return {"biosample": bios, "imp_biosample": imp, "assay": assay, "split": "V_",
            "n_points": n, "crps": crps, "crps_fg": crps}


def test_the_bias_is_measured_against_the_reference_coverage():
    rows = bias_table({2: [_rec("T_a", "V_a", "H3K4me3", 0.50)],
                       128: [_rec("T_a", "V_a", "H3K4me3", 0.66)]}, ref_level=128)
    assert len(rows) == 1
    assert rows[0]["ref"] == pytest.approx(0.66)
    assert rows[0]["bias@2"] == pytest.approx(-0.16)
    assert rows[0]["bias@128"] == pytest.approx(0.0)


def test_a_track_missing_from_one_coverage_is_dropped_from_all_of_them():
    """Not filled, not unioned, not carried at its reference value -- dropped.

    Averaging over a different track set at each coverage would make the coverage effect and the
    track-set effect the same number, which is precisely the confusion the table exists to resolve.
    """
    rows = bias_table({2: [_rec("T_a", "V_a", "H3K4me3", 0.50)],
                       128: [_rec("T_a", "V_a", "H3K4me3", 0.66),
                             _rec("T_b", "V_b", "ATAC-seq", 0.90)]}, ref_level=128)
    assert [r["imp_biosample"] for r in rows] == ["V_a"]


def test_every_track_in_the_table_has_a_value_at_every_coverage():
    """The invariant the drop enforces, asserted directly rather than inferred from the count."""
    levels = {2: [], 8: [], 128: []}
    for lv in levels:
        levels[lv] = [_rec("T_a", "V_a", "H3K4me3", 0.5 + lv / 1000),
                      _rec("T_b", "V_b", "ATAC-seq", 0.8 + lv / 1000)]
    levels[8] = levels[8][:1]                       # V_b vanishes at one coverage only
    rows = bias_table(levels, ref_level=128)
    assert rows and all(all(f"crps@{lv}" in r for lv in levels) for r in rows)
    assert {r["imp_biosample"] for r in rows} == {"V_a"}


def test_tracks_are_keyed_on_the_triple_not_on_the_assay():
    """Two cells share an assay name; keying on the assay alone would silently merge them."""
    rows = bias_table({2: [_rec("T_a", "V_a", "H3K4me3", 0.50),
                           _rec("T_b", "V_b", "H3K4me3", 0.70)],
                       128: [_rec("T_a", "V_a", "H3K4me3", 0.60),
                             _rec("T_b", "V_b", "H3K4me3", 0.90)]}, ref_level=128)
    assert len(rows) == 2
    got = sorted(r["bias@2"] for r in rows)
    assert got[0] == pytest.approx(-0.20) and got[1] == pytest.approx(-0.10)


def test_no_common_track_gives_an_empty_table_rather_than_a_wrong_one():
    rows = bias_table({2: [_rec("T_a", "V_a", "H3K4me3", 0.5)],
                       128: [_rec("T_b", "V_b", "ATAC-seq", 0.9)]}, ref_level=128)
    assert rows == []


def test_the_scorer_hands_back_the_rows_the_table_needs():
    """D4 makes the per-track score the primitive; the macro mean is a view of it.

    Calibration (a) could not attribute its own finding because the rows had already been collapsed.
    `quick_eval` needed `return_records=True` to hand them over and retired with `candi.eval` (D15);
    the monitor that replaced it carries them unconditionally, so the failure cannot recur.
    """
    from candi.monitor import DialResult

    d = DialResult(kind="impute", macro={"crps": 0.5},
                   per_track={"T_a|V_a|H3K4me3": {"crps": 0.5, "assay": "H3K4me3"}}, n_tracks=1)
    assert "per_track" in d.as_dict()
    assert d.as_dict()["per_track"]["T_a|V_a|H3K4me3"]["crps"] == 0.5
