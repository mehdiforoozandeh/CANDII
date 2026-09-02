"""t31 — declaring what a store eval imputes, and keeping the test set out of the loop.

D31 says the pairing is DECLARED, never inferred, because D16 makes biosample names opaque ids no
loader may parse. `tools/split_regime_by_panel.py` is the one place a person parses them, deliberately,
and writes the result somewhere it can be audited.

The claim these tests defend is not that suffix matching works — it is that the VALIDATION and TEST
splits cannot leak into one another. `B_` is scored once, at the end; a `B_` biosample appearing in
a regime that drives best-checkpoint selection would put the test set inside the loop, and nothing
downstream could get it back out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.split_regime_by_panel import TRUTH_PREFIX, main, pair_by_suffix   # noqa: E402

TRAIN = ["T_K562", "T_HepG2", "T_lonely"]
POOL = ["V_K562", "V_HepG2", "B_K562", "B_other"]


def test_validation_pairs_use_the_v_counterpart_only():
    pairs, unpaired = pair_by_suffix(TRAIN, POOL, "V_")
    assert pairs == [["T_HepG2", "V_HepG2"], ["T_K562", "V_K562"]]
    assert unpaired == ["T_lonely"]


def test_test_pairs_use_the_b_counterpart_only():
    pairs, _ = pair_by_suffix(TRAIN, POOL, "B_")
    assert pairs == [["T_K562", "B_K562"]]


def test_no_b_biosample_can_reach_a_validation_regime(tmp_path):
    """The separation that keeps `B_` clean. A leak here is invisible until publication."""
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"biosamples": {"train": TRAIN, "eval": POOL}}))
    out = tmp_path / "val.json"
    main(["--in", str(src), "--out", str(out), "--split", "validation"])
    obj = json.loads(out.read_text())
    truth = [b for _, b in obj["eval_pairs"]]
    assert truth and not any(b.startswith("B_") for b in truth)
    assert not any(b.startswith("B_") for b in obj["biosamples"]["eval"])


def test_the_eval_pool_is_exactly_the_declared_truth_biosamples(tmp_path):
    """A biosample in `eval` that is in no pair is a target the loader would slot and never score."""
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"biosamples": {"train": TRAIN, "eval": POOL}}))
    out = tmp_path / "val.json"
    main(["--in", str(src), "--out", str(out), "--split", "validation"])
    obj = json.loads(out.read_text())
    assert obj["biosamples"]["eval"] == [b for _, b in obj["eval_pairs"]]


def test_a_counterpart_listed_under_train_is_still_found(tmp_path):
    """A regime written for neither split may list every biosample under `train`."""
    pairs, _ = pair_by_suffix(["T_K562", "V_K562"], ["T_K562", "V_K562"], "V_")
    assert pairs == [["T_K562", "V_K562"]]


def test_a_regime_with_no_counterparts_refuses_rather_than_writing_an_empty_list(tmp_path):
    """An empty `eval_pairs` reads as 'declared, none' and would silently disable the eval."""
    src = tmp_path / "in.json"
    src.write_text(json.dumps({"biosamples": {"train": ["T_a"], "eval": ["B_b"]}}))
    with pytest.raises(SystemExit):
        main(["--in", str(src), "--out", str(tmp_path / "x.json"), "--split", "validation"])
    assert not (tmp_path / "x.json").exists()


def test_the_two_splits_are_the_only_ones_and_they_do_not_overlap():
    assert TRUTH_PREFIX == {"validation": "V_", "test": "B_"}
    assert len(set(TRUTH_PREFIX.values())) == 2
