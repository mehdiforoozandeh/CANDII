"""Close-out D2: an add-on score splits the other_experiment pair jobs (the cross-lab and repeat
correction) by pair class, under `other_experiment_gap_closed_by_pair_class`. A score without
`--addon` has no such key (its bytes are pinned by `test_onion_eval_addon.py`'s frozen digest)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_eval import metrics as Mx  # noqa: E402
from onion_eval import score as S  # noqa: E402
from tests.test_onion_eval_addon import addon, frozen  # noqa: E402,F401  (module fixtures)


def _rec(pid, knob, m, b, f, cell="C1", assay="H3K4me3", rung="other_experiment"):
    return {"pair_id": pid, "rung": rung, "knob": knob, "cell": cell, "assay": assay,
            "source": f"{pid}a", "target": f"{pid}b", "crps_model": m, "crps_depth_only": b,
            "crps_floor": f, "crps_model_oracle_scaled": None, "pred_total": 1.0,
            "source_total": 1.0, "log2_depth_ratio": 0.0}


def test_block_on_hand_records():
    recs = [_rec("p1", "experiment;lab;read_length", 0.6, 1.0, 0.2),
            _rec("p2", "experiment;lab", 0.8, 1.0, 0.2, cell="C2"),
            _rec("p3", "experiment", 1.0, 1.0, 0.5),
            _rec("p4", "experiment;lab;donor", 0.5, 0.4, 0.4),       # baseline not above floor
            _rec("p5", "", 0.1, 1.0, 0.0, rung="isogenic")]           # other rung: left out
    b = S.other_experiment_pair_class_block(recs)
    pc = b["per_pair_class"]
    assert set(pc) == {"experiment", "experiment;lab", "experiment;lab;donor"}
    # ratio of means: (1.0 - 0.7) / (1.0 - 0.2) on p1 + p2
    assert np.isclose(pc["experiment;lab"]["value"], 0.375)
    assert pc["experiment;lab"]["n_pairs"] == 2 and pc["experiment;lab"]["n_cells"] == 2
    assert pc["experiment"]["value"] == 0.0
    assert pc["experiment;lab;donor"]["value"] is None
    assert "not above the floor" in pc["experiment;lab;donor"]["reason"]
    assert b["pooled"]["n_pairs"] == 4
    assert np.isclose(b["pooled"]["value"], Mx.gap_closed([0.6, 0.8, 1.0, 0.5], [1, 1, 1, 0.4],
                                                          [0.2, 0.2, 0.5, 0.4]))
    pp = {x["pair_id"]: x for x in b["per_pair"]}
    assert set(pp) == {"p1", "p2", "p3", "p4"}
    assert pp["p1"]["pair_class"] == "experiment;lab" and np.isclose(pp["p1"]["gap_closed"], 0.5)
    assert pp["p4"]["gap_closed"] is None
    assert all(v["flag"] == "few pairs" for v in pc.values())     # under 5 pairs from 3 cells
    many = [_rec(f"q{i}", "experiment;lab", 0.6, 1.0, 0.2, cell=f"C{i % 3}") for i in range(5)]
    ok = S.other_experiment_pair_class_block(many)["per_pair_class"]["experiment;lab"]
    assert (ok["n_pairs"], ok["n_cells"], ok["flag"]) == (5, 3, "ok")
    four = S.other_experiment_pair_class_block(many[:4])["per_pair_class"]["experiment;lab"]
    assert four["flag"] == "few pairs"


def test_empty_block():
    b = S.other_experiment_pair_class_block([])
    assert b["pooled"]["value"] is None and b["per_pair_class"] == {} and b["per_pair"] == []
    assert b["pooled"]["flag"] == "few pairs"


def test_addon_score_has_the_block_and_frozen_score_does_not(frozen, addon):  # noqa: F811
    assert S.ADDON_PAIR_CLASS_KEY not in frozen["scores"]
    b = addon["scores"][S.ADDON_PAIR_CLASS_KEY]
    n = addon["scores"]["n_replicate_pairs"]
    assert b["pooled"]["n_pairs"] == n == len(b["per_pair"]) and n > 0
    assert sum(v["n_pairs"] for v in b["per_pair_class"].values()) == n
