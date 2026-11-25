import numpy as np

from onion_eval import covariate_prompt as CP


class Enc:
    def __init__(self):
        self.ids = {"lab": {"A": 1, "B": 2, "C": 3}}

    def cat_id(self, field, value):
        return self.ids.get(field, {}).get(value, 0)


def _native(pe=0.0, rl=36.0, dedup=1.0, mapq=30.0, lab=1.0):
    return np.array([25.0, pe, rl, dedup, mapq, 1, 1, lab, 2, 3, 4], dtype=np.float64)


def test_arms_change_one_column_each():
    nat = _native()
    a = CP.arms(nat, {"lab_id": 2, "platform_id": None, "antibody_id": 5})
    for name, row in a.items():
        diff = np.flatnonzero(row != nat)
        assert len(diff) == 1, name
    assert a["depth_up"][0] == 26.0 and a["depth_down"][0] == 24.0
    assert a["run_type_flip"][1] == 1.0 and a["read_length"][2] == 100.0
    assert a["dedup_flip"][3] == 0.0 and a["mapq_flip"][4] == 0.0
    assert a["lab_other"][7] == 2.0 and a["lab_unknown"][7] == 0.0
    assert "platform_other" not in a and "platform_unknown" in a


def test_unknown_native_skips_unknown_arm():
    a = CP.arms(_native(lab=0.0), {"lab_id": 2})
    assert "lab_unknown" not in a and "lab_other" in a


def test_check_directions():
    nat = _native()
    s = {"total_ratio": 2.0, "shape_r": 0.99, "n_ratio": 1.0}
    assert CP.check("depth_up", nat, s)
    assert not CP.check("depth_down", nat, s)
    assert CP.check("dedup_flip", nat, {"total_ratio": 1.1, "n_ratio": 1})
    assert CP.check("mapq_flip", nat, {"total_ratio": 1.1, "n_ratio": 1})
    assert CP.check("run_type_flip", nat, {"total_ratio": 0.6, "n_ratio": 1})
    assert CP.check("lab_unknown", nat, {"total_ratio": 1.05, "n_ratio": 0.9})
    assert not CP.check("lab_unknown", nat, {"total_ratio": 1.05, "n_ratio": 1.1})


def test_other_id_most_frequent_seen_value():
    rows = {i: {"in_eic": "train", "assay": "H3K4me3", "lab": l} for i, l in
            enumerate(["A", "B", "B", "C", "C", "C"])}
    rows["v"] = {"in_eic": "val", "assay": "H3K4me3", "lab": "B"}
    tgt = {"assay": "H3K4me3", "lab": "C"}
    assert CP.other_id(rows, Enc(), tgt, "lab_id") == 2
    assert CP.other_id({}, Enc(), tgt, "lab_id") is None


def test_summarize_ratios():
    mu0 = np.array([1.0, 2.0, 3.0]); n0 = np.array([2.0, 2.0, 2.0])
    s = CP.summarize(n0, mu0, n0, 2 * mu0)
    assert abs(s["total_ratio"] - 2) < 1e-12 and s["n_ratio"] == 1.0 and s["shape_r"] > 0.99
