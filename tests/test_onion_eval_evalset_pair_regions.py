"""EVd: a derived eval set names a nested subset of regions for the pair jobs (`pair_regions`),
keeps everything else of its base, and never changes the base file. Synthetic tables."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (6, "H3K27me3"))
BLIND = ((8, "H3K4me3"),)
RB = 64                                          # 46 candidate blocks on the synthetic genome


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = tmp_path_factory.mktemp("evd")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    d = root / "eval" / "synth"
    base = d / "evalset.json"
    E.build(t, base, n_regions=8, n_fit_regions=2, region_bins=RB, seed=0)
    E.superset(base, 32, d / "regions32.json")
    sup = d / "evalset_r32.json"
    E.build(t, sup, regions_file=d / "regions32.json")
    return {"root": root, "tables": t, "dir": d, "base": base, "sup": sup}


def test_256_regions_give_nested_positions():
    pos = {p: E.pair_positions(256, p) for p in (8, 16, 32)}
    assert pos[8] == list(range(0, 256, 32)) and len(pos[16]) == 16 and len(pos[32]) == 32
    assert set(pos[8]) < set(pos[16]) < set(pos[32])
    with pytest.raises(ValueError, match="divide"):
        E.pair_positions(256, 24)


def test_derive_keeps_base_and_adds_keys(fx):
    base, before = fx["base"], sha(fx["base"])
    out4, out2 = fx["dir"] / "evalset_p4.json", fx["dir"] / "evalset_p2.json"
    assert E.main(["derive", "--base", str(base), "--pair-regions", "4", "--out", str(out4)]) == 0
    E.derive(base, out2, pair_regions=2)
    bev, d4, d2 = (json.loads(p.read_text()) for p in (base, out4, out2))
    assert sha(base) == before                                  # the base is never changed
    assert sha(out4) != before
    assert d4["pair_regions"] == [0, 2, 4, 6] and d2["pair_regions"] == [0, 4]
    assert set(d2["pair_regions"]) < set(d4["pair_regions"])
    assert d4["pair_regions_p"] == 4 and d4["pair_regions_rule"] == E.PAIR_REGIONS_RULE
    assert d4["derived_from"] == {"path": str(base.resolve()), "sha256": before, "seed": 0}
    extra = {"pair_regions", "pair_regions_p", "pair_regions_rule", "derived_from"}
    assert set(d4) - set(bev) == extra
    assert {k: v for k, v in d4.items() if k not in extra} == bev   # targets, pairs, regions, sha
    for k in ("imputation_targets", "correction_pairs", "replicate_pairs", "regions",
              "fit_regions", "tables_sha256"):
        assert d4[k] == bev[k], k
    assert E.check(out4) == []


def test_superset_holds_the_base_pair_regions(fx):
    d = fx["dir"]
    der = d / "evalset_p4.json"
    if not der.exists():
        E.derive(fx["base"], der, pair_regions=4)
    out = d / "evalset_r32_p4.json"
    assert E.main(["derive", "--base", str(fx["sup"]), "--pair-regions-nested-in", str(der),
                   "--out", str(out)]) == 0
    dv, sv = json.loads(der.read_text()), json.loads(out.read_text())
    sreg = [tuple(b) for b in sv["regions"]]
    base_blocks = {tuple(dv["regions"][i]) for i in dv["pair_regions"]}
    sup_blocks = [sreg[i] for i in sv["pair_regions"]]
    assert base_blocks <= set(sup_blocks)
    old = {tuple(b) for b in dv["regions"]}
    others = [b for b in sreg if b not in old]                  # 24 new regions, step 8/4 = 2
    assert len(others) == 24
    assert set(sup_blocks) - base_blocks == set(others[::2])
    assert len(sv["pair_regions"]) == 4 + 12 and sv["pair_regions"] == sorted(sv["pair_regions"])
    assert sv["pair_regions_nested_in"]["sha256"] == sha(der)
    assert sv["derived_from"]["sha256"] == sha(fx["sup"]) and sv["pair_regions_p"] == 4
    assert E.check(out) == []


def test_refusals(fx, tmp_path):
    base, d = fx["base"], fx["dir"]
    der = d / "evalset_p4.json"
    if not der.exists():
        E.derive(base, der, pair_regions=4)
    with pytest.raises(ValueError, match="_p4.json"):
        E.derive(base, d / "evalset_x.json", pair_regions=4)
    with pytest.raises(ValueError, match="_p4.json"):
        E.derive(base, d / "evalset_p8.json", pair_regions=4)
    with pytest.raises(ValueError, match="already has pair_regions"):
        E.derive(der, d / "evalset_p4_p2.json", pair_regions=2)
    with pytest.raises(ValueError, match="divide"):
        E.derive(base, d / "evalset_p3.json", pair_regions=3)
    with pytest.raises(ValueError, match="blind"):
        E.derive(base, d / "evalset_blind_p4.json", pair_regions=4)
    with pytest.raises(ValueError, match="exactly one"):
        E.derive(base, d / "evalset_p4.json", pair_regions=4, nested_in=der)
    with pytest.raises(ValueError, match="not a derived"):
        E.derive(fx["sup"], d / "evalset_r32_p4.json", nested_in=base)
    with pytest.raises(ValueError, match="not a superset"):    # the base is no superset of sup
        sup_der = d / "evalset_r32_px_p4.json"
        E.derive(fx["sup"], sup_der, pair_regions=4)
        E.derive(base, d / "evalset_q_p4.json", nested_in=sup_der)
    with pytest.raises(SystemExit):                             # the CLI needs one of the two
        E.main(["derive", "--base", str(base), "--out", str(d / "evalset_p4.json")])
    # a blind base derives only into a blind name
    bl = d / "evalset_blind.json"
    E.build(fx["tables"], bl, n_regions=8, n_fit_regions=2, region_bins=RB, include_blind=True)
    with pytest.raises(ValueError, match="blind"):
        E.derive(bl, d / "evalset_p4b_p4.json", pair_regions=4)
    assert E.derive(bl, d / "evalset_blind_p4.json", pair_regions=4)["pair_regions"] == [0, 2, 4, 6]


def test_check_flags_bad_pair_regions(fx, tmp_path):
    der = fx["dir"] / "evalset_p4.json"
    if not der.exists():
        E.derive(fx["base"], der, pair_regions=4)
    ev = json.loads(der.read_text())
    for bad, msg in (([4, 2], "strictly increasing"), ([0, 99], "out of range"),
                     ("x", "non-empty list")):
        p = fx["dir"] / "evalset_bad_p4.json"
        p.write_text(json.dumps(dict(ev, pair_regions=bad)))
        assert any(msg in b for b in E.check(p)), msg
    p.write_text(json.dumps(dict(ev, derived_from=dict(ev["derived_from"], sha256="0" * 64))))
    assert any("sha256 differs" in b for b in E.check(p))
