"""ERx: the eval-set builder takes its regions from a file, and `superset` extends an eval set so
that the old regions stay a subset. Synthetic tables (tools/minicandi/synth.py)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.dataset import tables_sha256  # noqa: E402
from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (6, "H3K27me3"))
BLIND = ((8, "H3K4me3"),)
RB = 128                                        # 24 candidate blocks on the synthetic genome
#: sha256 of the default CLI build below, with the tmp root and the tables sha256 masked, taken
#: on the base commit 70ccbbd before ERx (the default path must stay byte-identical)
DEFAULT_DIGESTS = {
    "evalset.json": "ac222759714201043ceecee7dead86c13ae3f7ae8a827d5397851ca63f76a1d2",
    "scorable.tsv": "bf1ad1b901c92a5faa46337a891df6380dc6672b35b43ec4c13915bc496c08fe",
    "non_scorable_assays.tsv": "3dffce6efe51cd76b88227af984e33a1b65d7800e83c214aca4cf02b75fdbdfb",
}


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("erx")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    out = root / "eval" / "synth" / "evalset.json"
    assert E.main(["build", "--tables", str(t), "--n-regions", "4", "--n-fit-regions", "2",
                   "--region-bins", str(RB), "--out", str(out)]) == 0
    return root, t, out


def test_default_build_byte_identical_to_base_commit(base):
    root, t, out = base
    for name, want in DEFAULT_DIGESTS.items():
        text = (out.parent / name).read_text(encoding="utf-8")
        text = text.replace(tables_sha256(t), "<TSHA>")
        text = text.replace(str(root.resolve()), "<ROOT>").replace(str(root), "<ROOT>")
        assert hashlib.sha256(text.encode()).hexdigest() == want, name
    ev = json.loads(out.read_text())
    assert "regions_file" not in ev and ev["seed"] == 0


def test_superset_keeps_base_and_adds_in_permutation_order(base, tmp_path):
    root, t, out = base
    ev = json.loads(out.read_text())
    sup = tmp_path / "regions8.json"
    assert E.main(["superset", "--base", str(out), "--n-regions", "8", "--out", str(sup)]) == 0
    res = json.loads(sup.read_text())
    old = set(map(tuple, ev["regions"]))
    assert len(res["regions"]) == 8 and old <= set(map(tuple, res["regions"]))
    assert res["fit_regions"] == ev["fit_regions"] and res["region_bins"] == RB
    assert res["seed"] == 0 and res["regions"] == sorted(res["regions"])
    cands = E.candidate_blocks(root / "genome", RB)
    order = np.random.default_rng(0).permutation(len(cands))
    # the base is order[:4] + fit order[4:6]; the superset adds order[6:10], in that order
    assert old == {tuple(cands[i]) for i in order[:4]}
    assert res["added_regions"] == [cands[i] for i in order[6:10]]
    assert not set(map(tuple, res["regions"])) & set(map(tuple, res["fit_regions"]))
    assert res["superset_of_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    # and an eval set built from it holds the 4 and the same fit regions
    ev8 = E.build(t, tmp_path / "eval" / "synth" / "evalset8.json", regions_file=sup)
    assert ev8["regions"] == res["regions"] and ev8["fit_regions"] == ev["fit_regions"]
    assert ev8["regions_file_sha256"] == hashlib.sha256(sup.read_bytes()).hexdigest()
    assert E.check(tmp_path / "eval" / "synth" / "evalset8.json") == []
    # a superset of the superset-built eval set continues the same order
    res10 = E.superset(tmp_path / "eval" / "synth" / "evalset8.json", 10, tmp_path / "r10.json")
    assert res10["added_regions"] == [cands[i] for i in order[10:12]]


def test_build_from_existing_evalset_reproduces_regions(base, tmp_path):
    root, t, out = base
    ev = json.loads(out.read_text())
    again = tmp_path / "eval" / "synth" / "again.json"
    assert E.main(["build", "--tables", str(t), "--regions-file", str(out),
                   "--out", str(again)]) == 0
    ev2 = json.loads(again.read_text())
    for k in ("regions", "fit_regions", "region_bins", "seed", "n_candidate_regions",
              "imputation_targets", "correction_pairs", "scorable"):
        assert ev2[k] == ev[k], k
    assert ev2["regions_file"] == str(out.resolve())


def test_lists_used_as_they_are(base, tmp_path):
    root, t, out = base
    ev = json.loads(out.read_text())
    rf = tmp_path / "rf.json"
    rf.write_text(json.dumps({"regions": ev["regions"][::-1], "fit_regions": ev["fit_regions"],
                              "region_bins": RB}))
    ev2 = E.build(t, tmp_path / "eval" / "synth" / "x.json", regions_file=rf)
    assert ev2["regions"] == ev["regions"][::-1] and ev2["seed"] is None


def test_non_candidate_block_refused(base, tmp_path):
    root, t, out = base
    ev = json.loads(out.read_text())
    cands = set(map(tuple, E.candidate_blocks(root / "genome", RB)))
    for bad in (["chr1", 1], ["chrY", 0], ["chr99", 0]):            # unaligned, excluded, absent
        assert tuple(bad) not in cands
        rf = tmp_path / "bad.json"
        rf.write_text(json.dumps({"regions": ev["regions"][:3] + [bad],
                                  "fit_regions": ev["fit_regions"], "region_bins": RB}))
        with pytest.raises(ValueError, match="not a candidate block"):
            E.build(t, tmp_path / "eval" / "synth" / "bad_ev.json", regions_file=rf)
    rf.write_text(json.dumps({"regions": ev["regions"], "fit_regions": ev["regions"][:1],
                              "region_bins": RB}))
    with pytest.raises(ValueError, match="listed twice"):
        E.build(t, tmp_path / "eval" / "synth" / "bad_ev.json", regions_file=rf)
    rf.write_text(json.dumps({"regions": ev["regions"], "region_bins": RB}))
    with pytest.raises(ValueError, match="lacks"):
        E.build(t, tmp_path / "eval" / "synth" / "bad_ev.json", regions_file=rf)


@pytest.mark.parametrize("extra", [["--n-regions", "4"], ["--n-fit-regions", "2"], ["--seed", "0"],
                                   ["--regions-bed", "x.bed"]])
def test_refused_option_pairs(base, tmp_path, extra):
    root, t, out = base
    args = ["build", "--tables", str(t), "--regions-file", str(out),
            "--out", str(tmp_path / "eval" / "synth" / "r.json")] + extra
    with pytest.raises(ValueError, match="--regions-file is refused"):
        E.main(args)
    assert not (tmp_path / "eval" / "synth" / "r.json").exists()


def test_region_bins_must_match_file(base, tmp_path):
    root, t, out = base
    with pytest.raises(ValueError, match="region_bins"):
        E.build(t, tmp_path / "eval" / "synth" / "r.json", regions_file=out, region_bins=256)
