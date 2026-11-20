"""SC: eval-set options (--inputs-roles, --pair-splits, --regions-bed, blind names with a tag, the
read-overlap flag of testbed pairs). A default build must be byte-identical to the builder before
SC (the module at the base commit, read from git)."""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

#: the umbrella commit SC was cut from (the code before SC)
BASE = "6d5dac3"
VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (8, "H3K27ac"), (8, "DNase-seq"))
BLIND = ((8, "H3K4me3"),)
KW = dict(n_regions=2, n_fit_regions=1, region_bins=256)


def old_module(name: str, relpath: str, tmp: Path):
    """The module as it was at BASE (git show), imported under `name`; skip without git history."""
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BASE}:{relpath}"], check=True,
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # pragma: no cover - a kit copy without .git
        pytest.skip(f"git history of {BASE} not available: {e}")
    d = tmp / "sc_before"
    d.mkdir(exist_ok=True)
    p = d / f"{name}.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    root = tmp_path_factory.mktemp("evarms")
    return make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)


def _meas(t):
    return {r["measurement_id"]: r for r in tsv(t / "measurements.tsv")}


def test_default_build_is_byte_identical_to_before(tables, tmp_path):
    old = old_module("evalset_before_sc", "tools/onion_eval/evalset.py", tmp_path)
    for blind in (False, True):
        name = "evalset_blind.json" if blind else "evalset.json"
        a, b = tmp_path / "old" / name, tmp_path / "new" / name
        old.build(tables, a, include_blind=blind, seed=3, **KW)
        E.build(tables, b, include_blind=blind, seed=3, **KW)
        assert a.read_bytes() == b.read_bytes()
        for f in ("scorable.tsv", "non_scorable_assays.tsv"):
            assert (a.parent / f).read_bytes() == (b.parent / f).read_bytes()
    # explicit defaults equal no arguments
    c = tmp_path / "explicit" / "evalset.json"
    E.build(tables, c, seed=3, inputs_roles=("train",), pair_splits=None, regions_bed=None, **KW)
    assert c.read_bytes() == (tmp_path / "new" / "evalset.json").read_bytes()


def test_blind_name_with_tag(tables, tmp_path):
    assert E.is_blind_evalset_name("evalset_blind.json")
    assert E.is_blind_evalset_name("evalset_blind_v2.json")
    assert not E.is_blind_evalset_name("evalset_blindv2.json")
    assert not E.is_blind_evalset_name("evalset_blind_.json")
    assert not E.is_blind_evalset_name("evalset.json")
    ev = E.build(tables, tmp_path / "evalset_blind_tv.json", include_blind=True, **KW)
    assert ev["include_blind"] is True
    with pytest.raises(ValueError, match="evalset_blind"):
        E.build(tables, tmp_path / "evalset_blind_tv.json", **KW)
    with pytest.raises(ValueError, match="evalset_blind"):
        E.build(tables, tmp_path / "evalset_blindtv.json", include_blind=True, **KW)


def test_inputs_roles_train_val(tables, tmp_path):
    meas = _meas(tables)
    with pytest.raises(ValueError, match="include-blind"):
        E.build(tables, tmp_path / "evalset.json", inputs_roles=("train", "val"), **KW)
    with pytest.raises(ValueError, match="inputs_roles"):
        E.build(tables, tmp_path / "evalset_blind.json", include_blind=True,
                inputs_roles=("val",), **KW)
    base = E.build(tables, tmp_path / "a" / "evalset_blind.json", include_blind=True, **KW)
    out = tmp_path / "b" / "evalset_blind.json"
    ev = E.build(tables, out, include_blind=True, inputs_roles=("train", "val"), **KW)
    assert ev["inputs_roles"] == ["train", "val"] and "inputs_roles" not in base

    def by(e):
        return {(t["target"], t["regime"]): t["inputs"] for t in e["imputation_targets"]}
    old, new = by(base), by(ev)
    # s8 has no training DNase-seq track: its blind target gains an accessibility-only item
    assert set(new) - set(old) == {(k[0], "accessibility_only") for k in new
                                   if meas[k[0]]["in_eic"] == "blind"}
    blind = [k for k in new if meas[k[0]]["in_eic"] == "blind"]
    assert blind
    for k, inp in new.items():
        if meas[k[0]]["in_eic"] == "val":
            assert inp == old[k]                       # validation targets: training inputs only
    loao = next(k for k in blind if k[1] == "leave_one_assay_out")
    added = set(new[loao]) - set(old[loao])
    assert added and all(meas[m]["in_eic"] == "val" and meas[m]["cell"] == "s8" for m in added)
    assert {meas[m]["assay"] for m in added} == {"H3K27ac", "DNase-seq"}
    acc = next(k for k in blind if k[1] == "accessibility_only")
    assert meas[new[acc][0]]["assay"] == "DNase-seq" and meas[new[acc][0]]["in_eic"] == "val"
    assert E.check(out, allow_blind=True) == []
    # the check still refuses a validation input in an eval set built without the option
    ev2 = json.loads(out.read_text())
    ev2.pop("inputs_roles")
    p2 = tmp_path / "c" / "evalset_blind.json"
    p2.parent.mkdir()
    p2.write_text(json.dumps(ev2))
    assert any("not a training track" in b for b in E.check(p2, tables, allow_blind=True))


def test_pair_splits(tables, tmp_path):
    pairs = tsv(tables / "pairs.tsv")
    # one replicate pair per split so the replicate filter is exercised
    meas = _meas(tables)
    extra = []
    for split in ("train", "val", "blind"):
        r = next(m for m in meas.values() if m["in_eic"] == split and m["arm_knob"] == "")
        extra.append(dict(pairs[0], pair_id=f"rep_{split}", a=r["measurement_id"],
                          b=r["measurement_id"].replace(".base", ".mapq0"), rung="technical",
                          assay=r["assay"], cell=r["cell"], split=split, subset_pair="0",
                          never_trained="0"))
    t2 = tables.parent / "synth_rep"
    if not t2.exists():
        t2.mkdir()
        for f in tables.iterdir():
            (t2 / f.name).write_bytes(f.read_bytes())
        with open(t2 / "pairs.tsv", "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pairs[0]), delimiter="\t", lineterminator="\n")
            w.writerows(extra)
    full = E.build(t2, tmp_path / "a" / "evalset_blind.json", include_blind=True, **KW)
    assert {p["split"] for p in full["correction_pairs"]} == {"val", "blind"}
    assert {"rep_train", "rep_val", "rep_blind"} <= set(full["replicate_pairs"])
    ev = E.build(t2, tmp_path / "b" / "evalset_blind.json", include_blind=True,
                 pair_splits=("train", "val"), **KW)
    assert ev["pair_splits"] == ["train", "val"]
    assert {p["split"] for p in ev["correction_pairs"]} == {"val"}
    assert len(ev["correction_pairs"]) == sum(p["split"] == "val" for p in full["correction_pairs"])
    assert "rep_blind" not in ev["replicate_pairs"]
    assert {"rep_train", "rep_val"} <= set(ev["replicate_pairs"])
    assert any(t["split"] == "blind" for t in ev["imputation_targets"])   # targets untouched
    with pytest.raises(ValueError, match="unknown split"):
        E.build(t2, tmp_path / "c" / "evalset.json", pair_splits=("vali",), **KW)


def _write_bed(p, lines):
    p.write_text("".join(lines), encoding="utf-8")
    return p


def test_regions_bed_inside_and_exclude(tables, tmp_path):
    rb = KW["region_bins"]
    # chr1 bins [0, 1024) and [1024, 1536) as two touching intervals: merged, so a block across
    # the join is inside
    bed = _write_bed(tmp_path / "keep.bed", ["track name=x\n", "#c\n",
                                            f"chr1\t0\t{1024 * 25}\n",
                                            f"chr1\t{1024 * 25}\t{1536 * 25}\n"])
    assert E.read_bed(bed) == {"chr1": [(0, 1536 * 25)]}
    ev = E.build(tables, tmp_path / "in" / "evalset.json", regions_bed=bed, n_regions=3,
                 n_fit_regions=1, region_bins=rb)
    assert ev["regions_bed_mode"] == "inside" and ev["n_candidate_regions_before_bed"] > len(
        ev["regions"]) + len(ev["fit_regions"])
    for c, s in ev["regions"] + ev["fit_regions"]:
        assert c == "chr1" and s + rb <= 1536
    ex = E.build(tables, tmp_path / "out" / "evalset.json", regions_bed=bed, regions_exclude=True,
                 n_regions=3, n_fit_regions=1, region_bins=rb)
    assert ex["regions_bed_mode"] == "exclude"
    for c, s in ex["regions"] + ex["fit_regions"]:
        assert c == "chr2" or s >= 1536
    # a block that touches an interval by one bin is not "outside", and not "inside" either
    one = _write_bed(tmp_path / "one.bed", [f"chr1\t{300 * 25}\t{301 * 25}\n"])
    cands = [["chr1", 256], ["chr1", 512], ["chr2", 0]]
    assert E.filter_blocks(cands, E.read_bed(one), rb) == []
    assert E.filter_blocks(cands, E.read_bed(one), rb, exclude=True) == [["chr1", 512], ["chr2", 0]]
    with pytest.raises(ValueError, match="candidate blocks"):
        E.build(tables, tmp_path / "few" / "evalset.json", regions_bed=one, **KW)
    with pytest.raises(ValueError, match="regions-bed"):
        E.build(tables, tmp_path / "x" / "evalset.json", regions_exclude=True, **KW)


def make_testbed_tables(src: Path, dst: Path) -> Path:
    """A copy of `src` where cell s8's H3K4me3 base and its mapq0 arm are testbed-layer rows (the
    DT import: one experiment's reads, halves split per arm)."""
    if dst.exists():
        return dst
    dst.mkdir(parents=True)
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    rows = tsv(dst / "measurements.tsv")
    for r in rows:
        if r["cell"] == "s8" and r["assay"] == "H3K4me3" and r["arm_knob"] in ("", "mapq"):
            r["layer"] = "testbed"
    with open(dst / "measurements.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return dst


def test_testbed_pairs_flagged_not_read_disjoint(tables, tmp_path):
    t = make_testbed_tables(tables, tables.parent.parent / "tb" / "tables" / "synth")
    (t.parent.parent / "genome").mkdir(exist_ok=True)
    for f in (tables.parent.parent / "genome").iterdir():
        (t.parent.parent / "genome" / f.name).write_bytes(f.read_bytes())
    meas = _meas(t)
    ev = E.build(t, tmp_path / "evalset_blind.json", include_blind=True, **KW)
    flagged = [p for p in ev["correction_pairs"] if p.get("read_disjoint") == 0]
    assert flagged and ev["correction_pairs_not_read_disjoint"] == len(flagged)
    for p in flagged:
        a, b = meas[p["source"]], meas[p["target"]]
        assert a["experiment"] == b["experiment"] and "testbed" in (a["layer"], b["layer"])
        assert "share reads" in p["read_disjoint_reason"]
    others = [p for p in ev["correction_pairs"] if "read_disjoint" not in p]
    assert all("testbed" not in (meas[p["source"]]["layer"], meas[p["target"]]["layer"])
               or meas[p["source"]]["experiment"] != meas[p["target"]]["experiment"] for p in others)
    # a validation eval set of the same tables has no testbed pair, so no new key
    v = E.build(t, tmp_path / "v" / "evalset.json", **KW)
    assert "correction_pairs_not_read_disjoint" not in v
    assert all("read_disjoint" not in p for p in v["correction_pairs"])


def test_cli_passes_the_options(tables, tmp_path, capsys):
    bed = _write_bed(tmp_path / "k.bed", [f"chr1\t0\t{2048 * 25}\n"])
    out = tmp_path / "cli" / "evalset_blind_x.json"
    rc = E.main(["build", "--tables", str(tables), "--n-regions", "2", "--n-fit-regions", "1",
                 "--region-bins", "256", "--include-blind", "--inputs-roles", "train,val",
                 "--pair-splits", "train,val", "--regions-bed", str(bed), "--out", str(out)])
    assert rc == 0
    ev = json.loads(out.read_text())
    assert ev["inputs_roles"] == ["train", "val"] and ev["pair_splits"] == ["train", "val"]
    assert ev["regions_bed_mode"] == "inside"
    assert all(c == "chr1" for c, _ in ev["regions"])
