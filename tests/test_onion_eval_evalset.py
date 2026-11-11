"""E6 eval set builder and scorable set, on synthetic tables (tools/minicandi/synth.py)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi.synth import make_synthetic_tables  # noqa: E402
from onion_eval import evalset as E  # noqa: E402

# H3K4me3 and H3K27ac: 3 validation tracks each from 3 cells; H3K27me3: the planted rare assay
VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (6, "H3K27me3"))
BLIND = ((8, "H3K4me3"),)


def tsv(path):
    return list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("evalset")
    t = make_synthetic_tables(root, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=3, n_fit_regions=2, region_bins=512, seed=0)
    return root, t, out, ev


def test_scorable_set_and_planted_rare_assay(built):
    root, t, out, ev = built
    assert sorted(ev["scorable"]["imputation"]) == ["H3K27ac", "H3K4me3"]
    assert sorted(ev["scorable"]["correction"]) == ["H3K27ac", "H3K4me3"]
    rows = {r["assay"]: r for r in tsv(out.parent / "scorable.tsv")}
    assert tuple(tsv(out.parent / "scorable.tsv")[0].keys()) == E.SCORABLE_COLUMNS
    r = rows["H3K4me3"]
    assert (r["n_train"], r["n_val"], r["n_val_cells"], r["has_floors"]) == ("5", "3", "3", "1")
    rare = rows["H3K27me3"]
    assert rare["imputation_scorable"] == "0" and rare["correction_scorable"] == "0"
    assert "1 validation tracks with halves (need 3)" in rare["reason"]
    non = {r["assay"] for r in tsv(out.parent / "non_scorable_assays.tsv")}
    assert "H3K27me3" in non and "DNase-seq" in non and "H3K4me3" not in non
    # scorable outputs never land in the tables dir (tables_sha256 would move)
    assert not (t / "scorable.tsv").exists()


def test_training_track_threshold(tmp_path):
    # 8 cells: H3K4me3 keeps 3 validation tracks but only 4 training tracks -> not scorable
    t = make_synthetic_tables(tmp_path, n_cells=8, val_tracks=VAL, blind_tracks=((7, "H3K4me3"),))
    ev = E.build(t, tmp_path / "eval" / "synth" / "evalset.json", n_regions=2, n_fit_regions=1,
                 region_bins=512)
    assert ev["scorable"]["imputation"] == ["H3K27ac"]
    row = {r["assay"]: r for r in tsv(tmp_path / "eval" / "synth" / "scorable.tsv")}["H3K4me3"]
    assert "4 training tracks (need 5)" in row["reason"]


def test_regions_aligned_masked_disjoint_seeded(built):
    root, t, out, ev = built
    rb = ev["region_bins"]
    assert len(ev["regions"]) == 3 and len(ev["fit_regions"]) == 2
    assert all(s % rb == 0 for _, s in ev["regions"] + ev["fit_regions"])
    assert not set(map(tuple, ev["regions"])) & set(map(tuple, ev["fit_regions"]))
    with h5py.File(root / "genome" / "mask.h5", "r") as f:
        for c, s in ev["regions"]:
            assert f[c][s:s + rb].mean() >= 0.9
    again = E.build(t, root / "eval" / "again.json", n_regions=3, n_fit_regions=2, region_bins=512)
    assert again["regions"] == ev["regions"] and again["fit_regions"] == ev["fit_regions"]
    other = E.build(t, root / "eval" / "other.json", n_regions=3, n_fit_regions=2, region_bins=512,
                    seed=1)
    assert other["regions"] != ev["regions"] or other["fit_regions"] != ev["fit_regions"]


def test_imputation_targets_and_regimes(built):
    root, t, out, ev = built
    meas = {r["measurement_id"]: r for r in tsv(t / "measurements.tsv")}
    loao = [x for x in ev["imputation_targets"] if x["regime"] == "leave_one_assay_out"]
    acc = [x for x in ev["imputation_targets"] if x["regime"] == "accessibility_only"]
    assert len(loao) == len(VAL) and len(acc) == len(VAL)          # no val accessibility target
    for x in loao:
        tr = meas[x["target"]]
        assert tr["in_eic"] == "val" and tr["arm_knob"] == "" and x["split"] == "val"
        assert x["inputs"] and all(meas[m]["in_eic"] == "train" and meas[m]["arm_knob"] == ""
                                   and meas[m]["cell"] == tr["cell"]
                                   and meas[m]["assay"] != tr["assay"] for m in x["inputs"])
        assert len({meas[m]["assay"] for m in x["inputs"]}) == len(x["inputs"])
    for x in acc:
        assert [meas[m]["assay"] for m in x["inputs"]] == ["DNase-seq"]
    blind = [m for m, r in meas.items() if r["in_eic"] == "blind"]
    assert not {x["target"] for x in ev["imputation_targets"]} & set(blind)
    assert E.check(out) == []


def test_correction_pairs_follow_pairs_tsv(built):
    root, t, out, ev = built
    pairs = {p["pair_id"]: p for p in tsv(t / "pairs.tsv")}
    assert ev["correction_pairs"]
    for c in ev["correction_pairs"]:
        p = pairs[c["pair_id"]]
        assert (c["source"], c["target"]) == (p["a"], p["b"]) and c["split"] == "val"
        assert c["never_trained"] == int(p["never_trained"])
        assert c["subset_pair"] == int(p["subset_pair"])
    nt = [c for c in ev["correction_pairs"] if c["never_trained"] and not c["subset_pair"]]
    assert {c["knob"] for c in nt} <= {"mapq", "depth", "se"} and len(nt) >= 2


def test_pairs_without_halves_dropped_and_blind_flagged(tmp_path):
    t = make_synthetic_tables(tmp_path, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    rows = tsv(t / "measurements.tsv")
    victim = next(r for r in rows if r["in_eic"] == "val" and r["arm_knob"] == "mapq")
    victim["has_pseudoreps"] = "0"
    with open(t / "measurements.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "eval" / "synth" / "evalset_blind.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=512, include_blind=True)
    assert ev["correction_pairs_dropped"] > 0
    assert victim["measurement_id"] not in {m for c in ev["correction_pairs"]
                                            for m in (c["source"], c["target"])}
    bad = E.check(out)
    assert any(b.startswith("blind row") for b in bad)
    assert E.check(out, allow_blind=True) == []
    assert E.main(["check", "--evalset", str(out)]) == 1


def test_cli_build_with_strata(tmp_path, capsys):
    t = make_synthetic_tables(tmp_path, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    sp = tmp_path / "eval" / "strata" / "masks.h5"
    sp.parent.mkdir(parents=True)
    sizes = json.loads((tmp_path / "genome" / "chrom_sizes.json").read_text())
    with h5py.File(sp, "w") as f:
        for name in ("promoter", "background"):
            g = f.create_group(name)
            for c, bp in sizes.items():
                g.create_dataset(c, data=np.random.default_rng(0).random(bp // 25) < 0.5)
    out = tmp_path / "eval" / "synth" / "evalset.json"
    assert E.main(["build", "--tables", str(t), "--strata", str(sp), "--n-regions", "2",
                   "--n-fit-regions", "1", "--region-bins", "512", "--out", str(out)]) == 0
    ev = json.loads(out.read_text())
    assert ev["strata"] == ["background", "promoter"] and ev["strata_path"]
    assert set(ev) >= {"tables", "regions", "imputation_targets", "correction_pairs",
                       "replicate_pairs", "scorable", "strata"}
    assert set(ev["scorable"]) == {"imputation", "correction", "layers"}
    assert E.main(["check", "--evalset", str(out)]) == 0
    assert "imputation-scorable H3K4me3: 3 validation targets" in capsys.readouterr().out


# ------------------------------------------------------------------ review E6 regression tests

def _rewrite(t, rows):
    with open(t / "measurements.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def test_f13_missing_mask_raises(tmp_path):
    """Finding 13: without mask.h5 (or a chromosome in it) every block was accepted silently."""
    t = make_synthetic_tables(tmp_path, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    mask = tmp_path / "genome" / "mask.h5"
    with h5py.File(mask, "a") as f:
        del f["chr2"]
    with pytest.raises(KeyError, match="chr2"):
        E.build(t, tmp_path / "eval" / "synth" / "evalset.json", n_regions=2, n_fit_regions=1,
                region_bins=512)
    mask.unlink()
    with pytest.raises(FileNotFoundError, match="mask"):
        E.build(t, tmp_path / "eval" / "synth" / "evalset.json", n_regions=2, n_fit_regions=1,
                region_bins=512)


def test_mask_applied_and_tables_sha256_recorded(built):
    from minicandi.dataset import tables_sha256
    root, t, out, ev = built
    assert ev["mask_applied"] is True and ev["include_blind"] is False
    assert ev["tables_sha256"] == tables_sha256(t)


def test_f9_tables_change_is_caught(tmp_path):
    t = make_synthetic_tables(tmp_path, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    out = tmp_path / "eval" / "synth" / "evalset.json"
    E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=512)
    assert E.check(out) == []
    with open(t / "pairs.tsv", "a", encoding="utf-8") as fh:
        fh.write("")                                  # bytes unchanged: still passes
    assert E.check(out) == []
    rows = tsv(t / "measurements.tsv")
    rows[0]["lab"] = "elsewhere"
    _rewrite(t, rows)
    assert any("tables_sha256" in b for b in E.check(out))


def test_blind_evalset_path_convention(tmp_path):
    t = make_synthetic_tables(tmp_path, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    d = tmp_path / "eval" / "synth"
    with pytest.raises(ValueError, match="evalset_blind.json"):
        E.build(t, d / "evalset.json", n_regions=2, n_fit_regions=1, region_bins=512,
                include_blind=True)
    with pytest.raises(ValueError, match="evalset_blind.json"):
        E.build(t, d / "evalset_blind.json", n_regions=2, n_fit_regions=1, region_bins=512)
    ev = E.build(t, d / "evalset_blind.json", n_regions=2, n_fit_regions=1, region_bins=512,
                 include_blind=True)
    assert ev["include_blind"] is True
    assert any(x["split"] == "blind" for x in ev["imputation_targets"])


def test_f16_control_is_never_an_imputation_target(tmp_path):
    """Finding 16: a validation chipseq-control track became an imputation target."""
    t = make_synthetic_tables(tmp_path, n_cells=9, val_tracks=VAL + ((7, "chipseq-control"),),
                              blind_tracks=BLIND)
    out = tmp_path / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=512)
    meas = {r["measurement_id"]: r for r in tsv(t / "measurements.tsv")}
    assert all(meas[x["target"]]["assay"] != E.CONTROL for x in ev["imputation_targets"])
    assert [meas[x["target"]]["assay"] for x in ev["control_targets"]] == [E.CONTROL]
    ct = ev["control_targets"][0]
    assert ct["regime"] == "control" and ct["cell"] == "s7" and ct["inputs"]
    assert all(meas[m]["in_eic"] == "train" and meas[m]["assay"] != E.CONTROL for m in ct["inputs"])
    assert E.CONTROL not in ev["scorable"]["imputation"]
    row = {r["assay"]: r for r in tsv(out.parent / "scorable.tsv")}[E.CONTROL]
    assert "not an imputation target" in row["reason"]
    assert E.check(out) == []


def test_f11_targets_without_halves_kept_and_counted(tmp_path):
    """Finding 11: targets were not filtered or flagged on halves (a store: path never has them)."""
    t = make_synthetic_tables(tmp_path, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    rows = tsv(t / "measurements.tsv")
    v = [r for r in rows if r["in_eic"] == "val" and r["arm_knob"] == "" and r["assay"] == "H3K4me3"]
    v[0]["has_pseudoreps"] = "0"
    v[1]["counts_path"] = "store:/x/CANDI_STORE/eic:C02:H3K4me3"
    _rewrite(t, rows)
    ev = E.build(t, tmp_path / "eval" / "synth" / "evalset.json", n_regions=2, n_fit_regions=1,
                 region_bins=512)
    flags = {x["target"]: x["has_halves"] for x in ev["imputation_targets"]}
    assert flags[v[0]["measurement_id"]] is False and flags[v[1]["measurement_id"]] is False
    assert flags[v[2]["measurement_id"]] is True
    assert ev["targets_without_halves"] == sorted([v[0]["measurement_id"], v[1]["measurement_id"]])
    # the scorable rule counts only tracks with halves: H3K4me3 drops to 1 and is not scorable
    assert "H3K4me3" not in ev["scorable"]["imputation"]


def test_chrom_sizes_flat_and_nested_layouts(tmp_path):
    """The cluster's chrom_sizes.json is nested ({"assembly": ..., "chrom_sizes": {...}}); the flat
    reader died with `invalid literal for int() ... 'GRCh38'`. Both layouts give the same blocks."""
    make_synthetic_tables(tmp_path, n_cells=9, val_tracks=VAL, blind_tracks=BLIND)
    gdir = tmp_path / "genome"
    path = gdir / "chrom_sizes.json"
    flat = json.loads(path.read_text())
    assert all(isinstance(v, int) for v in flat.values())
    assert E.read_chrom_sizes(path) == flat
    want = E.candidate_blocks(gdir, 512)
    path.write_text(json.dumps({"assembly": "GRCh38", "chrom_sizes": flat}))
    assert E.read_chrom_sizes(path) == flat
    assert E.candidate_blocks(gdir, 512) == want and want
    path.write_text(json.dumps(["chr1", 100]))
    with pytest.raises(ValueError, match="chrom -> size"):
        E.read_chrom_sizes(path)
