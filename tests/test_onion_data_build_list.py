"""Tests for the EIC build list and the withheld design (t170, chunk D4).

Small synthetic ledgers, catalogs and portal probes; no network. Covers the priorities, the ledger
filter (match = 1, published experiment only), the control rule, the native route for restricted
reads, the pr1/pr2 request for validation rows, and the withheld constraints, with planted
violations that `check` must catch.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import build_list as B  # noqa: E402

# --------------------------------------------------------------------------------- fixtures

LEDGER_COLS = ("eic_track", "role", "cell", "assay", "assay_class", "experiment", "match", "lab",
               "depth", "encode_possible_controls", "in_testbed", "file_experiment")


def L(track, role, cell, assay, cls, exp, ctl="", match="1", testbed="0", file_exp=None,
      lab="labA", depth="1000"):
    return dict(zip(LEDGER_COLS, (track, role, cell, assay, cls, exp, match, lab, depth, ctl,
                                  testbed, file_exp or exp)))


LEDGER = [
    L("C01M22", "train", "A", "H3K4me3", "narrow_histone", "ENCSRTA1", ctl="ENCSRCT1"),
    L("C01M17", "train", "A", "H3K27me3", "broad_histone", "ENCSRTA2", ctl="ENCSRCT1",
      lab="labB"),
    L("C01M20", "train", "A", "H3K4me1", "broad_histone", "ENCSRTA3", ctl="ENCSRCT1",
      file_exp="ENCSRSTORE"),
    L("C02M22", "train", "B", "H3K4me3", "narrow_histone", "ENCSRTB1", ctl="ENCSRCT3"),
    L("C02M17", "train", "B", "H3K27me3", "broad_histone", "ENCSRTB2", ctl="ENCSRCT3"),
    L("C03M22", "val", "C", "H3K4me3", "narrow_histone", "ENCSRVC1", ctl="ENCSRCT1"),
    L("C03M17", "val", "C", "H3K27me3", "broad_histone", "ENCSRVC2", ctl="ENCSRCT1"),
    L("C04M22", "blind", "D", "H3K4me3", "narrow_histone", "ENCSRBD1", ctl="ENCSRCT2",
      testbed="1"),
    # control rows as the D3 ledger writes them (store tracks)
    L("C01CTL", "train", "A", "chipseq-control", "control", "ENCSRCT1"),
    L("C04CTL", "train", "D", "chipseq-control", "control", "ENCSRCT2", match="0"),
    L("C04CTL", "blind", "D", "chipseq-control", "control", "ENCSRCT2"),
]

FILE_COLS = ("file", "experiment", "library", "file_format", "output_type", "assembly", "run_type",
             "read_length", "platform", "paired_with", "bio_reps", "tech_reps", "file_size",
             "status", "paired_end", "url")


def se(acc, exp, rl=36, bio=1, tech=1, gb=1.0, status="released"):
    return dict(zip(FILE_COLS, (acc, exp, f"LIB{exp}{bio}{tech}", "fastq", "reads", "",
                                "single-ended", str(rl), "HiSeq 2000", "", str(bio),
                                f"{bio}_{tech}", str(int(gb * 1e9)), status, "", "")))


def pe(acc, exp, rl=76, bio=1, tech=1, gb=1.0):
    r1 = dict(zip(FILE_COLS, (acc + "1", exp, f"LIB{exp}{bio}{tech}", "fastq", "reads", "",
                              "paired-ended", str(rl), "HiSeq 4000", acc + "2", str(bio),
                              f"{bio}_{tech}", str(int(gb * 1e9)), "released", "1", "")))
    r2 = dict(r1, file=acc + "2", paired_with=acc + "1", paired_end="2")
    return [r1, r2]


def bam(acc, exp, bio=1, gb=0.5):
    return dict(zip(FILE_COLS, (acc, exp, "", "bam", "redacted unfiltered alignments", "GRCh38",
                                "", "", "", "", str(bio), "", str(int(gb * 1e9)), "released",
                                "", "")))


FILES = [
    # TA1: SE 76, two libraries in bio rep 1, library 1 sequenced in two runs
    se("F1", "ENCSRTA1", rl=76, tech=1), se("F2", "ENCSRTA1", rl=76, tech=1),
    se("F3", "ENCSRTA1", rl=76, tech=2),
    *pe("F4", "ENCSRTA2", rl=76),
    se("F5", "ENCSRTA3", rl=36), se("F5b", "ENCSRTA3", rl=36, bio=2),
    se("F6", "ENCSRSTORE", rl=36),
    *pe("F7", "ENCSRTB1", rl=100),
    se("F8", "ENCSRTB2", rl=36),                       # restricted: 403 below
    bam("F8BAM", "ENCSRTB2"),
    se("F9", "ENCSRVC1", rl=36),
    *pe("F10", "ENCSRVC2", rl=76),
    *pe("F11", "ENCSRBD1", rl=100),
    se("F12", "ENCSRCT1", rl=36), se("F13", "ENCSRCT2", rl=36), se("F14", "ENCSRCT3", rl=36),
]
RESTRICTED = {"F8"}


def probe_of(files, restricted=RESTRICTED):
    return {f["file"]: {"file": f["file"], "http": "403" if f["file"] in restricted else "200"}
            for f in files}


@pytest.fixture()
def res():
    return B.build(LEDGER, FILES, probe_of(FILES))


def rows_of(res, **kw):
    return [r for r in res["rows"] if all(r[k] == v for k, v in kw.items())]


def run_check(res, withheld=None, rows=None):
    lines = []
    rc = B.check(rows if rows is not None else res["rows"], LEDGER,
                 res["withheld"] if withheld is None else withheld, out=lines.append)
    return rc, lines


def line(lines, prefix):
    hits = [ln for ln in lines if ln.startswith(prefix)]
    assert len(hits) == 1, (prefix, lines)
    return hits[0]


def count(lines, prefix):
    return int(line(lines, prefix).split(":")[1].split()[0])


# ------------------------------------------------------------------------------- priorities

def test_priorities_and_order(res):
    pri = {(r["experiment"], r["unit"], r["arm"]): r["priority"] for r in res["rows"]}
    assert pri[("ENCSRVC1", "pooled", "base")] == "1"
    assert pri[("ENCSRVC2", "pooled", "base")] == "1"
    assert pri[("ENCSRVC2", "pooled", "se")] == "1"
    assert pri[("ENCSRCT1", "pooled", "base")] == "1"            # controls a validation exp
    assert pri[("ENCSRTA1", "pooled", "base")] == "2"
    assert pri[("ENCSRCT3", "pooled", "base")] == "2"            # controls training only
    assert pri[("ENCSRTB1", "pooled", "se")] == "3"
    assert pri[("ENCSRBD1", "pooled", "base")] == "4"
    assert pri[("ENCSRCT2", "pooled", "base")] == "4"            # controls blind only
    assert pri[("ENCSRTA1", "br1tr1", "base")] == "4"            # per-library unit
    assert pri[("ENCSRTA1", "br1tr1ln2", "base")] == "4"         # per-lane unit
    ps = [int(r["priority"]) for r in res["rows"]]
    assert ps == sorted(ps)
    assert [r["row"] for r in res["rows"]] == [str(i) for i in range(1, len(ps) + 1)]
    # within priority 1, every base row precedes every FASTQ-level arm row
    p1 = rows_of(res, priority="1")
    last_base = max(i for i, r in enumerate(p1) if r["arm"] == "base")
    first_arm = min(i for i, r in enumerate(p1) if r["arm"] != "base")
    assert last_base < first_arm


def test_every_pe_training_experiment_gets_se(res):
    for x in ("ENCSRTA2", "ENCSRTB1"):
        assert rows_of(res, experiment=x, arm="se", priority="3")
    assert not rows_of(res, experiment="ENCSRTA1", arm="se")      # single-end already


def test_crop_arms_only_where_reads_are_longer(res):
    assert rows_of(res, experiment="ENCSRTA1", arm="crop50")
    assert not rows_of(res, experiment="ENCSRTA3", arm="crop36")  # 36 bp reads
    assert not rows_of(res, experiment="ENCSRVC1", arm="crop36")


def test_build_list_columns_and_fastq_gb(res, tmp_path):
    p = tmp_path / "bl.tsv"
    B.write_tsv(p, B.BUILD_LIST_COLUMNS, res["rows"])
    with open(p) as fh:
        header = fh.readline().rstrip("\n").split("\t")
    assert tuple(header) == B.BUILD_LIST_COLUMNS
    r = rows_of(res, experiment="ENCSRTA1", unit="pooled", arm="base")[0]
    assert r["fastq_files"] == "F1;F2;F3" and float(r["fastq_gb"]) == pytest.approx(3.0)
    lane = rows_of(res, experiment="ENCSRTA1", unit="br1tr1ln2")[0]
    assert float(lane["fastq_gb"]) == pytest.approx(1.0)
    assert lane["fastq_files"] == "F1;F2"                         # the library's runs: k indexes them


# ----------------------------------------------------------------------------- ledger filter

def test_ledger_filter_published_experiment_only(res):
    xs = {r["experiment"] for r in res["rows"]}
    assert "ENCSRSTORE" not in xs          # the store file's experiment is never built
    assert "ENCSRTA3" in xs                # the published experiment is
    rc, lines = run_check(res)
    assert count(lines, "experiments outside ledger") == 0
    assert count(lines, "store-file experiments that are not the published experiment") == 0
    assert rc == 0, lines


def test_check_catches_experiment_outside_ledger(res):
    bad = res["rows"] + [dict(res["rows"][0], experiment="ENCSRALIEN", row="999")]
    rc, lines = run_check(res, rows=bad)
    assert count(lines, "experiments outside ledger") == 1 and rc == 1


def test_check_catches_store_file_experiment(res):
    bad = res["rows"] + [dict(res["rows"][0], experiment="ENCSRSTORE", row="999")]
    rc, lines = run_check(res, rows=bad)
    assert count(lines, "store-file experiments that are not the published experiment") == 1
    assert rc == 1


# ------------------------------------------------------------------------------ control rule

def test_control_roles(res):
    role = {r["experiment"]: r["role"] for r in res["rows"]}
    assert role["ENCSRCT1"] == "train"     # controls training and validation experiments
    assert role["ENCSRCT3"] == "train"
    assert role["ENCSRCT2"] == "blind"     # controls a blind experiment only: not training data
    vc = rows_of(res, experiment="ENCSRVC1", unit="pooled", arm="base")[0]
    assert vc["control_experiment"] == "ENCSRCT1"


def test_check_catches_training_control_of_blind_only(res):
    bad = [dict(r, role="train") if r["experiment"] == "ENCSRCT2" else r for r in res["rows"]]
    rc, lines = run_check(res, rows=bad)
    assert count(lines, "training controls that control no training experiment") == 1 and rc == 1


def test_blind_and_testbed_only_in_priority_4(res):
    rc, lines = run_check(res)
    assert count(lines, "blind or testbed experiments outside priority 4 or not blind") == 0
    bad = [dict(r, role="train", priority="2") if r["experiment"] == "ENCSRBD1" else r
           for r in res["rows"]]
    rc, lines = run_check(res, rows=bad)
    assert rc == 1
    assert count(lines, "EIC experiments under another role than the ledger's") >= 1


# ------------------------------------------------------------------------ restricted reads

def test_restricted_reads_take_the_native_route(res):
    r = rows_of(res, experiment="ENCSRTB2", unit="pooled", arm="base")[0]
    assert r["pipeline"] == B.NATIVE_PIPELINE and r["fastq_files"] == "F8BAM"
    assert "ENCSRTB2.pooled.native.base:" in r["expected_measurements"]
    assert not [x for x in rows_of(res, experiment="ENCSRTB2") if x["arm"] != "base"]
    assert {d["file"] for d in res["dropped"]} == {"F8"}


def test_no_fastq_and_no_bam_is_excluded():
    files = [f for f in FILES if f["file"] != "F8BAM"]
    res = B.build(LEDGER, files, probe_of(files))
    assert [e["experiment"] for e in res["excluded"]] == ["ENCSRTB2"]


def test_lost_mate_drops_the_pair():
    files = FILES + pe("F15", "ENCSRTA3", rl=36, bio=1, tech=2)
    res = B.build(LEDGER, files, probe_of(files, RESTRICTED | {"F152"}))
    r = rows_of(res, experiment="ENCSRTA3", unit="pooled", arm="base")[0]
    assert "F151" not in r["fastq_files"].split(";")


# --------------------------------------------------------------------------------- halves

def test_validation_rows_request_pr1_and_pr2(res):
    for r in res["rows"]:
        if r["role"] != "val" or r["unit"] != "pooled":
            continue
        for mid, prods in B.parse_tokens(r["expected_measurements"]):
            assert {"pr1", "pr2"} <= set(prods), mid
    base = rows_of(res, experiment="ENCSRVC2", unit="pooled", arm="base")[0]
    toks = dict(B.parse_tokens(base["expected_measurements"]))
    assert set(B.SIGNAL_PRODUCTS) <= set(toks["ENCSRVC2.pooled.uniform.base"])
    assert set(toks) >= {f"ENCSRVC2.pooled.uniform.{a}" for a in B.ALIGN_ARMS}
    arm = rows_of(res, experiment="ENCSRVC2", arm="se")[0]
    assert B.parse_tokens(arm["expected_measurements"]) == [
        ("ENCSRVC2.pooled.uniform.se", ("counts", "pr1", "pr2"))]


def test_check_catches_validation_view_without_pr2(res):
    bad = []
    for r in res["rows"]:
        if r["experiment"] == "ENCSRVC1" and r["arm"] == "base":
            r = dict(r, expected_measurements=r["expected_measurements"].replace(
                "ENCSRVC1.pooled.uniform.mapq0:counts+pr1+pr2",
                "ENCSRVC1.pooled.uniform.mapq0:counts+pr1"))
        bad.append(r)
    rc, lines = run_check(res, rows=bad)
    assert count(lines, "validation views without both pr1 and pr2") == 1
    # the nodedup>mapq0 pair loses its pr2 target, so the track drops below 2 pairs
    assert count(lines, "validation tracks with < 2 never-trained non-subset pairs") == 1
    assert rc == 1


# ------------------------------------------------------------------------- withheld design

def test_withheld_design_passes(res):
    rc, lines = run_check(res)
    assert count(lines, "validation tracks with < 2 never-trained non-subset pairs") == 0
    assert count(lines, "knobs absent from training") == 0
    assert rc == 0, lines
    # values are the kits' (knob, value) encodings, which minicandi.dataset.Withheld matches
    kv = {(w["source_knob"], w["source_value"], w["target_knob"], w["target_value"])
          for w in res["withheld"]}
    assert ("nodedup", "", "mapq", "0") in kv and ("se", "1", "crop", "50") in kv


def test_withheld_rows_block_in_the_dataset_loader(res):
    from minicandi.dataset import Withheld
    w = Withheld(res["withheld"])
    src = {"arm_knob": "nodedup", "arm_value": "", "assay_class": "narrow_histone"}
    tgt = {"arm_knob": "mapq", "arm_value": "0", "assay_class": "narrow_histone"}
    assert w.blocks(src, tgt)
    assert not w.blocks(tgt | {"arm_knob": "base", "arm_value": ""}, tgt)


def test_never_trained_counts(res):
    v = B.views_of(res["rows"])
    nt1 = B.never_trained_pairs(v["ENCSRVC1"], B.withheld_for(res["withheld"], "narrow_histone"))
    assert sorted(nt1) == [("mapq0", "nodedup"), ("nodedup", "mapq0")]
    nt2 = B.never_trained_pairs(v["ENCSRVC2"], B.withheld_for(res["withheld"], "broad_histone"))
    assert len(nt2) == 6


def test_planted_violation_too_few_pairs(res):
    keep = [w for w in res["withheld"] if not (w["source_knob"] == "mapq")]
    rc, lines = run_check(res, withheld=keep)
    assert count(lines, "validation tracks with < 2 never-trained non-subset pairs") == 1  # VC1
    assert rc == 1


def test_planted_depth_only_withheld_rows_never_count(res):
    depth_only = [{"source_knob": "base", "source_value": "", "target_knob": "depth",
                   "target_value": "0.50", "assay_class": c} for c in B.HISTONE] + \
                 [{"source_knob": "depth", "source_value": "0.25", "target_knob": "base",
                   "target_value": "", "assay_class": c} for c in B.HISTONE]
    keep = [w for w in res["withheld"] if w["source_knob"] != "mapq"] + depth_only
    rc, lines = run_check(res, withheld=keep)
    assert count(lines, "withheld rows that are pure depth pairs") == 4
    assert count(lines, "validation tracks with < 2 never-trained non-subset pairs") == 1
    assert rc == 1


def test_planted_violation_knob_absent_from_training(res):
    # withhold every pair that has se as its source: se is then never a training source
    arms = [a for a in B.ARM_VIEW if a != "se"]
    extra = [{"source_knob": "se", "source_value": "1", "target_knob": B.ARM_VIEW[a][0],
              "target_value": B.ARM_VIEW[a][1], "assay_class": "broad_histone"} for a in arms]
    rc, lines = run_check(res, withheld=res["withheld"] + extra)
    assert count(lines, "knobs absent from training") >= 1
    assert "broad_histone:se(no source)" in line(lines, "knobs absent from training")
    assert rc == 1


def test_validation_arm_never_built_in_training_is_absent(res):
    # a validation crop36 view in a class whose training holds no crop36 view
    rows = [r for r in res["rows"] if not (r["role"] == "train" and r["arm"] == "crop36")]
    rc, lines = run_check(res, rows=rows)
    assert count(lines, "knobs absent from training") >= 1 and rc == 1


def test_subset_pair_rule():
    assert B.is_subset_pair("base", "depth0.50") and B.is_subset_pair("depth0.25", "depth0.125")
    assert not B.is_subset_pair("depth0.50", "nodedup")
    assert not B.is_subset_pair("crop36", "crop50")


def test_cli_build_and_check(tmp_path):
    def w(name, cols, rows):
        B.write_tsv(tmp_path / name, cols, rows)
    w("ledger.tsv", LEDGER_COLS, LEDGER)
    (tmp_path / "cat").mkdir()
    w("cat/files.tsv", FILE_COLS, FILES)
    w("probe.tsv", B.PROBE_COLUMNS,
      [{"file": f["file"], "http": "403" if f["file"] in RESTRICTED else "200"} for f in FILES])
    assert B.main(["build", "--ledger", str(tmp_path / "ledger.tsv"), "--catalog",
                   str(tmp_path / "cat"), "--probe", str(tmp_path / "probe.tsv"), "--out",
                   str(tmp_path / "bl.tsv"), "--withheld-out", str(tmp_path / "wh.tsv"),
                   "--summary-dir", str(tmp_path / "sum")]) == 0
    assert B.main(["check", "--build-list", str(tmp_path / "bl.tsv"), "--ledger",
                   str(tmp_path / "ledger.tsv"), "--withheld", str(tmp_path / "wh.tsv")]) == 0
    with open(tmp_path / "sum" / "priority_summary.tsv") as fh:
        assert [r["priority"] for r in csv.DictReader(fh, delimiter="\t")] == ["1", "2", "3", "4"]


def test_rows_prints_a_priority_as_an_array_spec(tmp_path, capsys):
    """KF (t180): the submit chunk's one command per priority reads the rows from here."""
    from onion_data import build_list as BLM
    rows = [{"row": str(i), "priority": p} for i, p in
            enumerate([1, 1, 1, 2, 2, 1, 4, 4, 4, 4], start=1)]
    assert BLM.array_spec(rows, 1) == "1-3,6"
    assert BLM.array_spec(rows, "4") == "7-10"
    assert BLM.array_spec(rows, 3) == ""
    t = tmp_path / "bl.tsv"
    t.write_text("row\tpriority\n" + "".join(f"{r['row']}\t{r['priority']}\n" for r in rows))
    assert BLM.main(["rows", "--build-list", str(t), "--priority", "2"]) == 0
    assert capsys.readouterr().out.strip() == "4-5"
    with pytest.raises(SystemExit):
        BLM.main(["rows", "--build-list", str(t), "--priority", "3"])


def test_pending_skips_built_not_applicable_and_active_rows_and_splits_long_rows(tmp_path, capsys):
    """KF2 B2, M2, M4, M6: `submit` is safe to run twice; long rows get a longer limit."""
    import json
    from onion_data import build_list as BLM
    out, work = tmp_path / "meas", tmp_path / "work"

    def r(n, exp, unit="pooled", arm="base", files="F1", gb="1.0", pipe="chip-seq-pipeline2 v2.2.2",
          sig=False, pri="1"):
        mid = f"{exp}.{unit}.uniform.{arm}"
        return {"row": str(n), "priority": pri, "experiment": exp, "unit": unit, "arm": arm,
                "fastq_files": files, "fastq_gb": gb, "pipeline": pipe,
                "expected_measurements": f"{mid}:counts+pr1+pr2" + ("+signal" if sig else "")
                + (f";{exp}.pooled.uniform.nodedup:counts+pr1+pr2" if arm == "base" else "")}
    rows = [r(1, "BUILT"), r(2, "HALF"), r(3, "MIX", arm="se"), r(4, "BIG", files="F4;F5",
            sig=True), r(5, "RUNNING"), r(6, "NOCOUNT", files="F6", gb="2.0"), r(7, "P2", pri="2")]
    for mid in ("BUILT.pooled.uniform.base", "BUILT.pooled.uniform.nodedup",
                "HALF.pooled.uniform.base"):
        (out / mid).mkdir(parents=True)
        (out / mid / "provenance.json").write_text("{}")
    (work / "MIX" / "se").mkdir(parents=True)
    (work / "MIX" / "se" / "NOT_APPLICABLE").write_text(json.dumps({"reason": "mixed"}))
    # F4 + F5: 700 M reads of ChIP -> 700 * (0.0332 + 0.0158) + 0.585 = 34.9 h > 18 h
    counts = {"F1": 10_000_000, "F4": 350_000_000, "F5": 350_000_000}
    res = BLM.pending(rows, out, work, priority="1", active=[5], counts=counts)
    assert res["skipped"] == {1: "built", 3: "not_applicable", 5: "active"}
    assert res["limits"] == {"1-00:00:00": [2, 6], "3-00:00:00": [4]}
    assert res["signal"] == [4]
    m, h = BLM.projected_wall_h(rows[3], counts, 10.0)
    assert m == 700 and abs(h - (700 * (4.071 + 1.941) / 122.63 + 0.585)) < 1e-9
    m6, _ = BLM.projected_wall_h(rows[5], counts, 10.0)      # no read count: 2 GB x 10 M/GB
    assert m6 == 20
    err = capsys.readouterr().err
    assert "skip row 1 BUILT" in err and "not applicable" in err and "queued or running" in err
    assert "submit row 4 BIG" in err and "LONG" in err
    # an explicit row list, and the stdout plan the shell reads
    assert BLM.main(["pending", "--build-list", str(_tsv(tmp_path, rows)), "--priority", "all",
                     "--rows", "1-3,7", "--out", str(out), "--work", str(work)]) == 0
    assert capsys.readouterr().out.split("\n")[0] == "1-00:00:00\t2,7"
    # every expected measurement built: nothing to print (build_row.sh exits 0 at once)
    assert BLM.main(["pending", "--build-list", str(_tsv(tmp_path, rows)), "--rows", "1",
                     "--out", str(out), "--work", str(work)]) == 0
    assert capsys.readouterr().out == ""
    assert BLM.parse_spec("3,5-7") == [3, 5, 6, 7] and BLM.spec_of([7, 3, 6, 5]) == "3,5-7"


def _tsv(tmp_path, rows):
    p = tmp_path / "bl.tsv"
    cols = list(rows[0])
    p.write_text("\t".join(cols) + "\n" + "".join("\t".join(r[c] for c in cols) + "\n"
                                                  for r in rows))
    return p


# ------------------------------------------------- D6x: build_row.sh submit override, NOT_APPLICABLE

_BUILD_ROW = _ROOT / "slurm" / "onion_data" / "build_row.sh"


def _d6x_rows():
    def r(n, exp, arm="base", sig=False):
        mid = f"{exp}.pooled.uniform.{arm}"
        return {"row": str(n), "priority": "1", "experiment": exp, "assay_class": "narrow_histone",
                "unit": "pooled", "arm": arm, "fastq_files": "F1", "fastq_gb": "1.0",
                "pipeline": "chip-seq-pipeline2 v2.2.2",
                "expected_measurements": f"{mid}:counts+pr1+pr2" + ("+signal" if sig else "")}
    return [r(56, "DNASE", sig=True), r(102, "MIX", arm="se")]


def _run_build_row(args, env_extra, tmp_path):
    import os
    import subprocess
    env = dict(os.environ, ONION_OUT=str(tmp_path / "meas"), ONION_WORK=str(tmp_path / "work"),
               ONION_FASTQ_ROOT=str(tmp_path / "fq"), **env_extra)
    # output to files, not pipes: the script's 30 s memory sampler outlives it by one sleep
    log = tmp_path / "build_row.log"
    with open(log, "w") as out, open(str(log) + ".err", "w") as err:
        rc = subprocess.run(["bash", str(_BUILD_ROW), *args], env=env, stdout=out, stderr=err,
                            timeout=60).returncode
    return subprocess.CompletedProcess(args, rc, log.read_text(),
                                       Path(str(log) + ".err").read_text())


def test_submit_time_and_memory_override_reach_every_array(tmp_path):
    """D6x class C: ONION_TIME and ONION_MEM replace the projected limit and the header memory
    on the build array and the signal pass; without them nothing changes."""
    bl = _tsv(tmp_path, _d6x_rows())
    kit = _ROOT
    p = _run_build_row(["submit", str(kit), str(bl), "1", "50", "2", "56"],
                       {"ONION_SBATCH_TEST": "1", "ONION_TIME": "3-00:00:00",
                        "ONION_MEM": "128000M"}, tmp_path)
    assert p.returncode == 0, p.stderr
    build = [ln for ln in p.stdout.splitlines() if ln.startswith("rows 56")]
    assert build and "limit 3-00:00:00, mem 128000M" in build[0]
    assert "--time=3-00:00:00 --mem=128000M --array=56%50" in build[0]
    sig = [ln for ln in p.stdout.splitlines() if ln.startswith("signal pass")]
    assert sig and "--time=3-00:00:00 --mem=128000M --array=56%50" in sig[0]
    assert "--job-name=onion_signal" in sig[0]
    # stage array keeps its own resources
    assert "stage_fastq.sh" in build[0] and "--mem=128000M --array=56%2" not in build[0]
    q = _run_build_row(["submit", str(kit), str(bl), "1", "50", "2", "56"],
                       {"ONION_SBATCH_TEST": "1"}, tmp_path)
    assert q.returncode == 0, q.stderr
    line = [ln for ln in q.stdout.splitlines() if ln.startswith("rows 56")][0]
    assert "--time=1-00:00:00 --array=56%50" in line and "--mem" not in line


def test_not_applicable_row_exits_zero_without_verify(tmp_path):
    """D6x class B: a row whose align step writes NOT_APPLICABLE ends with exit 0, skipping the
    views step and verify (verify fails on an experiment with no measurement yet)."""
    import json
    import os
    bl = _tsv(tmp_path, _d6x_rows())
    stub = tmp_path / "bin"
    stub.mkdir()
    rd = tmp_path / "work" / "MIX" / "se"
    # the align step's python writes NOT_APPLICABLE (as fastq_uniform.run does) and returns 0
    (stub / "python").write_text(
        "#!/bin/bash\nmkdir -p %s && echo '%s' > %s/NOT_APPLICABLE\nexit 0\n"
        % (rd, json.dumps({"reason": "arm se: mixed"}), rd))
    (stub / "python").chmod(0o755)
    (tmp_path / "slurmtmp").mkdir()
    p = _run_build_row([str(_ROOT), str(bl), "102"],
                       {"SLURM_TMPDIR": str(tmp_path / "slurmtmp"),
                        "PATH": f"{stub}:{os.environ['PATH']}"}, tmp_path)
    assert "nothing to build, verify skipped" in p.stdout, p.stdout + p.stderr
    assert "=== verify" not in p.stdout and "views step" not in p.stdout
    assert p.stdout.strip().endswith("[onion-build] exit=0") and p.returncode == 0
    # a rerun exits 0 at step 0 (pending treats NOT_APPLICABLE as built)
    q = _run_build_row([str(_ROOT), str(bl), "102"],
                       {"SLURM_TMPDIR": str(tmp_path / "slurmtmp")}, tmp_path)
    assert q.returncode == 0 and "exit 0" in q.stdout
