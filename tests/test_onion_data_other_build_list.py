"""Tests for the build list of the second experiments (t262, chunk OXb).

Hand-written reviewed lists and catalogs; no network. Covers the row numbering and priorities, the
fixed column values (evalonly, pooled base, no control, no signal), the FASTQ and native routes,
the stop on a file mismatch, the batches, and the five check lines with planted violations.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from onion_data import build_list as B  # noqa: E402
from onion_data import other_build_list as OB  # noqa: E402

FILE_COLS = ("file", "experiment", "library", "file_format", "output_type", "assembly", "run_type",
             "read_length", "platform", "paired_with", "bio_reps", "tech_reps", "file_size",
             "status", "paired_end")


def fq(acc, exp, pe_mate="", end="", bio=1, gb=1.0, status="released"):
    return dict(zip(FILE_COLS, (acc, exp, f"LIB{exp}{bio}", "fastq", "reads", "",
                                "paired-ended" if pe_mate else "single-ended", "50", "HiSeq 2000",
                                pe_mate, str(bio), f"{bio}_1", str(int(gb * 1e9)), status, end)))


def bam(acc, exp, bio=1, gb=0.5):
    return dict(zip(FILE_COLS, (acc, exp, "", "bam", B.NATIVE_OUTPUT, "GRCh38", "", "", "", "",
                                str(bio), "", str(int(gb * 1e9)), "released", "")))


def sel(track, cls, assay, x, files, fb="", fb_files="", source="fastq", aligns="",
        fb_source="fastq", fb_aligns=""):
    return {"eic_track": track, "split": "train", "assay": assay, "assay_class": cls,
            "experiment": x, "fastq_files": files, "source": source, "alignment_files": aligns,
            "fallback": fb, "fallback_fastq_files": fb_files, "fallback_source": fb_source,
            "fallback_alignment_files": fb_aligns}


SELECTION = [
    sel("C02M22", "narrow_histone", "H3K4me3", "ENCSRX1", "ENCFF1A;ENCFF1B",
        fb="ENCSRF1", fb_files="ENCFFF1"),
    sel("C03M02", "accessibility_dnase", "DNase-seq", "ENCSRX2", "ENCFF2A"),
    sel("C27M17", "broad_histone", "H3K27me3", "ENCSRX3", "", source="redacted_bam",
        aligns="ENCFFB31;ENCFFB32", fb="ENCSRF3", fb_files="ENCFFF3"),
    sel("C05M01", "accessibility_atac", "ATAC-seq", "ENCSRX4", "ENCFF4A"),
]

CATALOG = [
    fq("ENCFF1A", "ENCSRX1", pe_mate="ENCFF1B", end="1", gb=2.0),
    fq("ENCFF1B", "ENCSRX1", pe_mate="ENCFF1A", end="2", gb=2.0),
    fq("ENCFF1Z", "ENCSRX1", gb=0.7),                 # not served (not in the reviewed list)
    fq("ENCFF2A", "ENCSRX2", gb=1.5),
    fq("ENCFF2OLD", "ENCSRX2", gb=9.0, status="archived"),
    fq("ENCFF3R", "ENCSRX3", gb=1.0, status="released"),  # restricted: not served
    bam("ENCFFB31", "ENCSRX3", bio=1), bam("ENCFFB32", "ENCSRX3", bio=2),
    fq("ENCFF4A", "ENCSRX4", gb=0.25),
    fq("ENCFFF1", "ENCSRF1", gb=3.0),
    fq("ENCFFF3", "ENCSRF3", gb=1.0),
]

LEDGER = [
    {"eic_track": "C02M22", "role": "train", "assay_class": "narrow_histone",
     "experiment": "ENCSRE1", "published_experiment": "ENCSRE1", "file_experiment": "ENCSRE1",
     "match": "1", "control_experiment": "ENCSRLC", "encode_possible_controls": "ENCSRC1"},
    {"eic_track": "C02CTL", "role": "train", "assay_class": "control", "experiment": "ENCSRC2",
     "published_experiment": "", "file_experiment": "", "match": "1", "control_experiment": "",
     "encode_possible_controls": ""},
]
EIC_ROWS = [{"row": str(i), "experiment": f"ENCSRB{i}", "assay_class": "narrow_histone",
             "control_experiment": "ENCSRBC", "pipeline": B.PIPELINE["narrow_histone"]}
            for i in range(1, 1218)]


@pytest.fixture()
def rows():
    return OB.build(SELECTION, CATALOG)


def by_x(rows):
    return {r["experiment"]: r for r in rows}


def test_numbering_priorities_and_fixed_values(rows):
    assert [(r["row"], r["priority"], r["experiment"]) for r in rows] == [
        ("2001", "5", "ENCSRX1"), ("2002", "5", "ENCSRX2"), ("2003", "5", "ENCSRX3"),
        ("2004", "5", "ENCSRX4"), ("3001", "6", "ENCSRF1"), ("3002", "6", "ENCSRF3")]
    for r in rows:
        assert r["role"] == "evalonly" and r["unit"] == "pooled" and r["arm"] == "base"
        assert r["control_experiment"] == "" and r["keep_bam"] == "0"
        layer = "native" if r["pipeline"] == B.NATIVE_PIPELINE else "uniform"
        assert r["expected_measurements"] == f"{r['experiment']}.pooled.{layer}.base:counts+pr1+pr2"
    x = by_x(rows)
    assert x["ENCSRF1"]["eic_track"] == "C02M22" and x["ENCSRX2"]["eic_track"] == "C03M02"


def test_pipelines_and_files(rows):
    x = by_x(rows)
    assert x["ENCSRX1"]["pipeline"] == B.PIPELINE["narrow_histone"]
    assert x["ENCSRX2"]["pipeline"] == B.PIPELINE["accessibility_dnase"]
    assert x["ENCSRX4"]["pipeline"] == B.PIPELINE["accessibility_atac"]
    assert x["ENCSRX1"]["fastq_files"] == "ENCFF1A;ENCFF1B"     # unserved ENCFF1Z left out
    assert x["ENCSRX1"]["fastq_gb"] == "4.000"
    assert x["ENCSRX2"]["fastq_files"] == "ENCFF2A"             # archived file left out
    assert x["ENCSRX3"]["pipeline"] == B.NATIVE_PIPELINE
    assert x["ENCSRX3"]["fastq_files"] == "ENCFFB31;ENCFFB32"
    assert x["ENCSRX3"]["fastq_gb"] == "1.000"


def test_written_file_has_the_build_list_columns(rows, tmp_path):
    p = tmp_path / "bx.tsv"
    B.write_tsv(p, B.BUILD_LIST_COLUMNS, rows)
    assert p.read_text().split("\n")[0].split("\t") == list(B.BUILD_LIST_COLUMNS)
    back = B.read_tsv(p)
    assert OB.check(back, SELECTION, LEDGER, EIC_ROWS, out=lambda s: None) == 0


def test_build_stops_when_catalog_differs_from_reviewed_list():
    bad = copy.deepcopy(SELECTION)
    bad[0]["fastq_files"] = "ENCFF1A"            # a paired-end mate missing from the list
    with pytest.raises(SystemExit):
        OB.build(bad, CATALOG)
    bad = copy.deepcopy(SELECTION)
    bad[1]["fastq_files"] = "ENCFF2A;ENCFF2NOPE"
    with pytest.raises(SystemExit):
        OB.build(bad, CATALOG)


def test_check_lines_all_zero(rows):
    lines = []
    assert OB.check(rows, SELECTION, LEDGER, EIC_ROWS, out=lines.append) == 0
    assert [l.split(":")[0] for l in lines] == [
        "rows outside the reviewed list", "experiments that are EIC experiments or EIC controls",
        "row numbers shared with the EIC build list", "rows with a role other than evalonly",
        "rows that request signal or name a control"]
    assert all(l.split(":")[1].strip().split()[0] == "0" for l in lines)


def _line(rows, prefix, eic_rows=EIC_ROWS):
    lines = []
    rc = OB.check(rows, SELECTION, LEDGER, eic_rows, out=lines.append)
    hit = [l for l in lines if l.startswith(prefix)]
    assert rc == 1 and len(hit) == 1
    return int(hit[0].split(":")[1].split()[0])


@pytest.mark.parametrize("acc", ["ENCSRE1", "ENCSRC1", "ENCSRC2", "ENCSRLC", "ENCSRB7", "ENCSRBC"])
def test_check_catches_eic_accessions(rows, acc):
    bad = copy.deepcopy(rows)
    bad[0]["experiment"] = acc
    assert _line(bad, "experiments that are EIC") == 1


def test_check_catches_planted_violations(rows):
    bad = copy.deepcopy(rows)
    bad[1]["row"] = "17"
    assert _line(bad, "row numbers shared") == 1
    bad = copy.deepcopy(rows)
    bad[2]["role"] = "train"
    assert _line(bad, "rows with a role other") == 1
    bad = copy.deepcopy(rows)
    bad[0]["expected_measurements"] = "ENCSRX1.pooled.uniform.base:counts+pr1+pr2+signal"
    assert _line(bad, "rows that request signal") == 1
    bad = copy.deepcopy(rows)
    bad[3]["control_experiment"] = "ENCSRC9"
    assert _line(bad, "rows that request signal") == 1
    bad = copy.deepcopy(rows)
    bad.append(dict(bad[0], row="2005", experiment="ENCSRZZ"))
    assert _line(bad, "rows outside the reviewed list") == 1
    bad = copy.deepcopy(rows)
    bad[4]["priority"] = "5"                                   # a fallback submitted as priority 5
    assert _line(bad, "rows outside the reviewed list") == 1
    bad = copy.deepcopy(rows)
    bad[0]["eic_track"] = "C03M02"
    assert _line(bad, "rows outside the reviewed list") == 1
    lines = []
    assert OB.check(rows[1:], SELECTION, LEDGER, EIC_ROWS, out=lines.append) == 1  # a row missing


def test_batches_cap_and_order(rows):
    b = OB.batches(rows)
    assert b == [{"batch": "1", "rows": "2001-2004", "n_rows": "4", "fastq_gb": "6.8"}]
    assert [x["rows"] for x in OB.batches(rows, cap_gb=4.0)] == ["2001", "2002-2004"]
    assert all(float(x["fastq_gb"]) <= 4.0 for x in OB.batches(rows, cap_gb=4.0))


def test_pending_sees_one_submit_per_priority5_row(rows, tmp_path, capsys):
    import io
    err = io.StringIO()
    res = B.pending(rows, tmp_path / "m", tmp_path / "w", "5", err=err)
    assert sorted(n for v in res["limits"].values() for n in v) == [2001, 2002, 2003, 2004]
    assert res["signal"] == []
    assert sum(1 for l in err.getvalue().splitlines() if l.startswith("submit ")) == 4
