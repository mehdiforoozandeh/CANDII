"""`tools/t112/arms.py` — the pinned t112 arm table and the task TSVs it expands into.

Every other t112 chunk builds from these rows, so the expected values below are typed in from the
plan's tables, not derived from the module: a wrong constant must fail here, not in a Nibi job.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "t112" / "arms.py"


def _load():
    spec = importlib.util.spec_from_file_location("t112_arms", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t112_arms"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def arms():
    return _load()


def _cli(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True)


def _parse(tsv: str) -> tuple[list[str], list[dict]]:
    lines = tsv.rstrip("\n").split("\n")
    header = lines[0].split("\t")
    return header, [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


CHIP = ("C19M16", "C40M17", "C40M18", "C07M20", "C19M22", "C07M29")


# --- constants ---------------------------------------------------------------------------------

def test_tracks_pinned(arms):
    exp = {  # cell, assay, treat acc, treat reads, ctl acc, fraglen, read_length, subsample
        "C19M16": ("C19", "H3K27ac", "ENCFF254LWX", 176237758, "ENCFF433TZR", 180, 101, 30000000),
        "C40M17": ("C40", "H3K27me3", "ENCFF581NOL", 207716224, "ENCFF337JNL", 200, 101, 30000000),
        "C40M18": ("C40", "H3K36me3", "ENCFF443KLR", 239733719, "ENCFF337JNL", 210, 101, 30000000),
        "C07M20": ("C07", "H3K4me1", "ENCFF458MVX", 205681163, "ENCFF164WQW", 215, 101, 30000000),
        "C19M22": ("C19", "H3K4me3", "ENCFF748TKZ", 174409214, "ENCFF433TZR", 205, 101, 30000000),
        "C07M29": ("C07", "H3K9me3", "ENCFF777XCR", 217273310, "ENCFF164WQW", 200, 101, 30000000),
        "C12M02": ("C12", "DNase-seq", "ENCFF211XVI", 134815660, None, 150, 76, 50000000),
    }
    assert list(arms.TRACKS) == list(exp)
    for track, e in exp.items():
        t = arms.TRACKS[track]
        got = (t["cell"], t["assay"], t["treat_acc"], t["treat_reads"], t["ctl_acc"], t["fraglen"],
               t["read_length"], t["subsample"])
        assert got == e, track
        assert t["run_type"] == "single-ended"
    eic = "/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline"
    assert arms.TRACKS["C19M16"]["treat_bam"] == \
        f"{eic}/results/C19M16/filter_shard0_ENCFF254LWX.merged.srt.nodup.bam"
    assert arms.TRACKS["C12M02"]["treat_bam"] == \
        f"{eic}/results/C12M02/filter_shard0_ENCFF211XVI.merged.srt.nodup.no_chrM_MT.bam"
    assert arms.TRACKS["C40M18"]["pval_bigwig"] == (
        f"{eic}/results/C40M18/macs2_signal_track_shard0_ENCFF443KLR.merged.srt.nodup.30M_x_"
        "ENCFF337JNL.merged.srt.nodup.pval.signal.bigwig")
    assert arms.TRACKS["C12M02"]["pval_bigwig"] == (
        f"{eic}/results/C12M02/macs2_signal_track_shard0_ENCFF211XVI.merged.srt.nodup.no_chrM_MT"
        ".50M.pval.signal.bigwig")
    # D1 is not decided here
    assert arms.TRACKS["C12M02"]["pipeline"] is None


def test_controls_pinned(arms):
    eic = "/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline"
    assert arms.CONTROLS == {
        "ENCFF433TZR": {"cell": "C19", "reads": 107039349, "read_length": 101,
                        "run_type": "single-ended",
                        "bam": f"{eic}/results/C19M16/filter_ctl_shard0_ENCFF433TZR.merged.srt.nodup.bam"},
        "ENCFF337JNL": {"cell": "C40", "reads": 150302589, "read_length": 101,
                        "run_type": "single-ended",
                        "bam": f"{eic}/results/C40M17/filter_ctl_shard0_ENCFF337JNL.merged.srt.nodup.bam"},
        "ENCFF164WQW": {"cell": "C07", "reads": 160028104, "read_length": 101,
                        "run_type": "single-ended",
                        "bam": f"{eic}/results/C07M20/filter_ctl_shard0_ENCFF164WQW.merged.srt.nodup.bam"},
    }
    for track in CHIP:  # every track's control is one of its own cell's
        t = arms.TRACKS[track]
        assert arms.CONTROLS[t["ctl_acc"]]["cell"] == t["cell"]


def test_ids_and_rounding(arms):
    assert arms.pid("C19M16", "depth", "7.5M") == "C19M16__depth__7.5M"
    assert arms.biosample("C19", "depth", "7.5M") == "CF_C19__depth__7.5M"
    assert arms.round_half_up(102.5) == 103
    assert arms.round_half_up(53519674.5) == 53519675
    assert arms.round_half_up(26759837.25) == 26759837
    assert arms.round_half_up(7.49) == 7
    assert arms.abproxy_n_ctl(0.5) == 15000000
    assert arms.abproxy_n_ctl(0.9) == 27000000


# --- rows --------------------------------------------------------------------------------------

def _by_pid(arms, **kw):
    return {r["pid"]: r for r in arms.rows("all", **kw)}


def test_counts_pinned(arms):
    def n(route, dnase, ratio, tracks):
        return len(arms.rows(route, dnase=dnase, ratio=ratio, tracks=tracks))
    assert n("bam", "none", "yes", None) == 84
    assert n("bam", "none", "no", None) == 72
    assert n("bam", "atac", "yes", ["C12M02"]) == 6
    assert n("bam", "atac", "no", ["C12M02"]) == 6
    assert n("fastq", "none", "yes", None) == 36
    assert n("fastq", "atac", "yes", ["C12M02"]) == 4
    assert n("all", "atac", "yes", None) == 130
    assert n("all", "atac", "no", None) == 118
    assert n("all", "none", "yes", None) == 120
    assert n("all", "none", "no", None) == 108
    assert n("all", "atac", "no", ["C12M02"]) == 10
    assert n("all", "none", "yes", ["C12M02"]) == 0


def test_dnase_none_drops_c12m02(arms):
    assert all(r["track"] != "C12M02" for r in arms.rows("all", dnase="none", ratio="yes"))


def test_ratio_no_drops_only_ratio(arms):
    yes = arms.rows("all", dnase="atac", ratio="yes")
    no = arms.rows("all", dnase="atac", ratio="no")
    assert [r for r in yes if r["arm"] != "ratio"] == no
    assert Counter(r["arm"] for r in yes)["ratio"] == 12


def test_per_track_arm_levels(arms):
    rs = arms.rows("all", dnase="atac", ratio="yes")
    got = {}
    for r in rs:
        got.setdefault(r["track"], []).append((r["arm"], r["level"], r["route"]))
    chip = [("base", "base", "bam"), ("depth", "15M", "bam"), ("depth", "7.5M", "bam"),
            ("depth", "3.75M", "bam"), ("pe", "pe", "fastq"), ("dedup", "off", "fastq"),
            ("abproxy", "f0.5", "bam"), ("abproxy", "f0.9", "bam"), ("crop", "36", "fastq"),
            ("crop", "50", "fastq"), ("mapq", "0", "fastq"), ("mapq", "10", "fastq"),
            ("ratio", "k0.5", "bam"), ("ratio", "k2", "bam"), ("ctlid", "other", "bam"),
            ("ctlid", "none", "bam"), ("ctldepth", "q0.5", "bam"), ("ctldepth", "q0.25", "bam"),
            ("extsize", "k0.5", "bam"), ("extsize", "k2", "bam")]
    dnase = [("base", "base", "bam"), ("depth", "15M", "bam"), ("depth", "7.5M", "bam"),
             ("depth", "3.75M", "bam"), ("pe", "pe", "fastq"), ("dedup", "off", "fastq"),
             ("mapq", "0", "fastq"), ("mapq", "10", "fastq"), ("extsize", "k0.5", "bam"),
             ("extsize", "k2", "bam")]
    assert list(got) == list(CHIP) + ["C12M02"]  # table order
    for track in CHIP:
        assert got[track] == chip, track
    assert got["C12M02"] == dnase


def test_route_filter_keeps_order(arms):
    all_rows = arms.rows("all", dnase="atac", ratio="yes")
    for route in ("bam", "fastq"):
        assert arms.rows(route, dnase="atac", ratio="yes") == [r for r in all_rows if r["route"] == route]


def test_ids_unique_and_formed(arms):
    rs = arms.rows("all", dnase="atac", ratio="yes")
    assert len({r["pid"] for r in rs}) == len(rs)
    assert len({r["biosample"] for r in rs}) == 3 * 20 + 10  # one per (cell, arm, level)
    for r in rs:
        assert r["pid"] == f"{r['track']}__{r['arm']}__{r['level']}"
        assert r["biosample"] == f"CF_{r['cell']}__{r['arm']}__{r['level']}"


def test_chip_knobs_and_values(arms):
    rs = _by_pid(arms, dnase="atac", ratio="yes")
    # knob and value independent of track
    for track in CHIP:
        v = lambda arm, lvl: (rs[f"{track}__{arm}__{lvl}"]["knob"], rs[f"{track}__{arm}__{lvl}"]["knob_value"])
        assert v("base", "base") == ("none", None)
        assert v("depth", "15M") == ("treatment_reads", 15000000)
        assert v("depth", "7.5M") == ("treatment_reads", 7500000)
        assert v("depth", "3.75M") == ("treatment_reads", 3750000)
        assert v("pe", "pe") == ("chip.paired_end", True)
        assert v("dedup", "off") == ("chip.no_dup_removal", True)
        assert v("abproxy", "f0.5") == ("control_read_fraction", 0.5)
        assert v("abproxy", "f0.9") == ("control_read_fraction", 0.9)
        assert v("crop", "36") == ("chip.crop_length", 36)
        assert v("crop", "50") == ("chip.crop_length", 50)
        assert v("mapq", "0") == ("chip.mapq_thresh", 0)
        assert v("mapq", "10") == ("chip.mapq_thresh", 10)
        assert v("ctlid", "none") == ("control_accession", "none")
        assert rs[f"{track}__ctlid__other"]["knob"] == "control_accession"
        assert rs[f"{track}__ratio__k2"]["knob"] == "macs2_ratio"
        assert rs[f"{track}__ctldepth__q0.5"]["knob"] == "ctl_subsample_reads"
        assert rs[f"{track}__extsize__k2"]["knob"] == "fraglen"
        assert rs[f"{track}__base__base"]["pipeline"] == "chip"


def test_ctlid_rotation(arms):
    rs = _by_pid(arms, dnase="none", ratio="yes")
    exp = {"C19M16": "ENCFF337JNL", "C19M22": "ENCFF337JNL", "C40M17": "ENCFF164WQW",
           "C40M18": "ENCFF164WQW", "C07M20": "ENCFF433TZR", "C07M29": "ENCFF433TZR"}
    for track, acc in exp.items():
        assert rs[f"{track}__ctlid__other"]["knob_value"] == acc
        assert acc != arms.TRACKS[track]["ctl_acc"]


def test_ctldepth_values(arms):
    rs = _by_pid(arms, dnase="none", ratio="yes")
    exp = {"ENCFF433TZR": (53519675, 26759837), "ENCFF337JNL": (75151295, 37575647),
           "ENCFF164WQW": (80014052, 40007026)}
    for track in CHIP:
        half, quarter = exp[arms.TRACKS[track]["ctl_acc"]]
        assert rs[f"{track}__ctldepth__q0.5"]["knob_value"] == half
        assert rs[f"{track}__ctldepth__q0.25"]["knob_value"] == quarter


def test_extsize_values(arms):
    rs = _by_pid(arms, dnase="atac", ratio="yes")
    exp = {"C19M16": (90, 360), "C40M17": (100, 400), "C40M18": (105, 420), "C07M20": (108, 430),
           "C19M22": (103, 410), "C07M29": (100, 400)}
    for track, (lo, hi) in exp.items():
        assert rs[f"{track}__extsize__k0.5"]["knob_value"] == lo
        assert rs[f"{track}__extsize__k2"]["knob_value"] == hi
    assert rs["C12M02__extsize__k0.5"]["knob"] == "atac.smooth_win"
    assert rs["C12M02__extsize__k0.5"]["knob_value"] == 75
    assert rs["C12M02__extsize__k2"]["knob_value"] == 300


def test_ratio_values(arms):
    rs = _by_pid(arms, dnase="none", ratio="yes")
    reads = {"ENCFF433TZR": 107039349, "ENCFF337JNL": 150302589, "ENCFF164WQW": 160028104}
    for track in CHIP:
        c = reads[arms.TRACKS[track]["ctl_acc"]]
        for lvl, k in (("k0.5", 0.5), ("k2", 2)):
            got = rs[f"{track}__ratio__{lvl}"]["knob_value"]
            assert isinstance(got, float)
            assert got == float(f"{k * 30000000 / c:.10g}")
            assert abs(got - k * 30000000 / c) < 1e-9
    assert rs["C19M16__ratio__k0.5"]["knob_value"] == 0.1401353814
    assert arms.ratio_value(1, 107039349) == 0.2802707629


def test_dnase_rows(arms):
    rs = [r for r in arms.rows("all", dnase="atac", ratio="yes") if r["track"] == "C12M02"]
    by = {(r["arm"], r["level"]): r for r in rs}
    assert all(r["pipeline"] == "atac" and r["assay"] == "DNase-seq" and r["cell"] == "C12" for r in rs)
    assert by[("base", "base")]["knob_value"] is None
    assert by[("depth", "3.75M")]["knob_value"] == 3750000
    assert (by[("pe", "pe")]["knob"], by[("pe", "pe")]["knob_value"]) == ("atac.paired_end", True)
    assert (by[("dedup", "off")]["knob"], by[("dedup", "off")]["knob_value"]) == ("atac.no_dup_removal", True)
    assert (by[("mapq", "0")]["knob"], by[("mapq", "0")]["knob_value"]) == ("atac.mapq_thresh", 0)
    assert by[("mapq", "10")]["knob_value"] == 10
    assert not {"abproxy", "ratio", "ctlid", "ctldepth", "crop"} & {r["arm"] for r in rs}


def test_bad_switches_refused(arms):
    with pytest.raises(ValueError):
        arms.rows("all", dnase="dnase", ratio="yes")
    with pytest.raises(ValueError):
        arms.rows("all", dnase="atac", ratio="maybe")
    with pytest.raises(ValueError):
        arms.rows("bams", dnase="atac", ratio="yes")
    with pytest.raises(ValueError):
        arms.rows("all", dnase="atac", ratio="yes", tracks=["C99M99"])
    with pytest.raises(TypeError):  # no default answer to D1 / D2
        arms.rows("all")


# --- CLI ---------------------------------------------------------------------------------------

def test_cli_count():
    p = _cli("count", "--dnase", "atac", "--ratio", "yes")
    assert p.returncode == 0 and p.stdout == "130\n"
    p = _cli("count", "--dnase", "none", "--ratio", "no")
    assert p.returncode == 0 and p.stdout == "108\n"


def test_cli_switches_required():
    assert _cli("count", "--ratio", "yes").returncode != 0
    assert _cli("count", "--dnase", "atac").returncode != 0
    assert _cli("rows", "--route", "all", "--dnase", "atac").returncode != 0
    assert _cli("rows", "--route", "all", "--dnase", "encode", "--ratio", "yes").returncode != 0


def test_cli_rows_tsv(arms):
    p = _cli("rows", "--route", "fastq", "--dnase", "none", "--ratio", "yes")
    assert p.returncode == 0
    header, rs = _parse(p.stdout)
    assert header == ["pid", "biosample", "track", "cell", "assay", "arm", "level", "route", "knob",
                      "knob_value", "pipeline"]
    assert len(rs) == 36
    assert [r["pid"] for r in rs] == [r["pid"] for r in arms.rows("fastq", dnase="none", ratio="yes")]
    first = rs[0]
    assert first == {"pid": "C19M16__pe__pe", "biosample": "CF_C19__pe__pe", "track": "C19M16",
                     "cell": "C19", "assay": "H3K27ac", "arm": "pe", "level": "pe", "route": "fastq",
                     "knob": "chip.paired_end", "knob_value": "true", "pipeline": "chip"}


def test_cli_knob_value_is_json(arms):
    p = _cli("rows", "--route", "all", "--dnase", "atac", "--ratio", "yes")
    _, rs = _parse(p.stdout)
    assert len(rs) == 130
    ref = {r["pid"]: r for r in arms.rows("all", dnase="atac", ratio="yes")}
    for r in rs:
        assert json.loads(r["knob_value"]) == ref[r["pid"]]["knob_value"]
    by = {r["pid"]: r["knob_value"] for r in rs}
    assert by["C19M16__base__base"] == "null"
    assert by["C19M16__ctlid__other"] == '"ENCFF337JNL"'
    assert by["C19M16__ctlid__none"] == '"none"'
    assert by["C19M22__extsize__k0.5"] == "103"


def test_cli_tracks_subset():
    p = _cli("rows", "--route", "all", "--dnase", "atac", "--ratio", "no", "--tracks", "C12M02,C19M16")
    assert p.returncode == 0
    _, rs = _parse(p.stdout)
    assert len(rs) == 18 + 10
    assert [r["track"] for r in rs] == ["C19M16"] * 18 + ["C12M02"] * 10  # table order, not arg order
    p = _cli("rows", "--route", "all", "--dnase", "atac", "--ratio", "no", "--tracks", "C99M99")
    assert p.returncode != 0
