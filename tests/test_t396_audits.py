"""t396 C5: the table audits (`tools/t396/audits.py`) on small synthetic tables and stores."""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import sys
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import t396.audits as au  # noqa: E402

MCOLS = ["measurement_id", "experiment", "assay", "in_eic", "layer", "arm_knob", "arm_value",
         "parent_measurement_id", "depth_reads", "depth_fragments", "run_type", "read_length",
         "pipeline", "dedup", "mapq_threshold", "counts_path"]


def _tsv(path: Path, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _meas(mid, run_type, reads, frags, in_eic="train", parent="", counts=None):
    return {"measurement_id": mid, "experiment": mid.split(".")[0], "assay": "H3K4me3",
            "in_eic": in_eic, "layer": "uniform", "parent_measurement_id": parent,
            "depth_reads": str(reads), "depth_fragments": str(frags), "run_type": run_type,
            "read_length": "36", "pipeline": "chip-seq-pipeline2",
            "counts_path": f"h5:{counts}" if counts else ""}


def _store(root: Path, mid: str, lpf: int, plus: int, minus: int, n_lines: int, pr1_file=None):
    d = root / mid
    d.mkdir(parents=True)
    (d / "starts.h5").write_bytes(b"x")
    s = {"n_lines": n_lines, "n_main": plus + minus, "n_reads_strand": [plus, minus]}
    (d / "provenance.json").write_text(json.dumps({
        "lines_per_fragment": lpf,
        "read_files": {"pr1": [pr1_file] if pr1_file else []},
        "outputs": {"starts.h5": s, "starts_pr1.h5": dict(s, n_lines=n_lines // 2)}}))


def _counts(root: Path, mid: str, chr1: np.ndarray, halves=True) -> Path:
    d = root / mid
    d.mkdir(parents=True)
    with h5py.File(d / "counts.h5", "w") as f:
        f["chr1"] = chr1.astype(np.uint32)
    if halves:
        (d / "pr1.h5").write_bytes(b"x")
        (d / "pr2.h5").write_bytes(b"x")
    return d / "counts.h5"


@pytest.fixture
def world(tmp_path):
    t, rs, cs = tmp_path / "tables", tmp_path / "rs", tmp_path / "counts"
    t.mkdir()
    rows = [
        _meas("E1.pooled.uniform.base", "paired-ended", 200, 100,
              counts=_counts(cs, "E1.pooled.uniform.base", np.array([3, 4, 5]))),
        _meas("E2.pooled.uniform.base", "paired-ended", 80, 80,
              counts=_counts(cs, "E2.pooled.uniform.base", np.array([1, 1]))),
        _meas("E3.pooled.uniform.base", "single-ended", 50, 50,
              counts=_counts(cs, "E3.pooled.uniform.base", np.array([2]), halves=False)),
        _meas("E4.pooled.uniform.base", "mixed", 60, 30),
        _meas("E5.pooled.testbed.base", "single-ended", 10, 10, in_eic="blind"),
        _meas("E1.pooled.uniform.se", "single-ended", 100, 100, parent="E1.pooled.uniform.base",
              counts=_counts(cs, "E1.pooled.uniform.se", np.array([1]))),
        _meas("E1.pooled.uniform.nodedup", "paired-ended", 300, 150,
              parent="E1.pooled.uniform.base",
              counts=_counts(cs, "E1.pooled.uniform.nodedup", np.array([9]))),
    ]
    _tsv(t / "measurements.tsv", MCOLS, rows)
    _store(rs, "E1.pooled.uniform.base", 2, 100, 100, 200)
    _store(rs, "E2.pooled.uniform.base", 1, 41, 39, 80)
    _store(rs, "E3.pooled.uniform.base", 1, 30, 20, 50)
    _store(rs, "E4.pooled.uniform.base", 1, 30, 30, 60)       # store disagrees with table ratio 2
    _store(rs, "E1.pooled.uniform.se", 1, 50, 50, 100)
    pcols = ["pair_id", "a", "b", "rung", "assay", "cell", "knobs_differing", "subset_pair",
             "split", "never_trained"]
    pairs = [
        {"pair_id": "p1", "a": "E1.pooled.uniform.base", "b": "E1.pooled.uniform.se",
         "rung": "designed", "knobs_differing": "se", "subset_pair": "0", "split": "val",
         "never_trained": "0"},
        {"pair_id": "p2", "a": "E1.pooled.uniform.base", "b": "E1.pooled.uniform.nodedup",
         "rung": "designed", "knobs_differing": "mapq+nodedup", "subset_pair": "0",
         "split": "val", "never_trained": "1"},
        {"pair_id": "p3", "a": "E1.pooled.uniform.base", "b": "E1.pooled.uniform.nodedup",
         "rung": "designed", "knobs_differing": "nodedup", "subset_pair": "0", "split": "train",
         "never_trained": "0"},
        {"pair_id": "p4", "a": "E3.pooled.uniform.base", "b": "E1.pooled.uniform.se",
         "rung": "designed", "knobs_differing": "se", "subset_pair": "0", "split": "val",
         "never_trained": "0"},
        {"pair_id": "p5", "a": "E1.pooled.uniform.base", "b": "E2.pooled.uniform.base",
         "rung": "designed", "knobs_differing": "depth", "subset_pair": "1", "split": "val",
         "never_trained": "0"},
    ]
    _tsv(t / "pairs.tsv", pcols, pairs)
    ev = {"correction_pairs": [
        {"source": "E1.pooled.uniform.base", "target": "E1.pooled.uniform.nodedup",
         "pair_id": "p2", "knob": "mapq+nodedup", "never_trained": 1, "subset_pair": 0,
         "split": "val"}]}
    return t, rs, ev


def test_log2_depth_is_the_encoder_column():
    assert au.log2_depth({"depth_reads": "1024"}) == pytest.approx(10.0)
    assert au.log2_depth({"depth_reads": "0"}) == pytest.approx(0.0)     # clamped at 1 read


def test_pe_depth_rows_and_summary(world):
    t, rs, _ = world
    rows = {r["measurement_id"]: r for r in au.pe_depth_rows(t, rs)}
    e1 = rows["E1.pooled.uniform.base"]
    assert e1["reads_per_fragment_table"] == 2.0
    assert e1["lines_per_fragment_store"] == 2
    assert e1["strand_balance"] == 0.0
    assert e1["chr1_overlap_total"] == 12
    assert e1["chr1_overlap_over_main_lines"] == pytest.approx(12 / 200)
    assert e1["log2_depth"] == pytest.approx(math.log2(200))
    assert e1["log2_fragments"] == pytest.approx(math.log2(100))
    assert rows["E2.pooled.uniform.base"]["strand_balance"] == pytest.approx(2 / 80)
    blind = rows["E5.pooled.testbed.base"]
    assert "in_start_store" not in blind and "chr1_overlap_total" not in blind
    assert rows["E4.pooled.uniform.base"]["in_count_store"] == 0

    summ = {s["run_type"]: s for s in au.pe_depth_summary(list(rows.values()))}
    pe = summ["paired-ended"]
    assert (pe["n_rows"], pe["n_ratio_1"], pe["n_ratio_2"]) == (3, 1, 2)
    assert pe["n_in_start_store"] == 2 and pe["n_store_lpf_2"] == 1
    assert pe["n_store_lines_eq_depth_reads"] == 2 and pe["n_strand_balance_0"] == 1
    se = summ["single-ended"]
    assert (se["n_rows"], se["n_rows_not_blind"], se["n_ratio_1"]) == (3, 2, 3)

    reasons = {r["measurement_id"]: r["reason"] for r in au.listed(list(rows.values()))}
    assert reasons["E2.pooled.uniform.base"] == "pe_ratio_1"
    assert reasons["E4.pooled.uniform.base"] == "mixed_ratio_2+store_lpf_ne_table_ratio"
    assert "E3.pooled.uniform.base" not in reasons


def test_pe_depth_cli_writes_files(world, tmp_path):
    t, rs, _ = world
    out = tmp_path / "out"
    au.main(["pe-depth", "--tables", str(t), "--rs-store", str(rs), "--out", str(out)])
    for f in ("rows.tsv", "summary.tsv", "listed_rows.tsv"):
        assert (out / f).stat().st_size > 0
    head = (out / "rows.tsv").read_text().splitlines()[0].split("\t")
    assert head == au.ROW_COLS


def test_count_mate_pairs():
    bedpe = ["chr1\t1\t37\tN\t1000\t+", "chr1\t90\t126\tN\t1000\t-",
             "chr1\t5\t41\tN\t1000\t+", "chr1\t50\t86\tN\t1000\t-"]
    assert au.count_mate_pairs(bedpe) == (4, 2)
    se = ["chr1\t1\t37\tN\t1000\t+", "chr1\t90\t126\tN\t1000\t+",      # same strand
          "chr1\t5\t41\tN\t1000\t+", "chr2\t50\t86\tN\t1000\t-",       # other chrom
          "chr1\t5\t41\tN\t1000\t+", "chr1\t5000\t5036\tN\t1000\t-",   # too far
          "chr1\t5\t41\tN\t1000\t+"]
    assert au.count_mate_pairs(se) == (7, 0)


def test_stream_counts_from_tar(tmp_path):
    lines = "".join(f"chr1\t{i}\t{i + 36}\tN\t1000\t{'+-'[i % 2]}\n" for i in range(11))
    gz = gzip.compress(lines.encode())
    tar = tmp_path / "t.tar"
    with tarfile.open(tar, "w") as tf:
        ti = tarfile.TarInfo("onion/work/E1/pr1.tagAlign.gz")
        ti.size = len(gz)
        tf.addfile(ti, io.BytesIO(gz))
    assert au._stream_counts(tar, "onion/work/E1/pr1.tagAlign.gz", str(tmp_path)) == (11, 5)


def test_member_index_and_pick(world, tmp_path):
    t, rs, _ = world
    m = tmp_path / "members"
    m.mkdir()
    (m / "members_onion-work__000.txt").write_text("onion/work/E1/pr1.tagAlign.gz\n")
    idx = au.member_index(m)
    assert idx == {"onion/work/E1/pr1.tagAlign.gz": "onion-work__000.tar"}
    pj = rs / "E1.pooled.uniform.base" / "provenance.json"
    p = json.loads(pj.read_text())
    p["read_files"]["pr1"] = ["/scratch/x/rs392/work/onion/work/E1/pr1.tagAlign.gz"]
    pj.write_text(json.dumps(p))
    rows = [dict(r, reads_per_fragment_table=float(r["depth_reads"]) / float(r["depth_fragments"]))
            for r in au.read_tsv(t / "measurements.tsv")]
    s = au.pick_samples(rows, rs, idx, per_class=2)
    assert [x["measurement_id"] for x in s] == ["E1.pooled.uniform.base"]
    assert s[0]["tar"] == "onion-work__000.tar" and s[0]["n_lines_store_pr1"] == 100


def test_pair_census(world, tmp_path):
    t, rs, ev = world
    rows = au.census_rows(t, ev, rs)
    by = {(r["source_file"], r["pair_id"]): r for r in rows}
    p1 = by[("pairs.tsv", "p1")]
    assert (p1["se"], p1["nodedup"], p1["halves_present"]) == (1, 0, 1)
    assert p1["both_in_start_store"] == 1 and p1["parents_in_both_stores"] == 1
    assert p1["widened_candidate"] == 1 and p1["headline_now"] == 0
    p2 = by[("pairs.tsv", "p2")]
    assert (p2["nodedup"], p2["mapq"], p2["headline_now"], p2["in_evalset"]) == (1, 1, 1, 1)
    assert p2["both_in_start_store"] == 0 and p2["widened_candidate"] == 1
    assert by[("pairs.tsv", "p3")]["widened_candidate"] == 0               # train split
    assert by[("pairs.tsv", "p4")]["widened_candidate"] == 0               # E3 has no halves
    assert by[("pairs.tsv", "p5")]["widened_candidate"] == 0               # depth only
    assert by[("evalset", "p2")]["headline_now"] == 1

    bc = au.by_covariate(rows)
    se_val = [r for r in bc if r["source_file"] == "pairs.tsv" and r["covariate"] == "se"]
    assert sum(r["n_pairs"] for r in se_val) == 2
    assert sum(r["n_widened_candidate"] for r in se_val) == 1

    out = tmp_path / "pc"
    evj = tmp_path / "ev.json"
    evj.write_text(json.dumps(ev))
    au.main(["pair-census", "--tables", str(t), "--evalset", str(evj), "--rs-store", str(rs),
             "--out", str(out)])
    census = au.read_tsv(out / "census.tsv")
    assert {r["pair_id"] for r in census} == {"p1", "p2", "p3", "p4"}
    assert {(r["source_file"], r["pair_id"]) for r in au.read_tsv(out / "candidates.tsv")} == {
        ("pairs.tsv", "p1"), ("pairs.tsv", "p2"), ("evalset", "p2")}
