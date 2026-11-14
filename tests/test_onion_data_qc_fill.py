"""Chunk QCt: the read-derived QC fill tool on tiny synthetic tagAligns (CPU, seconds).

The pipeline SIF is not on the laptop, so the xcor step is stubbed; the NRF/PBC arithmetic (the
pipeline's own awk), the tie of every value to the measurement's reads, and the side table are
what is tested. Measurement dirs are made by `align_views.py build` with the python splitter.
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from onion_data import align_views as AV  # noqa: E402
from onion_data import qc_fill as QF  # noqa: E402

SIZES = {"chr1": 200000, "chrM": 16000}
EXP = "ENCSRTEST"


def _write_ta(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for c, s, e, name, strand in rows:
            f.write(f"{c}\t{s}\t{e}\t{name}\t1000\t{strand}\n")
    return path


def _files(tmp_path, rows, name="a.tagAlign.gz"):
    p = _write_ta(tmp_path / name, rows)
    return [{"path": str(p), "lines": len(rows)}]


def _ref_pbc(keys):
    """Python reference of the pipeline's definitions: NRF = distinct/total, PBC1 = once/distinct,
    PBC2 = once/twice."""
    c = Counter(keys)
    mt, m0 = sum(c.values()), len(c)
    m1 = sum(1 for v in c.values() if v == 1)
    m2 = sum(1 for v in c.values() if v == 2)
    return mt, m0, m1, m2


# --------------------------------------------------------------------------------------------
# NRF / PBC arithmetic

def test_qc_columns_match_contract():
    from minicandi.contract import QC_COLUMNS
    assert QF.QC_COLUMNS == tuple(QC_COLUMNS)


def test_pbc_se_hand_count(tmp_path):
    rows = [("chr1", 100, 136, "N", "+"),                                   # A x1
            ("chr1", 200, 236, "N", "+"), ("chr1", 200, 236, "N", "+"),    # B x2
            ("chr1", 300, 336, "N", "-"), ("chr1", 300, 336, "N", "-"),
            ("chr1", 300, 336, "N", "-"),                                   # C x3
            ("chr1", 300, 336, "N", "+"),                                   # D x1 (other strand)
            ("chrM", 10, 46, "N", "+"), ("chrM", 10, 46, "N", "+")]        # mito: dropped
    b = QF.pbc(_files(tmp_path, rows), 1, tmp_path, sort_mem="16M")
    assert (b["mt"], b["m0"], b["m1"], b["m2"]) == (7, 4, 2, 1)
    assert b["nrf"] == "0.571429" and b["pbc1"] == "0.500000" and b["pbc2"] == "2.000000"


def test_pbc_se_matches_reference_on_random_reads(tmp_path):
    rng = random.Random(3)
    rows = [("chr1", (s := rng.randrange(0, 400)), s + 36, "N", rng.choice("+-"))
            for _ in range(2000)]
    b = QF.pbc(_files(tmp_path, rows), 1, tmp_path, sort_mem="16M")
    mt, m0, m1, m2 = _ref_pbc([(c, s, e, st) for c, s, e, _, st in rows])
    assert (b["mt"], b["m0"], b["m1"], b["m2"]) == (mt, m0, m1, m2)
    assert float(b["nrf"]) == pytest.approx(m0 / mt, abs=1e-6)
    assert float(b["pbc1"]) == pytest.approx(m1 / m0, abs=1e-6)
    assert float(b["pbc2"]) == pytest.approx(m1 / m2, abs=1e-6)


def test_pbc_pe_keys_on_both_mates(tmp_path):
    """PE: the key is (chr1, start1, chr2, end2, strand1, strand2) of the two mate lines."""
    def frag(s1, e2, st1="+", st2="-"):
        return [("chr1", s1, s1 + 50, "N", st1), ("chr1", e2 - 50, e2, "N", st2)]
    rows = (frag(100, 400) + frag(100, 400)          # one fragment, twice
            + frag(100, 450)                         # same mate-1 start, other end: distinct
            + frag(100, 400, "-", "+")               # same span, other strands: distinct
            + frag(900, 1200))                       # distinct
    b = QF.pbc(_files(tmp_path, rows), 2, tmp_path, sort_mem="16M")
    assert (b["mt"], b["m0"], b["m1"], b["m2"]) == (5, 4, 3, 1)
    rng = random.Random(5)
    rows, keys = [], []
    for _ in range(1500):
        s1, ln = rng.randrange(0, 300), rng.randrange(100, 104)
        st1 = rng.choice("+-")
        st2 = "-" if st1 == "+" else "+"
        rows += frag(s1, s1 + ln, st1, st2)
        keys.append(("chr1", s1, "chr1", s1 + ln, st1, st2))
    b = QF.pbc(_files(tmp_path, rows, "pe.tagAlign.gz"), 2, tmp_path, sort_mem="16M")
    assert (b["mt"], b["m0"], b["m1"], b["m2"]) == _ref_pbc(keys)


def test_pbc2_empty_when_no_position_has_two_reads(tmp_path):
    rows = [("chr1", s, s + 36, "N", "+") for s in (1, 2, 3)] + [("chr1", 9, 45, "N", "+")] * 3
    b = QF.pbc(_files(tmp_path, rows), 1, tmp_path, sort_mem="16M")
    assert b["m2"] == 0 and b["pbc2"] == "" and b["nrf"] == "0.666667"


def test_pbc_refuses_odd_pe_line_count(tmp_path):
    with pytest.raises(QF.Refused):
        QF.pbc(_files(tmp_path, [("chr1", 1, 37, "N", "+")] * 3), 2, tmp_path)


def test_parse_cc_qc():
    line = "x.tagAlign.gz\t15000000\t185\t0.21\t35\t0.19\t1500\t0.17\t1.2353\t1.1765\t1\n"
    assert QF.parse_cc_qc(line) == {"frag_len": "185", "nsc": "1.2353", "rsc": "1.1765"}


def test_rows_spec_roundtrip():
    idx = [0, 1, 2, 5, 7, 8, 12]
    assert QF.rows_spec(idx) == "0-2,5,7-8,12"
    assert QF.parse_rows("0-2,5,7-8,12%50") == idx


# --------------------------------------------------------------------------------------------
# measurement dirs and the tie to the reads

def _reads(n, seed, dup_every=4):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        s = rng.randrange(0, 190000)
        rows.append(("chr1", s, s + 36, f"r{i}", rng.choice("+-")))
        if i % dup_every == 0:
            rows.append(("chr1", s, s + 36, f"d{i}", rows[-1][4]))
    return rows


def _build(tmp_path, arms="base,depth0.50"):
    sz = tmp_path / "chrom_sizes.json"
    sz.write_text(json.dumps({"chrom_sizes": SIZES}))
    (tmp_path / "tmp").mkdir(exist_ok=True)
    kept = [r for r in _reads(600, 1) if r[3].startswith("r")]
    src = _write_ta(tmp_path / "src" / "kept.tagAlign.gz", kept)
    argv = ["build", "--experiment", EXP, "--assay", "H3K27ac", "--cell", "C01",
            "--eic-track", "C01M16", "--role", "train", "--arms", arms, "--splitter", "python",
            "--seed", "0", "--work", str(tmp_path / "work"), "--out", str(tmp_path / "out"),
            "--chrom-sizes", str(sz), "--tmp", str(tmp_path / "tmp"), "--read-length", "36",
            "--source", str(src)]
    assert AV.main(argv) == 0
    return tmp_path / "out", tmp_path / "work"


def _fake_nodedup(tmp_path, out, work, parent=None):
    """A nodedup measurement beside the base one: the kept reads plus their duplicates."""
    allr = _reads(600, 1)
    d = work / EXP / "pooled.uniform" / "nodedup"
    halves = {"pr1": allr[0::2], "pr2": allr[1::2]}
    rs = {"arm": "nodedup", "md5": {}}
    for h, rows in halves.items():
        rs[h] = str(_write_ta(d / f"{h}.tagAlign.gz", [(c, s, e, "N", st)
                                                       for c, s, e, _, st in rows]))
        rs["md5"][h] = AV.records.md5_file(rs[h])
    (d / "READSET.json").write_text(json.dumps(rs))
    mid = f"{EXP}.pooled.uniform.nodedup"
    m = out / mid
    m.mkdir()
    (m / "counts.h5").write_bytes(b"counts")
    (m / "meta.json").write_text(json.dumps(
        {"measurement_id": mid, "parent_measurement_id": parent or f"{EXP}.pooled.uniform.base"}))
    (m / "provenance.json").write_text(json.dumps(
        {"measurement_id": mid, "experiment": EXP, "unit": "pooled", "layer": "uniform",
         "arm": "nodedup", "lines_per_fragment": 1, "pipeline": {"name": "chip-seq-pipeline2"},
         "halves": {h: {"tagalign": [rs[h]], "n_lines": len(halves[h])} for h in halves}}))
    return allr


def _row(out, arm):
    return json.loads((out / f"{EXP}.pooled.uniform.{arm}" / "meta.json").read_text())


def _stub_xcor(calls):
    def fn(files, pipeline, paired_end, nth, work, cost):
        calls.append({"lines": sum(f["lines"] for f in files), "pipeline": pipeline,
                      "pe": paired_end, "md5": [f["md5"] for f in files]})
        return {"nsc": "1.2000", "rsc": "1.1000", "frag_len": "180", "command": "stub"}
    return fn


def test_measure_base_reads_nodedup_sibling(tmp_path):
    out, work = _build(tmp_path)
    allr = _fake_nodedup(tmp_path, out, work)
    calls = []
    rec = QF.measure(_row(out, "base"), tmp_path / "w", xcor_fn=_stub_xcor(calls))
    assert rec["status"] == "ok", rec["reason"]
    kept_lines = sum(1 for r in allr if r[3].startswith("r"))
    assert calls == [{"lines": kept_lines, "pipeline": "chip", "pe": False,
                      "md5": [f["md5"] for f in rec["kept_readset"]["files"]]}]
    mt, m0, m1, m2 = _ref_pbc([(c, s, e, st) for c, s, e, _, st in allr])
    assert rec["pbc"]["source_measurement_id"] == f"{EXP}.pooled.uniform.nodedup"
    assert (rec["pbc"]["mt"], rec["pbc"]["m0"], rec["pbc"]["m1"], rec["pbc"]["m2"]) \
        == (mt, m0, m1, m2)
    assert mt > m0      # the duplicates were seen
    v = rec["values"]
    assert v["nsc"] == "1.2000" and v["frag_len"] == "180" and v["frip"] == ""
    assert float(v["nrf"]) == pytest.approx(m0 / mt, abs=1e-6)
    assert rec["counts_sha256"] == _row(out, "base")["sha256"]


def test_measure_without_predup_reads_leaves_pbc_empty(tmp_path):
    out, work = _build(tmp_path)
    calls = []
    base = QF.measure(_row(out, "base"), tmp_path / "w", xcor_fn=_stub_xcor(calls))
    assert base["status"] == "ok" and base["pbc"] is None and "no nodedup" in base["pbc_reason"]
    assert base["values"]["nrf"] == base["values"]["pbc1"] == base["values"]["pbc2"] == ""
    _fake_nodedup(tmp_path, out, work)
    d = QF.measure(_row(out, "depth0.50"), tmp_path / "w", xcor_fn=_stub_xcor(calls))
    assert d["status"] == "ok" and d["pbc"] is None and d["values"]["nrf"] == ""
    assert calls[-1]["lines"] == int(_row(out, "depth0.50")["depth_reads"])


def test_measure_nodedup_arm_uses_its_own_reads(tmp_path):
    out, work = _build(tmp_path)
    allr = _fake_nodedup(tmp_path, out, work)
    mdir = out / f"{EXP}.pooled.uniform.nodedup"
    row = {"measurement_id": mdir.name, "counts_path": f"h5:{mdir}/counts.h5",
           "sha256": AV.sha256_file(mdir / "counts.h5")}
    calls = []
    rec = QF.measure(row, tmp_path / "w", xcor_fn=_stub_xcor(calls))
    assert rec["status"] == "ok", rec["reason"]
    assert rec["pbc"]["source_measurement_id"] == mdir.name
    assert (rec["pbc"]["mt"], rec["pbc"]["m0"], rec["pbc"]["m1"], rec["pbc"]["m2"]) \
        == _ref_pbc([(c, s, e, st) for c, s, e, _, st in allr])
    assert calls[0]["lines"] == len(allr)


def test_measure_refuses_reads_that_do_not_match(tmp_path):
    out, work = _build(tmp_path)
    row = _row(out, "base")
    calls = []
    bad_sha = {**row, "sha256": "0" * 64}
    rec = QF.measure(bad_sha, tmp_path / "w", xcor_fn=_stub_xcor(calls))
    assert rec["status"] == "refused" and "sha256" in rec["reason"]
    prov = json.loads((out / row["measurement_id"] / "provenance.json").read_text())
    pr1 = Path(prov["halves"]["pr1"]["tagalign"][0])
    _write_ta(pr1, [("chr1", 5, 41, "N", "+")])          # another run's reads
    rec = QF.measure(row, tmp_path / "w", xcor_fn=_stub_xcor(calls))
    assert rec["status"] == "refused" and "md5" in rec["reason"]
    assert calls == [] and all(v == "" for v in rec["values"].values())


def test_measure_refuses_nodedup_of_another_parent(tmp_path):
    out, work = _build(tmp_path)
    _fake_nodedup(tmp_path, out, work, parent=f"{EXP}.br1.uniform.base")
    rec = QF.measure(_row(out, "base"), tmp_path / "w", xcor_fn=_stub_xcor([]))
    assert rec["status"] == "refused" and "parent" in rec["reason"]


def test_measure_without_readset(tmp_path):
    rec = QF.measure({"measurement_id": "m", "counts_path": "store:abc", "sha256": ""},
                     tmp_path / "w", xcor_fn=_stub_xcor([]))
    assert rec["status"] == "no_readset"
    rec = QF.measure({"measurement_id": "m", "counts_path": f"h5:{tmp_path}/x/counts.h5"},
                     tmp_path / "w", xcor_fn=_stub_xcor([]))
    assert rec["status"] == "no_readset"


# --------------------------------------------------------------------------------------------
# run -> collect -> coverage

def _tables(tmp_path, rows):
    t = tmp_path / "tables"
    t.mkdir()
    cols = ["measurement_id", "in_eic", *QF.QC_COLUMNS, "counts_path", "sha256"]
    with open(t / "measurements.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    return t


def test_run_collect_coverage(tmp_path, monkeypatch, capsys):
    out, work = _build(tmp_path)
    _fake_nodedup(tmp_path, out, work)
    base, depth = _row(out, "base"), _row(out, "depth0.50")
    stale = {**depth, "measurement_id": f"{EXP}.pooled.uniform.depth0.50x"}
    blind = {"measurement_id": "Xblind", "in_eic": "blind", "counts_path": "store:x"}
    t = _tables(tmp_path, [base, depth, stale, blind])
    qc = tmp_path / "qc"
    monkeypatch.setattr(QF, "run_xcor", _stub_xcor([]))
    assert QF.main(["run", "--tables", str(t), "--rows", "0-1", "--work", str(tmp_path / "w"),
                    "--out", str(qc), "--sort-mem", "16M"]) == 0
    assert {p.name for p in qc.glob("*.json")} == {f"{base['measurement_id']}.json",
                                                    f"{depth['measurement_id']}.json"}
    # a record whose counts sha256 is not the row's is never collected
    rec = json.loads((qc / f"{depth['measurement_id']}.json").read_text())
    rec["measurement_id"] = stale["measurement_id"]
    rec["counts_sha256"] = "f" * 64
    (qc / f"{stale['measurement_id']}.json").write_text(json.dumps(rec))
    side = tmp_path / "eic_v1_qc.tsv"
    assert QF.main(["collect", "--tables", str(t), "--out", str(side), "--qc-dir", str(qc)]) == 0
    with open(side) as f:
        got = list(csv.DictReader(f, delimiter="\t"))
    assert list(got[0]) == ["measurement_id", *QF.QC_COLUMNS]
    by = {r["measurement_id"]: r for r in got}
    assert len(got) == 4
    assert by[base["measurement_id"]]["nsc"] == "1.2000" and by[base["measurement_id"]]["nrf"]
    assert by[depth["measurement_id"]]["nsc"] == "1.2000" and by[depth["measurement_id"]]["nrf"] == ""
    assert all(v == "" for k, v in by[stale["measurement_id"]].items() if k != "measurement_id")
    assert all(r["frip"] == "" for r in got)
    capsys.readouterr()
    assert QF.main(["coverage", "--tables", str(t), "--qc-table", str(side),
                    "--missing-rows"]) == 0
    lines = dict(ln.split("\t", 1) for ln in capsys.readouterr().out.strip().split("\n"))
    assert lines["nsc"] == "0.6667\t2/3" and lines["nrf"] == "0.3333\t1/3"
    assert lines["frip"] == "0.0000\t0/3" and lines["missing_rows"] == "1-2"
    # skip-if-done: a second run does not recompute
    monkeypatch.setattr(QF, "run_xcor", lambda *a, **k: pytest.fail("recomputed"))
    assert QF.main(["run", "--tables", str(t), "--rows", "0", "--work", str(tmp_path / "w"),
                    "--out", str(qc)]) == 0
