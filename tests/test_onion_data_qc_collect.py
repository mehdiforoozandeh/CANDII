"""FL: `qc_fill.py collect` keeps the build's QC values and writes the fragment-length side table.

Pins:
  - a row with no record, a failed record or a stale record (counts sha256 not the row's) keeps
    its `measurements.tsv` values in the QC side table (the defect: they were blanked, and
    `dataset.py::_apply_qc_table` overrides every listed row);
  - a row with an ok record takes the record's values; a cell the record leaves empty keeps the
    build's value; `frip` is always empty;
  - `--fraglen-out`: one row per ok record with a fragment length, the xcor fields of its
    `.cc.qc` line, the frozen `frag_len` beside it, and the record path and job ids as source;
    `signal_from_counts.load_fraglen_table` reads it.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_data import qc_fill as QF  # noqa: E402

COLS = ["measurement_id", "in_eic", "assay_class", "run_type", *QF.QC_COLUMNS, "counts_path",
        "sha256"]
CC = "kept.no_chrM.15M.tagAlign.gz\t15000000\t165\t0.359\t40\t0.324\t1500\t0.225\t1.59\t1.34\t1"


def _row(mid, **kw):
    r = {c: "" for c in COLS}
    r.update(measurement_id=mid, in_eic="train", assay_class="narrow_histone",
             run_type="single-ended", sha256=f"sha_{mid}")
    r.update(kw)
    return r


def _rec(mid, status="ok", sha=None, **vals):
    v = {c: "" for c in QF.QC_COLUMNS}
    v.update(vals)
    return {"schema": 1, "measurement_id": mid, "status": status, "values": v,
            "counts_sha256": sha or f"sha_{mid}", "slurm_job_ids": ["123_4"],
            "xcor": {"cc_qc": CC, "frag_len": v["frag_len"]}}


def _setup(tmp_path):
    t = tmp_path / "tables"
    t.mkdir()
    rows = [
        _row("built", nsc="1.5", rsc="1.2", frag_len="210", nrf="0.9", pbc1="0.9", pbc2="9"),
        _row("se_ok"),
        _row("se_partial", nrf="0.8"),
        _row("se_failed", frag_len=""),
        _row("se_stale", nsc="1.1"),
        _row("pe_mixed", run_type="mixed", frag_len="", frip="0.3"),
    ]
    with open(t / "measurements.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    qc = tmp_path / "qc"
    qc.mkdir()
    recs = [_rec("se_ok", nsc="1.59", rsc="1.34", frag_len="165", nrf="0.93", pbc1="0.94",
                 pbc2="15.7"),
            _rec("se_partial", nsc="1.0", rsc="1.0", frag_len="140"),   # no nrf/pbc (non-base arm)
            _rec("se_failed", status="failed"),
            _rec("se_stale", sha="other", nsc="2.0", rsc="2.0", frag_len="120")]
    for r in recs:
        (qc / f"{r['measurement_id']}.json").write_text(json.dumps(r))
    return t, qc, {r["measurement_id"]: r for r in rows}


def _read(p):
    with open(p) as f:
        return {r["measurement_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def test_collect_keeps_the_build_values(tmp_path, capsys):
    t, qc, rows = _setup(tmp_path)
    out = tmp_path / "eic_v1_qc.tsv"
    assert QF.main(["collect", "--tables", str(t), "--out", str(out), "--qc-dir", str(qc)]) == 0
    got = _read(out)
    assert set(got) == set(rows)
    for mid in ("built", "se_failed", "se_stale"):            # no usable record: build values
        assert {c: got[mid][c] for c in QF.QC_COLUMNS if c != "frip"} == \
               {c: rows[mid][c] for c in QF.QC_COLUMNS if c != "frip"}
    assert got["se_ok"]["frag_len"] == "165" and got["se_ok"]["nrf"] == "0.93"
    assert got["se_partial"]["frag_len"] == "140" and got["se_partial"]["nrf"] == "0.8"
    assert got["pe_mixed"]["frip"] == "0.3"                    # no record: the row as built
    msg = capsys.readouterr().out
    assert "filled 2" in msg and "kept_build_cells 1" in msg and "stale 1" in msg


def test_collect_fraglen_side_table(tmp_path):
    from onion_eval import signal_from_counts as SFC
    t, qc, rows = _setup(tmp_path)
    fl = tmp_path / "eic_v1_fraglen.tsv"
    assert QF.main(["collect", "--tables", str(t), "--out", str(tmp_path / "q.tsv"),
                    "--qc-dir", str(qc), "--fraglen-out", str(fl)]) == 0
    got = _read(fl)
    assert set(got) == {"se_ok", "se_partial"}                # ok records with a fragment length
    with open(fl) as f:
        assert next(csv.reader(f, delimiter="\t")) == list(QF.FRAGLEN_COLUMNS)
    r = got["se_ok"]
    assert (r["frag_len_estimated"], r["nsc"], r["rsc"], r["quality_tag"]) == ("165", "1.59",
                                                                              "1.34", "1")
    assert (r["xcor_num_reads"], r["phantom_peak"], r["min_corr"]) == ("15000000", "40", "0.225")
    assert r["frag_len_frozen"] == "" and r["run_type"] == "single-ended"
    assert r["source"] == f"{qc / 'se_ok.json'} job 123_4"
    tab = SFC.load_fraglen_table(fl)
    assert tab["se_ok"]["drop_reason"] == ""
    row, src, _ = SFC.with_fraglen(rows["se_ok"], tab)
    assert (row["frag_len"], src) == ("165", "estimated")
