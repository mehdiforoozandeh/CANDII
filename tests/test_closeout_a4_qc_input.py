"""Close-out item A4 (read-derived QC on the input side): absent encoding, the two leak rules in the
dataset, and the half mode and multi-directory collect of qc_fill.py. CPU, seconds."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from minicandi import contract as C  # noqa: E402
from minicandi import covariates as CV  # noqa: E402
from minicandi import dataset as D  # noqa: E402
from minicandi import synth  # noqa: E402
from onion_data import qc_fill as QF  # noqa: E402

import test_onion_data_qc_fill as T  # noqa: E402  (measurement-dir helpers)

L = 64
QI = {c: C.K_META + i for i, c in enumerate(C.QC_COLUMNS)}   # c_in index of each QC column


# ------------------------------------------------------------------ absent encoding (covariates)

def test_empty_qc_cell_is_absent_and_meta_keeps_its_centre():
    enc = CV.CovariateEncoder({"vocab": {}})
    row = {"nsc": "1.1", "rsc": "", "frag_len": "200"}
    q = enc.qc(row)
    assert q[0] == np.float32(1.1) and q[C.QC_COLUMNS.index("frag_len")] == 200.0
    assert all(q[C.QC_COLUMNS.index(c)] == CV.QC_ABSENT for c in ("rsc", "nrf", "pbc1", "frip"))
    assert enc.meta({})[C.META_COLUMNS.index("read_length")] == CV.NUMERIC["read_length"][0]
    assert CV.neutral_value("nsc") == CV.QC_ABSENT and CV.neutral_value("read_length") == 75.0


def test_embedding_tells_absent_from_a_typical_value():
    torch.manual_seed(0)
    emb = CV.CovariateEmbedding(C.K_META + C.K_QC, 6)
    assert len(emb.qc_pos) == C.K_QC
    assert emb.mlp[0].in_features == len(emb.num_idx) + C.K_QC + 8 * len(emb.cat_idx)
    enc = CV.CovariateEncoder({"vocab": {}})
    centre = {c: str(CV.NUMERIC[c][0]) for c in C.QC_COLUMNS}      # every QC cell at its centre
    absent = {}
    a = emb(torch.from_numpy(enc.c_in(centre, True))[None])
    b = emb(torch.from_numpy(enc.c_in(absent, True))[None])
    assert torch.isfinite(b).all() and not torch.allclose(a, b)
    meta = CV.CovariateEmbedding(C.K_META, 6)                      # the default build
    assert meta.qc_pos == [] and meta.mlp[0].in_features == len(meta.num_idx) + 8 * 6


def test_dropping_a_qc_field_reads_as_absent():
    c = torch.from_numpy(CV.CovariateEncoder({"vocab": {}}).c_in({"nsc": "1.5"}, True))[None, None]
    idx, _ = CV.drop_field_indices({"covariate_drop_fields": ["nsc"], "qc_on_input": True})
    cols = list(C.META_COLUMNS) + list(C.QC_COLUMNS)
    out = CV.drop_fields(c, torch.ones(1, 1, dtype=torch.bool), idx, cols)
    assert out[0, 0, QI["nsc"]] == CV.QC_ABSENT


# ------------------------------------------------------------------ dataset leak rules

@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    """Synthetic tables whose `mapq0` view rows are relabelled as `nodedup` children of the base,
    so a base -> nodedup pair exists."""
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("a4"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    p = td / "measurements.tsv"
    with open(p, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        cols, rows = r.fieldnames, list(r)
    for row in rows:
        if row["arm_knob"] == "mapq":
            row["arm_knob"], row["arm_value"] = "nodedup", ""
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    wh = td / "withheld.tsv"
    if wh.is_file():
        wh.write_text(wh.read_text().splitlines()[0] + "\n")
    return td


def _rows(tables):
    with open(tables / "measurements.tsv", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _qc_table(path, mids, nsc="1.5"):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(D.QC_TABLE_COLUMNS)
        for m in mids:
            w.writerow([m, nsc, "2.5", "0.91", "0.92", "33", "", "321"])
    return path


def _base_and_child(tables):
    rows = _rows(tables)
    child = next(r for r in rows if r["arm_knob"] == "nodedup" and r["in_eic"] == "train")
    base = next(r for r in rows if r["measurement_id"] == child["parent_measurement_id"])
    other = next(r for r in rows if r["parent_measurement_id"] == base["measurement_id"]
                 and r["arm_knob"] not in ("", "nodedup"))
    return base, child, other


def test_base_input_loses_pbc_when_its_nodedup_child_is_the_target(tables, tmp_path):
    base, child, other = _base_and_child(tables)
    qc = _qc_table(tmp_path / "qc.tsv", [r["measurement_id"] for r in _rows(tables)])
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "qc_on_input": True,
                                          "qc_table": str(qc)}, seed=1)
    chrom = sorted(ds.n_bins)[0]
    t = C.TRACK_INDEX[base["assay"]]
    w = ds.window(chrom, 0, [base["measurement_id"]], [child["measurement_id"]])
    v = w["c_in"][t]
    assert all(v[QI[c]] == CV.QC_ABSENT for c in D.PBC_COLUMNS)          # leak removed
    assert v[QI["nsc"]] == np.float32(1.5) and v[QI["frag_len"]] == 321.0  # own reads kept
    w2 = ds.window(chrom, 0, [base["measurement_id"]], [other["measurement_id"]])
    assert w2["c_in"][t, QI["nrf"]] == np.float32(0.91)                   # another target: kept
    w3 = ds.window(chrom, 0, [child["measurement_id"]], [base["measurement_id"]])
    assert w3["c_in"][t, QI["nrf"]] == np.float32(0.91)                   # nodedup input: own reads
    assert torch.equal(w["c_out"], ds.window(chrom, 0, [base["measurement_id"]],
                                             [child["measurement_id"]])["c_out"])
    off = D.OnionDataset(tables, "train", {"context_bins": L, "qc_table": str(qc)}, seed=1)
    wo = off.window(chrom, 0, [base["measurement_id"]], [child["measurement_id"]])
    assert wo["c_in"].shape[-1] == C.K_META and torch.equal(wo["c_out"], w["c_out"])


def test_pr1_input_reads_the_pr1_table_only(tables, tmp_path):
    base, child, other = _base_and_child(tables)
    mids = [r["measurement_id"] for r in _rows(tables)]
    full = _qc_table(tmp_path / "full.tsv", mids, nsc="1.5")
    pr1 = _qc_table(tmp_path / "pr1.tsv", [base["measurement_id"]], nsc="1.25")
    cfg = {"context_bins": L, "qc_on_input": True, "qc_table": str(full)}
    t = C.TRACK_INDEX[base["assay"]]
    with_pr1 = D.OnionDataset(tables, "val", dict(cfg, qc_table_pr1=str(pr1)), seed=1)
    without = D.OnionDataset(tables, "val", cfg, seed=1)
    chrom = sorted(with_pr1.n_bins)[0]
    kw = dict(input_file="pr1", target_file="pr2")
    a = with_pr1.window(chrom, 0, [base["measurement_id"]], [other["measurement_id"]], **kw)
    b = without.window(chrom, 0, [base["measurement_id"]], [other["measurement_id"]], **kw)
    assert a["c_in"][t, QI["nsc"]] == np.float32(1.25)          # the half's own value
    assert b["c_in"][t, QI["nsc"]] == np.float32(1.5)           # today's approximation: parent
    o = C.TRACK_INDEX[other["assay"]]
    c = with_pr1.window(chrom, 0, [other["measurement_id"]], [base["measurement_id"]], **kw)
    assert all(c["c_in"][o, QI[k]] == CV.QC_ABSENT for k in C.QC_COLUMNS)   # not listed: absent
    d = with_pr1.window(chrom, 0, [base["measurement_id"]], [other["measurement_id"]])
    assert d["c_in"][t, QI["nsc"]] == np.float32(1.5)           # a counts input ignores the table
    assert torch.equal(a["c_out"], b["c_out"])
    bad = tmp_path / "dup.tsv"
    _qc_table(bad, [base["measurement_id"]] * 2)
    with pytest.raises(ValueError):
        D.OnionDataset(tables, "val", dict(cfg, qc_table_pr1=str(bad)))


# ------------------------------------------------------------------ qc_fill half mode, collect

def test_measure_half_reads_the_half_files_only(tmp_path):
    out, work = T._build(tmp_path)
    allr = T._fake_nodedup(tmp_path, out, work)
    calls = []
    rec = QF.measure(T._row(out, "base"), tmp_path / "w", xcor_fn=T._stub_xcor(calls),
                     half="pr1")
    assert rec["status"] == "ok", rec["reason"] and rec["half"] == "pr1"
    assert {f["half"] for f in rec["kept_readset"]["files"]} == {"pr1"}
    assert {f["half"] for f in rec["pbc"]["files"]} == {"pr1"}
    assert rec["pbc"]["mt"] == len(allr[0::2])                 # the nodedup sibling's pr1 half
    prov = json.loads((out / T._row(out, "base")["measurement_id"] / "provenance.json").read_text())
    assert calls[0]["lines"] == prov["halves"]["pr1"]["n_lines"]
    with pytest.raises(ValueError):
        QF.measure(T._row(out, "base"), tmp_path / "w", xcor_fn=T._stub_xcor([]), half="pr3")


def _rec(d: Path, mid, sha, half="", status="ok", nsc="1.3"):
    d.mkdir(parents=True, exist_ok=True)
    v = {c: "" for c in QF.QC_COLUMNS}
    v.update(nsc=nsc, rsc="1.1", frag_len="150")
    (d / QF.record_name(mid, half or None)).write_text(json.dumps(
        {"measurement_id": mid, "half": half, "status": status, "counts_sha256": sha,
         "values": v}))


def _read(p):
    with open(p, newline="") as f:
        return {r["measurement_id"]: r for r in csv.DictReader(f, delimiter="\t")}


def test_collect_many_dirs_last_usable_record_wins_and_half_table(tmp_path, capsys):
    rows = [{"measurement_id": m, "in_eic": "train", "sha256": "s" + m, "nsc": b, "nrf": "0.8"}
            for m, b in (("A", ""), ("B", "1.9"), ("C", ""), ("D", ""))]
    t = T._tables(tmp_path, rows)
    d1, d2 = tmp_path / "q1", tmp_path / "q2"
    _rec(d1, "A", "sA", nsc="1.1")
    _rec(d2, "A", "sA", nsc="1.2")                 # later dir wins
    _rec(d1, "C", "sC", nsc="1.4")
    _rec(d2, "C", "sC", status="failed")           # a later failed record does not hide d1's
    _rec(d2, "D", "stale", nsc="9")                # stale: kept from measurements.tsv
    _rec(d2, "A", "sA", half="pr1", nsc="1.05")
    out = tmp_path / "qc.tsv"
    assert QF.main(["collect", "--tables", str(t), "--out", str(out),
                    "--qc-dir", str(d1), "--qc-dir", str(d2)]) == 0
    got = _read(out)
    assert got["A"]["nsc"] == "1.2" and got["C"]["nsc"] == "1.4" and got["D"]["nsc"] == ""
    assert got["B"]["nsc"] == "1.9" and got["A"]["nrf"] == "0.8"       # build values kept
    assert "no_record 1" in capsys.readouterr().out
    half = tmp_path / "qc_pr1.tsv"
    assert QF.main(["collect", "--tables", str(t), "--out", str(half), "--half", "pr1",
                    "--qc-dir", str(d1), "--qc-dir", str(d2)]) == 0
    h = _read(half)
    assert list(h) == ["A"] and h["A"]["nsc"] == "1.05" and h["A"]["nrf"] == ""   # no fallback
    one = tmp_path / "one.tsv"                     # one dir: the 2026-11-13 behaviour
    assert QF.main(["collect", "--tables", str(t), "--out", str(one), "--qc-dir", str(d1)]) == 0
    assert _read(one)["A"]["nsc"] == "1.1"
    with pytest.raises(SystemExit):
        QF.main(["collect", "--tables", str(t), "--out", str(one), "--half", "pr1",
                 "--fraglen-out", str(tmp_path / "fl.tsv")])


def test_qc_values_are_clipped_and_zero_fragment_length_is_absent(tmp_path):
    torch.manual_seed(0)
    emb = CV.CovariateEmbedding(C.K_META + C.K_QC, 6)
    enc = CV.CovariateEncoder({"vocab": {}})
    lim = CV.NUMERIC["nsc"][0] + CV.QC_Z_CLIP * CV.NUMERIC["nsc"][1]
    a = emb(torch.from_numpy(enc.c_in({"nsc": str(lim)}, True))[None])
    b = emb(torch.from_numpy(enc.c_in({"nsc": "40"}, True))[None])
    assert torch.allclose(a, b)                                   # beyond +5 z: same input
    rows = [{"measurement_id": "Z", "in_eic": "train", "sha256": "sZ"}]
    t = T._tables(tmp_path, rows)
    d = tmp_path / "q"
    d.mkdir()
    v = {c: "" for c in QF.QC_COLUMNS}
    v.update(nsc="1.6", rsc="1.9", frag_len="0")
    (d / "Z.json").write_text(json.dumps({"measurement_id": "Z", "status": "ok",
                                          "counts_sha256": "sZ", "values": v}))
    out = tmp_path / "qc.tsv"
    assert QF.main(["collect", "--tables", str(t), "--out", str(out), "--qc-dir", str(d)]) == 0
    got = _read(out)["Z"]
    assert got["frag_len"] == "" and got["nsc"] == "1.6"


# ------------------------------------------------------------------ review fixes (2026-11-17)

def test_guard_covers_a_replicate_base_and_the_pooled_nodedup_of_its_experiment(tables, tmp_path):
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "qc_on_input": True}, seed=1)
    full = {"nsc": "1.5", "rsc": "2", "nrf": "0.9", "pbc1": "0.9", "pbc2": "20", "frip": "",
            "frag_len": "180"}
    ds.rows["E1.br1.base"] = {"experiment": "E1", "arm_knob": "", **full}
    ds.rows["E1.br1.nodedup"] = {"experiment": "E1", "arm_knob": "nodedup", **full}
    ds.rows["E1.pooled.nodedup"] = {"experiment": "E1", "arm_knob": "nodedup", **full}
    ds.rows["E2.pooled.nodedup"] = {"experiment": "E2", "arm_knob": "nodedup", **full}
    r = ds.rows["E1.br1.base"]
    got = ds.qc_input_row("E1.br1.base", r, "counts", ["E1.pooled.nodedup"])
    assert all(got[c] == "" for c in D.PBC_COLUMNS) and got["nsc"] == "1.5"
    assert ds.qc_input_row("E1.br1.base", r, "counts", ["E2.pooled.nodedup"]) is r
    n = ds.rows["E1.br1.nodedup"]                    # a nodedup input: its own reads, kept
    assert ds.qc_input_row("E1.br1.nodedup", n, "counts", ["E1.pooled.nodedup"])["nrf"] == "0.9"


def test_qc_table_reaches_inputs_only_and_targets_keep_frozen_frag_len(tables, tmp_path):
    base, child, other = _base_and_child(tables)
    mids = [r["measurement_id"] for r in _rows(tables)]
    qc = _qc_table(tmp_path / "qc.tsv", mids)                      # frag_len 321 everywhere
    on = D.OnionDataset(tables, "val", {"context_bins": L, "qc_on_input": True,
                                        "qc_table": str(qc)}, seed=1)
    off = D.OnionDataset(tables, "val", {"context_bins": L}, seed=1)
    for m in mids:                                      # what score.py's signal routes read
        assert on.rows[m] == off.rows[m]
    chrom = sorted(on.n_bins)[0]
    t = C.TRACK_INDEX[base["assay"]]
    w = on.window(chrom, 0, [base["measurement_id"]], [other["measurement_id"]])
    v = off.window(chrom, 0, [base["measurement_id"]], [other["measurement_id"]])
    assert w["c_in"][t, QI["frag_len"]] == 321.0                    # the input reads the table
    for k in ("y_counts", "x_counts", "c_out"):
        assert torch.equal(w[k], v[k]), k


def test_collect_blanks_build_copies_of_non_base_views(tmp_path, capsys):
    cols = ["measurement_id", "in_eic", "arm_knob", *QF.QC_COLUMNS, "counts_path", "sha256"]
    t = tmp_path / "tables"
    t.mkdir()
    rows = [{"measurement_id": "B", "in_eic": "train", "arm_knob": "", "nsc": "1.4",
             "frag_len": "150", "sha256": "sB"},
            {"measurement_id": "X", "in_eic": "train", "arm_knob": "crop", "nsc": "1.4",
             "rsc": "1.2", "frag_len": "150", "sha256": "sX"},
            {"measurement_id": "Y", "in_eic": "train", "arm_knob": "crop", "nsc": "1.4",
             "nrf": "0.8", "sha256": "sY"}]
    with open(t / "measurements.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    d = tmp_path / "q"
    _rec(d, "Y", "sY", nsc="1.7")                      # own record: nsc, rsc, frag_len only
    out = tmp_path / "qc.tsv"
    assert QF.main(["collect", "--tables", str(t), "--out", str(out), "--qc-dir", str(d)]) == 0
    got = _read(out)
    assert got["B"]["nsc"] == "1.4" and got["B"]["frag_len"] == "150"      # base kept
    assert all(got["X"][c] == "" for c in QF.QC_COLUMNS)                    # copies blanked
    assert got["Y"]["nsc"] == "1.7" and got["Y"]["nrf"] == ""               # own record only
    assert "blanked_nonbase_build_cells 5" in capsys.readouterr().out
