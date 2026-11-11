"""miniCANDI OnionDataset on synthetic tables: roles, withheld.tsv, kinds, shapes, store: paths."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import dataset as D  # noqa: E402
from minicandi import synth  # noqa: E402

L = 64


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def train_ds(tables) -> D.OnionDataset:
    return D.OnionDataset(tables, "train", {"context_bins": L}, seed=0)


def _rows(tables):
    return {r["measurement_id"]: r for r in csv.DictReader(
        open(tables / "measurements.tsv", encoding="utf-8"), delimiter="\t")}


def _violations(rows, withheld, inputs, targets):
    """Independent of `Withheld.blocks`: literal string match on the synthetic tables."""
    out = []
    for i in inputs:
        for t in targets:
            si = (rows[i]["arm_knob"] or "base", rows[i]["arm_value"])
            st = (rows[t]["arm_knob"] or "base", rows[t]["arm_value"])
            classes = {rows[i]["assay_class"], rows[t]["assay_class"]}
            for w in withheld.rows:
                if ((w["source_knob"], w["source_value"]) == si
                        and (w["target_knob"], w["target_value"]) == st
                        and (w["assay_class"] == "*" or w["assay_class"] in classes)):
                    out.append((i, t))
    return out


# ------------------------------------------------------------------------------- roles


def test_10000_training_draws_hold_no_withheld_combination_and_no_val_or_blind_row(tables,
                                                                                train_ds):
    rows = _rows(tables)
    wh = D.Withheld(list(csv.DictReader(open(tables / "withheld.tsv"), delimiter="\t")))
    assert len(wh) == 2
    rng = np.random.default_rng(123)
    kinds = {1: 0, 2: 0, 3: 0}
    for _ in range(10_000):
        p = train_ds.plan(rng)
        ins = [e[1] for e in p["entries"] if e[1] is not None]
        outs = [e[2] for e in p["entries"]]
        assert all(rows[m]["in_eic"] == "train" for m in ins + outs)
        assert not _violations(rows, wh, ins, outs)
        for e in p["entries"]:
            kinds[e[3]] += 1
    assert min(kinds.values()) > 1000, kinds          # every kind is exercised


def test_the_withheld_test_is_not_vacuous(tables):
    """With withheld.tsv emptied, the same sampler does produce the withheld combinations."""
    rows = _rows(tables)
    wh = D.Withheld(list(csv.DictReader(open(tables / "withheld.tsv"), delimiter="\t")))
    ds = D.OnionDataset(tables, "train", {"context_bins": L}, seed=0)
    ds.withheld = D.Withheld([])
    rng = np.random.default_rng(123)
    hits = 0
    for _ in range(2_000):
        p = ds.plan(rng)
        ins = [e[1] for e in p["entries"] if e[1] is not None]
        hits += bool(_violations(rows, wh, ins, [e[2] for e in p["entries"]]))
    assert hits > 100


def test_materialised_windows_respect_withheld_and_roles(tables, train_ds):
    rows = _rows(tables)
    rng = np.random.default_rng(7)
    for _ in range(200):
        w = train_ds.materialize(train_ds.plan(rng), rng)
        m = w["meta"]
        assert not train_ds.window_violations(m["inputs"], m["targets"])
        assert all(rows[x]["in_eic"] == "train" for x in m["inputs"] + m["targets"])


def test_val_or_blind_row_requested_in_training_raises(tables, train_ds):
    rows = _rows(tables)
    val = next(m for m, r in rows.items() if r["in_eic"] == "val")
    blind = next(m for m, r in rows.items() if r["in_eic"] == "blind")
    for mid in (val, blind):
        with pytest.raises(D.RoleError):
            train_ds.read(mid, "chr1", 0)
        with pytest.raises(D.RoleError):
            train_ds.window("chr1", 0, [], [mid])
    with pytest.raises(D.RoleError):
        D.OnionDataset(tables, "blind", {"context_bins": L})
    vds = D.OnionDataset(tables, "val", {"context_bins": L})
    assert vds.read(val, "chr1", 0).shape == (L,)
    with pytest.raises(D.RoleError):
        vds.read(blind, "chr1", 0)
    bds = D.OnionDataset(tables, "blind", {"context_bins": L, "allow_blind": True})
    assert bds.read(blind, "chr1", 0).shape == (L,)


def test_explicit_withheld_window_in_training_raises(tables, train_ds):
    rows = _rows(tables)
    src = next(m for m, r in rows.items() if r["in_eic"] == "train" and r["arm_knob"] == "mapq"
               and r["assay_class"] == "narrow_histone")
    tgt = src.replace(".mapq0", ".se")
    with pytest.raises(D.WithheldError):
        train_ds.window("chr1", 0, [src], [tgt])
    vds = D.OnionDataset(tables, "val", {"context_bins": L})
    assert vds.window("chr1", 0, [src], [tgt])["y_kind"].sum() == C.KIND_CORRECT


# ------------------------------------------------------------------------------- contents


def test_batch_shapes_match_the_contract_exactly(tables):
    for qc, K in ((False, C.K_META), (True, C.K_META + C.K_QC)):
        ds = D.OnionDataset(tables, "train", {"context_bins": L, "qc_on_input": qc,
                                              "average_input": True}, seed=1)
        b = ds.batch(5)
        assert C.validate_batch(b, K) == (5, L)
        assert set(b) == set(C.REQUIRED) | {"avg"}
        shapes = C.expected_shapes(5, L, K)
        for k, v in b.items():
            if k != "meta":
                assert tuple(v.shape) == shapes[k] and v.dtype == C.DTYPES[k], k
        assert b["c_out"].shape[-1] == C.K_META


def test_masked_and_missing_channels_are_distinguished(tables, train_ds):
    rows = _rows(tables)
    rng = np.random.default_rng(3)
    present = {C.TRACK_INDEX[a] for a in synth.DEFAULT_ASSAYS}
    n_masked = 0
    for _ in range(100):
        p = train_ds.plan(rng)
        w = train_ds.materialize(p, rng)
        for t in range(C.A):
            if t not in present:                                   # missing: no track at all
                assert not w["x_avail"][t] and not w["y_avail"][t] and w["y_kind"][t] == 0
                assert (w["c_in"][t] == 0).all() and (w["c_out"][t] == 0).all()
        for a, v_in, v_out, kind in p["entries"]:
            t = C.TRACK_INDEX[a]
            if kind == C.KIND_IMPUTE:                              # masked: a target, no input
                n_masked += 1
                assert not w["x_avail"][t] and w["y_avail"][t]
                assert (w["c_in"][t] == 0).all() and (w["x_counts"][:, t] == 0).all()
                want = train_ds.encoder.c_out(rows[v_out])         # decoded with its own C'
                assert torch.equal(w["c_out"][t], torch.from_numpy(want))
                assert w["c_out"][t, 0] > 20
    assert n_masked > 50


def test_same_view_targets_hide_bins_and_loss_never_sees_a_visible_one(train_ds):
    b = train_ds.batch(64, np.random.default_rng(11))
    C.validate_batch(b)
    same = b["y_kind"] == C.KIND_SAME
    assert same.any()
    lm = C.loss_mask(b)
    visible_same = same[:, None, :] & ~b["x_bin_mask"]
    assert not (lm & visible_same).any()
    hidden = b["x_bin_mask"]
    assert (b["x_counts"][hidden] == 0).all()
    assert (b["y_counts"][hidden] > 0).any()                       # the truth is still there
    vis = same[:, None, :] & ~hidden
    assert torch.equal(b["x_counts"][vis], b["y_counts"][vis])


def test_pseudoreplicate_half_window_uses_half_depth(tables):
    rows = _rows(tables)
    vds = D.OnionDataset(tables, "val", {"context_bins": L})
    mid = next(m for m, r in rows.items() if r["in_eic"] == "val" and not r["arm_knob"])
    w = vds.window("chr1", 100, [mid], [mid], input_file="pr1", target_file="pr2",
                   target_depth_scale=0.5)
    t = C.TRACK_INDEX[rows[mid]["assay"]]
    assert w["y_kind"][t] == C.KIND_CORRECT                        # an independent draw
    assert w["c_out"][t, 0] == pytest.approx(w["c_in"][t, 0].item() - 1.0)
    full = vds.read(mid, "chr1", 100)
    assert np.array_equal(w["x_counts"][:, t].numpy() + w["y_counts"][:, t].numpy(), full)


def test_validation_mode_is_deterministic_from_the_evalset(tables):
    cfg = {"context_bins": L, "region_bins": 4 * L}
    a = D.OnionDataset(tables, "val", cfg)
    b = D.OnionDataset(tables, "val", cfg)
    ev = json.loads((tables.parent.parent / "eval" / tables.name / "evalset.json").read_text())
    jobs = len(ev["imputation_targets"]) + len(ev["correction_pairs"])
    items = a.evalset_items()
    assert len(items) == jobs * len(ev["regions"]) * (int(a.config["region_bins"]) // L)
    for wa, wb, _ in zip(a, b, range(12)):
        for k in wa:
            if k == "meta":
                assert wa[k] == wb[k]
            else:
                assert torch.equal(wa[k], wb[k])


def test_leave_one_cell_out_average(tables, train_ds):
    rows = _rows(tables)
    cell, chrom, start = "T_s0", "chr1", 300
    avg = train_ds.average(cell, chrom, start)
    t = C.TRACK_INDEX["H3K4me3"]
    others = [m for m, r in rows.items() if r["in_eic"] == "train" and r["cell"] != cell
              and r["assay"] == "H3K4me3" and not r["arm_knob"]]
    want = np.mean([train_ds.read(m, chrom, start) * 1e6 / float(rows[m]["depth_reads"])
                    for m in others], axis=0)
    assert len(others) == 3
    np.testing.assert_allclose(avg[:, t], want, rtol=1e-5)
    assert (avg[:, C.TRACK_INDEX["H3K9me3"]] == 0).all()


def test_tables_sha256_tracks_content(tables, tmp_path):
    h = D.tables_sha256(tables)
    assert h == D.tables_sha256(tables)
    cp = tmp_path / "t"
    shutil.copytree(tables, cp)
    with open(cp / "withheld.tsv", "a") as fh:
        fh.write("se\t\tbase\t\t*\n")
    assert D.tables_sha256(cp) != h


# ------------------------------------------------------------------------------- store: paths


def test_store_paths_and_read_free_depth_thinning(tmp_path):
    from tests.test_store_reader import make_store
    corpus = make_store(tmp_path / "s")
    tdir = tmp_path / "tables" / "pilot"
    tdir.mkdir(parents=True)
    rows = []
    for arm, value in (("", ""), ("depth", "0.5")):
        r = {k: "" for k in D.MEASUREMENT_COLUMNS}
        r.update({"measurement_id": f"E1.pooled.native.{'base' if not arm else 'depth0.5'}",
                  "experiment": "E1", "assay": "H3K4me3", "cell": "T_aa", "in_eic": "train",
                  "unit": "pooled", "layer": "native", "arm_knob": arm, "arm_value": value,
                  "depth_reads": "20000000",
                  "counts_path": f"store:{corpus}:T_aa:H3K4me3"})
        rows.append(r)
    r = dict(rows[0], measurement_id="E2.pooled.native.base", assay="chipseq-control",
             counts_path=f"store:{corpus}:T_aa:chipseq-control")
    rows.append(r)
    synth._write_tsv(tdir / "measurements.tsv", D.MEASUREMENT_COLUMNS, rows)
    (tdir / "covariates.json").write_text("{}")
    ds = D.OnionDataset(tdir, "train", {"context_bins": 256,
                                        "genome_dir": str(corpus.parent / "genome")})
    base = ds.read(rows[0]["measurement_id"], "chr1", 100)
    thin = ds.read(rows[1]["measurement_id"], "chr1", 100)
    assert base.sum() > 200
    assert (thin <= base).all() and np.array_equal(thin, ds.read(rows[1]["measurement_id"],
                                                                 "chr1", 100))
    assert abs(thin.sum() / base.sum() - 0.5) < 0.1
    ctl = ds.read("E2.pooled.native.base", "chr1", 100)
    assert ctl.shape == (256,) and ctl.dtype == np.int32
    with pytest.raises(ValueError):
        ds.read(rows[0]["measurement_id"], "chr1", 100, which="pr1")
    b = ds.batch(2)
    C.validate_batch(b)
