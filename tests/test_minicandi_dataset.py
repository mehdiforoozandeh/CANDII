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


def test_synth_splits_per_track_like_eic(tables):
    """Training cells also hold validation and blind tracks, so the role test below can fail."""
    rows = _rows(tables)
    by_cell = {}
    for r in rows.values():
        by_cell.setdefault(r["cell"], set()).add(r["in_eic"])
    mixed = [c for c, roles in by_cell.items() if "train" in roles and roles - {"train"}]
    assert len(mixed) >= 4, by_cell


def test_10000_training_draws_hold_no_withheld_combination_and_no_val_or_blind_row(tables,
                                                                                train_ds):
    rows = _rows(tables)
    wh = D.Withheld(list(csv.DictReader(open(tables / "withheld.tsv"), delimiter="\t")))
    assert len(wh) == 2
    rng = np.random.default_rng(123)
    kinds = {1: 0, 2: 0, 3: 0}
    cells_with_heldout = set()
    for _ in range(10_000):
        p = train_ds.plan(rng)
        ins = [e[1] for e in p["entries"] if e[1] is not None]
        outs = [e[2] for e in p["entries"]]
        assert all(rows[m]["in_eic"] == "train" for m in ins + outs)
        assert not _violations(rows, wh, ins, outs)
        for e in p["entries"]:
            kinds[e[3]] += 1
        if any(r["cell"] == p["cell"] and r["in_eic"] != "train" for r in rows.values()):
            cells_with_heldout.add(p["cell"])
    assert min(kinds.values()) > 1000, kinds          # every kind is exercised
    assert len(cells_with_heldout) >= 4               # drawn cells did hold val/blind tracks


def test_the_role_and_withheld_tests_are_not_vacuous(tables):
    """With roles ignored or withheld.tsv emptied, the same sampler does produce the leaks."""
    rows = _rows(tables)
    wh = D.Withheld(list(csv.DictReader(open(tables / "withheld.tsv"), delimiter="\t")))
    ds = D.OnionDataset(tables, "train", {"context_bins": L}, seed=0)
    ds.withheld = D.Withheld([])
    ds.train_tracks = ds.group_tracks(("train", "val", "blind"))
    rng = np.random.default_rng(123)
    wh_hits = role_hits = 0
    for _ in range(2_000):
        p = ds.plan(rng)
        ins = [e[1] for e in p["entries"] if e[1] is not None]
        outs = [e[2] for e in p["entries"]]
        wh_hits += bool(_violations(rows, wh, ins, outs))
        role_hits += any(rows[m]["in_eic"] != "train" for m in ins + outs)
    assert wh_hits > 100 and role_hits > 100


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
                                              "average_input": True, "batch_size": 5}, seed=1)
        b = next(iter(ds))                                  # iteration yields batches
        assert C.validate_batch(b, K) == (5, L)
        assert set(b) == set(C.REQUIRED) | {"avg"}
        shapes = C.expected_shapes(5, L, K)
        for k, v in b.items():
            if k != "meta":
                assert tuple(v.shape) == shapes[k] and v.dtype == C.DTYPES[k], k
        assert b["c_out"].shape[-1] == C.K_META
        for m in b["meta"]:
            assert len(m["measurements_in"]) == len(m["measurements_out"]) == C.A
        for bi, m in enumerate(b["meta"]):
            for t in range(C.A):
                assert bool(m["measurements_in"][t]) == bool(b["x_avail"][bi, t])
                assert bool(m["measurements_out"][t]) == bool(b["y_avail"][bi, t])


def test_masked_and_missing_channels_are_distinguished(tables, train_ds):
    rows = _rows(tables)
    rng = np.random.default_rng(3)
    n_masked = n_heldout_missing = 0
    for _ in range(100):
        p = train_ds.plan(rng)
        w = train_ds.materialize(p, rng)
        present = {C.TRACK_INDEX[a] for c, a in train_ds.train_tracks if c == p["cell"]}
        for t in range(C.A):
            if t not in present:                 # missing: no training track in this cell
                assert not w["x_avail"][t] and not w["y_avail"][t] and w["y_kind"][t] == 0
                assert (w["c_in"][t] == 0).all() and (w["c_out"][t] == 0).all()
                n_heldout_missing += any(r["cell"] == p["cell"] and r["in_eic"] != "train"
                                         and C.TRACK_INDEX[r["assay"]] == t
                                         for r in rows.values())
        for a, v_in, v_out, kind in p["entries"]:
            t = C.TRACK_INDEX[a]
            if kind == C.KIND_IMPUTE:                              # masked: a target, no input
                n_masked += 1
                assert not w["x_avail"][t] and w["y_avail"][t]
                assert (w["c_in"][t] == 0).all() and (w["x_counts"][:, t] == 0).all()
                want = train_ds.encoder.c_out(rows[v_out])         # decoded with its own C'
                assert torch.equal(w["c_out"][t], torch.from_numpy(want))
                assert w["c_out"][t, 0] > 20
    assert n_masked > 50 and n_heldout_missing > 10


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


def test_pseudoreplicate_halves_are_described_at_half_depth(tables):
    rows = _rows(tables)
    vds = D.OnionDataset(tables, "val", {"context_bins": L})
    mid = next(m for m, r in rows.items() if r["in_eic"] == "val" and not r["arm_knob"])
    full_log2 = vds.encoder.c_out(rows[mid])[0]
    w = vds.window("chr1", 100, [mid], [mid], input_file="pr1", target_file="pr2")
    t = C.TRACK_INDEX[rows[mid]["assay"]]
    assert w["y_kind"][t] == C.KIND_CORRECT                        # an independent draw
    assert w["c_in"][t, 0] == pytest.approx(full_log2 - 1.0)       # C follows the input file
    assert w["c_out"][t, 0] == pytest.approx(full_log2 - 1.0)
    full = vds.read(mid, "chr1", 100)
    assert np.array_equal(w["x_counts"][:, t].numpy() + w["y_counts"][:, t].numpy(), full)
    w1 = vds.window("chr1", 100, [mid], [mid])
    assert w1["c_in"][t, 0] == pytest.approx(full_log2)


def test_validation_mode_is_deterministic_and_has_all_three_tasks(tables):
    cfg = {"context_bins": L, "region_bins": 4 * L, "batch_size": 7}
    a = D.OnionDataset(tables, "val", cfg)
    b = D.OnionDataset(tables, "val", cfg)
    ev = json.loads((tables.parent.parent / "eval" / tables.name / "evalset.json").read_text())
    n_imp = len(ev["imputation_targets"])
    jobs = 2 * n_imp + len(ev["correction_pairs"])
    items = a.evalset_items()
    assert len(items) == jobs * len(ev["regions"]) * 4
    batches_a, batches_b = list(a), list(b)
    assert sum(x["x_counts"].shape[0] for x in batches_a) == len(items)
    assert batches_a[-1]["x_counts"].shape[0] == len(items) % 7 or len(items) % 7 == 0
    rows = _rows(tables)
    seen = set()
    for wa, wb in zip(batches_a, batches_b):
        C.validate_batch(wa)
        for k in wa:
            if k == "meta":
                assert wa[k] == wb[k]
            else:
                assert torch.equal(wa[k], wb[k])
        for bi, m in enumerate(wa["meta"]):
            seen.add(m["task"])
            t = C.TRACK_INDEX[rows[m["targets"][0]]["assay"]]
            if m["task"] != "correction":                          # context = training tracks
                assert all(rows[x]["in_eic"] == "train" for x in m["inputs"]
                           if x != m["targets"][0])
            if m["task"] == "correction":                          # PI ruling 6 floor layout
                assert m["input_file"] == "pr1" and m["target_file"] == "pr2"
                assert wa["y_kind"][bi, t] == C.KIND_CORRECT
                full = a.encoder.c_out(rows[m["targets"][0]])[0]
                assert wa["c_out"][bi, t, 0] == pytest.approx(full - 1.0)
            elif m["task"] == "reconstruction":
                assert wa["y_kind"][bi, t] == C.KIND_SAME
                assert wa["x_bin_mask"][bi, :, t].any() and not wa["x_bin_mask"][bi, :, t].all()
                assert C.loss_mask(wa)[bi, :, t].sum() == wa["x_bin_mask"][bi, :, t].sum()
            else:
                assert wa["y_kind"][bi, t] == C.KIND_IMPUTE
    assert seen == {"imputation", "correction", "reconstruction"}


def test_leave_one_cell_out_average_direct_and_precomputed(tables, tmp_path):
    rows = _rows(tables)
    cell, chrom, start = "s1", "chr1", 300
    ds = D.OnionDataset(tables, "train", {"context_bins": L})
    avg = ds.average(cell, chrom, start)
    t = C.TRACK_INDEX["H3K4me3"]
    others = [m for m, r in rows.items() if r["in_eic"] == "train" and r["cell"] != cell
              and r["assay"] == "H3K4me3" and not r["arm_knob"]]
    assert len(others) == 3                                       # s0 is val, s3 blind
    want = np.mean([ds.read(m, chrom, start) * 1e6 / float(rows[m]["depth_reads"])
                    for m in others], axis=0)
    np.testing.assert_allclose(avg[:, t], want, rtol=1e-5)
    assert (avg[:, C.TRACK_INDEX["H3K9me3"]] == 0).all()
    h5 = D.build_average_h5(tables, tmp_path / "avg.h5")
    ds2 = D.OnionDataset(tables, "train", {"context_bins": L, "average_h5": str(h5)})
    for c in ("s0", "s1", "s5"):
        np.testing.assert_allclose(ds2.average(c, chrom, start), ds.average(c, chrom, start),
                                   rtol=1e-4, atol=1e-3)


def test_control_flags(tables):
    ctl = C.CONTROL_INDEX
    for key, field in (("control_input", "x_avail"), ("predict_control", "y_avail")):
        ds = D.OnionDataset(tables, "train", {"context_bins": L, key: False}, seed=2)
        b = ds.batch(32)
        C.validate_batch(b)
        assert not b[field][:, ctl].any()
        if key == "control_input":
            assert (b["y_kind"][:, ctl][b["y_avail"][:, ctl]] == C.KIND_IMPUTE).all()
            assert (b["c_in"][:, ctl] == 0).all()
        else:
            assert (b["c_out"][:, ctl] == 0).all()
    on = D.OnionDataset(tables, "train", {"context_bins": L}, seed=2).batch(32)
    assert on["x_avail"][:, ctl].any() and on["y_avail"][:, ctl].any()


def test_two_step_base_keys(tables):
    rows = _rows(tables)
    for depth in ("native", "fixed"):
        ds = D.OnionDataset(tables, "train", {"context_bins": L, "two_step": True,
                                              "base_depth": depth}, seed=4)
        b = ds.batch(16)
        C.validate_batch(b)
        assert b["c_base"].shape == (16, C.A, C.K_META) and b["c_base"].dtype == torch.float32
        assert b["y_base"].shape == (16, L, C.A) and b["y_base"].dtype == torch.int32
        has = b["c_base"][..., 0] > 0
        assert torch.equal(has, b["y_avail"])                     # every synth track has a base
        for bi, m in enumerate(b["meta"]):
            for t in np.nonzero(b["y_avail"][bi].numpy())[0]:
                mid = m["measurements_out"][t]
                base = f"{rows[mid]['experiment']}.pooled.uniform.base"
                full = ds.read(base, m["chrom"], m["start_bin"])
                nat = ds.encoder.c_out(rows[base])[0]
                if depth == "native" or nat <= D.FIXED_BASE_LOG2_DEPTH[rows[base]["assay_class"]]:
                    assert np.array_equal(b["y_base"][bi, :, t].numpy(), full)
                    assert b["c_base"][bi, t, 0] == pytest.approx(nat)
                else:
                    assert (b["y_base"][bi, :, t].numpy() <= full).all()
                    assert b["c_base"][bi, t, 0] == 23.0
    plain = D.OnionDataset(tables, "train", {"context_bins": L}, seed=4).batch(2)
    assert "c_base" not in plain and "y_base" not in plain


def test_handle_cache_is_bounded(tables):
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "max_open_files": 3})
    for m in sorted(ds.allowed)[:20]:
        ds.read(m, "chr1", 0)
        assert len(ds._h5) <= 3


def test_tables_sha256_tracks_content(tables, tmp_path):
    h = D.tables_sha256(tables)
    assert h == D.tables_sha256(tables)
    cp = tmp_path / "t"
    shutil.copytree(tables, cp)
    with open(cp / "withheld.tsv", "a") as fh:
        fh.write("se\t\tbase\t\t*\n")
    assert D.tables_sha256(cp) != h


# ------------------------------------------------------------------------------- integration


class _ConstHead(C.MiniCandi):
    """A model from the shared components whose head is still zero-initialised."""

    def __init__(self, transform="local_ratio"):
        super().__init__()
        from minicandi import components as M
        d = 8
        self.tr = M.CountTransform(transform)
        self.inp = torch.nn.Linear(1, d)
        self.mask = M.MaskHandler("token", d)
        self.conv = torch.nn.Conv1d(C.A * d, C.A * d, 5, padding=2)
        self.head = M.NBHead(d, self.tr)

    def forward(self, batch):
        from minicandi import components as M
        B, Lb, A = batch["x_counts"].shape
        vis = M.MaskHandler.visible(batch["x_avail"], batch["x_bin_mask"])
        x = self.tr(batch["x_counts"], batch["c_in"][..., C.LOG2_DEPTH_COL], vis)
        h, _ = self.mask(self.inp(x[..., None]), batch["x_avail"], batch["x_bin_mask"])
        h = self.conv(h.reshape(B, Lb, -1).transpose(1, 2)).transpose(1, 2).reshape(B, Lb, A, -1)
        return self.head(h, batch["c_out"])


def test_batches_feed_the_merged_guardrail_and_baselines(tables):
    from onion_eval import baselines, guardrail
    cfg = {"context_bins": L, "region_bins": 4 * L, "batch_size": 8}
    vds = D.OnionDataset(tables, "val", cfg)
    val = list(vds)
    torch.manual_seed(0)
    model = _ConstHead().eval()
    with torch.no_grad():                                     # trunk random, head zero
        for p in model.conv.parameters():
            p.normal_(0, 0.3)
    res = guardrail.copy_leak(model, val[:6])
    assert res["part1_wiring_pass"], res
    assert np.isfinite(res["same_view_copy_gain"])            # same-view entries on validation
    # part 2 is not asserted: with ~400 hidden bins the hidden-minus-visible truth of a constant
    # model moves by +-15% across mask seeds (measured), above the 10% gate; E4 owns part 2.
    leaky = _ConstHead()
    with torch.no_grad():
        torch.nn.init.normal_(leaky.head.lin.weight, std=0.5)
    leaky.forward = lambda b, _m=leaky: _m.head(_m.inp(b["x_counts"].float()[..., None]),
                                                b["c_out"])
    assert not guardrail.copy_leak(leaky, val[:6])["part1_wiring_pass"]

    tds = D.OnionDataset(tables, "train", cfg, seed=5)
    train = [b for _, b in zip(range(6), tds)]
    marg = baselines.MarginalNB().fit(train).predict(val[0])
    pa = baselines.PositionAverage().fit(train).predict(val[0])
    yav = val[0]["y_avail"]
    for out in (marg, pa):
        assert out["mu"].shape == val[0]["y_counts"].shape
        assert torch.isfinite(out["mu"][:, :, :][yav[:, None, :].expand_as(out["mu"])]).any()
    corr = [b for b in val if (b["y_kind"] == C.KIND_CORRECT).any()]
    dos = baselines.DepthOnlyScaling().fit(corr)
    out = dos.predict(corr[0])
    k2 = (corr[0]["y_kind"] == C.KIND_CORRECT)[:, None, :].expand_as(out["mu"])
    assert torch.isfinite(out["mu"][k2]).all() and torch.isnan(out["mu"][~k2]).all()
