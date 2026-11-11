"""miniCANDI dataset arms (pinned interface E, chunk A2) on synthetic tables.

Defaults are byte-identical to the code before the arms; every arm keeps the leak rules: blind
rows, withheld combinations and (without `train_on_validation`) validation rows are unreachable
in train mode; the leave-one-cell-out average never holds the window's own cell; read-derived
QC never enters C'.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import h5py
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
DRAWS = 10_000

#: sha256 of 200 training batches (seed 3, `iter`) and of every validation batch, computed with
#: the dataset code BEFORE the arms were added (umbrella 6d5dac3), on these synthetic tables
REFERENCE = {
    ("plain", "train"): "a6953ed2f46e39a6968cda91505f9c51f42145e001cc1f1ef35f1a16b518bc81",
    ("plain", "val"): "afa051947a88b376715346cab55464601b523a4b017b3f5dd89aa68d5a315117",
    ("existing_keys", "train"): "61706912379e55511a6d7f03881661da38ed643c2e476ad768c2f7a3e4fd5bda",
    ("existing_keys", "val"): "72cb53a509c18db07d3b8e934414791859b44049cce5b72853f31cec1b06ec43",
}
REF_CONFIGS = {"plain": {"context_bins": L},
               "existing_keys": {"context_bins": L, "batch_size": 4, "qc_on_input": True,
                                 "two_step": True, "base_depth": "fixed",
                                 "average_input": True}}
NEW_KEYS_AT_DEFAULT = {"input_depth": "native", "mask_scheme": "uniform",
                       "bin_mask_shape": "random", "train_on_validation": False,
                       "dna_embedding_h5": None, "dna_shift_bp": 0, "qc_table": None}


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


def _rows(tables):
    return {r["measurement_id"]: r for r in csv.DictReader(
        open(tables / "measurements.tsv", encoding="utf-8"), delimiter="\t")}


def _withheld(tables):
    return D.Withheld(list(csv.DictReader(open(tables / "withheld.tsv"), delimiter="\t")))


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


def _digest(batches) -> str:
    h = hashlib.sha256()
    for b in batches:
        for k in sorted(b):
            h.update(k.encode())
            if k == "meta":
                h.update(json.dumps(b[k], sort_keys=True).encode())
            else:
                t = b[k].contiguous()
                h.update(str(t.dtype).encode() + str(tuple(t.shape)).encode())
                h.update(t.numpy().tobytes())
    return h.hexdigest()


def _digests(tables, cfg) -> dict:
    it = iter(D.OnionDataset(tables, "train", cfg, seed=3))
    return {"train": _digest(next(it) for _ in range(200)),
            "val": _digest(D.OnionDataset(tables, "val", dict(cfg, region_bins=4 * L), seed=3))}


# ------------------------------------------------------------------------------- defaults


@pytest.mark.parametrize("name", sorted(REF_CONFIGS))
def test_defaults_give_the_batches_of_the_code_before_the_arms(tables, name):
    got = _digests(tables, REF_CONFIGS[name])
    assert got == {s: REFERENCE[(name, s)] for s in ("train", "val")}
    explicit = _digests(tables, dict(REF_CONFIGS[name], **NEW_KEYS_AT_DEFAULT))
    assert explicit == got


def test_unknown_arm_values_raise(tables):
    for key in ("input_depth", "mask_scheme", "bin_mask_shape"):
        with pytest.raises(ValueError):
            D.OnionDataset(tables, "train", {"context_bins": L, key: "nope"})


# ------------------------------------------------------------------------------- input depth


def test_input_depth_fixed_never_raises_depth_and_c_in_states_the_thinned_depth(tables):
    cfg = {"context_bins": L, "batch_size": 16}
    nat = D.OnionDataset(tables, "train", cfg, seed=8)
    fix = D.OnionDataset(tables, "train", dict(cfg, input_depth="fixed"), seed=8)
    n_thin, ratios = 0, []
    for _ in range(10):
        a, b = nat.batch(16), fix.batch(16)                 # same plans: same generator draws
        C.validate_batch(b)
        for k in ("x_avail", "y_avail", "y_kind", "y_counts", "c_out", "x_bin_mask"):
            assert torch.equal(a[k], b[k]), k               # targets and C' untouched
        assert (b["x_counts"] <= a["x_counts"]).all()        # never deeper
        av = a["x_avail"]
        ref = torch.tensor([[D.FIXED_BASE_LOG2_DEPTH[C.assay_class(t)] for t in C.TRACKS]]
                           ).expand_as(av)
        d_nat, d_fix = a["c_in"][..., 0], b["c_in"][..., 0]
        assert torch.equal(d_fix[av], torch.minimum(d_nat, ref)[av])
        assert torch.equal(a["c_in"][..., 1:], b["c_in"][..., 1:])
        thin = av & (d_nat > ref)
        same = av & ~thin
        assert torch.equal(a["x_counts"][same[:, None, :].expand_as(a["x_counts"])],
                           b["x_counts"][same[:, None, :].expand_as(b["x_counts"])])
        for bi, t in thin.nonzero().tolist():
            n_thin += 1
            s_nat = int(a["x_counts"][bi, :, t].sum())
            if s_nat > 200:
                q = 2.0 ** float(ref[bi, t] - d_nat[bi, t])
                ratios.append(int(b["x_counts"][bi, :, t].sum()) / s_nat / q)
    assert n_thin > 50 and len(ratios) > 20
    assert abs(np.mean(ratios) - 1.0) < 0.05               # binomial thinning at the right q


def test_input_depth_fixed_on_validation_halves(tables):
    rows = _rows(tables)
    vds = D.OnionDataset(tables, "val", {"context_bins": L, "input_depth": "fixed"})
    deep = [m for m, r in rows.items() if r["in_eic"] == "val"
            and np.log2(float(r["depth_reads"])) - 1.0 > 23.0]
    assert deep
    m = deep[0]
    t = C.TRACK_INDEX[rows[m]["assay"]]
    w = vds.window("chr1", 100, [m], [m], input_file="pr1", target_file="pr2")
    assert w["c_in"][t, 0] == 23.0
    assert (w["x_counts"][:, t].numpy() <= vds.read(m, "chr1", 100, which="pr1")).all()
    assert np.array_equal(w["y_counts"][:, t].numpy(), vds.read(m, "chr1", 100, which="pr2"))
    again = vds.window("chr1", 100, [m], [m], input_file="pr1", target_file="pr2")
    assert torch.equal(w["x_counts"], again["x_counts"])  # seeded by measurement and locus


# ------------------------------------------------------------------------------- masking


@pytest.mark.parametrize("scheme", ["uniform", "dirichlet", "panel"])
def test_no_scheme_samples_a_withheld_combination_or_a_heldout_row(tables, scheme):
    rows, wh = _rows(tables), _withheld(tables)
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "mask_scheme": scheme}, seed=0)
    rng = np.random.default_rng(321)
    kinds = {1: 0, 2: 0, 3: 0}
    n_masked = []
    for _ in range(DRAWS):
        p = ds.plan(rng)
        ins = [e[1] for e in p["entries"] if e[1] is not None]
        outs = [e[2] for e in p["entries"]]
        assert all(rows[m]["in_eic"] == "train" for m in ins + outs)
        assert not _violations(rows, wh, ins, outs)
        assert ins                                          # at least one input per window
        for e in p["entries"]:
            kinds[e[3]] += 1
        n_masked.append(sum(e[3] == C.KIND_IMPUTE for e in p["entries"]))
    assert min(kinds.values()) > 500, kinds
    if scheme == "dirichlet":                              # alpha = 1: the share is uniform,
        assert np.mean(np.asarray(n_masked) == 0) > 0.05   # so all-visible windows occur


def test_panel_copies_another_training_cells_missing_assays(tables):
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "mask_scheme": "panel"}, seed=0)
    held = {c: {a for cc, a in ds.train_tracks if cc == c} for c in ds.train_cells}
    rng = np.random.default_rng(5)
    seen = set()
    for _ in range(2000):
        cell = ds.train_cells[int(rng.integers(len(ds.train_cells)))]
        order = sorted(held[cell])
        masked = frozenset(ds._masked_set(cell, order, rng))
        allowed = {frozenset(held[cell] - held[d]) for d in ds.train_cells if d != cell}
        assert masked in allowed
        seen.add(masked)
    assert len(seen) >= 3                                  # several donor patterns are copied


def test_bin_mask_span_is_spans_in_training_and_random_in_validation(tables):
    rows, wh = _rows(tables), _withheld(tables)
    cfg = {"context_bins": L, "bin_mask_shape": "span", "bin_mask_span_bins": 8}
    ds = D.OnionDataset(tables, "train", cfg, seed=1)
    fracs = []
    for _ in range(20):
        b = ds.batch(16)
        C.validate_batch(b)
        same = b["y_kind"] == C.KIND_SAME
        for bi, t in same.nonzero().tolist():
            col = b["x_bin_mask"][bi, :, t].numpy()
            assert col.any() and not col.all()
            runs = np.diff(np.flatnonzero(np.diff(np.r_[0, col.astype(int), 0])))[::2]
            assert (runs >= 8).all(), runs                 # contiguous spans, never single bins
            fracs.append(col.mean())
        for m in b["meta"]:
            assert all(rows[x]["in_eic"] == "train" for x in m["inputs"] + m["targets"])
            assert not _violations(rows, wh, m["inputs"], m["targets"])
    assert 0.12 < np.mean(fracs) < 0.3                     # about bin_mask_frac = 0.25
    vcfg = {"context_bins": L, "region_bins": 4 * L}
    assert (_digest(D.OnionDataset(tables, "val", dict(vcfg, bin_mask_shape="span"), seed=3))
            == REFERENCE[("plain", "val")])


# ------------------------------------------------------------------------------- roles


def test_train_on_validation_needs_final_model(tables):
    for mode in ("train", "val"):
        with pytest.raises(D.RoleError):
            D.OnionDataset(tables, mode, {"context_bins": L, "train_on_validation": True})
    with pytest.raises(D.RoleError):
        D.OnionDataset(tables, "train", {"context_bins": L, "final_model": True,
                                         "train_on_validation": True, "allow_blind": True}
                       ).read(next(m for m, r in _rows(tables).items()
                                   if r["in_eic"] == "blind"), "chr1", 0)


def test_train_on_validation_reads_val_rows_and_never_blind_rows(tables):
    rows, wh = _rows(tables), _withheld(tables)
    cfg = {"context_bins": L, "train_on_validation": True, "final_model": True,
           "allow_blind": True}
    ds = D.OnionDataset(tables, "train", cfg, seed=0)
    rng = np.random.default_rng(99)
    n_val = 0
    for _ in range(DRAWS):
        p = ds.plan(rng)
        ins = [e[1] for e in p["entries"] if e[1] is not None]
        outs = [e[2] for e in p["entries"]]
        roles = {rows[m]["in_eic"] for m in ins + outs}
        assert "blind" not in roles
        n_val += "val" in roles
        assert not _violations(rows, wh, ins, outs)       # withheld holds on val tracks too
    assert n_val > 500
    blind = next(m for m, r in rows.items() if r["in_eic"] == "blind")
    with pytest.raises(D.RoleError):
        ds.window("chr1", 0, [], [blind])
    for _ in range(50):
        m = ds.batch(4)["meta"]
        assert all(rows[x]["in_eic"] != "blind" for w in m for x in w["inputs"] + w["targets"])


# ------------------------------------------------------------------------------- DNA


def _write_emb(tables, d=6, layout="dir"):
    sizes = json.loads((tables.parent.parent / "genome" / "chrom_sizes.json").read_text())
    rng = np.random.default_rng(0)
    arrays = {c: rng.standard_normal((s // C.RES, d)).astype(np.float16)
              for c, s in sizes.items()}
    root = tables.parent.parent / f"dna_emb_{layout}"
    if layout == "dir":
        root.mkdir(exist_ok=True)
        for c, v in arrays.items():
            with h5py.File(root / f"{c}.h5", "w") as f:
                f.create_dataset("emb", data=v, chunks=(min(4096, v.shape[0]), d))
    else:
        with h5py.File(root.with_suffix(".h5"), "w") as f:
            for c, v in arrays.items():
                f.create_dataset(c, data=v)
        root = root.with_suffix(".h5")
    return root, arrays


@pytest.mark.parametrize("layout", ["dir", "file"])
def test_x_dna_emb_has_its_shape_and_equals_the_h5_slice(tables, layout):
    path, arrays = _write_emb(tables, layout=layout)
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "dna_embedding_h5": str(path)},
                        seed=2)
    b = ds.batch(5)
    C.validate_batch(b)
    assert b["x_dna_emb"].shape == (5, L, 6) and b["x_dna_emb"].dtype == torch.float32
    for bi, m in enumerate(b["meta"]):
        want = arrays[m["chrom"]][m["start_bin"]:m["start_bin"] + L].astype(np.float32)
        assert np.array_equal(b["x_dna_emb"][bi].numpy(), want)
    vb = next(iter(D.OnionDataset(tables, "val", {"context_bins": L, "region_bins": 4 * L,
                                                  "dna_embedding_h5": str(path)})))
    assert vb["x_dna_emb"].shape[1:] == (L, 6)
    plain = D.OnionDataset(tables, "train", {"context_bins": L}, seed=2).batch(5)
    assert "x_dna_emb" not in plain
    with pytest.raises(FileNotFoundError):
        D.OnionDataset(tables, "train", {"context_bins": L,
                                         "dna_embedding_h5": str(path) + ".missing"})


def test_dna_shift_reads_the_same_chromosome_elsewhere(tables):
    with h5py.File(tables.parent.parent / "genome" / "dna.h5", "r") as f:
        chr1 = f["chr1"][:]
    size = chr1.shape[0]
    shift = 1000 * C.RES + 7                               # not a whole bin, and wraps below
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "dna_shift_bp": shift})
    for start in (0, 300, size // C.RES - L):
        got = ds.dna("chr1", start)
        idx = (start * C.RES + shift + np.arange(L * C.RES)) % size
        assert np.array_equal(got, chr1[idx])
        assert not np.array_equal(got, chr1[start * C.RES:(start + L) * C.RES])
    neg = D.OnionDataset(tables, "train", {"context_bins": L, "dna_shift_bp": -shift})
    idx = (300 * C.RES - shift + np.arange(L * C.RES)) % size
    assert np.array_equal(neg.dna("chr1", 300), chr1[idx])
    path, arrays = _write_emb(tables)
    eds = D.OnionDataset(tables, "train", {"context_bins": L, "dna_shift_bp": shift,
                                           "dna_embedding_h5": str(path)})
    n = arrays["chr1"].shape[0]
    start = n - L                                          # the shifted rows wrap
    assert np.array_equal(eds.dna_emb("chr1", start), arrays["chr1"][
        (start + shift // C.RES + np.arange(L)) % n].astype(np.float32))


# ------------------------------------------------------------------------------- QC


def _write_qc_table(tables, path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(D.QC_TABLE_COLUMNS)
        for m in rows:
            w.writerow([m, "1.5", "2.5", "0.11", "0.22", "33", "", "321"])
        w.writerow(["not_a_measurement"] + ["1"] * 7)


def test_qc_table_changes_c_in_qc_columns_only_and_never_c_out(tables, tmp_path):
    rows = _rows(tables)
    qc = tmp_path / "qc.tsv"
    _write_qc_table(tables, qc, sorted(rows))
    for qc_on in (True, False):
        cfg = {"context_bins": L, "qc_on_input": qc_on, "batch_size": 8}
        a = D.OnionDataset(tables, "train", cfg, seed=6).batch(8)
        b = D.OnionDataset(tables, "train", dict(cfg, qc_table=str(qc)), seed=6).batch(8)
        for k in a:
            if k not in ("c_in", "meta"):
                assert torch.equal(a[k], b[k]), k          # c_out included: never QC
        assert torch.equal(a["c_in"][..., :C.K_META], b["c_in"][..., :C.K_META])
        if qc_on:
            av = b["x_avail"]
            assert not torch.equal(a["c_in"][..., C.K_META:][av], b["c_in"][..., C.K_META:][av])
            want = torch.tensor([1.5, 2.5, 0.11, 0.22, 33.0, 0.2, 321.0])  # frip empty: centre
            assert torch.allclose(b["c_in"][..., C.K_META:][av], want.expand(int(av.sum()), 7))
        else:
            assert torch.equal(a["c_in"], b["c_in"])
    ds = D.OnionDataset(tables, "train", {"context_bins": L, "qc_table": str(qc)})
    assert ds.qc_table_rows == len(rows) and ds.qc_table_unknown == 1
    bad = tmp_path / "bad.tsv"
    bad.write_text("measurement_id\tnsc\n", encoding="utf-8")
    with pytest.raises(ValueError):
        D.OnionDataset(tables, "train", {"context_bins": L, "qc_table": str(bad)})


# ------------------------------------------------------------------------------- average


def _record_rates(ds):
    seen = []
    orig = ds._rate

    def rec(mid, *a, **k):
        seen.append(mid)
        return orig(mid, *a, **k)
    ds._rate = rec
    return seen


def test_average_input_never_reads_the_windows_own_cell_or_a_heldout_row(tables):
    rows = _rows(tables)
    cfg = {"context_bins": L, "average_input": True, "region_bins": 4 * L, "batch_size": 4}
    for mode in ("train", "val"):
        ds = D.OnionDataset(tables, mode, cfg, seed=0)
        seen = _record_rates(ds)
        it = iter(ds)
        for _ in range(10):
            b = next(it)
            for m in b["meta"]:
                before = len(seen)
                ds.average(m["cell"], m["chrom"], m["start_bin"])
                got = seen[before:]
                assert got
                assert all(rows[x]["cell"] != m["cell"] for x in got)
                assert not set(got) & set(m["targets"])
                assert all(rows[x]["in_eic"] == "train" for x in got)


def test_average_under_train_on_validation_and_its_h5_guard(tables, tmp_path):
    rows = _rows(tables)
    tv = {"context_bins": L, "train_on_validation": True, "final_model": True}
    ds = D.OnionDataset(tables, "train", tv)
    t = C.TRACK_INDEX["H3K4me3"]
    val_s0 = next(m for m, r in rows.items() if r["cell"] == "s0" and r["assay"] == "H3K4me3"
                  and not r["arm_knob"])
    assert rows[val_s0]["in_eic"] == "val"
    for cell in ("s0", "s1"):
        others = [m for m, r in rows.items() if r["in_eic"] in ("train", "val")
                  and r["cell"] != cell and r["assay"] == "H3K4me3" and not r["arm_knob"]]
        assert (val_s0 in others) == (cell != "s0")
        assert all(rows[m]["in_eic"] != "blind" for m in others)
        want = np.mean([ds.read(m, "chr1", 200) * 1e6 / float(rows[m]["depth_reads"])
                        for m in others], axis=0)
        np.testing.assert_allclose(ds.average(cell, "chr1", 200)[:, t], want, rtol=1e-5)
    h5_tv = D.build_average_h5(tables, tmp_path / "avg_tv.h5", tv)
    with h5py.File(h5_tv, "r") as f:
        summed = [str(x) for x in f["H3K4me3"].attrs["measurement_ids"]]
    assert val_s0 in summed and all(rows[m]["in_eic"] != "blind" for m in summed)
    ds_h5 = D.OnionDataset(tables, "train", dict(tv, average_h5=str(h5_tv)))
    for cell in ("s0", "s1", "s5"):
        np.testing.assert_allclose(ds_h5.average(cell, "chr1", 200), ds.average(cell, "chr1", 200),
                                   rtol=1e-4, atol=1e-3)
    for mode in ("train", "val"):                          # val sums must not reach plain runs
        plain = D.OnionDataset(tables, mode, {"context_bins": L, "average_h5": str(h5_tv)})
        with pytest.raises(D.RoleError):
            plain.average("s1", "chr1", 200)
