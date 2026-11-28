"""t392 chunk C1, case (3): the read-start dataset behind `res_dataset.make_dataset`.

Written against `cruxvault/INSTRUCTIONS/t392_read_starts/INSTRUCTIONS.md` section 3 (work item 1:
strand channel through the store, the input contract and the average input; `res_dataset` must
accept `average_h5`), decisions D11 and D14, and the Interfaces lines of plan chunk C6, before
`tools/minicandi/rs_dataset.py` existed.

The fixture builds synthetic tables (`minicandi.synth`) with ATAC-seq, DNase-seq, two histone
marks and the control, and a hand-made read-start store for every measurement: per chromosome
`[n_bins, 2]` starts for `starts_pr1.h5` and `starts_pr2.h5`, and `starts.h5` their sum. The
store's starts are random, not derived from the overlap counts: the dataset only has to carry
them. The store is written with both strands for every assay (D11).
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

L = 64
STORE_FILES = {"counts": "starts.h5", "pr1": "starts_pr1.h5", "pr2": "starts_pr2.h5"}
UNSTRANDED = ("ATAC-seq", "DNase-seq")


def _C():
    from minicandi import contract
    return contract


def _write_starts(path: Path, arrays: dict, mid: str, depth: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for c, v in arrays.items():
            f.create_dataset(c, data=np.asarray(v, np.uint16), chunks=(min(4096, len(v)), 2),
                             compression="gzip", compression_opts=4, shuffle=True)
        f.attrs.update({"schema": 1, "counting_rule": "read_start", "resolution": 25,
                        "strand_columns": "plus,minus", "measurement_id": mid,
                        "depth_reads": depth})


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import build_average_h5 as onion_average_h5
    from minicandi.dataset import read_tsv
    from minicandi.synth import make_synthetic_tables
    C = _C()
    root = tmp_path_factory.mktemp("rs_ds")
    tables = make_synthetic_tables(
        root, n_cells=5, assays=("ATAC-seq", "DNase-seq", "H3K27ac", "H3K4me3", C.CONTROL),
        val_tracks=((0, "H3K4me3"),), blind_tracks=(), chrom_bins={"chr1": 1024, "chr2": 512})
    rows = read_tsv(tables / "measurements.tsv")
    store = root / "rs_store"
    rng = np.random.default_rng(0)
    starts = {}
    for r in rows:
        mid = r["measurement_id"]
        counts_h5 = Path(r["counts_path"].split(":", 1)[1])
        with h5py.File(counts_h5, "r") as f:
            n_bins = {c: f[c].shape[0] for c in f.keys()}
        lam = 2.0 if r["assay"] in UNSTRANDED else 1.0
        pr1 = {c: rng.poisson(lam, (n, 2)) for c, n in n_bins.items()}
        pr2 = {c: rng.poisson(lam, (n, 2)) for c, n in n_bins.items()}
        both = {c: pr1[c] + pr2[c] for c in n_bins}
        depth = float(r["depth_reads"] or 0) or 1.0
        for which, arr in (("counts", both), ("pr1", pr1), ("pr2", pr2)):
            _write_starts(store / mid / STORE_FILES[which], arr, mid, depth)
        starts[mid] = {"counts": both, "pr1": pr1, "pr2": pr2}
    ref_avg = onion_average_h5(tables, root / "eic_v1_average.h5")
    from minicandi.rs_dataset import build_average_h5
    rs_avg = Path(build_average_h5(ref_avg, store, tables, root / "rs_average.h5") or
                  root / "rs_average.h5")
    cfg = {"context_bins": L, "batch_size": 4, "average_input": True,
           "average_h5": str(ref_avg), "read_start": True, "read_start_store": str(store),
           "read_start_average_h5": str(rs_avg)}
    return {"root": root, "tables": tables, "rows": {r["measurement_id"]: r for r in rows},
            "store": store, "starts": starts, "ref_avg": ref_avg, "rs_avg": rs_avg, "cfg": cfg}


def _make(env, extra=None, seed=0, mode="train"):
    from minicandi.res_dataset import make_dataset
    return make_dataset(env["tables"], mode, {**env["cfg"], **(extra or {})}, seed)


@pytest.fixture(scope="module")
def batches(env):
    ds = _make(env, {"train_halves": "pairs"}, seed=3)
    it = iter(ds)
    return ds, [next(it) for _ in range(4)]


# --------------------------------------------------------------------------- entry and batch keys

def test_make_dataset_returns_the_read_start_dataset(env):
    from minicandi.dataset import OnionDataset
    from minicandi.rs_dataset import ReadStartDataset
    ds = _make(env)
    assert isinstance(ds, ReadStartDataset) and isinstance(ds, OnionDataset)


def test_batches_carry_the_strand_keys_with_contract_shapes(env, batches):
    _, bs = batches
    B = env["cfg"]["batch_size"]
    for b in bs:
        for key, dt in (("x_counts_sub", torch.int32), ("y_counts_sub", torch.int32),
                        ("avg_sub", torch.float32)):
            assert key in b, key
            assert tuple(b[key].shape) == (B, L, 36, 2), key
            assert b[key].dtype == dt, key
        assert "avg" in b and tuple(b["avg"].shape) == (B, L, 36)


def test_channel_sums_equal_the_counts_and_the_average(batches):
    _, bs = batches
    for b in bs:
        assert torch.equal(b["x_counts_sub"].sum(-1).to(b["x_counts"].dtype), b["x_counts"])
        assert torch.equal(b["y_counts_sub"].sum(-1).to(b["y_counts"].dtype), b["y_counts"])
        torch.testing.assert_close(b["avg_sub"].sum(-1), b["avg"].float(), rtol=1e-5, atol=1e-6)


def test_unstranded_tracks_have_all_starts_in_channel_zero(batches):
    C = _C()
    _, bs = batches
    seen = 0
    for b in bs:
        for a in UNSTRANDED:
            t = C.TRACK_INDEX[a]
            for key, tot in (("x_counts_sub", "x_counts"), ("y_counts_sub", "y_counts")):
                assert int(b[key][..., t, 1].abs().sum()) == 0, (a, key)
                assert torch.equal(b[key][..., t, 0].to(b[tot].dtype), b[tot][..., t])
                seen += int(b[tot][..., t].sum())
    assert seen > 0, "no unstranded track carried any start: the test would pass vacuously"


def test_stranded_tracks_keep_both_strands(batches):
    C = _C()
    _, bs = batches
    t = C.TRACK_INDEX["H3K27ac"]
    assert any(int(b["y_counts_sub"][..., t, 1].sum()) > 0 for b in bs)


def test_hidden_and_missing_inputs_are_zero_in_both_channels(batches):
    _, bs = batches
    for b in bs:
        hidden = b["x_bin_mask"] | ~b["x_avail"][:, None, :]
        assert int(b["x_counts_sub"][hidden].abs().sum()) == 0
        assert int(b["y_counts_sub"][~b["y_avail"][:, None, :].expand_as(b["y_counts"])]
                   .abs().sum()) == 0


def test_window_strands_are_the_store_slices_of_the_right_file(env, batches):
    """With train_halves pairs, a halved track reads pr1 as input and pr2 as target."""
    C = _C()
    _, bs = batches
    checked = 0
    for b in bs:
        for i, meta in enumerate(b["meta"]):
            halved = set(meta.get("halves_tracks") or ())
            s0, chrom = meta["start_bin"], meta["chrom"]
            for t in range(C.A):
                a = C.TRACKS[t]
                for key, mids, default, half in (
                        ("x_counts_sub", meta["measurements_in"], meta["input_file"], "pr1"),
                        ("y_counts_sub", meta["measurements_out"], meta["target_file"], "pr2")):
                    mid = mids[t]
                    if not mid:
                        continue
                    which = half if a in halved else default
                    want = env["starts"][mid][which][chrom][s0:s0 + L].astype(np.int64)
                    if a in UNSTRANDED:
                        want = np.stack([want.sum(1), np.zeros(len(want), np.int64)], 1)
                    got = b[key][i, :, t, :].numpy().astype(np.int64)
                    if key == "x_counts_sub":
                        got = got[~b["x_bin_mask"][i, :, t].numpy()]
                        want = want[~b["x_bin_mask"][i, :, t].numpy()]
                    np.testing.assert_array_equal(got, want, err_msg=f"{key} {a} {mid}")
                    checked += 1
    assert checked > 0


def test_read_is_the_strand_sum_and_read_strands_the_store_slice(env):
    ds = _make(env)
    mid = next(iter(env["starts"]))
    for which in ("counts", "pr1", "pr2"):
        want = env["starts"][mid][which]["chr1"][100:100 + L]
        got = ds.read_strands(mid, "chr1", 100, L, which)
        assert got.dtype == np.int32 and got.shape == (L, 2)
        np.testing.assert_array_equal(got, want.astype(np.int32))
        r = ds.read(mid, "chr1", 100, L, which)
        assert r.dtype == np.int32 and r.shape == (L,)
        np.testing.assert_array_equal(r, want.sum(1).astype(np.int32))


# --------------------------------------------------------------------------- the average input

def _expected_avg(env, ds, cell, chrom, start):
    """Leave-one-cell-out mean over training cells' base views of starts x 1e6 / depth_reads."""
    C = _C()
    out = np.zeros((L, C.A, 2), np.float64)
    by_track = {}
    for (c, a), m in ds.train_base.items():
        by_track.setdefault(a, []).append((c, m))
    for a, cm in by_track.items():
        others = [m for c, m in cm if c != cell and float(ds.rows[m]["depth_reads"] or 0)]
        if not others:
            continue
        acc = np.zeros((L, 2))
        for m in others:
            depth = float(ds.rows[m]["depth_reads"])
            acc += env["starts"][m]["counts"][chrom][start:start + L] * 1e6 / depth
        out[:, C.TRACK_INDEX[a], :] = acc / len(others)
    return out


def test_average_strands_leaves_the_windows_own_cell_out(env):
    ds = _make(env)
    cells = sorted({c for c, _ in ds.train_base})
    for cell in cells[:3]:
        got = np.asarray(ds.average_strands(cell, "chr1", 200, L), np.float64)
        want = _expected_avg(env, ds, cell, "chr1", 200)
        assert got.shape == want.shape
        np.testing.assert_allclose(got.sum(-1), want.sum(-1), rtol=1e-4, atol=1e-4)
        C = _C()
        stranded = [i for i, a in enumerate(C.TRACKS) if a not in UNSTRANDED]
        np.testing.assert_allclose(got[:, stranded], want[:, stranded], rtol=1e-4, atol=1e-4)


def _ids(attr) -> set:
    return {x.decode() if isinstance(x, bytes) else str(x) for x in np.atleast_1d(attr)}


def test_read_start_average_file_copies_the_sets_of_the_read_count_file(env):
    with h5py.File(env["ref_avg"], "r") as fr, h5py.File(env["rs_avg"], "r") as fs:
        assert set(fs.keys()) == set(fr.keys())
        for track in fr.keys():
            gr, gs = fr[track], fs[track]
            assert int(gs.attrs["n_cells"]) == int(gr.attrs["n_cells"]), track
            assert _ids(gs.attrs["measurement_ids"]) == _ids(gr.attrs["measurement_ids"]), track


def test_read_start_average_copies_a_non_default_set_from_the_reference(env, tmp_path):
    """The sets are read from the reference file, not recomputed from the tables: drop one
    measurement of one track in a copy of the reference and the read-start file follows."""
    import shutil
    from minicandi.rs_dataset import build_average_h5
    ref = tmp_path / "eic_v1_average_mod.h5"
    shutil.copy(env["ref_avg"], ref)
    track = "H3K27ac"
    with h5py.File(ref, "r+") as f:
        g = f[track]
        mids = sorted(_ids(g.attrs["measurement_ids"]))
        assert len(mids) >= 2
        kept = mids[:-1]
        g.attrs["measurement_ids"] = np.asarray(kept, dtype=h5py.string_dtype())
        g.attrs["n_cells"] = int(g.attrs["n_cells"]) - 1
        n_cells = int(g.attrs["n_cells"])
    out = tmp_path / "rs_average_mod.h5"
    out = Path(build_average_h5(ref, env["store"], env["tables"], out) or out)
    with h5py.File(out, "r") as fs:
        g = fs[track]
        assert _ids(g.attrs["measurement_ids"]) == set(kept)
        assert int(g.attrs["n_cells"]) == n_cells
        want = sum(env["starts"][m]["counts"]["chr1"] * 1e6 / float(env["rows"][m]["depth_reads"])
                   for m in kept)
        np.testing.assert_allclose(np.asarray(g["chr1"], np.float64), want, rtol=1e-5)


def test_read_start_average_file_holds_summed_strand_rates(env):
    with h5py.File(env["rs_avg"], "r") as fs:
        for track in fs.keys():
            g = fs[track]
            mids = sorted(_ids(g.attrs["measurement_ids"]))
            for chrom in ("chr1", "chr2"):
                d = g[chrom]
                assert d.dtype == np.float32
                n = env["starts"][mids[0]]["counts"][chrom].shape[0]
                assert d.shape == (n, 2)
                want = sum(env["starts"][m]["counts"][chrom] * 1e6
                           / float(env["rows"][m]["depth_reads"]) for m in mids)
                np.testing.assert_allclose(np.asarray(d, np.float64), want, rtol=1e-5)


# --------------------------------------------------------------------------- the sampler

def _seq(batch):
    out = []
    for i, m in enumerate(batch["meta"]):
        out.append((m["cell"], m["chrom"], int(m["start_bin"]), tuple(m["measurements_in"]),
                    tuple(m["measurements_out"]), batch["x_bin_mask"][i].numpy().tobytes()))
    return out


@pytest.mark.parametrize("extra", [{}, {"train_halves": "pairs"}], ids=["plain", "halves"])
def test_window_sequence_per_seed_equals_onion_dataset(env, extra):
    from minicandi.dataset import OnionDataset
    base = {k: v for k, v in env["cfg"].items() if not k.startswith("read_start")}
    for seed in (0, 1):
        od = iter(OnionDataset(env["tables"], "train", {**base, **extra}, seed))
        rd = iter(_make(env, extra, seed=seed))
        for _ in range(3):
            assert _seq(next(rd)) == _seq(next(od))


# --------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("extra", [
    {"input_depth": "fixed"}, {"aug_revcomp": True}, {"two_step": True}, {"res_bp": 50},
    {"res_in_sub": 5}, {"res_out_sub": 5},
], ids=lambda e: next(iter(e)))
def test_read_start_dataset_refuses_what_it_cannot_serve(env, extra):
    """Built directly, so the point-rule ResDataset's own refusal of average_h5 cannot be what
    raises (make_dataset would route res_bp / res_*_sub configs there)."""
    from minicandi.rs_dataset import ReadStartDataset
    with pytest.raises(ValueError) as e:
        ReadStartDataset(env["tables"], "train", {**env["cfg"], **extra}, 0)
    assert "['average_h5']" not in str(e.value)


def test_read_start_dataset_refuses_a_store_row(env, tmp_path):
    import shutil
    from minicandi.dataset import read_tsv
    from minicandi.rs_dataset import ReadStartDataset
    src = Path(env["tables"])
    dst = src.parent / (src.name + "_storerow")
    if not dst.exists():
        shutil.copytree(src, dst)
    rows = read_tsv(dst / "measurements.tsv")
    cols = list(rows[0])
    i = next(k for k, r in enumerate(rows) if r["assay"] == "H3K27ac")
    rows[i]["counts_path"] = f"store:{tmp_path / 'corpus'}:{rows[i]['cell']}:H3K27ac"
    with open(dst / "measurements.tsv", "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(r[c] for c in cols) + "\n")
    from minicandi.dataset import OnionDataset
    base = {k: v for k, v in env["cfg"].items() if not k.startswith("read_start")}
    OnionDataset(dst, "train", {**base, "average_input": False, "average_h5": None}, 0)
    # the parent accepts the table, so the refusal below is the read-start dataset's own
    with pytest.raises(ValueError):
        ReadStartDataset(dst, "train", env["cfg"], 0)
