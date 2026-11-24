"""t392 chunk C9: the read-start adapter between the model and the overlap-count scorer.

Pinned by plan chunk C9 (Interfaces and Tests), INSTRUCTIONS section 3 "Fair scoring" and
decision D9: a prediction converts to overlap counts with each measurement's measured average span
kernel. The batches are built by the scorer's own dataset (plain `OnionDataset`), the inner model
is a stub that records its input and returns fixed `mu_sub`, `n_sub`, and the expected output is
computed here a second way: a numpy convolution of the edge-padded starts with the kernel, and
the moment match var = sum of (w mu) + (w mu)^2 / n.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import read_start as RS  # noqa: E402

L = 64
STORE_FILES = {"counts": "starts.h5", "pr1": "starts_pr1.h5", "pr2": "starts_pr2.h5"}
UNSTRANDED = ("ATAC-seq", "DNase-seq")


def _write(path: Path, arrays: dict, mid: str, depth: float, kernel, n_reads) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for c, v in arrays.items():
            f.create_dataset(c, data=np.asarray(v, np.uint16))
        f.attrs.update({"schema": 1, "counting_rule": "read_start", "resolution": 25,
                        "strand_columns": "plus,minus", "measurement_id": mid,
                        "depth_reads": depth, "kernel": np.asarray(kernel, np.float64),
                        "n_reads_strand": np.asarray(n_reads, np.int64)})


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import build_average_h5 as onion_average_h5
    from minicandi.dataset import read_tsv
    from minicandi.rs_dataset import build_average_h5
    from minicandi.synth import make_synthetic_tables
    root = tmp_path_factory.mktemp("rs_adapter")
    tables = make_synthetic_tables(
        root, n_cells=4, assays=("ATAC-seq", "DNase-seq", "H3K27ac", "H3K4me3", C.CONTROL),
        val_tracks=((0, "H3K4me3"),), blind_tracks=(), chrom_bins={"chr1": 1024, "chr2": 512})
    rows = read_tsv(tables / "measurements.tsv")
    store = root / "rs_store"
    rng = np.random.default_rng(1)
    for r in rows:
        mid = r["measurement_id"]
        with h5py.File(r["counts_path"].split(":", 1)[1], "r") as f:
            n_bins = {c: f[c].shape[0] for c in f.keys()}
        depth = float(r["depth_reads"] or 0) or 1.0
        for which in STORE_FILES:
            arr = {c: rng.poisson(2.0, (n, 2)) + 1 for c, n in n_bins.items()}   # never 0
            kern = rng.uniform(0.05, 1.0, (2, 17))
            nr = rng.integers(1000, 9000, 2)
            _write(store / mid / STORE_FILES[which], arr, mid, depth, kern, nr)
    ref_avg = onion_average_h5(tables, root / "eic_v1_average.h5")
    rs_avg = root / "rs_average.h5"
    build_average_h5(ref_avg, store, tables, rs_avg)
    cfg = {"context_bins": L, "batch_size": 8, "average_input": True,
           "average_h5": str(ref_avg), "read_start": True, "read_start_store": str(store),
           "read_start_average_h5": str(rs_avg)}
    by = {(r["cell"], r["assay"]): r["measurement_id"] for r in rows}
    return {"tables": tables, "store": store, "cfg": cfg, "by": by}


class Stub(C.MiniCandi):
    """Records its input batch; returns fixed positive `mu_sub`, `n_sub` [B, L, A, 2]."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.seen = None
        self.out = None

    def forward(self, batch):
        self.seen = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
        B, Lb, A = batch["x_bin_mask"].shape
        g = torch.Generator().manual_seed(7)
        mu = 0.2 + 3.0 * torch.rand(B, Lb, A, 2, generator=g, dtype=torch.float64)
        n = 0.3 + 5.0 * torch.rand(B, Lb, A, 2, generator=g, dtype=torch.float64)
        self.out = {"mu_sub": mu, "n_sub": n}
        return dict(self.out, mu=mu.sum(-1), n=n.sum(-1))


def _adapter(env, **extra):
    from minicandi.rs_adapter import ReadStartAdapter
    return ReadStartAdapter(Stub(), {**env["cfg"], **extra}, tables=env["tables"]).eval()


def _scorer_ds(env):
    from minicandi.dataset import OnionDataset
    return OnionDataset(env["tables"], "val", env["cfg"], seed=0)


def _batch(env, starts, inputs, targets, chrom="chr1", mask_seed=0, **kw):
    ds = _scorer_ds(env)
    rng = np.random.default_rng(mask_seed)
    wins = [ds.window(chrom, s, inputs, targets, x_bin_mask=rng.random((L, C.A)) < 0.3, **kw)
            for s in starts]
    return C.collate(wins)


def _ids(env, cell):
    by = env["by"]
    return {a: by[(c, a)] for (c, a) in by if c == cell}


@pytest.fixture(scope="module")
def cell(env):
    return sorted({c for c, _ in env["by"]})[0]


@pytest.fixture(scope="module")
def imp_batch(env, cell):
    m = _ids(env, cell)
    inputs = [m["ATAC-seq"], m["H3K27ac"], m[C.CONTROL]]
    targets = [m["H3K4me3"], m["H3K27ac"]]
    return _batch(env, [i * L for i in range(8)], inputs, targets)


# --------------------------------------------------------------------------- reference


def _kernel_attrs(env, mid, which):
    with h5py.File(env["store"] / mid / STORE_FILES[which], "r") as f:
        return np.asarray(f.attrs["kernel"], float), np.asarray(f.attrs["n_reads_strand"], float)


def _expected(env, batch, mu_sub, n_sub, runs):
    """numpy: per run and track, edge-padded convolution with the measured kernel."""
    mu_sub, n_sub = mu_sub.numpy(), np.maximum(n_sub.numpy(), C.N_MIN)
    B, Lb, A, _ = mu_sub.shape
    mu_out, n_out = np.zeros((B, Lb, A)), np.zeros((B, Lb, A))
    for run in runs:
        meta = batch["meta"][run[0]]
        own = {}
        for t, m in enumerate(meta["measurements_out"]):
            if m:
                own[t] = _kernel_attrs(env, m, meta["target_file"])
        nr_mix = np.sum([nr for _, nr in own.values()], 0)
        k_mix = np.sum([k * nr[:, None] for k, nr in own.values()], 0) / nr_mix[:, None]
        mus = np.concatenate([mu_sub[i] for i in run])           # [n L, A, 2]
        ns = np.concatenate([n_sub[i] for i in run])
        for t in range(A):
            k, nr = own.get(t, (k_mix, nr_mix))
            if C.TRACKS[t] in UNSTRANDED:
                rows = [(0, (k * nr[:, None]).sum(0) / nr.sum())]
            else:
                rows = [(0, k[0]), (1, k[1])]
            m_t, v_t = 0.0, 0.0
            for s, w in rows:
                pm = np.pad(mus[:, t, s], 8, mode="edge")
                pn = np.pad(ns[:, t, s], 8, mode="edge")
                m_t = m_t + np.convolve(pm, w, "valid")
                v_t = v_t + np.convolve(pm * pm / pn, w * w, "valid")
            r = np.where(v_t > 0, m_t * m_t / np.where(v_t > 0, v_t, 1), C.N_MAX)
            r = np.clip(r, C.N_MIN, C.N_MAX)
            for j, i in enumerate(run):
                mu_out[i, :, t] = m_t[j * Lb:(j + 1) * Lb]
                n_out[i, :, t] = r[j * Lb:(j + 1) * Lb]
    return mu_out, n_out


# --------------------------------------------------------------------------- tests


def test_output_is_the_convolution_and_moment_match_of_the_stub(env, imp_batch):
    ad = _adapter(env)
    with torch.no_grad():
        o = ad(imp_batch)
    assert tuple(o["mu"].shape) == tuple(imp_batch["x_bin_mask"].shape)
    mu_e, n_e = _expected(env, imp_batch, ad.inner.out["mu_sub"], ad.inner.out["n_sub"],
                          [list(range(8))])
    np.testing.assert_allclose(o["mu"].numpy(), mu_e, rtol=1e-10)
    np.testing.assert_allclose(o["n"].numpy(), n_e, rtol=1e-9)


def test_output_equals_convert_nb_of_the_stub(env, imp_batch, cell):
    ad = _adapter(env)
    with torch.no_grad():
        o = ad(imp_batch)
    mu_sub, n_sub = ad.inner.out["mu_sub"], ad.inner.out["n_sub"]
    m = _ids(env, cell)
    t = C.TRACK_INDEX["H3K4me3"]
    k, _ = _kernel_attrs(env, m["H3K4me3"], "counts")
    mu, n = RS.convert_nb(mu_sub[:, :, t].reshape(8 * L, 2), n_sub[:, :, t].reshape(8 * L, 2),
                          torch.from_numpy(k), stranded=True)
    torch.testing.assert_close(o["mu"][:, :, t].reshape(-1), mu)
    torch.testing.assert_close(o["n"][:, :, t].reshape(-1), n)
    t = C.TRACK_INDEX["DNase-seq"]                    # no target: the read-weighted mix, unstranded
    ks = [_kernel_attrs(env, m[a], "counts") for a in ("H3K4me3", "H3K27ac")]
    nr = sum(x[1] for x in ks)
    kmix = sum(x[0] * x[1][:, None] for x in ks) / nr[:, None]
    kun = RS.unstranded_kernel(kmix, nr)
    mu, n = RS.convert_nb(mu_sub[:, :, t].reshape(8 * L, 2), n_sub[:, :, t].reshape(8 * L, 2),
                          kun, stranded=False)
    torch.testing.assert_close(o["mu"][:, :, t].reshape(-1), mu)
    torch.testing.assert_close(o["n"][:, :, t].reshape(-1), n)


def test_pair_window_reads_pr1_inputs_and_the_pr2_kernel(env, cell):
    m = _ids(env, cell)
    b = _batch(env, [0, L, 3 * L], [m["H3K27ac"]], [m["H3K27ac"]], input_file="pr1",
               target_file="pr2")
    ad = _adapter(env)
    with torch.no_grad():
        o = ad(b)
    mu_e, n_e = _expected(env, b, ad.inner.out["mu_sub"], ad.inner.out["n_sub"], [[0, 1], [2]])
    np.testing.assert_allclose(o["mu"].numpy(), mu_e, rtol=1e-10)
    np.testing.assert_allclose(o["n"].numpy(), n_e, rtol=1e-9)
    t = C.TRACK_INDEX["H3K27ac"]
    with h5py.File(env["store"] / m["H3K27ac"] / "starts_pr1.h5", "r") as f:
        raw = torch.from_numpy(f["chr1"][0:L].astype(np.int32))
    vis = ~b["x_bin_mask"][0, :, t]
    assert torch.equal(ad.inner.seen["x_counts_sub"][0, :, t][vis], raw[vis])


def test_inner_inputs_are_the_store_masked_and_folded(env, imp_batch, cell):
    b = dict(imp_batch, x_avail=imp_batch["x_avail"].clone())
    t_off = C.TRACK_INDEX["H3K27ac"]
    b["x_avail"][:, t_off] = False                  # available in the store, but marked missing
    ad = _adapter(env)
    with torch.no_grad():
        ad(b)
    x = ad.inner.seen["x_counts_sub"]
    hidden = b["x_bin_mask"] | ~b["x_avail"][:, None, :]
    assert x[hidden].abs().sum() == 0
    assert (x[:, :, t_off] == 0).all()
    assert torch.equal(ad.inner.seen["x_counts"], x.sum(-1))
    m = _ids(env, cell)
    for a in ("ATAC-seq", C.CONTROL):
        t = C.TRACK_INDEX[a]
        with h5py.File(env["store"] / m[a] / "starts.h5", "r") as f:
            raw = f["chr1"][0:8 * L].astype(np.int32).reshape(8, L, 2)
        if a in UNSTRANDED:
            raw = np.stack([raw.sum(-1), np.zeros_like(raw[..., 0])], -1)
        vis = ~hidden[:, :, t]
        assert vis.any() and hidden[:, :, t].any()
        assert torch.equal(x[:, :, t][vis], torch.from_numpy(raw)[vis])
    for a in ("DNase-seq", "H3K4me3"):              # not inputs: 0 whatever the store holds
        assert (x[:, :, C.TRACK_INDEX[a]] == 0).all()


def test_batch_x_counts_changes_nothing(env, imp_batch):
    ad = _adapter(env)
    with torch.no_grad():
        o1 = ad(imp_batch)
        seen1 = ad.inner.seen["x_counts_sub"]
        pert = dict(imp_batch, x_counts=torch.randint(0, 500, imp_batch["x_counts"].shape,
                                                      dtype=imp_batch["x_counts"].dtype))
        o2 = ad(pert)
    assert torch.equal(seen1, ad.inner.seen["x_counts_sub"])
    assert torch.equal(o1["mu"], o2["mu"]) and torch.equal(o1["n"], o2["n"])


def test_average_is_the_read_start_average(env, imp_batch):
    from minicandi.rs_dataset import ReadStartDataset
    ad = _adapter(env)
    with torch.no_grad():
        ad(imp_batch)
    ds = ReadStartDataset(env["tables"], "val", env["cfg"])
    want = np.stack([ds.average_strands(mt["cell"], mt["chrom"], mt["start_bin"], L)
                     for mt in imp_batch["meta"]])
    np.testing.assert_array_equal(ad.inner.seen["avg_sub"].numpy(), want)
    torch.testing.assert_close(ad.inner.seen["avg"], ad.inner.seen["avg_sub"].sum(-1))


def test_eight_consecutive_windows_have_two_open_edges(env, imp_batch, tmp_path, monkeypatch):
    log = tmp_path / "score" / "open_edges.json"
    monkeypatch.setenv("RS392_EDGE_LOG", str(log))
    ad = _adapter(env)
    with torch.no_grad():
        ad(imp_batch)
    assert (ad.n_open_edges, ad.n_windows) == (2, 8)
    assert json.loads(log.read_text()) == {"n_open_edges": 2, "n_windows": 8}


def test_gapped_windows_are_converted_apart(env, cell):
    m = _ids(env, cell)
    b = _batch(env, [0, 2 * L, 4 * L, 5 * L], [m["ATAC-seq"]], [m["H3K4me3"]])
    ad = _adapter(env)
    with torch.no_grad():
        o = ad(b)
    assert (ad.n_open_edges, ad.n_windows) == (6, 4)
    mu_e, n_e = _expected(env, b, ad.inner.out["mu_sub"], ad.inner.out["n_sub"],
                          [[0], [1], [2, 3]])
    np.testing.assert_allclose(o["mu"].numpy(), mu_e, rtol=1e-10)
    np.testing.assert_allclose(o["n"].numpy(), n_e, rtol=1e-9)


def test_dnase_target_uses_its_own_mixed_kernel(env):
    cells = sorted({c for c, _ in env["by"]})
    m = _ids(env, cells[1])
    b = _batch(env, [0, L], [m["H3K27ac"]], [m["DNase-seq"]], chrom="chr2")
    ad = _adapter(env)
    with torch.no_grad():
        o = ad(b)
    mu_e, n_e = _expected(env, b, ad.inner.out["mu_sub"], ad.inner.out["n_sub"], [[0, 1]])
    np.testing.assert_allclose(o["mu"].numpy(), mu_e, rtol=1e-10)
    np.testing.assert_allclose(o["n"].numpy(), n_e, rtol=1e-9)
    t = C.TRACK_INDEX["DNase-seq"]
    k, nr = _kernel_attrs(env, m["DNase-seq"], "counts")
    plus_only, _ = RS.convert_nb(ad.inner.out["mu_sub"][:, :, t].reshape(2 * L, 2),
                                 ad.inner.out["n_sub"][:, :, t].reshape(2 * L, 2),
                                 torch.from_numpy(k), stranded=False)
    assert not torch.allclose(o["mu"][:, :, t].reshape(-1), plus_only)   # the mix, not row 0


def test_refusals(env):
    with pytest.raises(ValueError, match="input_depth"):
        _adapter(env, input_depth="fixed")
    from minicandi.rs_adapter import ReadStartAdapter
    cfg = {k: v for k, v in env["cfg"].items() if k != "read_start"}
    with pytest.raises(ValueError, match="read_start"):
        ReadStartAdapter(Stub(), cfg, tables=env["tables"])


def test_load_run_wraps_a_read_start_run(env, tmp_path):
    from minicandi.registry import build_model, load_run
    from minicandi.rs_adapter import ReadStartAdapter
    cfg = {**env["cfg"], "rung": 2, "context_bins": 32, "param_budget": 2.0e5,
           "res_tables": str(env["tables"])}
    torch.manual_seed(0)
    model = build_model(cfg)
    run = tmp_path / "rs_run"
    run.mkdir()
    torch.save(C.make_ckpt(model, cfg, 0, "x", "x"), run / "ckpt.pt")
    back, config = load_run(run)
    assert isinstance(back, ReadStartAdapter) and config["read_start"]
    assert back.param_count() == model.param_count()
    plain = {k: v for k, v in cfg.items() if k != "read_start"}
    torch.save(C.make_ckpt(model, plain, 0, "x", "x"), run / "ckpt.pt")
    assert not isinstance(load_run(run)[0], ReadStartAdapter)


# --------------------------------------------------------------------------- staging (chunk C13fix)


def _forward_all(ad, batches):
    outs = []
    with torch.no_grad():
        for b in batches:
            o = ad(b)
            outs.append((o["mu"].clone(), o["n"].clone(), ad.inner.seen["x_counts_sub"].clone(),
                         ad.inner.seen["avg_sub"].clone()))
    return outs


def test_staged_read_starts_give_bit_identical_outputs(env, imp_batch, cell, tmp_path):
    m = _ids(env, cell)
    pair = _batch(env, [0, L, 3 * L], [m["H3K27ac"]], [m["H3K27ac"]], input_file="pr1",
                  target_file="pr2")
    ref = _forward_all(_adapter(env), [imp_batch, pair])
    ad = _adapter(env)
    info = ad.stage_read_starts(sorted(m.values()), tmp_path / "stage")
    assert info["skipped"] is None and info["files"] > 0 and info["bytes"] > 0
    got = _forward_all(ad, [imp_batch, pair])
    for a, b in zip(ref, got):
        for x, y in zip(a, b):
            assert torch.equal(x, y)
    assert str(ad.ds.rs_store).startswith(str(tmp_path / "stage"))
    assert str(ad.ds.rs_average_h5).startswith(str(tmp_path / "stage"))
    assert str(ad.store).startswith(str(tmp_path / "stage"))
    assert ad.config["read_start_store"] == env["cfg"]["read_start_store"]   # the run's config
    for mid in m.values():                     # every file of a needed measurement is a copy
        for f in STORE_FILES.values():
            p = ad.ds.rs_store / mid / f
            assert p.is_file() and not p.is_symlink() and not p.parent.is_symlink()
            assert p.read_bytes() == (env["store"] / mid / f).read_bytes()
    assert not ad.ds.rs_average_h5.is_symlink()
    assert ad.ds.rs_average_h5.read_bytes() == Path(env["cfg"]["read_start_average_h5"]).read_bytes()


def test_unstaged_measurements_are_read_through_links(env, cell, tmp_path):
    cells = sorted({c for c, _ in env["by"]})
    other = _ids(env, cells[1])
    b = _batch(env, [0, L], [other["H3K27ac"]], [other["DNase-seq"]], chrom="chr2")
    ref = _forward_all(_adapter(env), [b])
    ad = _adapter(env)
    ad.stage_read_starts(sorted(_ids(env, cell).values()), tmp_path / "stage")
    assert (ad.ds.rs_store / other["H3K27ac"]).is_symlink()
    got = _forward_all(ad, [b])
    for x, y in zip(ref[0], got[0]):
        assert torch.equal(x, y)


def test_staging_also_copies_the_average_sources_of_the_cell(env, cell, tmp_path):
    """The leave-one-cell-out average reads the window cell's own training measurement of every
    track, so staging one measurement of a cell copies those too."""
    m = _ids(env, cell)
    ad = _adapter(env)
    ad.stage_read_starts([m["H3K4me3"]], tmp_path / "stage")
    own = [x for (c, _), x in ad.ds.train_base.items() if c == cell]
    assert own
    for mid in own + [m["H3K4me3"]]:
        assert not (ad.ds.rs_store / mid).is_symlink()


def test_staging_is_skipped_without_space(env, cell, tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    du = namedtuple("du", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: du(1, 1, 0))
    ad = _adapter(env)
    info = ad.stage_read_starts(sorted(_ids(env, cell).values()), tmp_path / "stage")
    assert info["skipped"] and info["files"] == 0
    assert ad.ds.rs_store == env["store"] and ad.store == env["store"]


def test_predict_store_stage_calls_the_adapter_hook(env, cell, tmp_path):
    """tools/onion_eval/predict_store.py `_stage` stages a read-start run's own files; a model
    without the hook (every read-count run) gets the old result, key for key."""
    import types
    sys.path.insert(0, str(REPO / "tools" / "onion_eval"))
    import predict_store as P
    m = _ids(env, cell)
    reqs = [{"job": {"target": m["H3K4me3"], "inputs": [m["H3K27ac"], m["ATAC-seq"]]}}]

    def scorer(model):
        return types.SimpleNamespace(ds=_scorer_ds(env), model=model, _region=None, _rc={})
    plain = P._stage(scorer(Stub()), reqs, tmp_path / "a", lambda *x: None)
    assert set(plain) == {"directories", "bytes_copied", "seconds"}
    ad = _adapter(env)
    info = P._stage(scorer(ad), reqs, tmp_path / "b", lambda *x: None)
    assert set(info) == {"directories", "bytes_copied", "seconds", "read_starts"}
    assert info["read_starts"]["skipped"] is None
    assert str(ad.ds.rs_store).startswith(str(tmp_path / "b"))
