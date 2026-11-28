"""t396 iteration 2 chunk A2: the arm "count input, start target" (`input_space: "count"`).

Synthetic tables (`minicandi.synth`) with a hand-made read-start store (start files carry the
`kernel` and `n_reads_strand` attributes) and a count average file. Checks: the window input is
the count-store read and its target the start-store read; channel 1 of the input and of the
average is 0; the average is the count average; the config gate; one training step; the
adapter's input equals the dataset's; the arm config and its parameter count; and the kernel-head
arm's scoring path.
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

L = 64
STORE_FILES = {"counts": "starts.h5", "pr1": "starts_pr1.h5", "pr2": "starts_pr2.h5"}
CFG_DIR = REPO / "tools" / "minicandi" / "configs"
COUNT = {"input_space": "count"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.dataset import build_average_h5
    from minicandi.dataset import read_tsv
    from minicandi.synth import make_synthetic_tables
    root = tmp_path_factory.mktemp("count_in")
    tables = make_synthetic_tables(
        root, n_cells=5, assays=("ATAC-seq", "DNase-seq", "H3K27ac", "H3K4me3", C.CONTROL),
        val_tracks=((0, "H3K4me3"),), blind_tracks=(), chrom_bins={"chr1": 1024, "chr2": 512})
    rows = read_tsv(tables / "measurements.tsv")
    store = root / "rs_store"
    rng = np.random.default_rng(0)
    for r in rows:
        mid = r["measurement_id"]
        with h5py.File(Path(r["counts_path"].split(":", 1)[1]), "r") as f:
            n_bins = {c: f[c].shape[0] for c in f.keys()}
        for fn in STORE_FILES.values():
            k = rng.uniform(0.0, 1.0, (2, 17))
            k[0, :8] = 0.0
            k[1, 9:] = 0.0
            p = store / mid / fn
            p.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(p, "w") as f:
                for c, n in n_bins.items():
                    f.create_dataset(c, data=rng.poisson(1.0, (n, 2)).astype(np.uint16))
                f.attrs["kernel"] = k
                f.attrs["n_reads_strand"] = rng.integers(100, 1000, 2).astype(np.float64)
    avg = build_average_h5(tables, root / "count_average.h5")
    cfg = {"context_bins": L, "batch_size": 4, "read_start": True, "average_input": True,
           "average_h5": str(avg), "read_start_store": str(store), "train_halves": "pairs"}
    by = {(r["cell"], r["assay"]): r["measurement_id"] for r in rows}
    return {"tables": tables, "store": store, "cfg": cfg, "by": by}


def _ds(env, extra=None, mode="train", seed=5):
    from minicandi.res_dataset import make_dataset
    return make_dataset(env["tables"], mode, {**env["cfg"], **(extra or {})}, seed)


def _batches(env, extra=None, n=3, seed=5):
    ds = _ds(env, extra, seed=seed)
    it = iter(ds)
    return ds, [next(it) for _ in range(n)]


# ------------------------------------------------------------------------------- the window

def test_input_is_the_count_store_and_target_the_start_store(env):
    from minicandi.dataset import OnionDataset
    ds, bs = _batches(env, COUNT)
    n_in = n_out = 0
    for b in bs:
        for i, meta in enumerate(b["meta"]):
            halved = set(meta.get("halves_tracks") or ())
            mask = b["x_bin_mask"][i].numpy()
            for t, m in enumerate(meta["measurements_in"]):
                x = b["x_counts_sub"][i, :, t].numpy()
                if not m:
                    assert not x.any()
                    continue
                which = "pr1" if C.TRACKS[t] in halved else meta["input_file"]
                want = OnionDataset.read(ds, m, meta["chrom"], meta["start_bin"], L, which)
                want = np.where(mask[:, t], 0, want)
                np.testing.assert_array_equal(x[:, 0], want)
                np.testing.assert_array_equal(b["x_counts"][i, :, t].numpy(), want)
                n_in += 1
            for t, m in enumerate(meta["measurements_out"]):
                y = b["y_counts_sub"][i, :, t].numpy()
                if not m:
                    assert not y.any()
                    continue
                which = ("pr2" if meta["measurements_in"][t] and C.TRACKS[t] in halved
                         else meta["target_file"])
                want = ds.read_strands(m, meta["chrom"], meta["start_bin"], L, which)
                if C.TRACKS[t] in ("ATAC-seq", "DNase-seq"):
                    want = np.stack([want.sum(1), np.zeros(L, np.int64)], 1)
                np.testing.assert_array_equal(y, want)
                np.testing.assert_array_equal(b["y_counts"][i, :, t].numpy(), want.sum(1))
                n_out += 1
    assert n_in > 0 and n_out > 0


def test_channel_1_of_the_input_and_the_average_is_zero(env):
    _, bs = _batches(env, COUNT)
    for b in bs:
        assert not b["x_counts_sub"][..., 1].any()
        assert not b["avg_sub"][..., 1].any()
        assert torch.equal(b["avg"], b["avg_sub"].sum(-1))
        assert b["x_counts_sub"][..., 0].any()


def test_average_is_the_count_average(env):
    from minicandi.dataset import OnionDataset
    ds, bs = _batches(env, COUNT)
    plain = OnionDataset(env["tables"], "train", {k: v for k, v in env["cfg"].items()
                                                  if not k.startswith("read_start")}, 5)
    for b in bs:
        for i, meta in enumerate(b["meta"]):
            want = plain.average(meta["cell"], meta["chrom"], meta["start_bin"], L)
            np.testing.assert_allclose(b["avg_sub"][i, ..., 0].numpy(), want, rtol=1e-6)


def test_start_input_is_unchanged_by_the_absent_key(env):
    _, a = _batches(env)
    _, b = _batches(env, {"input_space": "start"})
    for x, y in zip(a, b):
        assert x["meta"] == y["meta"]
        for k in x:
            if k != "meta":
                assert torch.equal(x[k], y[k]), k


# ------------------------------------------------------------------------------- the gate

def test_make_dataset_accepts_the_arm_and_refuses_bad_configs(env):
    from minicandi.rs_dataset import ReadStartDataset
    assert isinstance(_ds(env, COUNT), ReadStartDataset)
    no_rs = {k: v for k, v in env["cfg"].items() if k != "read_start"}
    from minicandi.res_dataset import make_dataset
    with pytest.raises(ValueError, match="read_start"):
        make_dataset(env["tables"], "train", {**no_rs, **COUNT}, 0)
    with pytest.raises(ValueError, match="kernel_head"):
        _ds(env, {**COUNT, "kernel_head": True})
    with pytest.raises(ValueError, match="input_space"):
        _ds(env, {"input_space": "bins"})


def test_train_keeps_read_start_loss_and_takes_pearson_from_the_count(monkeypatch):
    from minicandi import res_dataset as R
    from minicandi import train as T
    from minicandi.read_start import read_start_loss
    seen = {}
    orig = T.set_pearson_stats

    def fake(config, *a, **k):
        seen["loss"] = T.total_loss
        seen["pearson"] = T.set_pearson_stats
        return {}

    monkeypatch.setattr(T, "train", fake)
    R.train({"read_start": True, **COUNT}, None, 0, None)
    assert seen["loss"] is read_start_loss
    assert seen["pearson"] is not orig and T.set_pearson_stats is orig
    with pytest.raises(ValueError, match="kernel_head"):
        R.train({"read_start": True, "kernel_head": True, **COUNT}, None, 0, None)


def _model_cfg(env, **extra):
    cfg = json.loads((CFG_DIR / "t396" / "count_in_start_out.json").read_text())
    cfg.update({k: env["cfg"][k] for k in ("average_h5", "read_start_store")})
    cfg.pop("read_start_average_h5")
    cfg.update({"context_bins": L, "width": 8, "train_halves": "pairs"}, **extra)
    return cfg


def test_one_training_step_gives_a_finite_loss(env):
    from minicandi import registry as R
    from minicandi.read_start import read_start_loss
    from minicandi.train import set_pearson_stats
    cfg = _model_cfg(env)
    ds = _ds(env, {k: v for k, v in cfg.items() if k != "read_start_average_h5"})
    b = ds.batch(2, np.random.default_rng([1, 9]))
    torch.manual_seed(0)
    m = R.build_model(cfg).train()
    set_pearson_stats(m, [b])
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    loss, _ = read_start_loss(m(b), b, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.track_in[0].weight.grad is not None
    opt.step()


# ------------------------------------------------------------------------------- scoring

def test_adapter_input_equals_the_dataset_input(env):
    from minicandi.rs_adapter import ReadStartAdapter
    cfg = {**env["cfg"], **COUNT}
    ds = _ds(env, COUNT, mode="val")
    cell = sorted({c for c, _ in env["by"]})[0]
    m = {a: env["by"][(c, a)] for (c, a) in env["by"] if c == cell}
    rng = np.random.default_rng(0)
    wins = [ds.window("chr1", s, [m["ATAC-seq"], m["H3K27ac"], m[C.CONTROL]],
                      [m["H3K4me3"], m["H3K27ac"]], x_bin_mask=rng.random((L, C.A)) < 0.3)
            for s in (0, L, 2 * L)]
    batch = C.collate(wins)
    seen = {}

    class Stub(C.MiniCandi):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def forward(self, b):
            seen.update(b)
            B, Lb, A = b["x_bin_mask"].shape
            one = torch.ones(B, Lb, A, 2, dtype=torch.float64)
            return {"mu_sub": one, "n_sub": one, "mu": one.sum(-1), "n": one.sum(-1)}

    ad = ReadStartAdapter(Stub(), cfg, tables=env["tables"]).eval()
    scorer_batch = dict(batch, x_counts=torch.zeros_like(batch["x_counts"]))
    with torch.no_grad():
        ad(scorer_batch)
    assert torch.equal(seen["x_counts_sub"], batch["x_counts_sub"])
    assert torch.equal(seen["x_counts"], batch["x_counts"])
    assert torch.equal(seen["avg_sub"], batch["avg_sub"])
    assert not seen["x_counts_sub"][..., 1].any()


def test_arm_config_and_param_count():
    from minicandi import registry as R
    d = CFG_DIR / "t396"
    ci = json.loads((d / "count_in_start_out.json").read_text())
    rs = json.loads((d / "start_in_start_out.json").read_text())
    assert "_todo" not in ci
    assert ci == {**rs, "input_space": "count"}
    torch.manual_seed(0)
    a = R.build_model(ci).param_count()
    torch.manual_seed(0)
    b = R.build_model(rs).param_count()
    assert a == b


def test_kernel_head_arm_scores_through_the_span_kernel_without_the_open_edge_rule():
    """The kernel-head arm (`start_in_count_out`) has start output, so `registry.load_run` wraps
    it in `ReadStartAdapter` like `start_in_start_out`: its count-space row is the adapter's
    span-kernel conversion (`read_start.convert_nb`, every open edge filled with the edge bin),
    i.e. the kernel layer of training without the open-edge rule of `kernel_head_loss` (edge bins
    carry no loss there). Here: a single window's adapter output equals `convert_nb` per track."""
    from minicandi import read_start as RS
    d = CFG_DIR / "t396"
    kh = json.loads((d / "start_in_count_out.json").read_text())
    assert kh["read_start"] is True and kh["kernel_head"] is True
    assert "input_space" not in kh
    rng = np.random.default_rng(3)
    k = rng.uniform(0.0, 1.0, (2, 17))
    k[0, :8] = 0.0
    k[1, 9:] = 0.0
    k = torch.from_numpy(k)
    mu = torch.from_numpy(rng.uniform(0.2, 3.0, (L, 2)))
    n = torch.from_numpy(rng.uniform(0.3, 5.0, (L, 2)))
    m1, n1 = RS.convert_nb(mu, n, k, stranded=True)
    assert m1.shape == (L,) and torch.isfinite(m1).all() and torch.isfinite(n1).all()
    # the open edges: the first and last bins equal a conversion of the edge-padded run
    pad = 8
    mp = torch.cat([mu[:1].expand(pad, 2), mu, mu[-1:].expand(pad, 2)])
    np_ = torch.cat([n[:1].expand(pad, 2), n, n[-1:].expand(pad, 2)])
    m2, _ = RS.convert_nb(mp, np_, k, stranded=True)
    torch.testing.assert_close(m1, m2[pad:-pad])
