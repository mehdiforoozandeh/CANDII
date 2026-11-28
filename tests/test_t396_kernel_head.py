"""t396 chunk C13: the kernel head (start input, count target), config-gated.

Synthetic tables (`minicandi.synth`) with a hand-made read-start store whose files carry the
`kernel` [2, 17] and `n_reads_strand` [2] attributes. Checks: the window keys and shapes, the
batched conversion against `read_start.convert_nb`, the loss ordering of a true and a shifted
model, gradients, the open edges, the config gate and the four arm configs.
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

L = 64
STORE_FILES = {"counts": "starts.h5", "pr1": "starts_pr1.h5", "pr2": "starts_pr2.h5"}
UNSTRANDED = ("ATAC-seq", "DNase-seq")
CFG_DIR = REPO / "tools" / "minicandi" / "configs"


def _kernel(rng):
    k = rng.uniform(0.0, 1.0, (2, 17))
    k[0, :8] = 0.0                       # plus-strand reads extend downstream
    k[1, 9:] = 0.0
    return k


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi import contract as C
    from minicandi.dataset import read_tsv
    from minicandi.synth import make_synthetic_tables
    root = tmp_path_factory.mktemp("kh")
    tables = make_synthetic_tables(
        root, n_cells=4, assays=("ATAC-seq", "DNase-seq", "H3K27ac", "H3K4me3", C.CONTROL),
        val_tracks=((0, "H3K4me3"),), blind_tracks=(), chrom_bins={"chr1": 1024, "chr2": 512})
    rows = read_tsv(tables / "measurements.tsv")
    store = root / "rs_store"
    rng = np.random.default_rng(0)
    attrs = {}
    for r in rows:
        mid = r["measurement_id"]
        with h5py.File(Path(r["counts_path"].split(":", 1)[1]), "r") as f:
            n_bins = {c: f[c].shape[0] for c in f.keys()}
        for which, fn in STORE_FILES.items():
            k, nr = _kernel(rng), rng.integers(100, 1000, 2).astype(np.float64)
            p = store / mid / fn
            p.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(p, "w") as f:
                for c, n in n_bins.items():
                    f.create_dataset(c, data=rng.poisson(1.0, (n, 2)).astype(np.uint16))
                f.attrs["kernel"] = k
                f.attrs["n_reads_strand"] = nr
            attrs[(mid, which)] = (k, nr)
    cfg = {"context_bins": L, "batch_size": 4, "read_start": True,
           "read_start_store": str(store), "train_halves": "pairs"}
    return {"tables": tables, "store": store, "attrs": attrs, "cfg": cfg,
            "rows": {r["measurement_id"]: r for r in rows}}


def _batches(env, extra=None, n=3, seed=5):
    from minicandi.res_dataset import make_dataset
    ds = make_dataset(env["tables"], "train", {**env["cfg"], **(extra or {})}, seed)
    it = iter(ds)
    return ds, [next(it) for _ in range(n)]


# ------------------------------------------------------------------------------- dataset keys

def test_window_keys_and_shapes(env):
    from minicandi import contract as C
    _, bs = _batches(env, {"kernel_head": True})
    B = env["cfg"]["batch_size"]
    for b in bs:
        assert b["y_counts_overlap"].dtype == torch.int32
        assert tuple(b["y_counts_overlap"].shape) == (B, L, C.A)
        assert b["y_kernel"].dtype == torch.float32
        assert tuple(b["y_kernel"].shape) == (B, C.A, 2, 17)


def test_window_values_are_the_count_store_and_the_target_kernels(env):
    from minicandi import contract as C
    from minicandi.dataset import OnionDataset
    from minicandi.read_start import unstranded_kernel
    ds, bs = _batches(env, {"kernel_head": True})
    checked = 0
    for b in bs:
        for i, meta in enumerate(b["meta"]):
            halved = set(meta.get("halves_tracks") or ())
            for t, m in enumerate(meta["measurements_out"]):
                yo = b["y_counts_overlap"][i, :, t].numpy()
                if not m:
                    assert not yo.any()
                    continue
                which = ("pr2" if meta["measurements_in"][t] and C.TRACKS[t] in halved
                         else meta["target_file"])
                want = OnionDataset.read(ds, m, meta["chrom"], meta["start_bin"], L, which)
                np.testing.assert_array_equal(yo, want)
                k, nr = env["attrs"][(m, which)]
                got = b["y_kernel"][i, t].numpy()
                if C.TRACKS[t] in UNSTRANDED:
                    np.testing.assert_allclose(got[0], unstranded_kernel(k, nr).numpy(),
                                               rtol=1e-6)
                    assert not got[1].any()
                else:
                    np.testing.assert_allclose(got, k, rtol=1e-6)
                checked += 1
    assert checked > 0


def test_gate_off_leaves_the_window_unchanged(env):
    _, off = _batches(env)
    _, on = _batches(env, {"kernel_head": True})
    for a, b in zip(off, on):
        assert "y_counts_overlap" not in a and "y_kernel" not in a
        assert set(b) - set(a) == {"y_counts_overlap", "y_kernel"}
        for k in a:
            if k != "meta":
                assert torch.equal(a[k], b[k]), k
        assert a["meta"] == b["meta"]


# ------------------------------------------------------------------------------- conversion

def test_batched_conversion_equals_convert_nb_per_track():
    from minicandi import contract as C
    from minicandi.kernel_head import convert_nb_batched
    from minicandi.read_start import convert_nb
    g = torch.Generator().manual_seed(0)
    B, Lw, A = 2, 40, C.A
    mu_sub = torch.rand(B, Lw, A, 2, generator=g, dtype=torch.float64) * 3
    n_sub = torch.rand(B, Lw, A, 2, generator=g, dtype=torch.float64) * 5 + 0.1
    ker = torch.rand(B, A, 2, 17, generator=g, dtype=torch.float64)
    ker[:, 0, 1] = 0.0                                   # track 0 as an unstranded track
    mu, n = convert_nb_batched(mu_sub, n_sub, ker)
    for b in range(B):
        for t in range(A):
            if t == 0:
                m1, n1 = convert_nb(mu_sub[b, :, t], n_sub[b, :, t], ker[b, t, 0], False)
            else:
                m1, n1 = convert_nb(mu_sub[b, :, t], n_sub[b, :, t], ker[b, t], True)
            assert torch.equal(mu[b, :, t], m1) and torch.equal(n[b, :, t], n1)


# ------------------------------------------------------------------------------- the loss

def _synthetic(shift: int = 0, seed: int = 1):
    """A start model and the batch whose overlap counts are its rounded conversion."""
    from minicandi import contract as C
    from minicandi.kernel_head import convert_nb_batched
    g = torch.Generator().manual_seed(seed)
    B, Lw, A = 2, 48, C.A
    starts = torch.rand(B, Lw, A, 2, generator=g, dtype=torch.float64) * 4 + 0.05
    ker = torch.rand(B, A, 2, 17, generator=g, dtype=torch.float64)
    n_sub = torch.full_like(starts, 20.0)
    mu, _ = convert_nb_batched(starts, n_sub, ker)
    batch = {"y_counts_overlap": mu.round().to(torch.int32), "y_kernel": ker.float(),
             "y_kind": torch.full((B, A), C.KIND_IMPUTE, dtype=torch.uint8),
             "y_avail": torch.ones(B, A, dtype=torch.bool),
             "x_bin_mask": torch.zeros(B, Lw, A, dtype=torch.bool),
             "y_counts": torch.zeros(B, Lw, A, dtype=torch.int32)}
    out = {"mu_sub": torch.roll(starts, shift, dims=1).clone().requires_grad_(True),
           "n_sub": n_sub.clone().requires_grad_(True)}
    return out, batch


def test_true_starts_beat_shifted_starts():
    from minicandi.kernel_head import kernel_head_loss
    lt, _ = kernel_head_loss(*_synthetic(0), {})
    ls, _ = kernel_head_loss(*_synthetic(3), {})
    assert float(lt) < float(ls)


def test_gradients_reach_mu_sub_and_n_sub():
    from minicandi.kernel_head import kernel_head_loss
    out, batch = _synthetic(2)
    loss, parts = kernel_head_loss(out, batch, {})
    loss.backward()
    for k in ("mu_sub", "n_sub"):
        g = out[k].grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, k
    assert "main" in parts


def test_edge_bins_carry_no_loss_and_no_gradient():
    from minicandi.kernel_head import convert_nb_batched, kernel_head_loss
    from minicandi.losses import total_loss
    from minicandi.read_start import REACH
    out, batch = _synthetic(1)
    base, _ = kernel_head_loss(out, batch, {})
    b2 = dict(batch, y_counts_overlap=batch["y_counts_overlap"].clone())
    b2["y_counts_overlap"][:, :REACH] += 50
    b2["y_counts_overlap"][:, -REACH:] += 50
    moved, _ = kernel_head_loss(out, b2, {})
    assert torch.equal(base, moved)
    # the gradient of the loss with respect to the converted NB is 0 on the edge bins
    mu, n = convert_nb_batched(out["mu_sub"].detach(), out["n_sub"].detach(), batch["y_kernel"])
    mu.requires_grad_(True)
    n.requires_grad_(True)
    Lw = mu.shape[1]
    sl = slice(REACH, Lw - REACH)
    ref, _ = total_loss({"mu": mu[:, sl], "n": n[:, sl]},
                        dict(batch, y_counts=batch["y_counts_overlap"][:, sl],
                             x_bin_mask=batch["x_bin_mask"][:, sl]), {})
    ref.backward()
    for g in (mu.grad, n.grad):
        assert not g[:, :REACH].any() and not g[:, -REACH:].any()
        assert g[:, sl].abs().sum() > 0
    assert torch.allclose(ref, base.to(ref.dtype))


# ------------------------------------------------------------------------------- gate, configs

def test_gate_refuses_kernel_head_without_read_start_and_placeholders(env):
    from minicandi.res_dataset import make_dataset
    cfg = {k: v for k, v in env["cfg"].items() if k != "read_start"}
    with pytest.raises(ValueError, match="read_start"):
        make_dataset(env["tables"], "train", {**cfg, "kernel_head": True}, 0)
    with pytest.raises(ValueError, match="placeholder"):
        make_dataset(env["tables"], "train", {**env["cfg"], "_todo": "x"}, 0)


@pytest.mark.parametrize("cfg,want", [({"read_start": True, "kernel_head": True}, "kernel_head"),
                                      ({"read_start": True}, "read_start"),
                                      ({}, "res_total_loss")])
def test_train_selects_the_loss_by_the_gate(monkeypatch, cfg, want):
    from minicandi import res_dataset as R
    from minicandi import train as T
    seen = {}

    def fake(config, *a, **k):
        seen["loss"] = T.total_loss
        return {}

    monkeypatch.setattr(T, "train", fake)
    R.train(cfg, None, 0, None)
    from minicandi.kernel_head import kernel_head_loss
    from minicandi.read_start import read_start_loss
    assert seen["loss"] is {"kernel_head": kernel_head_loss, "read_start": read_start_loss,
                            "res_total_loss": R.res_total_loss}[want]


def test_arm_configs():
    d = CFG_DIR / "t396"
    c = {n: json.loads((d / f"{n}.json").read_text()) for n in
         ("count_in_count_out", "start_in_start_out", "start_in_count_out", "count_in_start_out")}
    for name, v in c.items():
        assert v["windows_seen"] == 160000 and v["param_budget"] == 1e6, name
    recipe = json.loads((CFG_DIR / "recipe/scale/rung4/combo_w2x.json").read_text())
    assert c["count_in_count_out"] == recipe
    rs = c["start_in_start_out"]
    assert rs["read_start"] is True and "kernel_head" not in rs
    assert {k: v for k, v in rs.items() if not k.startswith("read_start")} == recipe
    assert c["start_in_count_out"] == {**rs, "kernel_head": True}
    assert "_todo" in c["count_in_start_out"]
    assert "_todo" not in c["start_in_count_out"]
