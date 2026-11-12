"""miniCANDI loss arms of stage 3 (chunk LS; plan Amendment 3 section B, pinned interface E, and
the key `loss_bin_equal` of the foreman amendment): each arm by hand, the gradient structure of the
bin-equal loss, the running mean of the equal assay weight, `pool_nb`, and the defaults bit for bit
over 5 training steps."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import losses as LS  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
DEFAULTS = {"loss_shape_weight": 0, "loss_assay_weight": "none", "loss_multiscale": [],
            "loss_bin_equal": False}


def _cfg(r: int = 1, **over) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    cfg.update(context_bins=L, param_budget=1.5e5, batch_size=4)
    cfg.update(over)
    return cfg


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_loss_arms"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def batch(tables) -> dict:
    return OnionDataset(tables, "train", _cfg(1), 0).batch(4, np.random.default_rng(0))


def _out(batch, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    shp = batch["y_counts"].shape
    return {"n": (torch.exp(torch.randn(shp, generator=g)) * 3).to(dtype),
            "mu": (torch.exp(torch.randn(shp, generator=g)) * 5).to(dtype)}


def _leaf(out):
    return {k: v.detach().clone().requires_grad_(True) for k, v in out.items()}


# ------------------------------------------------------------------- defaults


def test_defaults_are_bit_identical(batch):
    out = _out(batch)
    want, wp = LS.total_loss(out, batch, {})
    got, gp = LS.total_loss(out, batch, dict(DEFAULTS))
    assert torch.equal(got, want) and gp == wp
    want = C.nb_nll(out["n"], out["mu"], batch["y_counts"], C.loss_mask(batch))
    assert torch.equal(got, want)


def test_bad_values_are_refused(batch):
    out = _out(batch)
    for bad in ({"loss_assay_weight": "uniform"}, {"loss_shape_weight": -1},
                {"loss_multiscale": [1]}, {"loss_bin_equal": 1},
                {"loss_bin_equal": True, "loss": "nb_crps"}):
        with pytest.raises(ValueError):
            LS.total_loss(out, batch, bad)


def _old_step_loss(model, batch, config=None, parts=None):
    out = model(batch)
    return C.nb_nll(out["n"], out["mu"], batch["y_counts"], C.loss_mask(batch))


def test_explicit_defaults_train_as_before_for_5_steps(tables, tmp_path, monkeypatch):
    T.train(_cfg(1, windows_seen=20, **DEFAULTS), tables, 0, tmp_path / "new", log_every=1)
    monkeypatch.setattr(T, "step_loss", _old_step_loss)
    T.train(_cfg(1, windows_seen=20), tables, 0, tmp_path / "old", log_every=1)
    new = [json.loads(x) for x in (tmp_path / "new/log.jsonl").read_text().splitlines()]
    old = [json.loads(x) for x in (tmp_path / "old/log.jsonl").read_text().splitlines()]
    assert len(new) == len(old) == 5
    assert [x["loss"] for x in new] == [x["loss"] for x in old]
    assert [x["grad_norm"] for x in new] == [x["grad_norm"] for x in old]
    a = torch.load(tmp_path / "new/ckpt.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "old/ckpt.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------- bin-equal (Max's loss, NB)


def test_bin_equal_value_by_hand(batch):
    out = _out(batch, seed=1, dtype=torch.float64)
    mask = C.loss_mask(batch)
    y = batch["y_counts"].double()
    got, _ = LS.total_loss(out, batch, {"loss_bin_equal": True})
    n, mu = out["n"], out["mu"]
    nll = LS.nb_nll_per(n, mu, y)
    var = mu + mu * mu / n
    tot, k = 0.0, 0
    for b in range(mask.shape[0]):
        for a in range(mask.shape[2]):
            m = mask[b, :, a]
            if not m.any():
                continue
            vbar = float(var[b, m, a].mean())
            tot += float((nll[b, m, a] + 0.5 * (y[b, m, a] - mu[b, m, a]) ** 2 / vbar).sum())
            k += int(m.sum())
    assert float(got) == pytest.approx(tot / k, rel=1e-10)


def test_bin_equal_mean_gradient_comes_only_from_the_squared_term(batch):
    """(i) d/dmu is the squared term's alone, (iii') d/dn is the NB NLL's alone."""
    out = _leaf(_out(batch, seed=2, dtype=torch.float64))
    mask = C.loss_mask(batch)
    y = batch["y_counts"].double()
    LS.total_loss(out, batch, {"loss_bin_equal": True})[0].backward()
    k = int(mask.sum())
    n, mu = out["n"].detach(), out["mu"].detach()
    m = mask.double()
    vbar = ((mu + mu * mu / n) * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
    want_mu = torch.where(mask, (mu - y) / vbar / k, torch.zeros_like(mu))
    np.testing.assert_allclose(out["mu"].grad.numpy(), want_mu.numpy(), rtol=1e-9, atol=1e-15)
    nn_ = n.clone().requires_grad_(True)
    (LS.nb_nll_per(nn_[mask], mu[mask], y[mask]).sum() / k).backward()
    np.testing.assert_allclose(out["n"].grad.numpy(), nn_.grad.numpy(), rtol=1e-9, atol=1e-15)


def test_bin_equal_every_bin_of_a_track_has_the_same_weight(batch):
    """(ii) dLoss/dmu_i / (mu_i - y_i) is one constant per (window, track): 1 / (k vbar)."""
    out = _leaf(_out(batch, seed=3, dtype=torch.float64))
    mask = C.loss_mask(batch)
    y = batch["y_counts"].double()
    LS.total_loss(out, batch, {"loss_bin_equal": True})[0].backward()
    ratio = out["mu"].grad / (out["mu"].detach() - y)
    seen = 0
    for b in range(mask.shape[0]):
        for a in range(mask.shape[2]):
            m = mask[b, :, a] & ((out["mu"].detach() - y)[b, :, a].abs() > 1e-6)
            if int(m.sum()) < 2:
                continue
            r = ratio[b, m, a]
            assert float(r.max() - r.min()) <= 1e-9 * float(r.abs().max())
            seen += 1
    assert seen >= 4
    # and the NB NLL alone would give unequal weights: the arm changes something
    out2 = _leaf(_out(batch, seed=3, dtype=torch.float64))
    LS.total_loss(out2, batch, {})[0].backward()
    r2 = (out2["mu"].grad / (out2["mu"].detach() - y))[mask]
    assert float(r2.std()) > 1e-3 * float(r2.abs().mean())


# ------------------------------------------------------------------- assay weights


def _per_assay_means(per, mask):
    las = []
    for a in range(mask.shape[2]):
        m = mask[..., a]
        if m.any():
            las.append(float(per[..., a][m].double().mean()))
    return np.array(las)


def test_equal_assay_weight_running_mean(batch):
    LS.reset_assay_ema()
    cfg = {"loss_assay_weight": "equal"}
    mask = C.loss_mask(batch)
    o1, o2 = _out(batch, seed=4), _out(batch, seed=5)
    l1, _ = LS.total_loss(o1, batch, cfg)
    assert float(l1) == pytest.approx(1.0, rel=1e-6)     # first step: each assay over itself
    l2, _ = LS.total_loss(o2, batch, cfg)
    a1 = _per_assay_means(LS.nb_nll_per(o1["n"], o1["mu"], batch["y_counts"]), mask)
    a2 = _per_assay_means(LS.nb_nll_per(o2["n"], o2["mu"], batch["y_counts"]), mask)
    ema = 0.99 * a1 + 0.01 * a2
    assert float(l2) == pytest.approx(float(np.mean(a2 / ema)), rel=1e-5)
    # the running mean is per config object and carries no gradient
    o = _leaf(o2)
    LS.total_loss(o, batch, {"loss_assay_weight": "equal"})[0].backward()
    assert torch.isfinite(o["mu"].grad).all() and float(o["mu"].grad.abs().sum()) > 0
    LS.reset_assay_ema()


def test_learned_assay_weight_by_hand(batch):
    out = _out(batch, seed=6)
    mask = C.loss_mask(batch)
    lw = torch.linspace(-1.0, 1.0, C.A, requires_grad=True)
    got, _ = LS.total_loss(dict(out, assay_log_w=lw), batch, {"loss_assay_weight": "learned"})
    per = LS.nb_nll_per(out["n"], out["mu"], batch["y_counts"])
    tot, na = 0.0, 0
    for a in range(C.A):
        m = mask[..., a]
        if m.any():
            la = float(per[..., a][m].double().mean())
            tot += np.exp(float(lw[a])) * la - float(lw[a])
            na += 1
    assert float(got) == pytest.approx(tot / na, rel=1e-5)
    got.backward()
    absent = ~mask.any(0).any(0)
    assert absent.any() and float(lw.grad[absent].abs().sum()) == 0.0
    with pytest.raises(KeyError, match="assay_log_w"):
        LS.total_loss(out, batch, {"loss_assay_weight": "learned"})


# ------------------------------------------------------------------- shape and multiscale


def test_shape_term_by_hand(batch):
    out = _out(batch, seed=7, dtype=torch.float64)
    mask = C.loss_mask(batch)
    y = batch["y_counts"].double()
    base, _ = LS.total_loss(out, batch, {})
    got, parts = LS.total_loss(out, batch, {"loss_shape_weight": 5})
    tot = 0.0
    for b in range(mask.shape[0]):
        for a in range(mask.shape[2]):
            m = mask[b, :, a]
            if not m.any():
                continue
            yy, mm = y[b, m, a], out["mu"][b, m, a]
            p = mm / mm.sum()
            N = yy.sum()
            tot += float(-torch.lgamma(N + 1) + torch.lgamma(yy + 1).sum() - (yy * p.log()).sum())
    want = tot / int(mask.sum())
    assert parts["shape"] == pytest.approx(want, rel=1e-10)
    assert float(got) == pytest.approx(float(base) + 5 * want, rel=1e-10)
    # scale-free in mu: multiplying a track's mu by a constant leaves the shape term unchanged
    o2 = dict(out, mu=out["mu"] * 7.0)
    assert LS.shape_nll(o2["mu"], y, mask) == pytest.approx(want, rel=1e-9)


def test_pool_nb_formula():
    mu = torch.tensor([[1.0, 2.0, 3.0, 4.0, 9.0]], dtype=torch.float64)[..., None]
    n = torch.tensor([[2.0, 1.0, 5.0, 0.5, 9.0]], dtype=torch.float64)[..., None]
    mp, npool = LS.pool_nb(mu, n, 2)
    assert mp.shape == (1, 2, 1)                                # the trailing bin is dropped
    np.testing.assert_allclose(mp[0, :, 0].numpy(), [3.0, 7.0])
    np.testing.assert_allclose(npool[0, :, 0].numpy(),
                               [9 / (1 / 2 + 4 / 1), 49 / (9 / 5 + 16 / 0.5)])
    # equal sizes and means: the variance of the sum is the sum of the variances
    mp, npool = LS.pool_nb(torch.full((1, 4, 1), 3.0), torch.full((1, 4, 1), 2.0), 4)
    assert float(mp + mp ** 2 / npool) == pytest.approx(4 * (3 + 9 / 2))


def test_multiscale_by_hand(batch):
    out = _out(batch, seed=8, dtype=torch.float64)
    mask = C.loss_mask(batch).clone()
    b = dict(batch)
    # make one same-view target's mask cover a partial block, so block selection matters
    base, _ = LS.total_loss(out, b, {})
    got, parts = LS.total_loss(out, b, {"loss_multiscale": [4, 16]})
    y = b["y_counts"].double()
    tot = 0.0
    for k in (4, 16):
        mp, npool = LS.pool_nb(out["mu"], out["n"], k)
        B_, L_, A_ = mask.shape
        yp = y.reshape(B_, L_ // k, k, A_).sum(2)
        full = mask.reshape(B_, L_ // k, k, A_).all(2)
        v = float(LS.nb_nll_per(npool[full], mp[full], yp[full]).mean())
        assert parts[f"multiscale_{k}"] == pytest.approx(v, rel=1e-10)
        tot += v
        if k == 4:
            partial = mask.reshape(B_, L_ // k, k, A_).any(2) & ~full
            assert partial.any()                     # same-view targets: some blocks are partial
    assert float(got) == pytest.approx(float(base) + tot, rel=1e-10)


@pytest.mark.parametrize("arm", [{"loss_bin_equal": True}, {"loss_assay_weight": "equal"},
                                 {"loss_shape_weight": 5}, {"loss_multiscale": [4, 16]}])
def test_each_arm_trains_for_5_steps(arm, tables, tmp_path):
    LS.reset_assay_ema()
    T.train(_cfg(1, windows_seen=20, **arm), tables, 0, tmp_path / "run", log_every=1)
    rec = [json.loads(x) for x in (tmp_path / "run/log.jsonl").read_text().splitlines()]
    assert len(rec) == 5 and all(np.isfinite(r["loss"]) for r in rec)
    LS.reset_assay_ema()
