"""miniCANDI loss arms (chunk A1; plan pinned interface E, "Losses"): `losses.total_loss`, the NB
CRPS against the closed form, the PPML gradients by hand, the per-kind weights, the base leg, and
the default path bit for bit against the loss before this chunk."""
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

from candi.metrics import nb_crps as closed_form_nb_crps  # noqa: E402
from minicandi import contract as C  # noqa: E402
from minicandi import losses as LS  # noqa: E402
from minicandi import registry as R  # noqa: E402
from minicandi import synth  # noqa: E402
from minicandi import train as T  # noqa: E402
from minicandi.dataset import OnionDataset  # noqa: E402

L = 64
RUNGS = (1, 2, 5)


def _cfg(r: int, **over) -> dict:
    cfg = json.loads((_ROOT / f"tools/minicandi/configs/rung{r}/base.json").read_text())
    cfg.update(context_bins=L, param_budget=1.5e5, batch_size=4)
    cfg.update(over)
    return cfg


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> Path:
    td = synth.make_synthetic_tables(tmp_path_factory.mktemp("onion_losses"))
    synth.make_evalset(td, n_regions=2, region_bins=4 * L)
    return td


@pytest.fixture(scope="module")
def batch(tables) -> dict:
    return OnionDataset(tables, "train", _cfg(1, two_step=True), 0).batch(
        4, np.random.default_rng(0))


def _out(batch, seed=0, base=False):
    g = torch.Generator().manual_seed(seed)
    shp = batch["y_counts"].shape
    out = {"n": torch.exp(torch.randn(shp, generator=g)) * 3,
           "mu": torch.exp(torch.randn(shp, generator=g)) * 5}
    if base:
        out["n_base"] = torch.exp(torch.randn(shp, generator=g)) * 3
        out["mu_base"] = torch.exp(torch.randn(shp, generator=g)) * 5
    return out


# ------------------------------------------------------------------- NB CRPS


def test_nb_crps_matches_the_closed_form():
    n = np.array([0.5, 1.0, 2.0, 10.0, 50.0, 3.0, 0.2, 100.0, 5.0, 8.0])
    mu = np.array([0.1, 1.0, 5.0, 30.0, 200.0, 0.01, 40.0, 1000.0, 2.0, 12.0])
    y = np.array([0, 3, 0, 25, 180, 1, 0, 1100, 9, 12])
    p = n / (n + mu)
    want = closed_form_nb_crps(n, p, y)
    got = LS.nb_crps(torch.tensor(n), torch.tensor(mu), torch.tensor(y)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-9)   # the 10-sd tail cut: 1.1e-6 at n = 0.2, mu = 40


def test_nb_crps_far_from_zero_and_y_outside_the_window():
    """A window that starts above 0 (lo > 0) and observations below and above it."""
    n = np.array([100.0, 100.0, 100.0, 1000.0, 5.0])
    mu = np.array([5000.0, 5000.0, 5000.0, 300.0, 50.0])
    y = np.array([4500.0, 0.0, 20000.0, 290.0, 2000.0])
    want = closed_form_nb_crps(n, n / (n + mu), y)
    got = LS.nb_crps(torch.tensor(n), torch.tensor(mu), torch.tensor(y)).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-6)


def test_nb_crps_gaussian_fallback_past_the_window_cap():
    n, mu, y = np.array([10.0]), np.array([20000.0]), np.array([15000.0])
    assert np.sqrt(mu + mu ** 2 / n)[0] * 20 > LS.CRPS_K_CAP
    want = closed_form_nb_crps(n, n / (n + mu), y)
    got = LS.nb_crps(torch.tensor(n), torch.tensor(mu), torch.tensor(y)).numpy()
    np.testing.assert_allclose(got, want, rtol=0.15)   # skewed NB (n = 10): 11% off; a fallback only


def test_nb_crps_gradient_matches_finite_differences():
    n = torch.tensor([0.7, 2.0, 9.0], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([0.4, 6.0, 20.0], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([1.0, 2.0, 30.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda a, b: LS.nb_crps(a, b, y), (n, mu), eps=1e-6,
                                    atol=1e-5)


# ------------------------------------------------------------------- PPML


def test_poisson_mean_detached_size_gradients_by_hand():
    n = torch.tensor([2.0, 5.0], dtype=torch.float64, requires_grad=True)
    mu = torch.tensor([3.0, 0.5], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([4.0, 0.0], dtype=torch.float64)
    LS.poisson_mean_detached_size_per(n, mu, y).sum().backward()
    # mean: only the Poisson part, d/dmu (mu - y log mu) = 1 - y/mu
    np.testing.assert_allclose(mu.grad.numpy(), (1 - y / mu).detach().numpy(), rtol=1e-12)
    # size: only the NB part, d/dn of -[lgamma(y+n) - lgamma(n) + n log(n/(n+mu)) + y log(mu/(n+mu))]
    nn_, mm = n.detach(), mu.detach()
    dn = -(torch.digamma(y + nn_) - torch.digamma(nn_) + torch.log(nn_ / (nn_ + mm))
           + 1 - nn_ / (nn_ + mm) - y / (nn_ + mm))
    np.testing.assert_allclose(n.grad.numpy(), dn.numpy(), rtol=1e-10)


# ------------------------------------------------------------------- total_loss


def test_default_is_the_pre_A1_loss_bit_for_bit(batch):
    out = _out(batch)
    want = C.nb_nll(out["n"], out["mu"], batch["y_counts"], C.loss_mask(batch))
    for cfg in ({}, {"loss": "nb_nll"}, {"loss_weights": {"impute": 1, "same": 1.0}}):
        got, parts = LS.total_loss(out, batch, cfg)
        assert torch.equal(got, want), cfg
        assert parts["main"] == float(want)
        assert {"impute", "correct", "same"} <= set(parts)


@pytest.mark.parametrize("name", LS.LOSSES)
def test_kind_weights_by_hand(batch, name):
    out = _out(batch, seed=1)
    w = {"impute": 4.0, "correct": 0.5, "same": 2.0}
    got, parts = LS.total_loss(out, batch, {"loss": name, "loss_weights": w})
    mask = C.loss_mask(batch)
    per = LS.PER_BIN[name](out["n"], out["mu"], batch["y_counts"])
    kind = batch["y_kind"][:, None, :].expand_as(mask)
    tot = 0.0
    for kd, key in ((1, "impute"), (2, "correct"), (3, "same")):
        sel = mask & (kind == kd)
        assert sel.any()
        tot += w[key] * float(per[sel].double().sum())
        assert parts[key] == pytest.approx(float(per[sel].double().mean()), rel=1e-5)
    assert float(got) == pytest.approx(tot / int(mask.sum()), rel=1e-5)


def test_base_leg_by_hand(batch):
    out = _out(batch, seed=2, base=True)
    mask = C.loss_mask(batch)
    has = batch["c_base"][..., 0] != 0
    b = dict(batch)
    c_base = b["c_base"].clone()
    t0 = int(b["y_avail"][0].nonzero()[0])
    c_base[0, t0] = 0.0                                   # this track has no base view
    b["c_base"] = c_base
    has = c_base[..., 0] != 0
    main = C.nb_nll(out["n"], out["mu"], b["y_counts"], mask)
    base = C.nb_nll(out["n_base"], out["mu_base"], b["y_base"], mask & has[:, None, :])
    got, parts = LS.total_loss(out, b, {})
    assert torch.equal(got, main + base)
    assert parts["base"] == float(base)
    got3, _ = LS.total_loss(out, b, {"loss_weights": {"base": 3.0}})
    assert torch.allclose(got3, main + 3.0 * base)
    got0, _ = LS.total_loss(out, b, {"loss_weights": {"base": 0.0}})
    assert torch.equal(got0, main + 0.0 * base)
    # the excluded track's base counts move nothing; same-view targets use hidden bins only
    yb = b["y_base"].clone()
    yb[0, :, t0] += 100
    assert torch.equal(LS.total_loss(out, dict(b, y_base=yb), {})[0], got)
    same = (b["y_kind"] == C.KIND_SAME)
    bb, tt = same.nonzero()[0].tolist()
    vis = ~b["x_bin_mask"][bb, :, tt]
    yb = b["y_base"].clone()
    yb[bb, vis, tt] += 100
    assert torch.equal(LS.total_loss(out, dict(b, y_base=yb), {})[0], got)


def test_bad_config_is_refused(batch):
    out = _out(batch)
    with pytest.raises(ValueError, match="loss"):
        LS.total_loss(out, batch, {"loss": "mse"})
    with pytest.raises(ValueError, match="loss_weights"):
        LS.total_loss(out, batch, {"loss_weights": {"imputation": 2}})


# ------------------------------------------------------------------- training


@pytest.mark.parametrize("r", RUNGS)
@pytest.mark.parametrize("name", LS.LOSSES)
def test_the_loss_falls_over_30_steps(r, name, tables):
    """30 Adam steps on one fixed batch (two-step on, so the base leg trains too)."""
    cfg = _cfg(r, two_step=True, loss=name)
    b = OnionDataset(tables, "train", cfg, 0).batch(4, np.random.default_rng(0))
    torch.manual_seed(0)
    m = R.build_model(cfg).train()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    hist = []
    for _ in range(30):
        loss, _ = LS.total_loss(m(b), b, cfg)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(float(loss))
    assert np.isfinite(hist).all()
    assert hist[-1] < 0.9 * hist[0], (hist[0], hist[-1])


def _old_step_loss(model, batch, config=None, parts=None):
    """train.step_loss as it was before chunk A1."""
    out = model(batch)
    return C.nb_nll(out["n"], out["mu"], batch["y_counts"], C.loss_mask(batch))


@pytest.mark.parametrize("r", RUNGS)
def test_a_config_without_the_new_keys_trains_as_before(r, tables, tmp_path, monkeypatch):
    cfg = _cfg(r, windows_seen=20)
    T.train(cfg, tables, 0, tmp_path / "new", log_every=1)
    monkeypatch.setattr(T, "step_loss", _old_step_loss)
    T.train(cfg, tables, 0, tmp_path / "old", log_every=1)
    new = [json.loads(x) for x in (tmp_path / "new/log.jsonl").read_text().splitlines()]
    old = [json.loads(x) for x in (tmp_path / "old/log.jsonl").read_text().splitlines()]
    assert len(new) == len(old) == 5
    assert [x["loss"] for x in new] == [x["loss"] for x in old]
    assert [x["grad_norm"] for x in new] == [x["grad_norm"] for x in old]
    assert all(x["parts"]["main"] == x["loss"] for x in new)
    a = torch.load(tmp_path / "new/ckpt.pt", weights_only=False)["state_dict"]
    b = torch.load(tmp_path / "old/ckpt.pt", weights_only=False)["state_dict"]
    assert all(torch.equal(a[k], b[k]) for k in a)
