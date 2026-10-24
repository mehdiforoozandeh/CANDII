"""t118 rung D — `tools/t118/ladder/fforms/form_d.py`: FiLM-modulated dilated CNN.

No dependency on the harness (t118-C1): the recovery test trains theta and D's own weights directly
with a local NB NLL.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.ndimage import maximum_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import base  # noqa: E402
from ladder.fforms import form_d  # noqa: E402

STATS = {"knots_x": np.linspace(0.0, 5.0, 12), "n0": 5.0, "sigma0": 0.5}
C = 30


def _form(space="counts", seed=0):
    torch.manual_seed(seed)
    return base.load_form("D", space, STATS)


def _theta(form, b):
    return form.init_theta().unsqueeze(0).expand(b, -1).clone()


def _randomise_head(form, seed=1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        form.head.weight.copy_(0.3 * torch.randn(form.head.weight.shape, generator=g))
        form.head.bias.copy_(0.1 * torch.randn(form.head.bias.shape, generator=g))


def _nb_nll(loc, disp, y):
    mu = torch.exp(loc).clamp(1e-4, 1e6)
    n = torch.exp(disp).clamp(1e-3, 1e4)
    ll = (torch.lgamma(y + n) - torch.lgamma(n) - torch.lgamma(y + 1.0)
          + n * torch.log(n / (n + mu)) + y * torch.log(mu / (n + mu)))
    return -ll.mean()


@pytest.mark.parametrize("space", ["counts", "pval"])
def test_attributes_and_load_form(space):
    form = _form(space)
    assert isinstance(form, base.FForm) and isinstance(form, form_d.FormD)
    assert (form.rung, form.space, form.context) == ("D", space, 30)
    assert form.n_theta == 4 * 32 * 2 == 256
    th = form.init_theta()
    assert th.shape == (256,)
    assert torch.equal(th[:128], torch.ones(128)) and torch.equal(th[128:], torch.zeros(128))
    # the trunk and head are the form's own trainable parameters: 4 convs + head, weight + bias
    names = {n for n, _ in form.named_parameters()}
    assert names == {f"convs.{i}.{w}" for i in range(4) for w in ("weight", "bias")} | \
        {"head.weight", "head.bias"}
    assert [c.dilation[0] for c in form.convs] == [1, 2, 4, 8]
    assert all(c.kernel_size[0] == 5 and c.out_channels == 32 for c in form.convs)


@pytest.mark.parametrize("space", ["counts", "pval"])
@pytest.mark.parametrize("b,length", [(1, 1), (3, 17), (4, 2048)])
def test_shapes_with_halo(space, b, length):
    form = _form(space)
    x = torch.randn(b, length + 2 * C)
    loc, disp = form(x, _theta(form, b))
    assert loc.shape == (b, length) and disp.shape == (b, length)
    with pytest.raises(ValueError):
        form(torch.randn(b, 2 * C), _theta(form, b))


@pytest.mark.parametrize("space,s0", [("counts", 5.0), ("pval", 0.5)])
def test_identity_at_init(space, s0):
    form = _form(space)
    x = torch.randn(3, 500 + 2 * C) * 2.0
    th = _theta(form, 3) + 0.5 * torch.randn(3, 256)   # any theta: the zero head decides
    loc, disp = form(x, th)
    assert torch.equal(loc, x[:, C:-C])
    assert torch.allclose(disp, torch.full_like(disp, math.log(s0)))


def test_receptive_field_is_61_bins():
    form = _form().double()
    _randomise_head(form)
    th = _theta(form, 1).double() + 0.2 * torch.randn(1, 256, dtype=torch.float64)
    L = 200
    x = torch.randn(1, L + 2 * C, dtype=torch.float64)
    loc0, disp0 = form(x, th)
    j = 100                                   # output bin j sits at x index j + C
    for off, moves in ((30, True), (-30, True), (31, False), (-31, False)):
        xp = x.clone()
        xp[0, j + C + off] += 5.0
        loc1, disp1 = form(xp, th)
        d = max(abs(loc1[0, j] - loc0[0, j]).item(), abs(disp1[0, j] - disp0[0, j]).item())
        assert (d > 1e-6) if moves else (d < 1e-12), (off, d)
    # a delta at one input bin moves exactly the 61 output bins centred on it
    xp = x.clone()
    xp[0, j + C] += 5.0
    loc1, _ = form(xp, th)
    moved = torch.nonzero((loc1 - loc0).abs()[0] > 1e-12).flatten()
    assert moved.tolist() == list(range(j - 30, j + 31))


def test_gradient_reaches_head_then_film_and_trunk():
    form = _form()
    x = torch.randn(2, 64 + 2 * C)
    y = torch.poisson(torch.full((2, 64), 3.0))
    th = _theta(form, 2).requires_grad_(True)
    _nb_nll(*form(x, th), y).backward()
    assert form.head.weight.grad.abs().sum() > 0
    # zero head: nothing upstream of it receives gradient on the first step
    assert th.grad.abs().sum() == 0
    _randomise_head(form)
    form.zero_grad()
    th.grad = None
    _nb_nll(*form(x, th), y).backward()
    assert th.grad[:, :128].abs().sum() > 0 and th.grad[:, 128:].abs().sum() > 0
    for i, conv in enumerate(form.convs):
        assert conv.weight.grad.abs().sum() > 0, i
        assert conv.bias.grad.abs().sum() > 0, i


def test_describe():
    form = _form()
    th = form.init_theta()
    th[5] = 2.5          # gamma, layer 0, channel 5
    th[128 + 32 + 7] = -1.25   # beta, layer 1, channel 7
    d = form.describe(th)
    assert set(d) == {"film_gamma", "film_beta"}
    g, b = np.asarray(d["film_gamma"]), np.asarray(d["film_beta"])
    assert g.shape == (4, 32) and b.shape == (4, 32)
    assert g[0, 5] == 2.5 and b[1, 7] == -1.25
    assert (np.delete(g.ravel(), 5) == 1.0).all() and (np.delete(b.ravel(), 32 + 7) == 0.0).all()
    assert isinstance(d["film_gamma"][0][0], float)


def _fit(forward, params, x_all, y_all, steps, lr, rng, win=256, batch=32):
    opt = torch.optim.Adam(params, lr=lr)
    n = x_all.shape[0]
    for _ in range(steps):
        s = rng.integers(C, n - win - C, batch)
        xs = torch.stack([x_all[i - C:i + win + C] for i in s])
        ys = torch.stack([y_all[i:i + win] for i in s])
        loss = _nb_nll(*forward(xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()


def test_recovers_context_dependent_broadening_better_than_identity_and_affine():
    """Target = Poisson(max of the source over +-4 bins): every peak widens by 4 bins each side,
    so a bin's target depends on its neighbours. D must beat the identity and the best per-bin
    affine map (rung A's form, fitted here locally) on held-out bins."""
    rng = np.random.default_rng(0)
    n = 80_000
    lam = rng.gamma(0.6, 2.0 / 0.6, n)
    for s in rng.integers(0, n - 20, n // 200):
        lam[s:s + int(rng.integers(3, 12))] += rng.gamma(2.0, 10.0)
    src = rng.poisson(lam).astype(np.float64)
    tgt = rng.poisson(maximum_filter1d(src, size=9) + 0.1).astype(np.float64)
    x_all = torch.tensor(np.log1p(src), dtype=torch.float32)
    y_all = torch.tensor(tgt, dtype=torch.float32)
    split = 60_000
    x_tr, y_tr = x_all[:split], y_all[:split]
    x_te, y_te = x_all[split - C:], y_all[split:n - C]

    torch.manual_seed(0)
    form = base.load_form("D", "counts", STATS)
    theta = form.init_theta().clone().requires_grad_(True)

    def fwd_d(xs):
        return form(xs, theta.unsqueeze(0).expand(xs.shape[0], -1))

    with torch.no_grad():
        nll_identity = _nb_nll(*fwd_d(x_te.unsqueeze(0)), y_te.unsqueeze(0)).item()
    _fit(fwd_d, list(form.parameters()) + [theta], x_tr, y_tr, 200, 3e-3, rng)
    with torch.no_grad():
        nll_d = _nb_nll(*fwd_d(x_te.unsqueeze(0)), y_te.unsqueeze(0)).item()

    ab = torch.tensor([0.0, 1.0, math.log(5.0)], requires_grad=True)

    def fwd_a(xs):
        xc = xs[:, C:-C]
        return ab[0] + ab[1] * xc, ab[2].expand_as(xc)

    _fit(fwd_a, [ab], x_tr, y_tr, 300, 1e-2, rng)
    with torch.no_grad():
        nll_a = _nb_nll(*fwd_a(x_te.unsqueeze(0)), y_te.unsqueeze(0)).item()

    assert nll_d < nll_identity - 0.2, (nll_d, nll_a, nll_identity)
    assert nll_d < nll_a - 0.1, (nll_d, nll_a, nll_identity)
