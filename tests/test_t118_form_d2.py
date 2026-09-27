"""t118 rung D2 — `tools/t118/ladder/fforms/form_d2.py`: FiLM per bin (row 2 of rung D).

No dependency on the harness: the form is driven directly with a per-position theta [B, W, 256].
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import base  # noqa: E402
from ladder.fforms import form_d, form_d2  # noqa: E402

STATS = {"knots_x": np.linspace(0.0, 5.0, 12), "n0": 5.0, "sigma0": 0.5}
C = 30


def _form(space="counts", seed=0):
    torch.manual_seed(seed)
    return base.load_form("D2", space, STATS)


def _theta(form, b, w):
    return form.init_theta().reshape(1, 1, -1).expand(b, w, -1).clone()


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
    assert isinstance(form, base.FForm) and isinstance(form, form_d2.FormD2)
    assert (form.rung, form.space, form.context, form.n_theta) == ("D2", space, 30, 256)
    assert form_d2.FormD2.per_bin is True and form.per_bin is True
    th = form.init_theta()
    assert torch.equal(th[:128], torch.ones(128)) and torch.equal(th[128:], torch.zeros(128))
    names = {n for n, _ in form.named_parameters()}
    assert names == {f"convs.{i}.{w}" for i in range(4) for w in ("weight", "bias")} | \
        {"head.weight", "head.bias"}
    assert [c.dilation[0] for c in form.convs] == [1, 2, 4, 8]
    assert all(c.kernel_size[0] == 5 and c.out_channels == 32 for c in form.convs)
    assert form.head.weight.abs().sum() == 0 and form.head.bias.abs().sum() == 0


@pytest.mark.parametrize("space", ["counts", "pval"])
@pytest.mark.parametrize("b,length", [(1, 1), (3, 17), (4, 2048)])
def test_shapes_with_halo(space, b, length):
    form = _form(space)
    w = length + 2 * C
    loc, disp = form(torch.randn(b, w), _theta(form, b, w))
    assert loc.shape == (b, length) and disp.shape == (b, length)
    with pytest.raises(ValueError):                     # no bins left after the halo
        form(torch.randn(b, 2 * C), _theta(form, b, 2 * C))
    with pytest.raises(ValueError):                     # row-1 theta [B, 256]
        form(torch.randn(b, w), form.init_theta().expand(b, -1))
    with pytest.raises(ValueError):                     # theta over the centre only, no halo
        form(torch.randn(b, w), _theta(form, b, length))


@pytest.mark.parametrize("space,s0", [("counts", 5.0), ("pval", 0.5)])
def test_identity_at_init_whatever_theta(space, s0):
    form = _form(space)
    w = 500 + 2 * C
    x = torch.randn(3, w) * 2.0
    th = _theta(form, 3, w) + 0.5 * torch.randn(3, w, 256)   # varies per position too
    loc, disp = form(x, th)
    assert torch.equal(loc, x[:, C:-C])
    assert torch.allclose(disp, torch.full_like(disp, math.log(s0)))


@pytest.mark.parametrize("space", ["counts", "pval"])
def test_constant_theta_equals_form_d_exactly(space):
    d = base.load_form("D", space, STATS)
    d2 = _form(space, seed=7)
    _randomise_head(d)
    d2.load_state_dict(d.state_dict())
    assert d2.disp0 == d.disp0
    b, w = 3, 300 + 2 * C
    x = torch.randn(b, w) * 1.5
    th = d.init_theta().unsqueeze(0) + 0.3 * torch.randn(b, 256)      # one theta per row
    loc_d, disp_d = d(x, th)
    loc_2, disp_2 = d2(x, th.unsqueeze(1).expand(b, w, 256))
    assert torch.equal(loc_2, loc_d) and torch.equal(disp_2, disp_d)


def test_film_views_layout():
    form = _form()
    b, w = 2, 5
    th = torch.randn(b, w, 256)
    gamma, beta = form._film_bins(th)
    assert gamma.shape == beta.shape == (b, 4, 32, w)
    assert gamma.data_ptr() == th.data_ptr()            # a view, not a copy
    for i in range(4):
        for c in (0, 31):
            assert torch.equal(gamma[:, i, c, :], th[:, :, i * 32 + c])
            assert torch.equal(beta[:, i, c, :], th[:, :, 128 + i * 32 + c])


def test_receptive_field_is_61_bins():
    form = _form().double()
    _randomise_head(form)
    L = 200
    w = L + 2 * C
    th = _theta(form, 1, w).double() + 0.2 * torch.randn(1, w, 256, dtype=torch.float64)
    x = torch.randn(1, w, dtype=torch.float64)
    loc0, disp0 = form(x, th)
    j = 100                                   # output bin j sits at input index j + C
    for off, moves in ((30, True), (-30, True), (31, False), (-31, False)):
        xp = x.clone()
        xp[0, j + C + off] += 5.0
        loc1, disp1 = form(xp, th)
        d = max(abs(loc1[0, j] - loc0[0, j]).item(), abs(disp1[0, j] - disp0[0, j]).item())
        assert (d > 1e-6) if moves else (d < 1e-12), (off, d)
    # theta at one position enters after layer 0's conv, so it reaches 2 * (2 + 4 + 8) = 28 bins
    # each side (layers 1-3 only), and every halo position within that reach matters
    for off, moves in ((28, True), (-28, True), (29, False), (-29, False)):
        tp = th.clone()
        tp[0, j + C + off] += 0.5
        loc1, disp1 = form(x, tp)
        d = max(abs(loc1[0, j] - loc0[0, j]).item(), abs(disp1[0, j] - disp0[0, j]).item())
        assert (d > 1e-6) if moves else (d < 1e-12), ("theta", off, d)
    xp = x.clone()
    xp[0, j + C] += 5.0
    loc1, _ = form(xp, th)
    moved = torch.nonzero((loc1 - loc0).abs()[0] > 1e-12).flatten()
    assert moved.tolist() == list(range(j - 30, j + 31))


def test_gradient_reaches_head_then_film_and_trunk():
    form = _form()
    w = 64 + 2 * C
    x = torch.randn(2, w)
    y = torch.poisson(torch.full((2, 64), 3.0))
    th = _theta(form, 2, w).requires_grad_(True)
    _nb_nll(*form(x, th), y).backward()
    assert form.head.weight.grad.abs().sum() > 0
    assert th.grad.abs().sum() == 0                     # zero head: nothing upstream moves
    for conv in form.convs:
        assert conv.weight.grad is None or conv.weight.grad.abs().sum() == 0
    _randomise_head(form)
    form.zero_grad()
    th.grad = None
    _nb_nll(*form(x, th), y).backward()
    assert th.grad[..., :128].abs().sum() > 0 and th.grad[..., 128:].abs().sum() > 0
    # the halo's theta receives gradient too (it reaches the centre through later layers)
    assert th.grad[:, :C].abs().sum() > 0 and th.grad[:, -C:].abs().sum() > 0
    for i, conv in enumerate(form.convs):
        assert conv.weight.grad.abs().sum() > 0, i
        assert conv.bias.grad.abs().sum() > 0, i


def test_describe():
    form = _form()
    k = 3
    th = form.init_theta().unsqueeze(0).repeat(k, 1)
    th[0, 5] = 2.5                    # gamma, level 0, layer 0, channel 5: |γ−1| = 1.5
    th[2, 32 + 7] = 0.0               # gamma, level 2, layer 1, channel 7: |γ−1| = 1
    th[1, 128 + 3 * 32 + 1] = -1.25   # beta, level 1, layer 3, channel 1
    d = form.describe(th)
    assert set(d) == {"n_levels", "film_gamma_abs_dev", "film_beta_abs"}
    assert d["n_levels"] == k
    g, b = np.asarray(d["film_gamma_abs_dev"]), np.asarray(d["film_beta_abs"])
    assert g.shape == b.shape == (k, 4)
    want_g = np.zeros((k, 4))
    want_g[0, 0] = 1.5 / 32
    want_g[2, 1] = 1.0 / 32
    want_b = np.zeros((k, 4))
    want_b[1, 3] = 1.25 / 32
    np.testing.assert_allclose(g, want_g, rtol=0, atol=1e-15)
    np.testing.assert_allclose(b, want_b, rtol=0, atol=1e-15)
    assert isinstance(d["film_gamma_abs_dev"][0][0], float)
    assert isinstance(d["film_beta_abs"][0][0], float)
    with pytest.raises(ValueError):
        form.describe(form.init_theta())                # [256], not [K, 256]


def test_row1_form_d_untouched():
    assert form_d.FormD.rung == "D" and not getattr(form_d.FormD, "per_bin", False)
