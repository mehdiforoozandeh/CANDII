"""t118 rung A2 — `tools/t118/ladder/fforms/form_a2.py`: affine per bin against the row-2 contract.

Row-2 contract (plan_row2 R1 Interfaces): forward(x [B, W], theta [B, W, n_theta]) -> (loc [B, L],
disp [B, L]), W = L + 2*context; describe(theta [K, n_theta]) -> {"n_levels": K, ...}. No harness
(model.py) is used here: theta is built by hand, or by a two-level lookup on x standing in for g.
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
from ladder.fforms import form_a2  # noqa: E402

STATS = {"knots_x": np.linspace(0.0, 5.0, 12), "n0": 5.0, "sigma0": 0.5}


def planted_a(x: torch.Tensor) -> torch.Tensor:
    """The planted level-dependent shift: a(x) = 0.5 for x > 1, else 0."""
    return torch.where(x > 1.0, torch.tensor(0.5), torch.tensor(0.0))


# ---------------------------------------------------------------------------------------------
# contract
@pytest.mark.parametrize("space", ["counts", "pval"])
def test_build_and_load_form(space):
    f = base.load_form("A2", space, STATS)
    assert isinstance(f, form_a2.FormA2) and isinstance(f, base.FForm)
    assert (f.rung, f.space, f.n_theta, f.context, f.per_bin) == ("A2", space, 3, 0, True)
    assert form_a2.FormA2.per_bin is True
    assert isinstance(form_a2.build(space, STATS), base.FForm)


@pytest.mark.parametrize("space", ["counts", "pval"])
def test_shapes(space):
    f = form_a2.build(space, STATS)
    B, L = 4, 37
    x = torch.randn(B, L)
    theta = torch.randn(B, L, 3)
    loc, disp = f(x, theta)
    assert loc.shape == (B, L) and disp.shape == (B, L)
    # per-bin theta: each bin uses its own (a, b, d)
    assert torch.allclose(loc, theta[..., 0] + theta[..., 1] * x)
    assert torch.equal(disp, theta[..., 2])


@pytest.mark.parametrize("bad_x,bad_theta", [
    ((4, 37), (4, 3)),         # row-1 theta [B, n_theta]
    ((4, 37), (4, 37, 2)),     # wrong n_theta
    ((4, 37), (4, 36, 3)),     # theta length != W
    ((4, 37), (3, 37, 3)),     # batch mismatch
    ((37,), (37, 3)),          # x not [B, L]
    ((4, 37), (1, 4, 37, 3)),  # theta 4-d
])
def test_bad_shapes_raise(bad_x, bad_theta):
    f = form_a2.build("counts", STATS)
    with pytest.raises(ValueError):
        f(torch.randn(*bad_x), torch.randn(*bad_theta))


@pytest.mark.parametrize("space,d0", [("counts", 5.0), ("pval", 0.5)])
def test_init_theta_is_identity(space, d0):
    f = form_a2.build(space, STATS)
    th = f.init_theta()
    assert th.shape == (3,) and th.dtype == torch.float32
    assert torch.allclose(th, torch.tensor([0.0, 1.0, math.log(d0)]))
    x = torch.randn(3, 50) * 3
    loc, disp = f(x, th.expand(3, 50, 3))
    assert torch.equal(loc, x)
    assert torch.allclose(disp, torch.full_like(x, math.log(d0)))


def test_gradients_reach_theta():
    f = form_a2.build("counts", STATS)
    theta = f.init_theta().expand(2, 10, 3).clone().requires_grad_(True)
    loc, disp = f(torch.rand(2, 10) + 0.1, theta)
    (loc.sum() + disp.sum()).backward()
    assert theta.grad is not None and torch.all(theta.grad != 0)


# ---------------------------------------------------------------------------------------------
# planted level-dependent map
@pytest.mark.parametrize("space", ["counts", "pval"])
def test_planted_level_map_from_hand_built_theta(space):
    """theta_i = init_theta + (a(x_i), 0, 0) gives loc = x + a(x): a map no single line can make."""
    f = form_a2.build(space, STATS)
    x = torch.linspace(-2.0, 4.0, 121).reshape(1, -1).repeat(2, 1)
    theta = f.init_theta().expand(*x.shape, 3).clone()
    theta[..., 0] = planted_a(x)
    loc, disp = f(x, theta)
    assert torch.allclose(loc, x + planted_a(x))
    assert torch.allclose(loc[x <= 1.0], x[x <= 1.0])
    assert torch.allclose(loc[x > 1.0], x[x > 1.0] + 0.5)
    assert torch.equal(disp, theta[..., 2])


def test_planted_level_map_recovered_by_fit_pval():
    """Target = source · exp(a(x) + 0.1 noise). A two-level theta table gathered by (x > 1), standing
    in for g reading x, is fitted by log-normal NLL: a recovers 0 below and 0.5 above, b ≈ 1."""
    rng = np.random.default_rng(1182)
    x_np = rng.uniform(-2.0, 4.0, 100_000)
    x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
    t = x + planted_a(x) + 0.1 * torch.tensor(rng.standard_normal(x_np.size), dtype=torch.float32)
    f = form_a2.build("pval", STATS)
    table = f.init_theta().repeat(2, 1).requires_grad_(True)
    level = (x > 1.0).long()
    opt = torch.optim.Adam([table], lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        loc, disp = f(x, table[level])
        loss = (disp + 0.5 * ((t - loc) / torch.exp(disp)) ** 2).mean()
        loss.backward()
        opt.step()
    d = f.describe(table)
    assert d["n_levels"] == 2
    assert d["a"][0] == pytest.approx(0.0, abs=0.05), d
    assert d["a"][1] == pytest.approx(0.5, abs=0.05), d
    assert d["b"] == pytest.approx([1.0, 1.0], abs=0.05), d
    assert [math.exp(v) for v in d["disp"]] == pytest.approx([0.1, 0.1], abs=0.02), d


# ---------------------------------------------------------------------------------------------
# describe
@pytest.mark.parametrize("space,d0", [("counts", 5.0), ("pval", 0.5)])
def test_describe_keys(space, d0):
    f = form_a2.build(space, STATS)
    K = 12
    theta = f.init_theta().repeat(K, 1)
    theta[:, 0] = torch.arange(K, dtype=torch.float32) / 10
    d = f.describe(theta)
    assert set(d) == {"n_levels", "a", "b", "disp"}
    assert d["n_levels"] == K and type(d["n_levels"]) is int
    for key in ("a", "b", "disp"):
        assert isinstance(d[key], list) and len(d[key]) == K
        assert all(type(v) is float for v in d[key])
    assert d["a"] == pytest.approx([k / 10 for k in range(K)])
    assert d["b"] == [1.0] * K
    assert d["disp"] == pytest.approx([math.log(d0)] * K)
    with pytest.raises(ValueError):
        f.describe(torch.zeros(3))
    with pytest.raises(ValueError):
        f.describe(torch.zeros(K, 2))
