"""t125 rung A — `tools/t125/eta/fforms/form_a.py`: the row-1 affine f whose NB spread is set by level.

Pinned by plan_t125.md §t125-P1b and plan/ETA_OFFSET_AND_H_DESIGN.md ("A and B set it by level (a
12-knot curve over eta_in)"). theta = (a, b, d_0..d_11): loc = a + b·x; disp(x) = linear interpolation
of d_k over `stats["knots_x"]`, clamped to d_0 / d_11 outside the knot range (form B's rule).
The loader is `eta.model.load_form` (chunk P1c), so this test imports `eta.fforms.form_a.build` directly.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t118", ROOT / "tools" / "t125"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eta.fforms import form_a as eta_form_a  # noqa: E402
from ladder import base  # noqa: E402
from ladder.fforms import form_a as t118_form_a  # noqa: E402
from ladder.fforms import form_b as t118_form_b  # noqa: E402

N0, SIGMA0 = 5.0, 0.5
SPACES = ("counts", "pval")


def _stats(seed: int = 0) -> dict:
    """knots from a heavy-tailed count sample, shifted by a depth offset as eta_in is."""
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.gamma(0.5, 10.0, size=200_000))
    log_d = math.log(30e6 / 1e6)
    return {"knots_x": base.knot_stats(np.log1p(X) - log_d), "n0": N0, "sigma0": SIGMA0}


@pytest.fixture(scope="module")
def stats():
    return _stats()


def _grid(kx: np.ndarray, below: float = 2.0, above: float = 3.0, n: int = 4001) -> torch.Tensor:
    return torch.linspace(float(kx[0]) - below, float(kx[-1]) + above, n, dtype=torch.float32)


# ---- contract -----------------------------------------------------------------------------------

@pytest.mark.parametrize("space", SPACES)
def test_attributes_and_build(stats, space):
    f = eta_form_a.build(space, stats)
    assert isinstance(f, eta_form_a.FormA) and isinstance(f, base.FForm)
    assert (f.rung, f.space, f.n_theta, f.context, f.per_bin) == ("A", space, 14, 0, False)
    th = f.init_theta()
    assert th.shape == (14,) and th.dtype == torch.float32
    assert list(f.state_dict()) == []  # knots come from stats, not the checkpoint


@pytest.mark.parametrize("space", SPACES)
def test_init_theta_values(stats, space):
    f = eta_form_a.build(space, stats)
    d0 = math.log(N0 if space == "counts" else SIGMA0)
    expect = torch.tensor([0.0, 1.0] + [d0] * 12, dtype=torch.float32)
    assert torch.equal(f.init_theta(), expect)


@pytest.mark.parametrize("space", SPACES)
def test_shapes(stats, space):
    f = eta_form_a.build(space, stats)
    B, L = 5, 300
    x = torch.randn(B, L) * 2
    theta = f.init_theta().expand(B, -1) + 0.3 * torch.randn(B, 14)
    loc, disp = f(x, theta)
    assert loc.shape == (B, L) and disp.shape == (B, L)
    assert torch.isfinite(loc).all() and torch.isfinite(disp).all()


@pytest.mark.parametrize("x_shape, th_shape", [
    ((4, 10), (4, 3)),        # t118 form A's theta
    ((4, 10), (3, 14)),       # batch mismatch
    ((4, 10), (14,)),         # theta not 2-D
    ((40,), (1, 14)),         # x not 2-D
    ((4, 10), (4, 10, 14)),   # a row-2 per-bin theta
])
def test_bad_shapes_raise(stats, x_shape, th_shape):
    f = eta_form_a.build("counts", stats)
    with pytest.raises(ValueError):
        f(torch.zeros(x_shape), torch.zeros(th_shape))


@pytest.mark.parametrize("bad", [np.arange(11.0), np.arange(13.0),
                                 np.r_[np.arange(11.0), 10.0]])
def test_bad_knots_raise(bad):
    with pytest.raises(ValueError):
        eta_form_a.build("counts", {"knots_x": bad, "n0": N0, "sigma0": SIGMA0})


def test_bad_space_raises(stats):
    with pytest.raises(ValueError):
        eta_form_a.build("bogus", stats)


# ---- identity and reduction to t118 form A ------------------------------------------------------

@pytest.mark.parametrize("space", SPACES)
def test_identity_at_init_equals_t118_form_a(stats, space):
    f = eta_form_a.build(space, stats)
    old = t118_form_a.build(space, stats)
    x = _grid(stats["knots_x"]).reshape(1, -1)
    loc, disp = f(x, f.init_theta().reshape(1, -1))
    loc0, disp0 = old(x, old.init_theta().reshape(1, -1))
    assert torch.equal(loc, x)  # the identity map: loc = x
    assert torch.equal(loc, loc0) and torch.equal(disp, disp0)
    d0 = math.log(N0 if space == "counts" else SIGMA0)
    assert torch.equal(disp, torch.full_like(x, d0))


@pytest.mark.parametrize("space", SPACES)
def test_equal_dk_equals_t118_form_a_exactly(stats, space):
    f = eta_form_a.build(space, stats)
    old = t118_form_a.build(space, stats)
    g = torch.Generator().manual_seed(1)
    B = 6
    abd = torch.randn(B, 3, generator=g)
    theta = torch.cat([abd[:, :2], abd[:, 2:3].expand(B, 12)], dim=1)
    x = _grid(stats["knots_x"]).reshape(1, -1).expand(B, -1).contiguous()
    loc, disp = f(x, theta)
    loc0, disp0 = old(x, abd)
    assert torch.equal(loc, loc0) and torch.equal(disp, disp0)


# ---- the spread set by level --------------------------------------------------------------------

def test_planted_ramp_at_knots_and_clamped(stats):
    f = eta_form_a.build("counts", stats)
    kx = stats["knots_x"]
    d = torch.linspace(-1.0, 4.0, 12)
    theta = torch.cat([torch.tensor([0.3, 0.8]), d]).reshape(1, -1)
    # at the knots: disp equals d_k
    xk = torch.as_tensor(kx, dtype=torch.float32).reshape(1, -1)
    _, disp_k = f(xk, theta)
    assert torch.allclose(disp_k[0], d, atol=1e-5)
    # outside the knot range: clamped to d_0 below, d_11 above
    xo = torch.tensor([[float(kx[0]) - 5.0, float(kx[0]) - 0.01,
                        float(kx[-1]) + 0.01, float(kx[-1]) + 50.0]])
    _, disp_o = f(xo, theta)
    assert torch.allclose(disp_o[0], torch.tensor([d[0], d[0], d[11], d[11]]), atol=1e-6)
    # between two knots: the linear interpolation
    xm = 0.25 * kx[3] + 0.75 * kx[4]
    _, disp_m = f(torch.tensor([[xm]], dtype=torch.float32), theta)
    assert disp_m.item() == pytest.approx(0.25 * float(d[3]) + 0.75 * float(d[4]), abs=1e-5)
    # the mean is still affine in x, with no knots
    loc, _ = f(xo, theta)
    assert torch.allclose(loc, 0.3 + 0.8 * xo, atol=1e-6)


@pytest.mark.parametrize("space", SPACES)
def test_disp_equals_t118_form_b_rule(stats, space):
    """form B's dispersion rule, with form B's curve set to anything: disp must match."""
    f = eta_form_a.build(space, stats)
    fb = t118_form_b.build(space, stats)
    g = torch.Generator().manual_seed(2)
    B = 4
    d = torch.randn(B, 12, generator=g)
    theta = torch.cat([torch.randn(B, 2, generator=g), d], dim=1)
    theta_b = torch.cat([fb.init_theta()[:12].expand(B, -1), d], dim=1)
    x = _grid(stats["knots_x"]).reshape(1, -1).expand(B, -1).contiguous()
    _, disp = f(x, theta)
    _, disp_b = fb(x, theta_b)
    assert torch.allclose(disp, disp_b, atol=1e-6)


def test_gradients_reach_every_theta(stats):
    f = eta_form_a.build("counts", stats)
    kx = stats["knots_x"]
    # one x at each knot's midpoint to the next, plus both ends: every d_k gets weight
    xs = np.r_[kx, 0.5 * (kx[:-1] + kx[1:])]
    x = torch.as_tensor(xs, dtype=torch.float32).reshape(1, -1)
    theta = f.init_theta().reshape(1, -1).clone().requires_grad_(True)
    loc, disp = f(x, theta)
    (loc.sum() + disp.sum()).backward()
    assert theta.grad is not None and (theta.grad.abs() > 0).all()


# ---- describe -----------------------------------------------------------------------------------

@pytest.mark.parametrize("space", SPACES)
def test_describe(stats, space):
    f = eta_form_a.build(space, stats)
    theta = torch.cat([torch.tensor([0.5, 1.5]), torch.arange(12, dtype=torch.float32)])
    out = f.describe(theta)
    assert set(out) == {"a", "b", "disp", "knots_x"}
    assert out["a"] == pytest.approx(0.5) and out["b"] == pytest.approx(1.5)
    assert isinstance(out["a"], float) and isinstance(out["b"], float)
    assert out["disp"] == pytest.approx(list(range(12)))
    assert len(out["disp"]) == 12 and all(isinstance(v, float) for v in out["disp"])
    assert out["knots_x"] == pytest.approx(list(stats["knots_x"]))
    assert len(out["knots_x"]) == 12 and all(isinstance(v, float) for v in out["knots_x"])
    # a [1, 14] theta is accepted; a wrong count is not
    assert f.describe(theta.reshape(1, -1)) == out
    with pytest.raises(ValueError):
        f.describe(torch.zeros(3))
