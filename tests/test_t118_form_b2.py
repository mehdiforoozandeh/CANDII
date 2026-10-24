"""t118 ladder rung B2 — `tools/t118/ladder/fforms/form_b2.py`: rung B's curve with per-bin theta.

No harness import: B2 is checked against rung B (row 1) and against hand-built per-bin thetas.
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
from ladder.fforms import form_b, form_b2  # noqa: E402

N0, SIGMA0 = 5.0, 0.5


def _stats(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.gamma(0.5, 10.0, size=200_000))
    return {"knots_x": base.knot_stats(np.log1p(X)), "n0": N0, "sigma0": SIGMA0}


@pytest.fixture(scope="module")
def stats():
    return _stats()


def _grid(kx: np.ndarray, below: float = 2.0, above: float = 3.0, n: int = 4001) -> torch.Tensor:
    return torch.linspace(float(kx[0]) - below, float(kx[-1]) + above, n, dtype=torch.float32)


# ---- contract -----------------------------------------------------------------------------------

@pytest.mark.parametrize("space", ["counts", "pval"])
def test_load_form_and_attributes(stats, space):
    f = base.load_form("B2", space, stats)
    assert isinstance(f, form_b2.FormB2) and isinstance(f, base.FForm)
    assert (f.rung, f.space, f.n_theta, f.context) == ("B2", space, 24, 0)
    assert f.per_bin is True and form_b2.FormB2.per_bin is True
    assert f.init_theta().shape == (24,) and f.init_theta().dtype == torch.float32
    assert list(f.state_dict()) == []


def test_init_theta_equals_form_b(stats):
    for space in ("counts", "pval"):
        torch.testing.assert_close(form_b2.build(space, stats).init_theta(),
                                   form_b.build(space, stats).init_theta(), rtol=0, atol=0)


def test_shapes_and_bad_theta(stats):
    f = form_b2.build("counts", stats)
    B, L = 4, 300
    x = torch.randn(B, L) * 2 + 1
    theta = f.init_theta() + 0.3 * torch.randn(B, L, 24)
    loc, disp = f(x, theta)
    assert loc.shape == (B, L) and disp.shape == (B, L)
    assert torch.isfinite(loc).all() and torch.isfinite(disp).all()
    with pytest.raises(ValueError):
        f(x, theta[:, :, 0])                     # wrong rank: [B, L]
    with pytest.raises(ValueError):
        f(x, theta[:, :-1])                      # L mismatch


def test_bad_knots_rejected(stats):
    with pytest.raises(ValueError):
        form_b2.build("counts", {**stats, "knots_x": stats["knots_x"][:11]})
    kx = stats["knots_x"].copy()
    kx[5] = kx[4]
    with pytest.raises(ValueError):
        form_b2.build("counts", {**stats, "knots_x": kx})


# ---- theta constant over bins equals rung B exactly ---------------------------------------------

@pytest.mark.parametrize("space", ["counts", "pval"])
def test_constant_theta_equals_form_b_exactly(stats, space):
    fb, fb2 = form_b.build(space, stats), form_b2.build(space, stats)
    g = torch.Generator().manual_seed(5)
    B = 6
    theta = fb.init_theta() + 1.5 * torch.randn(B, 24, generator=g)
    x = torch.cat([_grid(stats["knots_x"], n=1001).expand(B, -1),
                   torch.rand(B, 200, generator=g) * 12 - 3], dim=1)
    L = x.shape[1]
    loc_b, disp_b = fb(x, theta)
    # expanded view and a materialised copy: both must match bit for bit
    for th2 in (theta.unsqueeze(1).expand(B, L, 24), theta.unsqueeze(1).repeat(1, L, 1)):
        loc, disp = fb2(x, th2)
        assert torch.equal(loc, loc_b) and torch.equal(disp, disp_b)


def test_constant_theta_gradient_matches_form_b(stats):
    fb, fb2 = form_b.build("counts", stats), form_b2.build("counts", stats)
    theta0 = fb.init_theta() + 0.2
    x = _grid(stats["knots_x"], 1.0, 1.0, n=500).unsqueeze(0)
    t1 = theta0.clone().unsqueeze(0).requires_grad_(True)
    loc, disp = fb(x, t1)
    (loc.square().sum() + disp.square().sum()).backward()
    t2 = theta0.clone().requires_grad_(True)
    loc2, disp2 = fb2(x, t2.expand(1, x.shape[1], 24))
    (loc2.square().sum() + disp2.square().sum()).backward()
    torch.testing.assert_close(t2.grad, t1.grad[0], rtol=1e-5, atol=1e-4)


# ---- identity at init ---------------------------------------------------------------------------

@pytest.mark.parametrize("space,disp0", [("counts", math.log(N0)), ("pval", math.log(SIGMA0))])
def test_identity_at_init(stats, space, disp0):
    f = form_b2.build(space, stats)
    kx = stats["knots_x"]
    outside = torch.cat([torch.linspace(float(kx[0]) - 5, float(kx[0]), 200),
                         torch.linspace(float(kx[-1]), float(kx[-1]) + 5, 200)])
    for x, tol in ((_grid(kx, 0.0, 0.0), 1e-5), (torch.as_tensor(kx, dtype=torch.float32), 1e-5),
                   (outside, 1e-4)):
        xb = x.unsqueeze(0)
        th = f.init_theta().expand(1, xb.shape[1], 24)
        loc, disp = f(xb, th)
        assert (loc[0] - x).abs().max().item() < tol
        assert (disp[0] - disp0).abs().max().item() < 1e-6


# ---- per-bin: each bin uses its own curve; non-monotone maps are representable ------------------

def test_each_bin_uses_its_own_curve(stats):
    """Bin j's output equals rung B run with bin j's theta alone."""
    fb, fb2 = form_b.build("counts", stats), form_b2.build("counts", stats)
    g = torch.Generator().manual_seed(7)
    B, L = 3, 40
    x = torch.rand(B, L, generator=g) * 8 - 1
    theta = fb.init_theta() + torch.randn(B, L, 24, generator=g)
    loc, disp = fb2(x, theta)
    lb, db = fb(x.reshape(-1, 1), theta.reshape(-1, 24))
    assert torch.equal(loc, lb.reshape(B, L)) and torch.equal(disp, db.reshape(B, L))


def test_non_monotone_map_representable(stats):
    """A decreasing map x -> loc = c - x: bin i picks the curve y = c - 2 x_i + x (slope 1 per bin,
    so every per-bin curve is monotone), whose value at x_i is c - x_i."""
    f = form_b2.build("counts", stats)
    kx = torch.as_tensor(stats["knots_x"], dtype=torch.float32)
    x = torch.linspace(float(kx[0]) + 0.01, float(kx[-1]) - 0.01, 300).unsqueeze(0)
    c = 4.0
    theta = f.init_theta().expand(1, x.shape[1], 24).clone()
    theta[..., 0] = theta[..., 0] + (c - 2.0 * x)          # shift each bin's identity curve
    loc, _ = f(x, theta)
    torch.testing.assert_close(loc, c - x, rtol=0, atol=1e-4)
    assert (torch.diff(loc, dim=1) < 0).all()
    # a bump: up then down across bins
    target = torch.sin(x)
    theta[..., 0] = f.init_theta()[0] + (target - x)
    loc, _ = f(x, theta)
    torch.testing.assert_close(loc, target, rtol=0, atol=1e-4)
    d = torch.diff(loc[0])
    assert (d > 0).any() and (d < 0).any()


def test_dispersion_per_bin(stats):
    f = form_b2.build("pval", stats)
    kx = stats["knots_x"]
    x = torch.as_tensor(np.concatenate([kx, [kx[0] - 2, kx[-1] + 2]]), dtype=torch.float32)
    x = x.unsqueeze(0)
    L = x.shape[1]
    theta = f.init_theta().expand(1, L, 24).clone()
    dvals = torch.arange(L, dtype=torch.float32) * 0.1 - 0.5
    theta[0, :, 12:] = dvals.unsqueeze(1)                    # bin j: flat disp = dvals[j]
    _, disp = f(x, theta)
    torch.testing.assert_close(disp[0], dvals, rtol=0, atol=1e-6)


def test_gradient_reaches_every_theta_entry(stats):
    f = form_b2.build("counts", stats)
    x = _grid(stats["knots_x"], 1.0, 1.0, n=2000).unsqueeze(0)
    theta = (f.init_theta() + 0.1).expand(1, x.shape[1], 24).clone().requires_grad_(True)
    loc, disp = f(x, theta)
    (loc.square().sum() + disp.square().sum()).backward()
    g = theta.grad[0]
    # each bin reaches y0, the steps up to its segment's upper knot, and its two disp knots
    assert (g[:, 0].abs() > 0).all()
    assert (g.abs().sum(dim=0) > 0).all()


# ---- describe -----------------------------------------------------------------------------------

def test_describe(stats):
    fb, fb2 = form_b.build("counts", stats), form_b2.build("counts", stats)
    g = torch.Generator().manual_seed(9)
    K = 5
    theta = fb.init_theta() + torch.randn(K, 24, generator=g)
    d = fb2.describe(theta)
    assert set(d) == {"n_levels", "knots_x", "curve_y", "disp"}
    assert d["n_levels"] == K and isinstance(d["n_levels"], int)
    assert len(d["knots_x"]) == 12 and all(isinstance(v, float) for v in d["knots_x"])
    assert len(d["curve_y"]) == K and len(d["disp"]) == K
    for k in range(K):
        db = fb.describe(theta[k])
        assert d["curve_y"][k] == db["curve_y"] and d["disp"][k] == db["disp"]
        assert all(isinstance(v, float) for v in d["curve_y"][k] + d["disp"][k])
    assert d["knots_x"] == fb.describe(theta[0])["knots_x"]
    di = fb2.describe(fb2.init_theta().unsqueeze(0).repeat(12, 1))
    assert di["n_levels"] == 12
    np.testing.assert_allclose(np.array(di["curve_y"]), np.tile(stats["knots_x"], (12, 1)),
                               atol=1e-5)
    np.testing.assert_allclose(np.array(di["disp"]), math.log(N0), atol=1e-6)
