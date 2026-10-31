"""t118 ladder rung B — `tools/t118/ladder/fforms/form_b.py`: per-bin monotone curve.

The training harness is not imported: every fit here optimises a free theta tensor against an NLL
written in this file.
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
from ladder.fforms import form_b  # noqa: E402

N0, SIGMA0 = 5.0, 0.5


def _stats(seed: int = 0) -> dict:
    """knots from a heavy-tailed count sample, the way C1 builds them (log1p space)."""
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
    f = base.load_form("B", space, stats)
    assert isinstance(f, form_b.FormB) and isinstance(f, base.FForm)
    assert (f.rung, f.space, f.n_theta, f.context) == ("B", space, 24, 0)
    assert f.init_theta().shape == (24,) and f.init_theta().dtype == torch.float32
    assert list(f.state_dict()) == []  # knots come from stats, not the checkpoint


@pytest.mark.parametrize("space", ["counts", "pval"])
def test_shapes(stats, space):
    f = form_b.build(space, stats)
    B, L = 5, 300
    x = torch.randn(B, L) * 2 + 1
    theta = f.init_theta().expand(B, -1) + 0.3 * torch.randn(B, 24)
    loc, disp = f(x, theta)
    assert loc.shape == (B, L) and disp.shape == (B, L)
    assert torch.isfinite(loc).all() and torch.isfinite(disp).all()
    d = f.describe(theta[0])
    assert set(d) == {"knots_x", "curve_y", "disp"}
    assert all(len(d[k]) == 12 and all(isinstance(v, float) for v in d[k]) for k in d)


def test_bad_knots_rejected(stats):
    with pytest.raises(ValueError):
        form_b.build("counts", {**stats, "knots_x": stats["knots_x"][:11]})
    kx = stats["knots_x"].copy()
    kx[5] = kx[4]
    with pytest.raises(ValueError):
        form_b.build("counts", {**stats, "knots_x": kx})


# ---- identity at init ---------------------------------------------------------------------------

@pytest.mark.parametrize("space,disp0", [("counts", math.log(N0)), ("pval", math.log(SIGMA0))])
def test_identity_at_init(stats, space, disp0):
    f = form_b.build(space, stats)
    kx = stats["knots_x"]
    th = f.init_theta().unsqueeze(0)
    # at the knots, between them, and beyond both ends (first/last slope = 1 at init)
    inside = _grid(kx, 0.0, 0.0)
    knots = torch.as_tensor(kx, dtype=torch.float32)
    outside = torch.cat([torch.linspace(float(kx[0]) - 5, float(kx[0]), 200),
                         torch.linspace(float(kx[-1]), float(kx[-1]) + 5, 200)])
    for x, tol in ((inside, 1e-5), (knots, 1e-5), (outside, 1e-4)):
        loc, disp = f(x.unsqueeze(0), th)
        assert (loc[0] - x).abs().max().item() < tol
        assert (disp[0] - disp0).abs().max().item() < 1e-6
    d = f.describe(f.init_theta())
    np.testing.assert_allclose(d["curve_y"], kx, atol=1e-5)
    np.testing.assert_allclose(d["disp"], disp0, atol=1e-6)


def test_identity_on_pval_knots():
    """pval knots span negative x (log(max(p, 1e-3)) >= -6.9); identity still holds."""
    rng = np.random.default_rng(3)
    p = np.maximum(rng.exponential(0.8, 100_000), 1e-3)
    st = {"knots_x": base.knot_stats(np.log(p)), "n0": N0, "sigma0": SIGMA0}
    f = form_b.build("pval", st)
    x = _grid(st["knots_x"], 1.0, 1.0).unsqueeze(0)
    loc, _ = f(x, f.init_theta().unsqueeze(0))
    assert (loc - x).abs().max().item() < 1e-5


# ---- monotonicity -------------------------------------------------------------------------------

def test_strictly_monotone_for_random_theta(stats):
    f = form_b.build("counts", stats)
    g = torch.Generator().manual_seed(0)
    theta = f.init_theta() + 2.0 * torch.randn(100, 24, generator=g)
    x = _grid(stats["knots_x"], n=801).expand(100, -1)
    loc, _ = f(x, theta)
    assert torch.isfinite(loc).all()
    assert (torch.diff(loc, dim=1) > 0).all()


@pytest.mark.parametrize("scale", [30.0, 1e3])
def test_monotone_for_extreme_theta(stats, scale):
    """Large |s|: softplus of a very negative step underflows below float32 resolution, so the
    curve can go flat there — never down."""
    f = form_b.build("counts", stats)
    g = torch.Generator().manual_seed(1)
    theta = scale * torch.randn(100, 24, generator=g)
    theta[:, 1:12] = scale * torch.sign(torch.randn(100, 11, generator=g))  # all steps at ±scale
    x = _grid(stats["knots_x"], n=801).expand(100, -1)
    loc, disp = f(x, theta)
    assert torch.isfinite(loc).all() and torch.isfinite(disp).all()
    assert (torch.diff(loc, dim=1) >= 0).all()
    # a curve whose steps are all large and positive stays strictly increasing
    th = torch.zeros(1, 24)
    th[0, 1:12] = scale
    loc, _ = f(x[:1], th)
    assert (torch.diff(loc, dim=1) > 0).all()


def test_extrapolation_uses_end_slopes(stats):
    f = form_b.build("counts", stats)
    kx = stats["knots_x"]
    g = torch.Generator().manual_seed(2)
    th = f.init_theta().unsqueeze(0) + torch.randn(1, 24, generator=g)
    y = torch.as_tensor(f.describe(th[0])["curve_y"], dtype=torch.float64)
    k = torch.as_tensor(kx)
    lo_slope = (y[1] - y[0]) / (k[1] - k[0])
    hi_slope = (y[-1] - y[-2]) / (k[-1] - k[-2])
    x = torch.tensor([[kx[0] - 3.0, kx[0] - 0.5, kx[-1] + 0.5, kx[-1] + 4.0]], dtype=torch.float32)
    loc, _ = f(x, th)
    want = [y[0] - 3.0 * lo_slope, y[0] - 0.5 * lo_slope, y[-1] + 0.5 * hi_slope,
            y[-1] + 4.0 * hi_slope]
    np.testing.assert_allclose(loc[0].numpy(), torch.stack(want).numpy(), rtol=1e-5, atol=1e-4)


# ---- dispersion by level ------------------------------------------------------------------------

def test_dispersion_interpolated_by_level(stats):
    f = form_b.build("pval", stats)
    kx = stats["knots_x"]
    th = f.init_theta().clone().unsqueeze(0)
    dk = torch.tensor([0.1 * k * k - 1.0 for k in range(12)])
    th[0, 12:] = dk
    mids = 0.5 * (kx[:-1] + kx[1:])
    q = kx[:-1] + 0.25 * np.diff(kx)
    x = torch.as_tensor(np.concatenate([kx, mids, q, [kx[0] - 2, kx[-1] + 2]]),
                        dtype=torch.float32).unsqueeze(0)
    _, disp = f(x, th)
    disp = disp[0]
    np.testing.assert_allclose(disp[:12].numpy(), dk.numpy(), atol=1e-6)
    np.testing.assert_allclose(disp[12:23].numpy(), (0.5 * (dk[:-1] + dk[1:])).numpy(), atol=1e-5)
    np.testing.assert_allclose(disp[23:34].numpy(), (0.75 * dk[:-1] + 0.25 * dk[1:]).numpy(),
                               atol=1e-5)
    assert disp[34].item() == pytest.approx(dk[0].item(), abs=1e-6)   # clamped below
    assert disp[35].item() == pytest.approx(dk[-1].item(), abs=1e-6)  # clamped above
    # the dispersion never touches loc
    loc_a, _ = f(x, th)
    loc_b, _ = f(x, f.init_theta().unsqueeze(0))
    torch.testing.assert_close(loc_a, loc_b)


def test_gradient_reaches_every_theta_entry(stats):
    f = form_b.build("counts", stats)
    theta = (f.init_theta() + 0.1).unsqueeze(0).repeat(2, 1).requires_grad_(True)
    x = _grid(stats["knots_x"], 1.0, 1.0, n=2000).unsqueeze(0).repeat(2, 1)
    loc, disp = f(x, theta)
    (loc.square().sum() + disp.square().sum()).backward()
    assert theta.grad is not None and (theta.grad.abs() > 0).all()


def test_batch_rows_are_independent(stats):
    f = form_b.build("counts", stats)
    g = torch.Generator().manual_seed(4)
    theta = f.init_theta() + torch.randn(3, 24, generator=g)
    x = torch.rand(3, 50, generator=g) * 6 - 1
    loc, disp = f(x, theta)
    for i in range(3):
        li, di = f(x[i:i + 1], theta[i:i + 1])
        torch.testing.assert_close(loc[i:i + 1], li)
        torch.testing.assert_close(disp[i:i + 1], di)


# ---- recovery: a monotone nonlinear map beats a single affine map -------------------------------

def _nb_nll(loc, disp, y):
    mu = loc.exp().clamp(1e-4, 1e6)
    n = disp.exp().clamp(1e-3, 1e4)
    return -(torch.lgamma(y + n) - torch.lgamma(n) - torch.lgamma(y + 1)
             + n * (n.log() - (n + mu).log()) + y * (mu.log() - (n + mu).log())).mean()


def _fit(forward, theta0, x, y, steps=600, lr=0.05):
    theta = theta0.clone().unsqueeze(0).requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = _nb_nll(*forward(x, theta), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return theta.detach()[0], _nb_nll(*forward(x, theta), y).item()


def _affine(x, th):  # rung A's map, restated locally (form_a is another chunk)
    return th[:, :1] + th[:, 1:2] * x, th[:, 2:3].expand_as(x)


def test_recovers_monotone_nonlinear_map_better_than_affine():
    torch.manual_seed(0)
    rng = np.random.default_rng(11)
    X = rng.poisson(rng.gamma(0.6, 8.0, size=200_000))
    Xt = rng.binomial(X, 0.5)                               # thinned source
    mu = 0.3 + 25.0 * Xt ** 2 / (Xt ** 2 + 16.0)           # monotone Hill curve: S-shaped in x
    Y = rng.negative_binomial(20.0, 20.0 / (20.0 + mu))
    x = torch.as_tensor(np.log1p(X), dtype=torch.float32).unsqueeze(0)
    y = torch.as_tensor(Y, dtype=torch.float32).unsqueeze(0)
    st = {"knots_x": base.knot_stats(np.log1p(X)), "n0": N0, "sigma0": SIGMA0}
    f = form_b.build("counts", st)

    _, nll_ident = _fit(f, f.init_theta(), x, y, steps=0)
    th_b, nll_b = _fit(f, f.init_theta(), x, y)
    _, nll_a = _fit(_affine, torch.tensor([0.0, 1.0, math.log(N0)]), x, y)
    assert nll_b < nll_a - 0.02 < nll_ident, (nll_b, nll_a, nll_ident)

    # the fitted curve tracks the empirical conditional mean where the data are dense
    loc, _ = f(x, th_b.unsqueeze(0))
    fit_mu = loc[0].exp().numpy()
    for v in range(0, 8):
        sel = X == v
        assert sel.sum() > 2000
        emp = Y[sel].mean()
        assert fit_mu[sel][0] == pytest.approx(emp, rel=0.1, abs=0.05), (v, fit_mu[sel][0], emp)
    # and it stays monotone after fitting
    xs = _grid(st["knots_x"], n=500).unsqueeze(0)
    assert (torch.diff(f(xs, th_b.unsqueeze(0))[0], dim=1) > 0).all()
