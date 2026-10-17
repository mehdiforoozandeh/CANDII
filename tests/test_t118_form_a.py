"""t118 rung A — `tools/t118/ladder/fforms/form_a.py`: per-bin affine f against the pinned FForm contract.

The planted-effect tests fit a free theta directly by NB / log-normal NLL written here (the C1-style
losses, restated), so they do not depend on the training harness.
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
from ladder.fforms import form_a  # noqa: E402

STATS = {"knots_x": np.linspace(0.0, 5.0, 12), "n0": 5.0, "sigma0": 0.5}
TOL = 0.15


# ---------------------------------------------------------------------------------------------
# the C1-style losses, restated locally
def nb_nll(loc, disp, y):
    """NB NLL, mean mu = exp(loc) clamped to [1e-4, 1e6], size n = exp(disp) clamped to [1e-3, 1e4]."""
    mu = torch.exp(loc).clamp(1e-4, 1e6)
    n = torch.exp(disp).clamp(1e-3, 1e4)
    ll = (torch.lgamma(y + n) - torch.lgamma(n) - torch.lgamma(y + 1.0)
          + n * (torch.log(n) - torch.log(n + mu)) + y * (torch.log(mu) - torch.log(n + mu)))
    return -ll.mean()


def lognormal_nll(loc, disp, y):
    """Log-normal NLL on t = log(max(y, 1e-3)): mu = loc, sigma = exp(disp)."""
    t = torch.log(y.clamp_min(1e-3))
    return (disp + 0.5 * ((t - loc) / torch.exp(disp)) ** 2 + t + 0.5 * math.log(2 * math.pi)).mean()


def fit_theta(form, x, y, nll, steps=300, lr=0.05):
    theta = form.init_theta().clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loc, disp = form(x, theta.unsqueeze(0).expand(x.shape[0], -1))
        loss = nll(loc, disp, y)
        loss.backward()
        opt.step()
    return form.describe(theta)


# ---------------------------------------------------------------------------------------------
# contract
@pytest.mark.parametrize("space", ["counts", "pval"])
def test_build_and_load_form(space):
    f = base.load_form("A", space, STATS)
    assert isinstance(f, form_a.FormA) and isinstance(f, base.FForm)
    assert (f.rung, f.space, f.n_theta, f.context) == ("A", space, 3, 0)
    assert isinstance(form_a.build(space, STATS), base.FForm)


@pytest.mark.parametrize("space", ["counts", "pval"])
def test_shapes(space):
    f = form_a.build(space, STATS)
    B, L = 4, 37
    x = torch.randn(B, L)
    theta = torch.randn(B, 3)
    loc, disp = f(x, theta)
    assert loc.shape == (B, L) and disp.shape == (B, L)
    # per-sample theta: each row uses its own (a, b, d)
    assert torch.allclose(loc, theta[:, :1] + theta[:, 1:2] * x)
    assert torch.equal(disp, theta[:, 2:3].expand(B, L))
    with pytest.raises(ValueError):
        f(x, torch.randn(B, 2))
    with pytest.raises(ValueError):
        f(x[0], theta)


@pytest.mark.parametrize("space,d0", [("counts", 5.0), ("pval", 0.5)])
def test_init_theta_is_identity(space, d0):
    f = form_a.build(space, STATS)
    th = f.init_theta()
    assert th.shape == (3,) and th.dtype == torch.float32
    assert torch.allclose(th, torch.tensor([0.0, 1.0, math.log(d0)]))
    x = torch.randn(3, 50) * 3
    loc, disp = f(x, th.unsqueeze(0).expand(3, -1))
    assert torch.equal(loc, x)
    assert torch.allclose(disp, torch.full_like(x, math.log(d0)))
    d = f.describe(th)
    assert set(d) == {"a", "b", "disp"} and all(type(v) is float for v in d.values())
    assert d["a"] == 0.0 and d["b"] == 1.0 and d["disp"] == pytest.approx(math.log(d0))


def test_gradients_reach_theta():
    f = form_a.build("counts", STATS)
    theta = f.init_theta().unsqueeze(0).repeat(2, 1).requires_grad_(True)
    loc, disp = f(torch.rand(2, 10), theta)
    (loc.sum() + disp.sum()).backward()
    assert theta.grad is not None and torch.all(theta.grad != 0)


# ---------------------------------------------------------------------------------------------
# planted effects
def test_planted_depth_ratio_counts():
    """Target = binomial thinning of the source at r = 0.5 over 200 k bins: a ≈ log r, b ≈ 1.

    The source is kept well above zero (gamma latent mean 200) because x = log1p(X), not log X: the
    best affine fit in log1p space has b > 1 and a < log r, by ~1/X (measured: mean 40 gives a - log r
    = -0.14, mean 200 gives -0.035; converged, 300 and 1500 steps agree). That bias is correct.
    """
    r = 0.5
    rng = np.random.default_rng(118)
    lam = rng.gamma(4.0, 200.0 / 4.0, 200_000)
    src = rng.poisson(lam)
    tgt = rng.binomial(src, r)
    x = torch.tensor(np.log1p(src), dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0)
    d = fit_theta(form_a.build("counts", STATS), x, y, nb_nll)
    assert d["b"] == pytest.approx(1.0, abs=TOL), d
    assert d["a"] == pytest.approx(math.log(r), abs=TOL), d
    # thinning given X is binomial, under-dispersed against Poisson: the NB size runs to large n
    assert d["disp"] > math.log(STATS["n0"]), d


def test_planted_affine_pval():
    """Target = 0.5 × source × lognormal noise (sigma 0.1): a ≈ log 0.5, b ≈ 1, sigma ≈ 0.1."""
    rng = np.random.default_rng(119)
    src = np.exp(rng.uniform(math.log(1e-2), math.log(50.0), 200_000))
    tgt = 0.5 * src * np.exp(0.1 * rng.standard_normal(src.size))
    x = torch.tensor(np.log(np.maximum(src, 1e-3)), dtype=torch.float32).unsqueeze(0)
    y = torch.tensor(tgt, dtype=torch.float32).unsqueeze(0)
    d = fit_theta(form_a.build("pval", STATS), x, y, lognormal_nll)
    assert d["b"] == pytest.approx(1.0, abs=TOL), d
    assert d["a"] == pytest.approx(math.log(0.5), abs=TOL), d
    assert math.exp(d["disp"]) == pytest.approx(0.1, abs=0.05), d
