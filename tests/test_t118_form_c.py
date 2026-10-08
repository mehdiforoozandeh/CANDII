"""t118 rung C — `tools/t118/ladder/fforms/form_c.py`: a 33-bin kernel, then rung B's curve.

The harness is not used: every fit here trains theta directly with a local NLL.
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
from ladder.fforms import form_c  # noqa: E402

N0, SIGMA0 = 5.0, 0.5
CTX = 16


def _source_counts(rng, n: int) -> np.ndarray:
    """Background Poisson plus scattered peaks — the shape of a ChIP track, roughly."""
    lam = np.full(n, 2.0)
    for c in rng.integers(0, n, size=n // 200):
        lo, hi = max(0, c - 10), min(n, c + 10)
        lam[lo:hi] += rng.uniform(10, 60)
    return rng.poisson(lam).astype(np.float64)


def _stats(x: np.ndarray) -> dict:
    return {"knots_x": base.knot_stats(x), "n0": N0, "sigma0": SIGMA0}


@pytest.fixture(scope="module")
def counts_x():
    return np.log1p(_source_counts(np.random.default_rng(0), 30_000))


@pytest.fixture(scope="module", params=["counts", "pval"])
def form(request, counts_x):
    return base.load_form("C", request.param, _stats(counts_x))


def _batch(x: np.ndarray, b: int, length: int, rng) -> torch.Tensor:
    starts = rng.integers(0, x.size - length - 2 * CTX, size=b)
    return torch.as_tensor(np.stack([x[s:s + length + 2 * CTX] for s in starts]), dtype=torch.float32)


def _ref_curve(xc: np.ndarray, knots: np.ndarray, curve_y: np.ndarray, d: np.ndarray):
    """Local stand-in for rung B's curve (numpy): linear interpolation, end-slope extrapolation,
    dispersion clamped."""
    loc = np.interp(xc, knots, curve_y)
    lo_slope = (curve_y[1] - curve_y[0]) / (knots[1] - knots[0])
    hi_slope = (curve_y[-1] - curve_y[-2]) / (knots[-1] - knots[-2])
    loc = np.where(xc < knots[0], curve_y[0] + lo_slope * (xc - knots[0]), loc)
    loc = np.where(xc > knots[-1], curve_y[-1] + hi_slope * (xc - knots[-1]), loc)
    return loc, np.interp(xc, knots, d)


# -- contract -----------------------------------------------------------------------------------

def test_load_form_and_attributes(form):
    assert isinstance(form, base.FForm)
    assert form.rung == "C" and form.n_theta == 57 and form.context == 16
    assert form.space in ("counts", "pval")
    assert tuple(form.init_theta().shape) == (57,)


def test_shapes_with_halo(form, counts_x):
    rng = np.random.default_rng(1)
    x = _batch(counts_x, 4, 100, rng)
    assert tuple(x.shape) == (4, 132)
    theta = form.init_theta().expand(4, -1) + 0.1 * torch.randn(4, 57)
    loc, disp = form(x, theta)
    assert tuple(loc.shape) == (4, 100) and tuple(disp.shape) == (4, 100)
    assert torch.isfinite(loc).all() and torch.isfinite(disp).all()
    with pytest.raises(ValueError):
        form(x, theta[:, :56])
    with pytest.raises(ValueError):
        form(x[:, :32], theta)


def test_identity_at_init(form, counts_x):
    rng = np.random.default_rng(2)
    x = _batch(counts_x, 3, 500, rng)
    x[0, 40] = -3.0   # below the first knot: the first segment's slope continues (slope 1)
    x[1, 60] = 12.0   # above the last knot
    theta = form.init_theta().expand(3, -1)
    loc, disp = form(x, theta)
    assert torch.max(torch.abs(loc - x[:, CTX:-CTX])).item() < 1e-4
    disp0 = math.log(N0 if form.space == "counts" else SIGMA0)
    assert torch.allclose(disp, torch.full_like(disp, disp0), atol=1e-6)
    desc = form.describe(form.init_theta())
    assert set(desc) == {"kernel", "knots_x", "curve_y", "disp"}
    assert desc["kernel"] == [1.0 if i == 16 else 0.0 for i in range(33)]
    assert np.allclose(desc["curve_y"], desc["knots_x"], atol=1e-5)
    assert len(desc["disp"]) == 12


def test_one_hot_kernel_gives_the_curve(form, counts_x):
    """Kernel one-hot at 16 + a random curve part == rung B's curve on the centre of x."""
    rng = np.random.default_rng(3)
    x = _batch(counts_x, 2, 400, rng)
    theta = form.init_theta().expand(2, -1).clone()
    theta[:, 33:] += torch.as_tensor(rng.normal(0, 0.7, size=(2, 24)), dtype=torch.float32)
    loc, disp = form(x, theta)
    knots = form.stats["knots_x"]
    for i in range(2):
        desc = form.describe(theta[i])
        ref_loc, ref_disp = _ref_curve(x[i, CTX:-CTX].numpy().astype(np.float64), knots,
                                       np.array(desc["curve_y"]), np.array(desc["disp"]))
        assert np.allclose(loc[i].numpy(), ref_loc, atol=1e-4)
        assert np.allclose(disp[i].numpy(), ref_disp, atol=1e-5)
        assert np.all(np.diff(desc["curve_y"]) >= 0)


def test_receptive_field_is_33_bins(form):
    """A delta in x moves loc only within +-16 bins of it."""
    length = 200
    x = torch.zeros(1, length + 2 * CTX)
    theta = form.init_theta().unsqueeze(0).clone()
    theta[0, :33] = torch.linspace(0.2, 1.0, 33)  # nonzero everywhere, asymmetric
    base_loc, _ = form(x, theta)
    x2 = x.clone()
    x2[0, CTX + 100] = 1.0
    loc, _ = form(x2, theta)
    moved = torch.nonzero(torch.abs(loc - base_loc)[0] > 1e-7).flatten().tolist()
    assert moved == list(range(100 - 16, 100 + 17))


def test_kernel_orientation(form):
    """w[k] weighs the bin k - 16 to the right: a kernel one-hot at 20 shifts x left by 4."""
    x = torch.as_tensor(np.random.default_rng(4).normal(1.0, 0.3, size=(1, 132)), dtype=torch.float32)
    theta = form.init_theta().unsqueeze(0).clone()
    theta[0, :33] = 0.0
    theta[0, 20] = 1.0
    loc, _ = form(x, theta)
    assert torch.allclose(loc[0], x[0, CTX + 4:CTX + 4 + 100], atol=1e-4)


def test_per_sample_kernels(form, counts_x):
    """Different theta rows -> different kernels, and no cross-talk between samples."""
    rng = np.random.default_rng(5)
    x1 = _batch(counts_x, 1, 300, rng)
    x = x1.expand(3, -1).clone()
    theta = form.init_theta().expand(3, -1).clone()
    theta[1, :33] = 1.0 / 33                   # box smoother
    theta[2, 15:18] = torch.tensor([-0.5, 2.0, -0.5])  # sharpener (unconstrained kernel)
    loc, disp = form(x, theta)
    assert not torch.allclose(loc[0], loc[1]) and not torch.allclose(loc[1], loc[2])
    for i in range(3):
        li, di = form(x[i:i + 1], theta[i:i + 1])
        assert torch.allclose(li[0], loc[i], atol=1e-6) and torch.allclose(di[0], disp[i], atol=1e-6)
    var = loc.var(dim=1)
    assert var[1] < var[0] < var[2]  # smoothing lowers variance, sharpening raises it


def test_smoothing_kernel_smooths(form, counts_x):
    rng = np.random.default_rng(6)
    x = _batch(counts_x, 4, 2000, rng)
    theta = form.init_theta().expand(4, -1).clone()
    theta[:, :33] = torch.as_tensor(np.hanning(35)[1:-1] / np.hanning(35)[1:-1].sum(), dtype=torch.float32)
    loc, _ = form(x, theta)
    assert (loc.var(dim=1) < x[:, CTX:-CTX].var(dim=1)).all()


def test_gradient_reaches_every_theta_entry(form, counts_x):
    rng = np.random.default_rng(7)
    # a wide batch spanning every knot segment, including the top ones
    knots = form.stats["knots_x"]
    x = np.linspace(knots[0] - 1.0, knots[-1] + 1.0, 4000 + 2 * CTX)[None, :]
    x = torch.cat([torch.as_tensor(x, dtype=torch.float32), _batch(counts_x, 1, 4000, rng)])
    theta = (form.init_theta().expand(2, -1)
             + 0.01 * torch.as_tensor(rng.normal(size=(2, 57)), dtype=torch.float32)).requires_grad_()
    loc, disp = form(x, theta)
    (loc.sum() + disp.sum()).backward()
    assert torch.isfinite(theta.grad).all()
    assert (theta.grad[0].abs() > 0).all()


# -- recovery ------------------------------------------------------------------------------------

def _nb_nll(y, loc, disp):
    n = torch.exp(disp)
    log_mu = loc
    log_n_mu = torch.logaddexp(disp, log_mu)
    ll = (torch.lgamma(y + n) - torch.lgamma(n) - torch.lgamma(y + 1)
          + n * (disp - log_n_mu) + y * (log_mu - log_n_mu))
    return -ll.mean()


def _lognormal_nll(log_y, loc, disp):
    return (0.5 * ((log_y - loc) / torch.exp(disp)) ** 2 + disp).mean()


def _fit(form, x, target, nll, free_kernel: bool, steps=400, lr=0.02):
    theta = form.init_theta().unsqueeze(0).clone().requires_grad_()
    mask = torch.ones(1, 57)
    if not free_kernel:
        mask[0, :33] = 0.0
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loc, disp = form(x, theta)
        loss = nll(target, loc, disp)
        loss.backward()
        theta.grad *= mask
        opt.step()
    with torch.no_grad():
        loc, disp = form(x, theta)
        return float(nll(target, loc, disp)), theta.detach()[0]


@pytest.mark.parametrize("space", ["counts", "pval"])
def test_recovers_a_smoothing_kernel(space):
    """Target = source smoothed by a known 9-bin kernel: the fitted kernel beats the identity
    kernel (curve and dispersion fitted in both) and lands near the true kernel."""
    rng = np.random.default_rng(11)
    n = 40_000
    k_true = np.zeros(33)
    k_true[12:21] = np.hanning(11)[1:-1] / np.hanning(11)[1:-1].sum()
    counts = _source_counts(rng, n + 2 * CTX)
    if space == "counts":
        x = np.log1p(counts)
        mu = np.exp(np.correlate(x, k_true, mode="valid"))
        target = torch.as_tensor(rng.poisson(mu), dtype=torch.float32)[None, :]
        nll = _nb_nll
    else:
        x = np.log(np.maximum(counts / 10.0, 1e-3))  # a -log10 p-like track, logged
        log_y = np.correlate(x, k_true, mode="valid") + rng.normal(0, 0.1, size=n)
        target = torch.as_tensor(log_y, dtype=torch.float32)[None, :]
        nll = _lognormal_nll
    form = base.load_form("C", space, _stats(x[CTX:-CTX]))
    xt = torch.as_tensor(x, dtype=torch.float32)[None, :]

    nll_kernel, theta = _fit(form, xt, target, nll, free_kernel=True)
    nll_ident, _ = _fit(form, xt, target, nll, free_kernel=False)
    assert nll_kernel < nll_ident - 0.02, (nll_kernel, nll_ident)
    w = np.array(form.describe(theta)["kernel"])
    assert np.corrcoef(w, k_true)[0, 1] > 0.9
    assert abs(w[12:21].sum() - 1.0) < 0.25
