"""t118 rung C2 — `tools/t118/ladder/fforms/form_c2.py`: a 33-bin kernel per bin, then B's curve per
bin. The harness is not used: theta [B, W, 57] is built directly in each test.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import base  # noqa: E402
from ladder.fforms import form_c2  # noqa: E402

N0, SIGMA0 = 5.0, 0.5
CTX = 16


def _source_counts(rng, n: int) -> np.ndarray:
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
def space(request):
    return request.param


@pytest.fixture(scope="module")
def form(space, counts_x):
    return base.load_form("C2", space, _stats(counts_x))


@pytest.fixture(scope="module")
def form_c(space, counts_x):
    return base.load_form("C", space, _stats(counts_x))


def _batch(x: np.ndarray, b: int, length: int, rng) -> torch.Tensor:
    starts = rng.integers(0, x.size - length - 2 * CTX, size=b)
    return torch.as_tensor(np.stack([x[s:s + length + 2 * CTX] for s in starts]), dtype=torch.float32)


def _random_row_theta(form, b: int, rng) -> torch.Tensor:
    """[B, 57]: a random kernel and curve around init (one row per sample)."""
    return form.init_theta().expand(b, -1) + torch.as_tensor(rng.normal(0, 0.3, size=(b, 57)),
                                                             dtype=torch.float32)


# -- contract -----------------------------------------------------------------------------------

def test_load_form_and_attributes(form):
    assert isinstance(form, base.FForm)
    assert form.rung == "C2" and form.n_theta == 57 and form.context == 16
    assert form_c2.FormC2.per_bin is True and form.per_bin is True
    assert tuple(form.init_theta().shape) == (57,)


def test_shapes_and_errors(form, counts_x):
    rng = np.random.default_rng(1)
    x = _batch(counts_x, 4, 100, rng)
    theta = form.init_theta().expand(4, 132, -1) + 0.1 * torch.randn(4, 132, 57)
    loc, disp = form(x, theta)
    assert tuple(loc.shape) == (4, 100) and tuple(disp.shape) == (4, 100)
    assert torch.isfinite(loc).all() and torch.isfinite(disp).all()
    with pytest.raises(ValueError):
        form(x, theta[..., :56])
    with pytest.raises(ValueError):
        form(x, theta[:, :100])  # theta must cover the haloed window
    with pytest.raises(ValueError):
        form(x, theta[:, 0])  # row-1 shaped theta
    with pytest.raises(ValueError):
        form(x[:, :32], theta[:, :32])


def test_identity_at_init(form, counts_x):
    rng = np.random.default_rng(2)
    x = _batch(counts_x, 3, 500, rng)
    x[0, 40] = -3.0   # below the first knot
    x[1, 60] = 12.0   # above the last knot
    theta = form.init_theta().expand(3, 532, -1)
    loc, disp = form(x, theta)
    assert torch.max(torch.abs(loc - x[:, CTX:-CTX])).item() < 1e-4
    disp0 = math.log(N0 if form.space == "counts" else SIGMA0)
    assert torch.allclose(disp, torch.full_like(disp, disp0), atol=1e-6)


# -- equals form C when theta is constant over bins -----------------------------------------------

def test_constant_theta_equals_form_c(form, form_c, counts_x):
    rng = np.random.default_rng(3)
    x = _batch(counts_x, 4, 700, rng)
    row = _random_row_theta(form, 4, rng)
    loc_c, disp_c = form_c(x, row)
    loc, disp = form(x, row.unsqueeze(1).expand(-1, x.shape[1], -1))
    assert torch.allclose(loc, loc_c, atol=1e-4, rtol=0)
    assert torch.allclose(disp, disp_c, atol=1e-4, rtol=0)
    # float64: the two reductions agree to rounding (float32 slack above is rounding times curve slope)
    x64, row64 = x.double(), row.double()
    loc_c, disp_c = form_c(x64, row64)
    loc, disp = form(x64, row64.unsqueeze(1).expand(-1, x.shape[1], -1))
    assert torch.allclose(loc, loc_c, atol=1e-12, rtol=0)
    assert torch.allclose(disp, disp_c, atol=1e-12, rtol=0)


def test_halo_handling_equals_conv1d(form):
    """Constant kernel, identity curve: loc = conv1d(x, w) with no padding, the centre L bins."""
    rng = np.random.default_rng(4)
    x = torch.as_tensor(rng.normal(1.0, 0.4, size=(3, 300 + 2 * CTX)), dtype=torch.float64)
    w = torch.as_tensor(rng.normal(0, 0.3, size=33), dtype=torch.float64)
    w[16] += 1.0
    row = form.init_theta().double().clone()
    row[:33] = w
    theta = row.expand(3, x.shape[1], -1)
    loc, _ = form(x, theta)
    ref = F.conv1d(x.unsqueeze(1), w.view(1, 1, 33)).squeeze(1)  # cross-correlation, no padding
    assert tuple(ref.shape) == (3, 300)
    # the init curve is the identity everywhere (end segments extrapolate with slope 1)
    assert torch.allclose(loc, ref, atol=1e-6, rtol=0)  # float32 knot buffer


def test_halo_theta_is_ignored(form, counts_x):
    """Only theta[:, 16:W-16] is read: the halo positions' theta change nothing and get no gradient."""
    rng = np.random.default_rng(5)
    x = _batch(counts_x, 2, 200, rng)
    theta = (form.init_theta().expand(2, 232, -1)
             + 0.2 * torch.as_tensor(rng.normal(size=(2, 232, 57)), dtype=torch.float32))
    loc, disp = form(x, theta)
    theta2 = theta.clone()
    theta2[:, :CTX] = torch.randn(2, CTX, 57) * 5
    theta2[:, -CTX:] = torch.randn(2, CTX, 57) * 5
    loc2, disp2 = form(x, theta2)
    assert torch.equal(loc, loc2) and torch.equal(disp, disp2)
    tg = theta.clone().requires_grad_()
    lg, dg = form(x, tg)
    (lg.sum() + dg.sum()).backward()
    assert (tg.grad[:, :CTX] == 0).all() and (tg.grad[:, -CTX:] == 0).all()
    assert torch.isfinite(tg.grad).all() and (tg.grad[:, CTX:-CTX].abs().sum(-1) > 0).all()


# -- per-bin behaviour ----------------------------------------------------------------------------

def test_kernel_chosen_by_the_bin_level(form, counts_x):
    """One-hot at 16 where the centre bin's x > 1, a 33-bin box where x <= 1; identity curve."""
    rng = np.random.default_rng(6)
    x = _batch(counts_x, 3, 800, rng).double()
    width = x.shape[1]
    theta = form.init_theta().double().expand(3, width, -1).clone()
    low = x <= 1.0
    theta[..., :33] = torch.where(low.unsqueeze(-1), torch.full((33,), 1.0 / 33, dtype=torch.float64),
                                  theta[..., :33])
    loc, _ = form(x, theta)
    centre = x[:, CTX:-CTX]
    box = x.unfold(1, 33, 1).mean(-1)
    expect = torch.where(centre > 1.0, centre, box)
    assert low[:, CTX:-CTX].any() and (~low[:, CTX:-CTX]).any()  # both branches exercised
    assert torch.allclose(loc, expect, atol=1e-6, rtol=0)


def test_kernel_orientation(form):
    """w_i[k] weighs the bin k - 16 to the right of i: one-hot at 20 shifts x left by 4."""
    x = torch.as_tensor(np.random.default_rng(7).normal(1.0, 0.3, size=(1, 132)), dtype=torch.float32)
    theta = form.init_theta().expand(1, 132, -1).clone()
    theta[..., :33] = 0.0
    theta[..., 20] = 1.0
    loc, _ = form(x, theta)
    assert torch.allclose(loc[0], x[0, CTX + 4:CTX + 4 + 100], atol=1e-4)


def test_theta_at_one_bin_moves_only_that_bin(form, counts_x):
    """Changing the centre theta at bin j (kernel and curve) changes loc/disp at j and nowhere else."""
    rng = np.random.default_rng(8)
    x = _batch(counts_x, 1, 300, rng)
    theta = form.init_theta().expand(1, 332, -1).clone()
    loc0, disp0 = form(x, theta)
    j = 150
    theta[0, CTX + j] += torch.as_tensor(rng.normal(0, 0.5, size=57), dtype=torch.float32)
    loc, disp = form(x, theta)
    moved = torch.nonzero((torch.abs(loc - loc0) + torch.abs(disp - disp0))[0] > 0).flatten().tolist()
    assert moved == [j]


def test_per_bin_curve_matches_row1_form_at_each_bin(form, form_c, counts_x):
    """Random theta per bin: loc/disp at bin i equal form C run with bin i's theta as a row."""
    rng = np.random.default_rng(9)
    length = 40
    x = _batch(counts_x, 2, length, rng)
    theta = (form.init_theta().expand(2, length + 2 * CTX, -1)
             + 0.3 * torch.as_tensor(rng.normal(size=(2, length + 2 * CTX, 57)), dtype=torch.float32))
    loc, disp = form(x, theta)
    for b in range(2):
        rows = theta[b, CTX:-CTX]  # [L, 57]
        lc, dc = form_c(x[b:b + 1].expand(length, -1), rows)  # [L, L]: row i uses bin i's theta
        assert torch.allclose(loc[b], lc.diagonal(), atol=1e-5)
        assert torch.allclose(disp[b], dc.diagonal(), atol=1e-5)


def test_no_cross_talk_between_samples(form, counts_x):
    rng = np.random.default_rng(10)
    x = _batch(counts_x, 3, 200, rng)
    theta = (form.init_theta().expand(3, 232, -1)
             + 0.2 * torch.as_tensor(rng.normal(size=(3, 232, 57)), dtype=torch.float32))
    loc, disp = form(x, theta)
    for i in range(3):
        li, di = form(x[i:i + 1], theta[i:i + 1])
        assert torch.allclose(li[0], loc[i], atol=1e-6) and torch.allclose(di[0], disp[i], atol=1e-6)


# -- describe -------------------------------------------------------------------------------------

def test_describe(form):
    k = 5
    rng = np.random.default_rng(11)
    theta = form.init_theta().expand(k, -1).clone()
    theta[1:] += torch.as_tensor(rng.normal(0, 0.3, size=(k - 1, 57)), dtype=torch.float32)
    desc = form.describe(theta)
    assert set(desc) == {"n_levels", "kernel", "knots_x", "curve_y", "disp"}
    assert desc["n_levels"] == k
    assert len(desc["kernel"]) == k and all(len(r) == 33 for r in desc["kernel"])
    assert len(desc["knots_x"]) == 12
    assert len(desc["curve_y"]) == k and all(len(r) == 12 for r in desc["curve_y"])
    assert len(desc["disp"]) == k and all(len(r) == 12 for r in desc["disp"])
    assert desc["kernel"][0] == [1.0 if i == 16 else 0.0 for i in range(33)]
    assert np.allclose(desc["curve_y"][0], desc["knots_x"], atol=1e-5)
    assert all(np.all(np.diff(r) >= 0) for r in desc["curve_y"])  # each level's curve is monotone
    assert np.allclose(desc["kernel"][3], theta[3, :33].double().numpy())
    disp0 = math.log(N0 if form.space == "counts" else SIGMA0)
    assert np.allclose(desc["disp"][0], disp0)
    assert all(isinstance(v, float) for v in desc["kernel"][2] + desc["curve_y"][2] + desc["disp"][2])
    with pytest.raises(ValueError):
        form.describe(theta[:, :56])
