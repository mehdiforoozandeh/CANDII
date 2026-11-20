"""E4 training-free baselines: known answers on hand-built interface-C batches."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import baselines as BL  # noqa: E402

A, K, K_META = 36, 18, 11


def batch(y, *, cells, starts, log2d_out, x=None, log2d_in=None, kind=1, chrom="chr1",
          pairs=None):
    """y, x: [B, L, A] int arrays; log2d_*: [B, A] log2 depths; kind: y_kind for available y."""
    y = np.asarray(y, np.int32)
    B, L, _ = y.shape
    x = np.zeros_like(y) if x is None else np.asarray(x, np.int32)
    c_out = np.zeros((B, A, K_META), np.float32)
    c_out[:, :, 0] = log2d_out
    c_in = np.zeros((B, A, K), np.float32)
    if log2d_in is not None:
        c_in[:, :, 0] = log2d_in
    y_avail = y.sum(1) >= 0
    meta = []
    for b in range(B):
        m = {"cell": cells[b], "chrom": chrom, "start_bin": starts[b]}
        if pairs is not None:
            m["measurements_in"], m["measurements_out"] = pairs[b]
        meta.append(m)
    return {
        "x_counts": torch.from_numpy(x), "x_avail": torch.from_numpy(x.sum(1) > 0),
        "x_bin_mask": torch.zeros(B, L, A, dtype=torch.bool),
        "x_dna": torch.zeros(B, L * 25, dtype=torch.uint8),
        "c_in": torch.from_numpy(c_in), "c_out": torch.from_numpy(c_out),
        "y_counts": torch.from_numpy(y), "y_avail": torch.from_numpy(y_avail),
        "y_kind": torch.from_numpy(np.where(y_avail, kind, 0).astype(np.uint8)),
        "meta": meta,
    }


# ---------------------------------------------------------------- marginal NB

def test_marginal_nb_is_constant_per_assay_and_tracks_assay_level():
    rng = np.random.default_rng(0)
    lam = np.linspace(1, 30, A)
    y = rng.poisson(lam, size=(6, 400, A))
    m = BL.MarginalNB().fit([batch(y, cells=list("abcdef"), starts=[0] * 6, log2d_out=np.zeros((6, A)))])
    out = m.predict(batch(y[:2], cells=["g", "h"], starts=[0, 0], log2d_out=np.zeros((2, A))))
    mu, n = out["mu"].numpy(), out["n"].numpy()
    assert mu.shape == (2, 400, A) and mu.dtype == np.float32
    assert np.all(mu == mu[:1, :1, :]) and np.all(n == n[:1, :1, :])
    # the CRPS-optimal constant sits near the Poisson mean (grid step 2^0.25)
    assert np.all(np.abs(np.log2(mu[0, 0] / lam)) < 0.5)
    assert np.all(n > 0)


# ---------------------------------------------------------------- position average

def test_position_average_is_leave_one_cell_out_and_depth_scaled():
    L = 8
    y = np.zeros((3, L, A), np.int64)
    y[0, :, 5], y[1, :, 5], y[2, :, 5] = 10, 20, 1000      # cell c is far off the others
    y[:, :, 6] = np.arange(L)[None, :] * np.array([1, 3, 5])[:, None]   # per-position profile
    tr = batch(y, cells=["a", "b", "c"], starts=[100, 100, 100], log2d_out=np.zeros((3, A)))
    pa = BL.PositionAverage().fit([tr])
    # target in cell c, depth 4x: the average uses a and b only -> (10 + 20)/2 * 4
    tgt = batch(np.zeros((1, L, A)), cells=["c"], starts=[100], log2d_out=np.full((1, A), 2.0))
    mu = pa.predict(tgt)["mu"].numpy()
    assert np.allclose(mu[0, :, 5], 15 * 4)
    assert np.allclose(mu[0, :, 6], np.maximum(np.arange(L) * 2 * 4, BL.PRED_FLOOR))
    # a never-seen cell averages all three
    new = batch(np.zeros((1, L, A)), cells=["z"], starts=[100], log2d_out=np.zeros((1, A)))
    assert np.allclose(pa.predict(new)["mu"].numpy()[0, :, 5], (10 + 20 + 1000) / 3)


def test_position_average_never_includes_target_cell():
    """Plant a huge value in the target cell only: its own LOO prediction must not see it."""
    rng = np.random.default_rng(1)
    L = 16
    y = rng.poisson(5, size=(4, L, A))
    y[3] = 10 ** 6
    cells = ["a", "b", "c", "d"]
    pa = BL.PositionAverage().fit([batch(y, cells=cells, starts=[0] * 4, log2d_out=np.zeros((4, A)))])
    for i, c in enumerate(cells[:3]):
        r = pa.loo_rate("chr1", 0, c, np.arange(L))
        others = [j for j in range(4) if j != i]
        assert np.allclose(r, y[others, :, 0].mean(0))
    r_d = pa.loo_rate("chr1", 0, "d", np.arange(L))
    assert r_d.max() < 100 and np.allclose(r_d, y[:3, :, 0].mean(0))


def test_position_average_uses_depth_in_rate_and_offset_windows():
    L = 4
    y = np.zeros((2, L, A), np.int64)
    y[0, :, 0], y[1, :, 0] = 8, 8
    # cell a at depth 2^3 covers bins 0..3; cell b at depth 2^1 covers bins 2..5
    lo = np.zeros((2, A)); lo[0, :], lo[1, :] = 3.0, 1.0
    pa = BL.PositionAverage().fit([batch(y, cells=["a", "b"], starts=[0, 2], log2d_out=lo)])
    r = pa.loo_rate("chr1", 0, "z", np.arange(6))
    assert np.allclose(r, [1, 1, 2.5, 2.5, 4, 4])     # rates a = 8/8, b = 8/2
    # bins no other cell covers fall back to the other cells' whole-track mean rate
    assert np.allclose(pa.loo_rate("chr1", 0, "a", np.arange(2)), 4.0)


# ---------------------------------------------------------------- depth-only scaling

def _thinning_pair(seed, p=0.5, L=20000):
    rng = np.random.default_rng(seed)
    x = np.zeros((1, L, A), np.int64)
    x[0, :, 3] = rng.negative_binomial(2, 2 / (2 + rng.gamma(1.0, 20, L)))
    y = np.zeros_like(x)
    y[0, :, 3] = rng.binomial(x[0, :, 3], p)
    li, lo = np.full((1, A), 20.0), np.full((1, A), 20.0 + np.log2(p))
    return x, y, li, lo


def test_depth_only_is_exact_on_binomial_thinning():
    x, y, li, lo = _thinning_pair(0)
    pairs = [(["s"] * A, ["t"] * A)]
    tr = batch(y, x=x, cells=["a"], starts=[0], log2d_out=lo, log2d_in=li, kind=2, pairs=pairs)
    d = BL.DepthOnlyScaling().fit([tr])
    out = d.predict(tr)
    mu = out["mu"].numpy()[0, :, 3]
    assert np.allclose(mu, np.maximum(0.5 * x[0, :, 3], BL.PRED_FLOOR), rtol=1e-6)
    # E[y | x] = p x: the residual mean is zero within 3 standard errors
    res = y[0, :, 3] - 0.5 * x[0, :, 3]
    assert abs(res.mean()) < 3 * res.std() / np.sqrt(res.size)
    # binomial thinning is under-dispersed vs Poisson, so the fitted size runs to the upper bound
    assert out["n"].numpy()[0, 0, 3] > 100
    # tracks that are not a correction pair are undefined, not zero
    assert not np.isnan(mu).any() and np.isnan(out["mu"].numpy()[0, :, 4]).all()


def test_depth_only_needs_a_fitted_pair_and_only_serves_kind_2():
    x, y, li, lo = _thinning_pair(1, L=500)
    tr = batch(y, x=x, cells=["a"], starts=[0], log2d_out=lo, log2d_in=li, kind=2,
               pairs=[(["s"] * A, ["t"] * A)])
    d = BL.DepthOnlyScaling().fit([tr])
    other = batch(y, x=x, cells=["a"], starts=[0], log2d_out=lo, log2d_in=li, kind=2,
                  pairs=[(["s"] * A, ["u"] * A)])
    with pytest.raises(KeyError):
        d.predict(other)
    imp = batch(y, x=x, cells=["a"], starts=[0], log2d_out=lo, log2d_in=li, kind=1,
                pairs=[(["s"] * A, ["t"] * A)])
    assert np.isnan(d.predict(imp)["mu"].numpy()).all()


def test_registry_names():
    assert set(BL.BASELINES) == {"marginal_nb", "position_average", "depth_only_scaling"}
