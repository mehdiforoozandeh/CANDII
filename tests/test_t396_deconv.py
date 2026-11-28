"""t396 C1: Richardson-Lucy deconvolution (tools/t396/deconv.py). Synthetic arrays only."""
import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
for _p in (_TOOLS / "t392", _TOOLS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from t396 import deconv as D  # noqa: E402

R = D.REACH


def _kernel2():
    """[2, 17] span kernel of a ~76 bp read: plus strand covers k = 0..3, minus k = -3..0."""
    k = np.zeros((2, D.KERNEL_WIDTH))
    k[0, R:R + 4] = [1.0, 1.0, 1.0, 0.04]
    k[1, R - 3:R + 1] = [0.04, 1.0, 1.0, 1.0]
    return k


W = D.mix_kernel(_kernel2(), [600, 400])


def _smooth(L=512, edge_zero=0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(L)
    s = 0.3 + sum(a * np.exp(-0.5 * ((x - m) / sd) ** 2)
                  for a, m, sd in zip(rng.uniform(1, 5, 6), rng.uniform(40, L - 40, 6),
                                      rng.uniform(12, 30, 6)))
    if edge_zero:
        s[:edge_zero] = 0.0
        s[-edge_zero:] = 0.0
    return s


def test_constants():
    assert D.N_ITER_PRIMARY == 50
    assert D.N_ITER_GRID == (10, 25, 50, 100, 200)
    from minicandi import contract as C
    assert (D.N_MIN, D.N_MAX) == (C.N_MIN, C.N_MAX)


def test_mix_kernel_matches_read_start():
    from minicandi.read_start import unstranded_kernel
    k, nr = _kernel2(), np.array([600.0, 400.0])
    np.testing.assert_allclose(D.mix_kernel(k, nr), unstranded_kernel(k, nr).numpy(), rtol=0, atol=1e-15)
    assert D.mix_kernel(k, nr).shape == (17,) and D.mix_kernel(k, nr).dtype == np.float64


def test_forward_equals_conv_interior():
    from conv_sparsity import conv
    s = np.random.default_rng(1).gamma(1.0, 2.0, 300)
    x = np.stack([s, np.zeros_like(s)], 1)
    w2 = np.zeros((2, 17))
    w2[0] = W
    ref = conv(x, w2)
    np.testing.assert_allclose(D.forward(s, W)[R:-R], ref[R:-R], rtol=1e-13)


def test_forward_edge_fill_and_batch():
    s = np.random.default_rng(2).gamma(1.0, 2.0, (3, 64))
    c = D.forward(s, W)
    pos = np.arange(64)
    want = sum(W[R + k] * s[:, np.clip(pos - k, 0, 63)] for k in range(-R, R + 1))
    np.testing.assert_allclose(c, want, rtol=1e-13)
    np.testing.assert_allclose(c[1], D.forward(s[1], W), rtol=0)


def test_adjoint_is_transpose():
    rng = np.random.default_rng(3)
    s, x = rng.normal(size=(2, 50)), rng.normal(size=(2, 50))
    lhs = (D.forward(s, W) * x).sum(-1)
    rhs = (s * D.adjoint(x, W)).sum(-1)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)
    # interior equals the Terms formula sum_k w_k x[j + k]
    pos = np.arange(50)[R:-R]
    want = sum(W[R + k] * x[:, pos + k] for k in range(-R, R + 1))
    np.testing.assert_allclose(D.adjoint(x, W)[:, R:-R], want, rtol=1e-12)


def test_rl_recovers_smooth_profile():
    s = _smooth()
    c = D.forward(s, W)
    rec = D.richardson_lucy(c, W, 200)
    err = np.linalg.norm(rec - s) / np.linalg.norm(s)
    assert err < 1e-3, err


def test_rl_nonnegative_and_mass():
    s = _smooth(edge_zero=2 * R + 1, seed=4)
    rng = np.random.default_rng(5)
    c = rng.poisson(D.forward(s, W)).astype(float)          # noisy, with zeros
    c[:R], c[-R:] = 0.0, 0.0
    for n_iter in D.N_ITER_GRID:
        rec = D.richardson_lucy(c, W, n_iter)
        assert rec.min() >= 0
        assert abs(D.forward(rec, W).sum() - c.sum()) <= 1e-6 * c.sum()
        assert abs(rec[R:-R].sum() * W.sum() - c[R:-R].sum()) <= 1e-6 * c.sum()


def test_rl_batch_and_s0():
    s = np.stack([_smooth(256, seed=i) for i in range(3)])
    c = D.forward(s, W)
    rec = D.richardson_lucy(c, W, 10)
    assert rec.shape == (3, 256)
    np.testing.assert_allclose(rec[2], D.richardson_lucy(c[2], W, 10), rtol=1e-12)
    np.testing.assert_allclose(D.richardson_lucy(c, W, 0), c / W.sum())
    np.testing.assert_allclose(D.richardson_lucy(c, W, 5, s0=rec), D.richardson_lucy(c, W, 15),
                               rtol=1e-10)
    with pytest.raises(ValueError):
        D.richardson_lucy(-c, W, 1)


def _convert(s, n_s):
    from conv_sparsity import convert_nb_np
    w2 = np.zeros((2, 17))
    w2[0] = W
    z = np.zeros_like(s)
    return convert_nb_np(np.stack([s, z], 1), np.stack([n_s, np.ones_like(s)], 1), w2, False)


@pytest.mark.parametrize("n0", [0.5, 5.0, 80.0])
def test_match_size_constant_roundtrip(n0):
    s = _smooth(300, seed=6)
    mu_c, n_c = _convert(s, np.full_like(s, n0))
    np.testing.assert_allclose(D.match_size(s, W, mu_c, n_c), n0, rtol=1e-12)


def test_match_size_not_constant_for_varying_size():
    s = _smooth(300, seed=7)
    n_s = np.where(np.arange(300) < 150, 1.0, 20.0)
    mu_c, n_c = _convert(s, n_s)
    out = D.match_size(s, W, mu_c, n_c)
    assert out.max() - out.min() > 1.0
    assert np.all((out >= D.N_MIN) & (out <= D.N_MAX))


def test_match_size_zero_mass_gives_nmax():
    s = np.zeros(40)
    assert np.all(D.match_size(s, W, s, np.ones(40)) == D.N_MAX)
