"""t396 C3: covariance-aware start-to-count conversion (tools/t396/covconv.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
for p in (TOOLS, TOOLS / "t392"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from t396 import covconv as cc  # noqa: E402


def _kernel():
    """Reads cover about 4 bins: plus strand downstream, minus strand upstream; w <= 1."""
    w = np.zeros((2, cc.KW))
    w[0, cc.REACH + np.arange(0, 4)] = [1.0, 0.9, 0.6, 0.2]
    w[1, cc.REACH - np.arange(0, 4)] = [1.0, 0.8, 0.5, 0.3]
    return w


def test_constants_match_sources():
    cs = pytest.importorskip("conv_sparsity")
    assert (cc.N_MIN, cc.N_MAX, cc.REACH) == (cs.N_MIN, cs.N_MAX, cs.REACH)


@pytest.mark.parametrize("stranded", [True, False])
def test_identity_rho_equals_convert_nb_np(stranded):
    cs = pytest.importorskip("conv_sparsity")
    rng = np.random.default_rng(0)
    L = 60
    mu_sub = rng.gamma(1.0, 2.0, (L, 2))
    n_sub = rng.gamma(1.0, 1.0, (L, 2)) + 1e-4       # some below N_MIN, to hit the floor
    w = rng.random((2, cc.KW))
    m0, n0 = cs.convert_nb_np(mu_sub, n_sub, w, stranded)
    m1, n1 = cc.convert_nb_cov(mu_sub, n_sub, w, stranded, cc.identity_rho())
    np.testing.assert_allclose(m1, m0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(n1, n0, rtol=1e-12, atol=0)


def test_all_ones_rho_constant_size():
    rng = np.random.default_rng(1)
    L = 50
    mu_sub = rng.gamma(2.0, 1.5, (L, 2))
    n_sub = np.full((L, 2), 3.7)
    _, n = cc.convert_nb_cov(mu_sub, n_sub, rng.random((2, cc.KW)), True, np.ones((2, 2, cc.NLAG)))
    np.testing.assert_allclose(n, 3.7, rtol=1e-12)


def test_poisson_gives_n_max_and_shape_errors():
    mu = np.ones((10, 2))
    _, n = cc.convert_nb_cov(mu, np.full((10, 2), 1e9), _kernel(), True, cc.identity_rho())
    assert np.all(n == cc.N_MAX)
    with pytest.raises(ValueError):
        cc.convert_nb_cov(mu, mu, _kernel(), True, np.zeros((2, 2, 5)))


def test_finish_rho_and_additivity():
    rng = np.random.default_rng(2)
    y = rng.poisson(3.0, (4, 40, 2)).astype(float)
    mu, n = np.full_like(y, 3.0), np.full_like(y, 2.0)
    num, den = cc.estimate_rho(y, mu, n)
    parts = [cc.estimate_rho(y[r], mu[r], n[r]) for r in range(4)]
    np.testing.assert_allclose(num, sum(p[0] for p in parts))
    np.testing.assert_allclose(den, sum(p[1] for p in parts))
    rho = cc.finish_rho(num, den)
    assert rho[0, 0, cc.MAXLAG] == rho[1, 1, cc.MAXLAG] == 1.0
    assert np.all(np.abs(rho) <= 1.0)
    # symmetry rho[s, s2, l] = rho[s2, s, -l]
    np.testing.assert_allclose(rho, np.transpose(rho, (1, 0, 2))[:, :, ::-1], atol=1e-12)


def _simulate(rng, R, L, mu0, a, block=5):
    """Starts X[r, i, s] ~ Poisson(mu0[s] * G), G ~ Gamma(a, 1/a) shared by both strands over
    blocks of `block` bins with a random offset per region: marginal NB(mu0, a), excess parts
    correlated within a block. Also the overlap counts Y[r, i] ~ Poisson(sum_k,s w mu0 G) (thinned
    starts, one source bin per k), for the interior bins."""
    off = rng.integers(0, block, R)
    nblk = (L + block) // block + 1
    G = rng.gamma(a, 1.0 / a, (R, nblk))
    idx = (np.arange(L)[None, :] + off[:, None]) // block
    rate = G[np.arange(R)[:, None], idx][..., None] * mu0[None, None, :]     # [R, L, 2]
    X = rng.poisson(rate).astype(float)
    w = _kernel()
    lam = np.zeros((R, L))
    for k in range(-cc.REACH, cc.REACH + 1):
        src = np.clip(np.arange(L) - k, 0, L - 1)
        for s in range(2):
            lam += w[s, cc.REACH + k] * rate[:, src, s]
    Y = rng.poisson(lam).astype(float)
    return X, Y, w


def test_simulated_correlated_starts():
    rng = np.random.default_rng(3)
    R, L, a = 4000, 120, 2.0
    mu0 = np.array([2.0, 1.5])
    X, Y, w = _simulate(rng, R, L, mu0, a)
    mu_sub = np.broadcast_to(mu0, (L, 2)).copy()
    n_sub = np.full((L, 2), a)
    num, den = cc.estimate_rho(X, np.broadcast_to(mu_sub, X.shape), np.broadcast_to(n_sub, X.shape))
    rho = cc.finish_rho(num, den)
    # true lag-l correlation within a 5-bin block with random offset: (5 - |l|) / 5
    assert rho[0, 0, cc.MAXLAG + 1] > 0.5 and rho[1, 1, cc.MAXLAG + 1] > 0.5
    assert abs(rho[0, 1, cc.MAXLAG] - 1.0) < 0.1
    assert abs(rho[0, 0, cc.MAXLAG + 8]) < 0.1

    edge = cc.REACH + cc.MAXLAG
    inner = slice(edge, L - edge)
    emp = Y[:, inner].var(axis=0, ddof=1).mean()
    m_c, n_c = cc.convert_nb_cov(mu_sub, n_sub, w, True, rho)
    m_i, n_i = cc.convert_nb_cov(mu_sub, n_sub, w, True, cc.identity_rho())
    var_c = (m_c + m_c ** 2 / n_c)[inner].mean()
    var_i = (m_i + m_i ** 2 / n_i)[inner].mean()
    np.testing.assert_allclose(m_c[inner].mean(), Y[:, inner].mean(), rtol=0.02)
    assert abs(var_c / emp - 1.0) < 0.05, (var_c, emp)
    assert var_i / emp < 0.8, (var_i, emp)


def test_fit_scale_sets_calibration_ratio_to_one():
    dev = pytest.importorskip("onion_eval.deviance")
    rng = np.random.default_rng(4)
    recs = []
    for _ in range(3):
        mu = rng.gamma(2.0, 3.0, 3000)
        y = rng.negative_binomial(1.0, 1.0 / (1.0 + mu)).astype(float)   # true size 1
        recs.append({"y": y, "mu": mu, "n": np.full_like(mu, 5.0)})     # stated size too large
    c = cc.fit_scale(recs)
    assert c > 1.5
    terms = [dev.target_terms(r["y"], r["mu"], cc.scale_size(r["n"], c)) for r in recs]
    assert abs(dev.calibration_ratio(terms) - 1.0) < 1e-3
    np.testing.assert_allclose(cc.scale_size(np.array([4.0]), 2.0), [2.0])
