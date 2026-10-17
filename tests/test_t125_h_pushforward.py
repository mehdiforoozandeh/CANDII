"""t125-P2c — the exact push-forward of an NB count through h: `tools/t125/hmap/pushforward.py`.

Written against the plan (`.orchestrate/plan_t125.md` § t125-P2c) and design note Part 2:
the NB support to a 1e-6 tail, the exact mean and CRPS of the pushed set, and the closed-form
Gaussian-mixture CRPS for the own-noise variant. Every closed form is checked against a
brute-force or Monte Carlo second method.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nbinom, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))
sys.path.insert(0, str(ROOT / "tools" / "t125"))

from candi.bench.distributional import p_from_mu  # noqa: E402
from candi.metrics import nb_crps  # noqa: E402
from hmap import pushforward as pf  # noqa: E402


# ---------------------------------------------------------------------------------------------
# brute-force references
# ---------------------------------------------------------------------------------------------


def brute_crps(v, w, y):
    """sum_i w_i |v_i - y| - 1/2 sum_ij w_i w_j |v_i - v_j|, one row, explicit double sum."""
    v = np.asarray(v, np.float64)
    w = np.asarray(w, np.float64)
    w = w / w.sum()
    e1 = float(np.sum(w * np.abs(v - y)))
    e2 = float(np.sum(w[:, None] * w[None, :] * np.abs(v[:, None] - v[None, :])))
    return e1 - 0.5 * e2


def gauss_crps(mu, sigma, y):
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / math.sqrt(math.pi))


# ---------------------------------------------------------------------------------------------
# nb_support
# ---------------------------------------------------------------------------------------------


def test_constants():
    assert pf.TAIL == 1e-6
    assert pf.K_MAX == 4096


MUS = np.array([1e-4, 0.0, 0.3, 1.0, 4.0, 25.0, 120.0, 900.0, 3.0, 50.0])
NS = np.array([1e4, 2.0, 0.5, 1.0, 10.0, 3.0, 50.0, 200.0, 1e-3 * 300, 1e4])


def test_nb_support_shapes_and_renormalised():
    k, w, valid = pf.nb_support(MUS, NS)
    n = MUS.size
    assert k.dtype == np.int64 and w.dtype == np.float64 and valid.dtype == bool
    assert k.shape == w.shape == valid.shape and k.shape[0] == n
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.all(w[~valid] == 0.0)
    assert np.all(w >= 0.0)
    # left-aligned, contiguous support: valid is a prefix, k steps by one inside it
    for i in range(n):
        m = int(valid[i].sum())
        assert valid[i, :m].all() and not valid[i, m:].any()
        assert np.all(np.diff(k[i, :m]) == 1)
    # K = the longest kept support
    assert k.shape[1] == int(valid.sum(axis=1).max())


def test_nb_support_tail_mass_and_pmf():
    k, w, valid = pf.nb_support(MUS, NS)
    p = p_from_mu(NS, MUS)
    for i in range(MUS.size):
        kk = k[i, valid[i]]
        lo, hi = int(kk[0]), int(kk[-1])
        dropped = (nbinom.cdf(lo - 1, NS[i], p[i]) if lo > 0 else 0.0) + nbinom.sf(hi, NS[i], p[i])
        assert dropped <= pf.TAIL * (1 + 1e-9), (i, lo, hi, dropped)
        # the weights are the NB pmf over the kept support, renormalised
        ref = nbinom.pmf(kk, NS[i], p[i])
        np.testing.assert_allclose(w[i, valid[i]], ref / ref.sum(), rtol=1e-9, atol=1e-15)


def test_nb_support_one_sided_starts_at_zero_and_is_minimal():
    """Within the cap the support is 0..k_i, k_i = nbinom.ppf(1 - tail, n, p) (minimal)."""
    mu = np.array([0.5, 3.0, 20.0, 80.0])
    n_size = np.array([1.0, 5.0, 2.0, 30.0])
    k, w, valid = pf.nb_support(mu, n_size)
    p = p_from_mu(n_size, mu)
    k_i = nbinom.ppf(1 - pf.TAIL, n_size, p).astype(np.int64)
    for i in range(mu.size):
        kk = k[i, valid[i]]
        assert kk[0] == 0 and kk[-1] == k_i[i]
        assert nbinom.sf(k_i[i] - 1, n_size[i], p[i]) > pf.TAIL     # one fewer would not do
    np.testing.assert_array_equal(pf.support_size(mu, n_size), k_i + 1)


def test_nb_support_bounded_and_capped_rows_flagged():
    """Large mu: a two-sided window keeps the row under the cap; a row that still does not fit is
    capped at k_max + 1 points, and valid's last column is True exactly for the capped rows."""
    k_max = 256
    mu = np.array([2.0, 400.0, 400.0, 5.0])
    n_size = np.array([5.0, 1e4, 0.5, 0.01])     # row 1 fits by window; rows 2, 3 do not fit
    size = pf.support_size(mu, n_size, k_max=k_max)
    k, w, valid = pf.nb_support(mu, n_size, k_max=k_max)
    capped = size > k_max
    assert capped.tolist() == [False, False, True, True]
    assert k.shape[1] == k_max + 1
    np.testing.assert_array_equal(valid[:, -1], capped)
    assert np.all(valid.sum(axis=1)[~capped] <= k_max)
    # row 1: two-sided window, not starting at 0, total dropped mass <= tail
    p = p_from_mu(n_size, mu)
    kk = k[1, valid[1]]
    assert kk[0] > 0
    dropped = nbinom.cdf(kk[0] - 1, n_size[1], p[1]) + nbinom.sf(kk[-1], n_size[1], p[1])
    assert dropped <= pf.TAIL
    # capped rows still carry renormalised weights on a contiguous window
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-12)
    # the capped window keeps most of the mass (it is placed around the mode)
    for i in (2, 3):
        kk = k[i, valid[i]]
        kept = nbinom.cdf(kk[-1], n_size[i], p[i]) - (nbinom.cdf(kk[0] - 1, n_size[i], p[i])
                                                        if kk[0] > 0 else 0.0)
        assert kept > 0.5


def test_nb_support_rejects_bad_input():
    with pytest.raises(ValueError):
        pf.nb_support(np.array([1.0, -1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        pf.nb_support(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        pf.nb_support(np.array([np.nan]), np.array([1.0]))


# ---------------------------------------------------------------------------------------------
# mean_discrete / crps_discrete
# ---------------------------------------------------------------------------------------------


def test_crps_discrete_equals_brute_force_random_rows():
    rng = np.random.default_rng(0)
    n, K = 40, 9
    v = rng.normal(0, 3, (n, K))
    v[:, 3] = v[:, 1]                         # ties
    w = rng.random((n, K))
    w[:, -2:] *= rng.random((n, 1)) > 0.5     # some padded zero weights
    w /= w.sum(axis=1, keepdims=True)
    y = rng.normal(0, 3, n)
    got = pf.crps_discrete(v, w, y)
    ref = np.array([brute_crps(v[i], w[i], y[i]) for i in range(n)])
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(pf.mean_discrete(v, w), (v * w).sum(axis=1), rtol=1e-13)


def test_crps_discrete_point_mass():
    v = np.array([[2.5, 7.0, -1.0], [4.0, 0.0, 0.0]])
    w = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    y = np.array([3.0, 10.0])
    np.testing.assert_allclose(pf.crps_discrete(v, w, y), [4.0, 6.0], atol=1e-14)


def test_crps_discrete_padding_ignored_even_if_nonfinite():
    v = np.array([[1.0, 3.0, np.nan, np.inf]])
    w = np.array([[0.25, 0.75, 0.0, 0.0]])
    y = np.array([2.0])
    np.testing.assert_allclose(pf.crps_discrete(v, w, y), [brute_crps([1, 3], [0.25, 0.75], 2.0)])
    np.testing.assert_allclose(pf.mean_discrete(v, w), [2.5])


def _monotone_h(k):
    return 1.7 * np.log1p(k) + 0.2 * k


def _nonmonotone_h(k):
    return np.abs(np.sin(0.7 * k)) * 5.0 + 0.1 * k


@pytest.mark.parametrize("h", [_monotone_h, _nonmonotone_h], ids=["monotone", "nonmonotone"])
def test_crps_of_pushed_nb_equals_brute_force(h):
    mu = np.array([0.4, 3.0, 12.0, 60.0])
    n_size = np.array([2.0, 0.8, 5.0, 40.0])
    y = np.array([0.5, 2.0, 7.5, 9.0])
    k, w, valid = pf.nb_support(mu, n_size)
    v = h(k.astype(np.float64))
    got = pf.crps_discrete(v, w, y)
    for i in range(mu.size):
        assert got[i] == pytest.approx(brute_crps(v[i, valid[i]], w[i, valid[i]], y[i]),
                                       rel=1e-11, abs=1e-12)


def test_crps_discrete_identity_h_equals_nb_crps():
    """v = k (the identity h): the pushed-set CRPS is the NB CRPS (candi.metrics.nb_crps) up to the
    truncated tail; at tail 1e-12 they agree within 1e-6."""
    mu = np.array([0.05, 0.7, 2.0, 9.0, 33.0, 150.0])
    n_size = np.array([0.6, 3.0, 1.0, 20.0, 4.0, 900.0])
    y = np.array([0.0, 1.0, 5.0, 8.0, 0.0, 170.0])
    k, w, valid = pf.nb_support(mu, n_size, tail=1e-12)
    got = pf.crps_discrete(k.astype(np.float64), w, y)
    ref = nb_crps(n_size, p_from_mu(n_size, mu), y)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)
    np.testing.assert_allclose(pf.mean_discrete(k.astype(np.float64), w),
                               n_size * (1 - p_from_mu(n_size, mu)) / p_from_mu(n_size, mu),
                               rtol=1e-8)


# ---------------------------------------------------------------------------------------------
# Gaussian mixture (own noise, on log(1 + P))
# ---------------------------------------------------------------------------------------------


def test_gauss_mixture_single_component_equals_gaussian_crps():
    rng = np.random.default_rng(1)
    n = 50
    mu = rng.normal(1, 2, (n, 1))
    sigma = np.exp(rng.normal(0, 1, (n, 1)))
    w = np.ones((n, 1))
    y = rng.normal(1, 3, n)
    got = pf.crps_gauss_mixture(mu, sigma, w, y)
    np.testing.assert_allclose(got, gauss_crps(mu[:, 0], sigma[:, 0], y), rtol=1e-12, atol=1e-13)


def test_gauss_mixture_three_components_matches_monte_carlo():
    mu = np.array([[0.0, 1.5, 4.0]])
    sigma = np.array([[0.5, 1.0, 0.3]])
    w = np.array([[0.5, 0.3, 0.2]])
    y = np.array([1.2])
    got = float(pf.crps_gauss_mixture(mu, sigma, w, y)[0])
    rng = np.random.default_rng(7)
    m = 2_000_000
    c1 = rng.choice(3, size=m, p=w[0])
    c2 = rng.choice(3, size=m, p=w[0])
    x1 = rng.normal(mu[0, c1], sigma[0, c1])
    x2 = rng.normal(mu[0, c2], sigma[0, c2])
    terms = np.abs(x1 - y[0]) - 0.5 * np.abs(x1 - x2)
    est, se = terms.mean(), terms.std(ddof=1) / math.sqrt(m)
    assert abs(got - est) < 3 * se, (got, est, se)


def test_gauss_mixture_chunking_and_padding():
    """Rows in chunks of any size give the same answer; zero-weight padding is ignored."""
    rng = np.random.default_rng(2)
    n, K = 37, 6
    mu = rng.normal(0, 2, (n, K))
    sigma = np.exp(rng.normal(-0.5, 0.5, (n, K)))
    w = rng.random((n, K))
    w[::3, 4:] = 0.0
    w /= w.sum(axis=1, keepdims=True)
    y = rng.normal(0, 2, n)
    full = pf.crps_gauss_mixture(mu, sigma, w, y, chunk=4096)
    small = pf.crps_gauss_mixture(mu, sigma, w, y, chunk=5)
    np.testing.assert_allclose(small, full, rtol=1e-13, atol=1e-14)
    # padded rows equal the same rows with the padding cut off
    i = 0
    mu_p = mu[i].copy()
    mu_p[4:] = np.nan
    trim = pf.crps_gauss_mixture(mu[i:i + 1, :4], sigma[i:i + 1, :4], w[i:i + 1, :4], y[i:i + 1])
    pad = pf.crps_gauss_mixture(mu_p[None], sigma[i:i + 1], w[i:i + 1], y[i:i + 1])
    np.testing.assert_allclose(pad, trim, rtol=1e-13)


def test_gauss_mixture_brute_double_sum():
    """The pair sum written out term by term with the A(m, s^2) kernel."""
    rng = np.random.default_rng(3)
    K = 5
    mu = rng.normal(0, 1, (1, K))
    sigma = np.exp(rng.normal(0, 0.5, (1, K)))
    w = rng.random((1, K))
    w /= w.sum()
    y = np.array([0.3])

    def A(m, s2):
        s = math.sqrt(s2)
        return m * (2 * norm.cdf(m / s) - 1) + 2 * s * norm.pdf(m / s)

    e1 = sum(w[0, a] * A(y[0] - mu[0, a], sigma[0, a] ** 2) for a in range(K))
    e2 = sum(w[0, a] * w[0, b] * A(mu[0, a] - mu[0, b], sigma[0, a] ** 2 + sigma[0, b] ** 2)
             for a in range(K) for b in range(K))
    assert float(pf.crps_gauss_mixture(mu, sigma, w, y)[0]) == pytest.approx(e1 - 0.5 * e2,
                                                                              rel=1e-12)


def test_mean_gauss_mixture_logp():
    mu = np.array([[0.2, 1.0], [2.0, 0.0]])
    sigma = np.array([[0.5, 0.1], [0.3, 1.0]])
    w = np.array([[0.4, 0.6], [1.0, 0.0]])
    ref = (w * (np.exp(mu + sigma ** 2 / 2) - 1)).sum(axis=1)
    np.testing.assert_allclose(pf.mean_gauss_mixture_logp(mu, sigma, w), ref, rtol=1e-14)


def test_own_noise_chain_end_to_end_monte_carlo():
    """The own-noise chain: Y = mu_h(k) + sigma_h(k) Z on log(1 + P), k ~ NB. The mixture CRPS over
    the NB support matches sampling k from the NB and Z from N(0, 1)."""
    mu_nb, n_nb = np.array([3.0]), np.array([2.0])
    k, w, valid = pf.nb_support(mu_nb, n_nb)
    kf = k.astype(np.float64)
    mu_h = 0.8 * np.log1p(kf) + 0.1
    sig_h = 0.2 + 0.05 * kf
    y = np.array([1.1])
    got = float(pf.crps_gauss_mixture(mu_h, sig_h, w, y)[0])
    rng = np.random.default_rng(11)
    m = 2_000_000
    p = p_from_mu(n_nb, mu_nb)[0]
    k1 = rng.negative_binomial(n_nb[0], p, m).astype(np.float64)
    k2 = rng.negative_binomial(n_nb[0], p, m).astype(np.float64)
    x1 = 0.8 * np.log1p(k1) + 0.1 + (0.2 + 0.05 * k1) * rng.standard_normal(m)
    x2 = 0.8 * np.log1p(k2) + 0.1 + (0.2 + 0.05 * k2) * rng.standard_normal(m)
    terms = np.abs(x1 - y[0]) - 0.5 * np.abs(x1 - x2)
    est, se = terms.mean(), terms.std(ddof=1) / math.sqrt(m)
    assert abs(got - est) < 3 * se, (got, est, se)
