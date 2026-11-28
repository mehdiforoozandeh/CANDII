"""Tests of tools/t396/caps.py: closed-form cap laws, fits, floors and gaps closed (t396 C4)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
from t396 import caps  # noqa: E402

K = 25


def _mc_distinct(c, a, K, n, rng):
    """Monte Carlo of the Dirichlet-multinomial occupancy: n draws of c reads on K positions."""
    return caps._draw_distinct(np.full(n, c, np.int64), a, K, rng)


def _simulate_pairs(n_bins, a, K, rng, q=1.0):
    """Dedup-off counts from a heterogeneous genome; the dedup-on track is the distinct positions
    among them (after thinning to one mate with probability q)."""
    lam = rng.gamma(0.4, 40.0, n_bins)                  # broad bin intensities, many deep bins
    off = rng.poisson(lam)
    m = rng.binomial(off, q) if q != 1.0 else off
    on = caps._draw_distinct(m, a, K, rng)
    return off, on


# ------------------------------------------------------------------ the laws
@pytest.mark.parametrize("c,a", [(5, 0.3), (30, 0.3), (30, 2.0), (80, 1.0), (200, 0.1)])
def test_dm_distinct_matches_monte_carlo(c, a):
    rng = np.random.default_rng(1)
    sim = _mc_distinct(c, a, K, 40000, rng)
    pred = float(caps.dm_distinct(c, a, K))
    assert abs(sim.mean() - pred) / pred < 0.01
    assert abs(sim.mean() - pred) < 4 * sim.std() / np.sqrt(sim.size)


def test_dm_distinct_limits():
    assert caps.dm_distinct(0, 0.5, K) == pytest.approx(0.0, abs=1e-12)
    assert caps.dm_distinct(1, 0.5, K) == pytest.approx(1.0, rel=1e-10)
    # small c relative to K and flat weights: almost every read kept
    assert caps.dm_distinct(2, 50.0, 1000.0) == pytest.approx(2.0, rel=1e-3)
    # large c: every position occupied (P(empty) falls as c^-a, so use a moderate a)
    assert caps.dm_distinct(1e6, 2.0, K) == pytest.approx(K, rel=1e-4)
    # a = inf is occupancy, and a large finite a approaches it
    c = np.arange(0, 300)
    occ = K * (1 - (1 - 1 / K) ** c)
    np.testing.assert_allclose(caps.dm_distinct(c, np.inf, K), occ, rtol=1e-12)
    np.testing.assert_allclose(caps.occupancy(c, K), occ, rtol=1e-12)
    np.testing.assert_allclose(caps.dm_distinct(c, 1e3, K), occ, rtol=2e-3)
    # more heterogeneity (smaller a) keeps fewer reads
    assert caps.dm_distinct(50, 0.1, K) < caps.dm_distinct(50, 1.0, K) < caps.occupancy(50, K)


def test_K_of():
    assert caps.K_of(1) == 25.0 and caps.K_of(3) == 75.0


def test_closed_form_matches_simulated_dedup():
    """Positions drawn with known heterogeneity, deduplicated per position; the mean kept count
    at each observed dedup-off count matches dm_distinct within Monte Carlo error."""
    rng = np.random.default_rng(7)
    a = 0.5
    off, on = _simulate_pairs(200000, a, K, rng)
    checked = 0
    for c in (3, 10, 25, 50, 100):
        sel = on[off == c]
        assert sel.size > 100
        se = sel.std() / np.sqrt(sel.size)
        assert abs(sel.mean() - caps.dm_distinct(c, a, K)) < 4 * se + 1e-9
        checked += 1
    assert checked == 5


def test_thinned_dm_distinct_matches_monte_carlo():
    rng = np.random.default_rng(3)
    a, q, c, n = 0.7, 0.5, 60, 40000
    m = rng.binomial(np.full(n, c), q)
    sim = caps._draw_distinct(m, a, K, rng)
    pred = float(caps.thinned_dm_distinct(c, q, a, K))
    assert abs(sim.mean() - pred) < 4 * sim.std() / np.sqrt(n)
    np.testing.assert_allclose(caps.thinned_dm_distinct(np.array([0, 4, 9]), 1.0, a, K),
                               caps.dm_distinct(np.array([0, 4, 9]), a, K))
    with pytest.raises(ValueError):
        caps.thinned_dm_distinct(np.array([1.5]), 0.5, a, K)


def test_start_cap_gamma_poisson():
    rng = np.random.default_rng(5)
    a, lam, n = 0.8, 40.0, 40000
    g = rng.gamma(a, 1 / a, size=(n, K))
    kept = (rng.poisson(lam / K * g) > 0).sum(1)
    pred = float(caps.start_cap(lam, a, K))
    assert abs(kept.mean() - pred) < 4 * kept.std() / np.sqrt(n)
    assert caps.start_cap(lam, np.inf, K) == pytest.approx(K * (1 - np.exp(-lam / K)))


def test_count_saturation():
    Kc = 2.0 * (76 + 25) / 25
    assert caps.count_saturation(10.0, 2.0, 76) == pytest.approx(Kc * (1 - np.exp(-10 / Kc)))
    assert caps.count_saturation(1e-6, 2.0, 76) == pytest.approx(1e-6, rel=1e-5)
    assert caps.count_saturation(1e7, 2.0, 76) == pytest.approx(Kc)


# ------------------------------------------------------------------ tables
def test_joint_hist_scopes():
    rng = np.random.default_rng(0)
    s = rng.poisson(5, 1000)
    t = rng.poisson(3, 1000)
    h = caps.joint_hist(s, t)
    assert sum(h["genome"].values()) == 1000
    assert sum(h["top1pct"].values()) == 10
    top_s = np.sort(s)[-10:]
    assert sorted(k[0] for k, n in h["top1pct"].items() for _ in range(n)) == sorted(top_s)
    # the scope is never chosen by the target: changing the target leaves the top sources alone
    h2 = caps.joint_hist(s, t * 0 + 1000)
    assert sorted(k[0] for k, n in h2["top1pct"].items() for _ in range(n)) == sorted(top_s)
    # explicit mask, merging, and the deviance from the table equals the bin-wise deviance
    mask = s > 7
    hm = caps.joint_hist(s, t, top_mask=mask)
    assert sum(hm["top1pct"].values()) == mask.sum()
    merged = caps.merge_hists([h["genome"], h["genome"]])
    assert sum(merged.values()) == 2000
    from onion_eval.deviance import deviance
    assert caps.deviance_of(h["genome"], lambda x: 0.6 * x) == pytest.approx(
        deviance(t, 0.6 * s), rel=1e-10)
    assert caps.depth_ratio(h["genome"]) == pytest.approx(t.sum() / s.sum())
    with pytest.raises(ValueError):
        caps.deviance_of(h, lambda x: x)


# ------------------------------------------------------------------ fits, floors, gaps
@pytest.mark.parametrize("a_true", [0.3, 2.0])
def test_fit_shape_recovers_a(a_true):
    rng = np.random.default_rng(11)
    off, on = _simulate_pairs(60000, a_true, K, rng)
    a_hat = caps.fit_shape(caps.joint_hist(off, on)["genome"], K)
    assert abs(a_hat - a_true) / a_true < 0.10


def test_fit_shape_thinned_and_lists():
    rng = np.random.default_rng(12)
    a_true = 0.6
    hs = []
    for _ in range(2):
        off, on = _simulate_pairs(40000, a_true, K, rng, q=0.5)
        hs.append(caps.joint_hist(off, on)["genome"])
    a_hat = caps.fit_shape(hs, [K, K], q=[0.5, 0.5])
    assert abs(a_hat - a_true) / a_true < 0.10


def test_fit_kappa_recovers_kappa():
    rng = np.random.default_rng(13)
    kappa, L = 1.5, 50
    s = rng.poisson(rng.gamma(0.5, 30.0, 50000))
    t = rng.poisson(caps.count_saturation(s, kappa, L))
    k_hat = caps.fit_kappa(caps.joint_hist(s, t)["genome"], L)
    assert abs(k_hat - kappa) / kappa < 0.10


def test_floor_below_wrong_law_and_gap_closed_is_one():
    rng = np.random.default_rng(21)
    a = 0.5
    off, on = _simulate_pairs(60000, a, K, rng)
    h = caps.joint_hist(off, on)["genome"]
    D_floor = caps.simulate_floor(h, a, K, 1.0, rng, n_rep=20)
    D_law = caps.deviance_of(h, lambda s: caps.dm_distinct(s, a, K))
    D_wrong = caps.deviance_of(h, lambda s: caps.occupancy(s, K))
    D_ident = caps.deviance_of(h, lambda s: s)
    r = caps.depth_ratio(h)
    D_depth = caps.deviance_of(h, lambda s: r * s)
    assert D_floor < D_wrong
    assert D_floor < D_depth < D_ident
    # the simulating law reaches its floor: the gap closed is 1 within the Monte Carlo error
    # (the realized deviance of one table scatters around its expectation by a few percent)
    for D_base in (D_ident, D_depth):
        assert caps.gap_closed(D_base, D_law, D_floor) == pytest.approx(1.0, abs=0.03)
    assert caps.gap_closed(D_depth, D_wrong, D_floor) < 0.97
    # the floor matches the realized law deviance within Monte Carlo error
    assert abs(D_law - D_floor) / D_floor < 0.03
    assert caps.gap_closed(10.0, 4.0, 2.0) == pytest.approx(0.75)


# ------------------------------------------------------------------ subsampled law (iteration 2)
@pytest.mark.parametrize("q_s,q_t,a,share", [(0.3, 0.7, 0.5, 1.0), (0.6, 0.4, 2.0, 1.0),
                                             (0.4, 0.9, 0.3, 0.5)])
def test_subsampled_dm_distinct_matches_monte_carlo(q_s, q_t, a, share):
    """Exact mean against draws of the generative model: unseen reads U ~ NegBinomial(c, q_s),
    m ~ Binomial(c + U, share) reads on 25 positions, distinct kept with probability q_t."""
    rng = np.random.default_rng(7)
    c = np.array([1, 3, 10, 40, 150])
    ex = caps.subsampled_dm_distinct(c, q_s, q_t, a, 25, share)
    mc = np.array([caps.draw_subsampled_distinct(np.full(40000, x), q_s, q_t, a, 25, rng,
                                                 share).mean() for x in c])
    assert np.all(np.abs(ex / mc - 1) < 0.02), (ex, mc)


def test_subsampled_dm_distinct_q1_limit_and_inputs():
    c = np.array([0, 1, 2, 7, 30, 200])
    for a in (0.2, 1.0, np.inf):
        assert np.allclose(caps.subsampled_dm_distinct(c, 1.0, 1.0, a, 25), caps.dm_distinct(c, a, 25))
    assert np.allclose(caps.subsampled_dm_distinct(c, 1.0, 1.0, 0.7, 25, 0.5),
                       caps.thinned_dm_distinct(c, 0.5, 0.7, 25))
    assert np.allclose(caps.subsampled_dm_distinct(c, 1.0, 0.4, 0.7, 25),
                       0.4 * caps.dm_distinct(c, 0.7, 25))
    assert abs(caps.subsampled_dm_distinct(np.array([0]), 0.3, 0.5, 1.0, 25)[0]) < 1e-9
    # the unseen reads raise the expectation: a q_source sample predicts more than c alone
    assert np.all(caps.subsampled_dm_distinct(c[1:], 0.5, 1.0, 1.0, 25)
                  > caps.dm_distinct(c[1:], 1.0, 25))
    with pytest.raises(ValueError):
        caps.subsampled_dm_distinct(np.array([1.5]), 0.5, 0.5, 1.0, 25)
    with pytest.raises(ValueError):
        caps.subsampled_dm_distinct(c, 0.0, 0.5, 1.0, 25)


def test_completion_prior():
    """A Gamma(alpha, beta) intensity prior: fit_completion_prior recovers it from source counts,
    and with it a source count of 0 predicts a positive target (the unseen reads of empty bins)."""
    rng = np.random.default_rng(11)
    lam = rng.gamma(0.5, 1 / 0.05, 200000)                    # alpha 0.5, rate 0.05: mean 10
    c = rng.poisson(0.3 * lam)
    al, be = caps.fit_completion_prior(c, q_source=0.3)
    assert al == pytest.approx(0.5, rel=0.05) and be == pytest.approx(0.05, rel=0.1), (al, be)
    v, w = np.unique(c, return_counts=True)
    assert np.allclose(caps.fit_completion_prior(v, w, 0.3), (al, be), rtol=1e-3)
    pri = (al, be)
    assert caps.subsampled_dm_distinct(np.array([0]), 0.3, 1.0, 1.0, 25, prior=pri)[0] > 0.1
    # Monte Carlo of the law with the prior, and E[M | c] against the truth of the simulation
    for x in (0, 2, 20):
        mc = caps.draw_subsampled_distinct(np.full(40000, x), 0.3, 0.7, 0.8, 25, rng,
                                           prior=pri).mean()
        ex = caps.subsampled_dm_distinct(np.array([x]), 0.3, 0.7, 0.8, 25, prior=pri)[0]
        assert abs(ex / mc - 1) < 0.02, (x, ex, mc)
        lo, pm = caps._completion_pmf(x, 0.3, 1.0, al, be)
        # M = c + U, U ~ Poisson((1 - q) lambda): E[M | c] = c + 0.7 E[lambda | c]
        assert np.dot(pm, np.arange(lo, lo + pm.size)) == pytest.approx(
            x + 0.7 * lam[c == x].mean(), rel=0.05)


def test_subsampled_count_saturation_and_fits():
    c = np.array([0.0, 5, 50, 500])
    assert np.allclose(caps.subsampled_count_saturation(c, 1.0, 1.0, 3.0, 50),
                       caps.count_saturation(c, 3.0, 50))
    assert np.allclose(caps.subsampled_count_saturation(c, 1.0, 0.8, 3.0, 50),
                       0.8 * caps.count_saturation(c, 3.0, 50))
    # with completion: below the plug-in of the mean (concave law), above the unsubsampled law
    sub = caps.subsampled_count_saturation(c[1:], 0.5, 1.0, 3.0, 50)
    assert np.all(sub <= caps.count_saturation(c[1:] / 0.5, 3.0, 50) + 1e-9)
    unsub = caps.count_saturation(c[1:], 3.0, 50)
    assert np.all(sub >= unsub - 1e-9) and sub[0] > unsub[0]
    # fit_shape_subsampled recovers a from targets drawn from the subsampled model
    rng = np.random.default_rng(3)
    s = rng.integers(0, 120, 20000)
    t = caps.draw_subsampled_distinct(s, 0.4, 0.8, 0.6, 25, rng)
    h = caps.joint_hist(s, t)["genome"]
    a = caps.fit_shape_subsampled(h, 25, 0.4, 0.8)
    assert 0.4 < a < 0.9, a
    assert np.isclose(caps.fit_shape_subsampled([h, h], [25, 25], [0.4, 0.4], [0.8, 0.8]), a,
                      rtol=1e-3)
    fl = caps.simulate_subsampled_floor(h, a, 25, 0.4, 0.8, rng=rng, n_rep=5)
    d_law = caps.deviance_of(h, lambda x: caps.subsampled_dm_distinct(x, 0.4, 0.8, a, 25))
    d_q1 = caps.deviance_of(h, lambda x: caps.dm_distinct(x, a, 25))
    assert fl < d_q1 and abs(d_law / fl - 1) < 0.1, (fl, d_law, d_q1)
    k = caps.fit_kappa_subsampled([h, h], [50, 50], [0.4, 0.4], [0.8, 0.8], prior=[(0, 0), (0, 0)])
    assert np.isfinite(k)
