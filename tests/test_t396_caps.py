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
