"""Known-answer tests for tools/onion_eval/metrics.py (the 7 headline metrics).

Each metric has: a known-answer case, a planted-error case (the metric moves in the stated
direction), an undefined case (NaN with a reason, no raise), and, where it applies, a tie case and a
3-tracks-per-assay case. Several values are recomputed by an independent formula.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nbinom

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import metrics as M  # noqa: E402
from candi.metrics import ece  # noqa: E402


def _undefined(v):
    return isinstance(v, M.Undefined) and math.isnan(v) and v.reason


# ------------------------------------------------------------------ to_space

def test_to_space_known_values():
    x = np.array([0.0, 1.0, np.e - 1])
    assert np.allclose(M.to_space(x, "counts_log1p"), [0.0, np.log(2), 1.0])
    assert np.allclose(M.to_space(x, "counts_log1p", depth=2.0), np.log1p(2 * x))
    assert np.allclose(M.to_space([0.0, np.sinh(1.0)], "signal_arcsinh"), [0.0, 1.0])
    with pytest.raises(ValueError):
        M.to_space(x, "asinh_p")
    with pytest.raises(ValueError):
        M.to_space(x, "signal_arcsinh", depth=2.0)


# ------------------------------------------------------------------ metric 1

def _orthogonal_tracks(M_, T, seed=0):
    """s and u_1..u_T: mean-zero, mutually orthogonal, each with variance exactly 1."""
    Z = np.random.default_rng(seed).normal(size=(M_, T + 1))
    Z -= Z.mean(0)
    Q, _ = np.linalg.qr(Z)
    Q *= np.sqrt(M_)
    return Q[:, 0], Q[:, 1:].T


def test_shared_specific_exact_split_three_tracks():
    # truth_t = s + u_t with Var(s) = Var(u_t) = 1 and exact orthogonality: V_sh = V_sp = 0.5 per target.
    s, U = _orthogonal_tracks(1024, 3)
    X = s + U
    out = M.shared_specific(X, X.copy(), groups=["a", "b", "c"], J=4, calibrated=False)
    assert np.allclose(out["V_sh"], 0.5, atol=1e-12) and np.allclose(out["V_sp"], 0.5, atol=1e-12)
    assert np.allclose(out["E_sp"], 0.0, atol=1e-12) and np.allclose(out["E_sh"], 0.0, atol=1e-12)
    assert M.specific_variance_recovered(out["E_sp"], out["V_sp"], np.zeros(3)) == pytest.approx(1.0)
    # planted error: predict the shared part only -> all specific variance is error, recovered = 0
    out2 = M.shared_specific(X, np.tile(s, (3, 1)), groups=["a", "b", "c"], J=4, calibrated=False)
    assert np.allclose(out2["E_sp"], 0.5, atol=1e-12) and np.allclose(out2["E_sh"], 0.0, atol=1e-12)
    assert M.specific_variance_recovered(out2["E_sp"], out2["V_sp"], 0.0) == pytest.approx(0.0, abs=1e-12)
    # by-level parts add up to the totals
    assert np.allclose(out2["by_level"]["E_sp"].sum(0), out2["E_sp"])


def test_shared_specific_calibrated_perfect_and_shared_only():
    s, U = _orthogonal_tracks(1 << 14, 3, seed=1)
    X = s + U
    # a monotone transform of the truth calibrates back to the truth: no error, level reported
    out = M.shared_specific(X, 3.0 * X + 1.0, groups=[0, 1, 2], J=6)
    assert np.allclose(out["E_sp"], 0.0, atol=1e-9) and np.all(out["level"] > 0)
    out2 = M.shared_specific(X, np.tile(s, (3, 1)), groups=[0, 1, 2], J=6)
    assert np.allclose(out2["E_sp"], 0.5, atol=0.02)


def test_shared_specific_undefined_cases():
    s, U = _orthogonal_tracks(256, 3)
    X = s + U
    X[2] = 0.0                               # zero-variance target
    out = M.shared_specific(X, X, groups=["a", "b", "a"], J=3, calibrated=False)
    assert out["reason"][2] == "truth has zero variance" and np.isnan(out["V_sp"][2])
    # targets 0 and 1 are of different groups, so they are still defined
    assert np.isfinite(out["V_sp"][:2]).all()
    one = M.shared_specific(X[:2], X[:2], groups=["a", "a"], J=3, calibrated=False)
    assert all(r == "no target of another group in this assay" for r in one["reason"])
    assert np.isnan(one["N"]).all()


def test_specific_variance_recovered_known_and_undefined():
    assert M.specific_variance_recovered([0.5], [1.0], [0.25]) == pytest.approx(2 / 3)
    # ratio of means, not mean of ratios: (0.2+0.6)/2=0.4 error, V mean 1.0, N 0 -> 0.6
    assert M.specific_variance_recovered([0.2, 0.6], [0.5, 1.5], [0.0, 0.0]) == pytest.approx(0.6)
    # non-finite targets dropped
    assert M.specific_variance_recovered([0.5, np.nan], [1.0, 1.0], [0.25, 0.25]) == pytest.approx(2 / 3)
    assert _undefined(M.specific_variance_recovered([0.5], [0.3], [0.3]))
    assert _undefined(M.specific_variance_recovered([np.nan], [1.0], [0.0]))
    # planted: more error -> lower
    assert M.specific_variance_recovered([0.6], [1.0], [0.25]) < M.specific_variance_recovered([0.5], [1.0], [0.25])


def test_noise_floor_spearman_brown():
    rng = np.random.default_rng(3)
    L = 1 << 16
    sig = rng.normal(size=L)
    sh2 = 1.0                                # half-depth noise variance; signal variance 1
    e1, e2 = rng.normal(scale=np.sqrt(sh2), size=(2, L))
    target = sig + 0.5 * (e1 + e2)           # full depth = both halves pooled
    N = M.noise_floor(sig + e1, sig + e2, target, depth_ratio=2.0, J=8)
    # independent formula: noise variance (sh2 / 2) over target variance (1 + sh2 / 2) = 1/3
    assert N == pytest.approx((sh2 / 2) / (1 + sh2 / 2), abs=0.02)
    assert M.noise_floor(sig, sig, sig, J=8) == pytest.approx(0.0, abs=1e-12)    # R = 1 -> no noise
    assert M.noise_floor(e1, e2, 0.5 * (e1 + e2), J=8) == pytest.approx(1.0, abs=0.03)  # R ~ 0
    assert _undefined(M.noise_floor(np.ones(64), sig[:64], sig[:64], J=3))


# ------------------------------------------------------------------ metric 2

def _rspec_data(L=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.gamma(0.5, 2.0, size=L)
    a[rng.random(L) < 0.3] = 0.0             # zero inflation: ties in a
    y = np.sqrt(a) + rng.normal(scale=0.5, size=L)
    return rng, a, y


def test_r_spec_average_and_linear_rescale_score_zero():
    _, a, y = _rspec_data()
    assert M.r_spec(y, a, a) == 0.0
    assert M.r_spec(y, 3.7 * a - 1.2, a) == 0.0
    assert M.r_spec(y, y, a) == pytest.approx(1.0)


def test_r_spec_planted_noise_lowers_it():
    rng, a, y = _rspec_data()
    r1 = M.r_spec(y, y + rng.normal(scale=0.3, size=y.size), a)
    r2 = M.r_spec(y, y + rng.normal(scale=1.0, size=y.size), a)
    assert 1.0 > r1 > r2 > 0.0


def test_r_spec_constant_average_equals_pearson_and_undefined():
    rng, _, y = _rspec_data()
    p = y + rng.normal(size=y.size)
    a0 = np.zeros_like(y)                    # all ties: the level function is a constant
    assert M.r_spec(y, p, a0) == pytest.approx(np.corrcoef(y, p)[0, 1], abs=1e-12)
    _, a, _ = _rspec_data()
    assert _undefined(M.r_spec(2 * a + 1, p, a))   # truth is a function of a


# ------------------------------------------------------------------ metrics 3, 4

def _exact_nb_crps(n, mu, y, kmax=2000):
    """Independent formula: CRPS = sum_k (F(k) - 1{k >= y})^2 for a discrete forecast."""
    k = np.arange(kmax)
    p = n / (n + mu)
    return np.array([np.sum((nbinom.cdf(k, ni, pi) - (k >= yi)) ** 2) for ni, pi, yi in zip(n, p, y)])


def test_crps_split_identity_and_independent_formula():
    rng = np.random.default_rng(4)
    mu = rng.gamma(2.0, 2.0, size=300)
    n = np.full(300, 2.0)
    y = rng.negative_binomial(n, n / (n + mu))
    s = M.crps_split(n, mu, y)
    assert s["crps"] == pytest.approx(s["oracle_scaled"] + s["scale_error"], abs=1e-15)
    assert s["crps"] == pytest.approx(_exact_nb_crps(n, mu, y).mean(), rel=1e-6)
    # planted: a 4x level error raises scale_error, oracle_scaled stays near the honest value
    s4 = M.crps_split(n, 4 * mu, y)
    assert s4["crps"] > s["crps"] and s4["scale_error"] > s["scale_error"] + 0.5
    assert s4["oracle_scaled"] == pytest.approx(s["oracle_scaled"], rel=0.05)
    sk = M.crps_skill(s, s4)
    assert sk["crps"] > 0 and sk["crps"] == pytest.approx(s4["crps"] - s["crps"])
    assert set(sk) == {"crps", "oracle_scaled", "scale_error"}
    assert _undefined(M.crps_split([], [], [])["crps"])


def test_gap_closed_known_tie_and_undefined():
    assert M.gap_closed(0.5, 1.0, 0.0) == pytest.approx(0.5)
    assert M.gap_closed(0.0, 1.0, 0.0) == pytest.approx(1.0)
    assert M.gap_closed(1.0, 1.0, 0.0) == pytest.approx(0.0)       # tie with the baseline
    assert M.gap_closed([0.6, 0.4], [1.0, 1.0], [0.2, 0.2]) == pytest.approx(0.625)   # ratio of means
    assert M.gap_closed(1.2, 1.0, 0.0) < 0                         # planted: worse than baseline
    assert _undefined(M.gap_closed(0.5, 0.3, 0.3))
    assert _undefined(M.gap_closed(np.nan, 1.0, 0.0))
    assert M.uncertainty_gap_closed(0.01, 0.04) == pytest.approx(0.75)
    assert _undefined(M.uncertainty_gap_closed(0.01, 0.0))


# ------------------------------------------------------------------ metric 5

def test_depth_law_error():
    src = np.array([100.0, 200.0, 50.0])
    r = np.array([1.0, -1.0, 0.0])
    assert M.depth_law_error(src * 2.0 ** r, src, r) == 0.0
    planted = src * 2.0 ** (r + np.array([1.0, -0.5, 2.0]))
    assert M.depth_law_error(planted, src, r) == pytest.approx(1.0)   # median of 1, 0.5, 2
    assert M.depth_law_error([0.0, 400.0], [100.0, 100.0], [1.0, 1.0]) == pytest.approx(1.0)  # zero total dropped
    assert _undefined(M.depth_law_error([0.0], [100.0], [0.0]))


# ------------------------------------------------------------------ metric 6

def test_pit_deviation_analytic_all_zero_counts():
    # y = 0 everywhere and F(0) = q: F_bar(u) = min(u / q, 1), so the deviation is known in closed form
    n, mu = np.full(50, 2.0), np.full(50, 2.0)
    q = nbinom.pmf(0, 2.0, 0.5)
    u = np.linspace(0, 1, 21)
    want = np.mean(np.abs(np.minimum(u / q, 1.0) - u))
    out = M.pit_deviation(n, mu, np.zeros(50))
    assert out["pooled"] == pytest.approx(want, abs=1e-9)


def test_pit_deviation_calibrated_planted_and_strata():
    rng = np.random.default_rng(5)
    L = 60000
    mu = rng.gamma(1.0, 3.0, size=L)
    n = np.full(L, 1.5)
    y = rng.negative_binomial(n, n / (n + mu))
    lab = np.where(np.arange(L) < L // 2, "lo", "hi")
    good = M.pit_deviation(n, mu, y, lab)
    assert good["pooled"] < 0.01
    # second method: equals candi.metrics.ece on the same bins
    assert good["pooled"] == pytest.approx(ece(n, n / (n + mu), y), abs=1e-12)
    mu_bad = np.where(lab == "hi", 3.0 * mu, mu)    # planted: stratum "hi" miscalibrated
    bad = M.pit_deviation(n, mu_bad, y, lab)
    assert bad["pooled"] > good["pooled"] and bad["worst_stratum_name"] == "hi"
    assert bad["worst_stratum"] > 5 * bad["per_stratum"]["lo"]
    marg = M.marginal_nb(y)
    pm = M.pit_deviation(np.full(L, marg["marg_n"]), np.full(L, marg["marg_mu"]), y)["pooled"]
    assert 0 < M.uncertainty_gap_closed(good["pooled"], pm) <= 1
    assert _undefined(M.pit_deviation([], [], [])["pooled"])


# ------------------------------------------------------------------ metric 7

def test_variance_ratio_analytic_and_planted():
    n, mu = np.full(4, 2.0), np.full(4, 4.0)                        # predicted var = 4 + 16/2 = 12
    a, b = np.array([0, 10, 3, 7.0]), np.array([4, 6, 7, 3.0])      # |a - b| = 4 -> replicate var 8
    out = M.variance_ratio(n, mu, a, b, strata=np.array([0, 0, 1, 1]))
    assert out["pooled"] == pytest.approx(12 / 8)
    assert out["per_stratum"]["0"] == pytest.approx(1.5)
    rng = np.random.default_rng(6)
    L = 200000
    mu = rng.gamma(2.0, 2.0, size=L)
    n = np.full(L, 3.0)
    ra, rb = (rng.negative_binomial(n, n / (n + mu)) for _ in range(2))
    honest = M.variance_ratio(n, mu, ra, rb)["pooled"]
    assert honest == pytest.approx(1.0, abs=0.03)
    assert M.variance_ratio(n / 3, mu, ra, rb)["pooled"] > honest + 0.5   # planted over-dispersion
    z = M.variance_ratio(n[:3], mu[:3], np.ones(3), np.ones(3))
    assert _undefined(z["pooled"]) and z["reason"]


# ------------------------------------------------------------------ retry 1: review defects

def test_to_space_rescale_keyword_and_input_checks():
    x = np.array([0.0, 1.0, 3.0])
    assert np.allclose(M.to_space(x, "counts_log1p", rescale=2.0), np.log1p(2 * x))
    assert np.allclose(M.to_space(x, "counts_log1p", 2.0), M.to_space(x, "counts_log1p", rescale=2.0))
    for bad in (3e7, 0.0, -1.0):                                  # an absolute depth is refused
        with pytest.raises(ValueError):
            M.to_space(x, "counts_log1p", rescale=bad)
    with pytest.raises(ValueError):
        M.to_space(x, "counts_log1p", 2.0, rescale=2.0)
    for bad_x in ([0.0, np.nan], [0.0, -1.0], [np.inf]):
        for sp in M.SPACES:
            with pytest.raises(ValueError):
                M.to_space(bad_x, sp)


def test_shared_specific_level_defined_in_both_modes():
    s, U = _orthogonal_tracks(1024, 3)
    X = s + U
    for cal in (True, False):
        out = M.shared_specific(X, 2.0 * X + 1.0, groups=[0, 1, 2], J=4, calibrated=cal)
        assert np.isfinite(out["level"]).all() and np.isfinite(out["offset"]).all()
        assert np.allclose(out["level"], out["offset"] + out["scale"] + out["nonlinear"])
    # 2X + 1: standardized offset (1 + mean)^2/Var = 0.5, scale (2 SD - SD)^2/Var = 1 (Var(x) = 2, mean 0)
    assert out["offset"] == pytest.approx(np.full(3, 0.5)) and out["scale"] == pytest.approx(np.ones(3))


def test_noise_floor_colored_signal_and_levels():
    rng = np.random.default_rng(7)
    L = 1 << 16
    sig = np.repeat(rng.normal(size=L // 256), 256) * 2.0         # low-frequency signal, variance ~4
    e1, e2 = rng.normal(size=(2, L))
    target = sig + 0.5 * (e1 + e2)
    noise = 0.5 * (e1 + e2)
    want = noise.var() / target.var()                             # independent formula
    out = M.noise_floor(sig + e1, sig + e2, target, 2.0, J=10, return_levels=True)
    assert out["N"] == pytest.approx(want, abs=0.01)
    assert M.noise_floor(sig + e1, sig + e2, target, 2.0, J=10) == out["N"]
    # fine levels are pure noise (R ~ 0), coarse levels pure signal (R ~ 1)
    assert out["R"][0] < 0.05 and out["R"][9] > 0.9
    assert list(out["n_coef"][:3]) == [L // 2, L // 4, L // 8] and out["n_coef"][-1] == L >> 10
    assert np.isclose(out["energy_share"].sum(), 1.0)


def test_crps_split_nan_mask_negative_and_same_estimator():
    rng = np.random.default_rng(8)
    mu = rng.gamma(2.0, 2.0, size=400)
    n = np.full(400, 2.0)
    y = rng.negative_binomial(n, n / (n + mu)).astype(float)
    ref = M.crps_split(n[1:], mu[1:], y[1:])
    mu_nan = mu.copy(); mu_nan[0] = np.nan
    got = M.crps_split(n, mu_nan, y)
    assert got["n_dropped"] == 1 and got["crps"] == pytest.approx(ref["crps"]) and not got["reason"]
    m = np.ones(400, bool); m[0] = False
    assert M.crps_split(n, mu, y, mask=m)["crps"] == pytest.approx(ref["crps"])
    allnan = M.crps_split(n, np.full(400, np.nan), y)
    assert _undefined(allnan["crps"]) and allnan["reason"]
    with pytest.raises(ValueError):
        M.crps_split(n, mu, -y - 1)
    # crps_approx goes to both numbers, so scale_error stays near the closed-form one
    samp = M.crps_split(n, mu, y, crps_approx=200)
    assert samp["crps"] == pytest.approx(samp["oracle_scaled"] + samp["scale_error"])
    assert abs(samp["scale_error"] - M.crps_split(n, mu, y)["scale_error"]) < 0.05


def test_pit_deviation_nan_and_mask():
    rng = np.random.default_rng(9)
    L = 5000
    mu = rng.gamma(1.0, 3.0, size=L)
    n = np.full(L, 1.5)
    y = rng.negative_binomial(n, n / (n + mu)).astype(float)
    lab = np.where(np.arange(L) % 2 == 0, "a", "b")
    ref = M.pit_deviation(n[1:], mu[1:], y[1:], lab[1:])
    yn = y.copy(); yn[0] = np.nan
    got = M.pit_deviation(n, mu, yn, lab)
    assert got["n_dropped"] == 1 and got["pooled"] == pytest.approx(ref["pooled"])
    assert got["per_stratum"] == pytest.approx(ref["per_stratum"])
    m = np.ones(L, bool); m[0] = False
    assert M.pit_deviation(n, mu, y, lab, mask=m)["pooled"] == pytest.approx(ref["pooled"])
    assert _undefined(M.pit_deviation(n, np.full(L, np.nan), y)["pooled"])
    with pytest.raises(ValueError):
        M.pit_deviation(n, mu, -y - 1)


def test_variance_ratio_unequal_depth_planted():
    rng = np.random.default_rng(10)
    L = 200000
    mu = rng.gamma(2.0, 2.0, size=L)
    n = np.full(L, 3.0)
    ra = rng.negative_binomial(n, n / (n + mu))
    rb2 = rng.poisson(rng.gamma(n, 2 * mu / n))                  # b at twice a's depth (NB(n, 2 mu))
    honest = M.variance_ratio(n, mu, ra, rng.negative_binomial(n, n / (n + mu)))["pooled"]
    matched = M.variance_ratio(n, mu, ra, rb2, depth_ratio_b=0.5)
    # thinning NB(n, 2mu) by 1/2 gives NB(n, mu): ratio back near 1, within seed noise of the honest one
    assert matched["pooled"] == pytest.approx(1.0, abs=0.03) and abs(matched["pooled"] - honest) < 0.04
    undeclared = M.variance_ratio(n, mu, ra, rb2)                 # depth mismatch not declared
    assert _undefined(undeclared["pooled"]) and "totals differ" in undeclared["reason"]
    # b shallower: a is thinned and mu with it
    shallow = M.variance_ratio(n, 2 * mu, rb2, ra, depth_ratio_b=2.0)
    assert shallow["pooled"] == pytest.approx(1.0, abs=0.03)


def test_variance_ratio_nan_replicate_and_negative():
    n, mu = np.full(5, 2.0), np.full(5, 4.0)
    a, b = np.array([0, 10, 3, 7.0, np.nan]), np.array([4, 6, 7, 3.0, 5.0])
    out = M.variance_ratio(n, mu, a, b)
    assert out["n_dropped"] == 1 and out["pooled"] == pytest.approx(12 / 8) and not out["reason"]
    allnan = M.variance_ratio(n, mu, np.full(5, np.nan), b)
    assert _undefined(allnan["pooled"]) and "finite" in allnan["reason"]
    with pytest.raises(ValueError):
        M.variance_ratio(n, mu, -np.ones(5), b)
