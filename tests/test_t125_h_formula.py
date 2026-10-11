"""t125 formula h — `tools/t125/hmap/formula.py`: the caller's Poisson upper tail at the local λ.

Written against the plan (chunk t125-P2b), `plan/ETA_OFFSET_AND_H_DESIGN.md` and the Phase 0 / PI
decisions in `plan/T125_STATUS.md`:
  pileup ≈ count × fraglen / (read_length + 24) × s_T (bin: the bin's count; frag: the mean count over
  the fragment window); λ = max(λ_bg, λ_frag, λ_1k, λ_10k) from the control, scaled to the smaller set
  (ratio V in ratio arms); no control → max(λ_bg(N_T), λ_10k of the treatment); P = −log10 P(X > floor(pileup)).

The test does not import P2a's modules (built in parallel): products are plain namespaces here.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mpmath
import numpy as np
import pytest
from scipy import stats
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))
sys.path.insert(0, str(ROOT / "tools" / "t125"))

from hmap import formula  # noqa: E402

LN10 = math.log(10.0)
G = 2.7e9


def info(**kw):
    """A minimal ProductInfo-like namespace (the P2a fields formula.py reads)."""
    base = dict(pid="X__base__base", assay="H3K27ac", depth_reads=30e6, read_length=101,
                fraglen=180, has_control=1, control_reads=40e6, ratio_V=None,
                control_read_length=101)
    base.update(kw)
    return SimpleNamespace(**base)


def brute_neglog10_sf(k, lam):
    """−log10 P(X > k) by a direct sum in log space (independent of scipy's sf)."""
    j = np.arange(int(k) + 1, int(k) + 1 + 5_000, dtype=np.float64)
    return -logsumexp(stats.poisson.logpmf(j, lam)) / LN10


def win_sum(x, w):
    """Centred window sum, zero outside the chromosome; window [i − w//2, i − w//2 + w)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(len(x))
    for i in range(len(x)):
        lo, hi = max(0, i - w // 2), min(len(x), i - w // 2 + w)
        out[i] = x[lo:hi].sum()
    return out


# ---------------------------------------------------------------------------------------------
# the tail
def test_hand_case_k10_lam2():
    p = formula.neglog10_sf_strict(np.array([10.0]), np.array([2.0]))
    assert p.dtype == np.float64
    assert p[0] == pytest.approx(-math.log10(stats.poisson.sf(10, 2.0)), rel=1e-12)
    direct = 1.0 - sum(math.exp(-2.0) * 2.0 ** j / math.factorial(j) for j in range(11))
    assert p[0] == pytest.approx(-math.log10(direct), rel=1e-6)
    assert p[0] == pytest.approx(brute_neglog10_sf(10, 2.0), rel=1e-10)


def test_phase0_number_is_the_strict_tail():
    # Phase 0: the caller gives 0.06315 at obs 0, λ = 2. That is −log10 P(X > 0) = −log10(1 − e^−2);
    # the non-strict tail P(X >= 0) = 1 would give 0.
    p = formula.neglog10_sf_strict(np.array([0.0]), np.array([2.0]))[0]
    assert p == pytest.approx(-math.log10(1.0 - math.exp(-2.0)), rel=1e-12)
    assert round(p, 5) == 0.06315
    assert -math.log10(stats.poisson.sf(-1, 2.0)) == 0.0   # the >= convention, for contrast


def test_floor_of_the_observed_value():
    p = formula.neglog10_sf_strict(np.array([2.0, 2.3, 2.999, 3.0]), np.full(4, 1.5))
    assert p[0] == p[1] == p[2]
    assert p[3] > p[2]


@pytest.mark.parametrize("k,lam", [(500, 1.0), (2000, 0.5), (5000, 3.0), (113_000, 1e5),
                                   (300, 1e-6)])
def test_tail_finite_and_exact_where_scipy_underflows(k, lam):
    p = formula.neglog10_sf_strict(np.array([float(k)]), np.array([lam]))[0]
    assert np.isfinite(p)
    mpmath.mp.dps = 50
    lam_m = mpmath.mpf(lam)
    term = mpmath.e ** (-lam_m) * lam_m ** (k + 1) / mpmath.factorial(k + 1)
    tot, i = mpmath.mpf(0), 0
    while True:
        tot += term
        term *= lam_m / (k + 2 + i)
        i += 1
        if term < tot * mpmath.mpf(10) ** -30:
            break
    exact = float(-mpmath.log10(tot))
    assert p == pytest.approx(exact, rel=1e-9)


def test_tail_continuous_across_the_underflow_switch():
    lam = 2.0
    ks = np.arange(0, 1500, dtype=np.float64)
    p = formula.neglog10_sf_strict(ks, np.full(len(ks), lam))
    assert np.all(np.isfinite(p))
    assert np.all(np.diff(p) > 0)
    for k in (150, 190, 191, 192, 193, 196, 197, 250, 400):  # the series takes over at k = 192
        assert p[k] == pytest.approx(brute_neglog10_sf(k, lam), rel=1e-9)


def test_lambda_zero_and_tiny():
    p = formula.neglog10_sf_strict(np.array([0.0, 0.0, 3.0]), np.array([0.0, 1e-30, 0.0]))
    assert p[0] == 0.0
    assert np.all(np.isfinite(p)) and np.all(p >= 0)
    # λ is clipped at 1e-12 from below
    assert p[1] == pytest.approx(-math.log10(-math.expm1(-1e-12)), rel=1e-9)
    assert p[2] == pytest.approx(brute_neglog10_sf(3, 1e-12), rel=1e-9)


def test_tail_monotone_in_k_and_broadcast():
    lam = np.array([0.3, 2.0, 40.0])
    k = np.tile(np.arange(0, 60, dtype=np.float64), (3, 1))
    p = formula.neglog10_sf_strict(k, lam[:, None])
    assert p.shape == (3, 60)
    assert np.all(np.diff(p, axis=1) >= 0)
    assert np.all(p >= 0)
    ref = -stats.poisson.logsf(k, lam[:, None]) / LN10
    ok = np.isfinite(ref) & (ref < 250)
    np.testing.assert_allclose(p[ok], ref[ok], rtol=1e-10, atol=1e-14)


# ---------------------------------------------------------------------------------------------
# geometry, scaling, λ
def test_frag_bins_and_window_sum_at_ends():
    assert formula.frag_bins(180) == 7        # round(7.2) = 7, odd
    assert formula.frag_bins(200) == 9        # round(8) = 8, made odd
    assert formula.frag_bins(150) == 7        # round(6) = 6, made odd
    assert formula.frag_bins(10) == 1
    x = np.arange(1, 11, dtype=np.float64)
    for w in (1, 3, 4, 7, 40):
        np.testing.assert_allclose(formula.window_sum(x, w), win_sum(x, w))
    # by hand: w = 3 at the ends sees two bins
    s = formula.window_sum(x, 3)
    assert s[0] == 1 + 2 and s[-1] == 9 + 10 and s[4] == 4 + 5 + 6


def test_scale_branches():
    # control larger: the control is scaled down to the treatment
    assert formula.scale(info(depth_reads=30e6, control_reads=40e6)) == pytest.approx((1.0, 0.75))
    # ctldepth q0.25: the treatment is larger and is scaled down
    sT, sC = formula.scale(info(depth_reads=30e6, control_reads=10e6))
    assert (sT, sC) == pytest.approx((1.0 / 3.0, 1.0))
    # ratio arm: V replaces N_T / N_C
    V = 2 * 30e6 / 107039349
    assert formula.scale(info(control_reads=107039349, ratio_V=V)) == pytest.approx((1.0, V))
    assert formula.scale(info(ratio_V=2.5)) == pytest.approx((0.4, 1.0))
    # no control: nothing is scaled
    assert formula.scale(info(has_control=0, control_reads=0)) == (1.0, 1.0)


def _ctl_windows(fr, w1, w10):
    return {"frag": np.asarray(fr, float), "1k": np.asarray(w1, float), "10k": np.asarray(w10, float)}


def test_local_lambda_with_control_by_hand():
    inf = info(depth_reads=30e6, control_reads=40e6, fraglen=180, control_read_length=101)
    sC = 0.75
    t = 25.0 / (101 + 24)                                  # tags per count
    bg = 180 * 40e6 * sC / G
    win = _ctl_windows([0, 200, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 30000])
    lam = formula.local_lambda(inf, win, None)
    assert lam.dtype == np.float64 and lam.shape == (4,)
    exp = [bg,
           max(bg, 200 * t * 180 / 180 * sC),
           max(bg, 1000 * t * 180 / 1000 * sC),
           max(bg, 30000 * t * 180 / 10000 * sC)]
    np.testing.assert_allclose(lam, exp, rtol=1e-12)
    assert lam[0] == pytest.approx(2.0)                    # 180 × 30e6 / 2.7e9: the background wins
    assert lam[1] > bg and lam[2] > bg and lam[3] > bg     # each window wins once
    # the control branch ignores any treatment window
    np.testing.assert_array_equal(formula.local_lambda(inf, win, np.full(4, 1e9)), lam)


def test_no_control_uses_the_10kb_treatment_window():
    inf = info(assay="DNase-seq", has_control=0, control_reads=0, depth_reads=50e6,
               read_length=76, fraglen=150, control_read_length=76)
    bg = 150 * 50e6 / G
    treat_10k = np.array([0.0, 1e4, 1e5, 1e6])
    lam = formula.local_lambda(inf, None, treat_10k)
    t = 25.0 / (76 + 24)
    np.testing.assert_allclose(lam, np.maximum(bg, treat_10k * t * 150 / 1e4), rtol=1e-12)
    assert lam[0] == pytest.approx(bg) and lam[3] > lam[2] > bg


def test_ratio_arm_uses_V():
    N_C = 107039349.0
    V = 2 * 30e6 / N_C
    inf_v = info(control_reads=N_C, ratio_V=V)
    inf_0 = info(control_reads=N_C, ratio_V=None)
    win = _ctl_windows([0, 500], [0, 0], [0, 0])
    lam_v = formula.local_lambda(inf_v, win, None)
    lam_0 = formula.local_lambda(inf_0, win, None)
    t = 25.0 / 125
    np.testing.assert_allclose(lam_v, [180 * N_C * V / G, max(180 * N_C * V / G, 500 * t * V)])
    np.testing.assert_allclose(lam_0, [180 * 30e6 / G, max(180 * 30e6 / G, 500 * t * 30e6 / N_C)])
    assert lam_v[1] == pytest.approx(2 * lam_0[1])


def test_q025_scales_the_treatment():
    inf = info(depth_reads=30e6, control_reads=10e6, read_length=101, fraglen=180)
    sT = 10e6 / 30e6
    np.testing.assert_allclose(formula.pileup(np.array([0.0, 3.0, 25.0]), inf),
                               np.array([0.0, 3.0, 25.0]) * 180 / 125 * sT)
    lam = formula.local_lambda(inf, _ctl_windows([0], [0], [0]), None)
    assert lam[0] == pytest.approx(180 * 10e6 / G)          # λ_bg from the (unscaled) control
    # and P itself uses the scaled pileup
    counts_t = np.array([0, 10, 50, 200], dtype=np.float64)
    counts_c = np.zeros(4)
    p = formula.P("bin", inf, counts_t, counts_c)
    exp = formula.neglog10_sf_strict(np.floor(counts_t * 180 / 125 * sT), np.full(4, lam[0]))
    np.testing.assert_allclose(p, exp, rtol=1e-12)


def test_pileup_unscaled_and_ratio():
    inf = info(depth_reads=30e6, control_reads=40e6)
    assert formula.pileup(10.0, inf) == pytest.approx(10 * 180 / 125)
    assert formula.pileup(10.0, info(ratio_V=4.0)) == pytest.approx(10 * 180 / 125 / 4.0)


# ---------------------------------------------------------------------------------------------
# P on tracks
def _manual_P(variant, inf, counts_t, counts_c, idx, treat=None):
    """P written out from the plan's definitions, loop-free but independent of formula.py."""
    counts_t = np.asarray(counts_t, float)
    treat = counts_t if treat is None else np.asarray(treat, float)
    rl = inf.read_length
    if inf.has_control:
        r = inf.ratio_V if inf.ratio_V is not None else inf.depth_reads / inf.control_reads
        sT, sC = (1.0, r) if r <= 1 else (1.0 / r, 1.0)
        t = 25.0 / (inf.control_read_length + 24)
        wf = formula.frag_bins(inf.fraglen)
        lam = np.maximum.reduce([
            np.full(len(counts_t), inf.fraglen * inf.control_reads * sC / G),
            win_sum(counts_c, wf) * t * inf.fraglen / inf.fraglen * sC,
            win_sum(counts_c, 40) * t * inf.fraglen / 1000 * sC,
            win_sum(counts_c, 400) * t * inf.fraglen / 10000 * sC])
    else:
        sT = 1.0
        t = 25.0 / (rl + 24)
        lam = np.maximum(inf.fraglen * inf.depth_reads / G,
                         win_sum(treat, 400) * t * inf.fraglen / 10000)
    if variant == "bin":
        c = counts_t
    else:
        wf = formula.frag_bins(inf.fraglen)
        c = (counts_t + win_sum(treat, wf) - treat) / wf
    pile = c * inf.fraglen / (rl + 24) * sT
    return np.array([brute_neglog10_sf(math.floor(pile[i]), lam[i]) for i in idx])


@pytest.mark.parametrize("variant", ["bin", "frag"])
@pytest.mark.parametrize("ctl", [True, False])
def test_P_matches_the_written_out_definition(variant, ctl):
    rng = np.random.default_rng(3)
    n = 900
    counts_t = rng.poisson(rng.gamma(0.6, 3.0, n)).astype(np.float64)
    counts_t[100:110] += 60
    if ctl:
        inf = info()
        counts_c = rng.poisson(1.5, n).astype(np.float64)
    else:
        inf = info(assay="DNase-seq", has_control=0, control_reads=0, depth_reads=50e6,
                   read_length=76, fraglen=150, control_read_length=76)
        counts_c = None
    p = formula.P(variant, inf, counts_t, counts_c)
    idx = np.r_[0:5, 95:115, n - 5:n]
    np.testing.assert_allclose(p[idx], _manual_P(variant, inf, counts_t, counts_c, idx),
                               rtol=1e-9, atol=1e-12)


def test_P_on_synthetic_products_finite_non_negative(tmp_path):
    from ladder import synth
    manifest, _ = synth.make_products(tmp_path / "prod", seed=0)
    rng = np.random.default_rng(11)
    for pid, inf in (("T1__base__base", info(fraglen=180, read_length=101)),
                     ("T1__depth__7.5M", info(depth_reads=7.5e6, fraglen=180, read_length=101)),
                     ("T2__base__base", info(assay="DNase-seq", has_control=0, control_reads=0,
                                             fraglen=150, read_length=76,
                                             control_read_length=76))):
        with np.load(tmp_path / "prod" / pid / "counts25.npz") as z:
            tracks = {c: z[c].astype(np.float64) for c in z.files}
        for chrom, ct in tracks.items():
            cc = rng.poisson(1.2, len(ct)).astype(np.float64) if inf.has_control else None
            for variant in ("bin", "frag"):
                p = formula.P(variant, inf, ct, cc)
                assert p.shape == ct.shape and p.dtype == np.float64
                assert np.all(np.isfinite(p)) and np.all(p >= 0)
                assert p.max() > 1.0                     # the planted peaks register
            # the chain path: f's mean as the treatment track
            pm = formula.P("frag", inf, ct, cc, treat_mean=ct * 0.5 + 0.1)
            assert np.all(np.isfinite(pm)) and np.all(pm >= 0)


@pytest.mark.parametrize("variant", ["bin", "frag"])
@pytest.mark.parametrize("ctl", [True, False])
@pytest.mark.parametrize("chain", [False, True])
def test_P_enumerated_equals_P_columnwise(variant, ctl, chain):
    rng = np.random.default_rng(5)
    n = 700
    counts_t = rng.poisson(rng.gamma(0.8, 2.5, n)).astype(np.float64)
    treat = rng.gamma(2.0, 1.0, n) if chain else None
    if ctl:
        inf = info(ratio_V=0.6)
        counts_c = rng.poisson(1.0, n).astype(np.float64)
    else:
        inf = info(has_control=0, control_reads=0, assay="CTCF")
        counts_c = None
    p = formula.P(variant, inf, counts_t, counts_c, treat_mean=treat)
    lam = formula.lambda_track(inf, counts_c, counts_t if treat is None else treat)
    rest = (formula.frag_rest(counts_t if treat is None else treat, inf)
            if variant == "frag" else None)
    k = np.stack([counts_t, counts_t + 1, np.zeros(n), counts_t], axis=1)
    pe = formula.P_enumerated(variant, inf, k, lam, rest)
    assert pe.shape == (n, 4) and pe.dtype == np.float64
    np.testing.assert_allclose(pe[:, 0], p, rtol=0, atol=0)
    np.testing.assert_allclose(pe[:, 3], p, rtol=0, atol=0)
    assert np.all(pe[:, 1] >= pe[:, 0]) and np.all(pe[:, 2] <= pe[:, 0])


def test_bin_and_frag_differ_only_through_the_window():
    rng = np.random.default_rng(9)
    n = 1200
    inf = info()
    counts_c = rng.poisson(1.0, n).astype(np.float64)
    # a constant treatment: the window mean equals the bin away from the chromosome ends
    const = np.full(n, 7.0)
    wf = formula.frag_bins(inf.fraglen)
    pb = formula.P("bin", inf, const, counts_c)
    pf = formula.P("frag", inf, const, counts_c)
    inner = slice(wf, n - wf)
    np.testing.assert_array_equal(pb[inner], pf[inner])
    # in general, frag = bin evaluated on the fragment-window mean, at the same λ
    counts_t = rng.poisson(3.0, n).astype(np.float64)
    pf = formula.P("frag", inf, counts_t, counts_c)
    mean_w = formula.window_sum(counts_t, wf) / wf
    lam = formula.lambda_track(inf, counts_c, counts_t)
    np.testing.assert_allclose(
        pf, formula.neglog10_sf_strict(np.floor(formula.pileup(mean_w, inf)), lam), rtol=0, atol=0)
    pb2 = formula.P("bin", inf, mean_w, counts_c)
    np.testing.assert_allclose(pf, pb2, rtol=0, atol=0)


def test_bad_variant_raises():
    with pytest.raises(ValueError):
        formula.P("window", info(), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        formula.P_enumerated("frag", info(), np.zeros((3, 2)), np.ones(3), None)
