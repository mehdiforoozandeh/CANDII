"""Known-answer tests for tools/onion_eval/signal_from_counts.py (plan chunk X2, decision D9b).

Hand cases against scipy for the Poisson tail and the NB expectation; the lambda rules held equal to
the repo's exact MACS2 functions they restate; the lambda-source label of every path; monotonicity in
the predicted count; the hotspot2 window and normalization on hand cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nbinom, poisson

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import signal_from_counts as S  # noqa: E402
from candi.store.pval_from_counts import (  # noqa: E402
    MACS2_GSIZE_HS,
    pval_from_counts,
    pval_from_counts_with_control,
)

CHIP_COV = {"depth_reads": 2.0e7, "read_length": 36, "frag_len": 180,
            "control_depth_reads": 3.0e7, "control_read_length": 50}
ATAC_COV = {"depth_reads": 5.0e7, "read_length": 76}
DNASE_COV = {"depth_reads": 4.0e7, "read_length": 36}


def _nb_sample_mean(mu, n, f):
    """Independent E_NB[f(k)] by a plain sum over k = 0..20000."""
    k = np.arange(20001)
    w = nbinom.pmf(k, n, n / (n + mu))
    k, w = k[w > 1e-30], w[w > 1e-30]          # past this the plain-float64 tail is inf; mass < 1e-27
    return float((w * f(k)).sum() / w.sum())


def _mp_neglog10_sf(k, lam):
    """-log10 P(Poisson(lam) > k) at 50 digits: the regularized lower gamma P(k + 1, lam)."""
    import mpmath
    mpmath.mp.dps = 50
    return np.array([float(-mpmath.log10(mpmath.gammainc(int(kk) + 1, 0, lam, regularized=True)))
                     for kk in np.atleast_1d(k)])


# ------------------------------------------------------------------------------------- the tail
def test_poisson_upper_tail_matches_scipy_on_hand_cases():
    """Strict tail P(X > k). A vanishing mean puts all NB mass on k = 0: -log10(1 - e^-lam).
    For k > 0 the tail the module sums is scipy's sf at floor(k * scale)."""
    lam = np.array([0.5, 2.0, 7.3])
    got = S.nb_expected_neglog10p(np.full(3, 1e-12), 1.0, 1.0, lam)
    assert np.allclose(got, -np.log10(1.0 - np.exp(-lam)), rtol=1e-9)
    from candi.store.pval_from_counts import log10_upper_tail
    k = np.array([3, 12, 40])
    assert np.allclose(log10_upper_tail(k, lam), -np.log10(poisson.sf(k, lam)), rtol=1e-9)
    assert not np.allclose(log10_upper_tail(k, lam), -np.log10(poisson.sf(k - 1, lam)))


def test_nb_expectation_matches_an_independent_sum():
    lam = 1.7
    for mu, n, scale in ((3.0, 2.0, 1.0), (40.0, 5.0, 3.0), (0.2, 0.5, 2.5)):
        want = _nb_sample_mean(mu, n, lambda k: _mp_neglog10_sf(np.floor(k * scale), lam))
        got = S.nb_expected_neglog10p(np.array([mu]), n, scale, lam)[0]
        assert got == pytest.approx(want, rel=1e-6)


def test_expectation_is_monotonic_in_the_predicted_count():
    mu = np.linspace(0.05, 60.0, 200)
    got = S.nb_expected_neglog10p(mu, 3.0, 2.0, 1.5)
    assert np.all(np.diff(got) >= -1e-9)
    assert got[-1] > got[0]


# ------------------------------------------------------------------------------------- lambda
def test_lambda_is_the_max_of_the_local_and_genome_wide_rates():
    pile = np.zeros(1200)
    pile[600] = 4000.0                         # local window mean = 4000 / 400 = 10 near bin 600
    lam = S.macs2_lambda(lambda_bg=2.0, treat_pileup=pile)
    assert lam[600] == pytest.approx(10.0)
    assert lam[0] == pytest.approx(2.0)        # window misses the spike: background wins
    lam_hi = S.macs2_lambda(lambda_bg=12.0, treat_pileup=pile)
    assert np.all(lam_hi == 12.0)


def test_no_control_lambda_restates_pval_from_counts():
    rng = np.random.default_rng(0)
    obs = rng.poisson(3.0, 3000).astype(np.int64)
    obs[1500:1510] += 60
    lam = S.macs2_lambda(lambda_bg=1.1, treat_pileup=obs)
    from candi.store.pval_from_counts import log10_upper_tail
    assert np.array_equal(log10_upper_tail(obs, lam).astype(np.float32),
                          pval_from_counts(obs, lambda_bg=1.1))


def test_with_control_lambda_restates_pval_from_counts_with_control():
    rng = np.random.default_rng(1)
    obs = rng.poisson(4.0, 3000).astype(np.int64)
    ctrl = rng.poisson(2.0, 3000).astype(np.int64)
    ctrl[100:140] += 30
    lam = S.macs2_lambda(lambda_bg=0.7, control_pileup=ctrl, ratio_treat2control=1.6)
    from candi.store.pval_from_counts import log10_upper_tail
    want = pval_from_counts_with_control(obs, ctrl, lambda_bg=0.7, ratio_treat2control=1.6)
    assert np.array_equal(log10_upper_tail(obs, lam).astype(np.float32), want)


def test_pileup_conversion_is_count_times_d_over_read_span():
    assert S.pileup_from_counts(np.array([10.0]), 150, 76)[0] == pytest.approx(10 * 150 / 100)


# ------------------------------------------------------------------------------------- routes
def test_chip_route_end_to_end_on_a_hand_case():
    """Flat control: lambda is the larger of lambda_bg and r * the control pileup."""
    L = 800
    mu = np.full(L, 5.0)
    ctrl = np.full(L, 2.0)
    sig, src = S.signal_track(mu, 1e12, "narrow_histone", ctrl, CHIP_COV)
    assert src == "observed_control" and sig.dtype == np.float32 and sig.shape == (L,)
    r = 2.0e7 / 3.0e7
    lam = max(180 * 2.0e7 / MACS2_GSIZE_HS, r * 2.0 * 180 / (50 + 24))
    # middle bins: the 1 kb and 10 kb windows are full, so their means equal the flat pileup
    want = _nb_sample_mean(5.0, 1e12, lambda k: -np.log10(poisson.sf(np.floor(k * 3.0), lam)))
    assert sig[L // 2] == pytest.approx(want, rel=1e-5)


def test_chip_with_a_predicted_control_is_labelled_so():
    cov = dict(CHIP_COV, control_is_predicted=True)
    _, src = S.signal_track(np.full(50, 3.0), 4.0, "broad_histone", np.full(50, 1.0), cov)
    assert src == "predicted_control"
    _, src = S.signal_track(np.full(50, 3.0), 4.0, "H3K27ac", np.full(50, 1.0), CHIP_COV)
    assert src == "observed_control"


def test_chip_without_a_control_is_refused():
    with pytest.raises(ValueError, match="control"):
        S.signal_track(np.full(10, 3.0), 4.0, "narrow_histone", None, CHIP_COV)


def test_atac_route_uses_d_150_and_no_control():
    L = 1000
    mu = np.full(L, 4.0)
    sig, src = S.signal_track(mu, 1e12, "accessibility_atac", None, ATAC_COV)
    assert src == "local_background"
    pile = 4.0 * 150 / (76 + 24)                       # = 6.0
    lam = max(150 * 5.0e7 / MACS2_GSIZE_HS, pile)      # flat track: local mean = pileup
    want = _nb_sample_mean(4.0, 1e12, lambda k: -np.log10(poisson.sf(np.floor(k * 1.5), lam)))
    assert sig[L // 2] == pytest.approx(want, rel=1e-5)
    with pytest.raises(ValueError, match="no control"):
        S.signal_track(mu, 4.0, "ATAC-seq", np.ones(L), ATAC_COV)


def test_atac_signal_rises_with_the_predicted_count_at_a_peak():
    base = np.full(1000, 2.0)
    lo, hi = base.copy(), base.copy()
    lo[500], hi[500] = 20.0, 40.0
    s_lo, _ = S.signal_track(lo, 5.0, "ATAC-seq", None, ATAC_COV)
    s_hi, _ = S.signal_track(hi, 5.0, "ATAC-seq", None, ATAC_COV)
    assert s_hi[500] > s_lo[500]


def test_dnase_route_is_hotspot2_normalized_density():
    mu = np.zeros(40)
    mu[20] = 60.0                                      # 60 * 25 / 60 = 25 tags in bin 20
    sig, src = S.signal_track(mu, 3.0, "accessibility_dnase", None, DNASE_COV)
    assert src == "hotspot2_density"
    per_m = 1e6 / 4.0e7
    want = np.zeros(40)
    want[17] = want[23] = 0.5 * 25 * per_m             # half of an edge bin in the 150 bp window
    want[18:23] = 25 * per_m
    assert np.allclose(sig, want, rtol=1e-6)
    # a flat track: the window holds exactly 150 bp of tags
    flat, _ = S.signal_track(np.full(40, 6.0), 3.0, "DNase-seq", None, DNASE_COV)
    assert flat[20] == pytest.approx(6.0 * 25 / 60 * 6 * per_m, rel=1e-6)


def test_every_label_is_one_of_the_four():
    outs = [S.signal_track(np.full(30, 2.0), 3.0, "DNase-seq", None, DNASE_COV)[1],
            S.signal_track(np.full(30, 2.0), 3.0, "ATAC-seq", None, ATAC_COV)[1],
            S.signal_track(np.full(30, 2.0), 3.0, "H3K4me3", np.ones(30), CHIP_COV)[1],
            S.signal_track(np.full(30, 2.0), 3.0, "H3K4me3", np.ones(30),
                           dict(CHIP_COV, control_is_predicted=True))[1]]
    assert set(outs) == set(S.LAMBDA_SOURCES)


def test_bad_inputs_are_refused():
    with pytest.raises(ValueError):
        S.route_of("chipseq-control")
    with pytest.raises(ValueError):
        S.route_of("RNA-seq")
    with pytest.raises(KeyError):
        S.signal_track(np.full(5, 1.0), 2.0, "ATAC-seq", None, {"depth_reads": 1e6})
    with pytest.raises(ValueError):
        S.nb_expected_neglog10p(np.array([0.0, 1.0]), 2.0, 1.0, 1.0)
