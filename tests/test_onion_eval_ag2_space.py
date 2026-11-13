"""Tests of tools/onion_eval/ag2_space.py (plan Amendment 5, pinned interface H, chunk AGs)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nbinom, pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import ag2_space as A  # noqa: E402
from onion_eval import signal_from_counts as S  # noqa: E402
from candi.store.pval_from_counts import MACS2_GSIZE_HS, log10_upper_tail  # noqa: E402

CHIP_COV = {"depth_reads": 2e7, "read_length": 76, "frag_len": 200,
            "control_depth_reads": 1e7, "control_read_length": 76}


def test_no_alphagenome_import():
    assert "alphagenome" not in sys.modules
    assert "import alphagenome" not in (REPO / "tools/onion_eval/ag2_space.py").read_text()


# ---------------------------------------------------------------- resample

def test_resample_mean_constant():
    x = np.full(40, 3.7)
    assert np.allclose(A.resample_mean(x, 0, 128, 0, 25, 200), 3.7)
    y = np.full(400, 3.7)
    assert np.allclose(A.resample_mean(y, 1000, 25, 1000, 128, 70), 3.7)


def test_resample_mean_conserves_integral_on_aligned_span():
    rng = np.random.default_rng(0)
    x128 = rng.gamma(2.0, 1.5, size=50)            # 50 * 128 = 6400 bp = 2 * lcm(128, 25)
    x25 = A.resample_mean(x128, 0, 128, 0, 25, 256)
    assert np.isclose(x25.sum() * 25, x128.sum() * 128, rtol=1e-12)
    back = A.resample_mean(x25, 0, 25, 0, 128, 50)
    assert np.isclose(back.sum() * 128, x128.sum() * 128, rtol=1e-12)


def test_resample_mean_hand_value_3_bins():
    # 128 bp bins [1, 2, 3] from 0 bp; 25 bp bins from 100 bp:
    # [100,125) in bin 0 -> 1; [125,150) = 3 bp of 1 + 22 bp of 2 -> 47/25; [150,175) -> 2
    got = A.resample_mean(np.array([1.0, 2.0, 3.0]), 0, 128, 100, 25, 3)
    assert np.allclose(got, [1.0, 47 / 25, 2.0], rtol=0, atol=1e-12)


def test_resample_mean_outside_source_is_nan_and_matches_max_to25_inside():
    x = np.arange(1.0, 6.0)                         # 5 bins, 640 bp
    got = A.resample_mean(x, 0, 128, 0, 25, 30)     # 750 bp: last bins run off the end
    F = np.concatenate([[0.0], np.cumsum(x)])       # Max's to25 (score_f1.py l.210-214)
    pos = np.minimum(np.arange(26) * 25 / 128.0, 5)
    to25 = np.diff(np.interp(pos, np.arange(6), F)) * 128 / 25
    assert np.allclose(got[:25], to25)
    assert np.isnan(got[-1])


def test_resample_sum_1bp_to_25bp():
    x = np.arange(100, dtype=float)
    assert np.allclose(A.resample_sum(x, 0, 1, 0, 25, 4), x.reshape(4, 25).sum(1))


# ---------------------------------------------------------------- fold change

def test_fold_change_flat_control_hand_value():
    L = 1000
    mu = np.full(L, 30.0)
    ctl = np.full(L, 5.0)
    # pileup scale 200 / (76 + 24) = 2: treatment pileup 60; control pileup 10 * r (= 2e7 / 1e7) = 20;
    # lambda_bg = 200 * 2e7 / 2.7e9 = 1.48 < 20; the local windows of a flat control are <= 20
    fe, src = A.fold_change_track(mu, ctl, CHIP_COV)
    assert fe.dtype == np.float32 and src == "observed_control"
    assert np.allclose(fe, 3.0, rtol=1e-6)


@pytest.mark.parametrize("predicted", [False, True])
def test_fold_change_lambda_source_matches_signal_track(predicted):
    rng = np.random.default_rng(1)
    mu = rng.gamma(2.0, 3.0, 800) + 0.1
    ctl = rng.gamma(2.0, 2.0, 800) + 0.1
    cov = {**CHIP_COV, "control_is_predicted": predicted}
    _, src = A.fold_change_track(mu, ctl, cov)
    _, ref = S.signal_track(mu, np.full_like(mu, 5.0), "narrow_histone", ctl, cov, mode="plug_in")
    assert src == ref


def test_fold_change_at_mu_equals_nb_mean_by_simulation():
    rng = np.random.default_rng(2)
    L, draws, n = 120, 2000, 4.0
    mu = rng.gamma(3.0, 4.0, L) + 1.0
    ctl = rng.gamma(2.0, 2.0, L) + 0.5
    at_mu, _ = A.fold_change_track(mu, ctl, CHIP_COV)
    k = nbinom.rvs(n, n / (n + mu), size=(draws, L), random_state=rng)
    mean = np.mean([A.fold_change_track(k[i].astype(float), ctl, CHIP_COV)[0] for i in range(draws)], axis=0)
    assert abs(mean.sum() / at_mu.sum() - 1) < 0.01
    assert np.max(np.abs(mean / at_mu - 1)) < 0.15     # per bin: 2000 draws, cv of NB(4, ~13) about 0.55


@pytest.mark.parametrize("rl", [36, 50, 101])
def test_normalized_insertions_sum_to_1e8(rl):
    # Simulate reads on a small genome and bin them by the onion rule (a read adds 1 to every 25 bp
    # bin it overlaps). Start offsets mod 25 are balanced, so the mean bins per read is exactly
    # (rl + 24) / 25 and the track must sum to 1e8 up to float rounding.
    rng = np.random.default_rng(3)
    nbins, n_reads = 20_000, 25 * 2_000
    genome_bp = nbins * 25
    base = rng.integers(0, (genome_bp - rl - 25) // 25, size=n_reads) * 25
    starts = base + np.arange(n_reads) % 25
    first, last = starts // 25, (starts + rl - 1) // 25
    counts = np.zeros(nbins + 1)
    np.add.at(counts, first, 1.0)
    np.add.at(counts, last + 1, -1.0)
    counts = np.cumsum(counts)[:nbins]
    assert counts.sum() == pytest.approx(n_reads * (rl + 24) / 25)
    v = A.normalized_insertions(counts, {"depth_reads": float(n_reads), "read_length": float(rl)})
    assert abs(v.astype(np.float64).sum() / 1e8 - 1) < 1e-6


def test_signal_track_plug_in_equals_score_with_this_modules_lambda():
    rng = np.random.default_rng(8)
    mu = rng.gamma(2.0, 4.0, 1500) + 0.1
    ctl = rng.gamma(2.0, 2.0, 1500) + 0.1
    lam, d, rl, _ = A._chip_lambda(mu, ctl, CHIP_COV, MACS2_GSIZE_HS)
    ref = log10_upper_tail(np.floor(S.pileup_from_counts(mu, d, rl)).astype(np.int64), lam)
    sig, _ = S.signal_track(mu, None, "narrow_histone", ctl, CHIP_COV, mode="plug_in")
    assert np.allclose(sig, ref.astype(np.float32), rtol=1e-6, atol=1e-6)
    cov = {"depth_reads": 3e7, "read_length": 50}
    lam, d, rl, _ = A._no_control_lambda(mu, cov, MACS2_GSIZE_HS)
    ref = log10_upper_tail(np.floor(S.pileup_from_counts(mu, d, rl)).astype(np.int64), lam)
    for assay in ("accessibility_atac", "accessibility_dnase"):
        sig, _ = S.signal_track(mu, None, assay, None, cov, mode="plug_in", dnase_route="macs2_no_control")
        assert np.allclose(sig, ref.astype(np.float32), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------- native_from_nb (adapter)

class _FakeDS:
    def __init__(self, ctl):
        self.rows = {"C1": {"measurement_id": "C1", "depth_reads": 1e7, "read_length": 76}}
        self._ctl = ctl

    def read(self, mid, chrom, start, nbins):
        assert mid == "C1"
        return self._ctl[chrom][start:start + nbins]


def _chip_row(ctl_id="C1"):
    return {"measurement_id": "T1", "assay": "H3K27ac", "assay_class": "narrow_histone",
            "depth_reads": 2e7, "read_length": 76, "frag_len": 200, "control_measurement_id": ctl_id}


def test_native_from_nb_chip_and_accessibility():
    rng = np.random.default_rng(4)
    rb = 600
    ctl = {"chr1": rng.gamma(2.0, 2.0, 3000) + 0.1}
    regions = [("chr1", 0), ("chr1", 1200)]
    mu = rng.gamma(2.0, 3.0, 2 * rb) + 0.1
    n = np.full_like(mu, 5.0)
    got, src = A.native_from_nb(n, mu, _chip_row(), ds=_FakeDS(ctl), regions=regions, rb=rb)
    cov = {**CHIP_COV, "control_is_predicted": False}
    ref = np.concatenate([A.fold_change_track(mu[:rb], ctl["chr1"][0:rb], cov)[0],
                          A.fold_change_track(mu[rb:], ctl["chr1"][1200:1800], cov)[0]])
    assert src == "observed_control" and np.array_equal(got, ref)
    _, src_pc = A.native_from_nb(n, mu, _chip_row(""), ds=_FakeDS(ctl), regions=regions, rb=rb,
                                 predicted_control=rng.gamma(2.0, 2.0, 2 * rb) + 0.1)
    assert src_pc == "predicted_control"
    arow = {"measurement_id": "A1", "assay": "ATAC-seq", "assay_class": "accessibility_atac",
            "depth_reads": 1e6, "read_length": 50, "control_measurement_id": ""}
    v, s = A.native_from_nb(n, mu, arow, ds=_FakeDS(ctl), regions=regions, rb=rb)
    assert s == "none" and np.allclose(v, A.normalized_insertions(mu, arow))


# ---------------------------------------------------------------- peak probability

def test_peak_probability_matches_simulation():
    rng = np.random.default_rng(5)
    rb = 400
    ctl = {"chr1": np.full(rb, 3.0)}
    mu = np.linspace(5.0, 60.0, rb)
    n = np.full(rb, 3.0)
    p = A.peak_probability_from_nb(n, mu, _chip_row(), ds=_FakeDS(ctl), regions=[("chr1", 0)], rb=rb)
    # simulate: draw counts, score them with the D9b score, count how often -log10 p >= 2
    lam = S.macs2_lambda(lambda_bg=200 * 2e7 / MACS2_GSIZE_HS,
                         control_pileup=S.pileup_from_counts(ctl["chr1"], 200, 76), ratio_treat2control=2.0)
    draws = 20000
    k = nbinom.rvs(n, n / (n + mu), size=(draws, rb), random_state=rng)
    score = log10_upper_tail(np.floor(k * 2.0).astype(np.int64), np.broadcast_to(lam, k.shape).copy())
    freq = (score >= 2.0).mean(axis=0)
    assert np.max(np.abs(p - freq)) < 0.01
    assert 0.05 < p.min() and p.max() < 0.999        # the case is not trivial


def test_peak_probability_limits_and_no_control_route():
    rb = 500
    ctl = {"chr1": np.full(rb, 3.0)}
    n = np.full(rb, 10.0)
    lo = A.peak_probability_from_nb(n, np.full(rb, 1e-3), _chip_row(), ds=_FakeDS(ctl), regions=[("chr1", 0)], rb=rb)
    hi = A.peak_probability_from_nb(n, np.full(rb, 1e4), _chip_row(), ds=_FakeDS(ctl), regions=[("chr1", 0)], rb=rb)
    assert np.all(lo < 1e-12) and np.all(hi == 1.0)     # NB(10, 1e-3) reaching k* ~ 10 is ~1e-39
    arow = {"measurement_id": "A1", "assay": "ATAC-seq", "assay_class": "accessibility_atac",
            "depth_reads": 1e7, "read_length": 50, "control_measurement_id": ""}
    mu = np.concatenate([np.full(250, 0.5), np.full(250, 40.0)])
    pa = A.peak_probability_from_nb(n, mu, arow, ds=_FakeDS(ctl), regions=[("chr1", 0)], rb=rb)
    assert pa.shape == (rb,) and np.all((pa >= 0) & (pa <= 1)) and pa[300] > pa[100]
    drow = {**arow, "assay": "DNase-seq", "assay_class": "accessibility_dnase"}
    with pytest.raises(ValueError):
        A.peak_probability_from_nb(n, mu, drow, ds=_FakeDS(ctl), regions=[("chr1", 0)], rb=rb,
                                   dnase_route="hotspot2_density")


# ---------------------------------------------------------------- isotonic recalibration

def test_isotonic_monotone_and_spearman():
    rng = np.random.default_rng(6)
    pred = rng.normal(size=20_000)
    truth = np.tanh(pred) + 0.3 * rng.normal(size=pred.size)
    cal = A.isotonic_recalibrate(pred, truth)
    assert cal.dtype == np.float64
    o = np.argsort(pred)
    assert np.all(np.diff(cal[o]) >= -1e-12)
    # non-decreasing map: ranks change only through ties that pooled and end bins create
    assert abs(spearmanr(cal, truth)[0] - spearmanr(pred, truth)[0]) < 2e-3
    assert pearsonr(cal, truth)[0] >= pearsonr(pred, truth)[0]


def test_isotonic_strictly_monotone_noiseless_reaches_pearson_1():
    rng = np.random.default_rng(7)
    pred = rng.uniform(-3, 3, size=2000)
    truth = np.exp(pred) ** 1.5
    cal = A.isotonic_recalibrate(pred, truth)       # 2000 bins of 1: exact
    assert pearsonr(cal, truth)[0] > 1 - 1e-12
    assert spearmanr(cal, truth)[0] == pytest.approx(spearmanr(pred, truth)[0], abs=1e-12)


def test_isotonic_refuses_non_finite_in_fit():
    pred = np.arange(10, dtype=float)
    truth = pred * 2.0
    for bad in ("pred", "truth"):
        p, t = pred.copy(), truth.copy()
        (p if bad == "pred" else t)[4] = np.nan
        with pytest.raises(ValueError):
            A.isotonic_recalibrate(p, t, n_bins=5)
    t = truth.copy()
    t[4] = np.inf
    with pytest.raises(ValueError):
        A.isotonic_recalibrate(pred, t, n_bins=5)
    valid = np.ones(10, bool)
    valid[4] = False
    t[4] = np.nan                                       # masked out of the fit: allowed
    assert np.all(np.isfinite(A.isotonic_recalibrate(pred, t, n_bins=9, valid=valid)))


def test_isotonic_valid_restricts_the_fit():
    pred = np.arange(10, dtype=float)
    truth = pred * 2.0
    truth[9] = -100.0
    valid = np.ones(10, bool)
    valid[9] = False
    cal = A.isotonic_recalibrate(pred, truth, n_bins=10, valid=valid)
    assert np.allclose(cal[:9], truth[:9]) and cal[9] == pytest.approx(16.0)
