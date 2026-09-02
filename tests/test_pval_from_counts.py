"""`pval_from_counts.py` — MACS2's Poisson rule, checked against the source it copies.

Two halves: the no-control rule first, then the with-control rule after the divider.

The three things worth a test are the three things a reimplementation gets wrong: which tail the
p-value is (MACS2 uses `P(X > k)`, strictly greater), which local window a no-control run uses
(llocal only — `--slocal` is "Invalid if there is no control data"), and whether the score
survives past float64 underflow (the corpus reaches `-log10 p` of 17,731, and `sf` flushes to
zero around 1e-308).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from candi.store.pval_from_counts import (
    MACS2_GSIZE_HS,
    MACS2_LLOCAL,
    MACS2_SLOCAL,
    genome_lambda_bg,
    local_lambda,
    log10_upper_tail,
    pval_from_counts,
    pval_from_counts_with_control,
)


def test_the_tail_is_strictly_greater_the_way_macs2_writes_it():
    """`get_pscore = -1*poisson_cdf(observed, expectation, False, True)`, and that upper-tail sum
    starts at `m = k+1`. So the score is `P(X > k)` — using `P(X >= k)` would shift every bin."""
    lam = np.full(6, 4.0)
    k = np.arange(6)
    got = log10_upper_tail(k, lam)
    assert np.allclose(got, -np.log10(poisson.sf(k, lam)))
    assert not np.allclose(got, -np.log10(poisson.sf(k - 1, lam)))


def test_an_empty_bin_still_scores_because_lambda_is_never_zero():
    """`-log10 P(X > 0) = -log10(1 - exp(-lam))`. MACS2's background is not zero, and neither is
    ours — a track whose empty bins read 0 has had that background removed by something else."""
    lam = np.array([0.25, 1.0, 3.0])
    assert np.allclose(log10_upper_tail(np.zeros(3, dtype=int), lam),
                       -np.log10(1.0 - np.exp(-lam)))


def test_the_score_survives_past_float64_underflow():
    """`scipy`'s `sf` is exactly 0 here, so a naive `-log10(sf)` is `inf` and the whole peak is
    lost. Reference values are the regularized lower incomplete gamma, `P(X > k) = P(k+1, lam)`."""
    k = np.array([3000, 10000, 50000])
    lam = np.array([0.3, 1.2, 0.5])
    assert (poisson.sf(k, lam) == 0).all()
    got = log10_upper_tail(k, lam)
    assert np.all(np.isfinite(got))
    # computed with mpmath at 60 dps: -log10(gammainc(k+1, 0, lam, regularized=True))
    assert np.allclose(got, [10703.384607, 34872.083777, 228293.241632], rtol=1e-9)


def test_local_lambda_is_a_centred_mean_over_the_full_window():
    """MACS2 divides by `llocal` whatever the window overlaps, so a bin near a chromosome end gets
    a genuinely smaller local lambda. Renormalising the edge would invent enrichment there."""
    c = np.arange(20, dtype=np.int32)
    got = local_lambda(c, 4)
    want = np.array([c[max(0, i - 2):min(20, i + 2)].sum() / 4.0 for i in range(20)])
    assert np.allclose(got, want)
    assert got[0] < c[:4].mean()          # the edge is not renormalised


def test_the_local_window_is_llocal_and_slocal_is_not_used():
    """`ctrl_d_s = [self.lregion,]` in `__call_peaks_wo_control`, and the line above it says
    'slocal and d-size local bias are not calculated!'. A 1000 bp window would be a different
    method, so the default has to be 10000 and nothing else."""
    assert MACS2_LLOCAL == 10000
    rng = np.random.default_rng(0)
    c = rng.poisson(2.0, 4000).astype(np.int32)
    at_llocal = pval_from_counts(c, lambda_bg=1.0, llocal=10000, resolution=25)
    at_slocal = pval_from_counts(c, lambda_bg=1.0, llocal=1000, resolution=25)
    assert not np.allclose(at_llocal, at_slocal)


def test_lambda_is_the_max_of_the_genome_wide_and_the_local_one():
    """`baseline_value` is documented as 'the minimum pileup', so the max happens inside the
    pileup. A quiet region must score against the genome-wide lambda, not against its own hole."""
    c = np.zeros(2000, dtype=np.int32)
    c[1000] = 40
    hot = pval_from_counts(c, lambda_bg=1e-6, llocal=10000, resolution=25)
    cold = pval_from_counts(c, lambda_bg=50.0, llocal=10000, resolution=25)
    assert hot[1000] > cold[1000]                       # a bigger lambda floor is less surprising
    assert np.allclose(cold[c == 0], -np.log10(1 - np.exp(-50.0)))


def test_lambda_bg_is_total_pileup_mass_over_the_effective_genome():
    """`lambda_bg = d * treat_total / gsize`. In the store's units that is
    `sum(counts) * resolution / gsize`, and `gsize` stays MACS2's EFFECTIVE size, not the
    assembly length — `-g hs` is 2.7e9 against a 3.1e9 assembly, a 15% difference in the floor."""
    assert MACS2_GSIZE_HS == 2.7e9
    assert genome_lambda_bg(1e8, resolution=25) == pytest.approx(1e8 * 25 / 2.7e9)


def test_a_zero_background_is_refused_rather_than_producing_infinities():
    """MACS2 asserts `lam > 0`. Poisson with lambda 0 makes every occupied bin infinitely
    significant, which is a silent all-peaks track rather than an error."""
    with pytest.raises(ValueError, match="lambda_bg"):
        pval_from_counts(np.zeros(10, dtype=np.int32), lambda_bg=0.0)


# --- the with-control branch -------------------------------------------------------------------
#
# What separates it from the no-control rule is not the tail but the lambda: with a control, the
# local lambda is read off the CONTROL and is a max over three windows (d, slocal, llocal) instead
# of one, each scaled by the treatment/control depth ratio. Every test below aims at that.


def test_the_with_control_lambda_is_the_hand_computed_max_of_four_terms():
    """`ctrl_d_s = [d, sregion, lregion]` with scales `[r, r*d/sregion, r*d/lregion]`, and
    `pileup_a_chromosome` maxes them together over a `lambda_bg` floor. Worked by hand on eight
    bins so the expected lambda is written out rather than recomputed by the code under test."""
    counts = np.array([0, 0, 5, 0, 0, 0, 0, 0], dtype=np.int32)
    control = np.array([1, 0, 0, 0, 0, 0, 0, 2], dtype=np.int32)
    # r * control                     = [2, 0, 0, 0, 0, 0, 0, 4]
    # r * 2-bin centred window mean   = [1, 1, 0, 0, 0, 0, 0, 2]
    # r * 4-bin centred window mean   = [.5,.5,.5, 0, 0, 0, 1, 1]
    # floor                           = 0.1
    want_lam = np.array([2.0, 1.0, 0.5, 0.1, 0.1, 0.1, 1.0, 4.0])
    got = pval_from_counts_with_control(
        counts, control, lambda_bg=0.1, ratio_treat2control=2.0,
        slocal=50, llocal=100, resolution=25,
    )
    assert np.allclose(got, -np.log10(poisson.sf(counts, want_lam)), rtol=1e-6)


def test_slocal_comes_back_when_a_control_is_present():
    """The no-control path says 'slocal and d-size local bias are not calculated!'. The
    with-control path adds `sregion` whenever `--slocal` is non-zero, and 1000 is its default,
    so switching it off has to change the answer."""
    assert MACS2_SLOCAL == 1000
    rng = np.random.default_rng(7)
    ctrl = rng.poisson(1.5, 4000).astype(np.int32)
    ctrl[1900:2100] += 20                        # a control bump narrower than llocal
    counts = rng.poisson(1.5, 4000).astype(np.int32)
    kw = dict(lambda_bg=0.5, ratio_treat2control=1.0, resolution=25)
    with_slocal = pval_from_counts_with_control(counts, ctrl, slocal=1000, llocal=10000, **kw)
    without = pval_from_counts_with_control(counts, ctrl, slocal=0, llocal=10000, **kw)
    assert not np.allclose(with_slocal, without)
    # slocal only ever ADDS a term to a max, so it can only lower the score
    assert np.all(with_slocal <= without + 1e-6)


def test_the_d_size_window_is_the_control_pileup_itself():
    """`ctrl_d_s` always starts at `[ self.d ]`, so a one-bin control spike raises the lambda at
    that bin alone. A rule that only had slocal and llocal would smear it over 40 bins."""
    counts = np.full(400, 3, dtype=np.int32)
    ctrl = np.zeros(400, dtype=np.int32)
    ctrl[200] = 500
    got = pval_from_counts_with_control(
        counts, ctrl, lambda_bg=0.5, ratio_treat2control=1.0,
        slocal=1000, llocal=10000, resolution=25,
    )
    assert got[200] == pytest.approx(-np.log10(poisson.sf(3, 500.0)), rel=1e-5)
    assert got[199] > got[200] and got[201] > got[200]


def test_the_treatment_neighbourhood_stops_mattering_once_there_is_a_control():
    """Every entry of `ctrl_d_s` is piled up from `self.ctrl`. So two treatments that agree bin by
    bin but differ everywhere else must score identically at the bins they share, which is exactly
    what the no-control rule cannot do."""
    ctrl = np.full(2000, 2, dtype=np.int32)
    a = np.zeros(2000, dtype=np.int32)
    a[1000] = 30
    b = a.copy()
    b[1100:1300] = 25                            # a big treatment feature inside llocal of bin 1000
    kw = dict(lambda_bg=0.4, ratio_treat2control=1.0, resolution=25)
    pa = pval_from_counts_with_control(a, ctrl, **kw)
    pb = pval_from_counts_with_control(b, ctrl, **kw)
    assert pa[1000] == pb[1000]
    # the no-control rule is the contrast: there the treatment's own neighbours move the score
    na = pval_from_counts(a, lambda_bg=0.4)
    nb = pval_from_counts(b, lambda_bg=0.4)
    assert na[1000] != nb[1000]


def test_the_depth_ratio_scales_the_local_terms_and_not_the_floor():
    """`ctrl_scale_s` carries `ratio_treat2control` on every entry; `lambda_bg` does not. Doubling
    the ratio therefore doubles the local lambda wherever it is already above the floor."""
    ctrl = np.full(600, 4, dtype=np.int32)
    counts = np.full(600, 10, dtype=np.int32)
    kw = dict(lambda_bg=1e-6, slocal=1000, llocal=10000, resolution=25)
    one = pval_from_counts_with_control(counts, ctrl, ratio_treat2control=1.0, **kw)
    two = pval_from_counts_with_control(counts, ctrl, ratio_treat2control=2.0, **kw)
    assert np.allclose(one, -np.log10(poisson.sf(10, 4.0)), rtol=1e-5)
    assert np.allclose(two, -np.log10(poisson.sf(10, 8.0)), rtol=1e-5)


def test_llocal_is_ignored_when_it_is_not_larger_than_slocal():
    """`if self.lregion and self.lregion > self.sregion:` — a `--llocal` equal to `--slocal` adds
    no second window, and the guard is what stops the same window being counted twice."""
    rng = np.random.default_rng(3)
    ctrl = rng.poisson(2.0, 3000).astype(np.int32)
    counts = rng.poisson(2.0, 3000).astype(np.int32)
    kw = dict(lambda_bg=0.3, ratio_treat2control=1.0, resolution=25)
    equal = pval_from_counts_with_control(counts, ctrl, slocal=1000, llocal=1000, **kw)
    dropped = pval_from_counts_with_control(counts, ctrl, slocal=1000, llocal=0, **kw)
    assert np.array_equal(equal, dropped)


def test_the_with_control_tail_is_strictly_greater_too():
    """`get_pscore` is the same function on both paths — `P(X > k)`, not `P(X >= k)`."""
    counts = np.arange(6, dtype=np.int32)
    ctrl = np.full(6, 4, dtype=np.int32)
    got = pval_from_counts_with_control(
        counts, ctrl, lambda_bg=1e-9, ratio_treat2control=1.0, slocal=0, llocal=0, resolution=25,
    )
    assert np.allclose(got, -np.log10(poisson.sf(counts, 4.0)), rtol=1e-6)


def test_the_with_control_inputs_that_would_be_silently_wrong_are_refused():
    """A depth ratio the caller forgot to compute rescales every lambda on the chromosome, and a
    control binned differently from the treatment lines the two up off by a bin. Both are loud."""
    c = np.zeros(10, dtype=np.int32)
    with pytest.raises(ValueError, match="lambda_bg"):
        pval_from_counts_with_control(c, c, lambda_bg=0.0, ratio_treat2control=1.0)
    with pytest.raises(ValueError, match="ratio_treat2control"):
        pval_from_counts_with_control(c, c, lambda_bg=1.0, ratio_treat2control=0.0)
    with pytest.raises(ValueError, match="binned the same"):
        pval_from_counts_with_control(c, np.zeros(11, dtype=np.int32),
                                      lambda_bg=1.0, ratio_treat2control=1.0)
    with pytest.raises(ValueError, match="llocal can't be smaller"):
        pval_from_counts_with_control(c, c, lambda_bg=1.0, ratio_treat2control=1.0,
                                      slocal=10000, llocal=1000)
    with pytest.raises(ValueError, match="under one 25 bp bin"):
        pval_from_counts_with_control(c, c, lambda_bg=1.0, ratio_treat2control=1.0,
                                      slocal=10, llocal=10000)
