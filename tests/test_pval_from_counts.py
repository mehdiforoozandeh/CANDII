"""`pval_from_counts.py` — MACS2's no-control Poisson rule, checked against the source it copies.

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
    genome_lambda_bg,
    local_lambda,
    log10_upper_tail,
    pval_from_counts,
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
