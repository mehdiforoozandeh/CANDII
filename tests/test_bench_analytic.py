"""Layer 2 — every metric against inputs whose answer is known before the code runs.

`EVAL_PLAN.md` §6. Layer 1 proves our E-block equals the organizers' E-block; it cannot prove either
is right, because two implementations of the same mistake agree perfectly. This layer is the check
that the *definitions* are what we think: inputs constructed so the answer is derivable on paper, and
asserted against that derivation rather than against a recorded output.

Every assertion below carries the derivation in its docstring. A test whose expected value is "what
it printed last time" is a change detector, not a verification, and none of those are here.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom

from candi.bench import binary as B
from candi.bench import distributional as D
from candi.bench import eic
from candi.bench import partitions as P

EXACT = 1e-12


# ===========================================================================
# E-block
# ===========================================================================

def test_mse_of_a_constant_offset_is_the_offset_squared() -> None:
    """ŷ = y + c  ⇒  mean((y − ŷ)²) = mean(c²) = c², for any y."""
    rng = np.random.default_rng(0)
    y = rng.random(1000) * 17.0
    for c in (0.0, 1.0, -2.5, 0.125):
        assert eic.mse(y, y + c) == pytest.approx(c * c, abs=EXACT)


def test_pearson_of_a_positive_affine_map_is_exactly_one() -> None:
    """ŷ = a·y + b with a > 0 is a perfect linear relationship, so r = 1 regardless of a and b."""
    rng = np.random.default_rng(1)
    y = rng.random(500) * 3.0
    for a, b in ((1.0, 0.0), (7.5, -2.0), (0.001, 100.0)):
        assert eic.gwcorr(y, a * y + b) == pytest.approx(1.0, abs=1e-12)
    assert eic.gwcorr(y, -2.0 * y + 1.0) == pytest.approx(-1.0, abs=1e-12)


def test_spearman_of_any_strictly_increasing_map_is_exactly_one() -> None:
    """Spearman reads ranks only, so it is 1 for exp, cube, and log alike — Pearson is not."""
    rng = np.random.default_rng(2)
    y = rng.random(500) * 4.0 + 0.1
    for f in (np.exp, lambda v: v ** 3, np.log, np.sqrt):
        assert eic.gwspear(y, f(y)) == pytest.approx(1.0, abs=1e-12)
    assert eic.gwcorr(y, np.exp(y)) < 0.999      # the contrast that makes the pair non-redundant


def test_msevar_with_unit_variance_is_exactly_mse() -> None:
    """var ≡ 1 ⇒ Σ(y−ŷ)²·1 / Σ1 = Σ(y−ŷ)²/N = mse. The weighting reduces to the unweighted mean."""
    rng = np.random.default_rng(3)
    y, p = rng.random(400) * 6, rng.random(400) * 6
    assert eic.msevar(y, p, np.ones(400)) == pytest.approx(eic.mse(y, p), abs=EXACT)


def test_msevar_with_a_one_hot_variance_is_that_positions_squared_error() -> None:
    """var = e_k ⇒ the weighted mean collapses to the single position k."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 5.0, 3.0, 0.0])
    var = np.array([0.0, 1.0, 0.0, 0.0])
    assert eic.msevar(y, p, var) == pytest.approx(9.0, abs=EXACT)     # (2−5)² = 9


def test_mse1obs_selects_exactly_the_top_ten_of_a_thousand() -> None:
    """N = 1000 ⇒ int(N·0.01) = 10. y = 0..999 has no ties, so the top 10 are y ∈ {990..999}.

    Error is 1 at those positions and 0 elsewhere, so mse1obs = 10·1/10 = 1 while mse = 10/1000.
    """
    y = np.arange(1000, dtype=float)
    p = y.copy()
    p[990:] += 1.0
    assert eic.mse1obs(y, p) == pytest.approx(1.0, abs=EXACT)
    assert eic.mse(y, p) == pytest.approx(0.01, abs=EXACT)


def test_mse1imp_ranks_by_the_prediction_not_the_truth() -> None:
    """The two top-1% measures disagree exactly when the model puts its mass in the wrong place.

    Truth peaks at the low indices; the prediction peaks at the high ones. mse1obs looks where the
    signal is, mse1imp looks where the model said it was, and neither sees the other's positions.
    """
    n = 1000
    y = np.zeros(n)
    y[:10] = 100.0                     # true signal, low indices
    p = np.zeros(n)
    p[-10:] = 100.0                    # predicted signal, high indices
    assert eic.mse1obs(y, p) == pytest.approx(100.0 ** 2, abs=1e-6)   # missed every true peak
    assert eic.mse1imp(y, p) == pytest.approx(100.0 ** 2, abs=1e-6)   # invented every called peak


def test_mse1obs_on_fewer_than_100_positions_scores_the_whole_array() -> None:
    """Quirk 4, pinned. int(99·0.01) = 0, and np.sort(y)[-0] is y_sorted[0] — the MINIMUM.

    So `y >= min` selects everything and mse1obs degenerates to mse. This is upstream behaviour and
    layer 1 confirms we match it; the test is here so that a future 'fix' has to argue with a
    docstring rather than a silent diff.
    """
    rng = np.random.default_rng(4)
    y, p = rng.random(99) * 5, rng.random(99) * 5
    assert eic.mse1obs(y, p) == pytest.approx(eic.mse(y, p), abs=EXACT)
    y100, p100 = rng.random(100) * 5, rng.random(100) * 5
    assert eic.mse1obs(y100, p100) != pytest.approx(eic.mse(y100, p100), abs=1e-6)


def test_a_duplicated_gene_leaves_msegene_unchanged() -> None:
    """Quirk 1: overlaps re-weight, they do not bias. Duplicating a line doubles sse AND n."""
    rng = np.random.default_rng(5)
    y, p = rng.random(4000) * 3, rng.random(4000) * 3
    td, pd_ = {"chrT": y}, {"chrT": p}
    one = ["chrT\t1000\t9000\tG0\t.\t+\n"]
    assert eic.msegene(td, pd_, ["chrT"], one) == pytest.approx(
        eic.msegene(td, pd_, ["chrT"], one * 2), abs=EXACT)
    assert eic.msegene(td, pd_, ["chrT"], one) == pytest.approx(
        eic.msegene(td, pd_, ["chrT"], one * 7), abs=EXACT)


def test_two_overlapping_genes_reweight_the_shared_region() -> None:
    """The overlap is counted twice; the answer is the n-weighted mean of the two gene bodies."""
    y = np.zeros(400)
    p = np.zeros(400)
    p[0:100] = 1.0                                     # squared error 1 on bins 0..99
    gene_a = "chrT\t0\t2475\tA\t.\t+\n"                # bins 0..99   (2475//25 + 1 = 100)
    gene_b = "chrT\t0\t4975\tB\t.\t+\n"                # bins 0..199
    td, pd_ = {"chrT": y}, {"chrT": p}
    # A alone: sse 100, n 100 -> 1.0.  B alone: sse 100, n 200 -> 0.5.
    assert eic.msegene(td, pd_, ["chrT"], [gene_a]) == pytest.approx(1.0, abs=EXACT)
    assert eic.msegene(td, pd_, ["chrT"], [gene_b]) == pytest.approx(0.5, abs=EXACT)
    # Together: sse 200, n 300 -> 2/3. Not the mean of 1.0 and 0.5.
    assert eic.msegene(td, pd_, ["chrT"], [gene_a, gene_b]) == pytest.approx(2 / 3, abs=EXACT)


def test_a_promoter_running_off_the_front_contributes_nothing() -> None:
    """Quirk 2. A + -strand gene starting in bin 0 has its promoter at [−80:0], which is empty.

    So the gene is silently absent from mseprom. Two genes, one of them off the front, must give
    exactly the value of the one that is not.
    """
    rng = np.random.default_rng(6)
    y, p = rng.random(4000) * 3, rng.random(4000) * 3
    td, pd_ = {"chrT": y}, {"chrT": p}
    off_front = "chrT\t0\t500\tOFF\t.\t+\n"            # start bin 0 -> promoter [−80:0] -> empty
    normal = "chrT\t50000\t60000\tOK\t.\t+\n"          # start bin 2000 -> promoter [1920:2000]
    assert eic.mseprom(td, pd_, ["chrT"], [normal]) == pytest.approx(
        eic.mseprom(td, pd_, ["chrT"], [off_front, normal]), abs=EXACT)


def test_msegene_counts_bins_past_the_end_of_the_array_but_scores_none_of_them() -> None:
    """Quirk 7, the asymmetry layer 1 caught. `msegene` counts `end − start` UNCLIPPED.

    A gene spanning bins 0..399 on a 200-bin array: sse comes from bins 0..199 only, but n is 400.
    With squared error exactly 1 everywhere, the answer is 200/400 = 0.5 — not 1.0. `mseprom` does
    the opposite and counts the clipped length, so the same construction there gives 1.0.
    """
    y = np.zeros(200)
    p = np.ones(200)
    td, pd_ = {"chrT": y}, {"chrT": p}
    gene = "chrT\t0\t9975\tG\t.\t+\n"                  # 9975//25 + 1 = 400 bins claimed
    assert eic.msegene(td, pd_, ["chrT"], [gene]) == pytest.approx(0.5, abs=EXACT)

    # The mseprom side of the asymmetry, plus quirk 8: an annotation that selects zero bins leaves
    # n == 0, and `sse / n` is np.float64(0)/0 -> nan (not a raise, and not 0.0). The matching
    # ZeroDivisionError case -- no annotation for the chromosome at all -- is in
    # test_bench_reference.py, asserted against the reference on both branches.
    minus = "chrT\t0\t4975\tG\t.\t-\n"                 # end bin 200 -> promoter [200:280] -> empty
    with np.errstate(invalid="ignore"):
        assert np.isnan(eic.mseprom(td, pd_, ["chrT"], [minus]))


def test_a_perfect_prediction_scores_zero_error_and_unit_correlation() -> None:
    """The trivial fixed point, asserted because a sign error hides everywhere else."""
    rng = np.random.default_rng(8)
    y = rng.random(2000) * 9
    # gene start at 5000 bp = bin 200, so the 80-bin promoter [120:200] is fully inside the array.
    # A start below bin 80 would make the promoter empty and `sse / n` a ZeroDivisionError (quirk 8).
    genes = ["chrT\t5000\t9000\tG\t.\t+\n"]
    enh = ["chrT\t2000\t3000\tE\t0\t.\t2000\t3000\t0,0,0\t1\t1000\t0\n"]
    s = eic.score_track({"chrT": y}, {"chrT": y.copy()}, ["chrT"],
                        gene_annotations=genes, enh_annotations=enh, var=np.ones(2000))
    for k in ("mse", "mseprom", "msegene", "mseenh", "mse1obs", "mse1imp", "msevar"):
        assert s[k] == pytest.approx(0.0, abs=EXACT), k
    assert s["gwcorr"] == pytest.approx(1.0, abs=1e-12)
    assert s["gwspear"] == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# D-block
# ===========================================================================

def test_gauss_crps_converges_to_absolute_error_as_sigma_vanishes() -> None:
    """A degenerate forecast is a point estimate, and CRPS of a point estimate is |y − µ|."""
    y = np.array([3.0, -2.0, 0.0])
    mu = np.array([0.0, 0.0, 0.0])
    for sigma in (1e-6, 1e-9, 1e-12):
        got = D.gauss_crps(mu, np.full(3, sigma), y)
        assert np.allclose(got, np.abs(y - mu), atol=1e-5)


def test_gauss_crps_at_the_mean_has_the_known_closed_form() -> None:
    """At y = µ, z = 0 and CRPS = σ·(0 + 2φ(0) − 1/√π) = σ·(2/√(2π) − 1/√π) = σ·(√2 − 1)/√π."""
    expected = (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi)
    for sigma in (0.5, 1.0, 4.0):
        got = float(D.gauss_crps(np.array([0.0]), np.array([sigma]), np.array([0.0]))[0])
        assert got == pytest.approx(sigma * expected, rel=1e-12)


def test_gauss_crps_is_minimised_by_the_true_sigma() -> None:
    """Propriety, empirically: for y ~ N(0, σ²), expected CRPS is lowest at the forecast σ."""
    rng = np.random.default_rng(9)
    y = rng.normal(0.0, 2.0, 200_000)
    scores = {s: D.gauss_crps_mean(np.zeros_like(y), np.full_like(y, s), y)
              for s in (0.5, 1.0, 2.0, 4.0, 8.0)}
    assert min(scores, key=scores.get) == 2.0


def test_pit_is_uniform_and_ece_vanishes_when_the_forecast_is_the_truth() -> None:
    """The calibration fixed point: draw y FROM the predicted NB and the PIT curve must be y = x.

    Asserted at two sample sizes an order apart, because the claim is that the error shrinks like
    1/√n — a constant bias would pass at one n and fail this.
    """
    rng = np.random.default_rng(10)
    errs = {}
    for n_pts in (20_000, 200_000):
        size = np.full(n_pts, 4.0)
        mu = np.full(n_pts, 6.0)
        p = D.p_from_mu(size, mu)
        y = nbinom.rvs(size, p, random_state=rng)
        errs[n_pts] = D.ece(size, p, y)
    assert errs[200_000] < errs[20_000]
    assert errs[200_000] < 0.005
    assert errs[20_000] < 0.02


def test_ece_is_large_when_the_forecast_is_deliberately_wrong() -> None:
    """The other half: a metric that returns ~0 for everything would pass the test above."""
    rng = np.random.default_rng(11)
    n_pts = 50_000
    size = np.full(n_pts, 4.0)
    y = nbinom.rvs(size, D.p_from_mu(size, np.full(n_pts, 6.0)), random_state=rng)
    wrong = D.p_from_mu(size, np.full(n_pts, 60.0))      # forecast 10x too high
    assert D.ece(size, wrong, y) > 0.3


def test_coverage_of_a_correct_forecast_approaches_the_nominal_level() -> None:
    """Discreteness makes NB coverage over-cover slightly; that is expected, and bounded."""
    rng = np.random.default_rng(12)
    size = np.full(100_000, 8.0)
    mu = np.full(100_000, 40.0)                  # large enough that discreteness is mild
    y = nbinom.rvs(size, D.p_from_mu(size, mu), random_state=rng)
    cov = D.coverage_nb(size, mu, y, level=0.95)
    assert 0.94 <= cov <= 0.97


def test_gaussian_coverage_of_a_correct_forecast_is_the_nominal_level() -> None:
    """No discreteness here, so this one should land on 0.95 rather than merely near it."""
    rng = np.random.default_rng(13)
    y = rng.normal(1.5, 3.0, 200_000)
    cov = D.coverage_gauss(np.full(200_000, 1.5), np.full(200_000, 3.0), y, level=0.95)
    assert cov == pytest.approx(0.95, abs=0.003)


def test_c_index_is_one_for_a_perfect_ranker_and_zero_for_a_reversed_one() -> None:
    """Concordance endpoints. σ small enough that the ordering of µ decides every pair."""
    y = np.arange(500, dtype=float)
    sigma = np.full(500, 1e-6)
    perfect = D.c_index_gauss(y, sigma, y, n_pairs=20_000, seed=0)
    reversed_ = D.c_index_gauss(-y, sigma, y, n_pairs=20_000, seed=0)
    assert perfect["c_index"] == pytest.approx(1.0, abs=1e-9)
    assert reversed_["c_index"] == pytest.approx(0.0, abs=1e-9)
    assert perfect["c_index_exact_per_pair"] is True


def test_c_index_is_one_half_when_the_forecast_carries_no_ordering() -> None:
    """Identical predictions everywhere ⇒ P(X_i > X_j) = 0.5 for every pair, exactly."""
    y = np.arange(500, dtype=float)
    flat = D.c_index_gauss(np.zeros(500), np.ones(500), y, n_pairs=20_000, seed=0)
    assert flat["c_index"] == pytest.approx(0.5, abs=1e-12)
    assert flat["c_index_se"] == pytest.approx(0.0, abs=1e-12)


def test_the_c_index_always_reports_its_sampling_error(caplog) -> None:
    """D3: the one sampled metric in the suite, and its SE is not optional.

    The SE must also behave like an SE — shrinking as 1/√n_pairs. Sixteen times the pairs should
    roughly halve it; asserted loosely because the pair draw is random.
    """
    rng = np.random.default_rng(14)
    y = rng.random(2000)
    small = D.c_index_gauss(y, np.ones(2000), y, n_pairs=2_000, seed=1)
    large = D.c_index_gauss(y, np.ones(2000), y, n_pairs=32_000, seed=1)
    for r in (small, large):
        assert set(("c_index", "c_index_se", "c_index_n_pairs", "c_index_n_drawn")) <= set(r)
        assert np.isfinite(r["c_index_se"])
    ratio = small["c_index_se"] / large["c_index_se"]
    assert 2.5 < ratio < 5.5, f"SE shrank by {ratio}x for 16x the pairs; expected ~4x"


def test_nb_c_index_agrees_with_the_exact_gaussian_one_in_the_regime_they_share() -> None:
    """A cross-check between two independent estimators, which is stronger than either alone.

    At large µ the NB is close to Gaussian, so the sampled NB concordance must land on the exact
    Gaussian value computed from the matching moments. It agrees to within the NB estimator's own
    reported standard error, which also validates that the SE is honest.
    """
    rng = np.random.default_rng(15)
    n_pos = 4000
    size = np.full(n_pos, 50.0)
    mu = rng.random(n_pos) * 200.0 + 50.0
    y = nbinom.rvs(size, D.p_from_mu(size, mu), random_state=rng).astype(float)

    nb = D.c_index_nb(size, mu, y, n_pairs=200_000, seed=2)
    sigma = np.sqrt(mu + mu ** 2 / size)                    # NB variance, matched moments
    gs = D.c_index_gauss(mu, sigma, y, n_pairs=200_000, seed=2)
    assert abs(nb["c_index"] - gs["c_index"]) < 6.0 * nb["c_index_se"]
    assert nb["c_index_exact_per_pair"] is False


def test_more_draws_per_pair_does_not_move_the_nb_c_index() -> None:
    """The claim behind using one draw per pair: per-pair noise averages out, so 1 ≈ 64.

    If this fails, the estimator is biased rather than merely noisy, and the docstring in
    `distributional.c_index_nb` is wrong.
    """
    rng = np.random.default_rng(16)
    size = np.full(2000, 6.0)
    mu = rng.random(2000) * 30 + 1
    y = nbinom.rvs(size, D.p_from_mu(size, mu), random_state=rng).astype(float)
    one = D.c_index_nb(size, mu, y, n_pairs=100_000, draws=1, seed=3)
    many = D.c_index_nb(size, mu, y, n_pairs=100_000, draws=64, seed=3)
    assert abs(one["c_index"] - many["c_index"]) < 4.0 * one["c_index_se"]


def test_the_oracle_rescale_recovers_a_planted_scale_error() -> None:
    """Plant a known 4x level error; the oracle must find c* = −2 and drive scale_error positive.

    This is what makes the split trustworthy: `crps_oracle_scaled` on the miscalibrated forecast must
    return to the CRPS of the correct one, because a single constant is all that was wrong.
    """
    rng = np.random.default_rng(17)
    n_pos = 30_000
    size = np.full(n_pos, 5.0)
    mu = np.full(n_pos, 10.0)
    y = nbinom.rvs(size, D.p_from_mu(size, mu), random_state=rng).astype(float)

    truth_crps = D.nb_crps_mean(size, mu, y)
    orc = D.oracle_scale(mu * 4.0, size, y, seed=0)
    assert orc["c_star"] == pytest.approx(-2.0, abs=0.05)          # 2^-2 = 1/4
    assert orc["crps_oracle_scaled"] == pytest.approx(truth_crps, rel=0.02)

    inflated = D.nb_crps_mean(size, mu * 4.0, y)
    assert inflated - orc["crps_oracle_scaled"] > 0.5 * truth_crps  # a large, fixable scale error


def test_the_marginal_bar_beats_the_legacy_median_baseline_it_replaced() -> None:
    """The honest bar must dominate the median rule by construction — it is entered as a candidate.

    And on an all-zero-median target, the legacy rule's degeneracy must be visible: it collapses to a
    near point mass, which is the failure that manufactured 'beats marginal 8/8'.
    """
    rng = np.random.default_rng(18)
    y = rng.negative_binomial(2, 0.5, 20_000).astype(float)
    m = D.marginal_nb(y)
    assert m["marg_crps"] <= m["marg_crps_legacy_median"] + 1e-12

    sparse = (rng.random(20_000) < 0.1) * rng.integers(1, 50, 20_000)
    ms = D.marginal_nb(sparse.astype(float))
    assert float(np.median(sparse)) == 0.0
    assert ms["marg_mu_legacy_median"] < 1e-3           # the point mass at zero
    assert ms["marg_crps"] < ms["marg_crps_legacy_median"]


def test_a_model_that_forecasts_the_truth_beats_the_marginal_bar() -> None:
    """A sanity direction: the bar must be beatable, or it is not a bar."""
    rng = np.random.default_rng(19)
    n_pos = 20_000
    size = np.full(n_pos, 3.0)
    mu = rng.random(n_pos) * 40 + 1.0                   # position-varying truth
    y = nbinom.rvs(size, D.p_from_mu(size, mu), random_state=rng).astype(float)
    assert D.nb_crps_mean(size, mu, y) < D.marginal_nb(y)["marg_crps"]


# ===========================================================================
# B-block
# ===========================================================================

def test_average_precision_endpoints_are_one_and_the_base_rate() -> None:
    """A perfect ranker gives 1.0; a ranker carrying no information gives the prevalence."""
    rng = np.random.default_rng(20)
    y = (rng.random(2000) < 0.05).astype(int)
    perfect = B.average_precision(y, y.astype(float))
    assert perfect == pytest.approx(1.0, abs=1e-12)

    tied = B.average_precision(y, np.ones(2000))        # every score equal: one threshold
    assert tied == pytest.approx(y.mean(), abs=1e-12)

    noise = np.mean([B.average_precision(y, rng.random(2000)) for _ in range(30)])
    assert abs(noise - y.mean()) < 0.02


def test_average_precision_is_hand_computable_on_a_four_point_case() -> None:
    """Scores 0.9, 0.8, 0.7, 0.6 with labels 1, 0, 1, 0.

    k=1: P=1/1, R=1/2 -> ΔR=1/2, term 1/2.  k=3: P=2/3, R=1 -> ΔR=1/2, term 1/3.
    AP = 1/2 + 1/3 = 5/6.
    """
    y = np.array([1, 0, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6])
    assert B.average_precision(y, s) == pytest.approx(5 / 6, abs=1e-12)


def test_average_precision_is_undefined_rather_than_zero_with_no_positives() -> None:
    """Precision has no meaning when nothing is positive; nan says so and 0.0 would lie."""
    assert np.isnan(B.average_precision(np.zeros(100, dtype=int), np.random.random(100)))


def test_peak_overlap_endpoints_and_a_known_partial_overlap() -> None:
    """Identical rankings overlap fully. Disjoint top sets overlap not at all."""
    y = np.arange(1000, dtype=float)
    assert B.peak_overlap(y, y, p=0.01) == pytest.approx(1.0, abs=EXACT)
    assert B.peak_overlap(y, -y, p=0.01) == pytest.approx(0.0, abs=EXACT)
    assert B.peak_overlap(y, y, p=0.0) == 0.0
    assert B.peak_overlap(y, y, p=1.0) == 1.0

    # top 10 of y is {990..999}; make the prediction rank {995..1004 mod n} highest -> 5 shared
    p = np.zeros(1000)
    p[995:] = 10.0                                   # 5 positions
    p[0:5] = 9.0                                     # 5 positions, not in the true top 10
    assert B.peak_overlap(y, p, p=0.01) == pytest.approx(0.5, abs=EXACT)


def test_the_correspondence_curve_of_a_perfect_ranker_is_the_identity() -> None:
    """Under a perfect ranking the top-p sets coincide, so overlap/N = p and the derivative is 1."""
    y = np.arange(1000, dtype=float)
    curve, deriv = B.correspondence_curve(y, y)
    for p, v in curve:
        assert v == pytest.approx(p, abs=0.001)
    assert np.allclose([d for _, d in deriv], 1.0, atol=0.02)


# ===========================================================================
# P-block
# ===========================================================================

def test_the_strength_grid_is_the_papers_grid() -> None:
    """'logarithmic bins of size 0.1 from 10⁻¹ to 10^2.5' — 35 bins, 36 edges."""
    e = P.strength_bin_edges()
    assert len(e) == 36
    assert e[0] == pytest.approx(0.1, rel=1e-12)
    assert e[-1] == pytest.approx(10 ** 2.5, rel=1e-12)
    assert np.allclose(np.diff(np.log10(e)), 0.1)


def test_accuracy_by_strength_is_one_everywhere_for_a_consistent_prediction() -> None:
    """If the prediction crosses the threshold exactly where the truth is a peak, every bin is 1.0."""
    rng = np.random.default_rng(21)
    sig = 10 ** (rng.random(20_000) * 3.5 - 1.0)
    peaks = sig >= P.BINARISE_THRESHOLD
    out = P.accuracy_by_strength(sig, peaks, sig, bin_by="obs")
    assert out["macro_accuracy"] == pytest.approx(1.0, abs=EXACT)
    assert out["pooled_accuracy"] == pytest.approx(1.0, abs=EXACT)
    assert out["n_occupied_bins"] > 30


def test_macro_accuracy_differs_from_pooled_accuracy_and_that_is_the_point() -> None:
    """The block exists because per-locus weighting is dominated by the crowded low bins.

    A model right on the sparse high-signal bins and wrong on the crowded low ones must score better
    under the macro average than under the pooled one, or the partitioning is doing nothing.
    """
    n = 100_000
    sig = np.r_[np.full(n - 200, 0.2), np.full(200, 100.0)]       # 99.8% low, 0.2% high
    peaks = sig >= P.BINARISE_THRESHOLD
    pred = sig.copy()
    pred[: n - 200] = 100.0                                       # wrong on every low position
    out = P.accuracy_by_strength(sig, peaks, pred, bin_by="obs")
    assert out["pooled_accuracy"] < 0.01
    assert out["macro_accuracy"] > 0.4


def test_specificity_score_counts_cell_types_with_a_peak() -> None:
    """A column sum over the cell-type x locus binary matrix. Nothing subtler than that."""
    m = np.array([[1, 0, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0]])
    assert list(P.specificity_scores(m)) == [2, 1, 3, 0]


def test_precision_recall_by_specificity_separates_a_baseline_from_a_real_model() -> None:
    """The measure's purpose, made into a test.

    An 'average activity' predictor calls a peak exactly where most cell types have one. It must
    score high recall in the high-specificity group and near-zero recall in the low ones — while a
    predictor that copies the truth scores 1.0 everywhere.
    """
    rng = np.random.default_rng(22)
    n_cells, n_loci = 10, 5000
    spec_true = rng.integers(0, n_cells + 1, n_loci)
    matrix = np.array([[1 if c < spec_true[i] else 0 for i in range(n_loci)]
                       for c in range(n_cells)])
    spec = P.specificity_scores(matrix)
    truth = matrix[0].astype(bool)                       # cell 0 peaks wherever spec >= 1

    avg_activity = spec >= (n_cells // 2)                # ignores the cell, calls the common sites
    pr = P.precision_recall_by_specificity(spec, truth, avg_activity)
    by_spec = dict(zip(pr["specificity"], pr["recall"]))
    assert by_spec[1] == pytest.approx(0.0, abs=1e-12)   # misses every cell-type-specific site
    assert by_spec[n_cells] == pytest.approx(1.0, abs=1e-12)

    perfect = P.precision_recall_by_specificity(spec, truth, truth)
    assert perfect["macro_recall"] == pytest.approx(1.0, abs=1e-12)
    assert perfect["macro_precision"] == pytest.approx(1.0, abs=1e-12)


def test_region_correlation_averages_within_regions_not_across_them() -> None:
    """The construction that separates shape from height, and the reason to average per region.

    Two regions with identical internal profiles but different heights. Pooled, a prediction that
    swaps the heights correlates poorly. Within each region the profile is perfect, so the per-region
    mean is 1.0 — and only the per-region number is a statement about shape.
    """
    profile = np.array([1.0, 3.0, 8.0, 3.0, 1.0])
    y = np.r_[profile, profile * 10.0]
    p = np.r_[profile * 10.0, profile]                   # heights swapped, shapes intact
    regions = [(0, 5), (5, 10)]
    assert P.region_correlation(y, p, regions)["mean_corr"] == pytest.approx(1.0, abs=1e-12)
    from candi.metrics import pearson as _pearson
    assert _pearson(y, p) < 0.5                          # pooled, it looks bad


def test_region_correlation_excludes_undefined_regions_rather_than_scoring_them_zero() -> None:
    """A flat region has no shape to get right; nan is the honest answer and 0.0 would be a claim."""
    y = np.r_[np.ones(5), np.array([1.0, 2.0, 3.0, 2.0, 1.0])]
    p = np.r_[np.ones(5), np.array([1.0, 2.0, 3.0, 2.0, 1.0])]
    out = P.region_correlation(y, p, [(0, 5), (5, 10)])
    assert out["n_regions"] == 2
    assert out["n_scored"] == 1
    assert out["n_undefined"] == 1
    assert out["mean_corr"] == pytest.approx(1.0, abs=1e-12)


def test_peak_regions_finds_runs_and_drops_the_shapeless_ones() -> None:
    """A single-bin peak has no profile; Pearson over one point is undefined, so it is excluded."""
    b = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1, 0])
    assert P.peak_regions(b, min_bins=2) == [(1, 4), (7, 9)]
    assert P.peak_regions(b, min_bins=1) == [(1, 4), (5, 6), (7, 9)]
    assert P.peak_regions(np.zeros(10, dtype=int)) == []


def test_promoter_windows_are_symmetric_and_differ_from_the_mseprom_region() -> None:
    """±2 kb around the start, not 2 kb upstream — the two regions are not interchangeable."""
    genes = ["chrT\t50000\t60000\tG\t.\t+\n"]
    w = P.promoter_windows(genes, "chrT", 10_000)
    assert w == [(2000 - 80, 2000 + 80)]
    assert (w[0][1] - w[0][0]) == 2 * P.PROM_LOC


# ===========================================================================
# the LOSS tier — pinned against the training loop's own arithmetic
# ===========================================================================
# `candi.bench` may NOT import `candi.train` (`tests/test_bench_harness.py` pins that in a
# subprocess), so `bench.distributional` carries its own numpy copy of the three NLLs the objective
# is built from. A copy with no pin is a copy that drifts, and the drift would be invisible: a
# val-loss that is off by a constant still goes down.
#
# A TEST IS ALLOWED TO IMPORT BOTH SIDES. The boundary is a property of the shipped package, not of
# the test suite, and this file is the only place the two implementations may meet. That is the whole
# design: the equivalence is checked once, here, instead of being asserted in a docstring.
#
# TOLERANCE, AND WHY IT IS NOT ZERO. The training terms run behind `precision.fp32_fence`, which
# casts to float32 by construction; the numpy copies are float64 throughout. So the two agree to
# float32's precision and the numpy side is the more accurate one — a bit-exact assertion here would
# be asserting that this file reproduces torch's rounding, which is not the claim.

TORCH_F32 = dict(rel=2e-6, abs=2e-6)


def _torch():
    import torch
    return torch


def test_numpy_nb_nll_equals_the_training_loops_nb_nll() -> None:
    """`D.nb_nll(n, mu, y)` == `train._elem_nb_nll(p, n, y).mean()` at `p = n/(n+mu)`.

    The `p` fed to torch is the one `p_from_mu` derives, because the claim under test is that the
    two FORMULAS agree — not that two different parameterisations of the same distribution do.
    """
    torch = _torch()
    from candi.train import _elem_nb_nll

    rng = np.random.default_rng(0)
    n = rng.uniform(0.05, 40.0, size=4096)
    mu = rng.uniform(1e-3, 500.0, size=4096)
    y = rng.poisson(mu).astype(np.float64)
    p = D.p_from_mu(n, mu)

    want = float(_elem_nb_nll(torch.tensor(p, dtype=torch.float32),
                              torch.tensor(n, dtype=torch.float32),
                              torch.tensor(y, dtype=torch.float32)).mean())
    assert D.nb_nll(n, mu, y) == pytest.approx(want, **TORCH_F32)


def test_numpy_nb_nll_matches_at_the_clamps_the_training_loop_applies() -> None:
    """The guards are the pin too: `n` under `eps`, `p` at both ends, a negative target.

    **HANDING TORCH float64 DOES NOT MAKE THIS A float64 COMPARISON.** `_elem_nb_nll` opens with
    `fp32_fence`, which casts every floating input to float32 unconditionally — that is the fence's
    job. So the torch side is float32 arithmetic whatever this file passes in, and the tolerance is
    float32's, elementwise rather than on a mean that could hide one bad position.

    `n` is kept under ~1e3 for a reason worth writing down: `log_prob` differences two lgammas, and
    at `n = 1e6` those are ~1.28e7 apart in float32, where one ulp IS 1.0. The float32 answer there
    is wrong by ~1 nat and the float64 one is right — a divergence in torch's precision, not in
    these two formulas, and a test that asserted on it would be pinning the wrong thing.
    """
    torch = _torch()
    from candi.train import _elem_nb_nll

    n = np.array([1e-9, 1e-6, 1.0, 1e3, 3.0])      # the first two are both floored to eps
    mu = np.array([1e-9, 1.0, 1e-9, 1.0, 5.0])     # p at both ends of its clamp
    y = np.array([0.0, 4.0, 0.0, 2.0, -3.0])       # a negative count is floored at 0 on both sides
    p = D.p_from_mu(n, mu)
    want = _elem_nb_nll(torch.tensor(p), torch.tensor(n), torch.tensor(y)).numpy()
    assert np.isfinite(want).all()

    # 5e-4 rather than float32's 1e-7 because the lgamma cancellation above is already visible at
    # n = 1e3: lgamma(1000) is ~5905, where a float32 ulp is 5e-4. The tolerance tracks that, so it
    # is the arithmetic's own floor and not a number tuned until the test went green.
    got = np.array([D.nb_nll(n[i:i + 1], mu[i:i + 1], y[i:i + 1]) for i in range(len(n))])
    np.testing.assert_allclose(got, want, rtol=5e-4, atol=1e-6)
    assert D.nb_nll(n, mu, y) == pytest.approx(float(want.mean()), rel=5e-4, abs=1e-6)


def test_numpy_gaussian_nll_equals_the_training_loops_gaussian_nll_including_the_constant() -> None:
    """`full=True`, so the `0.5*log(2*pi)` is in. Dropping it shifts every value by 0.9189385."""
    torch = _torch()
    from candi.train import _elem_gaussian_nll

    rng = np.random.default_rng(1)
    mu = rng.normal(0.0, 3.0, size=4096)
    var = rng.uniform(1e-8, 9.0, size=4096)        # spans torch's own 1e-6 clamp on `var`
    y = rng.normal(0.0, 3.0, size=4096)

    want = float(_elem_gaussian_nll(torch.tensor(mu, dtype=torch.float64),
                                    torch.tensor(var, dtype=torch.float64),
                                    torch.tensor(y, dtype=torch.float64)).mean())
    assert D.gaussian_nll(mu, var, y) == pytest.approx(want, rel=1e-9)

    # ... and the constant really is present, rather than the two copies agreeing on omitting it.
    got_without = float(np.mean(0.5 * (np.log(np.maximum(var, D.GAUSSIAN_VAR_EPS))
                                       + (mu - y) ** 2 / np.maximum(var, D.GAUSSIAN_VAR_EPS))))
    assert D.gaussian_nll(mu, var, y) - got_without == pytest.approx(
        0.5 * np.log(2 * np.pi), abs=1e-12)


def test_numpy_bernoulli_nll_equals_the_training_loops_bce_through_the_sigmoid() -> None:
    """The harness stores `sigmoid(logit)`, so the pin goes logit -> sigmoid -> this function."""
    torch = _torch()
    from candi.train import _elem_peak_bce

    rng = np.random.default_rng(2)
    logit = rng.uniform(-8.0, 8.0, size=4096)
    y = (rng.random(4096) < 0.2).astype(np.float64)

    want = float(_elem_peak_bce(torch.tensor(logit, dtype=torch.float64),
                                torch.tensor(y, dtype=torch.float64)).mean())
    prob = 1.0 / (1.0 + np.exp(-logit))
    assert D.bernoulli_nll(prob, y) == pytest.approx(want, rel=1e-9)


def test_bernoulli_nll_refuses_the_missing_sentinel_exactly_as_the_training_loop_does() -> None:
    """`y_peaks` holds -1 for an assay the biosample lacks; BCE against -1 is not defined."""
    with pytest.raises(ValueError, match="outside"):
        D.bernoulli_nll(np.array([0.5, 0.5]), np.array([1.0, -1.0]))


def test_the_signal_target_transform_moves_the_gaussian_loss_and_lands_on_torchs_value() -> None:
    """D30 — `arcsinh` is not `none`, and the transformed number is the one torch computes.

    This is the whole reason the loss path carries a transform: on a store the head is fit against
    `arcsinh(-log10 p)`, so scoring it against raw `-log10 p` gives a finite, plausible, wrong loss.
    """
    torch = _torch()
    from candi.train import _elem_gaussian_nll

    rng = np.random.default_rng(3)
    pval = rng.uniform(0.0, 60.0, size=2048)       # raw -log10 p, which reaches 0 constantly
    mu = np.abs(rng.normal(1.5, 1.0, size=2048))   # a softplus mean, so non-negative
    var = rng.uniform(1e-3, 2.0, size=2048)

    plain = D.gaussian_nll(mu, var, D.transform_signal_target(pval, "none"))
    bent = D.gaussian_nll(mu, var, D.transform_signal_target(pval, "arcsinh"))
    assert plain != pytest.approx(bent, rel=1e-3), "the transform did nothing"

    want = float(_elem_gaussian_nll(torch.tensor(mu, dtype=torch.float64),
                                    torch.tensor(var, dtype=torch.float64),
                                    torch.arcsinh(torch.tensor(pval, dtype=torch.float64))).mean())
    assert bent == pytest.approx(want, rel=1e-9)
    # `none` is the identity, bit for bit — the h5 path's whole arithmetic rests on that.
    np.testing.assert_allclose(D.transform_signal_target(pval, "none"), pval, rtol=0, atol=0)


def test_the_transform_vocabulary_matches_the_training_modules() -> None:
    """Two copies of a vocabulary drift. This is the one place they are compared."""
    from candi.train import SIGNAL_TARGET_TRANSFORMS as TRAIN_MODES

    assert D.SIGNAL_TARGET_TRANSFORMS == TRAIN_MODES
    with pytest.raises(ValueError, match="signal_target_transform"):
        D.transform_signal_target(np.array([1.0]), "sqrt")


# ---------------------------------------------------------------------------
# the OTHER direction — the benchmark path's inversion (the spaces contract)
# ---------------------------------------------------------------------------

def test_inverting_the_mean_undoes_the_target_transform_exactly() -> None:
    """`sinh(arcsinh(x)) = x` and `expm1(log1p(x)) = x` — the two are exact inverses on paper.

    The benchmark path's whole claim is that a prediction lands back in `-log10 p`, so the identity
    is checked on the mean, in both modes, at values that span the corpus range.
    """
    x = np.array([0.0, 1e-3, 0.5, 2.0, 17.0, 400.0, 17_731.0])
    for mode in ("arcsinh", "log1p"):
        mu_t = D.transform_signal_target(x, mode)
        mu_back, _ = D.invert_signal_prediction(mu_t, np.ones_like(x), mode)
        np.testing.assert_allclose(mu_back, x, rtol=1e-12, atol=1e-9)


def test_the_inverted_sigma_is_the_derivative_of_the_map_at_the_mean() -> None:
    """The delta method IS a first derivative: `σ' = |g'(µ)| σ`, `g = sinh` or `expm1`.

    Checked against a central finite difference rather than against a second copy of `cosh`, so a
    typo in the formula cannot be reproduced by the test.
    """
    mu = np.array([-1.0, 0.0, 0.7, 2.5])
    sigma = np.array([0.1, 0.25, 0.5, 1.0])
    h = 1e-6
    for mode, g in (("arcsinh", np.sinh), ("log1p", np.expm1)):
        _, s = D.invert_signal_prediction(mu, sigma, mode)
        deriv = (g(mu + h) - g(mu - h)) / (2 * h)
        np.testing.assert_allclose(s, np.abs(deriv) * sigma, rtol=1e-6)


def test_the_delta_method_is_honest_about_being_an_approximation() -> None:
    """`sinh` of a Gaussian is NOT a Gaussian, and the docstring says so. This is the size of it.

    Monte Carlo the true pushforward of `N(µ, σ²)` through `sinh`. At small `σ` the delta-method
    `(sinh µ, cosh µ · σ)` matches its mean and sd to a few percent; at large `σ` it does not, and
    the failure is one-sided — `sinh` is convex above 0, so the true mean exceeds `sinh µ`. A reader
    of a store-path `pit_ks` or `coverage_95` is reading the first case at best.
    """
    rng = np.random.default_rng(11)
    mu, n = 2.0, 400_000
    for sigma, tol in ((0.05, 0.02), (0.1, 0.05)):
        draws = np.sinh(rng.normal(mu, sigma, size=n))
        m, s = D.invert_signal_prediction(np.array([mu]), np.array([sigma]), "arcsinh")
        assert abs(draws.mean() - m[0]) / m[0] < tol
        assert abs(draws.std() - s[0]) / s[0] < tol
    # ... and at a σ the head could plausibly emit, the approximation is simply wrong.
    big = np.sinh(rng.normal(mu, 1.5, size=n))
    m, s = D.invert_signal_prediction(np.array([mu]), np.array([1.5]), "arcsinh")
    assert big.mean() > 1.5 * m[0], "sinh is convex here; the true mean must exceed sinh(mu)"
    assert big.std() > 1.5 * s[0]


def test_none_is_the_identity_and_hands_back_the_caller_s_own_arrays() -> None:
    """The h5 path's bit-identity rests on this: not "equal to", the SAME OBJECT.

    A copy in float64 would be equal here and could still change a downstream float32 reduction, so
    identity is the assertion rather than `allclose`.
    """
    mu = np.linspace(0.0, 3.0, 16, dtype=np.float32)
    sigma = np.full(16, 0.5, dtype=np.float32)
    m, s = D.invert_signal_prediction(mu, sigma, "none")
    assert m is mu and s is sigma


def test_the_inversion_refuses_a_mode_outside_the_vocabulary() -> None:
    """Same vocabulary as the forward direction, same refusal — a silent fall-through would score
    an unbent prediction against a raw truth and call it a benchmark number."""
    with pytest.raises(ValueError, match="signal_target_transform"):
        D.invert_signal_prediction(np.array([1.0]), np.array([1.0]), "sqrt")


def test_the_nll_eps_constants_are_the_training_loops_own_defaults() -> None:
    """A constant retyped is a constant that can drift; read it off the signature instead."""
    import inspect

    from candi.train import _elem_nb_nll

    assert inspect.signature(_elem_nb_nll).parameters["eps"].default == D.NLL_EPS
