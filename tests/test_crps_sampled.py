"""t56 — the sampled NB CRPS (`candi.metrics.nb_crps_sampled`) and its opt-in path through the bench.

Pure numerics; no store, no model, no h5. Three things are being pinned.

**It estimates the same number.** `nb_crps` is the definition, so every unbiasedness test here is
against `nb_crps` itself wherever the closed form is exact, and against an independent Poisson
CRPS in the large-n regime where `nb_crps` is itself a Poisson-limit approximation (t56).

**It is UNBIASED, not merely close.** The obvious estimator divides the pairwise sum by `k^2` and is
low by `E|X-X'| / 2k` for every k — an error that survives averaging over 124 M bins, which is the
only place this estimator is ever going to be used. `test_the_plug_in_form_is_biased_and_the_fair_
form_is_not` measures both against the closed form on the same draws, so a future "simplification"
back to `k^2` fails a test instead of quietly shifting a leaderboard.

**Its reach is exactly the CRPS family.** The bench tests at the bottom assert that turning the
approximation on moves `crps`, its two oracle-scaled companions and `scale_error`, and moves nothing
else in `nb_suite` — and that with it off, every key is the object the closed form returned.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from candi.bench import distributional as D
from candi.metrics import nb_crps, nb_crps_sampled


#: (n, mu) corners of the panel `t49` actually produced: mu spans 1e-6 to ~1e2 per track with tails
#: to 1e4, n runs from 0.046 up to the 1e4 Poisson floor the count baselines are capped at.
GRID = [(0.05, 0.01), (0.05, 100.0), (0.5, 1.0), (2.0, 10.0), (1.0, 0.05),
        (10.0, 2.0), (100.0, 300.0), (1e4, 2.0), (1e4, 1e3)]


def _mean_and_se(n, mu, y, *, k, seeds):
    p = n / (n + mu)
    v = np.array([float(nb_crps_sampled(n, p, y, k=k, seed=s)) for s in seeds])
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def _poisson_crps(mu: float, y: float) -> float:
    """CRPS of Poisson(mu) against integer y = sum_x (F(x) - 1{x >= y})^2.

    An INDEPENDENT reference, not a limit of `nb_crps`: it is a finite pmf sum and it does not
    overflow at the means used here, which is what lets it referee the n -> inf regime — where the
    closed form is itself a Poisson-limit approximation since t56, not an exact answer.
    """
    hi = max(int(mu + 12.0 * np.sqrt(mu) + 60.0), int(y) + 10)
    x = np.arange(hi + 1)
    return float(((poisson.cdf(x, mu) - (x >= y)) ** 2).sum())


# ---------------------------------------------------------------------------
# 1 — the estimator estimates the closed form
# ---------------------------------------------------------------------------

def test_sampled_agrees_with_the_closed_form_across_the_panel_range() -> None:
    seeds = range(48)
    for n, mu in GRID:
        for y in (0.0, 1.0, round(mu), round(3 * mu)):
            exact = float(nb_crps(n, n / (n + mu), y))
            if not np.isfinite(exact):
                continue
            got, se = _mean_and_se(n, mu, y, k=200, seeds=seeds)
            # 4 SE of the mean over 48 independent seeds, plus a floor for the bins where the score
            # itself is ~0 and the SE degenerates with it.
            assert abs(got - exact) <= 4.0 * se + 1e-6, (n, mu, y, exact, got, se)


def test_the_estimate_is_unbiased_at_every_k_not_just_large_k() -> None:
    """Bias must not be a function of k. It is the k(k-1) denominator that buys this."""
    n, mu, y = 2.0, 10.0, 8.0
    exact = float(nb_crps(n, n / (n + mu), y))
    for k in (2, 3, 5, 10, 50, 400):
        got, se = _mean_and_se(n, mu, y, k=k, seeds=range(400))
        assert abs(got - exact) <= 4.0 * se, (k, exact, got, se)


def test_the_plug_in_form_is_biased_and_the_fair_form_is_not() -> None:
    """The whole reason `_fair_crps` divides by k(k-1). Both forms, the SAME draws, k = 6.

    The plug-in `k^2` denominator keeps the k zero terms `|x_i - x_i|`, so it understates
    `E|X-X'|` by a factor (k-1)/k and therefore OVERSTATES CRPS by `E|X-X'| / 2k`. At k = 6 on this
    forecast that is a large multiple of the fair form's own residual, which is what the assertion
    at the end compares.
    """
    n, mu, y, k = 2.0, 10.0, 8.0, 6
    p = n / (n + mu)
    exact = float(nb_crps(n, p, y))
    rng = np.random.default_rng(7)
    x = rng.poisson(rng.gamma(np.full((60_000, k), n), (1.0 - p) / p)).astype(float)
    absdev = np.abs(x - y).mean(axis=1)
    pair = np.abs(x[:, :, None] - x[:, None, :]).sum(axis=(1, 2))
    fair = float((absdev - pair / (2.0 * k * (k - 1))).mean())
    plug = float((absdev - pair / (2.0 * k * k)).mean())
    gmd = 2.0 * (float(np.mean(absdev)) - exact)           # E|X-X'| implied by the closed form
    assert plug - exact == pytest.approx(gmd / (2 * k), rel=0.15), (plug, exact, gmd)
    assert abs(fair - exact) < 0.1 * abs(plug - exact), (fair, plug, exact)


def test_sampling_error_falls_as_one_over_sqrt_k() -> None:
    n, mu, y = 2.0, 10.0, 8.0
    p = n / (n + mu)
    sd = {k: float(np.std([float(nb_crps_sampled(n, p, y, k=k, seed=s)) for s in range(200)]))
          for k in (25, 100, 400)}
    assert sd[25] / sd[100] == pytest.approx(2.0, rel=0.25), sd
    assert sd[100] / sd[400] == pytest.approx(2.0, rel=0.25), sd


# ---------------------------------------------------------------------------
# 2 — the Poisson floor, where the closed form is NaN and this is the only instrument
# ---------------------------------------------------------------------------

def test_at_n_1e6_the_closed_form_and_the_sampler_both_land_on_the_poisson_crps() -> None:
    """`t49`'s pre-registered floor (RIVALS_PLAN.md §5.1), after the t56 fix.

    `nb_crps`'s hyp2f1 Gini term stops returning numbers long before n = 1e6, so the closed form
    used to be NaN here and this sampler was the only instrument — the reason this task existed.
    Since t56, `nb_crps` scores n > N_GINI_HYP2F1_MAX in the sd-standardized Poisson limit, so at
    the floor there are now three instruments and they must agree: NB(n, n/(n+mu)) tends to
    Poisson(mu), `_poisson_crps` computes that limit with no `candi.metrics` code in the loop, and
    both the closed form and the sampler must land on it.
    """
    for mu in (0.5, 2.0, 10.0, 100.0):
        for y in (0.0, 3.0, float(round(mu))):
            n = 1e6
            p = n / (n + mu)
            ref = _poisson_crps(mu, y)
            closed = float(nb_crps(n, p, y))
            assert np.isfinite(closed), (mu, y)                    # the t56 fix, pinned
            assert closed == pytest.approx(ref, rel=1e-3, abs=1e-6), (mu, y, closed, ref)
            got, se = _mean_and_se(n, mu, y, k=500, seeds=range(64))
            assert np.isfinite(got)
            assert abs(got - ref) <= 4.0 * se + 1e-6, (mu, y, got, se)


def test_a_whole_vector_at_the_poisson_floor_is_finite() -> None:
    rng = np.random.default_rng(0)
    mu = 10.0 ** rng.uniform(-2, 3, size=4000)
    n = np.full(4000, 1e6)
    y = np.round(mu * rng.uniform(0.0, 2.0, size=4000))
    assert np.isfinite(nb_crps_sampled(n, n / (n + mu), y, k=32, seed=0)).all()


# ---------------------------------------------------------------------------
# 3 — determinism, shape, refusals
# ---------------------------------------------------------------------------

def test_the_same_seed_gives_the_same_numbers_and_a_different_seed_does_not() -> None:
    rng = np.random.default_rng(3)
    mu = 10.0 ** rng.uniform(-2, 2, size=1500)
    n = 10.0 ** rng.uniform(-1, 4, size=1500)
    y = np.round(mu * rng.uniform(0.0, 2.0, size=1500))
    p = n / (n + mu)
    a = nb_crps_sampled(n, p, y, k=16, seed=11)
    assert np.array_equal(a, nb_crps_sampled(n, p, y, k=16, seed=11))
    assert not np.array_equal(a, nb_crps_sampled(n, p, y, k=16, seed=12))
    assert not np.array_equal(a, nb_crps_sampled(n, p, y, k=17, seed=11))


def test_it_broadcasts_and_keeps_the_caller_s_shape() -> None:
    n = np.full((3, 4), 2.0)
    mu = np.full((3, 4), 5.0)
    y = np.arange(4, dtype=float)
    out = nb_crps_sampled(n, n / (n + mu), y, k=8, seed=0)
    assert out.shape == (3, 4)
    assert float(nb_crps_sampled(2.0, 2.0 / 7.0, 1.0, k=8, seed=0)) == pytest.approx(
        float(nb_crps_sampled(np.array(2.0), np.array(2.0 / 7.0), np.array(1.0), k=8, seed=0)))


@pytest.mark.parametrize("k", [0, 1, -3])
def test_fewer_than_two_draws_is_refused(k) -> None:
    with pytest.raises(ValueError, match="k >= 2"):
        nb_crps_sampled(2.0, 0.2, 1.0, k=k)


def test_it_crosses_the_internal_chunk_boundary_without_changing_shape() -> None:
    """The chunk is a module constant, so a vector longer than it exercises the multi-pass path."""
    from candi.metrics import CRPS_SAMPLE_CHUNK
    k = 8
    m = (CRPS_SAMPLE_CHUNK // k) * 2 + 17
    n = np.full(m, 1.0)
    mu = np.full(m, 1.0)
    out = nb_crps_sampled(n, n / (n + mu), np.zeros(m), k=k, seed=0)
    assert out.shape == (m,) and np.isfinite(out).all()


# ---------------------------------------------------------------------------
# 4 — the bench path: opt-in, and reaching only the CRPS family
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def track():
    """One synthetic track: over-dispersed truth, a prediction that is off in level and in shape."""
    rng = np.random.default_rng(5)
    m = 3_000
    mu_true = 10.0 ** rng.uniform(-2, 1.5, size=m)
    y = rng.poisson(rng.gamma(1.5, mu_true / 1.5)).astype(float)
    return np.full(m, 2.0), mu_true * 1.6, y                      # (n, mu, y)


def test_crps_eval_returns_the_closed_form_itself_when_off() -> None:
    assert D.crps_eval(None) is nb_crps
    assert D.crps_eval(None, 7) is nb_crps


def test_the_default_nb_suite_is_untouched(track) -> None:
    n, mu, y = track
    a = D.nb_suite(n, mu, y, seed=0, n_pairs=2_000)
    b = D.nb_suite(n, mu, y, seed=0, n_pairs=2_000, crps_approx=None, crps_seed=3)
    assert set(a) == set(b)
    for key in a:
        assert repr(a[key]) == repr(b[key]), key


def test_approximating_moves_the_crps_family_and_nothing_else(track) -> None:
    n, mu, y = track
    exact = D.nb_suite(n, mu, y, seed=0, n_pairs=2_000)
    got = D.nb_suite(n, mu, y, seed=0, n_pairs=2_000, crps_approx=300, crps_seed=0)
    assert set(got) == set(exact)
    moved = {"crps", "crps_oracle_scaled", "crps_oracle_scaled_and_n", "scale_error",
             "c_star", "n_star_log2"}
    for key in set(exact) - moved:
        assert repr(got[key]) == repr(exact[key]), key
    for key in ("crps", "crps_oracle_scaled", "crps_oracle_scaled_and_n"):
        assert got[key] == pytest.approx(exact[key], rel=0.02), key
    # `marg_crps` is exact under both — a constant forecast is scored on its histogram, so there is
    # no CPU to buy back and `beats_marginal` keeps a bar with no sampling noise in it.
    assert got["marg_crps"] == exact["marg_crps"]


def test_scale_error_stays_the_difference_it_is_defined_to_be(track) -> None:
    """`scale_error` is DERIVED — `crps - crps_oracle_scaled` — so the identity must hold exactly
    whatever instrument produced the two terms. This is the cheap half of the decomposition check.
    """
    n, mu, y = track
    got = D.nb_suite(n, mu, y, seed=0, n_pairs=2_000, crps_approx=25, crps_seed=0)
    assert got["scale_error"] == pytest.approx(got["crps"] - got["crps_oracle_scaled"], rel=1e-12)


def test_the_two_crps_terms_share_their_draws_so_the_split_is_not_two_noises(track) -> None:
    """The expensive half: `scale_error` is only meaningful if its two terms are CORRELATED.

    `crps_eval` binds one seed for the whole track, so `crps` and `crps_oracle_scaled` are driven by
    the same Gamma stream — common random numbers. The correlation is partial rather than perfect
    (rescaling mu moves the Poisson stage's consumption, which desynchronises it), so what is
    asserted is the thing that matters: the spread of the DIFFERENCE across seeds is below the
    root-sum-square the two terms would give if they were independent.
    """
    n, mu, y = track
    rows = np.array([[(s := D.nb_suite(n, mu, y, seed=0, n_pairs=2_000, crps_approx=25,
                                       crps_seed=q))["crps"], s["crps_oracle_scaled"],
                      s["scale_error"]] for q in range(8)])
    independent = float(np.hypot(rows[:, 0].std(ddof=1), rows[:, 1].std(ddof=1)))
    assert float(rows[:, 2].std(ddof=1)) < independent, rows
    assert float(np.corrcoef(rows[:, 0], rows[:, 1])[0, 1]) > 0.2, rows


def test_seeds_are_reproducible_through_the_suite(track) -> None:
    n, mu, y = track
    kw = dict(seed=0, n_pairs=2_000, crps_approx=20)
    a = D.nb_suite(n, mu, y, crps_seed=1, **kw)
    assert a["crps"] == D.nb_suite(n, mu, y, crps_seed=1, **kw)["crps"]
    assert a["crps"] != D.nb_suite(n, mu, y, crps_seed=2, **kw)["crps"]
