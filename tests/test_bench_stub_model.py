"""Layer 3 — the C-block, against models whose conditioning we wrote ourselves.

`EVAL_PLAN.md` §6. Layers 1 and 2 verify *functions of arrays*. `covuse`, `covshare` and `covspec`
are not functions of arrays: they are properties of a **model**, measured by re-running it under
substituted covariates. Nothing an array-level test can do reaches them, and shipping them on the
strength of "the primitives are tested" would be shipping them unverified.

So the fixtures here are models. Each is a few lines, each has exactly one thing it does with its
covariates, and each therefore has a C-block answer that is known before the code runs:

- `DepthOnly` — `µ` depends on told depth and nothing else. Share must be (1, 0, 0, 0).
- `DepthAndAssay` — depth sets the level, assay identity sets the profile shape. `covspec`'s owner
  map must recover exactly that assignment.
- `IgnoresEverything` — `µ` is constant. This is the failure the conditional-generation literature
  names, `p(y|c) = p(y)`, and every instrument must say so.
- `Entangled` — every covariate perturbs every aspect. `covspec`'s gap must collapse.

The covariate matrix is built so that `run_type` is a deterministic function of `read_length`,
reproducing the real degeneracy `AGENTS.md` §329 records as
`H(run_type | assay_id, read_length) = 0.000 bits`. That is not incidental: it is the case that makes
the conditional null empty, and the block has to report it as degeneracy rather than as the model
ignoring the covariate.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import nbinom

from candi.bench import covariate as C

L = 32                      # positions per unit
DEPTHS = np.array([24.0, 23.0, 22.0, 21.0])
ASSAYS = np.array([0.0, 1.0, 2.0])
READ_LENGTHS = np.array([36.0, 50.0, 100.0, 150.0])


def _run_type(read_length: np.ndarray) -> np.ndarray:
    """Paired-end iff the read is long. Deterministic given read_length — the real degeneracy."""
    return (read_length >= 100.0).astype(float)


def make_covariates() -> np.ndarray:
    """`(48, 4)` — every (depth, assay, read_length) combination, run_type derived.

    Chosen so the conditional null is non-degenerate for depth, assay and read_length (each has
    alternatives inside every bucket of the other three) and **degenerate for run_type** (fixed once
    read_length is known). Both branches of `_conditional_resample` therefore get exercised.
    """
    rows = [(d, a, r, _run_type(np.array(r))[()])
            for d in DEPTHS for a in ASSAYS for r in READ_LENGTHS]
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# the stubs
# ---------------------------------------------------------------------------

def _profile(shape_id: float) -> np.ndarray:
    """A unit-mean profile selected by `shape_id`. Different ids, genuinely different shapes.

    `shape_id` is clipped to [0, 2] so the width `0.5 + s` stays positive: an unclipped `Entangled`
    fed this a large negative value, the Gaussian inverted, `exp` overflowed, and the whole
    `covshare` result came back nan. A fixture that can produce nan tests nothing.
    """
    s = float(np.clip(shape_id, 0.0, 2.0))
    x = np.linspace(-3.0, 3.0, L)
    base = np.exp(-((x - (s - 1.0)) ** 2) / (0.5 + s))
    return base / base.mean()


class DepthOnly:
    """`µ = 2^(depth − 24) · flat profile`. Reads column 0, ignores columns 1, 2 and 3."""

    def __call__(self, cov: np.ndarray):
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        level = 2.0 ** (cov[:, 0] - 24.0) * 40.0
        mu = level[:, None] * np.ones((1, L))
        return {"mu": mu, "n": np.full_like(mu, 5.0)}


class DepthAndAssay:
    """Depth sets the level; assay identity sets the profile shape. Two covariates, two aspects."""

    def __call__(self, cov: np.ndarray):
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        level = 2.0 ** (cov[:, 0] - 24.0) * 40.0
        shapes = np.stack([_profile(a) for a in cov[:, 1]], axis=0)
        mu = level[:, None] * shapes
        return {"mu": mu, "n": np.full_like(mu, 5.0)}


class IgnoresEverything:
    """`p(y|c) = p(y)`. The conditional model that learned to ignore its condition."""

    def __call__(self, cov: np.ndarray):
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        mu = np.full((len(cov), L), 40.0)
        return {"mu": mu, "n": np.full_like(mu, 5.0)}


# Fixed standardisation constants for `Entangled`. They must NOT be derived from the batch it is
# called with: `covshare` calls the predictor on small blocks, and a model that standardises
# against its own batch is not a function of its own row -- its prediction for one unit would
# change depending on
# what else was in the call. That is not a model, and every instrument here assumes it is one.
_ENT_MEAN = np.array([22.5, 1.0, 84.0, 0.5])
_ENT_STD = np.array([1.12, 0.82, 44.0, 0.5])


class Entangled:
    """Every covariate perturbs every aspect. Responsive, and specific to nothing."""

    def __call__(self, cov: np.ndarray):
        cov = np.atleast_2d(np.asarray(cov, dtype=np.float64))
        z = (cov - _ENT_MEAN) / _ENT_STD
        mix = z.sum(axis=1)
        level = 40.0 * np.exp(0.25 * mix)
        shapes = np.stack([_profile(1.0 + 0.5 * m) for m in mix], axis=0)
        mu = level[:, None] * shapes
        n = np.full_like(mu, 5.0) * np.exp(0.2 * mix)[:, None]
        return {"mu": mu, "n": n}


def make_target(cov: np.ndarray, seed: int = 0) -> np.ndarray:
    """Counts drawn from `DepthOnly`'s own forecast at the TRUE covariates.

    So the truth genuinely depends on depth and genuinely does not depend on anything else. Any
    instrument that reports otherwise is reporting an artefact of itself.
    """
    rng = np.random.default_rng(seed)
    p = DepthOnly()(cov)
    prob = C.p_from_mu(p["n"], p["mu"])
    return nbinom.rvs(p["n"], prob, random_state=rng).astype(float)


# ===========================================================================
# covuse — does the decoder use the covariate at all?
# ===========================================================================

@pytest.fixture(scope="module")
def cov():
    return make_covariates()


@pytest.fixture(scope="module")
def target(cov):
    return make_target(cov)


def test_covuse_finds_depth_and_only_depth_in_a_depth_only_model(cov, target) -> None:
    """The instrument's core claim, on a model where the answer is known by construction.

    Depth must be significant under both nulls. The other three must return p = 1.0 exactly — not
    "not significant", but exactly 1.0, because substituting a covariate the model never reads leaves
    the loss bit-identical, so every resampled loss ties the observed one.
    """
    out = C.covuse(DepthOnly(), cov, target, n_resamples=100, seed=0)

    assert out["depth"]["marginal_p"] < 0.05
    assert out["depth"]["conditional_p"] < 0.05
    assert out["depth"]["marginal_mean_d_crps"] > 0.0
    assert out["depth"]["uses_covariate"] is True

    for name in ("assay_id", "read_length", "run_type"):
        assert out[name]["marginal_p"] == 1.0, name
        assert out[name]["conditional_p"] == 1.0, name
        assert out[name]["marginal_mean_d_crps"] == pytest.approx(0.0, abs=1e-12), name
        assert out[name]["uses_covariate"] is False, name


def test_covuse_within_batch_tripwire_reads_exactly_zero(cov, target) -> None:
    """The structural null. Substituting the value already present can only give 0.0.

    A non-zero here is a bug in the probe, never a finding about the model, and it is the cheapest
    falsifier of the entire block — so it is asserted on every stub, not just the convenient one.
    """
    for model in (DepthOnly(), DepthAndAssay(), IgnoresEverything(), Entangled()):
        out = C.covuse(model, cov, target, n_resamples=5, seed=0)
        for name in C.COVARIATES:
            assert out[name]["within_batch_d_crps"] == 0.0, (type(model).__name__, name)


def test_covuse_reports_the_degenerate_conditional_null_rather_than_hiding_it(cov, target) -> None:
    """`run_type` is a deterministic function of `read_length`, so its conditional null is empty.

    The right behaviour is to say so. A p-value of 1.0 from an empty null means "this covariate adds
    nothing beyond the others", which is a different statement from "the model ignores it", and the
    flag is what keeps the two apart.
    """
    out = C.covuse(DepthOnly(), cov, target, n_resamples=20, seed=0)
    assert out["run_type"]["conditional_null_degenerate"] is True
    assert out["depth"]["conditional_null_degenerate"] is False
    assert out["assay_id"]["conditional_null_degenerate"] is False
    assert out["read_length"]["conditional_null_degenerate"] is False


def test_covuse_says_nothing_is_used_when_the_model_ignores_its_condition(cov, target) -> None:
    """`p(y|c) = p(y)` — the named failure mode of conditional generative models."""
    out = C.covuse(IgnoresEverything(), cov, target, n_resamples=50, seed=0)
    for name in C.COVARIATES:
        assert out[name]["marginal_p"] == 1.0, name
        assert out[name]["uses_covariate"] is False, name


def test_covuse_finds_two_covariates_when_the_model_uses_two(cov) -> None:
    """A model must not be able to pass by responding to exactly one thing."""
    two = DepthAndAssay()
    rng = np.random.default_rng(3)
    p = two(cov)
    y = nbinom.rvs(p["n"], C.p_from_mu(p["n"], p["mu"]), random_state=rng).astype(float)
    out = C.covuse(two, cov, y, n_resamples=100, seed=1)
    assert out["depth"]["marginal_p"] < 0.05
    assert out["assay_id"]["marginal_p"] < 0.05
    assert out["read_length"]["marginal_p"] == 1.0
    assert out["run_type"]["marginal_p"] == 1.0


# ===========================================================================
# covshare — what fraction of the output does each covariate own?
# ===========================================================================

def test_covshare_is_one_zero_zero_zero_for_a_depth_only_model(cov) -> None:
    """Shapley effects must put all of the level's variance on depth, and none anywhere else.

    The bar is 0.02 because that is what the bias-corrected estimator delivers at this sampling.
    Without the inner-sample correction in `covshare` the same call returns depth = 0.902 and hands
    0.098 to three covariates the model provably never reads -- so this tolerance is also the test
    that the correction is present.
    """
    s = C.covshare(DepthOnly(), cov, n_outer=48, n_inner=6, seed=0)
    # was `output_is_constant is False`; that flag is retired, and a total many error bars clear of
    # zero is the measurement it was standing in for.
    assert s["total_variance_unclamped"] > 3.0 * s["total_variance_se"]
    assert s["depth"] == pytest.approx(1.0, abs=0.02)
    for name in ("assay_id", "read_length", "run_type"):
        assert s[name] == pytest.approx(0.0, abs=0.02), name
    assert sum(s[n] for n in C.COVARIATES) == pytest.approx(1.0, abs=1e-9)


def test_covshare_always_sums_to_one_which_sobol_would_not(cov) -> None:
    """The property that motivated choosing Shapley over Sobol under dependent inputs (D10).

    `run_type` is fully determined by `read_length` here, so their contributions genuinely overlap.
    Sobol total-effect indices would each claim the shared variance and over-sum; Shapley splits it.
    """
    for model in (DepthOnly(), DepthAndAssay(), Entangled()):
        s = C.covshare(model, cov, n_outer=48, n_inner=6, seed=1)
        assert sum(s[n] for n in C.COVARIATES) == pytest.approx(1.0, abs=1e-9), type(model).__name__


def test_covshare_reports_a_constant_output_rather_than_dividing_by_zero(cov) -> None:
    """Zero total variance is the ignored-condition failure, not an undefined 0/0 to paper over.

    The `output_is_constant` boolean this used to assert is retired: it fired on the clamped total,
    so it could not tell a genuinely constant output from an effect below the estimator's resolution.
    What a constant output actually looks like is asserted directly instead — a naive variance of
    zero, a bias term of zero, and therefore an error bar of zero, none of which a merely
    unresolvable effect produces.
    """
    s = C.covshare(IgnoresEverything(), cov, n_outer=24, n_inner=4, seed=0)
    assert s["total_variance"] == pytest.approx(0.0, abs=1e-12)
    assert s["total_variance_naive"] == pytest.approx(0.0, abs=1e-12)
    assert s["total_variance_bias"] == pytest.approx(0.0, abs=1e-12)
    assert s["total_variance_se"] == pytest.approx(0.0, abs=1e-12)
    assert all(np.isnan(s[n]) for n in C.COVARIATES)
    assert "output_is_constant" not in s


def test_covshare_splits_the_share_when_two_covariates_matter(cov) -> None:
    """`DepthAndAssay` scalarised on `level` still puts everything on depth — assay moves shape only.

    Worth asserting because it is the subtle half of `covshare`: a share is a share *of the chosen
    aspect*, not of the model. Reading it as "the model ignores assay identity" would be wrong, and
    `covspec` is the instrument that shows why.
    """
    lvl = C.covshare(DepthAndAssay(), cov, scalar="level", n_outer=96, n_inner=12, seed=2)
    assert lvl["depth"] == pytest.approx(1.0, abs=0.05)

    shp = C.covshare(DepthAndAssay(), cov, scalar="shape", n_outer=96, n_inner=12, seed=2)
    assert shp["assay_id"] == pytest.approx(1.0, abs=0.05)


# ===========================================================================
# depthdir / depthcounterfact — does depth move it right?
# ===========================================================================

def test_depthdir_is_perfectly_monotone_for_a_model_that_scales_with_told_depth(cov) -> None:
    """Told half the depth, predict half the signal: every ladder step must fall."""
    out = C.depthdir(DepthOnly(), cov)
    assert out["monotone_frac"] == pytest.approx(1.0, abs=1e-12)
    assert out["mean_dose_response_corr"] == pytest.approx(1.0, abs=1e-9)
    levels = out["mean_level_by_told_depth"]
    assert all(levels[i] > levels[i + 1] for i in range(len(levels) - 1))
    # one log2 step of told depth is one log2 step of predicted level
    assert (levels[0] - levels[1]) == pytest.approx(np.log(2.0), abs=1e-9)


def test_depthdir_is_flat_for_a_model_that_ignores_depth(cov) -> None:
    """A flat response is monotone in the weak sense, so the flat-unit count is what exposes it.

    `monotone_frac` alone would score 1.0 here — every step is `<= 0` because every step is 0 — which
    is exactly why the correlation and the flat-unit count are reported beside it.
    """
    out = C.depthdir(IgnoresEverything(), cov)
    assert out["n_flat_units"] == len(cov)
    assert np.isnan(out["mean_dose_response_corr"])


def test_depthcounterfact_picks_the_true_level(cov) -> None:
    """The arm with real ground truth: score each told depth against the real downsampled track.

    Built so the right answer is unambiguous — the truth at level k is drawn from the forecast made
    when told k — and the constant answer is reported beside it, because 1/n_levels is what "always
    say level 1" scores and is not a chance baseline.

    **This is also the measurement of the perfect-model ceiling.** `DepthOnly` is exactly right here
    by construction, so whatever it scores IS the cap. With the fixed foreground it scores 1.000,
    mean over 20 seeds, at every `fg_frac` in (0.02, 0.1, 0.5, 1.0) — there is no ceiling below 1 to
    quote. Re-selecting the mask from the level being scored, which is what the retired `eval.py`
    S14 did, drops the same perfect model to 0.250 at `fg_frac` 0.02 and 0.1: the mask picks
    positions high by noise, every level then prefers the deepest told depth, and the statistic
    collapses onto the constant answer.
    """
    rng = np.random.default_rng(5)
    model = DepthOnly()
    # all 48 rows, not the first 8: the top 2% of 8x32 positions is 5 positions, and a CRPS mean over
    # 5 draws cannot separate adjacent told depths (it scores 0.75-0.887 for a perfect model).
    base = cov
    ladder = [0.0, -1.0, -2.0, -3.0]

    pred_by_told, true_by_level = {}, {}
    for off in ladder:
        c = np.array(base, copy=True)
        c[:, 0] = base[:, 0] + off
        p = model(c)
        pred_by_told[off] = p
        true_by_level[off] = nbinom.rvs(
            p["n"], C.p_from_mu(p["n"], p["mu"]), random_state=rng).astype(float)

    out = C.depthcounterfact(pred_by_told, true_by_level)
    assert out["frac_min_at_true"] == pytest.approx(1.0, abs=1e-12)
    assert out["n_levels"] == 4
    assert out["constant_answer_value"] == pytest.approx(0.25, abs=1e-12)
    # one mask, taken from the deepest truth (offset 0.0), not from each level in turn
    assert out["fg_level"] == pytest.approx(0.0, abs=1e-12)
    assert out["n_positions"] == len(base) * L
    assert 0 < out["n_fg"] <= round(0.02 * out["n_positions"])


def test_depthcounterfact_fails_a_depth_blind_model(cov) -> None:
    """A model whose prediction never moves cannot pick the true level better than by tie-breaking."""
    rng = np.random.default_rng(6)
    base = cov[:8]
    ladder = [0.0, -1.0, -2.0, -3.0]
    truth_model, blind = DepthOnly(), IgnoresEverything()

    pred_by_told, true_by_level = {}, {}
    for off in ladder:
        c = np.array(base, copy=True)
        c[:, 0] = base[:, 0] + off
        t = truth_model(c)
        true_by_level[off] = nbinom.rvs(
            t["n"], C.p_from_mu(t["n"], t["mu"]), random_state=rng).astype(float)
        pred_by_told[off] = blind(c)

    out = C.depthcounterfact(pred_by_told, true_by_level)
    assert out["frac_min_at_true"] <= 0.25 + 1e-12


# ===========================================================================
# covspec — does each covariate move what it should?
# ===========================================================================

def test_covspec_recovers_which_covariate_owns_which_aspect(cov) -> None:
    """The instrument's whole claim: depth owns the level, assay identity owns the shape."""
    out = C.covspec(DepthAndAssay(), cov, seed=0)
    assert out["owner_by_aspect"]["level"] == "depth"
    assert out["owner_by_aspect"]["shape"] == "assay_id"
    assert out["gap_by_aspect"]["level"] == pytest.approx(1.0, abs=1e-9)
    assert out["gap_by_aspect"]["shape"] == pytest.approx(1.0, abs=1e-9)


def test_covspec_gap_collapses_when_every_covariate_moves_everything(cov) -> None:
    """The contrast that makes the gap informative rather than decorative.

    `Entangled` is highly responsive — `covuse` finds every covariate significant — so a
    responsiveness metric would call it well-conditioned. The gap says otherwise, because no aspect
    has an owner.
    """
    entangled = C.covspec(Entangled(), cov, seed=0)
    specific = C.covspec(DepthAndAssay(), cov, seed=0)
    assert entangled["mean_gap"] < 0.5
    assert specific["mean_gap"] > 0.9
    assert entangled["mean_gap"] < specific["mean_gap"]


def test_covspec_matrix_is_zero_for_covariates_the_model_does_not_read(cov) -> None:
    """A row of exact zeros is the signature of an unread covariate, and must be exactly zero."""
    out = C.covspec(DepthOnly(), cov, seed=0)
    m = np.array(out["matrix"])
    for name in ("assay_id", "read_length", "run_type"):
        row = m[out["covariates"].index(name)]
        assert np.allclose(row, 0.0, atol=1e-12), name
    assert m[out["covariates"].index("depth"), out["aspects"].index("level")] > 0.0


# ===========================================================================
# depthblind / biokeep — invariance, and the guard it is never quoted without
# ===========================================================================

def _latents(kind: str, n_regions: int = 12, n_depths: int = 4, dim: int = 8, seed: int = 0):
    """Synthetic latents with a known invariance structure. Returns (Z, depth_label, region_label)."""
    rng = np.random.default_rng(seed)
    region_centre = rng.normal(0, 5.0, size=(n_regions, dim))
    depth_centre = rng.normal(0, 5.0, size=(n_depths, dim))
    Z, dep, reg = [], [], []
    for r in range(n_regions):
        for d in range(n_depths):
            if kind == "invariant":
                z = region_centre[r] + rng.normal(0, 0.2, dim)
            elif kind == "collapsed":
                z = rng.normal(0, 0.01, dim)
            elif kind == "depth_encoding":
                z = depth_centre[d] + rng.normal(0, 0.2, dim)
            else:
                raise ValueError(kind)
            Z.append(z)
            dep.append(d)
            reg.append(r)
    return np.array(Z), np.array(dep), np.array(reg)


def test_depthblind_passes_a_latent_that_is_depth_invariant_and_region_specific() -> None:
    """The good case: four depths of one region collapse together, different regions stay apart."""
    Z, dep, reg = _latents("invariant", seed=1)
    out = C.depthblind(Z, dep, reg, k=8, seed=0)
    assert out["kbet_rejection_rate"] < 0.25
    assert out["batch_asw"] > 0.8
    assert out["ilisi"] > 2.5                    # against a maximum of 4
    assert out["bio_silhouette"] > 0.8
    assert out["invariance_ok"] is True


def test_depthblind_fails_a_latent_that_encodes_depth() -> None:
    """The obvious bad case: the nuisance is exactly what the latent represents."""
    Z, dep, reg = _latents("depth_encoding", seed=2)
    out = C.depthblind(Z, dep, reg, k=8, seed=0)
    assert out["kbet_rejection_rate"] > 0.5
    assert out["batch_asw"] < 0.3
    # iLISI's floor is only reachable when k fits inside a batch; `ilisi_k` reports the k actually
    # used, and the assertion is against what that k can resolve rather than against a flat 1.0.
    assert out["ilisi_k"] <= len(Z) // out["ilisi_max"]
    assert out["ilisi"] < 0.5 * out["ilisi_max"]
    assert out["invariance_ok"] is False


def test_a_collapsed_encoder_passes_invariance_and_is_still_refused(cov) -> None:
    """**The test that proves the guard guards** (D13).

    A constant encoder is perfectly invariant to depth — kBET cannot reject, iLISI is maximal, batch
    ASW is ideal. Under the retired `eval.py` M3 rule it would pass, since `encoder_eff_rank > 1.0`
    is a bar a latent with two directions clears. Here the bio-conservation score is near zero and
    the verdict is False.
    """
    Z, dep, reg = _latents("collapsed", seed=3)
    out = C.depthblind(Z, dep, reg, k=8, seed=0)

    # every invariance instrument is satisfied ...
    assert out["kbet_rejection_rate"] < 0.25
    assert out["batch_asw"] > 0.8
    assert out["ilisi"] > 3.0

    # ... and the guard catches it anyway
    assert out["bio_silhouette"] < 0.25
    assert out["invariance_ok"] is False


def test_effective_rank_alone_would_not_have_caught_it() -> None:
    """Why the guard had to change rather than be re-thresholded.

    A collapsed latent whose residual noise is isotropic has a high effective rank — near the full
    dimension — because the spectrum of pure noise is flat. So `effective_rank > 1.0` passes
    comfortably on exactly the latent that carries no biology at all. The silhouette does not.
    """
    Z, dep, reg = _latents("collapsed", seed=4)
    guard = C.biokeep(Z, reg)
    assert guard["effective_rank"] > 1.0         # the old bar, cleared
    assert guard["bio_silhouette"] < 0.25        # the new one, failed


def test_depthblind_will_not_report_invariance_without_conservation() -> None:
    """The API refuses the shape that would let the two be separated in a report."""
    Z, dep, reg = _latents("invariant", seed=5)
    out = C.depthblind(Z, dep, reg, k=8, seed=0)
    for key in ("kbet_rejection_rate", "ilisi", "batch_asw", "bio_silhouette", "effective_rank"):
        assert key in out, key
    with pytest.raises(TypeError):
        C.depthblind(Z, dep)                  # bio_label is not optional


def test_kbet_rejects_at_the_nominal_rate_when_batches_are_drawn_identically() -> None:
    """kBET's null, checked directly: identical distributions must reject at ~alpha, not at 0."""
    rng = np.random.default_rng(6)
    Z = rng.normal(0, 1, size=(240, 6))
    batch = rng.integers(0, 4, size=240)         # label independent of position
    out = C.kbet(Z, batch, k=20, alpha=0.05, n_test=240, seed=0)
    assert out["kbet_rejection_rate"] < 0.15


def test_ilisi_endpoints_are_one_and_the_number_of_batches() -> None:
    """A bounded scale, which is the thing a cosine ratio against 0.3 never had."""
    rng = np.random.default_rng(7)
    Z = rng.normal(0, 1, size=(200, 4))
    mixed = rng.integers(0, 4, size=200)
    assert C.ilisi(Z, mixed, k=30) > 3.0

    separated = np.repeat(np.arange(4), 50)
    Z_sep = np.repeat(rng.normal(0, 20, size=(4, 4)), 50, axis=0) + rng.normal(0, 0.1, (200, 4))
    assert C.ilisi(Z_sep, separated, k=30) < 1.3
