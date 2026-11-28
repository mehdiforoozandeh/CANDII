"""t396 C2: tools/t396/spaces.py (three score spaces, the half-vs-half floor, the scorer's macro).

Requirements checked:
  - `aggregate` on records made by `deviance.target_terms` equals `deviance.gap` and
    `deviance.calibration_ratio` per assay, and its macro is the mean over the scorable assays;
  - an assay outside `scorable` does not move the scorable macro (it moves `all_assays`);
  - `to_1kb` of a constant, and the tail remainder is dropped;
  - `pool_size_1kb` follows the rule of `minicandi.losses.pool_nb`;
  - two Poisson halves give `ratio_to_halves < 1` for the true mean.
Synthetic arrays only. The package is imported as a namespace package (no tools/t396/__init__.py
is needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import deviance as DV  # noqa: E402
from t396 import spaces as SP  # noqa: E402

SCORABLE = [f"A{i}" for i in range(9)]


def _target(rng, assay, lam_scale=3.0, noise=0.3, shape=(4, 200)):
    lam = rng.gamma(2.0, lam_scale / 2.0, size=shape)
    f1, f2 = 0.45, 0.55
    y1 = rng.poisson(f1 * lam).astype(np.float64)
    y2 = rng.poisson(f2 * lam).astype(np.float64)
    y = y1 + y2
    mu = lam * np.exp(rng.normal(0.0, noise, size=shape))
    n = np.full(shape, 5.0)
    mu_ref = np.full(shape, lam.mean())
    r = SP.score_target(y, mu, n, mu_ref, y1, y2, (f1, f2))
    r["assay"] = assay
    return r


def _records(seed=0, assays=SCORABLE, per=3):
    rng = np.random.default_rng(seed)
    return [_target(rng, a, noise=0.1 + 0.05 * i) for i, a in enumerate(assays) for _ in range(per)]


def test_spaces_constants():
    assert SP.SPACES == ("overlap_count", "start_sum", "sum_1kb_count")
    assert SP.KB_BINS == 40


def test_score_target_keys_and_terms():
    rng = np.random.default_rng(1)
    r = _target(rng, "A0")
    y_keys = set(DV.target_terms(np.ones(10), np.ones(10), np.ones(10), mu_ref=np.ones(10)))
    assert y_keys <= set(r)
    assert {"D_halves", "D0_pr2", "D_model_pr2"} <= set(r)
    # D_halves by hand
    y1 = np.array([0.0, 2.0, 4.0]); y2 = np.array([1.0, 3.0, 5.0])
    t = SP.score_target(y1 + y2, np.ones(3), np.ones(3), np.ones(3), y1, y2, (0.25, 0.75))
    assert t["D_halves"] == pytest.approx(DV.deviance(y2, np.maximum(y1 * 3.0, DV.MU_MIN)))
    assert t["D_model_pr2"] == pytest.approx(DV.deviance(y2, 0.75 * np.ones(3)))
    assert t["D0_pr2"] == pytest.approx(DV.deviance(y2, np.full(3, 3.0)))


def test_aggregate_equals_deviance_gap_and_calibration():
    recs = _records()
    agg = SP.aggregate(recs, SCORABLE)
    per_skill, per_cal = [], []
    for a in SCORABLE:
        rs = [r for r in recs if r["assay"] == a]
        assert agg["per_assay"][a]["skill"] == pytest.approx(DV.gap(rs, "D_ref"), rel=1e-12)
        assert agg["per_assay"][a]["calibration_ratio"] == pytest.approx(
            DV.calibration_ratio(rs), rel=1e-12)
        assert agg["per_assay"][a]["r2"] == pytest.approx(
            1.0 - np.mean([r["D_model"] / r["D0"] for r in rs]), rel=1e-12)
        per_skill.append(DV.gap(rs, "D_ref"))
        per_cal.append(DV.calibration_ratio(rs))
    assert agg["scorable"]["skill"] == pytest.approx(np.mean(per_skill), rel=1e-12)
    assert agg["scorable"]["calibration_ratio"] == pytest.approx(np.mean(per_cal), rel=1e-12)
    assert agg["scorable"]["n_assays"] == 9
    assert agg["missing_scorable"] == []


def test_assay_outside_scorable_does_not_move_macro():
    recs = _records()
    base = SP.aggregate(recs, SCORABLE)
    rng = np.random.default_rng(7)
    extra = [_target(rng, "OUTSIDE", noise=1.5) for _ in range(3)]
    more = SP.aggregate(recs + extra, SCORABLE)
    for k in SP.METRICS:
        assert more["scorable"][k] == base["scorable"][k]
    assert more["all_assays"]["n_assays"] == 10
    assert more["all_assays"]["skill"] != base["all_assays"]["skill"]
    assert "OUTSIDE" in more["per_assay"]


def test_scorable_as_dict_and_missing_assay():
    recs = [r for r in _records() if r["assay"] != "A8"]
    agg = SP.aggregate(recs, {a: {} for a in SCORABLE})
    assert agg["missing_scorable"] == ["A8"]
    assert agg["scorable"]["n_assays"] == 8


def test_to_1kb_constant_and_tail():
    x = np.full((3, 16384), 2.0)
    out = SP.to_1kb(x)
    assert out.shape == (3, 16384 // 40)          # 409 sums; the 24-bin tail is dropped
    assert np.all(out == 80.0)
    y = np.arange(85, dtype=float)[None]
    np.testing.assert_allclose(SP.to_1kb(y), [[np.arange(40).sum(), np.arange(40, 80).sum()]])


def test_pool_size_1kb_rule():
    rng = np.random.default_rng(3)
    mu = rng.gamma(1.0, 2.0, size=(2, 120))
    n = rng.gamma(2.0, 2.0, size=(2, 120))
    b = mu.reshape(2, 3, 40); nb = n.reshape(2, 3, 40)
    want = b.sum(-1) ** 2 / (b * b / nb).sum(-1)
    np.testing.assert_allclose(SP.pool_size_1kb(mu, n), want, rtol=1e-12)
    # constant n and mu: the size of a sum of k iid NB(n, mu) is k * n
    np.testing.assert_allclose(SP.pool_size_1kb(np.full((1, 80), 3.0), np.full((1, 80), 2.0)),
                               [[80.0, 80.0]])


def test_pool_size_1kb_matches_losses_pool_nb():
    torch = pytest.importorskip("torch")
    try:
        from minicandi.losses import pool_nb
    except Exception as e:                     # pragma: no cover - environment without minicandi
        pytest.skip(f"minicandi.losses not importable: {e}")
    rng = np.random.default_rng(4)
    mu = rng.gamma(0.5, 2.0, size=(3, 170)); mu[0, :5] = 0.0
    n = rng.gamma(2.0, 2.0, size=(3, 170))
    _, n_ref = pool_nb(torch.tensor(mu), torch.tensor(n), SP.KB_BINS, dim=1)
    np.testing.assert_allclose(SP.pool_size_1kb(mu, n), n_ref.numpy(), rtol=1e-10)


def test_poisson_halves_ratio_below_one_for_true_mean():
    rng = np.random.default_rng(5)
    recs = []
    for a in SCORABLE:
        for _ in range(3):
            lam = rng.gamma(2.0, 2.0, size=(4, 400))
            f1, f2 = 0.4, 0.6
            y1 = rng.poisson(f1 * lam).astype(float)
            y2 = rng.poisson(f2 * lam).astype(float)
            r = SP.score_target(y1 + y2, lam, np.full_like(lam, 1e3), np.full_like(lam, lam.mean()),
                                y1, y2, (f1, f2))
            r["assay"] = a
            recs.append(r)
    agg = SP.aggregate(recs, SCORABLE)
    assert agg["scorable"]["ratio_to_halves"] < 1.0
    for a in SCORABLE:
        assert agg["per_assay"][a]["ratio_to_halves"] < 1.0
