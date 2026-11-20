"""Known-answer tests of the counts error budget (tools/onion_eval/error_budget.py).

Synthetic assay: T tracks in T cells over 8 regions of 16384 bins. Planted, in units of the truth's variance (1.0):
  truth x_t = S + U_t + noise_t      S shared (vs), U_t specific (vu), noise_t white (vn), the mean of two halves
  pred  y_t = a S + U_t + Z_t        Z_t specific variation the truth does not have (vz)
After the optimal affine calibration (slope b = (a vs + vu) / (a^2 vs + vu + vz)) the budget is, in expectation:
  shared:   V = vs, explained = vs, unexplained = 0, added = vs (1 - a b)^2
  specific: V = vu + vn, explained = vu^2 / (vu + vz), noise = vn, unexplained = vu vz / (vu + vz)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_p = Path(__file__).resolve().parents[1] / "tools"
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

from onion_eval import error_budget as EB  # noqa: E402

VS, VU, VN, VZ, A = 0.3, 0.4, 0.3, 0.2, 0.5
T, M = 6, 8 * EB.BLOCK


def _smooth(rng, n, w, var):
    v = np.convolve(rng.standard_normal(n + w), np.ones(w) / np.sqrt(w), mode="valid")[:n]
    return v * np.sqrt(var / v.var())


def _assay(seed):
    rng = np.random.default_rng(seed)
    S = _smooth(rng, M, 256, VS)
    U = np.stack([_smooth(rng, M, 16, VU) for _ in range(T)])
    Z = np.stack([_smooth(rng, M, 16, VZ) for _ in range(T)])
    n1 = rng.standard_normal((T, M)) * np.sqrt(2 * VN)        # half-depth noise: twice the pooled variance
    n2 = rng.standard_normal((T, M)) * np.sqrt(2 * VN)
    sig = S + U
    truth = sig + (n1 + n2) / 2
    pred = A * S + U + Z
    return truth, pred, sig + n1, sig + n2, np.array([f"cell{t}" for t in range(T)])


@pytest.fixture(scope="module")
def planted():
    truth, pred, h1, h2, cells = _assay(0)
    nbl = EB.noise_share_from_halves(h1, h2)
    return truth, pred, h1, h2, cells, nbl, EB.error_budget(truth, pred, cells, nbl)


def test_identities_hold(planted):
    truth, pred, _, _, cells, nbl, out = planted
    for o in (out, EB.error_budget(truth, pred, cells, None), EB.error_budget(truth, pred, cells, np.ones((T, EB.K)))):
        assert abs(o["V"] - (o["explained"] + o["unexplained"] + o["noise"])) < 1e-10
        assert abs(o["E"] - (o["unexplained"] + o["noise"] + o["added"])) < 1e-10
        assert abs(o["E"] - (o["shared"] + o["specific"])) < 1e-10
        assert abs(o["V"] - 1.0) < 1e-10                       # standardized truth: total energy 1
        assert abs(o["mse"] - (o["level"] + o["shared"] + o["specific"])) < 1e-8
        for p in EB.PARTS:
            for r in o["per_band"][p].values():
                assert abs(r["V"] - (r["explained"] + r["unexplained"] + r["noise"])) < 1e-10
                assert abs(r["E"] - (r["unexplained"] + r["noise"] + r["added"])) < 1e-10
                assert min(r["explained"], r["unexplained"], r["noise"]) >= 0


def test_planted_terms_recovered(planted):
    *_, out = planted
    b = (A * VS + VU) / (A * A * VS + VU + VZ)
    sh, sp = out["per_band"]["shared"], out["per_band"]["specific"]
    tot = lambda part, key: sum(r[key] for r in part.values())  # noqa: E731
    assert tot(sh, "V") == pytest.approx(VS, abs=0.03)
    assert tot(sh, "explained") == pytest.approx(VS, abs=0.03)
    assert tot(sh, "added") == pytest.approx(VS * (1 - A * b) ** 2, abs=0.03)
    assert tot(sp, "V") == pytest.approx(VU + VN, abs=0.03)
    assert tot(sp, "explained") == pytest.approx(VU ** 2 / (VU + VZ), abs=0.03)
    assert out["noise"] == pytest.approx(VN, abs=0.03)
    assert out["unexplained"] == pytest.approx(VU * VZ / (VU + VZ), abs=0.03)
    # white noise: 1/2 + 1/4 + 1/8 of its energy sits in the fine band (Haar levels 1-3)
    assert sp["fine"]["noise"] == pytest.approx(0.875 * VN, abs=0.03)
    sd_y = np.sqrt(A * A * VS + VU + VZ)                        # level = scale term (SD y - r SD x)^2; offset 0
    assert out["level"] == pytest.approx((sd_y - (A * VS + VU) / sd_y) ** 2, abs=0.005)


def test_level_planted_and_residual_budget_invariant(planted):
    truth, pred, _, _, cells, nbl, out = planted
    o2 = EB.error_budget(truth, 3.0 + 2.0 * pred, cells, nbl)
    sd_y = np.sqrt(A * A * VS + VU + VZ)
    r = (A * VS + VU) / sd_y
    assert o2["level"] == pytest.approx(9.0 + (2 * sd_y - r) ** 2, rel=0.03)
    for k in ("shared", "specific", "explained", "unexplained", "noise", "added"):
        assert o2[k] == pytest.approx(out[k], abs=1e-8)


def test_no_noise_term_folds_noise_into_unexplained(planted):
    truth, pred, _, _, cells, _, out = planted
    o0 = EB.error_budget(truth, pred, cells, None)
    assert o0["noise"] == 0.0
    assert o0["unexplained"] == pytest.approx(out["unexplained"] + out["noise"], abs=1e-10)


def test_one_cell_assay_has_no_shared_estimate(planted):
    truth, pred, _, _, _, nbl, _ = planted
    o = EB.error_budget(truth, pred, np.array(["c"] * T), nbl)
    assert np.isnan(o["shared"]) and np.isnan(o["specific"]) and np.isnan(o["explained"])
    assert np.isfinite(o["level"]) and np.isfinite(o["mse"])


def test_invalid_bins_are_ignored(planted):
    truth, pred, _, _, cells, nbl, out = planted
    valid = np.ones(M, bool); valid[::7] = False
    t2, p2 = truth.copy(), pred.copy()
    o1 = EB.error_budget(t2, p2, cells, nbl, valid=valid)
    t2[:, ~valid] = 1e3; p2[:, ~valid] = -1e3
    o2 = EB.error_budget(t2, p2, cells, nbl, valid=valid)
    for k in ("level", "shared", "specific", "explained", "noise", "added"):
        assert o2[k] == pytest.approx(o1[k], abs=1e-10)


def test_noise_share_from_halves_matches_spearman_brown():
    rng = np.random.default_rng(1)
    sig = rng.standard_normal((2, EB.BLOCK))                   # white signal, var 1; half noise var 1 => R = 0.5
    h1 = sig + rng.standard_normal((2, EB.BLOCK)); h2 = sig + rng.standard_normal((2, EB.BLOCK))
    share = EB.noise_share_from_halves(h1, h2)
    assert share.shape == (2, EB.K)
    # the fine levels hold most coefficients: share = 1 - 2 R / (1 + R) = 1/3 at R = 1/2
    assert share[:, :3].mean() == pytest.approx(1 / 3, abs=0.03)
