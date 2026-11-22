"""t370 ens: seed-ensemble moment matching, dispersion recalibration and its fit, and the combined
prediction store, checked against the formulas of the brief:
mean = mean of the means; variance = mean of the variances + variance of the means; n' = n / c."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from onion_eval import ensemble_store as ES  # noqa: E402
from onion_eval import predict_store as P  # noqa: E402


def nb_var(n, mu):
    return mu + mu ** 2 / n


def test_moment_match_mean_and_variance():
    rng = np.random.default_rng(1)
    ns = rng.uniform(0.2, 30, size=(3, 500))
    mus = rng.uniform(0.01, 50, size=(3, 500))
    n, m = ES.moment_match(ns, mus)
    want_m = mus.mean(0)
    want_v = nb_var(ns, mus).mean(0) + ((mus - want_m) ** 2).mean(0)
    np.testing.assert_allclose(m, want_m, rtol=1e-12)
    np.testing.assert_allclose(nb_var(n, m), want_v, rtol=1e-10)
    p = n / (n + m)                      # back to (n, p): the NB mean n (1 - p) / p is m
    np.testing.assert_allclose(n * (1 - p) / p, m, rtol=1e-10)


def test_moment_match_one_member_and_identical_members():
    ns = np.array([[2.0, 5.0]])
    mus = np.array([[3.0, 0.5]])
    n, m = ES.moment_match(ns, mus)
    np.testing.assert_allclose(n, ns[0])
    np.testing.assert_allclose(m, mus[0])
    n3, m3 = ES.moment_match(np.repeat(ns, 3, 0), np.repeat(mus, 3, 0))
    np.testing.assert_allclose(n3, ns[0], rtol=1e-12)
    np.testing.assert_allclose(m3, mus[0])


def test_moment_match_widens_with_disagreeing_means():
    ns = np.full((3, 1), 4.0)
    mus = np.array([[1.0], [5.0], [9.0]])
    n, m = ES.moment_match(ns, mus)
    assert nb_var(n, m)[0] > nb_var(ns, mus).mean()      # the spread of the means adds variance


def test_recalibrate_scales_the_dispersion():
    n, mu, c = np.array([2.0, 10.0]), np.array([4.0, 1.0]), 1.7
    n2 = ES.recalibrate(n, c)
    np.testing.assert_allclose(1 / n2, c / n)
    np.testing.assert_allclose(nb_var(n2, mu), mu + c * mu ** 2 / n)
    with pytest.raises(ValueError):
        ES.recalibrate(n, 0.0)


@pytest.mark.parametrize("c_true", [1.0, 3.0, 0.5])
def test_fit_factor_recovers_the_dispersion_scale(c_true):
    rng = np.random.default_rng(7)
    mu = rng.gamma(2.0, 3.0, size=60000)
    n_true = np.full(mu.size, 2.0)
    y = rng.negative_binomial(n_true, n_true / (n_true + mu)).astype(float)
    n_model = n_true * c_true            # dispersion off by 1 / c_true; the fit should return c_true
    r = ES.fit_factor(n_model, mu, y)
    assert abs(np.log2(r["factor"]) - np.log2(c_true)) < 0.15
    assert r["pit_at_factor"] <= r["pit_at_1"] + 1e-12
    assert not r["at_bound"]


def _store(d: Path, run: str, reqs, arrays, ckpt="a" * 64):
    d.mkdir(parents=True)
    head = {"format": 1, "run": run, "run_tables_sha256": "t", "evalset_sha256": "e",
            "tables_sha256": "t", "device": "cpu", "infer_batch": 8, "tf32": False, "amp": False,
            "n_shards": 2, "region_bins": 4, "n_requests_total": len(reqs),
            "requests_sha256": P.requests_sha256(reqs), P.CKPT_KEY: ckpt}
    (d / "STORE.json").write_text(json.dumps(head))
    for i in range(2):
        (d / f"PREDICTED_{i}of2.json").write_text(json.dumps(
            {k: head.get(k) for k in P.IDENTITY_KEYS + P.SETTING_KEYS}))
    (d / "guardrail.json").write_text(json.dumps({"copy_leak_pass": True}))
    for r in reqs:
        np.savez(d / P.key_file(r["key"]), key=np.array(r["key"]), **arrays[r["key"]])
    return head


def _reqs():
    return [{"key": "item/T1/leave_one_assay_out", "region_indices": [0, 1], "needs_ctl": True},
            {"key": "item/T1/accessibility_only", "region_indices": [0, 1], "needs_ctl": False},
            {"key": "pair/p1", "region_indices": [0], "needs_ctl": False}]


def _members(tmp_path, k=3):
    rng = np.random.default_rng(3)
    reqs = _reqs()
    dirs, arrs = [], []
    for s in range(k):
        a = {}
        for r in reqs:
            sh = (len(r["region_indices"]), 4)
            a[r["key"]] = {"n": rng.uniform(0.5, 9, sh).astype(np.float32),
                           "mu": rng.uniform(0.1, 9, sh).astype(np.float32)}
            if r["needs_ctl"]:
                a[r["key"]]["ctl_mu"] = rng.uniform(0.1, 2, sh).astype(np.float32)
        d = tmp_path / f"seed{s}"
        _store(d, f"/runs/r/seed{s}", reqs, a, ckpt=str(s) * 64)
        dirs.append(d)
        arrs.append(a)
    return reqs, dirs, arrs


def _factors(tmp_path, c):
    f = tmp_path / "factors.json"
    f.write_text(json.dumps({"regimes": ["leave_one_assay_out"], "target_assay": {"T1": "A"},
                             "factor_sets": {"ensemble": {"A": {"factor": c}},
                                             "seed0": {"A": {"factor": c}}}}))
    return f


def test_combine_ensemble_with_recalibration(tmp_path):
    reqs, dirs, arrs = _members(tmp_path)
    out = tmp_path / "ens3_recal_p16_sv"
    c = 2.0
    info = ES.combine(dirs, out, factors=_factors(tmp_path, c), factor_set="ensemble",
                      record=tmp_path / "rec" / "ENSEMBLE.json", log=lambda *m: None)
    st = P.PredictionStore(out, run="/runs/r/seed0", run_tables_sha256="t", evalset_sha256="e",
                           tables_sha256="t")
    assert st.complete(reqs)
    head = st.info
    assert head["run"] == "/runs/r/seed0" and head[P.CKPT_KEY] == "0" * 64
    assert len(head["ensemble"]["members"]) == 3
    for r in reqs:
        got = st.read(r["key"])
        ns = np.stack([a[r["key"]]["n"] for a in arrs]).astype(np.float64)
        ms = np.stack([a[r["key"]]["mu"] for a in arrs]).astype(np.float64)
        m = ms.mean(0)
        v = nb_var(ns, ms).mean(0) + ms.var(0)
        n = m ** 2 / (v - m)
        if r["key"] == "item/T1/leave_one_assay_out":
            n = n / c                       # recalibrated: this regime only
        np.testing.assert_allclose(got["mu"], m, rtol=1e-6)
        np.testing.assert_allclose(got["n"], n, rtol=1e-5)
        if r["needs_ctl"]:
            np.testing.assert_allclose(
                got["ctl_mu"], np.mean([a[r["key"]]["ctl_mu"] for a in arrs], 0), rtol=1e-6)
    assert info["n_requests_recalibrated"] == 1
    assert (tmp_path / "rec" / "ENSEMBLE.json").is_file()
    assert (out / "guardrail.json").is_file()
    again = ES.combine(dirs, out, log=lambda *m: None)          # a finished store is skipped
    assert again["n_requests"] == 3


def test_combine_single_member_recalibration_leaves_others_bit_identical(tmp_path):
    reqs, dirs, arrs = _members(tmp_path, k=1)
    out = tmp_path / "seed0_recal_p16_sv"
    ES.combine(dirs, out, factors=_factors(tmp_path, 0.5), factor_set="seed0", log=lambda *m: None)
    st = P.PredictionStore(out)
    for r in reqs:
        got, src = st.read(r["key"]), arrs[0][r["key"]]
        assert np.array_equal(got["mu"], src["mu"])
        if r["key"] == "item/T1/leave_one_assay_out":
            np.testing.assert_allclose(got["n"], src["n"].astype(np.float64) / 0.5, rtol=1e-7)
        else:
            assert np.array_equal(got["n"], src["n"])


def test_combine_refuses_mismatched_members_and_used_dirs(tmp_path):
    reqs, dirs, arrs = _members(tmp_path, k=2)
    h = json.loads((dirs[1] / "STORE.json").read_text())
    h["requests_sha256"] = "other"
    (dirs[1] / "STORE.json").write_text(json.dumps(h))
    with pytest.raises(ValueError, match="requests_sha256"):
        ES.combine(dirs, tmp_path / "x", log=lambda *m: None)
    used = tmp_path / "used"
    used.mkdir()
    (used / "junk").write_text("x")
    with pytest.raises(ValueError, match="not a finished"):
        ES.combine(dirs[:1], used, log=lambda *m: None)
    (dirs[0] / "PREDICTED_1of2.json").unlink()
    with pytest.raises(ValueError, match="not predicted"):
        ES.combine(dirs[:1], tmp_path / "y", log=lambda *m: None)


def test_combine_refuses_a_target_without_a_factor(tmp_path):
    reqs, dirs, arrs = _members(tmp_path, k=1)
    f = tmp_path / "factors.json"
    f.write_text(json.dumps({"regimes": ["leave_one_assay_out"], "target_assay": {"T9": "A"},
                             "factor_sets": {"seed0": {"A": {"factor": 2.0}}}}))
    with pytest.raises(ValueError, match="no dispersion factor"):
        ES.combine(dirs, tmp_path / "z", factors=f, factor_set="seed0", log=lambda *m: None)
