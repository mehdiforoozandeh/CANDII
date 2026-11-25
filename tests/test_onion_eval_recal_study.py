"""t370 recalstudy: the analyze conditions of recal_study.py on synthetic forwards (none,
same_cells, 2-fold cross-cell, shipped), and the pure helpers of its forward pass."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from onion_eval import ensemble_store as ES  # noqa: E402
from onion_eval import recal_study as R  # noqa: E402

K = 3
B = 3000
#: model n is the true n times this per cell, so the factor that calibrates a cell is this value
CELL_C = {"c0": 4.0, "c1": 1.0, "c2": 4.0, "c3": 1.0}


def _item(rng, kind, assay, cell, tag):
    mu = rng.gamma(2.0, 3.0, size=B)
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(np.float32)
    n = np.stack([np.full(B, 2.0 * CELL_C[cell] * (1.0 + 0.1 * k)) for k in range(K)])
    target = f"{assay}_{cell}_{tag}"
    key = f"item/{target}/leave_one_assay_out" if kind == "imputation" else f"pair/{target}"
    return {"kind": kind, "key": key, "cell": cell, "assay": assay, "target": target,
            "source": None if kind == "imputation" else "src", "regime": None,
            "y": y, "n": n.astype(np.float32), "mu": np.repeat(mu[None], K, 0).astype(np.float32)}


def _sets(seed=0):
    rng = np.random.default_rng(seed)
    fit = [_item(rng, "imputation", "A", c, "f") for c in CELL_C]
    fit += [_item(rng, "imputation", "B", c, "f") for c in ("c0", "c2")]     # fold 0 only
    fit += [_item(rng, "pair", "A", c, "f") for c in CELL_C]
    score = [_item(rng, "imputation", "A", c, "s") for c in CELL_C]
    score += [_item(rng, "imputation", "B", c, "s") for c in ("c0", "c1")]
    score += [_item(rng, "pair", "A", c, "s") for c in CELL_C]
    return fit, score


@pytest.fixture(scope="module")
def study():
    fit, score = _sets()
    return fit, score, R.analyze_items(fit, score, K, max_bins=100000)


def test_none_is_the_raw_pooled_pit(study):
    fit, score, res = study
    its = [it for it in score if it["kind"] == "imputation" and it["assay"] == "A"]
    y, n, mu, _ = R.pool(its)
    _, idx = ES.fit_index(y, n, mu, 100000)
    for k in range(K):
        want = ES.pit_dev(n[k, idx], mu[k, idx], y[idx])
        assert res["imputation"]["conditions"]["none"]["per_seed"][k]["per_assay"]["A"] == want


def test_cross_cell_fits_on_the_other_fold_only(study):
    fit, score, res = study
    r = res["imputation"]
    assert r["folds"] == {"c0": 0, "c1": 1, "c2": 0, "c3": 1}
    f0 = r["factors"]["cross_cell"]["fitted_on_fold0"]["A"]
    f1 = r["factors"]["cross_cell"]["fitted_on_fold1"]["A"]
    for k in range(K):     # fold 0 cells need c = 4 (times the seed's own 1 + 0.1 k), fold 1 c = 1
        assert abs(np.log2(f0[k]["factor"]) - np.log2(4.0 * (1 + 0.1 * k))) < 0.3
        assert abs(np.log2(f1[k]["factor"]) - np.log2(1.0 * (1 + 0.1 * k))) < 0.3
    # the cross-cell PIT, recomputed: cells of fold 0 take fold 1's factor and the reverse
    its = [it for it in score if it["kind"] == "imputation" and it["assay"] == "A"]
    y, n, mu, pos = R.pool(its)
    _, idx = ES.fit_index(y, n, mu, 100000)
    for k in range(K):
        per_item = np.array([(f1 if r["folds"][it["cell"]] == 0 else f0)[k]["factor"] for it in its])
        want = ES.pit_dev(n[k, idx] / per_item[pos[idx]], mu[k, idx], y[idx])
        got = r["conditions"]["cross_cell"]["per_seed"][k]["per_assay"]["A"]
        assert got == pytest.approx(want, abs=1e-12)


def test_same_cells_factor_is_fitted_on_all_fit_cells(study):
    fit, score, res = study
    its = [it for it in fit if it["kind"] == "imputation" and it["assay"] == "A"]
    y, n, mu, _ = R.pool(its)
    _, idx = ES.fit_index(y, n, mu, 100000)
    for k in range(K):
        want = ES.fit_factor(n[k, idx], mu[k, idx], y[idx])["factor"]
        assert res["imputation"]["factors"]["same_cells"]["A"][k]["factor"] == want
        assert "at_bound" in res["imputation"]["factors"]["same_cells"]["A"][k]


def test_missing_assay_in_the_other_fold_falls_back_to_one(study):
    fit, score, res = study
    r = res["imputation"]
    fb = r["fallback_c1"]["cross_cell"]
    assert [(x["assay"], x["cell"], x["fit_fold"]) for x in fb] == [("B", "c0", 1)]
    # B on c0 (c = 1) and on c1 (fold 0's factor), recomputed
    its = [it for it in score if it["kind"] == "imputation" and it["assay"] == "B"]
    y, n, mu, pos = R.pool(its)
    _, idx = ES.fit_index(y, n, mu, 100000)
    f0 = r["factors"]["cross_cell"]["fitted_on_fold0"]["B"]
    assert "B" not in r["factors"]["cross_cell"]["fitted_on_fold1"]
    for k in range(K):
        per_item = np.array([1.0, f0[k]["factor"]])
        want = ES.pit_dev(n[k, idx] / per_item[pos[idx]], mu[k, idx], y[idx])
        assert r["conditions"]["cross_cell"]["per_seed"][k]["per_assay"]["B"] == pytest.approx(want)


def test_shipped_uses_the_pair_factor_and_only_on_imputation(study):
    fit, score, res = study
    assert "shipped" in res["imputation"]["conditions"]
    assert "shipped" not in res["pair"]["conditions"]
    assert res["imputation"]["factors"]["shipped"]["A"] == res["pair"]["factors"]["same_cells"]["A"]
    assert [x["assay"] for x in res["imputation"]["fallback_c1"]["shipped"]] == ["B"]
    s = res["imputation"]["conditions"]["shipped"]["per_seed"][0]["per_assay"]["B"]
    assert s == res["imputation"]["conditions"]["none"]["per_seed"][0]["per_assay"]["B"]


def test_summary_mean_range_and_mu_unchanged(study):
    fit, score, res = study
    for kind, r in res.items():
        assert r["mu_unchanged"] is True
        for c, v in r["conditions"].items():
            ps = [s["mean_over_assays"] for s in v["per_seed"]]
            assert v["mean_over_assays"]["mean"] == pytest.approx(np.mean(ps))
            assert v["mean_over_assays"]["range"] == pytest.approx(max(ps) - min(ps))
            for a, s in v["per_assay"].items():
                vals = [p["per_assay"][a] for p in v["per_seed"]]
                assert s["range"] == pytest.approx(max(vals) - min(vals))
    assert res["imputation"]["bins"]["A"]["items"] == 4
    assert res["imputation"]["bins"]["A"]["used"] == 4 * B


def test_apply_factor_leaves_mu_and_refuses_nonpositive():
    n, mu = np.array([2.0, 4.0]), np.array([1.0, 3.0])
    n2, m2 = R.apply_factor(n, mu, np.array([2.0, 0.5]))
    assert m2 is mu
    np.testing.assert_allclose(n2, [1.0, 8.0])
    with pytest.raises(ValueError):
        R.apply_factor(n, mu, 0.0)


# ------------------------------------------------------------------------------------ forward helpers

def test_calibration_evalset_and_keys():
    ev = {"regions": [["chr1", 0]], "pair_regions": [0], "pair_regions_p": 0.1,
          "strata": ["x"], "region_bins": 4, "correction_pairs": [{"pair_id": "p"}]}
    cal = R.calibration_evalset(ev, [("chr2", 8), ("chr3", 16)], {"n_regions": 2})
    assert cal["regions"] == [["chr2", 8], ["chr3", 16]]
    assert "pair_regions" not in cal and "pair_regions_p" not in cal and cal["strata"] == []
    assert cal["correction_pairs"] == ev["correction_pairs"] and ev["pair_regions"] == [0]
    assert R.item_key("imputation", {"target": "T", "regime": "leave_one_assay_out"}) == \
        "item/T/leave_one_assay_out"
    assert R.item_key("pair", {"pair_id": "p7", "source": "S", "target": "T"}) == "pair/p7"
    assert R.item_key("pair", {"source": "S", "target": "T"}) == "pair/S>T"


def test_write_read_item_and_analyze_cli(tmp_path):
    fit, score = _sets(seed=5)
    for name, items in (("fit", fit), ("score", score)):
        d = tmp_path / name
        metas = []
        for it in items:
            meta = {k: it[k] for k in ("kind", "key", "cell", "assay", "target", "source", "regime")}
            f = R.write_item(d, meta, it["y"], it["n"], it["mu"])
            back = R.read_item(f)
            assert back["key"] == it["key"] and np.array_equal(back["n"], it["n"])
            metas.append(dict(meta, file=f.name))
        (d / R.INDEX).write_text(json.dumps({
            "runs": ["/r/s0", "/r/s1", "/r/s2"], "regions": [[name, 0]], "items": metas,
            "pair_target_assay": {it["key"][5:]: "A" for it in items if it["kind"] == "pair"}}))
    prefix = tmp_path / "out" / "study"
    assert R.analyze(["--fit", str(tmp_path / "fit"), "--score", str(tmp_path / "score"),
                      "--out", str(prefix), "--max-bins", "5000"]) == 0
    doc = json.loads((tmp_path / "out" / "study.json").read_text())
    assert set(doc["kinds"]) == {"imputation", "pair"} and doc["n_seeds"] == 3
    assert doc["kinds"]["imputation"]["bins"]["A"]["used"] == 5000
    assert "| none |" in (tmp_path / "out" / "study.md").read_text()
    pf = json.loads((tmp_path / "out" / "study.pair_factors.json").read_text())
    assert sorted(pf["factor_sets"]) == ["ensemble", "seed0", "seed1", "seed2"]
    assert len(pf["pair_target_assay"]) == 8           # fit and score pair keys, all assay A
    with pytest.raises(ValueError):
        R.write_item(tmp_path / "bad", {"key": "x"}, np.zeros(3), np.zeros((2, 4)), np.zeros((2, 4)))


# ------------------------------------------------------------------------------------ coverage

def _nb_items(rng, n_true, n_model, mu_scale=50.0, cells=("c0", "c1"), assay="A", kind="imputation",
              bins=20000, tag="s"):
    """Items with y ~ NB(n_true, mu) and a model that predicts (n_model, mu) for every seed."""
    out = []
    for cell in cells:
        mu = rng.gamma(2.0, mu_scale / 2.0, size=bins)
        y = rng.negative_binomial(n_true, n_true / (n_true + mu)).astype(np.float32)
        target = f"{assay}_{cell}_{tag}"
        out.append({"kind": kind, "key": f"item/{target}/leave_one_assay_out", "cell": cell,
                    "assay": assay, "target": target, "source": None, "regime": None, "y": y,
                    "n": np.full((K, bins), n_model, np.float32),
                    "mu": np.repeat(mu[None], K, 0).astype(np.float32)})
    return out


def _expected_coverage(n, mu, level):
    """Mean over bins of P(lower <= Y <= upper) under the model's own NB."""
    from scipy.stats import nbinom
    p = n / (n + mu)
    lo = nbinom.ppf((1 - level) / 2, n, p)
    hi = nbinom.ppf((1 + level) / 2, n, p)
    return float(np.mean(nbinom.cdf(hi, n, p) - np.where(lo > 0, nbinom.cdf(lo - 1, n, p), 0.0)))


def test_interval_rule_matches_scipy_ppf():
    from scipy.stats import nbinom
    rng = np.random.default_rng(1)
    n = rng.gamma(2.0, 1.0, 20000) + 0.05
    mu = rng.gamma(1.0, 20.0, 20000)
    y = rng.negative_binomial(n, n / (n + mu)).astype(np.float64)
    fy, fm = R.nb_cdf_pair(n, mu, y)
    tab = R.interval_table(fy, fm, {"all": np.ones(y.size, bool)})["all"]
    for L in R.LEVELS:
        p = n / (n + mu)
        lo, hi = nbinom.ppf((1 - L) / 2, n, p), nbinom.ppf((1 + L) / 2, n, p)
        t = tab[f"{L:.2f}"]
        assert t["coverage"] == np.mean((lo <= y) & (y <= hi))
        assert t["frac_above"] == np.mean(y > hi) and t["frac_below"] == np.mean(y < lo)
        assert t["coverage"] + t["frac_above"] + t["frac_below"] == pytest.approx(1.0, abs=1e-12)


def test_calibrated_model_covers_its_expected_level():
    rng = np.random.default_rng(2)
    its = _nb_items(rng, 3.0, 3.0, bins=30000)
    same = {"A": [{"factor": 1.0}] * K}
    r = R.coverage_assay("A", its, K, {"c0": 0, "c1": 1}, same, {0: same, 1: same})
    y, n, mu, _ = R.pool(its)
    assert r["mu_unchanged"] is True and r["bins"]["used"] == 60000
    for L in R.LEVELS:
        want = _expected_coverage(n[0], mu[0], L)
        se = np.sqrt(want * (1 - want) / y.size)
        for c in R.COV_CONDITIONS:
            got = r["tables"][c][0]["all"][f"{L:.2f}"]["coverage"]
            assert abs(got - want) < 4 * se
            assert want >= L - 1e-9           # a discrete central interval covers at least L
            assert want - L < 0.06            # mu ~ 50: little excess from discreteness


def test_too_narrow_n_undercovers_and_recalibration_restores():
    rng = np.random.default_rng(3)
    n_true, n_model = 2.0, 16.0           # the model's dispersion 1/n is 8 times too small
    its = _nb_items(rng, n_true, n_model, bins=30000)
    good = {"A": [{"factor": n_model / n_true}] * K}
    r = R.coverage_assay("A", its, K, {"c0": 0, "c1": 1}, good, {0: good, 1: good})
    y, n, mu, _ = R.pool(its)
    for L in R.LEVELS:
        lk = f"{L:.2f}"
        raw = r["tables"]["none"][0]["all"][lk]
        assert raw["coverage"] < L - 0.05
        want = _expected_coverage(n[0] / (n_model / n_true), mu[0], L)
        se = np.sqrt(want * (1 - want) / y.size)
        for c in ("same_cells", "cross_cell"):
            assert abs(r["tables"][c][0]["all"][lk]["coverage"] - want) < 4 * se
    assert r["mu_unchanged"] is True


def test_stratum_counts_follow_the_quantile_rule():
    rng = np.random.default_rng(4)
    its = _nb_items(rng, 1.0, 1.0, mu_scale=3.0, bins=5000)    # small counts: many ties
    one = {"A": [{"factor": 1.0}] * K}
    r = R.coverage_assay("A", its, K, {"c0": 0, "c1": 1}, one, {0: one, 1: one})
    y, n, mu, _ = R.pool(its)
    for (t, q) in R.TOPS:
        thr = np.quantile(y, q)
        assert r["obs_cuts"][f"obs_{t}"] == thr
        for k in range(K):
            tab = r["tables"]["none"][k]
            assert tab[f"obs_{t}"]["0.90"]["n_bins"] == int((y >= thr).sum())
            assert tab[f"pred_{t}"]["0.90"]["n_bins"] == int((mu[k] >= np.quantile(mu[k], q)).sum())
        assert tab["all"]["0.50"]["n_bins"] == y.size
    assert r["tables"]["none"][0]["obs_top10"]["0.90"]["n_bins"] > 0.1 * y.size   # ties kept


def test_coverage_cli_reuses_analyze_factors(tmp_path):
    fit, score = _sets(seed=7)
    for name, items in (("fit", fit), ("score", score)):
        d = tmp_path / name
        metas = []
        for it in items:
            meta = {k: it[k] for k in ("kind", "key", "cell", "assay", "target", "source", "regime")}
            metas.append(dict(meta, file=R.write_item(d, meta, it["y"], it["n"], it["mu"]).name))
        (d / R.INDEX).write_text(json.dumps({"runs": ["/r/s0", "/r/s1", "/r/s2"],
                                             "regions": [[name, 0]], "items": metas}))
    an = R.analyze_items(fit, score, K, 5000)
    for workers in (1, 2):
        prefix = tmp_path / f"out{workers}" / "cov"
        assert R.coverage(["--fit", str(tmp_path / "fit"), "--score", str(tmp_path / "score"),
                           "--out", str(prefix), "--max-bins", "5000",
                           "--workers", str(workers)]) == 0
        doc = json.loads(prefix.with_name("cov.json").read_text())
        for kind in ("imputation", "pair"):
            r = doc["kinds"][kind]
            assert r["mu_unchanged"] is True
            assert r["factors"] == json.loads(json.dumps({
                k: an[kind]["factors"][k] for k in ("same_cells", "cross_cell")}))
            assert r["fallback_c1"]["cross_cell"] == an[kind]["fallback_c1"]["cross_cell"]
            for c in R.COV_CONDITIONS:
                for st in R.STRATA:
                    for lk in doc["levels"]:
                        m = r["coverage"][c][st][lk]["mean_over_assays"]
                        tot = [m[x]["per_seed"] for x in R.COV_METRICS]
                        assert np.allclose(np.sum(tot, axis=0), 1.0)
        assert doc["kinds"]["imputation"]["bins"]["A"]["used"] == 4 * B     # no subsample
        if workers == 1:
            first = doc["kinds"]
        else:
            assert doc["kinds"] == first
    md = (tmp_path / "out1" / "cov.md").read_text()
    assert "lower <= y <= upper" in md and "| obs_top1 |" in md and "| A |" in md
