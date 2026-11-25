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
