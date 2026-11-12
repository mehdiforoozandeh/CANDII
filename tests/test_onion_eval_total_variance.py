"""Tests of the total-variance audit tool (plan Amendment 3, chunk TV2).

The simulator must reproduce its planted identity, and each check function must pass on consistent input and
flag a planted violation. No test asserts a defect of the audited tools (a fix chunk may repair them)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from onion_eval import error_budget as EB  # noqa: E402
from onion_eval import layers as LY  # noqa: E402
from onion_eval import total_variance as TV  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    return TV.simulate(TV.BASE_LAYERS, n_cells=12, seed=0)


def test_simulator_layer_sums_equal_planted_total(sim):
    P = sim["planted"]
    assert sum(P["terms"].values()) == pytest.approx(P["total"], rel=1e-12)       # closed form telescopes
    rt = TV.realized_terms(sim)
    assert rt["sum"] == pytest.approx(P["total"], rel=0.02)                       # own layer sums vs planted
    assert TV.empirical_total(sim) == pytest.approx(rt["sum"], rel=0.02)          # counts vs the latent chain
    assert rt["sampling"] == pytest.approx(P["terms"]["sampling"], rel=0.02)      # Poisson term = E[Y]


def test_simulator_pair_values_match_closed_form(sim):
    got = TV.pair_values(sim, "counts")
    for rung in TV.RUNGS:
        assert got[rung] == pytest.approx(sim["planted"]["pair_value"][rung], rel=0.06), rung
    assert {p["rung"] for p in sim["pairs"]} == set(LY.RUNGS)


def test_simulator_depth_and_crossing():
    s = TV.simulate(TV.BASE_LAYERS, {"experiment": [1.0, 2.0], "rung_b": {"technical": 2.0}}, n_cells=2,
                    seed=1)
    assert s["depth"][TV._leaf(0, 0, 1, 0, 0, 0)] == 2.0
    tech = [p for p in s["pairs"] if p["rung"] == "technical"]
    assert all(s["depth"][p["b"]] == 2 * s["depth"][p["a"]] for p in tech)
    a = TV.simulate(TV.BASE_LAYERS, n_cells=2, seed=3)
    b = TV.simulate(TV.BASE_LAYERS, {"scale": 0.5}, n_cells=2, seed=3, count_seed=99)
    leaf = TV._leaf(1, 1, 0, 1, 0, 1)
    np.testing.assert_allclose(b["rate"][leaf], 0.5 * a["rate"][leaf], rtol=1e-6)  # common latent rates


def _layers_rows(sim):
    ps = [p for p in sim["pairs"] if p["rung"] in ("pseudoreplicate", "sequencing", "technical", "isogenic")
          and p["cell"] in ("c0", "c1", "c2", "c3")]
    recs = TV._records(sim, ps, depth=0.5)
    for r in recs:
        r["assay"], r["assay_class"] = "A", "narrow_histone"
    return LY.assemble(recs, n_boot=50)


def test_telescoping_check_passes_and_flags(sim, tmp_path):
    rows = _layers_rows(sim)
    path = tmp_path / "layers.tsv"
    LY.write_tsv(rows, path)
    out = TV.telescoping_check(path)
    assert out and all(r["ok"] for r in out)
    assert {r["outermost"] for r in out} == {"isogenic"}
    bad = TV._read_layers(path)
    for r in bad:
        if r["rung"] == "technical" and r["band"] == "all":
            r["layer_increment"] = str(float(r["layer_increment"]) + 0.01)
    flagged = [r for r in TV.telescoping_check(bad) if not r["ok"]]
    assert flagged and all(r["group"].endswith("|all") for r in flagged)
    bad2 = TV._read_layers(path)
    for r in bad2:
        if r["rung"] == "isogenic":
            r["increment_over"] = "sequencing"
    assert not any(r["ok"] for r in TV.telescoping_check(bad2))


def _budget():
    rng = np.random.default_rng(0)
    M = 2 ** 14
    sig = rng.normal(size=M)
    truth = np.stack([sig + rng.normal(size=M) + 0.5 * rng.normal(size=M) for _ in range(4)])
    pred = np.stack([sig + 0.3 * rng.normal(size=M) for _ in range(4)])
    h1 = truth + rng.normal(size=truth.shape)
    h2 = truth + rng.normal(size=truth.shape)
    nbl = EB.noise_share_from_halves(h1, h2, 2.0)
    return EB.error_budget(truth, pred, np.array(["a", "a", "b", "b"]), noise_by_level=nbl)


def test_budget_check_passes_and_flags():
    bud = _budget()
    out = TV.budget_check(bud)
    assert len(out) == 4 and all(r["ok"] for r in out)
    scores = {"error_budget": {**{k: bud[k] for k in EB.PARTS + ("level", "explained", "unexplained", "noise",
                                                                    "added")},
                               "per_assay": {"H3K27ac": {k: bud[k] for k in ("level", "shared", "specific",
                                                                              "explained", "unexplained", "noise",
                                                                              "added")}}}}
    out = TV.budget_check(scores)
    assert {r["scope"] for r in out} == {"aggregate", "assay:H3K27ac"}
    assert all(r["ok"] for r in out if r["ok"] is not None)
    assert all(r["ok"] is None for r in out if r["identity"].startswith("level"))   # mse not in scores.json
    bad = dict(bud, noise=bud["noise"] + 0.05)
    assert not all(r["ok"] for r in TV.budget_check(bad) if r["ok"] is not None)
    bad = dict(bud, level=bud["level"] + 0.05)
    assert [r["ok"] for r in TV.budget_check(bad) if r["identity"].startswith("level")] == [False]


def test_per_rung_variance_ratio_calibrated_and_flags():
    rng = np.random.default_rng(4)
    M, n = 20000, 2.0
    mu = rng.gamma(2.0, 2.0, M)

    def nb(m):
        return rng.poisson(rng.gamma(n, m / n)).astype(float)
    # pairs drawn independently from the predictive law: calibrated at that rung
    pairs = {"technical": [(nb(mu), nb(mu)) for _ in range(3)],
             "pseudoreplicate": [(rng.poisson(mu), rng.poisson(mu))]}
    got = TV.per_rung_variance_ratio(np.full(M, n), mu, pairs)
    assert got["technical"]["ratio"] == pytest.approx(1.0, abs=0.05) and got["technical"]["n_pairs"] == 3
    # the planted violation: Poisson replicates are less variable than the predicted NB
    assert got["pseudoreplicate"]["ratio"] > 1.5
    # halves at half depth enter with mu_scale 0.5 (score.py:836)
    full = rng.poisson(mu)
    h1 = rng.binomial(full, 0.5)
    got = TV.per_rung_variance_ratio(np.full(M, 1e6), mu, [dict(rung="pseudoreplicate", a=h1, b=full - h1,
                                                                mu_scale=0.5)])
    assert got["pseudoreplicate"]["ratio"] == pytest.approx(1.0, abs=0.05)
