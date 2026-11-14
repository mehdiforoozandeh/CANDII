"""OF (PI ruling 47): the oracle floor of the correction gap closed.

Synthetic tables scored with the training-free baselines, then `oracle_floor.py compute` beside
them. Checks: depth-only scaling scores exactly 0 on the new scale; a forecast equal to the true
sampling distribution scores 1 in expectation (synthetic Poisson); the reference reproduces the
scorer's bins and pair order; report-time reading in select_rule and arm_report (composite version
2 with a reference, version 1 and the old output without one, a mix refused); a score without
per-pair records gives an undefined new metric with a reason, never the old value."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from onion_eval import oracle_floor as OF  # noqa: E402
from onion_eval import select_rule as R  # noqa: E402

VAL = ((0, "H3K4me3"), (1, "H3K4me3"), (2, "H3K4me3"), (3, "H3K27ac"), (4, "H3K27ac"),
       (5, "H3K27ac"), (7, "chipseq-control"))
CFG = {"context_bins": 128, "batch_size": 4}
KW = dict(n_boot=20, guardrail_windows=4, log=lambda *a: None)


# ------------------------------------------------------------------------------------------ pure math

def _poisson_crps(lam, y, kmax=200):
    """Exact CRPS of Poisson(lam) at integer y: sum_k (F(k) - 1{y <= k})^2."""
    k = np.arange(kmax)[None, :]
    F = st.poisson.cdf(k, lam[:, None])
    return ((F - (y[:, None] <= k)) ** 2).sum(1)


def test_true_distribution_scores_one_in_expectation():
    rng = np.random.default_rng(0)
    n = 200_000
    lam = rng.gamma(2.0, 2.0, n)                    # latent rate per bin
    y1, y2, y = (rng.poisson(lam) for _ in range(3))
    oracle = 0.5 * np.abs(y1 - y2).mean()
    true = _poisson_crps(lam, y).mean()
    worse = _poisson_crps(np.full(n, lam.mean()), y).mean()     # a depth-only stand-in
    g = (worse - true) / (worse - oracle)
    assert abs(g - 1.0) < 0.02, g
    # the old half-read floor (mean = the other half) sits above the oracle
    half = _poisson_crps(np.maximum(y1, 1e-3).astype(float), y).mean()
    assert half > oracle * 1.2


def _recs(vals):
    return [{"assay": a, "cell": c, "knob": k, "never_trained": 1, "subset_pair": 0,
             "nodedup_side": nd, "crps_model": m, "crps_depth_only": d, "crps_floor": f,
             "oracle": o} for a, c, k, nd, m, d, f, o in vals]


def test_aggregation_is_macro_of_per_assay_ratios_with_numerator_and_denominator():
    rs = _recs([("A", "c1", "crop", 0, 0.5, 1.0, 0.9, 0.0),
                ("A", "c2", "mapq+nodedup", 1, 0.5, 1.0, 0.9, 0.5),
                ("B", "c1", "crop", 0, 0.9, 1.0, 0.95, 0.8)])
    blk = OF.gap_closed_block(rs, ["A", "B"], n_boot=0)
    a = blk["all"]
    # A: (1 - .5) / (1 - .25) = 2/3; B: .1 / .2 = .5; macro .5833
    assert a["per_assay"] == pytest.approx({"A": 2 / 3, "B": 0.5})
    assert a["gap_closed_oracle"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert a["numerator"] == pytest.approx((0.5 + 0.1) / 2)
    assert a["denominator"] == pytest.approx((0.75 + 0.2) / 2)
    assert a["gap_closed_halfread_superseded"] == pytest.approx((0.5 / 0.1 + 0.1 / 0.05) / 2)
    b = blk["no_nodedup_side"]
    assert b["n_pairs"] == 2 and b["per_assay"] == pytest.approx({"A": 0.5, "B": 0.5})
    assert blk["all"]["per_knob"]["crop"]["value"] == pytest.approx((1.0 - 0.7) / (1.0 - 0.4))
    # the depth-only forecast is exactly 0
    d = OF.gap_closed_block([dict(r, crps_model=r["crps_depth_only"]) for r in rs], ["A", "B"],
                            n_boot=0)
    assert d["all"]["gap_closed_oracle"] == 0.0 and d["no_nodedup_side"]["gap_closed_oracle"] == 0.0


# ------------------------------------------------------------------------------------------ end to end

@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from minicandi.synth import make_synthetic_tables
    from onion_eval import evalset as E
    from onion_eval import score as S
    root = tmp_path_factory.mktemp("oracle")
    t = make_synthetic_tables(root, n_cells=6, val_tracks=VAL, blind_tracks=())
    out = root / "eval" / "synth" / "evalset.json"
    ev = E.build(t, out, n_regions=2, n_fit_regions=1, region_bins=256, seed=0)
    nt = [p for p in ev["correction_pairs"] if p["never_trained"] and not p["subset_pair"]][:12]
    ev["correction_pairs"] = nt
    out.write_text(json.dumps(ev))
    bdir = root / "eval" / "synth" / "baselines"
    res = {n: S.score(out, bdir / n / "seed0", baseline=n, config=CFG, **KW)
           for n in ("position_average", "depth_only_scaling", "pseudoreplicate_floor")}
    return {"root": root, "evalset": out, "ev": ev, "bdir": bdir, "res": res}


@pytest.fixture(scope="module")
def oracle(env):
    d = env["bdir"] / "oracle_floor" / "seed0"
    info = OF.compute(env["evalset"], d, log=lambda *a: None)
    return {"dir": d, "info": info}


def test_compute_matches_the_scorer_layout(env, oracle):
    o = OF.load_oracle(oracle["dir"])
    with np.load(env["bdir"] / "depth_only_scaling" / "seed0" / "pairs_per_region.npz") as z:
        assert list(o["pair_ids"]) == list(z["pair_ids"])
        assert list(o["region_indices"]) == list(z["region_indices"])
        assert np.array_equal(o["bins"], z["bins"])
    assert oracle["info"]["n_correction_pairs"] == len(env["ev"]["correction_pairs"])
    assert np.isfinite(o["crps_sum_oracle"]).all() and (o["crps_sum_oracle"] >= 0).all()
    assert set(np.unique(o["nodedup_side"])) <= {0, 1}
    # the oracle is half the mean |pr1 - pr2| of the target, read from the halves directly
    import h5py
    from minicandi.dataset import OnionDataset  # noqa: F401  (the scorer's loader built it)
    tables = Path(env["ev"]["tables_dir"]) if Path(env["ev"]["tables_dir"]).is_absolute() else None
    if tables is not None:
        import csv
        rows = {r["measurement_id"]: r for r in csv.DictReader(
            open(tables / "measurements.tsv"), delimiter="\t")}
        t = str(o["target"][0])
        p = Path(rows[t]["counts_path"][3:]).parent
        c, s = env["ev"]["regions"][int(o["region_indices"][0])]
        with h5py.File(p / "pr1.h5") as f1, h5py.File(p / "pr2.h5") as f2:
            a = np.zeros(256); b = np.zeros(256)
            e = min(int(s) + 256, f1[c].shape[0])
            a[: e - int(s)] = f1[c][int(s):e]; b[: e - int(s)] = f2[c][int(s):e]
        assert o["crps_sum_oracle"][0, 0] == pytest.approx(0.5 * np.abs(a - b).sum())


def test_depth_only_scores_exactly_zero_and_found_by_discovery(env, oracle):
    p = env["bdir"] / "depth_only_scaling" / "seed0" / "scores.json"
    s = json.loads(p.read_text())
    assert OF.find_oracle(p, s) == oracle["dir"]
    t = OF.seed_terms(p, s)
    assert t["composite_version"] == 2
    blk = t["block"]["all"]
    if blk["gap_closed_oracle"] is not None:
        assert blk["gap_closed_oracle"] == 0.0
        assert all(v == 0.0 for v in blk["per_assay"].values())
    else:                                  # every scorable assay needs a pair (require_all)
        assert "scorable assay with no pair" in blk["reason"]
        assert all(v == 0.0 for v in blk["per_assay"].values() if v is not None)
    assert all(v["value"] == 0.0 for v in blk["per_knob"].values())


def test_select_rule_reads_v2_and_keeps_v1_without_reference(env, oracle):
    p = env["bdir"] / "position_average" / "seed0" / "scores.json"
    v2 = R.load_seed(p)
    v1 = R.load_seed(p, oracle=None)
    assert v2["composite_version"] == 2 and v1["composite_version"] == 1
    assert v2["scorer_keys"]["composite_version"] == 2
    assert v2["scorer_keys"]["gap_closed_floor"] == "oracle"
    assert "composite_version" not in v1["scorer_keys"]
    old = json.loads(p.read_text())["selection"]["correction_gap_closed"]
    assert v1["goals"]["correction"] == R._num(old)
    assert v2["correction_halfread_superseded"] == R._num(old)
    t = OF.seed_terms(p, json.loads(p.read_text()))
    assert v2["goals"]["correction"] == t["value"]
    with pytest.raises(R.Refused, match="composite_version"):
        R.check_scorer_keys([(v1["path"], v1["scorer_keys"]), (v2["path"], v2["scorer_keys"])])
    with pytest.raises(R.Refused, match="composite"):
        R.check_composite_versions([v1, v2])


def test_missing_records_give_undefined_with_reason(env, oracle, tmp_path):
    src = env["bdir"] / "position_average" / "seed0"
    d = env["bdir"] / "position_average_norecords" / "seed0"
    d.mkdir(parents=True)
    shutil.copy(src / "scores.json", d / "scores.json")
    s = R.load_seed(d / "scores.json")
    assert s["composite_version"] == 2
    assert s["goals"]["correction"] is None and s["composite"] is None
    assert "no per-pair records" in s["goal_reasons"]["correction"]
    shutil.rmtree(d.parent)


def test_table_columns_only_with_a_reference(env, oracle, tmp_path):
    runs = tmp_path / "runs"
    for k in range(2):
        d = runs / "rung1_base" / f"seed{k}"
        d.mkdir(parents=True)
        for f in ("scores.json", "pairs_per_region.npz", "SCORED"):
            shutil.copy(env["bdir"] / "position_average" / "seed0" / f, d / f)
    rows = R.table(runs)
    assert rows and all(r["composite_version"] == 2 for r in rows)
    assert all("correction_gap_closed_halfread_superseded" in r for r in rows)
    rows1 = R.table(runs, oracle=None)
    assert all("composite_version" not in r for r in rows1)


def test_arm_report_headline_adds_oracle_columns(env, oracle, tmp_path):
    from onion_eval import arm_report as AR
    runs = tmp_path / "runs"
    d = runs / "rung1_base" / "seed0"
    d.mkdir(parents=True)
    for f in ("scores.json", "pairs_per_region.npz", "SCORED"):
        shutil.copy(env["bdir"] / "position_average" / "seed0" / f, d / f)
    rows, cols = AR.headline(runs, "", env["bdir"])
    assert "correction_gap_closed_oracle.mean" in cols
    assert "correction_gap_closed_oracle.numerator.mean" in cols
    r = next(x for x in rows if x["arm"] == "depth_only_scaling")
    assert r["correction_gap_closed_oracle.n_defined"] == 1 or r["composite_version"] == 2
    krows, kcols = AR.per_knob(runs, "", env["bdir"])
    assert any(x["pair_kind"] == "never_trained_non_subset_oracle" for x in krows)


def test_report_cli_prints_both_subsets_and_composites(env, oracle, tmp_path, capsys):
    p = env["bdir"] / "pseudoreplicate_floor" / "seed0" / "scores.json"
    po = tmp_path / "pairs.tsv"
    assert OF.main(["report", str(p), "--pairs-out", str(po)]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("scores\tsubset") and len(out) == 3
    assert {ln.split("\t")[1] for ln in out[1:]} == {"all", "no_nodedup_side"}
    rows = po.read_text().strip().splitlines()
    assert len(rows) == 1 + len(env["ev"]["correction_pairs"])


def test_select_rule_table_runs_as_a_script(env, oracle, tmp_path):
    import subprocess
    runs = tmp_path / "runs"
    d = runs / "rung1_base" / "seed0"
    d.mkdir(parents=True)
    for f in ("scores.json", "pairs_per_region.npz", "SCORED"):
        shutil.copy(env["bdir"] / "position_average" / "seed0" / f, d / f)
    r = subprocess.run([sys.executable, str(REPO / "tools" / "onion_eval" / "select_rule.py"),
                        "table", "--runs", str(runs)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    head = r.stdout.splitlines()[0].split("\t")
    assert "correction_gap_closed_halfread_superseded" in head and "composite_version" in head
