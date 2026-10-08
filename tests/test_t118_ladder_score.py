"""t118 ladder scorer and aggregation — `tools/t118/ladder/{score,aggregate}.py`.

The Predictor (t118-C1) is replaced by a stand-in with the pinned methods: its loc is the
source's log value plus log r, r = depth(C') / depth(C) x a per-model bias, so the depth law,
swap and identity cases have known answers.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import nbinom, norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

import baseline_rungs as br  # noqa: E402
from candi.bench.distributional import p_from_mu  # noqa: E402
from candi.metrics import nb_crps, spearman  # noqa: E402
from ladder import aggregate, data, pairs, score, synth  # noqa: E402

N_COUNTS = 50.0
SIGMA = 0.7
#: the size the scorer sees: exp of the float32 disp the Predictor returns
N_SEEN = float(np.exp(np.float64(np.float32(math.log(N_COUNTS)))))


class FakePredictor:
    """The pinned Predictor surface. counts: loc = log(max(X, 1e-12)) + log r, n = N_COUNTS;
    pval: loc = log(max(X, 1e-3)) + log r, sigma = SIGMA; r = depth(C')/depth(C) x bias."""

    def __init__(self, root, manifest, space, g_id, model="real", seed=0, bias=1.0, rung="A",
                 use_depth=True):
        rows = pairs.read_manifest(manifest)
        self.rung, self.space, self.g_id, self.model, self.seed = rung, space, g_id, model, seed
        self.corpus = data.Corpus(root)
        self.train_pairs = pairs.train_pairs(rows, g_id)
        self.fit_pids = pairs.fit_pids(rows, g_id)
        self.depth = {r["pid"]: float(r["depth"]) for r in rows}
        self.bias, self.use_depth = bias, use_depth

    def _log_r(self, cs, ct):
        r = self.depth[ct] / self.depth[cs] if self.use_depth else 1.0
        return math.log(r * self.bias)

    def theta(self, cs, ct):
        return np.array([self._log_r(cs, ct)])

    def describe(self, cs, ct):
        return {"log_r": self._log_r(cs, ct)}

    def predict(self, x_src_pid, cov_src_pid, cov_tgt_pid, chrom):
        x = self.corpus.get(x_src_pid, self.space, chrom).astype(np.float64)
        floor = 1e-12 if self.space == "counts" else 1e-3
        loc = np.log(np.maximum(x, floor)) + self._log_r(cov_src_pid, cov_tgt_pid)
        disp = np.full(x.shape, math.log(N_COUNTS if self.space == "counts" else SIGMA))
        return loc.astype(np.float32), disp.astype(np.float32)

    def mean(self, loc):
        return np.exp(loc)

    def quantiles(self, loc, disp, q):
        loc, disp = np.asarray(loc, np.float64), np.asarray(disp, np.float64)
        if self.space == "counts":
            n = score.spread_of("counts", disp)            # clipped, as the real Predictor does
            return nbinom.ppf(q, n, p_from_mu(n, np.exp(loc)))
        return np.exp(loc + np.exp(disp) * norm.ppf(q))


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    manifest, cov = synth.make_products(root, seed=0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\nchr21\t5000\t7500\nchr22\t0\t250\n")
    return root, manifest, cov, bed


def _rows(manifest):
    return pairs.read_manifest(manifest)


def _run_dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _eval_by_hand(root, pred, src, tgt, chroms, bed):
    xs, ys, ls = [], [], []
    for c in chroms:
        x = pred.corpus.get(src, pred.space, c)
        y = pred.corpus.get(tgt, pred.space, c)
        loc, _ = pred.predict(src, src, tgt, c)
        bad = data.read_blacklist_flags(bed, c, x.size)
        xs.append(x[~bad])
        ys.append(y[~bad])
        ls.append(loc[~bad].astype(np.float64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ls)


@pytest.fixture(scope="module")
def counts_scores(synth_dir, tmp_path_factory):
    root, manifest, _, bed = synth_dir
    pred = FakePredictor(root, manifest, "counts", "T1")
    d = tmp_path_factory.mktemp("runs") / "A_T1_counts_real_s0"
    d.mkdir()
    score.score_trained(pred, _rows(manifest), bed, d)
    return pred, d, json.loads((d / "scores.json").read_text())


# ---------------------------------------------------------------------------------------------
# score.py
# ---------------------------------------------------------------------------------------------


def test_trained_fields_equal_hand_computation(synth_dir, counts_scores):
    root, manifest, _, bed = synth_dir
    pred, _, s = counts_scores
    rec = next(r for r in s["records"] if r["kind"] == "trained" and r["eval"] == "score"
               and r["target_pid"] == "T1__pe__pe")
    x, y, loc = _eval_by_hand(root, pred, rec["source_pid"], rec["target_pid"],
                              pairs.SCORE_CHROMS, bed)
    mu = np.exp(loc)
    n = np.full(mu.shape, N_SEEN)
    crps = nb_crps(n, p_from_mu(n, mu), y.astype(np.float64))
    sub = br.subset_index(y)
    assert rec["n_blacklisted"] == 200 and rec["n_all"] == 1500 + 1200 - 200
    for sname, idx in sub.items():
        assert rec[f"n_{sname}"] == idx.size
        assert rec[f"crps_{sname}"] == pytest.approx(float(crps[idx].mean()), rel=1e-12)
        assert rec[f"spearman_{sname}"] == pytest.approx(spearman(mu[idx], y[idx].astype(float)),
                                                         rel=1e-12)
    split = br.scale_split(mu, y.astype(np.float64),
                           lambda m, yy: nb_crps(np.full(m.shape, N_SEEN),
                                                 p_from_mu(np.full(m.shape, N_SEEN), m), yy))
    assert rec["crps_oracle_scaled_all"] == pytest.approx(split["crps_oracle_scaled"], rel=1e-12)
    assert rec["c_star_all"] == split["c_star"]
    assert rec["describe"] == {"log_r": pytest.approx(0.0)}
    assert len(rec["pit_hist"]) == 20 and sum(rec["pit_hist"]) == rec["n_all"]
    val = next(r for r in s["records"] if r["kind"] == "trained" and r["eval"] == "val"
               and r["target_pid"] == "T1__pe__pe")
    assert val["n_blacklisted"] == 10 and val["n_all"] == 790
    assert "pit_hist" not in val and "c_star_all" not in val


def test_identity_prediction_matches_nosolution(synth_dir, tmp_path):
    """pval identity (loc = log max(X, 1e-3), one sigma) == baseline noSolution, exactly; counts
    with a huge NB size == the Poisson noSolution to its limit."""
    root, manifest, _, bed = synth_dir
    rows = _rows(manifest)
    pred = FakePredictor(root, manifest, "pval", "T1")
    d = _run_dir(tmp_path, "A_T1_pval_real_s0")
    score.score_trained(pred, rows, bed, d)
    s = json.loads((d / "scores.json").read_text())
    rec = next(r for r in s["records"] if r["kind"] == "trained" and r["eval"] == "score"
               and r["target_pid"] == "T1__extsize__k2")          # same depth: r = 1
    x, y, _ = _eval_by_hand(root, pred, rec["source_pid"], rec["target_pid"],
                            pairs.SCORE_CHROMS, bed)
    ref = br.score_rung(x.astype(np.float64), y, "pval", {"sigma": SIGMA}, br.subset_index(y))
    for sname in br.SUBSETS:
        assert rec[f"crps_{sname}"] == pytest.approx(ref[f"spread_crps_{sname}"], rel=1e-6)
    assert rec["crps_oracle_scaled_all"] == pytest.approx(ref["crps_oracle_scaled_all"], rel=1e-6)
    assert rec["c_star_all"] == pytest.approx(ref["c_star_all"])
    assert rec["spearman_all"] == pytest.approx(ref["spearman_all"], abs=1e-3)

    # counts: NB with n = 1e7 is the Poisson of the same mean to ~mu/n, on bins with X > 0
    # (below that, p_from_mu's clip at 1 - 1e-9 lifts the NB mean to n 1e-9: the known trap)
    corpus = data.Corpus(root)
    xc = corpus.get("T1__base__base", "counts", "chr19").astype(float)
    mu = xc[xc > 0]
    yy = corpus.get("T1__pe__pe", "counts", "chr19").astype(float)[xc > 0]
    nb = score.crps_given("counts", mu, np.full(mu.shape, 1e7), yy)
    assert np.allclose(nb, br.poisson_crps(mu, yy), rtol=1e-3, atol=1e-6)


def test_identical_flags_and_kinds(synth_dir, counts_scores):
    _, manifest, _, _ = synth_dir
    rows = _rows(manifest)
    _, _, s = counts_scores
    recs = s["records"]
    tr = [r for r in recs if r["kind"] == "trained"]
    assert len(tr) == 2 * len(pairs.train_pairs(rows, "T1")) == 16
    ext = [r for r in tr if "T1__extsize__k2" in (r["source_pid"], r["target_pid"])]
    assert ext and all(r["counts_identical"] and not r["pval_identical"] for r in ext)
    assert not any(r["counts_identical"] for r in tr if r["target_pid"] == "T1__pe__pe")
    sh = [r for r in recs if r["kind"] == "shuffle"]
    assert len(sh) == 8
    for r in sh:
        p = next(q for q in pairs.train_pairs(rows, "T1")
                 if (q["source_pid"], q["target_pid"]) == (r["source_pid"], r["target_pid"]))
        assert r["cov_tgt_pid"] == pairs.shuffle_target(rows, p, "counts", 0)
        assert r["cov_src_pid"] == r["source_pid"] and r["eval"] == "score"
    sw = [r for r in recs if r["kind"] == "swap"]
    assert sorted(r["source_pid"] for r in sw) == pairs.fit_pids(rows, "T1")
    assert all(r["cov_src_pid"] == r["cov_tgt_pid"] == r["source_pid"] == r["target_pid"]
               for r in sw)
    assert all(r[score.SWAP_KEY] < 1e-5 for r in sw)       # r = 1: prediction == X on X > 0
    assert s["run"] == {"run_name": "A_T1_counts_real_s0", "rung": "A", "g": "T1",
                        "g_version": "per_track", "space": "counts", "model": "real", "seed": 0,
                        "steps": None, "best_step": None, "val_nll": None}


def test_depth_law_recovers_ratio(counts_scores):
    _, _, s = counts_scores
    dl = [r for r in s["records"] if r["kind"] == "depthlaw"]
    assert len(dl) == 4                                   # base <-> 15M, base <-> 7.5M
    for r in dl:
        assert r["depth_source"] == "trained"
        assert abs(r["depth_log2_scale_pred"] - r["depth_log2_ratio_true"]) < 1e-6
    assert sorted(round(r["depth_log2_ratio_true"], 9) for r in dl) == [-2.0, -1.0, 1.0, 2.0]


def test_swap_and_figdata(synth_dir, counts_scores, tmp_path):
    root, manifest, _, bed = synth_dir
    _, d, s = counts_scores
    fd = np.load(d / "figdata.npz")
    metas = [k for k in fd.files if k.startswith("meta__")]
    snips = sorted(k for k in fd.files if k.startswith("snip__"))
    assert len(metas) == 8 and all(fd[k].shape == (3, 161) for k in metas)
    assert snips == [f"snip__{a}__{i}" for a in ("depth", "extsize", "pe") for i in range(3)]
    assert all(fd[k].shape == (5, 400) and fd[k].dtype == np.float32 for k in snips)
    assert len(s["snippets"]) == 9
    sn = {(x["arm"], x["i"]): x for x in s["snippets"]}
    assert sn[("depth", 0)]["target_pid"] == "T1__depth__15M"       # "15M" < "7.5M"
    assert all(x["chrom"] in pairs.SCORE_CHROMS for x in s["snippets"])
    assert (d / "SCORE_DONE").is_file()
    # a biased twin: swap ratio = |log bias|; twins write no meta/snip
    pred = FakePredictor(root, manifest, "counts", "T1", model="nocov", bias=1.5)
    d2 = _run_dir(tmp_path, "A_T1_counts_nocov_s0")
    score.score_trained(pred, _rows(manifest), bed, d2)
    s2 = json.loads((d2 / "scores.json").read_text())
    sw = [r for r in s2["records"] if r["kind"] == "swap"]
    assert all(r[score.SWAP_KEY] == pytest.approx(math.log(1.5), abs=1e-5) for r in sw)
    assert np.load(d2 / "figdata.npz").files == []
    assert all("pit_hist" in r for r in s2["records"]
               if r["kind"] == "trained" and r["eval"] == "score")


def test_law_pass(synth_dir, tmp_path):
    root, manifest, _, bed = synth_dir
    rows = _rows(manifest)
    pred = FakePredictor(root, manifest, "counts", "T1")
    d = _run_dir(tmp_path, "A_T1_counts_real_s0")
    score.score_law(pred, rows, bed, d, workers=2)
    law = json.loads((d / "law.json").read_text())
    lp = [r for r in law["records"] if r["kind"] == "law"]
    assert [(r["source_pid"], r["target_pid"]) for r in lp] == \
        [(p["source_pid"], p["target_pid"]) for p in pairs.law_pairs(rows, "T1")]
    assert len(lp) == 12 and all(r["direction"] == "arm_to_arm" for r in lp)
    dl = [r for r in law["records"] if r["kind"] == "depthlaw"]
    assert len(dl) == 2 and all(r["depth_source"] == "law" for r in dl)
    assert all(abs(r["depth_log2_scale_pred"] - r["depth_log2_ratio_true"]) < 1e-6 for r in dl)
    # the in-process path gives the same numbers as the forked pool
    d1 = _run_dir(tmp_path, "one/A_T1_counts_real_s0")
    score.score_law(pred, rows, bed, d1, workers=1)
    one = json.loads((d1 / "law.json").read_text())["records"]
    assert [r["crps_all"] for r in one] == [r["crps_all"] for r in law["records"]]
    assert (d / "LAW_DONE").is_file()


def test_cli_skips_when_done(tmp_path):
    d = _run_dir(tmp_path, "X")
    (d / "SCORE_DONE").write_text("x")
    for p in ("m", "c", "dd", "b"):
        (tmp_path / p).write_text("")
    assert score.main(["trained", str(tmp_path / "m"), str(tmp_path / "c"), str(tmp_path / "dd"),
                       str(tmp_path / "b"), str(d)]) == 0


# ---------------------------------------------------------------------------------------------
# aggregate.py on fabricated runs
# ---------------------------------------------------------------------------------------------

#: crps_all per (rung, model) per seed; crps_top1 = 2 x crps_all
FAB = {("A", "real"): [1.00, 1.10, 0.95], ("A", "nocov"): [2.0, 2.0, 2.0],
       ("A", "ids"): [1.05, 1.05, 1.05],
       ("B", "real"): [0.60, 0.62, 0.61], ("B", "nocov"): [2.0, 2.1, 2.0],
       ("B", "ids"): [0.9, 0.9, 0.9]}
#: chr22 crps_all (real): B is best with wobble 0.03, A within it -> A chosen
FAB_VAL = {"A": [1.0, 1.05, 1.02], "B": [0.985, 1.0, 1.015]}
DEPTH_DELTA = [0.01, 0.03, 0.02]         # log2_pred - log2_true per seed


def _fab_run(runs_dir: Path, rows, rung, model, seed):
    space, g = "counts", "T1"
    name = f"{rung}_{g}_{space}_{model}_s{seed}"
    d = runs_dir / name
    d.mkdir(parents=True)
    v = FAB[(rung, model)][seed]
    recs = []

    def rec(kind, ev, p, value, **extra):
        r = {"kind": kind, "eval": ev, **{k: p[k] for k in pairs.PAIR_KEYS},
             "cov_src_pid": p["source_pid"], "cov_tgt_pid": p["target_pid"]}
        for m in score.METRICS:
            r[m] = (2 * value if m == "crps_top1" else value) if m.startswith("crps") else 0.5
        r.update(extra)
        return r

    for p in pairs.train_pairs(rows, g):
        recs.append(rec("trained", "score", p, v, crps_oracle_scaled_all=0.9 * v,
                        scale_error_all=0.1 * v, c_star_all=0.0, pit_hist=[1] * 20,
                        describe={"a": 0.0, "b": 1.0}))
        recs.append(rec("trained", "val", p, FAB_VAL[rung][seed] if model == "real" else 9.0))
        recs.append(rec("shuffle", "score", p, v + 0.5 if model == "real" else v))
        if {p["arm_src"], p["arm_tgt"]} == {"base", "depth"}:
            t = math.log2(float(next(r for r in rows if r["pid"] == p["target_pid"])["depth"])
                          / float(next(r for r in rows if r["pid"] == p["source_pid"])["depth"]))
            recs.append(rec("depthlaw", "score", p, v, depth_source="trained",
                            depth_log2_ratio_true=t,
                            depth_log2_scale_pred=t + 0.5))      # never read by the check
    by_pid = {r["pid"]: r for r in pairs.usable_products(rows)}
    for pid in pairs.fit_pids(rows, g):
        p = pairs._pair(by_pid[pid], by_pid[pid], "swap")
        recs.append(rec("swap", "score", p, v, swap_median_abs_log_ratio=0.02 * (seed + 1)))
    law = [rec("law", "score", p, v + 0.1) for p in pairs.law_pairs(rows, g)]
    dep = {r["pid"]: float(r["depth"]) for r in rows}
    for p in pairs.law_pairs(rows, g):
        if p["arm_src"] == p["arm_tgt"] == "depth":
            t = math.log2(dep[p["target_pid"]] / dep[p["source_pid"]])
            law.append(rec("depthlaw", "score", p, v, depth_source="law", depth_log2_ratio_true=t,
                           depth_log2_scale_pred=t + DEPTH_DELTA[seed]))
    run = {"run_name": name, "rung": rung, "g": g, "g_version": "per_track", "space": space,
           "model": model, "seed": seed, "steps": 10, "best_step": 10, "val_nll": 1.0}
    (d / "scores.json").write_text(json.dumps({"run": run, "records": recs, "snippets": []}))
    (d / "law.json").write_text(json.dumps({"run": run, "records": law}))
    np.savez(d / "figdata.npz")


@pytest.fixture(scope="module")
def fabricated(synth_dir, tmp_path_factory):
    _, manifest, cov, _ = synth_dir
    rows = _rows(manifest)
    base = tmp_path_factory.mktemp("agg")
    runs = base / "runs"
    runs.mkdir()
    for rung, model in FAB:
        for seed in pairs.SEEDS:
            _fab_run(runs, rows, rung, model, seed)
    (runs / "empty_dir").mkdir()                    # a run with no scores.json: missing, not fatal
    refs = base / "refs.tsv"
    refs.write_text("track\tspace\trung\teval\tarm_pid\tspread_crps_all\tspread_crps_top1\t"
                    "spearman_all\n"
                    "T1\tcounts\tnoSolution\tscore\tT1__pe__pe\t3.0\t6.0\t0.4\n"
                    "T1\tcounts\tnoSolution\tscore\tT1__depth__15M\t5.0\t8.0\t0.6\n"
                    "T1\tcounts\tnoSolution\tval\tT1__pe__pe\t99\t99\t0.1\n"
                    "T1\tcounts\toracle\tscore\tT1__pe__pe\t99\t99\t0.1\n")
    agg = base / "agg"
    res = aggregate.aggregate(manifest, cov, runs, refs, agg)
    return res, agg, rows


def _check(res, name, rung="A", metric="crps_all", mc="narrow"):
    return next(c for c in res["checks"] if c["check"] == name and c["rung"] == rung
                and c["metric"] == metric and c["mark_class"] == mc)


def test_aggregate_schema_and_missing(fabricated):
    res, agg, rows = fabricated
    for k in ("created_utc", "runs_dir", "runs_present", "runs_missing", "refs", "per_pair",
              "per_track", "per_class", "checks", "knob_gain", "law_grid", "depth_law",
              "rung_choice", "figdata"):
        assert k in res
    assert len(res["runs_present"]) == 18
    assert "A_T2_counts_real_s0" in res["runs_missing"] and "C_all_pval_ids_s2" in res["runs_missing"]
    assert "A_T1_counts_real_s0" not in res["runs_missing"]
    assert len(res["runs_missing"]) == len(aggregate.expected_runs(rows)) - 18
    assert set(res["figdata"]) == set(res["runs_present"])
    for f in ("results.json", "results_summary.tsv", "checks_A.json", "checks_B.json",
              "checks_main.json"):
        assert (agg / f).is_file()
    text = (agg / "results.json").read_text()
    assert "\n " not in text                           # compact JSON
    on_disk = json.loads(text)
    assert {(r["kind"], r["eval"]) for r in on_disk["per_pair"]} == {("trained", "score"),
                                                                    ("swap", "score")}
    assert all(r["metric"] in score.METRICS for r in on_disk["per_class"] + on_disk["per_track"])
    import gzip
    with gzip.open(agg / "per_pair_rest.jsonl.gz", "rt") as fh:
        rest = [json.loads(line) for line in fh]
    assert {r["kind"] for r in rest} == {"trained", "shuffle", "law", "depthlaw"}
    assert {r["eval"] for r in rest if r["kind"] == "trained"} == {"val"}
    n_law = len(pairs.law_pairs(rows, "T1"))
    assert sum(r["kind"] == "law" for r in rest) == 18 * n_law
    assert all("describe" not in r for r in rest)
    assert "note" not in json.loads((agg / "checks_A.json").read_text())
    assert "note" not in json.loads((agg / "checks_main.json").read_text())
    pp = res["per_pair"][0]
    assert {"rung", "g", "g_version", "space", "model", "seed"} <= set(pp)
    assert res["refs"] == {"T1": {"counts": {"noSolution": {
        "crps_all": 4.0, "crps_top1": 7.0, "spearman_all": 0.5}}}}


def test_aggregate_missing_refs_is_not_fatal(synth_dir, tmp_path, capsys):
    _, manifest, cov, _ = synth_dir
    runs = tmp_path / "runs"
    runs.mkdir()
    res = aggregate.aggregate(manifest, cov, runs, tmp_path / "nope.tsv", tmp_path / "agg")
    assert res["refs"] == {} and res["runs_present"] == [] and res["checks"] == []
    assert "WARNING" in capsys.readouterr().err


def test_seed_wobble_and_class_values(fabricated):
    res, _, _ = fabricated
    row = next(r for r in res["per_class"] if r["rung"] == "A" and r["model"] == "real"
               and r["kind"] == "trained" and r["metric"] == "crps_all" and r["variant"] == "all")
    assert row["per_seed"] == pytest.approx([1.00, 1.10, 0.95])
    assert row["mean"] == pytest.approx(np.mean([1.00, 1.10, 0.95]))
    assert row["seed_wobble"] == pytest.approx(0.15)
    assert row["mark_class"] == "narrow" and row["g_version"] == "per_track"
    # nonidentical drops the count-identical extsize pairs (2 of 8 trained pairs)
    tr = [r for r in res["per_track"] if r["rung"] == "A" and r["model"] == "real"
          and r["seed"] == 0 and r["kind"] == "trained" and r["metric"] == "crps_all"]
    assert {r["variant"]: r["n_pairs"] for r in tr} == {"all": 8, "nonidentical": 6}
    assert aggregate.wobble([1.0, None, 1.3]) == pytest.approx(0.3)
    assert aggregate.wobble([1.0]) is None


def test_checks_values_bars_met(fabricated):
    res, agg, _ = fabricated
    mean = lambda v: float(np.mean(v))  # noqa: E731
    c = _check(res, "beatstwin")
    assert c["value"] == pytest.approx(2.0 - mean(FAB[("A", "real")]))
    assert c["bar"] == pytest.approx(0.30) and c["met"] is True
    c = _check(res, "beatstwin", metric="crps_top1")
    assert c["value"] == pytest.approx(2 * (2.0 - mean(FAB[("A", "real")])))
    assert c["bar"] == pytest.approx(0.60)
    c = _check(res, "lawtest_ids")                     # 1.15 - 1.1167 = 0.033 < 0.30
    assert c["value"] == pytest.approx(1.15 - (mean(FAB[("A", "real")]) + 0.1))
    assert c["met"] is False
    c = _check(res, "lawtest_nocov")
    assert c["value"] == pytest.approx(2.1 - (mean(FAB[("A", "real")]) + 0.1)) and c["met"]
    c = _check(res, "beatsbelow", rung="B")
    assert c["value"] == pytest.approx(mean(FAB[("A", "real")]) - mean(FAB[("B", "real")]))
    assert c["bar"] == pytest.approx(2 * 0.15) and c["met"] is True
    assert not [x for x in res["checks"] if x["check"] == "beatsbelow" and x["rung"] == "A"]
    c = _check(res, "shuffle")                         # D_nocov - D_real_shuffled
    assert c["value"] == pytest.approx(2.0 - (mean(FAB[("A", "real")]) + 0.5))
    assert c["bar"] == pytest.approx(0.30) and c["met"] is False
    c = _check(res, "swap", metric=score.SWAP_KEY)
    assert c["value"] == pytest.approx(0.04) and c["bar"] == 0.1 and c["met"] is True
    c = _check(res, "depthlaw", metric="depth_scale_rel_error")
    assert c["value"] == pytest.approx(abs(2 ** mean(DEPTH_DELTA) - 1))
    assert c["bar"] == 0.10 and c["met"] is True and c["space"] == "counts"
    assert c["components"]["n_pairs"] == 2            # the law depth->depth pairs only
    assert c["seed_wobble"] == pytest.approx(abs(2 ** 0.03 - 2 ** 0.01))
    for x in res["checks"]:
        assert set(x) == {"rung", "g_version", "space", "mark_class", "metric", "check", "value",
                          "bar", "met", "seed_wobble", "components"}
        assert x["met"] in (True, False)
    ca = json.loads((agg / "checks_A.json").read_text())
    assert ca["rung"] == "A" and {x["check"] for x in ca["checks"]} == \
        {"beatstwin", "lawtest_nocov", "lawtest_ids", "depthlaw"}


def test_rung_choice_and_main_checks(fabricated):
    res, agg, _ = fabricated
    ch = res["rung_choice"]["per_track|counts"]
    assert ch["val_by_rung"]["A"] == pytest.approx(np.mean(FAB_VAL["A"]))
    assert ch["wobble_by_rung"]["B"] == pytest.approx(0.03)
    assert ch["chosen"] == "A"                          # A is within B's wobble of B
    main = json.loads((agg / "checks_main.json").read_text())
    assert main["rung_choice"]["per_track|counts"]["chosen"] == "A"
    assert {x["rung"] for x in main["checks"]} == {"A"}
    assert {"shuffle", "swap", "beatstwin"} <= {x["check"] for x in main["checks"]}


def test_knob_gain_and_law_grid(fabricated):
    res, _, _ = fabricated
    kg = next(r for r in res["knob_gain"] if r["rung"] == "A" and r["arm"] == "pe"
              and r["metric"] == "crps_all")
    real = float(np.mean(FAB[("A", "real")]))
    assert kg["gain_rel"] == pytest.approx((2.0 - real) / 2.0)
    lg = [r for r in res["law_grid"] if r["rung"] == "A" and r["metric"] == "crps_all"]
    assert len(lg) == 12
    one = lg[0]
    assert one["gain_nocov"] == pytest.approx(2.1 - (real + 0.1))
    assert one["gain_ids"] == pytest.approx(1.15 - (real + 0.1))
    ext = [r for r in lg if r["arm_tgt"] == "extsize"]
    assert ext and all(r["identical_target"] is False for r in ext)   # extsize != any other arm
    dl = [r for r in res["depth_law"] if r["rung"] == "A" and r["model"] == "real"]
    assert len(dl) == (4 + 2) * 3 and {r["kind"] for r in dl} == {"trained", "law"}


def test_end_to_end_score_then_aggregate(synth_dir, tmp_path):
    """Real scorer output (3 models x 3 seeds, fake predictor) aggregates without error."""
    root, manifest, cov, bed = synth_dir
    rows = _rows(manifest)
    runs = tmp_path / "runs"
    for model, bias in (("real", 1.0), ("nocov", 1.6), ("ids", 1.2)):
        for seed in pairs.SEEDS:
            pred = FakePredictor(root, manifest, "counts", "T2", model=model, seed=seed,
                                 bias=1.0 if model == "real" else bias * (1 + 0.01 * seed))
            d = _run_dir(runs, f"A_T2_counts_{model}_s{seed}")
            score.score_trained(pred, rows, bed, d)
            score.score_law(pred, rows, bed, d)
    agg = tmp_path / "agg"
    assert aggregate.main([str(manifest), str(cov), str(runs), str(tmp_path / "none.tsv"),
                           str(agg)]) == 0
    res = json.loads((agg / "results.json").read_text())
    assert len(res["runs_present"]) == 9
    bt = [c for c in res["checks"] if c["check"] == "beatstwin" and c["metric"] == "crps_all"]
    assert len(bt) == 1 and bt[0]["mark_class"] == "DNase" and bt[0]["met"] is True
    dl = [c for c in res["checks"] if c["check"] == "depthlaw"]
    assert len(dl) == 1 and dl[0]["value"] < 1e-5 and dl[0]["met"] is True
    assert res["rung_choice"]["per_track|counts"]["chosen"] == "A"
    _c4_accepts(agg, "A")


def _c4_accepts(agg: Path, rung: str) -> None:
    """C4's schema check and report run on this aggregate (in-process, candii python)."""
    from ladder import figures, report
    errs = figures.check_schema(figures.load_results(agg))
    assert errs == [], errs[:5]
    text = report.build_report(agg, rung)
    assert "## Checks" in text and "## Depth law" in text


def test_c4_reads_the_aggregate(fabricated):
    _, agg, _ = fabricated
    _c4_accepts(agg, "A")
    _c4_accepts(agg, "B")


def test_spread_clipped_and_nonfinite_counted(synth_dir, tmp_path):
    root, manifest, _, bed = synth_dir
    rows = _rows(manifest)

    class Wild(FakePredictor):
        def predict(self, x_src_pid, cov_src_pid, cov_tgt_pid, chrom):
            loc, disp = super().predict(x_src_pid, cov_src_pid, cov_tgt_pid, chrom)
            disp = np.full_like(disp, 60.0)                       # n = e^60 before the clip
            if cov_tgt_pid == "T1__pe__pe":
                loc = loc.copy()
                loc[-3:] = np.nan                                 # three bad bins per chromosome
            return loc, disp

    pred = Wild(root, manifest, "counts", "T1")
    d = _run_dir(tmp_path, "A_T1_counts_real_s0")
    score.score_trained(pred, rows, bed, d)
    s = json.loads((d / "scores.json").read_text())
    tr = [r for r in s["records"] if r["kind"] == "trained"]
    ok = [r for r in tr if r["target_pid"] != "T1__pe__pe"]
    assert all(r["n_crps_nonfinite"] == 0 and r["crps_all"] is not None for r in ok)
    bad = [r for r in tr if r["target_pid"] == "T1__pe__pe" and r["eval"] == "score"]
    assert bad and all(r["n_crps_nonfinite"] == 6 and r["crps_all"] is None for r in bad)
    assert s["n_records_crps_nonfinite"] >= 2
    # the clip: disp = 60 is scored at n = N_MAX
    rec = next(r for r in ok if r["eval"] == "score" and r["target_pid"] == "T1__depth__15M")
    x, y, loc = _eval_by_hand(root, pred, rec["source_pid"], rec["target_pid"],
                              pairs.SCORE_CHROMS, bed)
    n = score.spread_of("counts", np.full(loc.shape, 60.0))
    assert np.allclose(n, score.N_MAX, rtol=1e-12)
    want = float(np.mean(nb_crps(n, p_from_mu(n, np.exp(loc)), y.astype(float))))
    assert rec["crps_all"] == pytest.approx(want, rel=1e-12)


def test_default_workers():
    assert 1 <= score.default_workers() <= (__import__("os").cpu_count() or 1)


def test_qm_curves(synth_dir, tmp_path):
    root, manifest, _, _ = synth_dir
    out = tmp_path / "qm.json"
    assert aggregate.main(["qm-curves", str(manifest), str(root), str(out)]) == 0
    q = json.loads(out.read_text())
    assert set(q) == {"counts", "pval"}
    assert set(q["counts"]["T1"]) == {"T1__depth__15M", "T1__extsize__k2", "T1__pe__pe"}
    c = q["counts"]["T1"]["T1__extsize__k2"]            # identical counts: the identity map
    assert c["knots_x"] == c["knots_y"]
    assert all(len(v["knots_x"]) <= aggregate.QM_MAX_KNOTS for t in q["pval"].values()
               for v in t.values())


def test_one_seed_gives_boolean_met_and_a_report(synth_dir, tmp_path):
    """With one seed no wobble exists: every bar-less check is met = False with its reason, and
    C4's schema check and report still run."""
    _, manifest, cov, _ = synth_dir
    rows = _rows(manifest)
    runs = tmp_path / "runs"
    runs.mkdir()
    for model in pairs.MODELS:
        _fab_run(runs, rows, "A", model, 0)
    agg = tmp_path / "agg"
    res = aggregate.aggregate(manifest, cov, runs, tmp_path / "none.tsv", agg)
    assert res["checks"] and all(isinstance(c["met"], bool) for c in res["checks"])
    nobar = [c for c in res["checks"] if c["bar"] is None]
    assert nobar and all(c["met"] is False and "met_reason" in c["components"] for c in nobar)
    _c4_accepts(agg, "A")
