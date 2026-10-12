"""t125 part 1 — `tools/t125/eta/score.py`: score one eta run (trained, shuffle, swap, depth law)
and the law pass (depth-only by default, `--law-scope all` for every arm->arm pair).

Written from plan/ETA_OFFSET_AND_H_DESIGN.md "Scoring" and the plan's pinned interfaces (chunk
t125-P1e). Runs are hand-written checkpoints of `eta.model.EtaLadder` on `ladder.synth` products
(P1d's trainer is built in parallel; a checkpoint is all the scorer reads).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t125", ROOT / "tools" / "t118"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import baseline_rungs as br  # noqa: E402
from ladder import base, data, encoding, synth  # noqa: E402
from ladder import model as lmodel  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402
from ladder import score as lscore  # noqa: E402
from eta import model as em  # noqa: E402
from eta import offset  # noqa: E402
from eta import pairs as ep  # noqa: E402
from eta import score as es  # noqa: E402

N0, SIGMA0 = 5.0, 0.5
G = "all"
MSE_KEYS = ("mse_log1p_all", "mse_log1p_nonzero", "mse_log1p_top1")
T1B, T1D15, T1D75 = "T1__base__base", "T1__depth__15M", "T1__depth__7.5M"


# ---------------------------------------------------------------------------------------------
# fixtures: synthetic products and hand-written runs
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_eta_score")
    manifest, cov = synth.make_products(root, seed=0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\nchr21\t5000\t7500\nchr22\t0\t250\n")
    rows = lpairs.read_manifest(manifest)
    corpus = data.Corpus(root)
    log_d = offset.depth_table(rows)
    xs = np.concatenate([offset.transform_eta(corpus.get(pid, "counts", "chr1"), ld)
                         for pid, ld in log_d.items()])
    stats = {"knots_x": base.knot_stats(xs), "n0": N0, "sigma0": SIGMA0}
    enc = encoding.Encoder.fit(encoding.read_covariates(cov), lpairs.fit_pids(rows, G))
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed, "rows": rows,
            "log_d": log_d, "stats": stats, "encoder": enc}


def _write_run(runs: Path, sd: dict, rung: str, model: str, perturb: bool = True,
               seed: int = 0) -> Path:
    torch.manual_seed(1234)               # g's own init draws from the global stream
    lad = em.EtaLadder(rung, sd["stats"])
    if perturb:
        gen = torch.Generator().manual_seed(21)
        with torch.no_grad():
            for p in list(lad.g.parameters()) + list(lad.form.parameters()):
                p.add_(0.05 * torch.randn(p.shape, generator=gen))
    rows = sd["rows"]
    name = ep.run_name(rung, G, model, seed)
    config = {
        "run_name": name, "rung": rung, "g": G, "space": "counts", "model": model, "seed": seed,
        "n_cov": encoding.N_COV, "fit_pids": lpairs.fit_pids(rows, G),
        "train_pairs": lpairs.train_pairs(rows, G),
        "manifest_md5": lmodel.md5_path(sd["manifest"]),
        "covariates_md5": lmodel.md5_path(sd["cov"]),
        "log_depth": dict(sd["log_d"]),
        "t125": {"offset": "log1p(X) - log(d/1e6)", "d_unit": 1e6},
        **({"row": 2} if lad.per_bin else {}),
    }
    ckpt = {"g": {k: v.clone() for k, v in lad.g.state_dict().items()},
            "form": {k: v.clone() for k, v in lad.form.state_dict().items()},
            "stats": {"knots_x": [float(v) for v in sd["stats"]["knots_x"]], "n0": N0,
                      "sigma0": SIGMA0},
            "encoder": sd["encoder"].state(), "config": config, "step": 5, "val_nll": 1.5}
    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, d / "ckpt.pt")
    (d / "config.json").write_text(json.dumps(config))
    (d / "train_log.tsv").write_text("step\tloss\tval_nll\tseconds\n5\t2.0\t1.5\t0.1\n")
    return d


def _pred(sd, run_dir):
    return em.load_run(run_dir, sd["root"], sd["cov"], sd["manifest"], device="cpu")


@pytest.fixture(scope="module")
def runs(synth_dir, tmp_path_factory):
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(scope="module")
def scored_a(synth_dir, runs):
    d = _write_run(runs, synth_dir, "A", "real")
    pred = _pred(synth_dir, d)
    es.score_trained(pred, synth_dir["rows"], synth_dir["bed"], d)
    return pred, d, json.loads((d / "scores.json").read_text())


@pytest.fixture(scope="module")
def scored_a2_nocov(synth_dir, runs):
    d = _write_run(runs, synth_dir, "A2", "nocov")
    pred = _pred(synth_dir, d)
    es.score_trained(pred, synth_dir["rows"], synth_dir["bed"], d)
    return pred, d, json.loads((d / "scores.json").read_text())


def _eval_by_hand(sd, pred, x_pid, y_pid, cov_src, cov_tgt, chroms=lpairs.SCORE_CHROMS):
    """(x, y, loc, disp) over the chromosomes, blacklist removed, one direct pred.predict each."""
    xs, ys, ls, ds = [], [], [], []
    for c in chroms:
        x = pred.corpus.get(x_pid, "counts", c)
        y = pred.corpus.get(y_pid, "counts", c)
        loc, disp = pred.predict(x_pid, cov_src, cov_tgt, c)
        keep = ~data.read_blacklist_flags(sd["bed"], c, x.size)
        xs.append(x[keep])
        ys.append(y[keep])
        ls.append(np.asarray(loc, np.float64)[keep])
        ds.append(np.asarray(disp, np.float64)[keep])
    cat = np.concatenate
    return cat(xs), cat(ys), cat(ls), cat(ds)


def _mse_by_hand(pred, loc, y) -> dict:
    m = np.exp(np.clip(loc, math.log(lmodel.MU_MIN), math.log(lmodel.MU_MAX)))
    sq = (np.log1p(m) - np.log1p(y.astype(np.float64))) ** 2
    return {f"mse_log1p_{s}": float(sq[idx].mean())
            for s, idx in br.subset_index(y).items()}


# ---------------------------------------------------------------------------------------------
# trained: record kinds and counts
# ---------------------------------------------------------------------------------------------


def test_trained_kinds_and_counts(synth_dir, scored_a):
    pred, d, s = scored_a
    rows = synth_dir["rows"]
    tp = lpairs.train_pairs(rows, G)
    recs = s["records"]
    by = lambda k, ev=None: [r for r in recs if r["kind"] == k and (ev is None or r["eval"] == ev)]  # noqa: E731
    assert len(by("trained", "score")) == len(tp) and len(by("trained", "val")) == len(tp)
    n_shuffle = sum(1 for p in tp if _has_shuffle(rows, p, pred.seed))
    assert len(by("shuffle")) == n_shuffle == len(tp) - len(s["skipped"])
    assert len(by("swap")) == len(lpairs.fit_pids(rows, G))
    n_depth = sum(lscore._is_depth_pair(p, "trained") for p in tp)
    assert n_depth == 8
    dl = by("depthlaw")
    assert len(dl) == n_depth and all(r["depth_source"] == "trained" for r in dl)
    assert {r["kind"] for r in recs} == {"trained", "shuffle", "swap", "depthlaw"}
    assert s["t125"]["metrics_added"] == list(MSE_KEYS)
    assert "law_scope" in s["t125"]
    for k in ("created_utc", "run", "config", "records", "snippets", "skipped",
              "n_records_crps_nonfinite", "timing"):
        assert k in s
    assert s["run"]["rung"] == "A" and s["run"]["g_version"] == "across"
    assert s["run"]["run_name"] == "A_all_counts_real_s0"
    for r in recs:
        for k in MSE_KEYS:
            assert k in r
        assert r["crps_all"] is not None and math.isfinite(r["crps_all"])
        assert isinstance(r[es.EXPLODE_KEY], bool)
        assert r[es.EXPLODE_KEY] == (r["crps_all"] > 20)
    tr = by("trained", "score")[0]
    for k in ("crps_oracle_scaled_all", "scale_error_all", "c_star_all", "describe", "pit_hist",
              "crps_top1", "n_top1", "spearman_all", "n_blacklisted", "n_crps_nonfinite"):
        assert k in tr
    assert all(lscore.SWAP_KEY in r for r in by("swap"))
    assert "n_records_crps_gt_20" not in s
    ck = s["crps_by_kind"]
    assert set(ck) == {"trained_score", "val", "shuffle", "swap"}
    for label, rs in (("trained_score", by("trained", "score")), ("val", by("trained", "val")),
                      ("shuffle", by("shuffle")), ("swap", by("swap"))):
        assert ck[label] == {"n_records": len(rs),
                             "n_crps_gt_20": sum(r[es.EXPLODE_KEY] for r in rs),
                             "n_crps_nonfinite": sum(r["n_crps_nonfinite"] > 0 for r in rs)}
    assert sum(c["n_records"] for c in ck.values()) == len(recs) - len(dl)   # no depthlaw copy
    assert (d / "SCORE_DONE").is_file()


def _has_shuffle(rows, p, seed):
    try:
        lpairs.shuffle_target(rows, p, "counts", seed)
    except ValueError:
        return False
    return True


def test_shuffle_and_swap_use_target_depth(synth_dir, scored_a):
    """shuffle: d' = the wrong target's depth; swap: C' = C so d' = d (the record's numbers equal
    a direct predict with those pids)."""
    pred, _, s = scored_a
    for kind in ("shuffle", "swap"):
        r = next(r for r in s["records"] if r["kind"] == kind)
        x, y, loc, _ = _eval_by_hand(synth_dir, pred, r["source_pid"], r["target_pid"],
                                     r["cov_src_pid"], r["cov_tgt_pid"])
        want = _mse_by_hand(pred, loc, y)
        for k in MSE_KEYS:
            assert r[k] == pytest.approx(want[k], rel=1e-9, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# the added metric: MSE on log(1 + counts) of the predicted mean
# ---------------------------------------------------------------------------------------------


def test_mse_log1p_equals_numpy(synth_dir, scored_a):
    pred, _, s = scored_a
    for r in [r for r in s["records"] if r["kind"] == "trained"][:6]:
        chroms = lpairs.SCORE_CHROMS if r["eval"] == "score" else lpairs.VAL_CHROMS
        x, y, loc, disp = _eval_by_hand(synth_dir, pred, r["source_pid"], r["target_pid"],
                                        r["source_pid"], r["target_pid"], chroms)
        want = _mse_by_hand(pred, loc, y)
        for k in MSE_KEYS:
            assert r[k] == pytest.approx(want[k], rel=1e-9, abs=1e-12)
        # CRPS untouched by the addition: the t118 metric fields on the same arrays
        fields, _ = lscore.metric_fields(pred, loc, disp, y, split=r["eval"] == "score")
        for k in ("crps_all", "crps_nonzero", "crps_top1"):
            assert r[k] == pytest.approx(fields[k], rel=1e-12)
        if r["eval"] == "score":
            assert r["crps_oracle_scaled_all"] == pytest.approx(fields["crps_oracle_scaled_all"])
            assert r["scale_error_all"] == pytest.approx(fields["scale_error_all"])


class _StubPred:
    space = "counts"

    @staticmethod
    def mean(loc):
        return np.exp(np.clip(np.asarray(loc, np.float64), math.log(lmodel.MU_MIN),
                              math.log(lmodel.MU_MAX)))


def test_metric_fields_eta_hand_case_and_explosion():
    y = np.array([0, 0, 1, 2, 3, 5, 8, 400] + [1] * 192, np.float32)
    loc = np.log(np.full(y.shape, 2.0))
    disp = np.full(y.shape, math.log(50.0))
    f, internals = es.metric_fields_eta(_StubPred(), loc, disp, y)
    sq = (np.log1p(2.0) - np.log1p(y.astype(np.float64))) ** 2
    sub = br.subset_index(y)
    assert sub["top1"].size == 2
    for s in ("all", "nonzero", "top1"):
        assert f[f"mse_log1p_{s}"] == pytest.approx(float(sq[sub[s]].mean()), rel=1e-12)
    assert f[es.EXPLODE_KEY] is False
    # a mean far off the target: CRPS over 20 is flagged
    big = np.full(y.shape, 500.0, np.float32)
    f2, _ = es.metric_fields_eta(_StubPred(), np.log(np.full(y.shape, 0.5)), disp, big)
    assert f2["crps_all"] > 20 and f2[es.EXPLODE_KEY] is True
    assert es.EXPLODE_AT == 20.0


def test_crps_by_kind_counts_each_record_once():
    def rec(kind, ev, crps, nonfinite=0):
        return {"kind": kind, "eval": ev, "crps_all": crps, es.EXPLODE_KEY:
                None if crps is None else crps > 20, "n_crps_nonfinite": nonfinite}
    t = rec("trained", "score", 25.0)
    recs = [t, {**t, "kind": "depthlaw", "depth_source": "trained"},      # the copy: not counted
            rec("trained", "val", 30.0), rec("trained", "score", 1.0, nonfinite=3),
            rec("shuffle", "score", None, nonfinite=5), rec("swap", "score", 2.0),
            rec("shuffle", "score", 21.0)]
    ck = es.crps_by_kind(recs, ("trained_score", "val", "shuffle", "swap"))
    assert ck == {"trained_score": {"n_records": 2, "n_crps_gt_20": 1, "n_crps_nonfinite": 1},
                  "val": {"n_records": 1, "n_crps_gt_20": 1, "n_crps_nonfinite": 0},
                  "shuffle": {"n_records": 2, "n_crps_gt_20": 1, "n_crps_nonfinite": 1},
                  "swap": {"n_records": 1, "n_crps_gt_20": 0, "n_crps_nonfinite": 0}}
    law = es.crps_by_kind([rec("law", "score", 40.0), {**rec("law", "score", 40.0),
                                                       "kind": "depthlaw"}], ("law",))
    assert law == {"law": {"n_records": 1, "n_crps_gt_20": 1, "n_crps_nonfinite": 0}}


# ---------------------------------------------------------------------------------------------
# depth law: the identity map at init scales by d'/d
# ---------------------------------------------------------------------------------------------


def test_identity_depth_law(synth_dir, runs):
    """At init f is the identity, so mu = (1 + X) d'/d and log2(sum mu / sum X) =
    log2(d'/d) + log2(sum(1 + X) / sum X): the true ratio plus the +1 of log1p, exactly."""
    d = _write_run(runs / "identity", synth_dir, "A", "real", perturb=False)
    pred = _pred(synth_dir, d)
    es.score_trained(pred, synth_dir["rows"], synth_dir["bed"], d)
    s = json.loads((d / "scores.json").read_text())
    dl = [r for r in s["records"] if r["kind"] == "depthlaw"]
    assert len(dl) == 8
    depth = {r["pid"]: float(r["depth"]) for r in synth_dir["rows"]}
    for r in dl:
        true = math.log2(depth[r["target_pid"]] / depth[r["source_pid"]])
        assert r["depth_log2_ratio_true"] == pytest.approx(true, abs=1e-12)
        x, _, _, _ = _eval_by_hand(synth_dir, pred, r["source_pid"], r["target_pid"],
                                   r["source_pid"], r["target_pid"])
        sx = float(x.astype(np.float64).sum())
        plus_one = math.log2((sx + x.size) / sx)
        assert r["depth_log2_scale_pred"] == pytest.approx(true + plus_one, abs=1e-4)
    assert sorted(round(r["depth_log2_ratio_true"], 9) for r in dl) == [-2, -2, -1, -1, 1, 1, 2, 2]


# ---------------------------------------------------------------------------------------------
# row-2 nocov: loc cached BEFORE the offset, each target adds its own log d'
# ---------------------------------------------------------------------------------------------


def test_nocov_row2_cache_adds_each_targets_depth(synth_dir, scored_a2_nocov):
    pred, _, s = scored_a2_nocov
    assert pred.per_bin and pred.model == "nocov"
    recs = [r for r in s["records"] if r["kind"] == "trained" and r["eval"] == "score"
            and r["source_pid"] == T1B]
    tgts = {r["target_pid"] for r in recs}
    assert {T1D15, T1D75} <= tgts                 # one source, targets of different depths
    for r in recs:
        x, y, loc, disp = _eval_by_hand(synth_dir, pred, r["source_pid"], r["target_pid"],
                                        r["cov_src_pid"], r["cov_tgt_pid"])
        want = _mse_by_hand(pred, loc, y)
        fields, _ = lscore.metric_fields(pred, loc, disp, y)
        assert r["crps_all"] == pytest.approx(fields["crps_all"], rel=1e-12)
        for k in MSE_KEYS:
            assert r[k] == pytest.approx(want[k], rel=1e-9, abs=1e-12)
    # depth-law records of the twin follow its own d' too
    dl = {r["target_pid"]: r for r in s["records"]
          if r["kind"] == "depthlaw" and r["source_pid"] == T1B}
    assert dl[T1D15]["depth_log2_scale_pred"] - dl[T1D75]["depth_log2_scale_pred"] == \
        pytest.approx(1.0, abs=1e-4)


def test_pre_offset_cache_is_bit_exact(synth_dir, scored_a2_nocov, tmp_path):
    pred, _, _ = scored_a2_nocov
    view = es.PreOffsetCache(pred)
    for tgt in (T1D15, T1D75, T1B):                 # the first call fills, the rest reuse
        loc, disp = view.predict(T1B, T1B, tgt, "chr19")
        w_loc, w_disp = pred.predict(T1B, T1B, tgt, "chr19")
        assert loc.dtype == w_loc.dtype == np.float32
        np.testing.assert_array_equal(loc, w_loc)
        np.testing.assert_array_equal(disp, w_disp)
    assert list(view.cache) == [(T1B, "chr19")]
    with pytest.raises(ValueError):
        es.PreOffsetCache(_pred(synth_dir, _write_run(tmp_path, synth_dir, "A2", "real")))


def test_nocov_row2_records_equal_ungrouped(synth_dir, scored_a2_nocov, tmp_path):
    """The by-source path gives the same records as scoring each job on its own."""
    pred, _, s = scored_a2_nocov
    jobs = [{"kind": "trained", "eval": "score", "pair": p, "x_pid": p["source_pid"],
             "y_pid": p["target_pid"], "cov_src": p["source_pid"], "cov_tgt": p["target_pid"]}
            for p in lpairs.train_pairs(synth_dir["rows"], G)][:6]
    es._setup(pred, synth_dir["rows"], synth_dir["bed"])
    grouped = es._run_jobs(jobs, 1)
    lscore._CTX["by_source"] = False
    single = es._run_jobs(jobs, 1)
    for (a, _, _), (b, _, _) in zip(grouped, single):
        assert a[0]["crps_all"] == b[0]["crps_all"]
        assert a[0]["mse_log1p_all"] == b[0]["mse_log1p_all"]


# ---------------------------------------------------------------------------------------------
# figdata
# ---------------------------------------------------------------------------------------------


def test_figdata_keys(synth_dir, scored_a, scored_a2_nocov):
    _, d, s = scored_a
    fd = np.load(d / "figdata.npz")
    n_score = sum(1 for r in s["records"] if r["kind"] == "trained" and r["eval"] == "score")
    metas = [k for k in fd.files if k.startswith("meta__")]
    snips = sorted(k for k in fd.files if k.startswith("snip__"))
    assert len(metas) == n_score and all(fd[k].shape == (3, 161) for k in metas)
    assert snips == [f"snip__{a}__{i}" for a in ("depth", "extsize", "pe") for i in range(3)]
    assert all(fd[k].shape == (5, 400) and fd[k].dtype == np.float32 for k in snips)
    assert len(s["snippets"]) == 9
    _, d2, _ = scored_a2_nocov
    assert np.load(d2 / "figdata.npz").files == []


# ---------------------------------------------------------------------------------------------
# law pass
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rung", ("A", "A2"))
def test_law_scope_depth_and_all(rung, synth_dir, runs):
    rows = synth_dir["rows"]
    d = _write_run(runs / f"law_{rung}", synth_dir, rung, "real")
    pred = _pred(synth_dir, d)
    es.score_law(pred, rows, synth_dir["bed"], d)
    law = json.loads((d / "law.json").read_text())
    assert law["t125"]["law_scope"] == "depth"
    assert law["t125"]["metrics_added"] == list(MSE_KEYS)
    lp = [r for r in law["records"] if r["kind"] == "law"]
    assert [(r["source_pid"], r["target_pid"]) for r in lp] == \
        [(p["source_pid"], p["target_pid"]) for p in ep.law_pairs_depth(rows, G)]
    assert len(lp) == 4 and all(r["arm_src"] == r["arm_tgt"] == "depth" for r in lp)
    dl = [r for r in law["records"] if r["kind"] == "depthlaw"]
    assert len(dl) == 4 and all(r["depth_source"] == "law" for r in dl)
    assert all(k in r and isinstance(r[es.EXPLODE_KEY], bool) for r in lp for k in MSE_KEYS)
    assert law["crps_by_kind"] == {"law": {
        "n_records": 4, "n_crps_gt_20": sum(r[es.EXPLODE_KEY] for r in lp),
        "n_crps_nonfinite": sum(r["n_crps_nonfinite"] > 0 for r in lp)}}
    assert (d / "LAW_DONE").is_file()

    d_all = _write_run(runs / f"lawall_{rung}", synth_dir, rung, "real")
    es.score_law(_pred(synth_dir, d_all), rows, synth_dir["bed"], d_all, law_scope="all")
    law_all = json.loads((d_all / "law.json").read_text())
    assert law_all["t125"]["law_scope"] == "all"
    lp_all = [r for r in law_all["records"] if r["kind"] == "law"]
    assert [(r["source_pid"], r["target_pid"]) for r in lp_all] == \
        [(p["source_pid"], p["target_pid"]) for p in lpairs.law_pairs(rows, G)]
    assert len(lp_all) == 24
    # the depth->depth records are the same under both scopes
    dd = {(r["source_pid"], r["target_pid"]): r["crps_all"] for r in lp}
    for r in lp_all:
        if (r["source_pid"], r["target_pid"]) in dd:
            assert r["crps_all"] == dd[(r["source_pid"], r["target_pid"])]
    with pytest.raises(ValueError):
        es.score_law(pred, rows, synth_dir["bed"], d, law_scope="arm")


def test_law_workers_match(synth_dir, runs):
    rows = synth_dir["rows"]
    d1 = _write_run(runs / "w1", synth_dir, "B2", "nocov")
    es.score_law(_pred(synth_dir, d1), rows, synth_dir["bed"], d1, workers=1, law_scope="all")
    d2 = _write_run(runs / "w2", synth_dir, "B2", "nocov")
    es.score_law(_pred(synth_dir, d2), rows, synth_dir["bed"], d2, workers=2, law_scope="all")
    a = json.loads((d1 / "law.json").read_text())["records"]
    b = json.loads((d2 / "law.json").read_text())["records"]
    assert [(r["kind"], r["source_pid"], r["target_pid"], r["crps_all"], r["mse_log1p_all"])
            for r in a] == \
        [(r["kind"], r["source_pid"], r["target_pid"], r["crps_all"], r["mse_log1p_all"])
         for r in b]


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _cli_args(sd, d):
    return [str(sd["manifest"]), str(sd["cov"]), str(sd["root"]), str(sd["bed"]), str(d)]


def test_cli_trained_runs_then_skips(synth_dir, runs, capsys):
    d = _write_run(runs / "cli", synth_dir, "A", "nocov")
    assert es.main(["trained", *_cli_args(synth_dir, d), "--workers", "1"]) == 0
    assert (d / "SCORE_DONE").is_file() and (d / "scores.json").is_file()
    t0 = os.stat(d / "scores.json").st_mtime_ns
    capsys.readouterr()
    assert es.main(["trained", *_cli_args(synth_dir, d), "--workers", "1"]) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert os.stat(d / "scores.json").st_mtime_ns == t0


def test_cli_law_skips_when_done(synth_dir, tmp_path, capsys):
    d = tmp_path / "X"
    d.mkdir()
    (d / "LAW_DONE").write_text("x")
    assert es.main(["law", *_cli_args(synth_dir, d), "--law-scope", "all"]) == 0
    assert "nothing to do" in capsys.readouterr().out
    assert not (d / "law.json").exists()


def test_cli_script_mode_law_all(synth_dir, runs):
    """`python tools/t125/eta/score.py law ... --law-scope all` as a script (the SLURM path)."""
    d = _write_run(runs / "script", synth_dir, "A", "real")
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "t125" / "eta" / "score.py"), "law",
                        *_cli_args(synth_dir, d), "--workers", "1", "--law-scope", "all"],
                       capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert r.returncode == 0, r.stderr
    law = json.loads((d / "law.json").read_text())
    assert law["t125"]["law_scope"] == "all"
    assert sum(r["kind"] == "law" for r in law["records"]) == 24
    assert (d / "LAW_DONE").is_file()


# ---------------------------------------------------------------------------------------------
# a run trained by eta.train, then scored
# ---------------------------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
@pytest.mark.parametrize("rung,model", [("A", "real"), ("A2", "nocov")])
def test_trained_run_scores(rung, model, synth_dir, tmp_path):
    from eta import train as et
    sd = synth_dir
    d = et.run_training(sd["manifest"], sd["cov"], sd["root"], tmp_path, rung, G, model, 0,
                        max_steps=3, eval_every=3, device="cpu", window=256, batch=8,
                        knot_sample=20_000, knot_block=256)
    assert (d / "TRAIN_DONE").is_file() and d.name == ep.run_name(rung, G, model, 0)
    args = [str(sd["manifest"]), str(sd["cov"]), str(sd["root"]), str(sd["bed"]), str(d)]
    assert es.main(["trained", *args, "--workers", "1"]) == 0
    assert es.main(["law", *args, "--workers", "1", "--law-scope", "depth"]) == 0
    s = json.loads((d / "scores.json").read_text())
    law = json.loads((d / "law.json").read_text())
    assert (d / "SCORE_DONE").is_file() and (d / "LAW_DONE").is_file()
    assert s["run"]["rung"] == rung and s["run"]["model"] == model
    fig = np.load(d / "figdata.npz").files
    if model == "nocov":
        assert fig == []
    else:
        assert any(k.startswith("meta__") for k in fig)
    assert set(s["crps_by_kind"]) == {"trained_score", "val", "shuffle", "swap"}
    assert set(law["crps_by_kind"]) == {"law"} and law["t125"]["law_scope"] == "depth"
    assert law["crps_by_kind"]["law"]["n_records"] == len(ep.law_pairs_depth(sd["rows"], G))
    kinds = {r["kind"] for r in s["records"]}
    assert kinds == {"trained", "shuffle", "swap", "depthlaw"}
    assert {r["kind"] for r in law["records"]} == {"law", "depthlaw"}
    for r in s["records"] + law["records"]:
        for k in ("crps_all", "crps_nonzero", "crps_top1") + MSE_KEYS:
            assert r[k] is not None and math.isfinite(r[k]), (r["kind"], k)
        assert isinstance(r[es.EXPLODE_KEY], bool)
    for r in s["records"]:
        if r["kind"] == "trained" and r["eval"] == "score":
            for k in ("crps_oracle_scaled_all", "scale_error_all", "c_star_all", "pit_hist",
                      "describe"):
                assert k in r
            assert math.isfinite(r["crps_oracle_scaled_all"]) and math.isfinite(r["scale_error_all"])
    for r in s["records"] + law["records"]:
        if r["kind"] == "depthlaw":
            assert math.isfinite(r["depth_log2_scale_pred"])
            assert math.isfinite(r["depth_log2_ratio_true"])
