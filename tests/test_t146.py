"""t146 — the final CAC testbed kit: leakage audit, loss normalisation, chaining, the shifted
mappability channel, the decile strata, and a tiny end-to-end train + score per new code path on
synthetic products (CPU, seconds)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import data, encoding, pairs as lp, synth  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from t146 import features as F  # noqa: E402
from t146 import model as M  # noqa: E402
from t146 import score as S  # noqa: E402
from t146 import tasks as T  # noqa: E402
from t146 import train as TR  # noqa: E402

FIXTURE = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
COV = REPO / "tools" / "t118" / "covariates.tsv"


# ---- the task table -----------------------------------------------------------------------------

def test_task_table_counts_and_smoke_rows():
    t = T.tasks()
    assert len(t) == 42 and [r["index"] for r in t] == list(range(42))
    by = {}
    for r in t:
        by.setdefault(r["arm"], []).append(r)
    assert {a: len(v) for a, v in by.items()} == {"alone": 6, "matched": 3, "xassay": 9,
                                                   "denoise": 9, "consist": 6, "map": 3,
                                                   "mapshift": 3, "dna": 3}
    assert len({r["run_name"] for r in t}) == 42
    assert [t[i]["run_name"] for i in T.SMOKE_INDICES] == list(T.SMOKE_RUNS)


# ---- leakage audit ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fx():
    rows = lp.read_manifest(FIXTURE)
    enc = encoding.Encoder.fit(encoding.read_covariates(COV), T.fit_pids(rows))
    return rows, enc


def test_xassay_pairs_are_base_sourced_cross_assay_same_cell(fx):
    rows, _ = fx
    xa = T.xassay_pairs(rows)
    by = {r["pid"]: r for r in rows}
    assert len(xa) == 6 * 19             # 3 cells x 2 directions x (base + 18 non-held-out arms)
    for p in xa:
        s, t = by[p["source_pid"]], by[p["target_pid"]]
        assert s["arm"] == "base" and s["cell"] == t["cell"] and s["track"] != t["track"]
        assert s["assay"] != t["assay"] and (t["arm"], t["level"]) != ("depth", "7.5M")
    assert not any(by[p["source_pid"]]["track"] == "C12M02" or by[p["target_pid"]]["track"]
                   == "C12M02" for p in xa)


def test_leakage_audit_passes_and_catches_leaks(fx):
    rows, enc = fx
    law = T.law_pairs(rows)
    ok = T.leakage_audit(rows, enc, T.train_pairs(rows) + T.xassay_pairs(rows), law)
    assert ok["pass"] and ok["pid_overlap"] == [] and ok["covariate_vector_overlap"] == []
    leak = dict(law[5], direction="base_to_arm")
    with pytest.raises(ValueError, match="leakage"):
        T.leakage_audit(rows, enc, T.train_pairs(rows) + [leak], law)
    bad_src = dict(T.xassay_pairs(rows)[0])
    arm = next(r for r in rows if r["track"] == bad_src["track"] and r["arm"] == "crop")
    bad_src["source_pid"] = arm["pid"]
    with pytest.raises(ValueError):
        T.leakage_audit(rows, enc, [bad_src], law)
    # consist pairs query scored covariate pairs by design and are exempt
    cons = [dict(p, direction="consist") for p in T.consist_pairs(rows)]
    assert T.leakage_audit(rows, enc, T.train_pairs(rows) + cons, law)["pass"]
    assert all(not lp._held(p, T.HOLDOUT) for p in cons)


# ---- shifted mappability, strata ------------------------------------------------------------------

def test_shifted_window_is_a_circular_shift_with_zero_pads():
    m = np.arange(1000, dtype=np.float32)
    shift = 137
    full = F.shifted_window(m, 0, 1000, shift)
    assert np.array_equal(full, np.roll(m, shift))
    w = F.shifted_window(m, -5, 20, shift)
    assert np.all(w[:5] == 0) and np.array_equal(w[5:], np.roll(m, shift)[:20])
    w = F.shifted_window(m, 990, 1010, shift)
    assert np.array_equal(w[:10], np.roll(m, shift)[990:]) and np.all(w[10:] == 0)
    # the default is 5 Mb in 25 bp bins
    assert T.SHIFT_BINS * 25 == 5_000_000


def _bed(tmp_path, lines):
    p = tmp_path / "bl.bed"
    p.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in lines))
    return p


def test_decile_strata_per_1kb_block_blacklist_and_tail(tmp_path):
    rng = np.random.default_rng(0)
    n_bins = {"chr1": 40 * 300 + 7, "chr2": 40 * 200}
    maps = {c: rng.random(n).astype(np.float32) for c, n in n_bins.items()}
    bl = _bed(tmp_path, [("chr1", 0, 1), ("chr2", 40 * 25 * 10 + 3, 40 * 25 * 10 + 4)])
    dec, info = F.strata(maps, n_bins, bl)
    d1, d2 = dec["chr1"], dec["chr2"]
    assert d1.shape == (n_bins["chr1"],) and d1.dtype == np.int8
    assert np.all(d1[-7:] == -1)                      # the tail past the last whole block
    assert np.all(d1[:40] == -1) and np.all(d2[400:440] == -1)   # blacklisted blocks
    blocks = np.concatenate([d1[:12000].reshape(-1, 40), d2.reshape(-1, 40)])
    assert np.all(blocks == blocks[:, :1])            # one decile per block
    kept = blocks[:, 0][blocks[:, 0] >= 0]
    counts = np.bincount(kept, minlength=10)
    assert counts.min() >= 49 and counts.max() <= 50   # 498 kept blocks
    # deciles rank the 1 kb mean: dec0 holds the lowest mappability
    mean1k = maps["chr1"][:12000].reshape(-1, 40).mean(1)
    lo, hi = mean1k[d1[:12000:40] == 0], mean1k[d1[:12000:40] == 9]
    assert lo.max() < hi.min()
    assert info["n_blacklisted_blocks"] == 2


# ---- synthetic end to end -------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synth_kit(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("t146")
    old = synth.TRACK_SPECS["T2"]["cell"]
    synth.TRACK_SPECS["T2"]["cell"] = synth.TRACK_SPECS["T1"]["cell"]   # one cell, two assays
    try:
        manifest, covariates = synth.make_products(tmp / "products", seed=0)
    finally:
        synth.TRACK_SPECS["T2"]["cell"] = old
    cache = tmp / "cache"
    rows = lp.read_manifest(manifest)
    for i in range(len(rows)):
        assert ltrain.main(["cache", str(manifest), str(tmp / "products"), str(cache), str(i)]) == 0
    corpus = data.Corpus(cache)
    feats = tmp / "features"
    feats.mkdir()
    rng = np.random.default_rng(1)
    pid = rows[0]["pid"]
    for c in corpus.chroms(pid, "counts"):
        n = corpus.n_bins(pid, "counts", c)
        np.save(feats / f"map25__{c}.npy", rng.random(n).astype(np.float32))
        np.save(feats / f"dna__{c}.npy", rng.integers(0, 5, n * 25).astype(np.uint8))
        np.save(feats / f"dec__{c}.npy", rng.integers(-1, 10, n).astype(np.int8))
    (feats / "FEATURES_DONE").write_text("ok\n")
    bl = _bed(tmp, [("chr19", 100 * 25, 110 * 25)])
    return {"tmp": tmp, "manifest": manifest, "cov": covariates, "cache": cache, "feats": feats,
            "bl": bl, "rows": rows}


SMALL = {"window": 128, "batch": 4, "norm_batches": 2}


def _train(kit, name, steps=2, runs="runs"):
    return TR.run_training(kit["manifest"], kit["cov"], kit["cache"], kit["tmp"] / runs,
                           T.task_by_name(name), features_dir=kit["feats"], max_steps=steps,
                           eval_every=1, device="cpu", **SMALL)


@pytest.mark.parametrize("name", list(T.SMOKE_RUNS))
def test_every_code_path_trains_and_scores(synth_kit, name):
    rd = _train(synth_kit, name)
    cfg = json.loads((rd / "config.json").read_text())
    assert (rd / "TRAIN_DONE").is_file() and (rd / "ckpt.pt").is_file()
    assert json.loads((rd / "leakage_audit.json").read_text())["pass"]
    arm = T.ARMS[cfg["arm"]]
    # loss normalisation: the correction factor is 1 / L0_corr exactly where the design says
    if arm["corr_scale"] == "step0":
        assert cfg["corr_factor"] == pytest.approx(1.0 / cfg["norm"]["L0_corr"])
    else:
        assert cfg["corr_factor"] == 1.0
    if arm["aux"] in ("xassay", "denoise"):
        assert cfg["aux_factor"] == pytest.approx(cfg["weight"] / cfg["norm"]["L0_aux"])
    run = S.Run(rd, synth_kit["cache"], synth_kit["feats"])
    path = S.score_run(run, synth_kit["rows"], synth_kit["bl"], workers=1, law_stride=7,
                       mix_stride=2, mix_draws=2)
    recs = json.loads(path.read_text())["records"]
    routes = {(r["kind"], r["route"]) for r in recs}
    assert {("trained", "direct"), ("law", "direct"), ("shuffle", "direct")} <= routes
    if cfg["arm"] in T.CHAIN_ARMS:
        assert {("law", "chain"), ("law", "mix"), ("ceiling", "ceiling")} <= routes
    for r in recs:
        assert np.isfinite(r["crps_all"])
        assert r["crps_all"] == pytest.approx(r["crps_oracle_scaled_all"] + r["scale_error_all"])
        assert r["n_blacklisted"] == 10
        assert sum(r[f"n_dec{k}"] for k in range(10)) <= r["n_all"]


def test_step0_norm_is_the_initial_models_mean_loss(synth_kit):
    rows = synth_kit["rows"]
    corpus = data.Corpus(synth_kit["cache"])
    tp = T.train_pairs(rows)
    enc = encoding.Encoder.fit(encoding.read_covariates(synth_kit["cov"]), T.fit_pids(rows))
    chroms, lens = ltrain.training_chroms(corpus, tp[0]["source_pid"], "counts")
    torch.manual_seed(0)
    m = M.T146Model(aux_head=True)
    w = TR.Windows(corpus, chroms, lens, 128, None, (), False)
    bt = TR.Batches(m, w, enc, rows, tp, torch.device("cpu"), 4)
    n0 = TR.step0_losses(bt, "xassay", 0, 3)
    with torch.no_grad():
        r = np.random.default_rng([0, 7147, 0])
        want = np.mean([float(bt.corr(r)) for _ in range(3)])
        r = np.random.default_rng([0, 7147, 1])
        want_aux = np.mean([float(bt.xassay(r)) for _ in range(3)])
    assert n0["L0_corr"] == pytest.approx(want) and n0["L0_aux"] == pytest.approx(want_aux)
    # matched and xassay with the same seed see the same L0_corr (the identity map at init)
    torch.manual_seed(0)
    m2 = M.T146Model(aux_head=False)
    bt2 = TR.Batches(m2, w, enc, rows, tp, torch.device("cpu"), 4)
    assert TR.step0_losses(bt2, None, 0, 3)["L0_corr"] == pytest.approx(n0["L0_corr"])


def test_alone_arm_initialises_exactly_as_old_rung_d():
    from ladder.model import Ladder
    torch.manual_seed(3)
    old = Ladder("D", "counts", {"knots_x": np.arange(12.0), "n0": 5.0, "sigma0": 0.5})
    torch.manual_seed(3)
    new = M.T146Model()
    so, sn = old.state_dict(), new.state_dict()
    assert list(so) == list(sn)
    assert all(torch.equal(so[k], sn[k]) for k in so)


def test_chaining_at_identity_and_its_shapes(synth_kit):
    """At initialisation every map is the identity (loc = log1p x): leg 1's mean is X + 1, and
    leg 2 fed it returns mean X + 2 — the chained route composes the two maps bin by bin."""
    rd = _train(synth_kit, "alone_s0", steps=0, runs="runs_init")
    run = S.Run(rd, synth_kit["cache"], synth_kit["feats"])
    rows = synth_kit["rows"]
    p = T.law_pairs(rows)[3]
    base = T.base_pid_of(rows, p["source_pid"])
    job = {"x_pid": p["source_pid"], "y_pid": p["target_pid"], "cov_src": p["source_pid"],
           "cov_tgt": p["target_pid"], "base": base, "law_index": 3, "mix_draws": 4}
    X = np.asarray(run.corpus.get(p["source_pid"], "counts", "chr19"), np.float64)
    m_d, n_d = S.route_nb(run, "direct", job, "chr19", 0)
    m_c, n_c = S.route_nb(run, "chain", job, "chr19", 0)
    m_x, n_x = S.route_nb(run, "mix", job, "chr19", 0)
    for a in (m_d, n_d, m_c, n_c, m_x, n_x):
        assert a.shape == X.shape and np.all(np.isfinite(a))
    np.testing.assert_allclose(m_d, X + 1, rtol=1e-5)
    np.testing.assert_allclose(m_c, X + 2, rtol=1e-4)
    assert np.all(n_x >= M.N_MIN) and np.all(n_x <= M.N_MAX)
    # the mixture's mean is the mean of leg 2 over draws of leg 1: E[draw] + 1 = X + 2 on average
    assert abs(m_x.mean() - (X + 2).mean()) < 0.1 * (X + 2).mean()


def test_consistency_loss_at_init_is_the_identity_gap(synth_kit):
    rows = synth_kit["rows"]
    corpus = data.Corpus(synth_kit["cache"])
    tp = T.train_pairs(rows)
    enc = encoding.Encoder.fit(encoding.read_covariates(synth_kit["cov"]), T.fit_pids(rows))
    chroms, lens = ltrain.training_chroms(corpus, tp[0]["source_pid"], "counts")
    torch.manual_seed(0)
    m = M.T146Model()
    w = TR.Windows(corpus, chroms, lens, 128, None, (), False)
    bt = TR.Batches(m, w, enc, rows, tp, torch.device("cpu"), 4)
    lc = float(bt.consist(np.random.default_rng(5)))
    # direct loc = log1p(x), chained loc = log1p(x + 1): the gap per bin is log((x+2)/(x+1))^2
    r = np.random.default_rng(5)
    pi = r.integers(len(bt.cp), size=4)
    ci, st = w.positions(r, 4)
    x = np.stack([np.asarray(corpus.window(bt.cp[i]["source_pid"], "counts", chroms[c], s, s + 128),
                             np.float64) for i, c, s in zip(pi, ci, st)])
    want = np.mean(np.log((x + 2) / (x + 1)) ** 2)
    assert lc == pytest.approx(want, rel=1e-4)


def test_shifted_input_reaches_the_model_shifted(synth_kit):
    """The mapshift arm's whole-chromosome input equals np.roll(map25, SHIFT) and its training
    windows read the same array."""
    feats = F.Features(synth_kit["feats"])
    m = np.asarray(feats.map_full("chr1"))
    ex, _ = TR.full_inputs(feats, ("mapshift",), False, "chr1", m.shape[0])
    assert np.array_equal(ex[0], np.roll(m, T.SHIFT_BINS % m.shape[0]))
    w = feats.window("mapshift", "chr1", 50, 90)
    assert np.array_equal(w, ex[0][50:90])


def test_committed_task_table_matches_tasks_py():
    want = "\t".join(T.TASK_KEYS) + "\n" + T._tsv(T.tasks(), T.TASK_KEYS)
    assert (REPO / "tools" / "t146" / "tasks.tsv").read_text() == want


def test_tables_from_the_scored_runs(synth_kit, tmp_path):
    """Runs after the code-path tests in this module (their scored runs are the input)."""
    import pandas as pd
    from t146 import tables
    runs = synth_kit["tmp"] / "runs"
    if not list(runs.glob("*/scores.json")):
        pytest.skip("no scored runs (run the whole module)")
    assert tables.main([str(runs), str(tmp_path)]) == 0
    p = pd.read_csv(tmp_path / "pooled.tsv", sep="\t")
    a = p[(p.arm == "alone") & (p.group == "never") & (p.knob == "all")]
    assert set(a.route) == {"direct", "chain", "mix", "ceiling"}
    np.testing.assert_allclose(a.crps_all, a.crps_oracle_scaled_all + a.scale_error_all, rtol=1e-3)
    assert (tmp_path / "cells.tsv").is_file() and (tmp_path / "deciles.tsv").is_file()
