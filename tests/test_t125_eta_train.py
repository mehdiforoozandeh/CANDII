"""t125 part 1 — `tools/t125/eta/train.py`: one (rung, g, model, seed) run with the depth offset.

Written from plan/ETA_OFFSET_AND_H_DESIGN.md "Part 1" and the plan's pinned interfaces (chunk
t125-P1d): t118's schedule and sampler streams, x = eta_in = log1p(X) - log d_src, the target NB
at log mu' = log d' + eta_out, spread knots built on eta_in, the nocov twin keeps the offset (PI).
Runs on CPU on `ladder.synth.make_products`.
"""
from __future__ import annotations

import json
import math
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

from ladder import base, data, encoding, synth  # noqa: E402
from ladder import model as lmodel  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from eta import offset  # noqa: E402
from eta import model as em  # noqa: E402
from eta import pairs as ep  # noqa: E402
from eta import train as et  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
PY = sys.executable
TRAIN_PY = ROOT / "tools" / "t125" / "eta" / "train.py"
FIXTURE_MANIFEST = ROOT / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
FAST = {"window": 256, "batch": 8, "knot_sample": 20_000, "knot_block": 256}
DEV = torch.device("cpu")

T118_CONFIG_KEYS = {
    "run_name", "rung", "g", "space", "model", "seed", "window", "batch", "lr", "warmup",
    "grad_clip", "dtype", "max_steps", "eval_every", "patience", "knot_sample", "knot_block", "n0",
    "sigma0", "device", "n_cov", "cov_fields", "context", "n_theta", "train_chroms", "val_chroms",
    "score_chroms", "sampling", "nocov_rng", "knots", "git_sha", "manifest", "manifest_md5",
    "covariates", "covariates_md5", "data_dir", "fit_pids", "train_pairs", "ids_perm", "knots_x",
}


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth_eta_train")
    manifest, cov = synth.make_products(root / "products", 0)
    rows = lpairs.read_manifest(manifest)
    return {"root": root, "products": root / "products", "manifest": manifest, "cov": cov,
            "rows": rows, "log_d": offset.depth_table(rows)}


def _train(prod, rung, model, runs="runs", seed=0, g="all", steps=5, **kw):
    return et.run_training(prod["manifest"], prod["cov"], prod["products"], prod["root"] / runs,
                           rung, g, model, seed, max_steps=steps, eval_every=kw.pop("eval_every", 5),
                           device="cpu", **FAST, **kw)


# -- constants ------------------------------------------------------------------------------------

def test_defaults_are_t118s():
    assert et.DEFAULTS == ltrain.DEFAULTS
    assert et.DEFAULTS is not ltrain.DEFAULTS
    for k, v in {"window": 2048, "batch": 32, "lr": 1e-3, "warmup": 100, "grad_clip": 1.0,
                 "max_steps": 4000, "eval_every": 250, "patience": 4, "knot_sample": 4_000_000,
                 "knot_block": 1024, "n0": 5.0, "sigma0": 0.5}.items():
        assert et.DEFAULTS[k] == v


def test_run_name():
    assert et.run_name("A", "C19M16", "real", 0) == "A_C19M16_counts_real_s0"
    assert et.run_name("Aold", "all", "nocov", 2) == "Aold_all_counts_nocov_s2"


# -- knots on eta_in ------------------------------------------------------------------------------

def test_knot_sample_is_t118s_draw_transformed_per_block(prod):
    corpus = data.Corpus(prod["products"])
    fit = lpairs.fit_pids(prod["rows"], "all")
    chroms, lens = ltrain.training_chroms(corpus, fit[0], "counts")
    n, block = 5000, 256
    got = et.knot_sample_eta(corpus, fit, chroms, lens, n, block, prod["log_d"])
    # independent: rng 0, n_blocks draws of pid, then chrom ∝ length, then u; each block eta'd
    rng = np.random.default_rng(0)
    nb = -(-n // block)
    pid_i = rng.integers(len(fit), size=nb)
    chrom_i = rng.choice(len(chroms), size=nb, p=lens / lens.sum())
    u = rng.random(nb)
    parts = []
    for p, c, uu in zip(pid_i, chrom_i, u):
        s = int(uu * (max(0, int(lens[c]) - block) + 1))
        X = corpus.window(fit[p], "counts", chroms[c], s, s + block).astype(np.float64)
        parts.append(np.log1p(X) - math.log(float(
            next(r for r in prod["rows"] if r["pid"] == fit[p])["depth"]) / 1e6))
    want = np.concatenate(parts)[:n]
    assert got.dtype == np.float64 and got.shape == (n,)
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
    # the raw-x sample of t118 differs by exactly the per-block depth shift
    raw = ltrain.knot_sample(corpus, fit, "counts", chroms, lens, n, block)
    shift = np.repeat([prod["log_d"][fit[p]] for p in pid_i], block)[:n]
    np.testing.assert_allclose(got, raw - shift, rtol=0, atol=1e-5)


# -- sampler --------------------------------------------------------------------------------------

def _samplers(prod, model):
    corpus = data.Corpus(prod["products"])
    tp = lpairs.train_pairs(prod["rows"], "all")
    chroms, lens = ltrain.training_chroms(corpus, tp[0]["source_pid"], "counts")
    s = et.EtaSampler(corpus, tp, chroms, lens, 64, 3, 11, model, prod["log_d"])
    t = ltrain.Sampler(corpus, "counts", tp, chroms, lens, 64, 3, 11, model, None)
    return tp, s, t


@pytest.mark.parametrize("model", ["real", "nocov"])
def test_sampler_streams_match_t118(prod, model):
    tp, s, t = _samplers(prod, model)
    for _ in range(3):
        pi, ci, xs, ys, mask, ld = s.draw(6)
        pi0, ci0, xs0, ys0, mask0 = t.draw(6)
        assert np.array_equal(pi, pi0) and np.array_equal(ci, ci0)
        assert np.array_equal(ys, ys0) and np.array_equal(mask, mask0)
        assert xs.dtype == np.float32 and xs.shape == (6, 64 + 6)
        assert ld.dtype == np.float32 and ld.shape == (6,)
        for b in range(6):
            src, tgt = tp[pi[b]]["source_pid"], tp[pi[b]]["target_pid"]
            # x0 = float32(log1p X); eta = log1p X - log d_src
            np.testing.assert_allclose(xs[b], xs0[b].astype(np.float64) - prod["log_d"][src],
                                       rtol=0, atol=1e-5)
            assert ld[b] == np.float32(prod["log_d"][tgt])


def test_nocov_sampler_same_window_stream_as_real(prod):
    _, real, _ = _samplers(prod, "real")
    tp, nocov, _ = _samplers(prod, "nocov")
    redrawn = False
    for _ in range(4):
        a, b = real.draw(8), nocov.draw(8)
        for i in (0, 2, 3, 4, 5):           # pair, x, y, mask, ld_tgt: same stream
            assert np.array_equal(a[i], b[i])
        assert np.array_equal(a[1], a[0])    # real: the pair's own covariates
        redrawn |= not np.array_equal(b[1], b[0])
        # the offset belongs to the counts: ld_tgt is the target of the sample's own pair
        assert np.array_equal(b[5], np.array([prod["log_d"][tp[k]["target_pid"]] for k in b[0]],
                                             dtype=np.float32))
    assert redrawn


# -- validation (the reviewer's hazard: never cache after the offset) -----------------------------

def _stats(prod, fit):
    corpus = data.Corpus(prod["products"])
    xs = np.concatenate([offset.transform_eta(corpus.get(p, "counts", "chr1"), prod["log_d"][p])
                         for p in fit])
    return {"knots_x": base.knot_stats(xs), "n0": 5.0, "sigma0": 0.5}


def _perturbed(rung, stats, seed=3):
    lad = em.EtaLadder(rung, stats)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in list(lad.g.parameters()) + list(lad.form.parameters()):
            p.add_(0.05 * torch.randn(p.shape, generator=gen))
    return lad.eval()


def _manual_val(lad, corpus, tp, cov, log_d, average):
    out = []
    for i, p in enumerate(tp):
        X = corpus.get(p["source_pid"], "counts", "chr22")
        Y = corpus.get(p["target_pid"], "counts", "chr22")
        ls, lt = log_d[p["source_pid"]], log_d[p["target_pid"]]
        if lad.per_bin:
            c = cov if average else cov[i:i + 1]
            loc, disp = em.predict_chrom_bins_eta(lad, X, ls, lt, c, DEV, average=average)
        else:
            th = lmodel.nocov_theta(lad, cov) if average else lad.theta(cov[i:i + 1])[0]
            loc, disp = em.predict_chrom_eta(lad, X, ls, lt, th, DEV)
        y = torch.from_numpy(np.asarray(Y, dtype=np.float32))
        out.append(float(lad.nll(torch.from_numpy(loc), torch.from_numpy(disp), y)))
    return float(np.mean(out))


@pytest.mark.parametrize("rung,average", [("A", False), ("A", True), ("B2", False), ("B2", True),
                                          ("C", True)])
def test_validate_equals_per_pair_loop_with_each_pairs_own_depth(prod, rung, average):
    corpus = data.Corpus(prod["products"])
    tp = lpairs.train_pairs(prod["rows"], "all")
    fit = lpairs.fit_pids(prod["rows"], "all")
    enc = encoding.Encoder.fit(encoding.read_covariates(prod["cov"]), fit)
    cov = torch.from_numpy(np.stack([enc.encode_pair(p["source_pid"], p["target_pid"])
                                     for p in tp]))
    lad = _perturbed(rung, _stats(prod, fit))
    # a source with targets of different depths exists, so a per-source cache after the offset
    # would be wrong
    by_src = {}
    for p in tp:
        by_src.setdefault(p["source_pid"], set()).add(prod["log_d"][p["target_pid"]])
    assert any(len(v) > 1 for v in by_src.values())
    want = _manual_val(lad, corpus, tp, cov, prod["log_d"], average)
    cache = {} if (lad.per_bin and average) else None
    got = et.validate_eta(lad, corpus, tp, cov, prod["log_d"], DEV, average, level_cache=cache)
    assert got == pytest.approx(want, rel=1e-6, abs=1e-6)
    if cache is not None:                    # the cache is reused on a second call
        assert cache
        again = et.validate_eta(lad, corpus, tp, cov, prod["log_d"], DEV, average,
                                level_cache=cache)
        assert again == pytest.approx(want, rel=1e-6, abs=1e-6)
    assert lad.training is False


def test_validate_restores_training_mode(prod):
    corpus = data.Corpus(prod["products"])
    tp = lpairs.train_pairs(prod["rows"], "T1")
    fit = lpairs.fit_pids(prod["rows"], "T1")
    enc = encoding.Encoder.fit(encoding.read_covariates(prod["cov"]), fit)
    cov = torch.from_numpy(np.stack([enc.encode_pair(p["source_pid"], p["target_pid"])
                                     for p in tp]))
    lad = em.EtaLadder("A", _stats(prod, fit)).train()
    et.validate_eta(lad, corpus, tp, cov, prod["log_d"], DEV, False)
    assert lad.training is True


# -- whole runs -----------------------------------------------------------------------------------

RUNS = [("A", "real"), ("A", "nocov"), ("Aold", "real"), ("Aold", "nocov"),
        ("B2", "real"), ("B2", "nocov")]


@pytest.fixture(scope="module")
def runs(prod):
    return {(r, m): _train(prod, r, m) for r, m in RUNS}


@pytest.mark.parametrize("rung,model", RUNS)
def test_run_dir_files_and_config(prod, runs, rung, model):
    rd = runs[(rung, model)]
    assert rd == prod["root"] / "runs" / f"{rung}_all_counts_{model}_s0"
    for f in ("config.json", "ckpt.pt", "train_log.tsv", "timing.json", "TRAIN_DONE"):
        assert (rd / f).is_file(), f
    cfg = json.loads((rd / "config.json").read_text())
    missing = T118_CONFIG_KEYS - set(cfg)
    assert not missing, missing
    assert cfg["rung"] == rung and cfg["model"] == model and cfg["space"] == "counts"
    assert cfg["g"] == "all" and cfg["seed"] == 0
    assert cfg["run_name"] == rd.name
    assert cfg["form_module"] == ep.FORM_MODULE[rung]
    assert cfg["t125"] == {"offset": "log1p(X) - log(d/1e6)", "d_unit": 1e6}
    assert ("row" in cfg) == (rung == "B2")
    if rung == "B2":
        assert cfg["row"] == 2
    fit = lpairs.fit_pids(prod["rows"], "all")
    assert cfg["fit_pids"] == fit
    assert set(fit) <= set(cfg["log_depth"])
    for p in fit:
        assert cfg["log_depth"][p] == pytest.approx(prod["log_d"][p], abs=1e-12)
    assert cfg["max_steps"] == 5 and cfg["window"] == FAST["window"]
    assert cfg["train_chroms"] == ["chr1", "chr2", "chrX"]
    assert cfg["val_chroms"] == ["chr22"] and cfg["score_chroms"] == ["chr19", "chr21"]
    log = (rd / "train_log.tsv").read_text().splitlines()
    assert log[0] == "step\tloss\tval_nll\tseconds"
    assert [int(r.split("\t")[0]) for r in log[1:]] == [0, 5]
    timing = json.loads((rd / "timing.json").read_text())
    assert set(timing) >= {"setup", "train", "total", "peak_rss_mb", "steps_run", "best_step",
                           "val_nll"}
    assert timing["steps_run"] == 5
    ck = torch.load(rd / "ckpt.pt", map_location="cpu", weights_only=True)
    assert set(ck) == {"g", "form", "stats", "encoder", "config", "step", "val_nll"}
    assert ck["config"] == cfg
    assert ck["step"] == timing["best_step"] and ck["val_nll"] == timing["val_nll"]
    assert np.isfinite(ck["val_nll"])
    # knots are built on eta_in: they sit below the raw log1p levels by about log d
    assert ck["stats"]["knots_x"] == cfg["knots_x"]
    assert len(cfg["knots_x"]) == 12
    assert min(cfg["knots_x"]) < -1.0          # zero counts at d ~ 7.5-30 M -> eta <= -2


@pytest.mark.parametrize("rung,model", RUNS)
def test_ckpt_reloads_and_predicts_chr22(prod, runs, rung, model):
    rd = runs[(rung, model)]
    pred = em.load_run(rd, prod["products"], prod["cov"], prod["manifest"], device="cpu")
    assert pred.run_name == rd.name and pred.model == model and pred.rung == rung
    p = lpairs.train_pairs(prod["rows"], "all")[0]
    loc, disp = pred.predict(p["source_pid"], p["source_pid"], p["target_pid"], "chr22")
    n = data.Corpus(prod["products"]).n_bins(p["source_pid"], "counts", "chr22")
    assert loc.shape == (n,) and disp.shape == (n,)
    assert np.all(np.isfinite(loc)) and np.all(np.isfinite(disp))
    # the reloaded model scores chr22 to the val_nll the run recorded for its best state
    corpus = data.Corpus(prod["products"])
    tp = pred.train_pairs
    cov = torch.from_numpy(np.stack([pred.encoder.encode_pair(q["source_pid"], q["target_pid"])
                                     for q in tp]))
    v = et.validate_eta(pred.ladder, corpus, tp, cov, pred.log_d, DEV, model == "nocov")
    assert v == pytest.approx(pred.val_nll, rel=1e-5)


def test_train_done_skips(prod, runs):
    rd = runs[("A", "real")]
    before = (rd / "ckpt.pt").stat().st_mtime_ns
    assert _train(prod, "A", "real") == rd
    assert (rd / "ckpt.pt").stat().st_mtime_ns == before


@pytest.mark.parametrize("rung,model", [("A", "real"), ("B2", "nocov")])
def test_same_seed_bit_identical(prod, runs, rung, model):
    rd2 = _train(prod, rung, model, runs="runs_again")
    a = torch.load(runs[(rung, model)] / "ckpt.pt", map_location="cpu", weights_only=True)
    b = torch.load(rd2 / "ckpt.pt", map_location="cpu", weights_only=True)
    for part in ("g", "form"):
        assert a[part].keys() == b[part].keys()
        for k in a[part]:
            assert torch.equal(a[part][k], b[part][k]), (part, k)
    assert a["val_nll"] == b["val_nll"] and a["step"] == b["step"]


def test_training_moves_the_model(prod):
    """A run with enough steps leaves the init: g's weights change and val_nll is finite."""
    rd = _train(prod, "A", "real", runs="runs_long", g="T1", steps=30, eval_every=10, lr=1e-2)
    log = [r.split("\t") for r in (rd / "train_log.tsv").read_text().splitlines()[1:]]
    assert [int(r[0]) for r in log][:2] == [0, 10]
    assert all(np.isfinite(float(r[2])) for r in log)
    assert all(np.isfinite(float(r[1])) for r in log[1:])
    assert min(float(r[2]) for r in log[1:]) < float(log[0][2])


def test_bad_arguments(prod, tmp_path):
    with pytest.raises(ValueError):
        et.run_training(prod["manifest"], prod["cov"], prod["products"], tmp_path, "A", "all",
                        "ids", 0, max_steps=1, device="cpu", **FAST)
    with pytest.raises((KeyError, ValueError)):
        et.run_training(prod["manifest"], prod["cov"], prod["products"], tmp_path, "E", "all",
                        "real", 0, max_steps=1, device="cpu", **FAST)
    with pytest.raises(KeyError):
        et.run_training(prod["manifest"], prod["cov"], prod["products"], tmp_path, "A", "all",
                        "real", 0, max_steps=1, device="cpu", bogus=1, **FAST)


# -- the task index -------------------------------------------------------------------------------

def test_task_of_index_on_fixture_manifest():
    assert et.task_of_index(FIXTURE_MANIFEST, 18)["run_name"] == "A_C19M16_counts_real_s0"
    assert et.task_of_index(FIXTURE_MANIFEST, 402)["run_name"] == "Aold_C19M16_counts_real_s0"
    assert et.task_of_index(FIXTURE_MANIFEST, 431)["run_name"] == ep.tasks()[431]["run_name"]
    for bad in (-1, 432):
        with pytest.raises(IndexError):
            et.task_of_index(FIXTURE_MANIFEST, bad)


def _cli(*args):
    import os
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run([PY, str(TRAIN_PY), *map(str, args)], capture_output=True, text=True,
                          env=env, cwd=str(ROOT), timeout=300)


def test_cli_train_index_18_resolves_and_skips_on_done(tmp_path):
    """Dry: TRAIN_DONE is planted, so the CLI resolves the name and exits 0 without training."""
    rd = tmp_path / "runs" / "A_C19M16_counts_real_s0"
    rd.mkdir(parents=True)
    (rd / "TRAIN_DONE").write_text("planted\n")
    r = _cli("train-index", FIXTURE_MANIFEST, "/nonexistent.tsv", "/nonexistent", tmp_path / "runs",
             18)
    assert r.returncode == 0, r.stderr
    assert "A_C19M16_counts_real_s0: TRAIN_DONE present" in r.stdout
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["A_C19M16_counts_real_s0"]


def test_cli_train_index_out_of_range_fails(tmp_path):
    r = _cli("train-index", FIXTURE_MANIFEST, "/x.tsv", "/x", tmp_path / "runs", 432)
    assert r.returncode != 0 and "432" in r.stderr


def test_cli_train_runs_a_short_run(prod):
    r = _cli("train", prod["manifest"], prod["cov"], prod["products"], prod["root"] / "runs_cli",
             "Aold", "T2", "nocov", 1, "--max-steps", 2, "--eval-every", 1, "--device", "cpu")
    # FAST is not exposed on the CLI: the default 4 M-bin knot sample runs on the synthetic set
    assert r.returncode == 0, r.stderr
    rd = prod["root"] / "runs_cli" / "Aold_T2_counts_nocov_s1"
    assert (rd / "TRAIN_DONE").is_file()
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["max_steps"] == 2 and cfg["eval_every"] == 1 and cfg["device"] == "cpu"
