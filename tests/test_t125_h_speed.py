"""t125-P2h: the exact speed-up of h training on f's NB and of the chain scoring changes no number.

PI decision 2026-10-16: make fnb training and chain scoring fast enough to run with no change to
any number; tests prove bit-equality, or <= 1e-7 relative where only the float reduction order (or
a device's libm) changes. Every fast path is compared here with the reference path that stays in
the code (`fast=False`, the numpy CPU scorer), on the synthetic set-up of tests/test_t125_h_score.py:
- the distinct-row helpers and the formula on distinct (lambda, floor(pileup)) pairs: bit-equal;
- `fnb_flat_fast` on a CPU device: the same chunks, features, weights, counts, bit for bit;
- `fnb_batch_loss` / `validate_fnb` fast vs reference: the same loss and gradient, bit for bit;
- a short fnb training run (`run_training`, FAST vs not): the same checkpoint and val loss;
- the threaded f-NB draw (cache-mode corpus): the same batch;
- the chain (`score_context_fast` on CPU torch) vs the numpy reference: <= 1e-9 relative on every
  record of formula, learned push and learned own-noise h, also with capped rows and alone;
- a checkpoint written by the pre-change train.py (git edc7329) loads and chain-scores with the
  current code exactly as with the pre-change score.py (skipped without git or that commit);
- on a CUDA device (Nibi only): the fast paths against the reference on the same device.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools" / "t125", ROOT / "tools" / "t118", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import data as ldata  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from eta import model as em  # noqa: E402
from hmap import data as hd  # noqa: E402
from hmap import formula, products  # noqa: E402
from hmap import model as hm  # noqa: E402
from hmap import pushforward as pf  # noqa: E402
from hmap import score  # noqa: E402
from hmap import train as ht  # noqa: E402
from test_t125_h_score import S, _h_dir  # noqa: E402,F401  (the shared synthetic set-up)

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
BASE_SHA = "edc7329"
CUDA = torch.cuda.is_available()
REL = 1e-9


def _pred(S, data_dir=None):
    return em.load_run(S["f_dir"], data_dir or S["prod"], S["cov"], S["manifest"], device="cpu")


def _draw(S, batch=6, window=256, seed=3, corpus=None, pred=None, workers=1):
    corpus = corpus or hd.HCorpus(S["prod"], S["ctl"])
    pred = pred or _pred(S)
    tp = list(pred.config["train_pairs"])
    chroms, lens = ltrain.training_chroms(corpus.data, tp[0]["source_pid"], hd.SPACE_COUNTS)
    src = hd.FnbSource(corpus, S["infos"], pred, tp, chroms, lens, window, seed, workers=workers)
    return src.draw(batch)


def _close(a, b, rel=REL, abs_=1e-12):
    if a is None or b is None:
        return a is b
    if isinstance(a, bool) or isinstance(a, int) and isinstance(b, int):
        return a == b
    if not (math.isfinite(a) and math.isfinite(b)):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


# -- helpers --------------------------------------------------------------------------------------


def test_unique_pairs_round_trip_by_bit_pattern():
    rng = np.random.default_rng(0)
    a = rng.choice([0.5, 1.25, -0.0, 0.0, np.nan, 3.0], size=500)
    b = rng.choice([1.0, 2.0, 1e-3], size=500)
    au, bu, inv = pf.unique_pairs(a, b)
    assert au.size == bu.size <= 18
    np.testing.assert_array_equal(au[inv].view(np.int64), a.view(np.int64))
    np.testing.assert_array_equal(bu[inv].view(np.int64), b.view(np.int64))


def test_support_size_unique_equals_support_size():
    rng = np.random.default_rng(1)
    mu = rng.choice(np.exp(rng.normal(0, 2, 40)), size=3000)
    n = rng.choice(np.exp(rng.normal(0, 2, 40)), size=3000)
    for k_max in (pf.K_MAX, 20):
        np.testing.assert_array_equal(pf.support_size_unique(mu, n, k_max=k_max),
                                      pf.support_size(mu, n, k_max=k_max))


@pytest.mark.parametrize("variant", ["bin", "frag"])
def test_formula_on_distinct_pairs_is_p_enumerated(S, variant):
    """Bit for bit, including the far tail (scipy's logsf underflow branch)."""
    rng = np.random.default_rng(2)
    for pid in ("T1__base__base", "T2__base__base"):
        info = S["infos"][pid]
        r, K = 300, 40
        start = rng.choice([0, 0, 5, 300, 4000], size=r)
        k = start[:, None] + np.arange(K)[None, :]
        lam = rng.choice(np.exp(rng.normal(0.5, 1.5, 25)), size=r)
        rest = rng.choice(rng.gamma(2.0, 3.0, 30), size=r)
        want = np.log1p(formula.P_enumerated(variant, info, k, lam,
                                             rest if variant == "frag" else None))
        kt = torch.from_numpy(k.astype(np.float64))
        rt = torch.from_numpy(rest) if variant == "frag" else None
        cache: dict = {}
        for c in (None, cache, cache):             # no cache, a cold one, a warm one
            got = hd.formula_log1p_t(variant, info, kt, lam, rt, cache=c)
            np.testing.assert_array_equal(got.numpy(), want)
        flat = hd.formula_log1p_t(variant, info, kt.reshape(-1)[::7], lam, rt, cache=cache,
                                  row_e=torch.arange(r).repeat_interleave(K)[::7])
        np.testing.assert_array_equal(flat.numpy(), want.reshape(-1)[::7])
        assert (want > np.log1p(300.0)).any()      # P > 300: the series branch was exercised


def test_gauss_mixture_torch_twins():
    rng = np.random.default_rng(3)
    n, K = 60, 9
    mu = rng.normal(0, 1, (n, K))
    sg = np.exp(rng.normal(-1, 0.5, (n, K)))
    w = rng.random((n, K)) * (rng.random((n, K)) > 0.3)
    w[0] = 0.0
    y = rng.normal(0, 1, n)
    t = [torch.from_numpy(v) for v in (mu, sg, w, y)]
    np.testing.assert_allclose(pf.crps_gauss_mixture_t(*t, chunk=7, pair_elems=200).numpy(),
                               pf.crps_gauss_mixture(mu, sg, w, y, chunk=7), rtol=1e-12,
                               equal_nan=True)
    dup = torch.from_numpy(np.arange(n) % 7)        # rows 7 apart: same (mu, sigma, w) row
    td = [torch.from_numpy(v[np.arange(n) % 7]) for v in (mu, sg, w)] + [t[3]]
    np.testing.assert_allclose(pf.crps_gauss_mixture_t(*td, row_key=dup).numpy(),
                               pf.crps_gauss_mixture_t(*td).numpy(), rtol=1e-13, equal_nan=True)
    np.testing.assert_allclose(pf.mean_gauss_mixture_logp_t(*t[:3]).numpy(),
                               pf.mean_gauss_mixture_logp(mu, sg, w), rtol=1e-12, equal_nan=True)


# -- h training on f's NB ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fb(S):
    return _draw(S)


def _flat_blocks(fb, rung, with_formula, device, **kw):
    """(feat [r, K, F], w [r, K], y [r]) per chunk of `fnb_flat_fast`, and its counts."""
    out, cnt = [], {"n_capped": 0, "n_k_sum": 0, "n_k_max": 0}
    for fl in hd.fnb_flat_fast(fb, rung, with_formula, device, **kw):
        for e0, r0, r, K in fl.blocks:
            out.append((fl.feat[e0:e0 + r * K].view(r, K, -1).cpu().numpy(),
                        fl.w[e0:e0 + r * K].view(r, K).cpu().numpy(),
                        fl.y[r0:r0 + r].cpu().numpy()))
        cnt["n_capped"] += fl.n_capped
        cnt["n_k_sum"] += fl.n_k_sum
        cnt["n_k_max"] = max(cnt["n_k_max"], fl.n_k_max)
    return out, cnt


@pytest.mark.parametrize("rung,with_formula,k_max,elems", [
    (1, False, pf.K_MAX, 300), (5, False, pf.K_MAX, 1 << 18), (5, True, pf.K_MAX, 300),
    (5, True, 4, 50)])
def test_fast_chunks_equal_the_reference_bit_for_bit(fb, rung, with_formula, k_max, elems):
    ref = list(hd.fnb_chunks(fb, rung, with_formula, elems=elems, k_max=k_max))
    for flat in (1, 5000, 1 << 22):                # runs of 1 chunk, of a few, of all
        new, cnt = _flat_blocks(fb, rung, with_formula, "cpu", elems=elems, k_max=k_max,
                                flat_elems=flat)
        assert len(ref) == len(new) > 0
        assert cnt == {"n_capped": sum(c.n_capped for c in ref),
                       "n_k_sum": sum(int(c.n_k.sum()) for c in ref),
                       "n_k_max": max(int(c.n_k.max()) for c in ref)}
        if k_max == 4:
            assert cnt["n_capped"] > 0
        for a, (feat, w, y) in zip(ref, new):
            assert a.feat.dtype == feat.dtype == np.float32
            np.testing.assert_array_equal(feat.view(np.int32), a.feat.view(np.int32))
            np.testing.assert_array_equal(w, a.w.astype(np.float32))
            np.testing.assert_array_equal(y, a.y)


def _model(variant, rung, noise, seed=0):
    torch.manual_seed(seed)
    h = hm.HModel(variant, rung, noise, hm.N_IN[rung])
    with torch.no_grad():
        for p in h.parameters():
            p.normal_(0.0, 0.1)
    return h


def _std(fb, rung, formula_):
    fixed = fb.feat_fixed[:, :hd.N_FIXED_R1] if rung == 1 else fb.feat_fixed
    cols = [np.log1p(fb.mu)[:, None].astype(np.float32), fixed]
    if formula_:
        cols.append(np.zeros((fb.y.shape[0], 1), np.float32))
    return hm.Standardiser().fit(np.concatenate(cols, axis=1), hm.continuous_cols(rung))


@pytest.mark.parametrize("variant,rung,noise,rule", [
    ("affine", 1, False, "expected"), ("mlp", 5, False, "expected"),
    ("formula_mlp", 5, True, "expected"), ("formula_affine", 5, True, "mixture")])
def test_fnb_batch_loss_and_gradient_bit_equal(fb, variant, rung, noise, rule):
    formula_ = variant.startswith("formula")
    std = _std(fb, rung, formula_)
    out = {}
    for fast in (False, True):
        h = _model(variant, rung, noise)
        loss, st = ht.fnb_batch_loss(h, std, fb, rung, formula_, noise, rule, "cpu", elems=500,
                                     fast=fast)
        out[fast] = (loss, st, [p.grad.clone() for p in h.parameters()])
    assert out[True][0] == out[False][0] and math.isfinite(out[True][0])
    assert out[True][1] == out[False][1]
    for g1, g0 in zip(out[True][2], out[False][2]):
        assert torch.equal(g1, g0)


def test_validate_fnb_fast_bit_equal(S):
    pred = _pred(S)
    corpus = hd.HCorpus(S["prod"], S["ctl"])
    tp = list(pred.config["train_pairs"])[:4]
    f_val = ht.f_val_pass(pred, tp)
    fb = _draw(S, pred=pred)
    for variant, rung, noise in (("mlp", 5, True), ("formula_mlp", 5, False)):
        h = _model(variant, rung, noise)
        std = _std(fb, rung, variant.startswith("formula"))
        f_ = variant.startswith("formula")
        ref = ht.validate_fnb(h, std, corpus, S["infos"], pred, tp, rung, f_, "expected", "cpu",
                              val_chunk=200, elems=400, f_val=f_val)
        cache: dict = {}
        for _ in range(2):                         # the second pass reads the kept plans
            assert ht.validate_fnb(h, std, corpus, S["infos"], pred, tp, rung, f_, "expected",
                                   "cpu", val_chunk=200, elems=400, f_val=f_val, fast=True,
                                   cache=cache) == ref
        assert cache["plans"] and (cache["formula"] if f_ else True)


@pytest.mark.parametrize("variant,rung,noise,rule,flat", [
    ("affine", 1, False, "expected", 1 << 22), ("mlp", 5, True, "mixture", 700),
    ("formula_mlp", 5, True, "expected", 1)])
def test_flat_validation_loss_bit_equal(S, variant, rung, noise, rule, flat):
    """One product per batch, chunks laid out flat in runs of <= `flat` entries (1: one chunk)."""
    pred = _pred(S)
    corpus = hd.HCorpus(S["prod"], S["ctl"])
    p = next(q for q in pred.config["train_pairs"] if q["track"] == "T1")
    info = S["infos"][p["target_pid"]]
    loc, disp = pred.predict(p["source_pid"], p["source_pid"], p["target_pid"], "chr1")
    mu, nn = hd.nb_params(pred, loc, disp)
    n = mu.shape[0]
    fb = hd.concat_fnb([hd.fnb_bins(corpus, info, "chr1", 0, n, 0, n, mu, nn)], [info], [0])
    f_ = variant.startswith("formula")
    h, std = _model(variant, rung, noise), _std(fb, rung, f_)
    for k_max in (pf.K_MAX, 5):
        ref = ht.fnb_batch_loss(h, std, fb, rung, f_, noise, rule, "cpu", elems=300,
                                k_max=k_max, backward=False)
        new = ht.fnb_val_loss_flat(h, std, fb, rung, f_, noise, rule, "cpu", elems=300,
                                   k_max=k_max, flat_elems=flat, fcache={})
        assert new == ref


def _run(S, runs, fast, monkeypatch, **kw):
    monkeypatch.setattr(ht, "FAST", fast)
    return ht.run_training(S["manifest"], S["cov"], S["prod"], S["ctl"], runs, kw.pop("variant"),
                           kw.pop("rung"), kw.pop("noise"), "fnb", 0, fnb_run=S["f_dir"],
                           max_steps=4, eval_every=2, device="cpu", control_rl_tsv=S["rl"],
                           window=256, batch=4, **kw)


def _same_run_dirs(d0: Path, d1: Path):
    c0, c1 = (torch.load(d / "ckpt.pt", map_location="cpu", weights_only=True) for d in (d0, d1))
    assert set(c0) == set(c1) == {"h", "std", "config", "step", "val_loss"}
    assert c0["step"] == c1["step"] and c0["val_loss"] == c1["val_loss"]
    assert c0["std"] == c1["std"]
    assert set(c0["h"]) == set(c1["h"])
    for k in c0["h"]:
        assert torch.equal(c0["h"][k], c1["h"][k]), k
    j0, j1 = (json.loads((d / "config.json").read_text()) for d in (d0, d1))
    j0.pop("git_sha"), j1.pop("git_sha")
    assert j0 == j1
    assert c0["config"].keys() == c1["config"].keys()
    l0, l1 = ([ln.split("\t")[:3] for ln in (d / "train_log.tsv").read_text().splitlines()]
              for d in (d0, d1))
    assert l0 == l1


@pytest.mark.parametrize("variant,rung,noise", [("mlp", 5, 0), ("formula_affine", 5, 1)])
def test_short_fnb_training_run_is_unchanged(S, tmp_path, monkeypatch, variant, rung, noise):
    d0 = _run(S, tmp_path / "ref", False, monkeypatch, variant=variant, rung=rung, noise=noise)
    d1 = _run(S, tmp_path / "fast", True, monkeypatch, variant=variant, rung=rung, noise=noise)
    _same_run_dirs(d0, d1)
    t = json.loads((d1 / "timing.json").read_text())
    assert t["step_seconds"] >= t["draw_seconds"] >= 0.0


@pytest.fixture(scope="module")
def cache(S, tmp_path_factory):
    """A cache-mode corpus of the synthetic products (the threaded draws need one)."""
    d = tmp_path_factory.mktemp("cache")
    for r in S["rows"]:
        ldata.build_cache(S["prod"], d, r["pid"], r)
    return d


def test_threaded_draw_equals_the_serial_draw(S, cache):
    corpus = hd.HCorpus(cache, S["ctl"])
    assert corpus.data.is_cache
    pred = _pred(S, data_dir=cache)
    a = _draw(S, batch=12, corpus=corpus, pred=pred, workers=1)
    b = _draw(S, batch=12, corpus=corpus, pred=pred, workers=5)
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y)
        else:
            assert [i.pid for i in x] == [i.pid for i in y]


def test_training_with_threaded_draws_ahead_is_unchanged(S, cache, tmp_path, monkeypatch):
    """A cache-mode corpus: windows in threads, the next batch drawn during the step."""
    runs = {}
    for workers in (1, 4):
        monkeypatch.setattr(ht, "DRAW_WORKERS", workers)
        runs[workers] = ht.run_training(
            S["manifest"], S["cov"], cache, S["ctl"], tmp_path / str(workers), "formula_mlp", 5,
            1, "fnb", 0, fnb_run=S["f_dir"], max_steps=4, eval_every=2, device="cpu",
            control_rl_tsv=S["rl"], window=256, batch=4)
    _same_run_dirs(runs[1], runs[4])


# -- the chain --------------------------------------------------------------------------------------


def _few_pairs(monkeypatch, n=3):
    """Score only n training pairs of each track (T1 with a control, T2 without): faster tests."""
    real = em.load_run

    def load_run(*a, **k):
        pred = real(*a, **k)
        tp = list(pred.train_pairs)
        pred.train_pairs = ([q for q in tp if q["track"] == "T1"][:n]
                            + [q for q in tp if q["track"] != "T1"][:n])
        return pred
    monkeypatch.setattr(em, "load_run", load_run)


def _chain_records(S, specs, out, fast, monkeypatch):
    real = score.score_context

    def forced(hs, ctx, device=None, fast_=None):
        return real(hs, ctx, device="cpu", fast=fast)
    monkeypatch.setattr(score, "score_context", forced)
    fd = score.score_chain(S["manifest"], S["cov"], S["prod"], S["ctl"], S["bed"], S["f_dir"],
                           out, specs, workers=1, control_rl_tsv=S["rl"])
    monkeypatch.setattr(score, "score_context", real)
    return {p.name: json.loads((p / "scores.json").read_text())["records"]
            for p in Path(fd).iterdir() if p.is_dir()}


def _assert_records_close(ref: dict, new: dict, rel=REL):
    assert ref.keys() == new.keys()
    for name in ref:
        assert len(ref[name]) == len(new[name]) > 0
        for a, b in zip(ref[name], new[name]):
            assert a.keys() == b.keys()
            for k in a:
                if isinstance(a[k], float) or isinstance(b[k], float):
                    assert _close(a[k], b[k], rel=rel), (name, k, a[k], b[k])
                else:
                    assert a[k] == b[k], (name, k)


def test_chain_fast_equals_reference(S, tmp_path, monkeypatch):
    """formula bin / frag, a learned push h (rung 1 and 5), own-noise h (rung 5, and rung 1 whose
    pair term is computed once per distinct row) and an fnb-trained h."""
    _few_pairs(monkeypatch)
    specs = S["specs"] + [str(_h_dir(tmp_path / "h", "h_mlp_r1_noise_true_s0", "mlp", 1, True, 3))]
    ref = _chain_records(S, specs, tmp_path / "ref", False, monkeypatch)
    new = _chain_records(S, specs, tmp_path / "fast", True, monkeypatch)
    assert "h_mlp_r1_noise_true_s0" in new
    _assert_records_close(ref, new)


def test_chain_fast_equals_reference_with_capped_rows(S, tmp_path, monkeypatch):
    """Also with many small chunks (the formula values kept across chunks, rung-1 rows shared).
    The two paths now cut different chunks, so the float32 h forward runs on other batch shapes
    (a BLAS result may change in the last bit): 1e-7, the PI's reduction-order tolerance."""
    _few_pairs(monkeypatch)
    monkeypatch.setattr(score, "K_MAX", 3)
    monkeypatch.setattr(score, "CHUNK_BINS", 300)
    monkeypatch.setattr(score, "FAST_ELEM_BUDGET", 700)
    specs = ["formula:frag", "formula:bin", str(S["h_mlp"]), str(S["h_noise"]),
             str(_h_dir(tmp_path / "h", "h_affine_r1_noise_true_s1", "affine", 1, True, 4))]
    ref = _chain_records(S, specs, tmp_path / "ref", False, monkeypatch)
    new = _chain_records(S, specs, tmp_path / "fast", True, monkeypatch)
    assert sum(r["n_capped"] for r in ref["h_formula_frag"]) > 0
    _assert_records_close(ref, new, rel=1e-7)


def test_alone_fast_equals_reference(S, tmp_path, monkeypatch):
    real = score.score_context
    out = {}
    for fast in (False, True):
        monkeypatch.setattr(score, "score_context",
                            lambda hs, ctx, device=None, fast_=None, f=fast: real(
                                hs, ctx, device="cpu", fast=f))
        recs = {}
        for spec in (S["specs"][1], str(S["h_noise"])):
            p = score.score_alone(S["manifest"], S["cov"], S["prod"], S["ctl"], S["bed"], spec,
                                  tmp_path / str(fast), workers=1, control_rl_tsv=S["rl"])
            recs[spec] = json.loads(Path(p).read_text())["records"]
        out[fast] = recs
    monkeypatch.setattr(score, "score_context", real)
    _assert_records_close(out[False], out[True])


# -- a checkpoint of the pre-change train.py -----------------------------------------------------------


def _pre_change_module(path_in_repo: str, name: str, tmp_path: Path):
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE_SHA}:{path_in_repo}"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {BASE_SHA} not available (a kit without .git)")
    f = tmp_path / f"{name}.py"
    # the module inserts tools/t118 and tools/t125 by its own path: keep it pointing at the repo
    f.write_text(src.replace("Path(__file__).resolve().parents[2]",
                             f"Path({str(ROOT / 'tools')!r})"))
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pre_change_checkpoint_loads_and_chain_scores_identically(S, tmp_path, monkeypatch):
    train_pre = _pre_change_module("tools/t125/hmap/train.py", "train_pre", tmp_path)
    score_pre = _pre_change_module("tools/t125/hmap/score.py", "score_pre", tmp_path)
    kw = dict(max_steps=4, eval_every=2, device="cpu", control_rl_tsv=S["rl"], window=256, batch=4)
    runs_pre, runs_new = tmp_path / "runs_pre", tmp_path / "runs_new"
    old = [train_pre.run_training(S["manifest"], S["cov"], S["prod"], S["ctl"], runs_pre,
                                  "formula_mlp", 5, 1, "true", 0, **kw),
           train_pre.run_training(S["manifest"], S["cov"], S["prod"], S["ctl"], runs_pre,
                                  "mlp", 5, 0, "fnb", 0, fnb_run=S["f_dir"], **kw)]
    monkeypatch.setattr(ht, "FAST", True)
    new = [ht.run_training(S["manifest"], S["cov"], S["prod"], S["ctl"], runs_new,
                           "formula_mlp", 5, 1, "true", 0, **kw),
           ht.run_training(S["manifest"], S["cov"], S["prod"], S["ctl"], runs_new,
                           "mlp", 5, 0, "fnb", 0, fnb_run=S["f_dir"], **kw)]
    for d0, d1 in zip(old, new):                   # the format and the numbers are unchanged
        _same_run_dirs(d0, d1)
    specs = ["formula:frag"] + [str(d) for d in old]
    _few_pairs(monkeypatch)
    fd_pre = score_pre.score_chain(S["manifest"], S["cov"], S["prod"], S["ctl"], S["bed"],
                                   S["f_dir"], tmp_path / "chain_pre", specs, workers=1,
                                   control_rl_tsv=S["rl"])
    pre = {p.name: json.loads((p / "scores.json").read_text())["records"]
           for p in Path(fd_pre).iterdir() if p.is_dir()}
    ref = _chain_records(S, specs, tmp_path / "chain_ref", False, monkeypatch)
    assert ref == pre                              # the CPU path: bit for bit
    new_fast = _chain_records(S, specs, tmp_path / "chain_fast", True, monkeypatch)
    _assert_records_close(pre, new_fast)


# -- CUDA (Nibi) -----------------------------------------------------------------------------------


@pytest.mark.skipif(not CUDA, reason="no CUDA device")
def test_cuda_fast_paths_against_the_reference_on_the_same_device(S, fb, tmp_path, monkeypatch):
    """GPU: the fast chunk features equal the CPU reference except log1p of the fragment-window
    mean (counted, at most 1 float32 ulp); the loss on the GPU equals the reference loss on the GPU
    within 1e-6; the chain records within 1e-7."""
    ref = list(hd.fnb_chunks(fb, 5, True, elems=1 << 18))
    new, _ = _flat_blocks(fb, 5, True, "cuda", elems=1 << 18)
    n_diff = 0
    for a, (bf, bw, by) in zip(ref, new):
        np.testing.assert_array_equal(bw, a.w.astype(np.float32))
        cols = [c for c in range(bf.shape[-1]) if c != hd.K_FRAG_COL]
        np.testing.assert_array_equal(bf[..., cols], a.feat[..., cols])
        d = np.abs(bf[..., hd.K_FRAG_COL].view(np.int32).astype(np.int64)
                   - a.feat[..., hd.K_FRAG_COL].view(np.int32))
        assert d.max() <= 1
        n_diff += int((d > 0).sum())
    print(f"log1p_k_frag float32 entries differing by 1 ulp: {n_diff}")
    std = _std(fb, 5, True)
    for noise in (False, True):
        losses = []
        for fast in (False, True):
            h = _model("formula_mlp", 5, noise).cuda()
            losses.append(ht.fnb_batch_loss(h, std, fb, 5, True, noise, "expected", "cuda",
                                            fast=fast)[0])
        assert losses[1] == pytest.approx(losses[0], rel=1e-6)
    real = score.score_context
    out = {}
    for fast in (False, True):
        monkeypatch.setattr(score, "score_context",
                            lambda hs, ctx, device=None, fast_=None, f=fast: real(
                                hs, ctx, device="cuda", fast=f))
        hs = [score.load_h(s, "cuda") for s in S["specs"]]
        score._setup(S["manifest"], S["cov"], S["prod"], S["ctl"], S["bed"], hs, S["rl"],
                     pred=_pred(S), f_run_name=S["f_dir"].name, device="cuda")
        out[fast] = {}
        for pair in list(score._CTX["pred"].train_pairs)[:4]:
            for name, rec in score.chain_job(pair).items():
                out[fast].setdefault(name, []).append(rec)
    monkeypatch.setattr(score, "score_context", real)
    _assert_records_close(out[False], out[True], rel=1e-7)
