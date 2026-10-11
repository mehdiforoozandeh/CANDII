"""t125 part 2 — h's two training sources (`hmap/data.py`) and the training loop (`hmap/train.py`).

Written from plan/ETA_OFFSET_AND_H_DESIGN.md "Part 2", plan/T125_STATUS.md "PI decisions" and the
plan's pinned interfaces (chunk t125-P2e):
- the true source draws windows (pid uniform, chromosome proportional to length, start uniform,
  `default_rng(seed)`) and gives P2a's features of the TRUE counts plus log1p of P2b's
  fragment-window formula; the expectations below recompute them on the WHOLE chromosome, so the
  halo the source cuts around each window is checked to be wide enough;
- the f-NB source draws pairs uniform and gives f's NB on the target, the no-control λ window and
  the fragment window built on f's predicted mean (P2a's chain rule);
- the f-NB losses use the exact NB support of P2c, chunked by support size; the chunked loss must
  equal the dense one, and capped rows (support > K_MAX) are counted and written to the run;
- training: the t118 g settings (Adam 1e-3, warmup 100, clip 1, 4000 steps, eval every 250 on
  chr22, patience 4), standardiser fitted once on 64 drawn windows.
Runs on CPU on `ladder.synth.make_products` with a Poisson control written for track T1 here.
"""
from __future__ import annotations

import csv
import hashlib
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

from ladder import data as ldata  # noqa: E402
from ladder import model as lmodel  # noqa: E402
from ladder import pairs as lpairs  # noqa: E402
from ladder import synth  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from eta import model as em  # noqa: E402
from eta import pairs as ep  # noqa: E402
from eta import train as et  # noqa: E402
from hmap import controls, features, formula, products, pushforward, tasks  # noqa: E402
from hmap import data as hd  # noqa: E402
from hmap import model as hm  # noqa: E402
from hmap import train as ht  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
PY = sys.executable
TRAIN_PY = ROOT / "tools" / "t125" / "hmap" / "train.py"
N_BINS = {"chr1": 2500, "chr2": 1800, "chr19": 600, "chr21": 500, "chr22": 700, "chrX": 500}
F_FAST = {"window": 256, "batch": 8, "knot_sample": 20_000, "knot_block": 256}
H_FAST = {"window": 256, "batch": 4}
F_RUN = "A_all_counts_real_s0"


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Synthetic products, a Poisson control for every T1 product, the control cache, a 3-step f."""
    root = tmp_path_factory.mktemp("synth_h_train")
    prod = root / "products"
    manifest, cov = synth.make_products(prod, 0, n_bins=N_BINS)
    with open(manifest, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    rng = np.random.default_rng(21)
    ctl = {}
    for r in rows:
        if r["track"] != "T1":
            continue
        arrays = {c: rng.poisson(1.5, n).astype(np.uint32) for c, n in N_BINS.items()}
        path = prod / r["pid"] / "control_counts25.npz"
        np.savez_compressed(path, **arrays)
        r["control_counts25_md5"] = _md5(path)
        ctl[r["pid"]] = arrays
    rl_tsv = root / "control_read_length.tsv"
    with open(rl_tsv, "w") as fh:
        fh.write("pid\tcontrol_tagalign\tcontrol_read_length_mean\tn_lines_read\n")
        for pid in sorted(ctl):
            fh.write(f"{pid}\tsynthetic\t101.0\t100000\n")
    with open(manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    ctl_cache = root / "ctl_cache"
    for r in rows:
        controls.build_control_cache(prod, ctl_cache, r["pid"], r)
    f_run = et.run_training(manifest, cov, prod, root / "eta_runs", "A", "all", "real", 0,
                            max_steps=3, eval_every=3, device="cpu", **F_FAST)
    infos = products.load(manifest, cov, rl_tsv)
    return {"root": root, "products": prod, "manifest": manifest, "cov": cov, "rl_tsv": rl_tsv,
            "ctl_cache": ctl_cache, "ctl": ctl, "rows": lpairs.read_manifest(manifest),
            "infos": infos, "f_run": f_run}


@pytest.fixture(scope="module")
def pred(env):
    return em.load_run(env["f_run"], env["products"], env["cov"], env["manifest"], device="cpu")


def _corpus(env):
    return hd.HCorpus(env["products"], env["ctl_cache"])


def _train(env, variant, rung, noise, source, runs="runs", seed=0, steps=5, eval_every=5, **kw):
    fnb_run = env["f_run"] if source == "fnb" else None
    return ht.run_training(env["manifest"], env["cov"], env["products"], env["ctl_cache"],
                           env["root"] / runs, variant, rung, noise, source, seed,
                           fnb_run=fnb_run, max_steps=steps, eval_every=eval_every, device="cpu",
                           control_rl_tsv=env["rl_tsv"], **{**H_FAST, **kw})


def _nb(pred_, loc, disp):
    mu = pred_.mean(loc)
    n = np.exp(np.clip(np.asarray(disp, np.float64), math.log(lmodel.N_MIN),
                       math.log(lmodel.N_MAX)))
    return mu, n


# -- constants -----------------------------------------------------------------------------------

def test_defaults_are_the_t118_g_settings():
    assert ht.DEFAULTS == ltrain.DEFAULTS and ht.DEFAULTS is not ltrain.DEFAULTS
    for k, v in {"window": 2048, "batch": 32, "lr": 1e-3, "warmup": 100, "grad_clip": 1.0,
                 "max_steps": 4000, "eval_every": 250, "patience": 4}.items():
        assert ht.DEFAULTS[k] == v
    assert ht.VAL_CHUNK == 65536
    assert ht.N_STD_WINDOWS == 64
    assert hd.HALO >= features.W_10K // 2 + 1        # the widest window (10 kb) fits the halo


# -- HCorpus -------------------------------------------------------------------------------------

def test_hcorpus_reads_counts_pval_and_control(env):
    hc = _corpus(env)
    raw = ldata.Corpus(env["products"])
    pid = "T1__depth__15M"
    for c in N_BINS:
        np.testing.assert_array_equal(hc.counts(pid, c), raw.get(pid, "counts", c))
        np.testing.assert_array_equal(hc.pval(pid, c), raw.get(pid, "pval", c))
        np.testing.assert_array_equal(hc.control(pid, c), env["ctl"][pid][c])
        assert hc.n_bins(pid, c) == N_BINS[c]
    assert hc.control("T2__base__base", "chr1") is None
    assert set(hc.chroms(pid)) == set(N_BINS)


# -- the true source -----------------------------------------------------------------------------

def _windows(seed, n_items, chroms, lens, W, batch):
    rng = np.random.default_rng(seed)
    pi = rng.integers(n_items, size=batch)
    ci = rng.choice(len(chroms), size=batch, p=lens / lens.sum())
    u = rng.random(batch)
    out = []
    for b in range(batch):
        n = int(lens[ci[b]])
        out.append((int(pi[b]), chroms[ci[b]], int(u[b] * (max(0, n - W) + 1)), n))
    return out


def test_true_source_equals_whole_chromosome_features(env):
    hc = _corpus(env)
    pids = [r["pid"] for r in lpairs.usable_products(env["rows"])]
    chroms, lens = ltrain.training_chroms(ldata.Corpus(env["products"]), pids[0], "counts")
    W, B, seed = 256, 6, 3
    src = hd.TrueSource(hc, env["infos"], pids, chroms, lens, W, seed)
    tb = src.draw(B)
    assert tb.feat_r1.shape == (B * W, 9) and tb.feat_r5.shape == (B * W, 21)
    assert tb.formula_col.shape == (B * W,) and tb.y.shape == (B * W,) and tb.mask.shape == (B * W,)
    np.testing.assert_array_equal(tb.feat_r1, tb.feat_r5[:, :9])
    for b, (p, chrom, s, n) in enumerate(_windows(seed, len(pids), chroms, lens, W, B)):
        info = env["infos"][pids[p]]
        ct = ldata.Corpus(env["products"]).get(info.pid, "counts", chrom)
        cc = env["ctl"][info.pid][chrom] if info.has_control else None
        fixed = features.fixed_features(info, ct, cc, 5)
        want = np.concatenate([np.log1p(ct.astype(np.float64))[:, None], fixed], axis=1)
        sl = slice(b * W, (b + 1) * W)
        np.testing.assert_allclose(tb.feat_r5[sl], want[s:s + W], rtol=1e-6, atol=1e-6)
        fcol = np.log1p(formula.P("frag", info, ct, cc))
        np.testing.assert_allclose(tb.formula_col[sl], fcol[s:s + W], rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(tb.y[sl], ldata.Corpus(env["products"]).get(
            info.pid, "pval", chrom)[s:s + W])
        assert tb.mask[sl].all()


def test_true_source_masks_bins_past_the_chromosome_end(env):
    hc = _corpus(env)
    lens = np.array([N_BINS["chrX"]], dtype=np.int64)
    src = hd.TrueSource(hc, env["infos"], ["T1__base__base"], ["chrX"], lens, 600, 0)
    tb = src.draw(2)
    m = tb.mask.reshape(2, 600)
    assert m[:, :500].all() and not m[:, 500:].any()
    assert np.all(tb.y.reshape(2, 600)[:, 500:] == 0)
    assert np.all(np.isfinite(tb.feat_r5)) and np.all(np.isfinite(tb.formula_col))


def test_true_source_without_formula(env):
    hc = _corpus(env)
    lens = np.array([N_BINS["chr1"]], dtype=np.int64)
    src = hd.TrueSource(hc, env["infos"], ["T2__base__base"], ["chr1"], lens, 128, 0,
                        with_formula=False)
    assert src.draw(2).formula_col is None


# -- the f-NB source -----------------------------------------------------------------------------

def _fnb_source(env, pred_, W=256, seed=5):
    hc = _corpus(env)
    tp = pred_.config["train_pairs"]
    chroms, lens = ltrain.training_chroms(ldata.Corpus(env["products"]), tp[0]["source_pid"],
                                          "counts")
    return hd.FnbSource(hc, env["infos"], pred_, tp, chroms, lens, W, seed), tp, chroms, lens


def test_fnb_source_equals_whole_chromosome_prediction(env, pred):
    W, B, seed = 256, 4, 5
    src, tp, chroms, lens = _fnb_source(env, pred, W, seed)
    fb = src.draw(B)
    raw = ldata.Corpus(env["products"])
    off = 0
    for p, chrom, s, n in _windows(seed, len(tp), chroms, lens, W, B):
        pair = tp[p]
        info = env["infos"][pair["target_pid"]]
        loc, disp = pred.predict(pair["source_pid"], pair["source_pid"], pair["target_pid"], chrom)
        mu, nn = _nb(pred, loc, disp)
        cc = env["ctl"][info.pid][chrom] if info.has_control else None
        fixed = features.fixed_features(info, mu, cc, 5, treat_mean=mu)
        wf = features.frag_bins(info.fraglen)
        rest = features.window_sum(mu, wf) - mu
        cnt = features.window_count(mu.shape[0], wf)
        lam = formula.lambda_track(info, cc, mu)
        m = min(W, n - s)
        sl = slice(off, off + m)
        np.testing.assert_allclose(fb.mu[sl], mu[s:s + m], rtol=1e-5)
        np.testing.assert_allclose(fb.n_size[sl], nn[s:s + m], rtol=1e-5)
        np.testing.assert_allclose(fb.feat_fixed[sl], fixed[s:s + m], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(fb.frag_rest[sl], rest[s:s + m], rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(fb.frag_cnt[sl], cnt[s:s + m])
        np.testing.assert_allclose(fb.lam[sl], lam[s:s + m], rtol=1e-5)
        np.testing.assert_array_equal(fb.y[sl], raw.get(info.pid, "pval", chrom)[s:s + m])
        assert all(fb.infos[i].pid == info.pid for i in fb.info_idx[sl])
        off += m
    assert off == fb.y.shape[0]


def _dense_feat(fb, idx, k, rung, with_formula):
    """The chain's per-k features, by hand: log1p k, the fixed columns, k_frag per k, formula."""
    r, K = k.shape
    fixed = fb.feat_fixed[idx][:, :8] if rung == 1 else fb.feat_fixed[idx]
    cols = [np.log1p(k.astype(np.float64))[:, :, None],
            np.broadcast_to(fixed[:, None, :].astype(np.float64), (r, K, fixed.shape[1]))]
    feat = np.concatenate(cols, axis=2)
    if rung == 5:
        j = features.FEATURES_R5.index("log1p_k_frag")
        feat[:, :, j] = np.log1p((k + fb.frag_rest[idx][:, None]) / fb.frag_cnt[idx][:, None])
    if with_formula:
        fcol = np.empty((r, K))
        for row in range(r):
            info = fb.infos[fb.info_idx[idx[row]]]
            fcol[row] = formula.P_enumerated("frag", info, k[row:row + 1],
                                             fb.lam[idx][row:row + 1],
                                             fb.frag_rest[idx][row:row + 1])[0]
        feat = np.concatenate([feat, np.log1p(fcol)[:, :, None]], axis=2)
    return feat.astype(np.float32)


@pytest.mark.parametrize("variant,rung,noise,rule", [
    ("formula_mlp", 5, 0, "expected"), ("formula_affine", 5, 1, "mixture"),
    ("mlp", 1, 1, "expected"), ("affine", 5, 0, "expected")])
def test_chunked_fnb_loss_equals_the_dense_loss(env, pred, variant, rung, noise, rule):
    src, *_ = _fnb_source(env, pred, W=128, seed=9)
    fb = src.draw(3)
    torch.manual_seed(1)
    form = variant.startswith("formula")
    h = hm.HModel(variant, rung, bool(noise), hm.N_IN[rung])
    with torch.no_grad():                       # move off the zero init so the test bites
        for p in h.parameters():
            p.add_(0.05 * torch.randn_like(p))
    n_feat = h.n_feat
    std = hm.Standardiser().fit(np.random.default_rng(0).normal(size=(50, n_feat)),
                                hm.continuous_cols(rung))
    N = fb.y.shape[0]
    idx = np.arange(N)
    k, w, valid = pushforward.nb_support(fb.mu, fb.n_size)
    x = std.apply(torch.from_numpy(_dense_feat(fb, idx, k, rung, form)))
    mu, ls = h(x)
    wt = torch.from_numpy(w).float()
    yt = torch.from_numpy(fb.y)
    want = (hm.loss_noise_fnb(mu, ls, wt, yt, rule) if noise else hm.loss_push_fnb(mu, wt, yt))
    for elems in (64, 4096, 1 << 22):
        got, st = ht.fnb_batch_loss(h, std, fb, rung, form, bool(noise), rule, torch.device("cpu"),
                                    elems=elems, backward=False)
        assert math.isclose(got, float(want), rel_tol=1e-5, abs_tol=1e-6), (elems, got, float(want))
        assert st["n_bins"] == N and st["n_capped"] == 0
        np.testing.assert_allclose(st["n_k_sum"], pushforward.support_size(fb.mu, fb.n_size).sum())


def test_chunks_bound_memory_and_count_capped_rows(env, pred):
    src, *_ = _fnb_source(env, pred, W=128, seed=2)
    fb = src.draw(4)
    k_max = 8
    size = pushforward.support_size(fb.mu, fb.n_size, k_max=k_max)
    assert (size > k_max).sum() > 0               # the synthetic peaks need more than 8 points
    seen, capped = [], 0
    for ch in hd.fnb_chunks(fb, 5, True, elems=200, k_max=k_max):
        r, K = ch.w.shape
        assert ch.feat.shape == (r, K, 22)
        assert K <= k_max + 1
        assert r == 1 or r * K <= 200
        np.testing.assert_allclose(ch.w.sum(axis=1), 1.0, rtol=1e-6)
        capped += ch.n_capped
        seen.append(ch.idx)
    allidx = np.concatenate(seen)
    assert np.array_equal(np.sort(allidx), np.arange(fb.y.shape[0]))
    assert capped == int((size > k_max).sum())


# -- training ------------------------------------------------------------------------------------

CONFIG_KEYS = {"h_name", "variant", "rung", "noise", "source", "seed", "fnb_run", "fnb_noise_loss",
               "features", "DEFAULTS", "git_sha", "manifest_md5", "covariates_md5", "n_in"}
TIMING_KEYS = {"setup", "train", "total", "peak_rss_mb", "steps_run", "best_step", "val_loss"}


@pytest.mark.parametrize("variant,rung,noise,source,rule", [
    ("affine", 1, 0, "true", None), ("mlp", 5, 1, "true", None),
    ("formula_mlp", 5, 0, "fnb", None), ("formula_affine", 5, 1, "fnb", "mixture")])
def test_train_writes_the_run_dir(env, variant, rung, noise, source, rule):
    kw = {"fnb_noise_loss": rule} if rule else {}
    run_dir = _train(env, variant, rung, noise, source, **kw)
    tag = "true" if source == "true" else F_RUN
    name = tasks.h_name(variant, rung, noise, tag, 0)
    assert run_dir.name == name
    for f in ("config.json", "ckpt.pt", "train_log.tsv", "timing.json", "TRAIN_DONE"):
        assert (run_dir / f).is_file(), f
    cfg = json.loads((run_dir / "config.json").read_text())
    assert CONFIG_KEYS <= set(cfg)
    assert cfg["h_name"] == name and cfg["variant"] == variant and cfg["rung"] == rung
    assert cfg["noise"] == noise and cfg["seed"] == 0
    assert cfg["DEFAULTS"]["lr"] == 1e-3 and cfg["DEFAULTS"]["patience"] == 4
    feats = list(features.FEATURES[rung]) + ([features.FEATURE_FORMULA]
                                            if variant.startswith("formula") else [])
    assert cfg["features"] == feats
    if source == "true":
        assert cfg["source"] == "true" and cfg["fnb_run"] is None
        assert cfg["pids"] == [r["pid"] for r in lpairs.usable_products(env["rows"])]
    else:
        assert cfg["source"] == f"fnb:{F_RUN}"
        assert Path(cfg["fnb_run"]) == Path(env["f_run"])
        assert cfg["fnb_noise_loss"] == (rule or "expected")
        f_cfg = json.loads((Path(env["f_run"]) / "config.json").read_text())
        assert cfg["train_pairs"] == f_cfg["train_pairs"]
    timing = json.loads((run_dir / "timing.json").read_text())
    assert TIMING_KEYS <= set(timing)
    assert math.isfinite(timing["val_loss"])
    if source == "fnb":
        assert timing["n_k_mean"] >= 1
        assert isinstance(timing["n_capped"], int) and timing["n_capped"] >= 0
        assert timing["n_capped"] == timing["n_capped_train"] + timing["n_capped_val"]
    log = (run_dir / "train_log.tsv").read_text().splitlines()
    assert log[0].split("\t") == ["step", "loss", "val_loss", "seconds"]
    assert all(math.isfinite(float(l.split("\t")[2])) for l in log[1:])

    # the ckpt reloads and h runs on chr22 features
    h, std, ck_cfg, step, val = hm.load_ckpt(run_dir / "ckpt.pt")
    assert (h.variant, h.rung, h.noise) == (variant, rung, bool(noise))
    assert step == timing["best_step"] and math.isclose(val, timing["val_loss"])
    info = env["infos"]["T1__base__base"]
    ct = ldata.Corpus(env["products"]).get(info.pid, "counts", "chr22")
    cc = env["ctl"][info.pid]["chr22"]
    fx = features.fixed_features(info, ct, cc, rung)
    feat = np.concatenate([features.k_column(ct)[:, None], fx], axis=1)
    if variant.startswith("formula"):
        feat = np.concatenate([feat, np.log1p(formula.P("frag", info, ct, cc))[:, None]], axis=1)
    with torch.no_grad():
        mu, ls = h(std.apply(torch.from_numpy(feat.astype(np.float32))))
    assert mu.shape == (N_BINS["chr22"],) and torch.isfinite(mu).all()
    assert (ls is not None) == bool(noise)


def test_affine_loss_decreases_over_50_steps(env):
    run_dir = _train(env, "affine", 1, 0, "true", runs="runs50", steps=50, eval_every=10,
                     batch=8)
    rows = [l.split("\t") for l in (run_dir / "train_log.tsv").read_text().splitlines()[1:]]
    val = [float(r[2]) for r in rows]
    loss = [float(r[1]) for r in rows[1:]]
    assert len(val) == 6 and all(math.isfinite(v) for v in val + loss)
    assert val[-1] < val[0]
    assert loss[-1] < loss[0]
    timing = json.loads((run_dir / "timing.json").read_text())
    assert timing["steps_run"] == 50 and timing["best_step"] > 0


def _ckpt_tensors(run_dir):
    return torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=True)


@pytest.mark.parametrize("variant,rung,noise,source", [("mlp", 1, 1, "true"),
                                                       ("formula_affine", 5, 0, "fnb")])
def test_same_seed_is_bit_identical(env, variant, rung, noise, source):
    a = _train(env, variant, rung, noise, source, runs="det_a", steps=3, eval_every=3)
    b = _train(env, variant, rung, noise, source, runs="det_b", steps=3, eval_every=3)
    ca, cb = _ckpt_tensors(a), _ckpt_tensors(b)
    assert ca["h"].keys() == cb["h"].keys()
    for key in ca["h"]:
        assert torch.equal(ca["h"][key], cb["h"][key]), key
    assert ca["std"] == cb["std"] and ca["val_loss"] == cb["val_loss"]


def test_train_done_skips(env):
    run_dir = _train(env, "affine", 5, 1, "true", runs="skip", steps=2, eval_every=2)
    before = (run_dir / "ckpt.pt").stat().st_mtime_ns
    again = _train(env, "affine", 5, 1, "true", runs="skip", steps=2, eval_every=2)
    assert again == run_dir and (run_dir / "ckpt.pt").stat().st_mtime_ns == before


def test_validation_is_independent_of_the_chunking(env, pred):
    hc = _corpus(env)
    torch.manual_seed(0)
    h = hm.HModel("formula_mlp", 5, True, 21)
    with torch.no_grad():
        for p in h.parameters():
            p.add_(0.05 * torch.randn_like(p))
    std = hm.Standardiser().fit(np.random.default_rng(1).normal(size=(40, 22)),
                                hm.continuous_cols(5))
    dev = torch.device("cpu")
    pids = ["T1__base__base", "T2__pe__pe"]
    a = ht.validate_true(h, std, hc, env["infos"], pids, 5, True, dev, val_chunk=65536)
    b = ht.validate_true(h, std, hc, env["infos"], pids, 5, True, dev, val_chunk=123)
    assert math.isclose(a, b, rel_tol=1e-6)
    tp = pred.config["train_pairs"][:3]
    fa, sa = ht.validate_fnb(h, std, hc, env["infos"], pred, tp, 5, True, "expected", dev,
                             val_chunk=65536, elems=1 << 22)
    fb_, sb = ht.validate_fnb(h, std, hc, env["infos"], pred, tp, 5, True, "expected", dev,
                              val_chunk=150, elems=300)
    assert math.isclose(fa, fb_, rel_tol=1e-5)
    assert sa["n_capped"] == sb["n_capped"] == 0 and sa["n_bins"] == 3 * N_BINS["chr22"]


def test_rejects_bad_arguments(env):
    with pytest.raises(ValueError):
        _train(env, "formula_affine", 1, 0, "true", runs="bad")
    with pytest.raises(ValueError):
        _train(env, "affine", 3, 0, "true", runs="bad")
    with pytest.raises(ValueError):
        ht.run_training(env["manifest"], env["cov"], env["products"], env["ctl_cache"],
                        env["root"] / "bad", "affine", 1, 0, "fnb", 0, device="cpu",
                        control_rl_tsv=env["rl_tsv"])
    with pytest.raises(ValueError):
        _train(env, "affine", 1, 1, "fnb", runs="bad", fnb_noise_loss="median")
    with pytest.raises(ValueError):
        _train(env, "affine", 1, 0, "pseudo", runs="bad")


# -- task tables ---------------------------------------------------------------------------------

def test_task_of_index_resolves_the_pinned_names(tmp_path):
    t = ht.task_of_index("true", 0)
    assert t["h_name"] == "h_affine_r1_push_true_s0" and t["source"] == "true"
    assert t["fnb_run"] is None
    t = ht.task_of_index("true", 35)
    assert t["h_name"] == "h_formula_mlp_r5_noise_true_s2"
    assert (t["variant"], t["rung"], t["noise"], t["seed"]) == ("formula_mlp", 5, 1, 2)
    t = ht.task_of_index("fnb", 0, eta_runs=tmp_path)
    assert t["h_name"] == "h_affine_r1_push_A_C07M20_counts_real_s0_s0"
    assert t["source"] == "fnb" and Path(t["fnb_run"]) == tmp_path / "A_C07M20_counts_real_s0"
    first_f = ep.f_runs_seed0()[0]
    assert first_f == "A_C07M20_counts_real_s0"
    t = ht.task_of_index("fnb", 767, eta_runs=tmp_path)
    assert t["h_name"] == tasks.tasks_fnb(ep.f_runs_seed0())[767]["h_name"]
    t = ht.task_of_index("fnb", 1, eta_runs=tmp_path, fnb_seeds=3)
    assert t["seed"] == 1
    with pytest.raises(ValueError, match="eta-runs"):
        ht.task_of_index("fnb", 0)
    for table, i in (("true", 36), ("fnb", 768), ("true", -1)):
        with pytest.raises(IndexError):
            ht.task_of_index(table, i, eta_runs=tmp_path)
    with pytest.raises(ValueError):
        ht.task_of_index("alone", 0)


# -- CLI -----------------------------------------------------------------------------------------

def _env_vars():
    e = dict(os.environ)
    e["PYTHONPATH"] = str(ROOT / "src")
    return e


def test_cli_train_and_train_index(env):
    runs = env["root"] / "cli_runs"
    base = [PY, str(TRAIN_PY)]
    common = [str(env["manifest"]), str(env["cov"]), str(env["products"]), str(env["ctl_cache"]),
              str(runs)]
    knobs = ["--max-steps", "2", "--eval-every", "2", "--device", "cpu",
             "--control-rl-tsv", str(env["rl_tsv"])]
    r = subprocess.run(base + ["train"] + common + ["affine", "1", "0", "true", "0"] + knobs,
                       capture_output=True, text=True, env=_env_vars())
    assert r.returncode == 0, r.stderr
    assert (runs / "h_affine_r1_push_true_s0" / "TRAIN_DONE").is_file()
    # train-index 0 of the true table is the same run: TRAIN_DONE makes it a no-op
    r = subprocess.run(base + ["train-index"] + common + ["true", "0"] + knobs,
                       capture_output=True, text=True, env=_env_vars())
    assert r.returncode == 0, r.stderr
    assert "TRAIN_DONE present" in r.stdout
    # an fnb run through the CLI on the synthetic f
    r = subprocess.run(base + ["train"] + common + ["affine", "1", "1", "fnb", "0", "--fnb-run",
                                                    str(env["f_run"]), "--fnb-noise-loss",
                                                    "mixture"] + knobs,
                       capture_output=True, text=True, env=_env_vars())
    assert r.returncode == 0, r.stderr
    d = runs / f"h_affine_r1_noise_{F_RUN}_s0"
    assert (d / "TRAIN_DONE").is_file()
    assert json.loads((d / "config.json").read_text())["fnb_noise_loss"] == "mixture"
    assert "n_capped" in json.loads((d / "timing.json").read_text())
