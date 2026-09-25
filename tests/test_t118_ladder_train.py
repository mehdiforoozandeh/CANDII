"""t118 ladder training harness — `tools/t118/ladder/{encoding,model,train}.py`, `fforms/form_identity.py`.

Runs on CPU on `synth.make_products`. The real rungs live in C2a-d; here a 2-parameter affine stub
(and a 5-bin box stub with a context halo) are registered as `ladder.fforms.form_z*` modules so
`base.load_form` finds them by name, exactly as it finds a rung.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import lognorm, nbinom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import base, data, encoding, model, pairs, synth, train  # noqa: E402

PY = sys.executable
TRAIN_PY = ROOT / "tools" / "t118" / "ladder" / "train.py"
FAST = {"window": 256, "batch": 16, "knot_sample": 20_000, "knot_block": 256}


# -- test-only forms ------------------------------------------------------------------------------

class AffineStub(base.FForm):
    """loc = a + b x, disp = log n0 / log sigma0 fixed."""
    rung, n_theta, context = "zaffine", 2, 0

    def __init__(self, space, stats):
        super().__init__(space, stats)
        self.d0 = math.log(self.stats["n0"] if space == "counts" else self.stats["sigma0"])

    def init_theta(self):
        return torch.tensor([0.0, 1.0])

    def forward(self, x, theta):
        return theta[:, :1] + theta[:, 1:2] * x, torch.full_like(x, self.d0)

    def describe(self, theta):
        return {"a": float(theta[0]), "b": float(theta[1])}


class BoxStub(AffineStub):
    """loc = a + 5-bin box mean of x (context 2): exercises the halo in `predict_chrom`."""
    rung, n_theta, context = "zbox", 2, 2

    def forward(self, x, theta):
        k = torch.full((1, 1, 5), 0.2, dtype=x.dtype)
        xc = torch.nn.functional.conv1d(x[:, None, :], k)[:, 0, :]
        return theta[:, :1] + xc, torch.full_like(xc, self.d0)


for _name, _cls in (("zaffine", AffineStub), ("zbox", BoxStub)):
    _mod = types.ModuleType(f"ladder.fforms.form_{_name}")
    _mod.build = (lambda c: (lambda space, stats: c(space, stats)))(_cls)
    sys.modules[_mod.__name__] = _mod


STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    manifest, cov = synth.make_products(root / "products", 0)
    return {"root": root, "products": root / "products", "manifest": manifest, "cov": cov,
            "rows": pairs.read_manifest(manifest), "cov_rows": encoding.read_covariates(cov)}


@pytest.fixture(scope="module")
def affine_run(prod):
    """real, T1, counts, the affine stub: the planted depth effect has to be learned by g."""
    run_dir = train.run_training(prod["manifest"], prod["cov"], prod["products"],
                                 prod["root"] / "runs_aff", "zaffine", "T1", "counts", "real", 0,
                                 max_steps=400, eval_every=100, device="cpu", lr=1e-2, **FAST)
    return run_dir


# -- encoder --------------------------------------------------------------------------------------

def test_encoder_vector_layout_and_zscores(prod):
    fit = pairs.fit_pids(prod["rows"], "all")
    enc = encoding.Encoder.fit(prod["cov_rows"], fit)
    V = np.stack([enc.encode(p) for p in fit])
    assert V.shape == (len(fit), 20) and V.dtype == np.float32
    by = {r["pid"]: r for r in prod["cov_rows"]}
    # z columns: mean 0 over the fit pids (or all zero when constant)
    for j in (0, 1, 4, 8):
        assert abs(V[:, j].mean()) < 1e-5
    assert np.allclose(V[:, 1][V[:, 1] != 0].std(), 1.0, atol=1e-5)          # read_length 101/76
    # z* columns: 0 for no-control products (T2, DNase)
    noctl = [i for i, p in enumerate(fit) if by[p]["has_control"] == "0"]
    assert noctl and np.all(V[noctl, 5:8] == 0)
    assert np.all(V[:, 9] == [float(by[p]["has_control"]) for p in fit])
    assert np.all(V[:, 10:13].sum(1) == 1) and np.all(V[:, 13:20].sum(1) == 1)
    t2 = fit.index("T2__base__base")
    assert V[t2, 12] == 1 and V[t2, 13] == 1                     # ctl none, assay DNase-seq
    t1 = fit.index("T1__base__base")
    assert V[t1, 10] == 1 and V[t1, 14] == 1                     # ctl matched, assay H3K27ac
    pe = fit.index("T1__pe__pe")
    assert V[pe, 2] == 1 and V[t1, 2] == 0
    # the depth z-score orders the depth arms
    d = {p: V[fit.index(p), 0] for p in ("T1__depth__7.5M", "T1__depth__15M", "T1__base__base")}
    assert d["T1__depth__7.5M"] < d["T1__depth__15M"] < d["T1__base__base"]
    # g's input is the concatenation, never a difference
    pair = enc.encode_pair("T1__base__base", "T1__depth__15M")
    assert pair.shape == (40,)
    assert np.array_equal(pair[:20], enc.encode("T1__base__base"))
    assert np.array_equal(pair[20:], enc.encode("T1__depth__15M"))
    # state round trip (json-safe)
    enc2 = encoding.Encoder.from_state(json.loads(json.dumps(enc.state())))
    assert all(np.array_equal(enc.encode(p), enc2.encode(p)) for p in by)


def test_per_track_encoder_constant_assay_zeroes(prod):
    fit = pairs.fit_pids(prod["rows"], "T2")
    enc = encoding.Encoder.fit(prod["cov_rows"], fit)
    V = np.stack([enc.encode(p) for p in fit])
    assert np.all(V[:, 1] == 0)                                  # read_length constant -> 0
    assert np.all(V[:, 5:8] == 0)                                # no control anywhere


# -- losses, g ------------------------------------------------------------------------------------

def test_nb_nll_matches_scipy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 60, 500).astype(np.float64)
    mu = rng.gamma(2.0, 3.0, 500)
    n = rng.gamma(2.0, 2.0, 500)
    got = model.nb_nll(torch.tensor(np.log(mu)), torch.tensor(np.log(n)), torch.tensor(y))
    want = -nbinom.logpmf(y, n, n / (n + mu))
    assert np.allclose(got.numpy(), want, rtol=1e-8, atol=1e-8)


def test_lognormal_nll_matches_scipy_with_floor_in_loss_only():
    rng = np.random.default_rng(1)
    y = np.concatenate([np.zeros(50), rng.gamma(1.0, 2.0, 450)])
    loc = rng.normal(0, 1, 500)
    sig = rng.gamma(2.0, 0.3, 500)
    got = model.lognormal_nll(torch.tensor(loc), torch.tensor(np.log(sig)), torch.tensor(y))
    want = -lognorm.logpdf(np.maximum(y, 1e-3), s=sig, scale=np.exp(loc))
    assert np.allclose(got.numpy(), want, rtol=1e-8, atol=1e-8)
    assert np.isfinite(got.numpy()).all()


def test_g_starts_at_init_theta_for_every_input():
    torch.manual_seed(0)
    lad = model.Ladder("zaffine", "counts", STATS)
    assert lad.g.net[0].in_features == 40 and lad.g.net[2].in_features == 64
    th = lad.theta(torch.randn(7, 40))
    assert torch.equal(th, torch.tensor([[0.0, 1.0]]).expand(7, 2))
    lad_id = model.Ladder("identity", "pval", STATS)
    x = torch.randn(3, 50)
    loc, disp = lad_id(torch.randn(3, 40), x)
    assert torch.equal(loc, x) and torch.allclose(disp, torch.full_like(x, math.log(0.5)))


def test_predict_chrom_chunks_and_halo():
    lad = model.Ladder("zbox", "counts", STATS)
    X = np.random.default_rng(2).poisson(3.0, 1003).astype(np.uint32)
    theta = torch.tensor([0.3, 0.0])
    loc, disp = model.predict_chrom(lad, X, theta, torch.device("cpu"), chunk=97)
    xp = np.log1p(np.concatenate([np.zeros(2), X, np.zeros(2)]))
    want = 0.3 + np.convolve(xp, np.full(5, 0.2), mode="valid")
    assert loc.shape == (1003,) and disp.shape == (1003,)
    assert np.allclose(loc, want, atol=1e-5)


# -- twins ----------------------------------------------------------------------------------------

def test_ids_permutation_is_a_fixed_derangement_and_sampler_assignments(prod):
    for n, seed in ((38, 0), (14, 1), (242, 2), (8, 0)):
        perm = pairs.ids_permutation(n, seed)
        assert sorted(perm) == list(range(n)) and not np.any(perm == np.arange(n))
        assert np.array_equal(perm, pairs.ids_permutation(n, seed))
    corpus = data.Corpus(prod["products"])
    tp = pairs.train_pairs(prod["rows"], "T1")
    chroms, lens = train.training_chroms(corpus, tp[0]["source_pid"], "counts")
    assert set(chroms) == {"chr1", "chr2", "chrX"}
    perm = pairs.ids_permutation(len(tp), 0)
    draws = {}
    for m in pairs.MODELS:
        s = train.Sampler(corpus, "counts", tp, chroms, lens, 128, 0, 0, m, perm)
        draws[m] = [s.draw(16) for _ in range(5)]
    for k in range(5):
        pi_r, ci_r, x_r, y_r, _ = draws["real"][k]
        for m in ("nocov", "ids"):           # the same window stream for all three models
            pi, _, x, y, _ = draws[m][k]
            assert np.array_equal(pi, pi_r) and np.array_equal(x, x_r) and np.array_equal(y, y_r)
        assert np.array_equal(ci_r, pi_r)
        assert np.array_equal(draws["ids"][k][1], perm[pi_r])
    ci_n = np.concatenate([d[1] for d in draws["nocov"]])
    pi_n = np.concatenate([d[0] for d in draws["nocov"]])
    assert np.mean(ci_n == pi_n) < 0.5


def test_nocov_twin_ignores_the_covariate_table(prod):
    """The nocov loss curve is the same (within tolerance) for two unrelated covariate tables."""
    rng = np.random.default_rng(5)
    rows2 = [dict(r) for r in prod["cov_rows"]]
    for r in rows2:
        r["depth_log2"] = repr(float(rng.normal(20, 5)))
        r["read_length"] = repr(float(rng.integers(20, 200)))
        r["extsize_k"] = repr(float(rng.uniform(0.2, 3)))
        r["run_type_pe"] = str(int(rng.integers(2)))
    curves = []
    for i, cov_rows in enumerate((prod["cov_rows"], rows2)):
        rd = train.run_training(prod["manifest"], prod["cov"], prod["products"],
                                prod["root"] / f"runs_nocov{i}", "zaffine", "T1", "counts",
                                "nocov", 0, max_steps=200, eval_every=50, device="cpu", lr=1e-2,
                                cov_rows=cov_rows, **FAST)
        log = np.loadtxt(rd / "train_log.tsv", skiprows=1)
        curves.append(log[:, 2])
    a, b = curves
    assert a.shape == b.shape and a[0] == b[0]               # step 0: g ignores input at init
    assert np.max(np.abs(a - b) / np.abs(a)) < 0.02


# -- the whole loop -------------------------------------------------------------------------------

def test_planted_depth_effect_is_recovered(prod, affine_run):
    """g must learn, from the covariates alone, the total-count scale of each depth pair."""
    pr = model.load_run(affine_run, prod["products"], prod["cov"], prod["manifest"], device="cpu")
    base_pid = "T1__base__base"
    for arm, r in (("T1__depth__15M", 0.5), ("T1__depth__7.5M", 0.25)):
        for src, tgt, ratio in ((base_pid, arm, r), (arm, base_pid, 1.0 / r)):
            X = np.concatenate([pr.corpus.get(src, "counts", c) for c in pairs.SCORE_CHROMS])
            mu = np.concatenate([pr.mean(pr.predict(src, src, tgt, c)[0])
                                 for c in pairs.SCORE_CHROMS])
            assert abs(mu.sum() / X.sum() / ratio - 1) < 0.2, (src, tgt, mu.sum() / X.sum())
    # the offset a follows the depth ratio
    assert pr.theta(base_pid, "T1__depth__7.5M")[0] < pr.theta(base_pid, "T1__depth__15M")[0]
    val = np.loadtxt(affine_run / "train_log.tsv", skiprows=1)[:, 2]
    assert pr.val_nll == val.min() < val[0]


def test_checkpoint_round_trips_through_load_run(prod, affine_run):
    ck = affine_run / "ckpt.pt"
    assert ck.stat().st_size < 100_000
    ckpt = torch.load(ck, weights_only=True)
    assert set(ckpt) >= {"g", "form", "stats", "encoder", "config", "step", "val_nll"}
    for f in ("config.json", "train_log.tsv", "timing.json", "TRAIN_DONE"):
        assert (affine_run / f).is_file()
    timing = json.loads((affine_run / "timing.json").read_text())
    assert set(timing) >= {"setup", "train", "total", "peak_rss_mb"}
    cfg = json.loads((affine_run / "config.json").read_text())
    for k in ("git_sha", "manifest_md5", "covariates_md5", "fit_pids", "train_pairs", "window",
              "batch", "lr", "max_steps", "eval_every", "patience"):
        assert k in cfg
    pr = model.load_run(affine_run, prod["products"], prod["cov"], prod["manifest"], device="cpu")
    assert (pr.rung, pr.space, pr.g_id, pr.model, pr.seed) == ("zaffine", "counts", "T1", "real", 0)
    assert pr.fit_pids == pairs.fit_pids(prod["rows"], "T1")
    assert pr.train_pairs == json.loads(json.dumps(pairs.train_pairs(prod["rows"], "T1")))
    assert isinstance(pr.corpus, data.Corpus)
    assert pr.step == ckpt["step"] and pr.val_nll == ckpt["val_nll"]
    # theta = the checkpoint's g on an independently fitted encoder's vector
    enc = encoding.Encoder.fit(prod["cov_rows"], pairs.fit_pids(prod["rows"], "T1"))
    g = model.G(40, 2, torch.zeros(2))
    g.load_state_dict(ckpt["g"])
    src, tgt = "T1__base__base", "T1__pe__pe"
    with torch.no_grad():
        want = g(torch.from_numpy(enc.encode_pair(src, tgt))[None])[0].numpy()
    assert np.allclose(pr.theta(src, tgt), want, atol=1e-6)
    assert pr.describe(src, tgt) == pytest.approx({"a": float(want[0]), "b": float(want[1])})
    # predict with the covariates of p equals theta(p, p) applied by hand
    p = "T1__depth__15M"
    a, b = pr.theta(p, p)
    for chrom in ("chr19", "chr1"):
        loc, disp = pr.predict(p, p, p, chrom)
        X = pr.corpus.get(p, "counts", chrom)
        assert loc.dtype == np.float32 and loc.shape == X.shape
        assert np.allclose(loc, a + b * np.log1p(X.astype(np.float64)), atol=1e-5)
        assert np.allclose(disp, math.log(5.0))
    # helpers
    mu = pr.mean(loc)
    assert np.allclose(mu, np.exp(loc.astype(np.float64)))
    q = pr.quantiles(loc[:200], disp[:200], [0.05, 0.5, 0.95])
    assert q.shape == (3, 200) and np.all(q[0] <= q[1]) and np.all(q[1] <= q[2])
    assert pr.quantiles(loc[:5], disp[:5], 0.5).shape == (5,)


def test_ids_predictor_uses_the_permuted_labels_and_pval_round_trip(prod):
    rd = train.run_training(prod["manifest"], prod["cov"], prod["products"], prod["root"] / "rids",
                            "zaffine", "T2", "pval", "ids", 1, max_steps=40, eval_every=20,
                            device="cpu", lr=1e-2, **FAST)
    pr = model.load_run(rd, prod["products"], prod["cov"], prod["manifest"], device="cpu")
    tp = pairs.train_pairs(prod["rows"], "T2")
    perm = pairs.ids_permutation(len(tp), 1)
    assert pr.config["ids_perm"] == perm.tolist()
    for i, p in enumerate(tp):
        q = tp[perm[i]]
        assert pr.cov_pids(p["source_pid"], p["target_pid"]) == (q["source_pid"], q["target_pid"])
    assert pr.cov_pids("T2__pe__pe", "T2__pe__pe") == ("T2__pe__pe", "T2__pe__pe")   # swap
    p0, q0 = tp[0], tp[perm[0]]
    with torch.no_grad():
        want = pr.ladder.theta(torch.from_numpy(
            pr.encoder.encode_pair(q0["source_pid"], q0["target_pid"]))[None])[0].numpy()
    assert np.allclose(pr.theta(p0["source_pid"], p0["target_pid"]), want, atol=1e-7)
    s = "T2__base__base"
    a, b = pr.theta(s, s)
    loc, disp = pr.predict(s, s, s, "chr21")
    X = pr.corpus.get(s, "pval", "chr21").astype(np.float64)
    assert np.allclose(loc, a + b * np.log(np.maximum(X, 1e-3)), atol=1e-5)
    assert np.allclose(pr.mean(loc), np.exp(loc.astype(np.float64)))
    q = pr.quantiles(loc[:10], disp[:10], 0.5)
    assert np.allclose(q, np.exp(loc[:10].astype(np.float64)), rtol=1e-6)


def test_cli_train_on_synth_writes_checkpoint_and_skips_when_done(prod):
    runs = prod["root"] / "runs_cli"
    args = [PY, str(TRAIN_PY), "train", str(prod["manifest"]), str(prod["cov"]),
            str(prod["products"]), str(runs), "identity", "all", "counts", "real", "0",
            "--max-steps", "30", "--eval-every", "10", "--device", "cpu"]
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    t0 = time.time()
    out = subprocess.run(args, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert time.time() - t0 < 60
    rd = runs / "identity_all_counts_real_s0"
    for f in ("config.json", "ckpt.pt", "train_log.tsv", "timing.json", "TRAIN_DONE"):
        assert (rd / f).is_file(), f
    log = list(csv.DictReader(open(rd / "train_log.tsv"), delimiter="\t"))
    assert [int(r["step"]) for r in log] == [0, 10, 20, 30]
    assert json.loads((rd / "config.json").read_text())["max_steps"] == 30
    pr = model.load_run(rd, prod["products"], prod["cov"], prod["manifest"], device="cpu")
    assert len(pr.train_pairs) == 16 and pr.theta("T1__base__base", "T2__pe__pe").shape == (1,)
    mtime = (rd / "ckpt.pt").stat().st_mtime
    again = subprocess.run(args, capture_output=True, text=True, env=env)
    assert again.returncode == 0 and "TRAIN_DONE present" in again.stdout
    assert (rd / "ckpt.pt").stat().st_mtime == mtime


def test_cli_cache_then_train_from_the_memmap_cache(prod):
    cache = prod["root"] / "cache"
    rows = sorted(prod["rows"], key=lambda r: r["pid"])
    env = {"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}
    out = subprocess.run([PY, str(TRAIN_PY), "cache", str(prod["manifest"]), str(prod["products"]),
                          str(cache), "0"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    assert (cache / f"{rows[0]['pid']}__counts.json").is_file()
    assert (cache / f"{rows[0]['pid']}__pval.json").is_file()
    for r in rows[1:]:
        data.build_cache(prod["products"], cache, r["pid"], r)
    assert data.Corpus(cache).is_cache
    rd = train.run_training(prod["manifest"], prod["cov"], cache, prod["root"] / "runs_cache",
                            "zaffine", "T1", "counts", "real", 0, max_steps=20, eval_every=10,
                            device="cpu", **FAST)
    rp = train.run_training(prod["manifest"], prod["cov"], prod["products"],
                            prod["root"] / "runs_npz", "zaffine", "T1", "counts", "real", 0,
                            max_steps=20, eval_every=10, device="cpu", **FAST)
    # the cache and the npz products are the same data: identical training
    assert (rd / "train_log.tsv").read_text().split("\n")[1].split("\t")[:3] == \
        (rp / "train_log.tsv").read_text().split("\n")[1].split("\t")[:3]
    assert np.allclose(np.loadtxt(rd / "train_log.tsv", skiprows=1)[:, 2],
                       np.loadtxt(rp / "train_log.tsv", skiprows=1)[:, 2])
