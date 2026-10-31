"""t118 row 2, the shuffled-bin-value twin (model "xshuf") — the draw (`ladder/xshuf.py`), the Ladder's
`x_g`, `predict_chrom_bins(x_g_rng=)`, the Sampler's sixth entry, `run_training`, the Predictor and
the twin's task table. The row-2 real path and the row-1 / row-2 tables must not move.

Rule under test (PI 2026-09-28, option (a)): g reads x at a uniformly random bin of the same
chromosome of the same source track; training draws are independent uniform with replacement from
a fixed per-(pid, chromosome) pool; prediction uses one fixed seeded permutation per chromosome;
pads read their own padded value.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import base, data, model, pairs, synth, train, xshuf  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
PY = sys.executable
PAIRS_PY = ROOT / "tools" / "t118" / "ladder" / "pairs.py"
MANIFEST = ROOT / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
ROW1_TASKS_MD5 = "110feba71ebf5f52a4e9cf9e177f443c"
ROW2_TASKS_MD5 = "991cef470efbeaeedb279e62ac45de45"
STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
FAST = {"window": 256, "batch": 16, "knot_sample": 20_000, "knot_block": 256}
CPU = torch.device("cpu")


# -- test-only per-bin forms (own names, so they never clash with other test files) --------------

class XBinShift(base.FForm):
    """Row 2, context 0: loc = x + theta[..., 0], disp = theta[..., 1]."""
    rung, n_theta, context, per_bin = "zxbin", 2, 0, True

    def init_theta(self):
        d0 = self.stats["n0"] if self.space == "counts" else self.stats["sigma0"]
        return torch.tensor([0.0, math.log(d0)])

    def forward(self, x, theta):
        assert theta.shape == (*x.shape, self.n_theta)
        c = self.context
        L = x.shape[1] - 2 * c
        return self._loc(x, theta), theta[:, c:c + L, 1]

    def _loc(self, x, theta):
        return x + theta[..., 0]

    def describe(self, theta):
        t = theta.detach().double().cpu()
        return {"n_levels": int(t.shape[0]), "shift": t[:, 0].tolist(), "disp": t[:, 1].tolist()}


class XBinBox(XBinShift):
    """Row 2, context 2: loc = 5-bin box mean of (x + theta[..., 0]), so the halo's theta counts."""
    rung, context = "zxbinbox", 2

    def _loc(self, x, theta):
        z = (x + theta[..., 0])[:, None, :]
        return torch.nn.functional.conv1d(z, torch.full((1, 1, 5), 0.2, dtype=z.dtype))[:, 0, :]


for _name, _cls in (("zxbin", XBinShift), ("zxbinbox", XBinBox)):
    _mod = types.ModuleType(f"ladder.fforms.form_{_name}")
    _mod.build = (lambda c: (lambda space, stats: c(space, stats)))(_cls)
    sys.modules[_mod.__name__] = _mod


def _trained_like(lad, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        torch.nn.init.normal_(lad.g.net[-1].weight, std=0.3)
    return lad


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("xshuf")
    manifest, cov = synth.make_products(root / "products", 0)
    return {"root": root, "products": root / "products", "manifest": manifest, "cov": cov,
            "corpus": data.Corpus(root / "products")}


# -- draw_positions ---------------------------------------------------------------------------------

def test_constants_pinned():
    assert xshuf.RULE == "same_chrom_uniform"
    assert xshuf.POOL_BINS == 1 << 17 == 131_072
    assert (xshuf.SALT_POOL, xshuf.SALT_TRAIN, xshuf.SALT_PRED) == (7122, 7123, 7124)
    assert pairs.MODEL_XSHUF == "xshuf"
    assert pairs.MODELS == ("real", "nocov", "ids")


def test_draw_positions_training_is_uniform_with_replacement():
    out = xshuf.draw_positions(np.random.default_rng(3), 100, 10_000)
    assert out.shape == (10_000,) and out.dtype == np.int64
    assert out.min() >= 0 and out.max() < 100
    assert np.array_equal(out, np.random.default_rng(3).integers(100, size=10_000))
    assert np.array_equal(out, xshuf.draw_positions(np.random.default_rng(3), 100, 10_000))
    assert len(np.unique(out)) == 100 < out.size              # with replacement: values repeat
    small = xshuf.draw_positions(np.random.default_rng(4), 5, 3)
    assert small.shape == (3,) and set(small) <= set(range(5))
    one = xshuf.draw_positions(np.random.default_rng(5), 1, 7)   # the bin itself is not excluded
    assert np.array_equal(one, np.zeros(7, dtype=np.int64))


def test_draw_positions_fixed_is_one_permutation():
    out = xshuf.draw_positions(np.random.default_rng(6), 1003, 1003, fixed=True)
    assert out.shape == (1003,) and out.dtype == np.int64
    assert np.array_equal(np.sort(out), np.arange(1003))
    assert np.array_equal(out, np.random.default_rng(6).permutation(1003))
    assert not np.array_equal(out, np.arange(1003))
    with pytest.raises(ValueError):
        xshuf.draw_positions(np.random.default_rng(6), 1003, 1002, fixed=True)
    with pytest.raises(ValueError):
        xshuf.draw_positions(np.random.default_rng(6), 0, 5)


def test_pred_rng_per_chromosome_and_seed():
    def perm(seed, chrom, n=500):
        return xshuf.draw_positions(xshuf.pred_rng(seed, chrom), n, n, fixed=True)
    a = perm(0, "chr19")
    assert np.array_equal(a, perm(0, "chr19"))                   # repeats exactly
    assert not np.array_equal(a, perm(0, "chr21"))               # differs by chromosome
    assert not np.array_equal(a, perm(1, "chr19"))               # differs by seed
    want = np.random.default_rng([0, 7124, pairs.MAIN_CHROMS.index("chr19")]).permutation(500)
    assert np.array_equal(a, want)
    for bad in ("chrY", "chrM", "chr23", "T1"):
        with pytest.raises(ValueError):
            xshuf.pred_rng(0, bad)
    tr = xshuf.train_rng(2).integers(1 << 30, size=4)
    assert np.array_equal(tr, np.random.default_rng([2, 7123]).integers(1 << 30, size=4))


# -- Pool -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("space", ["counts", "pval"])
def test_pool_whole_and_sampled(prod, space):
    corpus = prod["corpus"]
    pids = ["T1__pe__pe", "T1__base__base"]
    chroms, _ = train.training_chroms(corpus, "T1__base__base", space)
    assert chroms == ["chr1", "chr2", "chrX"]
    whole = xshuf.Pool(corpus, space, pids, chroms, 0)
    assert list(whole.values) == [(p, c) for p in sorted(pids) for c in chroms]
    for (pid, chrom), v in whole.values.items():
        X = corpus.get(pid, space, chrom)
        assert v.dtype == X.dtype and np.array_equal(v, X)       # shorter than n_bins: whole
        assert whole.n_source(pid, chrom) == X.shape[0]
    d = whole.describe()
    assert d == {"rule": "same_chrom_uniform", "pool_bins": 1 << 17, "n_pairs_pid_chrom": 6,
                 "salt_pool": 7122, "salt_train": 7123, "salt_pred": 7124}

    small = xshuf.Pool(corpus, space, pids, chroms, 3, n_bins=1000)   # chr1, chr2 > 1000; chrX 900
    rng = np.random.default_rng([3, 7122])
    for pid in sorted(pids):
        for chrom in chroms:
            X = np.asarray(corpus.get(pid, space, chrom))
            v = small.values[(pid, chrom)]
            assert v.dtype == X.dtype                            # raw, untransformed
            if X.shape[0] <= 1000:
                assert np.array_equal(v, X)
            else:
                idx = np.sort(rng.integers(X.shape[0], size=1000))   # the fixed rng order
                assert small.n_source(pid, chrom) == 1000
                assert np.array_equal(v, X[idx])
    assert small.describe()["pool_bins"] == 1000


# -- the Ladder and predict_chrom_bins --------------------------------------------------------------

@pytest.mark.parametrize("rung", ["zxbin", "zxbinbox", "A2", "D2"])
def test_ladder_forward_reads_x_g_in_g_only(rung):
    lad = _trained_like(model.Ladder(rung, "counts", STATS), seed=1)
    c = lad.context
    torch.manual_seed(2)
    cov = torch.randn(3, 40)
    x = torch.rand(3, 64 + 2 * c) * 4
    xg = torch.rand(3, 64 + 2 * c) * 4
    with torch.no_grad():
        got = lad(cov, x, x_g=xg)
        want = lad.form(x, lad.g(cov, xg))
        same = lad(cov, x)
        assert all(torch.equal(a, b) for a, b in zip(lad(cov, x, x_g=None), same))
        assert all(torch.equal(a, b) for a, b in zip(same, lad.form(x, lad.g(cov, x))))
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    if rung.startswith("zx"):                                  # D2's head may ignore theta at init
        assert not torch.equal(got[0], same[0])
    with pytest.raises(ValueError):
        lad(cov, x, x_g=xg[:, :-1])


def test_row1_ladder_rejects_x_g():
    lad = model.Ladder("A", "counts", STATS)
    cov, x = torch.randn(2, 40), torch.rand(2, 32)
    with pytest.raises(ValueError):
        lad(cov, x, x_g=x)
    lad(cov, x)


def _padded(X, c, space="counts"):
    return model.transform_x(np.concatenate([np.zeros(c), np.asarray(X, float), np.zeros(c)]), space)


@pytest.mark.parametrize("rung", ["zxbin", "zxbinbox", "C2"])
def test_predict_chrom_bins_with_the_fixed_permutation(rung):
    lad = _trained_like(model.Ladder(rung, "counts", STATS), seed=3)
    c = lad.context
    n = 1003
    X = np.random.default_rng(4).poisson(3.0, n).astype(np.uint32)
    X[::50] = 40
    torch.manual_seed(5)
    cov = torch.randn(1, 40)
    xf = torch.from_numpy(_padded(X, c))[None]
    perm = xshuf.draw_positions(xshuf.pred_rng(0, "chr19"), n, n, fixed=True)
    xg = xf.clone()
    xg[0, c:c + n] = xf[0, c + torch.from_numpy(perm)]           # bins permuted, pads themselves
    with torch.no_grad():
        want = lad.form(xf, lad.g(cov, xg))
        plain = lad.form(xf, lad.g(cov, xf))
    for chunk in (97, n, model.CHUNK):
        loc, disp = model.predict_chrom_bins(lad, X, cov, CPU, chunk=chunk,
                                             x_g_rng=xshuf.pred_rng(0, "chr19"))
        assert loc.shape == (n,) and loc.dtype == np.float32
        np.testing.assert_allclose(loc, want[0][0].numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(disp, want[1][0].numpy(), rtol=1e-5, atol=1e-5)
        l0, d0 = model.predict_chrom_bins(lad, X, cov, CPU, chunk=chunk)
        l1, d1 = model.predict_chrom_bins(lad, X, cov, CPU, chunk=chunk, x_g_rng=None)
        assert np.array_equal(l0, l1) and np.array_equal(d0, d1)
        np.testing.assert_allclose(l0, plain[0][0].numpy(), rtol=1e-5, atol=1e-5)
        assert not np.allclose(loc, l0)
    # the chunk padding past the chromosome end reads itself: a chunk not dividing n still agrees
    loc_a, _ = model.predict_chrom_bins(lad, X, cov, CPU, chunk=400,
                                        x_g_rng=xshuf.pred_rng(0, "chr19"))
    np.testing.assert_allclose(loc_a, want[0][0].numpy(), rtol=1e-5, atol=1e-5)


# -- Sampler ----------------------------------------------------------------------------------------

def _sampler(prod, window, context, pool, seed=0, space="counts"):
    rows = pairs.read_manifest(prod["manifest"])
    tp = pairs.train_pairs(rows, "T1")
    chroms, lens = train.training_chroms(prod["corpus"], tp[0]["source_pid"], space)
    return train.Sampler(prod["corpus"], space, tp, chroms, lens, window, context, seed, "xshuf",
                         pairs.ids_permutation(len(tp), seed), xshuf_pool=pool), tp, chroms


def test_sampler_stream_unchanged_and_x_g_from_the_pool(prod):
    corpus = prod["corpus"]
    rows = pairs.read_manifest(prod["manifest"])
    fit = pairs.fit_pids(rows, "T1")
    chroms, _ = train.training_chroms(corpus, "T1__base__base", "counts")
    pool = xshuf.Pool(corpus, "counts", fit, chroms, 0)
    for window, c in ((256, 0), (256, 30), (5000, 7)):      # 5000 > every chromosome: pads both sides
        plain, tp, chs = _sampler(prod, window, c, None)
        twin, _, _ = _sampler(prod, window, c, pool)
        for _ in range(3):
            a, b = plain.draw(8), twin.draw(8)
            assert len(a) == 5 and len(b) == 6
            for u, v in zip(a, b[:5]):
                assert np.array_equal(u, v)                     # the window stream is untouched
            xs, xs_g = b[2], b[5]
            assert xs_g.shape == xs.shape and xs_g.dtype == np.float32
            assert np.array_equal(b[1], b[0])                   # real covariates
            assert not np.array_equal(xs_g, xs)
    # exact reproduction of one batch, pads and bins
    twin, tp, chs = _sampler(prod, 5000, 7, pool, seed=4)
    rng_main = np.random.default_rng(4)
    pi = rng_main.integers(len(tp), size=6)
    _, lens = train.training_chroms(corpus, tp[0]["source_pid"], "counts")
    ci = rng_main.choice(len(chs), size=6, p=lens / lens.sum())
    pi2, cov_i, xs, ys, mask, xs_g = twin.draw(6)
    assert np.array_equal(pi, pi2)
    rng_x = np.random.default_rng([4, 7123])
    for k in range(6):
        pid, chrom = tp[pi[k]]["source_pid"], chs[ci[k]]
        n = int(lens[ci[k]])                                    # window 5000 > n: s = 0
        pad = model.transform_x(np.zeros(1), "counts")[0]
        assert np.all(xs_g[k, :7] == pad) and np.array_equal(xs_g[k, :7], xs[k, :7])
        assert np.all(xs_g[k, 7 + n:] == pad) and np.array_equal(xs_g[k, 7 + n:], xs[k, 7 + n:])
        pos = rng_x.integers(pool.n_source(pid, chrom), size=n)
        want = model.transform_x(pool.values[(pid, chrom)][pos], "counts")
        assert np.array_equal(xs_g[k, 7:7 + n], want)
        assert not np.array_equal(xs_g[k, 7:7 + n], xs[k, 7:7 + n])
        assert np.array_equal(xs[k, 7:7 + n], model.transform_x(corpus.get(pid, "counts", chrom),
                                                                "counts"))
    assert twin.cov_index(np.array([3, 1])).tolist() == [3, 1]


# -- run_training, the Predictor --------------------------------------------------------------------

def _run(prod, rung, model_name, sub, space="counts"):
    return train.run_training(prod["manifest"], prod["cov"], prod["products"], prod["root"] / sub,
                              rung, "T1", space, model_name, 0, max_steps=5, eval_every=5,
                              device="cpu", lr=1e-2, **FAST)


def _ckpt_tensors(run_dir):
    ck = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=True)
    return {f"{part}.{k}": v for part in ("g", "form") for k, v in ck[part].items()}, ck


def test_run_training_twin_and_the_real_path_unchanged(prod):
    real1 = _run(prod, "A2", "real", "real_before")
    twin = _run(prod, "A2", "xshuf", "twin")
    real2 = _run(prod, "A2", "real", "real_after")
    t1, c1 = _ckpt_tensors(real1)
    t2, c2 = _ckpt_tensors(real2)
    assert t1.keys() == t2.keys() and all(torch.equal(t1[k], t2[k]) for k in t1)
    assert c1["val_nll"] == c2["val_nll"] and c1["step"] == c2["step"]
    cfg_real = json.loads((real1 / "config.json").read_text())
    assert "xshuf" not in cfg_real and cfg_real["model"] == "real" and cfg_real["row"] == 2
    assert "xshuf_pool" not in json.loads((real1 / "timing.json").read_text())

    cfg = json.loads((twin / "config.json").read_text())
    assert cfg["model"] == "xshuf" and cfg["row"] == 2 and twin.name == "A2_T1_counts_xshuf_s0"
    assert cfg["xshuf"]["rule"] == "same_chrom_uniform"
    assert cfg["xshuf"]["pool_bins"] == 1 << 17 and cfg["xshuf"]["salt_pred"] == 7124
    assert cfg["xshuf"]["n_pairs_pid_chrom"] == len(cfg["fit_pids"]) * len(cfg["train_chroms"])
    assert [k for k in cfg if k != "xshuf"] == list(cfg_real)    # same keys, same order
    timing = json.loads((twin / "timing.json").read_text())
    assert isinstance(timing["xshuf_pool"], float) and timing["xshuf_pool"] >= 0
    tt, _ = _ckpt_tensors(twin)
    assert not all(torch.equal(tt[k], t1[k]) for k in t1)       # g read other x

    pr = model.load_run(twin, prod["products"], prod["cov"], prod["manifest"], device="cpu")
    assert pr.model == "xshuf" and pr.per_bin and pr._cov_map == {}
    assert pr._nocov_cov is None and pr._nocov_theta is None
    src, tgt = "T1__base__base", "T1__depth__15M"
    X = pr.corpus.get(src, "counts", "chr19")
    cov1 = torch.from_numpy(pr.encoder.encode_pair(src, tgt)).reshape(1, -1)
    loc, disp = pr.predict(src, src, tgt, "chr19")
    wl, wd = model.predict_chrom_bins(pr.ladder, X, cov1, CPU, x_g_rng=xshuf.pred_rng(0, "chr19"))
    assert np.array_equal(loc, wl) and np.array_equal(disp, wd)
    assert np.array_equal(pr.predict(src, src, tgt, "chr19")[0], loc)   # deterministic
    pl, _ = model.predict_chrom_bins(pr.ladder, X, cov1, CPU)
    assert not np.array_equal(loc, pl)
    assert pr.theta(src, tgt).shape == (12, pr.n_theta)
    assert pr.describe(src, tgt)["n_levels"] == 12


@pytest.mark.parametrize("rung,space", [("D2", "pval"), ("zxbinbox", "counts")])
def test_run_training_twin_with_a_halo(prod, rung, space):
    rd = _run(prod, rung, "xshuf", f"twin_{rung}", space)
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["model"] == "xshuf" and cfg["row"] == 2 and cfg["context"] > 0
    assert (rd / "TRAIN_DONE").is_file()
    val = [float(line.split("\t")[2]) for line in (rd / "train_log.tsv").read_text().splitlines()[1:]]
    assert val and all(np.isfinite(val))


@pytest.mark.parametrize("rung", ["A", "identity"])
def test_run_training_twin_needs_a_row2_rung(prod, rung):
    with pytest.raises(ValueError, match="model 'xshuf' needs a row-2 rung"):
        _run(prod, rung, "xshuf", "bad")
    with pytest.raises(ValueError):
        _run(prod, "A2", "shuffled", "bad")


def test_validate_uses_the_fixed_permutation(prod):
    lad = _trained_like(model.Ladder("zxbinbox", "counts", STATS), seed=9)
    rows = pairs.read_manifest(prod["manifest"])
    tp = pairs.train_pairs(rows, "T1")[:3]
    torch.manual_seed(10)
    cov = torch.randn(3, 40)
    corpus = prod["corpus"]
    plain = train.validate(lad, corpus, tp, cov, "counts", CPU)
    assert plain == train.validate(lad, corpus, tp, cov, "counts", CPU, xshuf_seed=None)
    got = train.validate(lad, corpus, tp, cov, "counts", CPU, xshuf_seed=2)
    per = []
    for i, p in enumerate(tp):
        X = corpus.get(p["source_pid"], "counts", "chr22")
        Y = torch.from_numpy(np.array(corpus.get(p["target_pid"], "counts", "chr22"), np.float32))
        loc, disp = model.predict_chrom_bins(lad, X, cov[i:i + 1], CPU,
                                             x_g_rng=xshuf.pred_rng(2, "chr22"))
        per.append(float(lad.nll(torch.from_numpy(loc), torch.from_numpy(disp), Y)))
    assert got == float(np.mean(per)) and got != plain


# -- task table -------------------------------------------------------------------------------------

def _pinned_table(rungs) -> str:
    lines = ["index\trung\tg\tspace\tmodel\tseed\trun_name"]
    rows = []
    for ri, r in enumerate(rungs):
        for gi, g in enumerate(pairs.G_IDS):
            for si, sp in enumerate(("counts", "pval")):
                for mi, m in enumerate(("real", "nocov", "ids")):
                    for s in (0, 1, 2):
                        rows.append((ri * 144 + gi * 18 + si * 9 + mi * 3 + s,
                                     f"{r}\t{g}\t{sp}\t{m}\t{s}\t{r}_{g}_{sp}_{m}_s{s}"))
    lines += [f"{i}\t{rest}" for i, rest in sorted(rows)]
    return "".join(line + "\n" for line in lines)


def _pinned_xshuf_table() -> str:
    lines = ["index\trung\tg\tspace\tmodel\tseed\trun_name"]
    for ri, r in enumerate(("A2", "B2", "C2", "D2")):
        for gi, g in enumerate(pairs.G_IDS):
            for si, sp in enumerate(("counts", "pval")):
                for s in (0, 1, 2):
                    lines.append(f"{ri * 48 + gi * 6 + si * 3 + s}\t{r}\t{g}\t{sp}\txshuf\t{s}\t"
                                 f"{r}_{g}_{sp}_xshuf_s{s}")
    return "".join(line + "\n" for line in lines)


def _cli(*args):
    return subprocess.run([PY, str(PAIRS_PY), *args], capture_output=True, text=True)


def test_row1_and_row2_tables_unchanged():
    t1, t2 = _pinned_table(pairs.RUNGS), _pinned_table(pairs.RUNGS_ROW2)
    assert hashlib.md5(t1.encode()).hexdigest() == ROW1_TASKS_MD5
    assert hashlib.md5(t2.encode()).hexdigest() == ROW2_TASKS_MD5
    assert pairs._tsv(pairs.tasks(), pairs.TASK_KEYS) == t1.split("\n", 1)[1]
    assert pairs._tsv(pairs.tasks(row=2), pairs.TASK_KEYS) == t2.split("\n", 1)[1]
    assert pairs.tasks(None, 2, xshuf=False) == pairs.tasks(row=2)
    if MANIFEST.is_file():
        assert _cli("tasks", str(MANIFEST)).stdout == t1
        assert _cli("tasks", str(MANIFEST), "--row", "2").stdout == t2
        assert _cli("count", str(MANIFEST)).stdout.splitlines()[-1] == "tasks 576"


def test_xshuf_table():
    t = pairs.tasks(row=2, xshuf=True)
    assert len(t) == 192 and [r["index"] for r in t] == list(range(192))
    assert all(tuple(r) == pairs.TASK_KEYS and r["model"] == "xshuf" for r in t)
    assert t[18]["run_name"] == "A2_C19M16_counts_xshuf_s0"
    assert [t[i]["run_name"] for i in (66, 114, 162)] == [
        "B2_C19M16_counts_xshuf_s0", "C2_C19M16_counts_xshuf_s0", "D2_C19M16_counts_xshuf_s0"]
    assert t[21]["run_name"] == "A2_C19M16_pval_xshuf_s0"
    assert all(r["run_name"] == train.run_name(r["rung"], r["g"], r["space"], r["model"], r["seed"])
               for r in t)
    assert pairs._tsv(t, pairs.TASK_KEYS) == _pinned_xshuf_table().split("\n", 1)[1]
    with pytest.raises(ValueError):
        pairs.tasks(row=1, xshuf=True)
    if MANIFEST.is_file():
        out = _cli("tasks", str(MANIFEST), "--row", "2", "--xshuf")
        assert out.returncode == 0 and out.stdout == _pinned_xshuf_table()
        assert out.stdout.splitlines()[19] == "18\tA2\tC19M16\tcounts\txshuf\t0\tA2_C19M16_counts_xshuf_s0"
        for extra in ([], ["--row", "1"]):
            bad = _cli("tasks", str(MANIFEST), *extra, "--xshuf")
            assert bad.returncode == 2 and bad.stdout == "" and "--row 2" in bad.stderr


def test_train_index_xshuf_flag(prod, monkeypatch):
    got = []
    monkeypatch.setattr(train, "run_training", lambda *a, **k: got.append(a[4:9]))
    args = [str(prod["manifest"]), str(prod["cov"]), str(prod["products"]), "runs"]
    train.main(["train-index", *args, "18", "--row", "2", "--xshuf"])
    train.main(["train-index", *args, "162", "--row", "2", "--xshuf"])
    train.main(["train-index", *args, "54", "--row", "2"])
    assert got == [("A2", "C19M16", "counts", "xshuf", 0), ("D2", "C19M16", "counts", "xshuf", 0),
                   ("A2", "C19M16", "counts", "real", 0)]
    with pytest.raises(IndexError):
        train.main(["train-index", *args, "192", "--row", "2", "--xshuf"])
    for extra in ([], ["--row", "1"]):
        with pytest.raises(SystemExit) as e:
            train.main(["train-index", *args, "18", *extra, "--xshuf"])
        assert e.value.code == 2
