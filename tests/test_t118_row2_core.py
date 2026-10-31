"""t118 row 2 core harness — per-bin g (`model.GBin`), `theta_levels`, `predict_chrom_bins`, the row-2
Predictor, `pairs.tasks(row=2)`, and the row-1 regression.

The row-2 forms A2..D2 are built elsewhere; these tests use `form_identity` and two minimal per-bin
forms defined here (registered as `ladder.fforms.form_z*` so `base.load_form` finds them by name).
"""
from __future__ import annotations

import hashlib
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

from ladder import base, encoding, model, pairs, synth, train  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
PY = sys.executable
PAIRS_PY = ROOT / "tools" / "t118" / "ladder" / "pairs.py"
MANIFEST = ROOT / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
ROW1_TASKS_MD5 = "110feba71ebf5f52a4e9cf9e177f443c"
STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
FAST = {"window": 256, "batch": 16, "knot_sample": 20_000, "knot_block": 256}
CPU = torch.device("cpu")


# -- test-only per-bin forms ------------------------------------------------------------------------

class BinShift(base.FForm):
    """Row 2, context 0: loc = x + theta[..., 0], disp = theta[..., 1]."""
    rung, n_theta, context, per_bin = "zbin", 2, 0, True

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


class BinBox(BinShift):
    """Row 2, context 2: loc = 5-bin box mean of (x + theta[..., 0]), so the halo's theta counts."""
    rung, context = "zbinbox", 2

    def _loc(self, x, theta):
        z = (x + theta[..., 0])[:, None, :]
        return torch.nn.functional.conv1d(z, torch.full((1, 1, 5), 0.2, dtype=z.dtype))[:, 0, :]


for _name, _cls in (("zbin", BinShift), ("zbinbox", BinBox)):
    _mod = types.ModuleType(f"ladder.fforms.form_{_name}")
    _mod.build = (lambda c: (lambda space, stats: c(space, stats)))(_cls)
    sys.modules[_mod.__name__] = _mod


def _trained_like(lad, seed=0):
    """Give g's last layer random weights so theta really varies with x and the covariates
    (`lad` is a Ladder or a bare g)."""
    net = lad.g.net if isinstance(lad, model.Ladder) else lad.net
    torch.manual_seed(seed)
    with torch.no_grad():
        torch.nn.init.normal_(net[-1].weight, std=0.3)
    return lad


def _padded_x(X, c, space="counts"):
    return torch.from_numpy(model.transform_x(np.concatenate([np.zeros(c), X, np.zeros(c)]), space))


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    manifest, cov = synth.make_products(root / "products", 0)
    return {"root": root, "products": root / "products", "manifest": manifest, "cov": cov}


# -- GBin -------------------------------------------------------------------------------------------

def test_gbin_identity_at_init_for_random_x():
    torch.manual_seed(0)
    lad = model.Ladder("zbin", "counts", STATS)
    assert lad.per_bin and isinstance(lad.g, model.GBin)
    assert lad.g.net[0].in_features == 41 and lad.g.net[2].in_features == 64
    x = torch.randn(5, 37) * 4
    th = lad.theta(torch.randn(5, 40), x)
    assert th.shape == (5, 37, 2)
    assert torch.equal(th, lad.form.init_theta().expand(5, 37, 2))
    loc, disp = lad(torch.randn(5, 40), x)
    assert torch.equal(loc, x) and torch.allclose(disp, torch.full_like(x, math.log(5.0)))
    with pytest.raises(ValueError):
        lad.theta(torch.randn(5, 40))                       # row 2 needs x


def test_gbin_first_layer_equals_the_concatenated_input():
    lad = _trained_like(model.Ladder("zbin", "pval", STATS))
    torch.manual_seed(1)
    cov, x = torch.randn(3, 40), torch.randn(3, 11)
    full = torch.cat([x[:, :, None], cov[:, None, :].expand(-1, 11, -1)], -1)   # [3, 11, 41]
    torch.testing.assert_close(lad.g(cov, x), lad.g.net(full), rtol=1e-5, atol=1e-6)
    assert lad.g.levels(cov[:1], x[0]).shape == (1, 11, 2)


def test_theta_levels_equals_direct_per_bin_evaluation():
    lad = _trained_like(model.Ladder("zbin", "counts", STATS))
    torch.manual_seed(2)
    cov = torch.randn(4, 40)
    xv = torch.linspace(-1, 8, 23)
    direct = torch.stack([torch.stack([lad.g(cov[j:j + 1], xv[u].reshape(1, 1))[0, 0]
                                       for u in range(23)]) for j in range(4)])   # [4, 23, 2]
    got = model.theta_levels(lad, xv, cov[:1])
    assert got.shape == (23, 2)
    torch.testing.assert_close(got, direct[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(model.theta_levels(lad, xv, cov, average=True), direct.mean(0),
                               rtol=1e-5, atol=1e-6)
    d = direct.detach().numpy()
    assert np.ptp(d[0, :, 0]) > 0 and np.ptp(d[:, 0, 0]) > 0      # theta varies with x and with C
    with pytest.raises(ValueError):
        model.theta_levels(lad, xv, cov)                    # k > 1 needs average=True
    old = model.LEVEL_ROWS
    try:                                                    # blocking over levels changes nothing
        model.LEVEL_ROWS = 5
        torch.testing.assert_close(model.theta_levels(lad, xv, cov, average=True),
                                   direct.mean(0), rtol=1e-5, atol=1e-6)
    finally:
        model.LEVEL_ROWS = old


@pytest.mark.parametrize("rung", ["zbin", "zbinbox"])
def test_predict_chrom_bins_equals_form_of_g_on_a_whole_chromosome(rung):
    lad = _trained_like(model.Ladder(rung, "counts", STATS), seed=3)
    c = lad.context
    X = np.random.default_rng(4).poisson(3.0, 1003).astype(np.uint32)
    X[::50] = 40
    torch.manual_seed(5)
    cov = torch.randn(1, 40)
    xf = _padded_x(X, c)[None]                              # the zero halo, as in training
    with torch.no_grad():
        want_loc, want_disp = lad.form(xf, lad.g(cov, xf))
    for chunk in (97, 1003, model.CHUNK):
        loc, disp = model.predict_chrom_bins(lad, X, cov, CPU, chunk=chunk)
        assert loc.shape == (1003,) and loc.dtype == np.float32
        np.testing.assert_allclose(loc, want_loc[0].numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(disp, want_disp[0].numpy(), rtol=1e-5, atol=1e-5)


def test_averaged_twin_equals_the_explicit_mean_over_pairs():
    lad = _trained_like(model.Ladder("zbinbox", "pval", STATS), seed=6)
    X = np.random.default_rng(7).exponential(2.0, 700)
    torch.manual_seed(8)
    cov = torch.randn(9, 40)
    xf = _padded_x(X, lad.context, "pval")[None]
    with torch.no_grad():
        theta_bar = torch.stack([lad.g(cov[j:j + 1], xf)[0] for j in range(9)]).mean(0)[None]
        want_loc, want_disp = lad.form(xf, theta_bar)
    loc, disp = model.predict_chrom_bins(lad, X, cov, CPU, average=True, chunk=128)
    np.testing.assert_allclose(loc, want_loc[0].numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(disp, want_disp[0].numpy(), rtol=1e-5, atol=1e-5)


# -- task table -------------------------------------------------------------------------------------

def _pinned_table(rungs) -> str:
    """The task table TSV straight from the pinned formula, independent of `pairs.tasks`."""
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


def test_row1_task_table_is_byte_identical():
    assert pairs.RUNGS == ("A", "B", "C", "D")
    want = _pinned_table(("A", "B", "C", "D"))
    assert hashlib.md5(want.encode()).hexdigest() == ROW1_TASKS_MD5
    assert pairs.tasks() == pairs.tasks(None, 1) == pairs.tasks(row=1)
    if MANIFEST.is_file():
        for extra in ([], ["--row", "1"]):
            out = subprocess.run([PY, str(PAIRS_PY), "tasks", str(MANIFEST), *extra], check=True,
                                 capture_output=True, text=True).stdout
            assert out == want
        count = subprocess.run([PY, str(PAIRS_PY), "count", str(MANIFEST)], check=True,
                               capture_output=True, text=True).stdout
        assert count.splitlines()[-1] == "tasks 576"


def test_row2_task_table_names():
    assert pairs.RUNGS_ROW2 == ("A2", "B2", "C2", "D2")
    assert pairs.ROW1_OF == {"A2": "A", "B2": "B", "C2": "C", "D2": "D"}
    assert [pairs.row_of(r) for r in ("A", "D", "A2", "D2")] == [1, 1, 2, 2]
    for bad in ("identity", "E", "a2", "X"):
        with pytest.raises(ValueError):
            pairs.row_of(bad)
    t1, t2 = pairs.tasks(), pairs.tasks(row=2)
    assert len(t2) == 576 and [t["index"] for t in t2] == list(range(576))
    assert t2[54]["run_name"] == "A2_C19M16_counts_real_s0"
    for a, b in zip(t1, t2):
        assert pairs.ROW1_OF[b["rung"]] == a["rung"]
        assert {k: v for k, v in a.items() if k not in ("rung", "run_name")} == \
            {k: v for k, v in b.items() if k not in ("rung", "run_name")}
        assert b["run_name"] == f"{b['rung']}_{b['g']}_{b['space']}_{b['model']}_s{b['seed']}"
    with pytest.raises(ValueError):
        pairs.tasks(row=3)
    if MANIFEST.is_file():
        out = subprocess.run([PY, str(PAIRS_PY), "tasks", str(MANIFEST), "--row", "2"],
                             check=True, capture_output=True, text=True).stdout
        assert out == _pinned_table(pairs.RUNGS_ROW2)


def test_train_index_row_flag_selects_the_table(prod, monkeypatch):
    got = []
    monkeypatch.setattr(train, "run_training", lambda *a, **k: got.append(a[4:9]))
    args = [str(prod["manifest"]), str(prod["cov"]), str(prod["products"]), "runs", "54"]
    train.main(["train-index", *args])
    train.main(["train-index", *args, "--row", "2"])
    assert got == [("A", "C19M16", "counts", "real", 0), ("A2", "C19M16", "counts", "real", 0)]


# -- row-1 regression -------------------------------------------------------------------------------

@pytest.mark.parametrize("rung,space", [("identity", "counts"), ("A", "counts"), ("A", "pval")])
def test_row1_ladder_forward_and_predict_chrom_unchanged(rung, space):
    torch.manual_seed(11)
    lad = model.Ladder(rung, space, STATS)
    form = base.load_form(rung, space, STATS)
    torch.manual_seed(11)                                   # construction draws the same numbers
    g = model.G(40, form.n_theta, form.init_theta())
    assert not lad.per_bin and type(lad.g) is model.G and lad.g.net[0].in_features == 40
    sd = lad.g.state_dict()
    assert sd.keys() == g.state_dict().keys()
    assert all(torch.equal(sd[k], v) for k, v in g.state_dict().items())
    _trained_like(lad, seed=12)
    _trained_like(g, seed=12)
    torch.manual_seed(13)
    cov, x = torch.randn(4, 40), torch.randn(4, 64).abs()
    with torch.no_grad():
        want = form(x, g.net(cov))
        got = lad(cov, x)
        assert torch.equal(lad.theta(cov), g.net(cov))
        assert torch.equal(lad.theta(cov, x), g.net(cov))  # row 1 ignores x
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])
    X = np.random.default_rng(14).poisson(2.0, 500).astype(np.uint32)
    theta = lad.theta(cov[:1])[0].detach()
    loc, disp = model.predict_chrom(lad, X, theta, CPU, chunk=97)
    xt = torch.from_numpy(model.transform_x(X, space))[None]
    with torch.no_grad():
        wl, wd = form(xt, g.net(cov[:1]))
    assert np.array_equal(loc, wl[0].numpy()) and np.array_equal(disp, wd[0].numpy())


# -- the row-2 run end to end -----------------------------------------------------------------------

@pytest.mark.parametrize("model_name", ["real", "nocov", "ids"])
def test_row2_run_trains_and_predicts_per_bin(prod, model_name):
    rd = train.run_training(prod["manifest"], prod["cov"], prod["products"], prod["root"] / "r2",
                            "zbinbox", "T1", "counts", model_name, 0, max_steps=40, eval_every=20,
                            device="cpu", lr=1e-2, **FAST)
    import json
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["row"] == 2 and list(cfg)[:7] == ["run_name", "rung", "g", "space", "model",
                                                 "seed", "row"]
    pr = model.load_run(rd, prod["products"], prod["cov"], prod["manifest"], device="cpu")
    assert pr.per_bin and isinstance(pr.ladder.g, model.GBin)
    tp = pr.train_pairs
    cov_all = torch.from_numpy(np.stack([pr.encoder.encode_pair(p["source_pid"], p["target_pid"])
                                         for p in tp]))
    knots = torch.as_tensor(pr.ladder.form.stats["knots_x"], dtype=torch.float32)
    src, tgt = "T1__base__base", "T1__depth__15M"
    X = pr.corpus.get(src, "counts", "chr19")
    loc, disp = pr.predict(src, src, tgt, "chr19")
    th = pr.theta(src, tgt)
    assert th.shape == (12, pr.n_theta)
    if model_name == "nocov":
        assert pr._nocov_cov.shape == (len(tp), 40) and pr._nocov_theta is None
        want = model.predict_chrom_bins(pr.ladder, X, cov_all, CPU, average=True)
        with torch.no_grad():
            th_want = pr.ladder.g.levels(cov_all, knots).mean(0).numpy()
        loc2, _ = pr.predict(src, "T2__pe__pe", "T1__pe__pe", "chr19")
        assert np.array_equal(loc, loc2)                    # covariates never consulted
    else:
        s, t = pr.cov_pids(src, tgt)
        cov1 = torch.from_numpy(pr.encoder.encode_pair(s, t)).reshape(1, -1)
        want = model.predict_chrom_bins(pr.ladder, X, cov1, CPU)
        with torch.no_grad():
            th_want = pr.ladder.g.levels(cov1, knots)[0].numpy()
    assert np.array_equal(loc, want[0]) and np.array_equal(disp, want[1])
    np.testing.assert_allclose(th, th_want, rtol=1e-6, atol=1e-6)
    d = pr.describe(src, tgt)
    assert d["n_levels"] == 12 and len(d["shift"]) == 12
    assert d["levels_x"] == [float(v) for v in pr.ladder.form.stats["knots_x"]]
    # flat window at each level: box mean of (level + shift(level)) = level + shift(level)
    np.testing.assert_allclose(d["response"]["loc"], np.array(d["levels_x"]) + th[:, 0],
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(d["response"]["disp"], th[:, 1], rtol=1e-6, atol=1e-6)


def test_row1_config_has_no_row_key(prod):
    rd = train.run_training(prod["manifest"], prod["cov"], prod["products"], prod["root"] / "r1",
                            "identity", "T2", "counts", "real", 0, max_steps=2, eval_every=1,
                            device="cpu", **FAST)
    import json
    cfg = json.loads((rd / "config.json").read_text())
    assert "row" not in cfg and list(cfg)[6] == "window"
    assert set(train.DEFAULTS) <= set(cfg) and encoding.N_COV == cfg["n_cov"]
