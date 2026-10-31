"""t128 training loop and CLI (`tools/t128/train_ds.py`, `tools/t128/train.py`).

  * the loss is the pinned weighted sum of its logged terms, for every setting; beta and KL;
  * deterministic settings are deterministic, the VAE setting is not;
  * own-rule validation (PI ruling (iii)): the L1+L2 rule adds the cross terms to the L2 rule;
  * short runs of settings 1 (real), 3 (nocov) and 2 (DSBOX) write a t126-shaped run dir that
    `load_run`, `score_trained` and `score_law(split=True)` read; C_clean init zeros for form A,
    the base's knobs inside the box for form D (PI ruling (ii));
  * `train.py train-index` maps a task row onto `run_training_ds`.
"""
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

from ladder import data, encoding, model, pairs, score, synth  # noqa: E402
from ladder.fforms import form_ds  # noqa: E402
from t128 import pairs as tp  # noqa: E402
from t128 import train as tcli  # noqa: E402
from t128 import train_ds  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
N = encoding.N_COV
W = 64
C = form_ds.CONTEXT
SMALL = dict(max_steps=6, eval_every=3, device="cpu", window=256, batch=4, knot_sample=20_000,
             knot_block=256)


def _ladder(rung="DS", seed=0) -> model.Ladder:
    torch.manual_seed(seed)
    lad = model.Ladder(rung, "counts", STATS)
    with torch.no_grad():
        lad.g.mlp[-1].weight.normal_(0.0, 0.3)
        lad.g.mlp[-1].bias.normal_(0.5, 0.3)
        lad.form.f.head.weight.normal_(0.0, 0.3)
        lad.form.f.head.bias.normal_(0.0, 0.3)
        lad.g.cclean.normal_(0.0, 1.0)
        if lad.g.box:
            lad.g.box_lo.fill_(-1.0)
            lad.g.box_hi.fill_(2.0)
    return lad


def _batch(k=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    cov = torch.randn(k, 2 * N, generator=gen)
    for off in (0, N):
        cov[:, off + form_ds.N_KNOB: off + N] = 0.0
        cov[:, off + form_ds.N_KNOB + 1] = 1.0                 # one assay for c and c'
    cnt = lambda: torch.poisson(torch.full((k, W + 2 * C), 3.0), generator=gen)  # noqa: E731
    xs_c, xt_c = cnt(), cnt()
    mask = torch.ones(k, W, dtype=torch.bool)
    mask[0, W // 2:] = False
    return cov, torch.log1p(xs_c), torch.log1p(xt_c), xs_c[:, C:-C], xt_c[:, C:-C], mask


def test_beta_and_kl():
    assert train_ds.beta(0) == 0.0 and train_ds.beta(500) == 0.5
    assert train_ds.beta(1000) == 1.0 and train_ds.beta(4000) == 1.0
    z = torch.zeros(2, 5)
    assert float(train_ds.kl_gauss(z, z, torch.ones(2, 5, dtype=torch.bool))) == 0.0
    assert float(train_ds.kl_gauss(z + 1.0, z)) == pytest.approx(0.5)


@pytest.mark.parametrize("setting", tp.SETTINGS, ids=lambda s: s["arm"])
def test_loss_is_the_pinned_weighted_sum(setting):
    lad = _ladder(setting["rung"], 1)
    step = 500
    torch.manual_seed(0)
    total, t = train_ds.sym_loss(lad, *_batch(), setting, step)
    assert t["L1"] == t["cross_st"] + t["cross_ts"] and t["L2"] == t["self_s"] + t["self_t"]
    want = t["L2"]
    if setting["loss"] in ("L12", "L123"):
        want = want + t["L1"]
    if setting["loss"] == "L123":
        want = want + setting["lam"] * t["mse_z"]
    if setting["vae"]:
        assert t["beta"] == 0.5 and float(t["kl"]) > 0.0
        want = want + 0.5 * t["kl"]
    else:
        assert float(t["kl"]) == 0.0
    assert torch.allclose(total, want, rtol=0, atol=1e-6)
    assert float(t["mse_z"]) > 0.0                              # logged even at weight 0


def test_deterministic_vs_vae():
    lad = _ladder("DS", 2)
    b = _batch(seed=3)
    s1, s8 = tp.SETTING_BY_ARM["DS"], tp.SETTING_BY_ARM["DSvae"]
    _, a = train_ds.sym_loss(lad, *b, s1, 10)
    _, a2 = train_ds.sym_loss(lad, *b, s1, 10)
    assert all(torch.equal(torch.as_tensor(a[k]), torch.as_tensor(a2[k])) for k in a)
    _, v = train_ds.sym_loss(lad, *b, s8, 10)
    _, v2 = train_ds.sym_loss(lad, *b, s8, 10)
    assert not torch.equal(v["cross_st"], v2["cross_st"])


def test_loss_rejects_unknown():
    with pytest.raises(ValueError):
        train_ds.sym_loss(_ladder(), *_batch(), dict(tp.SETTINGS[0], loss="L9"), 1)


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t128train")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed}


def test_validate_own_rules(prod):
    rows = pairs.read_manifest(prod["manifest"])
    tpairs = pairs.train_pairs(rows, "T1", tp.HOLDOUT)
    fit = pairs.fit_pids(rows, "T1", tp.HOLDOUT)
    enc = encoding.Encoder.fit(encoding.read_covariates(prod["cov"]), fit)
    cov = torch.from_numpy(np.stack([enc.encode_pair(p["source_pid"], p["target_pid"])
                                     for p in tpairs]))
    corpus = data.Corpus(prod["root"] / "products")
    lad = _ladder("DS", 4)
    l2 = train_ds.validate_own(lad, corpus, tpairs, cov, "cpu", "L2 NLL")
    l12 = train_ds.validate_own(lad, corpus, tpairs, cov, "cpu", "L1+L2 NLL")
    assert np.isfinite(l2) and l12 > l2
    with pytest.raises(ValueError):
        train_ds.validate_own(lad, corpus, tpairs, cov, "cpu", "cross")


@pytest.fixture(scope="module")
def runs(prod):
    out = {}
    for arm, model_name in (("DS", "real"), ("DSrt", "nocov"), ("DSbox", "real")):
        out[arm] = train_ds.run_training_ds(
            prod["manifest"], prod["cov"], prod["root"] / "products", prod["root"] / "runs",
            tp.SETTING_BY_ARM[arm], "T1", model_name, 0, holdout=tp.HOLDOUT, **SMALL)
    return out


@pytest.mark.parametrize("arm,model_name", [("DS", "real"), ("DSrt", "nocov"), ("DSbox", "real")])
def test_run_dir(prod, runs, arm, model_name):
    run_dir = runs[arm]
    setting = tp.SETTING_BY_ARM[arm]
    assert run_dir.name == f"{arm}_T1_counts_{model_name}_s0"
    for f in ("config.json", "ckpt.pt", "train_log.tsv", "timing.json", "TRAIN_DONE", "cclean.json"):
        assert (run_dir / f).is_file()
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["rung"] == setting["rung"] and cfg["arm"] == arm and cfg["setting"] == setting
    assert cfg["holdout"] == [["depth", "7.5M"]]
    assert cfg["val_nll_rule"] == setting["val_nll_rule"] and cfg["kl_warmup_steps"] == 1000
    assert all("7.5M" not in p["source_pid"] + p["target_pid"] for p in cfg["train_pairs"])
    assert not any("7.5M" in p for p in cfg["fit_pids"]) and len(cfg["heldout_pairs"]) == 2
    log = (run_dir / "train_log.tsv").read_text().splitlines()
    assert log[0].split("\t") == ["step", "loss", "mse_z", "kl", "val_nll", "val_cross_nll",
                                  "seconds"]
    assert [int(r.split("\t")[0]) for r in log[1:]] == [0, 3, 6]
    init = np.array([cfg["cclean_init"][a] for a in encoding.ASSAYS])
    if setting["cclean"] == "A":
        assert np.allclose(init, 0.0, atol=1e-6) and "box_lo" not in cfg
    else:
        enc = encoding.Encoder.fit(encoding.read_covariates(prod["cov"]), cfg["fit_pids"])
        a = encoding.ASSAYS.index(synth.TRACK_SPECS["T1"]["assay"])
        assert np.allclose(init[a], enc.encode("T1__base__base")[:13], atol=1e-6)
        lo = np.array([cfg["box_lo"][k] for k in encoding.ASSAYS])
        hi = np.array([cfg["box_hi"][k] for k in encoding.ASSAYS])
        assert (lo <= init + 1e-6).all() and (init <= hi + 1e-6).all()
    cc = json.loads((run_dir / "cclean.json").read_text())
    assert cc["box"] == (setting["cclean"] == "D") and set(cc["assays"]) == set(encoding.ASSAYS)
    timing = json.loads((run_dir / "timing.json").read_text())
    pred = model.load_run(run_dir, prod["root"] / "products", prod["cov"], prod["manifest"],
                          device="cpu")
    assert pred.val_nll == timing["val_nll"]
    if model_name == "nocov":
        return
    rows = pairs.read_manifest(prod["manifest"])
    score.score_trained(pred, rows, prod["bed"], run_dir)
    recs = json.loads((run_dir / "scores.json").read_text())["records"]
    assert {(r["source_pid"], r["target_pid"]) for r in recs if r["kind"] == "heldout"} == {
        (p["source_pid"], p["target_pid"]) for p in cfg["heldout_pairs"]}
    score.score_law(pred, rows, prod["bed"], run_dir, split=True)
    assert (run_dir / "LAW_DONE").is_file()


def test_train_index_cli(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(train_ds, "run_training_ds", lambda *a, **k: calls.append((a, k)))
    assert tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), "57",
                      "--max-steps", "2", "--device", "cpu"]) == 0
    (m, c, d, r, setting, g, model_name, seed), kw = calls[0]
    t = tp.tasks()[57]
    assert setting == tp.SETTING_BY_ARM["DSbox"] and (g, model_name, seed) == (
        t["g"], t["model"], t["seed"])
    assert kw["holdout"] == tp.HOLDOUT and kw["max_steps"] == 2 and kw["device"] == "cpu"
    with pytest.raises(IndexError):
        tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), "240"])
