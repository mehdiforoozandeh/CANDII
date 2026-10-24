"""t126 rung D — encoder-decoder D (form DED, `GEncDecD`, `t126/train_ded.py`).

  * the self path dec(c) o enc(c) is exactly what the self terms of the loss train;
  * an arm->arm prediction (the Ladder / Predictor path) equals dec(enc(X1, c1), c2);
  * g_enc reads c only, g_dec reads c' only; no skip from x (the output sees x only through Z);
  * the four-term loss is the sum of its four NLLs plus lambda * mse(Z, Z');
  * the hold-out removes every 7.5M pair from training, and a short train + score + law works.
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

from ladder import encoding, model, pairs, score, synth  # noqa: E402
from ladder.fforms import form_ded  # noqa: E402
from t126 import pairs as tp  # noqa: E402
from t126 import train_ded  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
N = encoding.N_COV
C = form_ded.CONTEXT


def _ladder(seed=0) -> model.Ladder:
    """A DED Ladder with random (non-zero) g last layers and head, so nothing is trivially 0."""
    torch.manual_seed(seed)
    lad = model.Ladder("DED", "counts", STATS)
    with torch.no_grad():
        for net in (lad.g.g_enc, lad.g.g_dec):
            net[-1].weight.normal_(0.0, 0.3)
            net[-1].bias.normal_(0.5, 0.3)
        lad.form.head.weight.normal_(0.0, 0.3)
        lad.form.head.bias.normal_(0.0, 0.3)
    return lad


def _covs(k, seed):
    return torch.randn(k, N, generator=torch.Generator().manual_seed(seed))


def _x(k, w, seed):
    return torch.log1p(torch.poisson(torch.full((k, w), 3.0),
                                     generator=torch.Generator().manual_seed(seed)))


def test_shape_and_init():
    lad = model.Ladder("DED", "counts", STATS)
    assert isinstance(lad.g, model.GEncDecD)
    assert lad.context == 30 and lad.n_theta == 256
    th = lad.theta(torch.cat([_covs(4, 1), _covs(4, 2)], 1))
    assert torch.equal(th, lad.form.init_theta().expand(4, -1))
    x = _x(4, 300, 0)
    loc, disp = lad(torch.cat([_covs(4, 1), _covs(4, 2)], 1), x)
    assert loc.shape == (4, 300 - 2 * C)
    assert torch.equal(loc, torch.zeros_like(loc))          # zero head, no skip: constant at init
    assert torch.allclose(disp, torch.full_like(disp, float(np.log(5.0))))


def test_self_path_is_dec_c_of_enc_c():
    lad = _ladder(1)
    c = _covs(6, 3)
    x = _x(6, 250, 1)
    loc, disp = lad(torch.cat([c, c], 1), x)
    l2, d2 = lad.form.decode(lad.form.encode(x, lad.g.g_enc(c)), lad.g.g_dec(c))
    assert torch.allclose(loc, l2, atol=1e-6) and torch.allclose(disp, d2, atol=1e-6)
    # the loss's self term on (X, c) is this map's NLL against X itself
    y = torch.expm1(x[:, C:-C])
    mask = torch.ones_like(y, dtype=torch.bool)
    _, terms = train_ded.four_term_loss(lad, torch.cat([c, _covs(6, 4)], 1), x, _x(6, 250, 2),
                                        y, torch.expm1(_x(6, 250, 2)[:, C:-C]), mask, 0.0)
    assert torch.allclose(terms["self_s"], lad.nll(loc, disp, y, mask), atol=1e-6)


def test_arm_to_arm_is_dec_c2_of_enc_x1_c1():
    lad = _ladder(2)
    c1, c2 = _covs(5, 5), _covs(5, 6)
    x1 = _x(5, 260, 3)
    loc, disp = lad(torch.cat([c1, c2], 1), x1)
    l2, d2 = lad.form.decode(lad.form.encode(x1, lad.g.g_enc(c1)), lad.g.g_dec(c2))
    assert torch.allclose(loc, l2, atol=1e-6) and torch.allclose(disp, d2, atol=1e-6)
    # the whole-chromosome predictor path (chunks + halo) gives the same centre bins
    theta = lad.theta(torch.cat([c1[:1], c2[:1]], 1))[0]
    X = torch.expm1(x1[0]).numpy()
    lp, _ = model.predict_chrom(lad, X, theta, "cpu", chunk=100)
    assert np.allclose(lp[C:-C], loc[0].detach().numpy(), atol=1e-5)


def test_each_g_reads_one_side():
    lad = _ladder(3)
    c, ct, other = _covs(4, 7), _covs(4, 8), _covs(4, 9)
    enc, dec = form_ded.FormDED.split_theta(lad.theta(torch.cat([c, ct], 1)))
    enc2, dec2 = form_ded.FormDED.split_theta(lad.theta(torch.cat([c, other], 1)))
    enc3, dec3 = form_ded.FormDED.split_theta(lad.theta(torch.cat([other, ct], 1)))
    assert torch.equal(enc, enc2) and not torch.allclose(dec, dec2)
    assert torch.equal(dec, dec3) and not torch.allclose(enc, enc3)
    assert torch.equal(form_ded.FormDED.join_theta(enc, dec), lad.theta(torch.cat([c, ct], 1)))


def test_no_skip_from_x():
    lad = _ladder(4)
    with torch.no_grad():
        for conv in lad.form.enc:
            conv.weight.zero_()
    cov = torch.cat([_covs(3, 1), _covs(3, 2)], 1)
    a, _ = lad(cov, _x(3, 200, 1))
    b, _ = lad(cov, _x(3, 200, 2))
    assert torch.allclose(a, b)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_four_term_loss_is_the_sum(lam):
    lad = _ladder(5)
    c, ct = _covs(4, 1), _covs(4, 2)
    xs, xt = _x(4, 220, 1), _x(4, 220, 2)
    ys, yt = torch.expm1(xs[:, C:-C]), torch.expm1(xt[:, C:-C])
    mask = torch.ones_like(ys, dtype=torch.bool)
    total, t = train_ded.four_term_loss(lad, torch.cat([c, ct], 1), xs, xt, ys, yt, mask, lam)
    want = t["cross_st"] + t["cross_ts"] + t["self_s"] + t["self_t"] + lam * t["mse_z"]
    assert torch.allclose(total, want)
    l, d = lad(torch.cat([c, ct], 1), xs)
    assert torch.allclose(t["cross_st"], lad.nll(l, d, yt, mask), atol=1e-6)
    l, d = lad(torch.cat([ct, c], 1), xt)
    assert torch.allclose(t["cross_ts"], lad.nll(l, d, ys, mask), atol=1e-6)
    assert float(t["mse_z"]) > 0


def test_task_table_d():
    t = tp.tasks(table="D")
    assert len(t) == 240 == len({r["run_name"] for r in t})
    assert t[18]["run_name"] == "D_C19M16_counts_real_s0"
    assert t[66]["run_name"] == "DEDl0_C19M16_counts_real_s0"
    assert {r["rung"] for r in t[:96]} == {"D", "DEDl0"}
    assert {r["rung"] for r in t[96:]} == {"DEDl0.1", "DEDl1", "DEDl10"}
    assert tp.LAMBDAS == {"DEDl0": 0.0, "DEDl0.1": 0.1, "DEDl1": 1.0, "DEDl10": 10.0}
    assert len(tp.tasks()) == 144                            # the rung-A table is unchanged


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t126dsynth")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed}


@pytest.mark.parametrize("model_name,lam", [("real", 0.0), ("nocov", 1.0)])
def test_train_score_law_with_holdout(prod, model_name, lam):
    runs = prod["root"] / f"runs_{model_name}"
    arm = "DEDl0" if lam == 0 else "DEDl1"
    run_dir = train_ded.run_training_ded(prod["manifest"], prod["cov"], prod["root"] / "products",
                                         runs, arm, "T1", model_name, 0, lam, holdout=tp.HOLDOUT,
                                         max_steps=6, eval_every=3, device="cpu", window=256,
                                         batch=4, knot_sample=20_000, knot_block=256)
    assert run_dir.name == f"{arm}_T1_counts_{model_name}_s0"
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["rung"] == "DED" and cfg["lam"] == lam and cfg["holdout"] == [["depth", "7.5M"]]
    assert all("7.5M" not in p["source_pid"] + p["target_pid"] for p in cfg["train_pairs"])
    assert len(cfg["heldout_pairs"]) == 2 and "T1__depth__7.5M" not in cfg["fit_pids"]
    log = (run_dir / "train_log.tsv").read_text().splitlines()
    assert log[0].split("\t") == ["step", "loss", "mse_z", "val_nll", "seconds"]
    pred = model.load_run(run_dir, prod["root"] / "products", prod["cov"], prod["manifest"],
                          device="cpu")
    rows = pairs.read_manifest(prod["manifest"])
    score.score_trained(pred, rows, prod["bed"], run_dir)
    recs = json.loads((run_dir / "scores.json").read_text())["records"]
    assert {(r["source_pid"], r["target_pid"]) for r in recs if r["kind"] == "heldout"} == {
        (p["source_pid"], p["target_pid"]) for p in cfg["heldout_pairs"]}
    score.score_law(pred, rows, prod["bed"], run_dir)
    assert (run_dir / "LAW_DONE").is_file()
