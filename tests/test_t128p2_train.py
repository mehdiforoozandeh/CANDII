"""t128 phase 2 training (`tools/t128/train_ds.py` additions, `tools/t128p2/train.py`).

  * loss L12A: total = L1 + L2 + anchor; the anchor is the NB NLL of the base counts given the
    arm's encoder head (by hand from `encode_nb` and the pair's direction); `src_is_base` needed;
    the middle is log1p of the head's mean;
  * a DSfixed run keeps C_clean frozen at the base's encoded knobs; a DSanchor run writes a
    t126-shaped run dir with config, train_terms.tsv, and is scored by `score_trained` / `score_law`;
  * `max_steps` comes from the setting when present; phase-1 settings keep DEFAULTS' 4000;
  * `train.py train-index` trains a DS arm with `run_training_ds` and arm D with `run_training`.
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
from ladder import train as ltrain  # noqa: E402
from ladder.fforms import form_ds  # noqa: E402
from t128 import pairs as tp1  # noqa: E402
from t128 import train_ds  # noqa: E402
from t128p2 import pairs as tp  # noqa: E402
from t128p2 import train as tcli  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only",
                                        "ignore:.*slow, meant for synthetic data")
STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
N = encoding.N_COV
W = 64
C = form_ds.CONTEXT
CTR = slice(30, -30)
SMALL = dict(max_steps=6, eval_every=3, device="cpu", window=256, batch=4, knot_sample=20_000,
             knot_block=256)
TINY = {k: v for k, v in SMALL.items() if k not in ("max_steps", "eval_every", "device")}


def _ladder(rung="DSANC", seed=0) -> model.Ladder:
    torch.manual_seed(seed)
    lad = model.Ladder(rung, "counts", STATS)
    with torch.no_grad():
        lad.g.mlp[-1].weight.normal_(0.0, 0.3)
        lad.g.mlp[-1].bias.normal_(0.5, 0.3)
        lad.form.f.head.weight.normal_(0.0, 0.3)
        lad.form.f.head.bias.normal_(0.0, 0.3)
        lad.g.cclean.normal_(0.0, 1.0)
    return lad


def _batch(k=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    cov = torch.randn(k, 2 * N, generator=gen)
    for off in (0, N):
        cov[:, off + form_ds.N_KNOB: off + N] = 0.0
        cov[:, off + form_ds.N_KNOB + 1] = 1.0
    cnt = lambda: torch.poisson(torch.full((k, W + 2 * C), 3.0), generator=gen)  # noqa: E731
    xs_c, xt_c = cnt(), cnt()
    mask = torch.ones(k, W, dtype=torch.bool)
    mask[0, W // 2:] = False
    return cov, torch.log1p(xs_c), torch.log1p(xt_c), xs_c[:, C:-C], xt_c[:, C:-C], mask


def _thetas(lad, cov):
    c, ct = cov[:, :N], cov[:, N:]
    cc = lad.g.cclean_of(c)
    return lad.g.film(c, cc), lad.g.film(ct, cc)


def test_constants():
    assert train_ds.LOSSES == ("L12", "L2", "L123", "L12A")
    assert train_ds.CCLEAN_FORMS == ("A", "D", "F")
    assert train_ds.VAL_RULES == ("L1+L2 NLL", "L2 NLL", "L1+L2+anchor NLL")


def test_l12a_total_and_anchor_by_hand():
    lad = _ladder("DSANC", 1)
    cov, xs, xt, ys, yt, mask = _batch()
    sib = torch.tensor([True, False, True])
    s = tp.SETTING_BY_ARM["DSanchor"]
    total, t = train_ds.sym_loss(lad, cov, xs, xt, ys, yt, mask, s, 500, src_is_base=sib)
    assert torch.allclose(total, t["L1"] + t["L2"] + t["anchor"], rtol=0, atol=1e-6)
    assert float(t["kl"]) == 0.0 and float(t["mse_z"]) > 0.0
    te, tet = _thetas(lad, cov)
    loc_s, disp_s = lad.form.encode_nb(xs, te)
    loc_t, disp_t = lad.form.encode_nb(xt, tet)
    m = mask.double()
    want = 0.0
    for b in range(3):
        if sib[b]:      # base -> arm: the arm (target) head predicts the base (source) counts
            per = model.nb_nll(loc_t[b, CTR], disp_t[b, CTR], ys[b])
        else:           # arm -> base: the arm (source) head predicts the base (target) counts
            per = model.nb_nll(loc_s[b, CTR], disp_s[b, CTR], yt[b])
        want += float((per.double() * m[b]).sum())
    want /= float(m.sum())
    assert float(t["anchor"]) == pytest.approx(want, rel=1e-5)
    # the other direction gives another value: the direction is read
    _, t2 = train_ds.sym_loss(lad, cov, xs, xt, ys, yt, mask, s, 500, src_is_base=~sib)
    assert float(t2["anchor"]) != pytest.approx(float(t["anchor"]), rel=1e-6)


def test_l12a_needs_src_is_base_and_other_losses_log_zero_anchor():
    lad = _ladder("DSANC", 2)
    with pytest.raises(ValueError):
        train_ds.sym_loss(lad, *_batch(), tp.SETTING_BY_ARM["DSanchor"], 1)
    for arm in ("DSfree", "DSfixed"):
        total, t = train_ds.sym_loss(_ladder("DS", 2), *_batch(), tp.SETTING_BY_ARM[arm], 1,
                                     src_is_base=torch.tensor([True, False, True]))
        assert float(t["anchor"]) == 0.0
        assert torch.allclose(total, t["L1"] + t["L2"], rtol=0, atol=1e-6)


def test_middle_is_log1p_of_the_head_mean():
    lad = _ladder("DSANC", 3)
    cov, xs, *_ = _batch()
    te, _ = _thetas(lad, cov)
    loc, disp = lad.form.encode_nb(xs, te)
    z, d = lad.form.encode(xs, te)
    assert torch.allclose(z, torch.log1p(torch.exp(loc)), rtol=1e-6, atol=1e-6)
    assert torch.equal(d, disp)
    assert (model.MU_MIN, model.MU_MAX) == (1e-4, 1e6)


@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t128p2train")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed}


def test_validate_own_anchor_rule(prod):
    from ladder import data
    rows = pairs.read_manifest(prod["manifest"])
    tpairs = pairs.train_pairs(rows, "T1", tp.HOLDOUT)
    fit = pairs.fit_pids(rows, "T1", tp.HOLDOUT)
    enc = encoding.Encoder.fit(encoding.read_covariates(prod["cov"]), fit)
    cov = torch.from_numpy(np.stack([enc.encode_pair(p["source_pid"], p["target_pid"])
                                     for p in tpairs]))
    corpus = data.Corpus(prod["root"] / "products")
    lad = _ladder("DSANC", 4)
    l12 = train_ds.validate_own(lad, corpus, tpairs, cov, "cpu", "L1+L2 NLL")
    a1 = train_ds.validate_own(lad, corpus, tpairs, cov, "cpu", "L1+L2+anchor NLL")
    a2 = train_ds.validate_own(lad, corpus, tpairs, cov, "cpu", "L1+L2+anchor NLL")
    assert np.isfinite(a1) and a1 > l12 and a1 == a2


@pytest.fixture(scope="module")
def runs(prod):
    return {arm: train_ds.run_training_ds(
        prod["manifest"], prod["cov"], prod["root"] / "products", prod["root"] / "runs",
        tp.SETTING_BY_ARM[arm], "T1", "real", 0, holdout=tp.HOLDOUT, **SMALL)
        for arm in ("DSfixed", "DSanchor")}


def test_dsfixed_cclean_stays_at_the_base(prod, runs):
    run_dir = runs["DSfixed"]
    cfg = json.loads((run_dir / "config.json").read_text())
    cc = json.loads((run_dir / "cclean.json").read_text())
    assert cfg["cclean_frozen"] is True and cc["frozen"] is True and cc["box"] is False
    init = np.array([cfg["cclean_init"][a] for a in encoding.ASSAYS])
    after = np.array([cc["assays"][a] for a in encoding.ASSAYS])
    assert np.allclose(after, init, atol=1e-6)
    enc = encoding.Encoder.fit(encoding.read_covariates(prod["cov"]), cfg["fit_pids"])
    for track in ("T1",):
        a = encoding.ASSAYS.index(synth.TRACK_SPECS[track]["assay"])
        assert np.allclose(init[a], enc.encode(f"{track}__base__base")[:13], atol=1e-6)
    ck = torch.load(run_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert np.allclose(ck["g"]["cclean"].double().numpy(), init, atol=1e-6)
    assert "anchor" not in cfg and "middle" not in cfg


def test_dsanchor_run_dir(prod, runs):
    run_dir = runs["DSanchor"]
    assert run_dir.name == "DSanchor_T1_counts_real_s0"
    for f in ("config.json", "ckpt.pt", "train_log.tsv", "train_terms.tsv", "timing.json",
              "TRAIN_DONE", "cclean.json"):
        assert (run_dir / f).is_file()
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["rung"] == "DSANC" and cfg["max_steps"] == 6 and cfg["cclean_frozen"] is True
    assert cfg["holdout"] == [["depth", "7.5M"]]
    assert cfg["val_nll_rule"] == "L1+L2+anchor NLL" and cfg["loss"].startswith("L12A: L1 + L2 + L_anchor")
    assert cfg["anchor"] == "NLL(X_base | encode_nb(x_arm; g(c_arm, C_clean)))[centre bins]"
    assert cfg["middle"] == "log1p(clamp(exp(loc_e), 1e-4, 1e6))"
    log = (run_dir / "train_log.tsv").read_text().splitlines()
    assert log[0] == train_ds.LOG_HEADER
    terms = (run_dir / "train_terms.tsv").read_text().splitlines()
    assert terms[0] == "step\tL1\tL2\tanchor\tmse_z\tkl"
    rows = [r.split("\t") for r in terms[1:]]
    assert [int(r[0]) for r in rows] == [0, 3, 6]
    assert all(np.isfinite(float(r[3])) and float(r[3]) > 0.0 for r in rows[1:])
    pred = model.load_run(run_dir, prod["root"] / "products", prod["cov"], prod["manifest"],
                          device="cpu")
    mrows = pairs.read_manifest(prod["manifest"])
    score.score_trained(pred, mrows, prod["bed"], run_dir)
    recs = json.loads((run_dir / "scores.json").read_text())["records"]
    assert {(r["source_pid"], r["target_pid"]) for r in recs if r["kind"] == "heldout"} == {
        (p["source_pid"], p["target_pid"]) for p in cfg["heldout_pairs"]}
    score.score_law(pred, mrows, prod["bed"], run_dir, split=True)
    assert (run_dir / "LAW_DONE").is_file()


def test_max_steps_from_the_setting(prod, tmp_path):
    s = dict(tp.SETTING_BY_ARM["DSfixed"], max_steps=7)
    run_dir = train_ds.run_training_ds(prod["manifest"], prod["cov"], prod["root"] / "products",
                                       tmp_path, s, "T1", "real", 0, holdout=tp.HOLDOUT,
                                       eval_every=7, device="cpu", **TINY)
    assert json.loads((run_dir / "timing.json").read_text())["steps_run"] == 7
    assert json.loads((run_dir / "config.json").read_text())["max_steps"] == 7


def test_phase1_setting_keeps_defaults_max_steps(prod, tmp_path, monkeypatch):
    assert "max_steps" not in tp1.SETTINGS[0] and ltrain.DEFAULTS["max_steps"] == 4000
    monkeypatch.setitem(ltrain.DEFAULTS, "max_steps", 2)
    run_dir = train_ds.run_training_ds(prod["manifest"], prod["cov"], prod["root"] / "products",
                                       tmp_path, tp1.SETTINGS[0], "T1", "real", 0,
                                       holdout=tp1.HOLDOUT, eval_every=2, device="cpu", **TINY)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["max_steps"] == 2 and cfg["cclean_frozen"] is False   # DEFAULTS', not a setting's
    assert json.loads((run_dir / "timing.json").read_text())["steps_run"] == 2


def test_train_index_cli(prod, tmp_path, monkeypatch):
    rows4 = [dict(t, g="T1", run_name=t["run_name"].replace(t["g"], "T1"))
             for t in (tp.tasks()[i] for i in (9, 33, 57, 81))]
    monkeypatch.setattr(tp, "tasks", lambda rows=None: rows4)
    for k, v in TINY.items():
        monkeypatch.setitem(ltrain.DEFAULTS, k, v)
    args = [str(prod["manifest"]), str(prod["cov"]), str(prod["root"] / "products"),
            str(tmp_path)]
    opts = ["--max-steps", "2", "--eval-every", "1", "--device", "cpu"]
    assert tcli.main(["train-index", *args, "1", *opts]) == 0
    assert tcli.main(["train-index", *args, "3", *opts]) == 0
    for name in ("DSfixed_T1_counts_real_s0", "D_T1_counts_real_s0"):
        assert (tmp_path / name / "TRAIN_DONE").is_file()
    cfg = json.loads((tmp_path / "D_T1_counts_real_s0" / "config.json").read_text())
    assert cfg["rung"] == "D" and cfg["max_steps"] == 2 and cfg["holdout"] == [["depth", "7.5M"]]
    with pytest.raises(IndexError):
        tcli.main(["train-index", *args, "4"])


def test_train_index_cli_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(train_ds, "run_training_ds", lambda *a, **k: calls.append(("ds", a, k)))
    monkeypatch.setattr(ltrain, "run_training", lambda *a, **k: calls.append(("d", a, k)))
    for i in (57, 81):
        assert tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), str(i)]) == 0
    (k1, a1, kw1), (k2, a2, kw2) = calls
    assert k1 == "ds" and a1[4] == tp.SETTING_BY_ARM["DSanchor"] and a1[5:] == ("C19M16", "real", 0)
    assert kw1["max_steps"] == 12000 and kw1["holdout"] == tp.HOLDOUT
    assert k2 == "d" and a2[4:] == ("D", "C19M16", "counts", "real", 0)
    assert kw2["max_steps"] == 12000 and kw2["holdout"] == tp.HOLDOUT
    with pytest.raises(IndexError):
        tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), "96"])
