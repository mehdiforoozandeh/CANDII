"""t126 — the encoder-decoder rung A (plan/T126_ENCDEC_A.md): g's algebra, the 7.5M hold-out, the
task table, and one short train + score on synthetic products.

  * identity at c' = c, for any phi / psi weights (beta learned and beta = 1);
  * composition: the map a -> b equals a -> base followed by base -> b;
  * beta = 1 gives b = 1 exactly and a = alpha(c') - alpha(c);
  * at initialisation theta = rung A's init_theta for every covariate pair;
  * the hold-out removes every 7.5M pair from training and keeps it in scoring (heldout records,
    the law pairs), and with no hold-out every t118 function is unchanged.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import encoding, model, pairs, score, synth, train  # noqa: E402
from t126 import pairs as tp  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
STATS = {"knots_x": np.linspace(0, 5, 12), "n0": 5.0, "sigma0": 0.5}
N = encoding.N_COV


def _ladder(rung: str, seed: int = 0) -> model.Ladder:
    """A Ladder whose g has random (non-zero) last layers, so the algebra is not trivially 0."""
    torch.manual_seed(seed)
    lad = model.Ladder(rung, "counts", STATS)
    with torch.no_grad():
        for net in (lad.g.phi, lad.g.psi):
            net[-1].weight.normal_(0.0, 0.5)
            net[-1].bias.normal_(0.0, 0.5)
    return lad


def _covs(k: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(k, N, generator=torch.Generator().manual_seed(seed))


def _pair(c, ct):
    return torch.cat([c, ct], dim=1)


@pytest.mark.parametrize("rung", ["AED", "AED1"])
def test_identity_at_same_covariates(rung):
    lad = _ladder(rung)
    c = _covs(16)
    th = lad.theta(_pair(c, c))
    assert torch.allclose(th[:, 0], torch.zeros(16), atol=1e-6)
    assert torch.allclose(th[:, 1], torch.ones(16), atol=1e-6)
    x = torch.rand(16, 50) * 5
    loc, _ = lad(_pair(c, c), x)
    assert torch.allclose(loc, x, atol=1e-5)


@pytest.mark.parametrize("rung", ["AED", "AED1"])
def test_composition_through_base(rung):
    lad = _ladder(rung, seed=3)
    ca, cbase, cb = _covs(8, 1), _covs(8, 2), _covs(8, 3)
    x = torch.rand(8, 40) * 6
    th_ab = lad.theta(_pair(ca, cb))
    th_a0 = lad.theta(_pair(ca, cbase))
    th_0b = lad.theta(_pair(cbase, cb))
    direct = th_ab[:, :1] + th_ab[:, 1:2] * x
    via = th_0b[:, :1] + th_0b[:, 1:2] * (th_a0[:, :1] + th_a0[:, 1:2] * x)
    assert torch.allclose(direct, via, atol=1e-4)
    # composed (a, b) equal the direct ones; d depends on the target only
    b = th_0b[:, 1] * th_a0[:, 1]
    a = th_0b[:, 0] + th_0b[:, 1] * th_a0[:, 0]
    assert torch.allclose(th_ab[:, 1], b, atol=1e-5)
    assert torch.allclose(th_ab[:, 0], a, atol=1e-4)
    assert torch.allclose(th_ab[:, 2], th_0b[:, 2], atol=1e-6)


def test_beta_one_is_a_pure_shift():
    lad = _ladder("AED1", seed=5)
    assert lad.g.phi[-1].out_features == 1
    c, ct = _covs(32, 1), _covs(32, 2)
    th = lad.theta(_pair(c, ct))
    assert torch.equal(th[:, 1], torch.ones(32))
    alpha, _ = lad.g.latent(c)
    alpha_t, _ = lad.g.latent(ct)
    assert torch.allclose(th[:, 0], alpha_t - alpha, atol=1e-6)


def test_beta_learned_is_positive_ratio():
    lad = _ladder("AED", seed=6)
    c, ct = _covs(32, 1), _covs(32, 2)
    th = lad.theta(_pair(c, ct))
    _, lb = lad.g.latent(c)
    _, lbt = lad.g.latent(ct)
    assert (th[:, 1] > 0).all()
    assert torch.allclose(th[:, 1], torch.exp(lbt - lb), atol=1e-6)
    assert not torch.allclose(th[:, 1], torch.ones(32))


@pytest.mark.parametrize("rung", ["AED", "AED1"])
def test_init_is_rung_a_identity(rung):
    lad = model.Ladder(rung, "counts", STATS)
    th = lad.theta(_pair(_covs(10, 1), _covs(10, 2)))
    want = lad.form.init_theta()
    assert torch.allclose(th, want.expand(10, -1), atol=0)
    assert torch.allclose(want, torch.tensor([0.0, 1.0, math.log(5.0)]))


def test_rung_a_g_unchanged():
    lad = model.Ladder("A", "counts", STATS)
    assert type(lad.g) is model.G


# ---- hold-out and task table on the real manifest -------------------------------------------

def test_holdout_on_the_t112_manifest():
    rows = pairs.read_manifest(FIXTURE_MANIFEST)
    for g in pairs.G_IDS:
        full = pairs.train_pairs(rows, g)
        assert pairs.train_pairs(rows, g, ()) == full
        assert pairs.fit_pids(rows, g, ()) == pairs.fit_pids(rows, g)
        kept = tp.train_pairs(rows, g)
        held = tp.heldout_pairs(rows, g)
        n_tracks = len(pairs.TRACKS) if g == "all" else 1
        assert len(held) == 2 * n_tracks
        assert len(kept) + len(held) == len(full)
        for p in kept:
            assert "7.5M" not in (p["level_src"], p["level_tgt"])
            assert "__depth__7.5M" not in p["source_pid"] + p["target_pid"]
        for p in held:
            assert {p["arm_src"], p["arm_tgt"]} == {"base", "depth"}
            assert "7.5M" in (p["level_src"], p["level_tgt"])
        fit = pairs.fit_pids(rows, g, tp.HOLDOUT)
        assert not [pid for pid in fit if pid.endswith("__depth__7.5M")]
        # 7.5M arm->arm pairs stay in the law pass (never trained, scored)
        law = pairs.law_pairs(rows, g)
        tracks = pairs.TRACKS if g == "all" else (g,)
        want = sum(2 * (len(pairs._by_track(rows, t)[1]) - 1) for t in tracks)
        assert sum("7.5M" in (p["level_src"], p["level_tgt"]) for p in law) == want


def test_task_table():
    t = tp.tasks()
    assert len(t) == 144 == len({r["run_name"] for r in t})
    assert {r["space"] for r in t} == {"counts"}
    assert t[18]["run_name"] == "A_C19M16_counts_real_s0"
    assert t[66]["run_name"] == "AED_C19M16_counts_real_s0"
    assert t[114]["run_name"] == "AED1_C19M16_counts_real_s0"
    assert t[45]["run_name"] == "A_all_counts_nocov_s0"


# ---- one short train + score on synthetic products ------------------------------------------

@pytest.fixture(scope="module")
def prod(tmp_path_factory):
    root = tmp_path_factory.mktemp("t126synth")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed}


@pytest.mark.parametrize("rung,model_name", [("AED", "real"), ("AED1", "nocov"), ("A", "real")])
def test_train_and_score_with_holdout(prod, rung, model_name):
    runs = prod["root"] / f"runs_{rung}_{model_name}"
    run_dir = train.run_training(prod["manifest"], prod["cov"], prod["root"] / "products", runs,
                                 rung, "T1", "counts", model_name, 0, max_steps=12, eval_every=6,
                                 device="cpu", holdout=tp.HOLDOUT, window=256, batch=8,
                                 knot_sample=20_000, knot_block=256)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["holdout"] == [["depth", "7.5M"]]
    tr = {(p["source_pid"], p["target_pid"]) for p in cfg["train_pairs"]}
    ho = {(p["source_pid"], p["target_pid"]) for p in cfg["heldout_pairs"]}
    assert ho == {("T1__base__base", "T1__depth__7.5M"), ("T1__depth__7.5M", "T1__base__base")}
    assert not tr & ho and all("7.5M" not in s + t for s, t in tr)
    assert "T1__depth__7.5M" not in cfg["fit_pids"]
    pred = model.load_run(run_dir, prod["root"] / "products", prod["cov"], prod["manifest"],
                          device="cpu")
    rows = pairs.read_manifest(prod["manifest"])
    score.score_trained(pred, rows, prod["bed"], run_dir)
    recs = json.loads((run_dir / "scores.json").read_text())["records"]
    held = [r for r in recs if r["kind"] == "heldout"]
    assert {(r["source_pid"], r["target_pid"]) for r in held} == ho
    assert all(r["eval"] == "score" and r["crps_all"] is not None for r in held)
    dl = [r for r in recs if r["kind"] == "depthlaw" and r["depth_source"] == "heldout"]
    assert len(dl) == 2
    trained = {(r["source_pid"], r["target_pid"]) for r in recs if r["kind"] == "trained"}
    assert trained == tr
    swaps = {r["source_pid"] for r in recs if r["kind"] == "swap"}
    assert "T1__depth__7.5M" not in swaps
    score.score_law(pred, rows, prod["bed"], run_dir)
    law = json.loads((run_dir / "law.json").read_text())["records"]
    assert any("7.5M" in r["source_pid"] or "7.5M" in r["target_pid"] for r in law)


def test_no_holdout_config_unchanged(prod):
    runs = prod["root"] / "runs_noholdout"
    run_dir = train.run_training(prod["manifest"], prod["cov"], prod["root"] / "products", runs,
                                 "A", "T1", "counts", "real", 0, max_steps=2, eval_every=2,
                                 device="cpu", window=256, batch=8, knot_sample=20_000,
                                 knot_block=256)
    cfg = json.loads((run_dir / "config.json").read_text())
    assert "holdout" not in cfg and "heldout_pairs" not in cfg
    assert len(cfg["train_pairs"]) == len(pairs.train_pairs(pairs.read_manifest(prod["manifest"]),
                                                            "T1"))
