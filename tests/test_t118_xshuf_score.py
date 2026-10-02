"""t118 row 2, shuffled-bin twin (model "xshuf") — `score.py trained` and `score.py law` on a twin run.

Real 5-step CPU runs on `synth.make_products`, scored through the CLI (`score.main`), with the FAST
settings of `tests/test_t118_row2_score.py`. At scoring the twin's g reads x through one fixed
seeded permutation per chromosome (`xshuf.pred_rng`). The in-process regression: an A2 real run is
scored before any twin is trained or scored, and an unscored copy of it is scored after; the two
record sets must be equal (the row-2 real records do not move).
"""
from __future__ import annotations

import collections
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "t118"))

from ladder import model, pairs, score, synth, train, xshuf  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only")
FAST = {"window": 256, "batch": 16, "knot_sample": 20_000, "knot_block": 256}
G = "T1"
VOLATILE = ("created_utc", "timing")
PER_FORM = {"A2": {"a", "b", "disp"}, "D2": {"film_gamma_abs_dev", "film_beta_abs"}}


@pytest.fixture(scope="module")
def kit(tmp_path_factory):
    root = tmp_path_factory.mktemp("xshufscore")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\nchr21\t5000\t7500\nchr22\t0\t250\n")
    return {"root": root, "products": root / "products", "manifest": manifest, "cov": cov,
            "bed": bed, "rows": pairs.read_manifest(manifest)}


def _train(kit, rung, model_name, sub="runs"):
    return train.run_training(kit["manifest"], kit["cov"], kit["products"], kit["root"] / sub,
                              rung, G, "counts", model_name, 0, max_steps=5, eval_every=5,
                              device="cpu", lr=1e-2, **FAST)


def _score(kit, run_dir, cmd, workers=1):
    argv = [cmd, str(kit["manifest"]), str(kit["cov"]), str(kit["products"]), str(kit["bed"]),
            str(run_dir), "--workers", str(workers)]
    assert score.main(argv) == 0
    name = "scores.json" if cmd == "trained" else "law.json"
    return json.loads((run_dir / name).read_text())


def _score_both(kit, run_dir, workers=1):
    s = _score(kit, run_dir, "trained", workers)
    law = _score(kit, run_dir, "law", workers)
    with np.load(run_dir / "figdata.npz") as z:
        figs = {k: z[k] for k in z.files}
    return {"run_dir": run_dir, "scores": s, "law": law, "figs": figs}


def _kinds(recs):
    return collections.Counter((r["kind"], r["eval"]) for r in recs)


def _stable(doc):
    return {k: v for k, v in doc.items() if k not in VOLATILE}


def _load(kit, run_dir):
    return model.load_run(run_dir, kit["products"], kit["cov"], kit["manifest"], device="cpu")


def _copy(run_dir, dest):
    shutil.copytree(run_dir, dest)          # unscored copy: same name, same ckpt
    return dest


@pytest.fixture(scope="module")
def runs(kit):
    """A2 real scored first; then the A2 and D2 twins; then a copy of A2 real re-scored."""
    real = _train(kit, "A2", "real")
    real_after = _copy(real, kit["root"] / "rescore_real" / real.name)
    out = {"real_before": _score_both(kit, real)}
    for rung in ("A2", "D2"):
        rd = _train(kit, rung, pairs.MODEL_XSHUF)
        twin_copy = _copy(rd, kit["root"] / f"rescore_{rung}" / rd.name)
        out[rung] = _score_both(kit, rd)
        out[f"{rung}_again"] = _score_both(kit, twin_copy, workers=2)
    out["real_after"] = _score_both(kit, real_after)
    return out


# ---------------------------------------------------------------------------------------------


def test_row2_real_records_unchanged_by_twin(runs):
    b, a = runs["real_before"], runs["real_after"]
    assert _stable(a["scores"]) == _stable(b["scores"])
    assert _stable(a["law"]) == _stable(b["law"])
    assert sorted(a["figs"]) == sorted(b["figs"]) and b["figs"]
    for k in b["figs"]:
        assert np.array_equal(a["figs"][k], b["figs"][k]), k


@pytest.mark.parametrize("rung", ["A2", "D2"])
def test_twin_record_kinds_and_counts_match_real(kit, runs, rung):
    real, tw = runs["real_before"], runs[rung]
    assert _kinds(tw["scores"]["records"]) == _kinds(real["scores"]["records"])
    assert _kinds(tw["law"]["records"]) == _kinds(real["law"]["records"])
    kinds = {r["kind"] for r in tw["scores"]["records"]}
    assert {"trained", "shuffle", "swap"} <= kinds
    assert len([r for r in tw["scores"]["records"] if r["kind"] == "shuffle"]) \
        == len(pairs.train_pairs(kit["rows"], G))
    assert len(tw["law"]["records"]) == len(real["law"]["records"]) > 0
    assert tw["scores"]["skipped"] == real["scores"]["skipped"]
    for doc in (tw["scores"], tw["law"]):
        info = doc["run"]
        assert set(info) == set(real["scores"]["run"])
        assert info["model"] == "xshuf" and info["rung"] == rung and info["g"] == G
        assert info["run_name"] == f"{rung}_{G}_counts_xshuf_s0"
    cfg = json.loads((tw["run_dir"] / "config.json").read_text())
    assert cfg["model"] == "xshuf" and cfg["xshuf"]["rule"] == xshuf.RULE


@pytest.mark.parametrize("rung", ["A2", "D2"])
def test_twin_describe_crps_and_empty_figdata(runs, rung):
    tw = runs[rung]
    recs = tw["scores"]["records"] + tw["law"]["records"]
    described = [r for r in recs if "describe" in r]
    assert described and all(r["kind"] == "trained" for r in described)
    for r in described:
        d = r["describe"]
        assert set(d) == {"n_levels", "levels_x", "response"} | PER_FORM[rung]
        assert d["n_levels"] == 12 and len(d["levels_x"]) == 12
        assert set(d["response"]) == {"loc", "disp"}
        assert all(len(v) == 12 and all(math.isfinite(x) for x in v)
                   for v in d["response"].values())
    for r in recs:
        assert r["n_crps_nonfinite"] == 0
        for s in score.SUBSETS:
            assert r[f"crps_{s}"] is not None and math.isfinite(r[f"crps_{s}"]), (r["kind"], s)
    assert tw["scores"]["n_records_crps_nonfinite"] == 0
    assert tw["law"]["n_records_crps_nonfinite"] == 0
    swaps = [r for r in tw["scores"]["records"] if r["kind"] == "swap"]
    assert swaps and all(r[score.SWAP_KEY] is not None for r in swaps)
    assert (tw["run_dir"] / "SCORE_DONE").is_file() and (tw["run_dir"] / "LAW_DONE").is_file()
    assert tw["figs"] == {}                  # a twin: figdata is written empty
    assert tw["scores"]["snippets"] == []


@pytest.mark.parametrize("rung", ["A2", "D2"])
def test_twin_scoring_is_deterministic(runs, rung):
    """A copy of the twin scored again, with 2 forked workers, gives the same records."""
    a, b = runs[rung], runs[f"{rung}_again"]
    assert _stable(a["scores"]) == _stable(b["scores"])
    assert _stable(a["law"]) == _stable(b["law"])


def test_twin_predict_uses_the_fixed_permutation(kit, runs):
    rd = runs["A2"]["run_dir"]
    p1, p2 = _load(kit, rd), _load(kit, rd)
    assert p1.model == "xshuf"
    pr = pairs.train_pairs(kit["rows"], G)[0]
    src, tgt = pr["source_pid"], pr["target_pid"]
    l1, d1 = p1.predict(src, src, tgt, "chr19")
    l2, d2 = p2.predict(src, src, tgt, "chr19")
    assert np.array_equal(l1, l2) and np.array_equal(d1, d2)
    # predict is predict_chrom_bins with the run seed's permutation of chr19
    X = p1.corpus.get(src, p1.space, "chr19")
    cov, avg = p1._cov_t(src, tgt)
    assert not avg
    lr, dr = model.predict_chrom_bins(p1.ladder, X, cov, p1.device,
                                      x_g_rng=xshuf.pred_rng(p1.seed, "chr19"))
    assert np.array_equal(l1, lr) and np.array_equal(d1, dr)
    # another seed's permutation, and no permutation, give other predictions
    lo, _ = model.predict_chrom_bins(p1.ladder, X, cov, p1.device,
                                     x_g_rng=xshuf.pred_rng(p1.seed + 1, "chr19"))
    ln, _ = model.predict_chrom_bins(p1.ladder, X, cov, p1.device)
    assert not np.array_equal(l1, lo)
    assert not np.array_equal(l1, ln)
