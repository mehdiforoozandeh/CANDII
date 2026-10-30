"""t136 entry point (`tools/t136/train.py train-index`) and `run_training(..., extra_pairs=...)`.

  * a task row maps onto `ladder.train.run_training` with form rung D, phase 2's D sizes, the arm
    name, 12,000 steps, the depth-7.5M hold-out and the arm's extra arm->arm pairs;
  * end to end on synthetic products: Darm1 and Darm2 train small, config.json records the arm and
    the extra pairs (appended after the base pairs), the run scores (trained; law with the CRPS
    split, whose records contain the extra pairs the readings subtract);
  * the default path: no `extra_pairs` gives `config["extra_pairs"] == []` and a checkpoint
    bit-equal to `extra_pairs=[]` and to the golden captured at 7004fa4 before the edit;
  * an extra pair touching a held-out product raises ValueError.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import model, pairs, score, synth  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from t136 import pairs as tp  # noqa: E402
from t136 import train as tcli  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only",
                                        "ignore:.*slow, meant for synthetic data")
SMALL = dict(window=256, batch=4, knot_sample=20_000, knot_block=256)
SPLIT_KEYS = {"crps_oracle_scaled_all", "scale_error_all", "c_star_all"}
#: synthetic products have no depth 3.75M arm: Darm1's endpoints become 15M and pe here
SYN_ENDPOINTS = (("depth", "15M"), ("pe", "pe"))
#: sha256 of a default run at HEAD 7004fa4 BEFORE the extra_pairs edit (ckpt g + form tensors and
#: the step / loss / val columns of train_log.tsv); `_golden_run` below, same settings
GOLDEN_SHA = "97854b186d58f07fa8f737d171da352f54b3ab22d9d06e12d65afd037ac22516"
GOLDEN_ENV = ("2.6.0", "Darwin", "arm64")


def _key(p):
    return (p["source_pid"], p["target_pid"])


def test_train_index_cli_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ltrain, "run_training", lambda *a, **k: calls.append((a, k)))
    rows = pairs.read_manifest(REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv")
    monkeypatch.setattr(tp._lp, "read_manifest", lambda path: rows)
    for i in tp.SMOKE:
        assert tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), str(i)]) == 0
    for (a, k), arm in zip(calls, tp.ARMS):
        assert a == ("m.tsv", "c.tsv", "d", str(tmp_path), "D", "C19M16", "counts", "real", 0)
        extra = k.pop("extra_pairs")
        assert k == {"max_steps": 12000, "eval_every": None, "device": "auto",
                     "holdout": tp.HOLDOUT, "g_hidden": 64, "f_channels": 32, "arm": arm}
        assert extra == tp.extra_pairs(rows, "C19M16", arm)
    calls.clear()
    assert tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), "41",
                      "--max-steps", "300", "--eval-every", "100"]) == 0
    (a, k), = calls
    assert a[5] == tp.G_IDS[-1] and a[8] == 2 and k["max_steps"] == 300 and k["arm"] == "Darm2"
    for bad in ("42", "-1"):
        with pytest.raises(IndexError):
            tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), bad])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("t136train")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    rows2 = [dict(t, g="T1", run_name=t["run_name"].replace(t["g"], "T1"))
             for t in (tp.tasks()[i] for i in (0, 21))]
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(tp, "tasks", lambda rows=None: rows2)
        mp.setattr(tp, "DARM1_ENDPOINTS", SYN_ENDPOINTS)
        for k, v in SMALL.items():
            mp.setitem(ltrain.DEFAULTS, k, v)
        args = [str(manifest), str(cov), str(root / "products"), str(root / "runs")]
        opts = ["--max-steps", "4", "--eval-every", "2", "--device", "cpu"]
        for i in ("0", "1"):
            assert tcli.main(["train-index", *args, i, *opts]) == 0
        rows = pairs.read_manifest(manifest)
        extras = {a: tp.extra_pairs(rows, "T1", a) for a in tp.ARMS}
    finally:
        mp.undo()
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed, "rows": rows,
            "data": root / "products", "extras": extras,
            "Darm1": root / "runs" / "Darm1_T1_counts_real_s0",
            "Darm2": root / "runs" / "Darm2_T1_counts_real_s0"}


@pytest.mark.parametrize("arm", tp.ARMS)
def test_run_dir_config(runs, arm):
    run_dir = runs[arm]
    assert (run_dir / "TRAIN_DONE").is_file()
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["rung"] == "D" and cfg["arm"] == arm and cfg["max_steps"] == 4
    assert (cfg["g_hidden"], cfg["f_channels"], cfg["n_theta"]) == (64, 32, 256)
    assert cfg["holdout"] == [["depth", "7.5M"]]
    extra = cfg["extra_pairs"]
    assert extra == runs["extras"][arm] and all(p["direction"] == "arm_to_arm" for p in extra)
    if arm == "Darm1":
        assert len(extra) == 2
        assert {_key(p) for p in extra} == {("T1__depth__15M", "T1__pe__pe"),
                                            ("T1__pe__pe", "T1__depth__15M")}
    base = pairs.train_pairs(runs["rows"], "T1", [["depth", "7.5M"]])
    assert cfg["train_pairs"] == base + extra
    held = {r["pid"] for r in runs["rows"] if r["arm"] == "depth" and r["level"] == "7.5M"}
    assert held and not any({p["source_pid"], p["target_pid"]} & held for p in cfg["train_pairs"])
    assert cfg["fit_pids"] == pairs.fit_pids(runs["rows"], "T1", [["depth", "7.5M"]])


def test_darm1_scores_trained_and_law(runs):
    run_dir = runs["Darm1"]
    pred = model.load_run(run_dir, runs["data"], runs["cov"], runs["manifest"], device="cpu")
    score.score_trained(pred, runs["rows"], runs["bed"], run_dir)
    assert (run_dir / "SCORE_DONE").is_file()
    recs = json.loads((run_dir / "scores.json").read_text())["records"]
    assert any(r["kind"] == "trained" for r in recs) and any(r["kind"] == "heldout" for r in recs)
    score.score_law(pred, runs["rows"], runs["bed"], run_dir, split=True)
    law = [r for r in json.loads((run_dir / "law.json").read_text())["records"]
           if r["kind"] == "law"]
    assert law and all(SPLIT_KEYS <= set(r) for r in law)
    assert {_key(p) for p in runs["extras"]["Darm1"]} <= {_key(r) for r in law}


# ---- the default path and the guard ---------------------------------------------------------

def _golden_run(root, **kw):
    m, c = synth.make_products(root / "products", 0)
    rd = ltrain.run_training(m, c, root / "products", root / "runs", "D", "T1", "counts", "real",
                             0, max_steps=4, eval_every=2, device="cpu",
                             holdout=(("depth", "7.5M"),), g_hidden=64, f_channels=32,
                             **SMALL, **kw)
    ck = torch.load(rd / "ckpt.pt", weights_only=False)
    h = hashlib.sha256()
    for part in ("g", "form"):
        for k in sorted(ck[part]):
            h.update(k.encode())
            h.update(ck[part][k].detach().cpu().contiguous().numpy().tobytes())
    log = [ln.split("\t") for ln in (rd / "train_log.tsv").read_text().splitlines()[1:]]
    h.update(json.dumps([[r[0], r[1], r[2]] for r in log]).encode())
    return rd, h.hexdigest()


def test_default_path_is_bit_exact(tmp_path):
    rd, sha = _golden_run(tmp_path / "a")
    cfg = json.loads((rd / "config.json").read_text())
    assert cfg["extra_pairs"] == [] and "arm" not in cfg
    _, sha_empty = _golden_run(tmp_path / "b", extra_pairs=[])
    assert sha_empty == sha
    if (torch.__version__.split("+")[0], platform.system(), platform.machine()) == GOLDEN_ENV:
        assert sha == GOLDEN_SHA
    else:
        pytest.skip(f"golden captured on torch/os/arch {GOLDEN_ENV}; equality with [] checked")


def test_extra_pair_with_a_heldout_product_raises(tmp_path):
    m, c = synth.make_products(tmp_path / "products", 0)
    rows = pairs.read_manifest(m)
    bad = [p for p in pairs.law_pairs(rows, "T1") if p["source_pid"] == "T1__depth__7.5M"][:1]
    assert bad
    with pytest.raises(ValueError, match="held-out"):
        ltrain.run_training(m, c, tmp_path / "products", tmp_path / "runs", "D", "T1", "counts",
                            "real", 0, max_steps=2, eval_every=2, device="cpu",
                            holdout=(("depth", "7.5M"),), extra_pairs=bad, **SMALL)
