"""t128 phase 3 entry point (`tools/t128p3/train.py train-index`).

  * a task row maps onto `ladder.train.run_training` with form rung D, the arm's sizes, the arm
    name, 15,000 steps and the depth-7.5M hold-out; an index out of range raises IndexError;
  * end to end on synthetic products: both arms train small, config.json records the sizes and the
    arm, `model.load_run` rebuilds each at its size, the Dbigf run scores (trained, law with the
    CRPS split) and reads through the base (`t128.viabase.run`).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import model, pairs, score, synth  # noqa: E402
from ladder import train as ltrain  # noqa: E402
from t128 import viabase  # noqa: E402
from t128p3 import pairs as tp  # noqa: E402
from t128p3 import train as tcli  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:.*meant for synthetic data only",
                                        "ignore:.*slow, meant for synthetic data")
SMALL = dict(window=256, batch=4, knot_sample=20_000, knot_block=256)
SPLIT_KEYS = {"crps_oracle_scaled_all", "scale_error_all", "c_star_all"}
SIZES = {"Dbigg": (256, 32, 256), "Dbigf": (64, 64, 512)}


def test_train_index_cli_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ltrain, "run_training", lambda *a, **k: calls.append((a, k)))
    for i in (9, 33):
        assert tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), str(i)]) == 0
    for (a, k), arm in zip(calls, ("Dbigg", "Dbigf")):
        s = tp.SETTING_BY_ARM[arm]
        assert a == ("m.tsv", "c.tsv", "d", str(tmp_path), "D", "C19M16", "counts", "real", 0)
        assert k == {"max_steps": 15000, "eval_every": None, "device": "auto",
                     "holdout": tp.HOLDOUT, "g_hidden": s["g_hidden"],
                     "f_channels": s["f_channels"], "arm": arm}
    with pytest.raises(IndexError):
        tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), "48"])
    with pytest.raises(IndexError):
        tcli.main(["train-index", "m.tsv", "c.tsv", "d", str(tmp_path), "-1"])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("t128p3train")
    manifest, cov = synth.make_products(root / "products", 0)
    bed = root / "blacklist.bed"
    bed.write_text("chr19\t0\t2500\n")
    rows2 = [dict(t, g="T1", run_name=t["run_name"].replace(t["g"], "T1"))
             for t in (tp.tasks()[i] for i in (9, 33))]
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(tp, "tasks", lambda rows=None: rows2)
        for k, v in SMALL.items():
            mp.setitem(ltrain.DEFAULTS, k, v)
        args = [str(manifest), str(cov), str(root / "products"), str(root / "runs")]
        opts = ["--max-steps", "4", "--eval-every", "2", "--device", "cpu"]
        for i in ("0", "1"):
            assert tcli.main(["train-index", *args, i, *opts]) == 0
        with pytest.raises(IndexError):
            tcli.main(["train-index", *args, "2", *opts])
    finally:
        mp.undo()
    return {"root": root, "manifest": manifest, "cov": cov, "bed": bed,
            "data": root / "products", "rows": pairs.read_manifest(manifest),
            "Dbigg": root / "runs" / "Dbigg_T1_counts_real_s0",
            "Dbigf": root / "runs" / "Dbigf_T1_counts_real_s0"}


@pytest.mark.parametrize("arm", ["Dbigg", "Dbigf"])
def test_run_dir_config(runs, arm):
    run_dir = runs[arm]
    assert (run_dir / "TRAIN_DONE").is_file()
    cfg = json.loads((run_dir / "config.json").read_text())
    assert cfg["rung"] == "D" and cfg["arm"] == arm and cfg["max_steps"] == 4
    assert (cfg["g_hidden"], cfg["f_channels"], cfg["n_theta"]) == SIZES[arm]
    assert cfg["holdout"] == [["depth", "7.5M"]]
    held = {r["pid"] for r in runs["rows"] if r.get("arm") == "depth" and r.get("level") == "7.5M"}
    assert held
    assert not any({p["source_pid"], p["target_pid"]} & held for p in cfg["train_pairs"])


@pytest.mark.parametrize("arm", ["Dbigg", "Dbigf"])
def test_load_run_rebuilds_sizes(runs, arm):
    gh, fc, nt = SIZES[arm]
    pred = model.load_run(runs[arm], runs["data"], runs["cov"], runs["manifest"], device="cpu")
    assert pred.ladder.g.net[0].out_features == gh
    assert pred.ladder.form.channels == fc and pred.ladder.n_theta == nt


def test_dbigf_scores_and_reads_through_the_base(runs):
    run_dir = runs["Dbigf"]
    pred = model.load_run(run_dir, runs["data"], runs["cov"], runs["manifest"], device="cpu")
    score.score_trained(pred, runs["rows"], runs["bed"], run_dir)
    assert (run_dir / "scores.json").is_file()
    score.score_law(pred, runs["rows"], runs["bed"], run_dir, split=True)
    assert (run_dir / "LAW_DONE").is_file()
    law = [r for r in json.loads((run_dir / "law.json").read_text())["records"]
           if r["kind"] == "law"]
    assert law and all(SPLIT_KEYS <= set(r) for r in law)
    out = runs["root"] / "t128p3" / "viabase" / run_dir.name
    path = viabase.run(runs["manifest"], runs["cov"], runs["data"], runs["bed"], run_dir, out)
    assert path == out / "law.json" and (out / "LAW_DONE").is_file()
    doc = json.loads(path.read_text())
    vlaw = [r for r in doc["records"] if r["kind"] == "law"]
    assert len(vlaw) == len(pairs.law_pairs(runs["rows"], "T1"))
    assert all(SPLIT_KEYS <= set(r) for r in vlaw)
    assert json.loads((out / "viabase.json").read_text())["old_run_dir"] == str(run_dir)
