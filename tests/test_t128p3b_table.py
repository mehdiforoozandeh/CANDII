"""t128p3b-B1 — the 42-row phase-3b task table, `train.py train-index-3b` and the two job scripts.

  * the table: 2 arms (Dbigf, D) x the 7 per-track g x seeds 3, 4, 5; anchors 0 / 21; the D run
    names equal phase 2's `D_<g>_counts_real_s<seed>`; phase 3's `tasks()` is unchanged;
  * `train-index-3b`: a row maps onto `ladder.train.run_training` with the arm's sizes and steps,
    arm=None for D;
  * the scripts: #SBATCH lines, the t128p3/seeds3b out-dir guard and DRY_RUN command lines.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import train as ltrain  # noqa: E402
from t128p3 import pairs as tp  # noqa: E402
from t128p3 import train as tcli  # noqa: E402

SLURM = REPO / "slurm" / "t128p3"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
SCRIPTS = {"train": SLURM / "train_3b.sh", "law": SLURM / "law_3b.sh"}
G0 = tp.G_IDS_3B[0]
ANCHORS = {0: f"Dbigf_{G0}_counts_real_s3", 21: f"D_{G0}_counts_real_s3"}
STEPS = {"Dbigf": 15000, "D": 12000}
SIZES = {"Dbigf": (64, 64, 512), "D": (64, 32, 256)}
OUT = "/project/def-maxwl/mforooz/t128p3/seeds3b"


# ---- the table -------------------------------------------------------------------------------

def test_constants_and_settings():
    assert tp.SEEDS_3B == (3, 4, 5)
    assert tp.G_IDS_3B == tp._lp.TRACKS and len(tp.G_IDS_3B) == 7 and "all" not in tp.G_IDS_3B
    assert [s["arm"] for s in tp.SETTINGS_3B] == ["Dbigf", "D"]
    for s in tp.SETTINGS_3B:
        assert (s["g_hidden"], s["f_channels"], s["n_theta"]) == SIZES[s["arm"]]
        assert s["max_steps"] == STEPS[s["arm"]] and s["rung"] == "D" and s["loss"] == "nb"
        assert s["n_theta"] == 2 * 4 * s["f_channels"]
    assert tp.SETTING_BY_ARM_3B["Dbigf"] == {**tp.SETTING_BY_ARM["Dbigf"],
                                              "number": tp.SETTING_BY_ARM_3B["Dbigf"]["number"]}


def test_task_table_3b():
    t = tp.tasks_3b()
    assert len(t) == 42 == len({r["run_name"] for r in t})
    assert [r["index"] for r in t] == list(range(42))
    assert all(list(r) == list(tp.TASK_KEYS) for r in t)
    assert {i: t[i]["run_name"] for i in ANCHORS} == ANCHORS
    assert not any(r["g"] == "all" for r in t)
    for ai, s in enumerate(tp.SETTINGS_3B):
        for gi, g in enumerate(tp.G_IDS_3B):
            for si, seed in enumerate(tp.SEEDS_3B):
                r = t[ai * 21 + gi * 3 + si]
                assert (r["rung"], r["g"], r["seed"]) == (s["arm"], g, seed)
                assert r["form_rung"] == "D" and r["model"] == "real" and r["space"] == "counts"
                assert (r["g_hidden"], r["f_channels"]) == SIZES[s["arm"]][:2]
                assert r["run_name"] == f"{s['arm']}_{g}_counts_real_s{seed}"
                assert r["run_name"] == ltrain.run_name(s["arm"], g, "counts", "real", seed)
    # no overlap with phase 3's runs or phase 2's reference runs (seeds 0-2)
    names = {r["run_name"] for r in t}
    assert not names & {r["run_name"] for r in tp.tasks()}
    assert not names & set(tp.REFERENCE_RUNS)


def test_phase3_table_unchanged():
    t = tp.tasks()
    assert len(t) == 48
    assert t[9]["run_name"] == "Dbigg_C19M16_counts_real_s0"
    assert t[33]["run_name"] == "Dbigf_C19M16_counts_real_s0"


def test_cli_tasks3b_tsv():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "t128p3" / "pairs.py"), "tasks3b",
                        str(FIXTURE_MANIFEST)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 43 and lines[0].split("\t") == list(tp.TASK_KEYS)
    assert [lines[1 + i].split("\t")[7] for i in ANCHORS] == list(ANCHORS.values())


# ---- train-index-3b --------------------------------------------------------------------------

def test_train_index_3b_arguments(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ltrain, "run_training", lambda *a, **k: calls.append((a, k)))
    for i in ANCHORS:
        assert tcli.main(["train-index-3b", "m.tsv", "c.tsv", "d", str(tmp_path), str(i)]) == 0
    for (a, k), arm in zip(calls, ("Dbigf", "D")):
        gh, fc, _ = SIZES[arm]
        assert a == ("m.tsv", "c.tsv", "d", str(tmp_path), "D", G0, "counts", "real", 3)
        assert k == {"max_steps": STEPS[arm], "eval_every": None, "device": "auto",
                     "holdout": tp.HOLDOUT, "g_hidden": gh, "f_channels": fc,
                     "arm": None if arm == "D" else arm}
    calls.clear()
    assert tcli.main(["train-index-3b", "m.tsv", "c.tsv", "d", str(tmp_path), "41",
                      "--max-steps", "300", "--eval-every", "100"]) == 0
    (a, k), = calls
    assert a[5] == tp.G_IDS_3B[-1] and a[8] == 5 and k["max_steps"] == 300 and k["arm"] is None
    for bad in ("42", "-1"):
        with pytest.raises(IndexError):
            tcli.main(["train-index-3b", "m.tsv", "c.tsv", "d", str(tmp_path), bad])


# ---- scripts ---------------------------------------------------------------------------------

def _sbatch(name: str) -> list[str]:
    return [ln for ln in SCRIPTS[name].read_text().splitlines() if ln.startswith("#SBATCH")]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name,time", [("train", "12:00:00"), ("law", "6:00:00")])
def test_sbatch_lines(name, time):
    lines = _sbatch(name)
    assert "#SBATCH --exclude=fc10713,fc10715" in lines
    assert f"#SBATCH --time={time}" in lines and sum("--time" in ln for ln in lines) == 1
    assert "#SBATCH --account=def-maxwl" in lines
    assert any("gres" in ln for ln in lines) is (name == "train")
    t = SCRIPTS[name].read_text()
    assert 'pairs.py" tasks3b "$MANIFEST"' in t and 'pairs.py" tasks "' not in t


def _dry(name: str, args: list[str], index: int | None) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"))
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


HEAD = "/products/MANIFEST.tsv /kit/tools/t118/covariates.tsv /cache"


@pytest.mark.parametrize("index", sorted(ANCHORS))
def test_dry_run_train(index):
    run = ANCHORS[index]
    r = _dry("train", ["/kit", "/products", "/cache", OUT, "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        f"python3 /kit/tools/t128p3/train.py train-index-3b {HEAD} {OUT}/runs {index}",
        f"python3 /kit/tools/t118/ladder/score.py trained {HEAD} /bl.bed {OUT}/runs/{run} "
        "--workers 4"]


@pytest.mark.parametrize("index", sorted(ANCHORS))
def test_dry_run_law(index):
    run = ANCHORS[index]
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", OUT], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        f"python3 /kit/tools/t118/ladder/score.py law {HEAD} /bl.bed {OUT}/runs/{run} "
        "--workers 8 --split",
        f"python3 /kit/tools/t128/viabase.py run {HEAD} /bl.bed {OUT}/runs/{run} "
        f"{OUT}/viabase/{run} --workers 8"]


REFUSED = ["/project/def-maxwl/mforooz/t128p3", "/project/def-maxwl/mforooz/t128p2",
           "/project/def-maxwl/mforooz/t128p3/runs", "/project/def-maxwl/mforooz/t128p3/smoke",
           "/x/seeds3b", "/x/t128p3/seeds3bx"]
ACCEPTED = [OUT, OUT + "/smoke"]


@pytest.mark.parametrize("out", REFUSED + ACCEPTED)
def test_out_dir_guard(out):
    ok = out in ACCEPTED
    for r in (_dry("train", ["/kit", "/p", "/c", out, "/bl.bed"], 0),
              _dry("law", ["/kit", "/p", "/c", "/bl.bed", out], 0)):
        assert (r.returncode == 0) is ok, (out, r.returncode, r.stderr)
        if not ok:
            assert r.returncode == 2 and r.stdout == ""


def test_an_index_outside_the_table_fails():
    assert _dry("train", ["/kit", "/p", "/c", OUT, "/bl.bed"], 42).returncode == 2
    assert _dry("law", ["/kit", "/p", "/c", "/bl.bed", OUT], 42).returncode == 2
