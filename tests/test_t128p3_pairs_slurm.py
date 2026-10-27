"""t128p3-C2 — the 2-arm x 48-row phase-3 task table and the three Fir job scripts.

  * the table: 48 rows, anchors 9/33, form rung D everywhere, g_hidden / f_channels per arm, the
    run names equal `ladder.train.run_name(arm, ...)`; REFERENCE_RUNS = the 24 phase-2 D runs;
  * the scripts: #SBATCH lines, the t128p3 out-dir guard and, through DRY_RUN=1, the command lines
    and skip rules (the scripts cannot run here: no SLURM, no modules).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tools" / "t118", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ladder import train as ladder_train  # noqa: E402
from t128 import viabase  # noqa: E402
from t128p3 import pairs as tp  # noqa: E402

SLURM = REPO / "slurm" / "t128p3"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
SCRIPTS = {name: SLURM / f"{name}.sh" for name in ("train", "law", "verify")}
GRES = "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
EXCLUDE = "#SBATCH --exclude=fc10713,fc10715"
ANCHORS = {9: "Dbigg_C19M16_counts_real_s0", 33: "Dbigf_C19M16_counts_real_s0"}
RUN33 = ANCHORS[33]
SIZES = {"Dbigg": (256, 32), "Dbigf": (64, 64)}


# ---- settings and the task table -------------------------------------------------------------

def test_settings():
    S = tp.SETTINGS
    assert [s["number"] for s in S] == [1, 2]
    assert tp.ARMS == ("Dbigg", "Dbigf")
    keys = ["number", "arm", "rung", "g_hidden", "f_channels", "n_theta", "loss", "max_steps",
            "val_nll_rule"]
    assert all(list(s) == keys for s in S)
    got = [tuple(s[k] for k in keys) for s in S]
    assert got == [(1, "Dbigg", "D", 256, 32, 256, "nb", 15000, "t118 cross NLL"),
                   (2, "Dbigf", "D", 64, 64, 512, "nb", 15000, "t118 cross NLL")]
    assert all(s["n_theta"] == 2 * 4 * s["f_channels"] for s in S)
    assert tp.SETTING_BY_ARM == {s["arm"]: s for s in S}
    assert not any("_" in a for a in tp.ARMS), "run names split on _"


def test_constants():
    assert tp.HOLDOUT == (("depth", "7.5M"),)
    assert tp.SEEDS == (0, 1, 2) and tp.SPACE == "counts" and tp.MAX_STEPS == 15000
    assert tp.G_IDS == tp._lp.G_IDS and len(tp.G_IDS) == 8 and tp.G_IDS[3] == "C19M16"
    assert tp.TASK_KEYS == ("index", "rung", "form_rung", "g", "space", "model", "seed",
                            "run_name", "setting", "g_hidden", "f_channels")
    assert tp.SMOKE == (9, 33) and tp.FULL_ARRAY == "0-47"
    assert tp.ARM_ARRAYS == {"Dbigg": "0-23", "Dbigf": "24-47"}


def test_task_table():
    t = tp.tasks()
    assert len(t) == 48 == len({r["run_name"] for r in t})
    assert [r["index"] for r in t] == list(range(48))
    assert {i: t[i]["run_name"] for i in ANCHORS} == ANCHORS
    assert all(list(r) == list(tp.TASK_KEYS) for r in t)
    for ai, arm in enumerate(tp.ARMS):
        for gi, g in enumerate(tp.G_IDS):
            for seed in tp.SEEDS:
                r = t[ai * 24 + gi * 3 + seed]
                assert (r["rung"], r["g"], r["seed"]) == (arm, g, seed)
                assert r["form_rung"] == "D" and r["setting"] == ai + 1
                assert (r["g_hidden"], r["f_channels"]) == SIZES[arm]
                assert r["model"] == "real" and r["space"] == "counts"
                assert r["run_name"] == f"{arm}_{g}_counts_real_s{seed}"
                assert r["run_name"] == ladder_train.run_name(arm, g, "counts", "real", seed)
    lo, hi = (int(x) for x in tp.ARM_ARRAYS["Dbigf"].split("-"))
    assert {r["rung"] for r in t[lo:hi + 1]} == {"Dbigf"}
    assert tp.tasks(tp._lp.read_manifest(FIXTURE_MANIFEST)) == t


def test_reference_runs_are_the_phase2_d_runs():
    assert len(tp.REFERENCE_RUNS) == 24
    assert list(tp.REFERENCE_RUNS) == [v["run_name"] for v in viabase.VIABASE_RUNS]
    want = [f"D_{g}_counts_real_s{s}" for g in tp.G_IDS for s in tp.SEEDS]
    assert list(tp.REFERENCE_RUNS) == want
    assert not set(tp.REFERENCE_RUNS) & {r["run_name"] for r in tp.tasks()}


def test_cli_tasks_tsv():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "t128p3" / "pairs.py"), "tasks",
                        str(FIXTURE_MANIFEST)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 49 and lines[0].split("\t") == list(tp.TASK_KEYS)
    rows = {int(c[0]): c for c in (ln.split("\t") for ln in lines[1:])}
    assert [rows[i][k] for i in (9, 33) for k in (7, 9, 10)] == [
        "Dbigg_C19M16_counts_real_s0", "256", "32", "Dbigf_C19M16_counts_real_s0", "64", "64"]
    assert {c[2] for c in rows.values()} == {"D"}


def test_cli_pairs_hold_out_7p5m():
    py, tool = sys.executable, str(REPO / "tools" / "t128p3" / "pairs.py")
    tr = subprocess.run([py, tool, "train-pairs", str(FIXTURE_MANIFEST), "C19M16"],
                        capture_output=True, text=True, timeout=60)
    ho = subprocess.run([py, tool, "heldout-pairs", str(FIXTURE_MANIFEST), "C19M16"],
                        capture_output=True, text=True, timeout=60)
    assert tr.returncode == 0 and ho.returncode == 0, tr.stderr + ho.stderr
    assert "7.5M" not in tr.stdout
    held = ho.stdout.splitlines()[1:]
    assert held and all("7.5M" in ln for ln in held)


# ---- script text -----------------------------------------------------------------------------

def _text(name: str) -> str:
    return SCRIPTS[name].read_text(encoding="utf-8")


def _sbatch(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if ln.startswith("#SBATCH")]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_gres_only_on_train():
    assert [ln for ln in _text("train").splitlines() if "gres" in ln] == [GRES]
    for name in ("law", "verify"):
        assert not any("gres" in ln for ln in _sbatch(name))


@pytest.mark.parametrize("name,time,cpus,mem", [("train", "1-00:00:00", 4, "16000M"),
                                                ("law", "12:00:00", 8, "24000M"),
                                                ("verify", "3:00:00", 8, "32000M")])
def test_sbatch_lines(name, time, cpus, mem):
    lines = _sbatch(name)
    assert "#SBATCH --account=def-maxwl" in lines
    assert f"#SBATCH --job-name=t128p3_{name}" in lines
    assert len(re.findall(r"job-name=", _text(name))) == 1
    assert EXCLUDE in lines and sum("--exclude" in ln for ln in lines) == 1
    assert f"#SBATCH --time={time}" in lines and sum("--time" in ln for ln in lines) == 1
    assert f"#SBATCH --cpus-per-task={cpus}" in lines and f"#SBATCH --mem={mem}" in lines
    assert not any("--partition" in ln or "--export" in ln for ln in lines)
    outs = [ln for ln in lines if "--output" in ln or "--error" in ln]
    if name == "verify":
        assert outs == ["#SBATCH --output=/project/def-maxwl/mforooz/t128p3/verify/%x_%j.out"]
    else:
        assert outs == [], "the submitter gives the log path"


@pytest.mark.parametrize("name", SCRIPTS)
def test_house_rules(name):
    t = _text(name)
    assert t.startswith("#!/bin/bash\n")
    assert ("set -euo pipefail" if name != "verify" else "set -uo pipefail") in t
    assert "--export=" not in t
    assert "afterok:" not in t and "/scratch" not in t
    assert "c128" not in t and "EIC_REPRO" not in t, "no Nibi node names or Nibi default paths"
    assert "tools/t128p2/" not in t, "phase 3 runs its own tools"


@pytest.mark.parametrize("name", ("train", "law"))
def test_the_venv_recipe_is_the_pinned_one(name):
    t = _text(name)
    for piece in ('PY_MODULES="python/3.10.13"',
                  'virtualenv --no-download "$SLURM_TMPDIR/venv"',
                  'pip install --no-index -r "$KIT/requirements-fir.txt"',
                  'pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"',
                  'export PYTHONPATH="$KIT/src"',
                  '"$KIT"/src/*) ;;',
                  'python3 "$KIT/tools/t128p3/pairs.py" tasks "$MANIFEST"'):
        assert piece in t, (name, piece)
    assert t.index('"${DRY_RUN:-0}" = 1') < t.index("module load $PY_MODULES")


def test_verify_runs_the_five_sets_on_its_own_kit():
    t = _text("verify")
    assert 'K="${1:?' in t
    assert 'export PYTHONPATH="$K/src"' in t and '"$K"/src/*) ;;' in t
    assert "for set in t128p3 t128p2 t128 t126 t118; do" in t
    assert "tests/test_${set}_*.py" in t
    assert 'echo "RESULT $set rc=$RC' in t
    assert t.index("bin/activate") < t.index("python3 -m pytest")


def test_headers_document_smoke_full_and_aftercorr():
    t = _text("train")
    for piece in ("--array=9,33%2", "--array=0-23%12", "--array=24-47%12", "/smoke", "300 100",
                  "--time=3:00:00", "--dependency=aftercorr:$TRAIN", "LIVE"):
        assert piece in t, piece
    lw = _text("law")
    for piece in ("--array=9,33%2", "--array=0-23%12", "--array=24-47%12",
                  "--dependency=aftercorr:$TRAIN", "LIVE"):
        assert piece in lw, piece


# ---- DRY_RUN ---------------------------------------------------------------------------------

def _dry(name: str, args: list[str], index: int | None = 33) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"))
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


TRAIN_ARGS = ["/kit", "/products", "/cache", "/out/t128p3", "/bl.bed"]
TRAIN_LINES = [
    "python3 /kit/tools/t128p3/train.py train-index /products/MANIFEST.tsv "
    "/kit/tools/t118/covariates.tsv /cache /out/t128p3/runs 33",
    "python3 /kit/tools/t118/ladder/score.py trained /products/MANIFEST.tsv "
    "/kit/tools/t118/covariates.tsv /cache /bl.bed /out/t128p3/runs/Dbigf_C19M16_counts_real_s0 "
    "--workers 4"]


def test_dry_run_train_prints_the_two_pinned_lines_for_task_33():
    r = _dry("train", TRAIN_ARGS)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == TRAIN_LINES


def test_dry_run_train_step_overrides():
    r = _dry("train", TRAIN_ARGS + ["300", "100"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [TRAIN_LINES[0] + " --max-steps 300 --eval-every 100",
                                     TRAIN_LINES[1]]
    r = _dry("train", TRAIN_ARGS + ["300"])
    assert r.stdout.splitlines()[0] == TRAIN_LINES[0] + " --max-steps 300"
    assert _dry("train", TRAIN_ARGS + ["3x"]).returncode == 2


@pytest.mark.parametrize("index", sorted(ANCHORS))
def test_dry_run_train_maps_every_smoke_index(index):
    r = _dry("train", ["/kit", "/p", "/c", "/out/t128p3", "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[-1].endswith(f"/bl.bed /out/t128p3/runs/{ANCHORS[index]} --workers 4")


def test_dry_run_train_skips_a_scored_run(tmp_path):
    out = tmp_path / "t128p3"
    run = out / "runs" / RUN33
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("train", ["/kit", "/p", "/c", str(out), "/bl.bed"])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_train_blacklist_is_required():
    assert _dry("train", ["/kit", "/p", "/c", "/out/t128p3"]).returncode != 0
    assert "DEFAULT_BLACKLIST" not in _text("train")


def _law(out, index=33):
    return _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(out)], index)


def _law_lines(out: str, run: str) -> list[str]:
    head = "/products/MANIFEST.tsv /kit/tools/t118/covariates.tsv /cache /bl.bed"
    return [f"python3 /kit/tools/t118/ladder/score.py law {head} {out}/runs/{run} --workers 8 --split",
            f"python3 /kit/tools/t128/viabase.py run {head} {out}/runs/{run} {out}/viabase/{run} "
            "--workers 8"]


def test_dry_run_law_prints_the_two_pinned_lines_for_task_33():
    r = _law("/out/t128p3")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/score.py law /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /bl.bed /out/t128p3/runs/Dbigf_C19M16_counts_real_s0 "
        "--workers 8 --split",
        "python3 /kit/tools/t128/viabase.py run /products/MANIFEST.tsv /kit/tools/t118/covariates.tsv "
        "/cache /bl.bed /out/t128p3/runs/Dbigf_C19M16_counts_real_s0 "
        "/out/t128p3/viabase/Dbigf_C19M16_counts_real_s0 --workers 8"]
    assert "exit 3" in r.stderr   # no SCORE_DONE: a real task would fail loudly


@pytest.mark.parametrize("index", sorted(ANCHORS))
def test_dry_run_law_skip_rules(tmp_path, index):
    out = tmp_path / "t128p3"
    name = ANCHORS[index]
    run, vb = out / "runs" / name, out / "viabase" / name
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    want = _law_lines(str(out), name)
    r = _law(out, index)
    assert r.returncode == 0 and r.stdout.splitlines() == want and "exit 3" not in r.stderr
    (run / "LAW_DONE").touch()
    r = _law(out, index)
    assert r.returncode == 0 and r.stdout.splitlines() == want[1:]
    vb.mkdir(parents=True)
    (vb / "LAW_DONE").touch()
    r = _law(out, index)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_dry_run_law_viabase_mark_alone(tmp_path):
    """Only the via-base mark: the law step still runs, the via-base step does not."""
    out = tmp_path / "t128p3"
    (out / "viabase" / RUN33).mkdir(parents=True)
    (out / "viabase" / RUN33 / "LAW_DONE").touch()
    r = _law(out)
    assert r.returncode == 0 and r.stdout.splitlines() == _law_lines(str(out), RUN33)[:1]


REFUSED = ["/project/def-maxwl/mforooz/t128p2", "/project/def-maxwl/mforooz/t128p2/smoke",
           "/project/def-maxwl/mforooz/t128", "/project/def-maxwl/mforooz/t128/runs",
           "/x/t126/D", "/project/def-maxwl/mforooz/t118/ladder", "/project/def-maxwl/mforooz/t125",
           "/x/t128p3x", "/x/out"]
ACCEPTED = ["/project/def-maxwl/mforooz/t128p3", "/project/def-maxwl/mforooz/t128p3/smoke"]


@pytest.mark.parametrize("out", REFUSED + ACCEPTED)
def test_out_dir_guard(out):
    ok = out in ACCEPTED
    for r in (_dry("train", ["/kit", "/p", "/c", out, "/bl.bed"]),
              _dry("law", ["/kit", "/p", "/c", "/bl.bed", out])):
        assert (r.returncode == 0) is ok, (out, r.returncode, r.stderr)
        if not ok:
            assert r.returncode == 2 and r.stdout == ""


@pytest.mark.parametrize("name,args", [("train", TRAIN_ARGS),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o/t128p3"])])
def test_array_scripts_refuse_to_run_without_a_task_index(name, args):
    assert _dry(name, args, None).returncode != 0


def test_an_index_outside_the_table_fails():
    assert _dry("train", TRAIN_ARGS, 48).returncode == 2
    assert _dry("law", ["/kit", "/p", "/c", "/b", "/o/t128p3"], 48).returncode == 2


def test_missing_positionals_fail():
    assert _dry("law", ["/kit", "/p", "/c", "/o/t128p3"]).returncode != 0
