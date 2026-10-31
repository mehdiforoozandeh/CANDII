"""t128-C2 — the 8 settings, the 240-row task table and the three Nibi job scripts of t128.

The settings must match the spec table (plan/T126_ENCDEC_D2.md "Phase 1 settings") and the PI's
selection rule (each setting selects on its own loss's NLL terms; L1+L2+L3 on L1+L2 NLL). The
scripts cannot run here (no SLURM, no modules), so this pins what can be checked off-cluster: the
#SBATCH lines, the house rules, the out-dir guard and, through DRY_RUN=1, the command lines and
the skip rules.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SLURM = REPO / "slurm" / "t128"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
sys.path.insert(0, str(REPO / "tools"))
from t128 import pairs as tp  # noqa: E402

SCRIPTS = {name: SLURM / f"{name}.sh" for name in ("train", "law", "verify")}
GRES = "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
EXCLUDE = "#SBATCH --exclude=c128,c166,c537,c371,c329"
BLACKLIST = "/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed"
ANCHORS = {18: "DS_C19M16_counts_real_s0", 57: "DSbox_C19M16_counts_real_s0",
           90: "DSrt_C19M16_counts_real_s0", 129: "DSrtvae_C19M16_counts_real_s0",
           153: "DSl0.1_C19M16_counts_real_s0", 177: "DSl1_C19M16_counts_real_s0",
           201: "DSl10_C19M16_counts_real_s0", 225: "DSvae_C19M16_counts_real_s0"}
RUN18 = "DS_C19M16_counts_real_s0"


def _text(name: str) -> str:
    return SCRIPTS[name].read_text(encoding="utf-8")


def _sbatch(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if ln.startswith("#SBATCH")]


def _dry(name: str, args: list[str], index: int | None = 18) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"))
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


def _expand(spec: str) -> set[int]:
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out |= set(range(int(a), int(b or a) + 1))
    return out


# ---- settings --------------------------------------------------------------------------------

def test_settings_match_the_spec_table():
    S = tp.SETTINGS
    assert len(S) == 8 and [s["number"] for s in S] == list(range(1, 9))
    assert tp.ARMS == ("DS", "DSbox", "DSrt", "DSrtvae", "DSl0.1", "DSl1", "DSl10", "DSvae")
    keys = {"number", "arm", "rung", "cclean", "loss", "lam", "vae", "twin", "val_nll_rule"}
    assert all(set(s) == keys for s in S)
    got = [(s["rung"], s["cclean"], s["loss"], s["lam"], s["vae"], s["twin"]) for s in S]
    assert got == [("DS", "A", "L12", 0.0, False, True),
                   ("DSBOX", "D", "L12", 0.0, False, False),
                   ("DS", "A", "L2", 0.0, False, True),
                   ("DS", "A", "L2", 0.0, True, False),
                   ("DS", "A", "L123", 0.1, False, False),
                   ("DS", "A", "L123", 1.0, False, False),
                   ("DS", "A", "L123", 10.0, False, False),
                   ("DS", "A", "L12", 0.0, True, False)]
    assert [s["number"] for s in S if s["rung"] == "DSBOX"] == [2]
    assert tp.SETTING_BY_ARM == {s["arm"]: s for s in S}
    assert not any("_" in a for a in tp.ARMS), "run names split on _"


def test_each_setting_records_its_own_selection_rule():
    """PI ruling (iii): no lam and no KL in the selection NLL; L1+L2+L3 selects on L1+L2."""
    rule = {s["arm"]: s["val_nll_rule"] for s in tp.SETTINGS}
    assert rule == {"DS": "L1+L2 NLL", "DSbox": "L1+L2 NLL", "DSrt": "L2 NLL",
                    "DSrtvae": "L2 NLL", "DSl0.1": "L1+L2 NLL", "DSl1": "L1+L2 NLL",
                    "DSl10": "L1+L2 NLL", "DSvae": "L1+L2 NLL"}


def test_constants():
    assert tp.HOLDOUT == (("depth", "7.5M"),)
    assert tp.SEEDS == (0, 1, 2) and tp.SPACE == "counts"
    assert len(tp.G_IDS) == 8 and tp.G_IDS[-1] == "all"
    assert tp.TASK_KEYS == ("index", "rung", "g", "space", "model", "seed", "run_name", "setting")


# ---- the task table --------------------------------------------------------------------------

def test_task_table():
    t = tp.tasks()
    assert len(t) == 240 == len({r["run_name"] for r in t})
    assert [r["index"] for r in t] == list(range(240))
    assert {i: t[i]["run_name"] for i in ANCHORS} == ANCHORS
    assert tuple(sorted(ANCHORS)) == tp.SMOKE
    assert all(set(r) == set(tp.TASK_KEYS) for r in t)
    for r in t:
        assert r["run_name"] == f"{r['rung']}_{r['g']}_counts_{r['model']}_s{r['seed']}"
        assert tp.SETTING_BY_ARM[r["rung"]]["number"] == r["setting"]
    nocov = [r for r in t if r["model"] == "nocov"]
    assert len(nocov) == 48 and {r["setting"] for r in nocov} == {1, 3}
    assert [r["setting"] for r in t] == sorted(r["setting"] for r in t)   # setting slowest
    assert {s: sum(r["setting"] == s for r in t) for s in range(1, 9)} == {
        1: 48, 2: 24, 3: 48, 4: 24, 5: 24, 6: 24, 7: 24, 8: 24}
    assert tp.tasks(tp._lp.read_manifest(FIXTURE_MANIFEST)) == t


def test_full_array_and_smoke_cover_the_table_once():
    full = _expand(tp.FULL_ARRAY)
    assert full.isdisjoint(tp.SMOKE)
    assert full | set(tp.SMOKE) == set(range(240))


def test_cli_tasks_tsv():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "t128" / "pairs.py"), "tasks",
                        str(FIXTURE_MANIFEST)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert len(lines) == 241 and lines[0].split("\t") == list(tp.TASK_KEYS)
    rows = [ln.split("\t") for ln in lines[1:]]
    assert {int(c[0]): c[6] for c in rows if int(c[0]) in ANCHORS} == ANCHORS


def test_cli_pairs_hold_out_7p5m():
    py, tool = sys.executable, str(REPO / "tools" / "t128" / "pairs.py")
    tr = subprocess.run([py, tool, "train-pairs", str(FIXTURE_MANIFEST), "C19M16"],
                        capture_output=True, text=True, timeout=60)
    ho = subprocess.run([py, tool, "heldout-pairs", str(FIXTURE_MANIFEST), "C19M16"],
                        capture_output=True, text=True, timeout=60)
    assert tr.returncode == 0 and ho.returncode == 0, tr.stderr + ho.stderr
    assert "7.5M" not in tr.stdout
    held = ho.stdout.splitlines()[1:]
    assert held and all("7.5M" in ln for ln in held)


# ---- script text -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_gres_only_on_train():
    assert [ln for ln in _text("train").splitlines() if "gres" in ln] == [GRES]
    for name in ("law", "verify"):
        assert not any("gres" in ln for ln in _sbatch(name))


@pytest.mark.parametrize("name,time,cpus,mem", [("train", "4:00:00", 4, "16000M"),
                                                ("law", "6:00:00", 8, "24000M"),
                                                ("verify", "3:00:00", 8, "32000M")])
def test_sbatch_lines(name, time, cpus, mem):
    lines = _sbatch(name)
    assert "#SBATCH --account=def-maxwl" in lines
    assert f"#SBATCH --job-name=t128_{name}" in lines
    assert len(re.findall(r"job-name=", _text(name))) == 1
    assert EXCLUDE in lines and sum("--exclude" in ln for ln in lines) == 1
    assert f"#SBATCH --time={time}" in lines
    assert f"#SBATCH --cpus-per-task={cpus}" in lines and f"#SBATCH --mem={mem}" in lines
    assert not any("--partition" in ln or "--export" in ln for ln in lines)
    outs = [ln for ln in lines if "--output" in ln or "--error" in ln]
    if name == "verify":
        assert outs == ["#SBATCH --output=/project/def-maxwl/mforooz/t128/verify/%x_%j.out"]
    else:
        assert outs == [], "the submitter gives the log path"


@pytest.mark.parametrize("name", SCRIPTS)
def test_house_rules(name):
    t = _text(name)
    assert t.startswith("#!/bin/bash\n")
    assert ("set -euo pipefail" if name != "verify" else "set -uo pipefail") in t
    assert "--export=" not in t
    code = [ln for ln in t.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("/usr/bin/time" in ln for ln in code)
    assert "afterok:" not in t
    assert "/scratch" not in t


@pytest.mark.parametrize("name", ("train", "law"))
def test_the_venv_recipe_is_the_pinned_one(name):
    t = _text(name)
    for piece in ('PY_MODULES="python/3.10.13"',
                  'virtualenv --no-download "$SLURM_TMPDIR/venv"',
                  'pip install --no-index -r "$KIT/requirements-fir.txt"',
                  'pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"',
                  'export PYTHONPATH="$KIT/src"',
                  '"$KIT"/src/*) ;;'):
        assert piece in t, (name, piece)
    assert t.index('"${DRY_RUN:-0}" = 1') < t.index("module load $PY_MODULES")
    assert "tools/t128/pairs.py" in t and "tools/t126" not in t


def test_verify_runs_the_three_sets_on_its_own_kit():
    t = _text("verify")
    assert 'K="${1:?' in t
    assert 'export PYTHONPATH="$K/src"' in t and '"$K"/src/*) ;;' in t
    assert "for set in t128 t126 t118; do" in t
    assert 'echo "RESULT $set rc=$RC' in t
    assert t.index("bin/activate") < t.index("python3 -m pytest")


def test_headers_document_the_capped_arrays_and_aftercorr():
    smoke = "--array=18,57,90,129,153,177,201,225%12"
    full = f"--array={tp.FULL_ARRAY}%12"
    for name in ("train", "law"):
        t = _text(name)
        assert smoke in t and full in t and "sbatch --parsable" in t
        assert "--dependency=aftercorr:$TRAIN" in t and "LIVE" in t


# ---- DRY_RUN ---------------------------------------------------------------------------------

def test_dry_run_train_prints_the_two_pinned_lines_for_task_18():
    r = _dry("train", ["/kit", "/products", "/cache", "/out/t128"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t128/train.py train-index /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /out/t128/runs 18",
        "python3 /kit/tools/t118/ladder/score.py trained /products/MANIFEST.tsv "
        f"/kit/tools/t118/covariates.tsv /cache {BLACKLIST} /out/t128/runs/{RUN18} --workers 4"]


@pytest.mark.parametrize("index", sorted(ANCHORS))
def test_dry_run_train_maps_every_smoke_index(index):
    r = _dry("train", ["/kit", "/p", "/c", "/out/t128", "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines()[-1].endswith(f"/bl.bed /out/t128/runs/{ANCHORS[index]} --workers 4")


def test_dry_run_train_skips_a_scored_run(tmp_path):
    out = tmp_path / "t128"
    run = out / "runs" / RUN18
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("train", ["/kit", "/p", "/c", str(out)])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def _law(out: Path) -> subprocess.CompletedProcess:
    return _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(out)])


def test_dry_run_law_prints_law_then_latent():
    r = _law(Path("/out/t128"))
    assert r.returncode == 0, r.stderr
    run = f"/out/t128/runs/{RUN18}"
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/score.py law /products/MANIFEST.tsv "
        f"/kit/tools/t118/covariates.tsv /cache /bl.bed {run} --workers 8 --split",
        "python3 /kit/tools/t128/latent.py run /products/MANIFEST.tsv "
        f"/kit/tools/t118/covariates.tsv /cache /bl.bed {run} --workers 8"]
    assert "exit 3" in r.stderr   # no SCORE_DONE: a real task would fail loudly


def test_dry_run_law_skip_rules(tmp_path):
    out = tmp_path / "t128"
    run = out / "runs" / RUN18
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _law(out)
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 2 and "exit 3" not in r.stderr
    (run / "LAW_DONE").touch()
    r = _law(out)
    assert r.returncode == 0
    lines = r.stdout.splitlines()
    assert len(lines) == 1 and lines[0].startswith("python3 /kit/tools/t128/latent.py run ")
    (run / "LATENT_DONE").touch()
    r = _law(out)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


@pytest.mark.parametrize("out,ok", [("/project/def-maxwl/mforooz/t128", True),
                                    ("/project/def-maxwl/mforooz/t126/D", False),
                                    ("/project/def-maxwl/mforooz/t118/ladder", False),
                                    ("/project/def-maxwl/mforooz/t125", False),
                                    ("/x/t128x", False), ("/x/out", False)])
def test_out_dir_guard(out, ok):
    rt = _dry("train", ["/kit", "/p", "/c", out])
    rl = _dry("law", ["/kit", "/p", "/c", "/bl.bed", out])
    for r in (rt, rl):
        assert (r.returncode == 0) is ok, (out, r.returncode, r.stderr)
        if not ok:
            assert r.returncode == 2 and r.stdout == ""


@pytest.mark.parametrize("name,args", [("train", ["/kit", "/p", "/c", "/o/t128"]),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o/t128"])])
def test_array_scripts_refuse_to_run_without_a_task_index(name, args):
    assert _dry(name, args, None).returncode != 0


def test_an_index_outside_the_table_fails():
    assert _dry("train", ["/kit", "/p", "/c", "/o/t128"], 240).returncode == 2
    assert _dry("law", ["/kit", "/p", "/c", "/b", "/o/t128"], 240).returncode == 2


def test_missing_positionals_fail():
    assert _dry("train", ["/kit", "/p", "/c"]).returncode != 0
    assert _dry("law", ["/kit", "/p", "/c", "/o/t128"]).returncode != 0
