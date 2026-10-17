"""t118-C5 — the four Nibi job scripts of the architecture ladder, checked as text and in DRY_RUN.

The scripts cannot run here (no SLURM, no modules), so this pins what can be checked off-cluster:
the resource lines the hard rules fix (one MIG-slice gres line on the GPU script and none on the
CPU scripts, the account, the job names, a walltime, no exported variables, no /usr/bin/time, no
afterok, no /scratch), the per-task venv recipe, and — through DRY_RUN=1 — that the index -> run
mapping comes from `pairs.py tasks` and that finished work is skipped.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SLURM = REPO / "slurm" / "t118"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"
sys.path.insert(0, str(REPO / "tools" / "t118"))
from ladder import pairs  # noqa: E402

SCRIPTS = {name: SLURM / f"ladder_{name}.sh" for name in ("cache", "train", "law", "agg")}
CPU = ("cache", "law", "agg")
GRES = "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
BLACKLIST = "/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed"
PILOT = {54: "A_C19M16_counts_real_s0", 57: "A_C19M16_counts_nocov_s0",
         60: "A_C19M16_counts_ids_s0", 63: "A_C19M16_pval_real_s0",
         66: "A_C19M16_pval_nocov_s0", 69: "A_C19M16_pval_ids_s0"}


def _text(name: str) -> str:
    return SCRIPTS[name].read_text(encoding="utf-8")


def _sbatch(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if ln.startswith("#SBATCH")]


def _dry(name: str, args: list[str], index: int | None = 54, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"), **env)
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


# ---- text checks -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_gpu_script_has_exactly_the_mig_slice_gres_line():
    gres = [ln for ln in _text("train").splitlines() if "gres" in ln]
    assert gres == [GRES]


@pytest.mark.parametrize("name", CPU)
def test_cpu_scripts_request_no_gres(name):
    assert "SBATCH --gres" not in _text(name)
    assert not any("gres" in ln for ln in _sbatch(name))


@pytest.mark.parametrize("name", SCRIPTS)
def test_account_time_and_no_partition(name):
    lines = _sbatch(name)
    assert "#SBATCH --account=def-maxwl" in lines
    assert any(re.fullmatch(r"#SBATCH --time=\d+:\d\d:\d\d", ln) for ln in lines)
    assert not any("--partition" in ln for ln in lines)
    assert not any("--output" in ln or "--error" in ln for ln in lines), "the submitter gives the log path"


def test_job_names():
    names = {}
    for name in SCRIPTS:
        found = re.findall(r"job-name=(\S+)", _text(name))
        assert len(found) == 1, (name, found)
        names[name] = found[0]
    assert names == {n: f"t118L_{n}" for n in SCRIPTS}
    assert not any(v.startswith("t112_") for v in names.values())


@pytest.mark.parametrize("name", SCRIPTS)
def test_house_rules(name):
    t = _text(name)
    assert t.startswith("#!/bin/bash\n")
    assert "set -euo pipefail" in t
    assert "--export=" not in t
    code = [ln for ln in t.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("/usr/bin/time" in ln for ln in code), "peak memory comes from sacct MaxRSS"
    assert "afterok:" not in t
    assert "/scratch" not in t, "every output goes under the positional out dir on /project"


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_venv_recipe_is_the_pinned_one(name):
    t = _text(name)
    for piece in ('PY_MODULES="python/3.10.13"',
                  'virtualenv --no-download "$SLURM_TMPDIR/venv"',
                  'pip install --no-index -r "$KIT/requirements-fir.txt"',
                  'pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"',
                  'export PYTHONPATH="$KIT/src"',
                  '"$KIT"/src/*) ;;'):
        assert piece in t, (name, piece)
    # DRY_RUN exits before the module load
    assert t.index('"${DRY_RUN:-0}" = 1') < t.index("module load $PY_MODULES")


@pytest.mark.parametrize("name,array", [("cache", "--array=0-129%40"), ("train", "--array=0-575%40"),
                                        ("law", "--array=0-575%40")])
def test_headers_document_the_capped_array_submission(name, array):
    t = _text(name)
    assert f"sbatch --test-only {array}" in t
    assert f"sbatch --parsable" in t and array in t


def test_law_is_chained_with_aftercorr_in_the_documented_submission():
    assert "--dependency=aftercorr:$TRAIN" in _text("law")
    assert "--dependency=aftercorr:$TRAIN" in _text("train")


# ---- the task table --------------------------------------------------------------------------

def test_task_table_from_the_fixture_manifest():
    rows = pairs.read_manifest(FIXTURE_MANIFEST)
    t = pairs.tasks(rows)
    assert len(t) == 576
    assert t[0]["run_name"] == "A_C07M20_counts_real_s0"
    assert t[-1]["run_name"] == "D_all_pval_ids_s2"
    assert {i: t[i]["run_name"] for i in PILOT} == PILOT
    counts = {g: len(pairs.train_pairs(rows, g)) for g in pairs.TRACKS}
    assert counts == {"C07M20": 38, "C07M29": 38, "C12M02": 14, "C19M16": 38, "C19M22": 38,
                      "C40M17": 38, "C40M18": 38}


def test_array_ranges_match_the_tables():
    rows = pairs.read_manifest(FIXTURE_MANIFEST)
    assert len(pairs.tasks(rows)) - 1 == 575
    assert len(rows) - 1 == 129   # the cache array covers all 130 products


# ---- DRY_RUN ---------------------------------------------------------------------------------

def test_dry_run_train_prints_train_then_score_for_task_54():
    r = _dry("train", ["/kit", "/products", "/cache", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/train.py train-index /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /out/runs 54",
        "python3 /kit/tools/t118/ladder/score.py trained /products/MANIFEST.tsv "
        f"/kit/tools/t118/covariates.tsv /cache {BLACKLIST} /out/runs/A_C19M16_counts_real_s0 "
        "--workers 4",
    ]


@pytest.mark.parametrize("index", sorted(PILOT))
def test_dry_run_train_uses_a_real_kit_and_manifest(tmp_path, index):
    products = tmp_path / "products"
    products.mkdir()
    shutil.copy(FIXTURE_MANIFEST, products / "MANIFEST.tsv")
    r = _dry("train", [str(REPO), str(products), "/c", str(tmp_path / "out"), "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    last = r.stdout.splitlines()[-1]
    assert last.endswith(f"/bl.bed {tmp_path}/out/runs/{PILOT[index]} --workers 4")


def test_dry_run_train_skips_a_scored_run(tmp_path):
    run = tmp_path / "runs" / "A_C19M16_counts_real_s0"
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("train", ["/kit", "/products", "/cache", str(tmp_path)])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_dry_run_law_prints_the_law_command_for_task_54():
    r = _dry("law", ["/kit", "/products", "/cache", "/blacklist.bed", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/score.py law /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /blacklist.bed /out/runs/A_C19M16_counts_real_s0 "
        "--workers 8"]
    assert "exit 3" in r.stderr   # no SCORE_DONE under /out: a real task would fail loudly


def test_dry_run_law_skips_when_law_done(tmp_path):
    run = tmp_path / "runs" / "A_C19M16_counts_real_s0"
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(tmp_path)])
    assert r.returncode == 0 and r.stdout.count("score.py law") == 1 and "exit 3" not in r.stderr
    (run / "LAW_DONE").touch()
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(tmp_path)])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_dry_run_cache():
    r = _dry("cache", ["/kit", "/products", "/cache"], 7)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/train.py cache /products/MANIFEST.tsv /products /cache 7"]


def test_dry_run_agg():
    r = _dry("agg", ["/kit", "/products", "/out", "/refs.tsv", "B"], None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/aggregate.py /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /out/runs /refs.tsv /out/agg",
        "python3 /kit/tools/t118/ladder/aggregate.py qm-curves /products/MANIFEST.tsv /products "
        "/out/agg/qm_curves.json",
        "python3 /kit/tools/t118/ladder/figures.py /out/agg B --refs-qm /out/agg/qm_curves.json",
        "python3 /kit/tools/t118/ladder/report.py /out/agg B",
    ]


def test_agg_rejects_an_unknown_rung():
    r = _dry("agg", ["/kit", "/products", "/out", "/refs.tsv", "E"], None)
    assert r.returncode == 2


@pytest.mark.parametrize("name,args", [("train", ["/kit", "/p", "/c", "/o"]),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o"]),
                                       ("cache", ["/kit", "/p", "/c"])])
def test_array_scripts_refuse_to_run_without_a_task_index(name, args):
    assert _dry(name, args, None).returncode != 0


def test_an_index_outside_the_table_fails():
    assert _dry("train", ["/kit", "/p", "/c", "/o"], 576).returncode == 2


def test_missing_positionals_fail():
    assert _dry("train", ["/kit", "/p", "/c"]).returncode != 0
    assert _dry("law", ["/kit", "/p", "/c", "/o"]).returncode != 0
