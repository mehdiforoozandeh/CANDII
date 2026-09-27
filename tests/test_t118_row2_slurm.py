"""t118 row 2 — the four Nibi job scripts of row 2, checked as text and in DRY_RUN.

Mirrors tests/test_t118_ladder_slurm.py for slurm/t118/row2_{train,train_cpu,law,agg}.sh: the
resource lines the hard rules fix (one MIG-slice gres line on the GPU script and none on the CPU
scripts, the account, the excluded nodes, the job names, a walltime, no exported variables, no
/usr/bin/time, no afterok, no aftercorr, no /scratch), the per-task venv recipe, the row-2 out
dir, and — through DRY_RUN=1 — that the index -> run mapping comes from `pairs.py tasks --row 2`
and that finished work is skipped.
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

SCRIPTS = {name: SLURM / f"row2_{name}.sh" for name in ("train", "train_cpu", "law", "agg")}
ARRAYS = ("train", "train_cpu", "law")
CPU = ("train_cpu", "law", "agg")
GRES = "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
BLACKLIST = "/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed"
OUT = "/project/def-maxwl/mforooz/t118/row2"
CACHE = "/project/def-maxwl/mforooz/t118/ladder/cache"
JOB_NAMES = {"train": "t118L_r2train", "train_cpu": "t118L_r2traincpu", "law": "t118L_r2law",
             "agg": "t118L_r2agg"}
RESOURCES = {"train": ("3:00:00", "4", "16000M"), "train_cpu": ("8:00:00", "16", "64000M"),
             "law": ("3:00:00", "8", "24000M"), "agg": ("2:00:00", "4", "16000M")}
# the four row-2 rungs at g C19M16, counts, real, seed 0 (the R8 pilot indices)
PILOT = {54: "A2_C19M16_counts_real_s0", 198: "B2_C19M16_counts_real_s0",
         342: "C2_C19M16_counts_real_s0", 486: "D2_C19M16_counts_real_s0"}


def _text(name: str) -> str:
    return SCRIPTS[name].read_text(encoding="utf-8")


def _sbatch(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if ln.startswith("#SBATCH")]


def _header(name: str) -> str:
    """The comment block above `set -euo pipefail`."""
    return _text(name).split("set -euo pipefail")[0]


def _dry(name: str, args: list[str], index: int | None = 54, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"), **env)
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


def _train_lines(kit, manifest, cache, out, index, run, blacklist=BLACKLIST, cpu=False):
    dev = " --device cpu" if cpu else ""
    return [
        f"python3 {kit}/tools/t118/ladder/train.py train-index {manifest} "
        f"{kit}/tools/t118/covariates.tsv {cache} {out}/runs {index} --row 2{dev}",
        f"python3 {kit}/tools/t118/ladder/score.py trained {manifest} "
        f"{kit}/tools/t118/covariates.tsv {cache} {blacklist} {out}/runs/{run} --workers 4",
    ]


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
def test_account_exclude_time_and_no_partition(name):
    lines = _sbatch(name)
    assert "#SBATCH --account=def-maxwl" in lines
    assert "#SBATCH --exclude=c128,c166,c537" in lines
    assert any(re.fullmatch(r"#SBATCH --time=\d+:\d\d:\d\d", ln) for ln in lines)
    assert not any("--partition" in ln for ln in lines)
    assert not any("--output" in ln or "--error" in ln for ln in lines), "the submitter gives the log path"


@pytest.mark.parametrize("name", SCRIPTS)
def test_resources(name):
    time, cpus, mem = RESOURCES[name]
    lines = _sbatch(name)
    assert f"#SBATCH --time={time}" in lines
    assert f"#SBATCH --cpus-per-task={cpus}" in lines
    assert f"#SBATCH --mem={mem}" in lines


def test_job_names():
    names = {}
    for name in SCRIPTS:
        found = re.findall(r"job-name=(\S+)", _text(name))
        assert len(found) == 1, (name, found)
        names[name] = found[0]
    assert names == JOB_NAMES


@pytest.mark.parametrize("name", SCRIPTS)
def test_house_rules(name):
    t = _text(name)
    assert t.startswith("#!/bin/bash\n")
    assert "set -euo pipefail" in t
    assert "--export=" not in t
    code = [ln for ln in t.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("/usr/bin/time" in ln for ln in code), "peak memory comes from sacct MaxRSS"
    assert "afterok" not in t
    assert "aftercorr" not in t, "row 2 submits the law array directly"
    assert "/scratch" not in t, "every output goes under the positional out dir on /project"


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_venv_recipe_is_the_pinned_one(name):
    t = _text(name)
    for piece in ('PY_MODULES="python/3.10.13"',
                  'virtualenv --no-download "$SLURM_TMPDIR/venv"',
                  'pip install --no-index -r "$KIT/requirements-fir.txt"',
                  'pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"',
                  'ls "$KIT"/wheels/x_transformers-*-py3-none-any.whl',
                  'export PYTHONPATH="$KIT/src"',
                  '"$KIT"/src/*) ;;'):
        assert piece in t, (name, piece)
    # DRY_RUN exits before the module load
    assert t.index('"${DRY_RUN:-0}" = 1') < t.index("module load $PY_MODULES")


@pytest.mark.parametrize("name", SCRIPTS)
def test_headers_document_the_row2_out_dir(name):
    h = _header(name)
    assert f"OUT={OUT}" in h
    if name != "agg":
        assert f"CACHE={CACHE}" in h
    assert "--exclude=c128,c166,c537" in h


@pytest.mark.parametrize("name", ARRAYS)
def test_array_scripts_use_the_row2_table_and_their_own_tasks_tsv(name):
    t = _text(name)
    assert t.count('pairs.py" tasks "$TABLE_MANIFEST" --row 2)') == 1
    assert t.count('pairs.py" tasks "$MANIFEST" --row 2 > "$TMP_TABLE"') == 1
    assert 'TASKS_TSV="$OUT/tasks.tsv"' in t
    assert re.search(r'pairs\.py" tasks "\$[A-Z_]+"(?! --row 2)', t) is None


@pytest.mark.parametrize("name,array", [("train", "--array=0-575%40"),
                                        ("train_cpu", "--array=0-575%40"),
                                        ("law", "--array=0-575%40")])
def test_headers_document_the_capped_array_submission(name, array):
    h = _header(name)
    assert f"sbatch --test-only {array}" in h
    assert f"sbatch --parsable {array}" in h


def test_train_cpu_differs_from_train_only_where_it_should():
    gpu, cpu = _text("train"), _text("train_cpu")
    assert "--device cpu" not in gpu
    assert cpu.count('"$IDX" --row 2 --device cpu') == 1
    assert 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" ' \
           'MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"' in cpu
    body = lambda t: t.split("set -euo pipefail", 1)[1]  # noqa: E731
    norm = (body(cpu).replace(" --device cpu", "").replace("row2_train_cpu", "row2_train")
            .replace('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" '
                     'MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"\n', ""))
    assert norm == body(gpu)


def test_law_is_submitted_directly_and_agg_is_chained_afterany():
    assert "--dependency" not in _text("law")
    assert "SUBMIT THIS ARRAY DIRECTLY" in _text("law")
    assert "status file" in _text("law")
    assert "--dependency=afterany:$PREV" in _text("agg")


def test_row1_scripts_are_not_the_row2_scripts():
    for n in ("train", "law", "agg"):
        assert "row2" not in (SLURM / f"ladder_{n}.sh").read_text(encoding="utf-8")


# ---- the task table --------------------------------------------------------------------------

def test_row2_task_table_from_the_fixture_manifest():
    rows = pairs.read_manifest(FIXTURE_MANIFEST)
    t = pairs.tasks(rows, row=2)
    assert len(t) == 576
    assert t[0]["run_name"] == "A2_C07M20_counts_real_s0"
    assert t[-1]["run_name"] == "D2_all_pval_ids_s2"
    assert {i: t[i]["run_name"] for i in PILOT} == PILOT


def test_the_pairs_cli_prints_the_row2_table():
    r = subprocess.run([sys.executable, str(REPO / "tools/t118/ladder/pairs.py"), "tasks",
                        str(FIXTURE_MANIFEST), "--row", "2"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=120)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    head = lines[0].split("\t")
    by_index = {int(ln.split("\t")[head.index("index")]): ln.split("\t")[head.index("run_name")]
                for ln in lines[1:]}
    assert len(by_index) == 576 and {i: by_index[i] for i in PILOT} == PILOT


# ---- DRY_RUN ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ("train", "train_cpu"))
def test_dry_run_train_prints_train_then_score_for_task_54(name):
    r = _dry(name, ["/kit", "/products", "/cache", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == _train_lines("/kit", "/products/MANIFEST.tsv", "/cache",
                                                 "/out", 54, PILOT[54], cpu=name == "train_cpu")


def test_dry_run_acceptance_line():
    # the plan's acceptance check: the kit is this checkout, products do not exist
    r = _dry("train", [str(REPO), "/nonexistent", "/cache", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == _train_lines(str(REPO), "/nonexistent/MANIFEST.tsv", "/cache",
                                                 "/out", 54, "A2_C19M16_counts_real_s0")


@pytest.mark.parametrize("index", sorted(PILOT))
def test_dry_run_train_uses_a_real_kit_and_manifest(tmp_path, index):
    products = tmp_path / "products"
    products.mkdir()
    shutil.copy(FIXTURE_MANIFEST, products / "MANIFEST.tsv")
    r = _dry("train", [str(REPO), str(products), "/c", str(tmp_path / "out"), "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    first, last = r.stdout.splitlines()
    assert first.endswith(f"{tmp_path}/out/runs {index} --row 2")
    assert last.endswith(f"/bl.bed {tmp_path}/out/runs/{PILOT[index]} --workers 4")


@pytest.mark.parametrize("name", ("train", "train_cpu"))
def test_dry_run_train_skips_a_scored_run(tmp_path, name):
    run = tmp_path / "runs" / PILOT[54]
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry(name, ["/kit", "/products", "/cache", str(tmp_path)])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_dry_run_train_does_not_skip_on_the_row1_run(tmp_path):
    run = tmp_path / "runs" / "A_C19M16_counts_real_s0"
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("train", ["/kit", "/products", "/cache", str(tmp_path)])
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 2


def test_dry_run_law_prints_the_law_command_for_task_54():
    r = _dry("law", ["/kit", "/products", "/cache", "/blacklist.bed", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/score.py law /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /blacklist.bed /out/runs/A2_C19M16_counts_real_s0 "
        "--workers 8"]
    assert "exit 3" in r.stderr   # no SCORE_DONE under /out: a real task would fail loudly


def test_dry_run_law_skips_when_law_done(tmp_path):
    run = tmp_path / "runs" / PILOT[54]
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(tmp_path)])
    assert r.returncode == 0 and r.stdout.count("score.py law") == 1 and "exit 3" not in r.stderr
    (run / "LAW_DONE").touch()
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(tmp_path)])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


@pytest.mark.parametrize("rung", ("A2", "B2", "C2", "D2"))
def test_dry_run_agg_rung(rung):
    r = _dry("agg", ["/kit", "/products", "/out", "/refs.tsv", "/row1/runs", rung], None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/aggregate.py /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /out/runs /refs.tsv /out/agg "
        "--also-runs /row1/runs --rows 1,2",
        "python3 /kit/tools/t118/ladder/aggregate.py qm-curves /products/MANIFEST.tsv /products "
        "/out/agg/qm_curves.json",
        f"python3 /kit/tools/t118/ladder/figures.py /out/agg {rung} --refs-qm /out/agg/qm_curves.json",
        f"python3 /kit/tools/t118/ladder/report.py /out/agg {rung}",
    ]


def test_dry_run_agg_grid():
    r = _dry("agg", ["/kit", "/products", "/out", "/refs.tsv", "/row1/runs", "grid"], None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/aggregate.py /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /out/runs /refs.tsv /out/agg "
        "--also-runs /row1/runs --rows 1,2",
        "python3 /kit/tools/t118/ladder/report.py grid /out/agg",
    ]


@pytest.mark.parametrize("rung", ("A", "D", "E2", "x"))
def test_agg_rejects_an_unknown_rung(rung):
    r = _dry("agg", ["/kit", "/products", "/out", "/refs.tsv", "/row1/runs", rung], None)
    assert r.returncode == 2


@pytest.mark.parametrize("name,args", [("train", ["/kit", "/p", "/c", "/o"]),
                                       ("train_cpu", ["/kit", "/p", "/c", "/o"]),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o"])])
def test_array_scripts_refuse_to_run_without_a_task_index(name, args):
    assert _dry(name, args, None).returncode != 0


@pytest.mark.parametrize("name,args", [("train", ["/kit", "/p", "/c", "/o"]),
                                       ("train_cpu", ["/kit", "/p", "/c", "/o"]),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o"])])
def test_an_index_outside_the_table_fails(name, args):
    assert _dry(name, args, 576).returncode == 2


def test_missing_positionals_fail():
    assert _dry("train", ["/kit", "/p", "/c"]).returncode != 0
    assert _dry("train_cpu", ["/kit", "/p", "/c"]).returncode != 0
    assert _dry("law", ["/kit", "/p", "/c", "/o"]).returncode != 0
    assert _dry("agg", ["/kit", "/p", "/o", "/r", "A2"], None).returncode != 0
