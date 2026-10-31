"""t125 part 1 — the four Nibi job scripts of the depth-offset count grid, checked as text and in
DRY_RUN.

Mirrors tests/test_t118_row2_slurm.py for slurm/t125/eta_{train,train_cpu,law,agg}.sh: the resource
lines the hard rules fix (one MIG-slice gres line on the GPU script and none on the CPU scripts, the
account, the excluded nodes incl. c371, the job names, a walltime, no exported variables, no
/usr/bin/time, no afterok, no /scratch), the per-task venv recipe and kit pin, the t125 out dir,
the law scope fixed in the script, and — through DRY_RUN=1 — that the index -> run mapping comes
from `tools/t125/eta/pairs.py tasks` (written out independently below) and that finished work is
skipped.
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
SLURM = REPO / "slurm" / "t125"
FIXTURE_MANIFEST = REPO / "tests" / "fixtures" / "t112_meta" / "MANIFEST.tsv"

SCRIPTS = {name: SLURM / f"eta_{name}.sh" for name in ("train", "train_cpu", "law", "agg")}
ARRAYS = ("train", "train_cpu", "law")
CPU = ("train_cpu", "law", "agg")
GRES = "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
EXCLUDE = "--exclude=c128,c166,c537,c371"
BLACKLIST = "/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed"
OUT = "/project/def-maxwl/mforooz/t125/eta"
CACHE = "/project/def-maxwl/mforooz/t118/ladder/cache"
OLD_GRID_JSON = "/project/def-maxwl/mforooz/t118/row2/agg/results.json"
JOB_NAMES = {"train": "t125_etatrain", "train_cpu": "t125_etatraincpu", "law": "t125_etalaw",
             "agg": "t125_etaagg"}
RESOURCES = {"train": ("3:00:00", "4", "16000M"), "train_cpu": ("8:00:00", "16", "64000M"),
             "law": ("1:00:00", "8", "24000M"), "agg": ("2:00:00", "4", "16000M")}
N_TASKS = 432


def _expected_table() -> dict[int, str]:
    """The plan's task table, written out here (not imported): rows 1 and 2, then Aold."""
    g_ids = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18", "all")  # ladder.pairs.G_IDS
    models, seeds = ("real", "nocov"), (0, 1, 2)
    out = {}
    for ri, rung in enumerate(("A", "B", "C", "D", "A2", "B2", "C2", "D2")):
        for gi, g in enumerate(g_ids):
            for mi, m in enumerate(models):
                for s in seeds:
                    out[ri * 48 + gi * 6 + mi * 3 + s] = f"{rung}_{g}_counts_{m}_s{s}"
    for gi, g in enumerate(g_ids):
        for mi, m in enumerate(models):
            for s in seeds:
                out[384 + gi * 6 + mi * 3 + s] = f"Aold_{g}_counts_{m}_s{s}"
    return out


TABLE = _expected_table()
PINNED = {18: "A_C19M16_counts_real_s0", 42: "A_all_counts_real_s0",
          402: "Aold_C19M16_counts_real_s0"}


def _text(name: str) -> str:
    return SCRIPTS[name].read_text(encoding="utf-8")


def _sbatch(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if ln.startswith("#SBATCH")]


def _header(name: str) -> str:
    """The comment block above `set -euo pipefail`."""
    return _text(name).split("set -euo pipefail")[0]


def _code(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if not ln.lstrip().startswith("#")]


def _dry(name: str, args: list[str], index: int | None = 18, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"), **env)
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


def _train_lines(kit, manifest, cache, out, index, run, blacklist=BLACKLIST, cpu=False):
    dev = " --device cpu" if cpu else ""
    return [
        f"python3 {kit}/tools/t125/eta/train.py train-index {manifest} "
        f"{kit}/tools/t118/covariates.tsv {cache} {out}/runs {index}{dev}",
        f"python3 {kit}/tools/t125/eta/score.py trained {manifest} "
        f"{kit}/tools/t118/covariates.tsv {cache} {blacklist} {out}/runs/{run} --workers 4",
    ]


# ---- the table the scripts read --------------------------------------------------------------

def test_the_pairs_cli_gives_the_written_out_table():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "t125" / "eta" / "pairs.py"), "tasks",
                        str(FIXTURE_MANIFEST)], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0, r.stderr
    rows = r.stdout.splitlines()
    head = rows[0].split("\t")
    assert head[0] == "index" and head[-1] == "run_name"
    got = {int(ln.split("\t")[0]): ln.split("\t")[-1] for ln in rows[1:]}
    assert len(got) == N_TASKS
    assert got == TABLE
    for i, run in PINNED.items():
        assert got[i] == run


# ---- text checks -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_gpu_script_has_exactly_the_mig_slice_gres_line():
    gres = [ln for ln in _text("train").splitlines() if "gres" in ln]
    assert gres == [GRES]
    assert _text("train").count("SBATCH --gres") == 1


@pytest.mark.parametrize("name", CPU)
def test_cpu_scripts_request_no_gres(name):
    assert _text(name).count("SBATCH --gres") == 0
    assert not any("gres" in ln for ln in _sbatch(name))


@pytest.mark.parametrize("name", SCRIPTS)
def test_account_exclude_time_and_no_partition(name):
    lines = _sbatch(name)
    assert "#SBATCH --account=def-maxwl" in lines
    assert f"#SBATCH {EXCLUDE}" in lines
    assert any(re.fullmatch(r"#SBATCH --time=\d+:\d\d:\d\d", ln) for ln in lines)
    assert not any("--partition" in ln for ln in lines)
    assert not any("--output" in ln or "--error" in ln for ln in lines), "the submitter gives the log path"


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_exclude_in_the_script_names_c371(name):
    for m in re.findall(r"--exclude=(\S+)", _text(name)):
        assert m == "c128,c166,c537,c371", (name, m)


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
    code = _code(name)
    assert not any("/usr/bin/time" in ln for ln in code), "peak memory comes from sacct MaxRSS"
    assert "afterok" not in t
    assert "aftercorr" not in t
    assert "--dependency" not in t, "t125 never chains a job; finished work is checked by files"
    assert "/scratch" not in t, "every output goes under the positional out dir on /project"
    assert "LAW_SCOPE" not in t, "the law scope is fixed in the script, never read from the env"


@pytest.mark.parametrize("name", SCRIPTS)
def test_the_venv_recipe_is_the_pinned_one(name):
    t = _text(name)
    for piece in ('PY_MODULES="python/3.10.13"',
                  'virtualenv --no-download "$SLURM_TMPDIR/venv"',
                  'source "$SLURM_TMPDIR/venv/bin/activate"',
                  'pip install --no-index -r "$KIT/requirements-fir.txt"',
                  'pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"',
                  'ls "$KIT"/wheels/x_transformers-*-py3-none-any.whl',
                  'export PYTHONPATH="$KIT/src"',
                  '"$KIT"/src/*) ;;'):
        assert piece in t, (name, piece)
    # the kit-pin form only: the t125 CLIs put tools/t118 and tools/t125 on sys.path themselves
    assert "$KIT/tools/t118:" not in t and "$KIT/tools/t125:" not in t
    assert ":$KIT/tools" not in t
    # DRY_RUN exits before the module load
    assert t.index('"${DRY_RUN:-0}" = 1') < t.index("module load $PY_MODULES")


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_call_only_t125_clis(name):
    code = "\n".join(_code(name))
    assert "tools/t118/ladder/" not in code, "t125 jobs run the t125 CLIs"
    assert "tools/t125/eta/" in code


@pytest.mark.parametrize("name", SCRIPTS)
def test_headers_document_the_t125_out_dir(name):
    h = _header(name)
    assert f"OUT={OUT}" in h
    assert f"CACHE={CACHE}" in h
    assert EXCLUDE in h
    assert "%40" in h or name == "agg"
    assert "T125_STATUS.md" in h
    # no t118 out dir is ever written: the only t118 paths are the read-only cache and old grid
    for m in re.findall(r"/project/def-maxwl/mforooz/t118/\S*", _text(name)):
        assert m.startswith(CACHE) or m.startswith(OLD_GRID_JSON), (name, m)


@pytest.mark.parametrize("name", ARRAYS)
def test_array_scripts_use_the_t125_table_and_their_own_tasks_tsv(name):
    t = _text(name)
    assert t.count('"$TOOLS/tools/t125/eta/pairs.py" tasks "$TABLE_MANIFEST")') == 1
    assert t.count('python3 "$KIT/tools/t125/eta/pairs.py" tasks "$MANIFEST" > "$TMP_TABLE"') == 1
    assert 'TASKS_TSV="$OUT/tasks.tsv"' in t
    assert "--row" not in t
    # created once, never replaced; a differing table is an error
    assert 'mv -n "$TMP_TABLE" "$TASKS_TSV"' in t
    assert 'cmp -s "$TMP_TABLE" "$TASKS_TSV"' in t


@pytest.mark.parametrize("name", ARRAYS)
def test_headers_document_the_capped_array_submission(name):
    h = _header(name)
    assert "sbatch --test-only --array=0-431%40" in h
    assert "sbatch --parsable" in h


def test_train_cpu_differs_from_train_only_where_it_should():
    gpu, cpu = _text("train"), _text("train_cpu")
    assert "--device cpu" not in gpu.split("set -euo pipefail", 1)[1]
    assert cpu.count('"$IDX" --device cpu') == 1
    assert 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" ' \
           'MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"' in cpu
    body = lambda t: t.split("set -euo pipefail", 1)[1]  # noqa: E731
    norm = (body(cpu).replace(" --device cpu", "").replace("eta_train_cpu", "eta_train")
            .replace('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" '
                     'MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"\n', ""))
    assert norm == body(gpu)


def test_law_carries_the_depth_scope_and_is_submitted_directly():
    t = _text("law")
    code = "\n".join(_code("law"))
    assert code.count("--law-scope depth") == 2   # the DRY_RUN echo and the real call
    assert "--law-scope all" not in code
    assert "SUBMIT THIS ARRAY DIRECTLY" in t
    assert "exit 3" in t


def test_agg_reads_the_old_grid_json_of_t118_row2():
    code = "\n".join(_code("agg"))
    assert f"OLD_GRID_JSON={OLD_GRID_JSON}" in code
    assert '--old-grid-tsv "$OLD_GRID_TSV" --old-grid-json "$OLD_GRID_JSON"' in code


# ---- DRY_RUN ---------------------------------------------------------------------------------

def test_dry_run_train_acceptance_form():
    """Acceptance criterion 7: the kit is this checkout, the manifest falls back to the fixture."""
    r = _dry("train", [str(REPO), "/nonexistent", "/cache", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == _train_lines(str(REPO), "/nonexistent/MANIFEST.tsv", "/cache",
                                                 "/out", 18, "A_C19M16_counts_real_s0")


def test_dry_run_train_from_a_kit_without_tools_falls_back_to_this_checkout():
    r = _dry("train", ["/kit", "/products", "/cache", "/out", "/bl.bed"], 402)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == _train_lines("/kit", "/products/MANIFEST.tsv", "/cache", "/out",
                                                 402, "Aold_C19M16_counts_real_s0", "/bl.bed")


def test_dry_run_train_cpu():
    r = _dry("train_cpu", ["/kit", "/products", "/cache", "/out"], 42)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == _train_lines("/kit", "/products/MANIFEST.tsv", "/cache", "/out",
                                                 42, "A_all_counts_real_s0", cpu=True)


@pytest.mark.parametrize("index", sorted(PINNED) + [0, 383, 384, 431])
def test_dry_run_train_uses_a_real_kit_and_manifest(tmp_path, index):
    products = tmp_path / "products"
    products.mkdir()
    shutil.copy(FIXTURE_MANIFEST, products / "MANIFEST.tsv")
    r = _dry("train", [str(REPO), str(products), "/c", str(tmp_path / "out"), "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    first, last = r.stdout.splitlines()
    assert first.endswith(f"{tmp_path}/out/runs {index}")
    assert last.endswith(f"/bl.bed {tmp_path}/out/runs/{TABLE[index]} --workers 4")


@pytest.mark.parametrize("name", ("train", "train_cpu"))
def test_dry_run_train_skips_a_scored_run(tmp_path, name):
    run = tmp_path / "runs" / PINNED[18]
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry(name, ["/kit", "/products", "/cache", str(tmp_path)])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_dry_run_train_does_not_skip_on_train_done_alone(tmp_path):
    run = tmp_path / "runs" / PINNED[18]
    run.mkdir(parents=True)
    (run / "TRAIN_DONE").touch()
    r = _dry("train", ["/kit", "/products", "/cache", str(tmp_path)])
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 2


@pytest.mark.parametrize("name,args", [
    ("train", ["/kit", "/p", "/c", "/project/def-maxwl/mforooz/t118/row2"]),
    ("train_cpu", ["/kit", "/p", "/c", "/project/def-maxwl/mforooz/t118/ladder"]),
    ("law", ["/kit", "/p", "/c", "/b", "/project/def-maxwl/mforooz/t118"]),
    ("agg", ["/kit", "/p", "/project/def-maxwl/mforooz/t118/row2", "/old.tsv"])])
def test_an_out_dir_under_t118_is_refused(name, args):
    """t125 run names equal t118 row 1's: a t118 SCORE_DONE must never make a t125 task skip."""
    r = _dry(name, args, None if name == "agg" else 18)
    assert r.returncode == 2 and r.stdout == "" and "t118" in r.stderr


def test_dry_run_law_prints_the_law_command_for_task_18():
    r = _dry("law", ["/kit", "/products", "/cache", "/blacklist.bed", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t125/eta/score.py law /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /blacklist.bed /out/runs/A_C19M16_counts_real_s0 "
        "--workers 8 --law-scope depth"]
    assert "exit 3" in r.stderr   # no SCORE_DONE under /out: a real task would fail loudly


def test_dry_run_law_skips_when_law_done(tmp_path):
    run = tmp_path / "runs" / PINNED[402]
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(tmp_path)], 402)
    assert r.returncode == 0 and r.stdout.count("score.py law") == 1 and "exit 3" not in r.stderr
    (run / "LAW_DONE").touch()
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(tmp_path)], 402)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_dry_run_agg():
    r = _dry("agg", ["/kit", "/products", "/out", "/old.tsv"], None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t125/eta/aggregate.py /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /out/runs /out/agg --old-grid-tsv /old.tsv "
        f"--old-grid-json {OLD_GRID_JSON}"]


@pytest.mark.parametrize("name,args", [("train", ["/kit", "/p", "/c", "/o"]),
                                       ("train_cpu", ["/kit", "/p", "/c", "/o"]),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o"])])
def test_array_scripts_refuse_to_run_without_a_task_index(name, args):
    assert _dry(name, args, None).returncode != 0


@pytest.mark.parametrize("name,args", [("train", ["/kit", "/p", "/c", "/o"]),
                                       ("train_cpu", ["/kit", "/p", "/c", "/o"]),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o"])])
@pytest.mark.parametrize("index", (N_TASKS, 1000))
def test_an_index_outside_the_table_fails(name, args, index):
    assert _dry(name, args, index).returncode == 2


@pytest.mark.parametrize("name,args", [("train", ["/kit", "/p", "/c", "/o"]),
                                       ("law", ["/kit", "/p", "/c", "/b", "/o"])])
def test_a_non_numeric_index_fails(name, args):
    assert _dry(name, args, None, SLURM_ARRAY_TASK_ID="18a").returncode == 2


def test_missing_positionals_fail():
    assert _dry("train", ["/kit", "/p", "/c"]).returncode != 0
    assert _dry("train_cpu", ["/kit", "/p", "/c"]).returncode != 0
    assert _dry("law", ["/kit", "/p", "/c", "/o"]).returncode != 0
    assert _dry("agg", ["/kit", "/p", "/o"], None).returncode != 0
