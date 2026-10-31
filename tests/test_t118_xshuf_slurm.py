"""t118 row 2, shuffled-bin twin — the four Nibi job scripts of the twin, as text and in DRY_RUN.

Mirrors tests/test_t118_row2_slurm.py for slurm/t118/xshuf_{train,train_cpu,law,agg}.sh: the
resource lines the hard rules fix (one MIG-slice gres line on the GPU script and none on the CPU
scripts, the account, the excluded nodes, the job names, a walltime, no exported variables, no
/usr/bin/time, no afterok, no aftercorr, no /scratch), the per-task venv recipe, the twin's out dir,
and — through DRY_RUN=1 — that the index -> run mapping comes from `pairs.py tasks --row 2 --xshuf`
and that finished work is skipped. The last block pins that each twin script differs from its
row-2 original only in the expected lines (a line whitelist), so a later edit to one of the pair
cannot drift silently.
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

NAMES = ("train", "train_cpu", "law", "agg")
SCRIPTS = {name: SLURM / f"xshuf_{name}.sh" for name in NAMES}
ROW2 = {name: SLURM / f"row2_{name}.sh" for name in NAMES}
ARRAYS = ("train", "train_cpu", "law")
CPU = ("train_cpu", "law", "agg")
GRES = "#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
BLACKLIST = "/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed"
OUT = "/project/def-maxwl/mforooz/t118/row2_xshuf"
ROW2_OUT = "/project/def-maxwl/mforooz/t118/row2"
CACHE = "/project/def-maxwl/mforooz/t118/ladder/cache"
JOB_NAMES = {"train": "t118L_xstrain", "train_cpu": "t118L_xstraincpu", "law": "t118L_xslaw",
             "agg": "t118L_xsagg"}
ROW2_JOB_NAMES = {"train": "t118L_r2train", "train_cpu": "t118L_r2traincpu", "law": "t118L_r2law",
                  "agg": "t118L_r2agg"}
RESOURCES = {"train": ("3:00:00", "4", "16000M"), "train_cpu": ("8:00:00", "16", "64000M"),
             "law": ("3:00:00", "8", "24000M"), "agg": ("2:00:00", "4", "16000M")}
N_TASKS = 192
# the four twin rungs at g C19M16, counts, seed 0 (the smoke indices)
SMOKE = {18: "A2_C19M16_counts_xshuf_s0", 66: "B2_C19M16_counts_xshuf_s0",
         114: "C2_C19M16_counts_xshuf_s0", 162: "D2_C19M16_counts_xshuf_s0"}


def _text(name: str) -> str:
    return SCRIPTS[name].read_text(encoding="utf-8")


def _row2_text(name: str) -> str:
    return ROW2[name].read_text(encoding="utf-8")


def _sbatch(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]


def _header(name: str) -> str:
    """The comment block above `set -euo pipefail`."""
    return _text(name).split("set -euo pipefail")[0]


def _body(text: str) -> str:
    return text.split("set -euo pipefail", 1)[1]


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
        f"python3 {kit}/tools/t118/ladder/train.py train-index {manifest} "
        f"{kit}/tools/t118/covariates.tsv {cache} {out}/runs {index} --row 2 --xshuf{dev}",
        f"python3 {kit}/tools/t118/ladder/score.py trained {manifest} "
        f"{kit}/tools/t118/covariates.tsv {cache} {blacklist} {out}/runs/{run} --workers 4",
    ]


# ---- text checks -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", NAMES)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_the_gpu_script_has_exactly_the_mig_slice_gres_line():
    gres = [ln for ln in _text("train").splitlines() if "gres" in ln]
    assert gres == [GRES]
    assert _text("train").count("SBATCH --gres") == 1


@pytest.mark.parametrize("name", CPU)
def test_cpu_scripts_request_no_gres(name):
    assert "SBATCH --gres" not in _text(name)
    assert not any("gres" in ln for ln in _sbatch(_text(name)))


@pytest.mark.parametrize("name", NAMES)
def test_account_exclude_time_and_no_partition(name):
    lines = _sbatch(_text(name))
    assert "#SBATCH --account=def-maxwl" in lines
    assert "#SBATCH --exclude=c128,c166,c537" in lines
    assert any(re.fullmatch(r"#SBATCH --time=\d+:\d\d:\d\d", ln) for ln in lines)
    assert not any("--partition" in ln for ln in lines)
    assert not any("--output" in ln or "--error" in ln for ln in lines), "the submitter gives the log path"


@pytest.mark.parametrize("name", NAMES)
def test_resources(name):
    time, cpus, mem = RESOURCES[name]
    lines = _sbatch(_text(name))
    assert f"#SBATCH --time={time}" in lines
    assert f"#SBATCH --cpus-per-task={cpus}" in lines
    assert f"#SBATCH --mem={mem}" in lines


def test_job_names():
    names = {}
    for name in NAMES:
        found = re.findall(r"job-name=(\S+)", _text(name))
        assert len(found) == 1, (name, found)
        names[name] = found[0]
    assert names == JOB_NAMES
    assert all(n.startswith("t118L_") for n in names.values())


@pytest.mark.parametrize("name", NAMES)
def test_house_rules(name):
    t = _text(name)
    assert t.startswith("#!/bin/bash\n")
    assert "set -euo pipefail" in t
    assert "--export=" not in t
    code = [ln for ln in t.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("/usr/bin/time" in ln for ln in code), "peak memory comes from sacct MaxRSS"
    assert "afterok" not in t
    assert "aftercorr" not in t, "the twin submits the law array directly"
    assert "/scratch" not in t, "every output goes under the positional out dir on /project"


@pytest.mark.parametrize("name", NAMES)
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
    # the venv block is the row-2 script's, line for line
    block = lambda s: s[s.index("set +u; module load"):s.index("*) echo \"candi does not import")]  # noqa: E731
    assert block(t) == block(_row2_text(name))


@pytest.mark.parametrize("name", ARRAYS)
def test_headers_document_the_twin_out_dir(name):
    h = _header(name)
    assert f"OUT={OUT}" in h
    assert f"OUT={ROW2_OUT}\n" not in h and f"OUT={ROW2_OUT};" not in h
    assert f"CACHE={CACHE}" in h
    assert "--exclude=c128,c166,c537" in h


def test_agg_header_documents_the_row2_out_dir_and_both_runs_dirs():
    h = _header("agg")
    assert f"OUT={ROW2_OUT}\n" in h, "the twin's aggregate writes into the row-2 agg dir"
    assert f"XSHUF_RUNS={OUT}/runs" in h
    assert "ROW1_RUNS=/project/def-maxwl/mforooz/t118/ladder/runs" in h
    assert "--exclude=c128,c166,c537" in h
    assert "row2_agg.sh" in h and "afterany" in h and "all scored" in h


def test_train_header_documents_the_pool_read_and_the_smoke_indices():
    h = _header("train")
    assert "--array=18,66,114,162%4" in h
    assert "60 GB" in h and "minutes" in h
    assert "192 tasks" in h


@pytest.mark.parametrize("name", ARRAYS)
def test_array_scripts_use_the_twin_table_and_their_own_tasks_tsv(name):
    t = _text(name)
    assert t.count('pairs.py" tasks "$TABLE_MANIFEST" --row 2 --xshuf)') == 1
    assert t.count('pairs.py" tasks "$MANIFEST" --row 2 --xshuf > "$TMP_TABLE"') == 1
    assert 'TASKS_TSV="$OUT/tasks.tsv"' in t
    assert re.search(r'pairs\.py" tasks "\$[A-Z_]+"(?! --row 2 --xshuf)', t) is None


@pytest.mark.parametrize("name", ("train", "train_cpu"))
def test_train_scripts_call_train_index_with_the_twin_flags(name):
    t = _text(name)
    dev = " --device cpu" if name == "train_cpu" else ""
    assert t.count(f'"$IDX" --row 2 --xshuf{dev}\n') == 1
    assert t.count(f'$IDX --row 2 --xshuf{dev}"\n') == 1


@pytest.mark.parametrize("name", ARRAYS)
def test_headers_document_the_capped_array_submission(name):
    h = _header(name)
    assert "sbatch --test-only --array=0-191%40" in h
    assert "sbatch --parsable --array=0-191%40" in h
    assert "0-575" not in h


def test_train_cpu_differs_from_train_only_where_it_should():
    gpu, cpu = _text("train"), _text("train_cpu")
    assert "--device cpu" not in gpu
    assert cpu.count('"$IDX" --row 2 --xshuf --device cpu') == 1
    assert 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" ' \
           'MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"' in cpu
    norm = (_body(cpu).replace(" --device cpu", "").replace("xshuf_train_cpu", "xshuf_train")
            .replace('export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" '
                     'MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"\n', ""))
    assert norm == _body(gpu)


def test_law_is_submitted_directly_and_agg_is_chained_afterany():
    assert "--dependency" not in _text("law")
    assert "SUBMIT THIS ARRAY DIRECTLY" in _text("law")
    assert "status file" in _text("law")
    assert "--dependency=afterany:$PREV" in _text("agg")


# ---- the twin scripts differ from the row-2 scripts only where they should -------------------

# (row-2 text, twin text) pairs: the only body edits allowed. Everything else must be identical.
_BODY_EDITS = {
    "train": [('usage: row2_train.sh', 'usage: xshuf_train.sh'),
              ('=== t118 row2_train idx', '=== t118 xshuf_train idx'),
              ('"$TABLE_MANIFEST" --row 2)', '"$TABLE_MANIFEST" --row 2 --xshuf)'),
              ('$IDX --row 2"', '$IDX --row 2 --xshuf"'),
              ('"$MANIFEST" --row 2 > "$TMP_TABLE"', '"$MANIFEST" --row 2 --xshuf > "$TMP_TABLE"'),
              ('"$IDX" --row 2\n', '"$IDX" --row 2 --xshuf\n')],
    "train_cpu": [('usage: row2_train_cpu.sh', 'usage: xshuf_train_cpu.sh'),
                  ('=== t118 row2_train_cpu idx', '=== t118 xshuf_train_cpu idx'),
                  ('"$TABLE_MANIFEST" --row 2)', '"$TABLE_MANIFEST" --row 2 --xshuf)'),
                  ('$IDX --row 2 --device cpu"', '$IDX --row 2 --xshuf --device cpu"'),
                  ('"$MANIFEST" --row 2 > "$TMP_TABLE"',
                   '"$MANIFEST" --row 2 --xshuf > "$TMP_TABLE"'),
                  ('"$IDX" --row 2 --device cpu\n', '"$IDX" --row 2 --xshuf --device cpu\n')],
    "law": [('usage: row2_law.sh', 'usage: xshuf_law.sh'),
            ('=== t118 row2_law idx', '=== t118 xshuf_law idx'),
            ('"$TABLE_MANIFEST" --row 2)', '"$TABLE_MANIFEST" --row 2 --xshuf)'),
            ('"$MANIFEST" --row 2 > "$TMP_TABLE"', '"$MANIFEST" --row 2 --xshuf > "$TMP_TABLE"')],
    "agg": [('usage: row2_agg.sh <kit_dir> <products_dir> <out_dir> <refs_tsv> <row1_runs_dir> <rung',
             'usage: xshuf_agg.sh <kit_dir> <products_dir> <row2_out_dir> <refs_tsv> <row1_runs_dir> '
             '<xshuf_runs_dir> <rung'),
            ('RUNG="${6:?$USAGE}"\n', 'XSHUF_RUNS="${6:?$USAGE}"\nRUNG="${7:?$USAGE}"\n'),
            ('--also-runs $ROW1_RUNS --rows 1,2"',
             '--also-runs $ROW1_RUNS --also-runs $XSHUF_RUNS --rows 1,2 --xshuf"'),
            ('--also-runs "$ROW1_RUNS" --rows 1,2\n',
             '--also-runs "$ROW1_RUNS" --also-runs "$XSHUF_RUNS" --rows 1,2 --xshuf\n'),
            ('{ echo "no row-1 runs dir $ROW1_RUNS" >&2; exit 2; }\n',
             '{ echo "no row-1 runs dir $ROW1_RUNS" >&2; exit 2; }\n'
             '[ -d "$XSHUF_RUNS" ] || { echo "no twin runs dir $XSHUF_RUNS" >&2; exit 2; }\n'),
            ('=== t118 row2_agg rung', '=== t118 xshuf_agg rung')],
}


@pytest.mark.parametrize("name", NAMES)
def test_body_differs_from_row2_only_in_the_whitelisted_lines(name):
    row2, twin = _body(_row2_text(name)), _body(_text(name))
    expected = row2
    for old, new in _BODY_EDITS[name]:
        assert row2.count(old) == 1, (name, old)
        expected = expected.replace(old, new)
    assert twin == expected


@pytest.mark.parametrize("name", NAMES)
def test_sbatch_lines_differ_from_row2_only_in_the_job_name(name):
    row2 = [ln.replace(ROW2_JOB_NAMES[name], JOB_NAMES[name]) for ln in _sbatch(_row2_text(name))]
    assert _sbatch(_text(name)) == row2


def test_row2_scripts_are_not_the_twin_scripts():
    for n in NAMES:
        assert "xshuf" not in _row2_text(n)


# ---- the task table --------------------------------------------------------------------------

def test_twin_task_table_from_the_fixture_manifest():
    rows = pairs.read_manifest(FIXTURE_MANIFEST)
    t = pairs.tasks(rows, row=2, xshuf=True)
    assert len(t) == N_TASKS
    assert t[0]["run_name"] == "A2_C07M20_counts_xshuf_s0"
    assert t[-1]["run_name"] == "D2_all_pval_xshuf_s2"
    assert {i: t[i]["run_name"] for i in SMOKE} == SMOKE


def test_the_pairs_cli_prints_the_twin_table():
    r = subprocess.run([sys.executable, str(REPO / "tools/t118/ladder/pairs.py"), "tasks",
                        str(FIXTURE_MANIFEST), "--row", "2", "--xshuf"], capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=120)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[19] == "18\tA2\tC19M16\tcounts\txshuf\t0\tA2_C19M16_counts_xshuf_s0"
    head = lines[0].split("\t")
    by_index = {int(ln.split("\t")[head.index("index")]): ln.split("\t")[head.index("run_name")]
                for ln in lines[1:]}
    assert len(by_index) == N_TASKS and {i: by_index[i] for i in SMOKE} == SMOKE


# ---- DRY_RUN ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ("train", "train_cpu"))
def test_dry_run_train_prints_train_then_score_for_task_18(name):
    r = _dry(name, ["/kit", "/products", "/cache", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == _train_lines("/kit", "/products/MANIFEST.tsv", "/cache",
                                                 "/out", 18, SMOKE[18], cpu=name == "train_cpu")


def test_dry_run_acceptance_line():
    # the plan's acceptance check: the kit is this checkout, products do not exist
    r = _dry("train", [str(REPO), "/nonexistent", "/cache", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == _train_lines(str(REPO), "/nonexistent/MANIFEST.tsv", "/cache",
                                                 "/out", 18, "A2_C19M16_counts_xshuf_s0")


@pytest.mark.parametrize("index", sorted(SMOKE))
def test_dry_run_train_uses_a_real_kit_and_manifest(tmp_path, index):
    products = tmp_path / "products"
    products.mkdir()
    shutil.copy(FIXTURE_MANIFEST, products / "MANIFEST.tsv")
    r = _dry("train", [str(REPO), str(products), "/c", str(tmp_path / "out"), "/bl.bed"], index)
    assert r.returncode == 0, r.stderr
    first, last = r.stdout.splitlines()
    assert first.endswith(f"{tmp_path}/out/runs {index} --row 2 --xshuf")
    assert last.endswith(f"/bl.bed {tmp_path}/out/runs/{SMOKE[index]} --workers 4")


@pytest.mark.parametrize("name", ("train", "train_cpu"))
def test_dry_run_train_skips_a_scored_run(tmp_path, name):
    run = tmp_path / "runs" / SMOKE[18]
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry(name, ["/kit", "/products", "/cache", str(tmp_path)])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


def test_dry_run_train_does_not_skip_on_the_row2_real_run(tmp_path):
    run = tmp_path / "runs" / "A2_C19M16_counts_real_s0"
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("train", ["/kit", "/products", "/cache", str(tmp_path)])
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 2


def test_dry_run_law_prints_the_law_command_for_task_18():
    r = _dry("law", ["/kit", "/products", "/cache", "/blacklist.bed", "/out"])
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "python3 /kit/tools/t118/ladder/score.py law /products/MANIFEST.tsv "
        "/kit/tools/t118/covariates.tsv /cache /blacklist.bed /out/runs/A2_C19M16_counts_xshuf_s0 "
        "--workers 8"]
    assert "exit 3" in r.stderr   # no SCORE_DONE under /out: a real task would fail loudly


def test_dry_run_law_for_task_162_is_d2():
    r = _dry("law", ["/kit", "/products", "/cache", "/blacklist.bed", "/out"], 162)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith(f"/out/runs/{SMOKE[162]} --workers 8")


def test_dry_run_law_skips_when_law_done(tmp_path):
    run = tmp_path / "runs" / SMOKE[18]
    run.mkdir(parents=True)
    (run / "SCORE_DONE").touch()
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(tmp_path)])
    assert r.returncode == 0 and r.stdout.count("score.py law") == 1 and "exit 3" not in r.stderr
    (run / "LAW_DONE").touch()
    r = _dry("law", ["/kit", "/products", "/cache", "/bl.bed", str(tmp_path)])
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


_AGG_ARGS = ["/kit", "/products", "/out", "/refs.tsv", "/row1/runs", "/xshuf/runs"]
_AGG_LINE = ("python3 /kit/tools/t118/ladder/aggregate.py /products/MANIFEST.tsv "
             "/kit/tools/t118/covariates.tsv /out/runs /refs.tsv /out/agg "
             "--also-runs /row1/runs --also-runs /xshuf/runs --rows 1,2 --xshuf")


@pytest.mark.parametrize("rung", ("A2", "B2", "C2", "D2"))
def test_dry_run_agg_rung(rung):
    r = _dry("agg", [*_AGG_ARGS, rung], None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        _AGG_LINE,
        "python3 /kit/tools/t118/ladder/aggregate.py qm-curves /products/MANIFEST.tsv /products "
        "/out/agg/qm_curves.json",
        f"python3 /kit/tools/t118/ladder/figures.py /out/agg {rung} --refs-qm /out/agg/qm_curves.json",
        f"python3 /kit/tools/t118/ladder/report.py /out/agg {rung}",
    ]


def test_dry_run_agg_grid():
    r = _dry("agg", [*_AGG_ARGS, "grid"], None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [_AGG_LINE, "python3 /kit/tools/t118/ladder/report.py grid /out/agg"]


@pytest.mark.parametrize("rung", ("A", "D", "E2", "x"))
def test_agg_rejects_an_unknown_rung(rung):
    r = _dry("agg", [*_AGG_ARGS, rung], None)
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
    assert _dry(name, args, N_TASKS).returncode == 2


def test_missing_positionals_fail():
    assert _dry("train", ["/kit", "/p", "/c"]).returncode != 0
    assert _dry("train_cpu", ["/kit", "/p", "/c"]).returncode != 0
    assert _dry("law", ["/kit", "/p", "/c", "/o"]).returncode != 0
    # the row-2 agg call shape (no twin runs dir) is refused: A2 lands in the runs-dir slot
    assert _dry("agg", ["/kit", "/p", "/o", "/r", "/row1", "A2"], None).returncode != 0
