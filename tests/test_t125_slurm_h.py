"""t125 part 2 — the five Nibi job scripts of the map h, checked as text and in DRY_RUN.

Mirrors tests/test_t125_slurm_eta.py for slurm/t125/h_{ctl_cache,train,alone,chain,agg}.sh: the
resource lines the hard rules fix (one MIG-slice gres line on h_chain.sh and on h_train.sh, the
latter since t125-P2h moved the fnb fits onto the slice, and none elsewhere, the
account, the excluded nodes incl. c371, the job names, a walltime, no exported variables, no
/usr/bin/time, no afterok, no /scratch), the per-task venv recipe and kit pin, the t125 out dirs,
and — through DRY_RUN=1 — that the index -> h / f mapping comes from `tools/t125/hmap/tasks.py`
(the expected names are written out independently below), that each h name becomes the right
h_spec, and that finished work is skipped.
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

SCRIPTS = {name: SLURM / f"h_{name}.sh" for name in ("ctl_cache", "train", "alone", "chain", "agg")}
ARRAYS = ("ctl_cache", "train", "alone", "chain")
CPU = ("ctl_cache", "alone", "agg")
GPU = ("train", "chain")
GRES_VALUE = "--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
GRES = f"#SBATCH {GRES_VALUE}"
EXCLUDE = "--exclude=c128,c166,c537,c371"
OUT_H = "/project/def-maxwl/mforooz/t125/h"
CTL = "/project/def-maxwl/mforooz/t125/ctl_cache"
CACHE = "/project/def-maxwl/mforooz/t118/ladder/cache"
OLD_GRID_JSON = "/project/def-maxwl/mforooz/t118/row2/agg/results.json"
JOB_NAMES = {"ctl_cache": "t125_hctl", "train": "t125_htrain", "alone": "t125_halone",
             "chain": "t125_hchain", "agg": "t125_hagg"}
RESOURCES = {"ctl_cache": ("0:30:00", "2", "8000M"), "train": ("4:00:00", "8", "24000M"),
             "alone": ("2:00:00", "8", "32000M"), "chain": ("6:00:00", "4", "32000M"),
             "agg": ("2:00:00", "4", "16000M")}
COV = "/kit/tools/t118/covariates.tsv"

# ---- the plan's tables, written out here (not imported) --------------------------------------

G_IDS = ("C07M20", "C07M29", "C12M02", "C19M16", "C19M22", "C40M17", "C40M18", "all")
F_RUNS = [f"{rung}_{g}_counts_real_s0" for rung in ("A", "B", "C", "D", "A2", "B2", "C2", "D2")
          for g in G_IDS]                                     # eta.pairs.f_runs_seed0, 64
H_LEARNED = ([(v, r, n) for v in ("affine", "mlp") for r in (1, 5) for n in (0, 1)]
             + [(v, 5, n) for v in ("formula_affine", "formula_mlp") for n in (0, 1)])


def _h(variant, rung, noise, tag, seed):
    return f"h_{variant}_r{rung}_{'noise' if noise else 'push'}_{tag}_s{seed}"


TRUE_H = [_h(v, r, n, "true", s) for (v, r, n) in H_LEARNED for s in (0, 1, 2)]       # 36


def _fnb_h(f_run, seeds=(0,)):
    return [_h(v, r, n, f_run, s) for (v, r, n) in H_LEARNED for s in seeds]


ALONE = ["formula:bin", "formula:frag"] + TRUE_H                                       # 38


def _chain_specs(f_run, out, seeds=(0,)):
    return (["formula:bin", "formula:frag"] + [f"{out}/runs_true/{h}" for h in TRUE_H]
            + [f"{out}/runs_fnb/{h}" for h in _fnb_h(f_run, seeds)])


# ---- helpers ---------------------------------------------------------------------------------

def _text(name: str) -> str:
    return SCRIPTS[name].read_text(encoding="utf-8")


def _sbatch(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if ln.startswith("#SBATCH")]


def _header(name: str) -> str:
    return _text(name).split("set -euo pipefail")[0]


def _code(name: str) -> list[str]:
    return [ln for ln in _text(name).splitlines() if not ln.lstrip().startswith("#")]


def _dry(name: str, args: list[str], index: int | None = 0, **env) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if k not in ("SLURM_ARRAY_TASK_ID", "DRY_RUN")}
    e.update(DRY_RUN="1", PYTHON=sys.executable, PYTHONPATH=str(REPO / "src"), **env)
    if index is not None:
        e["SLURM_ARRAY_TASK_ID"] = str(index)
    return subprocess.run(["bash", str(SCRIPTS[name]), *args], env=e, capture_output=True,
                          text=True, timeout=120)


def _train_line(table, index, out="/out", extra=""):
    return (f"python3 /kit/tools/t125/hmap/train.py train-index /p/MANIFEST.tsv {COV} /c /ctl "
            f"{out}/runs_{table} {table} {index}{extra}")


def _alone_line(spec, out="/out"):
    return (f"python3 /kit/tools/t125/hmap/score.py alone /p/MANIFEST.tsv {COV} /c /ctl /bl.bed "
            f"{spec} {out}/alone --workers 8")


def _chain_args(out="/out", eta="/eta", seeds=None):
    return ["/kit", "/p", "/c", "/ctl", "/bl.bed", eta, out] + ([str(seeds)] if seeds else [])


def _chain_split(line: str):
    """(the words before the first --h, the --h specs in order, the words after the last spec)."""
    words = line.split()
    head = words[:words.index("--h")]
    specs, i = [], len(head)
    while i < len(words) and words[i] == "--h":
        specs.append(words[i + 1])
        i += 2
    return head, specs, words[i:]


# ---- the tables the scripts read -------------------------------------------------------------

def test_the_tasks_cli_gives_the_written_out_tables():
    def run(*a):
        r = subprocess.run([sys.executable, str(REPO / "tools" / "t125" / "hmap" / "tasks.py"), *a],
                           capture_output=True, text=True, timeout=120,
                           env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    true = run("true", str(FIXTURE_MANIFEST))
    assert [ln.split("\t")[-1] for ln in true[1:]] == TRUE_H
    fnb = run("fnb", str(FIXTURE_MANIFEST))
    assert len(fnb) - 1 == 768
    assert [ln.split("\t")[-1] for ln in fnb[1:]] == [h for f in F_RUNS for h in _fnb_h(f)]
    alone = run("alone", str(FIXTURE_MANIFEST))
    assert [ln.split("\t")[4] for ln in alone[1:3]] == ["formula:bin", "formula:frag"]
    assert [ln.split("\t")[-1] for ln in alone[3:]] == TRUE_H


# ---- text checks -----------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    r = subprocess.run(["bash", "-n", str(SCRIPTS[name])], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize("name", GPU)
def test_the_gpu_scripts_request_one_mig_slice(name):
    """Acceptance criterion 8 as amended by t125-P2h: `grep -c 'SBATCH --gres'` is 1 on
    h_chain.sh and h_train.sh and 0 elsewhere."""
    assert [ln for ln in _sbatch(name) if "gres" in ln] == [GRES]
    assert _text(name).count("SBATCH --gres") == 1


@pytest.mark.parametrize("name", CPU)
def test_cpu_scripts_request_no_gres(name):
    assert _text(name).count("SBATCH --gres") == 0
    assert not any("gres" in ln for ln in _sbatch(name))


def test_h_train_documents_why_it_requests_the_slice():
    h = _header("train")
    assert "t125-P2h" in h and "MIG slice" in h
    assert "GPU-SLICE VARIANT" not in h


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
    assert not any("/usr/bin/time" in ln for ln in _code(name)), "peak memory comes from sacct MaxRSS"
    assert "afterok" not in t and "aftercorr" not in t
    assert "--dependency" not in t, "t125 never chains a job; finished work is checked by files"
    assert "/scratch" not in t, "every output goes under the positional out dirs on /project"


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
    assert "$KIT/tools/t118:" not in t and "$KIT/tools/t125:" not in t
    assert ":$KIT/tools" not in t
    assert t.index('"${DRY_RUN:-0}" = 1') < t.index("module load $PY_MODULES")


@pytest.mark.parametrize("name", SCRIPTS)
def test_scripts_call_only_t125_hmap_clis(name):
    code = "\n".join(_code(name))
    assert "tools/t118/ladder/" not in code
    assert "tools/t125/hmap/" in code


@pytest.mark.parametrize("name", SCRIPTS)
def test_headers_document_the_t125_out_dirs(name):
    h = _header(name)
    assert f"OUT_H={OUT_H}" in h
    assert f"CTL={CTL}" in h
    assert EXCLUDE in h
    assert "T125_STATUS.md" in h
    assert "sbatch --test-only" in h and "sbatch --parsable" in h
    for m in re.findall(r"/project/def-maxwl/mforooz/t118/\S*", _text(name)):
        assert m.startswith(CACHE) or m.startswith(OLD_GRID_JSON), (name, m)


@pytest.mark.parametrize("name,table", [("train", "tasks_true.tsv"), ("alone", "tasks_alone.tsv"),
                                        ("chain", "tasks_chain.tsv")])
def test_array_scripts_keep_one_shared_table(name, table):
    t = _text(name)
    assert table in t
    assert 'mv -n "$TMP_TABLE" "$TASKS_TSV"' in t
    assert 'cmp -s "$TMP_TABLE" "$TASKS_TSV"' in t


def test_h_agg_reads_the_old_grid_json_of_t118_row2():
    code = "\n".join(_code("agg"))
    assert f"OLD_GRID_JSON={OLD_GRID_JSON}" in code
    assert '--old-grid-tsv "$OLD_GRID_TSV" --old-grid-json "$OLD_GRID_JSON" --eta-runs "$ETA_RUNS"' in code


@pytest.mark.parametrize("name", ARRAYS)
def test_array_scripts_exit_3_or_rely_on_the_cli_for_unfinished_inputs(name):
    t = _text(name)
    if name in ("train", "alone", "chain"):
        assert "exit 3" in t and "TRAIN_DONE" in t


# ---- DRY_RUN: h_ctl_cache.sh -----------------------------------------------------------------

@pytest.mark.parametrize("index", (0, 5, 129))
def test_dry_run_ctl_cache(index):
    r = _dry("ctl_cache", ["/kit", "/p", "/ctl"], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        f"python3 /kit/tools/t125/hmap/controls.py cache /p/MANIFEST.tsv /p /ctl {index}"]


def test_dry_run_ctl_cache_reads_a_real_manifest(tmp_path):
    products = tmp_path / "products"
    products.mkdir()
    shutil.copy(FIXTURE_MANIFEST, products / "MANIFEST.tsv")
    assert _dry("ctl_cache", [str(REPO), str(products), "/ctl"], 129).returncode == 0
    assert _dry("ctl_cache", [str(REPO), str(products), "/ctl"], 130).returncode == 2


@pytest.mark.parametrize("index", (130, 1000))
def test_dry_run_ctl_cache_index_outside_the_manifest_fails(index):
    assert _dry("ctl_cache", ["/kit", "/p", "/ctl"], index).returncode == 2


# ---- DRY_RUN: h_train.sh ---------------------------------------------------------------------

@pytest.mark.parametrize("index", (0, 8, 35))
def test_dry_run_train_true(index):
    r = _dry("train", ["/kit", "/p", "/c", "/ctl", "/out", "true"], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [_train_line("true", index)]


@pytest.mark.parametrize("seeds,index", [(1, 0), (1, 767), (3, 2303)])
def test_dry_run_train_fnb(seeds, index):
    r = _dry("train", ["/kit", "/p", "/c", "/ctl", "/out", "fnb", "/eta", str(seeds)], index)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        _train_line("fnb", index, extra=f" --eta-runs /eta --fnb-seeds {seeds}")]
    f_run = F_RUNS[index // (12 * seeds)]
    assert f"/eta/{f_run}/TRAIN_DONE" in r.stderr and "exit 3" in r.stderr


def test_dry_run_train_fnb_seeds_default_to_one():
    r = _dry("train", ["/kit", "/p", "/c", "/ctl", "/out", "fnb", "/eta"], 3)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [_train_line("fnb", 3, extra=" --eta-runs /eta --fnb-seeds 1")]


def test_dry_run_train_fnb_no_note_when_the_f_run_is_trained(tmp_path):
    f = tmp_path / "eta" / F_RUNS[0]
    f.mkdir(parents=True)
    (f / "TRAIN_DONE").touch()
    r = _dry("train", ["/kit", "/p", "/c", "/ctl", "/out", "fnb", str(tmp_path / "eta"), "1"], 0)
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 1 and "exit 3" not in r.stderr


@pytest.mark.parametrize("table,index,h", [("true", 8, TRUE_H[8]),
                                           ("fnb", 13, _fnb_h(F_RUNS[1])[1])])
def test_dry_run_train_skips_a_trained_h(tmp_path, table, index, h):
    run = tmp_path / f"runs_{table}" / h
    run.mkdir(parents=True)
    (run / "TRAIN_DONE").touch()
    args = ["/kit", "/p", "/c", "/ctl", str(tmp_path), table] + (["/eta"] if table == "fnb" else [])
    r = _dry("train", args, index)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


@pytest.mark.parametrize("args,index", [
    (["/kit", "/p", "/c", "/ctl", "/out", "true"], 36),
    (["/kit", "/p", "/c", "/ctl", "/out", "fnb", "/eta", "1"], 768),
    (["/kit", "/p", "/c", "/ctl", "/out", "fnb", "/eta", "3"], 2304),
    (["/kit", "/p", "/c", "/ctl", "/out", "fnb"], 0),                 # fnb needs eta_runs_dir
    (["/kit", "/p", "/c", "/ctl", "/out", "fnb", "/eta", "2"], 0),    # fnb_seeds 1 or 3
    (["/kit", "/p", "/c", "/ctl", "/out", "true", "/eta"], 0),        # true takes no eta runs
    (["/kit", "/p", "/c", "/ctl", "/out", "law"], 0)])                # table true or fnb
def test_dry_run_train_bad_input_fails(args, index):
    assert _dry("train", args, index).returncode == 2


# ---- DRY_RUN: h_alone.sh ---------------------------------------------------------------------

@pytest.mark.parametrize("index", (0, 1, 2, 37))
def test_dry_run_alone(index):
    r = _dry("alone", ["/kit", "/p", "/c", "/ctl", "/bl.bed", "/out"], index)
    assert r.returncode == 0, r.stderr
    spec = ALONE[index] if index < 2 else f"/out/runs_true/{ALONE[index]}"
    assert r.stdout.splitlines() == [_alone_line(spec)]
    assert ("exit 3" in r.stderr) == (index >= 2)   # formulas need no trained h


def test_dry_run_alone_pinned_names():
    assert _dry("alone", ["/kit", "/p", "/c", "/ctl", "/bl.bed", "/out"], 0).stdout.split()[8] \
        == "formula:bin"
    assert _dry("alone", ["/kit", "/p", "/c", "/ctl", "/bl.bed", "/out"], 2).stdout.split()[8] \
        == "/out/runs_true/h_affine_r1_push_true_s0"


def test_dry_run_alone_skips_a_scored_h_and_needs_no_note_when_trained(tmp_path):
    h = TRUE_H[0]
    (tmp_path / "runs_true" / h).mkdir(parents=True)
    (tmp_path / "runs_true" / h / "TRAIN_DONE").touch()
    args = ["/kit", "/p", "/c", "/ctl", "/bl.bed", str(tmp_path)]
    r = _dry("alone", args, 2)
    assert r.returncode == 0 and len(r.stdout.splitlines()) == 1 and "exit 3" not in r.stderr
    (tmp_path / "alone" / h).mkdir(parents=True)
    (tmp_path / "alone" / h / "ALONE_DONE").touch()
    r = _dry("alone", args, 2)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr
    (tmp_path / "alone" / "h_formula_bin").mkdir(parents=True)
    (tmp_path / "alone" / "h_formula_bin" / "ALONE_DONE").touch()
    r = _dry("alone", args, 0)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


@pytest.mark.parametrize("index", (38, 1000))
def test_dry_run_alone_index_outside_the_table_fails(index):
    assert _dry("alone", ["/kit", "/p", "/c", "/ctl", "/bl.bed", "/out"], index).returncode == 2


# ---- DRY_RUN: h_chain.sh ---------------------------------------------------------------------

def test_dry_run_chain_index_0_pinned():
    """chain index 0 -> f A_C07M20_counts_real_s0 with 50 --h entries at fnb_seeds 1."""
    r = _dry("chain", _chain_args(seeds=1), 0)
    assert r.returncode == 0, r.stderr
    (line,) = r.stdout.splitlines()
    head, specs, tail = _chain_split(line)
    assert head == ["python3", "/kit/tools/t125/hmap/score.py", "chain", "/p/MANIFEST.tsv", COV,
                    "/c", "/ctl", "/bl.bed", "/eta/A_C07M20_counts_real_s0", "/out/chain"]
    assert len(specs) == 50
    assert specs == _chain_specs("A_C07M20_counts_real_s0", "/out")
    assert tail == ["--workers", "4"]


def test_dry_run_chain_learned_formula_h_are_run_dirs_not_formulas():
    _, specs, _ = _chain_split(_dry("chain", _chain_args(), 0).stdout)
    assert [s for s in specs if s.startswith("formula:")] == ["formula:bin", "formula:frag"]
    assert "/out/runs_true/h_formula_affine_r5_push_true_s0" in specs
    assert "/out/runs_fnb/h_formula_mlp_r5_noise_A_C07M20_counts_real_s0_s0" in specs


@pytest.mark.parametrize("index", (3, 63))
def test_dry_run_chain_f_runs_follow_f_runs_seed0(index):
    r = _dry("chain", _chain_args(), index)
    assert r.returncode == 0, r.stderr
    head, specs, _ = _chain_split(r.stdout)
    assert head[8] == f"/eta/{F_RUNS[index]}"
    assert specs == _chain_specs(F_RUNS[index], "/out")


def test_dry_run_chain_three_fnb_seeds():
    _, specs, _ = _chain_split(_dry("chain", _chain_args(seeds=3), 0).stdout)
    assert len(specs) == 2 + 36 + 36
    assert specs == _chain_specs(F_RUNS[0], "/out", seeds=(0, 1, 2))


def test_dry_run_chain_notes_missing_train_done_then_skips_on_chain_done(tmp_path):
    out, eta = tmp_path / "h", tmp_path / "eta"
    args = _chain_args(out=str(out), eta=str(eta), seeds=1)
    r = _dry("chain", args, 0)
    assert r.returncode == 0 and "48 h run dirs lack TRAIN_DONE" in r.stderr
    assert f"{eta}/{F_RUNS[0]}/TRAIN_DONE" in r.stderr
    for spec in _chain_specs(F_RUNS[0], str(out)):
        if not spec.startswith("formula:"):
            Path(spec).mkdir(parents=True)
            (Path(spec) / "TRAIN_DONE").touch()
    (eta / F_RUNS[0]).mkdir(parents=True)
    (eta / F_RUNS[0] / "TRAIN_DONE").touch()
    r = _dry("chain", args, 0)
    assert r.returncode == 0 and "exit 3" not in r.stderr and len(r.stdout.splitlines()) == 1
    (out / "chain" / F_RUNS[0]).mkdir(parents=True)
    (out / "chain" / F_RUNS[0] / "CHAIN_DONE").touch()
    r = _dry("chain", args, 0)
    assert r.returncode == 0 and r.stdout == "" and "skip" in r.stderr


@pytest.mark.parametrize("args,index", [(_chain_args(), 64), (_chain_args(), 1000),
                                        (_chain_args(seeds=2), 0)])
def test_dry_run_chain_bad_input_fails(args, index):
    assert _dry("chain", args, index).returncode == 2


# ---- DRY_RUN: h_agg.sh -----------------------------------------------------------------------

def test_dry_run_agg():
    r = _dry("agg", ["/kit", "/p", "/out", "/old.tsv", "/eta"], None)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        f"python3 /kit/tools/t125/hmap/aggregate.py /p/MANIFEST.tsv {COV} /out /out/agg "
        f"--old-grid-tsv /old.tsv --old-grid-json {OLD_GRID_JSON} --eta-runs /eta"]


# ---- refusals --------------------------------------------------------------------------------

T118 = "/project/def-maxwl/mforooz/t118"


@pytest.mark.parametrize("name,args", [
    ("ctl_cache", ["/kit", "/p", f"{T118}/ladder/cache"]),
    ("train", ["/kit", "/p", "/c", "/ctl", f"{T118}/h", "true"]),
    ("train", ["/kit", "/p", "/c", "/ctl", "/out", "fnb", f"{T118}/row2/runs"]),
    ("alone", ["/kit", "/p", "/c", "/ctl", "/bl.bed", T118]),
    ("chain", _chain_args(out=f"{T118}/h")),
    ("chain", _chain_args(eta=f"{T118}/ladder/runs")),
    ("agg", ["/kit", "/p", f"{T118}/row2", "/old.tsv", "/eta"])])
def test_a_t118_dir_is_refused(name, args):
    """t125's f run names equal t118's: a t118 dir must never be read as an f run or written."""
    r = _dry(name, args, None if name == "agg" else 0)
    assert r.returncode == 2 and r.stdout == "" and "t118" in r.stderr


@pytest.mark.parametrize("name,args", [("ctl_cache", ["/kit", "/p", "/ctl"]),
                                       ("train", ["/kit", "/p", "/c", "/ctl", "/o", "true"]),
                                       ("alone", ["/kit", "/p", "/c", "/ctl", "/b", "/o"]),
                                       ("chain", _chain_args())])
def test_array_scripts_need_a_numeric_task_index(name, args):
    assert _dry(name, args, None).returncode != 0
    assert _dry(name, args, None, SLURM_ARRAY_TASK_ID="3a").returncode == 2


def test_missing_positionals_fail():
    assert _dry("ctl_cache", ["/kit", "/p"]).returncode != 0
    assert _dry("train", ["/kit", "/p", "/c", "/ctl", "/o"]).returncode != 0
    assert _dry("alone", ["/kit", "/p", "/c", "/ctl", "/b"]).returncode != 0
    assert _dry("chain", ["/kit", "/p", "/c", "/ctl", "/b", "/eta"]).returncode != 0
    assert _dry("agg", ["/kit", "/p", "/o", "/old.tsv"], None).returncode != 0
