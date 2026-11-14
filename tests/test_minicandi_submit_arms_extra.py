"""Pinned interface K (chunk SAx): the scoring options of `slurm/minicandi/submit_arms.sh`
(EXTRA_SCORE_ARGS, NICE, SCORE_CPUS, SCORE_DEVICE, SCORE_ONLY). Run off-cluster in DRY_RUN and
against a fake `sbatch` / `squeue` on PATH. Nothing is submitted anywhere.

The default path is pinned by sha256 digests of the dry-run output, written by running the
script as merged at base commit 34f7711 (L1a's SCORE_MEM edit, 5dfee5f) on the same cases."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "slurm" / "minicandi" / "submit_arms.sh"
GPU = "--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"

FAKE_SBATCH = """#!/bin/bash
echo "$*" >> "$FAKE_LOG"
n=$(wc -l < "$FAKE_LOG")
echo "$((1000 + n));nibi"
"""
# FAKE_JOBS lines: "<job id> <job name> <command>". -o '%j' prints the names, any other format
# ('%i %j %o', the SCORE_ONLY query) prints the whole lines.
FAKE_SQUEUE = """#!/bin/bash
f="${FAKE_JOBS:-/dev/null}"
case "$*" in
  *%i*) cat "$f" ;;
  *) awk '{print $2}' "$f" ;;
esac
exit 0
"""
CLEARED = ("DRY_RUN", "SEEDS", "EXTRA_TRAIN_ARGS", "SCORE_TIME", "SCORE_MEM", "ONION_EXCLUDE",
           "DEVICE", "CLUSTER", "CC_CLUSTER", "EXTRA_SCORE_ARGS", "NICE", "SCORE_CPUS",
           "SCORE_DEVICE", "SCORE_ONLY", "ONION_STAGE", "TRAIN_TIME", "HOLD")


@pytest.fixture()
def env(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in (("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    (tmp_path / "tables").mkdir()
    (tmp_path / "evalset.json").write_text("{}")
    e = {k: v for k, v in os.environ.items() if k not in CLEARED}
    e.update(PATH=f"{bin_}:{e['PATH']}", PYTHON=sys.executable, CLUSTER="nibi",
             FAKE_LOG=str(tmp_path / "sbatch.log"), USER="tester")
    return tmp_path, e


def run(tmp, env, rung, *arms, evalset=True, ok=True, **over):
    e = {**env, **{k: str(v) for k, v in over.items()}}
    ev = str(tmp / "evalset.json") if evalset else ""
    p = subprocess.run(["bash", str(SCRIPT), str(REPO), str(tmp / "tables"), ev,
                        str(tmp / "runs"), str(rung), *arms],
                       env=e, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, p.stderr
        return p.stdout
    return p


def sbatch_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if re.search(r"(^|\s)sbatch\s", ln)]


def _opt(line: str, name: str) -> str:
    m = re.search(rf"--{name}=(\S+)", line)
    return m.group(1) if m else ""


def _jobs(tmp, *lines):
    (tmp / "jobs").write_text("".join(f"{ln}\n" for ln in lines))
    return {"FAKE_JOBS": tmp / "jobs"}


# ---- the default path: byte for byte the output of the base script --------------------------

def _case_setup(tmp, name):
    """Run-dir state of the 'state' case: seed0 scored, seed1 live, seed2 trained."""
    if name != "state":
        return {}
    for s, f in ((0, "SCORED"), (2, "DONE")):
        (tmp / f"runs/rung2_base/seed{s}").mkdir(parents=True)
        (tmp / f"runs/rung2_base/seed{s}/{f}").write_text("")
    return _jobs(tmp, "77 mc_rung2_base_s1 train.sh")


CASES = {
    "rung1_nibi": dict(rung=1, arms=(), over={}),
    "rung1_fir": dict(rung=1, arms=(), over={"CLUSTER": "fir"}),
    "rung2_all": dict(rung=2, arms=(), over={}),
    "rung5_overrides": dict(rung=5, arms=("base",), over={
        "SEEDS": "0", "DEVICE": "cuda", "SCORE_TIME": "02:00:00",
        "EXTRA_TRAIN_ARGS": "--max-windows 4000"}),
    "rung1_score_mem": dict(rung=1, arms=("base",), over={
        "SCORE_MEM": "128000", "SCORE_TIME": "1-00:00:00"}),
    "rung2_exclude": dict(rung=2, arms=("base",), over={"ONION_EXCLUDE": "fc10713, fc10715"}),
    "rung2_no_evalset": dict(rung=2, arms=("base",), over={}, evalset=False),
    "state": dict(rung=2, arms=("base",), over={}),
}
# sha256 of the normalized dry-run stdout of the script at base 34f7711 (before chunk SAx)
BASE_DIGESTS = {
    "rung1_fir": "6e25b1b9d9942756686ddb63bd0aedc607413ed61ee9fde6378a8506fe0a54d1",
    "rung1_nibi": "1d71ad00b8703eb74402b2951fc5bb0b87fbf0f91bc463e652e20e115392d2f6",
    "rung1_score_mem": "e21e03665854fe12bfb737c17e9153c38499e1839dc5327d4c3b053548db64d7",
    "rung2_all": "2b195b82bf59748d561fb617261b9f0b8c1f2db45295842ab99c3cca4174278c",
    "rung2_exclude": "aeba1818dc20e003ffa9c9c51b054a4c3c0b0e89d78b4cd9c8e5b91601de1a45",
    "rung2_no_evalset": "635f25c3432539deb012b2d1f4326218945eff66d3395c89aaaa9e80a44de813",
    "rung5_overrides": "c2fbd2f9a64cdb63ec932ecb7b0870a0a3c21129f766980a171c3a7678b35eb6",
    "state": "14f04d7769453ff306cfa3db8a815bca55a89d35582650c6a4aa0aab8c4cf3d7",
}


def _digest(tmp, e, name):
    c = CASES[name]
    extra = _case_setup(tmp, name)
    out = run(tmp, {**e, **{k: str(v) for k, v in extra.items()}}, c["rung"], *c["arms"],
              evalset=c.get("evalset", True), DRY_RUN=1, **c["over"])
    norm = out.replace(str(tmp), "<TMP>").replace(str(REPO), "<REPO>")
    return hashlib.sha256(norm.encode()).hexdigest()


@pytest.mark.parametrize("name", sorted(CASES))
def test_default_output_equals_the_base_script_byte_for_byte(env, name):
    tmp, e = env
    assert _digest(tmp, e, name) == BASE_DIGESTS[name]


# ---- the new variables ------------------------------------------------------------------------

def test_extra_score_args_end_the_scoring_line_only(env):
    tmp, e = env
    base = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1, SEEDS="0"))
    lines = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1, SEEDS="0",
                             EXTRA_SCORE_ARGS="--per-region"))
    assert lines[0] == base[0]                              # training line unchanged
    assert lines[1] == base[1] + " --per-region"
    assert lines[1].endswith(f"run {tmp}/runs/rung1_base/seed0 --per-region")


def test_extra_score_args_split_into_words_when_submitted(env):
    tmp, e = env
    run(tmp, e, 2, "base", SEEDS="0", EXTRA_SCORE_ARGS="--per-region --workers 4")
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert log[1].endswith("seed0 --per-region --workers 4") and "per-region" not in log[0]


def test_nice_on_both_lines(env):
    tmp, e = env
    lines = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1, NICE=300))
    assert len(lines) == 6 and all(_opt(ln, "nice") == "300" for ln in lines)
    lines = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1))
    assert not any("--nice" in ln for ln in lines)


def test_score_mem_still_works_with_the_new_variables(env):
    tmp, e = env
    tr, sc = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1, SEEDS="0", SCORE_MEM=16000,
                              NICE=300, EXTRA_SCORE_ARGS="--per-region"))
    assert _opt(sc, "mem") == "16000M" and _opt(tr, "mem") == "10000M"


def test_score_device_cpu_turns_a_gpu_scoring_into_a_cpu_job(env):
    tmp, e = env
    tr, sc = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1, SEEDS="0", SCORE_DEVICE="cpu"))
    assert GPU in tr                                         # training stays on the slice
    assert "--gres=none" in sc and GPU not in sc
    assert _opt(sc, "cpus-per-task") == "2" and _opt(sc, "mem") == "8000M"
    assert _opt(sc, "time") == "08:00:00"


def test_score_device_gpu_on_a_cpu_rung_and_score_cpus(env):
    tmp, e = env
    tr, sc = sbatch_lines(run(tmp, e, 2, "base", DRY_RUN=1, SEEDS="0", SCORE_DEVICE="gpu"))
    assert "--gres=none" in tr and GPU in sc and _opt(sc, "mem") == "24000M"
    assert _opt(sc, "cpus-per-task") == "2"                 # Nibi's slice CPU count
    _, sc = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1, SEEDS="0", SCORE_DEVICE="cpu",
                             SCORE_CPUS=8, SCORE_MEM=16000))
    assert _opt(sc, "cpus-per-task") == "8" and _opt(sc, "mem") == "16000M"


def test_bad_score_device_is_refused(env):
    tmp, e = env
    p = run(tmp, e, 1, "base", DRY_RUN=1, SCORE_DEVICE="cuda", ok=False)
    assert p.returncode == 2 and "SCORE_DEVICE" in p.stderr


def test_score_only_needs_an_evalset(env):
    tmp, e = env
    p = run(tmp, e, 1, "base", evalset=False, DRY_RUN=1, SCORE_ONLY=1, ok=False)
    assert p.returncode == 2 and "SCORE_ONLY" in p.stderr


def _score_only_state(tmp):
    """seed0 trained (DONE), seed1 training running as job 501, seed2 nothing,
    seed3 scored, seed4 trained and a scoring job already queued."""
    for s, f in ((0, "DONE"), (3, "SCORED"), (4, "DONE")):
        d = tmp / f"runs/rung1_base/seed{s}"
        d.mkdir(parents=True)
        (d / f).write_text("")
    return _jobs(tmp, "501 mc_rung1_base_s1 /k/slurm/minicandi/train.sh /k cfg t 1 run gpu",
                 "502 mc_rung1_base_s4 /k/slurm/onion_eval/score.sh /k ev run run",
                 "503 mc_other_s0 /k/slurm/minicandi/train.sh")


def test_score_only_dry_run(env):
    tmp, e = env
    out = run(tmp, e, 1, "base", DRY_RUN=1, SCORE_ONLY=1, SEEDS="0 1 2 3 4",
              EXTRA_SCORE_ARGS="--per-region", NICE=300, **_score_only_state(tmp))
    lines = sbatch_lines(out)
    assert len(lines) == 2 and all("score.sh" in ln and "train.sh" not in ln for ln in lines)
    s0, s1 = lines
    assert "seed0 --per-region" in s0 and "--dependency" not in s0
    assert "--dependency=afterok:501 --kill-on-invalid-dep=yes" in s1 and "seed1 " in s1
    assert all(_opt(ln, "nice") == "300" for ln in lines)
    assert "skip mc_rung1_base_s2: training neither done nor queued or running" in out
    assert "skip mc_rung1_base_s3: " in out and "SCORED exists" in out
    assert "skip mc_rung1_base_s4: a scoring job of that name is queued or running" in out
    assert not (tmp / "sbatch.log").exists()


def test_score_only_submits_scoring_jobs_and_logs_them(env):
    tmp, e = env
    out = run(tmp, e, 1, "base", SCORE_ONLY=1, SEEDS="0 1 2 3 4", **_score_only_state(tmp))
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert len(log) == 2 and all("score.sh" in ln for ln in log)
    assert "--dependency" not in log[0]
    assert "--dependency=afterok:501 --kill-on-invalid-dep=yes" in log[1]
    rows = (tmp / "runs/SUBMITTED.tsv").read_text().splitlines()
    assert [r.split("\t")[:4] for r in rows[1:]] == [
        ["rung1_base", "0", "-", "1001"], ["rung1_base", "1", "-", "1002"]]
    assert "submitted 2 runs, skipped 3" in out


def test_score_only_without_squeue_refuses_a_real_submission(env, tmp_path):
    tmp, e = env
    (tmp / "bin" / "squeue").unlink()
    e = {**e, "PATH": f"{tmp / 'bin'}:/usr/bin:/bin"}
    p = run(tmp, e, 1, "base", SCORE_ONLY=1, ok=False)
    assert p.returncode == 2 and not (tmp / "sbatch.log").exists()


def test_train_time_and_hold(env):
    tmp, e = env
    base = sbatch_lines(run(tmp, e, 2, "base", DRY_RUN=1, SEEDS="0"))
    tr, sc = sbatch_lines(run(tmp, e, 2, "base", DRY_RUN=1, SEEDS="0", TRAIN_TIME="04:00:00",
                              HOLD=1))
    assert _opt(base[0], "time") != "04:00:00" and _opt(tr, "time") == "04:00:00"
    assert _opt(sc, "time") == _opt(base[1], "time")                # scoring time unchanged
    assert " --hold " in tr and " --hold " in sc and "--hold" not in " ".join(base)
    assert tr.replace(" --hold", "").replace("--time=04:00:00", f"--time={_opt(base[0], 'time')}") \
        == base[0]
    p = run(tmp, e, 2, "base", DRY_RUN=1, TRAIN_TIME="4h", ok=False)
    assert p.returncode == 2 and "TRAIN_TIME" in p.stderr
