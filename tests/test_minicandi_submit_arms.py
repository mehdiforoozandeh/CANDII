"""miniCANDI ladder kit (chunk LK): `slurm/minicandi/submit_arms.sh`, run off-cluster in DRY_RUN
and against a fake `sbatch` / `squeue` on PATH. Nothing is submitted anywhere."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "slurm" / "minicandi" / "submit_arms.sh"
GPU = "--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"
N_ARMS = {1: 8, 2: 6, 3: 7, 4: 9, 5: 10, 6: 9}

FAKE_SBATCH = """#!/bin/bash
echo "$*" >> "$FAKE_LOG"
n=$(wc -l < "$FAKE_LOG")
echo "$((1000 + n));nibi"
"""
# lists the job names that the fake sbatch saw, as squeue -o %j would
FAKE_SQUEUE = """#!/bin/bash
[ -f "$FAKE_LOG" ] && grep -o -- '--job-name=[^ ]*' "$FAKE_LOG" | sed 's/--job-name=//'
cat "${FAKE_LIVE:-/dev/null}"
exit 0
"""


@pytest.fixture()
def env(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in (("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    (tmp_path / "tables").mkdir()
    (tmp_path / "evalset.json").write_text("{}")
    e = {k: v for k, v in os.environ.items()
         if k not in ("DRY_RUN", "SEEDS", "EXTRA_TRAIN_ARGS", "SCORE_TIME", "ONION_EXCLUDE",
                      "DEVICE", "CLUSTER", "CC_CLUSTER")}
    e.update(PATH=f"{bin_}:{e['PATH']}", PYTHON=sys.executable, CLUSTER="nibi",
             FAKE_LOG=str(tmp_path / "sbatch.log"), USER="tester")
    return tmp_path, e


def run(tmp, env, rung, *arms, evalset=True, **over):
    e = {**env, **{k: str(v) for k, v in over.items()}}
    ev = str(tmp / "evalset.json") if evalset else ""
    p = subprocess.run(["bash", str(SCRIPT), str(REPO), str(tmp / "tables"), ev,
                        str(tmp / "runs"), str(rung), *arms],
                       env=e, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    return p.stdout


def sbatch_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if re.search(r"(^|\s)sbatch\s", ln)]


def _opt(line: str, name: str) -> str:
    m = re.search(rf"--{name}=(\S+)", line)
    return m.group(1) if m else ""


@pytest.mark.parametrize("cluster,cpus", [("nibi", "2"), ("fir", "1")])
def test_dry_run_rung1_gpu_slice_with_the_clusters_cpus(env, cluster, cpus):
    tmp, e = env
    out = run(tmp, {**e, "CLUSTER": cluster}, 1, DRY_RUN=1)
    lines = sbatch_lines(out)
    assert len(lines) == 2 * 3 * N_ARMS[1]
    for tr, sc in zip(lines[::2], lines[1::2]):
        assert "slurm/minicandi/train.sh" in tr and "slurm/onion_eval/score.sh" in sc
        for ln in (tr, sc):
            assert GPU in ln and _opt(ln, "cpus-per-task") == cpus
            assert "--exclude" not in ln
        assert tr.rstrip().endswith(" gpu") and _opt(tr, "mem") == "10000M"
        assert _opt(sc, "time") == "08:00:00"
        assert "--dependency=afterok:" in sc and "--kill-on-invalid-dep=yes" in sc
        assert _opt(tr, "job-name") == _opt(sc, "job-name")
        assert 'EXTRA_TRAIN_ARGS="--workers 0"' in tr
    assert not (tmp / "runs").exists()          # a dry run writes nothing
    assert "--time=03:07:00" in lines[0]        # 3 x 80000 x 0.0466 s = 3.1 h
    mc = [ln for ln in lines if "rung1_matched_compute/seed0 " in ln][0]
    assert _opt(mc, "time") == "01:00:00"      # 3 x 24565 x 0.0466 s = 0.95 h -> the 1 h floor


@pytest.mark.parametrize("rung", [2, 3, 4, 5, 6])
def test_dry_run_cpu_rungs(env, rung):
    tmp, e = env
    lines = sbatch_lines(run(tmp, e, rung, DRY_RUN=1))
    assert len(lines) == 2 * 3 * N_ARMS[rung]
    want_score = "1-00:00:00" if rung == 5 else "08:00:00"
    for tr, sc in zip(lines[::2], lines[1::2]):
        assert "--gres=none" in tr and _opt(tr, "cpus-per-task") == "12"
        assert _opt(tr, "mem") == "48000M" and tr.rstrip().endswith(" cpu")
        assert 'EXTRA_TRAIN_ARGS="--workers 8"' in tr
        assert "--gres=none" in sc and _opt(sc, "cpus-per-task") == "2"
        assert _opt(sc, "time") == want_score and GPU not in sc


def test_exclude_list_on_every_line(env):
    tmp, e = env
    lines = sbatch_lines(run(tmp, e, 2, "base", DRY_RUN=1,
                             ONION_EXCLUDE="fc10713, fc10715,fc30601"))
    assert len(lines) == 6
    assert all(_opt(ln, "exclude") == "fc10713,fc10715,fc30601" for ln in lines)


def test_named_json_arm_runs_without_the_rung_prefix(env):
    tmp, e = env
    cfg = REPO / "tools/minicandi/configs/arms/rung3/base.json"
    out = run(tmp, e, 3, f"final_candidate={cfg}", DRY_RUN=1, SEEDS="0")
    lines = sbatch_lines(out)
    assert len(lines) == 2
    run_dir = f"{tmp}/runs/final_candidate/seed0"
    assert f" {cfg} " in lines[0] and f" {run_dir} " in lines[0]
    assert lines[1].rstrip().endswith(f"run {run_dir}")
    assert _opt(lines[0], "job-name") == "mc_final_candidate_s0"


def test_overrides_score_time_device_and_max_windows(env):
    tmp, e = env
    lines = sbatch_lines(run(tmp, e, 5, "base", DRY_RUN=1, SEEDS="0", DEVICE="cuda",
                             SCORE_TIME="02:00:00", EXTRA_TRAIN_ARGS="--max-windows 4000"))
    tr, sc = lines
    assert GPU in tr and GPU in sc and _opt(sc, "cpus-per-task") == "2"
    assert _opt(sc, "time") == "02:00:00" and _opt(tr, "time") == "01:00:00"
    assert 'EXTRA_TRAIN_ARGS="--workers 0 --max-windows 4000"' in tr


def test_empty_evalset_submits_training_only(env):
    tmp, e = env
    lines = sbatch_lines(run(tmp, e, 2, "base", evalset=False, DRY_RUN=1))
    assert len(lines) == 3 and all("train.sh" in ln for ln in lines)


def test_skips_scored_runs_and_live_jobs_and_scores_a_trained_run(env):
    tmp, e = env
    (tmp / "runs/rung2_base/seed0").mkdir(parents=True)
    (tmp / "runs/rung2_base/seed0/SCORED").write_text("")
    (tmp / "runs/rung2_base/seed2").mkdir(parents=True)
    (tmp / "runs/rung2_base/seed2/DONE").write_text("")
    (tmp / "live").write_text("mc_rung2_base_s1\n")
    out = run(tmp, e, 2, "base", DRY_RUN=1, FAKE_LIVE=tmp / "live")
    assert "skip mc_rung2_base_s0: " in out and "SCORED exists" in out
    assert "skip mc_rung2_base_s1: a job of that name is queued or running" in out
    lines = sbatch_lines(out)
    assert len(lines) == 1 and "score.sh" in lines[0] and "--dependency" not in lines[0]


def test_submission_is_logged_and_a_rerun_submits_nothing(env):
    tmp, e = env
    out = run(tmp, e, 1, "base", "dna_off", SEEDS="0 1")
    log = (tmp / "sbatch.log").read_text().splitlines()
    assert len(log) == 8
    assert "--dependency=afterok:1001 " in log[1]   # score waits on its own training job
    rows = (tmp / "runs/SUBMITTED.tsv").read_text().splitlines()
    assert rows[0] == "arm\tseed\ttrain_job\tscore_job\ttime"
    assert [r.split("\t")[:4] for r in rows[1:]] == [
        ["rung1_base", "0", "1001", "1002"], ["rung1_base", "1", "1003", "1004"],
        ["rung1_dna_off", "0", "1005", "1006"], ["rung1_dna_off", "1", "1007", "1008"]]
    assert "submitted 4 runs, skipped 0" in out
    out2 = run(tmp, e, 1, "base", "dna_off", SEEDS="0 1")   # fake squeue now lists all 4 names
    assert "submitted 0 runs, skipped 4" in out2
    assert len((tmp / "sbatch.log").read_text().splitlines()) == 8
    assert len((tmp / "runs/SUBMITTED.tsv").read_text().splitlines()) == 5


def test_unknown_cluster_refuses_a_gpu_job(env):
    tmp, e = env
    e = {**e, "CLUSTER": "elsewhere"}
    p = subprocess.run(["bash", str(SCRIPT), str(REPO), str(tmp / "tables"), "",
                        str(tmp / "runs"), "1", "base"], env={**e, "DRY_RUN": "1"},
                       capture_output=True, text=True)
    assert p.returncode != 0 and "cluster" in p.stderr
    assert not (tmp / "sbatch.log").exists()
