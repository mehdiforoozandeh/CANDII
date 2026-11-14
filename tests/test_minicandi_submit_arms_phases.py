"""Chunk SP3 (Amendment 10, interface M): two-phase scoring in `slurm/minicandi/submit_arms.sh`
(SCORE_PHASES, SCORE_SHARDS, SCORE_SUBDIR, PREDICT_ONLY, METRICS_ONLY, HOLD, REFS_FROM, ...) and
the metric phase of `slurm/onion_eval/score.sh`. Run off-cluster in DRY_RUN and against a fake
`sbatch` / `squeue` on PATH. Nothing is submitted anywhere. The default path (no new variable) is
pinned byte for byte by the digests of tests/test_minicandi_submit_arms_extra.py."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "slurm" / "minicandi" / "submit_arms.sh"
SCORE_SH = REPO / "slurm" / "onion_eval" / "score.sh"
GPU = "--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1"

# logs the arguments and the phase variables the job would see (--export=ALL)
FAKE_SBATCH = """#!/bin/bash
echo "$* |ENV shard=${ONION_SHARD:-} batch=${ONION_INFER_BATCH:-} dev=${ONION_DEVICE:-}\
 phase=${ONION_SCORE_PHASE:-} store=${ONION_PRED_STORE:-} refs=${ONION_REFS_FROM:-}\
 keep=${ONION_KEEP_PRED_STORE:-}" >> "$FAKE_LOG"
n=$(wc -l < "$FAKE_LOG")
echo "$((1000 + n));nibi"
"""
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
           "SCORE_DEVICE", "SCORE_ONLY", "ONION_STAGE", "SCORE_PHASES", "SCORE_SHARDS",
           "SCORE_SUBDIR", "PREDICT_ONLY", "METRICS_ONLY", "PRED_JOBS", "SCORE_INFER_BATCH",
           "SCORE_PRED_DEVICE", "SCORE_PRED_CPUS", "SCORE_PRED_MEM", "SCORE_PRED_TIME",
           "REFS_FROM", "KEEP_PRED_STORE", "HOLD", "PRED_STORE_ROOT", "ONION_SCORE_PHASE",
           "ONION_PRED_STORE", "ONION_REFS_FROM", "ONION_KEEP_PRED_STORE")


@pytest.fixture()
def env(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name, text in (("sbatch", FAKE_SBATCH), ("squeue", FAKE_SQUEUE)):
        (bin_ / name).write_text(text)
        (bin_ / name).chmod(0o755)
    (tmp_path / "tables").mkdir()
    (tmp_path / "refs").mkdir()
    (tmp_path / "evalset.json").write_text("{}")
    e = {k: v for k, v in os.environ.items() if k not in CLEARED}
    e.update(PATH=f"{bin_}:{e['PATH']}", PYTHON=sys.executable, CLUSTER="nibi",
             FAKE_LOG=str(tmp_path / "sbatch.log"), USER="tester",
             PRED_STORE_ROOT=str(tmp_path / "ps"))
    return tmp_path, e


def run(tmp, env, rung, *arms, ok=True, phases=True, **over):
    base = {"SCORE_PHASES": 1, "REFS_FROM": tmp / "refs", "SEEDS": "0"} if phases else {}
    e = {**env, **{k: str(v) for k, v in {**base, **over}.items()}}
    p = subprocess.run(["bash", str(SCRIPT), str(REPO), str(tmp / "tables"),
                        str(tmp / "evalset.json"), str(tmp / "runs"), str(rung), *arms],
                       env=e, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, p.stderr
        return p.stdout
    return p


def sbatch_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if re.search(r"(^|\s)sbatch\s", ln)]


def job_lines(out: str) -> list[list[str]]:
    return [ln.split() for ln in out.splitlines()
            if re.fullmatch(r"(\d+|<[a-z]+( \d+)?>) \S+ \d+ (train|predict|metrics) \S+", ln)]


def _opt(line: str, name: str) -> str:
    m = re.search(rf"--{name}=(\S+)", line)
    return m.group(1) if m else ""


def _done(tmp, *seeds, arm="rung1_base"):
    for s in seeds:
        d = tmp / f"runs/{arm}/seed{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "DONE").write_text("")


def _store(tmp, n, done, arm="rung1_base", seed=0, sub=""):
    d = tmp / "ps" / "runs" / arm / f"seed{seed}"
    d = d / sub if sub else d
    d.mkdir(parents=True)
    (d / "STORE.json").write_text(json.dumps({"format": 1, "n_shards": n}))
    for i in range(done):
        (d / f"PREDICTED_{i}of{n}.json").write_text("{}")
    return d


# ---- job shapes -------------------------------------------------------------------------------

def test_gpu_one_shard_lines_resources_and_dependencies(env):
    tmp, e = env
    out = run(tmp, e, 1, "base", DRY_RUN=1)
    tr, pr, me = sbatch_lines(out)
    assert "train.sh" in tr and GPU in tr
    run0 = f"{tmp}/runs/rung1_base/seed0"
    store = f"{tmp}/ps/runs/rung1_base/seed0"
    assert pr.startswith("ONION_SHARD=0/1 ONION_INFER_BATCH=32 ONION_DEVICE=cuda sbatch ")
    assert GPU in pr and _opt(pr, "cpus-per-task") == "4" and _opt(pr, "mem") == "16000M"
    assert _opt(pr, "time") == "02:00:00" and _opt(pr, "dependency") == "afterok:<train>"
    assert "--kill-on-invalid-dep=yes" in pr and "--export=ALL" in pr
    assert pr.endswith(f"slurm/onion_eval/predict.sh {REPO} {run0} {tmp}/evalset.json {store}")
    assert me.startswith(f"ONION_SCORE_PHASE=metrics ONION_PRED_STORE={store} "
                         f"ONION_REFS_FROM={tmp}/refs sbatch ")
    assert "--gres=none" in me and _opt(me, "cpus-per-task") == "16"
    assert _opt(me, "mem") == "64000M" and _opt(me, "time") == "03:00:00"
    assert _opt(me, "dependency") == "afterok:<predict"          # "<predict 0>" has a space
    assert "--dependency=afterok:<predict 0> --kill-on-invalid-dep=yes" in me
    assert me.endswith(f"score.sh {REPO} {tmp}/evalset.json {run0} run {run0}")
    assert "ONION_KEEP_PRED_STORE" not in me
    assert job_lines(out) == [["<train>", "rung1_base", "0", "train", "-"],
                              ["<predict", "0>", "rung1_base", "0", "predict", "0/1"],
                              ["<metrics>", "rung1_base", "0", "metrics", "-"]]
    assert not (tmp / "sbatch.log").exists() and not (tmp / "runs").exists()


def test_gpu_eight_shards(env):
    tmp, e = env
    out = run(tmp, e, 1, "base", DRY_RUN=1, SCORE_SHARDS=8, SCORE_INFER_BATCH=32)
    lines = sbatch_lines(out)
    assert len(lines) == 10
    preds = lines[1:9]
    for i, ln in enumerate(preds):
        assert ln.startswith(f"ONION_SHARD={i}/8 ONION_INFER_BATCH=32 ONION_DEVICE=cuda ")
        assert f"--output={tmp}/runs/rung1_base/seed0/predict_{i}of8_%j.out" in ln and GPU in ln
    want = ":".join(f"<predict {i}>" for i in range(8))
    assert f"--dependency=afterok:{want} --kill-on-invalid-dep=yes" in lines[9]


def test_cpu_predict(env):
    tmp, e = env
    for rung, over in ((1, {"SCORE_PRED_DEVICE": "cpu"}), (2, {})):
        _, pr, me = sbatch_lines(run(tmp, e, rung, "base", DRY_RUN=1, **over))
        assert pr.startswith("ONION_SHARD=0/1 ONION_INFER_BATCH=8 ONION_DEVICE=cpu sbatch ")
        assert "--gres=none" in pr and GPU not in pr
        assert _opt(pr, "cpus-per-task") == "8" and _opt(pr, "mem") == "32000M"
        assert _opt(pr, "time") == "06:00:00" and "--gres=none" in me


def test_pred_overrides_and_gpu_predict_on_a_cpu_rung(env):
    tmp, e = env
    _, pr, me = sbatch_lines(run(tmp, e, 4, "base", DRY_RUN=1, SCORE_PRED_DEVICE="cuda",
                                 SCORE_PRED_CPUS=2, SCORE_PRED_MEM=12000,
                                 SCORE_PRED_TIME="00:40:00", SCORE_INFER_BATCH=256,
                                 SCORE_CPUS=8, SCORE_MEM=32000, SCORE_TIME="04:00:00"))
    assert GPU in pr and "ONION_INFER_BATCH=256 ONION_DEVICE=cuda" in pr
    assert (_opt(pr, "cpus-per-task"), _opt(pr, "mem"), _opt(pr, "time")) == \
        ("2", "12000M", "00:40:00")
    assert (_opt(me, "cpus-per-task"), _opt(me, "mem"), _opt(me, "time")) == \
        ("8", "32000M", "04:00:00")


def test_score_subdir_store_and_score_paths(env):
    tmp, e = env
    _, pr, me = sbatch_lines(run(tmp, e, 4, "base", DRY_RUN=1, SCORE_SUBDIR="p256"))
    run0 = f"{tmp}/runs/rung4_base/seed0"
    store = f"{tmp}/ps/runs/rung4_base/seed0/p256"
    assert pr.endswith(f" {store}") and f"--output={run0}/p256/predict_0of1_%j.out" in pr
    assert f"ONION_PRED_STORE={store} " in me
    assert me.endswith(f"score.sh {REPO} {tmp}/evalset.json {run0}/p256 run {run0}")
    assert f"--output={run0}/p256/metrics_%j.out" in me


def test_named_arm_store_path_uses_the_run_name(env):
    tmp, e = env
    cfg = REPO / "tools/minicandi/configs/arms/rung4/base.json"
    _, pr, _ = sbatch_lines(run(tmp, e, 4, f"rg_no_eval={cfg}", DRY_RUN=1, SCORE_SUBDIR="p16"))
    assert pr.endswith(f" {tmp}/ps/runs/rg_no_eval/seed0/p16")


def test_score_subdir_skip_rule(env):
    tmp, e = env
    (tmp / "runs/rung4_base/seed0").mkdir(parents=True)
    (tmp / "runs/rung4_base/seed0/SCORED").write_text("")      # the legacy score: not p16's
    out = run(tmp, e, 4, "base", DRY_RUN=1, SCORE_SUBDIR="p16")
    assert len(sbatch_lines(out)) == 3
    (tmp / "runs/rung4_base/seed0/p16").mkdir()
    (tmp / "runs/rung4_base/seed0/p16/SCORED").write_text("")
    out = run(tmp, e, 4, "base", DRY_RUN=1, SCORE_SUBDIR="p16")
    assert not sbatch_lines(out) and "p16/SCORED exists" in out


def test_score_subdir_on_the_legacy_line(env):
    tmp, e = env
    _, sc = sbatch_lines(run(tmp, e, 2, "base", DRY_RUN=1, phases=False, SEEDS="0",
                             SCORE_SUBDIR="p16"))
    run0 = f"{tmp}/runs/rung2_base/seed0"
    assert sc.endswith(f"score.sh {REPO} {tmp}/evalset.json {run0}/p16 run {run0}")
    assert f"--output={run0}/p16/score_%j.out" in sc


def test_bad_score_subdir_is_refused(env):
    tmp, e = env
    for bad in ("..", "a/b", "/abs"):
        p = run(tmp, e, 1, "base", DRY_RUN=1, SCORE_SUBDIR=bad, ok=False)
        assert p.returncode == 2 and "SCORE_SUBDIR" in p.stderr


def test_keep_store_extra_args_nice_hold(env):
    tmp, e = env
    lines = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1, KEEP_PRED_STORE=1, NICE=300, HOLD=1,
                             EXTRA_SCORE_ARGS="--per-region"))
    assert len(lines) == 3
    assert all(_opt(ln, "nice") == "300" and " --hold " in ln for ln in lines)
    tr, pr, me = lines
    assert "ONION_KEEP_PRED_STORE=1 sbatch" in me and me.endswith(" --per-region")
    assert "--per-region" not in pr and "--per-region" not in tr
    assert not any("--hold" in ln for ln in sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1)))


# ---- modes ------------------------------------------------------------------------------------

def test_predict_only(env):
    tmp, e = env
    _done(tmp, 0)
    out = run(tmp, e, 1, "base", DRY_RUN=1, PREDICT_ONLY=1, SCORE_SHARDS=2, REFS_FROM="",
              SEEDS="0 1")
    lines = sbatch_lines(out)
    assert len(lines) == 2 and all("predict.sh" in ln and "--dependency" not in ln for ln in lines)
    assert "skip mc_rung1_base_s1: training neither done nor queued or running" in out
    assert [j[-2:] for j in job_lines(out)] == [["predict", "0/2"], ["predict", "1/2"]]


def test_predict_only_waits_on_a_live_training(env):
    tmp, e = env
    (tmp / "jobs").write_text("501 mc_rung1_base_s0 /k/slurm/minicandi/train.sh /k\n")
    lines = sbatch_lines(run(tmp, e, 1, "base", DRY_RUN=1, PREDICT_ONLY=1,
                             FAKE_JOBS=tmp / "jobs"))
    assert len(lines) == 1 and "--dependency=afterok:501 --kill-on-invalid-dep=yes" in lines[0]


def test_metrics_only_needs_a_complete_store_or_pred_jobs(env):
    tmp, e = env
    _done(tmp, 0)
    _store(tmp, 2, 1)
    out = run(tmp, e, 1, "base", DRY_RUN=1, METRICS_ONLY=1)
    assert not sbatch_lines(out) and "is not complete" in out
    out = run(tmp, e, 1, "base", DRY_RUN=1, METRICS_ONLY=1, PRED_JOBS="701:702")
    (me,) = sbatch_lines(out)
    assert "score.sh" in me and "--dependency=afterok:701:702 --kill-on-invalid-dep=yes" in me
    (tmp / "ps/runs/rung1_base/seed0/PREDICTED_1of2.json").write_text("{}")
    out = run(tmp, e, 1, "base", DRY_RUN=1, METRICS_ONLY=1)
    (me,) = sbatch_lines(out)
    assert "ONION_SCORE_PHASE=metrics" in me and "--dependency" not in me
    assert job_lines(out) == [["<metrics>", "rung1_base", "0", "metrics", "-"]]


def test_metrics_only_ignores_live_predict_jobs_but_not_a_live_metrics_job(env):
    tmp, e = env
    _done(tmp, 0, 1)
    (tmp / "jobs").write_text(
        "601 mc_rung1_base_s0 /k/slurm/onion_eval/predict.sh /k r ev st\n"
        "602 mc_rung1_base_s1 /k/slurm/onion_eval/score.sh /k ev o run r\n")
    out = run(tmp, e, 1, "base", DRY_RUN=1, METRICS_ONLY=1, PRED_JOBS="601", SEEDS="0 1",
              FAKE_JOBS=tmp / "jobs")
    (me,) = sbatch_lines(out)
    assert "seed0 run" in me and "--dependency=afterok:601 " in me
    assert "skip mc_rung1_base_s1: a scoring job of that name is queued or running" in out


def test_refusals(env):
    tmp, e = env
    for over, word in (({"REFS_FROM": ""}, "REFS_FROM"),
                       ({"REFS_FROM": tmp / "nope"}, "REFS_FROM"),
                       ({"METRICS_ONLY": 1, "REFS_FROM": ""}, "REFS_FROM"),
                       ({"PREDICT_ONLY": 1, "METRICS_ONLY": 1}, "exclude"),
                       ({"SCORE_PRED_DEVICE": "gpu"}, "SCORE_PRED_DEVICE"),
                       ({"SCORE_SHARDS": 0}, "SCORE_SHARDS"),
                       ({"PRED_JOBS": "12"}, "PRED_JOBS"),
                       ({"METRICS_ONLY": 1, "PRED_JOBS": "x1"}, "PRED_JOBS")):
        p = run(tmp, e, 1, "base", DRY_RUN=1, ok=False, **over)
        assert p.returncode == 2 and word in p.stderr, (over, p.stderr)
    assert not (tmp / "sbatch.log").exists()


# ---- a submission against the fake sbatch -----------------------------------------------------

def test_submission_prints_job_lines_and_chains_the_real_ids(env):
    tmp, e = env
    out = run(tmp, e, 1, "base", SCORE_SHARDS=2, SCORE_SUBDIR="p16", KEEP_PRED_STORE=1)
    assert job_lines(out) == [["1001", "rung1_base", "0", "train", "-"],
                              ["1002", "rung1_base", "0", "predict", "0/2"],
                              ["1003", "rung1_base", "0", "predict", "1/2"],
                              ["1004", "rung1_base", "0", "metrics", "-"]]
    log = (tmp / "sbatch.log").read_text().splitlines()
    store = f"{tmp}/ps/runs/rung1_base/seed0/p16"
    assert "--dependency=afterok:1001 " in log[1] and "--dependency=afterok:1001 " in log[2]
    assert log[1].endswith("|ENV shard=0/2 batch=32 dev=cuda phase= store= refs= keep=")
    assert log[2].endswith("|ENV shard=1/2 batch=32 dev=cuda phase= store= refs= keep=")
    assert "--dependency=afterok:1002:1003 --kill-on-invalid-dep=yes" in log[3]
    assert log[3].endswith(f"phase=metrics store={store} refs={tmp}/refs keep=1")
    assert "shard= " in log[3] and "phase= " in log[0]
    assert (tmp / "runs/rung1_base/seed0/p16").is_dir()
    rows = (tmp / "runs/SUBMITTED.tsv").read_text().splitlines()
    assert rows[1].split("\t")[:4] == ["rung1_base", "0", "1001", "predict=1002:1003;metrics=1004"]


# ---- score.sh, the metric phase ---------------------------------------------------------------

def _score_sh(tmp, *args, **env_over):
    e = {k: v for k, v in os.environ.items() if k not in CLEARED}
    e.update(DRY_RUN="1", **{k: str(v) for k, v in env_over.items()})
    e.pop("SLURM_CPUS_PER_TASK", None)
    e.pop("SLURM_TMPDIR", None)
    (tmp / "ev.json").write_text("{}")
    return subprocess.run(["bash", str(SCORE_SH), "/kit", str(tmp / "ev.json"), "/out", *args],
                          env=e, capture_output=True, text=True)


def test_score_sh_metric_phase_and_legacy_call(tmp_path):
    p = _score_sh(tmp_path, "run", "/r", "--per-region")
    assert p.returncode == 0
    assert p.stdout.strip() == (f"python3 /kit/tools/onion_eval/score.py --run /r --evalset "
                                f"{tmp_path}/ev.json --out /out --workers 1 --per-region")
    p = _score_sh(tmp_path, "run", "/r", "--per-region", ONION_SCORE_PHASE="metrics",
                  ONION_PRED_STORE="/ps/x", ONION_REFS_FROM="/b")
    assert p.returncode == 0
    assert p.stdout.strip() == (f"python3 /kit/tools/onion_eval/score.py --run /r --evalset "
                                f"{tmp_path}/ev.json --out /out --workers 1 --phase metrics "
                                f"--pred-store /ps/x --refs-from /b --per-region")


def test_score_sh_metric_phase_refusals(tmp_path):
    for args, over, word in (
            (("run", "/r"), {"ONION_REFS_FROM": "/b"}, "ONION_PRED_STORE"),
            (("run", "/r"), {"ONION_PRED_STORE": "/s"}, "ONION_REFS_FROM"),
            (("baseline", "marginal_nb"), {"ONION_PRED_STORE": "/s", "ONION_REFS_FROM": "/b"},
             "baseline"),
            (("run", "/r"), {"ONION_SCORE_PHASE": "predict"}, "ONION_SCORE_PHASE")):
        over = {"ONION_SCORE_PHASE": "metrics", **over}
        p = _score_sh(tmp_path, *args, **over)
        assert p.returncode == 2 and word in p.stderr, (args, p.stderr)


def test_score_sh_removes_the_store_only_after_scored_and_without_keep():
    text = SCORE_SH.read_text()
    i = text.index('"${CMD[@]}"\nif [ "$PHASE" = metrics ]')
    block = text[i:text.index("=== done", i)]
    for part in ('"${ONION_KEEP_PRED_STORE:-0}" != 1', '-f "$OUT/SCORED"',
                 '-f "$ONION_PRED_STORE/STORE.json"', 'rm -rf -- "$ONION_PRED_STORE"'):
        assert part in block
