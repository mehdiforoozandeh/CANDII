"""Task t370, chunk C4: the recipe arm table (`tools/minicandi/recipe_arms.py`), its generated
configs (`tools/minicandi/configs/recipe/`), the stage entry point
(`configs/recipe/submit_recipe.sh`) and the two additions to `slurm/minicandi/submit_arms.sh`
(JOB_PREFIX, the flock append). Off-cluster, DRY_RUN or a fake `sbatch`; nothing is submitted.

The model code that reads the keys of chunks C1 and C2 is not imported here.

The default path of submit_arms.sh (JOB_PREFIX unset) is pinned by sha256 digests of the dry-run
output of the script at base commit 57f3c9e8 (before this chunk) on the cases of CASES."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "tools") not in sys.path:
    sys.path.insert(0, str(REPO / "tools"))

from minicandi import contract as C  # noqa: E402
from minicandi import mix_arms as M  # noqa: E402
from minicandi import recipe_arms as RA  # noqa: E402

SUBMIT = REPO / "slurm/minicandi/submit_arms.sh"
RECIPE = REPO / "tools/minicandi/configs/recipe/submit_recipe.sh"
CONFIGS = REPO / "tools/minicandi/configs"

# ------------------------------------------------------------------------------ the arm table

PINNED = {
    "skip": {"skipfilm_ccp": {"skip_film": "c_cprime"}, "headskip_off": {"head_skip": "off"},
             "headskip_film": {"head_skip": "film"},
             "skipfilm_ccp_headskip_film": {"skip_film": "c_cprime", "head_skip": "film"}},
    "placement": {f"film_{p}": {"film_placement": p}
                  for p in ("encoder", "decoder", "bottleneck", "all")},
    "nocov": {"nocov": {"covariate_drop_fields": list(C.META_COLUMNS)}},
    "cac": {"cac_curve": {"cac_pre": "curve", "cac_post": "curve"},
            "cac_cnn": {"cac_pre": "cnn", "cac_post": "cnn"},
            "cac_curve_corefilm_off": {"cac_pre": "curve", "cac_post": "curve",
                                       "covariate_drop_fields": list(C.META_COLUMNS)},
            "cac_cnn_corefilm_off": {"cac_pre": "cnn", "cac_post": "cnn",
                                     "covariate_drop_fields": list(C.META_COLUMNS)}},
    "probe": {"probe_base": {}},
    "halves": {"halves": {"train_halves": "pairs"}},
    "control": {"ctl_holdout": {"control_holdout_cells": RA.CONTROL_HOLDOUT},
                "ctl_holdout_w4": {"control_holdout_cells": RA.CONTROL_HOLDOUT,
                                   "control_loss_weight": 4.0},
                "ctl_holdout_separate": {"control_holdout_cells": RA.CONTROL_HOLDOUT,
                                         "control_branch": "separate"}},
}
_HV = {"train_halves": "pairs"}
#: decision D35: the screen winner halves plus one U-Net switch each, U-Net only, no stage
PINNED["combo"] = {
    "halves_mix_avg": {**_HV, "average_input": True, "average_h5": M.AVERAGE_H5},
    "halves_mix_pearson": {**_HV, "count_transform": "pearson"},
    "halves_mix_fixdepth": {**_HV, "input_depth": "fixed"},
    "halves_headskip_film": {**_HV, "head_skip": "film"},
    "halves_headskip_off": {**_HV, "head_skip": "off"},   # D41: the copy-path test
    "halves_mix_avg_pearson": {**_HV, "average_input": True, "average_h5": M.AVERAGE_H5,
                               "count_transform": "pearson"}}   # PI 2026-11-22: full combination
_REC = {**_HV, "count_transform": "pearson"}
_CTL = {"control_holdout_cells": RA.CONTROL_HOLDOUT}
PINNED["reccontrol"] = {"rec_ctlw4": {**_REC, "control_loss_weight": 4.0},   # gate G5
                        "rec_ctl_holdout": {**_REC, **_CTL},
                        "rec_ctl_holdout_w4": {**_REC, **_CTL, "control_loss_weight": 4.0}}
PINNED["scale"] = {"scale_p2x": {**_REC, "param_budget": 2.0e6},   # D42
                   "scale_p4x": {**_REC, "param_budget": 4.0e6},
                   "scale_w2x": {**_REC, "windows_seen": 160000}}
PINNED_MODELS = {"combo": (4,), "scale": (4,), "reccontrol": (4,), "skip": (4,), "placement": (2, 3, 4), "nocov": (2, 3, 4), "cac": (2, 3, 4),
                 "probe": (4,), "halves": (2, 3, 4), "control": (2, 3, 4)}
#: the mix arms of the plan, per model
PINNED_MIX = {2: {"mix_svonly"}, 3: {"mix_concat", "mix_fixdepth", "mix_imp4", "mix_crps",
                                     "mix_svonly"}, 4: {"mix_avg", "mix_pearson", "mix_fixdepth"}}


def test_models_and_meta_columns():
    assert RA.MODELS == M.ARCH == {2: "convae", 3: "convattn", 4: "unet"}
    assert RA.META_COLUMNS == C.META_COLUMNS and len(RA.META_COLUMNS) == 11


@pytest.mark.parametrize("family", sorted(PINNED))
def test_pinned_families(family):
    have = {a: x for a, x in RA.ARMS.items() if x["family"] == family}
    assert set(have) == set(PINNED[family])
    for a, x in have.items():
        assert x["keys"] == PINNED[family][a], a
        assert x["models"] == PINNED_MODELS[family], a


def test_mix_arms_follow_the_pool():
    mix = {a: x for a, x in RA.ARMS.items() if x["family"] == "mix"}
    for r, want in PINNED_MIX.items():
        assert {a for a, x in mix.items() if r in x["models"]} == want
    short = {v[0]: sw for sw, v in M.SWITCHES.items()}
    for a, x in mix.items():
        sw = short[a.removeprefix("mix_")]
        assert x["keys"] == M.SWITCHES[sw][2]
        assert x["models"] == tuple(r for r in (2, 3, 4) if sw in M.POOL[r])
    assert set(RA.ARMS) == set().union(*map(set, PINNED.values())) | set(mix)


def test_stages():
    assert RA.STAGES["probe"] == ("probe_base",)
    assert set(RA.STAGES["s1a"]) == set(PINNED["skip"]) | set(PINNED["placement"]) | {"nocov",
                                                                                       "halves"}
    assert set(RA.STAGES["s1b"]) == ({"cac_curve", "cac_curve_corefilm_off"}
                                     | {a for a, x in RA.ARMS.items() if x["family"] == "mix"})
    assert set(RA.STAGES["s1c"]) == {"cac_cnn", "cac_cnn_corefilm_off"}
    # arm-models per stage (the plan: C12 19 arm-models, C13 15, C27 6; C29 adds 3 to s1a)
    assert [len(RA.select(s)) for s in ("probe", "s1a", "s1b", "s1c")] == [1, 22, 15, 6]
    # C15: the control family is in no stage; C18 runs it with STAGE=arms
    assert not any(RA.stages_of(a) for a in PINNED["control"])
    assert len(RA.select(arms=list(PINNED["control"]))) == 9
    assert [RA.run_name(r, a) for r, a in RA.select(arms=["halves"])] == \
        ["rcp_halves_convae", "rcp_halves_convattn", "rcp_halves_unet"]


def test_train_time_and_run_names():
    long = {a for a in RA.ARMS if RA.train_time(a) == "08:00:00"}
    assert long == {"mix_avg", "cac_cnn", "cac_cnn_corefilm_off", "halves_mix_avg",
                    "halves_mix_avg_pearson", "scale_p2x", "scale_w2x"}
    assert RA.train_time("scale_p4x") == "16:00:00"   # D42
    assert all(RA.train_time(a) == "03:00:00" for a in set(RA.ARMS) - long - {"scale_p4x"})
    for arm, x in RA.ARMS.items():
        for r in x["models"]:
            assert RA.run_name(r, arm) == f"rcp_{arm}_{RA.MODELS[r]}"


@pytest.mark.parametrize("rung,arm", [(r, a) for a, x in RA.ARMS.items() for r in x["models"]])
def test_config_is_base_plus_the_arms_keys(rung, arm):
    base = json.loads((CONFIGS / f"rung{rung}/base_heldout.json").read_text())
    cfg = RA.config(rung, arm)
    keys = RA.ARMS[arm]["keys"]
    assert {k: v for k, v in cfg.items() if k not in keys} == \
        {k: v for k, v in base.items() if k not in keys}
    assert {k: cfg[k] for k in keys} == keys
    # base keys keep their order; the arm's new keys follow
    assert list(cfg)[:len(base)] == list(base)


def test_config_refuses_a_model_outside_the_arm():
    with pytest.raises(ValueError):
        RA.config(2, "skipfilm_ccp")


def test_files_on_disk_equal_write():
    want = RA.files()
    for p, body in want.items():
        assert p.is_file() and p.read_text(encoding="utf-8") == body, p
    have = {p for p in RA.OUT.rglob("*") if p.is_file()}
    assert have - set(want) == {RA.OUT / "submit_recipe.sh"}


def test_list_cli():
    p = subprocess.run([sys.executable, str(REPO / "tools/minicandi/recipe_arms.py"), "list",
                        "--stage", "s1c", "--models", "2 4"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    rows = [ln.split("\t") for ln in p.stdout.splitlines()]
    assert [(r[0], r[1]) for r in rows] == [("2", "cac_cnn"), ("4", "cac_cnn"),
                                            ("2", "cac_cnn_corefilm_off"),
                                            ("4", "cac_cnn_corefilm_off")]
    assert all(r[2].startswith("rcp_") and (REPO / r[3]).is_file() and r[4] == "08:00:00"
               for r in rows)


# ---------------------------------------------------------------- submit_arms.sh: JOB_PREFIX

FAKE_SBATCH = """#!/bin/bash
echo "$*" >> "$FAKE_LOG"
n=$(wc -l < "$FAKE_LOG")
echo "$((1000 + n));fir"
"""
FAKE_SQUEUE = """#!/bin/bash
f="${FAKE_JOBS:-/dev/null}"
case "$*" in
  *%i*) cat "$f" ;;
  *) awk '{print $2}' "$f" ;;
esac
exit 0
"""
#: a real exclusive lock (fcntl), as util-linux `flock <file> <command> ...`; logs its lock file
FAKE_FLOCK = f"""#!{sys.executable}
import fcntl, os, subprocess, sys
with open(os.environ["FLOCK_LOG"], "a") as log:
    log.write(sys.argv[1] + "\\n")
with open(sys.argv[1], "a") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    sys.exit(subprocess.call(sys.argv[2:]))
"""
CLEARED = ("DRY_RUN", "SEEDS", "EXTRA_TRAIN_ARGS", "SCORE_TIME", "SCORE_MEM", "ONION_EXCLUDE",
           "DEVICE", "CLUSTER", "CC_CLUSTER", "EXTRA_SCORE_ARGS", "NICE", "SCORE_CPUS",
           "SCORE_DEVICE", "SCORE_ONLY", "ONION_STAGE", "SCORE_PHASES", "SCORE_SHARDS",
           "SCORE_SUBDIR", "PREDICT_ONLY", "METRICS_ONLY", "PRED_JOBS", "SCORE_INFER_BATCH",
           "SCORE_PRED_DEVICE", "SCORE_PRED_CPUS", "SCORE_PRED_MEM", "SCORE_PRED_TIME",
           "REFS_FROM", "KEEP_PRED_STORE", "HOLD", "PRED_STORE_ROOT", "TRAIN_TIME",
           "SCORE_SAME_VIEW", "FOREIGN_COVARIATES", "FOREIGN_SEED", "SCORE_INPUT_DEPTH",
           "JOB_PREFIX", "O", "K", "STAGE", "SUB", "ARMS", "MODELS", "FOREIGN")
CFG4 = str(CONFIGS / "rung4/base_heldout.json")
PH = {"SCORE_PHASES": "1", "SCORE_SHARDS": "2", "SCORE_SAME_VIEW": "1", "SEEDS": "0 1"}


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
    e.update(PATH=f"{bin_}:{e['PATH']}", PYTHON=sys.executable, CLUSTER="fir",
             FAKE_LOG=str(tmp_path / "sbatch.log"), USER="tester",
             PRED_STORE_ROOT=str(tmp_path / "ps"), REFS_FROM=str(tmp_path / "refs"))
    return tmp_path, e


def submit(script, tmp, e, *arms, rung=4, ok=True, **over):
    ee = {**e, **{k: str(v) for k, v in over.items()}}
    p = subprocess.run(["bash", str(script), str(REPO), str(tmp / "tables"),
                        str(tmp / "evalset.json"), str(tmp / "runs"), str(rung), *arms],
                       env=ee, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, p.stderr
        return p.stdout
    return p


def _setup(tmp, name, prefix="mc_"):
    """Run-dir state and live jobs of a case; returns extra env."""
    run = tmp / "runs/x"
    if name in ("foreign_done", "plain_state"):
        (run / "seed0").mkdir(parents=True, exist_ok=True)
        (run / "seed0/DONE").write_text("")
    jobs = {"native_live": [f"77 {prefix}x_s0 train.sh"],
            "score_only_live": [f"77 {prefix}x_s0 train.sh", f"78 {prefix}x_s1 score.sh"],
            "plain_state": [f"79 {prefix}x_s1 train.sh"]}.get(name)
    if not jobs:
        return {}
    (tmp / "jobs").write_text("".join(j + "\n" for j in jobs))
    return {"FAKE_JOBS": str(tmp / "jobs")}


CASES = {
    "plain": {},
    "plain_state": {},
    "phases_p16": {**PH, "SCORE_SUBDIR": "p16_sv"},
    "native_live": {**PH, "SCORE_SUBDIR": "p16_sv_native_depth", "SCORE_INPUT_DEPTH": "native"},
    "foreign_done": {**PH, "SCORE_SUBDIR": "p16_foreign_out_keep_depth",
                     "FOREIGN_COVARIATES": "out_keep_depth"},
    "score_only_live": {**PH, "SCORE_SUBDIR": "p256_sv", "SCORE_ONLY": "1"},
}
#: sha256 of the normalized dry-run stdout of submit_arms.sh at base 57f3c9e8 (before chunk C4)
BASE_DIGESTS = {
    "foreign_done": "c7223f2c06b02668b967070afbd01987396562cc9ab9090a7d62461b9391909e",
    "native_live": "2444c389ce6193895d95cabe731c238bc0473511262a75956d78793c30397f05",
    "phases_p16": "e9cb5200f142262308f1badd89c9623bea15738be9b04815ea4153644f7f486a",
    "plain": "f8f029fa4757ed3ef25716e0cdbca5ae2d20600b9f79ba1a81bc457626127d26",
    "plain_state": "6c2a1026f88a19a2c5feb2e7e9729a50eac27fb4f85f87bc5fd4fb6868545621",
    "score_only_live": "9006e5f498efbc682ef4a661abdd9a26f5fc47c5039f249fa79b524a79656b6d",
}


def _out(script, tmp, e, name, **over):
    extra = _setup(tmp, name, prefix=over.pop("_prefix", "mc_"))
    out = submit(script, tmp, {**e, **extra}, f"x={CFG4}", DRY_RUN=1, **CASES[name], **over)
    return out.replace(str(tmp), "<TMP>").replace(str(REPO), "<REPO>")


@pytest.mark.parametrize("name", sorted(CASES))
def test_job_prefix_unset_is_the_base_script_byte_for_byte(env, name):
    tmp, e = env
    out = _out(SUBMIT, tmp, e, name)
    assert hashlib.sha256(out.encode()).hexdigest() == BASE_DIGESTS[name], out
    assert "--job-name=mc_x_s" in out


@pytest.mark.parametrize("name", sorted(CASES))
def test_job_prefix_empty_renames_every_job_and_every_live_match(env, name):
    tmp, e = env
    base = _out(SUBMIT, tmp, e, name)
    for p in (tmp / "runs").glob("*/*/DONE"):
        p.unlink()
    out = _out(SUBMIT, tmp, e, name, JOB_PREFIX="", _prefix="")
    # the live jobs of the case carry the same new names, so every skip and every dependency
    # is found as before: the output differs only in the job names
    assert out == base.replace("mc_x_s", "x_s")
    assert "mc_" not in out


def test_job_prefix_rcp(env):
    tmp, e = env
    out = submit(SUBMIT, tmp, e, f"rcp_a_unet={CFG4}", DRY_RUN=1, JOB_PREFIX="", SEEDS="0")
    names = re.findall(r"--job-name=(\S+)", out)
    assert names and set(names) == {"rcp_a_unet_s0"}


# ------------------------------------------------------- submit_arms.sh: the SUBMITTED.tsv lock

def _tsv_rows(tmp):
    return [ln.split("\t") for ln in (tmp / "runs/SUBMITTED.tsv").read_text().splitlines()]


@pytest.mark.parametrize("phases", [False, True])
def test_append_runs_under_flock_and_the_line_is_unchanged(env, phases):
    tmp, e = env
    over = {**PH, "SCORE_SUBDIR": "p16_sv"} if phases else {"SEEDS": "0 1"}
    # without flock on PATH: the unlocked fallback
    bare = {**e, "PATH": ":".join(p for p in e["PATH"].split(":")
                                  if not (Path(p) / "flock").exists())}
    submit(SUBMIT, tmp, bare, f"x={CFG4}", DRY_RUN=0, **over)
    plain = _tsv_rows(tmp)
    (tmp / "runs/SUBMITTED.tsv").unlink()
    (tmp / "sbatch.log").unlink()
    (tmp / "bin/flock").write_text(FAKE_FLOCK)
    (tmp / "bin/flock").chmod(0o755)
    for p in (tmp / "runs").rglob("*"):
        if p.is_file() and p.name not in ("SUBMITTED.tsv",):
            p.unlink()
    submit(SUBMIT, tmp, {**e, "FLOCK_LOG": str(tmp / "flock.log")}, f"x={CFG4}",
           DRY_RUN=0, **over)
    locked = _tsv_rows(tmp)
    assert plain[0] == locked[0] == ["arm", "seed", "train_job", "score_job", "time"]
    assert [r[:4] for r in plain] == [r[:4] for r in locked] and len(locked) == 3
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", r[4]) for r in locked[1:])
    assert (tmp / "flock.log").read_text().splitlines() == \
        [str(tmp / "runs/SUBMITTED.tsv.lock")] * 2


def test_two_submissions_at_once_write_one_header_and_whole_lines(env):
    tmp, e = env
    (tmp / "bin/flock").write_text(FAKE_FLOCK)
    (tmp / "bin/flock").chmod(0o755)
    ee = {**e, "FLOCK_LOG": str(tmp / "flock.log"), "DRY_RUN": "0",
          "SEEDS": " ".join(map(str, range(8)))}
    ps = [subprocess.Popen(["bash", str(SUBMIT), str(REPO), str(tmp / "tables"),
                            str(tmp / "evalset.json"), str(tmp / "runs"), "4", f"{n}={CFG4}"],
                           env=ee, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
          for n in ("a", "b")]
    assert all(p.wait() == 0 for p in ps)
    rows = _tsv_rows(tmp)
    assert sum(r[0] == "arm" for r in rows) == 1 and rows[0][0] == "arm"
    assert sorted((r[0], r[1]) for r in rows[1:]) == \
        sorted((n, str(s)) for n in "ab" for s in range(8))
    assert all(len(r) == 5 for r in rows)


# ------------------------------------------------------------------ configs/recipe/submit_recipe.sh

def _onion(tmp):
    o = tmp / "O"
    (o / "tables/eic_v1_2").mkdir(parents=True)
    (o / "tables/eic_v1_2/measurements.tsv").write_text("")
    for ev, b in (("evalset_p16.json", "baselines_p16"), ("evalset.json", "baselines")):
        (o / "eval/eic_v1_2" / ev).parent.mkdir(parents=True, exist_ok=True)
        (o / "eval/eic_v1_2" / ev).write_text("{}")
        for n in ("marginal_nb", "position_average", "depth_only_scaling",
                  "pseudoreplicate_floor"):
            (o / "eval/eic_v1_2" / b / n / "seed0").mkdir(parents=True)
            (o / "eval/eic_v1_2" / b / n / "seed0/SCORED").write_text("")
        (o / "eval/eic_v1_2" / b / "oracle_floor/seed0").mkdir(parents=True)
        (o / "eval/eic_v1_2" / b / "oracle_floor/seed0/oracle_floor.json").write_text("{}")
    return o


def recipe(tmp, e, ok=True, **over):
    ee = {**e, "O": str(tmp / "O"), "K": str(REPO), **{k: str(v) for k, v in over.items()}}
    p = subprocess.run(["bash", str(RECIPE)], env=ee, capture_output=True, text=True)
    if ok:
        assert p.returncode == 0, p.stdout + p.stderr
        return p.stdout
    return p


def _excl():
    t = (CONFIGS / "base_on_version.sh").read_text()
    return re.search(r"^\s*export ONION_EXCLUDE=(\S+)", t, re.M).group(1)


@pytest.mark.parametrize("stage", ["probe", "s1a", "s1b", "s1c"])
def test_each_stage_dry_run_names_only_rcp_jobs(env, stage):
    tmp, e = env
    _onion(tmp)
    out = recipe(tmp, e, STAGE=stage, SUB="p16")
    assert not (tmp / "sbatch.log").exists() and "BLOCKED;" not in out
    names = re.findall(r"--job-name=(\S+)", out)
    runs = {RA.run_name(r, a) for r, a in RA.select(stage)}
    assert names and all(n.startswith("rcp_") for n in names)
    assert {n.rsplit("_s", 1)[0] for n in names} == runs
    # 1 training + 4 predict + 1 metrics per run-seed
    assert len(names) == 6 * 3 * len(runs)
    for ln in (ln for ln in out.splitlines() if " sbatch " in f" {ln}"):
        assert f"--exclude={_excl()}" in ln and "--nice=100" in ln
    for r, a in RA.select(stage):
        tr = [ln for ln in out.splitlines()
              if f"--job-name={RA.run_name(r, a)}_s0 " in ln and "train.sh" in ln]
        assert len(tr) == 1 and f"--time={RA.train_time(a)}" in tr[0]
        assert str(RA.config_path(r, a)) in tr[0]
    assert re.search(rf"jobs {len(names)} \(train {3 * len(runs)}, ", out)


def test_probe_cost_uses_the_measured_unit_costs(env):
    tmp, e = env
    _onion(tmp)
    out = recipe(tmp, e, STAGE="probe", SUB="p16")
    # U-Net, 3 seeds: training 7.06, screen predict 10.78 and metrics 8.27 per run-seed
    assert f"projected {3 * (7.06 + 10.78 + 8.27):.1f} CPU-h" in out
    assert "--output=" + str(tmp / "O/runs_rcp/rcp_probe_base_unet/seed0/p16_sv/") in out
    assert "evalset_p16.json" in out and "baselines_p16" in out


@pytest.mark.parametrize("sub,subdir,shards", [("p256", "p256_sv", 8),
                                               ("native", "p16_sv_native_depth", 4),
                                               ("native256", "p256_sv_native_depth", 8),
                                               ("foreign", "p16_foreign_out_keep_depth", 4)])
def test_score_only_tiers(env, sub, subdir, shards):
    tmp, e = env
    o = _onion(tmp)
    for s in (0, 1, 2):
        for m in ("convattn", "unet"):
            (o / f"runs_rcp/rcp_mix_fixdepth_{m}/seed{s}").mkdir(parents=True)
            (o / f"runs_rcp/rcp_mix_fixdepth_{m}/seed{s}/DONE").write_text("")
    over = {"FOREIGN": "out_keep_depth"} if sub == "foreign" else {}
    out = recipe(tmp, e, STAGE="arms", ARMS="mix_fixdepth", SUB=sub, **over)
    names = re.findall(r"--job-name=(\S+)", out)
    assert names and all(n.startswith("rcp_mix_fixdepth_") for n in names)
    assert "train.sh" not in out
    assert len(names) == 2 * 3 * (shards + 1)
    assert f"/{subdir}/metrics_" in out and f"ONION_SHARD=0/{shards} " in out
    p256 = sub in ("p256", "native256")
    assert ("--time=06:00:00 --cpus-per-task=16" in out) == p256
    refs = "baselines" if p256 else "baselines_p16"
    assert f"ONION_REFS_FROM={o / 'eval/eic_v1_2' / refs} " in out


def test_refusals(env):
    tmp, e = env
    _onion(tmp)
    assert recipe(tmp, e, ok=False, STAGE="s1a", SUB="p16", ARMS="nocov").returncode == 2
    assert recipe(tmp, e, ok=False, STAGE="arms", SUB="p16").returncode == 2
    assert recipe(tmp, e, ok=False, STAGE="arms", SUB="p16", ARMS="bogus").returncode == 2
    assert recipe(tmp, e, ok=False, STAGE="probe", SUB="p32").returncode == 2
    assert recipe(tmp, e, ok=False, STAGE="probe", SUB="foreign").returncode == 2
    assert recipe(tmp, e, ok=False, STAGE="probe", SUB="p16", FOREIGN="x").returncode == 2
    # a real submission refuses while a reference term is missing
    (tmp / "O/eval/eic_v1_2/baselines_p16/marginal_nb/seed0/SCORED").unlink()
    p = recipe(tmp, e, ok=False, STAGE="probe", SUB="p16", DRY_RUN=0)
    assert p.returncode == 3 and not (tmp / "sbatch.log").exists()


def test_refuses_a_run_name_without_rcp(env, tmp_path):
    """The guard of submit_recipe.sh, driven through a kit whose recipe_arms.py lists a bad name."""
    tmp, e = env
    _onion(tmp)
    kit = tmp_path / "kit"
    (kit / "tools/minicandi/configs").mkdir(parents=True)
    (kit / "slurm/minicandi").mkdir(parents=True)
    (kit / "slurm/minicandi/submit_arms.sh").write_text("echo SHOULD-NOT-RUN\n")
    (kit / "tools/minicandi/configs/base_on_version.sh").write_text(
        (CONFIGS / "base_on_version.sh").read_text())
    (kit / "tools/minicandi/recipe_arms.py").write_text(
        "print('4\\tprobe_base\\tmc_probe_base_unet\\tx.json\\t03:00:00')\n")
    p = recipe(tmp, e, ok=False, K=kit, STAGE="probe", SUB="p16")
    assert p.returncode == 2 and "SHOULD-NOT-RUN" not in p.stdout
