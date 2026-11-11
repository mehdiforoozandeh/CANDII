"""t152 chunk K0: the ONION kit tools (push, remote check, remote run) and their job scripts.

Offline checks only. Every tool honours ONION_DRY_RUN=1, under which it resolves the cluster,
validates its arguments and prints the command it would send, without touching ssh.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "onion_kit"
SLURM = ROOT / "slurm" / "onion_kit"
PUSH, CHECK, RUN = TOOLS / "push_kit.sh", TOOLS / "remote_check.sh", TOOLS / "remote_run.sh"
JOBS = [SLURM / "check.sh", SLURM / "run.sh", SLURM / "probe.sh"]
UMBRELLA = Path("/Users/mforooz/Desktop/research/libbrechteam@sfu/CANDII/.claude/worktrees/"
                "onion/.orchestrate/cluster")
KIT = "/project/def-maxwl/mforooz/onion_kit/K0"


def _run(script: Path, *args: str, cwd: Path, cluster_env: str | None = None):
    env = {k: v for k, v in os.environ.items() if k != "ONION_CLUSTER"}
    env["ONION_DRY_RUN"] = "1"
    if cluster_env is not None:
        env["ONION_CLUSTER"] = cluster_env
    return subprocess.run(["bash", str(script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=60)


def _cluster(out: str) -> str:
    return re.search(r"^cluster=(\w+)$", out, re.M).group(1)


@pytest.mark.parametrize("script", [PUSH, CHECK, RUN, *JOBS], ids=lambda p: p.name)
def test_bash_syntax(script: Path):
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


@pytest.mark.parametrize("script,args", [
    (PUSH, ["K0"]), (CHECK, ["K0", "-c", "pass"]), (RUN, ["K0", "1", "4000", "00:10:00", "-c", "pass"]),
], ids=["push", "check", "run"])
def test_cluster_lookup_order(script, args, tmp_path):
    (tmp_path / ".orchestrate").mkdir()
    (tmp_path / ".orchestrate" / "cluster").write_text("fir\n")
    assert _cluster(_run(script, *args, cwd=tmp_path).stdout) == "fir"            # ./.orchestrate
    assert _cluster(_run(script, *args, cwd=tmp_path, cluster_env="nibi").stdout) == "nibi"  # env wins
    r = _run(script, "--cluster", "fir", *args, cwd=tmp_path, cluster_env="nibi")
    assert _cluster(r.stdout) == "fir"                                              # flag wins
    bad = _run(script, "--cluster", "cedar", *args, cwd=tmp_path)
    assert bad.returncode == 2 and "unknown cluster" in bad.stderr


def test_umbrella_fallback(tmp_path):
    """From a chunk worktree (no .orchestrate/, no env) the umbrella checkout's file decides."""
    r = _run(PUSH, "K0", cwd=tmp_path)
    if UMBRELLA.is_file():
        assert _cluster(r.stdout) == UMBRELLA.read_text().strip()
    else:
        assert r.returncode == 2 and "no cluster" in r.stderr


def test_push_kit_rsync_line(tmp_path):
    r = _run(PUSH, "--cluster", "nibi", "K0", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert f"dest={KIT}" in r.stdout
    line = r.stdout.strip().splitlines()[-1]
    for flag in ("--no-g", "--no-p", "--delete", "BatchMode=yes", f"nibi:{KIT}/"):
        assert flag in line
    assert int(re.search(r"^files=(\d+)$", r.stdout, re.M).group(1)) > 100
    text = PUSH.read_text()
    assert r"^(\.git/|\.claude/|\.orchestrate/|cruxvault/raw/)" in text
    assert "P /wheels" in text and "/project/def-maxwl/mforooz/onion_kit/wheels" in text


@pytest.mark.parametrize("kit_id", ["", "../x", "a b", "wheels", "logs", "-x"])
def test_push_kit_rejects_bad_ids(kit_id, tmp_path):
    assert _run(PUSH, "--cluster", "fir", kit_id, cwd=tmp_path).returncode == 2


def test_remote_check_quotes_python_args(tmp_path):
    code = 'import h5py, torch, candi; print(candi.__file__, "x y", $HOME)'
    r = _run(CHECK, "--cluster", "nibi", "K0", "-c", code, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    sub = r.stdout.strip().splitlines()[-1]
    assert "--gres=none" in sub and "--time=00:30:00" in sub and "sbatch --parsable" in sub
    tail = sub.split(f"'{KIT}/slurm/onion_kit/check.sh' '{KIT}'", 1)[1]
    got = subprocess.run(["bash", "-c", f"printf '%s\\n' {tail}"], capture_output=True, text=True)
    assert got.stdout.splitlines() == ["-c", code]


def test_remote_check_options_and_wait_form(tmp_path):
    r = _run(CHECK, "--cluster", "fir", "--time", "02:00:00", "--cpus", "8", "--mem", "32000",
             "main", "-m", "pytest", "tests/", "-q", cwd=tmp_path)
    assert "--time=02:00:00 --cpus-per-task=8 --mem=32000M" in r.stdout
    w = _run(CHECK, "--cluster", "nibi", "--wait", "123456", cwd=tmp_path)
    assert w.returncode == 0 and "wait 123456" in w.stdout
    for bad in (["--wait", "12ab"], ["K0"], ["--time", "1h", "K0", "-c", "pass"],
                ["--mem", "8G", "K0", "-c", "pass"], ["--max-wait", "x", "K0", "-c", "pass"]):
        assert _run(CHECK, "--cluster", "nibi", *bad, cwd=tmp_path).returncode == 2, bad


def test_remote_check_bounded_wait_text():
    text = CHECK.read_text()
    assert "exit 75" in text and "Do not resubmit" in text and "--wait $JOB" in text
    assert r"\[onion-check\] exit=" in text and "exit 69" in text


def test_remote_run_cpu_and_gpu(tmp_path):
    cpu = _run(RUN, "--cluster", "fir", "D1", "4", "16000", "02:00:00", "x.py", "--a", "b c",
               cwd=tmp_path)
    assert cpu.returncode == 0, cpu.stderr
    assert "--gres=none" in cpu.stdout and "fc10713" not in cpu.stdout
    assert "--cpus-per-task=4 --mem=16000M" in cpu.stdout and "--time=02:00:00" in cpu.stdout
    gpu = _run(RUN, "--cluster", "fir", "--gpu", "T1", "1", "10000", "1-00:00:00", "-c", "pass",
               cwd=tmp_path)
    assert "--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --exclude=fc10713,fc10715" in gpu.stdout
    gn = _run(RUN, "--cluster", "nibi", "--gpu", "T1", "2", "10000", "01:00:00", "-c", "pass",
              cwd=tmp_path)
    assert "nvidia_h100_80gb_hbm3_1g.10gb:1" in gn.stdout and "--exclude" not in gn.stdout
    for bad in (["K0", "0", "4000", "00:10:00", "-c", "1"], ["K0", "1", "4G", "00:10:00", "-c", "1"],
                ["K0", "1", "4000", "10m", "-c", "1"], ["K0", "1", "4000", "00:10:00"]):
        assert _run(RUN, "--cluster", "nibi", *bad, cwd=tmp_path).returncode == 2, bad


@pytest.mark.parametrize("job", JOBS, ids=lambda p: p.name)
def test_job_headers(job: Path):
    text = job.read_text()
    assert "#SBATCH --account=def-maxwl" in text
    assert re.search(r"^#SBATCH --time=\d\d:\d\d:\d\d$", text, re.M)
    assert "#SBATCH --gres=none" in text
    assert "--partition" not in text


@pytest.mark.parametrize("job", [SLURM / "check.sh", SLURM / "run.sh"], ids=lambda p: p.name)
def test_python_jobs_follow_the_kit_template(job: Path):
    text = job.read_text()
    for needle in ("module load python/3.10.13", 'virtualenv --no-download "$SLURM_TMPDIR/venv"',
                   '"$KIT/requirements-fir.txt"', '--find-links "$KIT/wheels"',
                   '"$KIT/requirements-pypi.txt"', 'cd "$KIT"', 'python "$@"'):
        assert needle in text, needle
    assert text.index("bin/activate") < text.index('source "$KIT/slurm/_kit_pin.sh"')
    marker = "[onion-check] exit=$rc" if job.name == "check.sh" else "[onion-run] exit=$rc"
    assert marker in text


def test_probe_covers_internet_and_write():
    text = (SLURM / "probe.sh").read_text()
    assert "https://www.encodeproject.org/?format=json" in text
    assert "encode-public.s3.amazonaws.com" in text
    assert "count=1024" in text and "1073741824" in text and 'rm -f "$F"' in text
    for line in ("internet: yes", "internet: no", "write_test: pass", "write_test: fail"):
        assert line in text
