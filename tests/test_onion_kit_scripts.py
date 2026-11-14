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


def _run(script: Path, *args: str, cwd: Path, cluster_env: str | None = None,
         exclude_env: str | None = ""):
    """exclude_env: the ONION_EXCLUDE value ('' = no default exclude list; None = unset, so the
    exclude files decide)."""
    env = {k: v for k, v in os.environ.items() if k not in ("ONION_CLUSTER", "ONION_EXCLUDE")}
    env["ONION_DRY_RUN"] = "1"
    if cluster_env is not None:
        env["ONION_CLUSTER"] = cluster_env
    if exclude_env is not None:
        env["ONION_EXCLUDE"] = exclude_env
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
    assert "--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" in gpu.stdout
    assert re.search(r"sbatch .* --exclude=fc10713,fc10715 ", gpu.stdout)
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


# ------------------------------------------------------------------ F2: exclude list and retry

def _exclude(out: str):
    m = re.search(r"^mkdir .*sbatch .*?(?: --exclude=(\S+))? '/project", out, re.M)
    assert m, out
    return m.group(1)


@pytest.mark.parametrize("script,args", [
    (CHECK, ["K0", "-c", "pass"]), (RUN, ["K0", "1", "4000", "00:10:00", "-c", "pass"]),
], ids=["check", "run"])
def test_exclude_lookup_order_and_flag(script, args, tmp_path):
    (tmp_path / ".orchestrate").mkdir()
    (tmp_path / ".orchestrate" / "exclude_fir").write_text("# bad nodes\nfc1, fc2\nfc3 # hung\n")
    r = _run(script, "--cluster", "fir", *args, cwd=tmp_path, exclude_env=None)
    assert _exclude(r.stdout) == "fc1,fc2,fc3"                                # ./.orchestrate
    assert _exclude(_run(script, "--cluster", "nibi", *args, cwd=tmp_path, exclude_env=None)
                    .stdout) in (None, _umbrella_exclude("nibi"))           # per cluster
    r = _run(script, "--cluster", "fir", *args, cwd=tmp_path, exclude_env="fc9")
    assert _exclude(r.stdout) == "fc9"                                        # env wins
    r = _run(script, "--cluster", "fir", "--exclude", "fc3,fc4", *args, cwd=tmp_path,
             exclude_env=None)
    assert _exclude(r.stdout) == "fc1,fc2,fc3,fc4"                            # flag adds
    assert _exclude(_run(script, "--cluster", "fir", *args, cwd=tmp_path).stdout) is None
    assert "ONION_EXCLUDE_NODES='fc1,fc2,fc3'" in _run(script, "--cluster", "fir", *args,
                                                      cwd=tmp_path, exclude_env=None).stdout
    bad = _run(script, "--cluster", "fir", "--exclude", "fc1;rm", *args, cwd=tmp_path)
    assert bad.returncode == 2 and "bad --exclude" in bad.stderr


def _umbrella_exclude(cluster):
    f = UMBRELLA.parent / f"exclude_{cluster}"
    if not f.is_file():
        return None
    names = re.findall(r"[A-Za-z0-9._-]+", re.sub(r"#.*", "", f.read_text()))
    return ",".join(dict.fromkeys(names))


def test_exclude_umbrella_fallback(tmp_path):
    r = _run(CHECK, "--cluster", "fir", "K0", "-c", "pass", cwd=tmp_path, exclude_env=None)
    assert _exclude(r.stdout) == _umbrella_exclude("fir")


@pytest.mark.parametrize("script,args", [
    (CHECK, ["K0", "-c", "pass"]), (RUN, ["K0", "1", "4000", "00:10:00", "-c", "pass"]),
], ids=["check", "run"])
def test_resubmit_options_match_the_sbatch_line(script, args, tmp_path):
    out = _run(script, "--cluster", "nibi", *args, cwd=tmp_path).stdout
    opts = re.search(r"ONION_RESUBMIT='([^']*)'", out).group(1)
    assert f"sbatch --parsable --export=ALL {opts} '" in out
    assert "unset ONION_RETRY_OF" in out


def _stub_bin(tmp_path: Path, venv_ok: bool) -> Path:
    b = tmp_path / "bin"
    b.mkdir()
    stubs = {
        "virtualenv": "exit 0" if venv_ok else "echo 'OSError: [Errno 5] Input/output error: /cvmfs' >&2; exit 1",
        "sbatch": f'echo "$*" > {tmp_path}/sbatch_args; echo "${{ONION_RETRY_OF:-}}|${{ONION_EXCLUDE_NODES:-}}" > {tmp_path}/sbatch_env; echo 4242',
        "scontrol": "exit 0", "module": "exit 0", "pip": "exit 0",
    }
    for name, body in stubs.items():
        (b / name).write_text("#!/bin/bash\n" + body + "\n")
        (b / name).chmod(0o755)
    return b


@pytest.mark.parametrize("job,tag", [(SLURM / "check.sh", "onion-check"), (SLURM / "run.sh", "onion-run")],
                         ids=["check", "run"])
def test_job_retries_once_on_another_node_when_setup_fails(job, tag, tmp_path):
    b = _stub_bin(tmp_path, venv_ok=False)
    env = dict(os.environ, PATH=f"{b}:{os.environ['PATH']}", SLURM_JOB_ID="100",
               SLURM_TMPDIR=str(tmp_path), ONION_RESUBMIT="--time=00:10:00 --output=x_%j.out",
               ONION_EXCLUDE_NODES="fcA")
    env.pop("ONION_RETRY_OF", None)
    r = subprocess.run(["bash", str(job), str(tmp_path / "kit"), "-c", "print(1)"], env=env,
                       capture_output=True, text=True, timeout=60)
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    assert r.returncode == 3
    assert f"[{tag}] retried-as=4242 host={host}" in r.stdout and f"[{tag}] exit=3" in r.stdout
    assert f"[{tag}] env-ready" not in r.stdout
    args = (tmp_path / "sbatch_args").read_text().split()
    assert args[:5] == ["--parsable", "--export=ALL", "--time=00:10:00", "--output=x_%j.out",
                        f"--exclude=fcA,{host}"]
    assert args[5:] == [str(tmp_path / "kit" / "slurm/onion_kit" / job.name), str(tmp_path / "kit"),
                        "-c", "print(1)"]
    assert (tmp_path / "sbatch_env").read_text().strip() == f"100|fcA,{host}"
    (tmp_path / "sbatch_args").unlink()
    env["ONION_RETRY_OF"] = "99"                                    # a retry never retries again
    r = subprocess.run(["bash", str(job), str(tmp_path / "kit"), "-c", "print(1)"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "retried-as" not in r.stdout
    assert not (tmp_path / "sbatch_args").exists()


def test_remote_check_follows_the_retry(tmp_path):
    """The first job died in setup and resubmitted itself; remote_check prints both ids, waits on
    the retry and returns its exit code."""
    b = tmp_path / "bin"
    b.mkdir()
    (b / "ssh").write_text(r"""#!/bin/bash
for a in "$@"; do last="$a"; done
case "$*" in *"-O check"*) exit 0 ;; esac
case "$last" in
  *sbatch*) echo 111 ;;
  *squeue*) ;;
  *check_111.out*) printf 'check.sh: venv setup failed on fcX
[onion-check] retried-as=222 host=fcX
[onion-check] exit=3
' ;;
  *check_222.out*) printf 'hello
[onion-check] env-ready host=fcY
[onion-check] exit=0
' ;;
esac
""")
    (b / "ssh").chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ONION_")}
    env.update(PATH=f"{b}:{os.environ['PATH']}", ONION_EXCLUDE="fc1")
    r = subprocess.run(["bash", str(CHECK), "--cluster", "fir", "K0", "-c", "pass"], cwd=tmp_path,
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--exclude=fc1" in r.stdout.splitlines()[0]                      # printed sbatch line
    assert "Jobs: 111 222" in r.stdout and "excluding fcX" in r.stdout
    assert "hello" in r.stdout and "[onion-check]" not in r.stdout


def test_umbrella_exclude_fir_lists_the_dead_nodes():
    names = (_umbrella_exclude("fir") or "").split(",")
    if not (UMBRELLA.parent / "exclude_fir").is_file():
        pytest.skip("no umbrella checkout on this machine")
    assert set(names) >= {"fc10713", "fc10715", "fc30601", "fc30568", "fc30605", "fc30564",
                          "fc30654", "fc30435", "fc30562", "fc30355"}


# ------------------------------------------------------------------ F3: dependent jobs retry too

SCONTROL_RECORD = """JobId=100 JobName=p2a_guard_rung1
   Account=def-maxwl_cpu QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   RunTime=00:00:05 TimeLimit=01:30:00 TimeMin=N/A
   ReqNodeList=(null) ExcNodeList=fc[10713,10715]
   NumNodes=1 NumCPUs=4 NumTasks=1 CPUs/Task=4 ReqB:S:C:T=0:0:*:*
   MinCPUsNode=4 MinMemoryNode=16000M MinTmpDiskNode=0
   StdOut=/logs/check_100.out
"""


def test_raw_sbatch_check_job_rebuilds_its_retry_options(tmp_path):
    """DEFECTS item 1: a guardrail job chained by raw `sbatch --dependency=afterok` had no
    ONION_RESUBMIT, so a CVMFS death in setup was never retried. check.sh now rebuilds the options
    (no dependency, no account) and the exclude list from `scontrol show job`."""
    b = _stub_bin(tmp_path, venv_ok=False)
    rec = tmp_path / "record.txt"
    rec.write_text(SCONTROL_RECORD)
    (b / "scontrol").write_text(f"#!/bin/bash\ncat {rec}\n")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ONION_")}
    env.update(PATH=f"{b}:{os.environ['PATH']}", SLURM_JOB_ID="100", SLURM_TMPDIR=str(tmp_path))
    r = subprocess.run(["bash", str(SLURM / "check.sh"), str(tmp_path / "kit"), "-c", "print(1)"],
                       env=env, capture_output=True, text=True, timeout=60)
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    assert r.returncode == 3 and "[onion-check] retried-as=4242" in r.stdout, r.stdout
    args = (tmp_path / "sbatch_args").read_text().split()
    assert args[:2] == ["--parsable", "--export=ALL"]
    opts = args[2:8]
    assert opts == ["--job-name=p2a_guard_rung1", "--time=01:30:00", "--cpus-per-task=4",
                    "--mem=16000M", "--gres=none", "--output=/logs/check_%j.out"]
    assert args[8] == f"--exclude=fc[10713,10715],{host}"
    assert not any(a.startswith(("--dependency", "--account")) for a in args)
    # an unreadable record leaves the job without a retry, as before
    (b / "scontrol").write_text("#!/bin/bash\nexit 0\n")
    (tmp_path / "sbatch_args").unlink()
    r = subprocess.run(["bash", str(SLURM / "check.sh"), str(tmp_path / "kit"), "-c", "print(1)"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and "retried-as" not in r.stdout
    assert not (tmp_path / "sbatch_args").exists()


def test_remote_run_after_adds_the_dependency_to_the_first_submission_only(tmp_path):
    r = _run(RUN, "--cluster", "fir", "--after", "3430066", "P2a", "4", "16000", "01:00:00",
             "tools/onion_eval/guardrail.py", "--run", "r", cwd=tmp_path, exclude_env="fcA")
    assert r.returncode == 0, r.stderr
    assert "--exclude=fcA --dependency=afterok:3430066 '" in r.stdout
    opts = re.search(r"ONION_RESUBMIT='([^']*)'", r.stdout).group(1)
    assert "dependency" not in opts
    assert _run(RUN, "--cluster", "fir", "--after", "12x", "P2a", "4", "16000", "01:00:00", "-c",
                "1", cwd=tmp_path).returncode == 2
