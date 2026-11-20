"""t152 chunk CT: compute and storage tally tool. Offline: every cluster reply is recorded text."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "onion_kit" / "compute_tally.py"
_spec = importlib.util.spec_from_file_location("compute_tally", TOOL)
ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ct)

# Recorded 2026-11-11 (Fir, with the login banner that `hpc run fir` prints first).
SACCT_FIR = """\
-------------------------------------------------------------------------------
There are messages associated with the following module(s):
nodejs/24.15.0:
   This module is deprecated.
3430066|p2a_pilot_rung1_s0|COMPLETED|00:31:36|1|26:49.945|billing=1742,cpu=1,gres/gpu:nvidia_h100_80gb_hbm3_1g.10gb=1,gres/gpu=1,mem=10000M,node=1
3430066.batch|batch|COMPLETED|00:31:36|1|26:49.945|cpu=1,gres/gpu:nvidia_h100_80gb_hbm3_1g.10gb=1,gres/gpu=1,mem=10000M,node=1
3430066.extern|extern|COMPLETED|00:31:37|1|00:00:00|billing=1742,cpu=1,gres/gpu:nvidia_h100_80gb_hbm3_1g.10gb=1,gres/gpu=1,mem=10000M,node=1
3432010|p2a_pilot_rung1_cpu_s0|COMPLETED|03:33:57|8|19:13:28|billing=8000,cpu=8,mem=32000M,node=1
3432010.batch|batch|COMPLETED|03:33:57|8|19:13:28|cpu=8,mem=32000M,node=1
3432010.extern|extern|COMPLETED|03:33:57|8|00:00:00|billing=8000,cpu=8,mem=32000M,node=1
"""
# Recorded 2026-11-11 (Nibi): an array task done, one running, plus a failed and a pending line.
SACCT_NIBI = """\
26853261_4|onion_build|COMPLETED|01:28:59|16|04:23:22|billing=16000,cpu=16,mem=62.50G,node=1
26853261_4.batch|batch|COMPLETED|01:28:59|16|04:23:22|cpu=16,mem=62.50G,node=1
26853261_4.extern|extern|COMPLETED|01:29:00|16|00:00:00|billing=16000,cpu=16,mem=62.50G,node=1
26853261_5|onion_build|FAILED|00:10:00|16|00:30:00|billing=16000,cpu=16,mem=62.50G,node=1
26853261_[6-9%50]|onion_build|PENDING|00:00:00|16|00:00:00|billing=16000,cpu=16,mem=62.50G,node=1
26853270_57|onion_build|RUNNING|01:59:24|16|00:00:00|billing=16000,cpu=16,mem=62.50G,node=1
26853270_57.batch|batch|RUNNING|01:59:24|16|00:00:00|cpu=16,mem=62.50G,node=1
26815002|onion_D2_catalog|CANCELLED by 3021|00:08:19|1|00:51.497|billing=1000,cpu=1,mem=4000M,node=1
26815002.batch|batch|CANCELLED|00:08:19|1|00:51.497|cpu=1,mem=4000M,node=1
"""
# Recorded 2026-11-11 (Nibi diskusage_report).
DISKUSAGE_NIBI = """\
                            Description                Space         # of files
                   /home (user mforooz)       626MiB/  50GiB        9212 / 500K
                /scratch (user mforooz)   -> 2737GiB/1024GiB          60K/1000K
           /project (project def-maxwl)      4215GiB/4657GiB         327K/ 500K
            /nearline (group def-maxwl)      4096  B/9537GiB           1 /1025

-> Usage near or over quota (warning threshold at 95% of the quota)
"""


def test_parse_duration():
    assert ct.parse_duration("03:33:57") == pytest.approx(3 + 33 / 60 + 57 / 3600)
    assert ct.parse_duration("26:49.945") == pytest.approx((26 * 60 + 49.945) / 3600)
    assert ct.parse_duration("1-02:00:00") == pytest.approx(26.0)
    assert ct.parse_duration("00:00:00") == 0.0


def test_parse_sacct_alloc_versus_used_and_gpu_slice():
    a = ct.parse_sacct(SACCT_FIR)
    assert set(a) == {"3430066", "3432010"}  # banner and step lines skipped
    cpu = a["3432010"]
    el = 3 + 33 / 60 + 57 / 3600
    assert cpu["cpu_h_alloc"] == pytest.approx(8 * el)  # AllocCPUS x Elapsed = 28.53
    assert cpu["cpu_h_used"] == pytest.approx(19 + 13 / 60 + 28 / 3600)  # TotalCPU = 19.22
    assert cpu["cpu_h_used"] < cpu["cpu_h_alloc"]
    gpu = a["3430066"]
    assert gpu["gpu_slice_h"] == pytest.approx(31 / 60 + 36 / 3600)
    assert gpu["gpu_full_h"] == 0


def test_used_comes_from_steps_when_allocation_line_reads_zero():
    # sacct -X on Nibi prints TotalCPU 00:00:00 on the allocation line; the steps hold the time.
    text = ("26853261_3|onion_build|COMPLETED|00:32:51|16|00:00:00|cpu=16,node=1\n"
            "26853261_3.batch|batch|COMPLETED|00:32:51|16|02:00:00|cpu=16,node=1\n"
            "26853261_3.0|python|COMPLETED|00:10:00|16|01:00:00|cpu=16,node=1\n")
    r = ct.parse_sacct(text)["26853261_3"]
    assert r["cpu_h_used"] == pytest.approx(3.0)
    assert r["parent"] == "26853261"


def test_parse_tres_full_gpu():
    assert ct.parse_tres("cpu=4,gres/gpu:h100=2,gres/gpu=2") == (0, 2)
    assert ct.parse_tres("cpu=4,gres/gpu=1") == (0, 1)
    assert ct.parse_tres("cpu=4,mem=1G") == (0, 0)


def test_ids_in_text_shorthands_and_false_positives():
    ids = lambda s: [j for _, j in ct.ids_in_text(s)]
    assert ids("rung1 s0-2 3438391/3/5; rung2 3439926") == ["3438391", "3438393", "3438395",
                                                            "3439926"]
    assert ids("jobs 26497666/7 on Nibi") == ["26497666", "26497667"]
    assert ids("jobs 3429537-3429540") == ["3429537", "3429538", "3429539", "3429540"]
    assert ids("4 shards: 26816540, 26818325.") == ["26816540", "26818325"]
    assert ids("merged at fe430f4 (8853068; 45 tests)") == []
    assert ids("merged (6cb1093; 8 tests; nibi job 26815002, 0.14 CPU-h)") == ["26815002"]
    assert ids("sha 1d51ed3935a1 12,345,678 and 3.1415926 and 123456789") == []
    assert ids("Fir job 60130071_4 and train_3430066.out") == ["60130071", "3430066"]


def _tree(tmp_path: Path):
    res = tmp_path / "results"
    (res / "t180").mkdir(parents=True)
    (res / "t180" / "SUBMITTED.md").write_text(
        "# EIC build (chunk D6a, Nibi, 2026-11-11)\n\n| 1 | 26853261: 3-31 | 1 day |\n")
    (res / "t181").mkdir()
    (res / "t181" / "SUBMITTED.tsv").write_text(
        "job_id\tcluster\tnote\n3430066\tFir\tgpu\n3432010\t\tcpu twin\n")
    (res / "t999").mkdir()
    (res / "t999" / "notes.md").write_text("job 11111111 is not in a SUBMITTED file\n")
    plan = tmp_path / "plan.md"
    plan.write_text(
        "## Goal\nsee job 22222222 outside the Log\n### D6a EIC build\n### P2a Pilot\n"
        "### X1 Scoring\n### X1c Collect\n## Log\n"
        "- 2026-11-11 X1 code accepted, merged (6d8a984; 12 tests). nibi shards: 26816540, "
        "26818325.\n"
        "- 2026-11-11 X1c DONE (nibi collect 26860155); same shards 26816540\n"
        "- 2026-11-11 The original X1 builder resumed (nibi check 26861850)\n"
        "- 2026-11-11 NETPROBE nibi 26824325\n"
        "- 2026-11-11 D6a DONE: P1 26853261 again\n"
        "- 2026-11-11 F2 Fir check 3436642 and smoke 26828055\n")
    tmap = tmp_path / "map.md"
    tmap.write_text("| chunk id | task id | title |\n|---|---|---|\n| D6a | t180 | b |\n"
                    "| P2a | t181 | p |\n")
    return res, plan, tmap


def test_collect_jobs_attribution(tmp_path):
    res, plan, tmap = _tree(tmp_path)
    rows = {(r["cluster"], r["job_id"]): r for r in ct.collect_jobs(res, plan, tmap)}
    assert ("nibi", "11111111") not in rows and ("nibi", "22222222") not in rows
    assert ("fir", "22222222") not in rows
    b = rows[("nibi", "26853261")]
    assert (b["chunk"], b["cluster_source"]) == ("D6a", "title")
    assert b["also_in"] == ""  # same chunk in the Log: not "also in"
    assert rows[("fir", "3430066")]["chunk"] == "P2a"
    assert rows[("fir", "3430066")]["cluster_source"] == "column"
    assert rows[("fir", "3432010")]["cluster"] == "fir"  # empty column -> the file guess
    s = rows[("nibi", "26816540")]
    assert s["chunk"] == "X1" and s["also_in"] == "X1c"
    assert rows[("nibi", "26861850")]["chunk"] == "X1"
    assert rows[("nibi", "26824325")]["chunk"] == "unattributed"
    assert rows[("fir", "3436642")]["chunk"] == "F2"
    # "Fir" is the nearest cluster word before 26828055 in its line.
    assert rows[("fir", "26828055")]["cluster_source"] == "line"


def test_tally_counts_and_totals():
    jobs = [dict(job_id="26853261", cluster="nibi", chunk="D6a"),
            dict(job_id="26853270", cluster="nibi", chunk="D6a"),
            dict(job_id="26815002", cluster="nibi", chunk="D2"),
            dict(job_id="26999999", cluster="nibi", chunk="D2"),
            dict(job_id="3432010", cluster="fir", chunk="P2a")]
    rows = {(r["chunk"], r["cluster"]): r for r in
            ct.tally(jobs, {"nibi": SACCT_NIBI, "fir": SACCT_FIR})}
    d6a = rows[("D6a", "nibi")]
    assert (d6a["allocations"], d6a["failed"], d6a["running"], d6a["pending"]) == (4, 1, 1, 1)
    alloc = 16 * (ct.parse_duration("01:28:59") + ct.parse_duration("00:10:00")
                  + ct.parse_duration("01:59:24"))
    assert float(d6a["cpu_h_alloc"]) == pytest.approx(alloc, abs=0.01)
    assert float(d6a["cpu_h_used"]) == pytest.approx(4 + 23 / 60 + 22 / 3600 + 0.5, abs=0.01)
    d2 = rows[("D2", "nibi")]
    assert (d2["cancelled"], d2["not_found"], d2["jobs_listed"]) == (1, 1, 2)
    tot = rows[("TOTAL", "nibi")]
    assert tot["allocations"] == 5 and tot["not_found"] == 1
    assert float(rows[("TOTAL", "fir")]["cpu_h_alloc"]) == pytest.approx(28.53, abs=0.01)
    assert rows[("TOTAL", "fir")]["used_over_alloc"] == "0.674"


def test_parse_diskusage_recorded():
    rows = {r["path"].split()[0]: r for r in ct.parse_diskusage(DISKUSAGE_NIBI, "nibi")}
    assert set(rows) == {"/home", "/scratch", "/project", "/nearline"}
    s = rows["/scratch"]
    assert (s["gib"], s["gib_quota"], s["files"], s["files_quota"]) == (
        "2737.0", "1024.0", "60000", "1000000")
    assert float(s["space_pct"]) > 100
    assert rows["/project"]["files_pct"] == "65.4"
    assert rows["/home"]["files"] == "9212"
    assert rows["/nearline"]["files"] == "1" and rows["/nearline"]["files_quota"] == "1025"


def _fake_runner(replies):
    calls = []

    def run(cluster, cmd):
        calls.append((cluster, cmd))
        return replies[cluster]
    return run, calls


def test_storage_and_milestone_end_to_end(tmp_path):
    res, plan, tmap = _tree(tmp_path)
    out = tmp_path / "ms"
    dirs = ("ONIONDIR|/project/def-maxwl/mforooz/ONION|1048576|300\n"
            "ONIONDIR|/nearline/def-maxwl/mforooz/onion|NA|NA\n")
    run, calls = _fake_runner({"nibi": SACCT_NIBI + dirs + DISKUSAGE_NIBI, "fir": SACCT_FIR})
    assert ct.main(["jobs", "--results", str(res), "--orchestrate", str(plan), "--task-map",
                    str(tmap), "--out", str(out / "jobs.tsv")], runner=run) == 0
    assert ct.main(["sacct", "--jobs", str(out / "jobs.tsv"), "--out", str(out / "tally.tsv")],
                   runner=run) == 0
    assert all("sacct -j " in c and " -X" not in c and "TotalCPU" in c for _, c in calls)
    assert ct.main(["storage", "--clusters", "nibi", "--out", str(out / "storage.tsv")],
                   runner=run) == 0
    assert "du -sk /scratch/mforooz/onion" in calls[-1][1] and "diskusage_report" in calls[-1][1]
    st = ct.read_tsv(out / "storage.tsv")
    onion = [r for r in st if r["kind"] == "dir"]
    assert onion[0]["gib"] == "1.00" and onion[1]["gib"] == "absent"
    md_path = out / "G1.md"
    rc = ct.main(["milestone", "--name", "G1", "--out", str(md_path)], runner=run)
    assert rc == 0
    md = md_path.read_text()
    assert "CPU-h allocated" in md and "CPU-h used" in md and "/scratch (user mforooz)" in md
    _, entry = ct.milestone("G1", ct.read_tsv(out / "jobs.tsv"), ct.read_tsv(out / "tally.tsv"),
                            st, {"out": str(md_path)}, when="2026-11-11 20:00 UTC")
    assert entry.startswith("## 2026-11-11 Milestone G1: compute and storage")
    assert "allocated (AllocCPUS x Elapsed)" in entry and "used (TotalCPU)" in entry
    assert "Quotas at or over 80%: nibi /scratch" in entry


def test_tunnel_down_exits_69(tmp_path, capsys):
    def down(cluster, cmd):
        raise ct.TunnelDown(cluster)
    assert ct.main(["storage", "--clusters", "nibi", "--out", str(tmp_path / "s.tsv")],
                   runner=down) == 69
    assert "hpc up nibi" in capsys.readouterr().err
    assert not (tmp_path / "s.tsv").exists()


def test_cli_hpc_exit_69_through_real_subprocess(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "hpc").write_text("#!/bin/sh\nexit 69\n")
    (fake / "hpc").chmod(0o755)
    env = dict(os.environ, PATH=f"{fake}{os.pathsep}{os.environ['PATH']}")
    p = subprocess.run([sys.executable, str(TOOL), "storage", "--clusters", "fir", "--out",
                        str(tmp_path / "s.tsv")], capture_output=True, text=True, env=env)
    assert p.returncode == 69 and "tunnel to fir is down" in p.stderr
