"""Tail watchdog (PI ruling 38): latest task per row, failure classes, done rule."""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from onion_data import tail_watch as T  # noqa: E402

SACCT = [
    "26853261_46|FAILED", "26853261_56|TIMEOUT", "26853275_[4,6,10-12,46%50]|PENDING",
    "27087748_96|COMPLETED", "27105143_396|RUNNING", "26855582_396|FAILED",
    "27081354_132|CANCELLED by 1234", "junk line", "27123484_[10,37,63,84]|PENDING",
]


def test_expand_spec():
    assert T.expand_spec("10,37-39%50") == [10, 37, 38, 39] and T.expand_spec("4") == [4]


def test_latest_task_per_row_takes_the_newest_job_and_expands_pending_arrays():
    got = T.latest_tasks(SACCT, [10, 11, 46, 56, 96, 132, 396, 999])
    assert got[46] == ("26853275_46", "PENDING")            # newer than the failed build
    assert got[10] == ("27123484_10", "PENDING") and got[11] == ("26853275_11", "PENDING")
    assert got[56] == ("26853261_56", "TIMEOUT")
    assert got[396] == ("27105143_396", "RUNNING") and got[132][1] == "CANCELLED"
    assert 999 not in got


def test_failure_classes_from_the_log():
    assert T.failure_class("RUNNING", "GC overhead limit exceeded") == ""
    assert T.failure_class("COMPLETED", "") == ""
    assert T.failure_class("FAILED", "java.lang.OutOfMemoryError: GC overhead limit exceeded") \
        == "java heap"
    assert T.failure_class("TIMEOUT", "") == "time limit"
    assert T.failure_class("FAILED", "CANCELLED AT x DUE TO TIME LIMIT") == "time limit"
    assert T.failure_class("OUT_OF_MEMORY", "") == "slurm memory"
    assert T.failure_class("FAILED", "[mem] oom_kill=1") == "slurm memory"
    assert T.failure_class("FAILED", "=== views step exit=75 now") == \
        "waiting for control (exit 75)"
    assert T.failure_class("FAILED", "ValueError: No reads found") == "other"


def test_status_done_rule_and_live_jobs():
    tail = [{"row": "56", "experiment": "E56", "eic_track": "C50M02", "cell": "C50",
             "role": "val", "assay": "DNase-seq", "priority": "1"},
            {"row": "96", "experiment": "E96", "eic_track": "C25M21", "cell": "C25",
             "role": "val", "assay": "H3K4me2", "priority": "1"},
            {"row": "396", "experiment": "E396", "eic_track": "C15M02", "cell": "C15",
             "role": "train", "assay": "DNase-seq", "priority": "2"}]
    latest = T.latest_tasks(SACCT, [56, 96, 396])
    logs = {"26853261_56": "error: *** JOB CANCELLED AT 2026 DUE TO TIME LIMIT ***"}
    table = T.status_table(tail, latest, {96: True}, logs)
    by = {t["row"]: t for t in table}
    assert by["56"]["failure_class"] == "time limit" and by["56"]["verified"] == "no"
    assert by["96"]["verified"] == "yes" and by["96"]["failure_class"] == ""
    assert by["396"]["state"] == "RUNNING" and not T.is_done(table)
    assert T.live_jobs(table) == ["27105143"]
    done = T.status_table(tail, {**latest, 396: ("27105143_396", "COMPLETED")}, {96: True}, logs)
    assert T.is_done(done) and T.live_jobs(done) == []
    # a tail row with no job at all is not done
    assert not T.is_done(T.status_table(tail[:1], {}, {}, {}))


def test_cli_init_and_status_write_files(tmp_path, monkeypatch):
    import subprocess
    bl = tmp_path / "bl.tsv"
    bl.write_text("row\tpriority\texperiment\teic_track\trole\tassay\tunit\tarm\t"
                  "expected_measurements\n"
                  "56\t1\tE56\tC50M02\tval\tDNase-seq\tpooled\tbase\tE56.pooled.uniform.base:counts\n"
                  "96\t1\tE96\tC25M21\tval\tH3K4me2\tpooled\tcrop50\tE96.pooled.uniform.crop50:counts\n")
    td = tmp_path / "tail"
    assert T.main(["init", "--build-list", str(bl), "--rows", "56,96", "--out-dir", str(td)]) == 0
    meas = tmp_path / "meas"
    (meas / "E96.pooled.uniform.crop50").mkdir(parents=True)
    (meas / "E96.pooled.uniform.crop50" / "provenance.json").write_text("{}")

    def fake_run(cmd, **kw):
        assert cmd[0] == "sacct"
        return subprocess.CompletedProcess(cmd, 0, "26853261_56|RUNNING\n27087748_96|COMPLETED\n", "")
    monkeypatch.setattr(T.subprocess, "run", fake_run)
    args = ["status", "--build-list", str(bl), "--out-dir", str(td), "--measurements", str(meas),
            "--work", str(tmp_path / "work"), "--logs", str(tmp_path)]
    assert T.main(args) == 3
    st = (td / "TAIL_STATUS.tsv").read_text().splitlines()
    assert st[0].split("\t")[:3] == ["row", "experiment", "eic_track"] and len(st) == 3
    assert not (td / "TAIL_DONE").exists()

    def fake_done(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, "26853261_56|TIMEOUT\n27087748_96|COMPLETED\n", "")
    monkeypatch.setattr(T.subprocess, "run", fake_done)
    assert T.main(args) == 0
    assert "56(time limit)" in (td / "TAIL_DONE").read_text() or \
        "56(TIMEOUT)" in (td / "TAIL_DONE").read_text()
