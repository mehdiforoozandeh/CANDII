"""Tail watchdog of the EIC build (PI ruling 38, 2026-11-13): eic_v1 freezes with what is verified;
the rows still building (the tail) are watched until each is terminal, then the dataset is updated
with notes. This module only reads: it never submits a build row and never deletes anything.

    tail_watch.py init   --build-list B --rows SPEC --out-dir D
        write D/tail_rows.tsv (row, experiment, eic_track, cell, role, assay, priority)
    tail_watch.py status --build-list B --out-dir D [--measurements M --work W --logs L]
        write D/TAIL_STATUS.tsv: per tail row its latest onion_build / onion_signal array task
        (sacct, any kit), the task's state, whether the row is verified (every expected
        measurement has provenance.json, or NOT_APPLICABLE: build_list.row_state), and the
        failure class read from the task's log. When every tail row is verified or its latest
        task is terminal, also write D/TAIL_DONE and exit 0; else print `live <job ids>` (the
        array jobs still queued or running) and exit 3.

slurm/onion_data/tail_watch.sh runs `status` as a 1-CPU job and resubmits itself while the tail
is not done (at most once per 6 h). Stdlib only (runs under the login python3).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_list as BL  # noqa: E402  (stdlib only: row_state, parse_spec, read_tsv)

OUT = "/scratch/mforooz/onion/measurements"
WORK = "/scratch/mforooz/onion/work"
LOGS = "/project/def-maxwl/mforooz/ONION/logs"
SINCE = "2026-11-11"
JOB_NAMES = "onion_build,onion_signal"
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "CANCELLED", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
TAIL_COLS = ("row", "experiment", "eic_track", "cell", "role", "assay", "priority")
STATUS_COLS = TAIL_COLS + ("job", "state", "verified", "failure_class", "checked_utc")


def expand_spec(spec: str) -> list[int]:
    """An array index spec as sacct prints it: `3`, `1-4`, `1,3-5%50`."""
    spec = spec.split("%")[0]
    out = []
    for part in spec.split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def latest_tasks(sacct_lines, rows) -> dict:
    """row -> (task id, state) of the newest array job holding that row. Lines are
    `JobID|State` (sacct -X -P, then squeue -r); a pending array prints as `<job>_[<spec>]`.
    For one job id a later line wins, so squeue (current) overrides sacct."""
    want, best = set(rows), {}
    for ln in sacct_lines:
        if "|" not in ln:
            continue
        jid, state = ln.split("|")[:2]
        state = state.split()[0] if state else ""
        m = re.fullmatch(r"(\d+)_(\d+)", jid)
        if m:
            items = [(int(m.group(1)), int(m.group(2)))]
        else:
            m = re.fullmatch(r"(\d+)_\[(.+)\]", jid)
            if not m:
                continue
            items = [(int(m.group(1)), r) for r in expand_spec(m.group(2))]
        for job, r in items:
            if r in want and (r not in best or job >= best[r][0]):
                best[r] = (job, state)
    return {r: (f"{job}_{r}", st) for r, (job, st) in best.items()}


def failure_class(state: str, log_text: str) -> str:
    """'' while queued, running or completed; else the class read from the log."""
    if state not in TERMINAL or state == "COMPLETED":
        return ""
    if "OutOfMemoryError" in log_text or "GC overhead limit exceeded" in log_text:
        return "java heap"
    if state == "TIMEOUT" or "DUE TO TIME LIMIT" in log_text:
        return "time limit"
    if state == "OUT_OF_MEMORY" or re.search(r"oom_kill=[1-9]", log_text):
        return "slurm memory"
    if re.search(r"exit=75\b", log_text):
        return "waiting for control (exit 75)"
    if state == "CANCELLED":
        return "cancelled"
    return "other"


def status_table(tail, latest, verified, logs) -> list[dict]:
    """One dict per tail row. `latest` from latest_tasks; `verified` row -> bool; `logs`
    task id -> log text (missing = '')."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = []
    for t in tail:
        r = int(t["row"])
        task, state = latest.get(r, ("", "NO_JOB"))
        ok = bool(verified.get(r))
        out.append({**{k: t.get(k, "") for k in TAIL_COLS}, "job": task, "state": state,
                    "verified": "yes" if ok else "no",
                    "failure_class": "" if ok else failure_class(state, logs.get(task, "")),
                    "checked_utc": now})
    return out


def is_done(table) -> bool:
    """Every tail row verified, or its latest task terminal (a row with no job is not done)."""
    return all(t["verified"] == "yes" or t["state"] in TERMINAL for t in table)


def live_jobs(table) -> list[str]:
    return sorted({t["job"].split("_")[0] for t in table
                   if t["verified"] == "no" and t["job"] and t["state"] not in TERMINAL})


def write_tsv(path, cols, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--build-list", required=True)
    i.add_argument("--rows", required=True)
    i.add_argument("--out-dir", required=True)
    s = sub.add_parser("status")
    s.add_argument("--build-list", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--measurements", default=OUT)
    s.add_argument("--work", default=WORK)
    s.add_argument("--logs", default=LOGS)
    a = ap.parse_args(argv)
    rows = {int(r["row"]): r for r in BL.read_tsv(a.build_list)}
    d = Path(a.out_dir)
    if a.cmd == "init":
        tail = []
        for n in BL.parse_spec(a.rows):
            r = rows[n]
            tail.append({"row": n, "experiment": r["experiment"], "eic_track": r["eic_track"],
                         "cell": (r["eic_track"] or "")[:3], "role": r["role"],
                         "assay": r["assay"], "priority": r["priority"]})
        d.mkdir(parents=True, exist_ok=True)
        write_tsv(d / "tail_rows.tsv", TAIL_COLS, tail)
        print(f"{len(tail)} tail rows -> {d / 'tail_rows.tsv'}")
        return 0
    tail = BL.read_tsv(d / "tail_rows.tsv")
    nums = [int(t["row"]) for t in tail]
    sa = subprocess.run(["sacct", "-u", os.environ.get("USER", ""), "-S", SINCE, "-X", "-n", "-P",
                         "--name", JOB_NAMES, "-o", "JobID,State"],
                        capture_output=True, text=True, check=True).stdout.splitlines()
    # sacct leaves out a job that has never been eligible (pending on a dependency): add squeue
    sq = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-r", "-n", JOB_NAMES,
                         "-o", "%i|%T"], capture_output=True, text=True, check=True)
    latest = latest_tasks(sa + sq.stdout.splitlines(), nums)
    verified = {n: BL.row_state(rows[n], a.measurements, a.work)[0] != "todo" for n in nums}
    logs = {}
    for n, (task, state) in latest.items():
        p = Path(a.logs) / f"build_{task}.out"
        if state in TERMINAL and p.is_file():
            logs[task] = p.read_text(errors="replace")
    table = status_table(tail, latest, verified, logs)
    write_tsv(d / "TAIL_STATUS.tsv", STATUS_COLS, table)
    n_ok = sum(t["verified"] == "yes" for t in table)
    print(f"tail: {len(table)} rows, {n_ok} verified -> {d / 'TAIL_STATUS.tsv'}")
    if is_done(table):
        bad = [t for t in table if t["verified"] == "no"]
        (d / "TAIL_DONE").write_text(
            f"{table[0]['checked_utc'] if table else ''}\t{n_ok} verified\t{len(bad)} not verified: "
            + ",".join(f"{t['row']}({t['failure_class'] or t['state']})" for t in bad) + "\n")
        print("TAIL_DONE written")
        return 0
    print("live " + ",".join(live_jobs(table)))
    return 3


if __name__ == "__main__":
    sys.exit(main())
