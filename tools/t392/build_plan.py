#!/usr/bin/env python3
"""Recount plan of task t392: which measurement each array task recounts, one plan file per source.

Build:
    build_plan.py --tables <eic_v1_2> --measurements <dir> [<dir> ...] --index <dir of onion-work__*.txt>
                  --testbed-root <dir> --gib-per-task 2.6 --out <dir>

Writes `plan_<source>.tsv` for each source (`plan_onion-work__000.tsv` .. `plan_onion-work__005.tsv`,
`plan_testbed.tsv`), columns `task, measurement_id, source, read_gib, n_tagaligns, lines_per_fragment`.
The read files of a measurement are the `pr1` and `pr2` lists of its `provenance.json::halves`.

Source of a measurement:
- a tar, when every read file, with the prefix `/scratch/mforooz/` cut (`--strip`), is a member in the
  tar index `<index>/<tar>.txt` (`tar -tv` lines; the member is the last field, the size the third);
- `testbed`, when every read file is `<testbed-root>/<parent folder>/<file name>`; its size is read there.
Any other case raises: a measurement whose files are in no source, or in two tars and in no single one.

Tasks are numbered from 1 (the SLURM array index) and packed inside one source: the number of tasks is
ceil(source GiB / --gib-per-task), and measurements go, largest first, to the task with the least GiB.

Members:
    build_plan.py members --plan <plan file> [--measurements <dir> ...] [--strip /scratch/mforooz/]
prints the tar members that the plan's measurements read, one per line, sorted, each once. Only the `pr1`
and `pr2` read files are printed. A testbed plan prints nothing (its files are not in a tar).

Standard library only, so it runs with any python3 on Fir or Nibi.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import sys
from pathlib import Path

GIB = 1024 ** 3
COLUMNS = ("task", "measurement_id", "source", "read_gib", "n_tagaligns", "lines_per_fragment")
TESTBED = "testbed"
HALVES = ("pr1", "pr2")
DEFAULT_MEASUREMENTS = ["/scratch/mforooz/onion/measurements"]
DEFAULT_STRIP = "/scratch/mforooz/"


def measurement_ids(tables: Path) -> list[str]:
    with open(tables / "measurements.tsv", newline="") as fh:
        return [r["measurement_id"] for r in csv.DictReader(fh, delimiter="\t")]


def provenance(mid: str, dirs: list[str]) -> dict:
    for d in dirs:
        p = Path(d) / mid / "provenance.json"
        if p.is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"{mid}: no provenance.json under {dirs}")


def read_files(prov: dict) -> list[str]:
    """The read files of the two halves, in order, each once."""
    halves = prov["halves"]
    out: list[str] = []
    for h in HALVES:
        for p in halves[h]["tagalign"]:
            if p not in out:
                out.append(p)
    return out


def member_of(path: str, strip: str) -> str:
    if not path.startswith(strip):
        raise ValueError(f"{path}: does not start with {strip}")
    return path[len(strip):]


def load_index(index_dir: Path) -> dict[str, dict[str, int]]:
    """{tar name: {member: bytes}} from `tar -tv` listings named `<tar>.txt`."""
    out: dict[str, dict[str, int]] = {}
    for f in sorted(index_dir.glob("onion-work__*.txt")):
        sizes: dict[str, int] = {}
        for line in f.read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) < 6 or not parts[0].startswith("-"):
                continue
            sizes[parts[5]] = int(parts[2])
        out[f.stem] = sizes
    if not out:
        raise FileNotFoundError(f"no onion-work__*.txt under {index_dir}")
    return out


def locate(files: list[str], index: dict[str, dict[str, int]], testbed_root: Path,
           strip: str) -> tuple[str, list[int]]:
    """(source, bytes per file) of one measurement's read files."""
    for tar, sizes in index.items():
        members = [f[len(strip):] if f.startswith(strip) else None for f in files]
        if all(m is not None and m in sizes for m in members):
            return tar, [sizes[m] for m in members]
    local = [testbed_root / Path(f).parent.name / Path(f).name for f in files]
    if all(p.is_file() for p in local):
        return TESTBED, [p.stat().st_size for p in local]
    raise LookupError(f"read files in no single source: {files}")


def pack(items: list[tuple[str, float]], gib_per_task: float) -> dict[str, int]:
    """{measurement: task}: ceil(total / gib_per_task) tasks, largest item first to the lightest task."""
    total = sum(g for _, g in items)
    n = max(1, math.ceil(total / gib_per_task))
    heap = [(0.0, t) for t in range(1, n + 1)]
    out: dict[str, int] = {}
    for mid, g in sorted(items, key=lambda x: (-x[1], x[0])):
        load, t = heapq.heappop(heap)
        out[mid] = t
        heapq.heappush(heap, (load + g, t))
    return out


def build(tables: Path, measurements: list[str], index_dir: Path, testbed_root: Path,
          gib_per_task: float, out: Path, strip: str = DEFAULT_STRIP) -> dict[str, list[dict]]:
    index = load_index(index_dir)
    rows: dict[str, list[dict]] = {s: [] for s in list(index) + [TESTBED]}
    for mid in measurement_ids(tables):
        prov = provenance(mid, measurements)
        files = read_files(prov)
        source, sizes = locate(files, index, testbed_root, strip)
        rows[source].append({"measurement_id": mid, "source": source,
                             "read_gib": sum(sizes) / GIB, "n_tagaligns": len(files),
                             "lines_per_fragment": int(prov.get("lines_per_fragment") or 1)})
    out.mkdir(parents=True, exist_ok=True)
    for source, rs in rows.items():
        tasks = pack([(r["measurement_id"], r["read_gib"]) for r in rs], gib_per_task)
        for r in rs:
            r["task"] = tasks[r["measurement_id"]]
        rs.sort(key=lambda r: (r["task"], r["measurement_id"]))
        with open(out / f"plan_{source}.tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(COLUMNS)
            for r in rs:
                w.writerow([r["task"], r["measurement_id"], r["source"], f"{r['read_gib']:.6f}",
                            r["n_tagaligns"], r["lines_per_fragment"]])
    return rows


def summary(rows: dict[str, list[dict]]) -> str:
    lines = ["source\tmeasurements\ttasks\tread_gib\tmax_task_gib"]
    tot_m = tot_t = 0
    tot_g = 0.0
    for source, rs in rows.items():
        loads: dict[int, float] = {}
        for r in rs:
            loads[r["task"]] = loads.get(r["task"], 0.0) + r["read_gib"]
        g = sum(loads.values())
        lines.append(f"{source}\t{len(rs)}\t{len(loads)}\t{g:.3f}\t{max(loads.values(), default=0):.3f}")
        tot_m, tot_t, tot_g = tot_m + len(rs), tot_t + len(loads), tot_g + g
    lines.append(f"total\t{tot_m}\t{tot_t}\t{tot_g:.3f}\t")
    return "\n".join(lines)


def members(plan: Path, measurements: list[str], strip: str = DEFAULT_STRIP) -> list[str]:
    with open(plan, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    out: set[str] = set()
    for r in rows:
        if r["source"] == TESTBED:
            continue
        for f in read_files(provenance(r["measurement_id"], measurements)):
            out.add(member_of(f, strip))
    return sorted(out)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "members":
        ap = argparse.ArgumentParser(prog="build_plan.py members")
        ap.add_argument("--plan", required=True, type=Path)
        ap.add_argument("--measurements", nargs="+", default=DEFAULT_MEASUREMENTS)
        ap.add_argument("--strip", default=DEFAULT_STRIP)
        a = ap.parse_args(argv[1:])
        for m in members(a.plan, a.measurements, a.strip):
            print(m)
        return 0
    ap = argparse.ArgumentParser(prog="build_plan.py")
    ap.add_argument("--tables", required=True, type=Path)
    ap.add_argument("--measurements", nargs="+", required=True)
    ap.add_argument("--index", required=True, type=Path)
    ap.add_argument("--testbed-root", required=True, type=Path)
    ap.add_argument("--gib-per-task", type=float, default=2.6)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--strip", default=DEFAULT_STRIP)
    a = ap.parse_args(argv)
    rows = build(a.tables, a.measurements, a.index, a.testbed_root, a.gib_per_task, a.out, a.strip)
    print(summary(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
