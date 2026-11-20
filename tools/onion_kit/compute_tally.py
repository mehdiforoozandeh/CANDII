#!/usr/bin/env python3
"""t152 chunk CT: compute and storage tally at each milestone (decision D3).

Four subcommands, run in this order:

  jobs       collect every Slurm job id written in a SUBMITTED.md / SUBMITTED.tsv under the results
             tree, or in a line of the plan's `## Log`, with its chunk and cluster -> jobs.tsv
  sacct      query sacct through `hpc run <cluster>` and write per chunk and cluster: jobs, CPU-h
             allocated (AllocCPUS x Elapsed), CPU-h used (TotalCPU), GPU-slice-h, failed -> tally.tsv
  storage    `du -sk` and `find | wc -l` on the ONION directories of pinned interface A, plus
             `diskusage_report`, on each cluster -> storage.tsv
  milestone  join the three files into <name>.md and print the notepad entry text

CPU-h allocated and CPU-h used are different numbers. Allocated is AllocCPUS x Elapsed (what the
fair-share is charged). Used is sacct TotalCPU (user + system CPU time of all steps). sacct only
fills TotalCPU when a job ends, and with `-X` it prints 00:00:00 for every job on Nibi (measured
2026-11-11), so this tool queries without `-X` and sums the step lines.

Exit 69 from `hpc` (tunnel down) stops the tool with exit 69 and a message.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CLUSTERS = ("nibi", "fir")
ONION_DIRS = (
    "/project/def-maxwl/mforooz/ONION",
    "/project/def-maxwl/mforooz/onion_kit",
    "/scratch/mforooz/onion",
    "/nearline/def-maxwl/mforooz/onion",
)
SACCT_FIELDS = "JobID,JobName,State,Elapsed,AllocCPUS,TotalCPU,AllocTRES"
SACCT_BATCH = 150
FAILED_STATES = {"FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
                 "PREEMPTED"}
LOG_STOP = {"PI", "PR", "NEW", "The", "A", "An"}
MILESTONES = ("G1", "G2", "G2b", "G3", "G3b", "G4", "final")

JOBS_COLS = ["job_id", "cluster", "chunk", "source", "cluster_source", "also_in"]
TALLY_COLS = ["chunk", "cluster", "jobs_listed", "allocations", "cpu_h_alloc", "cpu_h_used",
              "used_over_alloc", "gpu_slice_h", "gpu_full_h", "failed", "cancelled", "running",
              "pending", "not_found"]
STORAGE_COLS = ["cluster", "kind", "path", "gib", "files", "gib_quota", "files_quota",
                "space_pct", "files_pct"]


class TunnelDown(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------- jobs

# A job id is 7 or 8 digits, not part of a longer number, a word, a decimal or a 1,234 group.
# A trailing comma or full stop is punctuation and is allowed.
_PRE = r"(?<![0-9A-Za-z])(?<!\d[.,])"
_POST = r"(?![0-9A-Za-z]|[.,]\d)"
_ID = re.compile(_PRE + r"(\d{7,8})" + _POST)
_SHORT = re.compile(r"/(\d{1,6})" + _POST)
_RANGE = re.compile(_PRE + r"(\d{7,8})-(\d{7,8})" + _POST)
_CLUSTER = re.compile(r"\b(nibi|fir)\b", re.IGNORECASE)
_RANGE_MAX = 500
# An all-digit git sha where the Log writes `merged at <sha> (<sha>;` or `merged (<sha>;`.
_SHA_BEFORE = re.compile(r"(?:\bmerged(?:\s+at)?|\bcommit|\b[0-9a-f]{7,40})\s*\(\s*$|"
                         r"\b(?:merged(?:\s+at)?|commit)\s+$")


def ids_in_text(text: str) -> list[tuple[int, str]]:
    """Job ids in one line, as (position, id). Expands the shorthands the slips use:
    `3438391/3/5` -> 3438391, 3438393, 3438395; `3429537-3429543` -> every id in the range."""
    out: list[tuple[int, str]] = []
    seen_spans: list[tuple[int, int]] = []
    for m in _RANGE.finditer(text):
        a, b = int(m.group(1)), int(m.group(2))
        if len(m.group(1)) == len(m.group(2)) and 0 < b - a <= _RANGE_MAX:
            out += [(m.start(), str(i)) for i in range(a, b + 1)]
            seen_spans.append(m.span())
    for m in _ID.finditer(text):
        if any(s <= m.start() < e for s, e in seen_spans) or _SHA_BEFORE.search(text[:m.start()]):
            continue
        jid = m.group(1)
        out.append((m.start(), jid))
        pos = m.end()
        while True:
            s = _SHORT.match(text, pos)
            if not s or len(s.group(1)) >= len(jid):
                break
            out.append((m.start(), jid[: len(jid) - len(s.group(1))] + s.group(1)))
            pos = s.end()
    return sorted(out)


def _cluster_near(line: str, pos: int) -> str | None:
    before = [m for m in _CLUSTER.finditer(line) if m.start() < pos]
    if before:
        return before[-1].group(1).lower()
    after = [m for m in _CLUSTER.finditer(line) if m.start() > pos]
    return after[0].group(1).lower() if after else None


def _guess_cluster(jid: str) -> str:
    # Last resort only (flagged `guess`): on 2026-11-11 Fir ids had 7 digits, Nibi ids 8.
    return "nibi" if len(jid) >= 8 else "fir"


def _task_map(path: Path | None) -> dict[str, str]:
    """task id -> chunk id, from `| chunk | task | title |` rows."""
    out: dict[str, str] = {}
    if path is None or not path.is_file():
        return out
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and re.fullmatch(r"t\d+", cells[1]):
            out[cells[1]] = cells[0]
    return out


def _chunk_ids(plan_text: str) -> set[str]:
    return set(re.findall(r"^### ([A-Za-z0-9]+) ", plan_text, re.MULTILINE))


def _file_cluster(lines: list[str]) -> tuple[str | None, str]:
    for line in lines:
        if line.startswith("#"):
            m = _CLUSTER.search(line)
            if m:
                return m.group(1).lower(), "title"
            break
    counts: dict[str, int] = defaultdict(int)
    for m in _CLUSTER.finditer("\n".join(lines)):
        counts[m.group(1).lower()] += 1
    if counts:
        return max(sorted(counts), key=lambda c: counts[c]), "file"
    return None, ""


def _submitted_chunk(path: Path, results: Path, tmap: dict[str, str], known: set[str],
                     text: str) -> str:
    rel = path.relative_to(results).parts
    if rel and rel[0] in tmap:
        return tmap[rel[0]]
    m = re.search(r"\bchunk ([A-Za-z0-9]+)", text)
    if m and m.group(1) in known:
        return m.group(1)
    first = text.splitlines()[0] if text else ""
    for tok in re.findall(r"[A-Za-z0-9]+", first):
        if tok in known:
            return tok
    return rel[0] if rel else "unattributed"


def _from_submitted_tsv(path: Path, chunk: str, src: str) -> list[dict] | None:
    """A SUBMITTED.tsv with a job id column; None when it has none (then the text scan runs)."""
    rows = list(csv.DictReader(path.open(), delimiter="\t"))
    if not rows:
        return None
    cols = {c.lower().replace(" ", "_"): c for c in rows[0]}
    jcol = next((cols[k] for k in ("job_id", "jobid", "job") if k in cols), None)
    if jcol is None:
        return None
    ccol = cols.get("cluster")
    kcol = cols.get("chunk")
    out = []
    for i, r in enumerate(rows, start=2):
        for _, jid in ids_in_text(r.get(jcol) or ""):
            cl = (r.get(ccol) or "").strip().lower() if ccol else ""
            out.append(dict(job_id=jid, cluster=cl or None, cluster_source="column" if cl else "",
                            chunk=(r.get(kcol) or "").strip() if kcol else chunk,
                            source=f"{src}:{i}"))
    return out


def collect_jobs(results: Path, plan: Path, task_map: Path | None = None) -> list[dict]:
    plan_text = plan.read_text()
    known = _chunk_ids(plan_text)
    tmap = _task_map(task_map)
    found: list[dict] = []

    files = sorted(p for p in results.rglob("SUBMITTED*")
                   if p.is_file() and p.suffix in (".md", ".tsv"))
    for path in files:
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        src = str(path)
        chunk = _submitted_chunk(path, results, tmap, known, text)
        fcl, fsrc = _file_cluster(lines)
        recs = _from_submitted_tsv(path, chunk, src) if path.suffix == ".tsv" else None
        if recs is not None:
            for r in recs:
                if not r["chunk"]:
                    r["chunk"] = chunk
                if not r["cluster"]:
                    r["cluster"], r["cluster_source"] = (fcl, fsrc) if fcl else (
                        _guess_cluster(r["job_id"]), "guess")
            found += recs
            continue
        for n, line in enumerate(lines, start=1):
            for pos, jid in ids_in_text(line):
                cl, cs = _cluster_near(line, pos), "line"
                if cl is None:
                    cl, cs = (fcl, fsrc) if fcl else (_guess_cluster(jid), "guess")
                found.append(dict(job_id=jid, cluster=cl, cluster_source=cs, chunk=chunk,
                                  source=f"{src}:{n}"))

    in_log = False
    for n, line in enumerate(plan_text.splitlines(), start=1):
        if line.startswith("## "):
            in_log = line.strip() == "## Log"
            continue
        if not in_log or not line.startswith("- "):
            continue
        chunk = _log_chunk(line, known)
        for pos, jid in ids_in_text(line):
            cl, cs = _cluster_near(line, pos), "line"
            if cl is None:
                cl, cs = _guess_cluster(jid), "guess"
            found.append(dict(job_id=jid, cluster=cl, cluster_source=cs, chunk=chunk,
                              source=f"{plan}:{n}"))

    # One owner per (cluster, job id): the first mention (SUBMITTED files before Log lines).
    owner: dict[tuple[str, str], dict] = {}
    for r in found:
        key = (r["cluster"], r["job_id"])
        if key not in owner:
            owner[key] = dict(r, also_in=[])
        elif r["chunk"] != owner[key]["chunk"] and r["chunk"] not in owner[key]["also_in"]:
            owner[key]["also_in"].append(r["chunk"])
    out = []
    for r in owner.values():
        r["also_in"] = ",".join(r["also_in"])
        out.append(r)
    return out


def _log_chunk(line: str, known: set[str]) -> str:
    toks = [t.strip(":,;()") for t in line[2:].split()]
    if toks and re.fullmatch(r"\d{4}-\d{2}-\d{2}", toks[0]):
        toks = toks[1:]
    if toks and re.fullmatch(r"[A-Z][A-Za-z0-9]{0,4}", toks[0]) and toks[0] not in LOG_STOP:
        return toks[0]
    for t in toks[:8]:
        if t in known:
            return t
    return "unattributed"


def write_tsv(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_tsv(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ---------------------------------------------------------------------------------------- remote

def hpc_run(cluster: str, cmd: str) -> str:
    p = subprocess.run(["hpc", "run", cluster, cmd], capture_output=True, text=True)
    if p.returncode == 69:
        raise TunnelDown(cluster)
    if p.returncode != 0:
        raise RuntimeError(f"hpc run {cluster} exited {p.returncode}: {p.stderr.strip()[-500:]}")
    return p.stdout


# ---------------------------------------------------------------------------------------- sacct

def parse_duration(s: str) -> float:
    """Slurm [D-][HH:]MM:SS[.fff] (Elapsed, TotalCPU) -> hours."""
    s = s.strip()
    if not s or s in ("INVALID", "Unknown"):
        return 0.0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts
    return days * 24 + h + m / 60 + sec / 3600


def parse_tres(tres: str) -> tuple[int, int]:
    """AllocTRES -> (GPU slices, full GPUs). A typed gres like `1g.10gb` is a MIG slice."""
    kv = dict(x.split("=", 1) for x in tres.split(",") if "=" in x)
    slices = full = 0
    typed = {k: v for k, v in kv.items() if k.startswith("gres/gpu:")}
    if typed:
        for k, v in typed.items():
            n = int(re.match(r"\d+", v).group()) if re.match(r"\d+", v) else 0
            if re.search(r"\dg\.\d+gb", k):
                slices += n
            else:
                full += n
    elif "gres/gpu" in kv:
        full = int(re.match(r"\d+", kv["gres/gpu"]).group())
    return slices, full


def parse_sacct(text: str) -> dict[str, dict]:
    """sacct -n -P -o JobID,JobName,State,Elapsed,AllocCPUS,TotalCPU,AllocTRES (no -X) ->
    allocation id -> record. Lines that are not sacct rows (module banners) are skipped."""
    allocs: dict[str, dict] = {}
    steps: dict[str, float] = defaultdict(float)
    has_step: set[str] = set()
    for line in text.splitlines():
        f = line.split("|")
        if len(f) != 7 or not re.match(r"\d+", f[0]):
            continue
        jid, name, state, elapsed, ncpu, totcpu, tres = f
        if "." in jid:
            base = jid.split(".", 1)[0]
            steps[base] += parse_duration(totcpu)
            has_step.add(base)
            continue
        el = parse_duration(elapsed)
        sl, fu = parse_tres(tres)
        allocs[jid] = dict(job_id=jid, parent=re.match(r"\d+", jid).group(), name=name,
                           state=state.split()[0] if state else "", elapsed_h=el,
                           cpus=int(ncpu or 0), cpu_h_alloc=int(ncpu or 0) * el,
                           cpu_h_used=parse_duration(totcpu), gpu_slice_h=sl * el,
                           gpu_full_h=fu * el)
    for jid, rec in allocs.items():
        if jid in has_step:
            rec["cpu_h_used"] = steps[jid]
    return allocs


def sacct_text(cluster: str, ids: list[str], runner=hpc_run) -> str:
    out = []
    for i in range(0, len(ids), SACCT_BATCH):
        chunk = ",".join(ids[i:i + SACCT_BATCH])
        out.append(runner(cluster, f"sacct -j {chunk} -n -P -o {SACCT_FIELDS}"))
    return "\n".join(out)


def tally(jobs: list[dict], sacct_by_cluster: dict[str, str]) -> list[dict]:
    rows: dict[tuple[str, str], dict] = {}

    def row(chunk, cluster):
        k = (chunk, cluster)
        if k not in rows:
            rows[k] = {c: 0 for c in TALLY_COLS}
            rows[k].update(chunk=chunk, cluster=cluster)
        return rows[k]

    parsed = {c: parse_sacct(t) for c, t in sacct_by_cluster.items()}
    for j in jobs:
        cl, jid = j["cluster"], j["job_id"]
        r = row(j["chunk"], cl)
        r["jobs_listed"] += 1
        recs = [a for a in parsed.get(cl, {}).values() if a["parent"] == jid]
        if not recs:
            r["not_found"] += 1
            continue
        for a in recs:
            r["allocations"] += 1
            for k in ("cpu_h_alloc", "cpu_h_used", "gpu_slice_h", "gpu_full_h"):
                r[k] += a[k]
            st = a["state"]
            if st in FAILED_STATES:
                r["failed"] += 1
            elif st.startswith("CANCELLED"):
                r["cancelled"] += 1
            elif st == "RUNNING":
                r["running"] += 1
            elif st == "PENDING":
                r["pending"] += 1
    out = sorted(rows.values(), key=lambda r: (r["cluster"], r["chunk"]))
    for cl in sorted({r["cluster"] for r in out}):
        tot = {c: 0 for c in TALLY_COLS}
        tot.update(chunk="TOTAL", cluster=cl)
        for r in out:
            if r["cluster"] == cl:
                for c in TALLY_COLS[2:]:
                    tot[c] += r[c]
        out.append(tot)
    for r in out:
        r["used_over_alloc"] = (f"{r['cpu_h_used'] / r['cpu_h_alloc']:.3f}"
                                if r["cpu_h_alloc"] else "")
        for k in ("cpu_h_alloc", "cpu_h_used", "gpu_slice_h", "gpu_full_h"):
            r[k] = f"{r[k]:.2f}"
    return out


# -------------------------------------------------------------------------------------- storage

_UNIT = {"B": 1 / 2**30, "KiB": 1 / 2**20, "MiB": 1 / 2**10, "GiB": 1.0, "TiB": 2**10,
         "PiB": 2**20}
_COUNT = {"": 1, "K": 10**3, "M": 10**6, "G": 10**9}
_SPACE = re.compile(r"([\d.]+)\s*([KMGTP]?i?B)\s*/\s*([\d.]+)\s*([KMGTP]?i?B)")
_FILES = re.compile(r"([\d.]+)\s*([KMG]?)\s*/\s*([\d.]+)\s*([KMG]?)")


def parse_diskusage(text: str, cluster: str) -> list[dict]:
    """diskusage_report rows -> storage rows (kind = quota)."""
    out = []
    for line in text.splitlines():
        line = line.replace("->", " ")
        m = re.match(r"\s*(/\S+)\s+\(([^)]*)\)(.*)$", line)
        if not m:
            continue
        rest = m.group(3)
        s = _SPACE.search(rest)
        if not s:
            continue
        f = _FILES.search(rest[s.end():])
        used = float(s.group(1)) * _UNIT[s.group(2)]
        quota = float(s.group(3)) * _UNIT[s.group(4)]
        r = dict(cluster=cluster, kind="quota", path=f"{m.group(1)} ({m.group(2)})",
                 gib=f"{used:.1f}", gib_quota=f"{quota:.1f}",
                 space_pct=f"{100 * used / quota:.1f}" if quota else "",
                 files="", files_quota="", files_pct="")
        if f:
            fu = float(f.group(1)) * _COUNT[f.group(2)]
            fq = float(f.group(3)) * _COUNT[f.group(4)]
            r.update(files=str(int(fu)), files_quota=str(int(fq)),
                     files_pct=f"{100 * fu / fq:.1f}" if fq else "")
        out.append(r)
    return out


def storage_cmd(dirs=ONION_DIRS) -> str:
    parts = [f'if [ -d {d} ]; then echo "ONIONDIR|{d}|$(du -sk {d} 2>/dev/null | cut -f1)|'
             f'$(find {d} 2>/dev/null | wc -l)"; else echo "ONIONDIR|{d}|NA|NA"; fi'
             for d in dirs]
    return "; ".join(parts) + "; diskusage_report"


def parse_storage(text: str, cluster: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if line.startswith("ONIONDIR|"):
            _, d, kib, nfiles = line.split("|")
            present = kib.strip() not in ("", "NA")
            out.append(dict(cluster=cluster, kind="dir", path=d,
                            gib=f"{int(kib) / 2**20:.2f}" if present else "absent",
                            files=nfiles.strip() if present else "", gib_quota="",
                            files_quota="", space_pct="", files_pct=""))
    return out + parse_diskusage(text, cluster)


# ------------------------------------------------------------------------------------ milestone

def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def milestone(name: str, jobs: list[dict], tally_rows: list[dict], storage: list[dict],
              sources: dict[str, str], when: str | None = None) -> tuple[str, str]:
    when = when or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    date = when[:10]
    chunks = [r for r in tally_rows if r["chunk"] != "TOTAL"]
    totals = [r for r in tally_rows if r["chunk"] == "TOTAL"]
    md = [f"# Compute and storage at milestone {name}", "",
          f"Generated {when} by `tools/onion_kit/compute_tally.py`. Inputs: "
          + ", ".join(f"`{k}` = `{v}`" for k, v in sources.items()) + ".", "",
          "CPU-h allocated is AllocCPUS x Elapsed. CPU-h used is sacct TotalCPU (all steps). "
          "GPU-h is allocated GPUs x Elapsed. sacct fills TotalCPU only when a job ends, so a "
          "running job adds allocated CPU-h so far and 0 used CPU-h.", "",
          "## Compute per cluster", "",
          "| cluster | job ids | allocations | CPU-h allocated | CPU-h used | used / allocated "
          "| GPU-slice-h | full-GPU-h | failed | cancelled | running | pending | ids not found |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in totals:
        md.append(f"| {r['cluster']} | {r['jobs_listed']} | {r['allocations']} | "
                  f"{r['cpu_h_alloc']} | {r['cpu_h_used']} | {r['used_over_alloc']} | "
                  f"{r['gpu_slice_h']} | {r['gpu_full_h']} | {r['failed']} | {r['cancelled']} | "
                  f"{r['running']} | {r['pending']} | {r['not_found']} |")
    md += ["", "## Compute per chunk", "",
           "| chunk | cluster | job ids | allocations | CPU-h allocated | CPU-h used | "
           "GPU-slice-h | full-GPU-h | failed | running + pending | ids not found |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in chunks:
        md.append(f"| {r['chunk']} | {r['cluster']} | {r['jobs_listed']} | {r['allocations']} | "
                  f"{r['cpu_h_alloc']} | {r['cpu_h_used']} | {r['gpu_slice_h']} | "
                  f"{r['gpu_full_h']} | {r['failed']} | "
                  f"{int(_f(r['running'])) + int(_f(r['pending']))} | {r['not_found']} |")
    dirs = [s for s in storage if s["kind"] == "dir"]
    quotas = [s for s in storage if s["kind"] == "quota"]
    md += ["", "## Storage: ONION directories (pinned interface A)", "",
           "| cluster | directory | GiB | files and directories |", "|---|---|---|---|"]
    md += [f"| {s['cluster']} | `{s['path']}` | {s['gib']} | {s['files']} |" for s in dirs]
    md += ["", "## Storage: quotas (`diskusage_report`)", "",
           "| cluster | space | GiB used | GiB quota | % | files used | files quota | % |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {s['cluster']} | {s['path']} | {s['gib']} | {s['gib_quota']} | {s['space_pct']} "
           f"| {s['files']} | {s['files_quota']} | {s['files_pct']} |" for s in quotas]
    guesses = sum(1 for j in jobs if j.get("cluster_source") == "guess")
    shared = sum(1 for j in jobs if j.get("also_in"))
    md += ["", "## Caveats", "",
           f"- {len(jobs)} job ids were collected. {guesses} had no cluster name in their line or "
           "file; their cluster was guessed from the id length (Fir 7 digits, Nibi 8).",
           f"- {shared} job ids were cited by more than one chunk. Each counts once, for the "
           "first chunk that cites it (SUBMITTED files before Log lines).",
           "- Jobs that no SUBMITTED file or Log line cites are not counted. An id that sacct "
           "does not know (for example a count that looks like an id) is in `ids not found`.",
           "- `diskusage_report` shows the soft quota where a space has one. Nibi `/scratch` "
           "had a soft quota of 1024 GiB and a hard limit of 20480 GiB (`quota -s`, 2026-11-10)."]
    md_text = "\n".join(md) + "\n"

    def tot(k, cl=None):
        return sum(_f(r[k]) for r in totals if cl is None or r["cluster"] == cl)

    per_cl = "; ".join(f"{r['cluster']} {_f(r['cpu_h_alloc']):.1f} allocated, "
                       f"{_f(r['cpu_h_used']):.1f} used" for r in totals)
    big = [s for s in quotas if _f(s["space_pct"]) >= 80 or _f(s["files_pct"]) >= 80]
    onion_gib = sum(_f(s["gib"]) for s in dirs)
    onion_files = sum(int(_f(s["files"])) for s in dirs)
    entry = [f"## {date} Milestone {name}: compute and storage", "",
             f"Jobs: {int(tot('jobs_listed'))} job ids from {len({r['chunk'] for r in chunks})} "
             f"chunks, {int(tot('allocations'))} allocations "
             f"({int(tot('failed'))} failed, {int(tot('running')) + int(tot('pending'))} still "
             f"running or pending, {int(tot('not_found'))} ids not found by sacct).",
             f"CPU-h: {tot('cpu_h_alloc'):.1f} allocated (AllocCPUS x Elapsed), "
             f"{tot('cpu_h_used'):.1f} used (TotalCPU) ({per_cl}).",
             f"GPU-h: {tot('gpu_slice_h'):.1f} GPU-slice-h, {tot('gpu_full_h'):.1f} full-GPU-h.",
             f"Storage: ONION directories hold {onion_gib:.1f} GiB in {onion_files} files and "
             f"directories over {len({s['cluster'] for s in dirs})} clusters."]
    if big:
        entry.append("Quotas at or over 80%: " + "; ".join(
            f"{s['cluster']} {s['path']} space {s['space_pct']}%, files {s['files_pct']}%"
            for s in big) + ".")
    entry.append(f"Tables: `{sources.get('out', '')}`.")
    return md_text, "\n".join(entry) + "\n"


# ------------------------------------------------------------------------------------------ CLI

def main(argv=None, runner=hpc_run) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("jobs")
    a.add_argument("--results", required=True, type=Path)
    a.add_argument("--orchestrate", required=True, type=Path, help="the plan file with ## Log")
    a.add_argument("--task-map", type=Path,
                   default=Path("cruxvault/INSTRUCTIONS/peeling_the_onion/onion task map.md"))
    a.add_argument("--out", required=True, type=Path)
    b = sub.add_parser("sacct")
    b.add_argument("--jobs", required=True, type=Path)
    b.add_argument("--out", required=True, type=Path)
    c = sub.add_parser("storage")
    c.add_argument("--clusters", default=",".join(CLUSTERS))
    c.add_argument("--out", required=True, type=Path)
    d = sub.add_parser("milestone")
    d.add_argument("--name", required=True, choices=MILESTONES)
    d.add_argument("--jobs", type=Path, help="default: jobs.tsv beside --out")
    d.add_argument("--tally", type=Path, help="default: tally.tsv beside --out")
    d.add_argument("--storage", type=Path, help="default: storage.tsv beside --out")
    d.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    try:
        if args.cmd == "jobs":
            rows = collect_jobs(args.results, args.orchestrate, args.task_map)
            write_tsv(args.out, JOBS_COLS, rows)
            n = defaultdict(int)
            for r in rows:
                n[r["cluster_source"]] += 1
            print(f"jobs: {len(rows)} job ids -> {args.out} (cluster from "
                  + ", ".join(f"{k} {v}" for k, v in sorted(n.items())) + ")")
        elif args.cmd == "sacct":
            jobs = read_tsv(args.jobs)
            texts = {}
            for cl in sorted({j["cluster"] for j in jobs}):
                ids = sorted({j["job_id"] for j in jobs if j["cluster"] == cl})
                texts[cl] = sacct_text(cl, ids, runner)
            rows = tally(jobs, texts)
            write_tsv(args.out, TALLY_COLS, rows)
            for r in rows:
                if r["chunk"] == "TOTAL":
                    print(f"sacct {r['cluster']}: {r['allocations']} allocations, CPU-h "
                          f"{r['cpu_h_alloc']} allocated / {r['cpu_h_used']} used, GPU-slice-h "
                          f"{r['gpu_slice_h']}, failed {r['failed']}, not found {r['not_found']}")
        elif args.cmd == "storage":
            rows = []
            for cl in [x for x in args.clusters.split(",") if x]:
                rows += parse_storage(runner(cl, storage_cmd()), cl)
            write_tsv(args.out, STORAGE_COLS, rows)
            print(f"storage: {len(rows)} rows -> {args.out}")
        elif args.cmd == "milestone":
            base = args.out.parent
            paths = dict(jobs=args.jobs or base / "jobs.tsv",
                         tally=args.tally or base / "tally.tsv",
                         storage=args.storage or base / "storage.tsv")
            md, entry = milestone(args.name, read_tsv(paths["jobs"]), read_tsv(paths["tally"]),
                                  read_tsv(paths["storage"]),
                                  {**{k: str(v) for k, v in paths.items()}, "out": str(args.out)})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(md)
            print(entry, end="")
    except TunnelDown as e:
        print(f"compute_tally: the tunnel to {e} is down (hpc exit 69). Run: hpc up {e}; "
              "then re-run this command.", file=sys.stderr)
        return 69
    return 0


if __name__ == "__main__":
    sys.exit(main())
