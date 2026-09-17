"""t112 FASTQ arms — one Caper run per arm, from patched input JSON to harvested keepers.

Arms whose knob sits before alignment (pe, dedup, crop, mapq) re-run the whole ENCODE pipeline from
FASTQ. Each run is one pid from `arms.py rows --route fastq`, and it owns everything it touches:

    $CF/fastqarms/inputs/<pid>.json        the base input JSON with exactly one knob changed
    $CF/fastqarms/loc/<pid>/               Caper loc dir: hard links to the staged FASTQs + refs
    $CF/fastqarms/cromwell/<pid>/          Cromwell out dir (attempt n>1: <pid>__attempt<n>/)
    $CF/fastqarms/state/<pid>.json         staged | submitted | succeeded | failed | harvested
    $CF/fastqarms/harvest/<pid>/           copies of the keepers + HARVEST.tsv

Why not `$EIC/scripts/run_batches.py`: its results dir and BAM keepers are hard-wired to /project
(~330 GiB free), it deletes trees after harvest, and it finds a workflow by title *suffix*, which
collides across arms of one track. What is kept from it:

- **every `caper hpc submit` argument is one token.** Caper rewrites its argv into the leader
  script with a bare `" ".join()`, so a value with a space word-splits and the leader dies on
  `ambiguous option: --mem`. Leader resources stay in `~/.caper/default.conf`.
- **a private loc dir per run.** A shared one makes leaders race for autouri's file locks.
- **a scheduler we cannot reach is not evidence a job died** (`job_alive`).

The loc dir is filled with hard links, not copies: `$CF/fastq_cache/` holds every FASTQ at autouri's
own path `<md5 of the URL string>/<basename>` (`prestage_fastqs.py`), md5-verified and mode 0444, so
autouri logs `skipped due to name_size_match` and transfers nothing. The 0444 mode is the interlock:
a link to a writable file would let one run overwrite the bytes every other run reads. `stage`
refuses a source with any file not 0444.

Keepers are harvested from the producing call's own `execution/` tree (the final Done attempt named
in Cromwell's metadata), never from a downstream call's `inputs/`, and each basename is cross-checked
against that call's recorded output. This file never deletes anything: `cleanup-candidates` only
lists trees whose keepers re-verify.

Pipelines: histone ChIP (`chip-seq-pipeline2` v2.2.2) only. The DNase (atac) branch is added to
`PIPELINES`, `base_json_path` and `harvest_roles` by chunk C13; until then an atac row is refused.

Deliberately stdlib only: it runs on the Nibi login node.

    python3 tools/t112/fastq_arms.py {stage|submit|poll|harvest|cleanup-candidates} \\
        --rows rows.tsv --cf /scratch/mforooz/t112_cf --eic $EIC [--pids p1,p2]
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

EIC = "/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline"
CF = "/scratch/mforooz/t112_cf"
#: the genome bundle every EIC run localized, at autouri paths, mode 0444
#: (`$EIC/scripts/stage_reference_cache.py`).
REFCACHE = "/scratch/mforooz/EIC_REPRO/003/refcache"

HEADER = ("pid", "biosample", "track", "cell", "assay", "arm", "level", "route", "knob",
          "knob_value", "pipeline")

STATUSES = ("staged", "submitted", "succeeded", "failed", "harvested")

#: per `pipeline` column value. C13 adds "atac".
PIPELINES = {
    "chip": {
        "prefix": "chip",
        "wdl": "chip-seq-pipeline2/chip.wdl",
        "sif": "sif/chip-seq-pipeline_v2.2.2.sif",
    },
}

#: keys the pe arm's base JSON (`<track>.bwa.pe.json`) differs from the se one in, besides
#: description. Read off C19M16 on Nibi 2026-09-17.
PE_KEYS = ("paired_end", "ctl_paired_end", "fastqs_rep1_R1", "fastqs_rep1_R2",
           "ctl_fastqs_rep1_R1", "ctl_fastqs_rep1_R2")

HARVEST_COLUMNS = ("role", "src", "dest", "bytes", "md5")


def log(msg):
    print("%s  %s" % (time.strftime("%F %T"), msg), flush=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sh(argv, timeout=900):
    """Run argv (no shell), return (rc, stdout+stderr). Tests replace this."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def md5_file(path, bufsize=1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


# --- rows --------------------------------------------------------------------------------------

def read_rows(tsv) -> list[dict]:
    """`arms.py rows` TSV → dicts, knob_value parsed from its JSON text."""
    lines = Path(tsv).read_text().rstrip("\n").split("\n")
    header = tuple(lines[0].split("\t"))
    if header != HEADER:
        raise SystemExit(f"{tsv}: header {header} is not the arms.py header {HEADER}")
    rows = []
    for ln in lines[1:]:
        r = dict(zip(HEADER, ln.split("\t")))
        r["knob_value"] = json.loads(r["knob_value"])
        rows.append(r)
    return rows


def select_rows(rows, pids=None) -> list[dict]:
    bad = [r["pid"] for r in rows if r["route"] != "fastq"]
    if bad:
        raise SystemExit(f"not FASTQ-route rows: {bad[:5]} — use `arms.py rows --route fastq`")
    if pids is None:
        return rows
    known = {r["pid"] for r in rows}
    unknown = [p for p in pids if p not in known]
    if unknown:
        raise SystemExit(f"pids not in the rows TSV: {unknown}")
    return [r for r in rows if r["pid"] in pids]


def pipeline_spec(row) -> dict:
    spec = PIPELINES.get(row["pipeline"])
    if spec is None:
        raise SystemExit(f"{row['pid']}: pipeline {row['pipeline']!r} has no FASTQ branch here yet "
                         "(the atac/DNase branch is added by C13)")
    return spec


# --- input JSON --------------------------------------------------------------------------------

def base_json_path(row, eic_dir, tag) -> Path:
    pipeline_spec(row)
    return Path(eic_dir) / "inputs_bwa" / f"{row['track']}.bwa.{tag}.json"


def diff_keys(base: dict, new: dict) -> set:
    """Keys present in only one of the two, or whose values differ."""
    return {k for k in set(base) | set(new) if k not in base or k not in new or base[k] != new[k]}


def build_json(row, eic_dir) -> dict:
    """The arm's input JSON: its base JSON with one knob set, plus a unique title and description.

    Every arm starts from `<track>.bwa.se.json`. The `pe` arm takes only the run-type keys
    (`PE_KEYS`) from `<track>.bwa.pe.json`, not the whole file: on Nibi 2026-09-17 C07M29's se JSON
    carries `align_cpu 12, filter_cpu 8` (`apply_axis4.py`, added after the pe JSON was written) and
    its pe JSON does not, so starting from the pe file would move two more keys than the arm names.
    Key order is kept, so a new key lands at the end. Raises if the result differs from the se base
    in any key other than the ones the arm is allowed to move.
    """
    spec = pipeline_spec(row)
    pfx = spec["prefix"]
    se = json.loads(base_json_path(row, eic_dir, "se").read_text())
    new = dict(se)
    if row["arm"] == "pe":
        pe = json.loads(base_json_path(row, eic_dir, "pe").read_text())
        missing = [k for k in PE_KEYS if f"{pfx}.{k}" not in pe]
        if missing:
            raise SystemExit(f"{row['pid']}: pe base JSON lacks {missing}")
        for k in PE_KEYS:
            new[f"{pfx}.{k}"] = pe[f"{pfx}.{k}"]
    if not row["knob"].startswith(pfx + "."):
        raise SystemExit(f"{row['pid']}: knob {row['knob']!r} is not a {pfx}.* input")
    new[row["knob"]] = row["knob_value"]
    new[f"{pfx}.title"] = f"CF {row['pid']}"
    new[f"{pfx}.description"] = f"t112 counterfactual arm {row['pid']}"

    moved = {row["knob"], f"{pfx}.title", f"{pfx}.description"}
    if row["arm"] == "pe":
        moved |= {f"{pfx}.{k}" for k in PE_KEYS}
    got = diff_keys(se, new)
    if got != moved:
        raise SystemExit(f"{row['pid']}: input JSON moves {sorted(got)}, expected {sorted(moved)}")
    return new


def json_text(d: dict) -> str:
    return json.dumps(d, indent=2) + "\n"


# --- paths and state ---------------------------------------------------------------------------

def paths(cf, pid, attempt=1) -> dict:
    fa = Path(cf) / "fastqarms"
    return {
        "input": fa / "inputs" / f"{pid}.json",
        "loc": fa / "loc" / pid,
        "cromwell": fa / "cromwell" / (pid if attempt <= 1 else f"{pid}__attempt{attempt}"),
        "state": fa / "state" / f"{pid}.json",
        "harvest": fa / "harvest" / pid,
        "logs": fa / "logs",
    }


def read_state(cf, pid):
    p = paths(cf, pid)["state"]
    return json.loads(p.read_text()) if p.exists() else None


def write_state(cf, pid, **kw):
    st = read_state(cf, pid) or {"pid": pid, "status": None, "leader_job_id": None, "attempts": 0,
                                 "workflow_id": None, "metadata_json": None}
    st.update(kw)
    if st["status"] not in STATUSES:
        raise ValueError(f"{pid}: bad status {st['status']!r}")
    st["updated_utc"] = utc_now()
    write_atomic(paths(cf, pid)["state"], json.dumps(st, indent=2, sort_keys=True) + "\n")
    return st


# --- stage -------------------------------------------------------------------------------------

def _files(root: Path):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            yield Path(dirpath) / f


def check_readonly(src: Path):
    if not src.is_dir():
        raise SystemExit(f"stage: source {src} does not exist")
    bad = [p for p in _files(src) if stat.S_IMODE(p.lstat().st_mode) != 0o444]
    if bad:
        raise SystemExit(f"stage REFUSING: {len(bad)} file(s) under {src} are not mode 0444, e.g. "
                         f"{bad[0]} ({oct(stat.S_IMODE(bad[0].lstat().st_mode))}). A hard link to a "
                         "writable file lets one run overwrite the bytes every run reads.")


def link_tree(src: Path, dst: Path) -> int:
    """`cp -al src/. dst/`, resumable: links what is missing, refuses a same-name different file."""
    n = 0
    for f in _files(src):
        d = dst / f.relative_to(src)
        if d.exists():
            if not os.path.samefile(f, d):
                raise SystemExit(f"stage REFUSING: {d} exists and is not a hard link to {f}")
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        os.link(f, d)
        n += 1
    return n


def stage(row, cf, eic, refcache=REFCACHE):
    pid = row["pid"]
    st = read_state(cf, pid)
    if st and st["status"] != "staged":
        log(f"{pid}: skip stage, status {st['status']}")
        return st
    p = paths(cf, pid)
    sources = [Path(cf) / "fastq_cache", Path(refcache)]
    for src in sources:
        check_readonly(src)
        p["loc"].parent.mkdir(parents=True, exist_ok=True)
        if os.stat(src).st_dev != os.stat(p["loc"].parent).st_dev:
            raise SystemExit(f"stage: {src} and {p['loc']} are on different filesystems")

    text = json_text(build_json(row, eic))
    if p["input"].exists() and p["input"].read_text() != text:
        raise SystemExit(f"stage REFUSING: {p['input']} exists with different content")
    if not p["input"].exists():
        write_atomic(p["input"], text)

    p["loc"].mkdir(parents=True, exist_ok=True)
    n = sum(link_tree(src, p["loc"]) for src in sources)
    log(f"{pid}: staged ({n} new links in {p['loc']})")
    return write_state(cf, pid, status="staged", input_json=str(p["input"]), loc_dir=str(p["loc"]))


# --- submit ------------------------------------------------------------------------------------

def submit_command(row, cf, eic, attempt=1) -> list[str]:
    """argv for `caper hpc submit`. Every caper argument is a single token (see module docstring)."""
    spec = pipeline_spec(row)
    p = paths(cf, row["pid"], attempt)
    inner = " ".join([
        "source", shlex.quote(f"{eic}/runner/env.sh"), ">/dev/null", "2>&1", "&&",
        "cd", shlex.quote(str(p["logs"])), "&&",
        "caper", "hpc", "submit", shlex.quote(f"{eic}/{spec['wdl']}"),
        "-i", shlex.quote(str(p["input"])),
        "--leader-job-name", shlex.quote(f"t112_{row['pid']}"),
        "--local-loc-dir", shlex.quote(str(p["loc"])),
        "--local-out-dir", shlex.quote(str(p["cromwell"])),
        "--singularity", shlex.quote(f"{eic}/{spec['sif']}"),
    ])
    return ["bash", "-lc", inner]


def submit(row, cf, eic, retry=False, max_attempts=2):
    pid = row["pid"]
    st = read_state(cf, pid)
    if st is None:
        raise SystemExit(f"{pid}: not staged")
    if st["status"] == "failed" and not retry:
        log(f"{pid}: skip submit, status failed (pass --retry to resubmit)")
        return st
    if st["status"] not in ("staged", "failed"):
        log(f"{pid}: skip submit, status {st['status']}")
        return st
    attempt = st["attempts"] + 1
    if attempt > max_attempts:
        raise SystemExit(f"{pid}: attempt {attempt} would exceed --max-attempts {max_attempts}")
    spec = pipeline_spec(row)
    p = paths(cf, pid, attempt)
    for need in (p["input"], p["loc"], Path(eic) / spec["wdl"], Path(eic) / spec["sif"]):
        if not need.exists():
            raise SystemExit(f"{pid}: missing {need}")
    if p["cromwell"].exists() and any(p["cromwell"].iterdir()):
        raise SystemExit(f"{pid}: out dir {p['cromwell']} is not empty; refusing to reuse it")
    locks = sorted(glob.glob(str(p["loc"] / "**" / "*.lock"), recursive=True))
    if locks:
        raise SystemExit(f"{pid}: {len(locks)} autouri .lock file(s) in {p['loc']}, e.g. {locks[0]}. "
                         "A dead leader leaves these and the retry dies on them. Nothing is deleted "
                         "here: confirm no job holds them, remove them by hand, then resubmit.")
    p["cromwell"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)

    argv = submit_command(row, cf, eic, attempt)
    rc, out = sh(argv)
    write_atomic(p["logs"] / f"{pid}.submit{attempt}.log", f"$ {shlex.join(argv)}\nrc={rc}\n{out}")
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        raise SystemExit(f"{pid}: submit FAILED rc={rc}: {out.strip()[-800:]}")
    log(f"{pid}: submitted leader {m.group(1)} (attempt {attempt}) → {p['cromwell']}")
    return write_state(cf, pid, status="submitted", leader_job_id=m.group(1), attempts=attempt,
                       cromwell_dir=str(p["cromwell"]), workflow_id=None, metadata_json=None,
                       submitted_utc=utc_now())


# --- poll --------------------------------------------------------------------------------------

def job_alive(job_id) -> bool:
    """True if the leader is queued or running. Fail safe, as `run_batches.py::job_alive`.

    "Invalid job id specified" is SLURM saying the job left the queue: dead. Any other squeue
    failure says nothing about the job: assume alive. A false alive costs one poll; a false dead
    marks a live workflow failed.
    """
    rc, out = sh(["squeue", "-h", "-j", str(job_id), "-o", "%T"], timeout=120)
    txt = (out or "").strip()
    if rc == 0:
        return bool(txt)
    if "Invalid job id" in txt:
        return False
    log(f"squeue FAILED for job {job_id} (rc={rc}); assuming ALIVE. Output: {txt[:200]}")
    return True


def find_metadata(cromwell_dir):
    metas = sorted(glob.glob(os.path.join(cromwell_dir, "*", "*", "metadata.json")))
    if len(metas) > 1:
        raise SystemExit(f"{cromwell_dir}: {len(metas)} metadata.json files, expected one: {metas}")
    return metas[0] if metas else None


def poll(row, cf):
    pid = row["pid"]
    st = read_state(cf, pid)
    if st is None or st["status"] != "submitted":
        log(f"{pid}: skip poll, status {st and st['status']}")
        return st
    if job_alive(st["leader_job_id"]):
        log(f"{pid}: leader {st['leader_job_id']} alive")
        return st
    meta = find_metadata(st["cromwell_dir"])
    if meta is None:
        log(f"{pid}: leader {st['leader_job_id']} gone, no metadata.json → failed")
        return write_state(cf, pid, status="failed", note="leader gone, no metadata.json")
    d = json.loads(Path(meta).read_text())
    status = "succeeded" if d.get("status") == "Succeeded" else "failed"
    log(f"{pid}: workflow {d.get('id')} {d.get('status')} → {status}")
    return write_state(cf, pid, status=status, workflow_id=d.get("id"), metadata_json=meta,
                       note=f"workflow status {d.get('status')}")


# --- harvest -----------------------------------------------------------------------------------

def harvest_roles(row) -> list[tuple]:
    """(role, call, glob pattern, metadata output key) for one arm."""
    pipeline_spec(row)
    # with chip.no_dup_removal the filter task returns its .filt.bam as `nodup_bam`
    # (encode_task_filter.py: `if args.no_dup_removal: nodup_bam = filt_bam`)
    bam = "*.filt.bam" if row["arm"] == "dedup" else "*.nodup*.bam"
    roles = [
        ("treat_bam", "filter", bam, "nodup_bam"),
        ("treat_bai", "filter", bam + ".bai", "nodup_bai"),
        ("ctl_bam", "filter_ctl", bam, "nodup_bam"),
        ("ctl_bai", "filter_ctl", bam + ".bai", "nodup_bai"),
        ("treat_ta", "bam2ta", "*.tagAlign.gz", "ta"),
        ("ctl_ta", "bam2ta_ctl", "*.tagAlign.gz", "ta"),
        ("pval_bigwig", "macs2_signal_track", "*.pval.signal.bigwig", "pval_bw"),
        ("xcor_qc", "xcor", "*.cc.qc", "score"),
        ("qc_json", "qc_report", "qc.json", "qc_json"),
    ]
    if row["arm"] == "mapq":
        roles += [("unfiltered_bam", "align", "*.merged.srt.bam", "bam"),
                  ("unfiltered_ctl_bam", "align_ctl", "*.merged.srt.bam", "bam")]
    return roles


def final_call(meta: dict, name: str) -> dict:
    """The one Done attempt of a call. A single-replicate run has exactly one shard per call."""
    entries = [c for c in meta.get("calls", {}).get(name, []) if c.get("executionStatus") == "Done"]
    by_shard = {}
    for c in entries:
        s = c.get("shardIndex")
        if s not in by_shard or c.get("attempt", 1) > by_shard[s].get("attempt", 1):
            by_shard[s] = c
    if len(by_shard) != 1:
        raise SystemExit(f"call {name}: {len(by_shard)} Done shards in metadata, expected 1")
    return next(iter(by_shard.values()))


def find_one(call: dict, pattern: str, role: str) -> Path:
    """The single file matching `pattern` in the call's own execution/ (hard links collapse)."""
    root = Path(call["callRoot"]) / "execution"
    hits = {}
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d != "inputs"]
        for f in files:
            if fnmatch.fnmatch(f, pattern):
                fp = Path(dirpath) / f
                s = fp.stat()
                hits.setdefault((s.st_dev, s.st_ino), []).append(fp)
    if len(hits) != 1:
        raise SystemExit(f"harvest {role}: {len(hits)} distinct files match {pattern!r} under {root}, "
                         f"expected 1: {sorted(str(v[0]) for v in hits.values())}")
    # prefer execution/<name> over execution/glob-<hash>/<name>; both are one inode
    return min(next(iter(hits.values())), key=lambda x: len(x.parts))


def output_basenames(call: dict, key: str) -> set:
    v = (call.get("outputs") or {}).get(key)
    vals = v if isinstance(v, list) else [v]
    return {os.path.basename(x) for x in vals if isinstance(x, str)}


def copy_verified(src: Path, dest: Path) -> str:
    """Copy to `<dest>.part`, md5 both, rename. Returns the md5. Existing equal dest is kept."""
    want = md5_file(src)
    if dest.exists() and dest.stat().st_size == src.stat().st_size and md5_file(dest) == want:
        return want
    part = dest.with_name(dest.name + ".part")
    try:
        shutil.copyfile(src, part)
        got = md5_file(part)
        if got != want:
            raise SystemExit(f"harvest REFUSING: {src} → {part} md5 {got} != source {want}")
        os.replace(part, dest)
    finally:
        if part.exists():
            part.unlink()
    return want


def harvest(row, cf):
    pid = row["pid"]
    st = read_state(cf, pid)
    if st is None or st["status"] != "succeeded":
        log(f"{pid}: skip harvest, status {st and st['status']}")
        return st
    meta_path = Path(st["metadata_json"])
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "Succeeded":
        raise SystemExit(f"{pid}: {meta_path} status {meta.get('status')}, not Succeeded")
    pfx = pipeline_spec(row)["prefix"]

    plan = []
    for role, call_name, pattern, key in harvest_roles(row):
        call = final_call(meta, f"{pfx}.{call_name}")
        src = find_one(call, pattern, role)
        names = output_basenames(call, key)
        if src.name not in names:
            raise SystemExit(f"harvest {role}: found {src.name}, but {pfx}.{call_name} output "
                             f"{key} is {sorted(names)}")
        plan.append((role, src))
    plan.append(("metadata", meta_path))
    dests = [src.name for _, src in plan]
    if len(set(dests)) != len(dests):
        raise SystemExit(f"{pid}: harvest destination names collide: {dests}")

    out = paths(cf, pid)["harvest"]
    out.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(HARVEST_COLUMNS)]
    for role, src in plan:
        dest = out / src.name
        md5 = copy_verified(src, dest)
        lines.append("\t".join([role, str(src), str(dest), str(dest.stat().st_size), md5]))
        log(f"{pid}: {role} {dest.name} {dest.stat().st_size} B")
    write_atomic(out / "HARVEST.tsv", "\n".join(lines) + "\n")
    return write_state(cf, pid, status="harvested", harvest_tsv=str(out / "HARVEST.tsv"))


# --- cleanup candidates ------------------------------------------------------------------------

def tree_bytes(root) -> int:
    """Apparent size of a tree, each inode once. Hard links shared with other trees still count."""
    seen, n = set(), 0
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            s = os.lstat(os.path.join(dirpath, f))
            if stat.S_ISREG(s.st_mode) and (s.st_dev, s.st_ino) not in seen:
                seen.add((s.st_dev, s.st_ino))
                n += s.st_size
    return n


def harvest_verified(cf, pid) -> bool:
    """Re-open every keeper: present, same byte count and md5 as HARVEST.tsv recorded."""
    st = read_state(cf, pid)
    tsv = paths(cf, pid)["harvest"] / "HARVEST.tsv"
    if not st or st["status"] != "harvested" or not tsv.exists():
        return False
    lines = tsv.read_text().rstrip("\n").split("\n")
    if tuple(lines[0].split("\t")) != HARVEST_COLUMNS or len(lines) < 2:
        return False
    for ln in lines[1:]:
        r = dict(zip(HARVEST_COLUMNS, ln.split("\t")))
        dest = Path(r["dest"])
        if not dest.exists() or dest.stat().st_size != int(r["bytes"]) or md5_file(dest) != r["md5"]:
            return False
    return True


def cleanup_candidates(rows, cf) -> Path:
    """List every Cromwell tree and loc dir of these pids with its size. Deletes nothing."""
    fa = Path(cf) / "fastqarms"
    lines = ["pid\ttree\tbytes\tharvest_verified"]
    for row in rows:
        pid = row["pid"]
        trees = [fa / "cromwell" / pid] + sorted((fa / "cromwell").glob(f"{pid}__attempt*"))
        trees += [fa / "loc" / pid]
        trees = [t for t in trees if t.is_dir()]
        if not trees:
            continue
        ok = harvest_verified(cf, pid)
        for t in trees:
            lines.append(f"{pid}\t{t}\t{tree_bytes(t)}\t{str(ok).lower()}")
    out = fa / "cleanup_candidates.tsv"
    write_atomic(out, "\n".join(lines) + "\n")
    log(f"wrote {out} ({len(lines) - 1} trees); nothing deleted")
    return out


# --- CLI ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("stage", "submit", "poll", "harvest", "cleanup-candidates"))
    ap.add_argument("--rows", required=True, help="TSV from `arms.py rows --route fastq`")
    ap.add_argument("--cf", default=CF)
    ap.add_argument("--eic", default=EIC)
    ap.add_argument("--pids", default=None, help="comma-separated subset of the rows")
    ap.add_argument("--refcache", default=REFCACHE, help="stage: reference bundle to hard-link")
    ap.add_argument("--retry", action="store_true", help="submit: also resubmit failed pids")
    ap.add_argument("--max-attempts", type=int, default=2)
    args = ap.parse_args(argv)

    rows = select_rows(read_rows(args.rows), args.pids.split(",") if args.pids else None)
    if args.cmd == "cleanup-candidates":
        cleanup_candidates(rows, args.cf)
        return 0
    for row in rows:
        if args.cmd == "stage":
            stage(row, args.cf, args.eic, args.refcache)
        elif args.cmd == "submit":
            submit(row, args.cf, args.eic, retry=args.retry, max_attempts=args.max_attempts)
        elif args.cmd == "poll":
            poll(row, args.cf)
        elif args.cmd == "harvest":
            harvest(row, args.cf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
