"""t117 — ENCODE pseudoreplicates (pr1, pr2) of the 130 t112 counterfactual-arm products.

Each product gets two halves. Each half has the product's two arrays, made the product's way:

* **counts25.npz** — the store overlap rule (`tools/t112/bin25.py counts_from_tagalign`) on the
  half's tagAlign. The rule adds per line, so pr1 + pr2 == the product's counts25, bit for bit.
* **pval25.npz** — the product's own MACS2 signal command with the treatment tagAlign swapped for
  the half, then `bin25.pval_from_bigwig`. Every other flag (fraglen / smooth-win, control, control
  depth, --pval-thresh, --gensz, --mem-gb) is the product's.

THE SPLIT IS THE PIPELINE'S OWN. `encode_task_spr.py` from the product's SIF (md5
077aa84a7cdffc2e2c8489d514869eee, identical in the chip and atac images), with the flags the
pipeline itself recorded for that read set's run: `--pseudoreplication-random-seed 0` (the seed is
then the input's uncompressed byte count, `zcat -f | wc -c`), and `--paired-end` only where the run
was paired (the `pe` arms), which splits by pair. Every seed is recorded in `readset.json` and
`SEEDS.tsv`.

ONE SPLIT PER TREATMENT READ SET. A read set is the tagAlign MACS2 read as treatment:

| arm                                   | read set                    | where it is                  |
|---------------------------------------|-----------------------------|------------------------------|
| base, ratio, ctlid, ctldepth, extsize | the track's base            | `$CF/ta/<...>.tagAlign.gz`   |
| depth, abproxy                        | its own thinned/mixed reads | rebuilt, md5-checked (below) |
| pe, dedup, crop, mapq (FASTQ route)   | its own run's bam2ta        | `$CF/fastqarms/harvest/<pid>`|

The depth and abproxy tagAligns were built in `$SLURM_TMPDIR` by t112 and not kept. They are
rebuilt with the exact commands `bam_arm.plan` gives (the pipeline's `encode_task_subsample_ctl.py`,
seeded by its input's byte count, so deterministic), and the md5 of the rebuilt file must equal the
md5 the product's provenance recorded for the file MACS2 read. Otherwise the task stops.

For BAM-route products the command list `bam_arm.plan` gives today must equal the one the product's
provenance recorded, after swapping the old `$SLURM_TMPDIR` for this one. That proves the MACS2
settings used here are the product's.

THE RATIO ARM IS HELD. What `--ratio` a half should get (the product's literal number, or the same
k times the half's own treatment/control quotient) is a PI decision. `--ratio-mode` has no default
and `half` refuses a ratio product without it.

    python3 pseudoreps.py tasks    --out-dir DIR                  # readsets.tsv + halves.tsv
    python3 pseudoreps.py readset  --tasks readsets.tsv --index I [--tmp T]
    python3 pseudoreps.py half     --tasks halves.tsv   --index I [--tmp T] [--ratio-mode M]
    python3 pseudoreps.py check    [--pids P1,P2]                  # checks + MANIFEST.tsv + SEEDS.tsv
    python3 pseudoreps.py smoke    --pid C19M16__base__base --chrom chr21 [--tmp T]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

T112 = Path(__file__).resolve().parents[1] / "t112"
sys.path.insert(0, str(T112))
import arms  # noqa: E402
import bam_arm  # noqa: E402
import records  # noqa: E402

CF = bam_arm.CF
EIC = bam_arm.EIC
CHRSZ = bam_arm.CHRSZ
PR = f"{CF}/pseudoreps"
PRODUCTS = f"{CF}/products"
HALVES = ("pr1", "pr2")
SPR_SCRIPT = "encode_task_spr.py"
SPR_MD5 = "077aa84a7cdffc2e2c8489d514869eee"

#: arms that leave the treatment reads unchanged: they reuse the base read set's halves.
REUSE_BASE = ("base", "ratio", "ctlid", "ctldepth", "extsize")
#: BAM-route arms with their own treatment read set, rebuilt from the recorded commands.
OWN_BAM = ("depth", "abproxy")

READSET_HEADER = ("index", "readset", "owner_pid", "track", "route", "pipeline", "users")
HALF_HEADER = ("index", "pid", "half", "readset", "arm", "route", "pipeline")


# --------------------------------------------------------------------------------------------
# rows and read sets


def all_rows() -> list:
    """Every t112 product row, as `arms.py rows` writes it (knob_value as JSON text)."""
    return [bam_arm._tsv_row(r) for r in arms.rows("all", dnase="atac", ratio="yes")]


def readset_of(row: dict) -> str:
    """The pid whose treatment tagAlign this product's MACS2 read."""
    if row["route"] == "bam" and row["arm"] in REUSE_BASE:
        return arms.pid(row["track"], "base", "base")
    if row["route"] == "bam" and row["arm"] not in OWN_BAM:
        raise ValueError(f"{row['pid']}: bam arm {row['arm']!r} has no read-set rule")
    return row["pid"]


def task_tables(rows) -> tuple:
    """(readset rows, half rows). Read sets in first-seen order; halves pr1 then pr2 per pid."""
    users = {}
    by_pid = {r["pid"]: r for r in rows}
    for r in rows:
        users.setdefault(readset_of(r), []).append(r["pid"])
    rs = []
    for i, (rid, pids) in enumerate(users.items()):
        o = by_pid[rid]
        rs.append({"index": str(i), "readset": rid, "owner_pid": rid, "track": o["track"],
                   "route": o["route"], "pipeline": o["pipeline"], "users": ",".join(pids)})
    hs = []
    for r in rows:
        for h in HALVES:
            hs.append({"index": str(len(hs)), "pid": r["pid"], "half": h,
                       "readset": readset_of(r), "arm": r["arm"], "route": r["route"],
                       "pipeline": r["pipeline"]})
    return rs, hs


def write_tsv(path, header, rs) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)] + ["\t".join(r[k] for k in header) for r in rs]
    Path(path).write_text("\n".join(lines) + "\n")


def read_tsv(path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def row_of(pid: str) -> dict:
    for r in all_rows():
        if r["pid"] == pid:
            return r
    raise SystemExit(f"no t112 row for pid {pid}")


# --------------------------------------------------------------------------------------------
# what the product recorded


def provenance(pid, products=PRODUCTS) -> dict:
    return json.loads(Path(f"{products}/{pid}/provenance.json").read_text())


def recorded_md5(prov: dict, path) -> str:
    """The md5 the product's provenance recorded for a file of this basename (inputs or fed)."""
    hits = {i["md5"] for i in prov["inputs"] if os.path.basename(i["path"]) == os.path.basename(path)}
    if len(hits) != 1:
        raise SystemExit(f"{prov['pid']}: {len(hits)} recorded md5s for {os.path.basename(path)}")
    return hits.pop()


def old_tmp(prov: dict) -> str:
    """The `$SLURM_TMPDIR` a BAM-route product was built in, read off its first recorded command."""
    first = prov["commands"][0].split()
    if first[:2] != ["mkdir", "-p"] or not first[2].endswith("/work"):
        raise SystemExit(f"{prov['pid']}: first recorded command is not the plan's mkdir: {first}")
    return first[2][: -len("/work")]


def bam_plan(row: dict, tmp: str, products=PRODUCTS, ratio=None) -> dict:
    """`bam_arm.plan` for this row under `tmp`, refused unless it equals what the product recorded.

    The ratio arm's `--ratio` is passed in from the product's own covariates so the comparison does
    not re-count 100 M-line control files; the recorded command then pins it.
    """
    prov = provenance(row["pid"], products)
    if row["arm"] == "ratio" and ratio is None:
        cov = json.loads(Path(f"{products}/{row['pid']}/covariates.json").read_text())
        ratio = cov["knob_value"]
    p = bam_arm.plan(row, CF, EIC, tmp=tmp, products=products, ratio=ratio)
    old = old_tmp(prov)
    want = [c.replace(old, tmp) for c in prov["commands"]]
    if p["commands"] != want:
        diff = [(a, b) for a, b in zip(p["commands"], want) if a != b]
        raise SystemExit(f"{row['pid']}: bam_arm.plan no longer equals the recorded commands "
                         f"({len(p['commands'])} vs {len(want)}); first diff: {diff[:1]}")
    p["recorded_tmp"] = old
    return p


def harvest(pid, cf=CF) -> dict:
    """`{role: {src, dest, bytes, md5}}` from a FASTQ-route pid's HARVEST.tsv."""
    lines = Path(f"{cf}/fastqarms/harvest/{pid}/HARVEST.tsv").read_text().rstrip("\n").split("\n")
    cols = lines[0].split("\t")
    return {rec["role"]: rec for rec in (dict(zip(cols, ln.split("\t"))) for ln in lines[1:])}


def call_command(meta: dict, name: str) -> str:
    calls = meta["calls"][name]
    if len(calls) != 1:
        raise SystemExit(f"{name}: {len(calls)} calls, expected 1")
    return calls[0]["commandLine"]


def spr_flags(cmdline: str) -> dict:
    """`{seed, paired_end}` of a recorded `encode_task_spr.py` commandLine."""
    toks = cmdline.replace("\\\n", " ").split()
    if not any(SPR_SCRIPT in t for t in toks):
        raise SystemExit(f"not an spr command: {cmdline!r}")
    i = toks.index("--pseudoreplication-random-seed")
    return {"seed": int(toks[i + 1]), "paired_end": "--paired-end" in toks}


def readset_source(rs: dict, tmp: str, products=PRODUCTS, cf=CF, eic=EIC) -> dict:
    """Where a read set's treatment tagAlign is, how to build it if it must be rebuilt, its recorded
    md5, and the spr flags the pipeline recorded for that read set's run."""
    row = row_of(rs["owner_pid"])
    pfx = row["pipeline"]
    if row["route"] == "bam":
        p = bam_plan(row, tmp, products)
        prov = provenance(row["pid"], products)
        build = [c for c in p["commands"]
                 if bam_arm.SUBSAMPLE_SCRIPT in c or "| gzip -nc >" in c]
        if row["arm"] in OWN_BAM and not build:
            raise SystemExit(f"{row['pid']}: no rebuild commands in the plan")
        meta = json.loads(Path(f"{eic}/results/{row['track']}/metadata.json").read_text())
        spr_cmd = call_command(meta, f"{pfx}.spr")
        spr_from = f"{eic}/results/{row['track']}/metadata.json"
        ta = p["treatment"]
        md5 = recorded_md5(prov, ta)
    else:
        h = harvest(row["pid"], cf)
        meta = json.loads(Path(h["metadata"]["dest"]).read_text())
        spr_cmd = call_command(meta, f"{pfx}.spr")
        spr_from = h["metadata"]["dest"]
        ta, md5, build = h["treat_ta"]["dest"], h["treat_ta"]["md5"], []
        recorded = [os.path.basename(t) for t in spr_cmd.split() if t.endswith(".tagAlign.gz")]
        if recorded != [os.path.basename(ta)]:
            raise SystemExit(f"{row['pid']}: recorded spr ran on {recorded}, not {ta}")
    flags = spr_flags(spr_cmd)
    return {"row": row, "treatment": ta, "md5": md5, "build": build, "spr_recorded": spr_cmd,
            "spr_recorded_in": spr_from, **flags}


# --------------------------------------------------------------------------------------------
# execution helpers


def sh(cmd: str, cwd=None):
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=True, executable="/bin/bash")


def lines_and_bytes(path) -> tuple:
    out = subprocess.run(f"zcat -f {shlex.quote(str(path))} | wc -lc", shell=True, check=True,
                         capture_output=True, text=True).stdout.split()
    return int(out[0]), int(out[1])


def copy_verified(src, dst) -> str:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    part = f"{dst}.part"
    shutil.copyfile(src, part)
    a, b = records.md5_file(src), records.md5_file(part)
    if a != b:
        raise SystemExit(f"copy md5 mismatch {src} -> {dst}")
    os.replace(part, dst)
    return a


def job_ids() -> list:
    return bam_arm._job_ids()


def git_sha() -> str:
    kit = os.environ.get("KIT") or str(Path(__file__).resolve().parents[2])
    return os.environ.get("T117_GIT_SHA") or bam_arm._git_sha(kit)


def tmpdir(tmp=None) -> str:
    return str(tmp or os.environ.get("SLURM_TMPDIR") or tempfile.mkdtemp())


def readset_dir(rid, pr=PR) -> Path:
    return Path(f"{pr}/_readsets/{rid}")


def pr_name(ta, half) -> str:
    """The name `encode_task_spr.py` gives a half of `ta`."""
    return f"{os.path.basename(bam_arm.strip_ext_ta(ta))}.{half}.tagAlign.gz"


# --------------------------------------------------------------------------------------------
# stage A: one read set -> two halves (tagAligns + counts)


def spr_command(src: dict, ta: str, out_dir: str, sif: str, tmp: str) -> str:
    inner = (f"python3 $(which {SPR_SCRIPT}) {ta} --pseudoreplication-random-seed {src['seed']}"
             + (" --paired-end" if src["paired_end"] else "") + f" --out-dir {out_dir}")
    return bam_arm.apptainer(inner, sif, tmp, f"{tmp}/work")


def run_readset(rs: dict, tmp=None, pr=PR, products=PRODUCTS, chrsz=CHRSZ,
                chrom=None, ta_override=None) -> dict:
    """Split one read set. `chrom`/`ta_override` are the smoke's: split a one-chromosome copy."""
    import bin25

    tmp = tmpdir(tmp)
    src = readset_source(rs, tmp, products)
    row = src["row"]
    sif = bam_arm.SIF[row["pipeline"]]
    sif_md5, sif_sha = bam_arm.check_sif(sif, records.PIPELINE_REPO[row["pipeline"]])
    Path(f"{tmp}/work").mkdir(parents=True, exist_ok=True)
    Path(f"{tmp}/ta").mkdir(parents=True, exist_ok=True)
    commands = []

    ta = src["treatment"]
    if src["build"] and ta_override is None:
        for c in src["build"]:
            sh(c, cwd=f"{tmp}/work")
            commands.append(c)
    if ta_override is None:
        got = records.md5_file(ta)
        if got != src["md5"]:
            raise SystemExit(f"{rs['readset']}: treatment {ta} md5 {got} != recorded {src['md5']}")
    else:
        ta = ta_override
    n_lines, n_bytes = lines_and_bytes(ta)

    spr_out = f"{tmp}/spr"
    Path(spr_out).mkdir(parents=True, exist_ok=True)
    spr_md5 = subprocess.run(
        f"apptainer exec --cleanenv {sif} bash -c 'md5sum $(which {SPR_SCRIPT})'", shell=True,
        check=True, capture_output=True, text=True).stdout.split()[0]
    if spr_md5 != SPR_MD5:
        raise SystemExit(f"{SPR_SCRIPT} md5 {spr_md5} != pinned {SPR_MD5} in {sif}")
    cmd = spr_command(src, ta, spr_out, sif, tmp)
    sh(cmd, cwd=f"{tmp}/work")
    commands.append(cmd)

    sizes = bin25.load_chrsz(chrsz)
    if chrom:
        sizes = {chrom: sizes[chrom]}
    out = readset_dir(rs["readset"], pr)
    stage = Path(f"{tmp}/stage_rs")
    stage.mkdir(parents=True, exist_ok=True)
    halves = {}
    for h in HALVES:
        f = f"{spr_out}/{pr_name(ta, h)}"
        if not Path(f).is_file() or Path(f).stat().st_size == 0:
            raise SystemExit(f"spr wrote no {f}")
        nl, nb = lines_and_bytes(f)
        stats = {}
        bin25.write_npz(stage / f"{h}.counts25.npz",
                        bin25.counts_from_tagalign(f, sizes, stats=stats))
        halves[h] = {"tagalign": str(out / os.path.basename(f)), "n_lines": nl,
                     "uncompressed_bytes": nb, "n_lines_main": stats["n_lines_main"],
                     "local": f}
    if sum(v["n_lines"] for v in halves.values()) != n_lines:
        raise SystemExit(f"{rs['readset']}: pr1 + pr2 lines != {n_lines}")

    for h, v in halves.items():
        v["md5"] = copy_verified(v.pop("local"), v["tagalign"])
        v["counts25_md5"] = copy_verified(stage / f"{h}.counts25.npz", out / f"{h}.counts25.npz")
        v["counts25"] = str(out / f"{h}.counts25.npz")
    rec = {
        "schema": 1, "readset": rs["readset"], "owner_pid": rs["owner_pid"],
        "users": rs["users"].split(","), "route": row["route"], "pipeline": row["pipeline"],
        "sif": sif, "sif_md5": sif_md5, "sif_sha256": sif_sha, "spr_script_md5": spr_md5,
        "treatment": {"path": src["treatment"], "md5": src["md5"], "n_lines": n_lines,
                      "uncompressed_bytes": n_bytes, "rebuilt": bool(src["build"]),
                      "chrom_only": chrom},
        "seed": {"arg": src["seed"], "paired_end": src["paired_end"],
                 "effective": n_bytes if src["seed"] == 0 else src["seed"],
                 "rule": ("encode_task_spr.py: seed 0 means shuf --random-source is keyed by the "
                          "input's uncompressed byte count (zcat -f | wc -c)"),
                 "recorded_spr_command": src["spr_recorded"],
                 "recorded_in": src["spr_recorded_in"]},
        "commands": commands, "halves": halves, "slurm_job_ids": job_ids(),
        "code": {"git_sha": git_sha()}, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                      time.gmtime()),
    }
    (out / "readset.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[t117] readset {rs['readset']} done: {n_lines} lines -> "
          f"{halves['pr1']['n_lines']} + {halves['pr2']['n_lines']}", flush=True)
    return rec


# --------------------------------------------------------------------------------------------
# stage B: one (product, half) -> MACS2 on the half, pval25 + counts25 + provenance


def signal_plan(row: dict, pr_ta: str, tmp: str, products=PRODUCTS, chrsz=CHRSZ,
                ratio_mode=None) -> dict:
    """The product's MACS2 signal command with the treatment tagAlign swapped for `pr_ta`."""
    pfx = row["pipeline"]
    out = f"{tmp}/out"
    flag = bam_arm.FRAGLEN_FLAG[pfx]
    if row["route"] == "bam":
        p = bam_plan(row, tmp, products)
        control, fraglen = p["control"], p["fraglen"]
        parsed = bam_arm.parse_signal_command(bam_arm.signal_command_line(row["track"], EIC, pfx),
                                              pfx)
        tas = [pr_ta] + ([control] if control else [])
        pre, ratio = [], None
        if row["arm"] == "ratio":
            if ratio_mode not in ("literal", "k"):
                raise SystemExit(f"{row['pid']}: a ratio product needs --ratio-mode literal|k "
                                 "(a PI decision; there is no default)")
            ratio = (p["knob_value"] if ratio_mode == "literal"
                     else bam_arm.ratio_for(row["level"], pr_ta, control))
            copy = f"{tmp}/patched/{bam_arm.SIGNAL_SCRIPT['chip']}"
            new = bam_arm.ratio_new(ratio)
            pre = [f"mkdir -p {tmp}/patched",
                   bam_arm.apptainer(f"cp {bam_arm.RATIO_SCRIPT} {copy}", p["sif"], tmp,
                                     f"{tmp}/work"),
                   f"python3 -c {shlex.quote(bam_arm.PATCH_PY)} {copy} "
                   f"{shlex.quote(bam_arm.RATIO_OLD)} {shlex.quote(new)}"]
            inner = bam_arm.build_signal_command(
                parsed, tas, out, fraglen=fraglen, chrsz=chrsz, fraglen_flag=flag,
                invocation=[f"PYTHONPATH={bam_arm.CHIP_SRC_DIR}", "python3", copy])
            patch = {"script": bam_arm.RATIO_SCRIPT, "copy": copy, "old": bam_arm.RATIO_OLD,
                     "new": new, "expect_md5": bam_arm.RATIO_SCRIPT_MD5}
        else:
            inner = bam_arm.build_signal_command(parsed, tas, out, fraglen=fraglen, chrsz=chrsz,
                                                 fraglen_flag=flag)
            patch = None
        recorded = [c for c in p["commands"] if any(s in c for s in bam_arm.SIGNAL_SCRIPT.values())]
        source = {"kind": "product provenance.commands (bam route)",
                  "recorded_signal_command": recorded[-1] if recorded else None,
                  "recorded_tmp": p["recorded_tmp"]}
    else:
        h = harvest(row["pid"])
        meta = json.loads(Path(h["metadata"]["dest"]).read_text())
        cmdline = call_command(meta, f"{pfx}.macs2_signal_track")
        parsed = bam_arm.parse_signal_command(cmdline, pfx)
        control = h["ctl_ta"]["dest"] if "ctl_ta" in h else None
        rec_tas = [os.path.basename(t) for t in parsed["tas"]]
        want = [os.path.basename(h["treat_ta"]["dest"])] + (
            [os.path.basename(control)] if control else [])
        if rec_tas != want:
            raise SystemExit(f"{row['pid']}: recorded MACS2 read {rec_tas}, harvest has {want}")
        fraglen = dict(parsed["flags"])[flag]
        tas = [pr_ta] + ([control] if control else [])
        inner = bam_arm.build_signal_command(parsed, tas, out, fraglen=None, chrsz=chrsz,
                                             fraglen_flag=flag)
        pre, patch, ratio = [], None, None
        source = {"kind": "harvested Cromwell metadata.json (fastq route)",
                  "recorded_signal_command": cmdline, "metadata": h["metadata"]["dest"]}
    sif = bam_arm.SIF[pfx]
    cmd = bam_arm.apptainer(inner, sif, tmp, f"{tmp}/work")
    return {"sif": sif, "commands": [f"mkdir -p {tmp}/work {out}"] + pre + [cmd],
            "signal_command": cmd, "tas": tas, "control": control, "fraglen": fraglen,
            "bigwig": bam_arm.signal_bigwig(tas, out), "patch": patch, "ratio": ratio,
            "ratio_mode": ratio_mode if row["arm"] == "ratio" else None, "source": source}


def run_half(task: dict, tmp=None, pr=PR, products=PRODUCTS, chrsz=CHRSZ, ratio_mode=None,
             chrom=None, out_override=None, control_swap=None) -> dict:
    """One half of one product. `chrom`, `out_override` and `control_swap` (old path, new path)
    exist for the smoke only, which runs MACS2 on a one-chromosome copy of the control."""
    import bin25

    tmp = tmpdir(tmp)
    pid, half, rid = task["pid"], task["half"], task["readset"]
    row = row_of(pid)
    rsdir = readset_dir(rid, pr)
    rsrec = json.loads((rsdir / "readset.json").read_text())
    hrec = rsrec["halves"][half]
    pr_ta = hrec["tagalign"]
    if records.md5_file(pr_ta) != hrec["md5"]:
        raise SystemExit(f"{pid} {half}: {pr_ta} md5 != readset.json")

    sp = signal_plan(row, pr_ta, tmp, products, chrsz, ratio_mode)
    if control_swap and sp["control"]:
        a, b = control_swap
        sp["commands"] = [c.replace(a, b) for c in sp["commands"]]
        sp["tas"] = [b if t == a else t for t in sp["tas"]]
        sp["control"] = b
        sp["bigwig"] = bam_arm.signal_bigwig(sp["tas"], f"{tmp}/out")
    sif_md5, sif_sha = bam_arm.check_sif(sp["sif"], records.PIPELINE_REPO[row["pipeline"]])
    Path(f"{tmp}/work").mkdir(parents=True, exist_ok=True)
    patch = None
    for c in sp["commands"]:
        sh(c, cwd=f"{tmp}/work")
        if sp["patch"] and c.startswith("python3 -c"):
            patch = bam_arm.apply_patch(sp["patch"])
    if sp["patch"] and patch is None:
        raise SystemExit(f"{pid}: the --ratio patch step never ran")
    bw = Path(sp["bigwig"])
    if not bw.is_file() or bw.stat().st_size == 0:
        raise SystemExit(f"{pid} {half}: no bigwig at {bw}")

    sizes = bin25.load_chrsz(chrsz)
    if chrom:
        sizes = {chrom: sizes[chrom]}
    stage = Path(f"{tmp}/stage/{pid}/{half}")
    stage.mkdir(parents=True, exist_ok=True)
    pval = bin25.pval_from_bigwig(str(bw), sizes, f"{tmp}/bdg")
    import numpy as np
    n_nan = int(sum(np.isnan(v).sum() for v in pval.values()))
    bin25.write_npz(stage / "pval25.npz", pval)
    shutil.copyfile(hrec["counts25"], stage / "counts25.npz")

    dest = Path(out_override or f"{pr}/{pid}/{half}")
    control_md5 = records.md5_file(sp["control"]) if sp["control"] else None
    prov = {
        "schema": 1, "pid": pid, "half": half, "readset": rid, "arm": row["arm"],
        "level": row["level"], "route": row["route"],
        "product": {"dir": f"{products}/{pid}",
                    "counts25_md5": records.md5_file(f"{products}/{pid}/counts25.npz"),
                    "pval25_md5": records.md5_file(f"{products}/{pid}/pval25.npz")},
        "pipeline": {"repo": records.PIPELINE_REPO[row["pipeline"]],
                     "release": records.PIPELINES[records.PIPELINE_REPO[row["pipeline"]]][0],
                     "sif": sp["sif"], "sif_md5": sif_md5, "sif_sha256": sif_sha},
        "split": {"readset_json": str(rsdir / "readset.json"),
                  "tagalign": pr_ta, "tagalign_md5": records.md5_file(pr_ta),
                  "n_lines": hrec["n_lines"], "seed": rsrec["seed"],
                  "product_treatment": rsrec["treatment"]},
        "macs2": {"commands": sp["commands"], "treatment": pr_ta,
                  "control": sp["control"], "control_md5": control_md5,
                  "fraglen_or_smooth_win": sp["fraglen"], "ratio": sp["ratio"],
                  "ratio_mode": sp["ratio_mode"], "patched_script": patch,
                  "settings_from": sp["source"], "bigwig_md5": records.md5_file(bw)},
        "pval_n_nan": n_nan, "chrom_only": chrom,
        "outputs": [{"path": str(dest / n), "md5": records.md5_file(stage / n)}
                    for n in ("counts25.npz", "pval25.npz")],
        "slurm_job_ids": job_ids(), "code": {"git_sha": git_sha()},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (stage / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    for n in ("counts25.npz", "pval25.npz", "provenance.json"):   # the record last
        copy_verified(stage / n, dest / n)
    print(f"[t117] {pid} {half} done -> {dest}  (pval NaN: {n_nan})", flush=True)
    return prov


# --------------------------------------------------------------------------------------------
# checks, manifest, seeds


def check_pid(pid: str, pr=PR, products=PRODUCTS, chrom=None, half_dirs=None) -> dict:
    """Additivity, half depth, grid and NaN checks for one product."""
    import numpy as np
    import bin25

    prod = Path(f"{products}/{pid}")
    halves = half_dirs or {h: Path(f"{pr}/{pid}/{h}") for h in HALVES}
    res = {"pid": pid, "checks": {}, "problems": []}
    missing = [str(d / n) for d in halves.values() for n in
               ("counts25.npz", "pval25.npz", "provenance.json") if not (d / n).is_file()]
    if missing:
        res["problems"].append(f"missing: {missing}")
        res["pass"] = False
        return res
    c = bin25.read_npz(prod / "counts25.npz")
    pv = bin25.read_npz(prod / "pval25.npz")
    hc = {h: bin25.read_npz(d / "counts25.npz") for h, d in halves.items()}
    hp = {h: bin25.read_npz(d / "pval25.npz") for h, d in halves.items()}
    chroms = [chrom] if chrom else list(c)

    keys_ok = all(set(chroms) <= set(hc[h]) for h in HALVES)
    add = keys_ok and all(
        np.array_equal(hc["pr1"][k].astype(np.uint64) + hc["pr2"][k], c[k]) for k in chroms)
    res["checks"]["additivity"] = bool(add)

    prov = {h: json.loads((d / "provenance.json").read_text()) for h, d in halves.items()}
    total = prov["pr1"]["split"]["product_treatment"]["n_lines"]
    cov = json.loads((prod / "covariates.json").read_text())
    if not chrom and total != cov["depth"]:
        res["problems"].append(f"read-set lines {total} != product depth {cov['depth']}")
    dev = {h: abs(prov[h]["split"]["n_lines"] - total / 2) / (total / 2) for h in HALVES}
    res["half_lines"] = {h: prov[h]["split"]["n_lines"] for h in HALVES}
    res["product_lines"] = total
    res["half_rel_dev"] = dev
    res["checks"]["half_depth"] = bool(all(v <= 0.005 for v in dev.values()))

    grid = all(set(hp[h]) >= set(chroms) and set(hc[h]) >= set(chroms) for h in HALVES) and all(
        hp[h][k].shape == pv[k].shape and hc[h][k].shape == c[k].shape
        and hp[h][k].dtype == pv[k].dtype and hc[h][k].dtype == c[k].dtype
        for h in HALVES for k in chroms)
    if not chrom:
        grid = grid and all(list(hp[h]) == list(pv) and list(hc[h]) == list(c) for h in HALVES)
    res["checks"]["grid"] = bool(grid)
    nan = {h: int(sum(np.isnan(hp[h][k]).sum() for k in chroms)) for h in HALVES}
    res["pval_nan"] = nan
    res["checks"]["no_nan"] = all(v == 0 for v in nan.values())
    # sanity only, not a pass/fail: how each half's p track tracks the product's, on one chromosome
    k = chroms[0]
    res["pval_pearson_vs_product"] = {
        "chrom": k, **{h: float(np.corrcoef(hp[h][k].astype(np.float64),
                                             pv[k].astype(np.float64))[0, 1]) for h in HALVES}}
    res["pval_mean_ratio_vs_product"] = {
        h: float(np.mean(hp[h][k], dtype=np.float64) / np.mean(pv[k], dtype=np.float64))
        for h in HALVES}
    res["pass"] = all(res["checks"].values()) and not res["problems"]
    return res


def run_checks(pids=None, pr=PR, products=PRODUCTS) -> dict:
    rows = all_rows()
    want = [r["pid"] for r in rows if pids is None or r["pid"] in pids]
    out = Path(f"{pr}/checks")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for pid in want:
        r = check_pid(pid, pr, products)
        (out / f"{pid}.json").write_text(json.dumps(r, indent=1) + "\n")
        results.append(r)
        print(f"[t117] check {pid}: {'PASS' if r['pass'] else 'FAIL'} {r['checks']} "
              f"{r['problems']}", flush=True)
    names = ("additivity", "half_depth", "grid", "no_nan")
    summary = {"n": len(results), "all_pass": all(r["pass"] for r in results),
               "per_check": {n: [sum(1 for r in results if r["checks"].get(n)), len(results)]
                             for n in names},
               "failed": [r["pid"] for r in results if not r["pass"]],
               "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if pids is None:
        (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        write_manifest(rows, pr)
        write_seeds(pr)
    print(json.dumps(summary), flush=True)
    return summary


MANIFEST_HEADER = ("pid", "half", "readset", "arm", "level", "route", "pipeline", "n_lines",
                   "product_lines", "seed_effective", "paired_end", "control", "fraglen_or_smooth_win",
                   "counts25_md5", "pval25_md5", "tagalign", "slurm_job_ids", "dir")


def write_manifest(rows, pr=PR) -> None:
    lines = ["\t".join(MANIFEST_HEADER)]
    for r in rows:
        for h in HALVES:
            d = Path(f"{pr}/{r['pid']}/{h}")
            if not (d / "provenance.json").is_file():
                continue
            p = json.loads((d / "provenance.json").read_text())
            md5 = {os.path.basename(o["path"]): o["md5"] for o in p["outputs"]}
            vals = [r["pid"], h, p["readset"], r["arm"], r["level"], r["route"], r["pipeline"],
                    p["split"]["n_lines"], p["split"]["product_treatment"]["n_lines"],
                    p["split"]["seed"]["effective"], p["split"]["seed"]["paired_end"],
                    os.path.basename(p["macs2"]["control"]) if p["macs2"]["control"] else "none",
                    p["macs2"]["fraglen_or_smooth_win"], md5["counts25.npz"], md5["pval25.npz"],
                    p["split"]["tagalign"], ",".join(p["slurm_job_ids"]), str(d)]
            lines.append("\t".join(str(v) for v in vals))
    Path(f"{pr}/MANIFEST.tsv").write_text("\n".join(lines) + "\n")


SEEDS_HEADER = ("readset", "users", "treatment", "treatment_md5", "n_lines", "seed_arg",
                "seed_effective", "paired_end", "pr1_lines", "pr2_lines", "pr1_md5", "pr2_md5")


def write_seeds(pr=PR) -> None:
    lines = ["\t".join(SEEDS_HEADER)]
    for f in sorted(Path(f"{pr}/_readsets").glob("*/readset.json")):
        r = json.loads(f.read_text())
        vals = [r["readset"], ",".join(r["users"]), r["treatment"]["path"], r["treatment"]["md5"],
                r["treatment"]["n_lines"], r["seed"]["arg"], r["seed"]["effective"],
                r["seed"]["paired_end"], r["halves"]["pr1"]["n_lines"],
                r["halves"]["pr2"]["n_lines"], r["halves"]["pr1"]["md5"],
                r["halves"]["pr2"]["md5"]]
        lines.append("\t".join(str(v) for v in vals))
    Path(f"{pr}/SEEDS.tsv").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------------
# smoke: one product end to end on one chromosome


def smoke(pid: str, chrom: str, tmp=None, pr=PR, products=PRODUCTS) -> dict:
    """Split a one-chromosome copy of `pid`'s read set, run both halves' MACS2 on it with the
    product's control cut to the same chromosome, and check that chromosome against the product.

    Every step shares the one `tmp`, because each container binds only its own `tmp`."""
    tmp = tmpdir(tmp)
    spr_root = f"{pr}/smoke/{pid}__{chrom}"
    row = row_of(pid)
    rs = {"readset": readset_of(row), "owner_pid": readset_of(row), "users": pid}
    src = readset_source(rs, tmp, products)
    Path(f"{tmp}/c").mkdir(parents=True, exist_ok=True)
    ta_c = f"{tmp}/c/{os.path.basename(src['treatment'])}"
    sh(f"zcat {shlex.quote(src['treatment'])} | awk '$1==\"{chrom}\"' | gzip -nc > {ta_c}")
    rec = run_readset(rs, tmp, pr=spr_root, products=products, chrom=chrom,
                      ta_override=ta_c)
    # the control, cut to the same chromosome, at the same basename: MACS2 on one chromosome
    ctl = signal_plan(row, ta_c, tmp, products)["control"]
    ctl_c = None
    if ctl:
        ctl_c = f"{tmp}/c/ctl/{os.path.basename(ctl)}"
        Path(ctl_c).parent.mkdir(parents=True, exist_ok=True)
        sh(f"zcat {shlex.quote(ctl)} | awk '$1==\"{chrom}\"' | gzip -nc > {ctl_c}")
    dirs = {}
    for h in HALVES:
        dirs[h] = Path(f"{spr_root}/{pid}/{h}")
        run_half({"pid": pid, "half": h, "readset": rs["readset"]}, tmp, pr=spr_root,
                 products=products, chrom=chrom, out_override=dirs[h],
                 control_swap=(ctl, ctl_c) if ctl else None)
    res = check_pid(pid, spr_root, products, chrom=chrom, half_dirs=dirs)
    res["readset"] = {k: rec[k] for k in ("seed", "treatment")}
    Path(f"{spr_root}/SMOKE.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"smoke_pass": res["pass"], "checks": res["checks"],
                      "problems": res["problems"]}), flush=True)
    return res


# --------------------------------------------------------------------------------------------


def dry_plan(pr=PR) -> int:
    """Resolve every read set's source and every half's MACS2 command without running anything.

    Stdlib only, so it runs on the login node. It is what proves, before any compute, that the
    recorded commands of all 90 BAM-route products still equal `bam_arm.plan`, that every FASTQ
    harvest names the tagAligns its recorded MACS2 read, and that only the ratio halves are held.
    """
    rs, hs = task_tables(all_rows())
    tmp = "/TMP"
    bad, held = [], []
    for r in rs:
        try:
            s = readset_source(r, tmp)
            print(f"readset {r['readset']}: {s['treatment']} md5={s['md5']} seed={s['seed']} "
                  f"pe={s['paired_end']} rebuild={len(s['build'])}")
        except SystemExit as e:
            bad.append(f"readset {r['readset']}: {e}")
    for h in hs:
        if h["half"] != "pr1":
            continue
        row = row_of(h["pid"])
        pr_ta = f"{pr}/_readsets/{h['readset']}/HALF.pr1.tagAlign.gz"
        try:
            sp = signal_plan(row, pr_ta, tmp)
            print(f"half {h['pid']}: control={sp['control']} frag={sp['fraglen']}")
        except SystemExit as e:
            (held if row["arm"] == "ratio" and "ratio-mode" in str(e) else bad).append(
                f"half {h['pid']}: {e}")
    print(f"read sets {len(rs)}, halves {len(hs)}, held (ratio) {len(held)}, problems {len(bad)}")
    for b in bad:
        print("PROBLEM", b)
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("tasks")
    pt.add_argument("--out-dir", default=PR)
    for name in ("readset", "half"):
        p = sub.add_parser(name)
        p.add_argument("--tasks", required=True)
        p.add_argument("--index", required=True, type=int)
        p.add_argument("--tmp", default=None)
        p.add_argument("--pr", default=PR)
        if name == "half":
            p.add_argument("--ratio-mode", choices=("literal", "k"), default=None)
    pp = sub.add_parser("plan", help="dry run: resolve every read set and half, execute nothing")
    pp.add_argument("--pr", default=PR)
    pc = sub.add_parser("check")
    pc.add_argument("--pids", default=None)
    pc.add_argument("--pr", default=PR)
    ps = sub.add_parser("smoke")
    ps.add_argument("--pid", required=True)
    ps.add_argument("--chrom", required=True)
    ps.add_argument("--tmp", default=None)
    ps.add_argument("--pr", default=PR)
    args = ap.parse_args(argv)

    if args.cmd == "tasks":
        rs, hs = task_tables(all_rows())
        write_tsv(f"{args.out_dir}/readsets.tsv", READSET_HEADER, rs)
        write_tsv(f"{args.out_dir}/halves.tsv", HALF_HEADER, hs)
        print(f"{len(rs)} read sets, {len(hs)} halves -> {args.out_dir}")
        return 0
    if args.cmd in ("readset", "half"):
        rows = read_tsv(args.tasks)
        if not 0 <= args.index < len(rows):
            raise SystemExit(f"{args.tasks}: no row {args.index}")
        t = rows[args.index]
        if t["index"] != str(args.index):
            raise SystemExit(f"{args.tasks}: row {args.index} carries index {t['index']}")
        if args.cmd == "readset":
            run_readset(t, args.tmp, pr=args.pr)
        else:
            run_half(t, args.tmp, pr=args.pr, ratio_mode=args.ratio_mode)
        return 0
    if args.cmd == "plan":
        return dry_plan(args.pr)
    if args.cmd == "check":
        s = run_checks(args.pids.split(",") if args.pids else None, pr=args.pr)
        return 0 if s["all_pass"] else 1
    if args.cmd == "smoke":
        return 0 if smoke(args.pid, args.chrom, args.tmp, pr=args.pr)["pass"] else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
