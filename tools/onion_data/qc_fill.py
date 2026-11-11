"""Chunk QCt -- read-derived QC fill tool for the D6 arm "QC on input only".

The build leaves the QC columns of `measurements.tsv` (`minicandi.contract.QC_COLUMNS`) empty, or
fills them from somewhere else. This tool computes them again FROM THE READS OF THE MEASUREMENT
ITSELF and writes a side table (`measurement_id` + the 7 QC columns) that `config["qc_table"]`
reads. The frozen tables never change.

QC IS TIED TO THE READS THAT MADE IT. On Nibi, old QC files beside some BAMs describe another run
(memory: EIC_REPRO samstats are stale). So nothing here reads a QC file that it did not just make.
Per measurement, `run`:
  1. checks that the measurement dir's `provenance.json` names the row's `measurement_id` and that
     `counts.h5` has the row's `sha256`;
  2. reads the kept read set (the pr1 and pr2 tagAlign halves listed in `provenance.json`), and
     checks every file's md5 against the `READSET.json` beside it and every half's line count
     against `provenance.json`;
  3. computes NSC, RSC and the fragment length with the pipeline's own `encode_task_xcor.py` in the
     pipeline image (chip: chip.wdl xcor defaults; atac: atac.wdl defaults; DNase: the atac image
     and atac defaults, because the DNase image has no xcor step -- same route as PI ruling 19);
  4. computes NRF, PBC1, PBC2 with the awk of `encode_task_filter.py::pbc_qc_se/pbc_qc_pe`
     (chip-seq-pipeline2 v2.2.2) on the reads BEFORE duplicate removal: for a `base` measurement
     the sibling `nodedup` measurement (same experiment, unit, layer; its `meta.json` names the base
     as its parent), for a `nodedup` measurement its own reads; for every other arm, or when no
     nodedup measurement exists, the three columns stay empty;
  5. records the values, the md5s and line counts of every input file, the commands, the image
     sha256, the counts sha256, the cost and the job id in `<out>/<measurement_id>.json`.
`collect` refuses a record whose counts sha256 is not the row's (a rebuilt measurement).

Fixed facts and approximations (logged in every record):
* `frip` is empty for every row: no peaks are called.
* A pseudoreplicate half (`pr1`, `pr2`) has no row; the dataset gives it its parent measurement's
  QC row (approximation).
* PBC2 is empty when no position has exactly two reads (the pipeline writes -1 there).
* The xcor input is the measurement's kept read set. chip.wdl feeds xcor the non-deduplicated
  R1 reads trimmed to 50 bp for a FASTQ start; this tool measures the reads the measurement has.
* PE keys: consecutive tagAlign lines are the two mates (`-mate1` order); the key is
  (chr1, start1, chr2, end2, strand1, strand2), as `bedtools bamtobed -bedpe` columns
  1, 2, 4, 6, 9, 10. ATAC tagAligns carry the Tn5 shift, a fixed per-strand offset that keeps
  positions distinct except where the pipeline's clamp fires.

    python qc_fill.py coverage --tables DIR [--qc-table TSV] [--missing-rows]
    python qc_fill.py run --tables DIR --rows 0-9,12 --work W --out /scratch/mforooz/onion/qc/
    python qc_fill.py collect --tables DIR --out $O/tables/eic_v1_qc.tsv [--qc-dir DIR]

Row indices are 0-based data rows of `measurements.tsv` (the `run --rows` spec and the
`coverage --missing-rows` spec use the same index).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from onion_data import align_views as AV  # noqa: E402

bam_arm, records = AV.bam_arm, AV.records

#: = minicandi.contract.QC_COLUMNS (a test checks it; importing minicandi here would pull torch)
QC_COLUMNS = ("nsc", "rsc", "nrf", "pbc1", "pbc2", "frip", "frag_len")
INFORMATIVE = tuple(c for c in QC_COLUMNS if c != "frip")
OUT_DEFAULT = "/scratch/mforooz/onion/qc"
COVERED_ROLES = ("train", "val")
MITO = "chrM"
PIPELINE_KEY = {v: k for k, v in AV.PIPELINE_NAME.items()}
#: xcor settings of each pipeline (chip.wdl v2.2.2: xcor_subsample_reads 15000000,
#: chip_seq_type = pipeline_type, xcor_exclusion_range_min -500; atac.wdl v2.2.3: 25000000,
#: no chip_seq_type and no exclusion range). DNase uses the atac image and settings.
XCOR_ARGS = {"chip": "--subsample 15000000 --chip-seq-type histone --exclusion-range-min -500",
             "atac": "--subsample 25000000", "dnase": "--subsample 25000000"}
XCOR_IMAGE = {"chip": "chip", "atac": "atac", "dnase": "atac"}
APPROXIMATIONS = [
    "frip empty: no peaks are called",
    "pr1/pr2 halves use their parent measurement's QC row",
    "xcor input is the measurement's kept read set (not chip.wdl's non-dedup trimmed R1)",
]
#: encode_task_filter.py pbc_qc_se / pbc_qc_pe (chip-seq-pipeline2 v2.2.2), verbatim
PBC_AWK = ("""awk 'BEGIN{mt=0;m0=0;m1=0;m2=0} ($1==1){m1=m1+1} ($1==2){m2=m2+1} {m0=m0+1} """
           """{mt=mt+$1} END{m1_m2=-1.0; if(m2>0) m1_m2=m1/m2; m0_mt=0; if (mt>0) m0_mt=m0/mt; """
           """m1_m0=0; if (m0>0) m1_m0=m1/m0; printf "%d\\t%d\\t%d\\t%d\\t%f\\t%f\\t%f\\n","""
           """mt,m0,m1,m2,m0_mt,m1_m0,m1_m2}'""")
#: the key columns: SE tagAlign 1,2,3,6; PE (two lines pasted) chr1,start1,chr2,end2,str1,str2
KEY_AWK = {1: """awk 'BEGIN{OFS="\\t"} $1!="%s"{print $1,$2,$3,$6}'""",
           2: """awk 'BEGIN{OFS="\\t"} $1!="%s"{print $1,$2,$7,$9,$6,$12}'"""}


class Refused(Exception):
    """The reads cannot be tied to the measurement (checksum, line count or id mismatch)."""


class NoReadset(Exception):
    """The measurement has no kept read set this tool can read."""


# --------------------------------------------------------------------------------------------
# tables and row specs

def read_tsv(path) -> tuple:
    with open(path, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r.fieldnames or []), list(r)


def parse_rows(spec: str) -> list:
    """'0-3,7,9-10' -> [0, 1, 2, 3, 7, 9, 10] (a SLURM array spec without the % limit)."""
    out = []
    for part in spec.split("%")[0].split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def rows_spec(idx) -> str:
    idx = sorted(set(idx))
    parts, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        parts.append(str(idx[i]) if i == j else f"{idx[i]}-{idx[j]}")
        i = j + 1
    return ",".join(parts)


def with_qc_table(rows: list, qc_table) -> list:
    """The rows as `config["qc_table"]` makes them: a listed measurement's 7 QC columns are the
    side table's; an unlisted one keeps the columns of `measurements.tsv`."""
    if not qc_table:
        return rows
    _, qrows = read_tsv(qc_table)
    q = {r["measurement_id"]: r for r in qrows}
    return [{**r, **{c: q[r["measurement_id"]].get(c, "") for c in QC_COLUMNS}}
            if r["measurement_id"] in q else r for r in rows]


def coverage(rows: list) -> dict:
    """{column: (n with a value, n)} over the training and validation rows."""
    sel = [r for r in rows if r.get("in_eic") in COVERED_ROLES]
    return {c: (sum(1 for r in sel if (r.get(c) or "").strip()), len(sel)) for c in QC_COLUMNS}


def missing_rows(rows: list) -> list:
    """Indices of training and validation rows that lack any informative QC column."""
    return [i for i, r in enumerate(rows) if r.get("in_eic") in COVERED_ROLES
            and any(not (r.get(c) or "").strip() for c in INFORMATIVE)]


# --------------------------------------------------------------------------------------------
# tie the reads to the measurement

def measurement_dir(row: dict) -> Path:
    cp = row.get("counts_path") or ""
    if not cp.startswith("h5:"):
        raise NoReadset(f"counts_path {cp!r} is not a measurement dir (h5:...)")
    return Path(cp[3:]).parent


def load_prov(mdir: Path, mid: str) -> dict:
    f = mdir / "provenance.json"
    if not f.is_file():
        raise NoReadset(f"no provenance.json in {mdir}")
    prov = json.loads(f.read_text())
    if prov.get("measurement_id") != mid:
        raise Refused(f"{f} names {prov.get('measurement_id')!r}, the row is {mid!r}")
    if "halves" not in prov:
        raise NoReadset(f"{f} lists no read-set halves")
    return prov


def verify_readset(prov: dict) -> list:
    """Every half file of a measurement: md5 against its READSET.json, lines against
    provenance.json. Returns [{half, path, md5, lines}] in provenance order."""
    out = []
    for h in AV.HALVES:
        files = prov["halves"][h]["tagalign"]
        files = [files] if isinstance(files, str) else files
        n = 0
        for p in files:
            rsf = Path(p).parent / "READSET.json"
            if not Path(p).is_file() or not rsf.is_file():
                raise Refused(f"{p}: read set file or its READSET.json is missing")
            rs = json.loads(rsf.read_text())
            if rs.get(h) != str(p):
                raise Refused(f"{rsf}: half {h} is {rs.get(h)!r}, provenance lists {p}")
            md5 = records.md5_file(p)
            if md5 != rs["md5"][h]:
                raise Refused(f"{p}: md5 {md5} != READSET.json {rs['md5'][h]}")
            k = AV.count_lines(p)
            n += k
            out.append({"half": h, "path": str(p), "md5": md5, "lines": k})
        if n != int(prov["halves"][h]["n_lines"]):
            raise Refused(f"{prov['measurement_id']} {h}: {n} lines, provenance says "
                          f"{prov['halves'][h]['n_lines']}")
    return out


def check_counts(row: dict, mdir: Path) -> str:
    sha = AV.sha256_file(mdir / "counts.h5")
    if row.get("sha256") and sha != row["sha256"]:
        raise Refused(f"{mdir}/counts.h5 sha256 {sha} != the row's {row['sha256']}")
    return sha


def predup_readset(mid: str, mdir: Path, prov: dict) -> tuple:
    """(source measurement id, verified files) of the reads before duplicate removal, or
    (None, reason)."""
    arm = prov.get("arm")
    if arm == "nodedup":
        return mid, None     # the caller reuses the kept read set
    if arm != "base":
        return None, f"arm {arm!r}: no read set before duplicate removal"
    nmid = AV.measurement_id(prov["experiment"], prov["unit"], prov["layer"], "nodedup")
    ndir = mdir.parent / nmid
    if not (ndir / "provenance.json").is_file():
        return None, f"no nodedup measurement {nmid} beside {mdir}"
    meta = json.loads((ndir / "meta.json").read_text())
    if meta.get("parent_measurement_id") != mid:
        raise Refused(f"{ndir}/meta.json parent {meta.get('parent_measurement_id')!r} != {mid}")
    nprov = load_prov(ndir, nmid)
    if nprov.get("lines_per_fragment") != prov.get("lines_per_fragment"):
        raise Refused(f"{nmid}: lines_per_fragment differs from {mid}")
    return nmid, verify_readset(nprov)


# --------------------------------------------------------------------------------------------
# the two measurements

def pbc_command(files: list, lpu: int, work, sort_mem: str = "4G", mito: str = MITO) -> str:
    if lpu not in KEY_AWK:
        raise Refused(f"lines_per_fragment {lpu}")
    cat = "gzip -dc " + " ".join(shlex.quote(str(f)) for f in files)
    paste = " | paste - -" if lpu == 2 else ""
    return (f"{cat}{paste} | {KEY_AWK[lpu] % mito} | LC_ALL=C sort -S {sort_mem} "
            f"-T {shlex.quote(str(work))} | uniq -c | {PBC_AWK}")


def pbc(files: list, lpu: int, work, cost=None, sort_mem: str = "4G") -> dict:
    """NRF, PBC1, PBC2 (strings, the pipeline's %f) and the four counts."""
    total = sum(int(f["lines"]) for f in files)
    if total % lpu:
        raise Refused(f"{total} lines is not a multiple of lines_per_fragment {lpu}")
    cmd = pbc_command([f["path"] for f in files], lpu, work, sort_mem)
    t0 = time.time()
    p = subprocess.run(["/bin/bash", "-o", "pipefail", "-c", cmd], capture_output=True,
                       text=True)
    if p.returncode:
        raise RuntimeError(f"pbc step failed ({p.returncode}): {p.stderr[-2000:]}")
    if cost is not None:
        cost.steps.append({"step": "pbc", "kind": "shell", "wall_s": round(time.time() - t0, 2)})
    mt, m0, m1, m2, nrf, pbc1, pbc2 = p.stdout.split()
    return {"mt": int(mt), "m0": int(m0), "m1": int(m1), "m2": int(m2), "nrf": nrf,
            "pbc1": pbc1, "pbc2": pbc2 if int(m2) > 0 else "", "command": cmd}


_SIF_CACHE = {}


def xcor_command(ta, pipeline: str, paired_end: bool, nth: int, out_dir) -> str:
    return (f"python3 $(which encode_task_xcor.py) {ta}{' --paired-end' if paired_end else ''} "
            f"--mito-chr-name {MITO} {XCOR_ARGS[pipeline]} --nth {nth} --out-dir {out_dir}")


def run_xcor(files: list, pipeline: str, paired_end: bool, nth: int, work: Path, cost) -> dict:
    """NSC, RSC and fragment length from the pipeline's xcor task in the pipeline image."""
    img = XCOR_IMAGE[pipeline]
    if img not in _SIF_CACHE:
        _SIF_CACHE[img] = AV.sif_info(bam_arm.SIF[img], img)
    sif = _SIF_CACHE[img]
    ta = work / "kept.tagAlign.gz"
    cost.shell("cat_kept", "cat " + " ".join(shlex.quote(f["path"]) for f in files)
               + f" > {shlex.quote(str(ta))}")
    od = work / "xcor"
    od.mkdir(parents=True, exist_ok=True)
    inner = xcor_command(ta, pipeline, paired_end, nth, od)
    cost.shell("xcor", bam_arm.apptainer(inner, sif["path"], str(work), str(od)))
    qc = sorted(od.glob("*.cc.qc"))
    if len(qc) != 1:
        raise RuntimeError(f"xcor: expected one *.cc.qc in {od}, got {qc}")
    line = qc[0].read_text().split("\n")[0]
    out = parse_cc_qc(line)
    ta.unlink()
    return {**out, "command": inner, "cc_qc": line, "image": img, "sif_sha256": sif["sha256"]}


def parse_cc_qc(line: str) -> dict:
    """spp's .cc.qc (after the pipeline's sed): Filename numReads estFragLen corr_estFragLen
    PhantomPeak corr_phantomPeak argmin_corr min_corr NSC RSC QualityTag."""
    f = line.rstrip("\n").split("\t")
    if len(f) < 10:
        raise RuntimeError(f"cc.qc line has {len(f)} fields: {line!r}")
    return {"frag_len": str(int(float(f[2]))), "nsc": f[8], "rsc": f[9]}


def measure(row: dict, work: Path, nth: int = 4, xcor_fn=None, sort_mem: str = "4G") -> dict:
    xcor_fn = xcor_fn or run_xcor
    mid = row["measurement_id"]
    cost = AV.Cost()
    rec = {"schema": 1, "measurement_id": mid, "status": "ok", "reason": "",
           "values": {c: "" for c in QC_COLUMNS}, "approximations": APPROXIMATIONS,
           "slurm_job_ids": bam_arm._job_ids(), "code": {"git_sha": AV.git_sha()}}
    try:
        mdir = measurement_dir(row)
        prov = load_prov(mdir, mid)
        rec["counts_sha256"] = check_counts(row, mdir)
        pipe = PIPELINE_KEY.get((prov.get("pipeline") or {}).get("name"))
        if pipe is None:
            raise NoReadset(f"pipeline {prov.get('pipeline')!r} has no xcor route here")
        lpu = int(prov["lines_per_fragment"])
        kept = verify_readset(prov)
        rec["kept_readset"] = {"files": kept, "lines_per_fragment": lpu}
        src, pre = predup_readset(mid, mdir, prov)
        work.mkdir(parents=True, exist_ok=True)
        x = xcor_fn(kept, pipe, lpu == 2, nth, work, cost)
        rec["xcor"] = x
        rec["values"].update({c: x[c] for c in ("nsc", "rsc", "frag_len")})
        if src is None:
            rec["pbc"], rec["pbc_reason"] = None, pre
        else:
            files = kept if pre is None else pre
            b = pbc(files, lpu, work, cost, sort_mem)
            rec["pbc"] = {"source_measurement_id": src, "files": files, **b}
            rec["values"].update({c: b[c] for c in ("nrf", "pbc1", "pbc2")})
    except NoReadset as e:
        rec["status"], rec["reason"] = "no_readset", str(e)
    except Refused as e:
        rec["status"], rec["reason"] = "refused", str(e)
        rec["values"] = {c: "" for c in QC_COLUMNS}
    rec["cost"] = cost.steps
    rec["created_utc"] = AV.utc()
    return rec


def write_json(path: Path, rec: dict) -> None:
    tmp = path.with_name(f".{path.name}.partial")
    tmp.write_text(json.dumps(rec, indent=1) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------
# commands

def cmd_coverage(a) -> int:
    _, rows = read_tsv(Path(a.tables) / "measurements.tsv")
    rows = with_qc_table(rows, a.qc_table)
    for c, (k, n) in coverage(rows).items():
        print(f"{c}\t{(k / n if n else 0.0):.4f}\t{k}/{n}")
    if a.missing_rows:
        print(f"missing_rows\t{rows_spec(missing_rows(rows))}")
    return 0


def cmd_run(a) -> int:
    _, rows = read_tsv(Path(a.tables) / "measurements.tsv")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bad = 0
    for i in parse_rows(a.rows):
        if i >= len(rows):
            print(f"[qc_fill] row {i}: past the end ({len(rows)} rows)", flush=True)
            bad += 1
            continue
        row = rows[i]
        f = out / f"{row['measurement_id']}.json"
        if f.is_file() and json.loads(f.read_text()).get("status") in ("ok", "no_readset"):
            print(f"[qc_fill] row {i} {row['measurement_id']}: done, skipped", flush=True)
            continue
        work = Path(a.work) / f"row{i}"
        try:
            rec = measure(row, work, a.nth, sort_mem=a.sort_mem)
        except (Exception, SystemExit) as e:   # a failed step (Cost.shell exits): rerun by index
            rec = {"schema": 1, "measurement_id": row["measurement_id"], "status": "failed",
                   "reason": f"{type(e).__name__}: {e}", "values": {c: "" for c in QC_COLUMNS},
                   "slurm_job_ids": bam_arm._job_ids(), "created_utc": AV.utc()}
        finally:
            shutil.rmtree(work, ignore_errors=True)
        rec["row_index"] = i
        write_json(f, rec)
        print(f"[qc_fill] row {i} {row['measurement_id']}: {rec['status']} "
              f"{rec['reason'] or rec['values']}", flush=True)
        bad += rec["status"] in ("refused", "failed")
    return 1 if bad else 0


def cmd_collect(a) -> int:
    _, rows = read_tsv(Path(a.tables) / "measurements.tsv")
    qc_dir = Path(a.qc_dir)
    n = {"rows": len(rows), "filled": 0, "no_record": 0, "stale": 0, "not_ok": 0}
    out_rows = []
    for r in rows:
        mid = r["measurement_id"]
        vals = {c: "" for c in QC_COLUMNS}
        f = qc_dir / f"{mid}.json"
        if not f.is_file():
            n["no_record"] += 1
        else:
            rec = json.loads(f.read_text())
            if rec.get("measurement_id") != mid or rec.get("status") != "ok":
                n["not_ok"] += 1
            elif r.get("sha256") and rec.get("counts_sha256") != r["sha256"]:
                n["stale"] += 1
            else:
                vals.update({c: rec["values"].get(c, "") for c in QC_COLUMNS})
                vals["frip"] = ""
                n["filled"] += any(vals[c] for c in INFORMATIVE)
        out_rows.append({"measurement_id": mid, **vals})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.partial")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["measurement_id", *QC_COLUMNS], delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    os.replace(tmp, out)
    print(f"[qc_fill] collect {out}: " + ", ".join(f"{k} {v}" for k, v in n.items())
          + "; pr1/pr2 halves use their parent's row (approximation); frip empty", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--tables", required=True)
    c.add_argument("--qc-table")
    c.add_argument("--missing-rows", action="store_true",
                   help="also print the row spec of training/validation rows lacking a column")
    r = sub.add_parser("run")
    r.add_argument("--tables", required=True)
    r.add_argument("--rows", required=True)
    r.add_argument("--work", required=True)
    r.add_argument("--out", default=OUT_DEFAULT)
    r.add_argument("--nth", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    r.add_argument("--sort-mem", default="4G")
    k = sub.add_parser("collect")
    k.add_argument("--tables", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--qc-dir", default=OUT_DEFAULT)
    a = ap.parse_args(argv)
    return {"coverage": cmd_coverage, "run": cmd_run, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
