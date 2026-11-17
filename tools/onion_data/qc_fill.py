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
    python qc_fill.py collect --tables DIR --out $O/tables/eic_v1_qc.tsv [--qc-dir DIR ...]
        [--fraglen-out $O/tables/eic_v1_fraglen.tsv]
    python qc_fill.py run --half pr1 ...        # A4: measure the pr1 half only (<id>.pr1.json)
    python qc_fill.py collect --half pr1 ...    # A4: the pr1 side table (dataset qc_table_pr1)

Close-out item A4 (half mode). A scored correction source is read from its `pr1` half, and its
`pr2` half may share reads with the target. `--half pr1` measures xcor on the pr1 files only, and
NRF/PBC on the pr1 files of the nodedup sibling (a `nodedup` row: its own pr1 files), so no QC value
of the source comes from reads outside the half. The record is `<out>/<id>.pr1.json` with
`"half": "pr1"`. `collect --half pr1` lists only the rows with an ok pr1 record and never falls back
to `measurements.tsv` (those are full-track values). `--qc-dir` may be given more than once (the
2026-11-13 records on /scratch and a later pass elsewhere); a row takes its record from the LAST
directory that holds a usable one.

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


def verify_readset(prov: dict, halves=None) -> list:
    """Every half file of a measurement (or of the listed `halves` only): md5 against its
    READSET.json, lines against provenance.json. Returns [{half, path, md5, lines}] in provenance
    order."""
    out = []
    for h in (halves or AV.HALVES):
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


def predup_readset(mid: str, mdir: Path, prov: dict, halves=None) -> tuple:
    """(source measurement id, verified files) of the reads before duplicate removal, or
    (None, reason). `halves` limits the files to those halves (A4 half mode)."""
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
    return nmid, verify_readset(nprov, halves)


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


def measure(row: dict, work: Path, nth: int = 4, xcor_fn=None, sort_mem: str = "4G",
            half: str | None = None) -> dict:
    """One measurement's QC record. `half` (A4): measure that pseudoreplicate half only."""
    if half is not None and half not in AV.HALVES:
        raise ValueError(f"half {half!r} not in {AV.HALVES}")
    halves = (half,) if half else None
    xcor_fn = xcor_fn or run_xcor
    mid = row["measurement_id"]
    cost = AV.Cost()
    rec = {"schema": 1, "measurement_id": mid, "half": half or "", "status": "ok", "reason": "",
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
        kept = verify_readset(prov, halves)
        rec["kept_readset"] = {"files": kept, "lines_per_fragment": lpu}
        src, pre = predup_readset(mid, mdir, prov, halves)
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


def record_name(mid: str, half: str | None = None) -> str:
    """`<id>.json` (whole read set) or `<id>.<half>.json` (A4 half mode)."""
    return f"{mid}.{half}.json" if half else f"{mid}.json"


def pick_record(mid: str, sha: str, qc_dirs: list, half: str | None = None) -> tuple:
    """(kind, record, path) of measurement `mid` over `qc_dirs`: kind is "ok" (the LAST directory
    with a usable record wins), else the kind of the last record seen ("not_ok", "stale") or
    "no_record". A usable record names `mid`, has status ok, the row's counts sha256 (when the row
    has one) and the requested half."""
    kind, got = "no_record", (None, None)
    for d in qc_dirs:
        f = Path(d) / record_name(mid, half)
        if not f.is_file():
            continue
        rec = json.loads(f.read_text())
        if (rec.get("measurement_id") != mid or rec.get("status") != "ok"
                or (rec.get("half") or "") != (half or "")):
            k = "not_ok"
        elif sha and rec.get("counts_sha256") != sha:
            k = "stale"
        else:
            k = "ok"
        if k == "ok":
            kind, got = "ok", (rec, f)
        elif kind != "ok":
            kind = k
    return (kind, *got)


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
        f = out / record_name(row["measurement_id"], a.half)
        if f.is_file() and json.loads(f.read_text()).get("status") in ("ok", "no_readset"):
            print(f"[qc_fill] row {i} {row['measurement_id']}: done, skipped", flush=True)
            continue
        work = Path(a.work) / f"row{i}"
        try:
            rec = measure(row, work, a.nth, sort_mem=a.sort_mem, half=a.half)
        except (Exception, SystemExit) as e:   # a failed step (Cost.shell exits): rerun by index
            rec = {"schema": 1, "measurement_id": row["measurement_id"], "half": a.half or "",
                   "status": "failed",
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


#: the fields of spp's .cc.qc line (parse_cc_qc), as columns of the fragment-length side table
CC_QC_FIELDS = ("xcor_num_reads", "frag_len_xcor", "corr_est_frag_len", "phantom_peak",
                "corr_phantom_peak", "argmin_corr", "min_corr", "nsc", "rsc", "quality_tag")
FRAGLEN_COLUMNS = ("measurement_id", "frag_len_estimated", "estimate_kind", "frag_len_frozen",
                   "frag_len_signal_provenance", "frag_len_replicate_mean", "frag_len_xcor",
                   "nsc", "rsc", "quality_tag", "xcor_num_reads", "corr_est_frag_len",
                   "phantom_peak", "corr_phantom_peak", "argmin_corr", "min_corr", "run_type",
                   "in_eic", "assay_class", "unit", "source")
#: the order in which a fragment-length estimate is chosen (see fraglen_rows)
ESTIMATE_KINDS = ("signal_provenance", "replicate_mean", "xcor")
_CHIP = ("broad_histone", "narrow_histone")


def signal_provenance_fraglen(row: dict) -> tuple:
    """(fraglen, source) that the pipeline used for this measurement's own `signal.h5`, from its
    `provenance.json` (`signal.signal.h5.spec.fraglen`, align_views.chip_fraglen), else ("", "")."""
    try:
        f = measurement_dir(row) / "provenance.json"
    except NoReadset:
        return "", ""
    if not f.is_file():
        return "", ""
    try:
        prov = json.loads(f.read_text())
    except ValueError:
        return "", ""
    if prov.get("measurement_id") != row["measurement_id"]:
        return "", ""
    spec = (((prov.get("signal") or {}).get("signal.h5") or {}).get("spec") or {})
    v = spec.get("fraglen")
    return (str(int(v)), f"{f} signal.h5 spec ({spec.get('fraglen_source', '')})") \
        if v not in (None, "") else ("", "")


def replicate_mean_fraglen(row: dict, rows: list) -> tuple:
    """(rounded mean of the replicates' frozen frag_len, ids) for a pooled row: chip.wdl's rule
    for the pooled track (align_views.chip_fraglen). Replicates: unit br<i>, same experiment,
    layer and arm; ("", "") when none has a frag_len."""
    if row.get("unit") != "pooled":
        return "", ""
    key = (row.get("experiment"), row.get("layer"), row.get("arm_knob"), row.get("arm_value"))
    reps = [r for r in rows if (r.get("experiment"), r.get("layer"), r.get("arm_knob"),
                                r.get("arm_value")) == key
            and (r.get("unit") or "").startswith("br") and (r.get("frag_len") or "").strip()]
    if not reps:
        return "", ""
    vals = [float(r["frag_len"]) for r in reps]
    return str(int(round(sum(vals) / len(vals)))), ";".join(r["measurement_id"] for r in reps)


def fraglen_rows(rows: list, xcor: dict) -> list:
    """The fragment-length side table: one row per ChIP-seq measurement with at least one
    estimate. `frag_len_estimated` is the first of ESTIMATE_KINDS that exists: the fragment length
    the pipeline used for the measurement's own signal track (provenance.json), the rounded mean of
    the replicates' frozen frag_len (pooled rows), the xcor estimate of the QC fill. `xcor`:
    measurement_id -> (ok record, record path)."""
    out = []
    for r in rows:
        if r.get("assay_class") not in _CHIP:
            continue
        mid = r["measurement_id"]
        prov, prov_src = signal_provenance_fraglen(r)
        rep, rep_src = replicate_mean_fraglen(r, rows)
        vals, x_src = {}, ""
        if mid in xcor:
            rec, f = xcor[mid]
            cc = (rec.get("xcor") or {}).get("cc_qc", "").rstrip("\n").split("\t")
            vals = dict(zip(CC_QC_FIELDS, cc[1:11])) if len(cc) >= 11 else {}
            vals["frag_len_xcor"] = rec["values"].get("frag_len", "")
            x_src = f"{f} job {';'.join(rec.get('slurm_job_ids') or []) or 'none'}"
        got = {"signal_provenance": (prov, prov_src), "replicate_mean": (rep, f"replicates {rep_src}"),
               "xcor": (vals.get("frag_len_xcor", ""), x_src)}
        kind = next((k for k in ESTIMATE_KINDS if got[k][0]), None)
        if kind is None:
            continue
        out.append({**{c: "" for c in FRAGLEN_COLUMNS}, **vals, "measurement_id": mid,
                    "frag_len_estimated": got[kind][0], "estimate_kind": kind,
                    "frag_len_frozen": r.get("frag_len", ""), "frag_len_signal_provenance": prov,
                    "frag_len_replicate_mean": rep, "run_type": r.get("run_type", ""),
                    "in_eic": r.get("in_eic", ""), "assay_class": r.get("assay_class", ""),
                    "unit": r.get("unit", ""), "source": got[kind][1]})
    return out


def _write_tsv(out: Path, cols, rows) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.partial")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)


def cmd_collect(a) -> int:
    """The QC side table, one row per measurement. A cell comes from the measurement's ok record
    when the record has a value there; else it keeps the `measurements.tsv` value, so a value the
    build filled is never blanked (the side table overrides every listed row: dataset.py
    `_apply_qc_table`). `--fraglen-out` also writes the fragment-length side table
    (`fraglen_rows`): one row per ChIP-seq measurement with an estimate, its kind and source.
    `--half pr1` (A4): one row per measurement with an ok pr1 record only, values from the record
    alone (dataset `qc_table_pr1`)."""
    _, rows = read_tsv(Path(a.tables) / "measurements.tsv")
    qc_dirs = a.qc_dir or [OUT_DEFAULT]
    if a.half and a.fraglen_out:
        raise SystemExit("collect: --fraglen-out reads whole-read-set records; not with --half")
    n = {"rows": len(rows), "filled": 0, "no_record": 0, "stale": 0, "not_ok": 0,
         "kept_build_cells": 0}
    out_rows, xcor = [], {}
    for r in rows:
        mid = r["measurement_id"]
        vals = {c: "" for c in QC_COLUMNS} if a.half else {c: r.get(c, "") or "" for c in QC_COLUMNS}
        kind, rec, f = pick_record(mid, r.get("sha256") or "", qc_dirs, a.half)
        if kind != "ok":
            n[kind] += 1
            if a.half:
                continue
        else:
            for c in QC_COLUMNS:
                v = rec["values"].get(c, "") or ""
                if v:
                    vals[c] = v
                elif vals[c]:
                    n["kept_build_cells"] += 1
            vals["frip"] = ""
            n["filled"] += any(vals[c] for c in INFORMATIVE)
            if rec["values"].get("frag_len"):
                xcor[mid] = (rec, f)
        out_rows.append({"measurement_id": mid, **vals})
    out = Path(a.out)
    _write_tsv(out, ["measurement_id", *QC_COLUMNS], out_rows)
    msg = (f"[qc_fill] collect {out}{' half ' + a.half if a.half else ''}: "
           + ", ".join(f"{k} {v}" for k, v in n.items())
           + ("; only rows with an ok half record are listed (no fallback); frip empty"
              if a.half else
              "; rows without a usable record keep the measurements.tsv values; pr1/pr2 halves "
              "use their parent's row (approximation); frip empty"))
    if a.fraglen_out:
        fl_rows = fraglen_rows(rows, xcor)
        _write_tsv(Path(a.fraglen_out), FRAGLEN_COLUMNS, fl_rows)
        kinds = {k: sum(r["estimate_kind"] == k for r in fl_rows) for k in ESTIMATE_KINDS}
        msg += (f"; fragment lengths {len(fl_rows)} ChIP-seq rows "
                + ", ".join(f"{k} {v}" for k, v in kinds.items()) + f" -> {a.fraglen_out}")
    print(msg, flush=True)
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
    r.add_argument("--half", choices=("pr1", "pr2"), default=None,
                   help="A4: measure this pseudoreplicate half only (record <id>.<half>.json)")
    k = sub.add_parser("collect")
    k.add_argument("--tables", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--qc-dir", action="append", default=None,
                   help=f"record directory; repeatable, the last usable record wins "
                        f"(default {OUT_DEFAULT})")
    k.add_argument("--half", choices=("pr1", "pr2"), default=None,
                   help="A4: collect the half records into the half side table")
    k.add_argument("--fraglen-out", default=None,
                   help="also write the fragment-length side table (signal_from_counts "
                        "load_fraglen_table reads it)")
    a = ap.parse_args(argv)
    return {"coverage": cmd_coverage, "run": cmd_run, "collect": cmd_collect}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
