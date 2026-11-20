"""D7 (t168): reprocess one EIC experiment from FASTQ with the standard ENCODE-DCC pipeline of its
assay, in the pipeline's own alignment-only mode, and keep the alignment products.

Pipelines (decision D5; repositories checked out on Nibi at these tags, images built from the
WDL's own `encodedcc/<image>:<tag>`):

    histone ChIP-seq, control   chip-seq-pipeline2 v2.2.2   `chip.align_only = true`
    ATAC-seq                    atac-seq-pipeline  v2.2.3   `atac.align_only = true`

`align_only` is the pipelines' own switch (chip.wdl:159, atac.wdl:142): alignment, filtering,
tagAlign and the QC tasks run; peak calling, pseudoreplication and IDR do not. Everything else is
the WDL default. Two settings are set on purpose: no read subsampling (the earlier EIC_REPRO runs
used 30 M / 50 M; the uniform base keeps every read) and, for ATAC, `atac.auto_detect_adapter =
true` (ENCODE's own example input JSON, `example_input_json/ENCSR356KRQ_subsampled.json`).
DNase-seq is chunk D7d's (dnase-seq-pipeline v3.0.0, never ATAC in DNase mode); it is refused here.

One build-list row is one Caper run, `caper run` on the Local backend inside one SLURM job:
Cromwell's execution tree lives in `--tmp` (the job's `$SLURM_TMPDIR`) and dies with the job; only
the harvested keepers reach `--work`:

    <work>/<experiment>/<arm>/            unit `pooled`
    <work>/<experiment>/<unit>.<arm>/     every other unit
        input.json  arm.json  metadata.json  qc.json (if the run made one)
        rep<i>.unfiltered.bam  rep<i>.filtered.bam  rep<i>.filtered.bam.bai  rep<i>.tagAlign.gz
        rep<i>.xcor.qc (ChIP only)  HARVEST.tsv  ALIGN_DONE (written last)

Units (pinned interface B): `pooled` (every biological replicate, one pipeline replicate each; its
rep<i> products are the `br<i>` views), `br<i>` (that biological replicate alone), `br<i>tr<j>` (one
library), `br<i>tr<j>ln<k>` (the k-th sequencing run of that library: its k-th R1 FASTQ by
accession, with its mate). Arms: `base`; `se` (paired-end experiments only: mate 1 only, run as
single-end); `crop36`, `crop50` (only when every FASTQ is longer than the crop). ChIP crops with the
pipeline's own `chip.crop_length` (tolerance 2). atac.wdl v2.2.3 has no crop input, so an ATAC crop
runs the same Trimmomatic step (`encode_task_trimmomatic.py`, CROP and MINLEN = crop - 2) from the
chip-seq-pipeline2 v2.2.2 image on each FASTQ before the pipeline.

FASTQs are staged under `<fastq root>/<experiment>/` from the ENCODE portal, md5-checked against
the portal's `md5sum`, mode 0444, and deleted once every build-list row of that experiment has a
verified ALIGN_DONE.

Deliberately stdlib only: it runs under the Caper venv (python 3.11) on Nibi.

    fastq_uniform.py stage --experiment ENCSR... [--fastq-root DIR]
    fastq_uniform.py prepare-refs [--src REFCACHE] [--dest DIR]
    fastq_uniform.py input-json --build-list TSV --row I [--files-json F]
    fastq_uniform.py run --build-list TSV --row I --work DIR --tmp DIR [--cpus N]
    fastq_uniform.py verify --work DIR --experiment ENCSR... [--md5]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

EIC = "/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline"  # read only
REFCACHE = "/scratch/mforooz/EIC_REPRO/003/refcache"        # read only
REFS = "/project/def-maxwl/mforooz/ONION/pipelines/hg38"
FASTQ_ROOT = "/scratch/mforooz/onion/fastq"
WORK = "/scratch/mforooz/onion/work"
PORTAL = "https://www.encodeproject.org"
FASTQ_URL = PORTAL + "/files/{0}/@@download/{0}.fastq.gz"

#: repository, release tag, commit checked out under EIC_REPRO, image and its sha256 (Nibi,
#: 2026-11-10, `sha256sum sif/*.sif`; `git rev-parse HEAD`).
PIPELINES = {
    "chip": {
        "repo": "https://github.com/ENCODE-DCC/chip-seq-pipeline2",
        "name": "chip-seq-pipeline2", "tag": "v2.2.2",
        "commit": "26eeda81a0540dc793fc69b0c390d232ca7ca50a",
        "wdl": "chip-seq-pipeline2/chip.wdl",
        "sif": "sif/chip-seq-pipeline_v2.2.2.sif",
        "image": "encodedcc/chip-seq-pipeline:v2.2.2",
        "sha256": "20b7f35c39374d53ecf868e20303d26c2de641751f4b0c347c6785d15915ae97",
    },
    "atac": {
        "repo": "https://github.com/ENCODE-DCC/atac-seq-pipeline",
        "name": "atac-seq-pipeline", "tag": "v2.2.3",
        "commit": "47ba8dff9c332e24b48e767303e9fcac98589cf2",
        "wdl": "atac-seq-pipeline/atac.wdl",
        "sif": "sif/atac-seq-pipeline_v2.2.3.sif",
        "image": "encodedcc/atac-seq-pipeline:v2.2.3",
        "sha256": "3086dbe7a42ae7c8323aaf0f07a293dd858a70de2e5c57331d5ace97c5733637",
    },
}

#: the genome TSV the EIC runs used, as reconstructed by t112 C7
#: (`/scratch/mforooz/t112_cf/fastqarms/genome/{hg38.local.tsv,PROVENANCE.md}`): key -> path under
#: the refcache (`<md5 of the source URL>/<file>`), or a scalar.
GENOME_SCALARS = (("genome_name", "hg38"), ("gensz", "hs"), ("mito_chr_name", "chrM"),
                  ("regex_bfilt_peak_chr_name", r"chr[\dXY]+"))
GENOME_FILES = (
    ("ref_fa", "caf534ed3cf684406e731d19be272b4a/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz"),
    ("bwa_idx_tar", "8c692fba4640609720272154ab0faa30/ENCFF643CGH.tar.gz"),
    ("bowtie2_idx_tar", "3ff4ac4c3f59d096b1a3842a182072ae/ENCFF110MCL.tar.gz"),
    ("chrsz", "c52f52c7bfa357f55a39b1de7e4d0b0c/GRCh38_EBV.chrom.sizes.tsv"),
    ("blacklist", "f183dcba5d34f959d8b55ed438ee2e22/ENCFF356LFX.bed.gz"),
    ("ref_mito_fa", "f43b63a83784d3ec8055f1a22168ed89/GRCh38_no_alt_analysis_set_GCA_000001405.15_mito_only.fasta.gz"),
    ("bowtie2_mito_idx_tar", "df5193e07055d13c48be59bacd0f56b8/GRCh38_no_alt_analysis_set_GCA_000001405.15_mito_only_bowtie2_index.tar.gz"),
    ("dnase", "805e179275a9c0fb7a37def40c4312d1/ENCFF304XEX.bed.gz"),
    ("enh", "1d3aa436b05f16a509edb94789c061d3/ENCFF212UAV.bed.gz"),
    ("prom", "0cbd2c602ddad252bc39729fc8a29286/ENCFF140XLU.bed.gz"),
    ("tss", "0fa7d04b32e66fa02fb2c1ae39e41447/ENCFF493CCB.bed.gz"),
    ("reg2map", "3b39284516e676ea52238f0636c0bbbf/hg38_dnase_avg_fseq_signal_formatted.txt.gz"),
    ("reg2map_bed", "c73f434c3fa4f3f54bc2ecad09c065c2/hg38_celltype_compare_subsample.bed.gz"),
    ("roadmap_meta", "a9745b33b4ffdd83d7d2c5a7d3c8036a/hg38_dnase_avg_fseq_signal_metadata.txt"),
)

ARMS = {"base": ("", ""), "se": ("se", "1"), "crop36": ("crop", "36"), "crop50": ("crop", "50")}
CROP_TOL = 2           # chip.crop_length_tol default
MAPQ_THRESH = 30       # chip.wdl:179, atac.wdl:184 default
UNIT_RE = re.compile(r"^(pooled|br(\d+)(?:tr(\d+)(?:ln(\d+))?)?)$")
BUILD_LIST_REQUIRED = ("row", "experiment", "assay", "assay_class", "unit", "arm")
#: a run dir under work/<experiment>/: `<arm>` (unit pooled) or `<unit>.<arm>`. align_views'
#: read-set dirs (`<unit>.<layer>`) do not match.
RUN_DIR_RE = re.compile(r"^(?:br\d+(?:tr\d+(?:ln\d+)?)?\.)?(base|se|crop36|crop50)$")
#: a run dir that holds this file instead of ALIGN_DONE is a row that does not apply (kept, verified)
NOT_APPLICABLE = "NOT_APPLICABLE"
#: ENCODE4's own processing of an experiment whose biological replicate mixes single-end and
#: paired-end runs (portal, 2026-11-11: chip-seq-alignment-step v2 `derived_from` of ENCSR339FOO,
#: ENCSR082NQB, ENCSR389WWC, ENCSR447ZGY, ENCSR720GSE, ENCSR955IXZ, ENCSR976JHS lists one mate
#: per pair plus the single-end runs, for every replicate of the experiment).
MIXED_RUN_TYPE_NOTE = ("mixed single-end and paired-end runs: every replicate aligned as "
                       "single-end from mate 1 of each pair plus the single-end runs, as ENCODE4's "
                       "chip-seq-alignment-step v2 did for these experiments")

#: staging (Nibi -> ENCODE S3 is ~0.14 MB/s per TCP flow, linear in flows up to 64 at least:
#: NETPROBE 2026-11-11). Byte ranges per file and files at once; both are CLI parameters.
RANGES = 64
PARALLEL_FILES = 4
USER_AGENT = "Mozilla/5.0 (generic-client)"


class NotApplicable(Exception):
    """A build-list row whose arm does not exist for this experiment (written as NOT_APPLICABLE)."""


def log(msg):
    print("%s  %s" % (time.strftime("%F %T"), msg), flush=True)


def md5_file(path, bufsize=1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path, bufsize=1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp%d" % os.getpid())
    tmp.write_text(text)
    os.replace(tmp, path)


# --- build list and pipeline choice ------------------------------------------------------------

def read_build_list(tsv) -> list[dict]:
    lines = [ln for ln in Path(tsv).read_text().split("\n") if ln.strip()]
    header = lines[0].split("\t")
    missing = [c for c in BUILD_LIST_REQUIRED if c not in header]
    if missing:
        raise SystemExit(f"{tsv}: build list lacks columns {missing}")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def get_row(rows, i) -> dict:
    hits = [r for r in rows if r["row"] == str(i)]
    if len(hits) != 1:
        raise SystemExit(f"build list: {len(hits)} rows with row = {i}, expected 1")
    return hits[0]


def pipeline_key(row) -> tuple[str, str]:
    """(pipeline key, pipeline_type) for a build-list row."""
    cls, assay = row["assay_class"], row["assay"]
    if cls in ("broad_histone", "narrow_histone"):
        key, ptype = "chip", "histone"
    elif cls == "control" or assay == "chipseq-control":
        key, ptype = "chip", "control"
    elif cls == "accessibility_atac" or assay == "ATAC-seq":
        key, ptype = "atac", "atac"
    elif cls == "accessibility_dnase":
        raise SystemExit(f"row {row['row']}: DNase-seq belongs to the D7d kit (dnase-seq-pipeline "
                         "v3.0.0); this kit never runs the ATAC pipeline in DNase mode")
    else:
        raise SystemExit(f"row {row['row']}: assay_class {cls!r} has no FASTQ pipeline here")
    named = (row.get("pipeline") or "").strip()
    if named and key not in named:
        raise SystemExit(f"row {row['row']}: build list names pipeline {named!r}, assay_class "
                         f"{cls!r} needs {PIPELINES[key]['name']}")
    return key, ptype


def arm_dir(work, row) -> Path:
    unit, arm = row["unit"], row["arm"]
    return Path(work) / row["experiment"] / (arm if unit == "pooled" else f"{unit}.{arm}")


# --- portal file table -------------------------------------------------------------------------

def _get_json(url, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - network; retried, then raised
            if k == tries - 1:
                raise
            log(f"portal GET failed ({e}); retry {k + 1}")
            time.sleep(5 * (k + 1))


def portal_files(experiment) -> dict:
    """The experiment's FASTQ table from the portal: released files only (archived files are used
    only when the experiment has no released FASTQ, and that is recorded)."""
    url = (f"{PORTAL}/search/?type=File&dataset=/experiments/{experiment}/&file_format=fastq"
           "&frame=object&limit=all&format=json")
    graph = _get_json(url)["@graph"]
    rel = [f for f in graph if f.get("status") == "released"]
    note = ""
    if not rel:
        rel = [f for f in graph if f.get("status") == "archived"]
        note = "no released FASTQ; archived FASTQs used"
    files = []
    for f in rel:
        fc = (f.get("flowcell_details") or [{}])[0]
        tr = (f.get("technical_replicates") or [""])[0]
        files.append({
            "accession": f["accession"], "md5": f["md5sum"], "bytes": f["file_size"],
            "run_type": f.get("run_type", ""), "read_length": f.get("read_length"),
            "paired_end": str(f.get("paired_end") or ""),
            "paired_with": (f.get("paired_with") or "").strip("/").split("/")[-1],
            "bio_rep": (f.get("biological_replicates") or [None])[0],
            "tech_rep": int(tr.split("_")[1]) if "_" in tr else None,
            "platform": (f.get("platform") or "").strip("/").split("/")[-1],
            "flowcell": fc.get("flowcell", ""), "lane": fc.get("lane", ""),
            "status": f.get("status"), "read_count": f.get("read_count"),
            "url": (f.get("cloud_metadata") or {}).get("url") or FASTQ_URL.format(f["accession"]),
        })
    files.sort(key=lambda d: d["accession"])
    return {"experiment": experiment, "files": files, "note": note,
            "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def portal_file(accession) -> dict:
    """One file's record (a native redacted BAM): accession, md5, bytes, bio_rep, url, status."""
    f = _get_json(f"{PORTAL}/files/{accession}/?format=json&frame=object")
    ext = f.get("file_format", "bam")
    return {"accession": f["accession"], "md5": f["md5sum"], "bytes": f["file_size"],
            "bio_rep": (f.get("biological_replicates") or [None])[0],
            "output_type": f.get("output_type", ""), "status": f.get("status", ""),
            "assembly": f.get("assembly", ""), "file_format": ext,
            "mapped_read_length": f.get("mapped_read_length"),
            "mapped_run_type": f.get("mapped_run_type", ""),
            "analysis_step_version": f.get("analysis_step_version", ""),
            "url": ((f.get("cloud_metadata") or {}).get("url")
                    or f"{PORTAL}/files/{f['accession']}/@@download/{f['accession']}.{ext}")}


# --- unit, arm and input JSON ------------------------------------------------------------------

def unit_reps(files_json: dict, unit: str, restrict=None) -> list[dict]:
    """Pipeline replicates for a unit: [{rep, bio_rep, tech_reps, r1: [acc], r2: [acc]}].

    R1 lists are sorted by accession and R2[i] is R1[i]'s mate. `restrict` (the build list's
    `fastq_files`) limits the file set to those accessions (either mate names the pair)."""
    m = UNIT_RE.match(unit)
    if not m:
        raise SystemExit(f"unit {unit!r} is not pooled, br<i>, br<i>tr<j> or br<i>tr<j>ln<k>")
    files = {f["accession"]: f for f in files_json["files"]}
    r1s = [f for f in files_json["files"] if f["paired_end"] in ("", "1")]
    if restrict:
        keep = set(restrict)
        r1s = [f for f in r1s if f["accession"] in keep or f["paired_with"] in keep]
    _, bio, tech, lane = m.groups()
    if bio is not None:
        r1s = [f for f in r1s if f["bio_rep"] == int(bio)]
    if tech is not None:
        r1s = [f for f in r1s if f["tech_rep"] == int(tech)]
    if lane is not None:
        r1s = sorted(r1s, key=lambda f: f["accession"])
        k = int(lane)
        if not 1 <= k <= len(r1s):
            raise SystemExit(f"unit {unit}: library has {len(r1s)} runs, no run {k}")
        r1s = [r1s[k - 1]]
    if not r1s:
        raise SystemExit(f"unit {unit}: no FASTQ left for {files_json['experiment']}")
    reps = []
    for i, b in enumerate(sorted({f["bio_rep"] for f in r1s}), start=1):
        mine = sorted((f for f in r1s if f["bio_rep"] == b), key=lambda f: f["accession"])
        r2 = []
        for f in mine:
            if f["paired_end"] == "1":
                mate = files.get(f["paired_with"])
                if mate is None:
                    raise SystemExit(f"{f['accession']}: mate {f['paired_with']!r} not in the table")
                r2.append(mate["accession"])
        rep = {"rep": i, "bio_rep": b, "tech_reps": sorted({f["tech_rep"] for f in mine}),
               "r1": [f["accession"] for f in mine], "r2": r2}
        if r2 and len(r2) != len(mine):
            # ENCODE4 aligned such a replicate as single-end from mate 1 (MIXED_RUN_TYPE_NOTE);
            # `arm_settings` turns every replicate of the unit single-end.
            rep["r2"], rep["mixed_run_types"] = [], True
            rep["mates_dropped"] = r2
        reps.append(rep)
    if any(r.get("mixed_run_types") for r in reps):
        for r in reps:
            if r["r2"]:
                r["mates_dropped"], r["r2"] = r["r2"], []
    return reps


def arm_settings(files_json, reps, arm) -> dict:
    """What an arm does to the run, and the measurement covariates it produces."""
    if arm not in ARMS:
        raise SystemExit(f"arm {arm!r} is not one of {sorted(ARMS)}")
    files = {f["accession"]: f for f in files_json["files"]}
    used = [files[a] for r in reps for a in r["r1"] + r["r2"]]
    lengths = sorted({int(f["read_length"]) for f in used if f["read_length"]})
    pe = [bool(r["r2"]) for r in reps]
    knob, value = ARMS[arm]
    crop = 0
    mixed = any(r.get("mixed_run_types") for r in reps)
    if arm == "se":
        if mixed:
            raise NotApplicable("arm se: the base of this experiment is already mate 1 only "
                                f"({MIXED_RUN_TYPE_NOTE}), so se would repeat the base")
        if not any(pe):
            raise SystemExit("arm se: the experiment is single-end already")
        pe = [False] * len(reps)
    if knob == "crop":
        crop = int(value)
        if not lengths or lengths[0] <= crop:
            raise SystemExit(f"arm {arm}: read lengths {lengths} are not all longer than {crop}")
    read_length = crop if crop else (lengths[0] if lengths else None)
    return {
        "arm_knob": knob, "arm_value": value, "paired_end_per_rep": pe, "crop_length": crop,
        "run_type": ("paired-ended" if all(pe) else "single-ended" if not any(pe) else "mixed"),
        "read_length": read_length, "read_lengths_source": lengths,
        "platforms": sorted({f["platform"] for f in used if f["platform"]}),
        "dedup": "on", "mapq_threshold": MAPQ_THRESH,
        "mixed_run_types": MIXED_RUN_TYPE_NOTE if mixed else "",
    }


def fastq_path(fastq_root, experiment, acc) -> str:
    return str(Path(fastq_root) / experiment / f"{acc}.fastq.gz")


def build_input_json(row, files_json, genome_tsv, fastq_root=FASTQ_ROOT, eic=EIC,
                     cropped=None) -> dict:
    """The Caper input JSON for one build-list row. `cropped` maps an accession to the path the
    run reads instead of the staged file: its pre-cropped FASTQ (ATAC crop arms) or its repaired
    copy (prepare_reads); see input_paths."""
    key, ptype = pipeline_key(row)
    spec, pfx = PIPELINES[key], key
    restrict = [a for a in (row.get("fastq_files") or "").replace(",", ";").split(";") if a]
    reps = unit_reps(files_json, row["unit"], restrict or None)
    arm = arm_settings(files_json, reps, row["arm"])
    exp = row["experiment"]

    def path(acc):
        if cropped is not None:
            return cropped[acc]
        return fastq_path(fastq_root, exp, acc)

    d = {f"{pfx}.pipeline_type": ptype, f"{pfx}.align_only": True,
         f"{pfx}.genome_tsv": str(genome_tsv)}
    pe = arm["paired_end_per_rep"]
    if len(set(pe)) == 1:
        d[f"{pfx}.paired_end"] = pe[0]
    else:
        d[f"{pfx}.paired_ends"] = pe
    for r, is_pe in zip(reps, pe):
        d[f"{pfx}.fastqs_rep{r['rep']}_R1"] = [path(a) for a in r["r1"]]
        if is_pe:
            d[f"{pfx}.fastqs_rep{r['rep']}_R2"] = [path(a) for a in r["r2"]]
    if key == "chip" and arm["crop_length"]:
        d["chip.crop_length"] = arm["crop_length"]
        d["chip.crop_length_tol"] = CROP_TOL
    if key == "atac":
        d["atac.auto_detect_adapter"] = True
    # the WDL's own `singularity` default is an https URL that each task resolves by itself;
    # Caper's --singularity does not override it (Nibi 2026-11-11, job 26816044: "FATAL: the
    # requested image was not found"), so the local image goes in the input JSON, as EIC_REPRO did.
    d[f"{pfx}.singularity"] = str(Path(eic) / spec["sif"])
    d[f"{pfx}.title"] = f"ONION {exp} {row['unit']} {row['arm']}"
    d[f"{pfx}.description"] = f"t152 onion uniform base, build-list row {row['row']}"
    return d


def arm_record(row, files_json, reps, arm, job_id=None) -> dict:
    key, ptype = pipeline_key(row)
    spec = PIPELINES[key]
    return {
        "experiment": row["experiment"], "unit": row["unit"], "arm": row["arm"],
        "layer": "uniform", "assay": row["assay"], "assay_class": row["assay_class"],
        "eic_track": row.get("eic_track", ""), "build_list_row": row["row"],
        "pipeline": spec["name"], "pipeline_version": spec["tag"], "pipeline_commit": spec["commit"],
        "pipeline_type": ptype, "image": spec["image"], "image_digest": "sha256:" + spec["sha256"],
        "reps": reps, "portal_note": files_json.get("note", ""), "slurm_job_id": job_id,
        **arm,
    }


# --- stage -------------------------------------------------------------------------------------

def byte_ranges(size: int, n: int) -> list[tuple[int, int]]:
    """n contiguous inclusive (start, end) ranges covering [0, size)."""
    n = max(1, min(n, size))
    step = -(-size // n)
    return [(s, min(s + step, size) - 1) for s in range(0, size, step)]


def backoff_s(k: int, base=5.0, cap=300.0) -> float:
    """Exponential backoff with jitter for retry k (0-based): base * 2^k, capped, +-25%."""
    import random
    return min(cap, base * 2 ** k) * (0.75 + 0.5 * random.random())


def _fetch_range(url, part: Path, lo: int, hi: int, tries=8):
    """One byte range into `part` at offset lo (rangefetch.sh logic: curl -r lo-hi --retry)."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}",
                                                       "User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=300) as r, open(part, "r+b") as out:
                if r.status != 206 and not (lo == 0 and r.status == 200):
                    raise IOError(f"HTTP {r.status} for a range request")
                out.seek(lo)
                got = 0
                for chunk in iter(lambda: r.read(1 << 22), b""):
                    out.write(chunk)
                    got += len(chunk)
            if got != hi - lo + 1:
                raise IOError(f"range {lo}-{hi}: {got} bytes")
            return got
        except Exception as e:  # noqa: BLE001 - network; retried, then raised
            if k == tries - 1:
                raise
            w = backoff_s(k)
            log(f"range {lo}-{hi}: {e}; retry {k + 1} in {w:.0f} s")
            time.sleep(w)


def download(url, dest: Path, want_md5: str, size: int, ranges=RANGES, tries=3) -> dict:
    """`ranges` parallel HTTP byte ranges into one preallocated file, then the md5 of the whole
    against the portal's; mode 0444. Returns {"bytes", "seconds"}."""
    from concurrent.futures import ThreadPoolExecutor
    dest = Path(dest)
    for k in range(tries):
        part = dest.with_name(dest.name + ".part%d" % os.getpid())
        t0 = time.time()
        try:
            with open(part, "wb") as fh:
                fh.truncate(size)
            rr = byte_ranges(size, ranges)
            with ThreadPoolExecutor(len(rr)) as ex:
                for f in [ex.submit(_fetch_range, url, part, lo, hi) for lo, hi in rr]:
                    f.result()
            got = md5_file(part)
            if got != want_md5:
                raise IOError(f"md5 {got} != portal {want_md5}")
            os.chmod(part, 0o444)
            os.replace(part, dest)
            return {"bytes": size, "seconds": time.time() - t0}
        except Exception as e:  # noqa: BLE001 - network; retried, then raised
            if part.exists():
                part.unlink()
            if k == tries - 1:
                raise SystemExit(f"download {url} FAILED after {tries} tries: {e}")
            w = backoff_s(k, base=15)
            log(f"download {url}: {e}; retry {k + 1} in {w:.0f} s")
            time.sleep(w)


def fetch_many(items, ranges=RANGES, parallel_files=PARALLEL_FILES) -> dict:
    """Download `items` ({"url", "dest", "md5", "bytes"}) `parallel_files` at a time, each in
    `ranges` byte ranges; a file already present with the right size and md5 is kept. Returns the
    aggregate {"files", "bytes", "seconds", "mb_per_s", "ranges", "parallel_files"}."""
    from concurrent.futures import ThreadPoolExecutor
    todo = []
    for it in items:
        dest = Path(it["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == it["bytes"] and md5_file(dest) == it["md5"]:
            log(f"{dest.name}: present, md5 ok")
            continue
        if dest.exists():
            dest.chmod(0o644)
            dest.unlink()
        todo.append(it)
    t0 = time.time()

    def one(it):
        import fcntl
        dest = Path(it["dest"])
        # two rows of one experiment (base and an arm) may stage at once: one downloads, one waits
        with open(f"{dest}.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if dest.exists() and dest.stat().st_size == it["bytes"]:
                log(f"{dest.name}: staged by another job meanwhile")
                return None
            log(f"{dest.name}: downloading {it['bytes'] / 2**30:.2f} GiB in {ranges} ranges")
            r = download(it["url"], dest, it["md5"], it["bytes"], ranges=ranges)
        log(f"{dest.name}: {it['bytes'] / 1e6 / max(r['seconds'], 1e-9):.2f} MB/s")
        return r

    if todo:
        with ThreadPoolExecutor(max(1, min(parallel_files, len(todo)))) as ex:
            list(ex.map(one, todo))
    sec = time.time() - t0
    nb = sum(it["bytes"] for it in todo)
    st = {"files": len(todo), "bytes": nb, "seconds": round(sec, 1),
          "mb_per_s": round(nb / 1e6 / sec, 3) if todo and sec > 0 else None,
          "ranges": ranges, "parallel_files": parallel_files}
    log(f"staged {st['files']} file(s), {nb / 1e9:.3f} GB in {sec:.0f} s "
        f"({st['mb_per_s']} MB/s aggregate, {ranges} ranges x {parallel_files} files)")
    return st


def stage(experiment, fastq_root=FASTQ_ROOT, accessions=None, ranges=RANGES,
          parallel_files=PARALLEL_FILES) -> dict:
    """Fetch the portal table and every FASTQ (or `accessions` and their mates), md5-checked."""
    d = Path(fastq_root) / experiment
    d.mkdir(parents=True, exist_ok=True)
    fj = portal_files(experiment)
    want = fj["files"]
    if accessions:
        keep = set(accessions)
        want = [f for f in want if f["accession"] in keep or f["paired_with"] in keep]
    st = fetch_many([{"url": f["url"], "dest": d / f"{f['accession']}.fastq.gz", "md5": f["md5"],
                      "bytes": f["bytes"]} for f in want], ranges, parallel_files)
    fj["staged"] = [f["accession"] for f in want]
    fj["stage_wall_hours"] = st["seconds"] / 3600
    fj["stage_stats"] = st
    write_atomic(d / "files.json", json.dumps(fj, indent=1) + "\n")
    log(f"{experiment}: {len(want)} FASTQs staged and md5-verified in {d}")
    return fj


def row_kind(row) -> str:
    """`chip`, `atac`, `dnase` or `native` (ENCODE's redacted unfiltered BAM, PI ruling 16)."""
    p = (row.get("pipeline") or "").strip()
    if p == "native-redacted-bam":
        return "native"
    if row.get("assay_class") == "accessibility_dnase" or p.startswith("dnase-seq-pipeline"):
        return "dnase"
    return pipeline_key(row)[0]


def row_accessions(row) -> list:
    return [a for a in (row.get("fastq_files") or "").replace(",", ";").split(";") if a]


def stage_native(row, fastq_root=FASTQ_ROOT, ranges=RANGES, parallel_files=PARALLEL_FILES) -> dict:
    """The redacted unfiltered GRCh38 BAMs a native row names, md5-checked against the portal,
    under `<fastq root>/<experiment>/<accession>.bam`, with `bams.json` (one record per BAM)."""
    exp = row["experiment"]
    d = Path(fastq_root) / exp
    recs = [portal_file(a) for a in row_accessions(row)]
    for r in recs:
        if r["file_format"] != "bam" or r["assembly"] != "GRCh38":
            raise SystemExit(f"{r['accession']}: {r['file_format']} {r['assembly']}, not a GRCh38 BAM")
    st = fetch_many([{"url": r["url"], "dest": d / f"{r['accession']}.bam", "md5": r["md5"],
                      "bytes": r["bytes"]} for r in recs], ranges, parallel_files)
    rec = {"experiment": exp, "bams": recs, "stage_stats": st,
           "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_atomic(d / "bams.json", json.dumps(rec, indent=1) + "\n")
    return rec


def stage_row(row, fastq_root=FASTQ_ROOT, ranges=RANGES, parallel_files=PARALLEL_FILES,
              work=WORK) -> dict:
    """Stage what one build-list row reads: its FASTQs (any pipeline), or its native BAMs.
    Nothing when the row's run dir already holds ALIGN_DONE or NOT_APPLICABLE: the alignment no
    longer needs the FASTQs, and the cleanup may have deleted them (a second `submit`, KF2)."""
    if row_kind(row) != "native":
        d = arm_dir(work, row)
        done = [m for m in ("ALIGN_DONE", NOT_APPLICABLE) if (d / m).exists()]
        if done:
            log(f"row {row.get('row', '')}: {d / done[0]} exists, nothing to stage")
            return {"skipped": str(d / done[0])}
    if row_kind(row) == "native":
        return stage_native(row, fastq_root, ranges, parallel_files)
    return stage(row["experiment"], fastq_root, row_accessions(row) or None, ranges,
                 parallel_files)


def staged_ok(files_json, fastq_root, experiment, accessions) -> bool:
    by = {f["accession"]: f for f in files_json["files"]}
    for a in accessions:
        p = Path(fastq_path(fastq_root, experiment, a))
        if not p.exists() or p.stat().st_size != by[a]["bytes"]:
            return False
    return True


# --- references --------------------------------------------------------------------------------

def genome_tsv_text(refs_dir) -> str:
    lines = [f"{k}\t{v}" for k, v in GENOME_SCALARS]
    lines += [f"{k}\t{Path(refs_dir) / rel}" for k, rel in GENOME_FILES]
    return "\n".join(lines) + "\n"


def prepare_refs(src=REFCACHE, dest=REFS) -> Path:
    """Copy the EIC runs' hg38 bundle out of the purgeable refcache, md5-checked, and write
    `hg38.local.tsv` over the copies. Idempotent."""
    dest = Path(dest)
    rows = ["key\tpath\tbytes\tmd5"]
    for k, rel in GENOME_FILES:
        s, t = Path(src) / rel, dest / rel
        if not (t.exists() and t.stat().st_size == s.stat().st_size):
            t.parent.mkdir(parents=True, exist_ok=True)
            part = t.with_name(t.name + ".part")
            shutil.copyfile(s, part)
            os.replace(part, t)
        m = md5_file(t)
        if m != md5_file(s):
            raise SystemExit(f"prepare-refs: {t} md5 differs from {s}")
        os.chmod(t, 0o444)
        rows.append(f"{k}\t{t}\t{t.stat().st_size}\t{m}")
    write_atomic(dest / "REFS.tsv", "\n".join(rows) + "\n")
    write_atomic(dest / "hg38.local.tsv", genome_tsv_text(dest))
    log(f"refs ready under {dest}")
    return dest / "hg38.local.tsv"


# --- run ---------------------------------------------------------------------------------------

#: Caper's own jars, given as local files so Caper never "installs" them: with the default
#: (a GitHub URL) every `caper run` takes a file lock under ~/.caper on the NFS home, and jobs
#: starting together died on it ("OSError: [Errno 116] Stale file handle", KF job 26829982).
CROMWELL_JAR = "/home/mforooz/.caper/cromwell_jar/cromwell-82.jar"
WOMTOOL_JAR = "/home/mforooz/.caper/womtool_jar/womtool-82.jar"


def check_caper_jars(jars=None) -> None:
    """Fail fast, before staging or Cromwell starts, when a Caper jar named in the conf is
    missing (the jars live in the user's ~/.caper, outside the kit)."""
    jars = jars or (CROMWELL_JAR, WOMTOOL_JAR)
    missing = [str(j) for j in jars if not Path(j).is_file()]
    if missing:
        raise SystemExit("Caper jar(s) missing: " + ", ".join(missing) + ". Caper runs offline "
                         "from these local jars; restore them (e.g. `caper init local` on a login "
                         "node downloads cromwell-82 and womtool-82 into ~/.caper) and resubmit.")


def caper_conf_text() -> str:
    return (f"backend=local\ndisable-auto-update-check=True\ncromwell={CROMWELL_JAR}\n"
            f"womtool={WOMTOOL_JAR}\n")


def check_image(key, eic=EIC) -> str:
    spec = PIPELINES[key]
    got = sha256_file(Path(eic) / spec["sif"])
    if got != spec["sha256"]:
        raise SystemExit(f"{spec['sif']}: sha256 {got} != pinned {spec['sha256']}")
    return got


def crop_fastqs(reps, arm, fastq_root, experiment, tmp, cpus, eic=EIC, src=None) -> dict:
    """ATAC crop arm: crop each FASTQ (pair) with the chip image's Trimmomatic step. `src` maps
    an accession to a prepared copy (prepare_reads) to crop instead of the staged file."""
    sif = str(Path(eic) / PIPELINES["chip"]["sif"])
    out = Path(tmp) / "crop"
    out.mkdir(parents=True, exist_ok=True)
    src = src or {}

    def inp(acc):
        return src.get(acc) or fastq_path(fastq_root, experiment, acc)

    paths = {}
    for r in reps:
        pairs = list(zip(r["r1"], r["r2"])) if r["r2"] else [(a, None) for a in r["r1"]]
        for a, b in pairs:
            o1, o2 = out / a, out / (b or "none")
            o1.mkdir(exist_ok=True)
            cmd = ["apptainer", "exec", "--cleanenv", "-B", f"{fastq_root},{Path(tmp)}", sif,
                   "encode_task_trimmomatic.py", "--fastq1", inp(a),
                   "--crop-length", str(arm["crop_length"]), "--crop-length-tol", str(CROP_TOL),
                   "--out-dir-R1", str(o1), "--nth", str(cpus)]
            if b:
                o2.mkdir(exist_ok=True)
                cmd += ["--fastq2", inp(b), "--paired-end",
                        "--out-dir-R2", str(o2)]
            log("crop: " + " ".join(cmd))
            subprocess.run(cmd, check=True)
            got1 = sorted(o1.glob("*.fastq.gz"))
            if len(got1) != 1:
                raise SystemExit(f"crop {a}: {len(got1)} outputs in {o1}")
            paths[a] = str(got1[0])
            if b:
                got2 = sorted(o2.glob("*.fastq.gz"))
                if len(got2) != 1:
                    raise SystemExit(f"crop {b}: {len(got2)} outputs in {o2}")
                paths[b] = str(got2[0])
    return paths


# --- read preparation before Caper (D6x) ---------------------------------------------------------
#
# Two source defects, both repaired by a streaming rewrite into the job's tmp dir (the staged file
# keeps the portal's bytes and md5), recorded under arm.json `fastq_prep`:
#  (1) Broad paired-end FASTQs name reads `@<id>/1 1:X:0:<index>`. Bowtie2 strips `/1` `/2` only at
#      the end of the whole header line, so with a comment after it the SAM QNAMEs keep the suffix,
#      mates never match, and the pipeline's `samtools fixmate` keeps no pair (D6b-early: 27 rows;
#      Nibi probe 2026-11-13: 0 proper pairs with the suffix, 400 of 400 without). The suffix is
#      stripped from the name token of every header, only for paired-end runs whose first header
#      has it.
#  (2) A FASTQ that ENCODE itself serves truncated (md5 = portal md5, gzip ends early; ENCFF033XNT):
#      the complete records before the break are kept; its mate is cut to the same read count.
#      Below MIN_KEPT_FRACTION of the expected reads the row is refused.

MIN_KEPT_FRACTION = 0.5
SUFFIX_RE = re.compile(r"/[12]$")


def gz_stream(path, sink=None, bufsize=1 << 22) -> dict:
    """Decompress a (multi-member) gzip file as far as it goes, passing every decompressed chunk
    to `sink`. `gzip -cd` drops its last buffer at a break, so the kit reads with zlib itself.
    Returns {"complete", "records": complete 4-line records, "trailing_garbage_bytes"}. Bytes after
    a clean member end that do not start a new gzip member are trailing garbage: the stream counts
    as complete (gzip itself says "decompression OK, trailing garbage ignored"), and the count of
    those bytes is returned so the caller can record it. A member that starts and breaks off is a
    truncation."""
    import zlib
    lines, started, ok, garbage = 0, False, True, 0
    boundary = False                     # just after a clean member end
    d = zlib.decompressobj(31)
    with open(path, "rb") as fh:
        for buf in iter(lambda: fh.read(bufsize), b""):
            if garbage:
                garbage += len(buf)
                continue
            while buf:
                if boundary and not buf.startswith(b"\x1f\x8b"[:len(buf)]):
                    garbage = len(buf)
                    break
                try:
                    out = d.decompress(buf)
                except zlib.error:
                    if boundary:
                        garbage = len(buf)
                        break
                    ok = False
                    break
                started, boundary = True, False
                lines += out.count(b"\n")
                if sink is not None and out:
                    sink(out)
                if d.eof:
                    buf, d, started, boundary = d.unused_data, zlib.decompressobj(31), False, True
                else:
                    buf = b""
            if not ok:
                break
    if started and not d.eof:
        ok = False
    return {"complete": ok, "records": lines // 4, "trailing_garbage_bytes": garbage}


def gz_scan(path) -> dict:
    return gz_stream(path)


def gz_scan_cached(path) -> dict:
    """gz_scan, remembered beside the staged file (`.<name>.gzscan.json`, keyed by size and
    mtime) so that the rows of one experiment scan each file once."""
    p = Path(path)
    st = p.stat()
    key = {"bytes": st.st_size, "mtime": int(st.st_mtime)}
    memo = p.with_name(f".{p.name}.gzscan.json")
    try:
        got = json.loads(memo.read_text())
        if {k: got.get(k) for k in key} == key:
            return {k: got[k] for k in ("complete", "records", "trailing_garbage_bytes")}
    except (OSError, ValueError, KeyError):
        pass
    res = gz_scan(p)
    try:
        write_atomic(memo, json.dumps({**key, **res}) + "\n")
    except OSError:
        pass
    return res


def first_header(path) -> str:
    with gzip.open(path, "rt", errors="replace") as fh:
        return fh.readline().rstrip("\n")


def has_mate_suffix(path) -> bool:
    """The first header's name token ends in /1 or /2."""
    h = first_header(path)
    return bool(h.startswith("@") and h[1:].split() and SUFFIX_RE.search(h[1:].split()[0]))


#: awk over a decompressed FASTQ: keep at most `lim` records (0 = all), strip /1 /2 from the
#: name token when s = 1, and write "<records> <last name token>" to `cnt` at the end.
REWRITE_AWK = r"""
lim > 0 && NR > 4 * lim { exit }
NR % 4 == 1 {
  m = match($0, /[ \t]/); t = m ? substr($0, 1, m - 1) : $0; r = m ? substr($0, m) : ""
  if (s == 1 && t ~ /\/[12]$/) t = substr(t, 1, length(t) - 2)
  $0 = t r; last = t
}
{ print; c++ }
END { print int(c / 4), last > cnt }
"""


def rewrite_fastq(src, dest, records=0, strip_suffix=False) -> tuple[int, str]:
    """Stream src -> dest (gzip -1): the first `records` records (0 = all), with the /1 /2 name
    suffix stripped when asked. Returns (records written, last name token, `@` included)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cnt = dest.with_name(dest.name + ".count")
    cmd = (f"awk -v lim={int(records)} -v s={1 if strip_suffix else 0} -v cnt={sh_quote(cnt)} "
           f"{sh_quote(REWRITE_AWK)} | gzip -1 > {sh_quote(dest)}")
    proc = subprocess.Popen(["bash", "-c", cmd], stdin=subprocess.PIPE)

    class Done(Exception):
        pass

    def sink(chunk):
        try:
            proc.stdin.write(chunk)
        except BrokenPipeError:           # awk kept its `records` and exited
            raise Done
    try:
        gz_stream(src, sink)
    except Done:
        pass
    try:
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if proc.wait() != 0:
        raise SystemExit(f"rewrite {src} -> {dest}: awk | gzip exit {proc.returncode}")
    n, _, last = cnt.read_text().strip().partition(" ")
    cnt.unlink()
    return int(n), last


def sh_quote(x) -> str:
    import shlex
    return shlex.quote(str(x))


def _portal_read_count(acc):
    try:
        return _get_json(f"{PORTAL}/files/{acc}/?format=json&frame=object", tries=2).get("read_count")
    except Exception:  # noqa: BLE001 - provenance only; the mate's count stands in
        return None


def prepare_reads(reps, pe_per_rep, files_json, fastq_root, experiment, tmp, cpus=4,
                  read_count=_portal_read_count) -> tuple[dict, dict]:
    """Repair the two source defects above for the FASTQs one run reads. Returns (accession ->
    path of the repaired copy under `tmp`/prep, accession -> provenance record); both empty when
    no file needs a repair. Raises SystemExit when a truncated file keeps under
    MIN_KEPT_FRACTION of its expected reads, or when a cut pair's last names disagree."""
    from concurrent.futures import ThreadPoolExecutor
    by = {f["accession"]: f for f in files_json["files"]}
    units = []                                   # (r1, r2 or None)
    for r, is_pe in zip(reps, pe_per_rep):
        if is_pe:
            units += list(zip(r["r1"], r["r2"]))
        else:
            units += [(a, None) for a in r["r1"]]
    accs = [a for u in units for a in u if a]

    def path(a):
        return fastq_path(fastq_root, experiment, a)

    with ThreadPoolExecutor(max(1, min(cpus, len(accs)))) as ex:
        scans = dict(zip(accs, ex.map(lambda a: gz_scan_cached(path(a)), accs)))
    plan = []                                    # (unit, records to keep or 0, strip, notes)
    for a, b in units:
        bad = [x for x in (a, b) if x and not scans[x]["complete"]]
        strip = bool(b) and (has_mate_suffix(path(a)) or has_mate_suffix(path(b)))
        junk = [x for x in (a, b) if x and scans[x].get("trailing_garbage_bytes")]
        if not bad and not strip and not junk:
            continue
        keep, note = 0, {}
        if bad:
            keep = min(scans[x]["records"] for x in (a, b) if x)
            exp_n = None
            for x in bad + [y for y in (a, b) if y and y not in bad]:
                exp_n = (by.get(x) or {}).get("read_count") or read_count(x)
                if exp_n:
                    break
            frac = keep / exp_n if exp_n else None
            note = {"truncated_source": True, "truncated_files": bad, "reads_kept": keep,
                    "reads_expected": exp_n,
                    "kept_fraction": round(frac, 4) if frac is not None else None}
            log(f"truncated source FASTQ {bad}: {keep} complete records kept of {exp_n} expected")
            if frac is None or frac < MIN_KEPT_FRACTION:
                raise SystemExit(f"truncated source FASTQ {bad} of {experiment}: kept fraction "
                                 f"{frac} is under {MIN_KEPT_FRACTION}, or no portal read count "
                                 "exists for either mate; not built")
        plan.append(((a, b), keep, strip, note))
    if not plan:
        return {}, {}
    out = Path(tmp) / "prep"

    def one(item):
        (a, b), keep, strip, note = item
        res = {x: rewrite_fastq(path(x), out / f"{x}.fastq.gz", keep, strip) for x in (a, b) if x}
        return item, res

    paths, prov = {}, {}
    with ThreadPoolExecutor(max(1, min(cpus, len(plan)))) as ex:
        for ((a, b), keep, strip, note), res in ex.map(one, plan):
            want = keep or None
            names = set()
            for x, (n, last) in res.items():
                if want is None:
                    want_x = scans[x]["records"]
                else:
                    want_x = want
                if n != want_x:
                    raise SystemExit(f"prepare {x}: wrote {n} records, expected {want_x}")
                names.add(SUFFIX_RE.sub("", last))
            if b and len(names) != 1:
                raise SystemExit(f"prepare {a}/{b}: last read names differ after the cut: {names}")
            for x, (n, _) in res.items():
                paths[x] = str(out / f"{x}.fastq.gz")
                prov[x] = {"mate_of": (b if x == a else a) if b else None,
                           "read_names_normalised": strip,
                           "read_name_rule": ("trailing /1 or /2 stripped from the name token "
                                              "(Bowtie2 keeps it when a comment follows)")
                           if strip else "",
                           "truncated_source": bool(note), "records_written": n,
                           "source_md5": (by.get(x) or {}).get("md5", ""), **note}
                g = scans[x].get("trailing_garbage_bytes") or 0
                if g:
                    prov[x]["trailing_garbage_bytes"] = g
                    prov[x]["trailing_garbage_note"] = (
                        f"{g} bytes after the last complete gzip member are not gzip data; "
                        "dropped, the file counts as complete")
    log(f"prepared {len(paths)} FASTQ(s) under {out}: "
        + ", ".join(f"{x} (names {'stripped' if p['read_names_normalised'] else 'kept'}, "
                    f"truncated {p['truncated_source']})" for x, p in prov.items()))
    return paths, prov


def input_paths(key, arm, reps, fastq_root, experiment, prepared, crop) -> tuple:
    """(accession -> FASTQ path for the input JSON, or None for the staged paths; the crop_tool
    record, or '' when no crop ran). `crop()` runs the ATAC pre-crop and is called only for an
    ATAC crop arm; only then is a crop tool recorded (a repaired, uncropped run records none)."""
    paths, crop_tool = (dict(prepared) if prepared else None), ""
    if key == "atac" and arm["crop_length"]:
        paths = crop()
        crop_tool = ("encode_task_trimmomatic.py from " + PIPELINES["chip"]["image"]
                     + f" (CROP {arm['crop_length']}, MINLEN {arm['crop_length'] - CROP_TOL})")
    if paths is not None:
        for a in [a for r in reps for a in r["r1"] + r["r2"]]:
            paths.setdefault(a, fastq_path(fastq_root, experiment, a))
    return paths, crop_tool


def final_calls(meta, name) -> dict:
    """shard index -> the last Done attempt of a call."""
    best = {}
    for c in meta.get("calls", {}).get(name, []):
        if c.get("executionStatus") != "Done":
            continue
        s = c.get("shardIndex", -1)
        if s not in best or c.get("attempt", 1) > best[s].get("attempt", 1):
            best[s] = c
    return best


def harvest_plan(meta, key, n_reps) -> list[tuple[str, str]]:
    """(dest name, source path) for the keepers, from Cromwell's metadata outputs."""
    pfx = key
    want = [("align", "bam", "rep{}.unfiltered.bam"), ("filter", "nodup_bam", "rep{}.filtered.bam"),
            ("filter", "nodup_bai", "rep{}.filtered.bam.bai"), ("bam2ta", "ta", "rep{}.tagAlign.gz")]
    plan = []
    for call, out, name in want:
        shards = final_calls(meta, f"{pfx}.{call}")
        if sorted(shards) != list(range(n_reps)):
            raise SystemExit(f"{pfx}.{call}: Done shards {sorted(shards)}, expected {n_reps}")
        for s, c in shards.items():
            plan.append((name.format(s + 1), c["outputs"][out]))
    if key == "chip":
        for s, c in final_calls(meta, "chip.xcor").items():
            if (c.get("outputs") or {}).get("score"):
                plan.append((f"rep{s + 1}.xcor.qc", c["outputs"]["score"]))
    for c in final_calls(meta, f"{pfx}.qc_report").values():
        if (c.get("outputs") or {}).get("qc_json"):
            plan.append(("qc.json", c["outputs"]["qc_json"]))
    return plan


def copy_verified(src, dest: Path) -> str:
    want = md5_file(src)
    part = dest.with_name(dest.name + ".part")
    shutil.copyfile(src, part)
    if md5_file(part) != want:
        part.unlink()
        raise SystemExit(f"copy {src} -> {dest}: md5 mismatch")
    os.replace(part, dest)
    return want


def count_lines_gz(path) -> int:
    n = 0
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            n += chunk.count(b"\n")
    return n


def run(rows, row, work, tmp, cpus, fastq_root=FASTQ_ROOT, refs=REFS, eic=EIC, max_tasks=2):
    t_start = time.time()
    key, _ = pipeline_key(row)
    exp = row["experiment"]
    out = arm_dir(work, row)
    if (out / "ALIGN_DONE").exists() or (out / NOT_APPLICABLE).exists():
        log(f"{out}: ALIGN_DONE or {NOT_APPLICABLE} exists, nothing to do")
        cleanup_fastq(rows, exp, work, fastq_root)      # a rerun still frees what it can
        return 0
    out.mkdir(parents=True, exist_ok=True)
    genome_tsv = Path(refs) / "hg38.local.tsv"
    if not genome_tsv.exists():
        raise SystemExit(f"{genome_tsv} missing: run `fastq_uniform.py prepare-refs` first")
    check_caper_jars()
    check_image(key, eic)

    fj_path = Path(fastq_root) / exp / "files.json"
    restrict = [a for a in (row.get("fastq_files") or "").replace(",", ";").split(";") if a]
    fj = json.loads(fj_path.read_text()) if fj_path.exists() else None
    reps = unit_reps(fj, row["unit"], restrict or None) if fj else None
    need = [a for r in reps for a in r["r1"] + r["r2"]] if reps else []
    if fj is None or not staged_ok(fj, fastq_root, exp, need):
        log(f"{exp}: FASTQs not staged; staging inside this job")
        fj = stage(exp, fastq_root, restrict or None)
        reps = unit_reps(fj, row["unit"], restrict or None)
    t_staged = time.time()
    try:
        arm = arm_settings(fj, reps, row["arm"])
    except NotApplicable as e:
        write_not_applicable(out, row, str(e))
        cleanup_fastq(rows, exp, work, fastq_root)
        return 0
    job = os.environ.get("SLURM_JOB_ID")
    rec = arm_record(row, fj, reps, arm, job)
    rec["fastq_md5"] = {f["accession"]: f["md5"] for f in fj["files"]
                        if f["accession"] in {a for r in reps for a in r["r1"] + r["r2"]}}

    tmp = Path(tmp) / f"onion_{exp}_{row['unit']}_{row['arm']}"
    tmp.mkdir(parents=True, exist_ok=True)
    prepared, prep = prepare_reads(reps, arm["paired_end_per_rep"], fj, fastq_root, exp, tmp,
                                   cpus)
    if prep:
        rec["fastq_prep"] = prep
        rec["truncated_source"] = any(p["truncated_source"] for p in prep.values())
    cropped, crop_tool = input_paths(
        key, arm, reps, fastq_root, exp, prepared,
        lambda: crop_fastqs(reps, arm, fastq_root, exp, tmp, cpus, eic, src=prepared))
    if crop_tool:
        rec["crop_tool"] = crop_tool
    inp = build_input_json(row, fj, genome_tsv, fastq_root, eic, cropped)
    write_atomic(out / "input.json", json.dumps(inp, indent=2) + "\n")
    write_atomic(tmp / "input.json", json.dumps(inp, indent=2) + "\n")
    conf = tmp / "caper.conf"
    conf.write_text(caper_conf_text())
    spec = PIPELINES[key]
    meta_path = tmp / "metadata.json"
    binds = ",".join(sorted({str(Path(fastq_root)), str(Path(refs)), str(tmp)}))
    cmd = ["caper", "run", str(Path(eic) / spec["wdl"]), "-c", str(conf), "-b", "local",
           "-i", str(tmp / "input.json"), "--singularity", str(Path(eic) / spec["sif"]),
           "--local-out-dir", str(tmp / "out"), "--local-loc-dir", str(tmp / "loc"),
           "--local-hash-strat", "path+modtime", "--db", "in-memory",
           "--max-concurrent-tasks", str(max_tasks), "-m", str(meta_path),
           "--cromwell-stdout", str(tmp / "cromwell.out")]
    env = dict(os.environ, SINGULARITY_BINDPATH=binds, APPTAINER_BINDPATH=binds)
    log("caper: " + " ".join(cmd))
    t_align = time.time()
    rc = subprocess.run(cmd, cwd=str(tmp), env=env).returncode
    t_aligned = time.time()
    if not meta_path.exists():
        tail = (tmp / "cromwell.out").read_text()[-3000:] if (tmp / "cromwell.out").exists() else ""
        raise SystemExit(f"caper rc={rc}, no metadata.json. cromwell.out tail:\n{tail}")
    meta = json.loads(meta_path.read_text())
    if meta.get("status") != "Succeeded":
        shutil.copyfile(meta_path, out / "metadata.failed.json")
        raise SystemExit(f"workflow {meta.get('id')} {meta.get('status')} (caper rc={rc}); "
                         f"metadata kept at {out / 'metadata.failed.json'}")

    lines = ["name\tsrc\tbytes\tmd5"]
    for name, src in harvest_plan(meta, key, len(reps)):
        m = copy_verified(src, out / name)
        lines.append(f"{name}\t{src}\t{(out / name).stat().st_size}\t{m}")
        log(f"kept {name} {(out / name).stat().st_size} B")
    shutil.copyfile(meta_path, out / "metadata.json")
    write_atomic(out / "HARVEST.tsv", "\n".join(lines) + "\n")
    write_atomic(out / "arm.json", json.dumps(rec, indent=2) + "\n")

    n_reads = {f"rep{r['rep']}": count_lines_gz(out / f"rep{r['rep']}.tagAlign.gz") for r in reps}
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    done = {
        "cpu_hours": round((ru.ru_utime + ru.ru_stime) / 3600, 4),
        "cpu_hours_source": "getrusage(RUSAGE_CHILDREN) of this process; sacct of the job is the "
                            "authority",
        "wall_hours": round((t_aligned - t_staged) / 3600, 4),
        "align_wall_hours": round((t_aligned - t_align) / 3600, 4),
        "stage_wall_hours": round((t_staged - t_start) / 3600, 4),
        "total_wall_hours": round((time.time() - t_start) / 3600, 4),
        "max_rss_gb": round(ru.ru_maxrss / 2**20, 3),
        "n_reads": sum(n_reads.values()), "n_reads_per_rep": n_reads,
        "n_reads_definition": "lines of the filtered tagAlign (one per read; 2 per pair)",
        "slurm_job_id": job, "workflow_id": meta.get("id"), "cpus": cpus,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "truncated_source": rec.get("truncated_source", False),
        "fastq_prep": prep,
    }
    write_atomic(out / "ALIGN_DONE", json.dumps(done, indent=2) + "\n")
    log(f"ALIGN_DONE {out}: {json.dumps(done)}")
    shutil.rmtree(tmp, ignore_errors=True)
    cleanup_fastq(rows, exp, work, fastq_root)
    return 0


# --- verify and cleanup ------------------------------------------------------------------------

def write_not_applicable(out: Path, row, reason: str):
    """Record a row whose arm does not exist for the experiment; it counts as verified."""
    rec = {"experiment": row["experiment"], "unit": row["unit"], "arm": row["arm"],
           "build_list_row": row.get("row", ""), "reason": reason,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_atomic(Path(out) / NOT_APPLICABLE, json.dumps(rec, indent=2) + "\n")
    log(f"{out}: {NOT_APPLICABLE}: {reason}")


def read_dropped(d: Path) -> dict:
    """name -> record of keepers deliberately deleted after the views were built (DROPPED.tsv)."""
    p = Path(d) / "DROPPED.tsv"
    if not p.exists():
        return {}
    rows = [ln.split("\t") for ln in p.read_text().strip().split("\n")[1:] if ln]
    return {r[0]: r for r in rows}


def verify_arm(d: Path, md5=False) -> list[str]:
    probs = []
    if (d / NOT_APPLICABLE).exists():
        return []
    if not (d / "ALIGN_DONE").exists():
        return [f"{d}: no ALIGN_DONE"]
    h = d / "HARVEST.tsv"
    if not h.exists():
        return [f"{d}: no HARVEST.tsv"]
    rows = [ln.split("\t") for ln in h.read_text().strip().split("\n")[1:]]
    names = {r[0] for r in rows}
    reps = json.loads((d / "arm.json").read_text())["reps"]
    for r in reps:
        for stem in ("unfiltered.bam", "filtered.bam", "filtered.bam.bai", "tagAlign.gz"):
            if f"rep{r['rep']}.{stem}" not in names:
                probs.append(f"{d}: HARVEST.tsv lacks rep{r['rep']}.{stem}")
    dropped = read_dropped(d)
    for name, _src, size, m in rows:
        p = d / name
        if name in dropped and not p.exists():
            continue
        if not p.exists() or p.stat().st_size != int(size):
            probs.append(f"{p}: missing or size != {size}")
        elif md5 and md5_file(p) != m:
            probs.append(f"{p}: md5 != {m}")
    return probs


def cleanup_fastq(rows, experiment, work, fastq_root) -> bool:
    """Delete the staged FASTQs once every build-list row of the experiment verifies.

    `rows` is the whole build list (every priority), so the FASTQs stay while any row of the
    experiment -- a priority-3 arm row or a priority-4 library or lane row included -- has no
    ALIGN_DONE or NOT_APPLICABLE yet. Native rows read staged BAMs, not FASTQs, and never call this.
    Two rows of one experiment may finish at the same moment (KF2 review): the directory is first
    renamed to a private name, which only one caller can do; the other finds it gone and returns
    True. Missing files are fine."""
    native = {r["row"] for r in rows if r["experiment"] == experiment and row_kind(r) == "native"}
    mine = [r for r in rows if r["experiment"] == experiment and r["row"] not in native]
    probs = [p for r in mine for p in verify_arm(arm_dir(work, r))]
    if probs:
        log(f"{experiment}: FASTQs kept, {len(probs)} row check(s) open, e.g. {probs[0]}")
        return False
    d = Path(fastq_root) / experiment
    gone = d.with_name(f".{experiment}.deleting.{os.getpid()}.{time.time_ns()}")
    try:
        os.rename(d, gone)
    except FileNotFoundError:
        return True                     # never staged, or another row's cleanup took it
    shutil.rmtree(gone, ignore_errors=True)
    if gone.exists():
        log(f"{experiment}: WARNING {gone} not fully deleted")
    log(f"{experiment}: all {len(mine)} rows verified; deleted {d}")
    return True


# --- CLI ---------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="an experiment's FASTQs, or what one build-list row reads")
    s.add_argument("--experiment", action="append", default=[])
    s.add_argument("--build-list")
    s.add_argument("--row")
    s.add_argument("--fastq-root", default=FASTQ_ROOT)
    s.add_argument("--ranges", type=int, default=RANGES, help="byte ranges per file")
    s.add_argument("--parallel-files", type=int, default=PARALLEL_FILES)
    s.add_argument("--work", default=WORK, help="a row whose run dir is done stages nothing")
    sub.add_parser("check-jars", help="exit non-zero when Caper's local jars are missing")
    s = sub.add_parser("prepare-refs")
    s.add_argument("--src", default=REFCACHE)
    s.add_argument("--dest", default=REFS)
    s = sub.add_parser("input-json")
    s.add_argument("--build-list", required=True)
    s.add_argument("--row", required=True)
    s.add_argument("--files-json", default=None)
    s.add_argument("--fastq-root", default=FASTQ_ROOT)
    s.add_argument("--refs", default=REFS)
    s = sub.add_parser("run")
    s.add_argument("--build-list", required=True)
    s.add_argument("--row", required=True)
    s.add_argument("--work", default=WORK)
    s.add_argument("--tmp", default=os.environ.get("SLURM_TMPDIR"))
    s.add_argument("--cpus", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    s.add_argument("--max-concurrent-tasks", type=int, default=2)
    s.add_argument("--fastq-root", default=FASTQ_ROOT)
    s.add_argument("--refs", default=REFS)
    s = sub.add_parser("verify")
    s.add_argument("--work", default=WORK)
    s.add_argument("--experiment", required=True)
    s.add_argument("--md5", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "stage":
        if a.build_list:
            if not a.row:
                raise SystemExit("stage --build-list needs --row")
            stage_row(get_row(read_build_list(a.build_list), a.row), a.fastq_root, a.ranges,
                      a.parallel_files, a.work)
        elif not a.experiment:
            raise SystemExit("stage: give --experiment, or --build-list and --row")
        for e in a.experiment:
            stage(e, a.fastq_root, None, a.ranges, a.parallel_files)
        return 0
    if a.cmd == "check-jars":
        check_caper_jars()
        print("caper jars present: " + ", ".join((CROMWELL_JAR, WOMTOOL_JAR)))
        return 0
    if a.cmd == "prepare-refs":
        prepare_refs(a.src, a.dest)
        return 0
    if a.cmd == "input-json":
        row = get_row(read_build_list(a.build_list), a.row)
        fjp = a.files_json or str(Path(a.fastq_root) / row["experiment"] / "files.json")
        fj = json.loads(Path(fjp).read_text())
        print(json.dumps(build_input_json(row, fj, Path(a.refs) / "hg38.local.tsv",
                                          a.fastq_root), indent=2))
        return 0
    if a.cmd == "run":
        if not a.tmp:
            raise SystemExit("run: --tmp (or $SLURM_TMPDIR) is required; Cromwell never writes "
                             "to /project or /scratch")
        rows = read_build_list(a.build_list)
        return run(rows, get_row(rows, a.row), a.work, a.tmp, a.cpus, a.fastq_root, a.refs,
                   max_tasks=a.max_concurrent_tasks)
    if a.cmd == "verify":
        base = Path(a.work) / a.experiment
        dirs = sorted(p for p in base.iterdir() if p.is_dir() and RUN_DIR_RE.match(p.name)) \
            if base.exists() else []
        probs = [p for d in dirs for p in verify_arm(d, a.md5)]
        for p in probs:
            print(p)
        print(f"{a.experiment}: {len(dirs)} arm dir(s), {len(probs)} problem(s)")
        return 1 if probs or not dirs else 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
