"""DNase-seq FASTQ kit: ENCODE-DCC dnase-seq-pipeline v3.0.0, alignment only (t152, chunk D7d).

PI rule: DNase-seq is processed with `dnase-seq-pipeline` (github.com/ENCODE-DCC/dnase-seq-pipeline),
never with the ATAC pipeline's DNase mode.

The pipeline has no alignment-only switch: `dnase.wdl` scatters `dnase_replicate.wdl`, which always
calls hotspots, footprints and normalization after the filtered BAM. This kit runs the pipeline's
own first two stages and nothing else, through a two-call wrapper workflow (`wrapper_wdl()`)
written beside `dnase.wdl` inside the unpacked v3.0.0 tree:

    concatenate_trim_and_align_fastqs  ->  merge_mark_and_filter_bams

Both are the v3.0.0 workflows, unmodified, with the release's default machine sizes. References
are passed in their unpacked form (`References.indexed_fasta`, `.bwa_index`, `.nuclear_chroms`),
unpacked once by `setup` from the ENCODE4 reference set ENCSR083JOX; the hotspot references are
not needed before peak calling and are not passed.

Subcommands (stdlib only; `run` and `setup` run inside a Nibi job, see slurm/onion_data/fastq_dnase.sh):

    setup   --pipeline-dir P --scratch-refs R    pull the image, the release tarball and the
                                                 reference set; unpack the alignment refs to R;
                                                 validate the wrapper with womtool
    units   --experiment ENCSR...                print the experiment's units and their arms
    run     (--experiment E --unit U --arm A | --build-list TSV --row I) --work W [--dry-run]

Units (pinned interface B): `pooled` (one pipeline run over every biological replicate; it
yields `br<i>` and `pooled`), `br<i>tr<j>` (one library alone), `br<i>tr<j>ln<k>` (one FASTQ, or
one R1/R2 pair, of one library alone; k orders the library's files by accession).
Arms: `base`; `se` (paired-end reads only: mate 1 of every pair as single-end); `crop36`, `crop50`
(when the unit's shortest read is longer than the crop). A crop sets the replicate's
`read_length`, which the pipeline's own trim step (`trim_fastq_to_length`) applies.

Output, per run, under `<work>/<experiment>/<dir>/` where `<dir>` is `<arm>` for unit `pooled`
and `<unit>.<arm>` otherwise -- the SAME layout and names as the ChIP/ATAC kit (fastq_uniform.py),
so align_views.py reads both alike:

    rep<i>.unfiltered.bam     the pipeline's merged BAM (ENCODE "unfiltered alignments")
    rep<i>.filtered.bam(.bai) the pipeline's nuclear BAM (ENCODE "alignments"; QC-fail and
                              non-nuclear reads removed; duplicates kept, flagged 0x400)
    rep<i>.tagAlign.gz        reads of rep<i>.filtered.bam without flag 0x400, one line per read
                              (mates of a pair are two lines), `bedtools bamtobed` + the tagAlign awk
    rep<i>.duplication_metrics.txt
    input.json  arm.json (reps: rep i -> bio_rep, run type, read length, covariates)  HARVEST.tsv
    ALIGN_DONE (json: cpu_hours, wall_hours, max_rss_gb, n_reads, provenance; written last)

`rep<i>` is the i-th biological replicate of the unit in replicate-number order (arm.json maps it).
FASTQs are read from the shared staging dir `<fastq root>/<experiment>/` (fastq_uniform.py stage),
fetched there if absent, and deleted once every build-list row of the experiment verifies.
Cromwell runs in `--tmp` (the job's `$SLURM_TMPDIR`); only the files above reach `--work`.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fastq_uniform as FU  # noqa: E402  (sibling, stdlib only: staging, harvest, cleanup)

PIPELINE_REPO = "https://github.com/ENCODE-DCC/dnase-seq-pipeline"
PIPELINE_TAG = "v3.0.0"
PIPELINE_COMMIT = "c1082e63d8893d9520fd3f714d0580039f76aaf2"
PIPELINE_TARBALL_URL = PIPELINE_REPO + "/archive/refs/tags/" + PIPELINE_TAG + ".tar.gz"
IMAGE = "encodedcc/dnase-seq-pipeline:v3.0.0"
#: Docker Hub manifest digest of IMAGE, read 2026-11-10 (registry-1.docker.io, docker-content-digest)
IMAGE_DIGEST = "sha256:5d3983130a2b6d6b7b47ab9c0c4a0644a74b69dd556863e7ce7fbe6514a2ca3b"

PIPELINE_DIR = "/project/def-maxwl/mforooz/ONION/pipelines/dnase"
SCRATCH_REFS = "/scratch/mforooz/onion/pipelines/dnase/refs"
WORK = "/scratch/mforooz/onion/work"
SIF_NAME = "dnase-seq-pipeline_v3.0.0.sif"
TARBALL_NAME = "dnase-seq-pipeline_v3.0.0.tar.gz"
GENOME_NAME = "GRCh38"
CROMWELL = "/home/mforooz/.caper/cromwell_jar/cromwell-82.jar"
WOMTOOL = "/home/mforooz/.caper/womtool_jar/womtool-82.jar"
ENCODE = "https://www.encodeproject.org"

#: ENCODE reference set ENCSR083JOX, "Reference & index files used by ENCODE4 Uniform Processing
#: DNase-seq pipeline for GRCh38" (portal, read 2026-11-10). accession -> (output_type, ext, md5,
#: bytes). The `dnase.references` tar.gz form of the README maps onto these files.
REFERENCE_SET = "ENCSR083JOX"
REFERENCE_FILES = {
    "ENCFF499QKB": ("genome reference", "tar.gz", "6c4ee6c10d21e049d5391f1c2fbdbfbf", 872953187),
    "ENCFF884HOA": ("genome index", "tar.gz", "b51b024f1d0091b7c160bcfc85d17c94", 4318243849),
    "ENCFF762MJQ": ("chromosomes reference", "txt.gz", "21dd8c5316e69937fc88141269865f29", 80),
    "ENCFF119KWQ": ("bias models", "txt.gz", "3cc463a5bf73637abf4a8d3869fa923f", 38737),
    "ENCFF405EUN": ("hotspots1 reference", "tar.gz", "4162d405fe9897af1fcdccef0d6626d1", 46838657),
    "ENCFF180EJG": ("hotspots2 reference", "tar.gz", "141f59a0f2d4fa3037caa8f5bce3b370", 49306335),
    "ENCFF304SVB": ("hotspots1 reference", "tar.gz", "9732a40cde1e89e7c621c3bb936faf91", 5207385),
    "ENCFF162AKB": ("hotspots2 reference", "tar.gz", "58e25bcf87320824789aacb6dd2f8277", 6025381),
}
REF_INDEXED_FASTA, REF_BWA_INDEX, REF_NUCLEAR_CHROMS = "ENCFF499QKB", "ENCFF884HOA", "ENCFF762MJQ"

#: wdl/default_machine_sizes.json at v3.0.0, passed explicitly (its read_json default is relative)
MACHINE_SIZES = {
    "align": "large2x", "concatenate": "medium", "convert": "medium", "filter": "large",
    "footprint": "medium2x", "mark": "large", "merge": "large", "normalize": "medium2x",
    "peaks": "medium2x", "qc": "medium2x", "score": "medium", "trim": "large2x", "unpack": "medium",
}

ARMS = ("base", "se", "crop36", "crop50")
CROPS = {"crop36": 36, "crop50": 50}
FASTQ_STATUSES = ("released",)
#: used only when an experiment has no released FASTQ (7 training DNase-seq experiments of the
#: published list are archived; build list D4), as fastq_uniform.portal_files does
FASTQ_FALLBACK_STATUSES = ("archived",)
#: stampipes filter_reads.py --min_mapq default (dnase-seq-pipeline v3.0.0, in the image)
MAPQ_THRESH = 10
TAGALIGN_AWK = r"""awk 'BEGIN{OFS="\t"}{$4="N";$5="1000";print $0}'"""
BUILD_LIST_KEYS = ("row", "experiment", "unit", "arm")

WRAPPER_NAME = "dnase_align_only.wdl"


def wrapper_wdl() -> str:
    """The alignment-only workflow. It sits at the root of the v3.0.0 tree, beside dnase.wdl."""
    return """version 1.0

# t152 D7d: the first two stages of dnase_replicate.wdl (v3.0.0), unmodified, and nothing after.

import "wdl/workflows/concatenate_trim_and_align_fastqs.wdl" as raw_fastqs
import "wdl/workflows/merge_mark_and_filter_bams.wdl" as name_sorted_bams


workflow dnase_align_only {
    input {
        Array[Replicate] replicates
        References references
        MachineSizes machine_sizes
    }

    scatter (replicate in replicates) {
        call raw_fastqs.concatenate_trim_and_align_fastqs {
            input:
                replicate=replicate,
                references=references,
                machine_sizes=machine_sizes,
        }

        call name_sorted_bams.merge_mark_and_filter_bams {
            input:
                name_sorted_bams=concatenate_trim_and_align_fastqs.name_sorted_bams,
                references=references,
                machine_sizes=machine_sizes,
        }
    }

    output {
        Array[File] unfiltered_bam = merge_mark_and_filter_bams.unfiltered_bam
        Array[File] nuclear_bam = merge_mark_and_filter_bams.nuclear_bam
        Array[File] duplication_metrics = merge_mark_and_filter_bams.duplication_metrics
        Array[File?] trimstats = concatenate_trim_and_align_fastqs.trimstats
    }
}
"""


def log(msg):
    print("%s  %s" % (time.strftime("%F %T"), msg), flush=True)


def die(msg):
    raise SystemExit("fastq_dnase: " + msg)


# --- ENCODE metadata ---------------------------------------------------------------------------

def get_json(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": FU.USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 -- network: retry, then fail loudly
            if i == tries - 1:
                raise
            log(f"retry {url}: {e}")
            time.sleep(10 * (i + 1))


def fetch_metadata(experiment) -> dict:
    """{"experiment": embedded experiment json, "fastqs": [file objects]} from the portal."""
    exp = get_json(f"{ENCODE}/experiments/{experiment}/?format=json&frame=embedded")
    files = get_json(f"{ENCODE}/search/?type=File&dataset=/experiments/{experiment}/"
                     "&file_format=fastq&format=json&limit=all&frame=object")["@graph"]
    return {"experiment": exp, "fastqs": files}


def _acc(ref) -> str:
    return ref.rstrip("/").split("/")[-1]


def adapters_of(library) -> dict | None:
    seqs = {}
    for a in library.get("adapters") or []:
        t = a.get("type", "")
        if t.startswith("read1 3'"):
            seqs["sequence_R1"] = a["sequence"]
        elif t.startswith("read2 3'"):
            seqs["sequence_R2"] = a["sequence"]
    return seqs if set(seqs) == {"sequence_R1", "sequence_R2"} else None


def libraries(meta) -> list[dict]:
    """One dict per library: bio, tech, library accession, adapters, and its runs.

    A run is one SE FASTQ or one R1/R2 pair: {"R1": f, "R2": f|None, "read_length": int}, where f
    is the file object. Only FASTQs with status in FASTQ_STATUSES enter. Runs sort by R1 accession.
    """
    exp = meta["experiment"]
    reps = {}
    for r in exp.get("replicates", []):
        lib = r.get("library") or {}
        reps[r["@id"]] = {"bio": int(r["biological_replicate_number"]),
                          "tech": int(r["technical_replicate_number"]),
                          "library": lib.get("accession", ""),
                          "adapters": adapters_of(lib)}
    byacc = {f["accession"]: f for f in meta["fastqs"] if f.get("status") in FASTQ_STATUSES}
    if not byacc:
        byacc = {f["accession"]: f for f in meta["fastqs"]
                 if f.get("status") in FASTQ_FALLBACK_STATUSES}
    out = {}
    for f in sorted(byacc.values(), key=lambda f: f["accession"]):
        rep = reps.get(f.get("replicate"))
        if rep is None:
            die(f"{f['accession']}: replicate {f.get('replicate')} is not in the experiment")
        key = (rep["bio"], rep["tech"])
        lib = out.setdefault(key, dict(rep, runs=[]))
        pe = f.get("run_type") == "paired-ended"
        if pe and str(f.get("paired_end")) == "2":
            continue                      # taken with its mate 1
        mate = None
        if pe:
            mate = byacc.get(_acc(f.get("paired_with") or ""))
            if mate is None:
                die(f"{f['accession']}: paired-end mate 1 without a released mate 2")
        rl = [int(f["read_length"])] + ([int(mate["read_length"])] if mate else [])
        lib["runs"].append({"R1": f, "R2": mate, "read_length": min(rl)})
    return [out[k] for k in sorted(out)]


def units(meta) -> list[dict]:
    """Every unit of the experiment: {"unit", "libraries": [library dicts with selected runs]}."""
    libs = libraries(meta)
    us = [{"unit": "pooled", "libraries": libs}]
    for lib in libs:
        tag = f"br{lib['bio']}tr{lib['tech']}"
        us.append({"unit": tag, "libraries": [lib]})
        if len(lib["runs"]) > 1:
            for k, run in enumerate(lib["runs"], 1):
                us.append({"unit": f"{tag}ln{k}", "libraries": [dict(lib, runs=[run])]})
    return us


def unit_runs(u) -> list[dict]:
    return [r for lib in u["libraries"] for r in lib["runs"]]


def arms_for(u) -> list[str]:
    runs = unit_runs(u)
    if not runs:
        return []
    arms = ["base"]
    if any(r["R2"] is not None for r in runs):
        arms.append("se")
    shortest = min(r["read_length"] for r in runs)
    arms += [a for a, n in CROPS.items() if shortest > n]
    return arms


def select_unit(meta, unit) -> dict:
    for u in units(meta):
        if u["unit"] == unit:
            return u
    die(f"unit {unit!r} is not a unit of {meta['experiment'].get('accession')} "
        f"(have: {[u['unit'] for u in units(meta)]})")


# --- input JSON --------------------------------------------------------------------------------

def replicates_json(experiment, u, arm, fastq_path) -> list[dict]:
    """`dnase.replicates` for this unit and arm: one entry per biological replicate.

    `fastq_path(file_object) -> str` maps a FASTQ to its local path. PE runs of one replicate
    share one `adapters` entry (the pipeline's model); when its libraries disagree, the entry of
    the library with the most PE runs is used and recorded under `_adapters_note`.
    """
    if arm not in ARMS:
        die(f"unknown arm {arm!r} (known: {ARMS})")
    if arm not in arms_for(u):
        die(f"arm {arm} does not apply to unit {u['unit']} of {experiment} "
            f"(applies: {arms_for(u)})")
    bybio = {}
    for lib in u["libraries"]:
        bybio.setdefault(lib["bio"], []).append(lib)
    out = []
    for bio in sorted(bybio):
        libs = bybio[bio]
        se, pe, lengths, pe_libs = [], [], [], []
        for lib in libs:
            for r in lib["runs"]:
                lengths.append(r["read_length"])
                if r["R2"] is None:
                    se.append(fastq_path(r["R1"]))
                elif arm == "se":
                    se.append(fastq_path(r["R1"]))
                else:
                    pe.append({"R1": fastq_path(r["R1"]), "R2": fastq_path(r["R2"])})
                    pe_libs.append(lib)
        rep = {"accession": experiment, "number": bio,
               "read_length": CROPS.get(arm, min(lengths))}
        if pe:
            counts = {}
            for lib in pe_libs:
                counts[lib["library"]] = counts.get(lib["library"], 0) + 1
            if any(lib["adapters"] is None for lib in pe_libs):
                bad = sorted({lib["library"] for lib in pe_libs if lib["adapters"] is None})
                die(f"{experiment} br{bio}: paired-end library without read1/read2 3' adapters "
                    f"on the portal: {bad}")
            best = max(pe_libs, key=lambda lib: (counts[lib["library"]], lib["library"]))
            rep["adapters"] = best["adapters"]
            if len({json.dumps(lib["adapters"], sort_keys=True) for lib in pe_libs}) > 1:
                rep["_adapters_note"] = (f"libraries disagree on adapters; used {best['library']}")
            rep["pe_fastqs"] = pe
        if se:
            rep["se_fastqs"] = se
        out.append(rep)
    return out


def references_json(refs_dir) -> dict:
    """`dnase.references` in the unpacked form, from the files `setup` put under refs_dir."""
    d = Path(refs_dir)
    fa = str(d / f"{GENOME_NAME}.fa")
    return {
        "genome_name": GENOME_NAME,
        "indexed_fasta": {"fasta": fa, "fai": fa + ".fai"},
        "bwa_index": {"fasta": fa, **{k: f"{fa}.{k}" for k in ("amb", "ann", "bwt", "pac", "sa")}},
        "nuclear_chroms": str(d / "nuclear_chroms.txt"),
    }


def input_json(experiment, u, arm, fastq_path, refs_dir) -> dict:
    reps = replicates_json(experiment, u, arm, fastq_path)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in reps]
    return {"dnase_align_only.replicates": clean,
            "dnase_align_only.references": references_json(refs_dir),
            "dnase_align_only.machine_sizes": dict(MACHINE_SIZES)}


def out_dir_name(unit, arm) -> str:
    return arm if unit == "pooled" else f"{unit}.{arm}"


def product_names(unit, bios) -> list[str]:
    """Per-replicate product prefixes, as the ChIP/ATAC kit writes them: rep1..repN in
    biological-replicate order (arm.json maps rep i to its bio_rep)."""
    if unit != "pooled" and len(bios) != 1:
        die(f"unit {unit} spans {len(bios)} biological replicates")
    return [f"rep{i}" for i in range(1, len(bios) + 1)]


def read_build_list(tsv, row) -> tuple[list[dict], dict]:
    """(all rows, the row whose `row` column equals `row`) -- the same rule as fastq_uniform."""
    with open(tsv, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows or any(k not in rows[0] for k in BUILD_LIST_KEYS):
        die(f"{tsv}: needs columns {BUILD_LIST_KEYS}")
    hits = [r for r in rows if r["row"] == str(row)]
    if len(hits) != 1:
        die(f"{tsv}: {len(hits)} rows with row = {row}, expected 1")
    return rows, hits[0]


# --- file helpers ------------------------------------------------------------------------------

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


#: Nibi to S3 runs near 0.2 MB/s per connection and scales with connections (measured
#: 2026-11-11: 1 stream 0.20 MB/s, 8 streams 1.7 MB/s), so large files go in parallel ranges.
STREAMS = 32
CHUNK = 64 << 20


def _resolve(url):
    """Final URL after redirects, and its size (None when the server gives none)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": FU.USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        final = r.geturl()
        cr = r.headers.get("Content-Range", "")
    m = re.match(r"bytes 0-0/(\d+)$", cr)
    return final, (int(m.group(1)) if m else None)


def _fetch_ranges(url, part, streams=STREAMS, chunk=CHUNK):
    """Write url to part with up to `streams` parallel HTTP range requests (one stream if the
    server gives no size). Each range is retried; the caller checks the md5 of the whole."""
    from concurrent.futures import ThreadPoolExecutor
    final, size = _resolve(url)
    if size is None or size <= chunk:
        with urllib.request.urlopen(final, timeout=300) as r, open(part, "wb") as out:
            shutil.copyfileobj(r, out, 1 << 22)
        return
    with open(part, "wb") as out:
        out.truncate(size)
    state = {"url": final}

    def one(start):
        end = min(start + chunk, size) - 1
        for k in range(6):
            try:
                req = urllib.request.Request(state["url"], headers={"Range": f"bytes={start}-{end}",
                                                                   "User-Agent": FU.USER_AGENT})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"range {start}-{end}: {len(data)} bytes")
                fd = os.open(part, os.O_WRONLY)
                try:
                    os.pwrite(fd, data, start)
                finally:
                    os.close(fd)
                return len(data)
            except Exception as e:  # noqa: BLE001
                if k == 5:
                    raise
                if "403" in str(e):          # signed S3 URL expired: resolve again
                    state["url"] = _resolve(url)[0]
                time.sleep(5 * (k + 1))

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=streams) as ex:
        done = sum(ex.map(one, range(0, size, chunk)))
    log(f"fetched {done} bytes in {time.time() - t0:.0f} s ({done / max(time.time() - t0, 1) / 2**20:.1f} MB/s)")


def download(url, dest, md5=None, tries=4):
    """Stream url to dest (via dest.part); verify md5 when given. Skip if dest has that md5."""
    dest = Path(dest)
    if dest.exists() and (md5 is None or md5_file(dest) == md5):
        log(f"have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for i in range(tries):
        try:
            _fetch_ranges(url, part)
            got = md5_file(part)
            if md5 is not None and got != md5:
                raise IOError(f"md5 {got} != {md5}")
            os.replace(part, dest)
            log(f"got {dest.name} ({dest.stat().st_size} bytes)")
            return
        except Exception as e:  # noqa: BLE001
            log(f"download {url} attempt {i + 1}: {e}")
            if i == tries - 1:
                raise
            time.sleep(15 * (i + 1))


def sh(argv, cwd=None, check=True, **kw):
    log("$ " + " ".join(argv))
    p = subprocess.run(argv, cwd=cwd, **kw)
    if check and p.returncode != 0:
        die(f"command failed (exit {p.returncode}): {' '.join(argv)}")
    return p


def unpack_pipeline(tarball, dest) -> Path:
    """Extract the release tarball (top dir `dnase-seq-pipeline-3.0.0/`) and add the wrapper."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tf:
        top = {m.name.split("/")[0] for m in tf.getmembers()}
        if len(top) != 1:
            die(f"{tarball}: expected one top directory, got {sorted(top)[:5]}")
        tf.extractall(dest)
    root = dest / top.pop()
    if not (root / "dnase.wdl").exists():
        die(f"{root}: no dnase.wdl")
    (root / WRAPPER_NAME).write_text(wrapper_wdl())
    return root


def apptainer_bash(sif, script, binds, out_path=None):
    argv = ["apptainer", "exec", "--cleanenv"]
    for b in binds:
        argv += ["-B", b]
    argv += [str(sif), "bash", "-o", "pipefail", "-c", script]
    if out_path is None:
        return sh(argv, capture_output=True, text=True).stdout
    sh(argv)
    return None


# --- setup -------------------------------------------------------------------------------------

def cmd_setup(a):
    pdir, rdir, tmp = Path(a.pipeline_dir), Path(a.scratch_refs), Path(a.tmp)
    (pdir / "refs").mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    manifest = {"repository": PIPELINE_REPO, "tag": PIPELINE_TAG, "commit": PIPELINE_COMMIT,
                "image": IMAGE, "image_digest": IMAGE_DIGEST, "reference_set": REFERENCE_SET,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}

    tarball = pdir / TARBALL_NAME
    download(PIPELINE_TARBALL_URL, tarball)
    manifest["files"][TARBALL_NAME] = {"url": PIPELINE_TARBALL_URL,
                                       "sha256": sha256_file(tarball), "bytes": tarball.stat().st_size}

    sif = pdir / SIF_NAME
    if not sif.exists():
        env = dict(os.environ, APPTAINER_CACHEDIR=str(tmp / "apptainer_cache"),
                   APPTAINER_TMPDIR=str(tmp / "apptainer_tmp"))
        for k in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
            Path(env[k]).mkdir(parents=True, exist_ok=True)
        part = tmp / SIF_NAME
        sh(["apptainer", "pull", "--force", str(part), f"docker://encodedcc/dnase-seq-pipeline@{IMAGE_DIGEST}"],
           env=env)
        shutil.copyfile(part, sif)
    manifest["files"][SIF_NAME] = {"source": f"docker://encodedcc/dnase-seq-pipeline@{IMAGE_DIGEST}",
                                   "sha256": sha256_file(sif), "md5": md5_file(sif),
                                   "bytes": sif.stat().st_size}

    for acc, (otype, ext, md5, size) in REFERENCE_FILES.items():
        dest = pdir / "refs" / f"{acc}.{ext}"
        url = f"{ENCODE}/files/{acc}/@@download/{acc}.{ext}"
        download(url, dest, md5=md5)
        if dest.stat().st_size != size:
            die(f"{dest}: {dest.stat().st_size} bytes, portal says {size}")
        listing = []
        if ext == "tar.gz":
            with tarfile.open(dest) as tf:
                listing = [m.name for m in tf.getmembers() if m.isfile()]
        manifest["files"][f"refs/{acc}.{ext}"] = {"url": url, "output_type": otype, "md5": md5,
                                                  "bytes": size, "members": listing}

    # unpack the three references alignment needs; the tarballs on /project stay the authority
    for acc in (REF_INDEXED_FASTA, REF_BWA_INDEX):
        with tarfile.open(pdir / "refs" / f"{acc}.tar.gz") as tf:
            for m in tf.getmembers():
                if m.isfile() and not (rdir / Path(m.name).name).exists():
                    m.name = Path(m.name).name
                    tf.extract(m, rdir)
    with gzip.open(pdir / "refs" / f"{REF_NUCLEAR_CHROMS}.txt.gz", "rb") as src, \
            open(rdir / "nuclear_chroms.txt", "wb") as out:
        shutil.copyfileobj(src, out)
    refs = references_json(rdir)
    need = [refs["indexed_fasta"]["fai"], refs["nuclear_chroms"]] + list(refs["bwa_index"].values())
    missing = [p for p in need if not Path(p).exists()]
    if missing:
        die(f"unpacked references missing: {missing}")
    manifest["unpacked"] = {Path(p).name: {"path": p, "bytes": Path(p).stat().st_size}
                            for p in sorted(set(need))}

    # the wrapper must parse against the release's own WDL
    root = unpack_pipeline(tarball, tmp / "pipeline")
    p = sh(["java", "-jar", WOMTOOL, "validate", WRAPPER_NAME], cwd=root, check=False,
           capture_output=True, text=True)
    manifest["womtool_validate"] = {"rc": p.returncode, "out": (p.stdout + p.stderr).strip()[-2000:]}
    log(f"womtool validate {WRAPPER_NAME}: rc={p.returncode} {(p.stdout + p.stderr).strip()[-500:]}")
    p2 = sh(["java", "-jar", WOMTOOL, "validate", "dnase.wdl"], cwd=root, check=False,
            capture_output=True, text=True)
    manifest["womtool_validate_dnase_wdl"] = {"rc": p2.returncode}
    ver = apptainer_bash(sif, "samtools --version | head -1; bedtools --version; { bwa 2>&1 || true; } | grep Version",
                         binds=[])
    manifest["image_tools"] = ver.strip().splitlines()
    (pdir / "SETUP.json").write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"wrote {pdir / 'SETUP.json'}")
    if p.returncode != 0:
        die("womtool rejected the wrapper")


# --- run ---------------------------------------------------------------------------------------

def caper_conf(tmp, max_tasks) -> str:
    return "\n".join([
        "backend=local",
        f"local-out-dir={tmp}/cromwell_out",
        f"local-loc-dir={tmp}/cromwell_loc",
        f"cromwell={CROMWELL}",
        f"womtool={WOMTOOL}",
        f"max-concurrent-tasks={max_tasks}",
        "disable-auto-update-check=True",
        "",
    ])


def find_metadata(out_root) -> dict:
    found = sorted(Path(out_root).glob("dnase_align_only/*/metadata.json"),
                   key=lambda p: p.stat().st_mtime)
    if not found:
        die(f"no Cromwell metadata.json under {out_root}")
    return json.loads(found[-1].read_text())


def cmd_units(a):
    meta = fetch_metadata(a.experiment) if not a.metadata else json.loads(Path(a.metadata).read_text())
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(["experiment", "unit", "arms", "n_runs", "read_length_min", "paired"])
    for u in units(meta):
        runs = unit_runs(u)
        if not runs:
            continue
        w.writerow([a.experiment, u["unit"], ",".join(arms_for(u)), len(runs),
                    min(r["read_length"] for r in runs),
                    int(any(r["R2"] is not None for r in runs))])


def arm_record(experiment, unit, arm, reps, u, used_md5, row=None) -> dict:
    """arm.json in the ChIP/ATAC kit's form (fastq_uniform.arm_record), for align_views.py."""
    runs = unit_runs(u)
    pe = [bool(r.get("pe_fastqs")) and arm != "se" for r in reps]
    lengths = sorted({int(x["read_length"]) for x in runs})
    knob, value = {"base": ("", ""), "se": ("se", "1"), "crop36": ("crop", "36"),
                   "crop50": ("crop", "50")}[arm]
    return {
        "experiment": experiment, "unit": unit, "arm": arm, "layer": "uniform",
        "assay": "DNase-seq", "assay_class": "accessibility_dnase",
        "eic_track": (row or {}).get("eic_track", ""), "build_list_row": (row or {}).get("row", ""),
        "pipeline": "dnase-seq-pipeline", "pipeline_version": PIPELINE_TAG,
        "pipeline_commit": PIPELINE_COMMIT, "pipeline_type": "dnase", "image": IMAGE,
        "image_digest": IMAGE_DIGEST, "workflow": WRAPPER_NAME,
        "reps": [{"rep": i, "bio_rep": r["number"], "read_length": r["read_length"],
                  "paired_end": p, "se_fastqs": len(r.get("se_fastqs", [])),
                  "pe_pairs": len(r.get("pe_fastqs", []))}
                 for i, (r, p) in enumerate(zip(reps, pe), start=1)],
        "arm_knob": knob, "arm_value": value, "paired_end_per_rep": pe,
        "crop_length": CROPS.get(arm, 0),
        "run_type": ("paired-ended" if all(pe) else "single-ended" if not any(pe) else "mixed"),
        "read_length": min(r["read_length"] for r in reps),
        "read_lengths_source": lengths,
        "platforms": sorted({(x["R1"].get("platform") or "").strip("/").split("/")[-1]
                             for x in runs} - {""}),
        "dedup": "on", "mapq_threshold": MAPQ_THRESH, "fastq_md5": used_md5,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def cmd_run(a):
    rows = None
    row = None
    if a.build_list:
        rows, row = read_build_list(a.build_list, a.row)
        experiment, unit, arm = row["experiment"], row["unit"], row["arm"]
    else:
        experiment, unit, arm = a.experiment, a.unit, a.arm
    if not (experiment and unit and arm):
        die("give --experiment, --unit and --arm, or --build-list and --row")
    t0 = time.time()
    out = Path(a.work) / experiment / out_dir_name(unit, arm)
    if (out / "ALIGN_DONE").exists() and not a.force:
        log(f"{out}/ALIGN_DONE exists; skip")
        return
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    if a.metadata:
        meta = json.loads(Path(a.metadata).read_text())
    else:
        meta = fetch_metadata(experiment)
        (out / "encode_metadata.json").write_text(json.dumps(meta) + "\n")
    u = select_unit(meta, unit)
    fq_dir = Path(a.fastq_root) / experiment

    def fastq_path(f):
        return str(fq_dir / f"{f['accession']}.fastq.gz")

    inp = input_json(experiment, u, arm, fastq_path, a.scratch_refs)
    reps = replicates_json(experiment, u, arm, fastq_path)
    notes = {f"rep{i}": r["_adapters_note"] for i, r in enumerate(reps, 1)
             if "_adapters_note" in r}
    (out / "input.json").write_text(json.dumps(inp, indent=1) + "\n")
    bios = [r["number"] for r in reps]
    names = product_names(unit, bios)
    if a.dry_run:
        log(f"dry run: wrote {out / 'input.json'}")
        return

    # FASTQs from the shared staging dir (fastq_uniform.py stage); fetched there when absent
    need, used = [], {}
    for r in unit_runs(u):
        for f in (r["R1"], r["R2"]):
            if f is None or (arm == "se" and f is r["R2"]):
                continue
            used[f["accession"]] = f["md5sum"]
            url = (f.get("cloud_metadata") or {}).get("url") or ENCODE + f["href"]
            need.append({"url": url, "dest": fastq_path(f), "md5": f["md5sum"],
                         "bytes": int(f.get("file_size") or 0)})
    missing = [n for n in need if not (Path(n["dest"]).exists() and
                                       (not n["bytes"] or Path(n["dest"]).stat().st_size == n["bytes"]))]
    if missing:
        if all(n["bytes"] for n in missing):
            FU.fetch_many(missing, a.ranges, a.parallel_files)
        else:
            for n in missing:
                download(n["url"], n["dest"], md5=n["md5"])
    t_staged = time.time()
    inp_path = tmp / "input.json"
    inp_path.write_text(json.dumps(inp, indent=1) + "\n")

    root = unpack_pipeline(Path(a.pipeline_dir) / TARBALL_NAME, tmp / "pipeline")
    conf = tmp / "caper.conf"
    conf.write_text(caper_conf(tmp, a.max_concurrent_tasks))
    sif = Path(a.pipeline_dir) / SIF_NAME
    # Caper binds only the common roots of the input JSON's paths, so a task cannot follow the
    # soft link Cromwell makes to an upstream call's output under cromwell_out (smoke 26818194:
    # "pigz: abort: read error on concatenated.fastq.gz"). Apptainer adds these binds too.
    binds = sorted({str(tmp), os.path.realpath(tmp), "/scratch", "/project"})
    env = dict(os.environ, APPTAINER_BINDPATH=",".join(binds),
               SINGULARITY_BINDPATH=",".join(binds))
    sh(["caper", "run", WRAPPER_NAME, "-c", str(conf), "-b", "local", "-i", str(inp_path),
        "--singularity", str(sif), "--str-label", f"{experiment}_{out.name}"], cwd=root, env=env)
    t_aligned = time.time()

    md = find_metadata(tmp / "cromwell_out")
    if md.get("status") != "Succeeded":
        die(f"workflow status {md.get('status')}")
    outs = md["outputs"]
    unf, nuc = outs["dnase_align_only.unfiltered_bam"], outs["dnase_align_only.nuclear_bam"]
    dup = outs["dnase_align_only.duplication_metrics"]
    if not (len(unf) == len(nuc) == len(names)):
        die(f"{len(unf)} unfiltered and {len(nuc)} nuclear BAMs for {len(names)} replicates")
    n_reads = {}
    harvest = ["name\tsrc\tbytes\tmd5"]

    def keep(src, name):
        m = FU.copy_verified(src, out / name)
        harvest.append(f"{name}\t{src}\t{(out / name).stat().st_size}\t{m}")

    for name, ub, nb, dm in zip(names, unf, nuc, dup):
        keep(ub, f"{name}.unfiltered.bam")
        keep(nb, f"{name}.filtered.bam")
        keep(dm, f"{name}.duplication_metrics.txt")
        fb = out / f"{name}.filtered.bam"
        apptainer_bash(sif, f"samtools index {fb} {fb}.bai.part", binds, out_path=fb)
        os.replace(f"{fb}.bai.part", f"{fb}.bai")
        harvest.append(f"{name}.filtered.bam.bai\tsamtools index\t{Path(f'{fb}.bai').stat().st_size}"
                       f"\t{md5_file(f'{fb}.bai')}")
        ta = out / f"{name}.tagAlign.gz"
        apptainer_bash(sif, f"samtools view -b -F 1024 {fb} | "
                            f"bedtools bamtobed -i stdin | {TAGALIGN_AWK} | gzip -nc > {ta}.part",
                       binds, out_path=ta)
        os.replace(f"{ta}.part", ta)
        harvest.append(f"{name}.tagAlign.gz\tsamtools view -F 1024 | bamtobed\t"
                       f"{ta.stat().st_size}\t{md5_file(ta)}")
        with gzip.open(ta, "rb") as fh:
            n_reads[name] = sum(1 for _ in fh)
    (out / "HARVEST.tsv").write_text("\n".join(harvest) + "\n")
    (out / "metadata.json").write_text(json.dumps(md) + "\n")
    rec = arm_record(experiment, unit, arm, reps, u, used, row)
    rec["adapters_notes"] = notes
    (out / "arm.json").write_text(json.dumps(rec, indent=2) + "\n")

    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    done = {
        "experiment": experiment, "unit": unit, "arm": arm, "products": names,
        "cpu_hours": round((ru.ru_utime + ru.ru_stime) / 3600.0, 4),
        "cpu_hours_source": "getrusage(RUSAGE_CHILDREN) of this process; sacct of the job is the "
                            "authority",
        "wall_hours": round((t_aligned - t_staged) / 3600.0, 4),
        "stage_wall_hours": round((t_staged - t0) / 3600.0, 4),
        "total_wall_hours": round((time.time() - t0) / 3600.0, 4),
        "max_rss_gb": round(ru.ru_maxrss / 2**20, 3),
        "n_reads": sum(n_reads.values()), "n_reads_per_rep": n_reads,
        "n_reads_definition": "lines of the filtered tagAlign (one per read; 2 per pair)",
        "tagalign_rule": "filtered BAM without flag 0x400; bedtools bamtobed; one line per read",
        "pipeline": "dnase-seq-pipeline", "pipeline_version": PIPELINE_TAG,
        "pipeline_commit": PIPELINE_COMMIT, "image": IMAGE, "image_digest": IMAGE_DIGEST,
        "workflow": WRAPPER_NAME, "workflow_id": md.get("id"),
        "read_length": {f"rep{i}": r["read_length"] for i, r in enumerate(reps, 1)},
        "fastqs_md5": used, "adapters_notes": notes,
        "reference_set": REFERENCE_SET, "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (out / "ALIGN_DONE").write_text(json.dumps(done, indent=1) + "\n")
    log(f"ALIGN_DONE {out}: {json.dumps({k: done[k] for k in ('cpu_hours', 'wall_hours', 'n_reads')})}")
    if rows is not None:
        FU.cleanup_fastq(rows, experiment, a.work, a.fastq_root)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--pipeline-dir", default=PIPELINE_DIR)
    s.add_argument("--scratch-refs", default=SCRATCH_REFS)
    s.add_argument("--tmp", default=os.environ.get("SLURM_TMPDIR", "/tmp"))
    s = sub.add_parser("units")
    s.add_argument("--experiment", required=True)
    s.add_argument("--metadata")
    s = sub.add_parser("run")
    s.add_argument("--experiment")
    s.add_argument("--unit", default="pooled")
    s.add_argument("--arm", default="base")
    s.add_argument("--build-list")
    s.add_argument("--row", help="the build list's `row` column value")
    s.add_argument("--work", default=WORK)
    s.add_argument("--fastq-root", default=FU.FASTQ_ROOT)
    s.add_argument("--ranges", type=int, default=FU.RANGES)
    s.add_argument("--parallel-files", type=int, default=FU.PARALLEL_FILES)
    s.add_argument("--metadata", help="cached fetch_metadata() json (tests, reruns)")
    s.add_argument("--pipeline-dir", default=PIPELINE_DIR)
    s.add_argument("--scratch-refs", default=SCRATCH_REFS)
    s.add_argument("--tmp", default=os.environ.get("SLURM_TMPDIR", "/tmp"))
    s.add_argument("--max-concurrent-tasks", type=int, default=2)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    {"setup": cmd_setup, "units": cmd_units, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
