"""t167 (chunk D5) -- alignment-level view kit: one experiment's alignments -> measurement dirs.

For one experiment, from one source alignment file per biological replicate (a filtered tagAlign,
or a filtered BAM plus, for the BAM arms, the unfiltered BAM), this writes pinned-interface-B
measurement directories `<out>/<experiment>.<unit>.<layer>.<arm>/`:

    counts.h5  pr1.h5  pr2.h5  meta.json  provenance.json  [signal.h5 pr1_signal.h5 pr2_signal.h5]

`meta.json` holds this measurement's `measurements.tsv` row (`minicandi.dataset.MEASUREMENT_COLUMNS`,
all strings), i.e. its covariates. `provenance.json` holds sources, md5s, commands, the split, the
per-step cost and the job id. It is written last: a directory with it is complete (skip-if-exists).

THE SPLIT IS MADE ONCE PER PARENT, AT THE READ LEVEL, AND EVERY ARM REUSES IT. The pseudoreplicate
halves of a replicate's base read set are made first. Every arm is then built from the pr1 half
alone and from the pr2 half alone, so `<arm>/pr1` never shares a read with `<any arm>/pr2`
(addendum: scored correction pairs share no reads). An arm's `counts.h5` is `pr1 + pr2`.

* tagAlign source: the pipeline's own `encode_task_spr.py` in the pipeline SIF, seed 0 (the
  effective seed is then the input's uncompressed byte count), as t117 did (`tools/t117`), so the
  halves of a testbed tagAlign equal t117's halves. `--splitter python` is a seeded exact split for
  laptop tests; it is never the production split.
* BAM source: a tagAlign line carries no read name (`bam2ta` writes "N"), so spr's line split cannot
  follow a read into the nodedup or mapq0 read sets. Here every read set of the replicate is turned
  into a NAMED tagAlign (`bedtools bamtobed` plus the pipeline's awk, with the name kept), and each
  fragment goes to pr1 or pr2 by a seeded hash of its read name. A read present in two arms lands in
  the same half in both. Names are replaced by "N" in the half files, as `bam2ta` writes them.

Arms (plan D5): `base`; `depth0.50`, `depth0.25`, `depth0.125` by read-level nested thinning of each
half (never bin-level; exact counts round(q * n_fragments) per half, seeded; `reads_subset_of` set);
`nodedup` and `mapq0` by re-running the pipeline's own filter step (`encode_task_filter.py`, in the
SIF) on the unfiltered BAM with that one setting changed. `mapq0` is refused when the filter runs
with `--multimapping > 0` (the ATAC default 4), because the filter then ignores `--mapq-thresh`.
Units: one source -> `pooled` only (it is the only replicate). Two or more -> `br<i>` per replicate
and `pooled` as their union (pooled pr1 = the union of every replicate's pr1).

Counts: `tools/t112/bin25.py::counts_from_tagalign` (store overlap rule), 25 bp, the genome layer's
floor grid, uint32, chunks of at most 4096 bins. Signal (`--signal`, validation and blind
experiments): the pipeline's MACS2 signal step on the pooled base and on its two halves, the
recorded command of the base run (`--metadata`) with only the tagAligns swapped
(`bam_arm.build_signal_command`), the pval bigwig binned by `bin25.pval_from_bigwig`.

    python align_views.py build  --experiment ENCSR.. --source <path> [--source <path> ...] \\
        --assay H3K27ac --cell C19 --work /scratch/mforooz/onion/work \\
        --out /scratch/mforooz/onion/measurements [--signal --metadata M --control-tagalign C]
    python align_views.py verify --measurements DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
for _p in (TOOLS, TOOLS / "t112", TOOLS / "t117"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import bam_arm  # noqa: E402  (tools/t112, stdlib only)
import bin25  # noqa: E402
import records  # noqa: E402

GENOME_SIZES = "/project/def-maxwl/mforooz/ONION/genome/chrom_sizes.json"
RES = 25
CHUNK = 4096
HALVES = ("pr1", "pr2")
#: plan D5: the thinning levels, as written into the arm name.
DEPTH_LEVELS = (("0.50", 0.5), ("0.25", 0.25), ("0.125", 0.125))
BAM_ARMS = ("nodedup", "mapq0")
#: FASTQ-level arms (fastq_uniform / fastq_dnase); here they are only a source read set's label.
FASTQ_ARMS = ("se", "crop36", "crop50")
ALL_ARMS = ("base",) + tuple(f"depth{v}" for v, _ in DEPTH_LEVELS) + BAM_ARMS
PIPELINE_NAME = {"chip": "chip-seq-pipeline2", "atac": "atac-seq-pipeline",
                 "dnase": "dnase-seq-pipeline"}
PIPELINE_VERSION = {"chip": "v2.2.2", "atac": "v2.2.3", "dnase": "v3.0.0"}
#: the base run's MAPQ threshold: chip.wdl/atac.wdl `mapq_thresh` 30; DNase: stampipes
#: filter_reads.py --min_mapq default 10 (dnase-seq-pipeline v3.0.0 image)
BASE_MAPQ = {"chip": 30, "atac": 30, "dnase": 10}
DNASE_SIF = "/project/def-maxwl/mforooz/ONION/pipelines/dnase/dnase-seq-pipeline_v3.0.0.sif"
DNASE_SIF_SHA256 = "2cd73eb14798f90a3372eafdd5549768c0ae705da9c1a3d054cc0e6921680d1d"
DNASE_NUCLEAR_CHROMS = "/scratch/mforooz/onion/pipelines/dnase/refs/nuclear_chroms.txt"
#: atac-seq-pipeline v2.2.3 `encode_task_bam2ta.py::tn5_shift_ta`, verbatim (bam2ta is called
#: with the shift on for pipeline_type atac, atac.wdl:1204)
TN5_AWK = ("""awk 'BEGIN {OFS = "\\t"} {if ($6 == "+") {$2 = $2 + 4} else if ($6 == "-") """
           """{$3 = $3 - 5} if ($2 >= $3) { if ($6 == "+") {$2 = $3 - 1} else {$3 = $2 + 1} } """
           """print $0}'""")
#: `encode_task_filter.py` settings of a default run (chip.wdl call recorded in
#: EIC_REPRO results/C19M16/metadata.json; atac.wdl v2.2.3 input defaults).
FILTER_DEFAULTS = {
    "chip": {"multimapping": 0, "dup_marker": "picard", "mapq_thresh": 30, "filter_chrs": [],
             "mito_chr_name": "chrM"},
    "atac": {"multimapping": 4, "dup_marker": "picard", "mapq_thresh": 30,
             "filter_chrs": ["chrM", "MT"], "mito_chr_name": "chrM"},
}
SE_NAMED_AWK = """awk 'BEGIN{OFS="\\t"}{$5="1000";print $0}'"""
PE_NAMED_AWK = ("""awk 'BEGIN{OFS="\\t"}{printf "%s\\t%s\\t%s\\t%s\\t1000\\t%s\\n"""
                """%s\\t%s\\t%s\\t%s\\t1000\\t%s\\n",$1,$2,$3,$7,$9,$4,$5,$6,$7,$10}'""")


# --------------------------------------------------------------------------------------------
# small helpers


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def key(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_sizes(path) -> dict:
    """Chromosome sizes from the genome layer's `chrom_sizes.json`, or a 2-column TSV
    (`bin25.load_chrsz`, main chromosomes)."""
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        return {c: int(v) for c, v in d.get("chrom_sizes", d).items()}
    return bin25.load_chrsz(p)


def git_sha() -> str:
    kit = TOOLS.parent
    stamp = kit / ".kit_stamp"
    if stamp.is_file():
        return stamp.read_text().strip().splitlines()[0]
    out = subprocess.run(["git", "-C", str(kit), "rev-parse", "HEAD"], capture_output=True,
                         text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


class Cost:
    """Per-step wall, CPU and peak RSS. A shell step is measured with `os.wait4` on its own
    process tree; an in-process step by `getrusage(RUSAGE_SELF)` deltas (peak = process peak)."""

    def __init__(self):
        self.steps = []

    def shell(self, name: str, cmd: str, cwd=None):
        print(f"+ {cmd}", flush=True)
        t0 = time.time()
        p = subprocess.Popen(["/bin/bash", "-o", "pipefail", "-c", cmd], cwd=cwd)
        _, status, ru = os.wait4(p.pid, 0)
        p.returncode = os.waitstatus_to_exitcode(status)
        self.steps.append({"step": name, "kind": "shell", "wall_s": round(time.time() - t0, 2),
                           "cpu_s": round(ru.ru_utime + ru.ru_stime, 2),
                           "max_rss_gb": round(ru.ru_maxrss / 2**20, 3)})
        if p.returncode:
            raise SystemExit(f"step {name} failed ({p.returncode}): {cmd}")

    def parallel(self, jobs: list):
        """Run [(name, cmd, cwd)] at the same time; each process tree measured by `os.wait4`."""
        t0, procs = time.time(), {}
        for name, cmd, cwd in jobs:
            print(f"+ [{name}] {cmd}", flush=True)
            p = subprocess.Popen(["/bin/bash", "-o", "pipefail", "-c", cmd], cwd=cwd)
            procs[p.pid] = (name, cmd, time.time())
        failed = []
        while procs:
            pid, status, ru = os.wait4(-1, 0)
            if pid not in procs:
                continue
            name, cmd, ts = procs.pop(pid)
            self.steps.append({"step": name, "kind": "shell", "parallel": True,
                               "wall_s": round(time.time() - ts, 2),
                               "cpu_s": round(ru.ru_utime + ru.ru_stime, 2),
                               "max_rss_gb": round(ru.ru_maxrss / 2**20, 3)})
            if os.waitstatus_to_exitcode(status):
                failed.append(f"{name}: {cmd}")
        self.steps.append({"step": "parallel_group", "kind": "wall",
                           "members": [j[0] for j in jobs], "wall_s": round(time.time() - t0, 2)})
        if failed:
            raise SystemExit("parallel steps failed: " + "; ".join(failed))

    def py(self, name: str, fn, *a, **kw):
        r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.time()
        out = fn(*a, **kw)
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        self.steps.append({"step": name, "kind": "python", "wall_s": round(time.time() - t0, 2),
                           "cpu_s": round(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime, 2),
                           "max_rss_gb": round(r1.ru_maxrss / 2**20, 3)})
        return out


def _reader(path):
    return subprocess.Popen(["gzip", "-dcf", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20)


class _Writer:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = Path(path)
        self.part = Path(f"{path}.part")
        self._f = open(self.part, "wb")
        self._p = subprocess.Popen(["gzip", "-nc"], stdin=subprocess.PIPE, stdout=self._f,
                                   bufsize=1 << 20)
        self.write = self._p.stdin.write
        self.n = 0

    def close(self):
        self._p.stdin.close()
        if self._p.wait():
            raise SystemExit(f"gzip failed writing {self.path}")
        self._f.close()
        os.replace(self.part, self.path)


def _finish(proc, path):
    proc.stdout.close()
    if proc.wait():
        raise SystemExit(f"gzip -dcf failed reading {path}")


def count_lines(path) -> int:
    out = subprocess.run(f"gzip -dcf {shlex.quote(str(path))} | wc -l", shell=True, check=True,
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


# --------------------------------------------------------------------------------------------
# read-level operations (pure python; all streaming)


def _units(path, lpu: int):
    """Yield the source's fragments in file order: one line (SE) or two consecutive lines (PE,
    as `bam2ta_pe` writes a pair)."""
    p = _reader(path)
    it = iter(p.stdout)
    try:
        if lpu == 1:
            for line in it:
                yield line
        else:
            for a in it:
                b = next(it, None)
                if b is None:
                    raise SystemExit(f"{path}: odd line count with --paired-end")
                yield a + b
    finally:
        _finish(p, path)


def split_python(src, out1, out2, n_lines: int, lpu: int, rng) -> dict:
    """Seeded exact split of fragments: pr1 gets ceil(n/2) as `encode_task_spr.py` does. Laptop
    tests and fallback only; production tagAlign sources use spr in the SIF."""
    n = n_lines // lpu
    rank = rng.permutation(n)
    code = (rank < (n + 1) // 2).astype(np.uint8).tobytes()
    w = {1: _Writer(out1), 0: _Writer(out2)}
    u = 0
    for frag in _units(src, lpu):
        w[code[u]].write(frag)
        w[code[u]].n += lpu
        u += 1
    if u != n:
        raise SystemExit(f"{src}: {u} fragments read, {n} expected")
    for x in w.values():
        x.close()
    return {"pr1": w[1].n, "pr2": w[0].n}


def name_key(name: bytes) -> bytes:
    """The fragment's name: `bedtools bamtobed` (no -bedpe) appends /1 or /2 to a paired read's
    name, so a trailing /1 or /2 is dropped, uniformly, from every name of every arm."""
    return name[:-2] if name[-2:] in (b"/1", b"/2") else name


def name_half(name: bytes, salt: bytes) -> int:
    """0 -> pr1, 1 -> pr2: the low bit of a seeded hash of the read (fragment) name."""
    return hashlib.blake2b(salt + name_key(name), digest_size=1).digest()[0] & 1


def name_token(name: bytes) -> bytes:
    """64-bit hex token of a fragment name (the `--name-tokens` check mode keeps it in column 4)."""
    return hashlib.blake2b(name_key(name), digest_size=8).hexdigest().encode()


def split_salt(seed: int, experiment: str) -> bytes:
    """ONE split per experiment: the same read lands in the same half in every unit (pooled, br<i>,
    library, lane) and every arm (base, depth, nodedup, mapq0, se, crop) of the experiment."""
    return f"{seed}|{experiment}|".encode()


def split_named(src, out1, out2, salt: bytes, tokens: bool = False) -> dict:
    """Split a NAMED tagAlign (column 4 = read name) by `name_half`; both mates of a fragment carry
    its name, so a fragment never straddles the halves. Column 4 is written as "N" (`bam2ta`), or
    as `name_token` when `tokens` (verification runs only: lets `verify` prove disjointness)."""
    w = (_Writer(out1), _Writer(out2))
    p = _reader(src)
    for line in p.stdout:
        f = line.split(b"\t")
        x = w[name_half(f[3], salt)]
        f[3] = name_token(f[3]) if tokens else b"N"
        x.write(b"\t".join(f))
        x.n += 1
    _finish(p, src)
    for x in w:
        x.close()
    return {"pr1": w[0].n, "pr2": w[1].n}


def thin_nested(src, outs: dict, n_lines: int, lpu: int, rng) -> dict:
    """Read-level nested thinning of one half. `outs` maps level label -> path, levels from
    `DEPTH_LEVELS`. One seeded permutation ranks the fragments; level q keeps the fragments of rank
    < round(q * n). So every level is an exact-size subset of the next larger one."""
    n = n_lines // lpu
    rank = rng.permutation(n)
    levels = sorted(((q, lab) for lab, q in DEPTH_LEVELS if lab in outs), reverse=True)
    ks = [round_half_up(q * n) for q, _ in levels]
    code = np.zeros(n, dtype=np.uint8)
    for k in ks:
        code += rank < k
    code = code.tobytes()
    w = [_Writer(outs[lab]) for _, lab in levels]
    u = 0
    for frag in _units(src, lpu):
        for j in range(code[u]):
            w[j].write(frag)
            w[j].n += lpu
        u += 1
    if u != n:
        raise SystemExit(f"{src}: {u} fragments read, {n} expected")
    for x in w:
        x.close()
    return {lab: x.n for (_, lab), x in zip(levels, w)}


# --------------------------------------------------------------------------------------------
# pipeline steps inside the SIF (cluster only)


def sif_info(sif: str, pipeline: str) -> dict:
    if pipeline == "dnase":
        sha = sha256_file(sif)
        if sha != DNASE_SIF_SHA256:
            raise SystemExit(f"{sif}: sha256 {sha} != pinned {DNASE_SIF_SHA256}")
        return {"path": sif, "md5": records.md5_file(sif), "sha256": sha}
    md5, sha = bam_arm.check_sif(sif, records.PIPELINE_REPO[pipeline])
    return {"path": sif, "md5": md5, "sha256": sha}


def default_sif(pipeline: str) -> str:
    return DNASE_SIF if pipeline == "dnase" else bam_arm.SIF[pipeline]


def spr_split(cost: Cost, ta, out1, out2, *, sif, tmp, paired_end) -> dict:
    """`encode_task_spr.py` on `ta`, seed 0, as t117 (`pseudoreps.spr_command`)."""
    import pseudoreps
    spr_md5 = subprocess.run(
        f"apptainer exec --cleanenv {sif} bash -c 'md5sum $(which {pseudoreps.SPR_SCRIPT})'",
        shell=True, check=True, capture_output=True, text=True).stdout.split()[0]
    if spr_md5 != pseudoreps.SPR_MD5:
        raise SystemExit(f"{pseudoreps.SPR_SCRIPT} md5 {spr_md5} != pinned {pseudoreps.SPR_MD5}")
    out_dir = tempfile.mkdtemp(prefix="spr_", dir=tmp)
    Path(f"{tmp}/work").mkdir(parents=True, exist_ok=True)
    cmd = pseudoreps.spr_command({"seed": 0, "paired_end": paired_end}, str(ta), out_dir, sif,
                                 tmp)
    cost.shell("split_spr", cmd, cwd=f"{tmp}/work")
    n = {}
    for h, dst in zip(HALVES, (out1, out2)):
        src = f"{out_dir}/{pseudoreps.pr_name(ta, h)}"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, f"{dst}.part")
        os.replace(f"{dst}.part", dst)
        n[h] = count_lines(dst)
    shutil.rmtree(out_dir, ignore_errors=True)
    return {"lines": n, "command": cmd, "spr_script_md5": spr_md5,
            "effective_seed": int(subprocess.run(
                f"gzip -dcf {shlex.quote(str(ta))} | wc -c", shell=True, check=True,
                capture_output=True, text=True).stdout.split()[0])}


def filter_command(bam, out_dir, *, pipeline, arm, paired_end, chrsz, nth, mem_gb,
                   multimapping=None) -> str:
    """The pipeline's own filter step on the unfiltered BAM with ONE setting changed (`arm` base:
    none changed -- the native route makes its base this way)."""
    if pipeline == "dnase":
        raise ValueError("dnase: use dnase_filter_command")
    s = dict(FILTER_DEFAULTS[pipeline])
    if multimapping is not None:
        s["multimapping"] = multimapping
    nodup = False
    if arm == "nodedup":
        nodup = True
    elif arm == "mapq0":
        if s["multimapping"] > 0:
            raise SystemExit("mapq0 refused: with --multimapping > 0 encode_task_filter.py "
                             "ignores --mapq-thresh (plan D5 trap)")
        s["mapq_thresh"] = 0
    elif arm != "base":
        raise ValueError(arm)
    return (f"python3 $(which encode_task_filter.py) {bam}"
            f"{' --paired-end' if paired_end else ''} --multimapping {s['multimapping']}"
            f" --dup-marker {s['dup_marker']} --mapq-thresh {s['mapq_thresh']}"
            f" --filter-chrs {' '.join(s['filter_chrs'])} --chrsz {chrsz}"
            f"{' --no-dup-removal' if nodup else ''} --mito-chr-name {s['mito_chr_name']}"
            f" --mem-gb {mem_gb} --nth {nth} --picard-java-heap {max(1, int(mem_gb))}G"
            f" --out-dir {out_dir}")


#: dnase-seq-pipeline v3.0.0 `merge_mark_and_filter_bams` after its merge, task by task with the
#: release's parameters (wdl/workflows/mixed/{mark,filter}.wdl and the subworkflows they call;
#: machine sizes large/large-low-cpu): flag (stampipes filter_reads.py), coordinate sort, mate
#: cigar, MarkDuplicatesWithMateCigar, drop QC-fail (-F 512), drop non-nuclear (-F 4096).
DNASE_MARKDUP_REGEX = "'[a-zA-Z0-9]+:[0-9]+:[a-zA-Z0-9]+:[0-9]+:([0-9]+):([0-9]+):([0-9]+).*'"


def dnase_filter_command(unfiltered, out_dir, *, min_mapq, nth, nuclear_chroms=DNASE_NUCLEAR_CHROMS,
                         heap_gb=12) -> str:
    """The DNase pipeline's mark-and-filter on its merged (name-sorted, "unfiltered") BAM with
    `filter_reads.py --min_mapq` set; min_mapq 10 is the release default (an identity re-run).
    Writes `<out_dir>/nuclear.bam` (QC-fail and non-nuclear reads removed, duplicates flagged)."""
    o = str(out_dir)
    return " && ".join([
        f"cd {o}",
        f"python3 $(which filter_reads.py) {unfiltered} flagged.bam {nuclear_chroms}"
        f" --min_mapq {int(min_mapq)}",
        f"samtools sort -@ {nth} flagged.bam > sorted.bam",
        "rm -f flagged.bam",
        "java -jar $(which picard.jar) RevertOriginalBaseQualitiesAndAddMateCigar INPUT=sorted.bam"
        " OUTPUT=mate_cigar.bam RESTORE_ORIGINAL_QUALITIES=false VALIDATION_STRINGENCY=SILENT"
        " MAX_RECORDS_TO_EXAMINE=0",
        "rm -f sorted.bam",
        f"java -Xmx{int(heap_gb)}G -XX:ParallelGCThreads=1 -jar $(which picard.jar)"
        " MarkDuplicatesWithMateCigar INPUT=mate_cigar.bam OUTPUT=marked.bam"
        f" METRICS_FILE=MarkDuplicates.picard ASSUME_SORTED=true MINIMUM_DISTANCE=300"
        f" READ_NAME_REGEX={DNASE_MARKDUP_REGEX} VALIDATION_STRINGENCY=SILENT",
        "rm -f mate_cigar.bam",
        f"samtools view -F 512 -h -b -@ {nth} marked.bam > qc_fail_flag_filtered.bam",
        "rm -f marked.bam",
        f"samtools view -F 4096 -h -b -@ {nth} qc_fail_flag_filtered.bam > nuclear.bam",
        "rm -f qc_fail_flag_filtered.bam",
    ])


def named_ta_command(bam, out, *, paired_end, tmp, nth, pipeline="chip", drop_dups=False) -> str:
    """BAM -> tagAlign with read names kept: the pipeline's `bam2ta` (bamtobed and awk) minus
    `$4="N"`; atac adds the pipeline's Tn5 shift; dnase is the DNase kit's tagAlign rule (one line
    per read, from the nuclear BAM, flag 0x400 dropped when `drop_dups`)."""
    tn5 = f" | {TN5_AWK}" if pipeline == "atac" else ""
    if pipeline == "dnase":
        src = f"samtools view -b -F 1024 {bam} | bedtools bamtobed -i stdin" if drop_dups \
            else f"bedtools bamtobed -i {bam}"
        return f"{src} | {SE_NAMED_AWK} | gzip -nc > {out}"
    if not paired_end:
        return f"bedtools bamtobed -i {bam} | {SE_NAMED_AWK}{tn5} | gzip -nc > {out}"
    srt = f"{tmp}/nmsrt.{Path(str(bam)).stem}.{os.getpid()}.bam"
    return (f"samtools sort -n -@ {nth} -o {srt} {bam} && LC_COLLATE=C bedtools bamtobed -bedpe "
            f"-mate1 -i {srt} | {PE_NAMED_AWK}{tn5} | gzip -nc > {out} && rm -f {srt}")


#: Runs a pipeline signal script with its fold-enrichment branch skipped: every shell command of
#: the script runs as written except the four FE ones (`bdgcmp -m FE`, its slop/clip, sort and
#: bigwig), whose output (`*.fc.signal.bigwig`) nothing here reads. The p-value bigwig is the
#: script's own, byte for byte (checked on the D5 smoke's recorded md5s). FE is ~40% of a run.
SKIP_FE_PY = ("import importlib.util,os,sys;p=sys.argv[1];sys.path.insert(0,os.path.dirname(p));"
              "s=importlib.util.spec_from_file_location('sigtrack',p);"
              "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);o=m.run_shell_cmd;"
              "m.run_shell_cmd=lambda c,*a,**k:None if ('-m FE' in c or '_FE.bdg' in c "
              "or '.fc.signal' in c) else o(c,*a,**k);sys.argv=[p]+sys.argv[2:];m.main()")
SIGNAL_SCRIPT = {"chip": "encode_task_macs2_signal_track_chip.py",
                 "atac": "encode_task_macs2_signal_track_atac.py"}
#: pipeline defaults (chip.wdl: pval_thresh 0.01; atac.wdl: smooth_win 150, pval_thresh 0.01).
#: DNase-seq: PI ruling 19, the MACS2 no-control route = the ATAC recipe (--shift -75
#: --extsize 150, no control) run from the atac image on the DNase tagAlign.
SIGNAL_PVAL_THRESH = 0.01
SIGNAL_SMOOTH_WIN = 150


def pval_only_invocation(script: str) -> list:
    return ["python3", "-c", shlex.quote(SKIP_FE_PY), f"$(which {script})"]


def signal_inner(pipeline, ta, out_dir, *, chrsz, ctl=None, fraglen=None, ctl_subsample=0,
                 ctl_paired_end=False, mem_gb=8.0) -> str:
    """The signal step of `pipeline` (dnase -> the atac script, ruling 19) on one tagAlign, with
    the pipeline's default flags, FE branch skipped (`SKIP_FE_PY`)."""
    script = SIGNAL_SCRIPT["chip" if pipeline == "chip" else "atac"]
    parts = pval_only_invocation(script) + [str(ta)]
    if pipeline == "chip":
        if ctl is None or fraglen is None:
            raise SystemExit("chip signal needs a control tagAlign and the fragment length")
        parts += [str(ctl), "--gensz hs", f"--chrsz {chrsz}", f"--fraglen {int(fraglen)}",
                  f"--pval-thresh {SIGNAL_PVAL_THRESH}", f"--mem-gb {mem_gb}"]
        if ctl_subsample:
            parts += [f"--ctl-subsample {int(ctl_subsample)}"]
            if ctl_paired_end:
                parts += ["--ctl-paired-end"]
    else:
        parts += ["--gensz hs", f"--chrsz {chrsz}", f"--pval-thresh {SIGNAL_PVAL_THRESH}",
                  f"--smooth-win {SIGNAL_SMOOTH_WIN}", f"--mem-gb {mem_gb}"]
    return " ".join(parts + [f"--out-dir {out_dir}"])


def signal_command(tas, out_dir, *, pipeline, metadata, chrsz) -> tuple:
    """(inner command, bigwig path): the base run's recorded MACS2 signal command with only the
    tagAligns, `--chrsz` and `--out-dir` set (`bam_arm.build_signal_command`), FE skipped."""
    meta = json.loads(Path(metadata).read_text())
    calls = meta["calls"][bam_arm.SIGNAL_CALL[pipeline]]
    if len(calls) != 1:
        raise SystemExit(f"{metadata}: {len(calls)} signal calls, expected 1")
    parsed = bam_arm.parse_signal_command(calls[0]["commandLine"], pipeline)
    inner = bam_arm.build_signal_command(parsed, [str(t) for t in tas], out_dir, chrsz=chrsz,
                                         fraglen_flag=bam_arm.FRAGLEN_FLAG[pipeline],
                                         invocation=pval_only_invocation(SIGNAL_SCRIPT[pipeline]))
    return inner, bam_arm.signal_bigwig([str(t) for t in tas], out_dir)


# --------------------------------------------------------------------------------------------
# h5 products


def write_h5(path, arrays: dict, attrs: dict, dtype) -> None:
    import h5py
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for c, v in arrays.items():
            f.create_dataset(c, data=v.astype(dtype), chunks=(max(1, min(CHUNK, v.size)),),
                             compression="gzip", compression_opts=4, shuffle=True)
        f.attrs.update(attrs)


def read_h5(path, chroms=None) -> dict:
    import h5py
    with h5py.File(path, "r") as f:
        return {c: f[c][()] for c in (chroms or list(f.keys()))}


def counts_attrs(mid, n_lines, lpu) -> dict:
    return {"schema": 1, "measurement_id": mid, "depth_reads": int(n_lines),
            "depth_fragments": int(n_lines // lpu), "resolution": RES,
            "counting_rule": records.COUNT_RULE}


# --------------------------------------------------------------------------------------------
# build


def measurement_id(exp, unit, layer, arm) -> str:
    return f"{exp}.{unit}.{layer}.{arm}"


class ControlPending(Exception):
    """The signal of a ChIP experiment needs its control's read sets, which are not built yet."""


def arm_fields(arm: str, base: dict) -> dict:
    """The covariates an arm changes: (arm_knob, arm_value, dedup, mapq_threshold)."""
    same = {"dedup": base["dedup"], "mapq_threshold": base["mapq_threshold"]}
    if arm == "base":
        return {"arm_knob": "", "arm_value": "", **same}
    if arm.startswith("depth"):
        return {"arm_knob": "depth", "arm_value": arm[len("depth"):], **same}
    if arm == "nodedup":
        return {"arm_knob": "nodedup", "arm_value": "", "dedup": "0",
                "mapq_threshold": base["mapq_threshold"]}
    if arm == "mapq0":
        return {"arm_knob": "mapq", "arm_value": "0", "dedup": base["dedup"],
                "mapq_threshold": "0"}
    if arm == "se":
        return {"arm_knob": "se", "arm_value": "1", **same}
    if arm.startswith("crop"):
        return {"arm_knob": "crop", "arm_value": arm[len("crop"):], **same}
    raise ValueError(arm)


def make_meta(cfg: dict, unit: str, arm: str, depth_reads: int, lpu: int, *, has_signal: bool,
              counts_path, sha256: str, src: dict | None = None) -> dict:
    from minicandi import contract as C
    from minicandi.dataset import MEASUREMENT_COLUMNS
    exp, layer = cfg["experiment"], cfg["layer"]
    label = cfg.get("source_arm", "base")
    mid = measurement_id(exp, unit, layer, arm)
    base_mid = measurement_id(exp, unit, layer, "base")
    pooled_base = measurement_id(exp, "pooled", layer, "base")
    src = src or {}
    run_type = src.get("run_type") or cfg["run_type"]
    read_length = src.get("read_length") or cfg["read_length"]
    if arm == "se":
        run_type = "single-ended"
    if arm.startswith("crop"):
        read_length = arm[len("crop"):]
    m = re.match(r"^br(\d+)", unit)
    row = {k: "" for k in MEASUREMENT_COLUMNS}
    row.update(cfg["meta"])
    row.update({
        "measurement_id": mid, "experiment": exp, "assay": cfg["assay"],
        "assay_class": C.assay_class(cfg["assay"]), "cell": cfg["cell"],
        "eic_track": cfg["eic_track"], "in_eic": cfg["role"], "unit": unit,
        "bio_rep": m.group(1) if m else "", "layer": layer,
        "parent_measurement_id": "" if arm == "base" else base_mid,
        "reads_subset_of": (base_mid if arm.startswith("depth")
                            else pooled_base if arm == label == "base" and unit != "pooled"
                            and re.fullmatch(r"br\d+", unit) else ""),
        "depth_reads": str(depth_reads), "depth_fragments": str(depth_reads // lpu),
        "run_type": run_type, "read_length": str(read_length),
        "pipeline": cfg.get("pipeline_name") or PIPELINE_NAME[cfg["pipeline"]],
        "pipeline_version": (cfg.get("pipeline_version")
                             or records.PIPELINES[records.PIPELINE_REPO[cfg["pipeline"]]][0]),
        "image_digest": cfg.get("image_digest", ""),
        "control_measurement_id": cfg.get("control_measurement_id", ""),
        "has_pseudoreps": "1", "has_signal": "1" if has_signal else "0",
        "counts_path": f"h5:{Path(counts_path).resolve()}", "sha256": sha256,
    })
    row.update(arm_fields(arm, {"dedup": cfg["dedup"], "mapq_threshold": cfg["mapq_threshold"]}))
    return {k: str(row[k]) for k in MEASUREMENT_COLUMNS}


def readset_dir(cfg, unit, arm) -> Path:
    return Path(cfg["work"]) / cfg["experiment"] / f"{unit}.{cfg['layer']}" / arm


def _named_split(cfg, cost, unit, arm, bam, *, drop_dups, label=None) -> dict:
    """BAM -> named tagAlign (in tmp) -> the experiment-level name split into the arm's halves."""
    tmp, exp = cfg["tmp"], cfg["experiment"]
    named = Path(tmp) / f"{exp}.{unit}.{arm}.named.tagAlign.gz"
    pe = cfg["_src_pe"][unit]
    cost.shell(f"named_ta_{arm}_{unit}", bam_arm.apptainer(
        named_ta_command(bam, named, paired_end=pe, tmp=tmp, nth=cfg["nth"],
                         pipeline=cfg["pipeline"], drop_dups=drop_dups), cfg["sif"], tmp, tmp))
    d = readset_dir(cfg, unit, label or arm)
    lines = cost.py(f"split_named_{arm}_{unit}", split_named, named, d / "pr1.tagAlign.gz",
                    d / "pr2.tagAlign.gz", split_salt(cfg["seed"], exp), cfg.get("name_tokens"))
    named.unlink()
    return {"lines": lines,
            "split": {"method": "blake2b(salt + read name) low bit", "salt":
                      split_salt(cfg["seed"], exp).decode(),
                      "name_tokens": bool(cfg.get("name_tokens")),
                      "named_ta_command": named_ta_command(bam, named, paired_end=pe, tmp=tmp,
                                                           nth=cfg["nth"],
                                                           pipeline=cfg["pipeline"],
                                                           drop_dups=drop_dups)}}


def _filtered_bam(cfg, cost, unit, arm, unfiltered) -> tuple:
    """Re-run the pipeline's filter on the unfiltered BAM with the arm's one setting; returns
    (bam path, command, tmp dir to remove)."""
    tmp = cfg["tmp"]
    fdir = Path(tempfile.mkdtemp(prefix=f"filter_{arm}_{unit}_", dir=tmp))
    if cfg["pipeline"] == "dnase":
        mq = {"mapq0": 0, "base": BASE_MAPQ["dnase"]}[arm]
        fcmd = dnase_filter_command(unfiltered, fdir, min_mapq=mq, nth=cfg["nth"],
                                    nuclear_chroms=cfg.get("nuclear_chroms", DNASE_NUCLEAR_CHROMS))
        cost.shell(f"filter_{arm}_{unit}", bam_arm.apptainer(fcmd, cfg["sif"], tmp, str(fdir)))
        return fdir / "nuclear.bam", fcmd, fdir
    fcmd = filter_command(unfiltered, fdir, pipeline=cfg["pipeline"], arm=arm,
                          paired_end=cfg["_src_pe"][unit], chrsz=cfg["chrsz"], nth=cfg["nth"],
                          mem_gb=cfg["mem_gb"], multimapping=cfg["multimapping"])
    cost.shell(f"filter_{arm}_{unit}", bam_arm.apptainer(fcmd, cfg["sif"], tmp, str(fdir)))
    return filter_output(fdir, arm, FILTER_DEFAULTS[cfg["pipeline"]]["filter_chrs"]), fcmd, fdir


def filter_output(fdir, arm, filter_chrs) -> Path:
    """The final BAM `encode_task_filter.py` leaves: `*.nodup.bam` (`*.filt.bam` with
    --no-dup-removal), then `.no_<chrs>.bam` appended when --filter-chrs is set (atac: chrM MT,
    KF job 26830521: `rep1.unfiltered.nodup.no_chrM_MT.bam`)."""
    want = ".filt" if arm == "nodedup" else ".nodup"
    if filter_chrs:
        want += ".no_" + "_".join(filter_chrs)
    want += ".bam"
    bams = [b for b in Path(fdir).glob("*.bam") if b.name.endswith(want)]
    if len(bams) != 1:
        raise SystemExit(f"filter {arm}: expected one *{want} in {fdir}, got {bams}")
    return bams[0]


def build_readsets(cfg: dict, unit: str, src: dict, cost: Cost) -> dict:
    """Every arm's (pr1, pr2) tagAlign pair for one replicate, under `work/`. Skip-if-exists per
    arm (READSET.json is written last). The source's own read set is labelled
    `cfg["source_arm"]` (`base`, or a FASTQ-level arm `se`/`crop36`/`crop50`)."""
    lpu = cfg["_src_lpu"][unit]
    exp, seed, tmp = cfg["experiment"], cfg["seed"], cfg["tmp"]
    label = cfg.get("source_arm", "base")
    dnase = cfg["pipeline"] == "dnase"
    out = {}

    def done(arm):
        f = readset_dir(cfg, unit, arm) / "READSET.json"
        return json.loads(f.read_text()) if f.is_file() else None

    def record(arm, rec):
        d = readset_dir(cfg, unit, arm)
        rec.update({h: str(d / f"{h}.tagAlign.gz") for h in HALVES})
        rec["md5"] = {h: records.md5_file(rec[h]) for h in HALVES}
        rec["created_utc"] = utc()
        (d / "READSET.json").write_text(json.dumps(rec, indent=1) + "\n")
        out[arm] = rec

    d = readset_dir(cfg, unit, label)
    if done(label):
        out[label] = done(label)
    elif src["kind"] == "tagalign":
        ta = src["path"]
        rec = {"arm": label, "source": {"kind": "tagalign", "path": str(ta),
                                        "md5": records.md5_file(ta)}}
        n_src = count_lines(ta)
        if cfg["splitter"] == "spr":
            s = spr_split(cost, ta, d / "pr1.tagAlign.gz", d / "pr2.tagAlign.gz",
                          sif=cfg["sif"], tmp=tmp, paired_end=src.get("paired_end",
                                                                      cfg["paired_end"]))
            rec["split"] = {"method": "encode_task_spr.py", "seed_arg": 0,
                            "effective_seed": s["effective_seed"], "command": s["command"],
                            "spr_script_md5": s["spr_script_md5"], "paired_end": cfg["paired_end"]}
            lines = s["lines"]
        else:
            rng = np.random.default_rng([seed, key(exp, unit, "split")])
            lines = cost.py("split_python", split_python, ta, d / "pr1.tagAlign.gz",
                            d / "pr2.tagAlign.gz", n_src, lpu, rng)
            rec["split"] = {"method": "python exact seeded split (test only)", "seed": seed,
                            "rng_key": [seed, key(exp, unit, "split")]}
        if lines["pr1"] + lines["pr2"] != n_src:
            raise SystemExit(f"{ta}: pr1 + pr2 lines {lines} != {n_src}")
        rec["lines"] = lines
        record(label, rec)
    else:
        fdir, prov = None, {}
        if src.get("path"):
            bam = src["path"]
            prov["source"] = {"kind": "bam", "path": str(bam), "md5": records.md5_file(bam)}
        else:   # native route: ENCODE's unfiltered BAM through the pipeline's own filter
            bam, fcmd, fdir = _filtered_bam(cfg, cost, unit, "base", src["unfiltered"])
            prov["source"] = {"kind": "bam", "path": str(src["unfiltered"]),
                              "md5": records.md5_file(src["unfiltered"]),
                              "filter_command": fcmd}
        r = _named_split(cfg, cost, unit, label, bam, drop_dups=dnase)
        if fdir:
            shutil.rmtree(fdir, ignore_errors=True)
        pta = src.get("pipeline_ta")
        if pta and Path(pta).is_file():
            n_pta = count_lines(pta)
            if n_pta != r["lines"]["pr1"] + r["lines"]["pr2"]:
                raise SystemExit(f"{unit} {label}: named tagAlign has {r['lines']} lines, the "
                                 f"pipeline's {pta} has {n_pta}")
            prov["pipeline_tagalign"] = {"path": str(pta), "lines": n_pta}
        record(label, {"arm": label, **r, **prov})

    if cfg.get("check_refilter") and label == "base" and src["kind"] == "bam" \
            and src.get("unfiltered") and src.get("path"):
        f = readset_dir(cfg, unit, label) / "REFILTER_CHECK.json"
        if not f.is_file():
            bam, fcmd, fdir = _filtered_bam(cfg, cost, unit, "base", src["unfiltered"])
            nb = Path(tmp) / f"{exp}.{unit}.refilter.named.tagAlign.gz"
            cost.shell(f"named_ta_refilter_{unit}", bam_arm.apptainer(
                named_ta_command(bam, nb, paired_end=cfg["_src_pe"][unit], tmp=tmp,
                                 nth=cfg["nth"], pipeline=cfg["pipeline"], drop_dups=dnase),
                cfg["sif"], tmp, tmp))
            n_re, n_base = count_lines(nb), sum(out[label]["lines"].values())
            nb.unlink()
            shutil.rmtree(fdir, ignore_errors=True)
            rec = {"filter_command": fcmd, "lines_refilter": n_re, "lines_base": n_base,
                   "equal": n_re == n_base}
            f.write_text(json.dumps(rec, indent=1) + "\n")
            if n_re != n_base:
                raise SystemExit(f"{unit}: identity re-filter gives {n_re} lines, base {n_base}: "
                                 f"the arm filter command is not the pipeline's")

    depth_arms = [a for a in cfg["arms"] if a.startswith("depth")]
    if depth_arms and not all(done(a) for a in depth_arms):
        res = {}
        for h in HALVES:
            rng = np.random.default_rng([seed, key(exp, unit, h, "thin")])
            outs = {a[len("depth"):]: readset_dir(cfg, unit, a) / f"{h}.tagAlign.gz"
                    for a in depth_arms}
            res[h] = cost.py(f"thin_{h}", thin_nested, out[label][h], outs,
                             out[label]["lines"][h], lpu, rng)
        for a in depth_arms:
            lab = a[len("depth"):]
            record(a, {"arm": a, "lines": {h: res[h][lab] for h in HALVES},
                       "thinned_from": {h: out[label][h] for h in HALVES},
                       "method": ("nested exact thinning: rank fragments by a seeded permutation "
                                  "per half, keep rank < round(q * n_fragments)"),
                       "rng_key": {h: [seed, key(exp, unit, h, "thin")] for h in HALVES}})
    else:
        for a in depth_arms:
            out[a] = done(a)

    for a in (x for x in cfg["arms"] if x in BAM_ARMS):
        if done(a):
            out[a] = done(a)
            continue
        if src["kind"] != "bam" or not src.get("unfiltered"):
            print(f"[align_views] {exp} {unit}: arm {a} skipped: needs the unfiltered BAM "
                  f"(source is a {src['kind']})", flush=True)
            continue
        if a == "mapq0" and not dnase and cfg["multimapping"]:
            print(f"[align_views] {exp} {unit}: arm mapq0 skipped: the filter runs with "
                  f"--multimapping {cfg['multimapping']}, which ignores --mapq-thresh", flush=True)
            continue
        prov = {}
        if a == "nodedup" and dnase:
            # the DNase pipeline never removes duplicates (it flags them); its dedup is the kit's
            # tagAlign rule (-F 1024), so nodedup = the same nuclear BAM with flag 0x400 kept
            bam, fdir = src["path"], None
            prov["filter_command"] = "none: the pipeline's nuclear BAM, duplicates kept"
        else:
            bam, fcmd, fdir = _filtered_bam(cfg, cost, unit, a, src["unfiltered"])
            prov["filter_command"] = fcmd
        r = _named_split(cfg, cost, unit, a, bam, drop_dups=dnase and a != "nodedup")
        if fdir:
            shutil.rmtree(fdir, ignore_errors=True)
        record(a, {"arm": a, **r, **prov,
                   "source": {"kind": "bam", "path": str(src["unfiltered"]),
                              "md5": records.md5_file(src["unfiltered"])}})
    return out


def write_measurement(cfg, unit, arm, halves_counts: dict, half_tas: dict, half_lines: dict,
                      cost: Cost, extra_prov: dict, signal=None, src=None, lpu=None) -> str:
    """Stage one measurement dir under `out/.<mid>.partial`, then rename it into place."""
    mid = measurement_id(cfg["experiment"], unit, cfg["layer"], arm)
    final = Path(cfg["out"]) / mid
    stage = Path(cfg["out"]) / f".{mid}.partial"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    lpu = lpu or cfg["lpu"]
    n_total = half_lines["pr1"] + half_lines["pr2"]
    counts = {c: halves_counts["pr1"][c].astype(np.uint64) + halves_counts["pr2"][c]
              for c in halves_counts["pr1"]}
    for c, v in counts.items():
        if v.size and int(v.max()) > np.iinfo(np.uint32).max:
            raise SystemExit(f"{mid} {c}: count overflows uint32")
    write_h5(stage / "counts.h5", counts, counts_attrs(mid, n_total, lpu), np.uint32)
    for h in HALVES:
        write_h5(stage / f"{h}.h5", halves_counts[h], counts_attrs(mid, half_lines[h], lpu),
                 np.uint32)
    if signal:
        for name, arr in signal.items():
            write_h5(stage / name, arr, {"schema": 1, "measurement_id": mid, "resolution": RES,
                                         "kind": "-log10 p, MACS2 signal step, 25 bp mean"},
                     np.float32)
    meta = make_meta(cfg, unit, arm, n_total, lpu, has_signal=bool(signal),
                     counts_path=final / "counts.h5", sha256=sha256_file(stage / "counts.h5"),
                     src=src)
    (stage / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    prov = {
        "schema": 1, "measurement_id": mid, "experiment": cfg["experiment"], "unit": unit,
        "arm": arm, "layer": cfg["layer"], "lines_per_fragment": lpu,
        "halves": {h: {"tagalign": half_tas[h], "n_lines": half_lines[h]} for h in HALVES},
        "chrom_sizes": {"path": str(cfg["chrom_sizes_path"]), "sizes": cfg["sizes"]},
        "pipeline": {"name": meta["pipeline"], "version": meta["pipeline_version"],
                     "sif": cfg.get("sif_info")},
        "build_list_row": cfg.get("build_list_row"),
        "outputs": {n: records.md5_file(stage / n) for n in sorted(os.listdir(stage))
                    if n.endswith(".h5") or n == "meta.json"},
        "slurm_job_ids": bam_arm._job_ids(), "code": {"git_sha": git_sha()},
        "created_utc": utc(), **extra_prov,
    }
    (stage / "provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    shutil.rmtree(final, ignore_errors=True)
    os.replace(stage, final)
    print(f"[align_views] wrote {mid}: depth_reads={n_total}", flush=True)
    return mid


def build(cfg: dict) -> dict:
    """Read sets for every unit, then every wanted measurement dir. Returns {"written",
    "signal_pending"}; `cfg["want"]` (a set of (unit, arm), or None = every one) limits what is
    written."""
    cost = Cost()
    t0 = time.time()
    Path(cfg["tmp"]).mkdir(parents=True, exist_ok=True)     # build_row.sh passes $SLURM_TMPDIR/views
    sizes = cfg["sizes"]
    exp, layer = cfg["experiment"], cfg["layer"]
    sources = cfg["sources"]
    units = cfg.get("units") or (["pooled"] if len(sources) == 1
                                 else [f"br{i + 1}" for i in range(len(sources))])
    cfg["_src_pe"] = {u: s.get("paired_end", cfg["paired_end"]) for u, s in zip(units, sources)}
    cfg["_src_lpu"] = {u: (1 if cfg["pipeline"] == "dnase" else 2 if cfg["_src_pe"][u] else 1)
                       for u in units}
    if cfg["splitter"] == "spr" or any(s["kind"] == "bam" for s in sources):
        cfg["sif_info"] = sif_info(cfg["sif"], cfg["pipeline"])
        cfg["image_digest"] = cfg.get("image_digest") or "sha256:" + cfg["sif_info"]["sha256"]
    cfg["units"] = units
    want = cfg.get("want")
    label = cfg.get("source_arm", "base")
    multi = len(units) > 1
    written, pending = [], None

    def wanted(unit, arm):
        return want is None or (unit, arm) in want

    def exists(unit, arm):
        return (Path(cfg["out"]) / measurement_id(exp, unit, layer, arm)
                / "provenance.json").is_file()

    rs = {u: build_readsets(cfg, u, s, cost) for u, s in zip(units, sources)}
    arms = [label] + [a for a in cfg["arms"] if a != label]
    for arm in arms:
        todo = [u for u in units if arm in rs[u] and wanted(u, arm) and not exists(u, arm)]
        pooled = (multi and wanted("pooled", arm) and not exists("pooled", arm)
                  and all(arm in rs[u] for u in units))
        if not todo and not pooled:
            continue
        acc, lines, tas = {}, {h: 0 for h in HALVES}, {h: [] for h in HALVES}
        for u, src in zip(units, sources):
            if arm not in rs[u] or not (u in todo or pooled):
                continue
            hc = {h: cost.py(f"count_{arm}_{u}_{h}", bin25.counts_from_tagalign, rs[u][arm][h],
                             sizes) for h in HALVES}
            if u in todo:
                extra = {"readset": rs[u][arm], "references": {}}
                sig = None
                if u == "pooled" and arm == "base" and cfg["signal"]:
                    try:
                        sig = build_signal(cfg, {h: [rs[u]["base"][h]] for h in HALVES},
                                           cost, sources=[src])
                    except ControlPending as e:
                        pending = str(e)
                if not (u == "pooled" and arm == "base" and cfg["signal"] and sig is None):
                    if sig:
                        extra["signal"] = sig.pop("_prov")
                    if u == "pooled" and arm == "base":
                        extra["references"] = cfg["references"]
                    written.append(write_measurement(
                        cfg, u, arm, hc, {h: [rs[u][arm][h]] for h in HALVES}, rs[u][arm]["lines"],
                        cost, extra, sig, src=src, lpu=cfg["_src_lpu"][u]))
            if pooled:
                for h in HALVES:
                    acc[h] = hc[h] if h not in acc else {
                        c: (acc[h][c].astype(np.uint64) + hc[h][c]).astype(np.uint32)
                        for c in acc[h]}
                    lines[h] += rs[u][arm]["lines"][h]
                    tas[h].append(rs[u][arm][h])
            del hc
        if pooled:
            extra = {"pooled_from": [measurement_id(exp, u, layer, arm) for u in units],
                     "references": cfg["references"] if arm == "base" else {}}
            sig = None
            if cfg["signal"] and arm == "base":
                try:
                    sig = build_signal(cfg, tas, cost, sources=sources)
                except ControlPending as e:
                    pending = str(e)
            if not (cfg["signal"] and arm == "base" and sig is None):
                if sig:
                    extra["signal"] = sig.pop("_prov")
                written.append(write_measurement(cfg, "pooled", arm, acc, tas, lines, cost, extra,
                                                 sig, lpu=cfg["_src_lpu"][units[0]]))

    rec = {"experiment": exp, "build_list_row": cfg.get("build_list_row"), "written": written,
           "signal_pending": pending, "steps": cost.steps,
           "wall_s": round(time.time() - t0, 1),
           "self_max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
           "children_max_rss_gb": round(
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 3),
           "slurm_job_ids": bam_arm._job_ids(), "finished_utc": utc()}
    d = Path(cfg["work"]) / exp
    d.mkdir(parents=True, exist_ok=True)
    tag = f"{layer}.{cfg.get('build_list_row') or label}"
    (d / f"BUILD_COST.{tag}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[align_views] {exp}: {len(written)} measurements written, wall {rec['wall_s']} s"
          + (f"; SIGNAL PENDING: {pending}" if pending else ""), flush=True)
    return {"written": written, "signal_pending": pending}


def _cat(cost, paths, out) -> Path:
    if len(paths) == 1:
        return Path(paths[0])
    cost.shell(f"concat_{Path(out).name}", "cat " + " ".join(shlex.quote(str(x)) for x in paths)
               + f" > {out}")
    return Path(out)


def _bin_bigwig(args):
    bw, sizes, tmpdir = args
    return bin25.pval_from_bigwig(bw, sizes, tmpdir)


def build_signal(cfg, half_tas: dict, cost: Cost, sources=None) -> dict:
    """signal.h5, pr1_signal.h5, pr2_signal.h5 for the pooled base: the pipeline's MACS2 signal
    step, FE branch skipped (`SKIP_FE_PY`), the three runs at the same time, then the three
    bigwigs binned at the same time. `cfg["signal_spec"]` is `{"mode": "recorded", "metadata"}`
    (the base run's recorded command; testbed) or `{"mode": "pipeline", ...}` (the pipeline's
    default flags; chip adds the control and fragment length, `chip_signal_inputs`)."""
    from concurrent.futures import ProcessPoolExecutor
    tmp, p = Path(cfg["tmp"]), cfg["pipeline"]
    spec = dict(cfg.get("signal_spec") or {"mode": "recorded", "metadata": cfg.get("metadata")})
    Path(f"{tmp}/work").mkdir(parents=True, exist_ok=True)
    exp = cfg["experiment"]
    sources = sources or cfg["sources"]
    if spec["mode"] == "recorded" and len(sources) == 1 and sources[0]["kind"] == "tagalign":
        treat = {"signal.h5": Path(sources[0]["path"])}       # reproduces the base run
    else:
        treat = {"signal.h5": _cat(cost, half_tas["pr1"] + half_tas["pr2"],
                                   tmp / f"{exp}.pooled.tagAlign.gz")}
    for h in HALVES:
        treat[f"{h}_signal.h5"] = _cat(cost, half_tas[h], tmp / f"{exp}.{h}.tagAlign.gz")
    if spec["mode"] == "pipeline" and p == "chip":
        spec.update(chip_signal_inputs(cfg, cost, treat["signal.h5"], spec))
    ctl = spec.get("ctl_ta") if p == "chip" else None
    if spec["mode"] == "recorded" and p == "chip":
        ctl = cfg.get("control_tagalign")
        if not ctl:
            raise SystemExit("--signal for chip needs --control-tagalign")
    sif = cfg["signal_sif"] if spec["mode"] == "pipeline" and p == "dnase" else cfg["sif"]
    jobs, bws, prov = [], {}, {}
    for name, ta in treat.items():
        od = tmp / f"macs2_{name[:-3]}"
        shutil.rmtree(od, ignore_errors=True)
        od.mkdir(parents=True)
        if spec["mode"] == "recorded":
            inner, bw = signal_command([ta] + ([ctl] if ctl else []), od, pipeline=p,
                                       metadata=spec["metadata"], chrsz=cfg["chrsz"])
        else:
            inner = signal_inner(p, ta, od, chrsz=cfg["chrsz"], ctl=ctl,
                                 fraglen=spec.get("fraglen"),
                                 ctl_subsample=spec.get("ctl_subsample", 0),
                                 ctl_paired_end=spec.get("ctl_paired_end", False))
            bw = bam_arm.signal_bigwig([str(ta)] + ([str(ctl)] if ctl else []), od)
        cmd = bam_arm.apptainer(inner, sif, tmp, f"{tmp}/work")
        jobs.append((f"macs2_{name[:-3]}", cmd, f"{tmp}/work"))
        bws[name] = bw
        prov[name] = {"command": cmd, "treatment": str(ta), "control": str(ctl) if ctl else None}
    cost.parallel(jobs)
    t0 = time.time()
    names = list(treat)
    with ProcessPoolExecutor(len(names)) as ex:
        arrs = list(ex.map(_bin_bigwig, [(bws[n], cfg["sizes"], tmp / f"bdg_{n[:-3]}")
                                         for n in names]))
    cost.steps.append({"step": "bin_signal_parallel", "kind": "python",
                       "wall_s": round(time.time() - t0, 2)})
    out = {}
    for n, arr in zip(names, arrs):
        out[n] = arr
        prov[n]["bigwig_md5"] = records.md5_file(bws[n])
        prov[n]["route"] = ("pipeline signal script, FE branch skipped (SKIP_FE_PY); the three runs "
                            "concurrent")
        prov[n]["spec"] = {k: v for k, v in spec.items() if k != "ctl_ta"}
        shutil.rmtree(Path(bws[n]).parent, ignore_errors=True)
    if ctl:
        prov["control_md5"] = records.md5_file(ctl)
    out["_prov"] = prov
    return out


def control_replicates(work, control: str, layer: str, fastq_root=None) -> list | None:
    """The base read-set units a fully built control has: `["pooled"]` for one replicate, else
    `br<bio_rep>` per replicate, from the control's run record (`work/<ctl>/base/arm.json`; native:
    `<fastq root>/<ctl>/bams.json`). None when there is no run record yet."""
    if layer == "native":
        bj = Path(fastq_root) / control / "bams.json" if fastq_root else None
        if not (bj and bj.is_file()):
            return None
        bios = [r.get("bio_rep") for r in json.loads(bj.read_text())["bams"]]
    else:
        aj = Path(work) / control / "base" / "arm.json"
        if not aj.is_file():
            return None
        bios = [r.get("bio_rep") for r in json.loads(aj.read_text())["reps"]]
    return ["pooled"] if len(bios) == 1 else [f"br{b}" for b in bios]


def control_readsets(work, control: str, out, layer: str = "uniform", fastq_root=None) -> dict:
    """{unit: (pr1, pr2)} of a control experiment's base read sets, only when the control is
    completely built: its pooled base measurement exists (`<out>/<ctl>.pooled.<layer>.base/
    provenance.json`, written after every replicate's read set), or every replicate its run record
    lists has READSET.json. Otherwise {} and the caller raises ControlPending: a control row that
    runs beside its ChIP row may have br1 built and br2 not yet (KF2 review)."""
    found = {}
    d = Path(work) / control
    if d.is_dir():
        for u in sorted(d.iterdir()):
            m = re.fullmatch(rf"(pooled|br\d+)\.{re.escape(layer)}", u.name)
            if m and (u / "base" / "READSET.json").is_file():
                found[m.group(1)] = tuple(str(u / "base" / f"{h}.tagAlign.gz") for h in HALVES)
    if not found:
        return {}
    if (Path(out) / measurement_id(control, "pooled", layer, "base") / "provenance.json").is_file():
        return found
    want = control_replicates(work, control, layer, fastq_root)
    if want and set(want) <= set(found):
        return {u: found[u] for u in want}
    return {}


def chip_fraglen(cfg, cost) -> dict:
    """The pooled fragment length the chip pipeline uses for the pooled signal: the rounded mean
    of the replicates' xcor fragment lengths (`fraglen_mean`, chip.wdl). From the run's Cromwell
    metadata when it has them; else the pipeline's own xcor task on each replicate's nodedup read
    set (its xcor input for a BAM start, chip.wdl `bam2ta_no_dedup`)."""
    vals, how = [], ""
    md = cfg.get("run_metadata")
    if md and Path(md).is_file():
        meta = json.loads(Path(md).read_text())
        calls = {}
        for c in meta.get("calls", {}).get("chip.xcor", []):
            if c.get("executionStatus") == "Done" and (c.get("outputs") or {}).get("fraglen"):
                calls[c.get("shardIndex", -1)] = int(c["outputs"]["fraglen"])
        vals, how = [calls[k] for k in sorted(calls)], f"chip.xcor outputs in {md}"
    if not vals:
        tmp = Path(cfg["tmp"])
        for u in cfg["units"]:
            rsd = readset_dir(cfg, u, "nodedup")
            if not (rsd / "READSET.json").is_file():
                raise SystemExit(f"chip signal: no xcor fraglen in the run metadata and no nodedup "
                                 f"read set for {u}")
            ta = _cat(cost, [str(rsd / f"{h}.tagAlign.gz") for h in HALVES],
                      tmp / f"{cfg['experiment']}.{u}.nodedup.tagAlign.gz")
            od = tmp / f"xcor_{u}"
            od.mkdir(parents=True, exist_ok=True)
            inner = (f"python3 $(which encode_task_xcor.py) {ta} --mito-chr-name chrM "
                     f"--subsample 15000000 --chip-seq-type histone --exclusion-range-min -500 "
                     f"--subsample 15000000 --nth {cfg['nth']} --out-dir {od}")
            cost.shell(f"xcor_{u}", bam_arm.apptainer(inner, cfg["sif"], tmp, str(od)))
            fl = sorted(od.glob("*.cc.fraglen.txt"))
            vals.append(int(fl[0].read_text().split()[0]))
        how = "encode_task_xcor.py on each replicate's nodedup read set (chip.wdl xcor defaults)"
    mean = sum(vals) / len(vals)
    return {"fraglen": int(round(mean)), "fraglen_per_rep": vals, "fraglen_source": how}


def chip_signal_inputs(cfg, cost, pooled_ta, spec) -> dict:
    """Control and fragment length for a ChIP experiment's pooled signal, as chip.wdl makes them:
    the pooled control (all listed control experiments' base reads), subsampled by the pipeline's
    own `encode_task_choose_ctl.py` rule (always_use_pooled_ctl, ctl_depth_limit 200000000,
    exp_ctl_depth_ratio_limit 5.0), and the rounded-mean xcor fragment length."""
    tmp = Path(cfg["tmp"])
    ctl_units = {}
    for c in spec.get("controls") or []:
        r = control_readsets(cfg["work"], c, cfg["out"], (cfg.get("control_layers") or {}).get(
            c, "uniform"), cfg.get("fastq_root"))
        if not r:
            raise ControlPending(f"control {c} of {cfg['experiment']} is not completely built "
                                 f"(no pooled base measurement, or a replicate's read set missing)")
        ctl_units.update({f"{c}.{u}": v for u, v in r.items()})
    if not ctl_units:
        raise SystemExit(f"{cfg['experiment']}: chip signal needs a control_experiment")
    ctl_reps = [_cat(cost, list(v), tmp / f"ctl_{k}.tagAlign.gz") for k, v in sorted(ctl_units.items())]
    ctl_pooled = _cat(cost, [str(x) for x in ctl_reps], tmp / "ctl_pooled.tagAlign.gz")
    exp_reps = []
    for u in cfg["units"]:
        rsd = readset_dir(cfg, u, "base")
        exp_reps.append(_cat(cost, [str(rsd / f"{h}.tagAlign.gz") for h in HALVES],
                             tmp / f"exp_{u}.tagAlign.gz"))
    od = tmp / "choose_ctl"
    od.mkdir(parents=True, exist_ok=True)
    inner = ("python3 $(which encode_task_choose_ctl.py) --tas " + " ".join(map(str, exp_reps))
             + " --ctl-tas " + " ".join(map(str, ctl_reps)) + f" --ta-pooled {pooled_ta}"
             + f" --ctl-ta-pooled {ctl_pooled} --always-use-pooled-ctl --ctl-depth-ratio 1.2"
             + f" --ctl-depth-limit 200000000 --exp-ctl-depth-ratio-limit 5.0 --out-dir {od}")
    cost.shell("choose_ctl", bam_arm.apptainer(inner, cfg["sif"], tmp, str(od)))
    n = int((od / "chosen_ctl_subsample_pooled.txt").read_text().split()[0])
    out = {"ctl_ta": str(ctl_reps[0] if len(ctl_reps) == 1 else ctl_pooled),
           "ctl_subsample": n, "ctl_paired_end": bool(spec.get("ctl_paired_end")),
           "control_units": sorted(ctl_units), "choose_ctl_command": inner}
    out.update(chip_fraglen(cfg, cost))
    return out


# --------------------------------------------------------------------------------------------
# verify


def _npz_or_none(path):
    return bin25.read_npz(path) if path and Path(path).is_file() else None


def verify_measurement(d: Path, all_meta: dict, recount_chrom: str) -> list:
    """Problems of one measurement dir (empty list = pass)."""
    import h5py
    from minicandi.dataset import MEASUREMENT_COLUMNS
    probs = []
    need = ("counts.h5", "pr1.h5", "pr2.h5", "meta.json", "provenance.json")
    miss = [n for n in need if not (d / n).is_file()]
    if miss:
        return [f"missing {miss}"]
    meta = json.loads((d / "meta.json").read_text())
    prov = json.loads((d / "provenance.json").read_text())
    mid = d.name
    if list(meta) != list(MEASUREMENT_COLUMNS):
        probs.append("meta.json columns differ from MEASUREMENT_COLUMNS")
    if meta.get("measurement_id") != mid or prov.get("measurement_id") != mid:
        probs.append("measurement_id != directory name")
    sizes = prov["chrom_sizes"]["sizes"]
    arrs = {}
    for n in ("counts.h5", "pr1.h5", "pr2.h5"):
        with h5py.File(d / n, "r") as f:
            a = dict(f.attrs)
            if set(f.keys()) != set(sizes):
                probs.append(f"{n}: chromosomes {sorted(f.keys())} != chrom sizes")
            for c in f.keys():
                ds = f[c]
                if ds.shape != (sizes[c] // RES,):
                    probs.append(f"{n} {c}: shape {ds.shape} != ({sizes[c] // RES},)")
                if ds.dtype not in (np.uint16, np.uint32):
                    probs.append(f"{n} {c}: dtype {ds.dtype}")
                if ds.chunks is None or ds.chunks[0] > CHUNK:
                    probs.append(f"{n} {c}: chunks {ds.chunks} > {CHUNK}")
            arrs[n] = {c: f[c][()] for c in f.keys()}
            if (a.get("schema"), a.get("measurement_id"), a.get("resolution")) != (1, mid, RES):
                probs.append(f"{n}: root attrs {a}")
            arrs[n + ".depth"] = int(a.get("depth_reads", -1))
    if arrs["pr1.h5.depth"] + arrs["pr2.h5.depth"] != arrs["counts.h5.depth"]:
        probs.append("depth_reads: pr1 + pr2 != counts")
    if str(arrs["counts.h5.depth"]) != meta["depth_reads"]:
        probs.append("meta depth_reads != counts.h5 attrs")
    for c in arrs["counts.h5"]:
        if not np.array_equal(arrs["pr1.h5"][c].astype(np.uint64) + arrs["pr2.h5"][c],
                              arrs["counts.h5"][c]):
            probs.append(f"halves do not sum to counts on {c}")
            break
    if sha256_file(d / "counts.h5") != meta["sha256"]:
        probs.append("counts.h5 sha256 != meta.json")
    for n, m in prov.get("outputs", {}).items():
        if records.md5_file(d / n) != m:
            probs.append(f"{n}: md5 != provenance")

    # recount the kept half tagAligns on one chromosome
    if recount_chrom in sizes:
        for h in HALVES:
            tas = prov["halves"][h]["tagalign"]
            rc = sum(bin25.counts_from_tagalign(t, {recount_chrom: sizes[recount_chrom]})
                     [recount_chrom].astype(np.uint64) for t in tas)
            if not np.array_equal(rc, arrs[f"{h}.h5"][recount_chrom]):
                probs.append(f"{h}: {recount_chrom} != recount of {tas}")

    # depth arm: total within 0.1% of q x base, and a subset of the base half by half
    if meta["arm_knob"] == "depth":
        q = float(meta["arm_value"])
        base_id = meta["reads_subset_of"]
        bd = d.parent / base_id
        if base_id not in all_meta or not (bd / "counts.h5").is_file():
            probs.append(f"base {base_id} missing")
        else:
            base_reads = int(all_meta[base_id]["depth_reads"])
            got = int(meta["depth_reads"])
            # exact per half up to rounding: round(q * n) per half, so at most one fragment each
            slack = max(1e-3 * q * base_reads, 2 * int(prov.get("lines_per_fragment", 1)))
            if abs(got - q * base_reads) > slack:
                probs.append(f"depth {got} not within 0.1% of {q} x {base_reads}")
            for h in HALVES:
                b = read_h5(bd / f"{h}.h5")
                if any(np.any(arrs[f"{h}.h5"][c] > b[c]) for c in b):
                    probs.append(f"{h}: thinned counts exceed base")
    # covariates of the arm (meta.json) agree with the arm in the measurement id
    arm = mid.split(".", 3)[3] if mid.count(".") >= 3 else ""
    want = arm_fields(arm, {"dedup": "1", "mapq_threshold": meta["mapq_threshold"]}) \
        if arm and (arm in ALL_ARMS or arm in FASTQ_ARMS) else None
    if want:
        for k in ("arm_knob", "arm_value"):
            if meta[k] != want[k]:
                probs.append(f"meta {k} {meta[k]!r} != {want[k]!r} for arm {arm}")
        if (arm == "nodedup") != (meta["dedup"] == "0"):
            probs.append(f"meta dedup {meta['dedup']!r} wrong for arm {arm}")
        if arm == "mapq0" and meta["mapq_threshold"] != "0":
            probs.append("meta mapq_threshold != 0 for arm mapq0")
        if arm == "se" and meta["run_type"] != "single-ended":
            probs.append("meta run_type != single-ended for arm se")
        if arm.startswith("crop") and meta["read_length"] != arm[4:]:
            probs.append(f"meta read_length {meta['read_length']} != {arm[4:]} for arm {arm}")
        if arm.startswith("depth") and not meta["reads_subset_of"].endswith(".base"):
            probs.append("depth arm without reads_subset_of = its base")
    # nodedup keeps every read of the base: half by half, bin by bin, nodedup >= base
    if arm == "nodedup":
        bd = d.parent / meta["parent_measurement_id"]
        if (bd / "pr1.h5").is_file():
            for h in HALVES:
                b = read_h5(bd / f"{h}.h5")
                if any(np.any(arrs[f"{h}.h5"][c] < b[c]) for c in b):
                    probs.append(f"{h}: nodedup has fewer reads than base in some bin")

    # references (smoke): the testbed's counts and t117 halves of the same tagAlign
    ref = prov.get("references") or {}
    for name, path in (("counts.h5", ref.get("counts25")), ("pr1.h5", ref.get("pr1_counts25")),
                       ("pr2.h5", ref.get("pr2_counts25"))):
        r = _npz_or_none(path)
        if path and r is None:
            probs.append(f"reference {path} missing")
        if r is not None:
            common = [c for c in r if c in arrs[name]]
            if not common or not all(np.array_equal(r[c], arrs[name][c]) for c in common):
                probs.append(f"{name} != reference {path}")
    if meta["has_signal"] == "1":
        for n in ("signal.h5", "pr1_signal.h5", "pr2_signal.h5"):
            if not (d / n).is_file():
                probs.append(f"has_signal but no {n}")
                continue
            s = read_h5(d / n)
            if set(s) != set(arrs["counts.h5"]) or not all(np.isfinite(v).all() for v in s.values()):
                probs.append(f"{n}: grid or NaN")
        r = _npz_or_none(ref.get("pval25"))
        if r is not None:
            s = read_h5(d / "signal.h5")
            common = [c for c in r if c in s]
            dev = max(float(np.max(np.abs(s[c] - r[c]))) for c in common)
            if dev > 1e-4:
                probs.append(f"signal.h5 differs from reference pval25 by {dev}")
    return probs


def _first_col4(path) -> bytes:
    p = _reader(path)
    try:
        line = p.stdout.readline()
    finally:
        p.stdout.close()
        p.wait()
    f = line.split(b"\t")
    return f[3] if len(f) > 3 else b""


def experiment_checks(root: Path, mids: list, all_meta: dict, recount_chrom: str) -> list:
    """Checks across the measurements of one experiment: (1) pooled = sum of br<i> for every arm
    that has both, half by half; (2) when the halves carry name tokens (`--name-tokens`), no read
    of any pr1 half (any unit, any arm) is in any pr2 half, on `recount_chrom`."""
    probs = []
    by = {}
    for m in mids:
        exp, unit, layer, arm = m.split(".", 3)
        by.setdefault((layer, arm), {})[unit] = m
    for (layer, arm), units in by.items():
        brs = sorted(u for u in units if re.fullmatch(r"br\d+", u))
        if "pooled" not in units or not brs:
            continue
        for n in ("pr1.h5", "pr2.h5"):
            pooled = read_h5(root / units["pooled"] / n)
            parts = [read_h5(root / units[u] / n) for u in brs]
            for c in pooled:
                if not np.array_equal(pooled[c], sum(x[c].astype(np.uint64) for x in parts)):
                    probs.append(f"{units['pooled']} {n}: pooled != sum of {brs} on {c}")
                    break
    tas = {h: set() for h in HALVES}
    for m in mids:
        prov = json.loads((root / m / "provenance.json").read_text())
        for h in HALVES:
            tas[h].update(prov["halves"][h]["tagalign"])
    if tas["pr1"] and all(_first_col4(t) not in (b"N", b"") for t in tas["pr1"] | tas["pr2"]):
        tok = {}
        for h in HALVES:
            tok[h] = set()
            for t in sorted(tas[h]):
                p = _reader(t)
                for line in p.stdout:
                    if line.startswith(recount_chrom.encode() + b"\t"):
                        tok[h].add(line.split(b"\t")[3])
                _finish(p, t)
        both = tok["pr1"] & tok["pr2"]
        if both:
            probs.append(f"{len(both)} read names are in a pr1 half and a pr2 half ({recount_chrom})")
        else:
            print(f"  name tokens on {recount_chrom}: {len(tok['pr1'])} pr1, {len(tok['pr2'])} pr2, "
                  f"0 shared, across {len(tas['pr1'])} + {len(tas['pr2'])} half files")
    return probs


def verify(root, recount_chrom="chr21", experiments=None) -> int:
    root = Path(root)
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file()
                  and (not experiments or p.name.split(".")[0] in experiments))
    if not dirs:
        print(f"no measurement directories under {root}")
        return 1
    all_meta = {p.name: json.loads((p / "meta.json").read_text()) for p in dirs}
    by_exp, bad = {}, 0
    for p in dirs:
        probs = verify_measurement(p, all_meta, recount_chrom)
        by_exp.setdefault(all_meta[p.name]["experiment"], []).append((p.name, probs))
    for exp, rows in by_exp.items():
        xp = experiment_checks(root, [m for m, _ in rows], all_meta, recount_chrom)
        if xp:
            rows.append((f"{exp} (across measurements)", xp))
        nbad = sum(1 for _, pr in rows if pr)
        bad += nbad
        print(f"{exp}: {len(rows) - (1 if xp else 0)} measurements, {nbad} failing")
        for mid, pr in rows:
            for x in pr:
                print(f"  FAIL {mid}: {x}")
    if bad:
        print(f"{len(by_exp)} experiments, {bad} measurements fail")
        return 1
    print(f"{len(by_exp)} experiments, all checks pass")
    return 0


# --------------------------------------------------------------------------------------------
# CLI


def parse_source(s: str) -> dict:
    """`<path>.tagAlign.gz`, or `bam:<filtered.bam>` or `bam:<filtered.bam>,<unfiltered.bam>`."""
    if s.startswith("bam:"):
        parts = s[4:].split(",")
        if not 1 <= len(parts) <= 2:
            raise SystemExit(f"bad --source {s!r}")
        return {"kind": "bam", "path": parts[0], "unfiltered": parts[1] if len(parts) == 2 else None}
    return {"kind": "tagalign", "path": s}


def config_from_args(a) -> dict:
    arms = list(ALL_ARMS) if a.arms == "all" else a.arms.split(",")
    bad = [x for x in arms if x not in ALL_ARMS]
    if bad or "base" not in arms:
        raise SystemExit(f"--arms must include base and use {ALL_ARMS}; got {arms}")
    meta = {}
    for kv in a.meta:
        k, _, v = kv.partition("=")
        meta[k] = v
    sizes = load_sizes(a.chrom_sizes)
    refs = {k: v for k, v in (("counts25", a.reference_counts25), ("pval25", a.reference_pval25))
            if v}
    if a.reference_halves:
        refs.update({f"{h}_counts25": f"{a.reference_halves}/{h}.counts25.npz" for h in HALVES})
    for k, v in list(refs.items()):
        refs[k + "_md5"] = records.md5_file(v) if Path(v).is_file() else None
    return {
        "experiment": a.experiment, "sources": [parse_source(s) for s in a.source],
        "assay": a.assay, "cell": a.cell, "eic_track": a.eic_track, "role": a.role,
        "layer": a.layer, "arms": arms, "pipeline": a.pipeline, "paired_end": a.paired_end,
        "lpu": 2 if a.paired_end else 1,
        "run_type": "paired-ended" if a.paired_end else "single-ended",
        "read_length": a.read_length, "dedup": "1", "mapq_threshold": str(a.mapq_threshold),
        "multimapping": a.multimapping, "meta": meta, "work": a.work, "out": a.out,
        "sizes": sizes, "chrom_sizes_path": a.chrom_sizes, "splitter": a.splitter,
        "seed": a.seed, "tmp": a.tmp or os.environ.get("SLURM_TMPDIR") or tempfile.mkdtemp(),
        "sif": a.sif or default_sif(a.pipeline), "chrsz": a.chrsz, "nth": a.nth,
        "mem_gb": a.mem_gb, "signal": a.signal, "metadata": a.metadata,
        "control_tagalign": a.control_tagalign, "references": refs,
        "name_tokens": a.name_tokens, "check_refilter": False,
    }


# --------------------------------------------------------------------------------------------
# run: one build-list row -> its measurement directories (the bulk build's driver)


ARM_ORDER = ("base", "se", "crop36", "crop50") + tuple(f"depth{v}" for v, _ in DEPTH_LEVELS) \
    + BAM_ARMS


def parse_expected(s: str) -> dict:
    """`expected_measurements` of a build-list row -> {measurement_id: {products}}."""
    out = {}
    for item in (s or "").split(";"):
        if item.strip():
            mid, _, prods = item.partition(":")
            out[mid] = {x for x in prods.split("+") if x}
    return out


def _fu():
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import fastq_uniform
    return fastq_uniform


def control_layer(rows, control: str) -> str:
    """The layer a control experiment is built in: native when its base row is a native row."""
    FU = _fu()
    for r in rows:
        if r["experiment"] == control and r["unit"] == "pooled" and r["arm"] == "base":
            return "native" if FU.row_kind(r) == "native" else "uniform"
    return "uniform"


def config_from_row(a) -> dict | None:
    """The build config of one build-list row (None: the row's arm does not apply)."""
    FU = _fu()
    rows = FU.read_build_list(a.build_list)
    row = FU.get_row(rows, a.row)
    kind = FU.row_kind(row)
    exp = row["experiment"]
    layer = "native" if kind == "native" else "uniform"
    pipeline = "chip" if kind == "native" else kind
    expected = parse_expected(row.get("expected_measurements", ""))
    want, signal_wanted = set(), False
    for mid, prods in expected.items():
        e, unit, lay, arm = mid.split(".", 3)
        if e != exp or lay != layer:
            raise SystemExit(f"row {row['row']}: expected {mid} is not {exp} in layer {layer}")
        want.add((unit, arm))
        if "signal" in prods:
            if (unit, arm) != ("pooled", "base"):
                raise SystemExit(f"row {row['row']}: signal asked for {mid}, not the pooled base")
            signal_wanted = True
    if not want:
        raise SystemExit(f"row {row['row']}: no expected_measurements")
    rundir = FU.arm_dir(a.work, row)
    platform, run_metadata = "", None
    if kind != "native":
        if (rundir / FU.NOT_APPLICABLE).exists():
            print(f"[align_views] row {row['row']}: {FU.NOT_APPLICABLE}: "
                  f"{json.loads((rundir / FU.NOT_APPLICABLE).read_text())['reason']}")
            return None
        if not (rundir / "ALIGN_DONE").exists():
            raise SystemExit(f"{rundir}: no ALIGN_DONE (the FASTQ kit has not finished this row)")
        armj = json.loads((rundir / "arm.json").read_text())
        pe = armj.get("paired_end_per_rep") or [False] * len(armj["reps"])
        sources = []
        for r, is_pe in zip(armj["reps"], pe):
            k = r["rep"]
            src = {"kind": "bam", "path": str(rundir / f"rep{k}.filtered.bam"),
                   "unfiltered": str(rundir / f"rep{k}.unfiltered.bam"),
                   "pipeline_ta": str(rundir / f"rep{k}.tagAlign.gz"),
                   "bio_rep": r["bio_rep"], "paired_end": bool(is_pe),
                   "run_type": "paired-ended" if is_pe else "single-ended"}
            if r.get("read_length"):
                src["read_length"] = str(r["read_length"])
            sources.append(src)
        run_type, read_length = armj.get("run_type", ""), armj.get("read_length") or ""
        platform = ",".join(armj.get("platforms") or [])
        pipeline_name = armj.get("pipeline") or PIPELINE_NAME[pipeline]
        pipeline_version = armj.get("pipeline_version") or PIPELINE_VERSION[pipeline]
        image_digest = armj.get("image_digest", "")
        mapq = armj.get("mapq_threshold", BASE_MAPQ[pipeline])
        if (rundir / "metadata.json").is_file():
            run_metadata = str(rundir / "metadata.json")
        notes = {"mixed_run_types": armj.get("mixed_run_types", "")}
    else:
        bj = Path(a.fastq_root) / exp / "bams.json"
        if not bj.is_file():
            FU.stage_native(row, a.fastq_root)
        recs = sorted(json.loads(bj.read_text())["bams"], key=lambda r: (r["bio_rep"] or 0))
        sources = []
        for r in recs:
            bam = Path(a.fastq_root) / exp / f"{r['accession']}.bam"
            if not bam.is_file() or bam.stat().st_size != r["bytes"]:
                raise SystemExit(f"{bam}: missing or size != portal {r['bytes']}")
            sources.append({"kind": "bam", "path": None, "unfiltered": str(bam),
                            "bio_rep": r["bio_rep"], "paired_end": False,
                            "run_type": "single-ended", "portal": r})
        run_type = "single-ended"   # bwa samse (D4's read of ENCFF989MMP)
        read_length = str(recs[0].get("mapped_read_length") or "")
        pipeline_name = "native-redacted-bam"
        pipeline_version = (recs[0].get("analysis_step_version") or "").strip("/").split("/")[-1]
        image_digest, mapq = "", BASE_MAPQ["chip"]
        notes = {"native": "ENCODE redacted unfiltered GRCh38 BAM (PI ruling 16), filtered here by "
                           "chip-seq-pipeline2 v2.2.2 encode_task_filter.py at its defaults"}
    if row["unit"] == "pooled":
        units = (["pooled"] if len(sources) == 1
                 else [f"br{s['bio_rep']}" for s in sources])
        if len(set(units)) != len(units):
            raise SystemExit(f"row {row['row']}: replicates {units} are not distinct")
    else:
        if len(sources) != 1:
            raise SystemExit(f"row {row['row']}: unit {row['unit']} has {len(sources)} replicates")
        units = [row["unit"]]
    source_arm = row["arm"]
    arms = [x for x in ARM_ORDER if any(arm == x for _, arm in want)]
    if source_arm not in arms:
        arms.insert(0, source_arm)
    controls = [c for c in (row.get("control_experiment") or "").split(";") if c]
    sig = {"auto": signal_wanted, "on": True, "off": False}[a.signal]
    if sig and pipeline == "chip" and kind == "native" and "nodedup" not in arms:
        arms.append("nodedup")                  # the xcor input of a BAM start (chip.wdl)
    ctl_pe = False
    if controls:
        cj = Path(a.work) / controls[0] / "base" / "arm.json"
        if cj.is_file():
            ctl_pe = bool((json.loads(cj.read_text()).get("paired_end_per_rep") or [False])[0])
    m = re.match(r"^(C\d+)", row.get("eic_track", ""))
    sizes = load_sizes(a.chrom_sizes)
    cfg = {
        "experiment": exp, "sources": sources, "units": units, "assay": row["assay"],
        "cell": m.group(1) if m else "", "eic_track": row.get("eic_track", ""),
        "role": row.get("role", ""), "layer": layer, "arms": arms, "source_arm": source_arm,
        "want": want, "pipeline": pipeline, "pipeline_name": pipeline_name,
        "pipeline_version": pipeline_version, "image_digest": image_digest,
        "paired_end": any(s["paired_end"] for s in sources),
        "lpu": 1 if pipeline == "dnase" else (2 if sources[0]["paired_end"] else 1),
        "run_type": run_type, "read_length": str(read_length), "dedup": "1",
        "mapq_threshold": str(mapq), "multimapping": FILTER_DEFAULTS.get(
            pipeline, {}).get("multimapping", 0) if a.multimapping is None else a.multimapping,
        "meta": {"platform": platform} if platform else {},
        "control_measurement_id": (f"{controls[0]}.pooled.{control_layer(rows, controls[0])}.base"
                                   if controls else ""),
        "work": a.work, "out": a.out, "sizes": sizes, "chrom_sizes_path": a.chrom_sizes,
        "splitter": "spr", "seed": a.seed,
        "tmp": a.tmp or os.environ.get("SLURM_TMPDIR") or tempfile.mkdtemp(),
        "sif": default_sif(pipeline), "signal_sif": bam_arm.SIF["atac"], "chrsz": a.chrsz,
        "nth": a.nth, "mem_gb": a.mem_gb, "signal": sig,
        "signal_spec": {"mode": "pipeline", "controls": controls, "ctl_paired_end": ctl_pe},
        "run_metadata": run_metadata, "references": {}, "name_tokens": a.name_tokens,
        "check_refilter": a.check_refilter, "build_list_row": row["row"],
        "keep_bam": row.get("keep_bam", "1"), "rundir": str(rundir), "notes": notes,
        "nuclear_chroms": a.nuclear_chroms, "fastq_root": a.fastq_root,
        "control_layers": {c: control_layer(rows, c) for c in controls},
        "row": row, "rows": rows,
    }
    return cfg


def bam_readers_pending(cfg) -> list:
    """Build-list rows (any priority) that still read this row's run-dir BAMs: every row whose run
    dir is the same directory and whose expected measurements are not all written. A row reads only
    its own run dir (`fastq_uniform.arm_dir`): a priority-3 FASTQ arm row (`se`, `crop36`,
    `crop50`) and a priority-4 library or lane row each align their own FASTQs into their own run
    dir, and the depth, nodedup and mapq0 views come from the base row itself. So once the row's
    own measurements exist, no later row needs these BAMs."""
    FU = _fu()
    rd = Path(cfg["rundir"]).resolve()
    out = []
    for r in cfg.get("rows") or [cfg["row"]]:
        if FU.row_kind(r) == "native" or FU.arm_dir(cfg["work"], r).resolve() != rd:
            continue
        if (rd / FU.NOT_APPLICABLE).exists():
            continue
        if any(not (Path(cfg["out"]) / mid / "provenance.json").is_file()
               for mid in parse_expected(r.get("expected_measurements", ""))):
            out.append(r["row"])
    return out


def drop_bams(cfg) -> list:
    """After every wanted measurement of a `keep_bam = 0` row is written: delete the run dir's BAMs
    (the read sets under work/ stay and rebuild every measurement); DROPPED.tsv records them so the
    FASTQ kit's `verify` still passes."""
    FU = _fu()
    rd = Path(cfg["rundir"])
    harvest = rd / "HARVEST.tsv"
    if not harvest.is_file():
        return []
    rows = [ln.split("\t") for ln in harvest.read_text().strip().split("\n")[1:]]
    gone = [r for r in rows if r[0].endswith((".bam", ".bam.bai")) and (rd / r[0]).exists()]
    if not gone:
        return []
    old = FU.read_dropped(rd)
    lines = ["name\tbytes\tmd5\treason\tdropped_utc"] + [
        "\t".join([r[0], r[2], r[3], r[4], r[5]]) for r in old.values()]
    for r in gone:
        lines.append(f"{r[0]}\t{r[2]}\t{r[3]}\tkeep_bam=0, views built (align_views run)\t{utc()}")
    FU.write_atomic(rd / "DROPPED.tsv", "\n".join(lines) + "\n")
    for r in gone:
        (rd / r[0]).unlink()
    print(f"[align_views] dropped {len(gone)} BAM files from {rd} (keep_bam = 0)", flush=True)
    return [r[0] for r in gone]


def run_row(a) -> int:
    cfg = config_from_row(a)
    if cfg is None:
        return 0
    res = build(cfg)
    missing = [measurement_id(cfg["experiment"], u, cfg["layer"], arm) for u, arm in cfg["want"]
               if not (Path(cfg["out"]) / measurement_id(cfg["experiment"], u, cfg["layer"], arm)
                       / "provenance.json").is_file()]
    if res["signal_pending"]:
        print(f"[align_views] row {cfg['build_list_row']}: exit 75, signal pending "
              f"({res['signal_pending']}); rerun the row after its control is built", flush=True)
        return 75
    if missing:
        print(f"[align_views] row {cfg['build_list_row']}: NOT BUILT {missing}", flush=True)
        return 1
    if a.drop_bams and cfg["keep_bam"] == "0":
        readers = bam_readers_pending(cfg)
        if readers:
            print(f"[align_views] BAMs kept in {cfg['rundir']}: rows {readers} still read them",
                  flush=True)
        else:
            drop_bams(cfg)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="one experiment -> measurement directories")
    b.add_argument("--experiment", required=True)
    b.add_argument("--source", action="append", required=True,
                   help="one per biological replicate, in replicate order: a tagAlign path, or "
                        "bam:<filtered.bam>[,<unfiltered.bam>]")
    b.add_argument("--assay", required=True)
    b.add_argument("--cell", required=True)
    b.add_argument("--eic-track", default="")
    b.add_argument("--role", default="", choices=("", "train", "val", "blind"))
    b.add_argument("--layer", default="uniform", choices=("uniform", "native", "testbed"))
    b.add_argument("--arms", default="all", help=f"comma list from {ALL_ARMS}, or all")
    b.add_argument("--pipeline", default="chip", choices=("chip", "atac", "dnase"))
    b.add_argument("--paired-end", action="store_true")
    b.add_argument("--read-length", default="")
    b.add_argument("--mapq-threshold", type=int, default=30, help="the base run's filter setting")
    b.add_argument("--multimapping", type=int, default=None,
                   help="the filter's --multimapping (default: the pipeline's default)")
    b.add_argument("--meta", action="append", default=[],
                   help="extra measurements.tsv field, key=value (lab, platform, antibody, ...)")
    b.add_argument("--work", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--chrom-sizes", default=GENOME_SIZES)
    b.add_argument("--splitter", default="spr", choices=("spr", "python"))
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tmp", default=None)
    b.add_argument("--sif", default=None)
    b.add_argument("--chrsz", default=bam_arm.CHRSZ, help="chrom sizes TSV the SIF commands read")
    b.add_argument("--nth", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    b.add_argument("--mem-gb", type=float, default=16.0)
    b.add_argument("--signal", action="store_true",
                   help="validation and blind experiments: MACS2 signal of pooled base and halves")
    b.add_argument("--metadata", default=None, help="base run's Cromwell metadata.json")
    b.add_argument("--control-tagalign", default=None)
    b.add_argument("--reference-counts25", default=None, help="smoke: testbed counts25.npz")
    b.add_argument("--reference-halves", default=None,
                   help="smoke: dir with t117 pr1/pr2.counts25.npz of the same tagAlign")
    b.add_argument("--reference-pval25", default=None, help="smoke: testbed pval25.npz")
    b.add_argument("--name-tokens", action="store_true",
                   help="verification runs: keep a hash of each read name in the halves' column 4")
    r = sub.add_parser("run", help="one build-list row -> its measurement directories")
    r.add_argument("--build-list", required=True)
    r.add_argument("--row", required=True, help="the build list's `row` column value")
    r.add_argument("--work", default="/scratch/mforooz/onion/work")
    r.add_argument("--out", default="/scratch/mforooz/onion/measurements")
    r.add_argument("--fastq-root", default="/scratch/mforooz/onion/fastq")
    r.add_argument("--chrom-sizes", default=GENOME_SIZES)
    r.add_argument("--chrsz", default=bam_arm.CHRSZ)
    r.add_argument("--nuclear-chroms", default=DNASE_NUCLEAR_CHROMS)
    r.add_argument("--signal", default="auto", choices=("auto", "on", "off"),
                   help="auto: only when the row's expected products include signal")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tmp", default=None)
    r.add_argument("--nth", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    r.add_argument("--mem-gb", type=float, default=16.0)
    r.add_argument("--multimapping", type=int, default=None)
    r.add_argument("--name-tokens", action="store_true")
    r.add_argument("--check-refilter", action="store_true",
                   help="also re-run the filter unchanged and require the base's read count")
    r.add_argument("--drop-bams", action="store_true",
                   help="delete the run dir's BAMs once every view of a keep_bam = 0 row exists")
    v = sub.add_parser("verify", help="check every measurement directory under a root")
    v.add_argument("--measurements", required=True)
    v.add_argument("--recount-chrom", default="chr21")
    v.add_argument("--experiment", action="append", default=[],
                   help="limit to these experiments (repeatable)")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        return verify(a.measurements, a.recount_chrom, set(a.experiment) or None)
    if a.cmd == "run":
        return run_row(a)
    cfg = config_from_args(a)
    if a.signal and not a.metadata:
        raise SystemExit("--signal needs --metadata (the base run's recorded signal command)")
    if a.multimapping is None:
        cfg["multimapping"] = FILTER_DEFAULTS[a.pipeline]["multimapping"]
    build(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
