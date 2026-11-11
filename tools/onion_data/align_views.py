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
ALL_ARMS = ("base",) + tuple(f"depth{v}" for v, _ in DEPTH_LEVELS) + BAM_ARMS
PIPELINE_NAME = {"chip": "chip-seq-pipeline2", "atac": "atac-seq-pipeline"}
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


def name_half(name: bytes, salt: bytes) -> int:
    """0 -> pr1, 1 -> pr2: the low bit of a seeded hash of the read (fragment) name."""
    return hashlib.blake2b(salt + name, digest_size=1).digest()[0] & 1


def split_named(src, out1, out2, salt: bytes) -> dict:
    """Split a NAMED tagAlign (column 4 = read name) by `name_half`; both mates of a fragment carry
    its name, so a fragment never straddles the halves. Column 4 is written as "N" (`bam2ta`)."""
    w = (_Writer(out1), _Writer(out2))
    p = _reader(src)
    for line in p.stdout:
        f = line.split(b"\t")
        x = w[name_half(f[3], salt)]
        f[3] = b"N"
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
    md5, sha = bam_arm.check_sif(sif, records.PIPELINE_REPO[pipeline])
    return {"path": sif, "md5": md5, "sha256": sha}


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
    """The pipeline's own filter step on the unfiltered BAM with ONE setting changed."""
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
    else:
        raise ValueError(arm)
    return (f"python3 $(which encode_task_filter.py) {bam}"
            f"{' --paired-end' if paired_end else ''} --multimapping {s['multimapping']}"
            f" --dup-marker {s['dup_marker']} --mapq-thresh {s['mapq_thresh']}"
            f" --filter-chrs {' '.join(s['filter_chrs'])} --chrsz {chrsz}"
            f"{' --no-dup-removal' if nodup else ''} --mito-chr-name {s['mito_chr_name']}"
            f" --mem-gb {mem_gb} --nth {nth} --picard-java-heap {max(1, int(mem_gb))}G"
            f" --out-dir {out_dir}")


def named_ta_command(bam, out, *, paired_end, tmp, nth) -> str:
    """BAM -> tagAlign with read names kept: `bam2ta`'s bamtobed and awk, minus `$4="N"`."""
    if not paired_end:
        return f"bedtools bamtobed -i {bam} | {SE_NAMED_AWK} | gzip -nc > {out}"
    srt = f"{tmp}/nmsrt.{Path(bam).stem}.bam"
    return (f"samtools sort -n -@ {nth} -o {srt} {bam} && LC_COLLATE=C bedtools bamtobed -bedpe "
            f"-mate1 -i {srt} | {PE_NAMED_AWK} | gzip -nc > {out} && rm -f {srt}")


def signal_command(tas, out_dir, *, pipeline, metadata, chrsz) -> tuple:
    """(inner command, bigwig path): the base run's recorded MACS2 signal command with only the
    tagAligns, `--chrsz` and `--out-dir` set (`bam_arm.build_signal_command`)."""
    meta = json.loads(Path(metadata).read_text())
    calls = meta["calls"][bam_arm.SIGNAL_CALL[pipeline]]
    if len(calls) != 1:
        raise SystemExit(f"{metadata}: {len(calls)} signal calls, expected 1")
    parsed = bam_arm.parse_signal_command(calls[0]["commandLine"], pipeline)
    inner = bam_arm.build_signal_command(parsed, [str(t) for t in tas], out_dir, chrsz=chrsz,
                                         fraglen_flag=bam_arm.FRAGLEN_FLAG[pipeline])
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


def arm_fields(arm: str, base: dict) -> dict:
    """The covariates an arm changes: (arm_knob, arm_value, dedup, mapq_threshold)."""
    if arm == "base":
        return {"arm_knob": "", "arm_value": "", "dedup": base["dedup"],
                "mapq_threshold": base["mapq_threshold"]}
    if arm.startswith("depth"):
        return {"arm_knob": "depth", "arm_value": arm[len("depth"):], "dedup": base["dedup"],
                "mapq_threshold": base["mapq_threshold"]}
    if arm == "nodedup":
        return {"arm_knob": "nodedup", "arm_value": "", "dedup": "0",
                "mapq_threshold": base["mapq_threshold"]}
    if arm == "mapq0":
        return {"arm_knob": "mapq", "arm_value": "0", "dedup": base["dedup"],
                "mapq_threshold": "0"}
    raise ValueError(arm)


def make_meta(cfg: dict, unit: str, arm: str, depth_reads: int, lpu: int, *, has_signal: bool,
              counts_path, sha256: str) -> dict:
    from minicandi import contract as C
    from minicandi.dataset import MEASUREMENT_COLUMNS
    exp, layer = cfg["experiment"], cfg["layer"]
    mid = measurement_id(exp, unit, layer, arm)
    base_mid = measurement_id(exp, unit, layer, "base")
    pooled_base = measurement_id(exp, "pooled", layer, "base")
    row = {k: "" for k in MEASUREMENT_COLUMNS}
    row.update(cfg["meta"])
    row.update({
        "measurement_id": mid, "experiment": exp, "assay": cfg["assay"],
        "assay_class": C.assay_class(cfg["assay"]), "cell": cfg["cell"],
        "eic_track": cfg["eic_track"], "in_eic": cfg["role"], "unit": unit,
        "bio_rep": unit[2:] if unit.startswith("br") else "", "layer": layer,
        "parent_measurement_id": "" if arm == "base" else base_mid,
        "reads_subset_of": (base_mid if arm.startswith("depth")
                            else pooled_base if arm == "base" and unit != "pooled" else ""),
        "depth_reads": str(depth_reads), "depth_fragments": str(depth_reads // lpu),
        "run_type": cfg["run_type"], "read_length": cfg["read_length"],
        "pipeline": PIPELINE_NAME[cfg["pipeline"]],
        "pipeline_version": records.PIPELINES[records.PIPELINE_REPO[cfg["pipeline"]]][0],
        "image_digest": cfg.get("image_digest", ""),
        "has_pseudoreps": "1", "has_signal": "1" if has_signal else "0",
        "counts_path": f"h5:{Path(counts_path).resolve()}", "sha256": sha256,
    })
    row.update(arm_fields(arm, {"dedup": cfg["dedup"], "mapq_threshold": cfg["mapq_threshold"]}))
    return {k: str(row[k]) for k in MEASUREMENT_COLUMNS}


def readset_dir(cfg, unit, arm) -> Path:
    return Path(cfg["work"]) / cfg["experiment"] / f"{unit}.{cfg['layer']}" / arm


def build_readsets(cfg: dict, unit: str, src: dict, cost: Cost) -> dict:
    """Every arm's (pr1, pr2) tagAlign pair for one replicate, under `work/`. Skip-if-exists per
    arm (READSET.json is written last)."""
    lpu = cfg["lpu"]
    exp, seed, tmp = cfg["experiment"], cfg["seed"], cfg["tmp"]
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

    d = readset_dir(cfg, unit, "base")
    if done("base"):
        out["base"] = done("base")
    elif src["kind"] == "tagalign":
        ta = src["path"]
        rec = {"arm": "base", "source": {"kind": "tagalign", "path": str(ta),
                                         "md5": records.md5_file(ta)}}
        n_src = count_lines(ta)
        if cfg["splitter"] == "spr":
            s = spr_split(cost, ta, d / "pr1.tagAlign.gz", d / "pr2.tagAlign.gz",
                          sif=cfg["sif"], tmp=tmp, paired_end=cfg["paired_end"])
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
        record("base", rec)
    else:
        named = Path(tmp) / f"{exp}.{unit}.base.named.tagAlign.gz"
        cost.shell("named_ta_base", bam_arm.apptainer(
            named_ta_command(src["path"], named, paired_end=cfg["paired_end"], tmp=tmp,
                             nth=cfg["nth"]), cfg["sif"], tmp, tmp))
        salt = f"{seed}|{exp}|{unit}|".encode()
        lines = cost.py("split_named", split_named, named, d / "pr1.tagAlign.gz",
                        d / "pr2.tagAlign.gz", salt)
        named.unlink()
        record("base", {"arm": "base", "lines": lines,
                        "source": {"kind": "bam", "path": str(src["path"]),
                                   "md5": records.md5_file(src["path"])},
                        "split": {"method": "blake2b(salt + read name) low bit", "salt": salt.decode(),
                                  "named_ta_awk": SE_NAMED_AWK if not cfg["paired_end"] else PE_NAMED_AWK}})

    depth_arms = [a for a in cfg["arms"] if a.startswith("depth")]
    if depth_arms and not all(done(a) for a in depth_arms):
        res = {}
        for h in HALVES:
            rng = np.random.default_rng([seed, key(exp, unit, h, "thin")])
            outs = {a[len("depth"):]: readset_dir(cfg, unit, a) / f"{h}.tagAlign.gz"
                    for a in depth_arms}
            res[h] = cost.py(f"thin_{h}", thin_nested, out["base"][h], outs,
                             out["base"]["lines"][h], lpu, rng)
        for a in depth_arms:
            lab = a[len("depth"):]
            record(a, {"arm": a, "lines": {h: res[h][lab] for h in HALVES},
                       "thinned_from": {h: out["base"][h] for h in HALVES},
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
        fdir = Path(tempfile.mkdtemp(prefix=f"filter_{a}_", dir=tmp))
        fcmd = filter_command(src["unfiltered"], fdir, pipeline=cfg["pipeline"], arm=a,
                              paired_end=cfg["paired_end"], chrsz=cfg["chrsz"], nth=cfg["nth"],
                              mem_gb=cfg["mem_gb"], multimapping=cfg["multimapping"])
        cost.shell(f"filter_{a}", bam_arm.apptainer(fcmd, cfg["sif"], tmp, str(fdir)))
        want = ".filt.bam" if a == "nodedup" else ".nodup.bam"
        bams = [b for b in fdir.glob("*.bam") if b.name.endswith(want)]
        if len(bams) != 1:
            raise SystemExit(f"filter {a}: expected one *{want} in {fdir}, got {bams}")
        named = Path(tmp) / f"{exp}.{unit}.{a}.named.tagAlign.gz"
        cost.shell(f"named_ta_{a}", bam_arm.apptainer(
            named_ta_command(bams[0], named, paired_end=cfg["paired_end"], tmp=tmp,
                             nth=cfg["nth"]), cfg["sif"], tmp, tmp))
        salt = f"{seed}|{exp}|{unit}|".encode()
        d = readset_dir(cfg, unit, a)
        lines = cost.py(f"split_named_{a}", split_named, named, d / "pr1.tagAlign.gz",
                        d / "pr2.tagAlign.gz", salt)
        named.unlink()
        shutil.rmtree(fdir, ignore_errors=True)
        record(a, {"arm": a, "lines": lines, "filter_command": fcmd,
                   "source": {"kind": "bam", "path": str(src["unfiltered"]),
                              "md5": records.md5_file(src["unfiltered"])},
                   "split": {"method": "blake2b(salt + read name) low bit",
                             "salt": salt.decode()}})
    return out


def write_measurement(cfg, unit, arm, halves_counts: dict, half_tas: dict, half_lines: dict,
                      cost: Cost, extra_prov: dict, signal=None) -> str:
    """Stage one measurement dir under `out/.<mid>.partial`, then rename it into place."""
    mid = measurement_id(cfg["experiment"], unit, cfg["layer"], arm)
    final = Path(cfg["out"]) / mid
    stage = Path(cfg["out"]) / f".{mid}.partial"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    lpu = cfg["lpu"]
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
                     counts_path=final / "counts.h5", sha256=sha256_file(stage / "counts.h5"))
    (stage / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    prov = {
        "schema": 1, "measurement_id": mid, "experiment": cfg["experiment"], "unit": unit,
        "arm": arm, "layer": cfg["layer"], "lines_per_fragment": lpu,
        "halves": {h: {"tagalign": half_tas[h], "n_lines": half_lines[h]} for h in HALVES},
        "chrom_sizes": {"path": str(cfg["chrom_sizes_path"]), "sizes": cfg["sizes"]},
        "pipeline": {"name": PIPELINE_NAME[cfg["pipeline"]],
                     "version": records.PIPELINES[records.PIPELINE_REPO[cfg["pipeline"]]][0],
                     "sif": cfg.get("sif_info")},
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


def build(cfg: dict) -> list:
    cost = Cost()
    t0 = time.time()
    sizes = cfg["sizes"]
    exp, layer = cfg["experiment"], cfg["layer"]
    units = (["pooled"] if len(cfg["sources"]) == 1
             else [f"br{i + 1}" for i in range(len(cfg["sources"]))])
    if cfg["splitter"] == "spr" or any(s["kind"] == "bam" for s in cfg["sources"]):
        cfg["sif_info"] = sif_info(cfg["sif"], cfg["pipeline"])
        cfg["image_digest"] = "sha256:" + cfg["sif_info"]["sha256"]
    written, per_unit = [], {}

    def exists(unit, arm):
        return (Path(cfg["out"]) / measurement_id(exp, unit, layer, arm) / "provenance.json").is_file()

    for unit, src in zip(units, cfg["sources"]):
        rs = build_readsets(cfg, unit, src, cost)
        per_unit[unit] = rs
        for arm in [a for a in cfg["arms"] if a in rs]:
            want_signal = cfg["signal"] and arm == "base" and unit == "pooled"
            if exists(unit, arm):
                print(f"[align_views] skip existing {measurement_id(exp, unit, layer, arm)}")
                continue
            hc = {h: cost.py(f"count_{arm}_{h}", bin25.counts_from_tagalign, rs[arm][h], sizes)
                  for h in HALVES}
            extra = {"readset": rs[arm], "references": {}}
            sig = None
            if want_signal:
                sig = build_signal(cfg, [src], {h: [rs["base"][h]] for h in HALVES}, cost)
                extra["signal"] = sig.pop("_prov")
            if unit == "pooled" and arm == "base":
                extra["references"] = cfg["references"]
            written.append(write_measurement(cfg, unit, arm, hc, {h: [rs[arm][h]] for h in HALVES},
                                             rs[arm]["lines"], cost, extra, sig))

    if len(units) > 1:
        arms = [a for a in cfg["arms"] if all(a in per_unit[u] for u in units)]
        for arm in arms:
            if exists("pooled", arm):
                continue
            hc, lines, tas = {}, {}, {}
            for h in HALVES:
                parts = [read_h5(Path(cfg["out"]) / measurement_id(exp, u, layer, arm) / f"{h}.h5")
                         for u in units]
                hc[h] = {c: sum(p[c].astype(np.uint64) for p in parts).astype(np.uint32)
                         for c in parts[0]}
                lines[h] = sum(per_unit[u][arm]["lines"][h] for u in units)
                tas[h] = [per_unit[u][arm][h] for u in units]
            extra = {"pooled_from": [measurement_id(exp, u, layer, arm) for u in units],
                     "references": cfg["references"] if arm == "base" else {}}
            sig = None
            if cfg["signal"] and arm == "base":
                sig = build_signal(cfg, cfg["sources"], tas, cost)
                extra["signal"] = sig.pop("_prov")
            written.append(write_measurement(cfg, "pooled", arm, hc, tas, lines, cost, extra, sig))

    rec = {"experiment": exp, "written": written, "steps": cost.steps,
           "wall_s": round(time.time() - t0, 1),
           "self_max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
           "children_max_rss_gb": round(
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20, 3),
           "slurm_job_ids": bam_arm._job_ids(), "finished_utc": utc()}
    d = Path(cfg["work"]) / exp
    d.mkdir(parents=True, exist_ok=True)
    (d / f"BUILD_COST.{layer}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"[align_views] {exp}: {len(written)} measurements written, wall {rec['wall_s']} s",
          flush=True)
    return written


def build_signal(cfg, sources, half_tas: dict, cost: Cost) -> dict:
    """signal.h5, pr1_signal.h5, pr2_signal.h5 for the pooled base. The treatment of `signal.h5` is
    the source tagAlign itself when there is one tagAlign source (so it reproduces the base run),
    else the concatenation of the halves."""
    tmp, p = Path(cfg["tmp"]), cfg["pipeline"]
    if p == "chip" and not cfg["control_tagalign"]:
        raise SystemExit("--signal for chip needs --control-tagalign")
    ctl = [cfg["control_tagalign"]] if cfg["control_tagalign"] else []
    Path(f"{tmp}/work").mkdir(parents=True, exist_ok=True)

    def cat(paths, name):
        out = tmp / name
        cost.shell(f"concat_{name}", "cat " + " ".join(shlex.quote(str(x)) for x in paths)
                   + f" > {out}")
        return out

    treat = {"signal.h5": (sources[0]["path"] if len(sources) == 1
                           and sources[0]["kind"] == "tagalign"
                           else cat(half_tas["pr1"] + half_tas["pr2"],
                                    f"{cfg['experiment']}.pooled.tagAlign.gz"))}
    for h in HALVES:
        treat[f"{h}_signal.h5"] = (half_tas[h][0] if len(half_tas[h]) == 1
                                   else cat(half_tas[h], f"{cfg['experiment']}.{h}.tagAlign.gz"))
    out, prov = {}, {}
    for name, ta in treat.items():
        od = tmp / f"macs2_{name[:-3]}"
        od.mkdir(parents=True, exist_ok=True)
        inner, bw = signal_command([ta] + ctl, od, pipeline=p, metadata=cfg["metadata"],
                                   chrsz=cfg["chrsz"])
        cmd = bam_arm.apptainer(inner, cfg["sif"], tmp, f"{tmp}/work")
        cost.shell(f"macs2_{name[:-3]}", cmd, cwd=f"{tmp}/work")
        out[name] = cost.py(f"bin_{name[:-3]}", bin25.pval_from_bigwig, bw, cfg["sizes"],
                            tmp / "bdg")
        prov[name] = {"command": cmd, "treatment": str(ta), "control": ctl[0] if ctl else None,
                      "bigwig_md5": records.md5_file(bw), "recorded_in": cfg["metadata"]}
        shutil.rmtree(od, ignore_errors=True)
    if ctl:
        prov["control_md5"] = records.md5_file(ctl[0])
    out["_prov"] = prov
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
            if abs(got - q * base_reads) > 1e-3 * q * base_reads:
                probs.append(f"depth {got} not within 0.1% of {q} x {base_reads}")
            for h in HALVES:
                b = read_h5(bd / f"{h}.h5")
                if any(np.any(arrs[f"{h}.h5"][c] > b[c]) for c in b):
                    probs.append(f"{h}: thinned counts exceed base")
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


def verify(root, recount_chrom="chr21") -> int:
    root = Path(root)
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").is_file())
    if not dirs:
        print(f"no measurement directories under {root}")
        return 1
    all_meta = {p.name: json.loads((p / "meta.json").read_text()) for p in dirs}
    by_exp, bad = {}, 0
    for p in dirs:
        probs = verify_measurement(p, all_meta, recount_chrom)
        by_exp.setdefault(all_meta[p.name]["experiment"], []).append((p.name, probs))
    for exp, rows in by_exp.items():
        nbad = sum(1 for _, pr in rows if pr)
        bad += nbad
        print(f"{exp}: {len(rows)} measurements, {nbad} failing")
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
        "sif": a.sif or bam_arm.SIF[a.pipeline], "chrsz": a.chrsz, "nth": a.nth,
        "mem_gb": a.mem_gb, "signal": a.signal, "metadata": a.metadata,
        "control_tagalign": a.control_tagalign, "references": refs,
    }


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
    b.add_argument("--pipeline", default="chip", choices=("chip", "atac"))
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
    v = sub.add_parser("verify", help="check every measurement directory under a root")
    v.add_argument("--measurements", required=True)
    v.add_argument("--recount-chrom", default="chr21")
    a = ap.parse_args(argv)
    if a.cmd == "verify":
        return verify(a.measurements, a.recount_chrom)
    cfg = config_from_args(a)
    if a.signal and not a.metadata:
        raise SystemExit("--signal needs --metadata (the base run's recorded signal command)")
    if a.multimapping is None:
        cfg["multimapping"] = FILTER_DEFAULTS[a.pipeline]["multimapping"]
    build(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
