"""t112 — build one BAM-route counterfactual arm: its input tagAlign, then the pipeline's own
MACS2 signal step, then the 25 bp product.

A BAM-route arm changes the INPUT of the ENCODE signal step, or one flag of it, and runs that step
otherwise unchanged. Nothing here re-implements a pipeline step: the treatment tagAlign is thinned
or mixed with `encode_task_subsample_ctl.py` (the pipeline's own SE subsampler, `shuf` seeded by the
input's uncompressed byte count) inside the pipeline's own SIF, and the signal command is the
track's recorded `calls["chip.macs2_signal_track"][0].commandLine` with only the tagAlign paths,
`--fraglen` and `--out-dir` substituted.

Three details are load-bearing and were verified on Nibi on 2026-09-17:

* **`--cleanenv`.** Caper ran every task as `singularity exec --cleanenv`. Without it the host's
  exported `which` shell function leaks into the container and `python3 $(which <script>)` breaks
  (C5 hit this). Every command below therefore carries absolute paths and no host environment.
* **`--chrsz`.** The recorded command names the Cromwell input copy of `GRCh38_EBV.chrom.sizes.tsv`,
  and those Cromwell trees are gone. `CHRSZ` here is the refcache original; the basename is asserted
  to match what was recorded, so the substitution is a relocation, not a change of reference.
* **`--ratio`.** `macs2 callpeak` (2.2.4) uses `--ratio` only when it differs from 1.0, and it then
  replaces the ChIP/control depth ratio MACS2 would otherwise compute as
  `treat_total / control_total` — the tag totals of the two BED files, with `--keep-dup all` and no
  `--scale-to`/`--to-large`, so nothing is filtered and the larger file is scaled down. Probed in
  the SIF on a synthetic chr21 pair: passing that exact quotient reproduces `control_lambda.bdg`
  byte for byte, and any other value changes it. That is what the `ratio` arm turns.

  So `--ratio` is computed here as `k * lines(T) / lines(C)` from the two tagAligns this arm
  actually feeds MACS2, never from a pinned read count, and the computed value — not the rows TSV
  column — is what `covariates.knob_value` records, because it is the number MACS2 was given. `k`
  comes from the level (`k0.5`, `k2`). The quotient is rounded the way `arms.ratio_value` rounds
  it, so the two agree while the pinned counts are right and `records.validate` reports it when
  they are not — which is how the first wrong control count was caught (`$CF/ta/LINES.tsv`).

The product is written by `records.py` and the arrays by `bin25.py`; `counts25.npz` is always the
counts of the tagAlign MACS2 actually consumed, so arms 7-10 are count-identical to base by
construction. The base arm additionally bins the pipeline's own bigwig into `pval25.npz` and its
own rebuild into `<work>/<pid>/rebuild_pval25.npz`, which is what the `base_rebuild` check reads.

`plan_commands` is pure — given a row, a `$CF` and an `$EIC` it returns the exact shell strings
`run` will execute, in order, and those strings are what `provenance.commands` records. Binning is
not in that list: it is this module calling `bin25`, pinned by `provenance.code.git_sha`.

Only `pipeline == "chip"` rows are handled; C13 adds the atac branch for C12M02.

    python3 bam_arm.py plan  --rows ROWS.tsv --index I
    python3 bam_arm.py run   --rows ROWS.tsv --index I [--products DIR] [--work DIR] [--ta-dir DIR]
    python3 bam_arm.py smoke --part ab|c|ok [--cf DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arms  # noqa: E402  (sibling, stdlib only)
import records  # noqa: E402

CF = "/scratch/mforooz/t112_cf"
EIC = arms.EIC
CHRSZ = ("/scratch/mforooz/EIC_REPRO/003/refcache/c52f52c7bfa357f55a39b1de7e4d0b0c/"
         "GRCh38_EBV.chrom.sizes.tsv")

SIF = {"chip": f"{EIC}/sif/chip-seq-pipeline_v2.2.2.sif",
       "atac": f"{EIC}/sif/atac-seq-pipeline_v2.2.3.sif"}
#: `calls` key and script name of the signal step, per pipeline.
SIGNAL_CALL = {"chip": "chip.macs2_signal_track", "atac": "atac.macs2_signal_track"}
SIGNAL_SCRIPT = {"chip": "encode_task_macs2_signal_track_chip.py",
                 "atac": "encode_task_macs2_signal_track_atac.py"}
SUBSAMPLE_SCRIPT = "encode_task_subsample_ctl.py"

#: the signal script inside the chip SIF, and its md5 (verified 2026-09-17). The `ratio` arm copies
#: this file out, edits the one string below, and runs the copy with the SIF's `src` on PYTHONPATH.
CHIP_SRC_DIR = "/software/chip-seq-pipeline/src"
RATIO_SCRIPT = f"{CHIP_SRC_DIR}/{SIGNAL_SCRIPT['chip']}"
RATIO_SCRIPT_MD5 = "6039e1c382ef1e8afa3dbfd5ea103ba8"
RATIO_OLD = "--nomodel --shift {shiftsize} --extsize {extsize} --keep-dup all -B --SPMR'"


def ratio_new(r: float) -> str:
    """`RATIO_OLD` with `--ratio <r>` appended inside the same quoted string, at full precision.

    Appending a literal number rather than a `{}` field matters: the line it sits on is a `.format`
    argument list in the pipeline's source, and a new field would raise `KeyError` there.
    """
    return RATIO_OLD[:-1] + " --ratio " + repr(float(r)) + "'"

#: the one-liner that applies that edit. Run as written, so `provenance.commands` is literal.
PATCH_PY = ("import pathlib,sys;p=pathlib.Path(sys.argv[1]);s=p.read_text();"
            "assert s.count(sys.argv[2])==1,'expected exactly one occurrence';"
            "p.write_text(s.replace(sys.argv[2],sys.argv[3]))")

MAIN_CHROMS = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX",)

#: the ChIP arms this module builds, and the shape of each one's input.
BAM_ARMS = ("base", "depth", "abproxy", "ratio", "ctlid", "ctldepth", "extsize")


# --------------------------------------------------------------------------------------------
# names the pipeline's own helpers produce


def human_readable_number(num: int) -> str:
    """`encode_lib_common.human_readable_number`, verbatim: 15000000 -> '15M', 3750000 -> '3M'."""
    for unit in ['', 'K', 'M', 'G', 'T', 'P']:
        if abs(num) < 1000:
            return '{}{}'.format(num, unit)
        num = int(num / 1000.0)
    return '{}{}'.format(num, 'E')


def strip_ext_ta(ta) -> str:
    """`encode_lib_common.strip_ext_ta`, verbatim."""
    return re.sub(r'\.(tagAlign|TagAlign|ta|Ta)\.gz$', '', str(ta))


def subsample_output(ta, n: int, out_dir) -> str:
    """The path `subsample_ta_se(ta, n, non_mito=False, ..., out_dir)` writes."""
    prefix = os.path.join(str(out_dir), os.path.basename(strip_ext_ta(ta)))
    return '{}.{}.tagAlign.gz'.format(prefix, human_readable_number(n))


def treatment_tagalign(track: str, ta_dir) -> str:
    """C5's treatment tagAlign for a track: the BAM basename without the pipeline's shard prefix,
    plus the `--subsample` suffix `bam2ta` appends."""
    t = arms.TRACKS[track]
    stem = Path(t["treat_bam"]).name
    for pre in ("filter_shard0_",):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    stem = stem[:-len(".bam")]
    return f"{ta_dir}/{stem}.{human_readable_number(t['subsample'])}.tagAlign.gz"


def control_tagalign(acc: str, ta_dir) -> str:
    """C5's full control tagAlign (`bam2ta_ctl --subsample 0`, so no suffix)."""
    stem = Path(arms.CONTROLS[acc]["bam"]).name
    for pre in ("filter_ctl_shard0_",):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    return f"{ta_dir}/{stem[:-len('.bam')]}.tagAlign.gz"


def ctldepth_tagalign(acc: str, level: str, n: int, ta_dir) -> str:
    """C5's arm-9 thinned control: one file per (control, q), named by the same subsampler.

    `n` comes from the rows TSV. C5 rebuilds these files whenever that number is re-pinned, so if
    the computed name is absent and the directory holds exactly one tagAlign, that one is used —
    the directory, not the suffix, is what identifies (control, q).
    """
    d = f"{ta_dir}/ctldepth/{acc}__{level}"
    want = subsample_output(control_tagalign(acc, ta_dir), n, d)
    if not Path(want).is_file():
        there = sorted(Path(d).glob("*.tagAlign.gz")) if Path(d).is_dir() else []
        if len(there) == 1:
            return str(there[0])
    return want


# --------------------------------------------------------------------------------------------
# the recorded signal command


def signal_command_line(track: str, eic=EIC, pipeline: str = "chip") -> str:
    """`calls[<pipeline>.macs2_signal_track][0].commandLine` of a track's base run."""
    meta = json.loads(Path(f"{eic}/results/{track}/metadata.json").read_text())
    calls = meta["calls"][SIGNAL_CALL[pipeline]]
    if len(calls) != 1:
        raise ValueError(f"{track}: {len(calls)} macs2_signal_track calls, expected 1")
    return calls[0]["commandLine"]


def parse_signal_command(cmdline: str, pipeline: str = "chip") -> dict:
    """Split a recorded commandLine into `{invocation, tas, flags}`; `flags` keeps recorded order.

    The recorded form is `set -e`, then `python3 $(which <script>)`, then one or two tagAlign paths,
    then `--flag value` pairs. Everything but the tagAligns, `--fraglen` and `--out-dir` is carried
    through untouched by `build_signal_command`.
    """
    toks = cmdline.replace("\\\n", " ").replace("\n", " ").split()
    if toks[:2] == ["set", "-e"]:
        toks = toks[2:]
    script = SIGNAL_SCRIPT[pipeline]
    hit = [i for i, t in enumerate(toks) if script in t]
    if not hit:
        raise ValueError(f"{script} not in the recorded command: {cmdline!r}")
    invocation, rest = toks[:hit[0] + 1], toks[hit[0] + 1:]

    tas = []
    while rest and not rest[0].startswith("--"):
        tas.append(rest.pop(0))
    if not 1 <= len(tas) <= 2 or not all(t.endswith(".tagAlign.gz") for t in tas):
        raise ValueError(f"expected 1-2 tagAlign positionals, got {tas}")

    flags = []
    while rest:
        name = rest.pop(0)
        if not name.startswith("--"):
            raise ValueError(f"expected a flag, got {name!r} in {cmdline!r}")
        value = rest.pop(0) if rest and not rest[0].startswith("--") else None
        flags.append((name, value))
    return {"invocation": invocation, "tas": tas, "flags": flags}


def build_signal_command(parsed: dict, tas, out_dir, *, fraglen=None, chrsz=CHRSZ,
                         invocation=None) -> str:
    """The recorded command with only the tagAligns, `--fraglen`, `--chrsz` and `--out-dir` set.

    `--chrsz` is relocated, not changed: the recorded path is a Cromwell input copy that no longer
    exists, so its basename is asserted against `chrsz` and the path replaced.
    """
    out = list(invocation or parsed["invocation"]) + [str(t) for t in tas]
    seen = set()
    for name, value in parsed["flags"]:
        seen.add(name)
        if name == "--out-dir":
            continue
        if name == "--fraglen" and fraglen is not None:
            value = str(fraglen)
        if name == "--chrsz":
            if os.path.basename(value) != os.path.basename(chrsz):
                raise ValueError(f"--chrsz basename {value!r} != {chrsz!r}")
            value = str(chrsz)
        out.append(name)
        if value is not None:
            out.append(value)
    for required, value in (("--gensz", "hs"), ("--chrsz", None), ("--pval-thresh", "0.01"),
                            ("--fraglen", None)):
        if required not in seen:
            raise ValueError(f"recorded command has no {required}: {parsed['flags']}")
        if value is not None and dict(parsed["flags"])[required] != value:
            raise ValueError(f"recorded {required} is "
                             f"{dict(parsed['flags'])[required]!r}, not {value!r}")
    return " ".join(out + ["--out-dir", str(out_dir)])


def ratio_k(level: str) -> float:
    """The `k` of a ratio level: `k0.5` -> 0.5, `k2` -> 2.0."""
    m = re.fullmatch(r"k(\d+(?:\.\d+)?)", level)
    if not m:
        raise ValueError(f"ratio level {level!r} is not k<number>")
    return float(m.group(1))


def ratio_for(level: str, treatment, control, sig: int = 10) -> float:
    """`--ratio` for the ratio arm: `k` times the ChIP/control tag-total quotient MACS2 itself would
    compute for these two files, counted from the files rather than from any pinned read count.

    `sig` mirrors `arms.ratio_value`'s `:.10g`, in the same association order, so the value written
    to `covariates.knob_value` equals the rows TSV column exactly while still being derived from
    the bytes — if a pinned count ever drifts from its tagAlign again, `records.validate` says so.
    `sig=None` is the raw quotient, which at k=1 is bit-exactly MACS2's own default scaling.
    """
    x = ratio_k(level) * _lines(treatment) / _lines(control)
    return x if sig is None else float(f"{x:.{sig}g}")


def signal_bigwig(tas, out_dir) -> str:
    """The `*.pval.signal.bigwig` the signal script writes, from `macs2_signal_track`'s own naming."""
    base = os.path.basename(strip_ext_ta(tas[0]))
    prefix = base if len(tas) == 1 else f"{base}_x_{os.path.basename(strip_ext_ta(tas[1]))}"
    if len(prefix) > 200:
        prefix = f"{base}_x_control"
    return f"{out_dir}/{prefix}.pval.signal.bigwig"


# --------------------------------------------------------------------------------------------
# commands


def apptainer(inner: str, sif: str, tmp, workdir) -> str:
    """One pipeline command inside the SIF, with the binds and the cwd every t112 job uses."""
    return (f"apptainer exec --cleanenv -B /scratch -B /project -B {tmp} --pwd {workdir} "
            f"{sif} bash -c {shlex.quote(inner)}")


def plan(row: dict, cf=CF, eic=EIC, *, tmp=None, ta_dir=None, work=None, products=None,
         chrsz=CHRSZ, pipeline_pval=True, ratio=None) -> dict:
    """Everything one row needs: input paths, the shell commands in run order, and what to bin.

    It executes nothing. It reads the track's `metadata.json` under `eic`, and for the ratio arm
    the line counts of the two tagAligns (pass `ratio` to skip that); every other number comes from
    the row.
    """
    track, arm, level = row["track"], row["arm"], row["level"]
    pid = row["pid"]
    if row.get("route") != "bam":
        raise ValueError(f"{pid}: route {row.get('route')!r}, not 'bam'")
    if row.get("pipeline") != "chip":
        raise ValueError(f"{pid}: pipeline {row.get('pipeline')!r}; only 'chip' is implemented "
                         "(C13 adds the atac branch)")
    if arm not in BAM_ARMS:
        raise ValueError(f"{pid}: unknown bam arm {arm!r}")

    t = arms.TRACKS[track]
    kv = json.loads(row["knob_value"]) if isinstance(row["knob_value"], str) else row["knob_value"]
    tmp = str(tmp or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir())
    ta_dir = str(ta_dir or f"{cf}/ta")
    work = f"{work or f'{cf}/bamarms'}/{pid}"
    product = f"{products or f'{cf}/products'}/{pid}"
    sif = SIF["chip"]

    W = f"{tmp}/work"           # cwd of every container command
    TAOUT = f"{tmp}/ta"         # tagAligns this arm builds
    OUT = f"{tmp}/out"          # the signal script's --out-dir
    PATCHED = f"{tmp}/patched"  # the ratio arm's edited copy

    cmds = [f"mkdir -p {W} {TAOUT} {OUT} {work} {product}"]

    treatment = src_treatment = treatment_tagalign(track, ta_dir)
    control = control_tagalign(t["ctl_acc"], ta_dir)
    control_acc, control_source = t["ctl_acc"], "matched"
    fraglen = t["fraglen"]
    subsamples = []          # provenance.subsample_seed entries are filled in by run()
    patch = None

    if arm == "depth":
        out = subsample_output(treatment, kv, TAOUT)
        cmds.append(apptainer(
            f"python3 $(which {SUBSAMPLE_SCRIPT}) {treatment} --subsample {kv} --out-dir {TAOUT}",
            sif, tmp, W))
        subsamples.append(treatment)
        treatment = out

    elif arm == "abproxy":
        n_ctl = arms.abproxy_n_ctl(kv)
        n_treat = t["subsample"] - n_ctl
        a = subsample_output(treatment, n_treat, TAOUT)
        b = subsample_output(control, n_ctl, TAOUT)
        mixed = f"{TAOUT}/{os.path.basename(strip_ext_ta(treatment))}.abproxy_{level}.tagAlign.gz"
        cmds += [
            apptainer(f"python3 $(which {SUBSAMPLE_SCRIPT}) {treatment} "
                      f"--subsample {n_treat} --out-dir {TAOUT}", sif, tmp, W),
            apptainer(f"python3 $(which {SUBSAMPLE_SCRIPT}) {control} "
                      f"--subsample {n_ctl} --out-dir {TAOUT}", sif, tmp, W),
            apptainer(f"zcat {a} {b} | gzip -nc > {mixed}", sif, tmp, W),
        ]
        subsamples += [treatment, control]
        treatment = mixed

    elif arm == "ctlid":
        if kv == "none":
            control, control_acc, control_source = None, None, None
        else:
            control = control_tagalign(kv, ta_dir)
            control_acc, control_source = kv, "other"

    elif arm == "ctldepth":
        control = ctldepth_tagalign(t["ctl_acc"], level, kv, ta_dir)

    elif arm == "extsize":
        fraglen = kv

    tas = [treatment] + ([control] if control else [])
    parsed = parse_signal_command(signal_command_line(track, eic), "chip")

    if arm == "ratio":
        # never the rows TSV column: that is k * 30000000 / a pinned BAM read count, and the BAM
        # read count is not the control tagAlign's line count.
        kv = ratio if ratio is not None else ratio_for(level, treatment, control)

        copy = f"{PATCHED}/{SIGNAL_SCRIPT['chip']}"
        new = ratio_new(kv)
        cmds = [cmds[0] + f" {PATCHED}"]
        cmds += [
            apptainer(f"cp {RATIO_SCRIPT} {copy}", sif, tmp, W),
            f"python3 -c {shlex.quote(PATCH_PY)} {copy} {shlex.quote(RATIO_OLD)} "
            f"{shlex.quote(new)}",
        ]
        inner = build_signal_command(
            parsed, tas, OUT, fraglen=fraglen, chrsz=chrsz,
            invocation=[f"PYTHONPATH={CHIP_SRC_DIR}", "python3", copy])
        patch = {"script": RATIO_SCRIPT, "copy": copy, "old": RATIO_OLD, "new": new,
                 "expect_md5": RATIO_SCRIPT_MD5}
    else:
        inner = build_signal_command(parsed, tas, OUT, fraglen=fraglen, chrsz=chrsz)

    cmds.append(apptainer(inner, sif, tmp, W))
    bigwig = signal_bigwig(tas, OUT)
    cmds += [f"test -s {bigwig}", f"cp {bigwig} {work}/"]

    pipeline_bigwig = t["pval_bigwig"] if (arm == "base" and pipeline_pval) else None
    return {
        "pid": pid, "track": track, "arm": arm, "level": level, "knob_value": kv,
        "sif": sif, "tmp": tmp, "work": work, "product": product, "out_dir": OUT,
        "treatment": treatment, "control": control,
        "control_accession": control_acc, "control_source": control_source,
        "fraglen": fraglen, "commands": cmds,
        "bigwig": bigwig, "kept_bigwig": f"{work}/{os.path.basename(bigwig)}",
        "pipeline_bigwig": pipeline_bigwig,
        "rebuild_npz": f"{work}/rebuild_pval25.npz" if pipeline_bigwig else None,
        "subsample_inputs": subsamples, "patch": patch,
        "ratio_formula": ("k * lines(treatment tagAlign) / lines(control tagAlign)"
                          if arm == "ratio" else None),
        # `inputs` must already be on disk before the first command; `fed` is what MACS2 read,
        # which for depth and abproxy is a tagAlign these commands are about to build.
        "inputs": [x for x in dict.fromkeys([src_treatment, control, pipeline_bigwig]) if x],
        "fed": tas,
    }


def plan_commands(row: dict, cf=CF, eic=EIC, **kw) -> list:
    """The exact shell strings `run` executes for this row, in order."""
    return plan(row, cf, eic, **kw)["commands"]


# --------------------------------------------------------------------------------------------
# execution


def sh(cmd: str, cwd=None, check=True):
    """One recorded shell string, run as written."""
    print(f"+ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check)


def _uncompressed_bytes(path) -> int:
    out = subprocess.run(f"zcat -f {shlex.quote(str(path))} | wc -c", shell=True, check=True,
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def _lines(path) -> int:
    out = subprocess.run(f"zcat -f {shlex.quote(str(path))} | wc -l", shell=True, check=True,
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def check_sif(sif: str, repo: str = "ENCODE-DCC/chip-seq-pipeline2") -> tuple:
    """(md5, sha256) of the image, refusing to run if the md5 is not the pinned one."""
    md5 = records.md5_file(sif)
    pinned = records.PIPELINES[repo][1]
    if md5 != pinned:
        raise SystemExit(f"SIF md5 {md5} != pinned {pinned} for {sif}")
    sha = subprocess.run(["sha256sum", sif], check=True, capture_output=True,
                         text=True).stdout.split()[0]
    return md5, sha


def apply_patch(p: dict) -> dict:
    """Read back the ratio arm's copy, prove the edit is one line, and return the provenance block.

    Reversing the edit must reproduce the SIF's own script byte for byte: `p["expect_md5"]` is
    checked against the md5 of that reconstruction, so an image whose signal script has moved on
    stops the arm here instead of silently producing a product from a different recipe.
    """
    import difflib
    import hashlib
    copy = Path(p["copy"])
    patched = copy.read_text()
    original = patched.replace(p["new"], p["old"])
    if original.count(p["old"]) != 1 or p["new"] not in patched:
        raise SystemExit(f"{copy}: the --ratio edit is not the single expected substitution")
    original_md5 = hashlib.md5(original.encode()).hexdigest()
    if original_md5 != p["expect_md5"]:
        raise SystemExit(f"{copy}: unpatched md5 {original_md5} != pinned {p['expect_md5']}")
    diff = list(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                     "original", "patched", n=1))
    removed = [d for d in diff if d.startswith("-") and not d.startswith("---")]
    added = [d for d in diff if d.startswith("+") and not d.startswith("+++")]
    if len(removed) != 1 or len(added) != 1:
        raise SystemExit(f"{copy}: diff changed {len(removed)} lines, expected exactly 1")
    return {"original_md5": original_md5,
            "patched_md5": hashlib.md5(patched.encode()).hexdigest(),
            "diff": "".join(diff)}


def run(row: dict, cf=CF, eic=EIC, *, tmp=None, ta_dir=None, work=None, products=None,
        chrsz=CHRSZ, pipeline_pval=True, job_ids=None, code_dir=None, git_sha=None) -> dict:
    """Build one BAM-route arm end to end and write its product dir. Returns the plan it ran."""
    import bin25

    p = plan(row, cf, eic, tmp=tmp, ta_dir=ta_dir, work=work, products=products, chrsz=chrsz,
             pipeline_pval=pipeline_pval)
    md5, sha = check_sif(p["sif"])
    for f in p["inputs"]:
        if not Path(f).is_file():
            raise SystemExit(f"{p['pid']}: input missing: {f}")

    seeds = [{"file": f, "uncompressed_bytes": _uncompressed_bytes(f)}
             for f in p["subsample_inputs"]]
    Path(f"{p['tmp']}/work").mkdir(parents=True, exist_ok=True)
    patch = None
    for cmd in p["commands"]:
        sh(cmd, cwd=f"{p['tmp']}/work")
        if p["patch"] and cmd.startswith("python3 -c"):
            patch = apply_patch(p["patch"])   # before the signal command runs, not after
    if p["patch"] and patch is None:
        raise SystemExit(f"{p['pid']}: the --ratio patch step never ran")

    kept = Path(p["kept_bigwig"])
    if not kept.is_file() or kept.stat().st_size == 0:
        raise SystemExit(f"{p['pid']}: no bigwig at {kept}")

    # Arrays are built in $SLURM_TMPDIR and published only once they exist and are non-empty, so a
    # task that dies mid-write leaves no half-product for `records.validate` to accept.
    sizes = bin25.load_chrsz(chrsz)
    product = Path(p["product"])
    stage = Path(f"{p['tmp']}/stage/{p['pid']}")
    stage.mkdir(parents=True, exist_ok=True)
    staged = []                                   # (staged path, final path)

    tstats = {}
    bin25.write_npz(stage / "counts25.npz",
                    bin25.counts_from_tagalign(p["treatment"], sizes, stats=tstats))
    staged.append((stage / "counts25.npz", product / "counts25.npz"))
    control_reads = None
    if p["control"]:
        cstats = {}
        bin25.write_npz(stage / "control_counts25.npz",
                        bin25.counts_from_tagalign(p["control"], sizes, stats=cstats))
        control_reads = cstats["n_lines"]
        staged.append((stage / "control_counts25.npz", product / "control_counts25.npz"))

    pval_src = p["pipeline_bigwig"] or str(kept)
    bin25.write_npz(stage / "pval25.npz",
                    bin25.pval_from_bigwig(pval_src, sizes, f"{p['tmp']}/bdg"))
    staged.append((stage / "pval25.npz", product / "pval25.npz"))
    if p["rebuild_npz"]:
        bin25.write_npz(stage / "rebuild_pval25.npz",
                        bin25.pval_from_bigwig(kept, sizes, f"{p['tmp']}/bdg"))
        staged.append((stage / "rebuild_pval25.npz", Path(p["rebuild_npz"])))
    # provenance records the FINAL path but the md5 of the bytes in hand, since nothing is
    # published until every one of them exists.
    outputs = [(s, f) for s, f in staged] + [(kept, kept)]

    t = arms.TRACKS[p["track"]]
    records.write_covariates(
        stage / "covariates.json",
        pid=p["pid"], biosample=row["biosample"], track=p["track"], cell=row["cell"],
        assay=row["assay"], arm=p["arm"], level=p["level"], knob=row["knob"],
        knob_value=p["knob_value"], depth=tstats["n_lines"], read_length=t["read_length"],
        run_type=t["run_type"], fraglen=p["fraglen"],
        control=None if not p["control"] else {"accession": p["control_accession"],
                                               "source": p["control_source"],
                                               "reads": control_reads})
    code_dir = code_dir or os.environ.get("KIT") or str(Path(__file__).resolve().parents[2])
    git_sha = git_sha or os.environ.get("T112_GIT_SHA") or _git_sha(code_dir)
    records.write_provenance(
        stage / "provenance.json", pid=p["pid"], route="bam",
        pipeline={"repo": "ENCODE-DCC/chip-seq-pipeline2", "release": "v2.2.2", "sif": p["sif"],
                  "sif_md5": md5, "sif_sha256": sha},
        commands=p["commands"],
        inputs=[{"path": f, "md5": records.md5_file(f)}
                for f in dict.fromkeys(p["inputs"] + p["fed"])],
        outputs=[{"path": str(f), "md5": records.md5_file(s)} for s, f in outputs],
        subsample_seed=seeds, patched_script=patch, caper=None,
        slurm_job_ids=job_ids if job_ids is not None else _job_ids(),
        code={"snapshot_dir": code_dir, "git_sha": git_sha})
    staged += [(stage / n, product / n) for n in ("covariates.json", "provenance.json")]

    publish(staged)
    print(f"[t112] {p['pid']} done -> {product}", flush=True)
    return p


def publish(staged) -> None:
    """Copy `(staged, final)` pairs out of `$SLURM_TMPDIR`, refusing anything empty and verifying
    every copy by md5. The records are written last, so a product dir is never valid before it is
    complete."""
    import shutil
    for src, _ in staged:
        if not Path(src).is_file() or Path(src).stat().st_size == 0:
            raise SystemExit(f"refusing to publish: {src} is missing or empty")
    for src, dst in staged:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if records.md5_file(src) != records.md5_file(dst):
            raise SystemExit(f"copy md5 mismatch: {src} -> {dst}")


def _job_ids() -> list:
    a, i, j = (os.environ.get(k) for k in ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
                                           "SLURM_JOB_ID"))
    if a and i:
        return [f"{a}_{i}"]
    return [j] if j else []


def _git_sha(code_dir) -> str:
    f = Path(code_dir) / "GIT_SHA"
    if f.is_file():
        return f.read_text().strip()
    out = subprocess.run(["git", "-C", str(code_dir), "rev-parse", "HEAD"], capture_output=True,
                         text=True)
    if out.returncode:
        raise SystemExit(f"no git sha: {code_dir}/GIT_SHA missing and `git rev-parse` failed")
    return out.stdout.strip()


# --------------------------------------------------------------------------------------------
# rows


def read_rows(path) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if rows and tuple(rows[0]) != arms.HEADER:
        raise SystemExit(f"{path}: header {tuple(rows[0])} != {arms.HEADER}")
    return rows


def row_at(path, index: int) -> dict:
    rows = read_rows(path)
    if not 0 <= index < len(rows):
        raise SystemExit(f"{path}: no row at index {index} ({len(rows)} rows)")
    return rows[index]


# --------------------------------------------------------------------------------------------
# smoke (chunk C8; everything under $CF/smoke/C8/ except the genome-wide base rebuild)

#: one row per ChIP arm kind, for the chr21 structure smoke.
SMOKE_LEVELS = (("base", "base"), ("depth", "15M"), ("abproxy", "f0.5"), ("ratio", "k0.5"),
                ("ctlid", "other"), ("ctlid", "none"), ("ctldepth", "q0.5"), ("extsize", "k0.5"))
SMOKE_TRACK = "C19M16"


def smoke_rows(track=SMOKE_TRACK) -> list:
    want = set(SMOKE_LEVELS)
    return [r for r in arms.rows("bam", dnase="none", ratio="yes", tracks=[track])
            if (r["arm"], r["level"]) in want]


def _tsv_row(r: dict) -> dict:
    """An `arms.rows` dict as it appears in a rows TSV (knob_value as JSON text)."""
    return {k: (json.dumps(v) if k == "knob_value" else str(v)) for k, v in r.items()}


def make_chr21_tas(rows, cf, ta_dir, out_dir, tmp=None) -> list:
    """chr21-only copies, at the same basenames, of every tagAlign those rows read.

    The `ctldepth` inputs are NOT copied from `$CF/ta/ctldepth`: they are re-thinned here from the
    chr21 control copy with the pipeline's own subsampler, so the smoke does not depend on which
    N those files were last built at.
    """
    full, thinned = set(), {}
    for r in rows:
        p = plan(_tsv_row(r), cf, ta_dir=ta_dir, tmp=out_dir, products=out_dir, work=out_dir,
                 ratio=1.0)
        full.add(treatment_tagalign(r["track"], ta_dir))
        if p["control"] and "/ctldepth/" in p["control"]:
            acc, n = p["control_accession"], json.loads(_tsv_row(r)["knob_value"])
            full.add(control_tagalign(acc, ta_dir))
            thinned[f"{out_dir}/ctldepth/{acc}__{r['level']}"] = (acc, n)
        elif p["control"]:
            full.add(p["control"])

    made = []
    for src in sorted(full):
        dst = f"{out_dir}/{os.path.relpath(src, ta_dir)}"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if not Path(dst).is_file():
            sh(f"zcat {shlex.quote(src)} | awk '$1==\"chr21\"' | gzip -nc > {shlex.quote(dst)}")
        made.append(dst)

    tmp = str(tmp or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir())
    Path(f"{tmp}/work").mkdir(parents=True, exist_ok=True)
    for d, (acc, n) in sorted(thinned.items()):
        ctl21 = control_tagalign(acc, out_dir)
        want = subsample_output(ctl21, n, d)
        if not Path(want).is_file():
            Path(d).mkdir(parents=True, exist_ok=True)
            sh(apptainer(f"python3 $(which {SUBSAMPLE_SCRIPT}) {ctl21} --subsample {n} "
                         f"--out-dir {d}", SIF["chip"], tmp, f"{tmp}/work"))
        made.append(want)
    return made


def structure_problems(product: Path, row: dict, sizes: dict) -> list:
    """`records.product_problems` plus the array shape, dtype and range the store needs."""
    import numpy as np
    import bin25
    probs = records.product_problems(product, row)
    for name, dtype in (("counts25.npz", np.uint32), ("pval25.npz", np.float32),
                        ("control_counts25.npz", np.uint32)):
        f = product / name
        if not f.is_file():
            continue
        a = bin25.read_npz(f)
        if tuple(a) != MAIN_CHROMS:
            probs.append(f"{name}: chroms {tuple(a)} != {MAIN_CHROMS}")
        for c, v in a.items():
            if v.dtype != dtype:
                probs.append(f"{name}:{c}: dtype {v.dtype} != {dtype}")
            if v.shape[0] != sizes[c] // 25:
                probs.append(f"{name}:{c}: {v.shape[0]} bins != {sizes[c] // 25}")
            if not np.isfinite(v).all():
                probs.append(f"{name}:{c}: non-finite values")
            if v.min() < 0:
                probs.append(f"{name}:{c}: negative values")
    return probs


def smoke_ab(cf=CF, eic=EIC, chrsz=CHRSZ, tmp=None) -> dict:
    """(a) one row of every ChIP arm kind on chr21 copies; (b) the ratio arm at k=1 exactly."""
    import numpy as np
    import bin25

    smoke = Path(f"{cf}/smoke/C8")
    ta = f"{smoke}/ta"
    tmp = str(tmp or os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir())
    rows = smoke_rows()
    make_chr21_tas(rows, cf, f"{cf}/ta", ta, tmp=f"{tmp}/mk")
    sizes = bin25.load_chrsz(chrsz)

    # (a) structure
    detail = {}
    for i, r in enumerate(rows):
        tr = _tsv_row(r)
        p = run(tr, cf, eic, tmp=f"{tmp}/a{i}", ta_dir=ta, work=f"{smoke}/work_chr21",
                products=f"{smoke}/products_chr21", chrsz=chrsz, pipeline_pval=False)
        # the ratio arm's knob_value is counted from the tagAligns it was given, so on chr21 copies
        # it is legitimately not the genome-wide rows TSV number; every other column is compared.
        tr["knob_value"] = json.dumps(p["knob_value"])
        probs = structure_problems(Path(f"{smoke}/products_chr21/{r['pid']}"), tr, sizes)
        detail[r["pid"]] = probs
    part_a = {"pass": all(not v for v in detail.values()), "n_rows": len(rows),
              "problems": {k: v for k, v in detail.items() if v}}
    (smoke / "part_a.json").write_text(json.dumps(part_a, indent=1) + "\n")

    # (b) --ratio at exactly the quotient MACS2 computes for these inputs: patched == unpatched.
    base = smoke_rows()[0]
    t = arms.TRACKS[SMOKE_TRACK]
    T = treatment_tagalign(SMOKE_TRACK, ta)
    C = control_tagalign(t["ctl_acc"], ta)
    n_t, n_c = _lines(T), _lines(C)
    row_k1 = dict(base, arm="ratio", level="k1", knob="macs2_ratio", knob_value=None,
                  pid=arms.pid(SMOKE_TRACK, "ratio", "k1"),
                  biosample=arms.biosample(t["cell"], "ratio", "k1"))
    # "exact" is the raw quotient; "rounded" is that quotient through arms.ratio_value's `:.10g`,
    # which is the value the real k0.5 / k2 arms carry. Both are compared to the unpatched run, so
    # the smoke says whether the rounding alone can move the signal.
    variants = {"exact": ratio_for("k1", T, C, sig=None), "rounded": ratio_for("k1", T, C),
                "plain": None}
    out = {}
    for name, r in variants.items():
        tsvrow = _tsv_row(base if name == "plain" else row_k1)
        p = plan(tsvrow, cf, eic, tmp=f"{tmp}/b_{name}", ta_dir=ta, work=f"{smoke}/ratio_k1/{name}",
                 products=f"{smoke}/ratio_k1/{name}", chrsz=chrsz, pipeline_pval=False, ratio=r)
        Path(f"{p['tmp']}/work").mkdir(parents=True, exist_ok=True)
        for cmd in p["commands"]:
            sh(cmd, cwd=f"{p['tmp']}/work")
        out[name] = p["kept_bigwig"]

    binned = {k: bin25.pval_from_bigwig(v, sizes, f"{tmp}/bdg") for k, v in out.items()}

    def against_plain(name):
        d = max(float(np.max(np.abs(binned[name][c].astype(np.float64)
                                    - binned["plain"][c].astype(np.float64)))) for c in sizes)
        same = records.md5_file(out[name]) == records.md5_file(out["plain"])
        return d, same

    diff, same_md5 = against_plain("exact")
    diff10, same10 = against_plain("rounded")
    part_b = {
        "bit_identical": bool(diff == 0.0 and same_md5), "max_abs_diff": diff,
        "bigwig_md5_equal": same_md5,
        "bit_identical_rounded": bool(diff10 == 0.0 and same10), "max_abs_diff_rounded": diff10,
        "ratio": variants["exact"], "ratio_repr": repr(variants["exact"]),
        "ratio_rounded": variants["rounded"],
        "formula": "k * lines(treatment tagAlign) / lines(control tagAlign), k = 1",
        "n_lines_treatment": n_t, "n_lines_control": n_c,
        "treatment": T, "control": C,
        "note": ("k=1 must be the quotient MACS2 itself computes for THESE inputs "
                 "(treat_total/control_total, --keep-dup all, no --scale-to), so it is counted "
                 "from the chr21 copies fed to MACS2. The plan's 30000000/58073570 is not that "
                 "quotient for any input here: 58073570 was the control BAM's samstats total, "
                 "and its tagAlign has 107039349 lines ($CF/ta/LINES.tsv)."),
    }
    (smoke / "ratio_k1.json").write_text(json.dumps(part_b, indent=1) + "\n")
    return {"a": part_a, "b": part_b}


def smoke_c(cf=CF, eic=EIC, chrsz=CHRSZ, tmp=None) -> dict:
    """(c) C19M16 base genome-wide: the rebuilt bigwig against the pipeline's own, on chr21."""
    import numpy as np
    import bin25

    smoke = Path(f"{cf}/smoke/C8")
    row = _tsv_row(smoke_rows()[0])
    # part (a) runs the same pid on chr21, so the genome-wide product needs its own root; the
    # rebuild alone goes to $CF/bamarms/<pid>/, where the `base_rebuild` check reads it.
    p = run(row, cf, eic, tmp=tmp, ta_dir=f"{cf}/ta", work=f"{cf}/bamarms",
            products=f"{smoke}/products_gw", chrsz=chrsz, pipeline_pval=True)
    pipeline = bin25.read_npz(f"{p['product']}/pval25.npz")["chr21"]
    rebuild = bin25.read_npz(p["rebuild_npz"])["chr21"]
    same = pipeline.shape == rebuild.shape
    diff = (float(np.max(np.abs(pipeline.astype(np.float64) - rebuild.astype(np.float64))))
            if same else float("inf"))
    part_c = {"pid": p["pid"], "n_bins": int(rebuild.shape[0]),
              "n_bins_pipeline": int(pipeline.shape[0]), "same_n_bins": bool(same),
              "max_abs_diff": diff, "n_nan_rebuild": int(np.isnan(rebuild).sum()),
              "pass": bool(same and diff <= 1e-6),
              "rebuild_npz": p["rebuild_npz"], "pipeline_bigwig": p["pipeline_bigwig"]}
    (smoke / "base_rebuild_C19M16.json").write_text(json.dumps(part_c, indent=1) + "\n")
    return part_c


def smoke_ok(cf=CF) -> dict:
    """Collect the three parts into SMOKE_OK.json. stdlib only, so it runs on the login node."""
    smoke = Path(f"{cf}/smoke/C8")
    parts = {}
    for key, name in (("a", "part_a.json"), ("b", "ratio_k1.json"), ("c",
                                                                    "base_rebuild_C19M16.json")):
        f = smoke / name
        parts[key] = json.loads(f.read_text()) if f.is_file() else {"missing": str(f)}
    ok = {"pass": bool(parts["a"].get("pass") and parts["b"].get("bit_identical")
                       and parts["c"].get("pass")), "parts": parts}
    (smoke / "SMOKE_OK.json").write_text(json.dumps(ok, indent=1) + "\n")
    print(json.dumps({"pass": ok["pass"]}))
    return ok


# --------------------------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--rows", required=True)
        p.add_argument("--index", required=True, type=int)
        p.add_argument("--cf", default=CF)
        p.add_argument("--eic", default=EIC)
        p.add_argument("--chrsz", default=CHRSZ)
        p.add_argument("--ta-dir", default=None)
        p.add_argument("--work", default=None)
        p.add_argument("--products", default=None)
        p.add_argument("--tmp", default=None)
    ps = sub.add_parser("smoke")
    ps.add_argument("--part", choices=("ab", "c", "ok"), required=True)
    ps.add_argument("--cf", default=CF)
    ps.add_argument("--eic", default=EIC)
    ps.add_argument("--chrsz", default=CHRSZ)
    ps.add_argument("--tmp", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "smoke":
        if args.part == "ab":
            smoke_ab(args.cf, args.eic, args.chrsz, args.tmp)
        elif args.part == "c":
            smoke_c(args.cf, args.eic, args.chrsz, args.tmp)
        else:
            return 0 if smoke_ok(args.cf)["pass"] else 1
        return 0

    row = row_at(args.rows, args.index)
    kw = dict(tmp=args.tmp, ta_dir=args.ta_dir, work=args.work, products=args.products,
              chrsz=args.chrsz)
    if args.cmd == "plan":
        for c in plan_commands(row, args.cf, args.eic, **kw):
            print(c)
        return 0
    run(row, args.cf, args.eic, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
