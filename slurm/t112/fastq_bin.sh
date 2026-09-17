#!/bin/bash
# t112, chunks C12 and C14: one FASTQ-route product per array task, from that pid's harvested files.
#
# WHAT ONE TASK DOES. It reads its row from a rows TSV by index, reads that pid's
# `$CF/fastqarms/harvest/<pid>/HARVEST.tsv` for the keepers `fastq_arms.py harvest` copied out of
# the Cromwell tree, bins them at 25 bp with `tools/t112/bin25.py` (counts of the treatment
# tagAlign, counts of the control tagAlign, mean -log10 p of the pipeline's own pval bigwig), writes
# `covariates.json` and `provenance.json` with `tools/t112/records.py`, publishes the product, and
# validates it. The arrays are built in $SLURM_TMPDIR and copied out only once they exist and are
# non-empty; the two records are published LAST, so a product dir is never valid before it is
# complete (the C8 convention).
#
# THE HARVEST IS THE INPUT, NOT THE CROMWELL TREE. Every path this task reads comes from
# HARVEST.tsv's `dest` column, and `provenance.caper.metadata_json` names the harvested copy of
# metadata.json, not the one inside `cromwell/<pid>/`: under the PI's early-D3 ruling that tree is
# deleted (`fastq_arms.py cleanup`) as soon as the product validates, and provenance must keep
# pointing at a file that still exists.
#
# DNase (C14) IS THE SAME SCRIPT. An atac run has no control and, in dnase mode, no `xcor` call at
# all, so HARVEST.tsv has no `ctl_ta` and no `xcor_qc`: no `control_counts25.npz` is written,
# `covariates.control` is null, and fraglen is the run's own `atac.smooth_win` (150) instead of the
# xcor estimate. read_length and run_type come from `arms.py` (76, single-ended) exactly as for ChIP.
#
# NO --export. Alliance arrays lose a comma-valued --export variable, so the kit root, the rows TSV
# and $CF are POSITIONAL and the task index is $SLURM_ARRAY_TASK_ID. The kit root is an argument
# because SLURM runs its own spool copy of this file: deriving it from ${BASH_SOURCE[0]} gave every
# task `/var/spool/tools/t112/...: No such file` (C9, 2026-09-17).
#
# THE VENV. `bin25.py pval` imports `tools/dnase_macs2_pval.py`, which imports h5py and pandas at
# module level, and reads the bigwig with pyBigWig. scipy-stack supplies numpy and pandas; pyBigWig
# and h5py come from the Alliance wheelhouse with --no-index, because compute nodes have no
# internet. The venv is built in $SLURM_TMPDIR: never on the login node, which has no numpy. Nothing
# here runs inside a SIF, so apptainer is not loaded.
#
# DRY_RUN=1 prints the command list for one row and runs nothing (no venv, no writes). That is what
# `tests/test_t112_fastq_bin.py` exercises on the laptop.
#
# Usage, from the Nibi login node, after snapshotting the repo to $CF/code/<chunk>:
#   KIT=$CF/code/C12
#   mkdir -p $CF/logs/fastq_bin           # THE LAUNCHER MUST DO THIS. SLURM opens --output before
#                                         # the script body runs, so a job cannot create its own log
#                                         # directory: without it every task dies with no log.
#   R=$CF/fastqarms/rows_chip.tsv         # the same TSV C7 submitted from
#   sbatch --test-only --array=0-35 $KIT/slurm/t112/fastq_bin.sh $KIT $R
#   sbatch --parsable  --array=0-35 $KIT/slurm/t112/fastq_bin.sh $KIT $R
#SBATCH --account=def-maxwl
#SBATCH --job-name=t112_fastq_bin
#SBATCH --output=/scratch/mforooz/t112_cf/logs/fastq_bin/%x_%A_%a.out
#SBATCH --error=/scratch/mforooz/t112_cf/logs/fastq_bin/%x_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail

USAGE="usage: fastq_bin.sh <kit_dir> <rows.tsv> [cf_dir]   (DRY_RUN=1 prints the commands)"
KIT="${1:?$USAGE — kit_dir is the code snapshot, e.g. \$CF/code/C12}"
ROWS="${2:?$USAGE}"
CF="${3:-/scratch/mforooz/t112_cf}"
CHRSZ="${CHRSZ:-/scratch/mforooz/EIC_REPRO/003/refcache/c52f52c7bfa357f55a39b1de7e4d0b0c/GRCh38_EBV.chrom.sizes.tsv}"
DRY_RUN="${DRY_RUN:-0}"
PY_MODULES="StdEnv/2023 python/3.11 scipy-stack"

# Fail here, not 10 s into the venv build, if the kit is wrong or the old argument order was used.
for f in tools/t112/bin25.py tools/t112/records.py tools/t112/arms.py; do
    [ -f "$KIT/$f" ] || { echo "kit_dir '$KIT' has no $f. $USAGE" >&2; exit 2; }
done
KIT=$(cd "$KIT" && pwd)

: "${SLURM_ARRAY_TASK_ID:?array task only}"

# The row, by the pinned index rule: +2 skips the header and turns 0-based into sed's 1-based.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$ROWS")
[ -n "$LINE" ] || { echo "no row at index $SLURM_ARRAY_TASK_ID in $ROWS" >&2; exit 2; }
PID=$(cut -f1 <<< "$LINE")
ROUTE=$(cut -f8 <<< "$LINE")
PIPELINE=$(cut -f11 <<< "$LINE")
[ "$ROUTE" = "fastq" ] || { echo "$PID: route is '$ROUTE', not fastq — wrong rows TSV" >&2; exit 2; }

HARVEST="$CF/fastqarms/harvest/$PID"
HTSV="$HARVEST/HARVEST.tsv"
STATE="$CF/fastqarms/state/$PID.json"
WORK="$CF/fastqarms/bin/$PID"
PRODUCT="$CF/products/$PID"
STAGE="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/t112_fastq_bin/$PID"
CMDLOG="$STAGE/commands.txt"

for f in "$HTSV" "$STATE" "$CHRSZ"; do
    [ -f "$f" ] || { echo "$PID: missing $f" >&2; exit 2; }
done

echo "=== $PID  idx=$SLURM_ARRAY_TASK_ID  job ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-none}}_${SLURM_ARRAY_TASK_ID}  host=$(hostname)  $(date -u)"

# --- the keepers, by role, from HARVEST.tsv (role src dest bytes md5) ----------------------------
dest_of() { awk -F'\t' -v want="$1" 'NR > 1 && $1 == want {print $3}' "$HTSV"; }
TREAT_TA=$(dest_of treat_ta)
CTL_TA=$(dest_of ctl_ta)
PVAL_BW=$(dest_of pval_bigwig)
XCOR_QC=$(dest_of xcor_qc)

case "$PIPELINE" in
    chip)
        NEED="treat_ta ctl_ta pval_bigwig xcor_qc metadata" ;;
    atac)
        NEED="treat_ta pval_bigwig metadata"
        if [ -n "$CTL_TA$XCOR_QC" ]; then
            echo "$PID: an atac run has no control and no xcor call, but HARVEST.tsv lists them" >&2
            exit 2
        fi ;;
    *)
        echo "$PID: unknown pipeline '$PIPELINE'" >&2; exit 2 ;;
esac
for r in $NEED; do
    [ -n "$(dest_of "$r")" ] || { echo "$PID: HARVEST.tsv has no '$r' role — harvest first" >&2; exit 2; }
done

# --- every external command goes through run(): printed, recorded, then executed -----------------
run() {
    local line
    line=$(printf '%q ' "$@")
    echo "+ ${line% }"
    if [ "$DRY_RUN" != "1" ]; then
        printf '%s\n' "${line% }" >> "$CMDLOG"
        "$@"
    fi
}

if [ "$DRY_RUN" != "1" ]; then
    mkdir -p "$WORK" "$STAGE"
    df -h "$STAGE" | tail -1
    head -1 "$ROWS" > "$WORK/row.tsv"
    printf '%s\n' "$LINE" >> "$WORK/row.tsv"
    : > "$CMDLOG"

    set +u; module load $PY_MODULES; set -u
    virtualenv --no-download "$SLURM_TMPDIR/venv"
    source "$SLURM_TMPDIR/venv/bin/activate"
    pip install --no-index --upgrade pip
    pip install --no-index pyBigWig h5py
    python3 -c "import numpy, pandas, h5py, pyBigWig; print('venv ok', numpy.__version__, pyBigWig.__version__)"
    # PREPEND, never replace: `module load scipy-stack` puts numpy and pandas on PYTHONPATH, and
    # overwriting it makes `import numpy` fail inside the job (C8, 2026-09-17).
    export PYTHONPATH="$KIT/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# --- bin ----------------------------------------------------------------------------------------
BIN25="$KIT/tools/t112/bin25.py"
run python3 "$BIN25" counts --ta "$TREAT_TA" --chrsz "$CHRSZ" \
    --out "$STAGE/counts25.npz" --json "$STAGE/counts25_stats.json"
if [ -n "$CTL_TA" ]; then
    run python3 "$BIN25" counts --ta "$CTL_TA" --chrsz "$CHRSZ" \
        --out "$STAGE/control_counts25.npz" --json "$STAGE/control_counts25_stats.json"
fi
run python3 "$BIN25" pval --bigwig "$PVAL_BW" --chrsz "$CHRSZ" --tmpdir "$STAGE/bdg" \
    --out "$STAGE/pval25.npz" --json "$STAGE/pval25_stats.json"

# --- covariates.json and provenance.json --------------------------------------------------------
export T112_KIT="$KIT" T112_PID="$PID" T112_ROW="$LINE" T112_STAGE="$STAGE" \
       T112_PRODUCT="$PRODUCT" T112_HARVEST_TSV="$HTSV" T112_STATE="$STATE" T112_CMDLOG="$CMDLOG"
if [ -f "$KIT/GIT_SHA" ]; then T112_GIT_SHA=$(cat "$KIT/GIT_SHA"); export T112_GIT_SHA; fi
echo "+ python3 - # covariates.json + provenance.json via $KIT/tools/t112/records.py"
if [ "$DRY_RUN" != "1" ]; then
    printf '%s\n' "python3 - # covariates.json + provenance.json via records.py ($PID)" >> "$CMDLOG"
    python3 - <<'PY'
# >>> records step (extracted verbatim by tests/test_t112_fastq_bin.py) >>>
"""Write one FASTQ-route product's covariates.json and provenance.json into the stage dir.

Everything is read from files the job already has in hand: the rows TSV line, HARVEST.tsv, the
pid's state file, the harvested metadata.json and the stats JSONs bin25 just wrote. Paths named in
`outputs` are the FINAL product paths but the md5s are of the staged bytes, since nothing is
published until every one of them exists (`bam_arm.py::run`).
"""
import json
import os
import sys
from pathlib import Path

KIT = os.environ["T112_KIT"]
sys.path.insert(0, os.path.join(KIT, "tools", "t112"))
import arms      # noqa: E402
import records   # noqa: E402

pid = os.environ["T112_PID"]
stage = Path(os.environ["T112_STAGE"])
product = Path(os.environ["T112_PRODUCT"])
row = dict(zip(arms.HEADER, os.environ["T112_ROW"].split("\t")))
state = json.loads(Path(os.environ["T112_STATE"]).read_text())

lines = Path(os.environ["T112_HARVEST_TSV"]).read_text().rstrip("\n").split("\n")
cols = lines[0].split("\t")
harvested = {}
for ln in lines[1:]:
    rec = dict(zip(cols, ln.split("\t")))
    harvested[rec["role"]] = rec
# a keeper that lost bytes since the harvest verified it must never reach a product
for role in ("treat_ta", "ctl_ta", "pval_bigwig", "xcor_qc", "metadata"):
    rec = harvested.get(role)
    if rec and os.path.getsize(rec["dest"]) != int(rec["bytes"]):
        raise SystemExit(f"{pid}: {role} {rec['dest']} is {os.path.getsize(rec['dest'])} B, "
                         f"HARVEST.tsv recorded {rec['bytes']} B")

track = arms.TRACKS[row["track"]]
knob_value = json.loads(row["knob_value"])
repo = records.PIPELINE_REPO[row["pipeline"]]
release, sif_md5 = records.PIPELINES[repo]

# depth is the line count of the treatment tagAlign, as bin25 counted it.
depth = json.loads((stage / "counts25_stats.json").read_text())["n_lines"]
control = None
if "ctl_ta" in harvested:
    reads = json.loads((stage / "control_counts25_stats.json").read_text())["n_lines"]
    control = {"accession": track["ctl_acc"], "source": "matched", "reads": reads}

# fraglen: field 3 of the xcor .cc.qc (the estimated fragment length the pipeline itself used),
# or — DNase, which has no xcor call — the run's own smooth_win from arms.py.
if "xcor_qc" in harvested:
    fields = Path(harvested["xcor_qc"]["dest"]).read_text().split("\n")[0].split("\t")
    fraglen = int(str(fields[2]).split(",")[0])
else:
    fraglen = int(track["fraglen"])

records.write_covariates(
    stage / "covariates.json",
    pid=pid, biosample=row["biosample"], track=row["track"], cell=row["cell"],
    assay=row["assay"], arm=row["arm"], level=row["level"], knob=row["knob"],
    knob_value=knob_value, depth=depth,
    read_length=int(knob_value) if row["arm"] == "crop" else int(track["read_length"]),
    run_type="paired-ended" if row["arm"] == "pe" else track["run_type"],
    fraglen=fraglen, control=control)

input_json = state["input_json"]
sif = json.loads(Path(input_json).read_text())[f"{row['pipeline']}.singularity"]
meta = json.loads(Path(harvested["metadata"]["dest"]).read_text())

inputs = [{"path": input_json, "md5": records.md5_file(input_json)}]
if state.get("genome_tsv"):
    inputs.append({"path": state["genome_tsv"], "md5": state["genome_tsv_md5"]})
inputs += [{"path": harvested[r]["dest"], "md5": harvested[r]["md5"]}
           for r in ("treat_ta", "ctl_ta", "pval_bigwig", "xcor_qc", "metadata") if r in harvested]

names = ["counts25.npz"] + (["control_counts25.npz"] if control else []) + ["pval25.npz"]
outputs = [{"path": str(product / n), "md5": records.md5_file(stage / n)} for n in names]

job_ids = [str(state["leader_job_id"])] if state.get("leader_job_id") else []
array, task, job = (os.environ.get(k) for k in
                    ("SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_ID"))
if array and task:
    job_ids.append(f"{array}_{task}")
elif job:
    job_ids.append(job)

git_sha = os.environ.get("T112_GIT_SHA")
if not git_sha and (Path(KIT) / "GIT_SHA").is_file():
    git_sha = (Path(KIT) / "GIT_SHA").read_text().strip()

records.write_provenance(
    stage / "provenance.json", pid=pid, route="fastq",
    pipeline={"repo": repo, "release": release, "sif": sif, "sif_md5": sif_md5,
              "sif_sha256": None},
    commands=Path(os.environ["T112_CMDLOG"]).read_text().rstrip("\n").split("\n"),
    inputs=inputs, outputs=outputs, subsample_seed=[], patched_script=None,
    caper={"input_json": input_json,
           "workflow_id": state.get("workflow_id") or meta["id"],
           "metadata_json": harvested["metadata"]["dest"]},
    slurm_job_ids=job_ids,
    code={"snapshot_dir": KIT, "git_sha": git_sha})
print(f"[t112] {pid} records written -> {stage}")
# <<< records step <<<
PY
fi

# --- publish: the records LAST, so the dir is never valid before it is complete -----------------
PUBLISH="counts25.npz"
if [ -n "$CTL_TA" ]; then
    PUBLISH="$PUBLISH control_counts25.npz"
fi
PUBLISH="$PUBLISH pval25.npz covariates.json provenance.json"
for f in $PUBLISH; do
    if [ "$DRY_RUN" = "1" ]; then
        echo "+ cp $STAGE/$f $PRODUCT/$f   # non-empty check, then md5-verified"
        continue
    fi
    [ -s "$STAGE/$f" ] || { echo "$PID: refusing to publish: $STAGE/$f is missing or empty" >&2; exit 3; }
    mkdir -p "$PRODUCT"
    cp -p "$STAGE/$f" "$PRODUCT/$f.part"
    mv "$PRODUCT/$f.part" "$PRODUCT/$f"
    a=$(md5sum < "$STAGE/$f" | cut -d' ' -f1)
    b=$(md5sum < "$PRODUCT/$f" | cut -d' ' -f1)
    [ "$a" = "$b" ] || { echo "$PID: copy md5 mismatch for $f: $a != $b" >&2; exit 3; }
done

# --- validate this one product ------------------------------------------------------------------
run python3 "$KIT/tools/t112/records.py" validate --products "$CF/products" \
    --rows "$WORK/row.tsv" --expect 1

if [ "$DRY_RUN" != "1" ]; then
    cp -p "$STAGE"/*_stats.json "$CMDLOG" "$WORK/"
    ls -l "$PRODUCT"
fi
echo "=== $PID done $(date -u)"
