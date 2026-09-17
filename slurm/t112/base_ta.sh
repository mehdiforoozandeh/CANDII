#!/bin/bash
# t112, chunk C5: rebuild the exact tagAligns the base pipeline runs fed MACS2, with the pipeline's own
# bam2ta, inside the pipeline's own SIF; then the arm-9 (ctldepth) thinned controls with the
# pipeline's own control subsampler.
#
# WHY A REBUILD. The Cromwell trees of the 7 base runs are gone (checked 2026-09-17: no
# *.30M.tagAlign.gz, *.50M.tagAlign.gz or control tagAlign anywhere under /scratch/mforooz or
# $EIC/cromwell_out), so nothing to compare an md5 against. What survives is each run's
# metadata.json, whose calls["chip.bam2ta"] / ["chip.bam2ta_ctl"] / ["atac.bam2ta"] commandLine
# is copied here VERBATIM with one edit: the Cromwell input path becomes the bare BAM basename, and the
# command runs with cwd = a work dir holding a symlink of that name. The output prefix is derived from
# the basename, so names match the pipeline's (<ACC>.merged.srt.nodup.30M.tagAlign.gz).
#
# WHY IN THE SIF, AND WHY --cleanenv. The subsampler is
#   shuf -n N --random-source=<(openssl enc -aes-256-ctr -pass pass:$(zcat -f ta | wc -c) -nosalt </dev/zero)
# so the draw depends on the openssl and shuf builds as well as the input bytes; only the SIF has the
# ones the base runs used. Caper ran every task as `singularity exec --cleanenv`; without --cleanenv
# the host's exported `which` shell function leaks in, and `python3 $(which encode_task_bam2ta.py)`
# breaks (seen 2026-09-17).
#
# WHY ctldepth USES encode_task_subsample_ctl.py ON THE FULL CONTROL tagAlign. chip.ctl_subsample_reads
# is handed to bam2ta_ctl as --subsample, which runs subsample_ta_se(ta, N, non_mito=False, ...) on
# the full control tagAlign; encode_task_subsample_ctl.py runs the same function with the same
# arguments on a tagAlign of the same name and bytes. The command form is chip.wdl v2.2.2
# `task subsample_ctl` (cwd = output dir, no --out-dir).
#
# RE-RUN MODE. A task whose output tagAlign is already published is not rebuilt: the task measures
# the published file, refuses to go on unless its line count is the expected one, and skips bam2ta.
# So re-running a control task rebuilds only its arm-9 thinned controls, and cannot disturb the
# tagAligns MACS2 will read. The measurement is also the assertion that the arm-9 N were computed
# from the file being thinned: N = round_half_up(q * arms.CONTROLS[acc]["reads"]) and a control
# row's expected_lines is that same constant, so one comparison binds them.
#
# Usage, from the Nibi login node, after snapshotting the repo to $CF/code/C5:
#   bash   $CF/code/C5/slurm/t112/base_ta.sh --make-tasks          # tasks.tsv + cmd/*.sh from metadata
#   A=$(sbatch --parsable --array=0-9 $CF/code/C5/slurm/t112/base_ta.sh)
#   sbatch --dependency=afterany:$A --time=0:30:00 --cpus-per-task=1 --mem=2G \
#          $CF/code/C5/slurm/t112/base_ta.sh --collect                   # LINES.tsv, LINES_CHECK.tsv
# Re-run of the arm-9 thinning alone (rows 7-9 are the controls; 64G because the first pass peaked
# at 31.5 GiB of 32G on the 160M-line control):
#   A=$(sbatch --parsable --array=7-9 --mem=64G $CF/code/C5/slurm/t112/base_ta.sh)
#SBATCH --account=def-maxwl
#SBATCH --job-name=t112_base_ta
#SBATCH --output=/scratch/mforooz/t112_cf/logs/base_ta/%x_%A_%a.out
#SBATCH --error=/scratch/mforooz/t112_cf/logs/base_ta/%x_%A_%a.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

CF=/scratch/mforooz/t112_cf
CODE=$CF/code/C5
TA=$CF/ta
EIC=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline
TASKS=$CODE/tasks.tsv
CHIP_SIF_MD5=f6e408f7e77bafde4556883f33552191
ATAC_SIF_MD5=04d9fa482d67cf633845653750f58cef

# ---------------------------------------------------------------------------------------------------
if [ "${1:-}" = "--make-tasks" ]; then
  mkdir -p "$CODE/cmd"
  python3 - "$CODE" "$EIC" <<'PY'
import json, re, sys
from pathlib import Path
code, eic = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(code / "tools" / "t112"))
import arms

def calls(track):
    return json.load(open(f"{eic}/results/{track}/metadata.json"))["calls"]

def one_call(track, key):
    cs = [c for c in calls(track)[key] if c.get("executionStatus") == "Done"]
    assert len(cs) == 1, (track, key, len(cs))
    return cs[0]

def localise(cmdline, want_subsample):
    """The commandLine with its one Cromwell input BAM path cut to the basename; nothing else."""
    toks = re.findall(r"\S+\.bam(?=\s)", cmdline)
    assert len(toks) == 1, toks
    link = toks[0].rsplit("/", 1)[1]
    assert cmdline.count(toks[0]) == 1
    assert re.search(rf"--subsample {want_subsample}\s", cmdline), cmdline
    assert "--disable-tn5-shift" in cmdline and "--paired-end" not in cmdline, cmdline
    return cmdline.replace(toks[0], link), link

def normal(cmdline):  # for cross-track agreement checks only
    return re.sub(r"--mem-gb \S+", "--mem-gb X", re.sub(r"\S+\.bam(?=\s)", "BAM", cmdline))

sif = {"chip": f"{eic}/sif/chip-seq-pipeline_v2.2.2.sif", "atac": f"{eic}/sif/atac-seq-pipeline_v2.2.3.sif"}
rows = []
chip_ref = None
for track, t in arms.TRACKS.items():
    pipe = "chip" if t["pipeline"] == "chip" else "atac"   # C12M02: D1 = atac (PI 2026-09-17)
    c = one_call(track, f"{pipe}.bam2ta")
    assert c["runtimeAttributes"]["singularity"] == sif[pipe], c["runtimeAttributes"]
    cmd, link = localise(c["commandLine"], t["subsample"])
    if pipe == "chip":
        chip_ref = chip_ref or normal(c["commandLine"])
        assert normal(c["commandLine"]) == chip_ref, track
    assert link.startswith(t["treat_acc"] + "."), link
    name = f"{track}__treat"
    (code / "cmd" / f"{name}.sh").write_text(cmd + "\n")
    out = link[:-len(".bam")] + (".30M" if t["subsample"] == 30000000 else ".50M") + ".tagAlign.gz"
    rows.append([name, "treat", sif[pipe], t["treat_bam"], link, out, str(t["subsample"]),
                 str(t["treat_reads"]), "-", "-"])

levels = dict(next(a for a in arms.CHIP_ARMS if a[0] == "ctldepth")[3])
for acc, ctl in arms.CONTROLS.items():
    track = ctl["bam"].split("/results/")[1].split("/")[0]
    sibling = [k for k, t in arms.TRACKS.items() if t["ctl_acc"] == acc and k != track]
    assert len(sibling) == 1, (acc, sibling)
    c = one_call(track, "chip.bam2ta_ctl")
    assert c["runtimeAttributes"]["singularity"] == sif["chip"]
    cmd, link = localise(c["commandLine"], 0)
    assert normal(c["commandLine"]) == normal(one_call(sibling[0], "chip.bam2ta_ctl")["commandLine"])
    assert link == f"{acc}.merged.srt.nodup.bam", link
    name = f"{acc}__ctl"
    (code / "cmd" / f"{name}.sh").write_text(cmd + "\n")
    depth = ",".join(f"{lv}:{arms.round_half_up(q * ctl['reads'])}" for lv, q in levels.items())
    sib_bam = ctl["bam"].replace(f"/results/{track}/", f"/results/{sibling[0]}/")
    rows.append([name, "ctl", sif["chip"], ctl["bam"], link, link[:-len(".bam")] + ".tagAlign.gz",
                 str(ctl["reads"]), str(ctl["reads"]), depth, sib_bam])

with open(code / "tasks.tsv", "w") as f:
    f.write("name\tkind\tsif\tbam\tlink\tout\texpected_lines\tbam_reads\tctldepth\tsibling_bam\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(f"wrote {len(rows)} tasks to {code / 'tasks.tsv'}")
PY
  exit 0
fi

# ---------------------------------------------------------------------------------------------------
if [ "${1:-}" = "--collect" ]; then
  # Every artifact this step produces is written before the step can exit non-zero: a failing check
  # has to leave the evidence of its own failure behind. The first pass exited 1 inside the
  # line-count python, so CTL_BAM_IDENTITY.tsv below was never reached under `set -e`.
  if ls "$TA"/ctl_identity/*.tsv >/dev/null 2>&1; then
    { printf 'control\tpinned_bam\tpinned_md5\tsibling_bam\tsibling_md5\tidentical\n'; cat "$TA"/ctl_identity/*.tsv; } \
      > "$TA/CTL_BAM_IDENTITY.tsv"
  fi

  # LINES.tsv in task order (treatment 30M/50M, control, then its ctldepth files), and the check
  # of every line count against the expected value. A missing fragment is a MISSING row, not a skip.
  BAD=0
  python3 - "$TASKS" "$TA" <<'PY' || BAD=1
import sys
from pathlib import Path
tasks, ta = Path(sys.argv[1]), Path(sys.argv[2])
hdr, *rows = [l.rstrip("\n").split("\t") for l in open(tasks)]
want = []
for r in rows:
    d = dict(zip(hdr, r))
    want.append((f"{ta}/{d['out']}", int(d["expected_lines"])))
    if d["ctldepth"] != "-":
        for lv_n in d["ctldepth"].split(","):
            lv, n = lv_n.split(":")
            want.append((f"{ta}/ctldepth/{d['link'].split('.')[0]}__{lv}", int(n)))
frag = {}
for p in sorted((ta / "lines.d").glob("*.tsv")):
    path, lines, nbytes, md5, expected = open(p).read().rstrip("\n").split("\t")
    frag[path] = (lines, nbytes, md5)
bad = 0
with open(ta / "LINES.tsv", "w") as out, open(ta / "LINES_CHECK.tsv", "w") as chk:
    out.write("path\tlines\tuncompressed_bytes\tmd5\n")
    chk.write("path\texpected\tlines\tstatus\n")
    for key, n in want:
        hit = [p for p in frag if p == key or p.startswith(key + "/")]
        if len(hit) != 1:
            chk.write(f"{key}\t{n}\t\tMISSING\n"); bad += 1; continue
        lines, nbytes, md5 = frag[hit[0]]
        out.write(f"{hit[0]}\t{lines}\t{nbytes}\t{md5}\n")
        ok = int(lines) == n
        bad += not ok
        chk.write(f"{hit[0]}\t{n}\t{lines}\t{'OK' if ok else 'MISMATCH'}\n")
print(f"{len(want)} files expected, {bad} not OK")
sys.exit(1 if bad else 0)
PY

  # Every provenance JSON back out through records.write_provenance, which validates the record and
  # writes `created_utc` in the one form records.py itself writes. The first pass wrote
  # "2026-09-17T17:36:04Z", which datetime.fromisoformat rejects on python 3.10 (the laptop env),
  # so every record failed validation there. Only the timestamp changes; the md5s already in the
  # file stand, and nothing is re-hashed.
  python3 - "$TA" "$CODE" <<'PY'
import json, sys
from pathlib import Path
ta, code = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(code / "tools" / "t112"))
import records
n = 0
for p in sorted((ta / "provenance").glob("*.json")):
    d = json.loads(p.read_text())
    t = d.get("created_utc")
    if isinstance(t, str) and t.endswith("Z"):
        d["created_utc"] = t[:-1] + "+00:00"
    records.write_provenance(p, **d)   # raises ValueError naming the offending key
    n += 1
print(f"{n} provenance JSONs validated and normalised")
PY

  [ "$BAD" = 0 ] || { echo "line-count check failed; see $TA/LINES_CHECK.tsv" >&2; exit 1; }
  exit 0
fi

# ---------------------------------------------------------------------------------------------------
: "${SLURM_ARRAY_TASK_ID:?array task only (or --make-tasks / --collect)}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$TASKS")
[ -n "$LINE" ] || { echo "no task at index $SLURM_ARRAY_TASK_ID in $TASKS" >&2; exit 2; }
IFS=$'\t' read -r NAME KIND SIF BAM LINK OUT EXPECTED BAM_READS CTLDEPTH SIB_BAM <<< "$LINE"
JOBTAG="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

set +u; module load apptainer/1.3.5; set -u
mkdir -p "$TA/provenance" "$TA/lines.d" "$TA/ctldepth" "$TA/ctl_identity"
echo "=== $NAME ($KIND) job $JOBTAG on $(hostname) $(date -u) ==="
df -h "$SLURM_TMPDIR" | tail -1

case "$SIF" in
  *chip-seq-pipeline_v2.2.2.sif) REPO=ENCODE-DCC/chip-seq-pipeline2; REL=v2.2.2; PIN=$CHIP_SIF_MD5 ;;
  *atac-seq-pipeline_v2.2.3.sif) REPO=ENCODE-DCC/atac-seq-pipeline;  REL=v2.2.3; PIN=$ATAC_SIF_MD5 ;;
  *) echo "unknown SIF $SIF" >&2; exit 3 ;;
esac
SIF_MD5=$(md5sum "$SIF" | cut -d' ' -f1)
[ "$SIF_MD5" = "$PIN" ] || { echo "SIF md5 $SIF_MD5 != pinned $PIN" >&2; exit 3; }
SIF_SHA256=$(sha256sum "$SIF" | cut -d' ' -f1)

W=$SLURM_TMPDIR/work
mkdir -p "$W"
ln -s "$BAM" "$W/$LINK"
cp "$CODE/cmd/$NAME.sh" "$W/cmd.sh"
APPT="apptainer exec --cleanenv -B /scratch -B /project -B $SLURM_TMPDIR"

# stats <file> -> "lines<TAB>uncompressed_bytes<TAB>md5"
stats() {
  local lc md5
  lc=$(zcat -f "$1" | wc -lc | awk '{print $1"\t"$2}')
  md5=$(md5sum "$1" | cut -d' ' -f1)
  printf '%s\t%s\n' "$lc" "$md5"
}

# publish <tmp file> <final path> <expected lines> -> copies, verifies md5, writes the lines fragment
publish() {
  local src=$1 dst=$2 exp=$3 st md5
  st=$(stats "$src"); md5=$(cut -f3 <<< "$st")
  cp "$src" "$dst.partial"
  [ "$(md5sum "$dst.partial" | cut -d' ' -f1)" = "$md5" ] || { echo "copy md5 mismatch $dst" >&2; exit 4; }
  mv -f "$dst.partial" "$dst"
  printf '%s\t%s\t%s\n' "$dst" "$st" "$exp" > "$TA/lines.d/$(basename "$dst").tsv"
  echo "published $dst $st expected_lines=$exp"
}

# prov <out json> <input path> <input md5> <output path> <output md5> <seed file|-> <seed bytes|-> <cmd>...
prov() {
  python3 - "$@" <<'PY'
import os, sys
from pathlib import Path
code = os.environ["CODE"]
sys.path.insert(0, os.path.join(code, "tools", "t112"))
import records   # stdlib only; it validates the record and owns the created_utc format
out, ipath, imd5, opath, omd5, sfile, sbytes, *cmds = sys.argv[1:]
records.write_provenance(
    out,
    pid=os.path.basename(opath),  # a shared input tagAlign, not a product: named by its file
    route="bam",
    pipeline={"repo": os.environ["REPO"], "release": os.environ["REL"], "sif": os.environ["SIF"],
              "sif_md5": os.environ["SIF_MD5"], "sif_sha256": os.environ["SIF_SHA256"]},
    commands=cmds,
    inputs=[{"path": ipath, "md5": imd5}],
    outputs=[{"path": opath, "md5": omd5}],
    subsample_seed=[] if sfile == "-" else [{"file": sfile, "uncompressed_bytes": int(sbytes)}],
    patched_script=None,
    caper=None,
    slurm_job_ids=[os.environ["JOBTAG"]],
    code={"snapshot_dir": code,
          "git_sha": (Path(code) / "GIT_SHA").read_text().strip()},
)
PY
}
export REPO REL SIF SIF_MD5 SIF_SHA256 JOBTAG CODE

echo "--- md5 of input BAM"
BAM_MD5=$(md5sum "$BAM" | cut -d' ' -f1)
echo "$BAM_MD5  $BAM"

RUN_CMD="$APPT --pwd $W $SIF /bin/bash cmd.sh"
echo "--- $RUN_CMD"; cat "$W/cmd.sh"

# Re-run mode (see the header). A published output is measured, never rebuilt, and a line count
# other than the expected one stops the task before anything downstream is derived from it.
REUSE=no
if [ -s "$TA/$OUT" ]; then
  echo "--- $TA/$OUT is already published: measuring it instead of running bam2ta"
  ST=$(stats "$TA/$OUT")
  HAVE=$(cut -f1 <<< "$ST")
  [ "$HAVE" = "$EXPECTED" ] \
    || { echo "PUBLISHED_LINES_MISMATCH $TA/$OUT has $HAVE lines, expected $EXPECTED" >&2; exit 7; }
  printf '%s\t%s\t%s\n' "$TA/$OUT" "$ST" "$EXPECTED" > "$TA/lines.d/$(basename "$OUT").tsv"
  echo "reuse $TA/$OUT $ST expected_lines=$EXPECTED"
  REUSE=yes
fi

if [ "$KIND" = "treat" ]; then
  if [ "$REUSE" = yes ]; then
    echo "re-run: $NAME is already published and correct; nothing to rebuild"
    echo "=== $NAME done $(date -u) ==="
    exit 0
  fi
  # The subsample seed is the uncompressed byte count of the full tagAlign, which bam2ta deletes;
  # measure it with the same bamtobed | awk pipeline, in parallel with the real run.
  cat > "$W/seed.sh" <<'EOF'
bedtools bamtobed -i "$1" | awk 'BEGIN{OFS="\t"}{$4="N";$5="1000";print $0}' | wc -lc
EOF
  $APPT --pwd "$W" "$SIF" /bin/bash "$W/seed.sh" "$W/$LINK" > "$W/seed.txt" &
  SEED_PID=$!
  $APPT --pwd "$W" "$SIF" /bin/bash cmd.sh
  wait "$SEED_PID"
  read -r FULL_LINES FULL_BYTES < "$W/seed.txt"
  echo "full tagAlign: lines=$FULL_LINES uncompressed_bytes=$FULL_BYTES (BAM reads per arms.py: $BAM_READS)"
  [ "$FULL_LINES" = "$BAM_READS" ] || echo "WARNING FULLTA_MISMATCH lines $FULL_LINES != $BAM_READS"
  ls -la "$W"
  [ -s "$W/$OUT" ] || { echo "expected output $OUT not produced" >&2; exit 5; }
  publish "$W/$OUT" "$TA/$OUT" "$EXPECTED"
  prov "$TA/provenance/$OUT.json" "$BAM" "$BAM_MD5" "$TA/$OUT" "$(cut -f4 "$TA/lines.d/$OUT.tsv")" \
       "${LINK%.bam}.tagAlign.gz" "$FULL_BYTES" \
       "ln -s $BAM $W/$LINK" "$RUN_CMD" "$(cat "$W/cmd.sh")"
else
  if [ "$REUSE" = no ]; then
    $APPT --pwd "$W" "$SIF" /bin/bash cmd.sh
    ls -la "$W"
    [ -s "$W/$OUT" ] || { echo "expected output $OUT not produced" >&2; exit 5; }
    publish "$W/$OUT" "$TA/$OUT" "$EXPECTED"
    prov "$TA/provenance/$OUT.json" "$BAM" "$BAM_MD5" "$TA/$OUT" \
         "$(cut -f4 "$TA/lines.d/$OUT.tsv")" - - \
         "ln -s $BAM $W/$LINK" "$RUN_CMD" "$(cat "$W/cmd.sh")"
  fi
  CTL_LINES=$(cut -f2 "$TA/lines.d/$OUT.tsv")
  CTL_BYTES=$(cut -f3 "$TA/lines.d/$OUT.tsv")
  CTL_MD5=$(cut -f4 "$TA/lines.d/$OUT.tsv")
  # The arm-9 N below are round_half_up(q * arms.CONTROLS[acc]["reads"]) and EXPECTED is that same
  # constant, so this one comparison says the N were computed from the file about to be thinned.
  # The first pass thinned to N from a constant 58073570 while the tagAlign held 107039349 lines.
  [ "$CTL_LINES" = "$EXPECTED" ] \
    || { echo "CTLDEPTH_BASE_MISMATCH $TA/$OUT has $CTL_LINES lines but the arm-9 N were computed "\
"from $EXPECTED; refusing to thin" >&2; exit 7; }

  # The base run of the sibling track aligned this control itself; its MACS2 saw that BAM's tagAlign.
  SIB_MD5=$(md5sum "$SIB_BAM" | cut -d' ' -f1)
  ACC=${LINK%%.*}
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$ACC" "$BAM" "$BAM_MD5" "$SIB_BAM" "$SIB_MD5" \
    "$([ "$SIB_MD5" = "$BAM_MD5" ] && echo yes || echo no)" > "$TA/ctl_identity/$ACC.tsv"
  cat "$TA/ctl_identity/$ACC.tsv"

  # arm 9: one thinned control per level, from the published full control tagAlign.
  IFS=',' read -r -a LEVELS <<< "$CTLDEPTH"
  PIDS=()
  for LN in "${LEVELS[@]}"; do
    LV=${LN%%:*}; N=${LN##*:}
    D=$SLURM_TMPDIR/ctldepth/${ACC}__$LV
    mkdir -p "$D"
    printf 'python3 $(which encode_task_subsample_ctl.py) \\\n    %s \\\n    --subsample %s \\\n    \n' \
      "$TA/$OUT" "$N" > "$D/cmd.sh"
    echo "--- $APPT --pwd $D $SIF /bin/bash cmd.sh"; cat "$D/cmd.sh"
    $APPT --pwd "$D" "$SIF" /bin/bash cmd.sh > "$D/run.log" 2>&1 &
    PIDS+=("$!")
  done
  for P in "${PIDS[@]}"; do wait "$P"; done
  for LN in "${LEVELS[@]}"; do
    LV=${LN%%:*}; N=${LN##*:}
    D=$SLURM_TMPDIR/ctldepth/${ACC}__$LV
    cat "$D/run.log"
    shopt -s nullglob; F=("$D"/*.tagAlign.gz); shopt -u nullglob
    [ "${#F[@]}" -eq 1 ] && [ -s "${F[0]}" ] || { echo "ctldepth $LV: want one tagAlign, got ${F[*]:-none}" >&2; exit 6; }
    FD=$TA/ctldepth/${ACC}__$LV
    mkdir -p "$FD"
    B=$(basename "${F[0]}")
    publish "${F[0]}" "$FD/$B" "$N"
    prov "$TA/provenance/$B.json" "$TA/$OUT" "$CTL_MD5" "$FD/$B" "$(cut -f4 "$TA/lines.d/$B.tsv")" \
         "$TA/$OUT" "$CTL_BYTES" \
         "$APPT --pwd $D $SIF /bin/bash cmd.sh" "$(cat "$D/cmd.sh")"
  done
fi
echo "=== $NAME done $(date -u) ==="
