#!/bin/bash
# t25 — rebuild the pval layer of a CANDI_STORE corpus under the D25 arcsinh codec.
#
# This is `cruxvault/results/t12/t12_build_pval.sh` with three changes: the job name, the log paths,
# and the two codec flags below.  Everything else is deliberately byte-identical, because t12's
# sizing is MEASURED (450 tasks, mean 84 s, max 319 s, 0 failures, 10.57 node-hours) and this run
# does the same work plus ~7% larger writes.
#
# WHY THIS IS A REBUILD FROM SOURCE AND NOT A TRANSCODE (D29).  Transcoding the existing pval.h5
# would faithfully preserve all 3,046,724 bins that the old ceiling of 655.35 already clipped --- the
# information is gone from the file and only the source npz still has it.  There is no shortcut.
#
# --kinds pval ONLY.  writer.py::build_biosample loops over the kinds it was given and never opens a
# file for a kind it was not given, so counts.h5 and peaks.h5 are not read, not rewritten, not
# touched.  --overwrite is therefore scoped to pval.h5 alone, which makes a failed-task resubmit
# idempotent and keeps the transient disk cost at ONE FILE rather than 289 GB.
#
# --pval-scale / --pval-transform are the shipped defaults after t24.  They are passed EXPLICITLY
# anyway: a 455-file rebuild that silently inherited a default is a rebuild whose codec you have to
# go and measure afterwards.  This way the choice is in every task's sbatch log.
#
# DO NOT SUBMIT THIS BEFORE t24 IS MERGED AND $KIT IS SYNCED.  A half-updated kit produces a corpus
# in two codecs with no way to tell which biosample got which --- the `transform` root attr tells you
# what a file IS, not what it was supposed to be.
#
# --gres note, verbatim from t12: the def-maxwl_cpu fairshare is 0.088 against the GPU account's
# 0.435, so a plain CPU job effectively never starts.  The MIG slice idles for the duration; that is
# the intended trade.  Never use any other gres spec (project hard rule).
#
# %15 throttles concurrency.  Be a good citizen: the same Lustre source tree serves everything else.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t25_pval
#SBATCH --output=/project/def-maxwl/mforooz/CANDI_STORE/t25/logs/pval_%A_%a.out
#SBATCH --error=/project/def-maxwl/mforooz/CANDI_STORE/t25/logs/pval_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -uo pipefail

STORE=/project/def-maxwl/mforooz/CANDI_STORE
CORPUS="${CORPUS:?set CORPUS=eic, merged or eic_slice}"
SRC="${SRC:?set SRC=/project/6014832/mforooz/DATA_CANDI_EIC or ..._MERGED}"
KIT=/home/mforooz/projects/def-maxwl/mforooz/CANDII
LIST="${LIST:-$STORE/t25/${CORPUS}_biosamples.txt}"
PVAL_SCALE="${PVAL_SCALE:-2000}"
PVAL_TRANSFORM="${PVAL_TRANSFORM:-arcsinh}"

B=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
if [ -z "$B" ]; then
  echo "[t25] no biosample at index $SLURM_ARRAY_TASK_ID in $LIST" >&2; exit 2
fi

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate

cd "$KIT"
export PYTHONPATH="$KIT/src"

echo "[t25] host=$(hostname) corpus=$CORPUS biosample=$B idx=$SLURM_ARRAY_TASK_ID"
echo "[t25] codec=${PVAL_TRANSFORM} scale=${PVAL_SCALE} kit=$(git -C "$KIT" rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[t25] started $(date -Is)"
t0=$(date +%s)

python -m candi.store build-biosample \
  --source-root "$SRC" \
  --corpus-root "$STORE/$CORPUS" \
  --chrom-sizes "$STORE/genome/chrom_sizes.json" \
  --biosample "$B" \
  --kinds pval \
  --pval-scale "$PVAL_SCALE" \
  --pval-transform "$PVAL_TRANSFORM" \
  --overwrite
rc=$?

t1=$(date +%s)
echo "[t25] corpus=$CORPUS biosample=$B rc=$rc elapsed_s=$((t1-t0)) finished $(date -Is)"
ls -l "$STORE/$CORPUS/biosamples/$B/pval.h5" 2>/dev/null
exit $rc
