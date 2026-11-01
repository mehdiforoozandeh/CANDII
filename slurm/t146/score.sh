#!/bin/bash
# t146: score one trained run (tools/t146/score.py): trained, held-out, shuffle and every
# never-trained arm->arm pair, direct; through the base (chain, 16-draw mixture, ceiling) for the
# alone and consist arms; CRPS with its oracle_scaled / scale_error split and per mappability
# decile on every record. CPU only.
#
#   python3 $KIT/tools/t146/score.py run <manifest> <covariates.tsv> <cache_dir> <blacklist> \
#       <out_dir>/runs/<run_name> --features <features_dir> --workers $SLURM_CPUS_PER_TASK \
#       [--law-stride $7 [--mix-stride $8]]           -> scores.json, SCORE_DONE
# SCORE_DONE exists: exit 0. No TRAIN_DONE: exit 3. The run name comes from tasks.py.
#
# NO --export; positional only. Paths as slurm/t146/train.sh, plus
#   BLACKLIST=/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed (sha 31c69342)
# RESOURCES. 8 cores, 32 GB, --time from the header (override per smoke). Fewer than 12 tasks
# should import torch at once from /project: submit with %12 or less.
#   smoke: sbatch --array=<idx> --dependency=aftercorr:<train> ... score.sh ... $OUT/smoke ... 8 16
#   full:  sbatch --array=0-41%12 --dependency=aftercorr:<train> --output=$OUT/logs/%x_%A_%a.out \
#              $KIT/slurm/t146/score.sh $KIT $PRODUCTS $CACHE $OUT $FEATURES $BLACKLIST
#SBATCH --account=def-maxwl
#SBATCH --job-name=t146_score
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none

set -euo pipefail
BLACKLIST_SHA256_PREFIX=31c69342
USAGE="usage: t146/score.sh <kit_dir> <products_dir> <cache_dir> <out_dir> <features_dir> <blacklist.bed> [law_stride [mix_stride]]"
KIT="${1:?$USAGE}"
PRODUCTS="${2:?$USAGE}"
CACHE="${3:?$USAGE}"
OUT="${4:?$USAGE}"
FEATURES="${5:?$USAGE}"
BLACKLIST="${6:?$USAGE}"
LAW_STRIDE="${7:-1}"
MIX_STRIDE="${8:-1}"
IDX="${SLURM_ARRAY_TASK_ID:?array task only}"
[[ "$IDX" =~ ^[0-9]+$ ]] || { echo "SLURM_ARRAY_TASK_ID '$IDX' is not an index" >&2; exit 2; }
[[ "$LAW_STRIDE" =~ ^[0-9]+$ && "$MIX_STRIDE" =~ ^[0-9]+$ ]] || { echo "strides must be integers" >&2; exit 2; }
case "$OUT" in
  */t146|*/t146/*) ;;
  *) echo "out dir $OUT is not t146 or under it; t146 writes only there" >&2; exit 2 ;;
esac
MANIFEST="$PRODUCTS/MANIFEST.tsv"
COVARIATES="$KIT/tools/t118/covariates.tsv"
# the committed copy of `tasks.py tasks` (tests/test_t146.py holds it equal to the table)
RUN=$(awk -F'\t' -v i="$IDX" 'NR > 1 && $1 == i { print $7 }' "$KIT/tools/t146/tasks.tsv")
[ -n "$RUN" ] || { echo "no task $IDX" >&2; exit 2; }
RD="$OUT/runs/$RUN"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"
CMD=(python3 "$KIT/tools/t146/score.py" run "$MANIFEST" "$COVARIATES" "$CACHE" "$BLACKLIST" "$RD"
     --features "$FEATURES" --workers "$WORKERS" --law-stride "$LAW_STRIDE" --mix-stride "$MIX_STRIDE")
if [ "${DRY_RUN:-0}" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
if [ -f "$RD/SCORE_DONE" ]; then echo "skip $RUN: SCORE_DONE exists"; exit 0; fi
[ -f "$RD/TRAIN_DONE" ] || { echo "no $RD/TRAIN_DONE" >&2; exit 3; }
[ -s "$BLACKLIST" ] || { echo "blacklist $BLACKLIST missing or empty" >&2; exit 2; }
case "$(sha256sum "$BLACKLIST" | cut -d' ' -f1)" in
  "$BLACKLIST_SHA256_PREFIX"*) ;;
  *) echo "blacklist sha256 does not start $BLACKLIST_SHA256_PREFIX" >&2; exit 2 ;;
esac
echo "=== t146 score $RUN job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$IDX host=$(hostname) $(date -u)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$KIT/requirements-fir.txt"
pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
export PYTHONPATH="$KIT/src"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
"${CMD[@]}"
echo "=== done $(date -u)"
