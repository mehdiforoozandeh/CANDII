#!/bin/bash
# The 45 declared tracks, one array task per chromosome, into a single §4.1 root.
#
#   sbatch competitors/lavawizard/slurm/predict.sh                    # all 23 -> P2 (P1 inside it)
#   sbatch --array=20 competitors/lavawizard/slurm/predict.sh         # chr21 alone -> P1
#
# `--clip` is ON here and OFF on the anchor: PI ruling 2026-08-26, recorded in the manifest.
# The manifest is written by the chr21 task alone — 23 tasks racing on one json buys nothing.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t53_predict
#SBATCH --output=slurm-logs/%x_%A_%a.out
#SBATCH --error=slurm-logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
# fc30560 gave repeated Lustre OSError 108 on files its neighbours read fine (anchor run).
#SBATCH --exclude=fc30560
# %12: more than twelve concurrent torch imports off the shared /project venv fail with
# partial-module ImportErrors. The cap is the fix, and it costs only wall-clock.
#SBATCH --array=0-22%12
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"
C=${CHROMS[$SLURM_ARRAY_TASK_ID]:-}
[ -n "$C" ] || { echo "[error] array index $SLURM_ARRAY_TASK_ID names no chromosome" >&2; exit 2; }
PRED="${PRED:-$RUNS/pred}"
CKPT="${CKPT:-$RUNS/ckpt}"
MAN=(); [ "$C" = "chr21" ] && MAN=(--manifest)
echo "[predict] $C -> $PRED  host=$(hostname)"
srun python -u -m lavawizard.store_eic predict --regime "$REGIME" --chrom "$C" \
     --cache "$CACHE" --checkpoint "$CKPT/guacamole_${C}.pt" \
     --pred-root "$PRED" --device "${DEVICE:-cuda}" --clip "${MAN[@]}"
