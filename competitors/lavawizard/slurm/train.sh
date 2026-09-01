#!/bin/bash
# One Guacamole per chromosome, on our EIC, leave-one-out contributors. THREE array tasks.
#
#   sbatch competitors/lavawizard/slurm/train.sh
#   MAX_STEPS=50 sbatch --array=1 competitors/lavawizard/slurm/train.sh     # one-chromosome smoke
#
# Measured on the anchor, same code and the same schedule: 63 ms/step with TF32 on a 1g.10gb MIG
# slice, 1.58 GiB of the slice's 10, and 136.5 GPU-hours over 23 chromosomes — which is NO LONGER
# THE SCOPE. chr20+21+22 are 6,478,903 of the 121,241,684 genome bins (5.34 %), and the schedule
# is per-chromosome, so three tasks is ~10-12 GPU-h total, not 136.5. The 12 h per-task wall
# below still holds: it was sized for a single chromosome and these three are mid-sized. The RAM ask is
# set by the cache the sampler reads into memory (chr1 is 10.6 GiB of float32 plus 2.7 of int8) —
# a memmap on Lustre does not work here, see `preprocess.CachedChrom`.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t53_train
#SBATCH --output=slurm-logs/%x_%A_%a.out
#SBATCH --error=slurm-logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# fc30560 gave repeated Lustre OSError 108 on files its neighbours read fine (anchor run).
#SBATCH --exclude=fc30560
# THREE tasks, not 23: the list is the regime's eval_chroms (see _env.sh).
#SBATCH --array=0-2%3
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"
C=${CHROMS[$SLURM_ARRAY_TASK_ID]:-}
[ -n "$C" ] || { echo "[error] array index $SLURM_ARRAY_TASK_ID names no chromosome; this regime has $NCHROM (${CHROMS[*]})" >&2; exit 2; }
CKPT="${CKPT:-$RUNS/ckpt}"
EXTRA=(); [ -n "${MAX_STEPS:-}" ] && EXTRA=(--max-steps-per-stage "$MAX_STEPS")
echo "[train] $C  host=$(hostname)"; nvidia-smi -L || true
srun python -u -m lavawizard.train --cache "$CACHE" --chrom "$C" --out "$CKPT" \
     --contributor-mode loo --device "${DEVICE:-cuda}" "${EXTRA[@]}"
