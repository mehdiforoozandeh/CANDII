#!/bin/bash
# One Guacamole per chromosome, on our EIC, leave-one-out contributors. THREE array tasks.
#
#   sbatch competitors/lavawizard/slurm/train.sh
#   MAX_STEPS=50 sbatch --array=1 competitors/lavawizard/slurm/train.sh     # one-chromosome smoke
#   SELECT_EVERY=0 sbatch competitors/lavawizard/slurm/train.sh             # no selection at all
#
# THE ENTRY POINT IS `store_eic train`, NOT `lavawizard.train`. BENCHMARK_DESIGN.md §5 makes every
# trainable method select its checkpoint on the V_ panel by the same rule, and that rule is
# `candi.bench.external` over a written §4.1 root — so the run needs the store, which only
# `store_eic` may open. `lavawizard.train` still runs on its own for the Dataset-3 anchor, where
# there is no V_ panel and nothing to select.
#
# WALLTIME, SIZED FOR THIS SCOPE (the recorded 136.5 GPU-h is for 23 chromosomes and does not carry):
#   * 63 ms/step with TF32 on a 1g.10gb MIG slice, 1.58 GiB of the slice's 10 — measured on the
#     anchor, same code and the same schedule.
#   * The schedule is per-chromosome (`dataset3.UPSTREAM_HYPERPARAMS`): batch 10,000 and
#     200 + 800 epochs for each of chr20/21/22. Steps per epoch is ceil(n_bins / 10,000), so
#     chr20 = 258, chr21 = 187, chr22 = 204 -> 258,000 + 187,000 + 204,000 = 649,000 steps, which
#     is 11.4 GPU-h of TRAINING over the three tasks — the largest task, chr20, is ~4.5 h.
#   * Selection adds one V_ prediction pass plus one scoring pass per check, on that chromosome
#     alone. At SELECT_EVERY=50 that is 16 checks; `train_<chrom>.json` records `eval_seconds`
#     beside `seconds`, so the first run says what the cadence actually cost. The 12 h wall leaves
#     room for a selection budget up to ~1.5x the training time on the worst task.
# The RAM ask is set by the cache the sampler reads into memory (chr1 is 10.6 GiB of float32 plus
# 2.7 of int8) — a memmap on Lustre does not work here, see `preprocess.CachedChrom`.
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
# The panel is derived V_-only by `store_eic.selector`, so B_ is never opened during training.
SELECT_EVERY="${SELECT_EVERY:-50}"
# Counted in EPOCHS, as `candi.train --early-stop-epochs` counts it. Nothing is lost by stopping:
# the best weights are written the moment the metric improves.
EARLY_STOP="${EARLY_STOP:-0}"
echo "[train] $C  host=$(hostname)  select_every=$SELECT_EVERY"; nvidia-smi -L || true
srun python -u -m lavawizard.store_eic train --regime "$REGIME" --cache "$CACHE" --chrom "$C" \
     --out "$CKPT" --contributor-mode loo --device "${DEVICE:-cuda}" \
     --select-every "$SELECT_EVERY" --early-stop-epochs "$EARLY_STOP" "${EXTRA[@]}"
rc=$?

# §5 is satisfied only if a checkpoint was actually SELECTED on V_. A run that produced only
# last-epoch weights is a different object than the design asks for — say so rather than exit 0.
if [ "$SELECT_EVERY" != "0" ] && [ $rc -eq 0 ] && [ ! -f "$CKPT/guacamole_${C}.best.pt" ]; then
  echo "[train] WARNING: no $CKPT/guacamole_${C}.best.pt. This run did NOT select on V_, so it" >&2
  echo "[train]   does not satisfy BENCHMARK_DESIGN.md §5. Do not predict from it." >&2
  rc=90
fi
exit $rc
