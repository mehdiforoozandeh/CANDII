#!/bin/bash
# The two-stage fit on our EIC, leave-one-out contributors. ONE task, then THREE.
#
#   STAGE=shared sbatch --array=0 competitors/lavawizard/slurm/train.sh   # first, and once
#   STAGE=genome sbatch          competitors/lavawizard/slurm/train.sh   # then, one per eval chrom
#   MAX_STEPS=50 STAGE=shared sbatch --array=0 <this>                     # smoke
#
# WHY TWO STAGES: BENCHMARK_DESIGN.md §2 Rule 2, and _env.sh's header has the finding. In one line:
# upstream fits cell factors, assay factors and the network on the chromosome it then predicts, and
# those are transferable parameters. `shared` moves them to the regime's own training scope;
# `genome` freezes them and fits one eval chromosome's position tables, which Rule 2 calls
# inference. The genome stage cannot start without the shared checkpoint and refuses rather than
# falling back to a fresh init — a fresh init here would be the breach, quietly.
#
# THE ENTRY POINT IS `store_eic train`, NOT `lavawizard.train`. BENCHMARK_DESIGN.md §5 makes every
# trainable method select its checkpoint on the V_ panel by the same rule, and that rule is
# `candi.bench.external` over a written §4.1 root — so the run needs the store, which only
# `store_eic` may open. `lavawizard.train` still runs on its own for the Dataset-3 anchor, where
# there is no V_ panel and nothing to select.
#
# SELECTION ATTACHES TO THE GENOME STAGE, IN BOTH REGIMES, and `store_eic.train_shared` says why:
# under eic.pilot the shared scope is a packing of eighteen chromosomes and `score_external` scores
# whole chromosomes, so there is no panel; under eic.19 the scope IS chr19 and there would be.
# Selecting in one regime and not the other would put a difference into the ablation that is not
# the regime. The genome stage is also the only one whose checkpoint is predicted from.
#
# WALLTIME, SIZED FOR THIS SCOPE at the anchor's measured 63 ms/step with TF32 on a 1g.10gb MIG
# slice, 1.58 GiB of the slice's 10. Batch 10,000 and 200+800 epochs is the chr20/21/22 row of
# `dataset3.UPSTREAM_HYPERPARAMS`, which chr19 shares; the packed shared stem borrows it
# (`store_eic.shared_hparams_chrom`). Steps per epoch is ceil(scope / 10,000):
#
#   STAGE=shared   eic.19    2,344,704 bins -> 235 x 1000 = 235,000 steps = 4.1 GPU-h
#                  eic.pilot 1,023,489 bins -> 103 x 1000 = 103,000 steps = 1.8 GPU-h
#   STAGE=genome   chr20 258 / chr21 187 / chr22 204 steps per epoch, x 800 (NO stage 1 — the
#                  trunk it pretrains arrives frozen) = 206,400 + 149,600 + 163,200 = 519,200
#                  steps = 9.1 GPU-h over three tasks; the largest, chr20, is 3.6 h.
#
# So 13.2 GPU-h under eic.19 and 10.9 under eic.pilot, against 11.4 for the one-stage fit this
# replaces: the shared stage is added but stage 1 comes off each of the three eval chromosomes.
# Freezing saves less than it sounds — the position tables are 89 % of the parameters on chr20
# (72.9 M of 81.8 M), so Adam's cost barely moves and 63 ms/step is the figure to plan with.
# `train_<chrom>.json` records the real `ms_per_step` either way.
#
#   * Selection adds one V_ prediction pass plus one scoring pass per check, on that chromosome
#     alone. At SELECT_EVERY=50 that is 16 checks; `train_<chrom>.json` records `eval_seconds`
#     beside `seconds`, so the first run says what the cadence actually cost. The 12 h wall leaves
#     room for a selection budget up to ~2x the training time on the worst task.
# The RAM ask is set by the cache the sampler reads into memory (chr20 is 2.6 GiB of float32 plus
# 0.6 of int8 over 267 tracks) — a memmap on Lustre does not work here, see `preprocess.CachedChrom`.
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
# THREE tasks, not 23: the list is the regime's eval_chroms (see _env.sh). STAGE=shared is ONE.
#SBATCH --array=0-2%3
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"
STAGE="${STAGE:?set STAGE=shared|genome}"
CKPT="${CKPT:-$RUNS/ckpt}"
EXTRA=(); [ -n "${MAX_STEPS:-}" ] && EXTRA=(--max-steps-per-stage "$MAX_STEPS")

if [ "$STAGE" = "shared" ]; then
  echo "[train] $SHARED_STEM (the transferable half of $REGIME_NAME)  host=$(hostname)"; nvidia-smi -L || true
  srun python -u -m lavawizard.store_eic train --regime "$REGIME" --cache "$CACHE" \
       --stage shared --out "$CKPT" --contributor-mode loo --device "${DEVICE:-cuda}" \
       --select-every 0 "${EXTRA[@]}"
  rc=$?
  # The genome stage reads this file by name and refuses without it, so say plainly whether it is
  # there — a three-task array that all fail at minute one is worse than one job that failed loudly.
  if [ $rc -eq 0 ] && [ ! -f "$CKPT/guacamole_${SHARED_STEM}.pt" ]; then
    echo "[train] ERROR: no $CKPT/guacamole_${SHARED_STEM}.pt. The genome stage cannot run." >&2
    rc=91
  fi
  exit $rc
fi

C=${CHROMS[$SLURM_ARRAY_TASK_ID]:-}
[ -n "$C" ] || { echo "[error] array index $SLURM_ARRAY_TASK_ID names no chromosome; this regime has $NCHROM (${CHROMS[*]})" >&2; exit 2; }
# The panel is derived V_-only by `store_eic.selector`, so B_ is never opened during training.
SELECT_EVERY="${SELECT_EVERY:-50}"
# Counted in EPOCHS, as `candi.train --early-stop-epochs` counts it. Nothing is lost by stopping:
# the best weights are written the moment the metric improves.
EARLY_STOP="${EARLY_STOP:-0}"
echo "[train] $C stage=genome init=$CKPT/guacamole_${SHARED_STEM}.pt  host=$(hostname)  select_every=$SELECT_EVERY"; nvidia-smi -L || true
srun python -u -m lavawizard.store_eic train --regime "$REGIME" --cache "$CACHE" \
     --stage genome --chrom "$C" --out "$CKPT" --contributor-mode loo \
     --device "${DEVICE:-cuda}" --select-every "$SELECT_EVERY" \
     --early-stop-epochs "$EARLY_STOP" "${EXTRA[@]}"
rc=$?

# §5 is satisfied only if a checkpoint was actually SELECTED on V_. A run that produced only
# last-epoch weights is a different object than the design asks for — say so rather than exit 0.
if [ "$SELECT_EVERY" != "0" ] && [ $rc -eq 0 ] && [ ! -f "$CKPT/guacamole_${C}.best.pt" ]; then
  echo "[train] WARNING: no $CKPT/guacamole_${C}.best.pt. This run did NOT select on V_, so it" >&2
  echo "[train]   does not satisfy BENCHMARK_DESIGN.md §5. Do not predict from it." >&2
  rc=90
fi
exit $rc
