#!/bin/bash
# The declared tracks, one array task per eval chromosome, into a single §4.1 root.
#
# WARNING: the regime now declares 38 eval_pairs — 26 V_ AND 12 B_. §5 rules B_ is touched
# ONCE, at the very end, from the selected checkpoint. `store_eic.py predict` walks every
# declared pair, so running this against the shipped regime SPENDS the B_ touch. Derive a
# V_-only regime first (slurm/t81_train_candi.sh shows how) for anything but the final run.
#
#   sbatch competitors/lavawizard/slurm/predict.sh                    # the regime eval scope
#   sbatch --array=1 competitors/lavawizard/slurm/predict.sh          # one chromosome alone
#
# `--clip` is ON here and OFF on the anchor: PI ruling 2026-08-26, recorded in the manifest.
# The manifest is written by array task 0 alone — three tasks racing on one json buys nothing.
# Keyed on the INDEX, not on the name `chr21`: a name key silently writes no manifest at all
# the first time eval_chroms changes.
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
# THREE tasks, not 23: the list is the regime's eval_chroms (see _env.sh).
#SBATCH --array=0-2%3
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"
C=${CHROMS[$SLURM_ARRAY_TASK_ID]:-}
[ -n "$C" ] || { echo "[error] array index $SLURM_ARRAY_TASK_ID names no chromosome; this regime has $NCHROM (${CHROMS[*]})" >&2; exit 2; }
PRED="${PRED:-$RUNS/pred}"
CKPT="${CKPT:-$RUNS/ckpt}"
MAN=(); [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ] && MAN=(--manifest)
# §5: every board number comes from the checkpoint the run SELECTED on V_, so `.best.pt` is what is
# predicted from. The last-epoch file is used only when the run deliberately selected nothing
# (SELECT_EVERY=0, the anchor path), and the fallback is announced rather than silent — a B_ touch
# spent on unselected weights cannot be taken back.
W="$CKPT/guacamole_${C}.best.pt"
if [ ! -f "$W" ]; then
  W="$CKPT/guacamole_${C}.pt"
  echo "[predict] NO .best.pt for $C — predicting from the LAST-epoch checkpoint, which does not" >&2
  echo "[predict]   satisfy BENCHMARK_DESIGN.md §5. Correct only for a run with no selection." >&2
fi
echo "[predict] $C -> $PRED  weights=$(basename "$W")  host=$(hostname)"
srun python -u -m lavawizard.store_eic predict --regime "$REGIME" --chrom "$C" \
     --cache "$CACHE" --checkpoint "$W" \
     --pred-root "$PRED" --device "${DEVICE:-cuda}" --clip "${MAN[@]}"
