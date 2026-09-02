#!/bin/bash
# One PANEL's declared tracks, one array task per eval chromosome, into a single §4.1 root.
#
# THE PANEL AXIS, ADDED 2026-09-01, AND IT REPLACES A WARNING WITH A MECHANISM. The header used to
# say: the regime declares 38 eval_pairs, `store_eic.py predict` walks every one of them, so running
# this against the shipped regime SPENDS the B_ touch — derive a V_-only regime first. That put the
# rule in a comment and left the breach one `sbatch` away. It is the code now: this script derives a
# PANEL-only regime with `tools/declare_eval_pairs.py split` and predicts from that, always.
#
#   PANEL=V_   the default. Rerunnable, into /scratch. Everything but the final board row is
#              scored from this root.
#   PANEL=B_   needs B_ONCE=1 AND a root that does not exist yet, and writes to /project. §5 spends
#              the blind panel once, from the SELECTED checkpoint, at the end; the guard below
#              exits 4 rather than overwriting.
#
#   sbatch competitors/lavawizard/slurm/predict.sh                    # V_, the regime eval scope
#   sbatch --array=1 competitors/lavawizard/slurm/predict.sh          # one chromosome alone
#   sbatch --export=ALL,PANEL=B_,B_ONCE=1 competitors/lavawizard/slurm/predict.sh
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
# THREE tasks, not 23: the list is the regime's eval_chroms (see _env.sh). %3 sits under the %12
# torch-import cap the shared /project venv needs.
#SBATCH --array=0-2%3
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"
C=${CHROMS[$SLURM_ARRAY_TASK_ID]:-}
[ -n "$C" ] || { echo "[error] array index $SLURM_ARRAY_TASK_ID names no chromosome; this regime has $NCHROM (${CHROMS[*]})" >&2; exit 2; }
PRED="${PRED:-$PRED_PANEL}"
CKPT="${CKPT:-$RUNS/ckpt}"

# THE ONCE-ONLY B_ VERB.
#
# `manifest.json` is the proof that a B_ pass finished in this root, so its presence is the refusal.
# The marker is what lets the THREE TASKS OF ONE ARRAY through while still refusing a later,
# different one: every task of this array writes the same `.b_once.<array job id>` name, so a
# sibling finds a marker that is its own and a re-submission six weeks later finds one that is not.
#
# THE MARKER IS WRITTEN LAST, AFTER EVERY PRECONDITION. It used to be dropped here, before the panel
# regime was derived and before the weights were resolved — so a `split` that could not run left a
# `.b_once.*` in an empty root, and every LATER submission read it as "the blind panel has already
# been spent". A trivial failure must not poison a root that holds nothing. The refusals stay here,
# where they cost nothing and fire early.
if [ "$PANEL" = "B_" ]; then
  MARK="$PRED/.b_once.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
  if [ "${B_ONCE:-0}" != "1" ]; then
    echo "[predict] REFUSING PANEL=B_ without B_ONCE=1. §5 spends the blind panel once, from the" >&2
    echo "[predict]   SELECTED checkpoint, at the end. Say B_ONCE=1 to mean it." >&2
    exit 4
  fi
  if [ -e "$PRED/manifest.json" ] ||
     { compgen -G "$PRED/.b_once.*" >/dev/null 2>&1 && [ ! -e "$MARK" ]; }; then
    echo "[predict] REFUSING: $PRED already holds a B_ pass. B_ is written ONCE (§5) and a second" >&2
    echo "[predict]   one is a second look at the blind panel, not a retry. If that root really is" >&2
    echo "[predict]   a failed write, move it aside by hand and say so." >&2
    exit 4
  fi
fi

derive_panel_regime || exit 1

MAN=(); [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ] && MAN=(--manifest)
# §5: every board number comes from the checkpoint the run SELECTED on V_, so `.best.pt` is what is
# predicted from. The last-epoch file is used only when the run deliberately selected nothing
# (SELECT_EVERY=0, the anchor path), and the fallback is announced rather than silent — a B_ touch
# spent on unselected weights cannot be taken back.
#
# AND THE FALLBACK IS NOT A LICENCE TO PREDICT FROM NOTHING. With neither file present the task
# used to warn, keep an unreadable `$W`, claim the root with the `.b_once` marker, and only then
# die inside `store_eic predict` — spending the blind panel on a run that never trained. Exit here,
# BEFORE the marker, exactly as `competitors/avocado/slurm/predict.sh` exits 1 on a missing genome
# checkpoint.
W="$CKPT/guacamole_${C}.best.pt"
if [ ! -f "$W" ]; then
  W="$CKPT/guacamole_${C}.pt"
  if [ ! -f "$W" ]; then
    echo "[predict] $C: no $CKPT/guacamole_${C}.best.pt and no $W — this chromosome's fit never" >&2
    echo "[predict]   ran, or its checkpoints are under another CKPT. Nothing is predicted and no" >&2
    echo "[predict]   marker is written, so the B_ root is untouched and can still be claimed." >&2
    exit 1
  fi
  echo "[predict] NO .best.pt for $C — predicting from the LAST-epoch checkpoint, which does not" >&2
  echo "[predict]   satisfy BENCHMARK_DESIGN.md §5. Correct only for a run with no selection." >&2
fi
# Every precondition has passed, so this array is the B_ pass it says it is: claim the root now.
if [ "$PANEL" = "B_" ]; then
  mkdir -p "$PRED" && : > "$MARK" || exit 1
fi
echo "[predict] $C panel=$PANEL -> $PRED  weights=$(basename "$W")  host=$(hostname)"
srun python -u -m lavawizard.store_eic predict --regime "$PANEL_REGIME" --chrom "$C" \
     --cache "$CACHE" --checkpoint "$W" \
     --pred-root "$PRED" --device "${DEVICE:-cuda}" --clip "${MAN[@]}"
