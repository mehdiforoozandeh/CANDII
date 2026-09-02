#!/bin/bash
# Avocado stage 3 -- write the RIVALS_PLAN.md 4.1 prediction root for ONE PANEL, one eval
# chromosome per task.
#
# THREE array tasks, not 23. §4 blanks Avocado's `genome-wide` cell and rules that a blanked cell is
# not computed, so the root covers the regime's eval_chroms and nothing else. The old header's "the
# root covers all 23 chromosomes, which serves BOTH protocols: P1 ... P2 ..." described the retired
# P1/P2 naming (§9) and a scope that no longer exists.
#
# THE PANEL AXIS, ADDED 2026-09-01. `predict.py` walks `_expected(open_source(store=regime))`, which
# is every pair the regime DECLARES -- 26 V_ and 12 B_. Running it against the shipped regime spends
# the B_ touch that BENCHMARK_DESIGN.md §5 says happens once, at the end, from the selected
# checkpoint. So this script never predicts from the shipped regime: it derives a PANEL-only copy
# with `tools/declare_eval_pairs.py split` and predicts from that. No python changed -- a derived
# regime is the whole mechanism.
#
#   PANEL=V_   the default. Rerunnable, into /scratch. Everything but the final board row is
#              scored from this root.
#   PANEL=B_   needs B_ONCE=1 AND a root that does not exist yet, and writes to /project. The guard
#              below exits 4 rather than overwriting: a second B_ pass is not a retry, it is a
#              second look at the blind panel, and it cannot be taken back.
#
#   mkdir -p slurm-logs && sbatch --array=0-2%12 competitors/avocado/slurm/predict.sh
#   mkdir -p slurm-logs && sbatch --array=0-2%12 --export=ALL,PANEL=B_,B_ONCE=1 \
#       competitors/avocado/slurm/predict.sh
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_avo_pred
#SBATCH --output=slurm-logs/t81_avo_pred_%A_%a.out
#SBATCH --error=slurm-logs/t81_avo_pred_%A_%a.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

source "${SLURM_SUBMIT_DIR:-$PWD}/competitors/avocado/slurm/_env.sh"

CHROM="${EVAL_CHROMS[${SLURM_ARRAY_TASK_ID:-0}]:-}"
[ -z "$CHROM" ] && { echo "no chromosome for index ${SLURM_ARRAY_TASK_ID:-0}; this regime has ${#EVAL_CHROMS[@]} eval chromosome(s): ${EVAL_CHROMS[*]}"; exit 1; }

PRED="${PRED:-$PRED_PANEL}"
echo "[avo_pred] regime=$REGIME_NAME panel=$PANEL chrom=$CHROM pred=$PRED host=$(hostname)"

# THE ONCE-ONLY B_ VERB.
#
# `manifest.json` is the proof that a B_ pass finished in this root, so its presence is the
# refusal. The marker is what lets the THREE TASKS OF ONE ARRAY through while still refusing a
# later, different one: every task of this array writes the same `.b_once.<array job id>` name, so
# a sibling finds a marker that is its own and a re-submission six weeks later finds one that is
# not.
if [ "$PANEL" = "B_" ]; then
    MARK="$PRED/.b_once.${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
    if [ "${B_ONCE:-0}" != "1" ]; then
        echo "[avo_pred] REFUSING PANEL=B_ without B_ONCE=1. §5 spends the blind panel once, from" >&2
        echo "[avo_pred]   the SELECTED checkpoint, at the end. Say B_ONCE=1 to mean it." >&2
        exit 4
    fi
    if [ -e "$PRED/manifest.json" ] ||
       { compgen -G "$PRED/.b_once.*" >/dev/null 2>&1 && [ ! -e "$MARK" ]; }; then
        echo "[avo_pred] REFUSING: $PRED already holds a B_ pass. B_ is written ONCE (§5) and a" >&2
        echo "[avo_pred]   second one is a second look at the blind panel, not a retry. If that" >&2
        echo "[avo_pred]   root really is a failed write, move it aside by hand and say so." >&2
        exit 4
    fi
    mkdir -p "$PRED" && : > "$MARK"
fi

derive_panel_regime || exit 1

# A train.py job that hits MAXHOURS writes a .partial and exits 0 -- deliberately, so a resubmit
# continues it. That means `--dependency=afterok` on the genome arrays is NOT sufficient to prove
# the checkpoint exists, and a chained predict array would otherwise start against a chromosome
# that never finished. Refuse here, naming the chromosome, rather than letting torch.load raise
# somewhere less legible.
CK="$WS/ckpt/genome_${CHROM}.pt"
if [ ! -s "$CK" ]; then
    if [ -s "$CK.partial" ]; then
        echo "[avo_pred] $CHROM: only $CK.partial exists -- its genome fit stopped at a wall-clock" \
             "deadline and has not finished. Re-submit that chromosome's train.sh array task to" \
             "continue it, then re-run this one." >&2
    else
        echo "[avo_pred] $CHROM: no $CK and no .partial -- its genome fit never ran." >&2
    fi
    exit 1
fi

python "$AVO/predict.py" \
    --regime "$PANEL_REGIME" --chrom "$CHROM" \
    --shared "$WS/ckpt/shared_${SHARED_SCOPE}.pt" \
    --genome "$WS/ckpt/genome_${CHROM}.pt" \
    --out "$PRED" || exit 1

# The manifest, from array task 0 alone -- three tasks racing on one json buys nothing, and the B_
# guard above reads this file. Keyed on the INDEX, not on the name `chr20`: a name key silently
# writes no manifest at all the first time eval_chroms changes. Written AFTER this task's predict,
# so a root carrying a manifest is a root whose task 0 got to the end.
if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then
    python "$AVO/predict.py" --regime "$PANEL_REGIME" --out "$PRED" --write-manifest \
        --version 005-port \
        --notes "panel $PANEL, regime $REGIME_NAME, selected per-chromosome genome checkpoints" \
        || exit 1
fi
