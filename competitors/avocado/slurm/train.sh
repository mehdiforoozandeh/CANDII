#!/bin/bash
# Avocado stage 2 -- the paper's two-stage fit. One GPU per task.
#
#   MODE=shared   ONE task, on the regime's train_chroms[0], everything trainable, 60 epochs
#   MODE=genome   ONE TASK PER EVAL CHROMOSOME (three), shared parameters frozen, per-chromosome
#                 genomic factors, 30 epochs
#
# RETARGETED 2026-08-31. Both counts moved, and both for a ruled reason:
#
#   * The joint fit was on chr20. chr20 is now an EVAL chromosome and that stage fits transferable
#     parameters, so it violates Rule 2. It moves to the regime's train_chroms -- chr19 under
#     eic.19, matching CANDI's own dev scope (§3.2). Nothing is hard-coded: _env.sh reads it.
#   * The genome stage was `--array 0-22`. §4 blanks Avocado's genome-wide cell and rules that a
#     blanked cell is NOT COMPUTED, so Avocado is only ever predicted on chr20+21+22 -- three
#     genomic-factor fits per regime, not 23 (§12.2, corrected 2026-08-30). The other 20 would fit
#     factors for positions nothing is ever scored at.
#
# The joint fit's own chromosome comes free with the shared run and is NOT one of the three.
#
# 60/30 is the HALVED budget RIVALS_PLAN.md 7.1 asks for: 005 ran 120/60 and measured that its
# held-out MSE bottoms out near epoch 25-34 (0.07177) and drifts up afterwards (0.07572 at epoch
# 119). Half is not a saving on a converged run; it is the better run.
#
# CHECKPOINT SELECTION, RULED IN 2026-08-31 (PI): Avocado gets a real selection loop, built to work
# like CANDI's. SELECTEVERY epochs, train.py writes this model's V_ predictions to a §4.1 root and
# scores them with `candi.bench.external.score_external` -- the same function that scores CANDI --
# and keeps the best weights. `--out` ends up holding the SELECTED checkpoint; the last epoch is
# beside it as `.last.pt` and nothing downstream resolves that name.
#
# WHERE it selects is decided by which genomic factors exist, and that answers the open question
# this header used to carry:
#
#   * MODE=genome fits one whole eval chromosome, so it is scored on that chromosome's V_ panel.
#     This is the stage that produces the shipped predictions, so it is the one that must select.
#   * MODE=shared on a whole chromosome (chr19 under eic.19) is scored on chr19's V_ panel.
#   * MODE=shared under a `regions` regime holds factors ONLY inside the BED, and score_external
#     scores whole chromosomes. There is no V_ panel it can be scored on, so SELECTEVERY is forced
#     to 0 below and train.py refuses the combination outright rather than downgrading in silence.
#
# B_ IS NEVER READ. train.py derives a V_-only regime (26 pairs, 12 B_ pairs dropped) and scores
# against that, never against the shipped 38-pair file.
#
# COST. A V_ selection pass is a full scoring pass over one chromosome: ~30-40 min of CPU inside the
# GPU job, against ~7 h of training for a 60-epoch fit. SELECTEVERY=10 is the default for that
# reason -- see README.md §7 for the measurement and the arithmetic. Lengthen it before reaching
# for a bigger walltime.
#
# The gres is the AGENTS.md 3.13 MIG slice. 005 used a full H100 and estimated a 1g.10gb slice at
# ~7x slower -- MEASURE the smoke run's steps/s before trusting that number here, and if the
# projection does not fit, raise the rule rather than reaching for another spec.
#
#   sbatch --array=0-0  --export=ALL,MODE=shared,EPOCHS=60 <this>
#   sbatch --array=0-2  --export=ALL,MODE=genome,EPOCHS=30 <this>
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_avo_train
#SBATCH --output=slurm-logs/t81_avo_train_%A_%a.out
#SBATCH --error=slurm-logs/t81_avo_train_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6

source "${SLURM_SUBMIT_DIR:-$PWD}/competitors/avocado/slurm/_env.sh"

MODE="${MODE:?set MODE=shared|genome}"
SELECTEVERY="${SELECTEVERY:-10}"
case "$MODE" in
  shared) CHROM="$SHARED_SCOPE" ;;
  genome) CHROM="${EVAL_CHROMS[${SLURM_ARRAY_TASK_ID:-0}]:-}" ;;
  *) echo "MODE must be shared or genome, got $MODE" >&2; exit 2 ;;
esac
[ -z "$CHROM" ] && { echo "no chromosome for index ${SLURM_ARRAY_TASK_ID:-0}; this regime has ${#EVAL_CHROMS[@]} eval chromosome(s): ${EVAL_CHROMS[*]}"; exit 1; }

ARGS=(--regime "$REGIME" --chrom "$CHROM" --mode "$MODE"
      --data-root "$WS/binned"
      --out "$WS/ckpt/${MODE}_${CHROM}.pt"
      --log "$WS/logs/train_${MODE}_${CHROM}.jsonl"
      --epochs "${EPOCHS:?set EPOCHS}" --batch-positions "${BATCHPOS:-1024}"
      --lr "${LR:-1e-3}" --genome-lr "${GENOMELR:-1e-2}" --seed "${SEED:-0}"
      --max-hours "${MAXHOURS:-11.5}")
[ "$MODE" = "genome" ] && ARGS+=(--init "$WS/ckpt/shared_${SHARED_SCOPE}.pt")
[ -n "${SMOKESTEPS:-}" ] && ARGS+=(--smoke-steps "$SMOKESTEPS")

if [ "$CHROM" = "regions" ]; then
    ARGS+=(--positions "$WS/binned/regions_layout.csv")
    if [ "$SELECTEVERY" != "0" ]; then
        echo "[avo_train] SELECTEVERY forced to 0: the BED-scoped shared fit has no whole"
        echo "[avo_train]   chromosome to score a V_ panel on. The genome stage still selects."
        SELECTEVERY=0
    fi
fi
[ "$SELECTEVERY" != "0" ] && ARGS+=(--select-every "$SELECTEVERY"
                                    --select-metric "${SELECTMETRIC:-mse}"
                                    --select-patience "${SELECTPATIENCE:-$SELECTEVERY}")

echo "[avo_train] regime=$REGIME_NAME mode=$MODE chrom=$CHROM shared_scope=$SHARED_SCOPE select_every=$SELECTEVERY host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python "$AVO/train.py" "${ARGS[@]}"
rc=$?
[ $rc -ne 0 ] && exit $rc

# KEEP THE SELECTED CHECKPOINT. $WS is /scratch, which purges after 60 days, and the board row this
# fit produced outlives that. `--out` holds the SELECTED weights -- train.py leaves the last epoch
# beside it as `.last.pt` and nothing copies that -- so this is the file a re-score or a provenance
# question needs. A wall-clock kill leaves a `.partial` and no `--out`, so a stage that did not
# finish copies nothing and says so.
SEL="$WS/ckpt/${MODE}_${CHROM}.pt"
if [ -s "$SEL" ]; then
    mkdir -p "$CKPT_KEEP"
    cp -p "$SEL" "$CKPT_KEEP/" && echo "[avo_train] kept $CKPT_KEEP/$(basename "$SEL")"
else
    echo "[avo_train] no $SEL to keep -- this stage wrote no selected checkpoint" >&2
fi
exit $rc
