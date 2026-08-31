#!/bin/bash
# t81 — CANDI's two retrains, one per live regime (plan/BENCHMARK_DESIGN.md §12.2).
#
# #SBATCH skeleton from slurm/train.sh; --gres is the HARD-RULE MIG slice (AGENTS.md invariant 13)
# and is never any other spec. Two differences from train.sh, both deliberate:
#
#  1. PYTHONPATH IS SET, not unset. train.sh cds into $KIT and leans on the venv's editable
#     install, which pins a checkout that is not this one (t39 is open on exactly that). A retrain
#     whose code provenance is "whatever the venv points at" is not reproducible, so the checkout is
#     named explicitly and echoed with its git sha.
#  2. train.sh's eval flags (--eval-max-batches, --eval-budget, --m3-regions, --fg-frac, --n-boot)
#     are NOT carried. On the store path M1/M2/M3/S14 are not scored at all — `--store`'s own help
#     says "training only" — so those flags are inert here and carrying them would imply a scoring
#     block that does not run.
#
# MODE=probe  measures training throughput and exits. Runs 30 steps, then 90 steps, and DIFFERENCES
#             them, so the store's manifest read and window planning cancel out of the rate instead
#             of being amortised into it. There is no measured store-path training throughput on
#             record — G3 (§12.7) timed INFERENCE only and says so — and --full-coverage on chr19
#             across 51 T_ biosamples is ~155,700 windows/epoch, so the full job's walltime is
#             unknown until this runs. Cheap, and it is the number that sizes MODE=full.
# MODE=full   the retrain. Read the walltime off the probe before submitting; the 05:00:00 below is
#             train.sh's h5-path figure and is almost certainly wrong for 35 assays.
#
#   mkdir -p slurm-logs
#   MODE=probe sbatch slurm/t81_train_candi.sh          # start here
#   MODE=full  sbatch --time=HH:MM:SS slurm/t81_train_candi.sh
#
# Logs resolve against the SUBMITTING cwd, not the script's location, so mkdir -p slurm-logs first.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_candi
#SBATCH --output=slurm-logs/t81_train_%A_%a.out
#SBATCH --error=slurm-logs/t81_train_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1

set -uo pipefail

# --- edit these ---------------------------------------------------------------------------------
KIT="${KIT:-/project/def-maxwl/mforooz/CANDII_t78_code}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
OUT="${OUT:-/scratch/$USER/t81_candi/runs}"
MODE="${MODE:-probe}"
SEED="${SEED:-0}"
# §5's uniform rule selects the best checkpoint on V_. See the WARNING under [t81] below before
# changing this: as shipped, any non-zero value also LOGS B_ every eval, which §5 forbids.
EVAL_EVERY="${EVAL_EVERY:-3}"
# -------------------------------------------------------------------------------------------------

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export WANDB_MODE=disabled

if [ ! -d "$VENV" ]; then echo "[error] no venv at $VENV" >&2; exit 1; fi
source "$VENV/bin/activate"
export PYTHONPATH="$KIT/src"

cd "$KIT"
mkdir -p "$OUT"

REGIMES=(configs/regime.eic_19.json configs/regime.eic_pilot.json)
NAMES=(eic_19 eic_pilot)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
REGIME="${REGIMES[$IDX]}"
NAME="${NAMES[$IDX]}"

CODE_SHA="$(git rev-parse --short HEAD)"
echo "[t81] task=$IDX regime=$NAME mode=$MODE seed=$SEED code=$CODE_SHA host=$(hostname)"
echo "[t81] PYTHONPATH=$PYTHONPATH"
python -c "import candi, sys; print('[t81] candi from', candi.__file__)" || exit 1
python -c "import json; d=json.load(open('$REGIME')); print('[t81] kinds =', d['kinds'], '| train =', d['train_chroms'], '| eval =', d['eval_chroms'], '| pairs =', len(d.get('eval_pairs') or []))" || exit 1
nvidia-smi -L || true

# `--heads count,signal,peak` is what §12.2 asks of CANDI, and it is the reason the regime must
# declare the `pval` kind: the signal head's target is y_pval, and a regime that does not load the
# layer supervises it on a plane of zeros in silence (tests/test_train_store.py guards this).
# --signal-target-transform stays `auto`, which resolves to `arcsinh` on the store path; the
# RESOLVED value lands in the run config and two runs only compare on the signal head if it matches.
COMMON=(
  --store "$REGIME" --out-dir "$OUT"
  --heads count,signal,peak
  --weight-decay 0.0
  --dsf-sampling uniform --batch-size 8
  --seed "$SEED"
)

if [ "$MODE" = "probe" ]; then
  # Two runs, differenced. --eval-every 0 on purpose: this measures the TRAINING step rate, and a
  # mid-training eval pass would be counted as training time. It also means the probe reads no B_.
  for STEPS in 30 90; do
    T0=$(date +%s.%N)
    python -m candi.train "${COMMON[@]}" \
      --tag "probe_${NAME}_s${SEED}_${STEPS}" \
      --epochs 1 --steps-per-epoch "$STEPS" --eval-every 0 \
      > "$OUT/probe_${NAME}_${STEPS}.log" 2>&1
    rc=$?
    T1=$(date +%s.%N)
    D=$(python -c "print(f'{$T1 - $T0:.3f}')")
    echo "[probe] regime=$NAME steps=$STEPS rc=$rc wall_s=$D"
    if [ $rc -ne 0 ]; then echo "[probe] FAILED — tail of log:"; tail -30 "$OUT/probe_${NAME}_${STEPS}.log"; exit $rc; fi
    eval "W$STEPS=$D"
  done
  python - <<EOF
w30, w90 = $W30, $W90
steps, bs = 60, 8
dt = w90 - w30
rate = steps * bs / dt
print(f"[probe] regime=$NAME  startup={w30 - 30*bs/rate:8.1f}s  steady={dt:8.1f}s for {steps} steps")
print(f"[probe] TRAINING RATE = {rate:.2f} windows/s  ({rate/bs:.2f} steps/s)")
for label, wpe in (("chr19 full-coverage (51 T_ x 3053 win)", 155703),):
    ep = wpe / rate
    print(f"[probe] {label}: {ep/60:.1f} min/epoch -> 25 epochs = {25*ep/3600:.2f} h")
EOF
  echo "[probe] DONE regime=$NAME"
  exit 0
fi

if [ "$MODE" != "full" ]; then echo "[error] MODE must be probe or full, got $MODE" >&2; exit 1; fi

# WARNING, and it is a §5 compliance question the PI has not ruled on yet.
# --eval-every N drives best-checkpoint selection on V_imp_crps (train.py:1392) and writes
# {tag}.best.ckpt — that is §5's uniform selection rule, and every trainable method needs it.
# But the same hook ALSO computes and prints B_imp_crps every time (train.py:1398-1404), and §5's
# panel table lists B_ as "used during training: never" on a row that includes MONITORING. So
# EVAL_EVERY=3 satisfies the selection rule and violates the never-touch rule; EVAL_EVERY=0 does the
# reverse. There is no shipped flag that does both. Do not resolve this by picking one silently.
echo "[t81] eval_every=$EVAL_EVERY — best checkpoint selected on V_imp_crps, written to *.best.ckpt"
if [ "$EVAL_EVERY" != "0" ]; then
  echo "[t81] NOTE: the eval hook also logs B_imp_crps; see the WARNING in this script."
fi

python -m candi.train "${COMMON[@]}" \
  --tag "t81_${NAME}_s${SEED}" \
  --epochs 25 --full-coverage \
  --eval-every "$EVAL_EVERY" --eval-batch-size 4
rc=$?

echo "[t81] DONE regime=$NAME mode=full rc=$rc"
exit $rc
