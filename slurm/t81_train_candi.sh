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
# MODE=probe  measures training throughput and exits. Runs PROBE_LO steps, then PROBE_HI, and
#             DIFFERENCES them, so the store's manifest read and window planning cancel out of the
#             rate instead of being amortised into it. The two counts must be FAR apart: the first
#             attempt used 30 and 90 and was invalid, because startup is 40-55 s while 60 steps cost
#             under 10 s, so startup variance swamped the signal and one regime reported a negative
#             rate. 300 vs 1500 puts ~100 s of step work against ~10 s of startup noise. There is no measured store-path training throughput on
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
PROBE_LO="${PROBE_LO:-300}"
PROBE_HI="${PROBE_HI:-1500}"
# SELECT_ON=V derives a V_-only eval-pair list so the mid-training eval NEVER READS B_, which is
# what §5 asks for. SELECT_ON=all uses the shipped regime verbatim (26 V_ + 12 B_ pairs) and so
# scores B_ at every eval. Selection is on V_imp_crps either way, so the SELECTED CHECKPOINT is the
# same under both; the difference is whether B_ is touched, plus the eval work saved.
SELECT_ON="${SELECT_ON:-V}"
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
# NOTE: --store is deliberately NOT in here. probe mode uses $REGIME; full mode may use a derived
# V_-only copy, and bash array pattern substitution cannot rewrite a flag and its value together
# because they are separate elements.
COMMON=(
  --out-dir "$OUT"
  --heads count,signal,peak
  --weight-decay 0.0
  --dsf-sampling uniform --batch-size 8
  --seed "$SEED"
)

if [ "$MODE" = "probe" ]; then
  # Two runs, differenced. --eval-every 0 on purpose: this measures the TRAINING step rate, and a
  # mid-training eval pass would be counted as training time. It also means the probe reads no B_.
  for STEPS in "$PROBE_LO" "$PROBE_HI"; do
    T0=$(date +%s.%N)
    python -m candi.train "${COMMON[@]}" --store "$REGIME" \
      --tag "probe_${NAME}_s${SEED}_${STEPS}" \
      --epochs 1 --steps-per-epoch "$STEPS" --eval-every 0 \
      > "$OUT/probe_${NAME}_${STEPS}.log" 2>&1
    rc=$?
    T1=$(date +%s.%N)
    D=$(python -c "print(f'{$T1 - $T0:.3f}')")
    echo "[probe] regime=$NAME steps=$STEPS rc=$rc wall_s=$D"
    if [ $rc -ne 0 ]; then echo "[probe] FAILED — tail of log:"; tail -30 "$OUT/probe_${NAME}_${STEPS}.log"; exit $rc; fi
    if [ "$STEPS" = "$PROBE_LO" ]; then WLO="$D"; else WHI="$D"; fi
  done
  python - <<EOF
wlo, whi = $WLO, $WHI
nlo, nhi, bs = $PROBE_LO, $PROBE_HI, 8
dn = nhi - nlo
dt = whi - wlo
print(f"[probe] regime=$NAME  {nlo} steps={wlo:.1f}s  {nhi} steps={whi:.1f}s  delta={dt:.1f}s for {dn} steps")
# Refuse to report a rate the measurement cannot support. dt must be large against the startup
# variance we actually observed (about 5-15 s), or the difference is noise and not a rate.
if dt < 30.0:
    print(f"[probe] INVALID: delta {dt:.1f}s is not large against startup variance (~5-15s). "
          f"Raise PROBE_HI and re-run. NO RATE REPORTED.")
    raise SystemExit(3)
rate = dn * bs / dt
startup = wlo - nlo * bs / rate
print(f"[probe] startup={startup:.1f}s (paid once per run, not per epoch)")
print(f"[probe] TRAINING RATE = {rate:.2f} windows/s  ({rate/bs:.2f} steps/s)")
print(f"[probe] for reference, G3 measured INFERENCE at 185.6 windows/s on the same MIG slice")
# --full-coverage sizes an epoch as (train windows x T_ biosamples), not by --steps-per-epoch, so
# these are the numbers that set MODE=full's walltime. 51 T_ biosamples is from §5.1.
for label, wpe in (("chr19 (2,344,704 bins / 768 = 3,053 win x 51 T_)", 3053 * 51),
                   ("pilot (1,023,489 bins -> 1,294 win x 51 T_)", 1294 * 51)):
    ep = wpe / rate
    print(f"[probe] full-coverage {label}: {wpe:,} win/epoch = {ep/60:.1f} min/epoch "
          f"-> 25 epochs = {25*ep/3600:.2f} h")
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

# Derive a V_-only regime rather than shipping a second 340-line config that could drift from its
# original. `regions.bed` RESOLVES AGAINST THE REGIME FILE'S OWN DIRECTORY (store/regime.py:52), so
# the derived copy must carry an absolute BED path or the pilot regime fails its sha256 check.
TRAIN_REGIME="$REGIME"
if [ "$SELECT_ON" = "V" ]; then
  TRAIN_REGIME="$OUT/regime.${NAME}.vsel.json"
  python - "$REGIME" "$TRAIN_REGIME" <<'PYEOF' || exit 1
import json, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
d = json.loads(src.read_text())
pairs = [tuple(p) if not isinstance(p, dict) else (p["input"], p["target"]) for p in d["eval_pairs"]]
kept = [list(p) for p in pairs if str(p[1]).startswith("V_")]
dropped = [list(p) for p in pairs if not str(p[1]).startswith("V_")]
if not kept:
    sys.exit("no V_ pairs to select on")
d["eval_pairs"] = kept
# `eval_pairs` being set makes biosamples.eval inert (regime.py:501 — the pool comes from the
# pairs), but leaving 38 cells in a V_-only config would be a false claim about the eval split.
d["biosamples"]["eval"] = sorted({p[1] for p in kept})
if d.get("regions"):
    d["regions"]["bed"] = str((src.parent / d["regions"]["bed"]).resolve())
d["_comment"] = ("DERIVED at submit time by slurm/t81_train_candi.sh from " + src.name +
                 " — eval_pairs filtered to V_ targets only, so the mid-training eval never reads "
                 "B_ (BENCHMARK_DESIGN.md §5). Selection is on V_imp_crps in both cases, so this "
                 "does not change which checkpoint is selected. Do not edit; edit the source.")
dst.write_text(json.dumps(d, indent=2))
print(f"[t81] derived {dst.name}: kept {len(kept)} V_ pairs, dropped {len(dropped)} B_ pairs")
print(f"[t81]   eval biosamples now {len(d['biosamples']['eval'])} (V_ only)")
if d.get("regions"):
    print(f"[t81]   regions.bed rewritten absolute: {d['regions']['bed']}")
PYEOF
  # Prove it parses, and that the hash gate still passes on the rewritten BED path.
  python -c "
import sys; sys.path.insert(0, '$KIT/src')
from candi.store.regime import Regime
r = Regime.from_file('$TRAIN_REGIME')
tgts = {t.split('_')[0] + '_' for _, t in r.eval_pairs}
assert tgts == {'V_'}, f'derived regime still targets {tgts}'
print(f'[t81] derived regime OK: {len(r.eval_pairs)} pairs, targets {sorted(tgts)}, regions={bool(r.regions)}')
" || exit 1
else
  echo "[t81] SELECT_ON=$SELECT_ON — using the shipped regime verbatim; the eval WILL read B_ at"
  echo "[t81]   every checkpoint, which BENCHMARK_DESIGN §5 lists as never. Deliberate only."
fi

python -m candi.train "${COMMON[@]}" --store "$TRAIN_REGIME" \
  --tag "t81_${NAME}_s${SEED}" \
  --epochs 25 --full-coverage \
  --eval-every "$EVAL_EVERY" --eval-batch-size 4
rc=$?

echo "[t81] DONE regime=$NAME mode=full rc=$rc"
exit $rc
