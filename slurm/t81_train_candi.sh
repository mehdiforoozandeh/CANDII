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
# MODE=probe  measures training throughput and exits. ONE long run, and the rate is read from
#             train.py's OWN `wall=` line, so process startup is excluded by construction. There is
#             no measured store-path training throughput on record — G3 (§12.7) timed INFERENCE only
#             and says so — so this is the number that sizes MODE=full.
#
#             Differencing two runs was tried twice and failed both times, for two DIFFERENT reasons,
#             recorded so nobody retries it. 30 vs 90 steps: startup is 40-55 s and 60 steps cost
#             under 10 s, so startup variance swamped the signal and one regime reported a negative
#             rate. 300 vs 1500: BOTH ARRAY TASKS LANDED ON ONE NODE (fc11013) and competed for 8
#             CPUs and the same networked store — and the loader is CPU/IO bound (§12.7 puts it at
#             37 % of inference) — so the 1500-step run came back FASTER than the 300-step one,
#             123.6 s against 138.9 s, a 5.6x rate difference. Cold-cache warm-up pushes the same
#             way. Hence --array 0-1%1 below: the two regimes are SERIALISED, never co-scheduled.
#
#             The reported rate is an AVERAGE over the whole run and so includes warm-up, which
#             makes it a CONSERVATIVE FLOOR for a run of the SAME KIND.
#
#             IT IS NOT A FLOOR FOR --full-coverage, AND THIS COST A MIS-SIZED JOB. Measured
#             2026-08-31: the probe (--steps-per-epoch 3000, sampled windows) gave 102.74 windows/s,
#             while the real --full-coverage run measured live off its own step counter gave
#             45.9 windows/s — 0.45x, a 2.2x overestimate. Cause is cache locality: the probe
#             samples a small working set that stays cached, while --full-coverage sweeps all
#             155,703 windows across all 51 T_ biosamples every epoch and mostly misses. So the
#             projections below are multiplied by FC_FACTOR. Re-measure it, do not trust it.
# MODE=full   the retrain. Read the walltime off the probe before submitting; the 05:00:00 below is
#             train.sh's h5-path figure and is almost certainly wrong for 35 assays.
#
# THE MID-TRAINING SELECTION SCOPE IS A BED, AND THAT IS WHY MODE=full IS NOW AFFORDABLE. The pilot
# job died at its walltime DURING the final full-coverage check and so wrote no run json at all —
# every mid-training check was full coverage (91-94 min each) and there was no budget left. Under
# EVAL_REGIONS (the default below) a check costs ~9 min instead, and the END-OF-RUN check is full
# coverage whatever this says, which is the number that gets quoted. Set EVAL_REGIONS=full to go
# back to the old, expensive behaviour; nothing else about the run changes.
#
#   mkdir -p slurm-logs
#   MODE=probe sbatch slurm/t81_train_candi.sh          # start here, BOTH regimes, serialised
#   MODE=full  sbatch --time=HH:MM:SS slurm/t81_train_candi.sh
#
# --array=0-1%1 below runs BOTH regimes, one after the other, and never side by side (see the probe
# header for what co-scheduling cost us). To run ONE regime, override the array on the submit line:
#
#   MODE=full sbatch --array=0 --time=HH:MM:SS slurm/t81_train_candi.sh   # eic.19 only
#   MODE=full sbatch --array=1 --time=HH:MM:SS slurm/t81_train_candi.sh   # eic.pilot only
#
# `%1` is inert on a single-element array, so no `%1` is needed on those lines.
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
#SBATCH --array=0-1%1

set -uo pipefail

# --- edit these ---------------------------------------------------------------------------------
KIT="${KIT:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
OUT="${OUT:-/scratch/$USER/t81_candi/runs}"
MODE="${MODE:-probe}"
SEED="${SEED:-0}"
# §5's uniform rule selects the best checkpoint on V_. Under SELECT_ON=V (the default) the eval
# reads V_ only, so a non-zero value here no longer touches B_ — see the PI ruling below.
# It is also the resolution of EARLY_STOP: a stall is only visible on an eval.
EVAL_EVERY="${EVAL_EVERY:-3}"
# WHICH POSITIONS SELECT THE CHECKPOINT (train.py --eval-regions, t89). A seeded random window set,
# NOT the pilot's training BED: configs/regions/encode_pilot_hg38.bed was MEASURED to flip checkpoint
# selections on gaps of 0.069, because its bias drifts with the model and so does not cancel in a
# difference; a random set of the same size did not. See EVAL.md before choosing another one.
#
# The BED is a scope, not a panel — all 45 declared tracks are still scored end to end, over fewer
# POSITIONS. The END-OF-RUN check on the SELECTED checkpoint is FULL COVERAGE whatever this says,
# and that is the number a run reports.
#
# EVAL_REGIONS=full restores the old every-window-of-every-eval-chromosome behaviour and skips the
# assertion below. Any other value is a path to a BED.
EVAL_REGIONS="${EVAL_REGIONS:-$KIT/configs/regions/eval_random450_seed890217.bed}"
# The sha256 of the BED above, pinned HERE and checked against the RUN JSON after training, because
# `configs/regions/` is a mutable directory: a path alone would let two runs claim the same
# selection scope while selecting on different loci. Override only together with EVAL_REGIONS.
EVAL_SCOPE_SHA256="${EVAL_SCOPE_SHA256:-24d9cb9cf6f5db696e4a47e85960f4cee7cc1c98a5bdbd75f1976f755b214ea7}"
PROBE_STEPS="${PROBE_STEPS:-3000}"
# Ratio of real --full-coverage throughput to the sampled probe rate. MEASURED 0.45 on 2026-08-31
# (job 57620803_0, live step counter). See the probe header.
FC_FACTOR="${FC_FACTOR:-0.45}"
# EPOCHS/STEPS_PER_EPOCH exist so MODE=full can be DRESS-REHEARSED cheaply: setting
# STEPS_PER_EPOCH drops --full-coverage, so the whole full-mode path (regime derivation, the eval
# hook, V_ selection, the .best.ckpt write) runs in minutes instead of hours. Leave both unset for
# the real retrain.
# 25 is carried from slurm/train.sh and t22_equiv.sh — and the carry-over is NOT meaningful, which
# matters when sizing a walltime. There, 25 epochs x 200 steps = 5,000 steps = 40,000 windows for
# the WHOLE run, on an 8-assay h5. Here ONE --full-coverage epoch is 19,462 steps / 155,703
# windows — 4x that entire run. So 6 epochs here is already 23x the recorded recipe, 15 is 58x and
# 25 is 97x. Pick this off the V_ eval curve (does V_imp_crps still fall?), not off train.sh.
EPOCHS="${EPOCHS:-25}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-}"
# SELECT_ON=V derives a V_-only eval-pair list so the mid-training eval NEVER READS B_, which is
# what §5 asks for. SELECT_ON=all uses the shipped regime verbatim (26 V_ + 12 B_ pairs) and so
# scores B_ at every eval. Selection is on V_imp_crps either way, so the SELECTED CHECKPOINT is the
# same under both; the difference is whether B_ is touched, plus the eval work saved.
SELECT_ON="${SELECT_ON:-V}"
# Stop when V_imp_crps has not improved for MORE than this many EPOCHS (0 = off, run all $EPOCHS).
# Counted in epochs, so the resolution is EVAL_EVERY: at 3/3 the earliest stop is 6 epochs after the
# best. Nothing is lost — .best.ckpt is written the moment the metric improves and is what gets
# scored. Added because job 57620803_0 selected at epoch 2 and then ran nine more GPU-hours while
# V_imp_crps rose 0.5604 -> 0.5820 -> 0.5864, and nothing in the loop could end it.
EARLY_STOP="${EARLY_STOP:-3}"
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

# --- the mid-training selection scope -------------------------------------------------------------
# Resolved and PRINTED WITH ITS HASH in both modes: probe needs the region count to size a check,
# and full mode needs the hash in the log so the run can be read back without the submit line.
N_SCOPE_WINDOWS=0
if [ "$EVAL_REGIONS" = "full" ]; then
  echo "[t81] eval scope: FULL — every window of every eval chromosome, no --eval-regions passed."
  echo "[t81]   Each mid-training check is then a full-coverage pass (91-94 min measured), and the"
  echo "[t81]   run json's config.eval_scope says name=full. The sha assertion is SKIPPED."
else
  if [ ! -f "$EVAL_REGIONS" ]; then
    echo "[error] EVAL_REGIONS=$EVAL_REGIONS is not a file. Point it at a BED, or set" >&2
    echo "        EVAL_REGIONS=full to select on every window of every eval chromosome." >&2
    exit 1
  fi
  BED_SHA="$(sha256sum "$EVAL_REGIONS" | awk '{print $1}')"
  N_SCOPE_WINDOWS="$(awk '/^[[:space:]]*$/ {next} /^[[:space:]]*(#|track|browser)/ {next} {n++} END {print n+0}' "$EVAL_REGIONS")"
  echo "[t81] eval scope: $EVAL_REGIONS"
  echo "[t81]   sha256  = $BED_SHA"
  echo "[t81]   regions = $N_SCOPE_WINDOWS (this BED's regions are one 768-bin context window each)"
  if [ "$BED_SHA" != "$EVAL_SCOPE_SHA256" ]; then
    echo "[t81]   NOTE: that hash is NOT the pinned EVAL_SCOPE_SHA256=$EVAL_SCOPE_SHA256, so the" >&2
    echo "[t81]   post-run assertion will exit 91. EVAL_REGIONS and EVAL_SCOPE_SHA256 move together." >&2
  fi
fi

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
  # --eval-every 0 on purpose: this measures the TRAINING step rate, and a mid-training eval pass
  # would be counted as training time. It also means the probe reads no B_.
  LOG="$OUT/probe_${NAME}_${PROBE_STEPS}.log"
  python -m candi.train "${COMMON[@]}" --store "$REGIME" \
    --tag "probe_${NAME}_s${SEED}_${PROBE_STEPS}" \
    --epochs 1 --steps-per-epoch "$PROBE_STEPS" --eval-every 0 \
    > "$LOG" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[probe] FAILED rc=$rc — tail:"; tail -30 "$LOG"; exit $rc; fi
  # train.py's own summary: "... N steps, first nll=X last nll=Y wall=ZZZ.Zs". Reading ITS number
  # rather than timing the process is what keeps interpreter and import time out of the rate.
  WALL=$(grep -oE 'wall=[0-9.]+s' "$LOG" | tail -1 | sed 's/wall=//; s/s$//')
  NLL=$(grep -oE 'first nll=[0-9.]+ last nll=[0-9.]+' "$LOG" | tail -1)
  if [ -z "$WALL" ]; then echo "[probe] could not read wall= from $LOG"; tail -20 "$LOG"; exit 4; fi
  echo "[probe] regime=$NAME steps=$PROBE_STEPS wall=${WALL}s  ($NLL)"
  python - <<EOF
wall, steps, bs, fc = $WALL, $PROBE_STEPS, 8, $FC_FACTOR
epochs, eval_every, n_scope = $EPOCHS, $EVAL_EVERY, $N_SCOPE_WINDOWS
rate = steps * bs / wall
fc_rate = rate * fc
print(f"[probe] regime=$NAME  SAMPLED RATE = {rate:.2f} windows/s ({rate/bs:.2f} steps/s)")
print(f"[probe]   G3 measured INFERENCE at 185.6 windows/s on the same MIG slice, for scale.")
print(f"[probe]   x FC_FACTOR {fc} -> --full-coverage rate {fc_rate:.2f} windows/s. The factor is")
print(f"[probe]   why: a sampled probe keeps a small working set cached; full coverage does not.")
rate = fc_rate

# A MODE=full walltime is THREE costs, not one, and the pilot died because only the first was
# budgeted. The two check costs below are MEASURED and are NOT projected off the probe above: a
# check is an inference pass over the whole V_ panel and does not scale with the training rate.
#
#   full-coverage V_ check   91 min PER DIAL (the range on record is 91-94)
#   225-window scoped check  5.85 min total, of which ~206 s is SCOPE-INVARIANT setup
#
# so a scoped check is a fixed floor plus a per-window slope — 450 windows is NOT twice 225.
# train.py's own --eval-regions help calls that check "21% of one epoch". Do not read that as a
# constant: it is a ratio against the epoch of the run it was measured on, and the epoch here is
# sized by --full-coverage. The ratio printed below is against THIS regime's epoch.
FULL_CHECK_MIN = 91.0
FLOOR_MIN = 206.0 / 60.0
PER_WINDOW_MIN = (5.85 - FLOOR_MIN) / 225.0
scoped = (FLOOR_MIN + PER_WINDOW_MIN * n_scope) if n_scope else FULL_CHECK_MIN
n_checks = epochs // eval_every if eval_every else 0
# The END-OF-RUN check runs BOTH dials on the selected checkpoint — the imputation dial that
# selected it and the denoising dial that only watches — and is FULL COVERAGE whatever
# --eval-regions says. It is where the pilot ran out of time, so it is budgeted explicitly.
final = 2 * FULL_CHECK_MIN
scope_label = f"{n_scope} regions" if n_scope else "FULL coverage"
print(f"[probe]   sizing MODE=full: epochs={epochs} eval_every={eval_every} scope={scope_label}")
print(f"[probe]   (EARLY_STOP can only make it shorter, so this is the ceiling to request.)")

# --full-coverage sizes an epoch as (train windows x T_ biosamples), NOT by --steps-per-epoch.
# 51 T_ biosamples is §5.1; the window counts are §3 (chr19) and §3.1 (pilot, the 1,294 ruling).
for label, wpe in (("eic_19    chr19: 3,053 win x 51 T_", 3053 * 51),
                   ("eic_pilot pilot: 1,294 win x 51 T_", 1294 * 51)):
    ep = wpe / rate / 60.0
    train_min, check_min = epochs * ep, n_checks * scoped
    total = train_min + check_min + final
    ask = total * 1.25
    print(f"[probe]   full-coverage {label}: {wpe:,} win/epoch = {ep:6.1f} min/epoch")
    print(f"[probe]     training     {epochs:3d} ep x {ep:6.1f} min = {train_min/60:6.2f} h")
    print(f"[probe]     mid-training {n_checks:3d} ck x {scoped:6.1f} min = {check_min/60:6.2f} h"
          f"  ({scoped/ep:.2f} of one epoch each)")
    print(f"[probe]     final check    2 dials x {FULL_CHECK_MIN:5.1f} min = {final/60:6.2f} h"
          f"  (FULL COVERAGE, always)")
    print(f"[probe]     TOTAL {total/60:6.2f} h  ->  MODE=full sbatch "
          f"--time={int(ask//60):02d}:{int(ask%60):02d}:00  (1.25x margin)")
EOF
  echo "[probe] DONE regime=$NAME"
  exit 0
fi

if [ "$MODE" != "full" ]; then echo "[error] MODE must be probe or full, got $MODE" >&2; exit 1; fi

# RULED 2026-08-31 (PI), and this replaces the open question that stood here.
#
#   "never ever we use B_ in training — V_ is only for checkpoint selection and monitoring,
#    not training"
#
# So the strict reading of §5 is the right one: B_ is not merely kept out of selection, it is not
# READ. SELECT_ON=V is the compliant setting and is the default below; SELECT_ON=all now needs a
# stated reason, because it makes the eval read B_ at every checkpoint.
# V_ is likewise eval-only — it selects and it monitors, and no gradient is ever taken on it.
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

COVERAGE=(--full-coverage)
TAG="t81_${NAME}_s${SEED}"
if [ -n "$STEPS_PER_EPOCH" ]; then
  COVERAGE=(--steps-per-epoch "$STEPS_PER_EPOCH")
  TAG="dress_${NAME}_s${SEED}"
  echo "[t81] DRESS REHEARSAL: $EPOCHS epochs x $STEPS_PER_EPOCH steps, NOT --full-coverage."
  echo "[t81]   This is not a retrain. Its checkpoint is not a board checkpoint."
fi

# The scope flag is built as an array so `full` passes NOTHING rather than a sentinel string —
# train.py's own default IS full coverage, and a flag whose value means "unset" is the kind of thing
# that ends up in a run json as a claim. --eval-batch-size rides along only so the array is never
# empty, which `set -u` would otherwise trip over.
EVALFLAGS=(--eval-batch-size 4)
if [ "$EVAL_REGIONS" != "full" ]; then
  EVALFLAGS+=(--eval-regions "$EVAL_REGIONS")
fi

echo "[t81] launching: epochs=$EPOCHS coverage=${COVERAGE[*]} eval_every=$EVAL_EVERY early_stop=$EARLY_STOP tag=$TAG"
echo "[t81]   eval flags: ${EVALFLAGS[*]}"
python -m candi.train "${COMMON[@]}" --store "$TRAIN_REGIME" \
  --tag "$TAG" \
  --epochs "$EPOCHS" "${COVERAGE[@]}" \
  --eval-every "$EVAL_EVERY" "${EVALFLAGS[@]}" \
  --early-stop-epochs "$EARLY_STOP"
rc=$?
TRAIN_RC=$rc

# WHICH POSITIONS SELECTED THIS CHECKPOINT — asserted against the run json, not against the submit
# line. train.py hashes the BED into `config.eval_scope` and the monitor upgrades that block with
# the window and bin counts it actually planned, so the run json is the only place that records
# what was SCORED rather than what was ASKED FOR. Two runs are comparable on the mid-training curve
# only if this hash matches.
#
# THE JSON IS WRITTEN LAST, AFTER THE FINAL FULL-COVERAGE CHECK (train.py:2092). That is exactly
# how the pilot lost its json: it hit the walltime during that check, so a run that had trained for
# hours and selected a checkpoint recorded nothing about itself. A missing json is therefore NOT a
# reason to skip this assertion — it is the loudest thing it can report.
if [ "$EVAL_REGIONS" != "full" ] && [ "$TRAIN_RC" -eq 0 ]; then
  python - "$OUT/${TAG}.json" "$EVAL_SCOPE_SHA256" "$EVAL_REGIONS" <<'PYEOF'
import json, sys
from pathlib import Path

run_json, want, bed = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if not run_json.is_file():
    sys.exit(f"[t81] eval-scope assertion FAILED: {run_json} does not exist. train.py writes it "
             f"LAST, after the final full-coverage check, so this run either died in that check "
             f"(the pilot's failure — raise the walltime) or never reached it. There is no record "
             f"of which positions selected the checkpoint, so the checkpoint is not usable.")
try:
    cfg = (json.loads(run_json.read_text()).get("config") or {})
except Exception as e:
    sys.exit(f"[t81] eval-scope assertion FAILED: {run_json} is not readable json ({e}).")
scope = cfg.get("eval_scope")
if not isinstance(scope, dict):
    sys.exit(f"[t81] eval-scope assertion FAILED: {run_json} carries no config.eval_scope block. "
             f"--eval-regions {bed} was passed, so one was expected; a run json without it "
             f"predates the scope machinery or the monitor never opened.")
got = scope.get("sha256")
if scope.get("name") == "full":
    # train.py blanks --eval-regions back to `full` when --eval-every resolves to 0: there is no
    # mid-training check to scope, and it refuses to record a claim about a curve that does not
    # exist. So this is EVAL_EVERY=0 with a BED set, which is a contradiction in the submit line.
    sys.exit(f"[t81] eval-scope assertion FAILED: config.eval_scope says name='full', but "
             f"--eval-regions {bed} was passed. train.py blanks the scope when --eval-every is 0 — "
             f"there is no mid-training check to scope, and nothing selected this checkpoint. Set "
             f"EVAL_EVERY>0, or EVAL_REGIONS=full if a full-coverage selection is what you meant.")
if got != want:
    sys.exit(f"[t81] eval-scope assertion FAILED: config.eval_scope.sha256 = {got!r}, pinned "
             f"{want!r} (name={scope.get('name')!r}, bed={scope.get('bed')!r}). The checkpoint was "
             f"selected on DIFFERENT positions than this launcher claims. Do not compare it with "
             f"anything.")
n, tot = scope.get("scored_bins"), scope.get("full_bins")
frac = f", {scope.get('fraction'):.2%} of the eval chromosomes" if scope.get("fraction") else ""
print(f"[t81] eval-scope OK: sha256 {got[:12]} matches the pin; {scope.get('n_regions')} regions, "
      f"{n:,} of {tot:,} bins{frac}." if n and tot else
      f"[t81] eval-scope OK: sha256 {got[:12]} matches the pin.")
PYEOF
  if [ $? -ne 0 ]; then rc=91; fi
elif [ "$EVAL_REGIONS" = "full" ]; then
  echo "[t81] eval-scope assertion SKIPPED: EVAL_REGIONS=full, so selection was full coverage."
fi

# §5's uniform rule is only satisfied if a checkpoint was actually SELECTED on V_. If the eval hook
# never fired, there is no .best.ckpt and the run yields a last-epoch checkpoint instead — which is
# a different object than the design asks for. Say so loudly rather than let it pass as success.
# Gated on TRAIN_RC, not rc: a failed scope assertion must not hide a missing checkpoint.
if [ "$EVAL_EVERY" != "0" ] && [ "$TRAIN_RC" -eq 0 ]; then
  if [ -f "$OUT/${TAG}.best.ckpt" ]; then
    echo "[t81] OK: ${TAG}.best.ckpt exists — a checkpoint was selected on V_imp_crps."
  else
    echo "[t81] WARNING: eval_every=$EVAL_EVERY but NO ${TAG}.best.ckpt was written. The run did" >&2
    echo "[t81]   NOT select on V_, so it does not satisfy BENCHMARK_DESIGN §5. Do not use it." >&2
    rc=90
  fi
fi

echo "[t81] DONE regime=$NAME mode=full rc=$rc"
exit $rc
