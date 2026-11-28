#!/bin/bash
# Task t370 (miniCANDI recipe run), chunk C4: one entry point per stage for the recipe arms of
# tools/minicandi/recipe_arms.py (configs in this folder). Run on a Fir login node.
# DRY RUN BY DEFAULT: it prints the sbatch lines, the job count and the projected CPU-hours, and
# submits nothing. DRY_RUN=0 submits (the foreman's call only).
#
#   K=<kit dir> STAGE=probe|s1a|s1b|s1c|arms SUB=p16|p256|native|native256|foreign \
#     [ARMS="<arm> ..."] [MODELS="2 3 4"] [SEEDS="0 1 2"] [FOREIGN=<mode>] bash submit_recipe.sh
#
# STAGE: probe = family probe (probe_base, U-Net); s1a = families skip, placement, nocov;
#   s1b = cac_curve, cac_curve_corefilm_off and family mix; s1c = cac_cnn, cac_cnn_corefilm_off;
#   arms = the arms named in ARMS. MODELS: rungs (2 conv autoencoder, 3 conv + attention, 4 U-Net).
# SUB (scoring tier and form):
#   p16       train, then score the screen: 4 shards into p16_sv, eval set evalset_p16.json,
#             REFS_FROM=$O/eval/eic_v1_2/baselines_p16 (the options of configs/base_on_version.sh
#             stage runs).
#   p256      score only, the confirm: 8 shards into p256_sv, eval set evalset.json,
#             REFS_FROM=$O/eval/eic_v1_2/baselines, SCORE_TIME=06:00:00 (the options of
#             configs/ablations/covariate_fields/submit_a12.sh).
#   native    SCORE_INPUT_DEPTH=native into p16_sv_native_depth (screen options).
#   native256 SCORE_INPUT_DEPTH=native into p256_sv_native_depth (p256 options), for the finalists
#             with fixed input depth; gate G3 reads it.
#   foreign   FOREIGN_COVARIATES=$FOREIGN into p16_foreign_$FOREIGN (screen options).
# Every call to slurm/minicandi/submit_arms.sh uses JOB_PREFIX="" (job name = run dir name +
# _s<seed>), runs root $O/runs_rcp, tables $O/tables/eic_v1_2, the ONION_EXCLUDE list of
# configs/base_on_version.sh, and TRAIN_TIME (TRAIN_TIME_OVERRIDE wins) 03:00:00 (08:00:00 with the average input or a CNN
# correction layer; recipe_arms.py list gives it). A run name that does not start with rcp_ is
# refused (exit 2). submit_arms.sh is idempotent (SCORED, DONE and live jobs are skipped) and
# appends to $O/runs_rcp/SUBMITTED.tsv under flock.
# Projected CPU-hours: the measured unit costs of plan fact 9 (Fir sacct, eic_v1_2 base runs),
# per job line that submit_arms.sh prints (skipped runs cost nothing): training 5.98 / 14.34 /
# 7.06 per run-seed; screen predict 8.45 / 8.40 / 10.78 over 4 shards and metrics 8.29 / 8.16 /
# 8.27; confirm predict 75.93 / 81.43 / 100.87 over 8 shards and metrics 11.71 / 11.46 / 13.26
# (conv autoencoder / conv + attention / U-Net).
set -u
set -o pipefail
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
O="${O:-/project/def-maxwl/mforooz/ONION}"
K="${K:?set K=<kit dir>}"
V=eic_v1_2
STAGE="${STAGE:?set STAGE=probe, s1a, s1b, s1c or arms}"
SUBT="${SUB:?set SUB=p16, p256, native, native256 or foreign}"
export DRY_RUN="${DRY_RUN:-1}"
export SEEDS="${SEEDS:-0 1 2}"
MODELS="${MODELS:-2 3 4}"
PY="${PYTHON:-python3}"
RUNS=$O/runs_rcp
T=$O/tables/$V
SUBMIT="$K/slurm/minicandi/submit_arms.sh"
RA="$K/tools/minicandi/recipe_arms.py"
for p in "$SUBMIT" "$RA" "$K/tools/minicandi/configs/base_on_version.sh"; do
  [ -f "$p" ] || { echo "missing $p (kit $K)" >&2; exit 2; }
done
case $STAGE in
  probe|s1a|s1b|s1c) [ -z "${ARMS:-}" ] || { echo "ARMS goes with STAGE=arms" >&2; exit 2; }
                     SEL=(--stage "$STAGE") ;;
  arms) [ -n "${ARMS:-}" ] || { echo "STAGE=arms needs ARMS" >&2; exit 2; }
        SEL=(--arms "$ARMS") ;;
  *) echo "STAGE must be probe, s1a, s1b, s1c or arms" >&2; exit 2 ;;
esac

unset FOREIGN_COVARIATES FOREIGN_SEED SCORE_INPUT_DEPTH SCORE_ONLY SCORE_TIME
export SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 \
  SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 KEEP_PRED_STORE=0
case $SUBT in
  p16|native|foreign) TIER=screen; EV=$O/eval/$V/evalset_p16.json; B=$O/eval/$V/baselines_p16
                      export SCORE_SHARDS=4 ;;
  p256|native256) TIER=confirm; EV=$O/eval/$V/evalset.json; B=$O/eval/$V/baselines
                  export SCORE_SHARDS=8 SCORE_ONLY=1 SCORE_TIME=06:00:00 ;;
  *) echo "SUB must be p16, p256, native, native256 or foreign" >&2; exit 2 ;;
esac
case $SUBT in
  p16) SD=p16_sv ;;
  p256) SD=p256_sv ;;
  native) SD=p16_sv_native_depth; export SCORE_INPUT_DEPTH=native ;;
  native256) SD=p256_sv_native_depth; export SCORE_INPUT_DEPTH=native ;;
  foreign) [ -n "${FOREIGN:-}" ] || { echo "SUB=foreign needs FOREIGN=<mode>" >&2; exit 2; }
           SD="p16_foreign_$FOREIGN"; export FOREIGN_COVARIATES="$FOREIGN" ;;
esac
[ -z "${FOREIGN:-}" ] || [ "$SUBT" = foreign ] || { echo "FOREIGN goes with SUB=foreign" >&2; exit 2; }
export SCORE_SUBDIR=$SD REFS_FROM=$B
export PRED_STORE_ROOT="${PRED_STORE_ROOT:-/scratch/mforooz/onion/pred_store}"
EXC="$(sed -n 's/^ *export ONION_EXCLUDE=\([^ ]*\).*/\1/p' "$K/tools/minicandi/configs/base_on_version.sh" | head -n 1)"
[ -n "$EXC" ] || { echo "no ONION_EXCLUDE list in configs/base_on_version.sh" >&2; exit 2; }
export ONION_EXCLUDE="$EXC" ONION_STAGE=1 CLUSTER=fir NICE="${NICE:-100}" JOB_PREFIX=""

blocked=""
for p in "$T/measurements.tsv" "$EV" "$B/oracle_floor/seed0/oracle_floor.json"; do
  [ -e "$p" ] || blocked="$blocked; missing $p"
done
for n in marginal_nb position_average depth_only_scaling pseudoreplicate_floor; do
  [ -e "$B/$n/seed0/SCORED" ] || blocked="$blocked; $B/$n not SCORED"
done
if [ -n "$blocked" ]; then
  echo "BLOCKED${blocked}"
  [ "$DRY_RUN" = 1 ] || { echo "refused: real submission needs every precondition" >&2; exit 3; }
fi

LIST="$("$PY" "$RA" list "${SEL[@]}" --models "$MODELS")" || { echo "recipe_arms.py list failed" >&2; exit 2; }
[ -n "$LIST" ] || { echo "no arm-models for STAGE=$STAGE MODELS=$MODELS ARMS=${ARMS:-}" >&2; exit 2; }
OUTF="$(mktemp)"; trap 'rm -f "$OUTF"' EXIT
n_fail=0
while IFS=$'\t' read -r r arm run cfg tt; do
  case "$run" in rcp_*) ;; *) echo "refused: run name $run does not start with rcp_" >&2; exit 2 ;; esac
  c="$K/$cfg"
  m=$("$PY" -c "import json,os,sys;d=json.load(open(sys.argv[1]));print(' '.join(d[k] for k in ('average_h5',) if d.get(k) and not os.path.exists(d[k])))" "$c")
  if [ -n "$m" ]; then
    echo "BLOCKED $run: missing $m"
    [ "$DRY_RUN" = 1 ] || continue
  fi
  echo "## $run ($arm, rung $r) -> $SD"
  TRAIN_TIME=${TRAIN_TIME_OVERRIDE:-$tt} bash "$SUBMIT" "$K" "$T" "$EV" "$RUNS" "$r" "$run=$c" < /dev/null 2>&1 | { grep -v "^$" || true; } | tee -a "$OUTF"
  rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    n_fail=$((n_fail + 1)); echo "FAILED submit_arms.sh exit $rc: $run"
    [ "$DRY_RUN" = 1 ] || { echo "refused: stopping at the first failed submission" >&2; exit "$rc"; }
  fi
done <<< "$LIST"

"$PY" - "$OUTF" "$TIER" "$SCORE_SHARDS" "$STAGE" "$SUBT" "$DRY_RUN" "$n_fail" <<'EOF'
import re, sys
out, tier, nsh, stage, sub, dry, nfail = sys.argv[1], sys.argv[2], int(sys.argv[3]), *sys.argv[4:]
TRAIN = {"convae": 5.98, "convattn": 14.34, "unet": 7.06}
PRED = {"screen": {"convae": 8.45, "convattn": 8.40, "unet": 10.78},
        "confirm": {"convae": 75.93, "convattn": 81.43, "unet": 100.87}}
SHARDS = {"screen": 4, "confirm": 8}
MET = {"screen": {"convae": 8.29, "convattn": 8.16, "unet": 8.27},
       "confirm": {"convae": 11.71, "convattn": 11.46, "unet": 13.26}}
n = {"train": 0, "predict": 0, "metrics": 0}
cpu = 0.0
for ln in open(out, encoding="utf-8"):
    m = re.match(r"^(?:<[^>]*>|\d+) (rcp_\S+)_(convae|convattn|unet) \d+ (train|predict|metrics) \S+$", ln.strip())
    if not m:
        continue
    model, phase = m.group(2), m.group(3)
    n[phase] += 1
    cpu += (TRAIN[model] if phase == "train" else
            PRED[tier][model] / SHARDS[tier] if phase == "predict" else MET[tier][model])
if int(nfail):
    print(f"recipe: {nfail} submit_arms.sh calls FAILED (see FAILED lines)")
print(f"recipe STAGE={stage} SUB={sub} ({tier}): jobs {sum(n.values())} (train {n['train']}, "
      f"predict {n['predict']}, metrics {n['metrics']}); projected {cpu:.1f} CPU-h; DRY_RUN={dry}")
EOF
