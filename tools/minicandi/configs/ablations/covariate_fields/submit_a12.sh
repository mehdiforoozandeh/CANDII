#!/bin/bash
# Close-out item A12 (minimum useful covariate set): ONE submit script for every step. Run on a
# login node of the cluster that holds the runs (Fir: runs_fir). DRY RUN BY DEFAULT: it prints the
# sbatch lines, the job count and the CPU-hours, and submits nothing.
#
#   STEP=ref    bash submit_a12.sh   # 256-region scoring of the full-set runs (needed by every step)
#   STEP=screen bash submit_a12.sh   # screening: full-set runs scored with ONE field foreign/neutral
#   STEP=lofo   bash submit_a12.sh   # confirmation: leave-one-field-out trainings + 256 scoring
#   STEP=groups bash submit_a12.sh   # control: grouped removal of correlated fields
#   STEP=curve  bash submit_a12.sh   # exclusion curve: cumulative removal, needs CFG with curve_k*
#   DRY_RUN=0 STEP=... bash submit_a12.sh   # real submission -- the foreman's call only
#
# Inputs (env; defaults in brackets):
#   CLUSTER [fir]: fir -> runs root $O/runs_fir; nibi -> $O/runs.
#   K [/project/def-maxwl/mforooz/onion_kit/COA12]: the kit (code and configs).
#   V [eic_v1_2]: tables version; T=$O/tables/$V, eval set $O/eval/$V/evalset.json (256 regions),
#     reference terms REFS_FROM [$O/eval/$V/baselines] (the 256-region baselines and oracle floor).
#   REF2 / REF3 / REF4 [${V}_rung{2,3,4}_base_heldout]: run dir names of the full-set runs of the
#     conv autoencoder / conv + attention / U-Net (the recipe that mix-and-match selects, trained on
#     $V, 3 seeds). STEP=ref and STEP=screen score these runs; no training.
#   CFG [$K/tools/minicandi/configs/ablations/covariate_fields]: arm configs made by
#     tools/minicandi/covariate_field_arms.py {lofo,groups,curve} --base <rung2> --base <rung3>
#     --base <rung4> --out <CFG>. The kit's folder is generated from configs/rung{2,3,4}/
#     base_heldout.json; for another recipe, generate a new folder (new kit) and pass CFG.
#   FIELDS [the 10 META columns other than log2_depth]: screened fields (STEP=screen).
#   SIDES ["out in"], SCREEN_KINDS ["keep_depth"] (add "neutral" for neutral values; read with the
#     caveat of score.py FOREIGN_COLUMNS_DEFINITION), FOREIGN_SEED [0]: STEP=screen.
#   SEEDS ["0 1 2"]; ARMS [all arms of the step in CFG]: a subset, e.g. "lofo_lab_id lofo_donor_id".
#   PRED_STORE_ROOT [/scratch/mforooz/onion/pred_store]: one store per scoring (~55 GB, ~1,700
#     files; the metrics job removes it). On Nibi /scratch refuses writes (2026-11-17): real
#     submission there is refused unless PRED_STORE_ROOT is set to a writable path.
#
# Run dirs: training arms $RUNS/a12_<arm>_<convae|convattn|unet>/seed<s>, scored into p256_sv;
# screening scores $RUNS/<REFr>/seed<s>/p256_foreign_<side>_<kind>.<field>. Read with
# tools/onion_eval/covariate_ranking.py (runs | screen). Idempotent through submit_arms.sh.
#
# Cost (CPU-hours, measured on Fir, read with sacct; arithmetic in the item file):
#   training per run (eic_v1_2 base recipes, jobs 4472547-4472600): conv autoencoder 5.99,
#     conv + attention 14.34, U-Net 7.06 -> one arm on 3 models x 3 seeds = 82.2
#   256-region scoring per run (8 predict + 1 metrics jobs, eic_v1 base runs, jobs 4492698 ff.):
#     conv autoencoder 30.03, conv + attention 25.43, U-Net 53.43 -> 3 models x 3 seeds = 326.7
set -u
O=/project/def-maxwl/mforooz/ONION
K="${K:-/project/def-maxwl/mforooz/onion_kit/COA12}"
V="${V:-eic_v1_2}"
CLUSTER="${CLUSTER:-fir}"
case "$CLUSTER" in fir) RUNS=$O/runs_fir ;; nibi) RUNS=$O/runs ;; *) echo "CLUSTER must be fir or nibi" >&2; exit 2 ;; esac
STEP="${STEP:?set STEP=ref, screen, lofo, groups or curve}"
export DRY_RUN="${DRY_RUN:-1}"
T=$O/tables/$V
EV=$O/eval/$V/evalset.json
B="${REFS_FROM:-$O/eval/$V/baselines}"
CFG="${CFG:-$K/tools/minicandi/configs/ablations/covariate_fields}"
SUBMIT="$K/slurm/minicandi/submit_arms.sh"
declare -A ARCH=([2]=convae [3]=convattn [4]=unet)
declare -A REF=([2]="${REF2:-${V}_rung2_base_heldout}" [3]="${REF3:-${V}_rung3_base_heldout}" [4]="${REF4:-${V}_rung4_base_heldout}")
declare -A TRAIN_CPUH=([2]=5.99 [3]=14.34 [4]=7.06)
declare -A SCORE_CPUH=([2]=30.03 [3]=25.43 [4]=53.43)
FIELDS="${FIELDS:-run_type_pe read_length dedup_on mapq_threshold pipeline_id pipeline_version_id lab_id platform_id antibody_id donor_id}"
SIDES="${SIDES:-out in}"
SCREEN_KINDS="${SCREEN_KINDS:-keep_depth}"
export SEEDS="${SEEDS:-0 1 2}"
read -r -a SEED_LIST <<< "$SEEDS"
NS=${#SEED_LIST[@]}

module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
# ---------------------------------------------------------------- preconditions
blocked=""
[ -f "$SUBMIT" ] || blocked="$blocked; no kit $K (push it: tools/onion_kit/push_kit.sh --cluster $CLUSTER COA12)"
[ -d "$T" ] || blocked="$blocked; no tables $T"
[ -f "$EV" ] || blocked="$blocked; no 256-region eval set $EV"
[ -e "$B/oracle_floor/seed0/oracle_floor.json" ] || blocked="$blocked; no 256-region references $B (4 baselines + oracle floor)"
for n in marginal_nb position_average depth_only_scaling pseudoreplicate_floor; do
  [ -e "$B/$n/seed0/SCORED" ] || { blocked="$blocked; $B/$n not SCORED"; break; }
done
PSR="${PRED_STORE_ROOT:-/scratch/mforooz/onion/pred_store}"
if [ "$CLUSTER" = nibi ] && [ -z "${PRED_STORE_ROOT:-}" ]; then
  blocked="$blocked; Nibi /scratch refuses writes: set PRED_STORE_ROOT to a writable path"
fi
case $STEP in ref|screen)
  for r in 2 3 4; do
    for s in "${SEED_LIST[@]}"; do
      [ -f "$RUNS/${REF[$r]}/seed$s/DONE" ] || { blocked="$blocked; full-set run $RUNS/${REF[$r]}/seed$s not DONE"; break; }
    done
  done ;;
lofo|groups|curve) ;;
*) echo "STEP must be ref, screen, lofo, groups or curve" >&2; exit 2 ;;
esac
if [ -n "$blocked" ]; then
  echo "BLOCKED${blocked}"
  [ "$DRY_RUN" = 1 ] || { echo "refused: real submission needs every precondition" >&2; exit 3; }
  echo "(dry run continues; lines below that need a missing path print as submit_arms.sh refusals)"
fi

export ONION_EXCLUDE="${ONION_EXCLUDE:-fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649}"
[ "$CLUSTER" = fir ] || export ONION_EXCLUDE=""
export ONION_STAGE=1 CLUSTER NICE="${NICE:-100}" TRAIN_TIME="${TRAIN_TIME:-03:00:00}"
export SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SHARDS=8 SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 \
  SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 \
  SCORE_TIME=06:00:00 REFS_FROM="$B" KEEP_PRED_STORE=0 PRED_STORE_ROOT="$PSR"
unset FOREIGN_COVARIATES FOREIGN_SEED

n_train=0; n_score=0; cpu_train=0; cpu_score=0
add() {   # <rung> <trainings per seed 0|1>: count one run x seeds
  local r="$1" t="$2"
  n_score=$((n_score + NS)); cpu_score=$(python3 -c "print($cpu_score + $NS * ${SCORE_CPUH[$r]})")
  if [ "$t" = 1 ]; then
    n_train=$((n_train + NS)); cpu_train=$(python3 -c "print($cpu_train + $NS * ${TRAIN_CPUH[$r]})")
  fi
}
run_submit() {   # run_submit [VAR=value ...] -- <rung> <name=config>; a dry run prints the lines
  local envs=()
  while [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  env ${envs[@]+"${envs[@]}"} bash "$SUBMIT" "$K" "$T" "$EV" "$RUNS" "$@" 2>&1 | grep -v "^$"
}

case $STEP in
ref)
  for r in 2 3 4; do
    run_submit SCORE_ONLY=1 SCORE_SUBDIR=p256_sv -- "$r" "${REF[$r]}=$RUNS/${REF[$r]}/seed0/run.json"
    add "$r" 0
  done ;;
screen)
  FOREIGN_SEED="${FOREIGN_SEED:-0}"
  for kind in $SCREEN_KINDS; do
    for side in $SIDES; do
      for f in $FIELDS; do
        mode="${side}_${kind}.${f}"
        echo "## screen $mode -> p256_foreign_$mode"
        for r in 2 3 4; do
          run_submit FOREIGN_COVARIATES="$mode" FOREIGN_SEED="$FOREIGN_SEED" \
            SCORE_SUBDIR="p256_foreign_$mode" -- "$r" "${REF[$r]}=$RUNS/${REF[$r]}/seed0/run.json"
          add "$r" 0
        done
      done
    done
  done ;;
lofo|groups|curve)
  case $STEP in lofo) pre=lofo_ ;; groups) pre=group_ ;; curve) pre=curve_k ;; esac
  [ -f "$CFG/arms.tsv" ] || { echo "no $CFG/arms.tsv (run covariate_field_arms.py $STEP)" >&2; exit 2; }
  for r in 2 3 4; do
    for c in "$CFG/rung$r/$pre"*.json; do
      [ -e "$c" ] || { echo "no $STEP arm configs in $CFG/rung$r"; continue; }
      arm=$(basename "$c" .json)
      if [ -n "${ARMS:-}" ] && ! printf '%s\n' $ARMS | grep -qxF "$arm"; then continue; fi
      run_submit SCORE_SUBDIR=p256_sv -- "$r" "a12_${arm}_${ARCH[$r]}=$c"
      add "$r" 1
    done
  done ;;
esac

python3 - "$STEP" "$n_train" "$cpu_train" "$n_score" "$cpu_score" "$DRY_RUN" <<'EOF'
import sys
step, nt, ct, nsc, cs, dry = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
print(f"A12 STEP={step}: {nt} trainings ({ct:.1f} CPU-h) + {nsc} 256-region scorings "
      f"({cs:.1f} CPU-h, {9 * nsc} jobs: 8 predict + 1 metrics each) = {ct + cs:.1f} CPU-h; "
      f"sbatch jobs {nt + 9 * nsc}; peak prediction-store files if all scorings run at once "
      f"~{1700 * nsc:,}; DRY_RUN={dry}")
EOF
