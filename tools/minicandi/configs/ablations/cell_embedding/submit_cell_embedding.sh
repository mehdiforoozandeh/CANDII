#!/bin/bash
# Close-out item A6: the learned cell-type embedding arm (configs rung{2,3,4}/cell_embedding.json:
# the base recipe plus cell_embedding = true, cell_state_dropout = 0.1, n_cells = 51) on the
# conv autoencoder, conv + attention and U-Net, 3 seeds, 80,000 windows, tables eic_v1_2, scored on
# all 256 regions (evalset.json) into <run>/seed<s>/p256_sv. Specification: the item file
# "closeout A6 learned cell-type embedding.md", section "Specification as worked".
# Run on a login node. DRY RUN BY DEFAULT: it prints the checks and the sbatch lines, submits nothing.
#
#   bash submit_cell_embedding.sh                   # MODE=train: 9 trainings + their scoring
#   MODE=audit bash submit_cell_embedding.sh        # after the 9 trainings: one CPU job, the
#                                                   # zero-shot trap checks (cell_embedding_audit.py runs)
#   DRY_RUN=0 bash submit_cell_embedding.sh         # real submission -- the foreman's call only
#
# CLUSTER=fir|nibi (default: from the host name). Runs root: Fir runs_fir, Nibi runs. The reference
# runs eic_v1_2_rung{2,3,4}_base_heldout exist on Fir only, so the arm belongs on Fir (one cluster per
# comparison). Run names: <runs root>/eic_v1_2_abl_cell_embedding_<convae|convattn|unet>/seed<s>.
# Scoring: the options of slurm/onion_eval/e1_pair_check_p256.sh (256 pair regions, same-view pass,
# 8 CPU predict shards, inference batch 8, 16-CPU metrics job) with REFS_FROM = eval/<V>/baselines.
# Training: --time 03:00:00 (submit_all.sh; the eic_v1_2 base runs took 0.40 to 1.44 h on Fir, 12 CPUs).
#
# Preflight (both modes): the tables, the eval set, the 256-region baselines (4 SCORED + oracle_floor),
# the kit's configs, and n_cells of every config = the training cells of the tables. A real
# submission refuses when one is missing (exit 3). A dry run with a missing eval set or baselines
# prints the training lines only. The reference scorings <base run>/seed<s>/p256_sv/SCORED are not a
# precondition of submission; they are a precondition of the reading and are listed.
set -u
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
O=/project/def-maxwl/mforooz/ONION
K="${K:-/project/def-maxwl/mforooz/onion_kit/COA6}"
V="${V:-eic_v1_2}"
T=$O/tables/$V
EV=$O/eval/$V/evalset.json
B=$O/eval/$V/baselines
A=$K/tools/minicandi/configs/ablations/cell_embedding
MODE="${MODE:-train}"
export DRY_RUN="${DRY_RUN:-1}"
CL="${CLUSTER:-${CC_CLUSTER:-}}"
if [ -z "$CL" ]; then case "$(hostname)" in *nibi*) CL=nibi ;; *fir*) CL=fir ;; esac; fi
case "$CL" in
  fir) ROOT=$O/runs_fir
       export ONION_EXCLUDE=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649 ;;
  nibi) ROOT=$O/runs; unset ONION_EXCLUDE ;;
  *) echo "CLUSTER must be fir or nibi" >&2; exit 2 ;;
esac
declare -A ARCH=([2]=convae [3]=convattn [4]=unet)
SEED_LIST="${SEEDS:-0 1 2}"
echo "A6 cell embedding: MODE=$MODE CLUSTER=$CL DRY_RUN=$DRY_RUN kit=$K tables=$V runs root=$ROOT"

# ---------------------------------------------------------------- preflight
miss=()      # blocks scoring (and any real submission)
for p in "$T/measurements.tsv" "$EV" "$B/oracle_floor/seed0/oracle_floor.json"; do
  [ -e "$p" ] || miss+=("$p")
done
for b in marginal_nb position_average depth_only_scaling pseudoreplicate_floor; do
  [ -e "$B/$b/seed0/SCORED" ] || miss+=("$B/$b/seed0/SCORED")
done
[ -e "$T/measurements.tsv" ] || { echo "MISSING $T/measurements.tsv: nothing to do" >&2; exit 3; }
NC=$(python3 -c "import csv,sys;print(len({r['cell'] for r in csv.DictReader(open(sys.argv[1]),delimiter='\t') if r['in_eic']=='train' and r['cell']}))" "$T/measurements.tsv")
for r in 2 3 4; do
  c=$A/rung$r/cell_embedding.json
  [ -f "$c" ] || { echo "MISSING config $c (push a kit that holds it)" >&2; exit 3; }
  n=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('n_cells'))" "$c")
  [ "$n" = "$NC" ] || { echo "REFUSED $c: n_cells $n, tables $V have $NC training cells" >&2; exit 3; }
done
echo "OK n_cells = $NC in all 3 configs (training cells of $V)"
for p in ${miss[@]+"${miss[@]}"}; do echo "MISSING $p"; done
nref=0
for r in 2 3 4; do for s in $SEED_LIST; do
  p=$ROOT/${V}_rung${r}_base_heldout/seed$s/p256_sv/SCORED
  [ -e "$p" ] || { echo "READING NEEDS $p (reference scoring, not submitted here)"; nref=$((nref + 1)); }
done; done
case "$(dirname "${PRED_STORE_ROOT:-/scratch/mforooz/onion/pred_store}")" in /scratch*)
  echo "NOTE prediction stores go to ${PRED_STORE_ROOT:-/scratch/mforooz/onion/pred_store} (~55 GB, ~1,700 files per run while scoring); check that /scratch on $CL takes writes" ;;
esac
if [ "${#miss[@]}" -gt 0 ] && [ "$DRY_RUN" != 1 ]; then
  echo "REFUSED: ${#miss[@]} prerequisites missing (above)" >&2; exit 3
fi

case "$MODE" in
train)
  export ONION_STAGE=1 CLUSTER=$CL NICE="${NICE:-100}" SEEDS="$SEED_LIST" TRAIN_TIME=03:00:00
  if [ "${#miss[@]}" -eq 0 ]; then
    EVS=$EV
    export SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SUBDIR=p256_sv SCORE_SHARDS=8 \
      SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 \
      SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 SCORE_TIME=06:00:00 \
      REFS_FROM=$B KEEP_PRED_STORE=0
  else
    EVS=""
    echo "DRY RUN WITHOUT SCORING: the eval set or the baselines are missing; training lines only"
    unset SCORE_SAME_VIEW SCORE_PHASES SCORE_SUBDIR REFS_FROM
  fi
  n=0
  for r in 2 3 4; do
    bash $K/slurm/minicandi/submit_arms.sh $K $T "$EVS" $ROOT $r \
      "${V}_abl_cell_embedding_${ARCH[$r]}=$A/rung$r/cell_embedding.json" 2>&1 | grep -v "^$"
    n=$((n + 1))
  done
  echo "architectures submitted (or printed): $n x seeds ($SEED_LIST); reference scorings missing: $nref; DRY_RUN=$DRY_RUN"
  ;;
audit)
  OUT="${OUT:-$O/CO-A6/audit}"
  RUNS=(); notdone=0
  for r in 2 3 4; do for s in $SEED_LIST; do
    d=$ROOT/${V}_abl_cell_embedding_${ARCH[$r]}/seed$s
    [ -e "$d/DONE" ] || { echo "NOT TRAINED $d"; notdone=$((notdone + 1)); }
    RUNS+=(--run "$d")
  done; done
  [ ! -e "$OUT/audit.json" ] || { echo "$OUT/audit.json exists; set OUT to a new path" >&2; exit 2; }
  if [ "$notdone" -gt 0 ] && [ "$DRY_RUN" != 1 ]; then echo "REFUSED: $notdone runs not trained" >&2; exit 3; fi
  BODY="set -eu; module load python/3.10.13; virtualenv --no-download \$SLURM_TMPDIR/venv; \
source \$SLURM_TMPDIR/venv/bin/activate; pip install --no-index --upgrade pip; \
pip install --no-index -r $K/requirements-fir.txt; \
pip install --no-index --find-links $K/wheels -r $K/requirements-pypi.txt; cd $K; \
export PYTHONPATH=$K/src:$K/tools PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2; \
python $K/tools/minicandi/cell_embedding_audit.py runs --tables $T --evalset $EV ${RUNS[*]} \
--json $OUT/audit.json"
  CMD=(sbatch --account=def-maxwl --job-name=a6_cell_embedding_audit --time=00:30:00
       --cpus-per-task=2 --mem=8000M --gres=none "--output=$OUT/audit_%j.out" --wrap "$BODY")
  if [ "$DRY_RUN" = 1 ]; then echo "DRY RUN"; printf '%q ' "${CMD[@]}"; echo
  else mkdir -p "$OUT" && "${CMD[@]}"; fi
  ;;
*) echo "MODE must be train or audit" >&2; exit 2 ;;
esac
