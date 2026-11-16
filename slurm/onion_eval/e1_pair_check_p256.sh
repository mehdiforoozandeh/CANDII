#!/bin/bash
# Close-out item E1: the 16-versus-256 pair-region check, as amended on 2026-11-16
# (cruxvault/results/pair_region_check/PAIR_CHECK_PREREGISTRATION.md, section "Amendment of 2026-11-16").
#
#   bash slurm/onion_eval/e1_pair_check_p256.sh score   # step 1: 12 runs x (8 predict + 1 metrics) jobs
#   bash slurm/onion_eval/e1_pair_check_p256.sh check   # step 2: one CPU job, after all 12 are SCORED
#
# DRY RUN BY DEFAULT: prints the sbatch lines and submits nothing. GO=1 submits. Run it on a Fir
# login node from any directory. The kit (below) must hold `pair_regions_check.py run-v3`.
#
# Step 1 scores conv autoencoder, conv + attention and U-Net base recipes and the U-Net with DNA-only
# skip connections (3 seeds each) on eval/eic_v1/evalset.json (all 256 pair regions) into
# <run>/seed<s>/p256_sv, with the same-view pass, on CPU with inference batch 8 (the device and batch
# of the stored p16_sv scores), reference terms from eval/eic_v1/baselines. It reuses
# slurm/minicandi/submit_arms.sh exactly as runs_fir/score_final.sh does for p16_sv; only the eval
# set, REFS_FROM, SCORE_SUBDIR and the times differ. A run whose p256_sv/SCORED exists is skipped.
# The prediction store of a run (~55 GB, ~1,700 files on /scratch) is removed by its metrics job.
#
# Step 2 refuses until all 12 p256_sv/SCORED exist; it writes $OUT/PAIR_REGIONS_V3.md (first line
# "pair regions: <p>", second line "16-region screening scores settled: yes|no (...)") and check_v3.tsv.
set -u
MODE="${1:-score}"
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
O=/project/def-maxwl/mforooz/ONION
# score mode: kit COE1 (submitted 2026-11-16, Fir jobs 4451177 to 4451301); check mode: kit COE1b,
# which holds the reviewed run-v3 (consistency gates, pinned seeds, 72 cells). KIT overrides both.
case "$MODE" in check) K="${KIT:-/project/def-maxwl/mforooz/onion_kit/COE1b}" ;;
  *) K="${KIT:-/project/def-maxwl/mforooz/onion_kit/COE1}" ;; esac
EV=$O/eval/eic_v1/evalset.json
B=$O/eval/eic_v1/baselines
OUT="${OUT:-$O/pair_check/e1_v3}"
ARMS=(rung2_base_heldout rung3_base_heldout rung4_base_heldout final_rung4_mech_skip_dna)
[ -f "$K/tools/onion_eval/pair_regions_check.py" ] && grep -q "def run_check_v3" "$K/tools/onion_eval/pair_regions_check.py" \
  || { echo "kit $K has no run_check_v3; push it first" >&2; exit 2; }
GO="${GO:-0}"

case "$MODE" in
score)
  export ONION_EXCLUDE=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649
  export ONION_STAGE=1 CLUSTER=fir NICE="${NICE:-300}" SEEDS="0 1 2"
  export SCORE_ONLY=1 SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SUBDIR=p256_sv SCORE_SHARDS=8 \
    SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 \
    SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 SCORE_TIME=06:00:00 \
    REFS_FROM=$B KEEP_PRED_STORE=0
  if [ "$GO" = 1 ]; then unset DRY_RUN; else export DRY_RUN=1; echo "DRY RUN (GO=1 submits)"; fi
  cd $O/runs_fir || exit 2
  for d in "${ARMS[@]}"; do
    r=$(python3 -c "import json;print(json.load(open('$d/seed0/run.json'))['config']['rung'])")
    case $d in
      final_*) c=$K/tools/minicandi/configs/final/rung$r/${d#final_rung${r}_}.json ;;
      *) c=$d/seed0/run.json ;;
    esac
    [ -e "$c" ] || c=${c%.json}_equal.json; [ -e "$c" ] || { echo "NOCONFIG $d $c"; exit 2; }
    bash $K/slurm/minicandi/submit_arms.sh $K $O/tables/eic_v1 $EV $O/runs_fir $r $d=$c 2>&1 | grep -v "^$"
  done
  ;;
check)
  miss=0
  for d in "${ARMS[@]}"; do for s in 0 1 2; do
    [ -f "$O/runs_fir/$d/seed$s/p256_sv/SCORED" ] || { echo "not scored: $d/seed$s/p256_sv"; miss=1; }
  done; done
  [ "$miss" = 0 ] || { echo "step 2 waits for every p256_sv/SCORED" >&2; exit 2; }
  [ ! -e "$OUT/PAIR_REGIONS_V3.md" ] || { echo "$OUT/PAIR_REGIONS_V3.md exists; set OUT to a new path" >&2; exit 2; }
  A=(); for d in "${ARMS[@]}"; do A+=(--arm "$O/runs_fir/$d"); done
  BODY="set -eu; module load python/3.10.13; virtualenv --no-download \$SLURM_TMPDIR/venv; \
source \$SLURM_TMPDIR/venv/bin/activate; pip install --no-index --upgrade pip; \
pip install --no-index -r $K/requirements-fir.txt; \
pip install --no-index --find-links $K/wheels -r $K/requirements-pypi.txt; cd $K; \
export PYTHONPATH=$K/src:$K/tools PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2; \
python $K/tools/onion_eval/pair_regions_check.py run-v3 ${A[*]} --sub p256_sv --screening-sub p16_sv \
--oracle $B/oracle_floor/seed0 --evalset $EV --out $OUT"
  CMD=(sbatch --account=def-maxwl --job-name=e1_pair_check_v3 --time=04:00:00 --cpus-per-task=4
       --mem=16000M --gres=none "--output=$OUT/check_%j.out" --wrap "$BODY")
  if [ "$GO" = 1 ]; then mkdir -p "$OUT" && "${CMD[@]}"; else echo "DRY RUN (GO=1 submits)"; printf '%q ' "${CMD[@]}"; echo; fi
  ;;
*) echo "usage: $0 score|check" >&2; exit 2 ;;
esac
