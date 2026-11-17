#!/bin/bash
# Close-out D2: second experiments and cross-lab pairs. Run on the laptop; every job is submitted
# over ssh. DRY RUN BY DEFAULT: DRY_RUN=0 submits. Item file:
# cruxvault/INSTRUCTIONS/peeling_the_onion/closeout/closeout D2 second experiments and cross-lab pairs.md
#
#   bash slurm/onion_data/closeout_d2_submit.sh <stage> [nibi|fir]
#
# Stages (in this order; each one skips a job whose output exists or whose name is queued/running):
#   prep       1 job, 4 CPU x 3 h = 12 CPU-h at most: add-on tables <BASE>x (second experiments
#              with lab, antibody, donor, platform filled), check, add-on eval set, layer pair file,
#              pair smoke of every pair (slurm/onion_data/closeout_d2_prep.sh). Writes under $D only.
#   layers     1 job, 8 CPU x 12 h = 96 CPU-h at most (E7a measured about 50): layers.py run on the
#              add-on tables, all pairs (train and val) -> $D/layers/layers.tsv
#   baselines  5 jobs, 2 CPU x 14 h = 28 CPU-h each at most, 140 in all: marginal_nb,
#              position_average, depth_only_scaling, pseudoreplicate_floor (score.sh --addon) and the
#              retrieval baseline (baseline_retrieval.py --addon) -> $D/eval/<BASE>x/baselines/<name>/seed0
#   models     9 jobs (3 architectures x 3 seeds; RUNS overrides), 2 CPU x 14 h = 28 CPU-h each at
#              most, 252 in all: score.sh <EVX> <seed dir>/ox run <seed dir> --addon. A run must be
#              trained on <BASE> (the scorer refuses another tables_sha256). A seed dir that is
#              absent on the cluster is listed and skipped (eic_v1_2 runs live on Fir only today).
#   build      the second-experiment rows not built yet (ROWS; default the 17 DNase-seq rows), with
#              kit KX4 and ONION_TIME=3-00:00:00. Writes /scratch/mforooz/onion/{fastq,work,
#              measurements}: refused unless BUILD_OK=1 (the foreman, after the PI's go and after the
#              four archived scratch folders are removed: a work dir written before the removal
#              would be removed with them). Allocated cost up to 17 x 72 h x 16 CPU = 19,584 CPU-h.
#
# Names: O=/project/def-maxwl/mforooz/ONION, D=$O/CO-D2 (this item's root), K=$O/../onion_kit/$KIT_ID,
# BASE=eic_v1_2 (the filled tables version; PI ruling 2026-11-17: the corrected tables are the default).
set -euo pipefail
STAGE="${1:?usage: closeout_d2_submit.sh prep|layers|baselines|models|build [nibi|fir]}"
WHERE="${2:-nibi}"
DRY="${DRY_RUN:-1}"
KIT_ID="${KIT_ID:-COD2b}"
BASE="${BASE:-eic_v1_2}"
BASE_EVALSET_SHA="${BASE_EVALSET_SHA:-26f9e32f7cfd49f33b841925ff88901ba4b91c2df438ecc840e4074499f31c76}"  # Fir eval/eic_v1_2/evalset.json
O=/project/def-maxwl/mforooz/ONION
D="${D:-$O/CO-D2}"
K=/project/def-maxwl/mforooz/onion_kit/$KIT_ID
TX=$D/tables/${BASE}x
EVX=$D/eval/${BASE}x/evalset.json
RUNS_ROOT="${RUNS_ROOT:-$O/runs_fir}"
RUNS="${RUNS:-${BASE}_rung2_base_heldout ${BASE}_rung3_base_heldout ${BASE}_rung4_base_heldout}"
SEEDS="${SEEDS:-0 1 2}"
ROWS="${ROWS:-2015,2023,2028,2039,2040,2042,2049,2052,2067,2072,2075,2076,2077,2078,2079,2081,2084}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
SB="module load python/3.10.13 >/dev/null 2>&1; sbatch --parsable --account=def-maxwl --gres=none"
[ "$WHERE" = fir ] && SB+=" --exclude=fc10713,fc10715"
SCORE="--time=14:00:00 --cpus-per-task=2 --mem=20000M"   # E2 timing job 4448330: 11.7 h, RSS 12.3 GB, full eval set

remote() {   # run one command on the cluster, or print it
  if [ "$DRY" = 1 ]; then echo "[dry-run $WHERE] $1"; else "${SSH[@]}" "$WHERE" "$1"; fi
}
# 0 (skip) when <file> exists or a job named <name> is queued or running; read-only, also in a dry run
done_or_queued() {
  "${SSH[@]}" "$WHERE" "test -e $1 || [ -n \"\$(squeue -h -u \$USER -n $2)\" ]" 2>/dev/null
}
need() {     # stop unless every path exists on the cluster (read-only, also in a dry run)
  for p in "$@"; do
    "${SSH[@]}" "$WHERE" "test -e $p" || { echo "missing on $WHERE: $p" >&2; exit 1; }
  done
}

need "$K/.kit_stamp" "$O/tables/$BASE/measurements.tsv"
echo "kit $K: $("${SSH[@]}" "$WHERE" "head -1 $K/.kit_stamp")"
case "$STAGE" in
prep)
  need "$O/catalog/encode/experiments.tsv" "$O/eic/other_experiment_ledger.tsv" \
       "$O/eic/other_experiment_build_list.tsv" "$O/eval/strata/masks.h5"
  if done_or_queued "$D/eval/${BASE}x/evalset_layers.json" co_d2_prep; then echo "skip prep: done or queued"; exit 0; fi
  remote "mkdir -p $D/logs && $SB --job-name=co_d2_prep --time=03:00:00 --cpus-per-task=4 --mem=32000M --output=$D/logs/%x_%j.out $K/slurm/onion_data/closeout_d2_prep.sh $K $D $BASE $BASE_EVALSET_SHA"
  ;;
layers)
  need "$TX/pairs.tsv" "$D/eval/${BASE}x/evalset_layers.json"
  if done_or_queued "$D/layers/layers.tsv" co_d2_layers; then echo "skip layers: done or queued"; exit 0; fi
  remote "mkdir -p $D/layers && $SB --job-name=co_d2_layers --time=12:00:00 --cpus-per-task=8 --mem=32000M --output=$D/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/layers.py run --tables $TX --evalset $D/eval/${BASE}x/evalset_layers.json --strata $O/eval/strata/masks.h5 --out $D/layers/layers.tsv"
  ;;
baselines)
  need "$EVX"
  for b in marginal_nb position_average depth_only_scaling pseudoreplicate_floor; do
    out=$D/eval/${BASE}x/baselines/$b/seed0
    if done_or_queued "$out/scores.json" "co_d2_$b"; then echo "skip $b: done or queued"; continue; fi
    remote "mkdir -p $out && $SB --job-name=co_d2_$b $SCORE --output=$D/logs/%x_%j.out $K/slurm/onion_eval/score.sh $K $EVX $out baseline $b --addon"
  done
  out=$D/eval/${BASE}x/baselines/retrieval/seed0
  if done_or_queued "$out/scores.json" co_d2_retrieval; then echo "skip retrieval: done or queued"; else
    remote "mkdir -p $out && $SB --job-name=co_d2_retrieval $SCORE --output=$D/logs/%x_%j.out $K/slurm/onion_kit/run.sh $K tools/onion_eval/baseline_retrieval.py run --tables $TX --evalset $EVX --out $out --addon"
  fi
  ;;
models)
  need "$EVX"
  for run in $RUNS; do
    for s in $SEEDS; do
      sd=$RUNS_ROOT/$run/seed$s
      if ! "${SSH[@]}" "$WHERE" "test -f $sd/ckpt.pt && test -f $sd/DONE"; then
        echo "absent on $WHERE (copy it first): $sd"; continue
      fi
      if done_or_queued "$sd/ox/scores.json" "co_d2_${run}_s$s"; then echo "skip $run seed$s: done or queued"; continue; fi
      remote "mkdir -p $sd/ox && $SB --job-name=co_d2_${run}_s$s $SCORE --output=$D/logs/%x_%j.out $K/slurm/onion_eval/score.sh $K $EVX $sd/ox run $sd --addon --device cpu"
    done
  done
  ;;
build)
  KX4=/project/def-maxwl/mforooz/onion_kit/KX4
  BX=$O/eic/other_experiment_build_list.tsv
  if [ "${BUILD_OK:-0}" != 1 ]; then
    echo "build refused: it writes /scratch/mforooz/onion/{fastq,work}; set BUILD_OK=1 only after the PI's go"
    echo "and after the archived scratch folders are removed. It would run:"
    echo "  ONION_TIME=3-00:00:00 bash $KX4/slurm/onion_data/build_row.sh submit $KX4 $BX 5 50 2 $ROWS"
    exit 0
  fi
  remote "ONION_TIME=3-00:00:00 bash $KX4/slurm/onion_data/build_row.sh submit $KX4 $BX 5 50 2 $ROWS"
  ;;
*) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
