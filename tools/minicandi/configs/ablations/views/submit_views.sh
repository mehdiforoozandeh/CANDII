#!/bin/bash
# Close-out A3: the counterfactual-view ablation on the conv autoencoder and conv + attention, and
# the same-view score pass (p16_sv) of the five U-Net view arms that exist. Run on a Fir login node.
# DRY RUN BY DEFAULT: it prints the sbatch lines and submits nothing.
#
#   bash submit_views.sh            # dry run
#   DRY_RUN=0 bash submit_views.sh  # real submission -- the foreman's call only
#
# Part 1 (train + score): ablations/views/rung{2,3}/<arm>.json, 3 seeds, through submit_all.sh
#   (run dirs runs_fir/abl_views_<arm>_<convae|convattn>/seed<s>; 5 arms x 2 architectures x 3
#   seeds = 30 runs; per run 1 training + 4 predict + 1 metrics job). Reference: the existing
#   rung{2,3}_base_heldout (all views), already scored in p16_sv.
# Part 2 (score only): runs_fir/rung4_views_<arm>/seed{0,1,2} are trained and scored in p16 only
#   (no same-view pass, so no correction-only term). This scores them into p16_sv with the options
#   of runs_fir/score_final.sh (15 seeds; per seed 4 predict + 1 metrics job). The config is the
#   run's own seed0/run.json, as score_final.sh does for rung*_base_heldout.
# Idempotent through submit_arms.sh (a SCORED dir, or a queued job of the same name, is skipped).
set -u
O=/project/def-maxwl/mforooz/ONION
export K="${K:-/project/def-maxwl/mforooz/onion_kit/COA3}"
export DRY_RUN="${DRY_RUN:-1}"
[ -f "$K/tools/minicandi/configs/ablations/views/rung2/base_only.json" ] || { echo "kit $K has no view configs" >&2; exit 2; }

echo "## part 1: conv autoencoder and conv + attention, 5 view arms, train + score into p16_sv"
ABLATIONS=views bash "$K/tools/minicandi/configs/ablations/submit_all.sh"

echo "## part 2: U-Net view arms, score only into p16_sv"
(
  module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
  export ONION_EXCLUDE=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649
  export ONION_STAGE=1 CLUSTER=fir NICE=100 SEEDS="0 1 2"
  export SCORE_ONLY=1 SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SUBDIR=p16_sv SCORE_SHARDS=4 SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 REFS_FROM=$O/eval/eic_v1/baselines_p16 KEEP_PRED_STORE=0
  cd "$O/runs_fir" || exit 2
  for d in rung4_views_base_only rung4_views_base_depth rung4_views_base_fastq rung4_views_base_alignment rung4_views_all_same_view_only; do
    [ -f "$d/seed0/run.json" ] || { echo "MISSING $d/seed0/run.json"; continue; }
    bash "$K/slurm/minicandi/submit_arms.sh" "$K" "$O/tables/eic_v1" "$O/eval/eic_v1/evalset_p16.json" "$O/runs_fir" 4 "$d=$d/seed0/run.json" 2>&1 | grep -v "^$"
  done
)
echo "DRY_RUN=$DRY_RUN"
