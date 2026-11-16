#!/bin/bash
# Ten named ablations of miniCANDI (chunks AB and AB2; cov_injection added last): train and score every arm config under
# $K/tools/minicandi/configs/ablations/<ablation>/rung{2,3,4}/<arm>.json, 3 seeds, on Fir CPU.
# Run on a Fir login node. DRY RUN BY DEFAULT: it prints the sbatch lines and submits nothing.
#
#   bash submit_all.sh            # dry run (DRY_RUN=1)
#   DRY_RUN=0 bash submit_all.sh  # real submission -- the foreman's call only
#   ABLATIONS="control_input dna_off" bash submit_all.sh   # a subset
#
# Run directory: $O/runs_fir/abl_<ablation>_<arm>_<convae|convattn|unet>/seed<s>.
# The reference is the existing base run rung{2,3,4}_base_heldout (no new run).
# Scoring: the same options as runs_fir/score_final.sh (same-view pass into p16_sv, 4 CPU predict
# shards, 16-CPU metrics job), but with training (no SCORE_ONLY).
# Training time: 03:00:00 for every arm (the default limit of 5,856 s is too tight: the conv
# autoencoder base seed 2 trained for 5,691 s, Fir job 4246989); 06:00:00 for two_step (two
# forward legs, about 2 x the FLOPs per window) and average_input (an unmeasured extra read per
# window; the unstaged rung-1 read took 4.5 x the base time); 04:00:00 for loss_nb_crps (1.2 to
# 2 x slower per window in the smokes).
# An arm whose input file (dna_embedding_h5, average_h5) is missing on this cluster is skipped
# and printed as BLOCKED.
set -u
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
O=/project/def-maxwl/mforooz/ONION
K="${K:-/project/def-maxwl/mforooz/onion_kit/CM}"
A="$K/tools/minicandi/configs/ablations"
export DRY_RUN="${DRY_RUN:-1}"
export ONION_EXCLUDE=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649
export ONION_STAGE=1 CLUSTER=fir NICE=100 SEEDS="${SEEDS:-0 1 2}"
export SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SUBDIR=p16_sv SCORE_SHARDS=4 SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 REFS_FROM=$O/eval/eic_v1/baselines_p16 KEEP_PRED_STORE=0
[ -d "$A" ] || { echo "no ablation configs in kit $K" >&2; exit 2; }
declare -A ARCH=([2]=convae [3]=convattn [4]=unet)
ABL="${ABLATIONS:-control_input predict_control average_input two_step dna_source dna_off cov_embedding depth_input losses cov_injection}"
n=0
for ab in $ABL; do
  for r in 2 3 4; do
    for c in "$A/$ab/rung$r"/*.json; do
      [ -e "$c" ] || continue
      arm=$(basename "$c" .json)
      case $ab/$arm in
        two_step/*|average_input/*) export TRAIN_TIME=06:00:00 ;;
        losses/loss_nb_crps) export TRAIN_TIME=04:00:00 ;;
        *) export TRAIN_TIME=03:00:00 ;;
      esac
      miss=$(python3 -c "import json,os,sys;d=json.load(open(sys.argv[1]));print(' '.join(d[k] for k in ('dna_embedding_h5','average_h5') if d.get(k) and not os.path.exists(d[k])))" "$c")
      if [ -n "$miss" ]; then echo "BLOCKED abl_${ab}_${arm}_${ARCH[$r]}: missing $miss"; continue; fi
      bash $K/slurm/minicandi/submit_arms.sh $K $O/tables/eic_v1 $O/eval/eic_v1/evalset_p16.json $O/runs_fir $r "abl_${ab}_${arm}_${ARCH[$r]}=$c" 2>&1 | grep -v "^$"
      n=$((n+1))
    done
  done
done
echo "arms x architectures submitted (or printed): $n; DRY_RUN=$DRY_RUN"
