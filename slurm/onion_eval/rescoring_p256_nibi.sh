#!/bin/bash
# Close-out: 256-region rescoring (scorer sub-directory p256_sv) of the 7 decisive ablation arms
# on NIBI, from the Globus copy of the Fir runs in /project/def-maxwl/mforooz/ONION/runs_fir.
# PI rulings 2026-11-17: move to Nibi; about 2,225 CPU-hours approved. It duplicates the Fir jobs
# 4535347 to 4535923 (the PI cancels those). Needs writable /scratch (prediction stores).
#
#   ssh -o BatchMode=yes nibi 'bash -s' < slurm/onion_eval/rescoring_p256_nibi.sh             # dry run
#   ssh -o BatchMode=yes nibi 'DRY_RUN=0 bash -s' < slurm/onion_eval/rescoring_p256_nibi.sh   # submit, all jobs HELD
#   ssh -o BatchMode=yes nibi 'ARMS="scrambled views_base" DRY_RUN=0 bash -s' < ...           # only some arm groups
#
# Every job is submitted on hold (HOLD=1). Release run-seeds in batches (scontrol release <ids>),
# so that the live prediction stores (about 55 GB each) fit on /scratch.
# Env: KIT_ID (default CO8), ARMS (default: all 7 groups), DRY_RUN (default 1).
set -uo pipefail
O=/project/def-maxwl/mforooz/ONION; K=/project/def-maxwl/mforooz/onion_kit/${KIT_ID:-CO8}
ARMS="${ARMS:-scrambled views_base two_step_native two_step_fixed fixed_depth average_input impute_x4}"
[ -f "$K/.kit_stamp" ] || { echo "REFUSED: no kit $K"; exit 3; }
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES ONION_EXCLUDE
export ONION_STAGE=1 CLUSTER=nibi NICE=300 SEEDS="0 1 2" SCORE_ONLY=1 DRY_RUN="${DRY_RUN:-1}" \
  SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SUBDIR=p256_sv SCORE_SHARDS=8 SCORE_PRED_DEVICE=cpu \
  SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 SCORE_PRED_TIME=06:00:00 \
  SCORE_CPUS=16 SCORE_MEM=64000 SCORE_TIME=06:00:00 REFS_FROM=$O/eval/eic_v1/baselines \
  KEEP_PRED_STORE=0 PRED_STORE_ROOT=/scratch/mforooz/onion/pred_store HOLD=1
group() { case "$1" in
  scrambled)       echo "2:abl_scrambled_covariates_scrambled_covariates_convae 3:abl_scrambled_covariates_scrambled_covariates_convattn 4:abl_scrambled_covariates_scrambled_covariates_unet";;
  views_base)      echo "2:abl_views_base_only_convae 3:abl_views_base_only_convattn 4:rung4_views_base_only";;
  two_step_native) echo "2:abl_two_step_two_step_native_convae 3:abl_two_step_two_step_native_convattn 4:abl_two_step_two_step_native_unet";;
  two_step_fixed)  echo "2:abl_two_step_two_step_fixed_convae 3:abl_two_step_two_step_fixed_convattn 4:abl_two_step_two_step_fixed_unet";;
  fixed_depth)     echo "2:abl_depth_input_input_depth_fixed_convae 3:abl_depth_input_input_depth_fixed_convattn 4:abl_depth_input_input_depth_fixed_unet";;
  average_input)   echo "2:abl_average_input_average_input_convae 3:abl_average_input_average_input_convattn 4:abl_average_input_average_input_unet";;
  impute_x4)       echo "2:abl_losses_loss_impute_x4_convae 3:abl_losses_loss_impute_x4_convattn 4:abl_losses_loss_impute_x4_unet";;
  *) echo "unknown arm group $1" >&2; exit 2;; esac; }
for g in $ARMS; do
  for spec in $(group "$g"); do
    r=${spec%%:*}; d=${spec#*:}
    [ -f "$O/runs_fir/$d/seed0/run.json" ] || { echo "MISSING $O/runs_fir/$d/seed0/run.json"; continue; }
    echo "=== $g: rung $r $d"
    bash $K/slurm/minicandi/submit_arms.sh $K $O/tables/eic_v1 $O/eval/eic_v1/evalset.json $O/runs_fir $r "$d=$O/runs_fir/$d/seed0/run.json"
  done
done
