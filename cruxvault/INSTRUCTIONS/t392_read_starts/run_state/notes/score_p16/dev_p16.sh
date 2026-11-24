#!/bin/bash
# t392 C14a: the deviance pass of the read-start model on 16 regions. One metrics-only job per
# seed, reading the kept p16_sv prediction store (deviance changes the metrics phase only; the
# predict phase never sees it), writing <run>/seed<k>/p16_dev. ONION_SCORE_DEVIANCE=1 is the
# score.py switch (tools/onion_eval/score.py DEVIANCE_ENV). Settings otherwise as the p16_sv
# metrics job of submit_p16.sh. Reference deviance metrics job 5381562: 16 CPU, 45:14.
#   DEP_<k>=<headline metrics job id> SEEDS="0 1 2" [DRY_RUN=1] bash dev_p16.sh   (Fir login node)
# A DEP_<k> makes the job wait for that seed's headline metrics (afterok); unset = no dependency.
set -euo pipefail
KIT=/project/def-maxwl/mforooz/onion_kit/RS7
RUNS=/project/def-maxwl/mforooz/t392/runs
ARM=rs_halves_mix_avg_pearson_unet
EV=/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json
REFS=/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/baselines_p16
PS_ROOT=/scratch/mforooz/rs392/pred_store
EXCL=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649
DTIME="${DTIME:-01:30:00}"
DRY="${DRY_RUN:-0}"
for S in ${SEEDS:?set SEEDS}; do
  RUN=$RUNS/$ARM/seed$S
  OUT=$RUN/p16_dev
  STORE=$PS_ROOT/runs/$ARM/same_view/seed$S/p16_sv
  [ -f "$OUT/SCORED" ] && { echo "seed $S: $OUT/SCORED exists, skip"; continue; }
  mkdir -p "$OUT"
  depvar="DEP_$S"; DEP=()
  [ -n "${!depvar:-}" ] && DEP=("--dependency=afterok:${!depvar}" --kill-on-invalid-dep=yes)
  CMD=(sbatch --account=def-maxwl --parsable "--exclude=$EXCL" "--job-name=rs392_dev_s$S"
       "--time=$DTIME" --cpus-per-task=16 --mem=64000M --gres=none
       "--output=$OUT/metrics_%j.out" ${DEP[@]+"${DEP[@]}"} --export=ALL
       "$KIT/slurm/onion_eval/score.sh" "$KIT" "$EV" "$OUT" run "$RUN")
  ENVS=(ONION_SCORE_PHASE=metrics "ONION_PRED_STORE=$STORE" "ONION_REFS_FROM=$REFS"
        ONION_KEEP_PRED_STORE=1 ONION_SCORE_SAME_VIEW=1 ONION_SCORE_DEVIANCE=1)
  if [ "$DRY" = 1 ]; then echo "${ENVS[*]} ${CMD[*]}"; else
    id=$(env "${ENVS[@]}" "${CMD[@]}"); echo "$id rs392_dev_s$S $OUT"; fi
done
