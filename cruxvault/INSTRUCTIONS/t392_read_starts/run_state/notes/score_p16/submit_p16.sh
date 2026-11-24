#!/bin/bash
# t392 chunk C13a: 16-region headline scoring of the read-start model, per seed, on Fir.
# Run on a Fir login node (copy at /project/def-maxwl/mforooz/t392/submit_p16.sh):
#   SEEDS="0" DRY_RUN=1 bash submit_p16.sh      # print every sbatch line, submit nothing
#   SEEDS="0" bash submit_p16.sh                # submit
# Per seed: 4 predict jobs (slurm/onion_eval/predict.sh, the reference's shape: CPU, 8 CPUs,
# 32000M, batch 8, same-view on), then one metrics job through slurm/minicandi/submit_arms.sh
# (METRICS_ONLY=1, PRED_JOBS = the 4 predict ids, so the metrics line, SUBMITTED.tsv and the job
# name are submit_arms.sh's own), then one sanity job (sanity_job.sh beside this file, which runs
# tools/t392/sanity.py; copy on Fir /project/def-maxwl/mforooz/t392/sanity_job.sh).
# Why the predict lines are not submit_arms.sh's: each shard needs its own RS392_EDGE_LOG file
# (rs_adapter.py rewrites the file after every forward; 4 shards on one file would overwrite each
# other). submit_arms.sh exports one environment to all shards. The predict lines below are the
# ones `DRY_RUN=1 SCORE_PHASES=1 SCORE_SHARDS=4 SCORE_PRED_DEVICE=cpu ...` prints, plus that
# variable. Shard 0 also runs the copy-leak guardrail after its requests, so its edge counters
# include the guardrail windows; shards 1-3 hold scoring windows only.
# --time is cut from the defaults (predict 06:00:00, metrics 03:00:00) to fit the per-seed request
# cap of 60 CPU-h (reference elapsed: predict 0:51-1:09, metrics 0:48). A predict shard that times
# out resumes request by request on resubmission.
set -euo pipefail
KIT=/project/def-maxwl/mforooz/onion_kit/RS5
RUNS=/project/def-maxwl/mforooz/t392/runs
ARM=rs_halves_mix_avg_pearson_unet
CONFIG=$KIT/tools/minicandi/configs/t392/rs_halves_mix_avg_pearson.json
TABLES=/project/def-maxwl/mforooz/ONION/tables/eic_v1_2
EV=/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/evalset_p16.json
SUB=p16_sv
PS_ROOT=/scratch/mforooz/rs392/pred_store
REFS=/project/def-maxwl/mforooz/ONION/eval/eic_v1_2/baselines_p16
# the exclude list of the reference's p16_sv jobs (sacct 5264675)
EXCL=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649
NSH=4; PB=8; PTIME=01:20:00; MTIME=00:55:00
STIME=00:30:00; SCPU=4; SMEM=16000
DRY="${DRY_RUN:-0}"
read -r -a SL <<< "${SEEDS:?set SEEDS}"
LIVE="$(squeue -h -u "$USER" -t PENDING,RUNNING,CONFIGURING -o '%j')"
for S in "${SL[@]}"; do
  RUN=$RUNS/$ARM/seed$S; SOUT=$RUN/$SUB; JOB=rs392_${ARM}_s$S
  STORE=$PS_ROOT/$(basename "$RUNS")/$ARM/same_view/seed$S/$SUB
  [ -f "$RUN/DONE" ] || { echo "seed $S: no $RUN/DONE; not submitted" >&2; continue; }
  if [ -f "$SOUT/SCORED" ]; then echo "seed $S: $SOUT/SCORED exists; skip"; continue; fi
  if printf '%s\n' "$LIVE" | grep -qxF "$JOB"; then echo "seed $S: a job $JOB is live; skip"; continue; fi
  [ "$DRY" = 1 ] || mkdir -p "$SOUT"
  PIDS=()
  for ((i = 0; i < NSH; i++)); do
    PENV=("ONION_SHARD=$i/$NSH" "ONION_INFER_BATCH=$PB" ONION_DEVICE=cpu ONION_SCORE_SAME_VIEW=1
          "RS392_EDGE_LOG=$SOUT/open_edges_${i}of${NSH}.json")
    PRED=(sbatch --account=def-maxwl --parsable "--exclude=$EXCL" "--job-name=$JOB" "--time=$PTIME"
          --cpus-per-task=8 --mem=32000M --gres=none "--output=$SOUT/predict_${i}of${NSH}_%j.out"
          --export=ALL "$KIT/slurm/onion_eval/predict.sh" "$KIT" "$RUN" "$EV" "$STORE")
    if [ "$DRY" = 1 ]; then echo "${PENV[*]} ${PRED[*]}"; PJ=$((900 + i))
    else PJ="$(env "${PENV[@]}" "${PRED[@]}")"; PJ="${PJ%%;*}"
      [[ "$PJ" =~ ^[0-9]+$ ]] || { echo "sbatch predict $i gave no id: $PJ" >&2; exit 1; }
    fi
    echo "$PJ $ARM $S predict $i/$NSH"; PIDS+=("$PJ")
  done
  env -u RS392_EDGE_LOG SEEDS="$S" METRICS_ONLY=1 PRED_JOBS="${PIDS[*]}" SCORE_SUBDIR=$SUB \
      SCORE_SAME_VIEW=1 PRED_STORE_ROOT=$PS_ROOT KEEP_PRED_STORE=1 JOB_PREFIX=rs392_ \
      REFS_FROM=$REFS SCORE_TIME=$MTIME ONION_EXCLUDE=$EXCL CLUSTER=fir DRY_RUN=$DRY \
      bash "$KIT/slurm/minicandi/submit_arms.sh" "$KIT" "$TABLES" "$EV" "$RUNS" 4 "$ARM=$CONFIG"
  SAN=$RUN/sanity_p16
  [ "$DRY" = 1 ] || mkdir -p "$SAN"
  SC=(sbatch --account=def-maxwl --parsable "--exclude=$EXCL" "--job-name=rs392_sanity_s$S" "--time=$STIME"
      "--cpus-per-task=$SCPU" "--mem=${SMEM}M" --gres=none "--output=$SAN/sanity_%j.out" --export=ALL
      /project/def-maxwl/mforooz/t392/sanity_job.sh "$KIT" "$RUN" "$EV" "$SAN/sanity.json")
  if [ "$DRY" = 1 ]; then echo "${SC[*]}"; else
    SJ="$(env -u RS392_EDGE_LOG "${SC[@]}")"; echo "${SJ%%;*} $ARM $S sanity -"; fi
done
