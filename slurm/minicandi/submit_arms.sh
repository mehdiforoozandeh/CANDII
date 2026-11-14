#!/bin/bash
# miniCANDI ladder: submit the training and scoring jobs of one rung's arms (chunk LK). Run on a
# login node; it is not itself an sbatch job.
#
#   bash submit_arms.sh <kit dir> <tables dir> <evalset.json|""> <runs root> <rung> [arm ...]
#
# An arm is a name (config <kit>/tools/minicandi/configs/arms/rung<rung>/<arm>.json, run directory
# <runs root>/rung<rung>_<arm>/seed<seed>) or <name>=<json> (that config; run directory
# <runs root>/<name>/seed<seed>, no rung prefix). No arm: every arm of the rung
# (tools/minicandi/ladder_arms.py arms). Resources come from tools/minicandi/configs/budgets.json
# through `ladder_arms.py plan` (the rung argument picks the device, the scoring time and the
# measured speed): training time = 3 x projected wall, at least 1 h; scoring on the rung's device
# type (GPU form when cuda, else 2 CPUs), so every seed of a comparison is scored on one device.
#
# Env: SEEDS (default "0 1 2"); DRY_RUN=1 (print the sbatch lines, submit and write nothing);
#   EXTRA_TRAIN_ARGS (to train.sh, after "--workers <budgets workers>"); SCORE_TIME (overrides
#   budgets score_time); SCORE_MEM=<MB> (overrides the scoring --mem; eic_v1 scoring needs
#   128000); ONION_EXCLUDE=<node list> (--exclude on every line; on Fir the nodes of
#   .orchestrate/exclude_fir); DEVICE=cpu|cuda (overrides the rung's device); CLUSTER=fir|nibi
#   (default $CC_CLUSTER, else the host name; a GPU job needs it for its CPU count).
# Scoring and queue options (interface K, chunk SAx; unset = the lines above, byte for byte):
#   EXTRA_SCORE_ARGS (appended to the score.sh line after "run <run dir>", e.g. "--per-region";
#   split on spaces); NICE=<n> (--nice=<n> on the training and the scoring line; region runs use
#   300, below the ladder); SCORE_DEVICE=cpu|gpu (scoring form of ladder_arms.py plan for that
#   device instead of the rung's: cpu = --gres=none, 2 CPUs, 8000M; gpu = the 1g.10gb slice, the
#   cluster's slice CPUs, 24000M; SCORE_MEM still overrides the --mem); SCORE_CPUS=<n> (overrides
#   the scoring --cpus-per-task); SCORE_ONLY=1 (needs an eval set; submits no training: per run
#   it skips SCORED and a queued or running scoring job of the run's name, submits a scoring job
#   with --dependency=afterok on the run's queued or running training job (train.sh), or with no
#   dependency when DONE exists, else skips; squeue '%o' tells a training from a scoring job).
# Two-phase scoring (interface M, chunk SP3; unset = the lines above, byte for byte):
#   SCORE_PHASES=1: per run, SCORE_SHARDS (default 1) predict jobs (slurm/onion_eval/predict.sh,
#   ONION_SHARD=i/N) into the store $PRED_STORE_ROOT/<runs root name>/<run name>/seed<s>[/<SUB>]
#   (PRED_STORE_ROOT default /scratch/mforooz/onion/pred_store), then one metrics job (score.sh,
#   ONION_SCORE_PHASE=metrics) with --dependency=afterok on all of them. Predict device
#   SCORE_PRED_DEVICE=cuda|cpu (default: the rung's scoring device); cuda = the 1g.10gb slice,
#   4 CPUs, 16000M, 02:00:00, batch 32; cpu = --gres=none, 8 CPUs, 32000M, 06:00:00, batch 8;
#   SCORE_PRED_CPUS, SCORE_PRED_MEM (MB), SCORE_PRED_TIME, SCORE_INFER_BATCH override them.
#   The batch (32 or 8, or SCORE_INFER_BATCH) is always passed as ONION_INFER_BATCH, so
#   predict.sh's own default is never used: infer_batch is a comparability key of the store, so
#   every run of one comparison must use one batch. A store that is already complete
#   (STORE.json and every PREDICTED_<i>of<N>.json) gets no predict job; its metrics job waits on
#   nothing (PRED_JOBS ignored), and PREDICT_ONLY=1 skips the run.
#   Metrics: --gres=none, SCORE_CPUS (default 16), SCORE_MEM (default 64000), SCORE_TIME (default
#   03:00:00); REFS_FROM=<baselines dir> is required. KEEP_PRED_STORE=1 keeps the store after
#   SCORED (else the metrics job removes it). SCORE_SUBDIR=<SUB> scores to <run dir>/<SUB> (also
#   without SCORE_PHASES; the skip test is then <run dir>/<SUB>/SCORED). PREDICT_ONLY=1: the
#   predict jobs only (no REFS_FROM needed). METRICS_ONLY=1: the metrics job only; it needs a
#   complete store at submit time (STORE.json and every PREDICTED_<i>of<N>.json) or PRED_JOBS=
#   "<id> ..." (the predict job ids it waits for); a run with neither is skipped. Both imply
#   SCORE_PHASES=1 and submit no training (run selection as SCORE_ONLY). HOLD=1 adds --hold to
#   every sbatch line. Each job prints "<job id> <run name> <seed> <phase> <shard>" (phase
#   train|predict|metrics; shard i/N or -; a dry run prints <...> placeholders as ids).
# TRAIN_TIME=<[d-]hh:mm:ss> (chunk TRa; unset = the lines above, byte for byte) overrides the
#   training --time of ladder_arms.py plan, whose speeds are Fir's.
#
# Idempotent. Per (arm, seed): skip when <run dir>/SCORED exists (training only: DONE), or when a
# queued or running job of this user is named mc_<run dir name>_s<seed>; when DONE exists but not
# SCORED, submit the scoring job only; else a training job and a scoring job with
# --dependency=afterok (and --kill-on-invalid-dep=yes, so a failed training never leaves a pending
# job that blocks a resubmission). Both jobs carry that name. Each submission appends
# "arm seed train_job score_job time" to <runs root>/SUBMITTED.tsv.
set -euo pipefail
USAGE='usage: submit_arms.sh <kit dir> <tables dir> <evalset.json|""> <runs root> <rung> [arm ...]'
[ "$#" -ge 5 ] || { echo "$USAGE" >&2; exit 2; }
KIT="$1"; TABLES="$2"; EVALSET="$3"; ROOT="$4"; RUNG="$5"; shift 5
[[ "$RUNG" =~ ^[1-6]$ ]] || { echo "rung must be 1..6" >&2; exit 2; }
[ -d "$KIT" ] || { echo "no kit dir $KIT" >&2; exit 2; }
[ -d "$TABLES" ] || { echo "no tables dir $TABLES" >&2; exit 2; }
[ -z "$EVALSET" ] || [ -f "$EVALSET" ] || { echo "no evalset $EVALSET" >&2; exit 2; }
PY="${PYTHON:-python3}"
LADDER="$KIT/tools/minicandi/ladder_arms.py"
DRY="${DRY_RUN:-0}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2}"
EXCLUDE="${ONION_EXCLUDE:-}"; EXCLUDE="${EXCLUDE//[[:space:]]/}"
CL="${CLUSTER:-${CC_CLUSTER:-}}"
if [ -z "$CL" ]; then
  case "$(hostname)" in *nibi*) CL=nibi ;; *fir*) CL=fir ;; esac
fi
SDEV="${SCORE_DEVICE:-}"
case "$SDEV" in ""|cpu|gpu) ;; *) echo "SCORE_DEVICE must be cpu or gpu" >&2; exit 2 ;; esac
[ -z "${NICE:-}" ] || [[ "$NICE" =~ ^-?[0-9]+$ ]] || { echo "NICE must be an integer" >&2; exit 2; }
SONLY="${SCORE_ONLY:-0}"
[ "$SONLY" != 1 ] || [ -n "$EVALSET" ] || { echo "SCORE_ONLY=1 needs an eval set" >&2; exit 2; }
# two-phase scoring (SP3)
PHASES="${SCORE_PHASES:-0}"; PONLY="${PREDICT_ONLY:-0}"; MONLY="${METRICS_ONLY:-0}"
SUB="${SCORE_SUBDIR:-}"
[ -z "$SUB" ] || { [[ "$SUB" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ ]] && [ "$SUB" != .. ]; } || \
  { echo "SCORE_SUBDIR must be one plain directory name" >&2; exit 2; }
[ "$PONLY" != 1 ] || [ "$MONLY" != 1 ] || { echo "PREDICT_ONLY and METRICS_ONLY exclude each other" >&2; exit 2; }
if [ "$PONLY" = 1 ] || [ "$MONLY" = 1 ]; then PHASES=1; fi
PJ="${PRED_JOBS:-}"; read -r -a PRED_IDS <<< "${PJ//:/ }"
if [ "$PHASES" = 1 ]; then
  [ -n "$EVALSET" ] || { echo "SCORE_PHASES=1 needs an eval set" >&2; exit 2; }
  if [ "$PONLY" != 1 ]; then
    [ -n "${REFS_FROM:-}" ] || { echo "a metrics job needs REFS_FROM=<baselines dir>" >&2; exit 2; }
    [ -d "$REFS_FROM" ] || { echo "no REFS_FROM dir $REFS_FROM" >&2; exit 2; }
  fi
  case "${SCORE_PRED_DEVICE:-}" in ""|cuda|cpu) ;; *) echo "SCORE_PRED_DEVICE must be cuda or cpu" >&2; exit 2 ;; esac
  NSH="${SCORE_SHARDS:-1}"
  [[ "$NSH" =~ ^[1-9][0-9]*$ ]] || { echo "SCORE_SHARDS must be a positive integer" >&2; exit 2; }
  for x in ${PRED_IDS[@]+"${PRED_IDS[@]}"}; do
    [[ "$x" =~ ^[0-9]+$ ]] || { echo "PRED_JOBS must hold job ids" >&2; exit 2; }
  done
  [ "${#PRED_IDS[@]}" = 0 ] || [ "$MONLY" = 1 ] || { echo "PRED_JOBS goes with METRICS_ONLY=1" >&2; exit 2; }
  PS_ROOT="${PRED_STORE_ROOT:-/scratch/mforooz/onion/pred_store}"
  if [ "$PONLY" = 1 ] || [ "$MONLY" = 1 ]; then SONLY=1; fi   # no training: SCORE_ONLY's run selection
fi
# a complete prediction store: STORE.json and every PREDICTED_<i>of<N>.json (METRICS_ONLY=1)
store_complete() {
  "$PY" - "$1" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
try:
    n = int(json.loads((d / "STORE.json").read_text())["n_shards"])
except Exception:
    sys.exit(1)
sys.exit(0 if all((d / f"PREDICTED_{i}of{n}.json").is_file() for i in range(n)) else 1)
PYEOF
}
read -r -a SCORE_EXTRA <<< "${EXTRA_SCORE_ARGS:-}"
[ "$#" -gt 0 ] || read -r -a ARGS_ARMS <<< "$("$PY" "$LADDER" arms --rung "$RUNG")"
[ "$#" -eq 0 ] || ARGS_ARMS=("$@")

# names of this user's queued or running jobs (one per line); none when squeue is absent.
# SCORE_ONLY=1 also reads "<id> <name> <command>" lines (LIVEJ).
LIVEJ=""
if command -v squeue > /dev/null 2>&1; then
  LIVE="$(squeue -h -u "${USER:-$(id -un)}" -t PENDING,RUNNING,CONFIGURING -o '%j')"
  [ "$SONLY" != 1 ] || \
    LIVEJ="$(squeue -h -u "${USER:-$(id -un)}" -t PENDING,RUNNING,CONFIGURING -o '%i %j %o')"
else
  [ "$DRY" = 1 ] || { echo "squeue not found: cannot check live jobs" >&2; exit 2; }
  echo "note: squeue not found; live jobs not checked (dry run)"
  LIVE=""
fi

COMMON=(--account=def-maxwl --parsable)
[ -z "$EXCLUDE" ] || COMMON+=("--exclude=$EXCLUDE")
[ -z "${NICE:-}" ] || COMMON+=("--nice=$NICE")
[ "${HOLD:-0}" != 1 ] || COMMON+=(--hold)
[ -z "${TRAIN_TIME:-}" ] || [[ "$TRAIN_TIME" =~ ^([0-9]+-)?[0-9]+:[0-9]{2}:[0-9]{2}$ ]] || \
  { echo "TRAIN_TIME must be [d-]hh:mm:ss" >&2; exit 2; }
TSV="$ROOT/SUBMITTED.tsv"
n_sub=0; n_skip=0

for spec in "${ARGS_ARMS[@]}"; do
  if [[ "$spec" == *=* ]]; then
    NAME="${spec%%=*}"; CONFIG="${spec#*=}"; ARMDIR="$ROOT/$NAME"
  else
    NAME="$spec"; CONFIG="$KIT/tools/minicandi/configs/arms/rung$RUNG/$spec.json"
    ARMDIR="$ROOT/rung${RUNG}_$spec"
  fi
  [ -f "$CONFIG" ] || { echo "no config $CONFIG for arm $spec" >&2; exit 2; }
  read -r MODE GRES CPUS MEM WORKERS TTIME SGRES SCPUS SMEM STIME <<< "$("$PY" "$LADDER" plan \
      --rung "$RUNG" --config "$CONFIG" --device "${DEVICE:-}" --cluster "$CL" \
      --score-time "${SCORE_TIME:-}" --extra "${EXTRA_TRAIN_ARGS:-}")"
  [ -n "$STIME" ] || { echo "plan failed for $spec" >&2; exit 2; }
  if [ -n "$SDEV" ]; then   # the scoring form of the other device; training fields ignored
    read -r _ _ _ _ _ _ SGRES SCPUS SMEM STIME2 <<< "$("$PY" "$LADDER" plan \
        --rung "$RUNG" --config "$CONFIG" --device "${SDEV/gpu/cuda}" --cluster "$CL" \
        --score-time "${SCORE_TIME:-}" --extra "${EXTRA_TRAIN_ARGS:-}")"
    [ -n "${STIME2:-}" ] || { echo "scoring plan (SCORE_DEVICE=$SDEV) failed for $spec" >&2; exit 2; }
  fi
  SMEM="${SCORE_MEM:-$SMEM}"; SCPUS="${SCORE_CPUS:-$SCPUS}"; TTIME="${TRAIN_TIME:-$TTIME}"
  if [ "$PHASES" = 1 ]; then   # SP3: the predict and metrics job shapes of this arm
    PD="${SCORE_PRED_DEVICE:-}"
    if [ -z "$PD" ]; then if [ "$SGRES" = none ]; then PD=cpu; else PD=cuda; fi; fi
    PG=none
    if [ "$PD" = cuda ]; then
      PG="$SGRES"
      if [ "$PG" = none ]; then
        read -r _ _ _ _ _ _ PG _ _ _ <<< "$("$PY" "$LADDER" plan --rung "$RUNG" --config "$CONFIG" \
            --device cuda --cluster "$CL" --score-time "" --extra "${EXTRA_TRAIN_ARGS:-}")"
        [ -n "$PG" ] && [ "$PG" != none ] || { echo "no GPU form for the predict jobs of $spec" >&2; exit 2; }
      fi
      PCPUS="${SCORE_PRED_CPUS:-4}"; PMEM="${SCORE_PRED_MEM:-16000}"
      PTIME="${SCORE_PRED_TIME:-02:00:00}"; PB="${SCORE_INFER_BATCH:-32}"
    else
      PCPUS="${SCORE_PRED_CPUS:-8}"; PMEM="${SCORE_PRED_MEM:-32000}"
      PTIME="${SCORE_PRED_TIME:-06:00:00}"; PB="${SCORE_INFER_BATCH:-8}"
    fi
    MCPUS="${SCORE_CPUS:-16}"; MMEM="${SCORE_MEM:-64000}"; MTIME="${SCORE_TIME:-03:00:00}"
  fi
  for SEED in "${SEED_LIST[@]}"; do
    RUN="$ARMDIR/seed$SEED"
    JOB="mc_$(basename "$ARMDIR")_s$SEED"
    SOUT="$RUN${SUB:+/$SUB}"   # the score dir (SCORE_SUBDIR)
    if [ -n "$EVALSET" ] && [ -f "$SOUT/SCORED" ]; then
      echo "skip $JOB: $SOUT/SCORED exists"; n_skip=$((n_skip + 1)); continue
    fi
    if [ -z "$EVALSET" ] && [ -f "$RUN/DONE" ]; then
      echo "skip $JOB: $RUN/DONE exists (no eval set)"; n_skip=$((n_skip + 1)); continue
    fi
    if [ "$SONLY" != 1 ] && printf '%s\n' "$LIVE" | grep -qxF "$JOB"; then
      echo "skip $JOB: a job of that name is queued or running"; n_skip=$((n_skip + 1)); continue
    fi
    TID=""
    if [ "$SONLY" = 1 ]; then
      MINE="$(printf '%s\n' "$LIVEJ" | awk -v j="$JOB" '$2 == j')"
      TID="$(printf '%s\n' "$MINE" | awk '/train\.sh/ {print $1; exit}')"
      NOTSC='train\.sh'; [ "$MONLY" != 1 ] || NOTSC='train\.sh|predict\.sh'
      if printf '%s\n' "$MINE" | awk -v p="$NOTSC" 'NF && $0 !~ p' | grep -q .; then
        echo "skip $JOB: a scoring job of that name is queued or running"
        n_skip=$((n_skip + 1)); continue
      fi
      if [ "$MONLY" != 1 ] && [ -z "$TID" ] && [ ! -f "$RUN/DONE" ]; then
        echo "skip $JOB: training neither done nor queued or running"
        n_skip=$((n_skip + 1)); continue
      fi
    fi
    TRAIN=(sbatch "${COMMON[@]}" "--job-name=$JOB" "--time=$TTIME" "--cpus-per-task=$CPUS"
           "--mem=${MEM}M" "--gres=$GRES" "--output=$RUN/train_%j.out" --export=ALL
           "$KIT/slurm/minicandi/train.sh" "$KIT" "$CONFIG" "$TABLES" "$SEED" "$RUN" "$MODE")
    SCORE=(sbatch "${COMMON[@]}" "--job-name=$JOB" "--time=$STIME" "--cpus-per-task=$SCPUS"
           "--mem=${SMEM}M" "--gres=$SGRES" "--output=$SOUT/score_%j.out"
           "$KIT/slurm/onion_eval/score.sh" "$KIT" "$EVALSET" "$SOUT" run "$RUN"
           ${SCORE_EXTRA[@]+"${SCORE_EXTRA[@]}"})
    DO_TRAIN=1
    [ -f "$RUN/DONE" ] && DO_TRAIN=0
    [ "$SONLY" != 1 ] || DO_TRAIN=0
    DEP=()   # SCORE_ONLY=1 with a live training job: the scoring waits on it
    [ -z "$TID" ] || DEP=("--dependency=afterok:$TID" --kill-on-invalid-dep=yes)
    # train.sh reads EXTRA_TRAIN_ARGS from the exported environment (--export=ALL)
    TRAIN_EXTRA="--workers $WORKERS${EXTRA_TRAIN_ARGS:+ $EXTRA_TRAIN_ARGS}"
    if [ "$PHASES" = 1 ]; then   # SP3: training (as above), predict jobs, one metrics job
      RNAME="$(basename "$ARMDIR")"
      STORE="$PS_ROOT/$(basename "$ROOT")/$RNAME/seed$SEED${SUB:+/$SUB}"
      COMPLETE=0; store_complete "$STORE" && COMPLETE=1
      if [ "$MONLY" = 1 ] && [ "${#PRED_IDS[@]}" = 0 ] && [ "$COMPLETE" != 1 ]; then
        echo "skip $JOB: METRICS_ONLY and the store $STORE is not complete (set PRED_JOBS)"
        n_skip=$((n_skip + 1)); continue
      fi
      if [ "$PONLY" = 1 ] && [ "$COMPLETE" = 1 ]; then
        echo "skip $JOB: PREDICT_ONLY and the store $STORE is complete"
        n_skip=$((n_skip + 1)); continue
      fi
      # a complete store: no predict job, and the metrics job waits on nothing (PRED_JOBS ignored:
      # a finished predict job has left the scheduler and sbatch would refuse its afterok)
      [ "$COMPLETE" != 1 ] || echo "note $JOB: the store $STORE is complete; no predict job, the metrics job waits on none"
      [ "$DRY" = 1 ] || mkdir -p "$SOUT"
      TJ="-"; PIDS=(); MJ="-"
      if [ "$DO_TRAIN" = 1 ]; then
        if [ "$DRY" = 1 ]; then
          echo "EXTRA_TRAIN_ARGS=\"$TRAIN_EXTRA\" ${TRAIN[*]}"; TJ="<train>"
        else
          TJ="$(EXTRA_TRAIN_ARGS="$TRAIN_EXTRA" "${TRAIN[@]}")"; TJ="${TJ%%;*}"
          [[ "$TJ" =~ ^[0-9]+$ ]] || { echo "sbatch (train) gave no job id for $JOB: $TJ" >&2; exit 1; }
        fi
        echo "$TJ $RNAME $SEED train -"
        DEP=("--dependency=afterok:$TJ" --kill-on-invalid-dep=yes)
      fi
      if [ "$COMPLETE" = 1 ]; then
        PIDS=()
      elif [ "$MONLY" != 1 ]; then
        for ((i = 0; i < NSH; i++)); do
          PENV=("ONION_SHARD=$i/$NSH" "ONION_INFER_BATCH=$PB" "ONION_DEVICE=$PD")
          PRED=(sbatch "${COMMON[@]}" "--job-name=$JOB" "--time=$PTIME" "--cpus-per-task=$PCPUS"
                "--mem=${PMEM}M" "--gres=$PG" "--output=$SOUT/predict_${i}of${NSH}_%j.out"
                ${DEP[@]+"${DEP[@]}"} --export=ALL
                "$KIT/slurm/onion_eval/predict.sh" "$KIT" "$RUN" "$EVALSET" "$STORE")
          if [ "$DRY" = 1 ]; then
            echo "${PENV[*]} ${PRED[*]}"; PJID="<predict $i>"
          else
            PJID="$(env "${PENV[@]}" "${PRED[@]}")"; PJID="${PJID%%;*}"
            [[ "$PJID" =~ ^[0-9]+$ ]] || { echo "sbatch (predict $i/$NSH) gave no job id for $JOB: $PJID" >&2; exit 1; }
          fi
          PIDS+=("$PJID"); echo "$PJID $RNAME $SEED predict $i/$NSH"
        done
      else
        PIDS=(${PRED_IDS[@]+"${PRED_IDS[@]}"})
      fi
      if [ "$PONLY" != 1 ]; then
        MDEP=()
        if [ "${#PIDS[@]}" -gt 0 ]; then
          WAIT="$(IFS=:; echo "${PIDS[*]}")"
          MDEP=("--dependency=afterok:$WAIT" --kill-on-invalid-dep=yes)
        fi
        MENV=(ONION_SCORE_PHASE=metrics "ONION_PRED_STORE=$STORE" "ONION_REFS_FROM=$REFS_FROM")
        [ "${KEEP_PRED_STORE:-0}" != 1 ] || MENV+=(ONION_KEEP_PRED_STORE=1)
        MET=(sbatch "${COMMON[@]}" "--job-name=$JOB" "--time=$MTIME" "--cpus-per-task=$MCPUS"
             "--mem=${MMEM}M" --gres=none "--output=$SOUT/metrics_%j.out"
             ${MDEP[@]+"${MDEP[@]}"} --export=ALL
             "$KIT/slurm/onion_eval/score.sh" "$KIT" "$EVALSET" "$SOUT" run "$RUN"
             ${SCORE_EXTRA[@]+"${SCORE_EXTRA[@]}"})
        if [ "$DRY" = 1 ]; then
          echo "${MENV[*]} ${MET[*]}"; MJ="<metrics>"
        else
          MJ="$(env "${MENV[@]}" "${MET[@]}")"; MJ="${MJ%%;*}"
          [[ "$MJ" =~ ^[0-9]+$ ]] || { echo "sbatch (metrics) gave no job id for $JOB: $MJ" >&2; exit 1; }
        fi
        echo "$MJ $RNAME $SEED metrics -"
      fi
      if [ "$DRY" != 1 ]; then
        [ -f "$TSV" ] || printf 'arm\tseed\ttrain_job\tscore_job\ttime\n' > "$TSV"
        PL="$(IFS=:; echo "${PIDS[*]-}")"
        printf '%s\t%s\t%s\t%s\t%s\n' "$RNAME" "$SEED" "$TJ" "predict=${PL:--};metrics=$MJ" \
          "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TSV"
      fi
      n_sub=$((n_sub + 1)); continue
    fi
    if [ "$DRY" = 1 ]; then
      if [ "$DO_TRAIN" = 1 ]; then
        echo "EXTRA_TRAIN_ARGS=\"$TRAIN_EXTRA\" ${TRAIN[*]}"
        [ -z "$EVALSET" ] || echo "${SCORE[*]:0:3} --dependency=afterok:<train job> --kill-on-invalid-dep=yes ${SCORE[*]:3}"
      else
        if [ "$SONLY" = 1 ]; then
          echo "note $JOB: SCORE_ONLY; ${TID:+after training job $TID}${TID:-$RUN/DONE exists}"
        else
          echo "note $JOB: $RUN/DONE exists; scoring only"
        fi
        echo "${SCORE[*]:0:3}${TID:+ ${DEP[*]}} ${SCORE[*]:3}"
      fi
      n_sub=$((n_sub + 1)); continue
    fi
    mkdir -p "$SOUT"
    TJ="-"; SJ="-"
    if [ "$DO_TRAIN" = 1 ]; then
      TJ="$(EXTRA_TRAIN_ARGS="$TRAIN_EXTRA" "${TRAIN[@]}")"; TJ="${TJ%%;*}"
      [[ "$TJ" =~ ^[0-9]+$ ]] || { echo "sbatch (train) gave no job id for $JOB: $TJ" >&2; exit 1; }
      if [ -n "$EVALSET" ]; then
        SJ="$("${SCORE[@]:0:3}" "--dependency=afterok:$TJ" --kill-on-invalid-dep=yes "${SCORE[@]:3}")"
      fi
    elif [ -n "$EVALSET" ]; then
      SJ="$("${SCORE[@]:0:3}" ${DEP[@]+"${DEP[@]}"} "${SCORE[@]:3}")"
    fi
    SJ="${SJ%%;*}"
    [ -f "$TSV" ] || printf 'arm\tseed\ttrain_job\tscore_job\ttime\n' > "$TSV"
    printf '%s\t%s\t%s\t%s\t%s\n' "$(basename "$ARMDIR")" "$SEED" "$TJ" "$SJ" \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$TSV"
    echo "submit $JOB: train $TJ, score $SJ"
    n_sub=$((n_sub + 1))
  done
done
if [ "$DRY" = 1 ]; then
  echo "dry run: would submit $n_sub runs, skipped $n_skip"
else
  echo "submitted $n_sub runs, skipped $n_skip; log $TSV"
fi
