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
# Queue options (chunk TRa; unset = the lines above, byte for byte): TRAIN_TIME=<[d-]hh:mm:ss>
#   (overrides the training --time of ladder_arms.py plan, whose speeds are Fir's); HOLD=1
#   (--hold on every line: jobs wait held until `scontrol release`).
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
  for SEED in "${SEED_LIST[@]}"; do
    RUN="$ARMDIR/seed$SEED"
    JOB="mc_$(basename "$ARMDIR")_s$SEED"
    if [ -n "$EVALSET" ] && [ -f "$RUN/SCORED" ]; then
      echo "skip $JOB: $RUN/SCORED exists"; n_skip=$((n_skip + 1)); continue
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
      if printf '%s\n' "$MINE" | awk 'NF && !/train\.sh/' | grep -q .; then
        echo "skip $JOB: a scoring job of that name is queued or running"
        n_skip=$((n_skip + 1)); continue
      fi
      if [ -z "$TID" ] && [ ! -f "$RUN/DONE" ]; then
        echo "skip $JOB: training neither done nor queued or running"
        n_skip=$((n_skip + 1)); continue
      fi
    fi
    TRAIN=(sbatch "${COMMON[@]}" "--job-name=$JOB" "--time=$TTIME" "--cpus-per-task=$CPUS"
           "--mem=${MEM}M" "--gres=$GRES" "--output=$RUN/train_%j.out" --export=ALL
           "$KIT/slurm/minicandi/train.sh" "$KIT" "$CONFIG" "$TABLES" "$SEED" "$RUN" "$MODE")
    SCORE=(sbatch "${COMMON[@]}" "--job-name=$JOB" "--time=$STIME" "--cpus-per-task=$SCPUS"
           "--mem=${SMEM}M" "--gres=$SGRES" "--output=$RUN/score_%j.out"
           "$KIT/slurm/onion_eval/score.sh" "$KIT" "$EVALSET" "$RUN" run "$RUN"
           ${SCORE_EXTRA[@]+"${SCORE_EXTRA[@]}"})
    DO_TRAIN=1
    [ -f "$RUN/DONE" ] && DO_TRAIN=0
    [ "$SONLY" != 1 ] || DO_TRAIN=0
    DEP=()   # SCORE_ONLY=1 with a live training job: the scoring waits on it
    [ -z "$TID" ] || DEP=("--dependency=afterok:$TID" --kill-on-invalid-dep=yes)
    # train.sh reads EXTRA_TRAIN_ARGS from the exported environment (--export=ALL)
    TRAIN_EXTRA="--workers $WORKERS${EXTRA_TRAIN_ARGS:+ $EXTRA_TRAIN_ARGS}"
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
    mkdir -p "$RUN"
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
