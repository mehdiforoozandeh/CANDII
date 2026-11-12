#!/bin/bash
# miniCANDI: train one (arm, seed) run, then score it (tools/minicandi/train.py, then
# tools/onion_eval/score.py on the same run directory). One job, no array.
#
#   CPU form (the header below: 8 cores x 4000 MB, no GPU):
#     sbatch --time=03:00:00 --output=$OUT/train_%j.out \
#         $KIT/slurm/minicandi/train.sh $KIT $CONFIG $TABLES $SEED $OUT cpu [$EVALSET]
#   GPU form (the fixed slice only; 1 CPU on Fir, 2 on Nibi; on Fir also --exclude=fc10713,fc10715):
#     sbatch --time=03:00:00 --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 --cpus-per-task=1 \
#         --mem=10000M --output=$OUT/train_%j.out \
#         $KIT/slurm/minicandi/train.sh $KIT $CONFIG $TABLES $SEED $OUT gpu [$EVALSET]
#   -> $OUT/{ckpt.pt,run.json,log.jsonl,DONE} and, with EVALSET, $OUT/{scores.json,suffstats.npz,SCORED}
#
# Rerunning the same command resumes from $OUT/ckpt.pt (train.py) and skips training when DONE
# exists; scoring is skipped only when $OUT/SCORED exists (score.py writes it after scores.json;
# training's DONE never skips scoring). train.py gets --workers <allocated CPUs - 1> (at least 0):
# that many DataLoader worker processes draw the training batches (0.93 s per batch in-process
# against 0.081 s with 8 workers on Fir); a --workers in EXTRA_TRAIN_ARGS overrides it.
# EXTRA_TRAIN_ARGS (space-separated, e.g. "--max-windows 100000 --val-every 20000") is
# passed to train.py. Never pass a comma-valued variable through sbatch --export; export it in the
# shell and use --export=ALL. DRY_RUN=1 prints the commands and exits.
# Retry (as slurm/onion_kit/check.sh): when the environment setup fails (e.g. a CVMFS I/O error,
# exit 126, on a bad node) the job resubmits itself once, with its own sbatch options read from
# `scontrol show job` and this node added to --exclude, moves every pending job of this user that
# waits on it with afterok (submit_arms.sh's scoring job) onto the retry, logs
# `[minicandi-train] retried-as=<job id> host=<node>` and exits 3. A retried job ($ONION_RETRY_OF
# set) never retries again. A healthy setup runs exactly the commands it ran before.
#SBATCH --account=def-maxwl
#SBATCH --job-name=minicandi_train
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --gres=none

set -euo pipefail
ARGS=("$@")
USAGE="usage: minicandi/train.sh <kit_dir> <config.json> <tables_dir> <seed> <out_dir> <cpu|gpu> [evalset.json]"
KIT="${1:?$USAGE}"
CONFIG="${2:?$USAGE}"
TABLES="${3:?$USAGE}"
SEED="${4:?$USAGE}"
OUT="${5:?$USAGE}"
MODE="${6:?$USAGE}"
EVALSET="${7:-}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be an integer" >&2; exit 2; }
case "$MODE" in
  cpu) DEVICE=cpu ;;
  gpu) DEVICE=cuda ;;
  *) echo "mode must be cpu or gpu" >&2; exit 2 ;;
esac
read -r -a EXTRA <<< "${EXTRA_TRAIN_ARGS:-}"
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-1} - 1 ))
[ "$WORKERS" -ge 0 ] || WORKERS=0
TRAIN=(python3 "$KIT/tools/minicandi/train.py" --config "$CONFIG" --tables "$TABLES"
       --seed "$SEED" --out "$OUT" --device "$DEVICE" --workers "$WORKERS" ${EXTRA[@]+"${EXTRA[@]}"})
SCORE=(python3 "$KIT/tools/onion_eval/score.py" --run "$OUT" --evalset "$EVALSET" --out "$OUT")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "${TRAIN[*]}"
  [ -n "$EVALSET" ] && echo "${SCORE[*]}"
  exit 0
fi
if [ "$MODE" = gpu ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "mode gpu but the job has no GPU: submit with --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1" >&2
  exit 2
fi
mkdir -p "$OUT"
echo "=== minicandi train job ${SLURM_JOB_ID:-none} host=$(hostname) mode=$MODE $(date -u)"
setup() {
  local rc
  set +u; module load python/3.10.13; rc=$?; set -u
  [ "$rc" = 0 ] &&
  virtualenv --no-download "$SLURM_TMPDIR/venv" &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install --no-index --upgrade pip &&
  pip install --no-index -r "$KIT/requirements-fir.txt" &&
  pip install --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt"
}
options_from_job() {   # scontrol record on stdin -> line 1 sbatch options, line 2 ExcNodeList
  local rec name tl cpt mem out exc gres
  rec="$(tr ' ' '\n')"
  field() { printf '%s\n' "$rec" | sed -n "s/^$1=//p" | head -1; }
  name="$(field JobName)"; tl="$(field TimeLimit)"; cpt="$(field 'CPUs\/Task')"
  mem="$(field MinMemoryNode)"; out="$(field StdOut)"
  exc="$(field ExcNodeList)"; gres="$(field TresPerNode)"
  [ -n "$tl" ] && [ -n "$cpt" ] && [ -n "$mem" ] || return 1
  if [ -n "$out" ]; then out="${out//$SLURM_JOB_ID/%j}"; [[ "$out" == *%j* ]] || out="$out.retry_%j"; fi
  case "$gres" in gres/*) gres="--gres=${gres#gres/}" ;; *) gres="--gres=none" ;; esac
  printf '%s\n' "${name:+--job-name=$name }--time=$tl --cpus-per-task=$cpt --mem=$mem $gres${out:+ --output=$out}"
  [ "$exc" = "(null)" ] && exc=""
  printf '%s\n' "$exc"
}
retry_elsewhere() {   # at most once: a retried job never retries again
  [ -z "${ONION_RETRY_OF:-}" ] && [ -n "${SLURM_JOB_ID:-}" ] || return 1
  local derived opts ex host new dep
  derived="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | options_from_job)" ||
    { echo "train.sh: no job record; the retry was not submitted"; return 1; }
  opts="$(printf '%s\n' "$derived" | sed -n 1p)"
  ex="$(printf '%s\n' "$derived" | sed -n 2p)"
  host="$(hostname -s 2>/dev/null || hostname)"
  ex="${ex:+$ex,}$host"
  # opts is a word list of sbatch options (no spaces inside a value)
  # shellcheck disable=SC2086
  new="$( (deactivate 2>/dev/null || true
           ONION_RETRY_OF="$SLURM_JOB_ID" sbatch --parsable --export=ALL $opts --exclude="$ex" \
             "$KIT/slurm/minicandi/train.sh" "${ARGS[@]}") 2>&1 | tail -1 | cut -d';' -f1)"
  [[ "$new" =~ ^[0-9]+$ ]] || { echo "train.sh: the retry was not submitted: $new"; return 1; }
  echo "[minicandi-train] retried-as=$new host=$host"
  # a dependent (afterok) job would be killed when this job fails: point it at the retry.
  # Fail-safe (unverified on Fir and Nibi): one try per dependent, no loop, nothing cancelled.
  squeue -h -u "${USER:-$(id -un)}" -t PENDING -o '%i %E' 2>/dev/null |
    grep -E "afterok:${SLURM_JOB_ID}([^0-9]|\$)" | while read -r dep _; do
      if scontrol update "JobId=$dep" "Dependency=afterok:$new" > /dev/null 2>&1; then
        echo "[minicandi-train] dependent $dep now waits on $new"
      else
        echo "[minicandi-train] dependency-move-failed scoring_job=$dep retry_job=$new: submit the scoring for $new by hand"
      fi
    done
  return 0
}
if ! setup; then
  echo "train.sh: environment setup failed on $(hostname)"
  retry_elsewhere || true
  exit 3
fi
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
"${TRAIN[@]}"
if [ -z "$EVALSET" ]; then
  echo "no evalset given: scoring skipped"
elif [ -f "$OUT/SCORED" ]; then
  echo "skip scoring: $OUT/SCORED exists"
else
  "${SCORE[@]}"
fi
echo "=== done $(date -u)"
