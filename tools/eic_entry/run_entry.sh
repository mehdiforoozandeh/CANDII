#!/bin/bash
# EIC entry pipeline (t390) on Nibi: score miniCANDI blind predictions on the MOESM2 and MOESM4 boards.
#
#   bash run_entry.sh deploy [nibi|fir]                  (on the Mac) rsync the code subset to $CODE
#   bash run_entry.sh submit <pred> <cov.tsv> <out> [C##M## ...]
#        (on Nibi) one job per track (score_entry.py run), then one collect job (board TSVs, SUMMARY.md).
#        <pred> = predict.py's --out folder (counts/<track>.h5); <cov.tsv> = covariates, one row per
#        track (score_entry.py COV_COLUMNS); tracks default to every row of <cov.tsv>. Use a new <out>.
#   bash run_entry.sh test <out> [C##M## ...]            (on Nibi) observed counts in place of
#        predictions (score_entry.py make-test), the same job, and `check` vs rungs L6 / L7.
#   bash run_entry.sh sacct <ids>                        (on Nibi) wall time and peak memory -> <out>/sacct.tsv
#
# Validation set (the 45 EIC validation experiments, truth known; score_entry.py --board validation):
#   bash run_entry.sh val-prep                           (on Nibi, once) covariates of every validation
#        track -> $VAL/cov.tsv (+ cov.tsv.missing.tsv), and the truth binned into the board's cache.
#   BOARD=validation bash run_entry.sh submit <pred> $VAL/cov.tsv <out> [C##M## ...]
#        as above, scored on the validation truth only, placed among MOESM6's round-1 entries.
#   bash run_entry.sh val-test <out> <C##M##:avg|C##M## ...>   (on Nibi) observed store counts in
#        place of predictions (make-test --from-cov), run, check-val; `:avg` also builds and scores
#        the round-1 Average baseline (score_entry.py average). DEP=<job id> waits for val-prep.
#
# Jobs are named eicx_entry_*; CPU (--gres=none), account def-maxwl, 2 CPUs each. h5py is not in
# scipy-stack: each job makes a venv in $SLURM_TMPDIR and installs it from the Alliance wheelhouse
# (--no-index), the pattern of the miniCANDI jobs; numpy and scipy stay scipy-stack's.
set -euo pipefail
ROOT=${ENTRY_ROOT:-/scratch/mforooz/eicx/entry}     # ENTRY_ROOT: another code + output root (t390 span_fix)
CODE=$ROOT/code
VAL=/scratch/mforooz/eicx/val
SE="python $CODE/tools/eic_entry/score_entry.py"
ENV='source /etc/profile >/dev/null 2>&1; module load StdEnv/2023 python/3.11 scipy-stack; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}; virtualenv -q --no-download --system-site-packages $SLURM_TMPDIR/venv; source $SLURM_TMPDIR/venv/bin/activate; pip install -q --no-index h5py'

case $(hostname -f) in *fir*) export EIC_CLUSTER=fir ;; *) export EIC_CLUSTER=nibi ;; esac

sub() {   # sub <out> <name> <mem> <time> <dependency or ""> <command>
  local out=$1 name=$2 mem=$3 time=$4 dep=$5; shift 5
  local d=(); [ -n "$dep" ] && d=(--dependency=afterany:$dep)
  local cpus=2   # Fir: at most 4000 MB per core, else the job goes to cpularge (days of queue)
  [ "$EIC_CLUSTER" = fir ] && cpus=$(( (${mem%G} * 1024 + 3999) / 4000 ))
  sbatch --parsable --account=def-maxwl --gres=none --job-name="$name" --cpus-per-task=$cpus \
    --mem="$mem" --time="$time" --output="$out/logs/%x_%j.out" "${d[@]}" \
    --wrap="$ENV; set -euo pipefail; $*"
}

collect() {   # collect <out> <dependency ids>   (BOARD=validation: the validation board)
  local J
  J=$(sub "$1" eicx_entry_collect 8G 0:30:00 "$2" "$SE collect --out $1 ${BOARD:+--board $BOARD}")
  echo "$J"
}

case "${1:?usage}" in
deploy)
  W=$(cd "$(dirname "$0")/../.." && pwd)
  H=${2:-nibi}                                       # deploy [nibi|fir]
  ssh -o BatchMode=yes $H "mkdir -p $CODE"
  (cd "$W" && rsync -a --relative -e 'ssh -o BatchMode=yes' \
    tools/eic_entry/ tools/onion_eval/__init__.py tools/onion_eval/eic_score.py \
    tools/onion_eval/signal_from_counts.py tools/t112/bin25.py \
    src/candi/store/pval_from_counts.py src/candi/bench/annotations.py \
    src/candi/bench/assets/ $H:$CODE/)
  ;;
submit)
  PRED=${2:?pred}; COV=${3:?cov}; OUT=${4:?out}; shift 4
  TR=("$@"); [ ${#TR[@]} -gt 0 ] || TR=($(awk -F'\t' 'NR > 1 {print $1}' "$COV"))
  mkdir -p "$OUT/logs"; cp "$COV" "$OUT/covariates.tsv"
  IDS=""
  MEM=48G; [ "${BOARD:-}" = validation ] && MEM=24G   # measured peaks: validation 16 GB, blind (with QN) 23-25 GB
  MEM=${SCORE_MEM:-$MEM}                               # SCORE_MEM: override (t390 slope_cal, 20G)
  for T in "${TR[@]}"; do
    J=$(sub "$OUT" eicx_entry_$T $MEM 2:00:00 "" "$SE run --pred $PRED --cov $OUT/covariates.tsv --track $T --out $OUT ${MODE:+--mode $MODE} ${BOARD:+--board $BOARD}")
    IDS=${IDS:+$IDS:}$J; echo "$T $J" | tee -a "$OUT/JOBS.txt"
  done
  echo "collect $(collect "$OUT" "$IDS")" | tee -a "$OUT/JOBS.txt"
  ;;
test)
  OUT=${2:?out}; shift 2
  TR=("$@"); [ ${#TR[@]} -gt 0 ] || TR=(C19M22 C40M17 C12M02)
  mkdir -p "$OUT/logs" "$OUT/cov"
  IDS=""
  for T in "${TR[@]}"; do
    J=$(sub "$OUT" eicx_entry_test_$T 48G 2:00:00 "" "$SE make-test --track $T --pred $OUT/pred --cov $OUT/cov/$T.tsv && \
$SE run --pred $OUT/pred --cov $OUT/cov/$T.tsv --track $T --out $OUT && $SE check --track $T --out $OUT")
    IDS=${IDS:+$IDS:}$J; echo "test $T $J" | tee -a "$OUT/JOBS.txt"
  done
  echo "collect $(collect "$OUT" "$IDS")" | tee -a "$OUT/JOBS.txt"
  ;;
val-prep)
  mkdir -p "$VAL/logs"
  J=$(sub "$VAL" eicx_val_prep 64G 3:00:00 "" "$SE val-cov --cov $VAL/cov.tsv && $SE prep-val --workers 2")
  echo "val-prep $J" | tee -a "$VAL/JOBS.txt"
  ;;
val-test)
  OUT=${2:?out}; shift 2
  mkdir -p "$OUT/logs" "$OUT/cov" "$OUT/avg"
  IDS=""
  for S in "$@"; do
    T=${S%%:*}
    J=$(sub "$OUT" eicx_val_test_$T 48G 2:00:00 "${DEP:-}" "$SE make-test --track $T --from-cov $VAL/cov.tsv --pred $OUT/pred --cov $OUT/cov/$T.tsv && \
$SE run --board validation --pred $OUT/pred --cov $OUT/cov/$T.tsv --track $T --out $OUT && $SE check-val --track $T --cov $OUT/cov/$T.tsv --out $OUT")
    IDS=${IDS:+$IDS:}$J; echo "val-test $T $J" | tee -a "$OUT/JOBS.txt"
    if [ "$S" != "$T" ]; then
      J=$(sub "$OUT" eicx_val_avg_$T 48G 2:00:00 "${DEP:-}" "$SE average --track $T --out $OUT/avg")
      echo "average $T $J" | tee -a "$OUT/JOBS.txt"
    fi
  done
  echo "collect $(BOARD=validation collect "$OUT" "$IDS")" | tee -a "$OUT/JOBS.txt"
  ;;
collect)   # collect <out> [dependency ids]: rebuild the board after scores were copied in
  OUT=${2:?out}; mkdir -p "$OUT/logs"
  echo "collect $(collect "$OUT" "${3:-}")" | tee -a "$OUT/JOBS.txt"
  ;;
sacct)
  OUT=${2:?out}; IDS=${3:?ids}
  sacct -j "$IDS" -X -P --units=G -o JobID,JobName%30,State,Elapsed,AllocCPUS,ReqMem > "$OUT/sacct.tsv"
  sacct -j "$IDS" -P --units=G -o JobID,MaxRSS | grep batch >> "$OUT/sacct.tsv" || true
  cat "$OUT/sacct.tsv"
  ;;
esac
