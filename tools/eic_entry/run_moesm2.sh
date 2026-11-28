#!/bin/bash
# MOESM2 board (original 2019 paired-end blind truth) for the t390 ladder rungs, on Nibi.
#
#   bash run_moesm2.sh deploy        (on the Mac) rsync the code subset to $CODE on Nibi
#   bash run_moesm2.sh submit        (on Nibi) cache links, the verify job, the rung score job
#   bash run_moesm2.sh sacct <ids>   (on Nibi) wall time and peak memory -> $OUT/sacct.tsv
#
# The binned original truth is Max's sprint 04 cache (truth/<exp> linked, checked bit for bit by
# check-bins on the 3 scored experiments); the msevar weights are rebuilt here from it and
# compared with Max's (moesm2.py varcheck). Jobs are named eicx_m2_*; CPU, account def-maxwl.
set -euo pipefail
OUT=/scratch/mforooz/eicx/moesm2
CODE=$OUT/code
LP="python $CODE/tools/eic_entry/ladder_probe.py"
EIC="$LP eic_score"
TRUTH=/scratch/maxwl/epi_imputation/data/synapse/blind_truth      # eic_score.ORIGINAL_TRUTH
MAXC=/scratch/maxwl/epi_imputation/s04/cache/pe
C=$OUT/score_cache/pe
PUB=/project/def-maxwl/mforooz/ONION/eic/published/13059_2023_2915_MOESM2_ESM.csv
RUNGS=/scratch/mforooz/eicx/ladder/rungs
EXPS=C19M22,C40M17,C12M02
ENV='source /etc/profile >/dev/null 2>&1; module load StdEnv/2023 python/3.11 scipy-stack; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}'

sub() {   # sub <name> <cpus> <mem> <time> <dependency or ""> <command>
  local name=$1 cpus=$2 mem=$3 time=$4 dep=$5; shift 5
  local d=(); [ -n "$dep" ] && d=(--dependency=afterany:$dep)
  sbatch --parsable --account=def-maxwl --gres=none --job-name="$name" --cpus-per-task="$cpus" \
    --mem="$mem" --time="$time" --output="$OUT/logs/%x_%j.out" "${d[@]}" \
    --wrap="$ENV; set -euo pipefail; $*"
}

case "${1:?usage}" in
deploy)
  W=$(cd "$(dirname "$0")/../.." && pwd)
  hpc run nibi "mkdir -p $CODE $OUT/logs"
  (cd "$W" && rsync -a --relative -e 'ssh -o BatchMode=yes' \
    tools/eic_entry/ tools/onion_eval/__init__.py tools/onion_eval/eic_score.py \
    tools/onion_eval/signal_from_counts.py tools/t112/bin25.py \
    src/candi/store/pval_from_counts.py src/candi/bench/annotations.py \
    src/candi/bench/assets/ nibi:$CODE/)
  ;;
submit)
  mkdir -p $OUT/logs $OUT/scores $OUT/verify $OUT/qc $C/truth
  for d in $MAXC/truth/C*M*; do [ -e $C/truth/$(basename $d) ] || ln -s $d $C/truth/$(basename $d); done
  V="$EIC check-bins --bigwigs $TRUTH --reference $MAXC/truth --exps $EXPS || echo check-bins exit \$?; \
$EIC verify-published --truth $TRUTH --cache $C --published $PUB --exps $EXPS --teams 0,100 \
--report-dir $OUT/verify --workers 1 || echo verify-published exit \$?; \
python $CODE/tools/eic_entry/moesm2.py varcheck --ours $C/var --max $MAXC/var --out $OUT/qc/varcheck.json"
  J1=$(sub eicx_m2_verify 2 48G 3:00:00 "" "$V")
  S=""
  for R in L3 L5 L6 L7; do
    S="$S $EIC score --truth $TRUTH --cache $C --pred $RUNGS/$R --entry $R --out $OUT/scores/$R.tsv --workers 1 &&"
  done
  J2=$(sub eicx_m2_rungs 2 48G 3:00:00 "$J1" "$S true")
  echo "verify $J1 rungs $J2" | tee -a $OUT/JOBS.txt
  ;;
sacct)
  sacct -j "${2:?ids}" -P --units=G -o JobID,JobName,State,Elapsed,AllocCPUS,MaxRSS,ReqMem > $OUT/sacct.tsv
  cat $OUT/sacct.tsv
  ;;
esac
