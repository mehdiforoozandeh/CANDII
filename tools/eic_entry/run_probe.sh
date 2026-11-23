#!/bin/bash
# Observed-data ladder probe (t390) on Nibi: submit the sbatch chain for one blind experiment.
#
#   bash run_probe.sh deploy                 (on the Mac) rsync the code subset to $CODE on Nibi
#   bash run_probe.sh setup                  (on Nibi) SE-truth score cache + entrant SE scores
#   bash run_probe.sh exp <C##M##> [dep]     (on Nibi) the chain for one experiment
#   bash run_probe.sh score <C##M##> [dep]   (on Nibi) only the scoring job
#   bash run_probe.sh sacct <jobids>         (on Nibi) wall time and peak memory per step
#
# Every job is named eicx_*; account def-maxwl, CPU (--gres=none). Outputs under $OUT.
set -euo pipefail
OUT=/scratch/mforooz/eicx/ladder
CODE=$OUT/code
LP="python $CODE/tools/eic_entry/ladder_probe.py"
EIC="$LP eic_score"
RES=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/results
TA=/project/def-maxwl/mforooz/t112_cf/tagaligns/ta
ZEN=/scratch/maxwl/epi_imputation/s04/zenodo
REF=/scratch/maxwl/epi_imputation/s04/qnorm/ref
TQN=/scratch/mforooz/onion/eic/truth/qnorm
TSE=/scratch/mforooz/onion/eic/truth/se_unnormalized
CQN=$OUT/score_cache/qnorm                 # truth/ and var/ link to the onion qnorm cache
CSE=$OUT/score_cache/se_unnormalized
ONION_CQN=/scratch/mforooz/onion/eic/score_cache/qnorm
SIZES=$TSE/C19M22.bigwig                   # chromosome sizes (GRCh38) from any truth bigWig
ENV='source /etc/profile >/dev/null 2>&1; module load StdEnv/2023 python/3.11 scipy-stack; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}'

# experiment table: treatment tagAlign | control tagAlign or none | d | peak cap | pipeline stem
conf() {
  case "$1" in
    C19M22) echo "ENCFF748TKZ.merged.srt.nodup.30M ENCFF433TZR.merged.srt.nodup 205 500000 \
macs2_signal_track_shard0_ENCFF748TKZ.merged.srt.nodup.30M_x_ENCFF433TZR.merged.srt.nodup \
call_peak_shard0_ENCFF748TKZ.merged.srt.nodup.30M_x_ENCFF433TZR.merged.srt.nodup.pval0.01.500K.bfilt" ;;
    C40M17) echo "ENCFF581NOL.merged.srt.nodup.30M ENCFF337JNL.merged.srt.nodup 200 500000 \
macs2_signal_track_shard0_ENCFF581NOL.merged.srt.nodup.30M_x_ENCFF337JNL.merged.srt.nodup \
call_peak_shard0_ENCFF581NOL.merged.srt.nodup.30M_x_ENCFF337JNL.merged.srt.nodup.pval0.01.500K.bfilt" ;;
    C12M02) echo "ENCFF211XVI.merged.srt.nodup.no_chrM_MT.50M none 150 300000 \
macs2_signal_track_shard0_ENCFF211XVI.merged.srt.nodup.no_chrM_MT.50M \
call_peak_shard0_ENCFF211XVI.merged.srt.nodup.no_chrM_MT.50M.pval0.01.300K.bfilt" ;;
    *) echo "unknown experiment $1" >&2; return 1 ;;
  esac
}

sub() {   # sub <name> <cpus> <mem> <time> <dependency or ""> <command>
  local name=$1 cpus=$2 mem=$3 time=$4 dep=$5; shift 5
  local d=(); [ -n "$dep" ] && d=(--dependency=afterok:$dep)
  sbatch --parsable --account=def-maxwl --gres=none --job-name="$name" --cpus-per-task="$cpus" \
    --mem="$mem" --time="$time" --output="$OUT/logs/%x_%j.out" "${d[@]}" \
    --wrap="$ENV; set -euo pipefail; $*"
}

score_job() {   # score_job <exp> <dependency>: every rung vs both truths, then the rung comparisons
  local E=$1 ALL=$2 R P N X Y fx fy SC="" CMP=""
  for R in L1 L2 L3 L4 L5 L6 L7; do
    SC="$SC $EIC score --truth $TQN --cache $CQN --pred $OUT/rungs/$R --entry $R --out $OUT/scores/qn/$R.tsv --workers 1 &&"
    SC="$SC $EIC score --truth $TSE --cache $CSE --pred $OUT/rungs/$R --entry $R --out $OUT/scores/se/$R.tsv --workers 1 &&"
  done
  CMP=""
  for P in "L5_vs_L3 L5 L3" "L6_vs_L5 L6 L5" "L3_vs_L1 L3 L1" "L4_vs_L2 L4 L2" "L7_vs_L4 L7 L4" "L6_vs_L3 L6 L3"; do
    read -r N X Y <<< "$P"
    fx=$OUT/rungs/$X/$E.npy; [ -e $OUT/rungs/$X/$E.bigwig ] && fx=$OUT/rungs/$X/$E.bigwig
    fy=$OUT/rungs/$Y/$E.npy; [ -e $OUT/rungs/$Y/$E.bigwig ] && fy=$OUT/rungs/$Y/$E.bigwig
    CMP="$CMP $LP compare --a $fx --b $fy --name $N --out $OUT/qc/${E}_$N.json &&"
  done
  sub eicx_score_$E 2 48G 6:00:00 "$ALL" "$SC $CMP true"
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
setup)
  mkdir -p $OUT/logs $CQN/scores $CSE
  [ -e $CQN/truth ] || ln -s $ONION_CQN/truth $CQN/truth
  [ -e $CQN/var ] || ln -s $ONION_CQN/var $CQN/var
  sub eicx_setup_se 16 64G 6:00:00 "" "$LP entrants --truth $TSE --cache $CSE --exps C19M22,C40M17,C12M02 --workers 4"
  ;;
exp)
  E=${2:?exp}; DEP=${3:-}
  read -r TRT CTL D CAP PSIG PPEAK < <(conf "$E")
  mkdir -p $OUT/{starts,rungs/{L1,L2,L3,L4,L5,L6,L7},peaks,qc,scores/{qn,se}}
  ln -sf $ZEN/blind_se_reprocessed/$E.bigwig $OUT/rungs/L1/$E.bigwig
  ln -sf $RES/$E/$PSIG.pval.signal.bigwig $OUT/rungs/L3/$E.bigwig
  A=${E:3}
  J_ST=$(sub eicx_starts_${E}_trt 2 32G 3:00:00 "" "$LP starts --ta $TA/$TRT.tagAlign.gz --out $OUT/starts/$TRT --sizes-from $SIZES")
  DEPS=$J_ST
  if [ "$CTL" != none ]; then
    if [ -e $OUT/starts/$CTL.json ]; then J_SC=""; else
      J_SC=$(sub eicx_starts_${E}_ctl 2 48G 3:00:00 "" "$LP starts --ta $TA/$CTL.tagAlign.gz --out $OUT/starts/$CTL --sizes-from $SIZES"); DEPS=$DEPS:$J_SC; fi
    CTLP=$OUT/starts/$CTL
  else CTLP=none; fi
  J_L2=$(sub eicx_L2_$E 2 64G 4:00:00 "" "$LP qnorm --x $ZEN/blind_se_reprocessed/$E.bigwig --mask $ZEN/blind_se_reprocessed/$E.bed.gz --ref $REF/ref_$A.npz --verify $ZEN/blind_se_reprocessed_qnorm/$E.bigwig --out $OUT/rungs/L2/$E.npy --sizes-from $SIZES")
  J_L4=$(sub eicx_L4_$E 2 64G 4:00:00 "" "$LP qnorm --x $RES/$E/$PSIG.pval.signal.bigwig --mask $RES/$E/$PPEAK.narrowPeak.gz --ref $REF/ref_$A.npz --out $OUT/rungs/L4/$E.npy --sizes-from $SIZES")
  J_L5=$(sub eicx_L5_$E 2 64G 4:00:00 "$DEPS" "$LP formula --trt $OUT/starts/$TRT --ctl $CTLP --d $D --out $OUT/rungs/L5/$E.npy --sizes-from $SIZES")
  J_L6=$(sub eicx_L6_$E 2 48G 3:00:00 "$DEPS" "$LP counts_route --trt $OUT/starts/$TRT --ctl $CTLP --d $D --out $OUT/rungs/L6/$E.npy --sizes-from $SIZES && \
RL=\$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))[\"mean_read_length\"])' $OUT/starts/$TRT.json) && \
$LP callpeaks --x $OUT/rungs/L6/$E.npy --out $OUT/peaks/${E}_L6.bed --max-gap \$RL --min-len $D --cap $CAP && \
$LP callpeaks --x $OUT/rungs/L3/$E.bigwig --out $OUT/peaks/${E}_L3rule.bed --max-gap \$RL --min-len $D --cap $CAP && \
$LP peakcheck --a $OUT/peaks/${E}_L3rule.bed --b $RES/$E/$PPEAK.narrowPeak.gz --out $OUT/qc/${E}_peakcheck_L3rule_vs_pipeline.json --sizes-from $SIZES && \
$LP peakcheck --a $OUT/peaks/${E}_L6.bed --b $RES/$E/$PPEAK.narrowPeak.gz --out $OUT/qc/${E}_peakcheck_L6_vs_pipeline.json --sizes-from $SIZES")
  J_L7=$(sub eicx_L7_$E 2 64G 4:00:00 "$J_L6" "$LP qnorm --x $OUT/rungs/L6/$E.npy --mask $OUT/peaks/${E}_L6.bed --ref $REF/ref_$A.npz --out $OUT/rungs/L7/$E.npy --sizes-from $SIZES")
  ALL=$J_L2:$J_L4:$J_L5:$J_L7${DEP:+:$DEP}
  J_SCORE=$(score_job $E $ALL)
  echo "exp $E: starts $DEPS L2 $J_L2 L4 $J_L4 L5 $J_L5 L6 $J_L6 L7 $J_L7 score $J_SCORE" | tee -a $OUT/JOBS.txt
  ;;
score)
  score_job ${2:?exp} "${3:-}"
  ;;
sacct)
  shift
  sacct -j "$1" -X --format=JobID,JobName%28,State,Elapsed,TotalCPU,ReqMem,AllocCPUS -P
  sacct -j "$1" --format=JobID,MaxRSS -P | grep batch
  ;;
esac
