#!/bin/bash
# Observed-data ladder probe (t390) on Nibi: submit the sbatch chain for one blind experiment.
#
#   bash run_probe.sh deploy                 (on the Mac) rsync the code subset to $CODE on Nibi
#   bash run_probe.sh setup                  (on Nibi) SE-truth score cache + entrant SE scores
#   bash run_probe.sh exp <C##M##> [dep]     (on Nibi) the chain for one experiment
#   bash run_probe.sh score <C##M##> [dep]   (on Nibi) only the scoring job
#   bash run_probe.sh l4o <C##M##>          (on Nibi) extra rung: L3 normalized with the overlap peaks
#   RUNGS="L4o" bash run_probe.sh score <C##M##> <dep>   score only those rungs
#   bash run_probe.sh p2 <C##M##>           (on Nibi) probe 2: L7t, L7o, L6f, L7f, checks and scoring
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
BL=/scratch/mforooz/EIC_REPRO/003/refcache/f183dcba5d34f959d8b55ed438ee2e22/ENCFF356LFX.bed.gz   # the pipelines' blacklist (metadata.json call inputs)
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
  for R in ${RUNGS:-L1 L2 L3 L4 L5 L6 L7}; do
    SC="$SC $EIC score --truth $TQN --cache $CQN --pred $OUT/rungs/$R --entry $R --out $OUT/scores/qn/$R.tsv --workers 1 &&"
    SC="$SC $EIC score --truth $TSE --cache $CSE --pred $OUT/rungs/$R --entry $R --out $OUT/scores/se/$R.tsv --workers 1 &&"
  done
  CMP=""
  [ -n "${RUNGS:-}" ] && { sub eicx_score_$E 2 48G 6:00:00 "$ALL" "$SC true"; return; }
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
l4o)   # L4o: L3 normalized with the pipeline's pseudoreplicate-overlap peaks (the set Jacob's .bed.gz matches)
  E=${2:?exp}; A=${E:3}
  read -r TRT CTL D CAP PSIG PPEAK < <(conf "$E")
  mkdir -p $OUT/rungs/L4o
  PO=$RES/$E/overlap_pr_shard0_rep1-pr1_vs_rep1-pr2.overlap.bfilt.narrowPeak.gz
  J=$(sub eicx_L4o_$E 2 64G 3:00:00 "" "$LP qnorm --x $RES/$E/$PSIG.pval.signal.bigwig --mask $PO --ref $REF/ref_$A.npz --out $OUT/rungs/L4o/$E.npy --sizes-from $SIZES && \
$LP peakcheck --a $ZEN/blind_se_reprocessed/$E.bed.gz --b $PO --out $OUT/qc/${E}_peakcheck_jacob_vs_overlap.json --sizes-from $SIZES && \
$LP compare --a $OUT/rungs/L4o/$E.npy --b $OUT/rungs/L2/$E.npy --name L4o_vs_L2 --out $OUT/qc/${E}_L4o_vs_L2.json")
  echo $J
  ;;
score)
  score_job ${2:?exp} "${3:-}"
  ;;
p2)   # probe 2: L7t, L7o, L6f, L7f for one experiment (per-experiment rung folders: no scoring race)
  E=${2:?exp}; A=${E:3}
  read -r TRT CTL D CAP PSIG PPEAK < <(conf "$E")
  X=$OUT/rungs2/$E; H=$OUT/halves; Q=$OUT/qc
  mkdir -p $X/{L6,L6f,L7t,L7o,L7f,L7chk} $H $OUT/scores2/{qn,se} $OUT/peaks $Q
  [ "$CTL" != none ] && CTLP=$OUT/starts/$CTL || CTLP=none
  PO=$RES/$E/overlap_pr_shard0_rep1-pr1_vs_rep1-pr2.overlap.bfilt.narrowPeak.gz
  PP1=$(ls $RES/$E/call_peak_pr1_shard0_*.bfilt.narrowPeak.gz); PP2=$(ls $RES/$E/call_peak_pr2_shard0_*.bfilt.narrowPeak.gz)
  PPO=$RES/$E/$PPEAK.narrowPeak.gz
  RL=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["mean_read_length"])' $OUT/starts/$TRT.json)
  CP="--max-gap $RL --min-len $D --cap $CAP"
  J_SP=$(sub eicx_p2split_$E 2 40G 2:00:00 "" "$LP split --ta $TA/$TRT.tagAlign.gz --out $H/$TRT && \
$LP starts --ta $H/$TRT.pr1.tagAlign --out $OUT/starts/$TRT.pr1 --sizes-from $SIZES && \
$LP starts --ta $H/$TRT.pr2.tagAlign --out $OUT/starts/$TRT.pr2 --sizes-from $SIZES && rm -f $H/$TRT.pr1.tagAlign $H/$TRT.pr2.tagAlign")
  J_FU=$(sub eicx_p2full_$E 2 40G 2:00:00 "" "$LP counts_route --trt $OUT/starts/$TRT --ctl $CTLP --d $D --out $X/L6/$E.npy --sizes-from $SIZES && \
$LP arrdiff --a $X/L6/$E.npy --b $OUT/rungs/L6/$E.npy --out $Q/${E}_L6rerun_arrdiff.json && \
$LP callpeaks --x $X/L6/$E.npy --out $OUT/peaks/${E}_L6p2.bed $CP && \
$LP peakcheck --a $OUT/peaks/${E}_L6p2.bed --b $OUT/peaks/${E}_L6.bed --out $Q/${E}_peakcheck_L6p2_vs_L6.json --sizes-from $SIZES && \
$LP fragpile_route --trt $OUT/starts/$TRT --ctl $CTLP --d $D --out $X/L6f/$E.npy --sizes-from $SIZES && \
$LP callpeaks --x $X/L6f/$E.npy --out $OUT/peaks/${E}_L6f.bed $CP")
  HC=""
  for h in pr1 pr2; do
    HC="$HC $LP counts_route --trt $OUT/starts/$TRT.$h --ctl $CTLP --d $D --out $H/${E}_L6_$h.npy --sizes-from $SIZES && \
$LP callpeaks --x $H/${E}_L6_$h.npy --out $OUT/peaks/${E}_L6_$h.bed $CP && \
$LP fragpile_route --trt $OUT/starts/$TRT.$h --ctl $CTLP --d $D --out $H/${E}_L6f_$h.npy --sizes-from $SIZES && \
$LP callpeaks --x $H/${E}_L6f_$h.npy --out $OUT/peaks/${E}_L6f_$h.bed $CP &&"
  done
  J_HA=$(sub eicx_p2half_$E 2 40G 3:00:00 "$J_SP:$J_FU" "$HC \
$LP overlap --peak1 $OUT/peaks/${E}_L6_pr1.bed --peak2 $OUT/peaks/${E}_L6_pr2.bed --pooled $OUT/peaks/${E}_L6p2.bed --blacklist $BL --out $OUT/peaks/${E}_L7o.bed && \
$LP overlap --peak1 $OUT/peaks/${E}_L6f_pr1.bed --peak2 $OUT/peaks/${E}_L6f_pr2.bed --pooled $OUT/peaks/${E}_L6f.bed --blacklist $BL --out $OUT/peaks/${E}_L7f.bed && \
$LP overlap --peak1 $PP1 --peak2 $PP2 --pooled $PPO --blacklist $BL --out $OUT/peaks/${E}_port_on_pipeline.bed && \
$LP peakcheck --a $OUT/peaks/${E}_port_on_pipeline.bed --b $PO --out $Q/${E}_peakcheck_port_vs_overlap.json --sizes-from $SIZES && \
$LP peakcheck --a $OUT/peaks/${E}_L7o.bed --b $PO --out $Q/${E}_peakcheck_L7o_vs_overlap.json --sizes-from $SIZES && \
$LP peakcheck --a $OUT/peaks/${E}_L7f.bed --b $PO --out $Q/${E}_peakcheck_L7f_vs_overlap.json --sizes-from $SIZES && \
$LP peakcheck --a $OUT/peaks/${E}_L6f.bed --b $PPO --out $Q/${E}_peakcheck_L6f_vs_pipeline.json --sizes-from $SIZES")
  J_CT=$(sub eicx_p2ctr_$E 2 8G 0:30:00 "$J_HA" "bash $CODE/tools/eic_entry/overlap_container_check.sh $E $OUT $PP1 $PP2 $PPO $BL")
  J_QT=$(sub eicx_p2L7t_$E 2 64G 3:00:00 "$J_FU" "$LP qnorm --x $X/L6/$E.npy --mask $PO --ref $REF/ref_$A.npz --out $X/L7t/$E.npy --sizes-from $SIZES")
  J_QO=$(sub eicx_p2L7o_$E 2 64G 3:00:00 "$J_HA" "$LP qnorm --x $X/L6/$E.npy --mask $OUT/peaks/${E}_L7o.bed --ref $REF/ref_$A.npz --out $X/L7o/$E.npy --sizes-from $SIZES")
  J_QF=$(sub eicx_p2L7f_$E 2 64G 3:00:00 "$J_HA" "$LP qnorm --x $X/L6f/$E.npy --mask $OUT/peaks/${E}_L7f.bed --ref $REF/ref_$A.npz --out $X/L7f/$E.npy --sizes-from $SIZES")
  ln -sf $OUT/rungs/L7/$E.npy $X/L7chk/$E.npy
  SC=""
  for R in L7t L7o L6f L7f; do
    SC="$SC $EIC score --truth $TQN --cache $CQN --pred $X/$R --entry $R --out $OUT/scores2/qn/${E}_$R.tsv --workers 1 &&"
    SC="$SC $EIC score --truth $TSE --cache $CSE --pred $X/$R --entry $R --out $OUT/scores2/se/${E}_$R.tsv --workers 1 &&"
  done
  SC="$SC $EIC score --truth $TQN --cache $CQN --pred $X/L7chk --entry L7 --out $OUT/scores2/qn/${E}_L7chk.tsv --workers 1 &&"
  CMP=""
  r() { case $1 in L1|L3) echo $OUT/rungs/$1/$E.bigwig ;; L2|L4|L4o|L5|L6|L7) echo $OUT/rungs/$1/$E.npy ;; *) echo $X/$1/$E.npy ;; esac; }
  for P in "L6f L5" "L6f L3" "L7t L4o" "L7o L4o" "L7f L4o" "L7t L2" "L7o L2" "L7f L2" "L7o L7" "L7t L7"; do
    read -r P1 P2 <<< "$P"
    CMP="$CMP $LP compare --a $(r $P1) --b $(r $P2) --name ${P1}_vs_$P2 --out $Q/${E}_${P1}_vs_$P2.json &&"
  done
  PL="$LP placement --cache $ONION_CQN --truth-name qn --rung L7t=$OUT/scores2/qn/${E}_L7t.tsv L7o=$OUT/scores2/qn/${E}_L7o.tsv L6f=$OUT/scores2/qn/${E}_L6f.tsv L7f=$OUT/scores2/qn/${E}_L7f.tsv --out $OUT/scores2/placement_qn_$E.tsv && \
$LP placement --cache $CSE --truth-name se --rung L7t=$OUT/scores2/se/${E}_L7t.tsv L7o=$OUT/scores2/se/${E}_L7o.tsv L6f=$OUT/scores2/se/${E}_L6f.tsv L7f=$OUT/scores2/se/${E}_L7f.tsv --out $OUT/scores2/placement_se_$E.tsv"
  J_SC=$(sub eicx_p2score_$E 2 48G 4:00:00 "$J_QT:$J_QO:$J_QF:$J_FU" "$SC $CMP $PL")
  echo "p2 $E: split $J_SP full $J_FU half $J_HA ctr $J_CT L7t $J_QT L7o $J_QO L7f $J_QF score $J_SC" | tee -a $OUT/JOBS.txt
  ;;
pp)
  # post-prediction pileup (t390 postpile): variants A B C D of `postpile_route`, each vs L5 and L6f.
  # CHECK=1 also runs the `exact` variant and holds it bit-equal to L6f (the conversion identity).
  E=${2:?exp}
  read -r TRT CTL D CAP PSIG PPEAK < <(conf "$E")
  [ "$CTL" != none ] && CTLP=$OUT/starts/$CTL || CTLP=none
  X=$OUT/postpile/$E; Q=$OUT/qc; mkdir -p $X $Q
  L6F=$OUT/rungs2/$E/L6f/$E.npy; L5=$OUT/rungs/L5/$E.npy
  CMD="true"
  if [ "${CHECK:-0}" = 1 ]; then
    CMD="$CMD && $LP postpile_route --variant exact --trt $OUT/starts/$TRT --ctl $CTLP --d $D --out $X/exact/$E.npy --sizes-from $SIZES && \
$LP arrdiff --a $X/exact/$E.npy --b $L6F --out $Q/${E}_pp_exact_vs_L6f_arrdiff.json"
  fi
  VS="A B C D"; [ "$CTL" = none ] && VS="A B D"      # DNase-seq: C = B by definition (no strand rule)
  for V in $VS; do
    CMD="$CMD && $LP postpile_route --variant $V --trt $OUT/starts/$TRT --ctl $CTLP --d $D --out $X/$V/$E.npy --sizes-from $SIZES && \
$LP compare --a $X/$V/$E.npy --b $L5 --name pp${V}_vs_L5 --out $Q/${E}_pp${V}_vs_L5.json && \
$LP compare --a $X/$V/$E.npy --b $L6F --name pp${V}_vs_L6f --out $Q/${E}_pp${V}_vs_L6f.json"
  done
  CMD="$CMD && $LP compare --a $OUT/rungs/L6/$E.npy --b $L6F --name L6_vs_L6f --out $Q/${E}_L6_vs_L6f.json"
  J=$(sub eicx_pp_$E 2 40G 3:00:00 "" "$CMD")
  echo "pp $E: $J" | tee -a $OUT/JOBS.txt
  ;;
sacct)
  shift
  sacct -j "$1" -X --format=JobID,JobName%28,State,Elapsed,TotalCPU,ReqMem,AllocCPUS -P
  sacct -j "$1" --format=JobID,MaxRSS -P | grep batch
  ;;
esac
