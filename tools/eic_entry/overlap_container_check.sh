#!/bin/bash
# Probe 2 check: run the ENCODE pipeline's own encode_task_overlap.py (in the pipeline's container)
# on the same inputs as the Python port in ladder_probe.py `overlap`, and compare the interval sets.
#
#   bash overlap_container_check.sh <C##M##> <OUT> <pipeline pr1 peaks> <pr2 peaks> <pooled peaks> <blacklist>
#
# Three cases: (1) our L6 half peaks -> L7o, (2) our L6f half peaks -> L7f, (3) the pipeline's own
# bfilt pr1 / pr2 / pooled peaks. Writes <OUT>/qc/<exp>_container_overlap.json.
set -uo pipefail
E=$1; OUT=$2; PP1=$3; PP2=$4; PPO=$5; BL=$6
PIPE=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline
CHRSZ=/scratch/mforooz/EIC_REPRO/003/refcache/c52f52c7bfa357f55a39b1de7e4d0b0c/GRCh38_EBV.chrom.sizes.tsv
if [ "$E" = C12M02 ]; then SIF=$PIPE/sif/atac-seq-pipeline_v2.2.3.sif; SRC=$PIPE/atac-seq-pipeline/src
else SIF=$PIPE/sif/chip-seq-pipeline_v2.2.2.sif; SRC=$PIPE/chip-seq-pipeline2/src; fi
module load StdEnv/2023 apptainer/1.3.5
T=$OUT/ctr/$E; rm -rf $T; mkdir -p $T
np() { awk 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,".",$7,-1,-1,-1}' "$1" | gzip -nc > "$2"; }
iv() { zcat -f "$1" | cut -f1-3 | sort -u | md5sum | cut -d' ' -f1; }
nr() { zcat -f "$1" | cut -f1-3 | sort -u | wc -l; }
run() {  # run <case> <p1> <p2> <pooled>
  mkdir -p $T/$1
  apptainer exec -B /project,/scratch $SIF python3 $SRC/encode_task_overlap.py "$2" "$3" "$4" \
    --prefix $1 --peak-type narrowPeak --nonamecheck --blacklist $BL \
    --regex-bfilt-peak-chr-name 'chr[\dXY]+' --chrsz $CHRSZ --out-dir $T/$1 > $T/$1.log 2>&1
  echo $T/$1/$1.overlap.bfilt.narrowPeak.gz
}
for k in L6 L6f; do
  np $OUT/peaks/${E}_${k}_pr1.bed $T/${k}_pr1.np.gz; np $OUT/peaks/${E}_${k}_pr2.bed $T/${k}_pr2.np.gz
done
np $OUT/peaks/${E}_L6p2.bed $T/L6_pooled.np.gz; np $OUT/peaks/${E}_L6f.bed $T/L6f_pooled.np.gz
O1=$(run ours_L6 $T/L6_pr1.np.gz $T/L6_pr2.np.gz $T/L6_pooled.np.gz)
O2=$(run ours_L6f $T/L6f_pr1.np.gz $T/L6f_pr2.np.gz $T/L6f_pooled.np.gz)
O3=$(run pipeline $PP1 $PP2 $PPO)
PO=$(dirname $PPO)/overlap_pr_shard0_rep1-pr1_vs_rep1-pr2.overlap.bfilt.narrowPeak.gz
cat > $OUT/qc/${E}_container_overlap.json <<J
{"experiment": "$E", "sif": "$SIF", "src": "$SRC/encode_task_overlap.py",
 "L7o": {"container_n": $(nr $O1), "port_n": $(nr $OUT/peaks/${E}_L7o.bed), "identical": $([ "$(iv $O1)" = "$(iv $OUT/peaks/${E}_L7o.bed)" ] && echo true || echo false)},
 "L7f": {"container_n": $(nr $O2), "port_n": $(nr $OUT/peaks/${E}_L7f.bed), "identical": $([ "$(iv $O2)" = "$(iv $OUT/peaks/${E}_L7f.bed)" ] && echo true || echo false)},
 "pipeline_inputs": {"container_n": $(nr $O3), "port_n": $(nr $OUT/peaks/${E}_port_on_pipeline.bed), "pipeline_overlap_n": $(nr $PO),
   "container_eq_port": $([ "$(iv $O3)" = "$(iv $OUT/peaks/${E}_port_on_pipeline.bed)" ] && echo true || echo false),
   "container_eq_pipeline_overlap": $([ "$(iv $O3)" = "$(iv $PO)" ] && echo true || echo false)}}
J
cat $OUT/qc/${E}_container_overlap.json
