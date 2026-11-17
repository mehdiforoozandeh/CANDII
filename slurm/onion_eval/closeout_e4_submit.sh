#!/bin/bash
# Close-out E4: the replicate ladder and the crossed mixed model on the second-experiment add-on
# tables of D2. Run on the laptop; every job is submitted over ssh. DRY RUN BY DEFAULT: DRY_RUN=0
# submits. Item file:
# cruxvault/INSTRUCTIONS/peeling_the_onion/closeout/closeout E4 complete the replicate ladder.md
#
#   bash slurm/onion_eval/closeout_e4_submit.sh <stage> [nibi|fir]
#
# CPU only; writes only under $E (and its logs). Reads /scratch/mforooz/onion/measurements (counts,
# pr1/pr2 halves): it never reads onion/work, onion/fastq, t112_cf or EIC_REPRO.
# Stages (each skips a job whose output exists or whose name is queued or running):
#   smoke   4 jobs on SMOKE_ASSAYS (default H3K36me3): pair list, layers --pairs-out, mixed model
#           (band fine, --second-experiments, lab, 20 bootstrap draws), report. Allocated at most
#           1x0.5 + 4x1 + 2x2 + 1x0.5 = 9 CPU-h.
#   layers  3 jobs: pair list (1 CPU x 0.5 h), design (1 x 0.5 h), layers on the whole pair list
#           with --pairs-out (4 CPU x 3 h, 32000M; t305 measured 1 h 19 min on 4 CPUs for 3,908 pairs,
#           this list holds 3,979) -> 13 CPU-h allocated at most, about 6 expected.
#   mixed   4 jobs, one per Haar band (fine, peak, domain, broad): mixed_model.py on every assay
#           with --second-experiments --extra-factors lab, 200 bootstrap draws. MIXED_TIME and
#           MIXED_CPUS set the request (default 8 h x 4 CPU = 32 CPU-h per job, 128 in all at most);
#           set them from the smoke job's sacct before submitting.
#   report  1 job (1 CPU x 0.5 h): ladder.py report over the layers pair values, layers.tsv
#           (telescoping check), the 4 mixed-model tables and design.tsv -> $E/ladder/.
#
# Names: O=/project/def-maxwl/mforooz/ONION, E=$O/CO-E4, K=<kit root>/$KIT_ID,
# TX = the D2 add-on tables, EVL = the D2 layers eval set (holds no pseudoreplicate pair: the pair list
# adds them, ladder.py pairlist).
set -euo pipefail
STAGE="${1:?usage: closeout_e4_submit.sh smoke|layers|mixed|report [nibi|fir]}"
WHERE="${2:-nibi}"
DRY="${DRY_RUN:-1}"
KIT_ID="${KIT_ID:-COE4}"
O=/project/def-maxwl/mforooz/ONION
E="${E:-$O/CO-E4}"
K=/project/def-maxwl/mforooz/onion_kit/$KIT_ID
TX="${TX:-$O/CO-D2/tables/eic_v1_2x}"
EVL="${EVL:-$O/CO-D2/eval/eic_v1_2x/evalset_layers.json}"
STRATA=$O/eval/strata/masks.h5
SMOKE_ASSAYS="${SMOKE_ASSAYS:-H3K36me3}"
MIXED_TIME="${MIXED_TIME:-08:00:00}"
MIXED_CPUS="${MIXED_CPUS:-4}"
BANDS="fine peak domain broad"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15)
SB="module load python/3.10.13 >/dev/null 2>&1; sbatch --parsable --account=def-maxwl --gres=none"
[ "$WHERE" = fir ] && SB+=" --exclude=fc10713,fc10715"
RUN="$K/slurm/onion_kit/run.sh $K"

remote() {   # run one submit command on the cluster and print its job id, or print it (dry run)
  if [ "$DRY" = 1 ]; then echo "[dry-run $WHERE] $1" >&2; echo "DRY$RANDOM"; else "${SSH[@]}" "$WHERE" "$1"; fi
}
done_or_queued() {   # 0 (skip) when <file> exists or a job named <name> is queued or running
  "${SSH[@]}" "$WHERE" "test -e $1 || [ -n \"\$(squeue -h -u \$USER -n $2)\" ]" 2>/dev/null
}
need() {
  for p in "$@"; do "${SSH[@]}" "$WHERE" "test -e $p" || { echo "missing on $WHERE: $p" >&2; exit 1; }; done
}
dep() { [ -n "$1" ] && echo "--dependency=afterok:$1" || true; }

need "$K/.kit_stamp" "$TX/pairs.tsv" "$TX/measurements.tsv" "$EVL" "$STRATA"
echo "kit $K: $("${SSH[@]}" "$WHERE" "head -1 $K/.kit_stamp")"
# no output under /scratch: refuse a root there
case "$E" in /scratch/*) echo "E must not be on /scratch" >&2; exit 2 ;; esac

case "$STAGE" in
smoke)
  S=$E/smoke
  remote "mkdir -p $S $E/logs" >/dev/null
  j1=""; j2=""; j3=""
  if done_or_queued "$S/pairlist.json" co_e4_smoke_pl; then echo "skip smoke pair list"; else
    j1=$(remote "$SB --job-name=co_e4_smoke_pl --time=00:30:00 --cpus-per-task=1 --mem=4000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/ladder.py pairlist --evalset $EVL --tables $TX --assays $SMOKE_ASSAYS --out $S/pairlist.json --changes $S/pairlist_changes.tsv"); echo "pair list: $j1"; fi
  if done_or_queued "$S/layers.tsv" co_e4_smoke_layers; then echo "skip smoke layers"; else
    j2=$(remote "$SB --job-name=co_e4_smoke_layers $(dep "$j1") --time=01:00:00 --cpus-per-task=4 --mem=16000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/layers.py run --tables $TX --evalset $S/pairlist.json --strata $STRATA --out $S/layers.tsv --pairs-out $S/pair_values.tsv.gz"); echo "layers: $j2"; fi
  if done_or_queued "$S/mixed_fine.tsv" co_e4_smoke_mm; then echo "skip smoke mixed model"; else
    j3=$(remote "$SB --job-name=co_e4_smoke_mm --time=02:00:00 --cpus-per-task=2 --mem=16000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/mixed_model.py run --tables $TX --evalset $EVL --band fine --assays $SMOKE_ASSAYS --second-experiments --extra-factors lab --n-boot 20 --out $S/mixed_fine.tsv"); echo "mixed model: $j3"; fi
  d=$(echo "$j2 $j3" | xargs | tr ' ' ':')
  if done_or_queued "$S/ladder/ladder.md" co_e4_smoke_report; then echo "skip smoke report"; else
    j4=$(remote "$SB --job-name=co_e4_smoke_report $(dep "$d") --time=00:30:00 --cpus-per-task=1 --mem=4000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/ladder.py report --pairs $S/pair_values.tsv.gz --layers $S/layers.tsv --mixed $S/mixed_fine.tsv --n-boot 1000 --out-dir $S/ladder"); echo "report: $j4"; fi
  ;;
layers)
  L=$E/layers
  remote "mkdir -p $L $E/logs" >/dev/null
  j1=""
  if done_or_queued "$L/pairlist.json" co_e4_pairlist; then echo "skip pair list"; else
    j1=$(remote "$SB --job-name=co_e4_pairlist --time=00:30:00 --cpus-per-task=1 --mem=4000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/ladder.py pairlist --evalset $EVL --tables $TX --out $L/pairlist.json --changes $L/pairlist_changes.tsv"); echo "pair list: $j1"; fi
  if done_or_queued "$L/design.tsv" co_e4_design; then echo "skip design"; else
    jd=$(remote "$SB --job-name=co_e4_design --time=00:30:00 --cpus-per-task=1 --mem=4000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/ladder.py design --tables $TX --out $L/design.tsv"); echo "design: $jd"; fi
  if done_or_queued "$L/layers.tsv" co_e4_layers; then echo "skip layers"; else
    j2=$(remote "$SB --job-name=co_e4_layers $(dep "$j1") --time=03:00:00 --cpus-per-task=4 --mem=32000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/layers.py run --tables $TX --evalset $L/pairlist.json --strata $STRATA --out $L/layers.tsv --pairs-out $L/pair_values.tsv.gz"); echo "layers: $j2"; fi
  ;;
mixed)
  M=$E/mixed
  remote "mkdir -p $M $E/logs" >/dev/null
  for b in $BANDS; do
    if done_or_queued "$M/mixed_$b.tsv" "co_e4_mm_$b"; then echo "skip mixed $b"; continue; fi
    j=$(remote "$SB --job-name=co_e4_mm_$b --time=$MIXED_TIME --cpus-per-task=$MIXED_CPUS --mem=32000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/mixed_model.py run --tables $TX --evalset $EVL --band $b --second-experiments --extra-factors lab --n-boot 200 --out $M/mixed_$b.tsv"); echo "mixed $b: $j"
  done
  ;;
report)
  need "$E/layers/pair_values.tsv.gz" "$E/layers/layers.tsv" "$E/layers/design.tsv"
  MX=""   # the mixed-model tables that exist (the report runs without them, and says so)
  for b in $BANDS; do "${SSH[@]}" "$WHERE" "test -e $E/mixed/mixed_$b.tsv" && MX+=" $E/mixed/mixed_$b.tsv"; done
  [ -n "$MX" ] && MX="--mixed$MX" || echo "no mixed-model table yet: the report has no mixed-model section"
  if done_or_queued "$E/ladder/ladder.md" co_e4_report; then echo "skip report"; else
    j=$(remote "$SB --job-name=co_e4_report --time=00:30:00 --cpus-per-task=1 --mem=8000M --output=$E/logs/%x_%j.out $RUN tools/onion_eval/ladder.py report --pairs $E/layers/pair_values.tsv.gz --layers $E/layers/layers.tsv --design $E/layers/design.tsv $MX --n-boot 1000 --out-dir $E/ladder"); echo "report: $j"; fi
  ;;
*) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
