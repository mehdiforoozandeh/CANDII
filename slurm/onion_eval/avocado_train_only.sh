#!/bin/bash
# C16 (plan t370, D10): the ready command for a TRAINING-ONLY Avocado, scored on the EIC validation
# tracks beside miniCANDI. DRY_RUN=1 is the default. A real submission (DRY_RUN=0) is the foreman's,
# after a PI yes (plan, Decisions needed, item 2; about 86 GPU-h). Run on a Fir login node:
#
#   bash slurm/onion_eval/avocado_train_only.sh               # stage + print the sbatch chain
#   DRY_RUN=0 bash slurm/onion_eval/avocado_train_only.sh     # stage + submit it (PI yes only)
#
# It reuses experiment 005's own scripts unchanged in logic (Fir, read-only:
# $E/scripts/hpc_{prepare_grid.py,bin_tracks.slurm,train.slurm,predict.slurm,make_bigwig.slurm}).
# Those .slurm files hard-code Max's Nibi workspace (/scratch/maxwl/epi_imputation/{005_ws,005,
# data}) and log folder, so the script copies them into $ROOT/ws and rewrites only those paths
# (and the blind list name: the prediction targets are the validation tracks). What changes:
#   - the data: $ROOT/data/synapse/train_plus_val holds symlinks to the training bigWigs ONLY
#     (DATA_EIC_SYNAPSE/training_data), so no validation track can be read;
#   - the bridge: eic_bridge.csv with split V renamed VAL_HELDOUT, so hpc_bin_tracks.py (which
#     keeps split T and V) bins the training columns only, while hpc_train.py's index space (every
#     bridge row) still gives the validation cell types an embedding, learned from their training
#     tracks;
#   - the predictions: hpc_predict.py with --blind-list = the validation experiments
#     (inputs/bridge/val_experiments.txt), then hpc_make_bigwig.py -> $ROOT/sc/pred_bw/d3/<exp>.bigwig.
# Training arguments are 005's (scripts/04_train.sh): dataset d3, shared mode on chr20 with 120
# epochs, then genome mode on the 23 chromosomes with 60 epochs; batch 1024 positions; lr 1e-3;
# genome lr 1e-2; 11 h cap; one full H100 (gpu:h100:1).
# Variables: AVO_ROOT (default /scratch/mforooz/onion/avocado_train_only), EIC_DATA (default
# /project/def-maxwl/mforooz/DATA_EIC_SYNAPSE), AVO_EXP (005's folder), ONION_EXCLUDE (GPU and CPU
# jobs get --exclude), LOGS (default /project/def-maxwl/mforooz/ONION/logs).

set -euo pipefail
DRY_RUN="${DRY_RUN:-1}"
E="${AVO_EXP:-/project/def-maxwl/mforooz/MAX_EPI_IMPUTATION/experiments/005_2026-07-29_avocado_cross_dataset}"
D="${EIC_DATA:-/project/def-maxwl/mforooz/DATA_EIC_SYNAPSE}"
ROOT="${AVO_ROOT:-/scratch/mforooz/onion/avocado_train_only}"
LOGS="${LOGS:-/project/def-maxwl/mforooz/ONION/logs}"
WS="$ROOT/ws"; SC="$ROOT/sc"; DATA="$ROOT/data"
TRACKS="$DATA/synapse/train_plus_val"
EXCLUDE="${ONION_EXCLUDE:-}"; EXCLUDE="${EXCLUDE//[[:space:]]/}"

for f in scripts/hpc_env.sh scripts/hpc_prepare_grid.py scripts/hpc_bin_tracks.slurm \
         scripts/hpc_train.slurm scripts/hpc_predict.slurm scripts/hpc_make_bigwig.slurm \
         output/inputs/bridge/eic_bridge.csv; do
  [ -f "$E/$f" ] || { echo "missing $E/$f" >&2; exit 2; }
done
[ -d "$D/training_data" ] || { echo "missing $D/training_data" >&2; exit 2; }

# ---------------------------------------------------------------- stage (idempotent, no compute)
mkdir -p "$WS/scripts" "$WS/inputs/bridge" "$WS/inputs/grid" "$SC" "$TRACKS" "$LOGS"
ls "$D/training_data" | sed -n 's/^\(C[0-9][0-9]M[0-9][0-9]\)\.bigwig$/\1/p' | sort \
  > "$WS/inputs/train_only_experiments.txt"
NT=$(wc -l < "$WS/inputs/train_only_experiments.txt" | tr -d ' ')
while read -r x; do ln -sfn "$D/training_data/$x.bigwig" "$TRACKS/$x.bigwig"; done \
  < "$WS/inputs/train_only_experiments.txt"
cp "$E"/scripts/*.py "$E"/scripts/*.sh "$E"/scripts/*.slurm "$WS/scripts/"
for f in "$WS"/scripts/*.slurm; do
  sed -i.bak -e "s#/scratch/maxwl/epi_imputation/005_ws#$WS#g" \
             -e "s#/scratch/maxwl/epi_imputation/005#$SC#g" \
             -e "s#/scratch/maxwl/epi_imputation/data#$DATA#g" \
             -e "s#/scratch/maxwl/logs#$LOGS#g" \
             -e "s#blind_experiments\.txt#val_experiments.txt#g" "$f"
  rm -f "$f.bak"
done
awk -F, 'BEGIN{OFS=","} NR > 1 && $4 == "V" {$4 = "VAL_HELDOUT"} {print}' \
  "$E/output/inputs/bridge/eic_bridge.csv" > "$WS/inputs/bridge/eic_bridge.csv"
awk -F, 'NR > 1 && $4 == "VAL_HELDOUT" {print $1}' "$WS/inputs/bridge/eic_bridge.csv" \
  > "$WS/inputs/bridge/val_experiments.txt"
awk -F, 'NR > 1 && $4 == "T" {print $1}' "$WS/inputs/bridge/eic_bridge.csv" | sort \
  > "$WS/inputs/bridge/train_experiments.txt"
NV=$(wc -l < "$WS/inputs/bridge/val_experiments.txt" | tr -d ' ')
echo "chr20" > "$WS/inputs/grid/chrom20.txt"

# Consistency: the bridge's training rows are exactly the training folder; every validation cell
# type and assay has a training track (else Avocado has no embedding for it).
if ! cmp -s "$WS/inputs/bridge/train_experiments.txt" "$WS/inputs/train_only_experiments.txt"; then
  echo "the bridge's split-T rows differ from $D/training_data:" >&2
  diff "$WS/inputs/bridge/train_experiments.txt" "$WS/inputs/train_only_experiments.txt" | head >&2 || true
  exit 2
fi
NOEMB=$(awk -F, 'NR == FNR {if ($4 == "T") {c[$2] = 1; a[$3] = 1}; next}
                 FNR > 1 && $4 == "VAL_HELDOUT" && (!($2 in c) || !($3 in a)) {print $1}' \
        "$WS/inputs/bridge/eic_bridge.csv" "$WS/inputs/bridge/eic_bridge.csv" | sort -u | tr '\n' ' ')
echo "staged $WS: $NT training tracks (symlinked under $TRACKS), $NV validation targets"
[ -z "$NOEMB" ] || echo "warning: validation tracks whose cell type or assay has no training track: $NOEMB"

# ---------------------------------------------------------------- the chain
X=(); [ -z "$EXCLUDE" ] || X=(--exclude="$EXCLUDE")
COMMON=(--parsable --output="$LOGS/%x_%A_%a.out" --error="$LOGS/%x_%A_%a.err" ${X[@]+"${X[@]}"})
GRID_WRAP="NEED_TORCH=0 source $WS/scripts/hpc_env.sh && cd $WS && python3 scripts/hpc_prepare_grid.py --bigwig $TRACKS/$(head -1 "$WS/inputs/train_only_experiments.txt").bigwig --out $WS/inputs/grid && echo chr20 > $WS/inputs/grid/chrom20.txt"
TV="BATCHPOS=1024,LR=1e-3,GENOMELR=1e-2,MAXHOURS=11.0"

submit() {   # submit <var> <dependency or -> sbatch args...; prints the line; sets $<var>
  local var=$1 dep=$2; shift 2
  local line=(sbatch "${COMMON[@]}")
  [ "$dep" = - ] || line+=(--dependency="afterok:$dep")
  line+=("$@")
  echo "${line[*]}"
  if [ "$DRY_RUN" = 1 ]; then
    printf -v "$var" '%s' "<$var>"
  else
    printf -v "$var" '%s' "$("${line[@]}" | cut -d';' -f1)"
    echo "  -> ${!var}"
  fi
}

echo "DRY_RUN=$DRY_RUN; sbatch chain (dependencies on live jobs only):"
submit J_GRID - --job-name=rcp_avocado_grid --account=def-maxwl_cpu --time=00:20:00 --mem=8G \
  --cpus-per-task=1 --gres=none --output="$LOGS/%x_%j.out" --error="$LOGS/%x_%j.err" \
  --wrap "$GRID_WRAP"
submit J_BIN "$J_GRID" --job-name=rcp_avocado_bin --array=0-22 --export=ALL,DATASET=d3 \
  "$WS/scripts/hpc_bin_tracks.slurm"
submit J_SHARED "$J_BIN" --job-name=rcp_avocado_train_shared --array=0-0 --gres=gpu:h100:1 \
  --export="ALL,DATASET=d3,MODE=shared,CHROMFILE=$WS/inputs/grid/chrom20.txt,EPOCHS=120,$TV" \
  "$WS/scripts/hpc_train.slurm"
submit J_GENOME "$J_SHARED" --job-name=rcp_avocado_train_genome --array=0-22%12 --gres=gpu:h100:1 \
  --export="ALL,DATASET=d3,MODE=genome,CHROMFILE=$WS/inputs/grid/chroms.txt,EPOCHS=60,$TV" \
  "$WS/scripts/hpc_train.slurm"
submit J_PRED "$J_GENOME" --job-name=rcp_avocado_predict --array=0-22%12 --export=ALL,DATASET=d3 \
  "$WS/scripts/hpc_predict.slurm"
submit J_BW "$J_PRED" --job-name=rcp_avocado_bigwig --array="0-$((NV - 1))%12" \
  --export=ALL,DATASET=d3 "$WS/scripts/hpc_make_bigwig.slurm"
echo "then score (CPU job, after $J_BW):"
echo "python3 tools/onion_eval/eic_val_compare.py score --pred $SC/pred_bw/d3" \
     "--name Avocado_train_only --truth $D/validation_data --cache <cache> --out <tsv>"
