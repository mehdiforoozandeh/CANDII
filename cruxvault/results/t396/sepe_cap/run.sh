#!/bin/bash
# t396 C11 sepe_cap: the SE view against its PE or mixed parent, thinned to one mate and capped per
# position (start space), against the count-space law (h60 SE views; R44, R29, R30)
#
#   bash run.sh
#
# No arguments, no environment variables. Every step skips when its output exists.
# Steps (cluster: Fir; writes only in this folder, in $OUT and in $FLOOR on Fir):
#   1. fit     $OUT/per_assay.tsv missing -> sbatch --wait `sepe_cap.py fit` (one pooled a and
#              kappa, leave-one-experiment-out) on the 21 statistics files of $STATS
#              (array sc_sepe 6737536, see jobs.tsv)
#   2. floor   a $FLOOR/<se id>.npz missing -> sbatch --wait the floor-count array of
#              slurm/t396/sepe_cap.sh (the count-space law's attainable floor, every pair)
#   3. score   $OUT/table.tsv missing -> sbatch --wait `sepe_cap.py score` -> table.tsv,
#              pairs.tsv, pe_cap.tsv under $OUT
#   4. pull    each missing output -> rsync from $OUT on Fir
# Inputs read, never rebuilt: rows.tsv and jobs.tsv of this folder (C10b), UNIT_RULE.json of
# dedup_1bp (C12b), the read-start store /scratch/mforooz/rs392/store (t392) and the read-count
# store /scratch/mforooz/onion/measurements (t152).
# ssh exit 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=fir
KIT=/project/def-maxwl/mforooz/onion_kit/SC10b                 # .kit_stamp git=cce0535f
PY=/project/def-maxwl/mforooz/candi_venv/bin/python
STATS=/scratch/mforooz/sc396/caps/sepe
FLOOR=/scratch/mforooz/sc396/caps/sepe_floor
OUT=/project/def-maxwl/mforooz/t396/sepe_cap
ROWS=$KIT/cruxvault/results/t396/sepe_cap/rows.tsv
RULE=$KIT/cruxvault/results/t396/dedup_1bp/UNIT_RULE.json
N_PAIRS=21
GROUP=2                                                         # floor-count pairs per task
SB="--account=def-maxwl --gres=none --exclude=fc10713,fc10715 --cpus-per-task=2 --mem=8000M"
OUTPUTS=(per_assay.tsv table.tsv pairs.tsv pe_cap.tsv)

remote() {  # remote <cmd>: run on Fir; exit 255 = tunnel down
  local rc=0; ssh -o BatchMode=yes "$H" "$1" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  return "$rc"
}

need=0; for f in "${OUTPUTS[@]}"; do [ -s "$f" ] || need=1; done
if [ "$need" = 1 ]; then
  # 1. fit
  if ! remote "test -s $OUT/per_assay.tsv"; then
    remote "mkdir -p $OUT/logs && cd $OUT && sbatch --wait $SB --job-name=sc_sepe_fit --time=01:00:00 --output=$OUT/logs/%x_%j.out --wrap 'cd $KIT && $PY tools/t396/sepe_cap.py fit --stats $STATS/*.npz --rows $ROWS --out $OUT/per_assay.tsv'"
    echo "1 fit: ran on $H"
  fi
  # 2. floor
  n=$(remote "ls $FLOOR 2>/dev/null | grep -c npz" || true)
  if [ "${n:-0}" -lt "$N_PAIRS" ]; then
    last=$(( (N_PAIRS + GROUP - 1) / GROUP - 1 ))
    remote "mkdir -p $OUT/logs && export KIT=$KIT UNIT_RULE=$RULE FIT=$OUT/per_assay.tsv ROWS=$ROWS OUT_FLOOR=$FLOOR GROUP=$GROUP && sbatch --wait --export=ALL --job-name=sc_sepe --array=0-$last%12 --time=01:00:00 --output=$OUT/logs/%x_%A_%a.out $KIT/slurm/t396/sepe_cap.sh floor-count"
    echo "2 floor: ran on $H"
  fi
  # 3. score
  if ! remote "test -s $OUT/table.tsv"; then
    remote "cd $OUT && sbatch --wait $SB --job-name=sc_sepe_score --time=01:00:00 --output=$OUT/logs/%x_%j.out --wrap 'cd $KIT && $PY tools/t396/sepe_cap.py score --stats $STATS/*.npz --fit $OUT/per_assay.tsv --rows $ROWS --floor-count $FLOOR/*.npz --out $OUT'"
    echo "3 score: ran on $H"
  fi
fi

# 4. pull
for f in "${OUTPUTS[@]}"; do
  [ -s "$f" ] && continue
  rc=0; rsync -a -e 'ssh -o BatchMode=yes' "$H:$OUT/$f" "$f" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  [ "$rc" = 0 ] || { echo "rsync $OUT/$f failed" >&2; exit 1; }
  echo "4 pull: $f"
done
