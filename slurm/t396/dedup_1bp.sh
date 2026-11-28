#!/bin/bash
# t396 dedup at 1 bp (C12a, C12b). Two modes, one script.
#
#   untar: extract the base and nodedup tagAligns of the picked experiments from one onion-work tar.
#     export KIT=/project/def-maxwl/mforooz/onion_kit/SC12a
#     sbatch --job-name=sc_untar --export=ALL --output=<logs>/%x_%j.out \
#         $KIT/slurm/t396/dedup_1bp.sh untar
#   run: `dedup_1bp.py unit` and `shape` on every replicate and on the pooled set of one experiment.
#     export KIT=... EXP=ENCSR928HYM
#     sbatch --job-name=sc_1bp --export=ALL --cpus-per-task=2 --mem=8000M --time=<probe> \
#         --output=<logs>/%x_%j.out $KIT/slurm/t396/dedup_1bp.sh run
#
# Experiments: column 1 of $KIT/cruxvault/results/t396/dedup_1bp/picked.tsv.
# Tagaligns land at $WORK/onion/work/<experiment>/<unit>.<layer>/{base,nodedup}/pr{1,2}.tagAlign.gz.
# Run outputs: $OUT/<experiment>/{unit,shape}_<label>.json.
#SBATCH --account=def-maxwl
#SBATCH --job-name=sc_untar
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=none
#SBATCH --exclude=fc10713,fc10715

set -uo pipefail
MODE="${1:?usage: dedup_1bp.sh untar|run}"
: "${KIT:?export KIT}"
TAR="${TAR:-/scratch/mforooz/rs392/tars/onion-work__005.tar}"
MEMBERS_ALL="${MEMBERS_ALL:-/scratch/mforooz/rs392/plan/members_onion-work__005.txt}"
WORK="${WORK:-/scratch/mforooz/sc396/work}"
OUT="${OUT:-/project/def-maxwl/mforooz/t396/dedup_1bp}"
CHROM_SIZES="${CHROM_SIZES:-/project/def-maxwl/mforooz/ONION/genome/chrom_sizes.json}"
PICKED="$KIT/cruxvault/results/t396/dedup_1bp/picked.tsv"
PY=/project/def-maxwl/mforooz/candi_venv/bin/python

echo "=== sc396 dedup_1bp $MODE job ${SLURM_JOB_ID:-none} host=$(hostname) $(date -u)"
cd "$KIT" || exit 2
[ -f "$PICKED" ] || { echo "no $PICKED" >&2; exit 2; }
mkdir -p "$WORK" "$OUT"

case "$MODE" in
untar)
  MEMBERS="$OUT/members.txt"
  EXPS="$(awk -F'\t' 'NR > 1 {print $1}' "$PICKED" | paste -sd'|' -)"
  grep -E "^onion/work/($EXPS)/[^/]+/(base|nodedup)/pr[12]\.tagAlign\.gz$" "$MEMBERS_ALL" > "$MEMBERS"
  want=$(awk -F'\t' 'NR > 1 {s += $NF} END {print s}' "$PICKED")
  got=$(wc -l < "$MEMBERS")
  echo "members: $got (picked.tsv expects $want)"
  [ "$got" -eq "$want" ] || { echo "member count mismatch" >&2; exit 1; }
  tar -xf "$TAR" -C "$WORK" -T "$MEMBERS"
  r=$?
  missing=0
  while IFS= read -r m; do
    [ -s "$WORK/$m" ] || { echo "missing $m"; missing=$((missing + 1)); }
  done < "$MEMBERS"
  echo "tar exit=$r missing=$missing bytes=$(cd "$WORK" && tr '\n' '\0' < "$MEMBERS" | du -cb --files0-from=- | tail -1 | cut -f1)"
  [ "$r" -eq 0 ] && [ "$missing" -eq 0 ] || exit 1
  ;;
run)
  : "${EXP:?export EXP}"
  source "$(dirname "$PY")/activate"
  source "$KIT/slurm/_kit_pin.sh" || exit 1
  E="$WORK/onion/work/$EXP"
  mkdir -p "$OUT/$EXP"
  units=()
  for u in "$E"/*/; do units+=("$(basename "$u")"); done
  n=${#units[@]}
  [ "$n" -ge 1 ] || { echo "no units under $E" >&2; exit 1; }
  off_all=(); on_all=()
  for u in "${units[@]}"; do
    off=("$E/$u"/nodedup/pr{1,2}.tagAlign.gz); on=("$E/$u"/base/pr{1,2}.tagAlign.gz)
    off_all+=("${off[@]}"); on_all+=("${on[@]}")
    label="${u%%.*}"
    for cmd in unit shape; do
      f="$OUT/$EXP/${cmd}_${label}.json"
      [ -s "$f" ] && continue
      extra=(); [ "$cmd" = shape ] && extra=(--chrom-sizes "$CHROM_SIZES")
      /usr/bin/time -v "$PY" tools/t396/dedup_1bp.py "$cmd" --off "${off[@]}" --on "${on[@]}" \
        --experiment "$EXP" --label "$label" --n-units 1 --out "$f" "${extra[@]}" || exit 1
    done
  done
  if [ "$n" -ge 2 ]; then
    for cmd in unit shape; do
      f="$OUT/$EXP/${cmd}_pooled.json"
      [ -s "$f" ] && continue
      extra=(); [ "$cmd" = shape ] && extra=(--chrom-sizes "$CHROM_SIZES")
      /usr/bin/time -v "$PY" tools/t396/dedup_1bp.py "$cmd" --off "${off_all[@]}" --on "${on_all[@]}" \
        --experiment "$EXP" --label pooled --n-units "$n" --out "$f" "${extra[@]}" || exit 1
    done
  fi
  ;;
*) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "=== done $(date -u)"
