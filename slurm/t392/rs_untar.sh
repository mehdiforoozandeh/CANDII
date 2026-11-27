#!/bin/bash
# t392 C11a: extract the plan members of one onion-work tar on Fir.
#
#   sbatch --job-name=rs392_untar_<k> --output=<logs>/%x_%j.out \
#       <kit>/slurm/t392/rs_untar.sh <tar> <members.txt> <work root>
#
# <members.txt>: one tar member per line, the output of `build_plan.py members --plan <plan>`.
# The members land at <work root>/<member>, i.e., <work root>/onion/work/<experiment>/..., the
# prefix that `rs_recount.sh --work-root <work root>` reads. Exit 1 when a listed member is
# missing on disk after the extraction.
#SBATCH --account=def-maxwl
#SBATCH --job-name=rs392_untar
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=none

set -uo pipefail
USAGE="usage: rs_untar.sh <tar> <members.txt> <work root>"
TAR="${1:?$USAGE}"
MEMBERS="${2:?$USAGE}"
OUT="${3:?$USAGE}"
[ -f "$TAR" ] || { echo "no tar $TAR" >&2; exit 2; }
[ -s "$MEMBERS" ] || { echo "no members $MEMBERS" >&2; exit 2; }
mkdir -p "$OUT"
echo "=== rs392 untar job ${SLURM_JOB_ID:-none} host=$(hostname) tar=$TAR members=$(wc -l < "$MEMBERS") $(date -u)"
tar -xf "$TAR" -C "$OUT" -T "$MEMBERS"
r=$?
missing=0
while IFS= read -r m; do
  [ -f "$OUT/$m" ] || { echo "missing $m"; missing=$((missing + 1)); }
done < "$MEMBERS"
echo "tar exit=$r missing=$missing bytes=$(cd "$OUT" && tr '\n' '\0' < "$MEMBERS" | du -cb --files0-from=- | tail -1 | cut -f1)"
echo "=== done $(date -u)"
[ "$r" -eq 0 ] && [ "$missing" -eq 0 ] || exit 1
