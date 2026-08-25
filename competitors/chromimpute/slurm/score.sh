#!/bin/bash
#SBATCH --job-name=ci_score
#SBATCH --account=def-maxwl
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --time=6:00:00
#
# `collect.py`'s prediction root -> a scores json, through `candi.bench.external`. A compute job and
# not a login-node command: the scorer walks every bin of every declared track and reads truth out
# of the store, which is minutes to hours, not seconds.
#
#   CI_RUN=<rundir> CI_CHROMS=chr21 sbatch score.sh
#
# `CI_ALLOW_MISSING=1` for a partial root — the pilot covers 20 of the 45 declared targets, and
# without it `bench.external` refuses the run by design (the D2 lesson).
set -euo pipefail

REPO=${CI_REPO:-/project/def-maxwl/mforooz/CANDII_t51}
PY=${CI_PY:-/project/def-maxwl/mforooz/candi_venv/bin/python}
RUN=${CI_RUN:?set CI_RUN}
CHROMS=${CI_CHROMS:-chr21}
OUT=${CI_OUT:-$RUN/scores_$CHROMS.json}

if [ "$CHROMS" = "all" ]; then
  CHROMS=$(cut -f1 "$RUN/input/chrominfo.txt" | paste -sd, -)
fi
EXTRA=(); [ "${CI_ALLOW_MISSING:-0}" = "1" ] && EXTRA=(--allow-missing)

cd "$REPO"
PYTHONPATH=$REPO/src $PY -m candi.bench.external \
    --store "$REPO/configs/regime.eic_val.json" --pred "$RUN/pred" --out "$OUT" \
    --chroms "$CHROMS" "${EXTRA[@]}"
echo "wrote $OUT"
