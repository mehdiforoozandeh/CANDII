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
#   CI_RUN=<rundir> sbatch score.sh          # defaults to the regime eval scope
#
# `CI_ALLOW_MISSING=1` for a partial root — the pilot covers 20 of the 45 declared targets, and
# without it `bench.external` refuses the run by design (the D2 lesson).
#
# `CI_SIGMA=<sigma.json>` supplies the σ table. Without it a point-only rival carries the E and P
# blocks only and no `gauss_suite` — absent keys, not NaN. With it the pval arm becomes a
# homoscedastic Gaussian and the CRPS tier opens.
#
# DO NOT PASS A σ FROM `fit_sigma.py` AS IT STANDS. It reads a V_ scores json and squares its `mse`,
# so it is fitted on V_ eval-pair residuals — which BENCHMARK_DESIGN.md Rule 1 forbids at every
# stage including a σ-fit, and which §7 rules out explicitly ("σ is fit on training-set residuals
# only"). §12.2 declares every existing σ VOID. A training-residual σ is new work, not a flag.
set -euo pipefail

REPO=${CI_REPO:-/project/def-maxwl/mforooz/CANDII_t78_code}
PY=${CI_PY:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python}
REGIME=${CI_REGIME:-$REPO/configs/regime.eic_19.json}
RUN=${CI_RUN:?set CI_RUN}
CHROMS=${CI_CHROMS:-chr20,chr21,chr22}
OUT=${CI_OUT:-$RUN/scores_$CHROMS.json}

if [ "$CHROMS" = "all" ]; then
  CHROMS=$(cut -f1 "$RUN/input/chrominfo.txt" | paste -sd, -)
fi
EXTRA=(); [ "${CI_ALLOW_MISSING:-0}" = "1" ] && EXTRA+=(--allow-missing)
[ -n "${CI_SIGMA:-}" ] && EXTRA+=(--sigma-table "$CI_SIGMA")

cd "$REPO"
PYTHONPATH=$REPO/src $PY -m candi.bench.external \
    --store "$REGIME" --pred "$RUN/pred" --out "$OUT" \
    --chroms "$CHROMS" "${EXTRA[@]}"
echo "wrote $OUT"
