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
#   CI_RUN=<rundir> sbatch score.sh                                   # V_ against store truth
#   CI_RUN=<rundir> CI_PANEL=B_ sbatch score.sh                       # B_ against store truth
#   CI_RUN=<rundir> CI_PANEL=B_ CI_TRUTH=challenge sbatch score.sh    # B_ against the 2019 bigwigs
#
# THE SCOPE IS THE EVAL CHROMOSOMES AND NOTHING ELSE. §4 blanks ChromImpute's `genome-wide` cell
# and rules that a blanked cell is not computed, so this job never passes `--held-out-chroms`:
# there is no genome-wide aggregation for it to sit beside.
#
# `CI_ALLOW_MISSING=1` for a partial root — a pilot run covers 20 of the declared targets, and
# without it `bench.external` refuses the run by design (the D2 lesson).
#
# THE σ TABLE IS REQUIRED, AND IT MUST BE A TRAINING-RESIDUAL ONE. ChromImpute predicts a point in
# -log10 p and no spread, so the pval arm can only be scored as a Gaussian against a table handed
# in. §7: "σ is fit on training-set residuals only — never on V_, never on B_"; §12.2 declared
# every table squared off a V_ scores json VOID. This job checks `fitted_on` and refuses anything
# that does not start `training-residuals:` (exit 3). There is no override flag; the table comes
# from `competitors/chromimpute/slurm/sigma.sh`.
set -euo pipefail

# The one prefix a σ table must carry to be usable anywhere in this tree. Pinned; do not localise.
SIGMA_FITTED_ON_PREFIX="training-residuals:"

REPO=${CI_REPO:-/project/def-maxwl/mforooz/CANDII_main}
PY=${CI_PY:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python}
SRC_REGIME=${CI_REGIME:-$REPO/configs/regime.eic_19.json}
REGIME_NAME=$(basename "$SRC_REGIME" .json); REGIME_NAME=${REGIME_NAME#regime.}
PANEL=${CI_PANEL:-V_}
TRUTH=${CI_TRUTH:-store}
RUN=${CI_RUN:?set CI_RUN}
CHROMS=${CI_CHROMS:-chr20,chr21,chr22}

case "$PANEL" in V_|B_) ;; *) echo "[error] CI_PANEL must be V_ or B_, got '$PANEL'"; exit 2 ;; esac
case "$TRUTH" in
  store) ;;
  challenge) if [ "$PANEL" != "B_" ]; then
        echo "[error] CI_TRUTH=challenge exists only for B_: the challenge truth root holds the"
        echo "        2019 blind tracks, which are the B_ cells. Got CI_PANEL=$PANEL."
        exit 2
      fi ;;
  *) echo "[error] CI_TRUTH must be store or challenge, got '$TRUTH'"; exit 2 ;;
esac

if [ "$PANEL" = "B_" ]; then
  PRED_ROOT=${CI_PRED_ROOT:-/project/def-maxwl/mforooz/t81_pred_B/ChromImpute/$REGIME_NAME/B_}
else
  PRED_ROOT=${CI_PRED_ROOT:-/scratch/mforooz/t81_pred/ChromImpute/$REGIME_NAME/V_}
fi
SIGMA=${CI_SIGMA:-/project/def-maxwl/mforooz/t81_sigma/ChromImpute/sigma_$REGIME_NAME.json}
OUT=${CI_OUT:-/project/def-maxwl/mforooz/t81_scores/ChromImpute/$REGIME_NAME/$TRUTH.$PANEL.json}
TRUTH_ROOT=${CI_TRUTH_ROOT:-/project/def-maxwl/mforooz/t81_truth_challenge/B_}

if [ "$CHROMS" = "all" ]; then
  CHROMS=$(cut -f1 "$RUN/input/chrominfo.txt" | paste -sd, -)
fi

# The single-panel regime, derived rather than assumed present: this job may run long after the
# chain that made it, and re-deriving is cheap and idempotent. Everything the scorer reads — the
# declared pairs, the eval chromosomes, the region set — comes from THIS file.
REGIME=$RUN/regime.$REGIME_NAME.$PANEL.json
$PY "$REPO/tools/declare_eval_pairs.py" split \
    --regime "$SRC_REGIME" --panel "$PANEL" --out "$REGIME"

# --- the σ table -------------------------------------------------------------------------------
if [ ! -f "$SIGMA" ]; then
  echo "[ci_score] no σ table at $SIGMA. Run competitors/chromimpute/slurm/sigma.sh for this"
  echo "           regime first; §7 lets the pval arm be scored as a Gaussian only against a"
  echo "           table fit on TRAINING residuals."
  exit 3
fi
$PY - "$SIGMA" "$SIGMA_FITTED_ON_PREFIX" <<'PYEOF' || exit 3
import json, sys
from pathlib import Path
p, prefix = Path(sys.argv[1]), sys.argv[2]
d = json.loads(p.read_text())
got = str(d.get("fitted_on", ""))
if not got.startswith(prefix):
    sys.stderr.write(
        f"[ci_score] REFUSING {p}: fitted_on = {got!r}, which does not start {prefix!r}.\n"
        f"           BENCHMARK_DESIGN.md Rule 1 and §7 allow a σ fit on TRAINING residuals only;\n"
        f"           §12.2 declares every table fit on V_ or B_ eval-pair residuals VOID. There\n"
        f"           is no override flag. Refit with competitors/chromimpute/slurm/sigma.sh.\n")
    raise SystemExit(3)
print(f"[ci_score] σ OK: {p.name}  fitted_on = {got}")
PYEOF

EXTRA=(--sigma-table "$SIGMA")
[ "${CI_ALLOW_MISSING:-0}" = "1" ] && EXTRA+=(--allow-missing)
if [ "$TRUTH" = "challenge" ]; then
  if [ ! -f "$TRUTH_ROOT/manifest.json" ]; then
    echo "[error] CI_TRUTH=challenge but no manifest.json under $TRUTH_ROOT"; exit 2
  fi
  EXTRA+=(--truth-root "$TRUTH_ROOT")
  echo "[ci_score] truth: challenge bigwigs at $TRUTH_ROOT (count and peak arms are ABSENT there)"
fi

mkdir -p "$(dirname "$OUT")"
echo "[ci_score] panel=$PANEL truth=$TRUTH chroms=$CHROMS"
echo "[ci_score]   pred=$PRED_ROOT"
echo "[ci_score]   out=$OUT"

cd "$REPO"
PYTHONPATH=$REPO/src $PY -m candi.bench.external \
    --store "$REGIME" --pred "$PRED_ROOT" --out "$OUT" \
    --chroms "$CHROMS" "${EXTRA[@]}"
echo "wrote $OUT"
