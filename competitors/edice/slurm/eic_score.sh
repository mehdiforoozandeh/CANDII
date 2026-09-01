#!/bin/bash
# predict -> σ-fit -> score, in one job. Runs after eic_train.sh for the same N_TARGETS.
#
#   N_TARGETS=31 SCOPE=heldout   sbatch --time=03:00:00 competitors/edice/slurm/eic_score.sh
#   N_TARGETS=31 SCOPE=genomewide sbatch --time=12:00:00 --mem=96G competitors/edice/slurm/eic_score.sh
#
# RETARGETED 2026-08-31. `P1`/`P2` are RETIRED (BENCHMARK_DESIGN.md §9) and replaced by §4's two
# SCOPES of one prediction pass. eDICE is one of the methods whose `genome-wide` cell is PRINTED
# (§4), so unlike Avocado, ChromImpute and Lavawizard it does still predict genome-wide.
#   heldout    = the regime's eval_chroms, chr20+21+22, 6,478,903 bins. THE RANKED NUMBER.
#   genomewide = every chromosome the store carries, 121,241,684 bins — ~19x, and ~22 GB of npz.
#
# THE σ-FIT IS REFUSED, AND THAT IS THE RULING, NOT A BUG. `fit_sigma.py` fits on V_ eval-pair
# residuals. §7 rules "σ is fit on training-set residuals only — never on V_, never on B_", and
# §12.2 declares every existing σ VOID under Rule 1. A training-residual σ needs predictions on
# TRAINING tracks, which this prediction root does not contain, so it is new work, not a flag.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t52_edice_score
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -uo pipefail

if [ -z "${N_TARGETS:-}" ]; then
  echo "[error] N_TARGETS is required -- it names which trained model to score." >&2; exit 2
fi
SCOPE="${SCOPE:-heldout}"

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t78_code}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-$REPO/configs/regime.eic_19.json}"
REGIME_NAME="$(basename "$REGIME" .json)"; REGIME_NAME="${REGIME_NAME#regime.}"
RUNS="${RUNS:-/project/def-maxwl/mforooz/rivals_src/edice_runs}"
MODEL="${MODEL:-$RUNS/${REGIME_NAME}_nt${N_TARGETS}/model.pt}"
PRED="${PRED:-$RUNS/${REGIME_NAME}_nt${N_TARGETS}/preds_${SCOPE}}"
# The σ-table lives with the run, not with a scope -- see the header.
SIGMA="${SIGMA:-$RUNS/${REGIME_NAME}_nt${N_TARGETS}/sigma.json}"
SCORES="${SCORES:-$RUNS/${REGIME_NAME}_nt${N_TARGETS}/scores_${SCOPE}.json}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
source "$VENV/bin/activate"

cd "$REPO/competitors/edice"
export PYTHONPATH="$PWD:$REPO/src"

case "$SCOPE" in
  heldout)    CHROMS=() ;;                 # default: the regime's eval_chroms
  genomewide) CHROMS=(--chroms all) ;;
  *)  echo "[error] SCOPE must be heldout or genomewide, got '$SCOPE'" >&2; exit 2 ;;
esac

if [ ! -f "$MODEL" ]; then
  echo "[error] no model at $MODEL -- run eic_train.sh for N_TARGETS=$N_TARGETS first" >&2; exit 2
fi

echo "[edice] EIC $SCOPE  n_targets=$N_TARGETS  model=$MODEL  pred=$PRED"
echo "[edice] host=$(hostname)"; nvidia-smi -L || true

# Skip the predict pass when this root is ALREADY COMPLETE. The σ-fit died once on a signature
# bug after a finished predict, throwing away work that had succeeded; for P1 that was 5 minutes,
# for P2 it would be hours. "Complete" means the manifest lists exactly the chromosomes asked for
# AND every track directory holds every one of them -- a partial root is redone, never resumed,
# because a half-written npz is the failure mode the §4.1 grid assertion exists to catch.
# FORCE_PREDICT=1 overrides.
complete=no
if [ -z "${FORCE_PREDICT:-}" ] && [ -f "$PRED/manifest.json" ]; then
  if python - "$PRED" "$SCOPE" "$REGIME" <<'PYEOF'
import json, sys
from pathlib import Path
root, scope, regime_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
m = json.loads((root / "manifest.json").read_text())
have = list(m["chroms"])

# What THIS invocation would ask for. heldout = the regime's eval_chroms; genomewide = every
# chromosome the store carries. Checked rather than assumed: the root is keyed by scope today, but
# a root recorded under a regime with different eval_chroms must not be silently reused.
regime = json.loads(Path(regime_path).read_text())
if scope == "heldout":
    want = list(regime["eval_chroms"])
else:
    # candi is on PYTHONPATH already (exported above); a heredoc has no __file__ to derive it from.
    from candi.store.reader import CorpusStore
    from candi.store import layout as L
    want = L.sort_chroms(CorpusStore(regime["store"]).n_bins().keys())

dirs = [d for d in root.iterdir() if d.is_dir()]
same = have == want
whole = len(dirs) == m["n_tracks"] and all((d / f"{c}.npz").exists() for d in dirs for c in have)
ok = same and whole
why = "COMPLETE, skipping predict" if ok else (
    f"chroms differ (have {have}, want {want}), redoing predict" if not same
    else "INCOMPLETE, redoing predict")
print(f"[edice] existing root: {len(dirs)}/{m['n_tracks']} tracks x {len(have)} chroms -> {why}")
sys.exit(0 if ok else 1)
PYEOF
  then complete=yes; fi
fi

if [ "$complete" = "no" ]; then
  python run_eic.py predict \
    --regime "$REGIME" --model "$MODEL" --out "$PRED" "${CHROMS[@]}" || exit $?
fi

if [ ! -f "$SIGMA" ]; then
  if [ "${SIGMA_RULE1_OVERRIDE:-0}" != "1" ]; then
    echo "[error] REFUSING to fit $SIGMA: fit_sigma.py fits on V_ eval-pair residuals, which" >&2
    echo "        BENCHMARK_DESIGN.md Rule 1 forbids and §7 rules out explicitly. A training-" >&2
    echo "        residual σ needs predictions on TRAINING tracks, which this root does not" >&2
    echo "        contain. Raise it; do not override." >&2
    exit 3
  fi
  python fit_sigma.py --regime "$REGIME" --pred "$PRED" --out "$SIGMA" || exit $?
fi

mkdir -p "$(dirname "$SCORES")"
# `candi.bench.external` takes --chroms as ONE comma-separated string; run_eic.py takes a list. Read
# the chromosomes back out of the manifest the predict pass just wrote rather than re-deriving them,
# so the set scored is exactly the set emitted.
BENCH_CHROMS=$(python -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['chroms']))" \
                 "$PRED/manifest.json") || exit $?
python -m candi.bench.external \
  --store "$REGIME" --pred "$PRED" --out "$SCORES" --sigma-table "$SIGMA" --chroms "$BENCH_CHROMS"
rc=$?

echo "[edice] DONE eic-score $SCOPE n_targets=$N_TARGETS rc=$rc  scores=$SCORES"
exit $rc
