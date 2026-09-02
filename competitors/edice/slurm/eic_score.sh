#!/bin/bash
# predict -> score ONE panel, in one job. Runs after eic_train.sh (for the model) and sigma.sh
# (for the σ table) with the same N_TARGETS and REGIME.
#
#   N_TARGETS=31                           sbatch --time=03:00:00 competitors/edice/slurm/eic_score.sh
#   N_TARGETS=31 SCOPE=genomewide          sbatch --time=12:00:00 --mem=96G competitors/edice/slurm/eic_score.sh
#   N_TARGETS=31 PANEL=B_ B_ONCE=1         sbatch --time=03:00:00 competitors/edice/slurm/eic_score.sh
#   N_TARGETS=31 PANEL=B_ TRUTH=challenge  sbatch --time=03:00:00 competitors/edice/slurm/eic_score.sh
#
# PANEL IS THE WHOLE OF §5's "B_ IS TOUCHED ONCE". The live regimes declare 38 eval pairs in one
# file -- 26 V_ and 12 B_ -- so predicting from the shipped regime opens both. This job predicts
# from a DERIVED single-panel regime instead (tools/declare_eval_pairs.py split), and that derived
# file is the only thing run_eic.py ever sees. PANEL=V_ is the default and is re-runnable; PANEL=B_
# needs B_ONCE=1 on the launch line AND an absent prediction root, so a second B_ pass is a refusal
# (exit 4) and not a silent overwrite.
#
# SCOPE picks the chromosome set, not the output file. §4 gives eDICE a PRINTED genome-wide cell,
# so unlike Avocado, ChromImpute and Lavawizard it does still predict genome-wide.
#   heldout    = the regime's eval_chroms, chr20+21+22, 6,478,903 bins. THE RANKED NUMBER.
#   genomewide = every chromosome the store carries, 121,241,684 bins -- ~19x, and ~22 GB of npz.
# A genomewide pass adds --held-out-chroms, so its ONE scores json carries the held-out `macro` and
# `panels` blocks AND a `genome_wide` block. That is why both scopes write the same path: the
# genome-wide json is a superset of the held-out one, not a rival to it.
#
# THE σ TABLE IS AN INPUT AND IS NEVER FIT HERE. §7: "σ is fit on training-set residuals only --
# never on V_, never on B_", and §12.2 voided every table fit on eval-pair residuals. This job
# refuses any table whose `fitted_on` does not start `training-residuals:` (exit 3). There is no
# override; competitors/edice/slurm/sigma.sh is how a valid table is made.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_edice_score
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -uo pipefail

# The one prefix a σ table must carry to be usable anywhere in this tree. Pinned; do not localise.
SIGMA_FITTED_ON_PREFIX="training-residuals:"

if [ -z "${N_TARGETS:-}" ]; then
  echo "[error] N_TARGETS is required -- it names which trained model to score." >&2; exit 2
fi
SCOPE="${SCOPE:-heldout}"
PANEL="${PANEL:-V_}"
TRUTH="${TRUTH:-store}"

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-$REPO/configs/regime.eic_19.json}"
REGIME_NAME="$(basename "$REGIME" .json)"; REGIME_NAME="${REGIME_NAME#regime.}"
RUNS="${RUNS:-/scratch/mforooz/t81_rivals/eDICE}"
WS="${WS:-$RUNS/${REGIME_NAME}_nt${N_TARGETS}}"
# model.SELECTED.pt, not model.pt: §5 ranks the checkpoint the run chose on V_, and model.pt is
# always the last epoch. eic_train.sh fails the run if the selected file is missing, so there is
# no legitimate board run where this default does not exist.
MODEL="${MODEL:-$WS/model.selected.pt}"

case "$PANEL" in
  V_) ;;
  B_) if [ "${B_ONCE:-0}" != "1" ]; then
        echo "[error] PANEL=B_ needs B_ONCE=1 on the launch line. BENCHMARK_DESIGN §5 rules that" >&2
        echo "        B_ is predicted ONCE, at the very end, and the flag is how that decision" >&2
        echo "        lands in this job's log instead of only in someone's memory." >&2
        exit 2
      fi ;;
  *)  echo "[error] PANEL must be V_ or B_, got '$PANEL'" >&2; exit 2 ;;
esac
case "$SCOPE" in heldout|genomewide) ;;
  *) echo "[error] SCOPE must be heldout or genomewide, got '$SCOPE'" >&2; exit 2 ;;
esac
case "$TRUTH" in
  store) ;;
  challenge) if [ "$PANEL" != "B_" ]; then
        echo "[error] TRUTH=challenge exists only for B_: the challenge truth root holds the 2019" >&2
        echo "        blind tracks, which are the B_ cells. Got PANEL=$PANEL." >&2
        exit 2
      fi ;;
  *) echo "[error] TRUTH must be store or challenge, got '$TRUTH'" >&2; exit 2 ;;
esac

# The pinned roots. V_ is scratch and re-runnable; B_ is /project because it is written once and
# must outlive a scratch purge. A genome-wide root is a SIBLING of the panel root and never the
# same directory -- the panel root is what the once-guard and the board both read.
PRED_ROOT_V="/scratch/mforooz/t81_pred/eDICE/${REGIME_NAME}/V_"
PRED_ROOT_B="/project/def-maxwl/mforooz/t81_pred_B/eDICE/${REGIME_NAME}/B_"
if [ "$PANEL" = "B_" ]; then PRED_DEFAULT="$PRED_ROOT_B"; else PRED_DEFAULT="$PRED_ROOT_V"; fi
[ "$SCOPE" = "genomewide" ] && PRED_DEFAULT="${PRED_DEFAULT}.genomewide"
PRED="${PRED:-$PRED_DEFAULT}"
SIGMA="${SIGMA:-/project/def-maxwl/mforooz/t81_sigma/eDICE/sigma_${REGIME_NAME}.json}"
SCORES="${SCORES:-/project/def-maxwl/mforooz/t81_scores/eDICE/${REGIME_NAME}/${TRUTH}.${PANEL}.json}"
TRUTH_ROOT="${TRUTH_ROOT:-/project/def-maxwl/mforooz/t81_truth_challenge/B_}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
source "$VENV/bin/activate"

cd "$REPO/competitors/edice"
# The venv's editable install does NOT put `candi` on the path -- its .pth names a checkout that
# carries `candi_kit` instead. $REPO/src is not optional.
export PYTHONPATH="$PWD:$REPO/src"

if [ ! -f "$MODEL" ]; then
  echo "[error] no model at $MODEL -- run eic_train.sh for N_TARGETS=$N_TARGETS first" >&2; exit 2
fi

# --- the σ table, before a GPU is spent ---------------------------------------------------------
# Checked FIRST, because a two-hour predict pass that ends in a refusal is two hours of the wrong
# lesson. `fitted_on` is the assertion: chunk A's fitter writes "training-residuals: ..." and
# nothing else in this tree can.
if [ ! -f "$SIGMA" ]; then
  echo "[error] no σ table at $SIGMA. Run competitors/edice/slurm/sigma.sh for this regime" >&2
  echo "        first; §7 lets the pval arm be scored as a Gaussian only against a table fit on" >&2
  echo "        TRAINING residuals." >&2
  exit 3
fi
python - "$SIGMA" "$SIGMA_FITTED_ON_PREFIX" <<'PYEOF' || exit 3
import json, sys
from pathlib import Path
p, prefix = Path(sys.argv[1]), sys.argv[2]
d = json.loads(p.read_text())
got = str(d.get("fitted_on", ""))
if not got.startswith(prefix):
    sys.stderr.write(
        f"[error] REFUSING {p}: fitted_on = {got!r}, which does not start {prefix!r}.\n"
        f"        BENCHMARK_DESIGN.md Rule 1 and §7 allow a σ fit on TRAINING residuals only;\n"
        f"        §12.2 declares every table fit on V_ or B_ eval-pair residuals VOID. There is\n"
        f"        no override flag. Refit with competitors/edice/slurm/sigma.sh.\n")
    raise SystemExit(3)
print(f"[edice] σ OK: {p.name}  fitted_on = {got}")
PYEOF

# --- the panel regime ---------------------------------------------------------------------------
# Derived, never a second shipped config: two files declaring one panel drift, and the drift is
# silent. `split` rewrites regions.bed absolute, because a regime file's BED resolves against its
# own directory and this copy lives beside the run.
mkdir -p "$WS"
PANEL_REGIME="$WS/regime.${REGIME_NAME}.${PANEL}.json"
python "$REPO/tools/declare_eval_pairs.py" split \
    --regime "$REGIME" --panel "$PANEL" --out "$PANEL_REGIME" || exit $?
echo "[edice] panel regime: $PANEL_REGIME"

case "$SCOPE" in
  heldout)    CHROMS=() ;;                 # default: the derived regime's eval_chroms
  genomewide) CHROMS=(--chroms all) ;;
esac

# --- the once-only B_ guard ---------------------------------------------------------------------
# Exit 4, and B_ONCE=1 does NOT lift it: the flag says "this is the one B_ pass", and a root that
# already carries a manifest proves it is not. Overwriting is how a second, differently-selected
# checkpoint gets scored on the blind panel with nothing in the record to say so.
if [ "$PANEL" = "B_" ] && [ -f "$PRED/manifest.json" ]; then
  echo "[error] $PRED already holds a manifest.json. B_ is predicted ONCE (§5); this root has" >&2
  echo "        been written. Delete it deliberately, or score the root that is there." >&2
  exit 4
fi

echo "[edice] EIC $SCOPE  panel=$PANEL  truth=$TRUTH  n_targets=$N_TARGETS"
echo "[edice]   model=$MODEL"
echo "[edice]   pred=$PRED"
echo "[edice]   scores=$SCORES"
echo "[edice] host=$(hostname)"; nvidia-smi -L || true

# Skip the predict pass when this root is ALREADY COMPLETE. A downstream step died once after a
# finished predict and threw away work that had succeeded; for the held-out scope that is minutes,
# for genome-wide it is hours. "Complete" means the manifest lists exactly the chromosomes asked
# for AND every track directory holds every one of them -- a partial root is redone, never resumed,
# because a half-written npz is the failure mode the §4.1 grid assertion exists to catch.
# FORCE_PREDICT=1 overrides.
complete=no
if [ -z "${FORCE_PREDICT:-}" ] && [ -f "$PRED/manifest.json" ]; then
  if python - "$PRED" "$SCOPE" "$PANEL_REGIME" <<'PYEOF'
import json, sys
from pathlib import Path
root, scope, regime_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
m = json.loads((root / "manifest.json").read_text())
have = list(m["chroms"])

# What THIS invocation would ask for. heldout = the derived regime's eval_chroms; genomewide =
# every chromosome the store carries. Checked rather than assumed: the root is keyed by panel and
# scope today, but a root recorded under a regime with different eval_chroms must not be reused.
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
    --regime "$PANEL_REGIME" --model "$MODEL" --out "$PRED" "${CHROMS[@]}" || exit $?
fi

mkdir -p "$(dirname "$SCORES")"
# `candi.bench.external` takes --chroms as ONE comma-separated string; run_eic.py takes a list. Read
# the chromosomes back out of the manifest the predict pass just wrote rather than re-deriving them,
# so the set scored is exactly the set emitted.
BENCH_CHROMS=$(python -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['chroms']))" \
                 "$PRED/manifest.json") || exit $?
EXTRA=()
# The held-out chromosomes are the derived regime's own eval_chroms -- chr20,chr21,chr22 in both
# live regimes -- read from the file rather than typed here, so the flag cannot disagree with the
# config the predictions were made under.
if [ "$SCOPE" = "genomewide" ]; then
  HELD_OUT=$(python -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['eval_chroms']))" \
                 "$PANEL_REGIME") || exit $?
  EXTRA+=(--held-out-chroms "$HELD_OUT")
  echo "[edice] genome-wide pass: --held-out-chroms $HELD_OUT"
fi
if [ "$TRUTH" = "challenge" ]; then
  if [ ! -f "$TRUTH_ROOT/manifest.json" ]; then
    echo "[error] TRUTH=challenge but no manifest.json under $TRUTH_ROOT" >&2; exit 2
  fi
  EXTRA+=(--truth-root "$TRUTH_ROOT")
  echo "[edice] truth: challenge bigwigs at $TRUTH_ROOT (count and peak arms are ABSENT there)"
fi

python -m candi.bench.external \
  --store "$PANEL_REGIME" --pred "$PRED" --out "$SCORES" --sigma-table "$SIGMA" \
  --chroms "$BENCH_CHROMS" "${EXTRA[@]}"
rc=$?

echo "[edice] DONE eic-score scope=$SCOPE panel=$PANEL truth=$TRUTH rc=$rc  scores=$SCORES"
exit $rc
