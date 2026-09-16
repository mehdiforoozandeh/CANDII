#!/bin/bash
# predict -> score ONE panel, in one job. Runs after eic_train.sh (for the model) and sigma.sh
# (for the σ table) with the same N_TARGETS and REGIME.
#
#   N_TARGETS=31                                            sbatch --time=03:00:00 …/eic_score.sh
#   N_TARGETS=31 SCOPE=genomewide                           sbatch --time=12:00:00 --mem=96G …
#   N_TARGETS=31 PANEL=B_ SCOPE=genomewide B_ONCE=1         sbatch --time=12:00:00 --mem=96G …
#   N_TARGETS=31 PANEL=B_ SCOPE=genomewide TRUTH=challenge   sbatch --time=03:00:00 …
#
# THE LAST LINE IS A SCORE OF A ROOT THAT IS ALREADY THERE, AND PREDICTS NOTHING. This job does
# predict -> score in one command, so it has to be able to tell the two apart. It resolves $PRED,
# asks whether that root is COMPLETE, and only then decides:
#   complete             -> mode=score-only. No predict, no B_ guard, any TRUTH. This is how the
#                           challenge-truth score of the one B_ prediction is reached.
#   absent or partial    -> mode=predict+score. A prediction WOULD be written, so PANEL=B_ meets
#                           the once-only guard here: B_ONCE=1 required, and either B_ root
#                           already carrying a manifest is a refusal (exit 4).
#
# PANEL IS THE WHOLE OF §5's "B_ IS TOUCHED ONCE". The live regimes declare 38 eval pairs in one
# file -- 26 V_ and 12 B_ -- so predicting from the shipped regime opens both. This job predicts
# from a DERIVED single-panel regime instead (tools/declare_eval_pairs.py split), and that derived
# file is the only thing run_eic.py ever sees. PANEL=V_ is the default and is re-runnable; a B_
# PREDICTION needs B_ONCE=1 on the launch line AND both B_ prediction roots absent, so a second B_
# prediction is a refusal (exit 4) and not a silent overwrite. Scoring a complete B_ root spends no
# prediction and is therefore not refused -- the rule is one B_ PREDICTION, not one B_ invocation.
#
# TRUTH PICKS WHAT THE PREDICTIONS ARE COMPARED AGAINST, AND HOW WIDE THE SCORE IS.
#   store      = the CANDI store, over the chromosomes the PREDICTION manifest names.
#   challenge  = the 2019 blind bigwigs under $TRUTH_ROOT, over the chromosomes the TRUTH root's
#                manifest names -- chr20,chr21,chr22, because tools/challenge_bigwigs.py converted
#                the held-out chromosomes alone. A challenge pass is HELD-OUT ONLY: it never adds
#                --held-out-chroms and its json carries no `genome_wide` block, exactly like every
#                sibling method's challenge.B_.json. Asking for the prediction root's 23
#                chromosomes instead would send read_truth_root_arrays after npz files the truth
#                root does not have.
#
# SCOPE picks the chromosome set, not the output file. §4 gives eDICE a PRINTED genome-wide cell,
# so unlike Avocado, ChromImpute and Lavawizard it does still predict genome-wide.
#   heldout    = the regime's eval_chroms, chr20+21+22, 6,478,903 bins. THE RANKED NUMBER.
#   genomewide = every chromosome the store carries, 121,241,684 bins -- ~19x, and ~22 GB of npz.
# A genomewide pass adds --held-out-chroms, so its ONE scores json carries the held-out `macro` and
# `panels` blocks AND a `genome_wide` block. That is why both scopes write the same path: the
# genome-wide json is a superset of the held-out one, not a rival to it.
#
# TWO CONSEQUENCES OF THAT SUPERSET, BOTH ENFORCED BELOW.
#  * B_ IS GENOME-WIDE OR NOTHING (exit 2). The scope changes the prediction ROOT -- genomewide is
#    a `.genomewide` sibling -- so "B_ held-out" and "B_ genome-wide" are two predictions of B_,
#    each of which would pass a guard that watched only its own root. eDICE has a printed
#    genome-wide cell and the genome-wide pass yields the held-out macro anyway, so the ONE
#    permitted B_ prediction is the genome-wide one; SCOPE=heldout PANEL=B_ is refused. And the
#    once-guard watches BOTH roots, not the one this invocation happens to point at.
#  * A HELD-OUT SCORE NEVER OVERWRITES A GENOME-WIDE ONE (exit 5). Same path, smaller content: a
#    held-out pass run second would drop the `genome_wide` block and leave nothing in the file to
#    say a genome-wide pass had ever been made.
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
# Rows per forward pass in the predict pass. NOT a science knob: predict is `model.eval()` with no
# normalisation layer anywhere in the net and one independent row per bin, so the batch only sets
# how much VRAM one decoder activation costs -- `batch x n_targets x 2048 x 4 B`. Held here at the
# same value sigma.sh uses, so both stages of one method predict under one shape. run_eic.py's own
# default of 4096 fits the 45-track V_ panel (1.41 GiB) but not the 98-track σ panel (3.06 GiB) on
# the MIG slice; 1024 fits both and costs no measurable time, the pass being bin-count bound.
PREDICT_BATCH="${PREDICT_BATCH:-1024}"

# B_ONCE is NOT checked here. It licenses a B_ PREDICTION, and whether this invocation would make
# one is not known until $PRED has been resolved and asked whether it is already complete -- so the
# check lives with the rest of the once-only guard, below the completeness test.
case "$PANEL" in
  V_|B_) ;;
  *)  echo "[error] PANEL must be V_ or B_, got '$PANEL'" >&2; exit 2 ;;
esac
case "$SCOPE" in heldout|genomewide) ;;
  *) echo "[error] SCOPE must be heldout or genomewide, got '$SCOPE'" >&2; exit 2 ;;
esac
# B_ IS GENOME-WIDE OR NOTHING. The scope picks the prediction root as well as the chromosome set,
# so allowing both scopes on B_ would allow TWO B_ predictions -- one under `.../B_`, one under
# `.../B_.genomewide` -- and §5 permits one, ever. eDICE is the one method with a printed
# genome-wide cell (§4), and the genome-wide pass carries the ranked held-out `macro` and `panels`
# in the same scores json via --held-out-chroms, so the genome-wide scope is the strictly larger
# of the two and there is nothing a held-out B_ pass could add.
if [ "$PANEL" = "B_" ] && [ "$SCOPE" != "genomewide" ]; then
  echo "[error] SCOPE=$SCOPE with PANEL=B_ is refused. B_ is predicted ONCE, EVER (§5), and each" >&2
  echo "        scope writes its own prediction root, so a held-out B_ pass would SPEND that one" >&2
  echo "        prediction on the smaller scope. eDICE prints a genome-wide cell (§4) and its" >&2
  echo "        genome-wide scoring pass already emits the held-out macro and panels through" >&2
  echo "        --held-out-chroms. Relaunch with SCOPE=genomewide." >&2
  exit 2
fi
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
echo "[banner] code=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) kit=$REPO"

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

# --- the scores json a held-out pass would replace ------------------------------------------------
# Both scopes write the SAME path on purpose -- the genome-wide json is a superset of the held-out
# one -- and that makes the reverse order LOSSY. A held-out pass run after a genome-wide one would
# rewrite the file without its `genome_wide` block, and nothing left in the result would say a
# genome-wide pass had ever happened. Refuse (exit 5) rather than overwrite; a genome-wide pass may
# always replace a held-out json, because it carries everything the held-out one did.
# Checked HERE, beside the σ check, so the refusal costs no GPU.
#
# N/A UNDER TRUTH=challenge, and unreachable there: challenge forces PANEL=B_, which forces
# SCOPE=genomewide, so the condition below is false. It is also moot -- a challenge pass writes
# `challenge.B_.json`, a different file from the store pass's `store.B_.json`, and never carries a
# `genome_wide` block for a later pass to drop.
if [ "$SCOPE" = "heldout" ] && [ -f "$SCORES" ]; then
  python - "$SCORES" <<'PYEOF' || exit 5
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text())
except Exception as exc:                                   # a truncated json is not a genome-wide
    print(f"[edice] existing {p.name} does not parse ({exc}); a held-out pass will replace it.")
    raise SystemExit(0)
if "genome_wide" in d:
    sys.stderr.write(
        f"[error] REFUSING to overwrite {p}: it carries a `genome_wide` block, so it was written\n"
        f"        by a SCOPE=genomewide pass and already holds everything this held-out pass\n"
        f"        would write, plus the genome-wide aggregation. Running held-out over it would\n"
        f"        drop that block silently. Rerun with SCOPE=genomewide, or send this pass\n"
        f"        somewhere else with SCORES=<path>.\n")
    raise SystemExit(5)
print(f"[edice] existing {p.name} carries no genome_wide block; a held-out pass may replace it.")
PYEOF
fi

# --- the panel regime ---------------------------------------------------------------------------
# Derived, never a second shipped config: two files declaring one panel drift, and the drift is
# silent. `split` rewrites regions.bed absolute, because a regime file's BED resolves against its
# own directory and this copy lives beside the run.
mkdir -p "$WS"
PANEL_REGIME="$WS/regime.${REGIME_NAME}.${PANEL}.json"
python "$REPO/tools/declare_eval_pairs.py" split \
    --regime "$REGIME" --panel "$PANEL" --out "$PANEL_REGIME" || exit $?
echo "[edice] panel regime: $PANEL_REGIME"

# === DISPATCH: predict then score, or score a root that is already there ========================
# The order below is load-bearing.
#   (a) $PRED, resolved above, is the root this invocation would write;
#   (b) ask whether that root is already COMPLETE;
#   (c) complete -> predict nothing, and score it under whatever TRUTH was asked for;
#   (d) absent or partial -> a prediction WOULD be written, so PANEL=B_ meets the once-only guard.
# The guard used to sit BEFORE (b) and to fire on PANEL=B_ alone, so the store pass's own manifest
# made every later B_ invocation exit 4 -- including the challenge-truth score, which writes no
# prediction at all and therefore cannot spend the one B_ shot. Nothing about "B_ is predicted
# ONCE, EVER" is loosened: every invocation that would PREDICT is still checked against BOTH B_
# roots. All of this runs before the GPU is touched.

case "$SCOPE" in
  heldout)    CHROMS=() ;;                 # default: the derived regime's eval_chroms
  genomewide) CHROMS=(--chroms all) ;;
esac

# --- (b) is the root this invocation would write ALREADY COMPLETE? ------------------------------
# A downstream step died once after a finished predict and threw away work that had succeeded; for
# the held-out scope that is minutes, for genome-wide it is hours. "Complete" means the manifest
# lists exactly the chromosomes asked for AND every track directory holds every one of them -- a
# partial root is redone, never resumed, because a half-written npz is the failure mode the §4.1
# grid assertion exists to catch. FORCE_PREDICT=1 overrides, and for B_ then meets the guard below,
# which is what keeps it from re-predicting the blind panel.
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
# (c) vs (d), in one word, so a log says which of the two this job was.
if [ "$complete" = "yes" ]; then MODE=score-only; else MODE=predict+score; fi

# --- (d) the once-only B_ guard, over BOTH B_ ROOTS ---------------------------------------------
# Reached only when a PREDICTION would be written. Exit 4, and B_ONCE=1 does NOT lift the manifest
# clause: the flag says "this is the one B_ pass", and a root that already carries a manifest
# proves it is not. Overwriting is how a second, differently-selected checkpoint gets scored on the
# blind panel with nothing in the record to say so.
#
# EVERY B_ ROOT IS CHECKED, NOT THE ONE THIS INVOCATION WOULD WRITE. A guard that read only $PRED
# is no guard at all: SCOPE appends `.genomewide` to the root, so two launches with different
# scopes would each find their own root absent and each predict B_. The rule is one B_ prediction
# EVER, so any manifest under either root refuses the next one. $PRED is checked too, in case it
# was overridden on the launch line to a third place.
if [ "$PANEL" = "B_" ] && [ "$complete" = "no" ]; then
  if [ "${B_ONCE:-0}" != "1" ]; then
    echo "[error] a B_ PREDICTION needs B_ONCE=1 on the launch line. BENCHMARK_DESIGN §5 rules" >&2
    echo "        that B_ is predicted ONCE, at the very end, and the flag is how that decision" >&2
    echo "        lands in this job's log instead of only in someone's memory. Scoring a B_ root" >&2
    echo "        that is already COMPLETE predicts nothing and does not ask for the flag." >&2
    exit 4
  fi
  for B_ROOT in "$PRED_ROOT_B" "${PRED_ROOT_B}.genomewide" "$PRED"; do
    [ -f "$B_ROOT/manifest.json" ] || continue
    echo "[error] $B_ROOT already holds a manifest.json. B_ is predicted ONCE, EVER (§5), across" >&2
    echo "        both the held-out root and its .genomewide sibling; this one has been written." >&2
    echo "        Delete it deliberately, or SCORE the root that is there -- a complete root is" >&2
    echo "        scored by this job without predicting, under either TRUTH." >&2
    exit 4
  done
fi

# --- the chromosomes the SCORE pass runs over, when the truth is the challenge bigwigs ----------
# Resolved here rather than beside the score command so the banner can print them, and so a missing
# truth root refuses before a predict rather than after one. tools/challenge_bigwigs.py writes
# `chroms` into the truth root's manifest; the fallback is the held-out three, with a warning,
# because a truth root whose manifest lost that key is still the held-out conversion.
TRUTH_CHROMS="-"
if [ "$TRUTH" = "challenge" ]; then
  if [ ! -f "$TRUTH_ROOT/manifest.json" ]; then
    echo "[error] TRUTH=challenge but no manifest.json under $TRUTH_ROOT" >&2; exit 2
  fi
  TRUTH_CHROMS=$(python -c \
      "import json,sys; print(','.join(json.load(open(sys.argv[1])).get('chroms') or []))" \
      "$TRUTH_ROOT/manifest.json")
  if [ -z "$TRUTH_CHROMS" ]; then
    TRUTH_CHROMS="chr20,chr21,chr22"
    echo "[warn] $TRUTH_ROOT/manifest.json names no chroms; falling back to $TRUTH_CHROMS" >&2
  fi
fi

echo "[edice] EIC $SCOPE  panel=$PANEL  truth=$TRUTH  n_targets=$N_TARGETS  mode=$MODE"
echo "[edice]   model=$MODEL"
echo "[edice]   pred=$PRED"
echo "[edice]   scores=$SCORES"
echo "[edice]   predict_batch=$PREDICT_BATCH"
echo "[edice]   truth_chroms=$TRUTH_CHROMS"
echo "[edice] host=$(hostname)"; nvidia-smi -L || true

if [ "$MODE" = "predict+score" ]; then
  python run_eic.py predict \
    --regime "$PANEL_REGIME" --model "$MODEL" --out "$PRED" \
    --batch-size "$PREDICT_BATCH" "${CHROMS[@]}" || exit $?
fi

mkdir -p "$(dirname "$SCORES")"
EXTRA=()
if [ "$TRUTH" = "challenge" ]; then
  # HELD-OUT ONLY, for every panel. --held-out-chroms is what turns one pass into two aggregations
  # and adds the `genome_wide` block, and a challenge json must not carry one: the truth root
  # covers the held-out chromosomes alone, so there is no wider scope to aggregate over. Every
  # sibling method's challenge.B_.json is shaped the same way. The prediction manifest is NOT the
  # source of the chromosome set here -- a B_.genomewide root names 23, and asking
  # read_truth_root_arrays for those would send it after npz files the truth root does not have.
  BENCH_CHROMS="$TRUTH_CHROMS"
  EXTRA+=(--truth-root "$TRUTH_ROOT")
  echo "[edice] truth: challenge bigwigs at $TRUTH_ROOT (count and peak arms are ABSENT there)"
  echo "[edice] challenge pass: --chroms $BENCH_CHROMS, and no --held-out-chroms"
else
  # `candi.bench.external` takes --chroms as ONE comma-separated string; run_eic.py takes a list.
  # Read the chromosomes back out of the manifest under $PRED rather than re-deriving them, so the
  # set scored is exactly the set emitted.
  BENCH_CHROMS=$(python -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['chroms']))" \
                   "$PRED/manifest.json") || exit $?
  # The held-out chromosomes are the derived regime's own eval_chroms -- chr20,chr21,chr22 in both
  # live regimes -- read from the file rather than typed here, so the flag cannot disagree with the
  # config the predictions were made under.
  if [ "$SCOPE" = "genomewide" ]; then
    HELD_OUT=$(python -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['eval_chroms']))" \
                   "$PANEL_REGIME") || exit $?
    EXTRA+=(--held-out-chroms "$HELD_OUT")
    echo "[edice] genome-wide pass: --held-out-chroms $HELD_OUT"
  fi
fi

python -m candi.bench.external \
  --store "$PANEL_REGIME" --pred "$PRED" --out "$SCORES" --sigma-table "$SIGMA" \
  --chroms "$BENCH_CHROMS" "${EXTRA[@]}"
rc=$?

echo "[edice] DONE eic-score scope=$SCOPE panel=$PANEL truth=$TRUTH mode=$MODE rc=$rc  scores=$SCORES"
exit $rc
