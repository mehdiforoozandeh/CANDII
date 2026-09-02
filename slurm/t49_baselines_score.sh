#!/bin/bash
# Score the baseline prediction roots. ONE ARRAY TASK PER METHOD, either §4 scope.
#
#   mkdir -p slurm-logs
#   # held-out, eic_19, V_ — chr20+21+22 only, one aggregation
#   REGIME=configs/regime.eic_19.json PANEL=V_ SCOPE=heldout \
#       sbatch --array=0-4%5 --time=3:00:00 --mem=24G slurm/t49_baselines_score.sh
#   # genome-wide, after the generation array — all 23, and BOTH aggregations in one json
#   SCOPE=genomewide sbatch --array=0-4%5 --dependency=afterok:<gen_array_jobid> \
#       --time=5-18:00:00 --mem=96G slurm/t49_baselines_score.sh
#
# THE TWO SCOPES OF §4, AND WHY THE GENOME-WIDE PASS NAMES THE HELD-OUT CHROMOSOMES
# ---------------------------------------------------------------------------------
# `SCOPE=heldout` scores chr20+21+22 and the json carries one aggregation. `SCOPE=genomewide` scores
# all 23 AND passes `--held-out-chroms chr20,chr21,chr22`, which is what turns one pass into the
# held-out numbers plus a parallel `genome_wide` block (`plan/EVAL_PLAN.md`; `provenance.scope`
# records which, so an absent block is never ambiguous). Until 2026-09-01 this script passed 23
# chromosomes and no `--held-out-chroms`, so the genome-wide pass produced NO `genome_wide` block —
# and these five baselines are among the methods whose genome-wide cell §4 does not blank, so that
# block is a cell of the board rather than an extra.
#
# `--held-out-chroms` IS CHUNK J's FLAG ON `candi.bench.external`. It is already on `candi.bench.cli`
# and on `harness.run_bench`. On a checkout where `bench.external` has not got it yet this pass dies
# with an argparse error, which is the right failure: silently dropping it would produce a
# genome-wide job whose json has no genome-wide block, at ~54 h a method.
#
# THE COLLAPSED TWO ARE SCORED TWICE, OUT OF ONE ROOT — D1, 2026-09-01
# --------------------------------------------------------------------
# `avg` and `avg-arcsinh` are GENERATED once, under $COLLAPSE_REGIME (their prediction is a function
# of the contributors at the predicted position, so the regime's training loci cannot move it; the
# manifest carries `regime_independent`, written by the assertion `t49_baselines_p1.sh` runs).
# They are nevertheless SCORED once per regime, because `tools/leaderboard.py`'s
# `gate_row_against_board` requires a row's regime name to match the board it is added to. So the
# score file is addressed by the BOARD's regime and the prediction root by the GENERATION regime,
# and those two differ for exactly these two methods. `knn1`, `knn5` and `marginal` fit on
# `train_chroms` and have a root of their own per regime.
#
# A collapsed method whose root carries no `regime_independent` stamp is REFUSED against the other
# board (exit 5): the one number is printed in two rows only because a comparison licensed it.
#
# ONE TASK PER METHOD BECAUSE SCORING IS THE EXPENSIVE HALF, and by a lot. Measured on Fir, chr21,
# 45 declared tracks: generation is ~4.5 minutes for all five methods; `bench.external` is ~2.4
# minutes PER TRACK, so ~110 minutes per method. Scoring five methods inside one job is nine hours
# of serial work for something that is five independent runs.
#
# For a genome-wide pass the dependency must be `afterok` on the WHOLE generation array:
# `bench.external` scores a track over the CONCATENATION of every chromosome (the top-1 % thresholds
# of mse1obs/mse1imp are taken over all of them at once), so a root missing one chromosome cannot be
# scored at all. It is also why a genome-wide pass cannot be split by chromosome to make it cheaper:
# a per-chromosome top-1 % is a different metric, not a cheaper estimate of the same one.
#
# COST, MEASURED RATHER THAN GUESSED — and the reason `--time` is not defaulted for the genome-wide
# pass. chr21 is 1.87 M bins and the whole store is 121 M, so genome-wide is 65x per track: ~2.6 h
# per track and ~117 h for the 45 declared tracks. `avg-arcsinh` is pval-only — no NB CRPS and no
# `nbinom.ppf` over every bin — and comes in near 10 h. The 24 h default below is right for a
# held-out pass and FAR too short for a genome-wide one; the count arm needs the b5 bin. Mind the
# standing reservations: a 7-day request is refused outright with `ReqNodeNotAvail, Reserved for
# maintenance` whenever a maintenance reservation falls inside the window, so size the walltime to
# land BEFORE the next one rather than asking for the bin maximum.
#
# CRPS_APPROX=100 CRPS_SEED=0 SWITCHES THE COUNT ARM TO t56's FAIR-SAMPLED ESTIMATOR, which the PI
# approved for the genome-wide pass on 2026-08-26. That is what makes it affordable: ~54 h/method at
# k=100 against ~117 h exact. It needs a checkout carrying the t56 code — `bench.external` rejects
# the flag otherwise, which is the right failure, because a silent fall-back to the closed form
# would produce a 117 h job labelled as a 54 h one. Every score json it writes carries
# provenance.crps_estimator/crps_k/crps_seed, and a count-arm CRPS from such a json is an ESTIMATE
# and is quoted as one.
#
# The leaderboard is NOT assembled here: it needs every method's score file, so run
# `python -m competitors.baselines.leaderboard --protocol P1|P2 --scores ...` once the array is
# done. That step is json folding and takes seconds.
#
# WHY --gres ON A CPU-ONLY JOB: see slurm/bake.sh. (The two generation scripts dropped theirs. They
# load torch too — everything that imports `candi` does, through `candi/__init__.py` ->
# `candi.encoder` — but they never open CUDA, so the slice bought them nothing. This one is left on
# the GPU account's fairshare. Invariant 13: the 1g.10gb slice is the only sanctioned gres.)
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t49_score
#SBATCH --output=slurm-logs/t49_score_%A_%a.out
#SBATCH --error=slurm-logs/t49_score_%A_%a.err
#SBATCH --array=0-4%5
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4

set -uo pipefail

KIT="${KIT:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_19.json}"        # the BOARD's regime
PANEL="${PANEL:-V_}"
COLLAPSE_REGIME="${COLLAPSE_REGIME:-eic_19}"          # where the collapsed two were generated
V_PRED_ROOT="${V_PRED_ROOT:-/scratch/$USER/t81_pred}"
B_PRED_ROOT="${B_PRED_ROOT:-/project/def-maxwl/$USER/t81_pred_B}"
SCORES_ROOT="${SCORES_ROOT:-/project/def-maxwl/$USER/t81_scores}"
VARPOOL="${VARPOOL:-}"
# t56's fair-sampled NB-CRPS estimator (PI-approved at k=100, 2026-08-26). Empty = the closed form.
# Only reachable when the checkout carries the t56 code; on a checkout without it `bench.external`
# rejects the flag, which is the failure we want rather than a silent exact run.
CRPS_APPROX="${CRPS_APPROX:-}"
CRPS_SEED="${CRPS_SEED:-0}"
# §4's two scopes. `heldout` is the default because it is the cheap one and the one every board row
# needs; `genomewide` is opted into, sized by hand, and is the only one that names --held-out-chroms.
SCOPE="${SCOPE:-heldout}"
HELD_OUT="${HELD_OUT:-chr20,chr21,chr22}"
ALL_CHROMS="chr1,chr2,chr3,chr4,chr5,chr6,chr7,chr8,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr20,chr21,chr22,chrX"
case "$SCOPE" in
    heldout)    CHROMS="${CHROMS:-$HELD_OUT}"; HELD_ARG="" ;;
    genomewide) CHROMS="${CHROMS:-$ALL_CHROMS}"; HELD_ARG="--held-out-chroms $HELD_OUT" ;;
    *) echo "[t49-score] REFUSING: SCOPE=$SCOPE. §4 has two scopes: heldout, genomewide." >&2
       exit 2 ;;
esac

METHOD_LIST=(avg avg-arcsinh knn1 knn5 marginal)
METHOD="${METHOD:-${METHOD_LIST[${SLURM_ARRAY_TASK_ID:-0}]}}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONPATH="$KIT/src:$KIT"

case "$PANEL" in
    V_) PRED_BASE="$V_PRED_ROOT" ;;
    B_) PRED_BASE="$B_PRED_ROOT" ;;
    *)  echo "[t49-score] REFUSING: PANEL=$PANEL is not a panel; it is V_ or B_ (§5)." >&2; exit 2 ;;
esac
REGNAME="$(basename "$REGIME" .json)"; REGNAME="${REGNAME#regime.}"

# The board's regime names the ROW; the generation regime names the ROOT. They differ for the two
# collapsed methods and only for them.
case "$METHOD" in
    avg|avg-arcsinh)     PRED_REGNAME="$COLLAPSE_REGIME" ;;
    knn1|knn5|marginal)  PRED_REGNAME="$REGNAME" ;;
    *) echo "[t49-score] REFUSING: $METHOD is not a baseline method." >&2; exit 2 ;;
esac
PRED="$PRED_BASE/$METHOD/$PRED_REGNAME/$PANEL"
OUT="$SCORES_ROOT/$METHOD/$REGNAME/store.$PANEL.json"

if [ ! -f "$PRED/manifest.json" ]; then
    echo "[t49-score] REFUSING: no manifest at $PRED — that unit has not been generated." >&2
    exit 2
fi
# Printing one root's number in a board it was not generated under is licensed by the identity
# assertion and by nothing else.
if [ "$PRED_REGNAME" != "$REGNAME" ]; then
    python - "$PRED/manifest.json" "$REGNAME" <<'PYEOF' || exit 5
import json, sys
m = json.load(open(sys.argv[1]))
ok = m.get("regime_independent", {}).get("identical") is True
if not ok:
    sys.exit(f"[t49-score] REFUSING: {sys.argv[1]} carries no `regime_independent` stamp, so this "
             f"root was never shown to be identical under another regime and its number may not be "
             f"printed in the {sys.argv[2]} board. Regenerate with "
             f"--assert-regime-independent (BENCHMARK_DESIGN.md §12.2, D1).")
print(f"[t49-score] collapse asserted against {m['regime_independent']['asserted_against']} on "
      f"{m['regime_independent']['chrom']}")
PYEOF
fi

# The panel regime: the DECLARED eval_pairs filtered to this panel's targets, so `_expected` covers
# exactly the tracks the root holds. Both live regimes carry all 38 pairs (26 V_ + 12 B_). It goes
# beside the roots, not in $SLURM_TMPDIR, for the same reason the generators put it there: the score
# json records the regime it was given, and provenance that names a path on a vanished node-local
# disk is not provenance. Written through a rename so concurrent array tasks never read a half-file.
REG_DIR="$PRED_BASE/_regimes"
mkdir -p "$REG_DIR"
PANEL_REGIME="$REG_DIR/regime.$REGNAME.$PANEL.json"
python tools/declare_eval_pairs.py split --regime "$REGIME" --panel "$PANEL" \
    --out "$PANEL_REGIME.$$.tmp" && mv -f "$PANEL_REGIME.$$.tmp" "$PANEL_REGIME" \
    || { echo "[t49-score] could not derive the $PANEL regime" >&2; exit 2; }

echo "[t49-score] host=$(hostname) commit=$(git rev-parse --short HEAD) method=$METHOD"
echo "[t49-score] board=$REGNAME panel=$PANEL pred=$PRED out=$OUT"
echo "[t49-score] scope=$SCOPE chroms=$CHROMS held_out=${HELD_OUT:-none} arg=${HELD_ARG:-none}"
# Which CRPS estimator ran is a PI-ruled fact about the numbers, so it belongs in the log and not
# only in whoever's memory of the submit line. The score json carries its own stamp; this is the
# copy you can read after the job is gone.
if [ -n "$CRPS_APPROX" ]; then
    echo "[t49-score] crps=fair_sampled k=$CRPS_APPROX seed=$CRPS_SEED"
else
    echo "[t49-score] crps=closed_form"
fi

mkdir -p "$(dirname "$OUT")"
python -m candi.bench.external \
    --store "$PANEL_REGIME" --pred "$PRED" --out "$OUT" \
    --chroms "$CHROMS" $HELD_ARG ${VARPOOL:+--varpool "$VARPOOL"} \
    ${CRPS_APPROX:+--crps-approx "$CRPS_APPROX" --crps-seed "$CRPS_SEED"}
rc=$?
echo "[t49-score] method=$METHOD exit=$rc"
exit $rc
