#!/bin/bash
# The naive baseline suite — GENERATION, one job, one (regime, panel) unit.
#
#   mkdir -p slurm-logs
#   # eic_19, V_: all five methods, with the D1 identity assertion on the collapsed two
#   REGIME=configs/regime.eic_19.json PANEL=V_ sbatch slurm/t49_baselines_p1.sh
#   # eic_pilot, V_: ONLY the three that fit on the regime's training loci (D1)
#   REGIME=configs/regime.eic_pilot.json PANEL=V_ METHODS=knn1,knn5,marginal \
#       sbatch slurm/t49_baselines_p1.sh
#   # the once-only B_ pass, genome-wide, to /project
#   GW=$(printf 'chr%s,' 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X); GW=${GW%,}
#   PANEL=B_ B_ONCE=1 CHROMS=$GW sbatch slurm/t49_baselines_p1.sh
#   # stamp a root the p2 array built (one chromosome, two methods, minutes)
#   ASSERT_ONLY=1 METHODS=avg,avg-arcsinh PANEL=V_ sbatch slurm/t49_baselines_p1.sh
#
# HOW MANY TIMES EACH BASELINE RUNS — D1, SETTLED 2026-09-01
# ----------------------------------------------------------
# `plan/BENCHMARK_DESIGN.md` §12.2 first ruled all five naive baselines run ONCE rather than once
# per regime, "because their fit is regime-independent", and asked for an ASSERTION rather than an
# argument. Read against the code the claim is true of two of the five and false of three:
#
#   avg, avg-arcsinh   regime-independent. The contributor set is `biosamples.train` minus the
#                      target's cell type, and every written bin is a function of the contributors
#                      AT THE PREDICTED POSITION. No training locus enters. Collapse: correct.
#   knn1, knn5         `generate.similarity_table` correlates over `panel.train_chroms`. Different
#                      train chromosomes -> a different ranking -> different predictions.
#   marginal           `generate.fit_marginal` pools over `panel.train_chroms`.
#
# So the baselines are **8 method-regime units**, not 5: `avg` and `avg-arcsinh` once (under
# $COLLAPSE_REGIME, and this script refuses to generate them under any other), `knn1`, `knn5` and
# `marginal` once per regime. §12.2 and §12.3 now say so.
#
# THE ASSERTION EXISTS NOW and this script runs it, as a second pass over ONE chromosome once the
# main generation is done: `--assert-regime-independent $ASSERT_AGAINST --assert-only` re-predicts
# the collapsed methods under the OTHER regime INTO A TEMPORARY DIRECTORY, compares every array with
# `np.array_equal`, and writes `regime_independent` into their manifests. A difference exits **5**
# and stamps nothing — the collapse is licensed by the comparison, never by this comment.
# `ASSERT_AGAINST=none` skips it, and a root that carries no stamp is refused against the other
# board by t49_baselines_score.sh. `ASSERT_ONLY=1` runs this pass alone, which is how a root built
# by the p2 array gets its stamp.
#
# `--assert-only` IS LOAD-BEARING AND WAS MISSING UNTIL 2026-09-01. This pass used to re-run
# `--methods avg,avg-arcsinh` into the SAME roots, which OVERWROTE the checked chromosome's npz and
# then stamped `identical: true` on arrays that were not the ones the generation pass wrote (and,
# with no knn method in that second list, on arrays computed a third way again — see the D1 note in
# `generate.py`). The stamp is now the only mutation of the prediction root, and
# `tests/test_baselines.py::test_the_assertion_pass_changes_nothing_but_the_manifest` holds it there
# byte for byte.
#
# WHY eic_pilot IS STILL REFUSED FOR knn/marginal (exit 3)
# --------------------------------------------------------
# `competitors/baselines/generate.py` reads `train_chroms` RAW and has no `regions` support, so
# under `regime.eic_pilot.json` the kNN similarity table and the per-assay marginal would fit over
# 18 WHOLE chromosomes (~2.7 Gbp) instead of the 25,588,197 bp the regime declares — every other
# method under that regime sees the Pilot Regions only. That is a Rule 2 break, so the guard below
# refuses it, and D1's second pilot unit for those three is blocked until either `generate.py`
# honours `regions` or the PI rules the whole-chromosome fit acceptable and on the record. `avg` and
# `avg-arcsinh` are unaffected: no training locus enters them, which is why the guard fires only
# when a regime-dependent method is requested.
#
# CPU ONLY, AND NO GRES ANY MORE. The arithmetic is numpy over h5py reads and NOTHING HERE OPENS
# CUDA. It does load torch — `generate.py` imports `candi.store.reader`, and `candi/__init__.py`
# imports `candi.encoder`, which imports torch — so the job pays the import and the RSS, and that is
# the reason to keep the array cap on the p2 script. It never allocates a device. The MIG slice this
# script used to request was a fairshare workaround (see slurm/bake.sh) and it parked a 10 GB H100
# slice for a job that never opened CUDA. Invariant 13 allows the 1g.10gb slice or nothing; this is
# nothing.
#
# ONE POISSON FLOOR, THE PRE-REGISTERED ONE. This script used to generate at 1e6 AND 1e4 because
# `candi.metrics.nb_crps` returned NaN above ~2e4 and the count arm lost its whole distributional
# tier at the pre-registered value. t56 fixed that (`nb_crps` scores the large-dispersion case in
# the Poisson limit; `test_the_preregistered_poisson_floor_is_scoreable_by_candi_bench` pins it), so
# there is one floor again and it is §5.1's. Two floors could not share a pinned root anyway —
# `poisson_n` is in `_MANIFEST_IDENTITY`, so the second pass would refuse to merge.
#
# GENERATION ONLY. Scoring is `slurm/t49_baselines_score.sh`, one array task per method: measured on
# Fir, chr21, 45 declared tracks, generation is ~4.5 minutes for all five methods and
# `bench.external` is ~110 minutes PER METHOD. Scoring inside this job made it eighteen hours of
# serial work for something that is independent runs.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t49_p1
#SBATCH --output=slurm-logs/t49_p1_%j.out
#SBATCH --error=slurm-logs/t49_p1_%j.err
#SBATCH --time=11:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -uo pipefail

KIT="${KIT:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_19.json}"
PANEL="${PANEL:-V_}"                     # V_ or B_. Nothing else is a panel.
METHODS="${METHODS:-avg,avg-arcsinh,knn1,knn5,marginal}"
# The regime the two collapsed methods are generated under, once, and printed in both board rows.
COLLAPSE_REGIME="${COLLAPSE_REGIME:-eic_19}"
ASSERT_AGAINST="${ASSERT_AGAINST:-configs/regime.eic_pilot.json}"   # `none` to skip
ASSERT_ONLY="${ASSERT_ONLY:-0}"          # 1 = stamp an existing root and generate nothing
# The regime's eval scope. Genome-wide `V_` comes from the p2 array; genome-wide `B_` comes from
# this script with CHROMS set to all 23, because p2's array cannot hold the once-only B_ guard.
CHROMS="${CHROMS:-chr20,chr21,chr22}"
POISSON_N="${POISSON_N:-1e6}"            # §5.1's pre-registered floor, scoreable since t56
# The pinned prediction roots. `V_` is deletable scratch; `B_` lands on /project and is written once.
V_PRED_ROOT="${V_PRED_ROOT:-/scratch/$USER/t81_pred}"
B_PRED_ROOT="${B_PRED_ROOT:-/project/def-maxwl/$USER/t81_pred_B}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
# `_kit_pin.sh` pins `candi` to this checkout's src and proves it. `competitors` is not installed
# anywhere, so the repo root goes on the path too — AFTER src, so the pin's guarantee still holds.
export PYTHONPATH="$KIT/src:$KIT"

case "$PANEL" in
    V_) PRED_BASE="$V_PRED_ROOT" ;;
    B_) PRED_BASE="$B_PRED_ROOT" ;;
    *)  echo "[t49] REFUSING: PANEL=$PANEL is not a panel; it is V_ or B_ (§5)." >&2; exit 2 ;;
esac
REGNAME="$(basename "$REGIME" .json)"; REGNAME="${REGNAME#regime.}"

# Which of the requested methods collapse and which are fitted — the D1 split, the same two lists
# `competitors/baselines/generate.py` exports as REGIME_INDEPENDENT / REGIME_DEPENDENT.
COLLAPSED=""; DEPENDENT=""
for M in ${METHODS//,/ }; do
    case "$M" in
        avg|avg-arcsinh)     COLLAPSED="$COLLAPSED $M" ;;
        knn1|knn5|marginal)  DEPENDENT="$DEPENDENT $M" ;;
        *) echo "[t49] REFUSING: $M is not a baseline method." >&2; exit 2 ;;
    esac
done

if [ -n "$COLLAPSED" ] && [ "$REGNAME" != "$COLLAPSE_REGIME" ]; then
    echo "[t49] REFUSING:$COLLAPSED collapse to ONE run (D1) and that run is under" >&2
    echo "      $COLLAPSE_REGIME. Generating them again under $REGNAME would put two roots on disk" >&2
    echo "      for one unit. Drop them from METHODS, or set COLLAPSE_REGIME if the canonical" >&2
    echo "      regime has changed." >&2
    exit 2
fi

if [ -n "$DEPENDENT" ] && \
   python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('regions') else 1)" "$REGIME"; then
    echo "[t49] REFUSING: $REGIME declares a \`regions\` BED and$DEPENDENT fit on the regime's" >&2
    echo "      training loci. generate.py reads train_chroms RAW and has no regions support, so" >&2
    echo "      they would fit over WHOLE chromosomes instead of the declared region set. Rule 2" >&2
    echo "      break — see this script's header. Raise it; do not add a flag to get past it." >&2
    exit 3
fi

# The panel regime: the DECLARED eval_pairs filtered to this panel's targets. Both live regimes
# carry all 38 pairs (26 V_ + 12 B_), so a V_ run must be given a V_-only regime or it would predict
# B_ on the way past — and §5 touches B_ once, at the very end.
#
# IT LIVES BESIDE THE ROOTS, NOT IN $SLURM_TMPDIR, AND THAT IS LOAD-BEARING. `generate.py` records
# the regime PATH in each manifest and `regime` is one of the `_MANIFEST_IDENTITY` fields, so two
# passes into one root (this script for chr20-22 and the p2 array for the rest) must be handed the
# same path or the second refuses to merge. Written through a rename so concurrent tasks never see
# a half-file; the content is a deterministic function of $REGIME and $PANEL.
WORK="${SLURM_TMPDIR:-/tmp}/t49_${SLURM_JOB_ID:-$$}"
REG_DIR="$PRED_BASE/_regimes"
mkdir -p "$WORK" "$REG_DIR"
PANEL_REGIME="$REG_DIR/regime.$REGNAME.$PANEL.json"
python tools/declare_eval_pairs.py split --regime "$REGIME" --panel "$PANEL" \
    --out "$PANEL_REGIME.$$.tmp" && mv -f "$PANEL_REGIME.$$.tmp" "$PANEL_REGIME" \
    || { echo "[t49] could not derive the $PANEL regime" >&2; exit 2; }
python - "$PANEL_REGIME" "$PANEL" <<'PYEOF' || exit 2
import json, sys
pairs = json.load(open(sys.argv[1])).get("eval_pairs", [])
bad = [p for p in pairs if not str(p[1]).startswith(sys.argv[2])]
if bad or not pairs:
    sys.exit(f"[t49] REFUSING: {sys.argv[1]} carries {len(bad)} pair(s) outside {sys.argv[2]} "
             f"({len(pairs)} declared). The split did not do what its name says.")
PYEOF

if [ "$ASSERT_ONLY" = "1" ]; then
    if [ -z "$COLLAPSED" ]; then
        echo "[t49] REFUSING: ASSERT_ONLY=1 with METHODS=$METHODS has nothing to assert — the" >&2
        echo "      stamp belongs to avg and avg-arcsinh only." >&2
        exit 2
    fi
    if [ "$PANEL" = "B_" ]; then
        echo "[t49] REFUSING: ASSERT_ONLY=1 with PANEL=B_. The B_ pass is one job and stamps its" >&2
        echo "      own roots as it goes; there is no array to stamp after the fact." >&2
        exit 2
    fi
fi

# B_ IS WRITTEN ONCE (§5), TO /project. An explicit B_ONCE=1, and a root with no manifest in it.
if [ "$PANEL" = "B_" ]; then
    if [ "${B_ONCE:-0}" != "1" ]; then
        echo "[t49] REFUSING: PANEL=B_ without B_ONCE=1. B_ is predicted once, from the selected" >&2
        echo "      artefact, at the very end. Set B_ONCE=1 when this IS that run." >&2
        exit 4
    fi
    for M in ${METHODS//,/ }; do
        if [ -e "$PRED_BASE/$M/$REGNAME/B_/manifest.json" ]; then
            echo "[t49] REFUSING: $PRED_BASE/$M/$REGNAME/B_/manifest.json exists — that root has" >&2
            echo "      already been written. A second B_ pass is a second look at the blind panel." >&2
            exit 4
        fi
    done
fi

# The pinned roots are <Method>/<regime>/<panel> and `generate.py` writes <out>/<method>, so each
# method's root is reached through a symlink named for the method. One pass over the store then
# serves every method — which is the whole shape of generate.py, and a genome-wide pass reads
# ~370 GB, so five one-method passes would read it five times.
STAGE="$WORK/roots"
mkdir -p "$STAGE"
for M in ${METHODS//,/ }; do
    R="$PRED_BASE/$M/$REGNAME/$PANEL"
    mkdir -p "$R" && ln -sfn "$R" "$STAGE/$M" || exit 1
done

echo "[t49] host=$(hostname) commit=$(git rev-parse --short HEAD) regime=$REGNAME panel=$PANEL"
echo "[t49] methods=$METHODS chroms=$CHROMS poisson_n=$POISSON_N roots=$PRED_BASE/<M>/$REGNAME/$PANEL"
echo "[t49] assert_only=$ASSERT_ONLY assert_against=$ASSERT_AGAINST"

rc=0
if [ "$ASSERT_ONLY" != "1" ]; then
    python -m competitors.baselines.generate \
        --store "$PANEL_REGIME" --out "$STAGE" --chroms "$CHROMS" --methods "$METHODS" \
        --poisson-n "$POISSON_N"
    rc=$?
fi

# THE IDENTITY ASSERTION IS ITS OWN PASS, over ONE chromosome, AND IT GENERATES NOTHING INTO THE
# ROOT. `--assert-only` re-predicts the collapsed methods under $ASSERT_AGAINST into a temporary
# directory, compares every array, and writes `regime_independent` into their manifests — nothing
# else under the root is touched, so the arrays that get stamped are the arrays the pass above
# wrote. It is a separate pass rather than a flag on that one because `generate.py` refuses
# `--assert-regime-independent` outright when any fitted method is in the list, and refusing it
# there is the point: a mixed METHODS list still gets its stamp this way. Cost is one chromosome
# for two pval-cheap methods.
#
# ASSERT_ONLY=1 runs THIS AND NOTHING ELSE, which is how a root built by the p2 array gets stamped:
#   ASSERT_ONLY=1 METHODS=avg,avg-arcsinh REGIME=... PANEL=V_ sbatch slurm/t49_baselines_p1.sh
if [ $rc -eq 0 ] && [ -n "$COLLAPSED" ] && [ "$ASSERT_AGAINST" != "none" ]; then
    AREG="$(basename "$ASSERT_AGAINST" .json)"; AREG="${AREG#regime.}"
    # Beside the roots as well: the manifest records this file's NAME as `asserted_against`, and a
    # name that resolves to nothing is not provenance.
    ASSERT_REGIME="$REG_DIR/regime.$AREG.$PANEL.json"
    python tools/declare_eval_pairs.py split --regime "$ASSERT_AGAINST" --panel "$PANEL" \
        --out "$ASSERT_REGIME.$$.tmp" && mv -f "$ASSERT_REGIME.$$.tmp" "$ASSERT_REGIME" || exit 2
    echo "[t49] asserting$COLLAPSED regime-independent against $AREG on ${CHROMS%%,*}"
    python -m competitors.baselines.generate \
        --store "$PANEL_REGIME" --out "$STAGE" --chroms "${CHROMS%%,*}" \
        --methods "$(echo $COLLAPSED | tr ' ' ',')" --poisson-n "$POISSON_N" \
        --assert-regime-independent "$ASSERT_REGIME" --assert-only
    rc=$?
fi

# 5 is the identity assertion failing: a method claimed regime-independent was not. Not a crash.
[ $rc -eq 5 ] && echo "[t49] the collapse assertion FAILED — that method runs once per regime (D1)"
echo "[t49] exit=$rc"
for M in ${METHODS//,/ }; do
    ls -la "$PRED_BASE/$M/$REGNAME/$PANEL/manifest.json" 2>/dev/null
done
exit $rc
