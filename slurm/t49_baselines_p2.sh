#!/bin/bash
# The naive baselines, GENERATED genome-wide — §4's `genome-wide` scope, which these five do
# still print (unlike Avocado/ChromImpute/Lavawizard, whose cell §4 blanks).
#
#   mkdir -p slurm-logs
#   REGIME=configs/regime.eic_19.json PANEL=V_ sbatch --array=0-23 slurm/t49_baselines_p2.sh
#
# One array task per chromosome, all writing into the SAME pinned prediction roots — a track
# directory collects one `chr*.npz` per task and is only complete when every task has finished,
# which is why scoring is a separate, dependent job (`slurm/t49_baselines_score.sh`).
# `bench.external` refuses a track that covers only some of the scored chromosomes, so a partial
# array is a loud failure at score time rather than a quiet one.
#
# HOW MANY TIMES EACH BASELINE RUNS — D1, SETTLED 2026-09-01. `plan/BENCHMARK_DESIGN.md` §12.2 first
# ruled all five run ONCE rather than once per regime. That holds for `avg` and `avg-arcsinh`, whose
# every written bin is a function of the contributors AT THE PREDICTED POSITION, and it is false for
# `knn1`/`knn5` (`generate.similarity_table` correlates over `panel.train_chroms`) and `marginal`
# (`generate.fit_marginal` pools over them). Eight baseline method-regime units, not five.
# `slurm/t49_baselines_p1.sh`'s header carries the full statement, the identity assertion that
# licenses the collapse, and why `regime.eic_pilot.json` is still refused for the fitted three.
#
# THIS ARRAY DOES NOT STAMP `regime_independent`, and a root without that stamp is refused against
# the other board by `t49_baselines_score.sh`. After the array finishes, stamp it once:
#
#   ASSERT_ONLY=1 METHODS=avg,avg-arcsinh REGIME=$REGIME PANEL=V_ sbatch slurm/t49_baselines_p1.sh
#
# One chromosome, two pval-cheap methods, minutes. It is a separate job because the check is a
# read-modify-write on a shared manifest and 23 concurrent tasks would race it.
#
# THIS ARRAY IS `V_` ONLY. `B_` is written ONCE (§5) and the once-only guard is a check on the
# target root, which every task of an array would race. A genome-wide `B_` root comes from
# `t49_baselines_p1.sh` with `PANEL=B_ B_ONCE=1` and `CHROMS` set to all 23 — one job, one root, one
# manifest. Generation is the cheap half (~4.5 min for all five methods on chr21, so ~5 h
# genome-wide), so serialising it costs far less than a raced blind panel.
#
# THE POISSON FLOOR IS §5.1's PRE-REGISTERED 1e6 AGAIN. It was 1e4 here because
# `candi.metrics.nb_crps` returned NaN above ~2e4 and the count arm shipped with no CRPS tier at
# all; t56 fixed that (`test_the_preregistered_poisson_floor_is_scoreable_by_candi_bench`). The
# value is stamped in every manifest as `poisson_n`, and it is part of `_MANIFEST_IDENTITY`, so two
# tasks generating at different floors into one root fail loudly instead of merging.
#
# READ GENOME-WIDE NUMBERS WITH §4's IN-SAMPLE BADGE IN MIND, plus one specific to these baselines:
# the pass covers the regime's TRAIN chromosomes, so the kNN similarity ranking and the per-assay
# marginal are IN-SAMPLE there. `avg` is unaffected — its exclusion rule is over cells, not
# positions, which is the same fact that makes `avg` and `avg-arcsinh` the only two of the five that
# genuinely collapse to one run.
#
# CPU ONLY, AND NO GRES ANY MORE: `generate.py` is numpy + h5py and imports no torch. The MIG slice
# this script used to request was a fairshare workaround (slurm/bake.sh) for a job that never opened
# CUDA. Invariant 13 allows the 1g.10gb slice or nothing; this is nothing.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t49_p2gen
#SBATCH --output=slurm-logs/t49_p2gen_%A_%a.out
#SBATCH --error=slurm-logs/t49_p2gen_%A_%a.err
#SBATCH --time=11:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -uo pipefail

KIT="${KIT:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_19.json}"
PANEL="${PANEL:-V_}"
METHODS="${METHODS:-avg,avg-arcsinh,knn1,knn5,marginal}"
COLLAPSE_REGIME="${COLLAPSE_REGIME:-eic_19}"
POISSON_N="${POISSON_N:-1e6}"
V_PRED_ROOT="${V_PRED_ROOT:-/scratch/$USER/t81_pred}"

CHROM_LIST=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 \
            chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)
CHROM="${CHROM_LIST[${SLURM_ARRAY_TASK_ID:-20}]}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
module load StdEnv/2023 python/3.10.13 >/dev/null 2>&1
source "$VENV/bin/activate" || { echo "[error] no venv at $VENV" >&2; exit 1; }
cd "$KIT"
source "$KIT/slurm/_kit_pin.sh"
export PYTHONPATH="$KIT/src:$KIT"

if [ "$PANEL" != "V_" ]; then
    echo "[t49-p2] REFUSING: PANEL=$PANEL. This array is V_ only — B_ is written once and an" >&2
    echo "         array would race the once-only guard. Use t49_baselines_p1.sh with PANEL=B_," >&2
    echo "         B_ONCE=1 and CHROMS set to all 23." >&2
    exit 4
fi
REGNAME="$(basename "$REGIME" .json)"; REGNAME="${REGNAME#regime.}"

COLLAPSED=""; DEPENDENT=""
for M in ${METHODS//,/ }; do
    case "$M" in
        avg|avg-arcsinh)     COLLAPSED="$COLLAPSED $M" ;;
        knn1|knn5|marginal)  DEPENDENT="$DEPENDENT $M" ;;
        *) echo "[t49-p2] REFUSING: $M is not a baseline method." >&2; exit 2 ;;
    esac
done

if [ -n "$COLLAPSED" ] && [ "$REGNAME" != "$COLLAPSE_REGIME" ]; then
    echo "[t49-p2] REFUSING:$COLLAPSED collapse to ONE run (D1), under $COLLAPSE_REGIME. Drop" >&2
    echo "         them from METHODS for a $REGNAME pass." >&2
    exit 2
fi

if [ -n "$DEPENDENT" ] && \
   python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('regions') else 1)" "$REGIME"; then
    echo "[t49-p2] REFUSING: $REGIME declares a \`regions\` BED and$DEPENDENT fit on the regime's" >&2
    echo "         training loci. generate.py has no regions support, so they would fit over WHOLE" >&2
    echo "         chromosomes instead of the declared region set. Rule 2 break — raise it." >&2
    exit 3
fi

# The panel regime: the DECLARED eval_pairs filtered to V_. Both live regimes carry all 38 pairs
# (26 V_ + 12 B_), so without this the array would predict B_ on the way past.
#
# IT LIVES BESIDE THE ROOTS, NOT IN $SLURM_TMPDIR, AND THAT IS LOAD-BEARING. `generate.py` records
# the regime PATH in each manifest and `regime` is one of the `_MANIFEST_IDENTITY` fields, so every
# task of this array must be handed the SAME path or the second one to finish refuses to merge and
# the root loses a chromosome. $SLURM_TMPDIR is per node. Written through a rename so a task never
# reads a half-file; the content is a deterministic function of $REGIME and $PANEL.
WORK="${SLURM_TMPDIR:-/tmp}/t49_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
REG_DIR="$V_PRED_ROOT/_regimes"
mkdir -p "$WORK" "$REG_DIR"
PANEL_REGIME="$REG_DIR/regime.$REGNAME.$PANEL.json"
python tools/declare_eval_pairs.py split --regime "$REGIME" --panel "$PANEL" \
    --out "$PANEL_REGIME.$$.tmp" && mv -f "$PANEL_REGIME.$$.tmp" "$PANEL_REGIME" \
    || { echo "[t49-p2] could not derive the $PANEL regime" >&2; exit 2; }
python - "$PANEL_REGIME" "$PANEL" <<'PYEOF' || exit 2
import json, sys
pairs = json.load(open(sys.argv[1])).get("eval_pairs", [])
bad = [p for p in pairs if not str(p[1]).startswith(sys.argv[2])]
if bad or not pairs:
    sys.exit(f"[t49-p2] REFUSING: {sys.argv[1]} carries {len(bad)} pair(s) outside {sys.argv[2]} "
             f"({len(pairs)} declared). The split did not do what its name says.")
PYEOF

# The pinned roots are <Method>/<regime>/V_ and `generate.py` writes <out>/<method>, so each
# method's root is reached through a per-task symlink named for the method. One pass over the store
# then serves every method — this array reads ~370 GB in total and must not read it once per method.
STAGE="$WORK/roots"
mkdir -p "$STAGE"
for M in ${METHODS//,/ }; do
    R="$V_PRED_ROOT/$M/$REGNAME/$PANEL"
    mkdir -p "$R" && ln -sfn "$R" "$STAGE/$M" || exit 1
done

echo "[t49-p2] host=$(hostname) commit=$(git rev-parse --short HEAD) regime=$REGNAME panel=$PANEL"
echo "[t49-p2] chrom=$CHROM methods=$METHODS n=$POISSON_N roots=$V_PRED_ROOT/<M>/$REGNAME/$PANEL"

python -m competitors.baselines.generate \
    --store "$PANEL_REGIME" --out "$STAGE" --chroms "$CHROM" --methods "$METHODS" \
    --poisson-n "$POISSON_N"
rc=$?
echo "[t49-p2] chrom=$CHROM exit=$rc"
exit $rc
