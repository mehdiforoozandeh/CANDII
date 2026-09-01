#!/bin/bash
# The naive baselines, GENERATED genome-wide — §4's `genome-wide` scope, which these five do
# still print (unlike Avocado/ChromImpute/Lavawizard, whose cell §4 blanks).
#
# RETARGETED 2026-08-31 for plan/BENCHMARK_DESIGN.md's two live regimes, and READ THE NEXT
# PARAGRAPH BEFORE LAUNCHING — the §12.2 collapse does NOT hold for all five methods.
#
# §12.2 rules that the five naive baselines run ONCE, not once per regime, "because their fit is
# regime-independent, so is their output: the two regimes would produce byte-identical predictions",
# and asks for an ASSERTION rather than an argument. THE ASSERTION DOES NOT EXIST — there is no test
# in tests/ or in competitors/ that predicts under both regimes and compares — AND THE CODE SAYS THE
# CLAIM IS FALSE FOR THREE OF THE FIVE:
#
#   avg, avg-arcsinh   regime-independent. The contributor set is `biosamples.train` minus the
#                      target's cell type, and the prediction is a per-bin function of the
#                      contributors AT THE PREDICTED POSITION. No training locus enters. Collapse
#                      to one run: correct.
#   knn1, knn5         `generate.similarity_table` correlates over `panel.train_chroms`
#                      (generate.py:207-241). Different train_chroms -> a different similarity
#                      ranking -> different predictions. NOT regime-independent.
#   marginal           `generate.fit_marginal` pools over `panel.train_chroms`
#                      (generate.py:262-300). NOT regime-independent.
#
# So it is 2 runs collapsed and 3 runs x 2 regimes = 8 method-regime units, not 5 — unless the PI
# rules otherwise. And `generate.py` reads `train_chroms` RAW: it has no `regions` support at all,
# so under eic.pilot knn/marginal would fit over 18 WHOLE chromosomes (~2.7 Gbp) instead of the
# 25,588,197 bp the regime declares. That is a Rule 2 break, and it is why REGIME defaults to
# eic_19 here and eic_pilot is refused below.
#
#   mkdir -p slurm-logs && sbatch --array=0-22 slurm/t49_baselines_p2.sh
#
# One array task per chromosome, all writing into the SAME prediction roots — a track directory
# collects one `chr*.npz` per task and is only complete when every task has finished, which is why
# scoring is a separate, dependent job (`t49_baselines_p2_score.sh`). `bench.external` refuses a
# track that covers only some of the scored chromosomes, so a partial array is a loud failure at
# score time rather than a quiet one.
#
# THE POISSON FLOOR HERE IS 1e4, NOT §5.1's 1e6, and that is a deviation on the record.
# `candi.metrics.nb_crps` cannot evaluate 1e6 (see `competitors/baselines/README.md`), so at the
# pre-registered value P2's count arm would ship with no CRPS tier at all. P1 is generated at BOTH
# values so the PI can see the difference cheaply; P2 reads ~370 GB off the store per pass and is
# not run twice to make the same point. The value is stamped in every manifest as `poisson_n`.
#
# READ GENOME-WIDE NUMBERS WITH §4's IN-SAMPLE BADGE IN MIND, plus one specific to these baselines:
# the pass covers the regime's TRAIN chromosomes, so the kNN similarity ranking and the per-assay
# marginal are IN-SAMPLE there. `avg` is unaffected — its exclusion rule is over cells, not
# positions, which is the same fact that makes `avg` and `avg-arcsinh` the only two of the five that
# genuinely collapse to one run.
#
# WHY --gres ON A CPU-ONLY JOB: see slurm/bake.sh.
#SBATCH --account=def-maxwl
#SBATCH --job-name=candi_t49_p2gen
#SBATCH --output=slurm-logs/t49_p2gen_%A_%a.out
#SBATCH --error=slurm-logs/t49_p2gen_%A_%a.err
#SBATCH --time=11:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -uo pipefail

KIT="${KIT:-/project/def-maxwl/$USER/CANDII_t78_code}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME="${REGIME:-configs/regime.eic_19.json}"
PRED="${PRED:-/project/def-maxwl/$USER/t49_baselines/p2/preds}"
METHODS="${METHODS:-avg,avg-arcsinh,knn1,knn5,marginal}"
POISSON_N="${POISSON_N:-1e4}"

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

# The old guard refused regime.eic_test.json, the separate B-pair regime. THE LIVE REGIMES CARRY
# THE B_ PAIRS INSIDE THEM — eic_19 and eic_pilot each declare 38 eval_pairs, 26 V_ and 12 B_ — so
# a name check protects nothing any more. §5 rules B_ is touched ONCE, at the very end. Derive a
# V_-only regime the way slurm/t81_train_candi.sh does, and refuse a regime that still has B_ in it
# unless this IS the once-only B_ run.
case "$REGIME" in *eic_test*) echo "[t49] REFUSING: $REGIME is the B-pair regime (A4)"; exit 2;; esac
if [ "${ALLOW_B_PAIRS:-0}" != "1" ]; then
  python - "$REGIME" <<'PYEOF' || exit 2
import json, sys
d = json.load(open(sys.argv[1]))
b = [p for p in d.get("eval_pairs", []) if str(p[1]).startswith("B_")]
if b:
    sys.exit(f"[t49] REFUSING: {sys.argv[1]} declares {len(b)} B_ eval pair(s). BENCHMARK_DESIGN "
             f"\u00a75 touches B_ ONCE, from the selected checkpoint. Derive a V_-only regime "
             f"(see slurm/t81_train_candi.sh), or set ALLOW_B_PAIRS=1 if this IS the final B_ run.")
PYEOF
fi
if python -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('regions') else 1)" "$REGIME"; then
  echo "[t49] REFUSING: $REGIME declares a \`regions\` BED. competitors/baselines/generate.py reads" >&2
  echo "      train_chroms raw and has no regions support, so knn1/knn5/marginal would fit over" >&2
  echo "      WHOLE chromosomes instead of the Pilot Regions. Rule 2 break. Raise it." >&2
  exit 3
fi
echo "[t49-p2] host=$(hostname) commit=$(git rev-parse --short HEAD) chrom=$CHROM n=$POISSON_N"

mkdir -p "$PRED"
python -m competitors.baselines.generate \
    --store "$REGIME" --out "$PRED" --chroms "$CHROM" --methods "$METHODS" \
    --poisson-n "$POISSON_N"
rc=$?
echo "[t49-p2] chrom=$CHROM exit=$rc"
exit $rc
