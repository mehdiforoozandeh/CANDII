#!/bin/bash
# Avocado stage 3b -- the σ table, fit on TRAINING-track residuals. New 2026-09-01.
#
# WHY THIS STAGE EXISTS. CRPS needs a per-assay σ, and until now every σ in this repo was fit by
# the old per-method fitter, on the residuals of the V_ EVAL pairs. BENCHMARK_DESIGN.md §7 rules
# the opposite -- "sigma is fit on training-set residuals only -- never on V_, never on B_" -- and
# §12.2 declares every σ fit the old way VOID. It was never a flag to flip: the eval prediction
# root holds only the declared eval tracks, so there were no training-track predictions to take a
# residual against. Producing them is what this script does, and it is why the old per-method
# fitter is now dead code that no launcher calls.
#
# THE SHAPE, in three steps and no more:
#
#   1. DERIVE a training regime -- `tools/sigma_training_regime.py` samples 12 training cells with
#      seed 890217 and writes them into `biosamples.eval` with NO `eval_pairs` at all. A self-pair's
#      "imputation" target is a track the model was fit on, so the residual is a training residual
#      by construction and no V_ or B_ track is opened at any point.
#   2. PREDICT those self-pairs into the σ prediction root on /scratch. Avocado needs this
#      chromosome's genomic factors to predict it at all, and the eval-side fits cover chr20/21/22
#      only, so a σ chromosome gets its own frozen-trunk fit here (`sigma_<chrom>.pt`). That fit is
#      inference under Rule 2 -- per-chromosome position factors, nothing transferable -- and it
#      reads the SOURCE regime, never the derived one, so the training-column guard sees the split
#      it was written for.
#   3. FIT with `competitors.sigma_pass`, the one fitter every method now shares. It writes
#      `fitted_on: "training-residuals: ..."`, and every score.sh refuses a table whose `fitted_on`
#      does not start with that.
#
# ONE CHROMOSOME BY DEFAULT, and that is a sizing choice worth stating. A σ entry is one scalar per
# assay pooled over 12 cells; on chr19 that is ~2.3 M bins x 12 cells per assay, and adding
# chromosomes moves the third decimal while costing a full binning pass and a genomic-factor fit
# SIGMA_PREDICT_BATCHPOS (default 2048): positions per forward in the self-pair predict. The sigma
# panel is 98 tracks/cell (V_ is 45); at predict.py's default 8192 the bf16 activation is
# 8192 x 98 x 2048 x 2 B = 3.06 GiB and OOMs a 1g.10gb MIG slice (jobs 57865902/57865905,
# 2026-09-02). The batch only chunks a stateless forward, so the arrays are bit-identical.
# each. Set SIGMA_CHROMS=chr19,chr7 to widen it; the fitter records what it actually read.
#
# HOW THE SELF-PAIRING IS SPELLED, AND WHY AVOCADO NEEDS NO CHANGE FOR IT. `candi.store.regime`
# refuses a `[c, c]` pair outright, refuses an eval_pairs target that is also in `biosamples.train`
# (as does `competitors/avocado/index.py`), and refuses train_chroms overlapping eval_chroms. So the
# derived file carries NO eval_pairs and names the drawn cells in `biosamples.eval`, which is
# `bench.harness.StoreSource`'s documented no-pairing path: it self-pairs every cell of the pool and
# `targets` is every assay that cell holds. `predict.py` walks `_expected(open_source(...))` — the
# harness itself — so it reads that path for free and this launcher only names it.
#
# A METHOD THAT READS `eval_pairs` DIRECTLY DOES NOT GET IT FOR FREE, and Lavawizard's
# `store_eic._declared_tracks` is the one that did: on this shape it returned an empty track list
# and wrote a manifest over an empty root. It now takes the same self-pairing fallback (2026-09-01).
# ChromImpute reaches its targets through `targets_sigma.tsv`, written by its own σ launcher.
#
# SIGMA_ALLOW_MISSING=1 fits over the tracks the predict pass DID write, instead of refusing the
# root. Set it only after reading the predictor's own skip lines and confirming every skipped track
# is a rare-mark empty leave-one-out pool -- the json's `skipped_tracks` then names them and the
# board quotes them. Default 0, because the refusal is what catches a half-written root.
#
#   mkdir -p slurm-logs && sbatch competitors/avocado/slurm/sigma.sh
#   mkdir -p slurm-logs && sbatch --export=ALL,REGIME=$PWD/configs/regime.eic_pilot.json \
#       competitors/avocado/slurm/sigma.sh
#
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_avo_sigma
#SBATCH --output=slurm-logs/t81_avo_sigma_%j.out
#SBATCH --error=slurm-logs/t81_avo_sigma_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=6

source "${SLURM_SUBMIT_DIR:-$PWD}/competitors/avocado/slurm/_env.sh"
cd "$REPO"

SIGMA="${SIGMA:-$SIGMA_JSON}"
PRED="${PRED:-$TRAIN_PRED}"
SIGMA_REGIME="$WS/regime.${REGIME_NAME}.sigma_train.json"
SHARED_CK="$WS/ckpt/shared_${SHARED_SCOPE}.pt"
SIGMA_ALLOW_MISSING="${SIGMA_ALLOW_MISSING:-0}"
mkdir -p "$(dirname "$SIGMA")" "$PRED"
echo "[avo_sigma] sigma_allow_missing=$SIGMA_ALLOW_MISSING"

if [ -s "$SIGMA" ]; then
    echo "[avo_sigma] $SIGMA exists, keeping it (a refit would change what every CRPS means)"
    exit 0
fi
[ -s "$SHARED_CK" ] || { echo "[avo_sigma] no shared checkpoint at $SHARED_CK -- run train.sh MODE=shared first" >&2; exit 1; }

# 1. the training regime.
python "$REPO/tools/sigma_training_regime.py" \
    --regime "$REGIME" --n-cells 12 --seed 890217 --out "$SIGMA_REGIME" || exit 1
echo "[avo_sigma] training regime: $SIGMA_REGIME"

# The chromosomes to take residuals on, and the BED if the regime carries one. Read off the DERIVED
# file so that what is predicted and what is fit are read from one place.
_SIG_RG="$(python - "$SIGMA_REGIME" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(",".join(d["eval_chroms"]))
print((d.get("regions") or {}).get("bed", ""))
PYEOF
)" || exit 1
{ read -r _SIG_ALL; read -r _SIG_BED; } <<< "$_SIG_RG"
SIGMA_CHROMS="${SIGMA_CHROMS:-${_SIG_ALL%%,*}}"
IFS=, read -r -a SIG_CHROMS <<< "$SIGMA_CHROMS"
echo "[avo_sigma] residual chromosomes: ${SIG_CHROMS[*]}  (regime offers $_SIG_ALL)"

# 2. predict the self-pairs, one chromosome at a time.
for C in "${SIG_CHROMS[@]}"; do
    if [ ! -s "$WS/binned/${C}.npy" ]; then
        echo "[avo_sigma] binning $C"
        python "$AVO/bin_store.py" --regime "$REGIME" --out "$WS/binned" --chrom "$C" || exit 1
    fi
    CK="$WS/ckpt/sigma_${C}.pt"
    if [ ! -s "$CK" ]; then
        echo "[avo_sigma] fitting $C genomic factors (trunk frozen, no selection)"
        python "$AVO/train.py" --regime "$REGIME" --chrom "$C" --mode genome \
            --data-root "$WS/binned" --out "$CK" --init "$SHARED_CK" \
            --log "$WS/logs/sigma_${C}.jsonl" \
            --epochs "${SIGMA_EPOCHS:-30}" --batch-positions "${BATCHPOS:-1024}" \
            --lr "${LR:-1e-3}" --genome-lr "${GENOMELR:-1e-2}" --seed "${SEED:-0}" \
            --max-hours "${MAXHOURS:-5.0}" || exit 1
    fi
    echo "[avo_sigma] predicting the self-pairs on $C -> $PRED"
    python "$AVO/predict.py" --regime "$SIGMA_REGIME" --chrom "$C" \
        --shared "$SHARED_CK" --genome "$CK" --out "$PRED" \
        --batch-positions "${SIGMA_PREDICT_BATCHPOS:-2048}" || exit 1
done
python "$AVO/predict.py" --regime "$SIGMA_REGIME" --out "$PRED" --write-manifest \
    --version 005-port \
    --notes "training-residual σ pass, regime $REGIME_NAME, chroms ${SIGMA_CHROMS}" || exit 1

# 3. the fit itself. `--method` is the board slug, so the table says whose residuals it holds.
FIT=(--regime "$SIGMA_REGIME" --pred "$PRED" --out "$SIGMA" --method "$METHOD"
     --chroms "$SIGMA_CHROMS")
[ -n "$_SIG_BED" ] && FIT+=(--eval-regions "$_SIG_BED")
# See the header: on 1 the fitter fits the present tracks and names the rest in `skipped_tracks`.
[ "$SIGMA_ALLOW_MISSING" = "1" ] && FIT+=(--allow-missing)
python -m competitors.sigma_pass "${FIT[@]}" || exit 1

# It has to say what it is, or score.sh will refuse it in three hours' time instead of now.
python - "$SIGMA" "$SIGMA_FITTED_ON_PREFIX" <<'PYEOF' || exit 3
import json, sys
got = str(json.load(open(sys.argv[1]))["fitted_on"])
if not got.startswith(sys.argv[2]):
    sys.exit(f"[avo_sigma] the fitter wrote fitted_on={got!r}, which no score.sh will accept")
print(f"[avo_sigma] wrote {sys.argv[1]}: {got}")
PYEOF
