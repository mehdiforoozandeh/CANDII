#!/bin/bash
# The σ table, fit on TRAINING-track residuals. New 2026-09-01.
#
# WHY THIS STAGE EXISTS. CRPS needs a per-assay σ, and until now every σ in this repo was fit by
# the old per-method fitter, on the residuals of the V_ EVAL pairs. BENCHMARK_DESIGN.md §7 rules
# the opposite — "sigma is fit on training-set residuals only — never on V_, never on B_" — and
# §12.2 declares every σ fit the old way VOID. It was never a flag to flip: the eval prediction
# root holds only the declared eval tracks, so there were no training-track predictions to take a
# residual against. Producing them is what this script does, and it is why the old per-method
# fitter is now dead code that no launcher calls.
#
# THE SHAPE, in three steps and no more:
#
#   1. DERIVE a training regime — `tools/sigma_training_regime.py` samples 12 training cells with
#      seed 890217 and declares them as SELF pairs, [T_x, T_x]. A self-pair's "imputation" target is
#      a track the model was fit on, so the residual is a training residual by construction and no
#      V_ or B_ track is opened at any point.
#   2. PREDICT those self-pairs into the σ prediction root on /scratch. Lavawizard predicts a
#      chromosome from that chromosome's own position tables, and the eval-side fits cover
#      chr20/21/22 only, so a σ chromosome gets its own frozen-trunk `genome` fit here. That fit is
#      inference under Rule 2 — position tables, nothing transferable — and it runs with
#      --select-every 0, because a training chromosome has no V_ panel to select on.
#   3. FIT with `competitors.sigma_pass`, the one fitter every method now shares. It writes
#      `fitted_on: "training-residuals: ..."`, and every score.sh refuses a table whose `fitted_on`
#      does not start with that.
#
# THE CACHE AND THE POSITION TABLES READ THE SOURCE REGIME, NEVER THE DERIVED ONE. `train_columns`
# (store_eic.py:138) refuses a regime whose eval_pairs target a biosample that is also in
# biosamples.train — which is exactly what a self-pair is. Only the PREDICT step needs the derived
# regime, and predict goes through `_declared_tracks`, not `train_columns`, so the split each guard
# was written for is the split it sees.
#
# WHICH TRAINING CHROMOSOME, AND WHY IT IS NOT THE FIRST ONE. The σ chromosome must be a TRAINING
# chromosome of the regime whose `dataset3.UPSTREAM_HYPERPARAMS` row is the row the shared stem
# borrows (`store_eic.shared_hparams_chrom`). That row fixes the three position-factor widths,
# `model.Factors.out_features` sums them into `block1.dense`'s input width, and that weight is a
# TRANSFERABLE tensor — a stem fit under one row cannot load into a model built under another.
# `python -m lavawizard.sigma_chrom` is the whole rule: the default is the first training
# chromosome on that row, in the regime's own `train_chroms` order, and a SIGMA_CHROMS the operator
# names is held to the same rule and refused with exit 3 before any GPU is spent.
#
# It is a real constraint and not a formality. Fir job 57833682 (eic_pilot, 2026-09-02) took the
# old default — the FIRST `train_chroms` entry, chr1 — built its cache for 7 minutes, and died in
# `model.load_transferable` with `size mismatch for block1.dense.weight: [2048, 225] vs
# [2048, 175]`: chr1 is `(10, 10, 45)` and the stem's borrowed chr20 row is `(25, 30, 60)`. Under
# eic_19 the first training chromosome IS chr19, on the eval row, so the default had never had to
# choose. Both live regimes now answer chr19, and neither offers a second candidate — of
# eic_pilot's eighteen training chromosomes only chr19 carries the eval row.
#
# ONE CHROMOSOME BY DEFAULT, and that is a sizing choice worth stating. A σ entry is one scalar per
# assay pooled over 12 cells; on chr19 that is ~2.3 M bins x 12 cells per assay, and adding
# chromosomes moves the third decimal while costing a full cache build and a position-table fit
# each. SIGMA_CHROMS=chr19,chr7 is how one WOULD widen it, and it is refused here — chr7 is off the
# row. Widening is an affordance for a later regime, not one either live regime can use; the fitter
# records what it actually read.
#
# KNOWN INTEGRATION EDGE, recorded rather than worked around: `candi.store.regime:597` refuses a
# regime whose eval_pairs target a train biosample, and `:589` refuses train_chroms and eval_chroms
# that overlap. `tools/sigma_training_regime.py` is what has to come out the far side of both; this
# launcher only names it.
#
#   sbatch competitors/lavawizard/slurm/sigma.sh
#   sbatch --export=ALL,REGIME=$PWD/configs/regime.eic_pilot.json \
#       competitors/lavawizard/slurm/sigma.sh
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t53_sigma
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# fc30560 gave repeated Lustre OSError 108 on files its neighbours read fine (anchor run).
#SBATCH --exclude=fc30560
# `$0` is sbatch's spool copy, not this file, so the path comes from the submit directory.
ENV="${SLURM_SUBMIT_DIR:-$PWD}/competitors/lavawizard/slurm/_env.sh"
[ -f "$ENV" ] || { echo "[error] no $ENV -- submit from the repo root" >&2; exit 2; }
source "$ENV"

SIGMA="${SIGMA:-$SIGMA_JSON}"
PRED="${PRED:-$TRAIN_PRED}"
CKPT="${CKPT:-$RUNS/ckpt}"
SIGMA_REGIME="$RUNS/regime.${REGIME_NAME}.sigma_train.json"
SHARED_CK="$CKPT/guacamole_${SHARED_STEM}.pt"
mkdir -p "$(dirname "$SIGMA")" "$PRED"

if [ -s "$SIGMA" ]; then
  echo "[sigma] $SIGMA exists, keeping it (a refit would change what every CRPS means)"
  exit 0
fi
[ -s "$SHARED_CK" ] || { echo "[sigma] no shared checkpoint at $SHARED_CK — run STAGE=shared train.sh first" >&2; exit 1; }

# 1. the training regime.
python "$REPO/tools/sigma_training_regime.py" \
    --regime "$REGIME" --n-cells 12 --seed 890217 --out "$SIGMA_REGIME" || exit 1
echo "[sigma] training regime: $SIGMA_REGIME"

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

# WHICH of them, and the one constraint only this method has — see the header. Reads the SOURCE
# regime, because the row the stem borrows is the SOURCE's eval_chroms' row and the derived file's
# `eval_chroms` are the training slice. Seconds on a login node: json and two small tables, no
# torch. Line 1 of its stdout is the chromosomes, line 2 is the stem's row for the banner.
#
# `--chroms ""` and no `--chroms` mean the same thing to the module, which is what lets an unset
# SIGMA_CHROMS ride through one unconditional command line rather than an array this script would
# then have to expand under `set -u`.
if ! _SIG_PICK="$(python -m lavawizard.sigma_chrom --regime "$REGIME" \
                         --chroms "${SIGMA_CHROMS:-}")"; then
  echo "[sigma] refusing to spend a cache build and a GPU on a σ chromosome the shared stem cannot" \
       "transfer into. Set SIGMA_CHROMS to one of the chromosomes named above, or leave it unset." >&2
  exit 3
fi
{ read -r SIGMA_CHROMS; read -r _SIG_ROW; } <<< "$_SIG_PICK"
IFS=, read -r -a SIG_CHROMS <<< "$SIGMA_CHROMS"
# The chooser read the source and the predict/fit steps below read the derived file. They agree by
# construction (`sigma_training_regime.derive` sets eval_chroms = the source's train_chroms), and
# this is where a drift between the two would show up instead of in a cache build.
for C in "${SIG_CHROMS[@]}"; do
  case ",$_SIG_ALL," in
    *",$C,"*) ;;
    *) echo "[sigma] $C is on the stem's row but the derived regime offers only $_SIG_ALL" >&2
       exit 3 ;;
  esac
done
echo "[sigma] residual chromosomes: ${SIG_CHROMS[*]}  (regime offers $_SIG_ALL)"
echo "[sigma] on the shared stem's borrowed row: $_SIG_ROW"

# 2. predict the self-pairs, one chromosome at a time.
FIRST=1
for C in "${SIG_CHROMS[@]}"; do
  if [ ! -d "$CACHE/$C" ]; then
    echo "[sigma] caching $C from the store"
    srun python -u -m lavawizard.store_eic cache --regime "$REGIME" --chrom "$C" --cache "$CACHE" \
      || exit 1
  fi
  W="$CKPT/guacamole_${C}.pt"
  if [ ! -f "$W" ]; then
    echo "[sigma] fitting $C position tables (trunk frozen, no selection)"
    srun python -u -m lavawizard.store_eic train --regime "$REGIME" --cache "$CACHE" \
         --stage genome --chrom "$C" --init "$SHARED_CK" --out "$CKPT" \
         --contributor-mode loo --device "${DEVICE:-cuda}" --select-every 0 || exit 1
  fi
  MAN=(); [ "$FIRST" = "1" ] && MAN=(--manifest); FIRST=0
  echo "[sigma] predicting the self-pairs on $C -> $PRED"
  srun python -u -m lavawizard.store_eic predict --regime "$SIGMA_REGIME" --chrom "$C" \
       --cache "$CACHE" --checkpoint "$W" --pred-root "$PRED" \
       --device "${DEVICE:-cuda}" --clip "${MAN[@]}" || exit 1
done

# 3. the fit itself. `--method` is the board slug, so the table says whose residuals it holds.
FIT=(--regime "$SIGMA_REGIME" --pred "$PRED" --out "$SIGMA" --method "$METHOD"
     --chroms "$SIGMA_CHROMS")
[ -n "$_SIG_BED" ] && FIT+=(--eval-regions "$_SIG_BED")
srun python -u -m competitors.sigma_pass "${FIT[@]}" || exit 1

# It has to say what it is, or score.sh will refuse it in three hours' time instead of now.
python - "$SIGMA" "$SIGMA_FITTED_ON_PREFIX" <<'PYEOF' || exit 3
import json, sys
got = str(json.load(open(sys.argv[1]))["fitted_on"])
if not got.startswith(sys.argv[2]):
    sys.exit(f"[sigma] the fitter wrote fitted_on={got!r}, which no score.sh will accept")
print(f"[sigma] wrote {sys.argv[1]}: {got}")
PYEOF
