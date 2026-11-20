#!/bin/bash
# Close-out D2, step "prep" (one small CPU job; submitted by slurm/onion_data/closeout_d2_submit.sh).
#
#   sbatch --output=<log> closeout_d2_prep.sh <kit dir> <out root D> <base tables> <base evalset sha256>
#
# Writes only under D (about 20 files): an add-on of the filled tables version with the second
# experiments, its check, the add-on eval set, the layer pair file and the pair smoke of every
# other_experiment pair. Reads $O/tables/<base>, $O/catalog/encode, $O/eic/other_experiment_*.tsv,
# /scratch/mforooz/onion/measurements and /scratch/mforooz/onion/work (QC files only), read only.
#   1. evalset.py build on the base tables with the D1 arguments (256 regions, 16 fit regions,
#      seed 0) -> D/eval/<base>/evalset.json; its sha256 is compared with the frozen file on Fir
#      (argument 4). Equal = the same eval set; the add-on eval set then extends the frozen one.
#   2. build_tables.py addon --catalog -> D/tables/<base>x
#   3. check_dataset.py --addon-of <base> -> D/logs (exit code recorded, the job goes on)
#   4. evalset.py addon + check -> D/eval/<base>x/evalset.json
#   5. evalset.py build --pair-splits train,val -> D/eval/<base>x/evalset_layers.json
#   6. other_experiments.py pair-smoke on every other_experiment pair -> D/PAIR_SMOKE_ALL.md
#SBATCH --account=def-maxwl
#SBATCH --job-name=co_d2_prep
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --gres=none
set -uo pipefail
KIT="${1:?usage: closeout_d2_prep.sh <kit> <D> <base tables name> <base evalset sha256>}"
D="${2:?}"; BASE="${3:?}"; WANT_SHA="${4:?}"
O=/project/def-maxwl/mforooz/ONION
M=/scratch/mforooz/onion/measurements
W=/scratch/mforooz/onion/work
T=$O/tables/$BASE
TX=$D/tables/${BASE}x
EVB=$D/eval/$BASE/evalset.json
EVX=$D/eval/${BASE}x/evalset.json
echo "=== co_d2_prep job ${SLURM_JOB_ID:-} host=$(hostname) kit=$KIT base=$T out=$D $(date -u)"
cat "$KIT/.kit_stamp" 2>/dev/null || true
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv" >/dev/null &&
  source "$SLURM_TMPDIR/venv/bin/activate" &&
  pip install -q --no-index --upgrade pip &&
  pip install -q --no-index -r "$KIT/requirements-fir.txt" &&
  pip install -q --no-index --find-links "$KIT/wheels" -r "$KIT/requirements-pypi.txt" ||
  { echo "venv setup failed"; echo "[co-d2] exit=3"; exit 3; }
cd "$KIT"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source "$KIT/slurm/_kit_pin.sh"
mkdir -p "$D/eval/$BASE" "$D/eval/${BASE}x" "$D/tables"
# evalset.py reads the genome at <tables>/../../genome; one symlink makes D/tables/<x> resolve to $O/genome
[ -e "$D/genome" ] || ln -s $O/genome "$D/genome"
step() { echo; echo "=== step $1 $(date -u +%H:%M:%S)"; }

step "1 evalset build on $BASE"
[ -f "$EVB" ] || python tools/onion_eval/evalset.py build --tables "$T" \
  --strata $O/eval/strata/masks.h5 --n-regions 256 --n-fit-regions 16 --seed 0 --out "$EVB" || exit 1
got="$(sha256sum "$EVB" | cut -d' ' -f1)"
echo "evalset sha256 $got; frozen file on Fir $WANT_SHA; $([ "$got" = "$WANT_SHA" ] && echo SAME || echo DIFFERENT)"

step "2 build_tables addon --catalog"
[ -d "$TX" ] || python tools/onion_data/build_tables.py addon --base "$T" \
  --other-build-list $O/eic/other_experiment_build_list.tsv \
  --other-experiments $O/eic/other_experiment_ledger.tsv \
  --measurements $M --work $W --catalog $O/catalog/encode --out "$TX" || exit 1

step "3 check_dataset --addon-of"
python tools/onion_data/check_dataset.py --tables "$TX" --addon-of "$T" --genome-dir $O/genome
echo "check_dataset exit=$?"

step "4 add-on eval set"
[ -f "$EVX" ] || python tools/onion_eval/evalset.py addon --base "$EVB" --tables "$TX" --out "$EVX" || exit 1
python tools/onion_eval/evalset.py check --evalset "$EVX" --tables "$TX"
echo "evalset check exit=$?"

step "5 layer pair file (all pairs, train and val)"
[ -f "$D/eval/${BASE}x/evalset_layers.json" ] || python tools/onion_eval/evalset.py build \
  --tables "$TX" --strata $O/eval/strata/masks.h5 --n-regions 256 --n-fit-regions 16 --seed 0 \
  --pair-splits train,val --out "$D/eval/${BASE}x/evalset_layers.json"
rc=$?; echo "layer pair file exit=$rc"
[ "$rc" = 0 ] || { echo "[co-d2] exit=1"; exit 1; }

step "6 pair smoke on every other_experiment pair"
PAIRS="$(awk -F'\t' -v m="$M" 'NR>1 && $4=="other_experiment"{printf "%s%s/%s:%s/%s", s, m, $2, m, $3; s=","}' "$TX/pairs.tsv")"
if [ -n "$PAIRS" ]; then
  python tools/onion_data/other_experiments.py pair-smoke --pairs "$PAIRS" --evalset "$EVB" \
    --out "$D/PAIR_SMOKE_ALL.md" > "$D/pair_smoke_all.tsv"
  echo "pair smoke exit=$?"
else
  echo "no other_experiment pair in $TX/pairs.tsv"
fi
echo "=== done $(date -u)"
echo "[co-d2] exit=0"
