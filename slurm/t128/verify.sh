#!/bin/bash
# t128: tests on one kit — tests/test_t128_*.py, test_t126_*.py, test_t118_*.py (CPU), in that
# order, each set ending with a `RESULT <set> rc=<rc>` line. This is t126's verify_K2.sh with the
# kit as the one positional argument. Known flake on Nibi CPU (row-2 code, untouched since t118):
# tests/test_t118_xshuf_score.py::test_twin_scoring_is_deterministic[D2].
#
# NO --export. CPU only, 8 cores, 32 GB, 3 h; c128, c166, c537, c371, c329 excluded.
# Usage, from the Nibi login node (the log dir must exist):
#   mkdir -p /project/def-maxwl/mforooz/t128/verify
#   sbatch --parsable $KIT/slurm/t128/verify.sh $KIT
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128_verify
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --output=/project/def-maxwl/mforooz/t128/verify/%x_%j.out
set -uo pipefail
K="${1:?usage: t128/verify.sh <kit_dir>}"
cd "$K" || exit 2
K=$(pwd)
echo "=== t128 verify job ${SLURM_JOB_ID:-none} host=$(hostname) $(date -u) kit=$K sha=$(cat GIT_SHA 2>/dev/null || echo unknown)"
set +u; module load python/3.10.13; set -u
virtualenv --no-download "$SLURM_TMPDIR/venv"
source "$SLURM_TMPDIR/venv/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "$K/requirements-fir.txt" || { echo "FAIL venv fir"; exit 3; }
pip install --no-index --find-links "$K/wheels" -r "$K/requirements-pypi.txt" || { echo "FAIL venv pypi"; exit 3; }
export PYTHONPATH="$K/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
python3 -c 'import candi, pytest; print("venv ok", candi.__file__, pytest.__version__)'
# the library must come from this kit, not from some other install (tests/test_slurm_kit_pin.py)
case "$(python3 -c 'import candi; print(candi.__file__)')" in
  "$K"/src/*) ;;
  *) echo "candi does not import from $K/src" >&2; exit 3 ;;
esac
for set in t128 t126 t118; do
  T0=$SECONDS
  python3 -m pytest tests/test_${set}_*.py -q -p no:cacheprovider --basetemp="$SLURM_TMPDIR/bt_$set" > "$SLURM_TMPDIR/$set.log" 2>&1; RC=$?
  tail -15 "$SLURM_TMPDIR/$set.log"
  grep -E '^(FAILED|ERROR) ' "$SLURM_TMPDIR/$set.log" | sed "s/^/ID $set /"
  echo "RESULT $set rc=$RC ($((SECONDS-T0)) s)"
done
echo "=== verify done $(date -u)"
