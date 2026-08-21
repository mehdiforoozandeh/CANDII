#!/bin/bash
# t25 step 2 — rebuild the corpus manifest so it republishes the NEW codec, then verify.
#
# build-manifest reads `scale`, `transform` and `pval_clip_frac` off each pval.h5's root attrs and
# copies them into the per-track records (D25/D27).  A corpus whose files changed codec but whose
# manifest was not rebuilt keeps claiming `pval_scale: 100` --- and the manifest is exactly where a
# consumer looks to answer "what are the units", which is the complaint that started all of this.
#
# `verify` is membership-checked on the schema, not equality (D27): this rebuild moves pval.h5 to
# schema 2 and leaves counts.h5 / peaks.h5 at schema 1, which is a CORRECT corpus, not a mixed one
# in the dangerous sense.  It also now fails a schema-2 pval file that carries no `transform`.
# --gres note as in t25_rebuild_pval.sh (project hard rule: never another gres spec).
#SBATCH --account=def-maxwl
#SBATCH --job-name=t25_manifest
#SBATCH --output=/project/def-maxwl/mforooz/CANDI_STORE/t25/logs/manifest_%j.out
#SBATCH --error=/project/def-maxwl/mforooz/CANDI_STORE/t25/logs/manifest_%j.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

set -uo pipefail

STORE=/project/def-maxwl/mforooz/CANDI_STORE
CORPUS="${CORPUS:?set CORPUS=eic or merged}"
SRC="${SRC:?set SRC=...}"
# Overridable, and normally overridden. The shared clone at .../CANDII is whatever branch
# someone is working on; a 455-file rebuild must run from a checkout PINNED to the commit
# whose codec it is applying, or "which code built this file" is unanswerable afterwards.
KIT="${KIT:-/home/mforooz/projects/def-maxwl/mforooz/CANDII}"
CSV_ARGS="${CSV_ARGS:?set CSV_ARGS to the --metadata-csv flags, as t12 did}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; unset PYTHONPATH || true
export MPLBACKEND=Agg
source /project/6014832/mforooz/EpiDenoise/candi_venv/bin/activate
cd "$KIT"; export PYTHONPATH="$KIT/src"

echo "[t25] manifest corpus=$CORPUS host=$(hostname) started $(date -Is)"
python -m candi.store build-manifest \
  --corpus-root "$STORE/$CORPUS" --corpus "$CORPUS" \
  $CSV_ARGS \
  --source-root "$SRC"
rc_m=$?
echo "[t25] build-manifest rc=$rc_m $(date -Is)"

echo "[t25] ---------------- verify ----------------"
python -m candi.store verify --corpus-root "$STORE/$CORPUS"
rc_v=$?
echo "[t25] verify rc=$rc_v $(date -Is)"

echo "[t25] ---------------- sha256 ----------------"
sha256sum "$STORE/$CORPUS/manifest.json"

echo "[t25] DONE manifest=$rc_m verify=$rc_v $(date -Is)"
exit $(( rc_m | rc_v ))
