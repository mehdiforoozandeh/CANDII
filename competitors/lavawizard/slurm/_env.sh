# Shared environment for every Lavawizard job on Fir. Sourced, never run.
#
# `candi_venv` and not the port's own `torch_env`: `store_eic.py` imports `candi`, and `candi`
# imports `x_transformers`, which `torch_env` does not carry. The Dataset-3 side still runs in
# `torch_env` — see the README's split.
#
# RETARGETED 2026-08-31 for plan/BENCHMARK_DESIGN.md's two live regimes.
#
#   * REGIME was configs/regime.eic_val.json — train chr19, eval chr21, 26 V_ pairs. The live
#     regimes are eic_19 and eic_pilot, and they eval on chr20+chr21+chr22 (§4).
#   * CHROMS was a hard-coded list of all 23. §4 blanks Lavawizard's `genome-wide` cell and rules
#     that a blanked cell is NOT COMPUTED, so the only scope this method is ever predicted or
#     scored on is the regime's eval_chroms — three chromosomes, not 23. The list is now read off
#     the regime, so it cannot drift from it.
#
# THE REGIME AXIS IS REAL AS OF 2026-09-01, AND IT WAS NOT BEFORE.
#
# `train.py` used to fit ONE INDEPENDENT Guacamole PER CHROMOSOME — cell factors, assay factors,
# the dense network and the three position tables, all from a fresh init, all on that chromosome's
# own bins. The cell and assay factors are TRANSFERABLE parameters by BENCHMARK_DESIGN.md §2's own
# definition, so they were fit on chr20/21/22, the chromosomes the method is scored on. That broke
# Rule 2 in BOTH regimes, and it also meant `train_chroms` and `regions` were read by no lavawizard
# module at all: every key this method DID read was identical between eic.19 and eic.pilot, so the
# two rows would have been the same run twice.
#
# PI ruling 2026-09-01: a transferable stage, on Avocado's scheme (§3.2, §12.2).
#
#   STAGE=shared   ONE task. Fits everything on the regime's own training scope — chr19 under
#                  eic.19, the Pilot Regions of eighteen chromosomes under eic.pilot, packed onto
#                  one axis by `store_eic.shared_layout`. Neither scope holds a bin of chr20, 21
#                  or 22. This is the whole regime axis.
#   STAGE=genome   ONE TASK PER EVAL CHROMOSOME (three). Loads the shared half, FREEZES it, and
#                  fits that chromosome's position tables alone — inference under Rule 2, and open
#                  to every method. This is the stage that selects on V_ and the only stage whose
#                  checkpoint may be predicted from.
#
# This is new code, so the board row is OUR TWO-STAGE VARIANT of Lavawizard and not the published
# one. The 2019 submission stays on the board unmodified as one of the 23 anchor entrants, so both
# readings are available — see README.md.
set -uo pipefail
REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t78_code}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
RUNS="${RUNS:-/project/def-maxwl/mforooz/rivals_src/lavawizard_runs}"
REGIME="${REGIME:-$REPO/configs/regime.eic_19.json}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
module load StdEnv/2023 python/3.11 >/dev/null 2>&1 || true
source "$VENV/bin/activate"
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/competitors"

[ -s "$REGIME" ] || { echo "[env] no regime at $REGIME" >&2; exit 1; }
REGIME_NAME="$(basename "$REGIME" .json)"; REGIME_NAME="${REGIME_NAME#regime.}"
RUNS="$RUNS/$REGIME_NAME"
CACHE="${CACHE:-$RUNS/eic_cache}"

# The eval chromosomes, read off the regime. Plain json — no `import candi`, because this is
# sourced by the caching array too and concurrent torch imports off /project are what the %3 array
# cap exists for.
#
# THE `regions` REFUSAL THAT USED TO LIVE HERE IS GONE, and the transferable stage is why. It
# refused a BED regime whose eval chromosomes were not in its `train_chroms`, because a
# per-chromosome fit could only be fit on the chromosome it predicts and under eic.pilot that meant
# fitting the four Pilot Regions the regime CUT — the complement of the declared scope, wearing its
# name. Nothing is fit on an eval chromosome any more except position tables, so the case the
# refusal guarded cannot arise.
_lava_chroms() {
    python - "$REGIME" <<'PYJSON'
import json, sys
print(" ".join(json.load(open(sys.argv[1]))["eval_chroms"]))
PYJSON
}
CHROMS=($(_lava_chroms)) || exit 1
NCHROM=${#CHROMS[@]}

# The cache stem the shared stage trains on — `store_eic.SHARED_STEM`, never a chromosome name,
# because under eic.pilot the axis is a packing of eighteen of them. Written out here rather than
# read from Python for the same reason as above: this file must not import candi.
SHARED_STEM=shared

mkdir -p "$RUNS" slurm-logs
