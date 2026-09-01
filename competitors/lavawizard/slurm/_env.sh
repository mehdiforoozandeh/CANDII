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
# READ THIS BEFORE LAUNCHING: WHAT "REGIME" MEANS FOR LAVAWIZARD IS AN OPEN QUESTION.
# `train.py` fits ONE INDEPENDENT Guacamole PER CHROMOSOME — cell factors, assay factors, the dense
# network and the three genome-factor tables, all from a fresh init, all on that chromosome's own
# bins. The cell and assay factors are TRANSFERABLE parameters by §2's own definition, and under
# this scheme they are fit ON chr20/21/22 — the eval chromosomes. §2 exempts per-position adaptation
# (Avocado's genomic factors, ChromImpute's neighbour features); it does not exempt a cell
# embedding. So as the code stands, Lavawizard's two "regime" runs would differ in NOTHING —
# `train_chroms` never reaches this method — and the row's regime label would be a claim the run
# does not support. Splitting it into a joint fit on train_chroms plus per-chromosome genome factors
# is Avocado's scheme, not upstream Guacamole's, and is a PI decision plus new code. Raise it.
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

# The eval chromosomes, read off the regime. Plain json — no `import candi`, because this is sourced
# by the caching array too and concurrent torch imports off /project are what the %3 array cap
# exists for.
#
# A `regions` regime (eic.pilot) is refused: `store_eic.build_cache_from_store` caches a whole
# chromosome and `train.py` trains on all of it, so neither can express a BED-restricted scope.
_lava_chroms() {
    python - "$REGIME" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("regions"):
    sys.exit("REGIONS")
print(" ".join(d["eval_chroms"]))
PYEOF
}
_LAVA_CH="$(_lava_chroms)" || {
    echo "[env] $REGIME declares a \`regions\` BED (the eic.pilot Pilot-Region scope, D32)." >&2
    echo "[env] store_eic.py caches whole chromosomes and train.py trains on all of them, so" >&2
    echo "[env] Lavawizard CANNOT express this regime's training scope. Raise it." >&2
    exit 1
}
CHROMS=($_LAVA_CH)
unset _LAVA_CH
NCHROM=${#CHROMS[@]}

mkdir -p "$RUNS" slurm-logs
