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
# READ THIS BEFORE LAUNCHING: THE REGIME AXIS IS EMPTY FOR LAVAWIZARD. CONFIRMED 2026-09-01.
# `train.py` fits ONE INDEPENDENT Guacamole PER CHROMOSOME — cell factors, assay factors, the dense
# network and the three genome-factor tables, all from a fresh init, all on that chromosome's own
# bins. `store_eic.predict_chrom` then indexes the genome tables by that chromosome's own bin
# numbers and refuses a checkpoint from a different index space, and `predict.sh` passes
# `guacamole_$C.pt` for C in eval_chroms. So the model that predicts chr20 was FIT ON chr20.
#
# The cell and assay factors are TRANSFERABLE parameters by §2's own definition, and under this
# scheme they are fit on chr20/21/22 — the eval chromosomes. §2 exempts per-position adaptation
# (Avocado's genomic factors, ChromImpute's neighbour features); it does not exempt a cell
# embedding. Two consequences, both verified against the code rather than argued:
#
#   1. `train_chroms` and `regions` are read by NO lavawizard module (grep both, across the package
#      — the only hits are this file's own refusal). The two live regimes differ ONLY in those two
#      keys plus `_comment`; every key this method reads — store, assays, biosamples.train,
#      eval_pairs, eval_chroms — is identical. So `eic.19` and `eic.pilot` would be the same run,
#      same seed, same cache, and two identical rows under two regime labels.
#   2. This is not "no transferable parameters". It is transferable parameters fit at the wrong
#      loci, which breaks Rule 2 in BOTH regimes, not only in the ablation.
#
# Collapsing the axis to one row is a PI decision, not a launcher's. So is the fix — a joint fit on
# train_chroms plus per-chromosome genome factors, which is Avocado's scheme and not upstream
# Guacamole's. Raise it; one ruling settles the collapse and the pilot refusal below together.
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
# D32 IS IMPLEMENTED — `store_eic.contained_bins` resolves the BED through
# `candi.store.regime.RegionSet` and writes the contained bins into the cache, and the sampler walks
# them (`test_under_a_regions_regime_every_training_locus_lies_inside_the_bed`). What is refused
# below is narrower and is the header's finding, not a missing feature: a `regions` regime whose
# eval chromosomes are NOT in its `train_chroms`. Lavawizard can only fit a chromosome's model on
# that chromosome, and under eic.pilot the regions on chr20/21/22 are precisely the four the regime
# CUT (§3.1). Training there would fit the COMPLEMENT of the declared scope and call it eic.pilot.
# `store_eic` refuses the same case in Python; this is the same refusal before 3 array tasks queue.
_lava_chroms() {
    python - "$REGIME" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
ev = list(d["eval_chroms"])
if d.get("regions"):
    outside = [c for c in ev if c not in set(d.get("train_chroms") or [])]
    if outside:
        sys.exit("SCOPE " + ",".join(outside))
print(" ".join(ev))
PYEOF
}
_LAVA_CH="$(_lava_chroms)" || {
    echo "[env] $REGIME declares a \`regions\` BED (the eic.pilot Pilot-Region scope, D32) and" >&2
    echo "[env] its train_chroms do NOT contain the eval chromosomes this method must fit on." >&2
    echo "[env] The BED restriction itself is built and tested; the blocker is that Lavawizard's" >&2
    echo "[env] transferable parameters are fit on the eval chromosomes at all — see the header." >&2
    echo "[env] Raise it. One ruling settles this and the regime collapse together." >&2
    exit 1
}
CHROMS=($_LAVA_CH)
unset _LAVA_CH
NCHROM=${#CHROMS[@]}

mkdir -p "$RUNS" slurm-logs
