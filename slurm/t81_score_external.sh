#!/bin/bash
# t81 — score ONE §4.1 prediction root with `candi.bench.external`. Every method goes through this:
# CANDI, the four rivals, the five naive baselines and the 25 challenge entrants. One script, so a
# CANDI number and an entrant number differ in their INPUTS and in nothing else.
#
# ONE JOB = ONE (method, regime, panel, truth, scope). Scoring is the expensive half of the
# benchmark and the units are independent, so they are separate jobs rather than a loop inside one.
#
#   mkdir -p slurm-logs
#   # held-out (chr20-22), store truth, CANDI's V_ panel under eic.19
#   REGIME=/scratch/mforooz/t81_pred/CANDI/eic_19/regime.eic_19.V_.json \
#   PRED=/scratch/mforooz/t81_pred/CANDI/eic_19/V_ \
#   OUT=/project/def-maxwl/mforooz/t81_scores/CANDI/eic_19/store.V_.json \
#   SCOPE=heldout sbatch slurm/t81_score_external.sh
#
#   # genome-wide: 23 chromosomes, with chr20-22 broken out as `genome_wide`'s counterpart
#   ... SCOPE=genomewide sbatch --time=60:00:00 slurm/t81_score_external.sh
#
#   # a point-only method, so a sigma table supplies the Gaussian arm
#   ... SIGMA=/project/def-maxwl/mforooz/t81_sigma/Avocado/sigma_eic_19.json sbatch ...
#
#   # challenge truth (the Synapse bigwigs) instead of the store
#   ... TRUTH_ROOT=/project/def-maxwl/mforooz/t81_truth_challenge/B_ sbatch ...
#
# WALLTIME, MEASURED. One genome-wide pass is ~50 CPU-h on 4 cores and held-out (chr20-22) is
# 5.34 % of that, ~2.7 h. So 03:00:00 is the DEFAULT and is right for held-out only; a genome-wide
# pass needs --time=60:00:00 on the submit line, which is also what puts it in the b4 band (3 d).
# The partition is chosen by the walltime, not named here.
#
# WHY --chroms AND NOT A PER-CHROMOSOME SPLIT. `bench.external` scores a track over the
# CONCATENATION of its chromosomes — the top-1 % thresholds of mse1obs/mse1imp are taken over all of
# them at once — so a per-chromosome run is a DIFFERENT metric, not a cheaper estimate of this one.
# It is also why a prediction root missing a chromosome cannot be scored genome-wide at all.
#
# NO --gres HERE, unlike slurm/t49_baselines_score.sh and slurm/bake.sh. Those request the smallest
# MIG slice to route a CPU job through the def-maxwl_gpu account, whose fairshare is 5x the CPU
# account's. That trade is fine for a 20-minute bake and wrong for a 60-hour scoring pass: it would
# hold a GPU slice idle for two and a half days. If the CPU queue is jammed, add --gres on the
# submit line for the SHORT held-out runs only, and never any spec but the hard-rule slice.
#
# Logs resolve against the SUBMITTING cwd, not the script's location, so mkdir -p slurm-logs first.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_score
#SBATCH --output=slurm-logs/t81_score_%j.out
#SBATCH --error=slurm-logs/t81_score_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -uo pipefail

# --- edit these ---------------------------------------------------------------------------------
KIT="${KIT:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
# The PANEL-DERIVED regime, not a shipped one: it declares the pairs this root actually holds, and
# `bench.external` refuses a root that does not cover every declared pair.
REGIME="${REGIME:-}"
PRED="${PRED:-}"
OUT="${OUT:-}"
SCOPE="${SCOPE:-heldout}"                      # heldout | genomewide
SIGMA="${SIGMA:-}"                             # optional — a training-residual sigma table
TRUTH_ROOT="${TRUTH_ROOT:-}"                   # optional — the challenge truth root, else the store
HELD_OUT_CHROMS="${HELD_OUT_CHROMS:-chr20,chr21,chr22}"
VARPOOL="${VARPOOL:-}"
EXTRA="${EXTRA:-}"                             # anything else to pass through, e.g. --allow-missing
# -------------------------------------------------------------------------------------------------

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

# Absolute, because this job `cd`s into $KIT below and the checks above it do not. A relative path
# would be tested against the submitting directory and then read from a different one.
for v in REGIME PRED OUT; do
  if [ -z "${!v}" ]; then
    echo "[error] $v is required and has no sensible default. See the header for a submit line." >&2
    exit 1
  fi
  case "${!v}" in
    /*) ;;
    *) echo "[error] $v must be an absolute path, got '${!v}'. This job cds into \$KIT." >&2; exit 1;;
  esac
done
case "$SCOPE" in
  heldout|genomewide) ;;
  *) echo "[error] SCOPE must be heldout or genomewide, got '$SCOPE'" >&2; exit 1;;
esac
if [ ! -f "$REGIME" ]; then echo "[error] no regime at $REGIME" >&2; exit 1; fi
if [ ! -f "$PRED/manifest.json" ]; then
  echo "[error] $PRED carries no manifest.json, so it is not a §4.1 prediction root — or it is an" >&2
  echo "        INCOMPLETE one (every writer puts the manifest last). Refusing to score it." >&2
  exit 1
fi

if [ -n "$SIGMA" ] && [ ! -f "$SIGMA" ]; then echo "[error] no sigma table at $SIGMA" >&2; exit 1; fi

if [ ! -d "$VENV" ]; then echo "[error] no venv at $VENV" >&2; exit 1; fi
source "$VENV/bin/activate"
cd "$KIT" || { echo "[error] no checkout at $KIT" >&2; exit 1; }
source "$KIT/slurm/_kit_pin.sh"

# --- the sigma-table rule ------------------------------------------------------------------------
# A sigma fitted on the EVAL pairs is fitted on the answer sheet: it is a free look at the held-out
# truth, and a Gaussian arm calibrated that way flatters the method against every method that did
# not do it. Only a table fitted on TRAINING residuals is admissible, and the table says so in its
# own `fitted_on` string. Checked here rather than trusted, because the four legacy fit_sigma.py
# scripts still write eval-pair tables and they sit next to the good ones on disk. It runs after the
# venv is active only because `python` is the venv's; nothing else in the job has happened yet.
if [ -n "$SIGMA" ]; then
  python - "$SIGMA" <<'PYEOF' || exit 3
import json, sys
from pathlib import Path

PREFIX = "training-residuals:"
p = Path(sys.argv[1])
try:
    d = json.loads(p.read_text())
except Exception as e:
    sys.exit(f"[t81-score] REFUSING: {p} is not readable json ({e}).")
fitted = d.get("fitted_on")
if not isinstance(fitted, str) or not fitted.startswith(PREFIX):
    sys.exit(f"[t81-score] REFUSING sigma table {p}: fitted_on = {fitted!r}, which does not start "
             f"with {PREFIX!r}. A sigma fitted on the EVAL pairs is fitted on the answer sheet — it "
             f"is a free look at held-out truth and it flatters this method against every method "
             f"that did not take one. Refit on training residuals "
             f"(python -m competitors.sigma_pass) and rerun.")
print(f"[t81-score] sigma OK: {p.name} fitted_on {fitted!r}")
PYEOF
fi

mkdir -p "$(dirname "$OUT")"

# --- what gets scored, and where ------------------------------------------------------------------
# held-out is chr20-22 and nothing else. genome-wide is EVERY chromosome the store carries, with
# chr20-22 named again through --held-out-chroms so the result carries both aggregations: `macro`
# and `panels` stay held-out, and `genome_wide.macro` / `genome_wide.panels` are the 23-chromosome
# roll-up. One pass, two numbers — scoring twice would double a 50 CPU-h bill for the same reads.
ARGS=(--store "$REGIME" --pred "$PRED" --out "$OUT")
if [ "$SCOPE" = "heldout" ]; then
  ARGS+=(--chroms "$HELD_OUT_CHROMS")
else
  ALL_CHROMS="$(python - "$REGIME" <<'PYEOF'
import json, sys
from pathlib import Path
from candi.store import layout as L
d = json.loads(Path(sys.argv[1]).read_text())
# The genome layer is SHARED and lives one level above the corpus store (CANDI_STORE/genome, not
# CANDI_STORE/eic/genome) — layout.corpus_genome_dir is the helper for that; chrom_sizes_path is not.
sizes = L.load_chrom_sizes(L.corpus_genome_dir(d["store"]) / "chrom_sizes.json")
print(",".join(L.sort_chroms(sizes)))
PYEOF
)" || { echo "[error] could not read the store's chromosome list from $REGIME" >&2; exit 1; }
  # --held-out-chroms is candi.bench.external's genome-wide flag. On a checkout that predates it the
  # parser rejects the flag outright, which is the failure we want: a genome-wide pass that silently
  # dropped it would write a result labelled genome-wide whose `macro` block was genome-wide too,
  # and nothing in the json would say the held-out aggregation was missing.
  ARGS+=(--chroms "$ALL_CHROMS" --held-out-chroms "$HELD_OUT_CHROMS")
fi
if [ -n "$SIGMA" ]; then ARGS+=(--sigma-table "$SIGMA"); fi
if [ -n "$TRUTH_ROOT" ]; then ARGS+=(--truth-root "$TRUTH_ROOT"); fi
if [ -n "$VARPOOL" ]; then ARGS+=(--varpool "$VARPOOL"); fi

echo "[t81-score] host=$(hostname) commit=$(git rev-parse --short HEAD)"
echo "[t81-score] scope=$SCOPE regime=$REGIME"
echo "[t81-score] pred=$PRED"
echo "[t81-score] out=$OUT"
echo "[t81-score] truth=$([ -n "$TRUTH_ROOT" ] && echo "challenge ($TRUTH_ROOT)" || echo "store")"
echo "[t81-score] sigma=${SIGMA:-none}"
# shellcheck disable=SC2086
echo "[t81-score] python -m candi.bench.external ${ARGS[*]} $EXTRA"

# shellcheck disable=SC2086
python -m candi.bench.external "${ARGS[@]}" $EXTRA
rc=$?

if [ $rc -eq 0 ] && [ ! -f "$OUT" ]; then
  echo "[t81-score] WARNING: exit 0 but $OUT was not written." >&2
  rc=5
fi

echo "[t81-score] DONE scope=$SCOPE rc=$rc out=$OUT"
exit $rc
