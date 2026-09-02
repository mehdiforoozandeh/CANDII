#!/bin/bash
# eDICE retrained on our EIC (RIVALS_PLAN §7.3), after the Roadmap gate is recorded.
#
# DECIDED (PI, 2026-08-25): N_TARGETS=31 — the paper's 11.6% RATE, not its absolute 120.
# eDICE masked 120 of Roadmap's 1032 tracks per bin; our panel holds 267, so the absolute count
# would mask 45% and make our variant structurally harder than the published one. The flag stays
# REQUIRED even though the value is settled, so the number appears on the launch line and in this
# job's log — which is where the record of what was actually run lives.
#
#   mkdir -p slurm-logs
#   N_TARGETS=31 REGIME=$REPO/configs/regime.eic_19.json    sbatch competitors/edice/slurm/eic_train.sh
#   N_TARGETS=31 REGIME=$REPO/configs/regime.eic_pilot.json sbatch competitors/edice/slurm/eic_train.sh
#
# A CHECKPOINT IS NOW SELECTED ON V_, AND SELECTION COSTS MORE THAN TRAINING. `run_eic.py train`
# derives a V_-only regime, and every EVAL_EVERY epochs it writes that panel's predictions and
# scores them with `candi.bench.external` -- the same instrument that produces a board row, which
# is the whole of what §5 asks for. It is not free:
#
#   one V_ check = predict 45 tracks x 6,478,903 bins (chr20+21+22) + score them
#                = ~38 min GPU  +  ~73 min CPU  =  ~1.9 h
#
# Both halves are DERIVED FROM A MEASUREMENT, not guessed, and both are re-measurable:
#   * GPU: the Roadmap gate measured 0.255 s/batch(256) at 1032 supports / 120 targets on this MIG
#     slice. Our prediction shape is 267 supports / 45 targets, which benchmarks at 0.358x that
#     shape's forward cost, so 0.091 s/batch = 2,803 bins/s. Measure the real one off the
#     `[edice]   chrN: ... s` lines the predict pass prints and correct this.
#   * CPU: `score_external` measured at ~15 us per (track x bin) locally, so 45 x 6.48 M track-bins
#     is ~73 min. It lands within 20 % of CANDI's MEASURED 91 min for the same panel and scope
#     (§12.2, 2026-08-31), which is the check that the extrapolation is not nonsense.
#
# Training, by the same 0.255 s/batch anchor at the 236-support / 31-target training shape
# (0.272x): 3,691 bins/s, so 10.6 min per chr19 epoch and 4.6 min per pilot epoch.
#
#   regime      50 epochs of training   17 checks at EVAL_EVERY=3   total   requested
#   eic.19               8.8 h                    31.8 h            40.6 h   60:00:00
#   eic.pilot            3.8 h                    31.8 h            35.6 h   60:00:00
#
# 60 h is 40.6 h x 1.48. THE EVAL IS 78 % OF THE eic.19 RUN AND 89 % OF THE PILOT ONE -- raising
# EVAL_EVERY is the only lever that moves the band much (EVAL_EVERY=5 gives 11 checks and 29 h /
# 24 h), and it costs resolution on EARLY_STOP, which is counted in epochs. The default stays 3
# because that is CANDI's cadence and §5's "same rule" is worth more than the hours.
# EARLY_STOP usually ends the run far short of the band; model.best.pt is written the moment the
# metric improves, so a walltime kill still leaves a validly selected checkpoint.
#
# --gres is the AGENTS.md hard-rule MIG slice and is never any other spec.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t52_edice_eic
#SBATCH --output=slurm-logs/%x_%j.out
#SBATCH --error=slurm-logs/%x_%j.err
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

set -uo pipefail

if [ -z "${N_TARGETS:-}" ]; then
  echo "[error] N_TARGETS is required and has no default." >&2
  echo "        31  = eDICE's Roadmap masking RATE (11.6% of our 267-track panel)  <-- DECIDED," >&2
  echo "              PI 2026-08-25. This is the run to launch." >&2
  echo "        120 = eDICE's Roadmap ABSOLUTE count (masks 45% of our panel)" >&2
  echo "        See competitors/edice/README.md, 'Masking rate on our EIC'." >&2
  exit 2
fi

REPO="${REPO:-/project/def-maxwl/mforooz/CANDII_t78_code}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
# RETARGETED 2026-08-31: was configs/regime.eic_val.json (train chr19, eval chr21, 26 V_ pairs).
# The two live regimes are eic_19 and eic_pilot (BENCHMARK_DESIGN.md §3); run this once per regime.
REGIME="${REGIME:-$REPO/configs/regime.eic_19.json}"
REGIME_NAME="$(basename "$REGIME" .json)"; REGIME_NAME="${REGIME_NAME#regime.}"
OUT="${OUT:-/project/def-maxwl/mforooz/rivals_src/edice_runs/${REGIME_NAME}_nt${N_TARGETS}}"
EPOCHS="${EPOCHS:-50}"
# §5's uniform rule. EVAL_EVERY is CANDI's cadence; see the cost table in the header before
# changing it, and note EARLY_STOP's resolution is EVAL_EVERY, so a patience below it never fires.
EVAL_EVERY="${EVAL_EVERY:-3}"
EARLY_STOP="${EARLY_STOP:-3}"
# The macro pval key selection reads off `candi.bench.external`. NOT `crps`: eDICE emits a point
# and no sigma, so the instrument records a point-only track and never computes a distributional
# key. CANDI selects on count-arm crps, which eDICE has no head for -- so the panel, the
# instrument and the cadence are uniform and the KEY is not. That gap is on record for the PI.
SELECT_METRIC="${SELECT_METRIC:-mse}"

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
source "$VENV/bin/activate"

cd "$REPO/competitors/edice"
export PYTHONPATH="$PWD:$REPO/src"
mkdir -p "$OUT"

# D32 IS HONOURED NOW. This block used to REFUSE a `regions` regime, because reading train_chroms
# raw would have (a) broken Rule 2 -- fitting on 18 WHOLE chromosomes instead of the 25,588,197 bp
# of Pilot Regions the regime declares -- and (b) run out of host RAM doing it (~115 GB). Both are
# fixed: `eic_panel.train_bin_spans` restricts the read to the bins lying WHOLLY inside a region,
# and the read walks region by region so a slab is contiguous. The scope is printed, and it is the
# regime's own declared figure, so a wrong run is visible on the first line of the log.
python - "$REGIME" <<'PYEOF' || exit 3
import json, sys
from pathlib import Path
src = Path(sys.argv[1])
d = json.loads(src.read_text())
if not d.get("regions"):
    print(f"[edice] {src.name} declares no `regions`; training on whole chromosomes {d['train_chroms']}")
    raise SystemExit(0)
from eic_panel import train_bin_spans   # cwd is competitors/edice and is on PYTHONPATH
tot = sum(b - a for c in d["train_chroms"] for a, b in
          train_bin_spans(d, src, c, 10**9, 25))
print(f"[edice] regions {d['regions']['bed']} (D32, policy={d['regions'].get('policy','contain')}): "
      f"{tot:,} contained 25 bp bins over {len(d['train_chroms'])} train chromosomes")
PYEOF

echo "[edice] EIC train  n_targets=$N_TARGETS  epochs=$EPOCHS  regime=$REGIME  out=$OUT"
echo "[edice] select_on=V_  metric=$SELECT_METRIC  eval_every=$EVAL_EVERY  early_stop=$EARLY_STOP"
echo "[edice] host=$(hostname)"; nvidia-smi -L || true

# --train-chroms defaults to the regime's train_chroms, and --eval-chroms to its eval_chroms --
# the same V_ scope CANDI selects on. eDICE carries no positional parameters, so the training
# slice is a budget choice and not a leakage one -- see the README's "The one open decision".
python run_eic.py train \
  --regime "$REGIME" --out "$OUT" \
  --n-targets "$N_TARGETS" --epochs "$EPOCHS" \
  --eval-every "$EVAL_EVERY" --early-stop-epochs "$EARLY_STOP" \
  --select-metric "$SELECT_METRIC"
rc=$?

# §5 is only satisfied if a checkpoint was actually SELECTED on V_. Without model.selected.pt the
# run yields a last-epoch checkpoint, which is a different object than the design asks for. Say so
# loudly rather than let it pass as success -- the same gate slurm/t81_train_candi.sh applies.
if [ "$EVAL_EVERY" != "0" ] && [ $rc -eq 0 ]; then
  if [ -f "$OUT/model.selected.pt" ]; then
    echo "[edice] OK: model.selected.pt exists -- a checkpoint was selected on V_ $SELECT_METRIC."
    echo "[edice]   SCORE THAT FILE, not model.pt. eic_score.sh's MODEL= must name it."
  else
    echo "[edice] WARNING: eval_every=$EVAL_EVERY but NO model.selected.pt was written. This run" >&2
    echo "[edice]   did NOT select on V_, so it does not satisfy BENCHMARK_DESIGN §5. Do not use." >&2
    rc=90
  fi
fi

echo "[edice] DONE eic-train n_targets=$N_TARGETS rc=$rc"
exit $rc
