#!/bin/bash
# t81 — dump CANDI's predictions for ONE regime and ONE panel to a §4.1 prediction root, so
# `candi.bench.external` can score them the same way it scores every rival.
#
# ONE JOB = ONE (regime, panel). Not an array: the four units differ in walltime by 2x, in output
# ROOT (V_ goes to /scratch, B_ goes to /project because it is written once and must survive the
# purge), and B_ carries a guard the V_ runs must not inherit. An array index that means four
# different things is how a B_ root gets overwritten by a rerun of the V_ task.
#
#   mkdir -p slurm-logs
#   REGIME_NAME=eic_19    PANEL=V_ sbatch slurm/t81_predict_candi.sh
#   REGIME_NAME=eic_pilot PANEL=V_ sbatch slurm/t81_predict_candi.sh
#   REGIME_NAME=eic_19    PANEL=B_ B_ONCE=1 sbatch --time=06:00:00 slurm/t81_predict_candi.sh
#   REGIME_NAME=eic_pilot PANEL=B_ B_ONCE=1 sbatch --time=06:00:00 slurm/t81_predict_candi.sh
#
# B_ IS WRITTEN ONCE, EVER (BENCHMARK_DESIGN.md §5). The test panel is not a dial to turn while
# looking at the answer, so this refuses a B_ run twice over: once unless B_ONCE=1 is set on the
# submit line, and again if the root already carries a manifest.json. Both exit 4. There is no
# --force: the way to redo a B_ prediction is for the PI to move the old root aside, by hand, and
# say why. A V_ run has neither guard and may be rerun freely.
#
# WALLTIME. Measured: CANDI inference is 0.2363 GPU-h per GENOME-WIDE track on this MIG slice, at
# MaxRSS 4.2 GiB (hence --mem=8G, which is 2x headroom, not a guess). A TRACK is not a PAIR — it is
# (declared pair x an assay the target cell holds and the input cell does not) — and the V_ panel is
# 45 tracks out of 26 pairs, a 1.7x difference. Sizing off pairs is how a 14 h job gets an 8 h
# request. The 14:00:00 default below is 45 x 0.2363 x 1.3; the job prints the figure for the panel
# it actually derived, BEFORE the dump starts, so the next submit line can be read off this log.
#
# WHY GENOME-WIDE. --chroms is every chromosome the STORE carries, not the regime's eval_chroms,
# because §4's genome-wide scoring pass aggregates over all 23 and `bench.external` scores a track
# over the CONCATENATION of its chromosomes — a root missing one cannot be scored genome-wide at
# all, and re-dumping later costs the same GPU hours again.
#
# Logs resolve against the SUBMITTING cwd, not the script's location, so mkdir -p slurm-logs first.
#SBATCH --account=def-maxwl
#SBATCH --job-name=t81_pred_candi
#SBATCH --output=slurm-logs/t81_pred_%j.out
#SBATCH --error=slurm-logs/t81_pred_%j.err
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -uo pipefail

# --- edit these ---------------------------------------------------------------------------------
KIT="${KIT:-/project/def-maxwl/mforooz/CANDII_main}"
VENV="${VENV:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv}"
REGIME_NAME="${REGIME_NAME:-eic_19}"           # eic_19 | eic_pilot
PANEL="${PANEL:-V_}"                           # V_ | B_
B_ONCE="${B_ONCE:-0}"                          # must be 1 for PANEL=B_, on the submit line
SEED="${SEED:-0}"
CKPT_DIR="${CKPT_DIR:-/project/def-maxwl/mforooz/t81_checkpoints}"
# The SELECTED checkpoint — the .best.ckpt the training run wrote on V_imp_crps, never the
# last-epoch one. `bench.dump` loads it with strict=True, so a mismatched architecture is a loud
# failure rather than a silently different model.
CKPT="${CKPT:-$CKPT_DIR/t81_${REGIME_NAME}_s${SEED}.best.ckpt}"
# The run json whose `config.arch` rebuilds the EXACT model that wrote the checkpoint. Left empty
# it resolves below: the run's own json first, then the reconstructed `.arch.json` (which the pilot
# needs, because its run json was never written — see slurm/t81_train_candi.sh).
ARCH_FROM="${ARCH_FROM:-}"
# Where the derived panel regime is written. Empty resolves to the prediction root's PARENT, so the
# regime that produced a root sits beside it — which for B_ means /project, not scratch. The score
# pass takes this file as its --store, so a B_ regime left on scratch would be purged out from
# under the one panel that can never be regenerated.
WORKSPACE="${WORKSPACE:-}"
VERSION="${VERSION:-$(date +%F)}"
CHROMS="${CHROMS:-}"                           # empty = every chromosome the store carries
BATCH_WINDOWS="${BATCH_WINDOWS:-4}"
# -------------------------------------------------------------------------------------------------

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export WANDB_MODE=disabled

case "$PANEL" in
  V_|B_) ;;
  *) echo "[error] PANEL must be V_ or B_ (with the underscore), got '$PANEL'" >&2; exit 1;;
esac
case "$REGIME_NAME" in
  eic_19|eic_pilot) ;;
  *) echo "[error] REGIME_NAME must be eic_19 or eic_pilot, got '$REGIME_NAME'" >&2; exit 1;;
esac
REGIME="configs/regime.${REGIME_NAME}.json"

# THE PINNED ROOTS (BENCHMARK_DESIGN.md §4.1). V_ lives on scratch and may be regenerated; B_ lives
# on /project because it is written once and scratch is purged 60 days after the oldest file.
if [ "$PANEL" = "V_" ]; then
  OUT="${OUT:-/scratch/mforooz/t81_pred/CANDI/${REGIME_NAME}/V_}"
else
  OUT="${OUT:-/project/def-maxwl/mforooz/t81_pred_B/CANDI/${REGIME_NAME}/B_}"
fi
if [ -z "$WORKSPACE" ]; then WORKSPACE="$(dirname "$OUT")"; fi

# --- the once-only B_ guard, before anything is allocated or derived ------------------------------
if [ "$PANEL" = "B_" ]; then
  if [ "$B_ONCE" != "1" ]; then
    echo "[error] PANEL=B_ needs B_ONCE=1 on the SUBMIT LINE. §5 touches the test panel once, from" >&2
    echo "        the selected checkpoint. Typing it out is the point: it is a decision, not a" >&2
    echo "        default. Re-submit with:" >&2
    echo "        REGIME_NAME=$REGIME_NAME PANEL=B_ B_ONCE=1 sbatch slurm/t81_predict_candi.sh" >&2
    exit 4
  fi
  # THE MANIFEST, not the directory, is what says B_ was already spent: `bench.dump` writes it LAST,
  # so a root with track dirs and no manifest is a dump that was killed part-way. Keying on the
  # directory would make that case unrecoverable without the PI; keying on the manifest lets a
  # killed dump be rerun and still refuses a completed one.
  if [ -f "$OUT/manifest.json" ]; then
    echo "[error] $OUT/manifest.json already exists. B_ was already predicted for" >&2
    echo "        $REGIME_NAME, and §5 allows exactly one. There is no --force here: move the old" >&2
    echo "        root aside by hand, with the PI, and record why. REFUSING." >&2
    exit 4
  fi
  echo "[t81-pred] B_ONCE=1 and $OUT carries no manifest — this is the once-only test-panel run."
fi

if [ ! -d "$VENV" ]; then echo "[error] no venv at $VENV" >&2; exit 1; fi
source "$VENV/bin/activate"
cd "$KIT" || { echo "[error] no checkout at $KIT" >&2; exit 1; }
source "$KIT/slurm/_kit_pin.sh"

mkdir -p "$WORKSPACE" "$(dirname "$OUT")"

# --arch-from: the run's own json, else the reconstructed arch json. Refused rather than guessed —
# without it `bench.dump` falls back to the parser's DEFAULT architecture flags and `strict=True`
# would either fail confusingly or, worse, load a model that happens to have the same shapes.
if [ -z "$ARCH_FROM" ]; then
  for cand in "$CKPT_DIR/t81_${REGIME_NAME}_s${SEED}.json" \
              "$CKPT_DIR/t81_${REGIME_NAME}_s${SEED}.arch.json"; do
    if [ -f "$cand" ]; then ARCH_FROM="$cand"; break; fi
  done
fi
if [ -z "$ARCH_FROM" ] || [ ! -f "$ARCH_FROM" ]; then
  echo "[error] no --arch-from json. Looked for t81_${REGIME_NAME}_s${SEED}.json and" >&2
  echo "        t81_${REGIME_NAME}_s${SEED}.arch.json under $CKPT_DIR. The pilot's run json was" >&2
  echo "        never written (its job hit the walltime during the final check), so its" >&2
  echo "        architecture has to be reconstructed and proved with a strict load first." >&2
  exit 1
fi
if [ ! -f "$CKPT" ]; then echo "[error] no checkpoint at $CKPT" >&2; exit 1; fi

echo "[t81-pred] host=$(hostname) commit=$(git rev-parse --short HEAD)"
echo "[t81-pred] regime=$REGIME_NAME panel=$PANEL seed=$SEED version=$VERSION"
echo "[t81-pred] ckpt=$CKPT"
echo "[t81-pred] arch-from=$ARCH_FROM"
echo "[t81-pred] out=$OUT"
nvidia-smi -L || true

# --- derive the panel regime ---------------------------------------------------------------------
# The shipped regimes declare all 38 pairs, 26 V_ and 12 B_. A dump run against one of them would
# write BOTH panels into one root, which spends the B_ budget without anyone asking for it. The
# prefix is an explicit argument to `split`, never a default, for exactly that reason.
DERIVED="$WORKSPACE/regime.${REGIME_NAME}.${PANEL}.json"
python tools/declare_eval_pairs.py split --regime "$REGIME" --panel "$PANEL" --out "$DERIVED" \
  || { echo "[error] could not derive the $PANEL regime from $REGIME" >&2; exit 1; }

# What the derived regime actually says, read back from the FILE rather than assumed from the flag,
# plus the two numbers the dump needs: the chromosome list and the declared TRACK count. The track
# count is computed with the same public API `bench.dump` uses to enumerate what it must write
# (source.pairs / source.targets), so the walltime figure below cannot drift from the real work.
PLAN="$(python - "$DERIVED" "$PANEL" "$CHROMS" <<'PYEOF'
import json, sys
from pathlib import Path

from candi.bench.harness import open_source
from candi.store import layout as L
from candi.store.reader import CorpusStore

derived, panel, chroms_arg = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
d = json.loads(derived.read_text())
# Both spellings, the same way slurm/t81_train_candi.sh reads them: a regime may write a pair as a
# two-element list or as {"input", "target"}, and this refusal must not become a crash on the form.
pairs = [(str(p["input"]), str(p["target"])) if isinstance(p, dict) else (str(p[0]), str(p[1]))
         for p in d["eval_pairs"]]
bad = sorted({t for _, t in pairs if not t.startswith(panel)})
if bad:
    sys.exit(f"[t81-pred] REFUSING: the derived regime {derived.name} targets {bad}, which are not "
             f"{panel}. Predicting them would write the wrong panel into this root.")

if chroms_arg:
    chroms = [c.strip() for c in chroms_arg.split(",") if c.strip()]
else:
    # The genome layer is SHARED and lives one level ABOVE the corpus store: CANDI_STORE/genome,
    # not CANDI_STORE/eic/genome. `layout.corpus_genome_dir` is the helper that knows that;
    # `layout.chrom_sizes_path` takes CANDI_STORE itself and so pointed at a directory that has
    # never existed. Every submission leaving CHROMS unset died right here with exit 1, at PLAN
    # time — cheap to see once you read the log, and invisible until then (observed 2026-09-03).
    sizes = L.load_chrom_sizes(L.corpus_genome_dir(d["store"]) / "chrom_sizes.json")
    # THE SHARED LAYER CARRIES MORE THAN THE CORPUS DOES, so its list is not the default on its
    # own: CANDI_STORE/genome declares 24 (chrY included) and the eic corpus holds 23 (chr1-22,
    # chrX). A dump cannot write a track for a chromosome the store has no data for, and both
    # readers this job drives have to hold it — the corpus tracks AND the shared DNA/mask layer —
    # so the default is the INTERSECTION. `CorpusStore.n_bins()` is the record a corpus keeps of
    # what it holds (manifest `genome.n_bins`, else the h5 attrs of its first biosample), and it
    # is the rule the two rival §4.1 writers already derive their chromosome list from —
    # `competitors/edice/run_eic.py --chroms all` and the `Panel` of `competitors/baselines`, both
    # of them off the same n_bins map — so these roots come out over the same 23 those do.
    #
    # NOTE ON STYLE, because it costs an hour to rediscover: this heredoc sits inside a "$( )",
    # and bash pairs up quote characters in the BODY even though the delimiter is quoted. An odd
    # apostrophe anywhere below turns the next "(" into `syntax error near unexpected token`, at
    # PARSE time, for the whole script. Prose here therefore carries no apostrophe at all, and
    # the message below reads `store` out of a variable rather than subscripting with quotes.
    store = d["store"]
    have = CorpusStore(store).n_bins()
    ranked = L.sort_chroms(sizes)
    chroms = [c for c in ranked if c in have]
    dropped = [c for c in ranked if c not in have]
    if not chroms:
        sys.exit(f"[t81-pred] REFUSING: the shared genome layer at "
                 f"{L.corpus_genome_dir(store)} and the corpus at {store} have no chromosome in "
                 f"common, so this dump would write nothing at all.")
    if dropped:
        print(f"[t81-pred]   the shared genome layer also declares {dropped}, which this corpus "
              f"holds no data for — dropped from the default", file=sys.stderr)

src = open_source(store=str(derived), chroms=chroms)
try:
    tracks = sum(len(src.targets(p, "impute")) for p in src.pairs("impute"))
finally:
    src.close()

hours = 0.2363 * tracks * 1.3
print(",".join(chroms))
print(tracks)
print(f"{int(hours):02d}:{int(hours % 1 * 60):02d}:00")
print(f"[t81-pred] derived {derived.name}: {len(pairs)} {panel} pairs -> {tracks} declared tracks",
      file=sys.stderr)
print(f"[t81-pred]   chroms ({len(chroms)}): {','.join(chroms)}", file=sys.stderr)
print(f"[t81-pred]   WALLTIME for this panel = 0.2363 GPU-h/track x {tracks} x 1.3 = {hours:.2f} h.",
      file=sys.stderr)
print(f"[t81-pred]   If this job was submitted with less, it will be killed mid-dump and the root",
      file=sys.stderr)
print(f"[t81-pred]   will be INCOMPLETE (bench.dump writes the manifest last, so an incomplete root",
      file=sys.stderr)
print(f"[t81-pred]   carries none — which is what the B_ guard above keys on). Resubmit with",
      file=sys.stderr)
print(f"[t81-pred]   --time above that figure.", file=sys.stderr)
PYEOF
)" || exit 1
CHROMS="$(echo "$PLAN" | sed -n 1p)"
N_TRACKS="$(echo "$PLAN" | sed -n 2p)"
WANT_TIME="$(echo "$PLAN" | sed -n 3p)"
echo "[t81-pred] tracks=$N_TRACKS recommended --time=$WANT_TIME"

# --- the dump ------------------------------------------------------------------------------------
python -m candi.bench.dump \
  --store "$DERIVED" \
  --ckpt "$CKPT" \
  --arch-from "$ARCH_FROM" \
  --out "$OUT" \
  --method CANDI \
  --version "$VERSION" \
  --chroms "$CHROMS" \
  --batch-windows "$BATCH_WINDOWS"
rc=$?

if [ $rc -eq 0 ] && [ ! -f "$OUT/manifest.json" ]; then
  echo "[t81-pred] WARNING: dump returned 0 but $OUT/manifest.json is absent. bench.dump writes" >&2
  echo "[t81-pred]   the manifest LAST, so a root without one is incomplete and cannot be scored." >&2
  rc=5
fi

# The derived regime is named again here because it is the score pass's --store, and reading it off
# this line beats re-deriving it and hoping the two agree.
echo "[t81-pred] DONE regime=$REGIME_NAME panel=$PANEL rc=$rc out=$OUT derived=$DERIVED"
exit $rc
