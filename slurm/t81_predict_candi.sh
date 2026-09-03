#!/bin/bash
# t81 — dump CANDI's predictions for ONE regime and ONE panel to a §4.1 prediction root, so
# `candi.bench.external` can score them the same way it scores every rival.
#
# ONE JOB = ONE (regime, panel). Not an array: the four units differ in walltime by 2x, in output
# ROOT (V_ goes to /scratch, B_ goes to /project because it is written once and must survive the
# purge), and B_ carries a guard the V_ runs must not inherit. An array index that means four
# different things is how a B_ root gets overwritten by a rerun of the V_ task. (SHARD_CHROMS=1
# below IS an array, and does not break that rule: its index means ONE thing, a chromosome of the
# one unit the submit line already fixed, and every task writes into that unit's own root.)
#
#   mkdir -p slurm-logs
#   REGIME_NAME=eic_19    PANEL=V_ sbatch slurm/t81_predict_candi.sh
#   REGIME_NAME=eic_pilot PANEL=V_ sbatch slurm/t81_predict_candi.sh
#   REGIME_NAME=eic_19    PANEL=B_ B_ONCE=1 sbatch --time=06:00:00 slurm/t81_predict_candi.sh
#   REGIME_NAME=eic_pilot PANEL=B_ B_ONCE=1 sbatch --time=06:00:00 slurm/t81_predict_candi.sh
#
# THE CPU ROUTE (DEVICE + SHARD_CHROMS), and why it is sharded by chromosome.
#
# Every GPU predict job sat PENDING on def-maxwl_gpu with no start estimate while the CPU
# partitions started in minutes. Nothing here needs a GPU to be correct: `bench.dump` already takes
# a real --device, evaluation runs under `no_autocast` so no bf16 path is exercised, and nothing in
# `src/candi` calls `.cuda()`. What a CPU cannot do is a whole 45-track genome-wide pass in one
# job: the measured 10.6 GPU-h becomes an estimated 71-143 CPU-h, over the 60 h cpubase_bycore_b4
# band. So the pass is SHARDED BY CHROMOSOME -- 23 array tasks, one chromosome each, into the SAME
# root -- and the shards are assembled into one manifest afterwards.
#
#   export DEVICE=cpu SHARD_CHROMS=1 REGIME_NAME=eic_19 PANEL=V_
#   AID=$(sbatch --parsable --export=ALL --array=0-22%12 --gres=none --cpus-per-task=8 \
#         --mem=16G --time=12:00:00 slurm/t81_predict_candi.sh)
#   export SHARD_MERGE=1 SHARD_ARRAY=$AID
#   sbatch --export=ALL --dependency=afterok:$AID --gres=none --cpus-per-task=1 --mem=8G \
#         --time=00:20:00 slurm/t81_predict_candi.sh
#
# EXPORT, THEN `--export=ALL`. A comma-valued variable inside `--export=ALL,VAR=a,b,c` arrives
# truncated at the first comma, and CHROMS is comma-valued. `%12` is the cap the shared /project
# venv needs: more than 12 simultaneous torch imports off it fail with partial-module ImportErrors.
#
# THE SHARD INDEX NAMES A CHROMOSOME BY POSITION in the list the PLAN block below derives -- the
# same list, in the same order, an unsharded run would have dumped. The array must therefore be
# exactly as long as that list (23 for the eic corpus); a mismatch exits 6 rather than silently
# leaving a chromosome unpredicted or running the same one twice.
#
# THE MANIFEST IS THE WHOLE REASON FOR THE SECOND JOB. `bench.dump` builds `manifest.json` from the
# chromosomes THIS invocation was given, so 23 shards racing on one root would leave a manifest
# naming one chromosome -- and `slurm/t81_score_external.sh` reads `manifest.chroms` and scores
# exactly those. So in sharded mode each shard RENAMES the manifest it wrote to
# `$OUT/.shard_manifest.<chrom>.json` and the root carries none until every shard is done. Then the
# SHARD_MERGE=1 step takes any one of those as its template, VERIFIES that every declared track
# holds an npz for every planned chromosome, and writes the single manifest naming all 23. It adds
# `device=` and `sharded: true` with the array id to `notes`; no key the scorer reads changes.
#
# A FILE, NOT A DIRECTORY, and that is not cosmetic: `bench.external` lists every DIRECTORY under a
# prediction root and refuses the whole pass if one names no declared track (external.py, the
# `track directory(ies) name no declared pair` refusal). A `.shards/` holding pen would have cost
# the score pass, hours in. `.b_once.<id>` is a file for the same reason.
#
# AND THE B_ GUARD STILL HOLDS. Unsharded, a manifest in the root means B_ was spent -- unchanged.
# Sharded, that clause alone would refuse the array's own later tasks, so the guard keys on a
# `.b_once.<array job id>` MARKER as well (the same shape `competitors/*/slurm/predict.sh` use):
# every task of one array writes the same marker name, so a sibling finds its own and a later,
# different submission finds one that is not. The marker is written after every precondition and
# before the first byte is predicted: later than that and a second submission arriving mid-array
# would find nothing to refuse it; earlier and a split that cannot run would leave an empty root
# claimed for good. Because the root carries no manifest until the
# merge step, a half-finished array can never be read as a finished pass -- by the guard or by the
# scorer. SHARD_ARRAY presents an array's marker and is accepted ONLY with SHARD_MERGE=1, which
# predicts nothing; it is not a way to join an array that is still predicting.
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
DEVICE="${DEVICE:-}"                           # empty = bench.dump chooses (cuda if available)
SHARD_CHROMS="${SHARD_CHROMS:-0}"              # 1 = one array task per planned chromosome
SHARD_MERGE="${SHARD_MERGE:-0}"                # 1 = assemble the one manifest, predict nothing
SHARD_ARRAY="${SHARD_ARRAY:-}"                 # the array job id a SHARD_MERGE=1 step assembles
# -------------------------------------------------------------------------------------------------

export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
export WANDB_MODE=disabled

# THREADS, ON THE CPU ROUTE ONLY. Nothing in `candi` calls `torch.set_num_threads`, so the whole
# knob is the environment: torch takes its default thread count from OMP_NUM_THREADS at import
# (checked -- OMP_NUM_THREADS=3 gives torch.get_num_threads()==3), and a plain sbatch script is not
# handed one. Left unset the OpenMP runtime sizes itself off the node rather than the allocation,
# which on a shared cpubase node is oversubscription, and it is the one knob between a shard that
# fits its walltime and one that does not. Set on DEVICE=cpu only, so the GPU path is untouched.
if [ "$DEVICE" = "cpu" ]; then
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
  export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
fi

case "$PANEL" in
  V_|B_) ;;
  *) echo "[error] PANEL must be V_ or B_ (with the underscore), got '$PANEL'" >&2; exit 1;;
esac
case "$REGIME_NAME" in
  eic_19|eic_pilot) ;;
  *) echo "[error] REGIME_NAME must be eic_19 or eic_pilot, got '$REGIME_NAME'" >&2; exit 1;;
esac
REGIME="configs/regime.${REGIME_NAME}.json"

# --- what the sharding flags may and may not mean (exit 6) ---------------------------------------
if [ "$SHARD_MERGE" = "1" ]; then
  if [ "$SHARD_CHROMS" != "1" ]; then
    echo "[error] SHARD_MERGE=1 is the tail of a SHARD_CHROMS=1 run and assembles its shards." >&2
    echo "        Set SHARD_CHROMS=1 too, or drop it: an unsharded dump writes its own manifest." >&2
    exit 6
  fi
  if [ -z "$SHARD_ARRAY" ]; then
    echo "[error] SHARD_MERGE=1 needs SHARD_ARRAY=<array job id>: the merged manifest records" >&2
    echo "        which array wrote the root, and for PANEL=B_ that id is also the marker the" >&2
    echo "        once-only guard checks. Read it off the sbatch --parsable of the array." >&2
    exit 6
  fi
elif [ -n "$SHARD_ARRAY" ]; then
  echo "[error] SHARD_ARRAY names the array a SHARD_MERGE=1 step assembles, and predicts nothing." >&2
  echo "        It is not a way to join an array that is still predicting. Refusing." >&2
  exit 6
fi

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
  if [ "$SHARD_CHROMS" = "1" ]; then
    # SHARDED. 23 tasks of ONE array must all get through, and a later, different submission must
    # not. The marker is what tells them apart: every task of this array writes the same
    # `.b_once.<array job id>` name, so a sibling finds its own and a re-submission six weeks later
    # finds one that is not. A manifest still refuses, but only a submission carrying no marker of
    # its own -- and the root carries none until the SHARD_MERGE=1 step, so a half-finished array
    # cannot be read as a spent panel by this guard or by the scorer. The marker is DROPPED further
    # down, after every precondition and before the dump: a split that cannot run must not leave an
    # empty root claimed, and a second submission arriving mid-array must still find it there.
    MARK="$OUT/.b_once.${SHARD_ARRAY:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
    if { [ -f "$OUT/manifest.json" ] && [ ! -e "$MARK" ]; } ||
       { compgen -G "$OUT/.b_once.*" >/dev/null 2>&1 && [ ! -e "$MARK" ]; }; then
      echo "[error] $OUT already holds a B_ pass claimed by another array (no $MARK)." >&2
      echo "        B_ is written ONCE (§5) and a second one is a second look at the blind panel," >&2
      echo "        not a retry. There is no --force: move the old root aside by hand, with the" >&2
      echo "        PI, and record why. REFUSING." >&2
      exit 4
    fi
    echo "[t81-pred] B_ONCE=1, shard marker $MARK — this array is the once-only test-panel run."
  else
    if [ -f "$OUT/manifest.json" ]; then
      echo "[error] $OUT/manifest.json already exists. B_ was already predicted for" >&2
      echo "        $REGIME_NAME, and §5 allows exactly one. There is no --force here: move the old" >&2
      echo "        root aside by hand, with the PI, and record why. REFUSING." >&2
      exit 4
    fi
    echo "[t81-pred] B_ONCE=1 and $OUT carries no manifest — this is the once-only test-panel run."
  fi
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
echo "[t81-pred] device=${DEVICE:-auto} sharded=$SHARD_CHROMS merge=$SHARD_MERGE" \
     "array=${SHARD_ARRAY:-${SLURM_ARRAY_JOB_ID:--}} task=${SLURM_ARRAY_TASK_ID:--}" \
     "threads=${OMP_NUM_THREADS:-default}"
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

# --- shard index -> chromosome, by POSITION in the list the plan just derived ----------------------
# The same list, in the same order, an unsharded run would have dumped, so shard i and an unsharded
# run agree on what chromosome i is. The array must be exactly as long as the list: a shorter one
# leaves chromosomes unpredicted and a longer one runs indices that name nothing, and both are found
# out only when the merge step refuses hours later.
ALL_CHROMS="$CHROMS"
SHARD_CHROM=""
if [ "$SHARD_CHROMS" = "1" ] && [ "$SHARD_MERGE" != "1" ]; then
  IFS=',' read -r -a SHARD_LIST <<< "$ALL_CHROMS"
  N_SHARDS="${#SHARD_LIST[@]}"
  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "[error] SHARD_CHROMS=1 predicts ONE chromosome per array task and this job is not an" >&2
    echo "        array. Submit it with --array=0-$((N_SHARDS - 1))%12." >&2
    exit 6
  fi
  if [ -n "${SLURM_ARRAY_TASK_COUNT:-}" ] && [ "$SLURM_ARRAY_TASK_COUNT" != "$N_SHARDS" ]; then
    echo "[error] this array has $SLURM_ARRAY_TASK_COUNT tasks and the plan plans $N_SHARDS" >&2
    echo "        chromosomes ($ALL_CHROMS). One task is one chromosome, by position, so a" >&2
    echo "        mismatch either leaves chromosomes unpredicted or predicts one twice." >&2
    echo "        Resubmit with --array=0-$((N_SHARDS - 1))%12." >&2
    exit 6
  fi
  SHARD_CHROM="${SHARD_LIST[$SLURM_ARRAY_TASK_ID]:-}"
  if [ -z "$SHARD_CHROM" ]; then
    echo "[error] array index $SLURM_ARRAY_TASK_ID names no chromosome; the plan has $N_SHARDS" >&2
    echo "        ($ALL_CHROMS). Indices run 0-$((N_SHARDS - 1))." >&2
    exit 6
  fi
  CHROMS="$SHARD_CHROM"
  echo "[t81-pred] shard $SLURM_ARRAY_TASK_ID/$N_SHARDS -> $SHARD_CHROM (the --time above is the" \
       "GPU figure for the WHOLE panel, not for one chromosome)"
fi

# --- the dump ------------------------------------------------------------------------------------
# --device only when one was named, so the default path hands `bench.dump` the same argv it always
# did and keeps its own cuda-if-available choice. --notes rides along with it because the device a
# root was predicted on is the one thing about a CPU pass a reader cannot recover from the arrays.
DUMP_ARGS=()
if [ -n "$DEVICE" ]; then DUMP_ARGS+=(--device "$DEVICE" --notes "device=$DEVICE"); fi

# CLAIM THE B_ ROOT NOW, and not after the dump. Every precondition has passed — B_ONCE, the guard,
# the venv, the kit, the arch json, the checkpoint, the panel split, the plan, the shard index — so
# this array is the pass it says it is. Written before a byte is predicted because the marker is
# what makes a DIFFERENT submission arriving mid-array a refusal rather than a second B_ pass; a
# marker written afterwards would leave that whole window open.
if [ "$PANEL" = "B_" ] && [ "$SHARD_CHROMS" = "1" ] && [ "$SHARD_MERGE" != "1" ]; then
  mkdir -p "$OUT" && : > "$MARK" || exit 1
fi

rc=0
if [ "$SHARD_MERGE" != "1" ]; then
  python -m candi.bench.dump \
    --store "$DERIVED" \
    --ckpt "$CKPT" \
    --arch-from "$ARCH_FROM" \
    --out "$OUT" \
    --method CANDI \
    --version "$VERSION" \
    --chroms "$CHROMS" \
    --batch-windows "$BATCH_WINDOWS" \
    ${DUMP_ARGS[@]+"${DUMP_ARGS[@]}"}
  rc=$?
fi

if [ $rc -eq 0 ] && [ "$SHARD_CHROMS" = "1" ] && [ "$SHARD_MERGE" != "1" ]; then
  # A SHARD writes a manifest naming its ONE chromosome, because that is what it was given. Left in
  # the root, the last shard to finish would leave the scorer reading `chroms: [chr7]` and scoring
  # one chromosome as if it were the genome. So it is moved aside, and the root carries no manifest
  # until the merge step verifies the whole thing.
  if [ -f "$OUT/manifest.json" ]; then
    mv -f "$OUT/manifest.json" "$OUT/.shard_manifest.${SHARD_CHROM}.json" || rc=5
  elif compgen -G "$OUT/.shard_manifest.*.json" >/dev/null 2>&1; then
    # Two shards finishing in the same instant: the other one wrote the root manifest over ours and
    # moved it before we looked. Harmless -- the shard manifests differ only in `chroms`, which the
    # merge step derives from the plan and checks against the npz on disk, never from them.
    echo "[t81-pred] $SHARD_CHROM: a sibling shard moved the root manifest first; its arrays are" >&2
    echo "[t81-pred]   written and a shard-manifest template is already there. Continuing." >&2
  else
    echo "[t81-pred] WARNING: dump returned 0 but $OUT/manifest.json is absent and no shard" >&2
    echo "[t81-pred]   manifest is there. bench.dump writes the manifest LAST, so this shard is incomplete." >&2
    rc=5
  fi
elif [ $rc -eq 0 ] && [ "$SHARD_MERGE" != "1" ] && [ ! -f "$OUT/manifest.json" ]; then
  echo "[t81-pred] WARNING: dump returned 0 but $OUT/manifest.json is absent. bench.dump writes" >&2
  echo "[t81-pred]   the manifest LAST, so a root without one is incomplete and cannot be scored." >&2
  rc=5
fi

# --- the merge: ONE manifest over the whole root ---------------------------------------------------
# Takes any shard manifest as its template -- they are `bench.dump`'s own output and differ only in
# `chroms`, so nothing about the identity half is retyped here -- then REFUSES unless every declared
# track holds an npz for every planned chromosome. Only `chroms` and `notes` are rewritten; no key
# `slurm/t81_score_external.sh` reads changes shape.
if [ $rc -eq 0 ] && [ "$SHARD_MERGE" = "1" ]; then
  python - "$OUT" "$ALL_CHROMS" "${DEVICE:-auto}" "$SHARD_ARRAY" <<'PYEOF'
import json, os, sys
from pathlib import Path

out, chroms_arg, device, array_id = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
chroms = [c for c in chroms_arg.split(",") if c]
shards = sorted(out.glob(".shard_manifest.*.json"))
if not shards:
    print(f"[t81-pred] REFUSING: no {out}/.shard_manifest.*.json. Each shard of a SHARD_CHROMS=1 "
          f"array renames the manifest it wrote to one of those; with none there, no shard "
          f"finished and there is nothing to assemble.", file=sys.stderr)
    raise SystemExit(5)
man = json.loads(shards[0].read_text())
declared = [str(t) for t in (man.get("declared_tracks") or [])]
if not declared:
    print(f"[t81-pred] REFUSING: {shards[0]} names no declared_tracks, so there is nothing to "
          f"check the root against.", file=sys.stderr)
    raise SystemExit(5)
missing = [f"{t}/{c}.npz" for t in declared for c in chroms
           if not (out / t / f"{c}.npz").is_file()]
if missing:
    print(f"[t81-pred] REFUSING: {len(missing)} of {len(declared) * len(chroms)} shard outputs are "
          f"absent under {out} -- {missing[:5]}. A manifest naming all {len(chroms)} chromosomes "
          f"over a root that does not hold them is a panel scored with holes in it.",
          file=sys.stderr)
    raise SystemExit(5)
prev = str(man.get("notes") or "").strip()
note = f"device={device}; sharded: true; array={array_id}; shards={len(shards)}"
man["chroms"] = list(chroms)
# The shard already recorded the device, and this line says it again with the sharding beside it.
# Keeping both would put `device=cpu device=cpu; ...` into every score file that copies provenance.
man["notes"] = note if (not prev or prev in note) else (prev + " " + note)
tmp = out / "manifest.json.merge.tmp"
tmp.write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
os.replace(tmp, out / "manifest.json")
print(f"[t81-pred] merged {len(shards)} shard manifest(s) into {out}/manifest.json: "
      f"{len(chroms)} chromosomes x {len(declared)} declared tracks, {note}", file=sys.stderr)
PYEOF
  rc=$?
fi

# The derived regime is named again here because it is the score pass's --store, and reading it off
# this line beats re-deriving it and hoping the two agree.
echo "[t81-pred] DONE regime=$REGIME_NAME panel=$PANEL rc=$rc out=$OUT derived=$DERIVED" \
     "chroms=$CHROMS merge=$SHARD_MERGE"
exit $rc
