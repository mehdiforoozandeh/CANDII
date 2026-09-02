#!/bin/bash
# ChromImpute's σ table, fit on TRAINING residuals. BENCHMARK_DESIGN.md §7 and Rule 1.
#
#   CI_RUN=<a finished V_ run directory> bash competitors/chromimpute/slurm/sigma.sh
#
# Run from a login node like `submit.sh`: this script submits a three-link chain and returns.
#
# WHY THIS STAGE EXISTS. ChromImpute predicts a point in -log10 p and no spread, so the pval arm
# can only be scored as a distribution against a σ handed in. Every σ this tree used to produce was
# squared off a V_ scores json, which Rule 1 forbids and §12.2 declared VOID.
#
# WHY IT CANNOT REUSE THE BOARD'S PREDICTORS. `lists/train.txt` is `cut -f1,3 targets.tsv`, so its
# every (T_x, mark) is a mark T_x does NOT carry — the store holds no truth for it and there is no
# residual to take. A training residual needs the other case: (T_x, mark) for a mark T_x DOES
# carry. So the chain is
#
#   strain   Train a predictor per (sigma cell, mark it carries), into SIGMAPREDICTORDIR
#   sapply   Apply it on the TRAINING grid, into SIGMAIMPUTEDIR
#   fit      collect.py -> a §4.1 root -> competitors.sigma_pass -> the pinned σ table
#
# The traindata `strain` reads was already written by the run's `gtd` stage, on the training grid,
# so nothing here is fit on an eval chromosome.
#
# A `regions` REGIME IS RUN LIKE ANY OTHER (2026-09-01). Its training grid is one declared
# pseudo-chromosome per Pilot Region, and `collect.py --regime` maps each of those back to its
# chromosome and bin offset — so the fit below is handed the SOURCE regime, not the derived one.
set -euo pipefail

REPO=${CI_REPO:-/project/def-maxwl/mforooz/CANDII_main}
STORE=${CI_STORE:-/project/def-maxwl/mforooz/CANDI_STORE/eic}
PY=${CI_PY:-/project/def-maxwl/mforooz/EpiDenoise/candi_venv/bin/python}
SRC_REGIME=${CI_REGIME:-$REPO/configs/regime.eic_19.json}
REGIME_NAME=$(basename "$SRC_REGIME" .json); REGIME_NAME=${REGIME_NAME#regime.}
RUN=${CI_RUN:?set CI_RUN — the finished V_ run directory whose gtd output the fit reads}
JAR=${CI_JAR:-/project/def-maxwl/mforooz/tools/ChromImpute.jar}
ACCT=${CI_ACCT:-def-maxwl}
CHROMS=${CI_CHROMS:-chr20,chr21,chr22}
HERE=$REPO/competitors/chromimpute
STAGE=$HERE/slurm/stage.sh
# The seeded sample is pinned: 12 cells, seed 890217, the same two numbers for every method, so a
# σ difference between methods is a difference of method and not of which cells were drawn.
N_CELLS=${CI_SIGMA_CELLS:-12}
SIGMA_SEED=${CI_SIGMA_SEED:-890217}
TRAIN_PRED=${CI_TRAIN_PRED:-/scratch/mforooz/t81_sigma/ChromImpute/$REGIME_NAME/train_pred}
SIGMA_OUT=${CI_SIGMA_OUT:-/project/def-maxwl/mforooz/t81_sigma/ChromImpute/sigma_$REGIME_NAME.json}

for f in "$RUN/input/inputinfofile.txt" "$RUN/input/chrominfo.train.txt" "$RUN/lists/gtd.txt"; do
  [ -s "$f" ] || { echo "[error] $f is missing or empty — run submit.sh through gtd first"; exit 2; }
done
mkdir -p "$RUN/lists" "$RUN/logs" "$(dirname "$SIGMA_OUT")"

# --- 1. the seeded training regime ---------------------------------------------------------------
DRAW=$RUN/regime.$REGIME_NAME.sigma.draw.json
SIGMA_REGIME=$RUN/regime.$REGIME_NAME.sigma.json
$PY "$REPO/tools/sigma_training_regime.py" \
    --regime "$SRC_REGIME" --n-cells "$N_CELLS" --seed "$SIGMA_SEED" --out "$DRAW"
echo "[ci_sigma] drawn training regime: $DRAW"

# --- 2. the work lists, and the regime that matches them -----------------------------------------
# THE DRAW CARRIES NO PAIRS, AND THE NARROWED REGIME MUST NOT EITHER. `sigma_training_regime.py`
# writes `eval_pairs: []` and puts the drawn cells in `biosamples.eval`, because
# `candi.store.regime._parse_eval_pairs` REFUSES a pair of a cell with itself outright. A regime
# narrowed by rewriting eval_pairs to a list of self-pairs would not load at all, and the fit would
# die on its first read of the file this stage just wrote. On the empty-pairs shape
# `bench.harness.StoreSource` self-pairs every cell in `biosamples.eval` — its documented
# no-pairing path — and `sigma_pass` walks those. So: read the cells out of `biosamples.eval`, and
# narrow `biosamples.eval` and nothing else.
#
# A cell is kept ALL OR NOTHING. A self-pair's target panel is every assay the cell holds, so a
# cell whose marks are not all trainable would send the fitter looking for a track this chain never
# wrote. Dropping the whole cell keeps the regime and the prediction root describing the same thing.
cd "$HERE"
echo "[banner] code=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) kit=$REPO"
PYTHONPATH=$REPO/src:$HERE $PY - "$DRAW" "$SIGMA_REGIME" "$RUN" "$STORE" <<'PYEOF'
import json, sys
from pathlib import Path
from prepare import load_json, manifest_assays, write_targets, NOT_AN_ASSAY

draw_p, out_p, run, store = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
draw = json.loads(draw_p.read_text())
manifest = load_json(store / "manifest.json")

# The grid `sapply` writes wigs on: store chromosomes under a plain regime, one Pilot Region
# pseudo-chromosome per line under a `regions` one. `collect.py --regime` reads both.
grid = [ln.split("\t")[0] for ln in
        (run / "input" / "chrominfo.train.txt").read_text().splitlines() if ln.strip()]

# The marks `gtd` produced training data for. Read off the work list rather than off filenames in
# TRAINDATADIR, so the check does not depend on the jar's naming.
avail = {ln.split("\t")[0] for ln in (run / "lists" / "gtd.txt").read_text().splitlines() if ln.strip()}
allowed = [a for a in draw["assays"] if a not in NOT_AN_ASSAY]

# The drawn cells, off `biosamples.eval`. NOT off `eval_pairs`: the draw declares none, so reading
# pairs here would find no cells at all and the `len(kept) < 3` guard below would refuse every run.
cells = [str(c) for c in ((draw.get("biosamples") or {}).get("eval") or [])]
if not cells:
    sys.exit("[ci_sigma] REFUSING: the draw declares no `biosamples.eval`. That is where "
             "tools/sigma_training_regime.py puts the sampled training cells; an empty list means "
             "the draw failed, or its shape changed and this stage has not been told.")
if draw.get("eval_pairs"):
    sys.exit(f"[ci_sigma] REFUSING: the draw declares {len(draw['eval_pairs'])} eval_pairs. A sigma "
             f"regime is the NO-PAIRING shape: candi.store.regime refuses a self-pair, and a "
             f"cross-cell pair here would be an eval pair, which Rule 1 forbids a sigma to see.")

kept, dropped, items = [], [], []
for cell in cells:
    marks = [m for m in manifest_assays(manifest, cell, "pval") if m in allowed]
    missing = [m for m in marks if m not in avail]
    if not marks or missing:
        dropped.append((cell, missing or ["no pval mark in the regime's assays"]))
        continue
    kept.append(cell)
    items.extend((cell, m) for m in marks)

print(f"[ci_sigma] drawn cells: {len(cells)} | kept: {len(kept)} | dropped: {len(dropped)}")
for cell, why in dropped:
    print(f"[ci_sigma]   dropped {cell}: no training data for {why}")
if len(kept) < 3:
    sys.exit(f"[ci_sigma] REFUSING: only {len(kept)} cell(s) survive. A σ per assay off fewer than "
             f"three cells is not a spread estimate. Widen the gtd mark set and re-run.")

lines = "\n".join(f"{c}\t{m}" for c, m in items) + "\n"
(run / "lists" / "strain.txt").write_text(lines, encoding="utf-8")
(run / "lists" / "sapply.txt").write_text(lines, encoding="utf-8")
write_targets(run / "input" / "targets_sigma.tsv", [(c, c, m) for c, m in items])

out = json.loads(json.dumps(draw))
# Stays EMPTY. regime.py refuses a [cell, cell] pair, and the narrowing is expressed by which cells
# `biosamples.eval` names — StoreSource self-pairs exactly those.
out["eval_pairs"] = []
if kept != cells:
    out["biosamples"]["eval"] = list(kept)
    out["_comment"] = (str(draw.get("_comment", "")) + " NARROWED by "
                       "competitors/chromimpute/slurm/sigma.sh: a drawn cell is kept only when "
                       "every mark it carries has ChromImpute training data, because a self-pair's "
                       "target panel is all of the cell's marks and a partial cell would send the "
                       "fitter looking for a track that was never applied.")
out_p.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
print(f"[ci_sigma] {len(items)} (cell, mark) items over {len(kept)} cells -> {out_p.name}")
print(f"[ci_sigma] training grid: {grid}")
PYEOF

# --- 3. the chain --------------------------------------------------------------------------------
# THE STAGE ENVIRONMENT IS EXPORTED HERE, NOT LISTED ON THE `sbatch` LINE — the same repair as
# `submit.sh`, and the same rule behind it: `sbatch --export=<list>` is a COMMA-SEPARATED list of
# `NAME` / `NAME=VALUE` entries, so a comma-valued variable cannot be passed in it.
# `CI_CHROMS=chr20,chr21,chr22` inside the list reached the stage as `CI_CHROMS=chr20`, with
# `chr21` / `chr22` read as bare variable names that do not exist — silently. Measured on Fir, job
# 57806189 (cruxvault/results/t81/W3_EARLY.md §1).
#
# It bit this script less than `submit.sh` only by luck of which grid the σ stages loop over
# (`strain`/`sapply` use `TRAIN_LIST`, not `CHROM_LIST`), and it bites a pilot σ hard: under a
# `regions` regime the grid is a comma list of Pilot Region names, so the truncation would leave
# `sapply` writing one region's wig. `--export=ALL` hands the submitting shell's environment over
# verbatim, commas intact; the list it replaces already began with `ALL`, so nothing was scrubbed
# and nothing else the stages see changes.
#
# The explicit `export` keeps the values THIS SCRIPT's own rather than the operator's login shell's
# — every one was resolved from this file's own defaults at the top.
export CI_RUN="$RUN" CI_REPO="$REPO" CI_STORE="$STORE" CI_PY="$PY" CI_JAR="$JAR"
export CI_CHROMS="$CHROMS" CI_REGIME="$SIGMA_REGIME"
# `%THROTTLE` caps how many array tasks of one stage run at once. It is the SAME courtesy cap
# `submit.sh` uses on `gtd` and `apply`, and for the same reason: the jar holds one open gzip reader
# per compendium track, and an unthrottled array puts thousands of concurrent handles on Lustre.
#
# IT IS NOT THE TORCH-IMPORT CAP. Both stages below are java-only — `stage.sh` runs the jar for
# `strain` and `sapply` and imports nothing — so the twelve-way limit is irrelevant here. That
# limit lives on `CI_NSHARD=12` in `submit.sh`, where `prepare.py` DOES import torch off the shared
# /project venv and more than twelve concurrent imports return half-read modules. Move this number
# for queue etiquette; move that one never.
THROTTLE=${CI_THROTTLE:-10}

sub() {  # sub <stage> <time> <mem> <mx> [afterok-jobid]
  local stage=$1 time=$2 mem=$3 mx=$4 dep=${5:-}
  local n; n=$(wc -l < "$RUN/lists/$stage.txt")
  local depflag=(); [ -n "$dep" ] && depflag=(--dependency="afterok:$dep")
  # Exported, not listed, for the same reason as the block above: `sbatch --export=<list>` splits
  # its argument on commas, so no assignment goes on that line at all.
  export CI_STAGE="$stage" CI_MX="$mx"
  # `${depflag[@]+"${depflag[@]}"}` and not a bare `"${depflag[@]}"`: identical on Fir's bash, and
  # the only form bash 3.2 accepts for an EMPTY array under `set -u` — which is `strain`, the first
  # link of this chain, and is what the regression test in tests/ submits.
  sbatch --parsable --account="$ACCT" --nodes=1 --ntasks=1 --cpus-per-task=1 \
         --job-name="ci_$stage" --array="0-$((n - 1))%$THROTTLE" --time="$time" --mem="$mem" \
         --output="$RUN/logs/%x_%A_%a.out" --error="$RUN/logs/%x_%A_%a.err" \
         ${depflag[@]+"${depflag[@]}"} --export=ALL "$STAGE"
}

D1=$(sub strain 6:00:00 24000M 20000M)
printf '%-8s %s\n' "strain" "$D1"
D2=$(sub sapply 12:00:00 16000M 12000M "$D1")
printf '%-8s %s\n' "sapply" "$D2"

# The fit. `--eval-regions` is the regime's own BED, passed explicitly: redundant if `sigma_pass`
# reads `regions` off the regime, and load-bearing if it does not — and a fit outside the Pilot
# Regions would be taken on loci the model never trained on.
#
# `collect.py --regime` is handed $SRC_REGIME, the SOURCE regime — never $SIGMA_REGIME. The map from
# a region pseudo-chromosome back to its offset is `prepare.region_scope`, which cuts the BED to
# `train_chroms`, and the derived σ regime empties that list; collect.py refuses the derived file by
# name for exactly that reason.
#
# NO `--notes`. collect.py's own note records the grid it collapsed and says "NaN outside the
# regions", which under a region grid is the one fact a reader of this root needs; a note of ours
# would replace it with prose that cannot know how many regions there were.
cat > "$RUN/sigma_fit.sh" <<EOF
#!/bin/bash
set -euo pipefail
GRID=\$(cut -f1 $RUN/input/chrominfo.train.txt | paste -sd, -)
PYTHONPATH=$REPO/src $PY $HERE/collect.py --store $STORE \\
    --targets $RUN/input/targets_sigma.tsv --impute-dir $RUN/SIGMAIMPUTEDIR \\
    --pred-root $TRAIN_PRED --chroms "\$GRID" --jar $JAR --regime $SRC_REGIME
BED=\$(PYTHONPATH=$REPO/src $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
print((d.get('regions') or {}).get('bed',''))" $SIGMA_REGIME)
EXTRA=()
[ -n "\$BED" ] && EXTRA+=(--eval-regions "\$BED")
cd $REPO
# NO \`--chroms\`. \$GRID is the APPLY grid, whose names under a \`regions\` regime are region
# pseudo-chromosomes; collect.py has already written the arrays back onto real chromosomes, and
# sigma_pass would refuse those names as held out. Left off, it fits on the σ regime's own
# \`eval_chroms\` — which IS the source regime's \`train_chroms\`, the same loci.
PYTHONPATH=$REPO/src $PY -m competitors.sigma_pass \\
    --regime $SIGMA_REGIME --pred $TRAIN_PRED --out $SIGMA_OUT \\
    --method ChromImpute "\${EXTRA[@]}"
PYTHONPATH=$REPO/src $PY -c "
import json,sys
d=json.load(open(sys.argv[1]))
print('[ci_sigma] fitted_on =', d['fitted_on'])
print('[ci_sigma] assays    =', len(d['sigma']))" $SIGMA_OUT
EOF
chmod +x "$RUN/sigma_fit.sh"

D3=$(sbatch --parsable --account="$ACCT" --nodes=1 --ntasks=1 --cpus-per-task=4 \
     --job-name=ci_sigmafit --time=6:00:00 --mem=32000M \
     --output="$RUN/logs/%x_%j.out" --error="$RUN/logs/%x_%j.err" \
     --dependency="afterok:$D2" --wrap "$RUN/sigma_fit.sh")
printf '%-8s %s\n' "fit" "$D3"
echo "[ci_sigma] σ table will land at $SIGMA_OUT"
