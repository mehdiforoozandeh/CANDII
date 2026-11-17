#!/bin/bash
# Close-out A7 (peeling the onion): resolution 1 to 125 bp. Item file:
# cruxvault/INSTRUCTIONS/peeling_the_onion/closeout/closeout A7 resolution.md ("Specification as worked").
# Dry run is the default. Runs on a Nibi login node:
#
#   ssh -o BatchMode=yes nibi 'bash -s -- plan recount'          < slurm/onion_data/closeout_a7_submit.sh
#   ssh -o BatchMode=yes nibi 'DRY_RUN=0 bash -s -- plan recount' < slurm/onion_data/closeout_a7_submit.sh
#
# Parts, in their order of use (give one or more):
#   probe     (after plan) 1 job, 2 CPUs: writes $A7/regions.tsv (256 eval + 512 training regions of 409,600 bp),
#             recounts 2 views into $A7/probe_store with the in-run read-set check, and times one
#             forward + backward pass per window of every arm on rungs 2, 3, 4. Small (< 2 CPU-h).
#   plan      login node, read-only on the data: writes $A7/recount_plan.tsv, one line per view
#             (measurement id, regions role, halves, fragment length and its source, tagAlign bytes)
#             and the task number of the array (views grouped to about $TASK_GIB of tagAlign each).
#   recount   the array over the plan (recount.py run --verify-counts). It READS the half tagAligns
#             under /scratch/mforooz/onion/work, so it MUST finish before that folder leaves scratch.
#             It writes only $STORE (Nibi /project). Refused if $STORE holds a view already.
#   tables    eic_res_v1 tables from the store, `verify --all --no-rescan` (store only), the eval
#             set of the resolution study (64 eval + 16 fit regions drawn inside the 256 eval
#             regions of the store, seed 0) and its check. 1 job, 8 CPUs.
#   variance  scale_variance.py on the validation views (chains 1-5-25-125 and 1-5-25-50). 1 job.
#   arms      per (arm, rung, seed): train (res_dataset.py) -> predict (res_score.py, once) ->
#             tables at every size the output bin divides (25, 50, 125, 250 bp), with dependencies.
#             DESIGN=reduced (default): conv autoencoder, res 5, 25, 125, 3 seeds = 9 runs.
#             DESIGN=full: 8 arms on conv AE and U-Net, 6 arms on conv + attention, 3 seeds = 66 runs.
#
# Env: DRY_RUN (default 1: print, submit nothing), KIT_ID (default COA7), DESIGN (reduced|full),
#      SEEDS (default "0 1 2"), TASK_GIB (default 6), MAX_PAR (array throttle, default 40).
set -uo pipefail
DRY_RUN="${DRY_RUN:-1}"; KIT_ID="${KIT_ID:-COA7}"; DESIGN="${DESIGN:-reduced}"
SEEDS="${SEEDS:-0 1 2}"; TASK_GIB="${TASK_GIB:-6}"; MAX_PAR="${MAX_PAR:-40}"
PARTS="${*:-}"
[ -n "$PARTS" ] || { echo "usage: closeout_a7_submit.sh probe|plan|recount|tables|variance|arms ..."; exit 2; }
for p in $PARTS; do case "$p" in probe|plan|recount|tables|variance|arms) ;;
  *) echo "unknown part '$p'"; exit 2;; esac; done

O=/project/def-maxwl/mforooz/ONION
A7=$O/CO-A7                                    # this item's own folder (new paths only)
STORE=$A7/res_store                            # the 1 bp region store (pinned interface E layout)
KIT=/project/def-maxwl/mforooz/onion_kit/$KIT_ID
T=$O/tables/eic_v1_2                           # the current table version
EV=$O/eval/eic_v1/evalset.json                 # its 256 eval regions are the store's eval regions
MASK=$O/genome/mask.h5
W=/scratch/mforooz/onion/work                  # read only; leaves scratch (NIBI_NEARLINE_ARCHIVE.md)
M=/scratch/mforooz/onion/measurements          # eic_v1 counts.h5 of every row (stays on scratch)
RT=$A7/tables/eic_res_v1; REV=$A7/eval/eic_res_v1
RUNS=$A7/runs
PLAN=$A7/recount_plan.tsv

die() { echo "REFUSED: $*"; exit 3; }
case "$(hostname)" in *nibi*) ;; *) die "run this on a Nibi login node";; esac
[ -f "$KIT/.kit_stamp" ] || die "no kit $KIT (push it: bash tools/onion_kit/push_kit.sh --cluster nibi $KIT_ID)"
echo "=== closeout A7 $(date -u +%FT%TZ) parts: $PARTS  DRY_RUN=$DRY_RUN  DESIGN=$DESIGN  kit: $(head -1 "$KIT/.kit_stamp")"
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES

run() {   # print a command; run it unless dry; echo its stdout (the job id of sbatch --parsable)
  echo "    \$ $*" >&2
  [ "$DRY_RUN" = 0 ] || { echo "DRYJOB"; return 0; }
  "$@" || { local rc=$?; echo "    FAILED (exit $rc): $1" >&2; return $rc; }
}
sb() {    # sbatch through the kit's check.sh: <name> <time> <cpus> <mem> <dep|-> <python args...>
  local name="$1" t="$2" c="$3" m="$4" dep="$5"; shift 5
  local d=(); [ "$dep" = - ] || d=(--dependency="afterok:$dep")
  run sbatch --parsable --account=def-maxwl --job-name="onion_a7_$name" --time="$t" \
    --cpus-per-task="$c" --mem="$m" --gres=none "${d[@]}" --output="$A7/logs/%x_%j.out" \
    "$KIT/slurm/onion_kit/check.sh" "$KIT" "$@"
}
[ "$DRY_RUN" = 0 ] && mkdir -p "$A7/logs"

# ------------------------------------------------------------------ plan
PY_PLAN='
import csv, glob, os, statistics, sys
T, W, M, A7, PLAN, task_gib = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6])
rows = list(csv.DictReader(open(f"{T}/measurements.tsv"), delimiter="\t"))
need = [r for r in rows if r["in_eic"] in ("train", "val") and not r["arm_knob"]
        and (r["unit"] == "pooled" or r["in_eic"] == "val")]
fl_of = lambda r: float(r["frag_len"]) if r["frag_len"].strip() else None
by_exp = {}
for r in rows:
    if r["unit"] != "pooled" and fl_of(r) and not r["arm_knob"]:
        by_exp.setdefault((r["experiment"], r["layer"]), []).append(fl_of(r))
chip = [v for vs in by_exp.values() for v in vs]
def frag(r):
    if r["assay_class"].startswith("accessibility"):
        return 0, "accessibility"
    if fl_of(r):
        return round(fl_of(r)), "row"
    if r["assay_class"] != "control":
        v = by_exp.get((r["experiment"], r["layer"]))
        if v:
            return round(sum(v) / len(v)), "replicate_mean"
        return round(statistics.median(chip)), "chip_median"
    named = []
    for x in rows:
        if x["control_measurement_id"] == r["measurement_id"] and not x["arm_knob"]:
            v = fl_of(x) or (lambda vs: sum(vs) / len(vs) if vs else None)(by_exp.get((x["experiment"], x["layer"])))
            if v:
                named.append(v)
    if named:
        return round(statistics.median(named)), "named_by_median"
    return round(statistics.median(chip)), "chip_median"
def readset(r):
    e, u, l = r["experiment"], r["unit"], r["layer"]
    d = f"{W}/{e}/{u}.{l}/base"
    fs = [f"{d}/{h}.tagAlign.gz" for h in ("pr1", "pr2")]
    if all(os.path.isfile(f) for f in fs):
        return fs
    if r["unit"] == "pooled":
        ds = sorted(glob.glob(f"{W}/{e}/br*.{l}/base"))
        fs = [f"{x}/{h}.tagAlign.gz" for x in ds for h in ("pr1", "pr2")]
        if ds and all(os.path.isfile(f) for f in fs):
            return fs
    return []
out, missing, task, acc = [], [], 1, 0.0
for r in sorted(need, key=lambda r: (r["in_eic"], r["unit"] != "pooled", r["measurement_id"])):
    fs = readset(r)
    mid = r["measurement_id"]
    cv1 = f"{M}/{mid}/counts.h5"
    if not fs or not os.path.isfile(cv1):
        missing.append(r["measurement_id"]); continue
    gib = sum(os.path.getsize(f) for f in fs) / 2**30
    if acc and acc + gib > task_gib:
        task, acc = task + 1, 0.0
    acc += gib
    fl, src = frag(r)
    pooled_val = r["unit"] == "pooled" and r["in_eic"] == "val"
    out.append({"task": task, "measurement_id": r["measurement_id"], "experiment": r["experiment"],
                "unit": r["unit"], "layer": r["layer"], "split": r["in_eic"],
                "assay_class": r["assay_class"],
                "roles": "all" if r["unit"] == "pooled" else "eval",
                "halves": 1 if pooled_val else 0, "frag_len": fl, "frag_len_source": src,
                "counts_v1": cv1, "tagalign_gib": round(gib, 3), "n_tagaligns": len(fs)})
cols = list(out[0])
with open(PLAN + ".part", "w") as fh:
    fh.write("\t".join(cols) + "\n")
    for o in out:
        fh.write("\t".join(str(o[c]) for c in cols) + "\n")
os.replace(PLAN + ".part", PLAN)
from collections import Counter
gib = sum(o["tagalign_gib"] for o in out)
print(f"views {len(out)}, tasks {task}, tagAlign {gib:.1f} GiB, missing {len(missing)} {missing[:10]}")
print("by (split, unit, roles, halves):", dict(Counter((o["split"], o["unit"][:2], o["roles"], o["halves"]) for o in out)))
print("frag_len source:", dict(Counter(o["frag_len_source"] for o in out)))
'
case " $PARTS " in *" plan "*)
  echo "--- plan -> $PLAN"
  [ -d "$W" ] || die "$W is gone: the plan needs the read sets on scratch"
  if [ "$DRY_RUN" = 0 ]; then python3 -c "$PY_PLAN" "$T" "$W" "$M" "$A7" "$PLAN" "$TASK_GIB"
  else python3 -c "$PY_PLAN" "$T" "$W" "$M" "$A7" "${TMPDIR:-/tmp}/a7_plan_dry.tsv" "$TASK_GIB"
       PLAN_DRY="${TMPDIR:-/tmp}/a7_plan_dry.tsv"; echo "    (dry: plan written to $PLAN_DRY, not to $PLAN)"; fi ;;
esac

# ------------------------------------------------------------------ probe
# views of the probe, from the plan (run the part plan first): the first validation histone view
# with halves, and the first training histone view without (each with the plan's fragment length)
PY_PROBE='
import json, os, sys, time, subprocess
A7, EV, MASK, W, M, T = sys.argv[1:7]
views = sys.argv[7:]
rc = subprocess.run([sys.executable, "tools/onion_data/recount.py", "regions", "--evalset", EV,
                     "--mask", MASK, "--n-train", "512", "--seed", "0",
                     "--out", f"{A7}/regions.tsv"]).returncode
if rc: sys.exit(rc)
for v in views:
    mid, halves, fl = v.split(":")
    exp, unit, layer, _ = mid.split(".")
    t0 = time.time()
    rc = subprocess.run([sys.executable, "tools/onion_data/recount.py", "run", "--work", W,
                         "--experiment", exp, "--unit", unit, "--layer", layer,
                         "--regions", f"{A7}/regions.tsv", "--frag-len", fl, "--halves", halves,
                         "--out", f"{A7}/probe_store",
                         "--verify-counts", f"{M}/{mid}/counts.h5"]).returncode
    print(f"[probe] {mid} halves={halves} rc={rc} wall_s={time.time() - t0:.1f}", flush=True)
sys.path.insert(0, "tools")
import torch
from minicandi.registry import build_model
from minicandi import contract as C
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
for rung in (2, 3, 4):
    base = json.load(open(f"tools/minicandi/configs/rung{rung}/base_heldout.json"))
    for arm in sorted(os.listdir(f"tools/minicandi/configs/resolution/rung{rung}")):
        cfg = json.load(open(f"tools/minicandi/configs/resolution/rung{rung}/{arm}"))
        cfg.pop("res_store", None)
        torch.manual_seed(0)
        m = build_model(cfg)
        L, res, B = int(cfg["context_bins"]), int(cfg.get("res_bp", 25)), 8
        g = torch.Generator().manual_seed(0)
        b = {"x_counts": torch.randint(0, 6, (B, L, C.A), generator=g, dtype=torch.int32),
             "x_avail": torch.ones(B, C.A, dtype=torch.bool),
             "x_bin_mask": torch.rand(B, L, C.A, generator=g) < 0.3,
             "x_dna": torch.randint(0, 5, (B, L * res), generator=g).to(torch.uint8),
             "c_in": torch.randn(B, C.A, C.k_input(cfg), generator=g) + 24.0,
             "c_out": torch.randn(B, C.A, C.K_META, generator=g) + 24.0,
             "y_avail": torch.ones(B, C.A, dtype=torch.bool)}
        s = cfg.get("res_in_sub", 1)
        if s > 1:
            b["x_counts_sub"] = torch.randint(0, 2, (B, L, C.A, s), generator=g, dtype=torch.int32)
        if cfg.get("res_out_sub", 1) > 1:
            b["y_counts_sub"] = torch.randint(0, 2, (B, L, C.A, 5), generator=g, dtype=torch.int32)
        m.eval()
        with torch.no_grad():
            m(b); t0 = time.time(); m(b); m(b); fwd = (time.time() - t0) / (2 * B)
        m.train()
        t0 = time.time()
        for _ in range(2):
            o = m(b); (o["mu"].log().mean() + o["n"].log().mean()).backward()
        fb = (time.time() - t0) / (2 * B)
        print(f"[time] rung{rung} {arm[:-5]} params={m.param_count()} fwd_s_per_window={fwd:.4f} "
              f"fwd_bwd_s_per_window={fb:.4f}", flush=True)
'
case " $PARTS " in *" probe "*)
  P="${PLAN_DRY:-$PLAN}"
  echo "--- probe -> $A7/regions.tsv, $A7/probe_store (views from $P)"
  [ -d "$W" ] || die "$W is gone: the half tagAligns are on nearline only"
  [ -s "$P" ] || die "no plan $P (run the part plan first)"
  [ -e "$A7/probe_store" ] && die "$A7/probe_store exists (the probe ran)"
  PV=$(awk -F'\t' 'NR>1 && $6=="val" && $4=="pooled" && $7 ~ /histone/ {print $2":"$9":"$10; exit}' "$P")
  PT=$(awk -F'\t' 'NR>1 && $6=="train" && $4=="pooled" && $7 ~ /histone/ {print $2":0:"$10; exit}' "$P")
  echo "    probe views: $PV $PT"
  [ "$DRY_RUN" = 0 ] && mkdir -p "$A7/probe_store"
  sb probe 02:00:00 2 16000M - -c "$PY_PROBE" "$A7" "$EV" "$MASK" "$W" "$M" "$T" "$PV" "$PT" >/dev/null ;;
esac

# ------------------------------------------------------------------ recount
PY_TASK='
import csv, os, subprocess, sys
plan, W, regions, store = sys.argv[1:5]
task = os.environ["SLURM_ARRAY_TASK_ID"]
rc = 0
for r in csv.DictReader(open(plan), delimiter="\t"):
    if r["task"] != task:
        continue
    d = os.path.join(store, r["measurement_id"])
    if os.path.isfile(os.path.join(d, "provenance.json")):
        print("[a7] skip", r["measurement_id"], "done", flush=True); continue
    x = subprocess.run([sys.executable, "tools/onion_data/recount.py", "run", "--work", W,
                        "--experiment", r["experiment"], "--unit", r["unit"], "--layer", r["layer"],
                        "--regions", regions, "--frag-len", r["frag_len"], "--halves", r["halves"],
                        "--roles", r["roles"], "--out", store, "--verify-counts", r["counts_v1"]])
    rc = rc or x.returncode
sys.exit(rc)
'
case " $PARTS " in *" recount "*)
  P="${PLAN_DRY:-$PLAN}"
  echo "--- recount: $P -> $STORE"
  [ -d "$W" ] || die "$W is gone: restore the read sets from nearline first"
  [ -s "$P" ] || die "no plan $P (run the part plan first)"
  [ -s "$A7/regions.tsv" ] || [ "$DRY_RUN" = 1 ] || die "no $A7/regions.tsv (run the part probe first)"
  [ "$DRY_RUN" = 0 ] && ls "$STORE"/*/provenance.json >/dev/null 2>&1 && \
    die "$STORE already holds views; resubmit only the missing tasks by hand (each task skips done views)"
  NT=$(awk -F'\t' 'NR>1{print $1}' "$P" | sort -n | tail -1)
  NV=$(awk 'NR>1' "$P" | wc -l)
  echo "    $NV views in $NT tasks; 2 CPUs, 24000M, 08:00:00 each; throttle %$MAX_PAR"
  [ "$DRY_RUN" = 0 ] && mkdir -p "$STORE"
  run sbatch --parsable --account=def-maxwl --job-name=onion_a7_recount --time=08:00:00 \
    --cpus-per-task=2 --mem=24000M --gres=none --array="1-$NT%$MAX_PAR" \
    --output="$A7/logs/%x_%A_%a.out" "$KIT/slurm/onion_kit/check.sh" "$KIT" \
    -c "$PY_TASK" "$PLAN" "$W" "$A7/regions.tsv" "$STORE" >/dev/null ;;
esac

# ------------------------------------------------------------------ tables
PY_TABLES="
import json, subprocess, sys
def run(*a):
    print('+', ' '.join(a), flush=True)
    r = subprocess.run([sys.executable, *a])
    if r.returncode: sys.exit(r.returncode)
run('tools/onion_data/recount.py', 'tables', '--eic-v1', '$T', '--store', '$STORE', '--out', '$RT')
run('tools/onion_data/recount.py', 'verify', '--all', '--tables', '$RT', '--no-rescan', '--workers', '8')
regs = [l.split('\t') for l in open('$RT/regions.tsv').read().splitlines()[1:]]
with open('$RT/eval_regions.bed', 'w') as fh:
    for i, c, s, role in regs:
        if role == 'eval':
            fh.write(f'{c}\t{s}\t{int(s) + 409600}\n')
import os; os.makedirs('$REV', exist_ok=False)
run('tools/onion_eval/evalset.py', 'build', '--tables', '$RT', '--strata', '$O/eval/strata/masks.h5',
    '--regions-bed', '$RT/eval_regions.bed', '--n-regions', '64', '--n-fit-regions', '16',
    '--seed', '0', '--out', '$REV/evalset.json')
run('tools/onion_eval/evalset.py', 'check', '--evalset', '$REV/evalset.json', '--tables', '$RT')
"
case " $PARTS " in *" tables "*)
  echo "--- tables -> $RT, $REV/evalset.json"
  [ -e "$RT" ] && die "$RT exists (frozen tables are never overwritten)"
  sb tables 04:00:00 8 32000M - -c "$PY_TABLES" >/dev/null ;;
esac

# ------------------------------------------------------------------ variance
case " $PARTS " in *" variance "*)
  echo "--- variance -> $A7/scale_variance.tsv"
  sb variance 12:00:00 8 32000M - tools/onion_eval/scale_variance.py run --store "$STORE" \
    --tables "$RT" --chains 1,5,25,125 1,5,25,50 --out "$A7/scale_variance.tsv" >/dev/null ;;
esac

# ------------------------------------------------------------------ arms
case " $PARTS " in *" arms "*)
  C=$KIT/tools/minicandi/configs/resolution
  declare -A ARCH=([2]=convae [3]=convattn [4]=unet)
  if [ "$DESIGN" = reduced ]; then LIST="2:res5 2:res25 2:res125"
  else LIST=""; for r in 2 3 4; do for f in "$C/rung$r"/*.json; do LIST="$LIST $r:$(basename "$f" .json)"; done; done; fi
  [ -s "$REV/evalset.json" ] || [ "$DRY_RUN" = 1 ] || die "no $REV/evalset.json (run the part tables first)"
  n=0
  for ra in $LIST; do
    r=${ra%%:*}; arm=${ra#*:}; cfg="$C/rung$r/$arm.json"
    [ -f "$cfg" ] || die "no config $cfg in the kit"
    sizes=$(python3 -c "
import json,sys; c=json.load(open(sys.argv[1])); r=c.get('res_bp',25); o=25//c.get('res_out_sub',1) if c.get('res_out_sub',1)>1 else r
cache=max(o,25); print(' '.join(str(b) for b in (25,50,125,250) if b%o==0 and b%cache==0))" "$cfg")
    ttime=08:00:00; case $arm in res5_ctx) ttime=24:00:00;; esac
    for s in $SEEDS; do
      R=$RUNS/${arm}_${ARCH[$r]}/seed$s
      [ -e "$R/DONE" ] && { echo "    skip $R: DONE"; continue; }
      echo "  - ${arm} ${ARCH[$r]} seed $s: tables at $sizes bp"
      [ "$DRY_RUN" = 0 ] && mkdir -p "$R"
      jt=$(sb "tr_${arm}_${ARCH[$r]}_s$s" "$ttime" 12 32000M - tools/minicandi/res_dataset.py train \
        --config "$cfg" --tables "$RT" --seed "$s" --out "$R" --workers 11)
      jp=$(sb "pr_${arm}_${ARCH[$r]}_s$s" 24:00:00 8 32000M "$jt" tools/onion_eval/res_score.py predict \
        --run "$R" --tables "$RT" --evalset "$REV/evalset.json" --infer-batch 8)
      for b in $sizes; do
        sb "tb_${arm}_${ARCH[$r]}_s${s}_$b" 04:00:00 4 32000M "$jp" tools/onion_eval/res_score.py table \
          --run "$R" --at "$b" --out "$R/res_table_$b.json" >/dev/null
      done
      n=$((n+1))
    done
  done
  echo "    runs: $n (DESIGN=$DESIGN)" ;;
esac
echo "=== done; DRY_RUN=$DRY_RUN"
