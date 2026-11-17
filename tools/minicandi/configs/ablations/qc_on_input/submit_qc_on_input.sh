#!/bin/bash
# Close-out item A4: read-derived QC on the input side. Specification: the item file
# "closeout A4 QC metrics on the input side.md", section "Specification as worked".
# Run on a login node. DRY RUN BY DEFAULT: it prints the checks and the sbatch lines, submits nothing.
#
#   MODE=fill bash submit_qc_on_input.sh      # Nibi only. The two QC fill passes that READ the
#                                             # tagAligns under /scratch/mforooz/onion/work; they
#                                             # must run BEFORE that folder leaves scratch:
#                                             #   whole: the training/validation rows with no record
#                                             #          that lack a QC column or are a non-base view
#                                             #          (2026-11-13 array ran batches 1-2 only)
#                                             #   pr1:   the scored correction sources (275), pr1 half
#                                             # Records go to $SLURM_TMPDIR; each task writes ONE tar to
#                                             # $Q/fill/rec_<pass>_<k>.tar (Q below). Nothing on /scratch
#                                             # is written.
#   MODE=collect bash submit_qc_on_input.sh   # after fill: one small job writes the side tables
#                                             # $Q/tables/eic_v1_2_qc.tsv, eic_v1_2_qc_pr1.tsv and the
#                                             # fill-rate report $Q/tables/coverage.txt
#   MODE=train bash submit_qc_on_input.sh     # the arm: 3 architectures x 3 seeds, 80,000 windows,
#                                             # tables eic_v1_2, scored on all 256 regions (p256_sv)
#   DRY_RUN=0 MODE=... bash submit_qc_on_input.sh   # real submission -- the foreman's call only
#
# Q = /project/def-maxwl/mforooz/ONION/CO-A4 (same path on Fir and Nibi). PASSES="whole pr1" (default)
# limits MODE=fill to one pass. CHUNK=20 rows per array task; MAXPAR=25 tasks at a time.
# MODE=train: CLUSTER=fir|nibi as submit_cell_embedding.sh; the reference runs
# eic_v1_2_rung{2,3,4}_base_heldout are on Fir, so the arm belongs on Fir (one cluster per comparison);
# the two side tables must then be on Fir at the same path (Globus, the foreman).
set -u -o pipefail
module load python/3.10.13 >/dev/null 2>&1; unset EBPYTHONPREFIXES
O=/project/def-maxwl/mforooz/ONION
Q="${Q:-$O/CO-A4}"
K="${K:-/project/def-maxwl/mforooz/onion_kit/COA4}"
V="${V:-eic_v1_2}"
T=$O/tables/$V
#: the 953 records of the 2026-11-13 array (Nibi jobs 27164529, 27235942, 27332401), copied from
#: /scratch/mforooz/onion/qc into one tar on 2026-11-17 (md5 024964da982ce098ae6160426da50309)
OLD_TAR="${OLD_TAR:-$Q/qc_records_2026-11-13.tar}"
A=$K/tools/minicandi/configs/ablations/qc_on_input
MODE="${MODE:-fill}"
export DRY_RUN="${DRY_RUN:-1}"
CHUNK="${CHUNK:-20}"; MAXPAR="${MAXPAR:-25}"; PASSES="${PASSES:-whole pr1}"
CL="${CLUSTER:-${CC_CLUSTER:-}}"
if [ -z "$CL" ]; then case "$(hostname)" in *nibi*) CL=nibi ;; *fir*) CL=fir ;; esac; fi
echo "A4 QC on input: MODE=$MODE CLUSTER=$CL DRY_RUN=$DRY_RUN kit=$K tables=$V out=$Q"
[ -e "$T/measurements.tsv" ] || { echo "MISSING $T/measurements.tsv" >&2; exit 3; }
[ -f "$K/tools/onion_data/qc_fill.py" ] || { echo "MISSING kit $K (push it: push_kit.sh --cluster $CL COA4)" >&2; exit 3; }
grep -q -- "--half" "$K/tools/onion_data/qc_fill.py" || { echo "REFUSED: kit $K has no qc_fill.py --half (A4 code)" >&2; exit 3; }
EVS=$O/eval/$V/evalset.json
[ -e "$EVS" ] || EVS=$O/eval/eic_v1/evalset.json      # same rows and ids (checked 2026-11-17)

case "$MODE" in
fill)
  [ "$CL" = nibi ] || { echo "REFUSED: the read sets are on Nibi /scratch; MODE=fill runs on nibi" >&2; exit 3; }
  [ -f "$OLD_TAR" ] || { echo "MISSING $OLD_TAR (the 2026-11-13 records; needed to skip their rows)" >&2; exit 3; }
  [ -e "$EVS" ] || { echo "MISSING $EVS" >&2; exit 3; }
  PLAN=$(mktemp -d "${TMPDIR:-/tmp}/a4_plan.XXXXXX")
  # Row lists, chunked, and a check that every read-set file a row needs exists (read only).
  python3 - "$T/measurements.tsv" "$OLD_TAR" "$EVS" "$PLAN" "$CHUNK" <<'PY' || exit 3
import csv, json, os, sys, tarfile
from pathlib import Path
tab, old, evs, plan, chunk = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
QC = ("nsc", "rsc", "nrf", "pbc1", "pbc2", "frag_len")          # informative (frip is never filled)
rows = list(csv.DictReader(open(tab, newline=""), delimiter="\t"))
idx = {r["measurement_id"]: i for i, r in enumerate(rows)}
have = {Path(n).name[:-5] for n in tarfile.open(old).getnames() if n.endswith(".json")}
# rows lacking a QC column, plus every non-base view (its build QC cells are copies of the
# parent's and collect blanks them), unless the row has its own record already
whole = [i for i, r in enumerate(rows) if r["in_eic"] in ("train", "val")
         and r["measurement_id"] not in have
         and (any(not (r.get(c) or "").strip() for c in QC)
              or (r.get("arm_knob") or "").strip() not in ("", "base"))]
ev = json.load(open(evs))
src = sorted({idx[p["source"]] for p in ev["correction_pairs"]})
child = {r["parent_measurement_id"]: r["measurement_id"] for r in rows
         if (r.get("arm_knob") or "") == "nodedup" and r.get("parent_measurement_id")}

def files(mid, halves):
    r = rows[idx[mid]]
    cp = r.get("counts_path") or ""
    if not cp.startswith("h5:"):
        return None, f"counts_path {cp[:20]!r}"
    pf = Path(cp[3:]).parent / "provenance.json"
    if not pf.is_file():
        return None, f"no {pf}"
    p = json.load(open(pf))
    if (p.get("pipeline") or {}).get("name") == "native-redacted-bam":
        return [], "native-redacted-bam (no read-set route; record no_readset, seconds)"
    out = []
    for h in halves:
        f = p.get("halves", {}).get(h, {}).get("tagalign", [])
        out += [f] if isinstance(f, str) else f
    return out, ""

for name, sel, halves in (("whole", whole, ("pr1", "pr2")), ("pr1", src, ("pr1",))):
    need, miss, cheap, gb, notes = 0, [], 0, 0.0, {}
    for i in sel:
        mid = rows[i]["measurement_id"]
        fs, why = files(mid, halves)
        if fs is None:
            notes[why.split(" ")[0]] = notes.get(why.split(" ")[0], 0) + 1
            continue
        if not fs:
            cheap += 1
            continue
        if (rows[i].get("arm_knob") or "") == "" and mid in child:   # base: PBC from the nodedup sibling
            cf, _ = files(child[mid], halves)
            fs = fs + (cf or [])
        for f in fs:
            need += 1
            if os.path.isfile(f):
                gb += os.path.getsize(f) / 1e9
            else:
                miss.append(f)
    with open(plan / f"rows_{name}.txt", "w") as fh:
        for k in range(0, len(sel), chunk):
            part = sel[k:k + chunk]
            fh.write(",".join(str(x) for x in part) + "\n")
    roots = sorted({"/".join(f.split("/")[:5]) for f in miss[:50]})
    print(f"PASS {name}: rows {len(sel)}, chunks {(len(sel) + chunk - 1) // chunk}, "
          f"no-read-set rows {cheap}, other {notes or 0}; read-set files {need} "
          f"({gb:.0f} GB), missing {len(miss)} {roots}")
    print(f"PASS {name} roles: " + ", ".join(f"{k} {sum(rows[i]['in_eic'] == k for i in sel)}"
                                              for k in ("train", "val")))
PY
  for pass in $PASSES; do
    RF=$PLAN/rows_$pass.txt
    n=$(wc -l < "$RF" | tr -d ' ')
    [ "$n" -gt 0 ] || { echo "PASS $pass: nothing to run"; continue; }
    [ ! -e "$Q/fill/rows_$pass.txt" ] || { echo "REFUSED: $Q/fill/rows_$pass.txt exists: pass $pass was submitted already (set Q to a new path)" >&2; exit 3; }
    for k in $(seq 0 $((n - 1))); do
      [ ! -e "$Q/fill/rec_${pass}_${k}.tar" ] || { echo "REFUSED: $Q/fill/rec_${pass}_${k}.tar exists; a pass is never resubmitted (set Q to a new path)" >&2; exit 3; }
    done
    CMD=(sbatch --parsable --job-name=a4_qc_fill_$pass "--array=0-$((n - 1))%$MAXPAR"
         "--output=$Q/fill/logs/${pass}_%A_%a.out"
         $K/slurm/onion_data/qc_fill_chunks.sh $K $T $Q/fill/rows_$pass.txt $Q/fill $pass)
    echo "PASS $pass: $n array tasks of <= $CHUNK rows; rows file $Q/fill/rows_$pass.txt"
    if [ "$DRY_RUN" = 1 ]; then echo "DRY RUN"; printf '%q ' "${CMD[@]}"; echo
    else
      mkdir -p "$Q/fill/logs" && cp "$RF" "$Q/fill/rows_$pass.txt" && "${CMD[@]}"
    fi
  done
  rm -rf "$PLAN"
  ;;
collect)
  [ "$CL" = nibi ] || { echo "REFUSED: collect reads the fill tars on nibi" >&2; exit 3; }
  [ -f "$OLD_TAR" ] || { echo "MISSING $OLD_TAR" >&2; exit 3; }
  for f in "$Q/tables/${V}_qc.tsv" "$Q/tables/${V}_qc_pr1.tsv"; do
    [ ! -e "$f" ] || { echo "REFUSED: $f exists (never overwritten; set Q to a new path)" >&2; exit 3; }
  done
  nw=$(ls "$Q"/fill/rec_whole_*.tar 2>/dev/null | wc -l); np=$(ls "$Q"/fill/rec_pr1_*.tar 2>/dev/null | wc -l)
  echo "fill tars: whole $nw, pr1 $np (rows files: $(cat "$Q/fill/rows_whole.txt" 2>/dev/null | wc -l) and $(cat "$Q/fill/rows_pr1.txt" 2>/dev/null | wc -l) chunks)"
  BODY="set -eu; module load python/3.10.13; virtualenv --no-download \$SLURM_TMPDIR/venv; \
source \$SLURM_TMPDIR/venv/bin/activate; pip install -q --no-index --upgrade pip; \
pip install -q --no-index -r $K/requirements-fir.txt; \
pip install -q --no-index --find-links $K/wheels -r $K/requirements-pypi.txt; \
cd $K; export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1; \
R=\$SLURM_TMPDIR; mkdir -p \$R/old \$R/whole \$R/pr1 $Q/tables; tar -C \$R/old -xf $OLD_TAR; \
for t in $Q/fill/rec_whole_*.tar; do [ -e \"\$t\" ] && tar -C \$R/whole -xf \"\$t\"; done; \
for t in $Q/fill/rec_pr1_*.tar; do [ -e \"\$t\" ] && tar -C \$R/pr1 -xf \"\$t\"; done; \
python tools/onion_data/qc_fill.py collect --tables $T --qc-dir \$R/old --qc-dir \$R/whole \
--out $Q/tables/${V}_qc.tsv; \
python tools/onion_data/qc_fill.py collect --tables $T --half pr1 --qc-dir \$R/pr1 \
--out $Q/tables/${V}_qc_pr1.tsv; \
python tools/onion_data/qc_fill.py coverage --tables $T --qc-table $Q/tables/${V}_qc.tsv \
--missing-rows > $Q/tables/coverage.txt; cat $Q/tables/coverage.txt"
  CMD=(sbatch --account=def-maxwl --job-name=a4_qc_collect --time=00:30:00 --cpus-per-task=1
       --mem=4000M --gres=none "--output=$Q/fill/logs/collect_%j.out" --wrap "$BODY")
  if [ "$DRY_RUN" = 1 ]; then echo "DRY RUN"; printf '%q ' "${CMD[@]}"; echo
  else mkdir -p "$Q/fill/logs" && "${CMD[@]}"; fi
  ;;
train)
  B=$O/eval/$V/baselines
  EV=$O/eval/$V/evalset.json
  case "$CL" in
    fir) ROOT=$O/runs_fir
         export ONION_EXCLUDE=fc10713,fc10715,fc30601,fc30568,fc30605,fc30564,fc30654,fc30435,fc30562,fc30355,fc30570,fc30455,fc30659,fc30565,fc30613,fc30649 ;;
    nibi) ROOT=$O/runs; unset ONION_EXCLUDE
          [ -n "${PRED_STORE_ROOT:-}" ] || { echo "REFUSED: on nibi set PRED_STORE_ROOT explicitly (the default is on /scratch)" >&2; exit 3; } ;;
    *) echo "CLUSTER must be fir or nibi" >&2; exit 2 ;;
  esac
  declare -A ARCH=([2]=convae [3]=convattn [4]=unet)
  miss=()
  for p in "$EV" "$B/oracle_floor/seed0/oracle_floor.json" "$Q/tables/${V}_qc.tsv" "$Q/tables/${V}_qc_pr1.tsv"; do
    [ -e "$p" ] || miss+=("$p")
  done
  for b in marginal_nb position_average depth_only_scaling pseudoreplicate_floor; do
    [ -e "$B/$b/seed0/SCORED" ] || miss+=("$B/$b/seed0/SCORED")
  done
  for r in 2 3 4; do
    c=$A/rung$r/qc_on_input.json
    [ -f "$c" ] || { echo "MISSING config $c (push a kit that holds it)" >&2; exit 3; }
    for key in qc_table qc_table_pr1; do
      p=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$c" "$key")
      [ "$p" = "$Q/tables/${V}_qc.tsv" ] || [ "$p" = "$Q/tables/${V}_qc_pr1.tsv" ] ||
        { echo "REFUSED $c: $key = $p is not under $Q/tables" >&2; exit 3; }
    done
  done
  for p in ${miss[@]+"${miss[@]}"}; do echo "MISSING $p"; done
  for r in 2 3 4; do for s in ${SEEDS:-0 1 2}; do
    p=$ROOT/${V}_rung${r}_base_heldout/seed$s/p256_sv/SCORED
    [ -e "$p" ] || echo "READING NEEDS $p (reference scoring, not submitted here)"
  done; done
  if [ "${#miss[@]}" -gt 0 ] && [ "$DRY_RUN" != 1 ]; then
    echo "REFUSED: ${#miss[@]} prerequisites missing (above)" >&2; exit 3
  fi
  export ONION_STAGE=1 CLUSTER=$CL NICE="${NICE:-100}" SEEDS="${SEEDS:-0 1 2}" TRAIN_TIME=03:00:00
  export SCORE_SAME_VIEW=1 SCORE_PHASES=1 SCORE_SUBDIR=p256_sv SCORE_SHARDS=8 \
    SCORE_PRED_DEVICE=cpu SCORE_PRED_CPUS=8 SCORE_PRED_MEM=32000 SCORE_INFER_BATCH=8 \
    SCORE_PRED_TIME=06:00:00 SCORE_CPUS=16 SCORE_MEM=64000 SCORE_TIME=06:00:00 \
    REFS_FROM=$B KEEP_PRED_STORE=0
  EVA=$EV; [ -e "$EV" ] || { EVA=""; unset SCORE_SAME_VIEW SCORE_PHASES SCORE_SUBDIR REFS_FROM; echo "DRY RUN WITHOUT SCORING: $EV missing; training lines only"; }
  for r in 2 3 4; do
    bash $K/slurm/minicandi/submit_arms.sh $K $T "$EVA" $ROOT $r \
      "${V}_abl_qc_on_input_${ARCH[$r]}=$A/rung$r/qc_on_input.json" 2>&1 | { grep -v "^$" || true; }
    rc=${PIPESTATUS[0]}
    [ "$rc" = 0 ] || { echo "FAILED submit_arms.sh rung $r (exit $rc); stopping" >&2; exit "$rc"; }
  done
  echo "3 architectures x seeds (${SEEDS:-0 1 2}) submitted (or printed); DRY_RUN=$DRY_RUN"
  ;;
*) echo "MODE must be fill, collect or train" >&2; exit 2 ;;
esac
