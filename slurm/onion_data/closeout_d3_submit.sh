#!/bin/bash
# Close-out D3 (peeling the onion): the dataset update that brings the DNase-seq tail rows in.
# Item file: cruxvault/INSTRUCTIONS/peeling_the_onion/closeout/closeout D3 dataset update with DNase-seq.md
# Dry run is the default. Runs on a Nibi login node:
#
#   ssh -o BatchMode=yes nibi 'bash -s -- preserve verify'          < slurm/onion_data/closeout_d3_submit.sh
#   ssh -o BatchMode=yes nibi 'DRY_RUN=0 bash -s -- preserve verify' < slurm/onion_data/closeout_d3_submit.sh
#
# Parts, in their order of use (give one or more):
#   preserve   snapshot of the small work-folder files the table build reads (QC files, READSET.json,
#              NOT_APPLICABLE), then a check that the snapshot alone gives what the disk gives.
#              MUST run before /scratch/mforooz/onion/work leaves the disk. 1 job, 1 CPU, < 1 CPU-h.
#   verify     `align_views.py verify --recount-chrom chr21` on the experiments that verified after
#              the eic_v1 freeze (the D6u step 2 check). It reads the half tagAlign files in the work
#              folder, so it MUST also run before the work folder leaves. Array, 1 CPU per experiment.
#   candidate  a table build with --superset-of eic_v1 on what is verified now (work folder on the
#              disk, no snapshot), and its eval set, under $D3/ (not a version; nothing else reads it).
#              It proves the superset rule and gives the scorable list. 1 job, 16 CPUs, about 6 CPU-h.
#   reruns     the 4 tail rows D3 owns, once each, 7 days (rule B.5), from kit KF4: row 56 (validation
#              DNase-seq; failed on the full /scratch, not on its own cause), 132 (Java heap), 280 (time
#              limit), 396 (out of memory at 192000M). The 7 NODE_FAIL rows are D4's (its script); never
#              both. Needs a writable /scratch. Up to 4 x 16 CPUs x 168 h = 10,752 CPU-h.
#   version    the new version $VERSION: staging build --superset-of eic_v1 (with the snapshot when the
#              work folder is gone), fill_covariates join + apply (the D1 columns), check_dataset
#              against eic_v1_2 and eic_v1, eval set (256 regions) + 16-pair-region set + checks +
#              regions assertion + oracle floor on both. 1 job, 16 CPUs, at most 4 h (64 CPU-h).
#   baselines  the 4 baselines on both eval sets of $VERSION (8 jobs, score.sh, 16 CPUs each).
#
# Env: DRY_RUN (default 1: print what would run, submit nothing), KIT_ID (default COD3),
#      VERSION (default eic_v1_3), EXPS (verify: comma list, default the 10 experiments below),
#      FORCE=1 (version: build although tail rows are queued or running).
set -uo pipefail
DRY_RUN="${DRY_RUN:-1}"; KIT_ID="${KIT_ID:-COD3}"; VERSION="${VERSION:-eic_v1_3}"; FORCE="${FORCE:-0}"
PARTS="${*:-}"
[ -n "$PARTS" ] || { echo "usage: closeout_d3_submit.sh preserve|verify|candidate|reruns|version|baselines ..."; exit 2; }
for p in $PARTS; do case "$p" in preserve|verify|candidate|reruns|version|baselines) ;;
  *) echo "unknown part '$p'"; exit 2;; esac; done

O=/project/def-maxwl/mforooz/ONION
D3=$O/CO-D3                                   # this item's own folder (new paths only)
KIT=/project/def-maxwl/mforooz/onion_kit/$KIT_ID
KF4=/project/def-maxwl/mforooz/onion_kit/KF4
B=$O/eic/build_list.tsv
M=/scratch/mforooz/onion/measurements; W=/scratch/mforooz/onion/work
SNAP=$D3/work_snapshot_2026-11-17.json.gz
B_SHA=33c077f464138c57cbe31827857061b1524a22ed571d00a7ab8b7de81ddc0b83
#: experiments built and verified after the eic_v1 freeze (TAIL_STATUS.tsv of 2026-11-17T08:13Z; rows
#: 52, 57, 337, 381, 388, 389, 390, 391, 394, 395); the other 7 verified tail rows are in eic_v1
NEW_EXPS="ENCSR521AIV,ENCSR489NAM,ENCSR731QLJ,ENCSR237WJY,ENCSR184LMY,ENCSR620QNS,ENCSR083STA,ENCSR000FEK,ENCSR000ENI,ENCSR000EQB"
EXPS="${EXPS:-$NEW_EXPS}"
#: D3's reruns: priority|time|mem|rows|label (rule B.5; 192000M also gives Picard -Xmx58G)
RERUNS=(
"1|7-00:00:00|192000M|56|validation DNase-seq C50M02: its 7-day task 27124285_56 failed in 7 s on 'Disk quota exceeded'"
"2|7-00:00:00|192000M|132|training DNase-seq C18M02: Java heap (27081354_132)"
"2|7-00:00:00||280|training DNase-seq C02M02: 3-day limit (26855582_280); 774.8 M reads, under 900 M"
"2|7-00:00:00|384000M|396|training DNase-seq C15M02: MaxRSS 196,581,636 KB = its 192000M limit (27105143_396)"
)
D4_ROWS="113,211,216,232,384,385,387"         # D4's outage reruns: watched here, never submitted

die() { echo "REFUSED: $*"; exit 3; }
case "$(hostname)" in *nibi*) ;; *) die "run this on a Nibi login node";; esac
[ -f "$KIT/.kit_stamp" ] || die "no kit $KIT (push it: bash tools/onion_kit/push_kit.sh --cluster nibi $KIT_ID)"
echo "=== closeout D3 $(date -u +%FT%TZ) parts: $PARTS  DRY_RUN=$DRY_RUN  kit: $(cat "$KIT/.kit_stamp")"
[ "$(sha256sum "$B" | cut -d' ' -f1)" = "$B_SHA" ] || die "build_list.tsv changed"

run() {   # print a command; run it unless dry
  echo "    \$ $*"
  [ "$DRY_RUN" = 0 ] && "$@"
  return 0
}
sb() {    # sbatch through the kit's check.sh: <name> <time> <cpus> <mem> <python args...>
  local name="$1" t="$2" c="$3" m="$4"; shift 4
  run sbatch --parsable --account=def-maxwl --job-name="onion_d3_$name" --time="$t" \
    --cpus-per-task="$c" --mem="$m" --gres=none --output="$D3/logs/%x_%j.out" \
    "$KIT/slurm/onion_kit/check.sh" "$KIT" "$@"
}
scratch_writable() { local f="/scratch/mforooz/onion/.d3_write_test_$$"
  ( : > "$f" ) 2>/dev/null && rm -f "$f"; }
active_rows() {   # build-list rows of $B with a queued or running onion_* task
  squeue -u "$USER" -h -r -n onion_build,onion_signal,onion_stage -o '%K' 2>/dev/null \
    | grep -E '^[0-9]+$' | sort -un | paste -sd, -; }

[ "$DRY_RUN" = 0 ] && mkdir -p "$D3/logs"

# ------------------------------------------------------------------ preserve
PY_PRESERVE="
import subprocess, sys
bt = 'tools/onion_data/build_tables.py'
r = subprocess.run([sys.executable, bt, 'snapshot-work', '--work', '$W', '--out', '$SNAP'])
if r.returncode: sys.exit(r.returncode)
sys.exit(subprocess.run([sys.executable, bt, 'check-work-snapshot', '--snapshot', '$SNAP', '--work', '$W',
                         '--measurements', '$M', '--build-list', '$B']).returncode)
"
case " $PARTS " in *" preserve "*)
  echo "--- preserve: $SNAP"
  if [ -e "$SNAP" ]; then echo "    exists: $(ls -l "$SNAP")"
  elif [ ! -d "$W" ]; then die "$W is gone: restore it from nearline first (NIBI_NEARLINE_ARCHIVE.md)"
  else sb preserve 01:00:00 1 8000M -c "$PY_PRESERVE"; fi ;;
esac

# ------------------------------------------------------------------ verify
case " $PARTS " in *" verify "*)
  echo "--- verify (chr21 recount) on: $EXPS"
  [ -d "$W" ] || die "$W is gone: the half tagAligns are on nearline only"
  IFS=, read -r -a XS <<< "$EXPS"
  for x in "${XS[@]}"; do
    sb "verify_$x" 03:00:00 1 8000M tools/onion_data/align_views.py verify --measurements "$M" \
      --recount-chrom chr21 --experiment "$x"
  done ;;
esac

# ------------------------------------------------------------------ candidate
CAND=$D3/tables/${VERSION}_candidate; CEV=$D3/eval/${VERSION}_candidate
PY_SUMMARY='
import json, sys
from collections import Counter
for name in sys.argv[1:]:
    ev = json.load(open(name))
    imp = [t for t in ev["imputation_targets"] if t["split"] == "val"]
    loao = {(t["cell"], t["assay"]) for t in imp if t["regime"] == "leave_one_assay_out"}
    acc = {(t["cell"], t["assay"]) for t in imp if t["regime"] == "accessibility_only"}
    no = sorted(loao - acc)
    print(name, "imputation-scorable:", ",".join(ev["scorable"]["imputation"]))
    print(name, "correction-scorable:", ",".join(ev["scorable"]["correction"]))
    print(name, f"validation targets with no accessibility-only item: {len(no)} of {len(loao)}:",
          ", ".join(f"{c} {a}" for c, a in no))
    print(name, "validation targets per assay:", dict(sorted(Counter(a for _, a in loao).items())))
'
PY_CANDIDATE="
import subprocess, sys, json, hashlib, os
def run(*a):
    print('+', ' '.join(a), flush=True)
    r = subprocess.run([sys.executable, *a])
    if r.returncode: sys.exit(r.returncode)
run('tools/onion_data/build_tables.py', 'build', '--build-list', '$B', '--priorities', '1,2,3',
    '--measurements', '$M', '--work', '$W', '--testbed-rows', '$O/eic/testbed_rows.tsv',
    '--withheld', '$O/tables/eic_v1/withheld.tsv', '--ledger', '$O/eic/accession_ledger.tsv',
    '--recount-chrom', 'none', '--superset-of', '$O/tables/eic_v1', '--out', '$CAND')
os.makedirs('$CEV', exist_ok=False)
run('tools/onion_eval/evalset.py', 'build', '--tables', '$CAND', '--strata', '$O/eval/strata/masks.h5',
    '--n-regions', '256', '--n-fit-regions', '16', '--seed', '0', '--out', '$CEV/evalset.json')
run('tools/onion_eval/evalset.py', 'check', '--evalset', '$CEV/evalset.json', '--tables', '$CAND')
a, b = json.load(open('$O/eval/regions_v1/regions.json')), json.load(open('$CEV/evalset.json'))
bad = [k for k in ('regions', 'fit_regions', 'region_bins') if a[k] != b[k]]
print('regions equal to regions_v1:', not bad, bad)
run('-c', '''$PY_SUMMARY''', '$O/eval/eic_v1/evalset.json', '$CEV/evalset.json')
print('evalset sha256', hashlib.sha256(open('$CEV/evalset.json', 'rb').read()).hexdigest())
sys.exit(1 if bad else 0)
"
case " $PARTS " in *" candidate "*)
  echo "--- candidate: $CAND and $CEV (not a version)"
  [ -e "$CAND" ] && die "$CAND exists"
  [ -d "$W" ] || [ -e "$SNAP" ] || die "neither the work folder nor the snapshot"
  [ -d "$W" ] || die "the candidate reads the work folder itself; use part 'version' with the snapshot"
  # evalset.genome_dir(tables) = tables/../../genome: a link in $D3 points it at $O/genome
  [ -e "$D3/genome" ] || run ln -s "$O/genome" "$D3/genome"
  sb candidate 01:00:00 16 64000M -c "$PY_CANDIDATE" ;;
esac

# ------------------------------------------------------------------ reruns
case " $PARTS " in *" reruns "*)
  ACTIVE=$(active_rows)
  echo "--- reruns (kit KF4); queued or running onion_* tasks: ${ACTIVE:-none}"
  grep -q "git=0d578dc" "$KF4/.kit_stamp" || die "kit KF4 is not git 0d578dc"
  scratch_writable || die "/scratch refuses writes (quota); the reruns stage FASTQs and write work folders there"
  [ ! -e "$D3/reruns_submitted.tsv" ] || [ "$FORCE" = 1 ] \
    || die "$D3/reruns_submitted.tsv exists: rule B.5 allows one 7-day rerun per row (stop rule)"
  for c in "${RERUNS[@]}"; do
    IFS='|' read -r PRI TIME MEM ROWS LABEL <<< "$c"
    echo "    row $ROWS (priority $PRI, $TIME, mem ${MEM:-64000M header}): $LABEL"
    if [ "$DRY_RUN" != 0 ]; then
      python3 "$KF4/tools/onion_data/build_list.py" pending --build-list "$B" --priority "$PRI" --rows "$ROWS" \
        --out "$M" --work "$W" --files "$O/catalog/encode/files.tsv" ${ACTIVE:+--active "$ACTIVE"} 2>&1 >/dev/null \
        | sed 's/^/      /'
      env ONION_SBATCH_TEST=1 ONION_TIME="$TIME" ${MEM:+ONION_MEM=$MEM} \
        bash "$KF4/slurm/onion_data/build_row.sh" submit "$KF4" "$B" "$PRI" 50 2 "$ROWS" 2>/dev/null | sed 's/^/      /'
    else
      out=$(env ONION_TIME="$TIME" ${MEM:+ONION_MEM=$MEM} \
              bash "$KF4/slurm/onion_data/build_row.sh" submit "$KF4" "$B" "$PRI" 50 2 "$ROWS" 2>&1)
      echo "$out" | sed 's/^/      /'
      printf '%s\t%s\t%s\t%s\t%s\n' "$ROWS" "$(date -u +%FT%TZ)" "$TIME" "${MEM:-}" \
        "$(echo "$out" | grep -o 'array [0-9]*' | paste -sd' ' -)" >> "$D3/reruns_submitted.tsv"
    fi
  done ;;
esac

# ------------------------------------------------------------------ version
VT=$O/tables/$VERSION; VE=$O/eval/$VERSION; STG=$D3/staging/${VERSION}_counts
PY_VERSION="
import subprocess, sys, json, hashlib, os
def run(*a):
    print('+', ' '.join(a), flush=True)
    r = subprocess.run([sys.executable, *a])
    if r.returncode: sys.exit(r.returncode)
snap = ['--work-snapshot', '$SNAP'] if os.path.exists('$SNAP') else []
run('tools/onion_data/build_tables.py', 'build', '--build-list', '$B', '--priorities', '1,2,3',
    '--measurements', '$M', '--work', '$W', '--testbed-rows', '$O/eic/testbed_rows.tsv',
    '--withheld', '$O/tables/eic_v1/withheld.tsv', '--ledger', '$O/eic/accession_ledger.tsv',
    '--recount-chrom', 'none', '--superset-of', '$O/tables/eic_v1', '--out', '$STG', *snap)
run('tools/onion_data/fill_covariates.py', 'join', '--tables', '$STG', '--catalog', '$O/catalog/encode',
    '--out', '$D3/side_${VERSION}.tsv')
run('tools/onion_data/fill_covariates.py', 'apply', '--base', '$STG', '--side', '$D3/side_${VERSION}.tsv',
    '--out', '$VT')
for old in ('eic_v1_2', 'eic_v1'):
    run('tools/onion_data/check_dataset.py', '--tables', '$VT', '--superset-of', '$O/tables/' + old,
        '--covariate-columns', 'lab,antibody,donor,platform', '--workers', '16')
os.makedirs('$VE', exist_ok=False)
run('tools/onion_eval/evalset.py', 'build', '--tables', '$VT', '--strata', '$O/eval/strata/masks.h5',
    '--n-regions', '256', '--n-fit-regions', '16', '--seed', '0', '--out', '$VE/evalset.json')
run('tools/onion_eval/evalset.py', 'derive', '--base', '$VE/evalset.json', '--pair-regions', '16',
    '--out', '$VE/evalset_p16.json')
for f in ('evalset.json', 'evalset_p16.json'):
    run('tools/onion_eval/evalset.py', 'check', '--evalset', '$VE/' + f, '--tables', '$VT')
a, b = json.load(open('$O/eval/regions_v1/regions.json')), json.load(open('$VE/evalset.json'))
bad = [k for k in ('regions', 'fit_regions', 'region_bins') if a[k] != b[k]]
print('regions equal to regions_v1:', not bad, bad)
if bad: sys.exit(1)
run('tools/onion_eval/oracle_floor.py', 'compute', '--evalset', '$VE/evalset_p16.json',
    '--out', '$VE/baselines_p16/oracle_floor/seed0')
run('tools/onion_eval/oracle_floor.py', 'compute', '--evalset', '$VE/evalset.json',
    '--out', '$VE/baselines/oracle_floor/seed0')
run('-c', '''$PY_SUMMARY''', '$O/eval/eic_v1/evalset.json', '$VE/evalset.json')
sys.path.insert(0, 'tools'); from minicandi.dataset import tables_sha256
print('tables_sha256', tables_sha256('$VT'))
for f in ('evalset.json', 'evalset_p16.json'):
    print(f, 'sha256', hashlib.sha256(open('$VE/' + f, 'rb').read()).hexdigest())
"
case " $PARTS " in *" version "*)
  ACTIVE=$(active_rows)
  echo "--- version $VERSION: staging $STG, tables $VT, eval $VE"
  echo "    snapshot: $([ -e "$SNAP" ] && echo "$SNAP" || echo 'none (the work folder must be on the disk)')"
  echo "    queued or running onion_* tasks: ${ACTIVE:-none}"
  for t in "$VT" "$VE" "$STG"; do [ -e "$t" ] && die "$t exists (a version is never overwritten)"; done
  [ -e "$SNAP" ] || [ -d "$W" ] || die "neither the work folder nor the snapshot: run 'preserve' first"
  if [ -n "$ACTIVE" ] && [ "$FORCE" != 1 ]; then
    left=$(python3 - "56,132,280,396,$D4_ROWS" "$ACTIVE" <<'EOF'
import sys
def p(s):
    out = set()
    for t in s.split(","):
        a, _, b = t.partition("-"); out.update(range(int(a), int(b or a) + 1))
    return out
print(",".join(map(str, sorted(p(sys.argv[1]) & p(sys.argv[2])))))
EOF
)
    [ -z "$left" ] || die "tail rows still queued or running: $left (FORCE=1 builds with what is verified)"
  fi
  sb version 04:00:00 16 64000M -c "$PY_VERSION" ;;
esac

# ------------------------------------------------------------------ baselines
case " $PARTS " in *" baselines "*)
  echo "--- baselines on $VE"
  for spec in "evalset_p16.json|baselines_p16|03:00:00" "evalset.json|baselines|12:00:00"; do
    IFS='|' read -r EVF SUB T <<< "$spec"
    [ -f "$VE/$EVF" ] || [ "$DRY_RUN" != 0 ] || die "no $VE/$EVF: run part 'version' first"
    for bl in marginal_nb position_average depth_only_scaling pseudoreplicate_floor; do
      run sbatch --parsable --account=def-maxwl --job-name="onion_score_${VERSION}_${SUB#baselines}_$bl" \
        --time="$T" --cpus-per-task=16 --mem=64000M --gres=none --output="$O/logs/%x_%j.out" \
        "$KIT/slurm/onion_eval/score.sh" "$KIT" "$VE/$EVF" "$VE/$SUB/$bl/seed0" baseline "$bl"
    done
  done ;;
esac
echo "=== done. Dry run: $([ "$DRY_RUN" = 0 ] && echo no || echo 'yes (nothing submitted)')"
