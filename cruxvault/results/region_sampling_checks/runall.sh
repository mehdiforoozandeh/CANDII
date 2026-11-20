#!/bin/bash
# Region-sampling checks of the ablations (close-out item CO-REG) and the amended 16-versus-256
# pair-region check (close-out item E2, PAIR_REGIONS_V3.md). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Every step skips when its output exists, so a rerun of
# the finished folder only checks MANIFEST.tsv. Frozen: findings cite REGION_CHECKS.md. A redo is
# a new task folder.
#
# Steps (cluster: Fir; this script writes only in this folder):
#   1. signal     $RC/signal_v2/region_signal_summary.tsv   missing -> print the recorded command, stop
#   2. bootstrap  $RC/bootstrap_v2/bootstrap_effects.tsv    missing -> print the recorded command, stop
#   3. pull       each missing table of this folder          -> rsync from Fir
#   4. manifest   MANIFEST.tsv                               missing -> hash every output
#   5. check      always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# Steps 1 and 2 are the recorded runs of REGION_CHECKS.md (code tools/onion_eval/region_checks.py
# at $CODE_SHA, kit CO6, venv candi_venv). The script does not submit them: their inputs (the
# scored ablation runs under runs_fir) are owned by the ablation runs. check_v3.tsv and
# PAIR_REGIONS_V3.md are outputs of `pair_regions_check.py run-v3`; their Fir directory is not
# recorded in this folder, so a missing copy stops the script and names the command.
# Hand-written files (REGION_CHECKS.md) are not outputs.
# ssh exit 69 or 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=fir
CODE_SHA=c394eeef      # CO-REG review fixes: the v2 runs carry them (REGION_CHECKS.md, "Code")
K=/project/def-maxwl/mforooz/onion_kit/CO6
VENV=/project/def-maxwl/mforooz/candi_venv
RC=/project/def-maxwl/mforooz/ONION/region_checks
EVALSET=/project/def-maxwl/mforooz/ONION/eval/eic_v1/evalset_p16.json
RUNS=/project/def-maxwl/mforooz/ONION/runs_fir
SIGNAL=(region_signal_eval256.tsv region_signal_summary.tsv signal_tracks.tsv)
BOOT=(reproduction.tsv bootstrap_effects.tsv bootstrap_base_floor.tsv loro_regions.tsv)
V3=(check_v3.tsv PAIR_REGIONS_V3.md)

remote() {  # remote <command>; stops the script on a dead tunnel
  local rc=0
  ssh -o BatchMode=yes "$H" "$1" || rc=$?
  if [ "$rc" = 69 ] || [ "$rc" = 255 ]; then
    echo "ssh $H exited $rc: the tunnel is down. Run 'hpc up $H', then rerun." >&2
    exit "$rc"
  fi
  return "$rc"
}

sha() {  # sha <file> -> hex digest (macOS shasum or GNU sha256sum)
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

missing() {  # missing <file>...; true when any is absent here
  local f
  for f in "$@"; do [ -s "$f" ] || return 0; done
  return 1
}

run_cmd() {  # run_cmd <subcommand args>; the recorded command line of region_checks.py
  echo "source $VENV/bin/activate && PYTHONPATH=$K/tools:$K/src python $K/tools/onion_eval/region_checks.py $*"
}

# 1. signal
if missing "${SIGNAL[@]}"; then
  if remote "test -s $RC/signal_v2/region_signal_summary.tsv"; then echo "1 signal: have $RC/signal_v2"
  else echo "1 signal: $RC/signal_v2 is missing; on Fir run (kit $K at $CODE_SHA):" >&2
       run_cmd "signal --evalset $EVALSET --out $RC/signal_v2" >&2; exit 1; fi
else echo "1 signal: have the local copies"; fi

# 2. bootstrap
if missing "${BOOT[@]}"; then
  if remote "test -s $RC/bootstrap_v2/bootstrap_effects.tsv"; then echo "2 bootstrap: have $RC/bootstrap_v2"
  else echo "2 bootstrap: $RC/bootstrap_v2 is missing; on Fir run (kit $K at $CODE_SHA):" >&2
       run_cmd "bootstrap --runs $RUNS --out $RC/bootstrap_v2 --n-boot 5000" >&2; exit 1; fi
else echo "2 bootstrap: have the local copies"; fi

# 3. pull
for f in "${SIGNAL[@]}"; do
  [ -s "$f" ] && continue
  rsync -a -e 'ssh -o BatchMode=yes' "$H:$RC/signal_v2/$f" . || { echo "rsync $f failed" >&2; exit 1; }
  echo "3 pull: $f"
done
for f in "${BOOT[@]}"; do
  [ -s "$f" ] && continue
  rsync -a -e 'ssh -o BatchMode=yes' "$H:$RC/bootstrap_v2/$f" . || { echo "rsync $f failed" >&2; exit 1; }
  echo "3 pull: $f"
done
if missing "${V3[@]}"; then
  echo "3 pull: ${V3[*]} missing; rerun 'python tools/onion_eval/pair_regions_check.py run-v3'" \
       "on the 4 arms named in PAIR_REGIONS_V3.md (close-out item E2) and copy its --out here" >&2
  exit 1
fi

# 4. manifest
if [ ! -s MANIFEST.tsv ]; then
  {
    printf 'path\tbytes\tsha256\twhere\n'
    for f in "${SIGNAL[@]}" "${BOOT[@]}" "${V3[@]}"; do
      printf '%s\t%s\t%s\tgit\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(sha "$f")"
    done
  } > MANIFEST.tsv.tmp
  mv MANIFEST.tsv.tmp MANIFEST.tsv
  echo "4 manifest: wrote MANIFEST.tsv"
fi

# 5. check
bad=0; n=0
while IFS=$'\t' read -r path bytes want where; do
  [ "$path" = path ] && continue
  n=$((n + 1))
  if [ "$where" = git ]; then got=$(sha "$path")
  else got=$(remote "sha256sum ${where#*:} | cut -d' ' -f1"); fi
  if [ "$got" != "$want" ]; then echo "MISMATCH $path ($where)"; bad=$((bad + 1)); fi
done < MANIFEST.tsv
echo "5 check: $((n - bad)) of $n outputs match MANIFEST.tsv"
[ "$bad" = 0 ]
