#!/bin/bash
# t138 regional map (where split-half disagreement concentrates). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Every step skips when its output exists, so a rerun
# after a crash resumes at the first missing output, and a rerun of the finished folder only
# checks MANIFEST.tsv. Frozen: finding f25 cites DIGEST.md. A redo is a new task folder.
#
# Steps (cluster: Nibi; writes only under $T there and only in this folder here):
#   1. annotations   $T/annot/ANNOT.tsv          missing -> sbatch slurm/t138/annot.sh, wait
#   2. map           $T/out/map_check.txt        missing -> sbatch slurm/t138/map.sh, wait
#   3. pull          the small tables + job log  each missing file -> rsync from Nibi
#   4. manifest      MANIFEST.tsv                missing -> hash every output (local + Nibi)
#   5. check         always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# A step that must submit a job first builds the code kit at $KIT from git commit $KIT_SHA
# (the commit the recorded run used), so the rerun runs the recorded code, not today's tree.
# Inputs owned by other tasks are frozen there: the 1 kb bins (t137) and the products (t112).
# If they are gone, rerun that task first. Paths match the recorded submit line of job 25402043. Hand-written files (DIGEST.md, NIBI_PATH.txt) are not outputs.
# ssh exit 69 or 255 means the tunnel is down: run `hpc up nibi`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
REPO=$(cd ../../.. && pwd)

H=nibi
T=/project/def-maxwl/mforooz/t138
KIT_SHA=6743ab64644aaf16e5c1c222b8396bf7726bb233     # recorded run: kit t137/code/K2 at this sha
KIT=$T/kit
ANN=$T/annot
NP=/project/def-maxwl/mforooz/EIC_REPRO/003_pipeline/results
BINS=/project/def-maxwl/mforooz/t137/floor/bins1k
PRODUCTS=/project/def-maxwl/mforooz/t112_cf/products   # holds the product MANIFEST.tsv (t112)
BL=/project/def-maxwl/mforooz/EIC_REPRO/002/scripts/hg38_blacklist_v2.bed
OUT=$T/out
TABLES=(map_products.tsv map_classes.tsv map_context.tsv map_deciles.tsv map_check.txt
        config.json features.json)
REMOTE_OUTPUTS=("$OUT/features.npz")

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

kit() {  # build $KIT from git at $KIT_SHA unless it is already there at that sha
  if remote "test \"\$(cat $KIT/GIT_SHA 2>/dev/null)\" = $KIT_SHA"; then return 0; fi
  echo "building kit $KIT at $KIT_SHA"
  git -C "$REPO" archive "$KIT_SHA" tools slurm src pyproject.toml \
    | remote "rm -rf $KIT.tmp && mkdir -p $KIT.tmp && tar -x -C $KIT.tmp && echo $KIT_SHA > $KIT.tmp/GIT_SHA && rm -rf $KIT && mv $KIT.tmp $KIT"
}

submit_wait() {  # submit_wait <slurm script> <args...>; submits, polls sacct, fails on non-COMPLETED
  local id s
  remote "mkdir -p $T/logs"
  id=$(remote "sbatch --parsable --output=$T/logs/%x_%j.out $KIT/$1 ${*:2}")
  [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch $1 gave '$id'" >&2; exit 1; }
  echo "$(date -u +%FT%TZ) submitted $1 as job $id"
  while :; do
    s=$(remote "sacct -j $id -X -n -P -o State" | head -1 | tr -d ' ')
    case "$s" in
      COMPLETED) echo "job $id COMPLETED"; return 0 ;;
      PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") sleep 60 ;;
      *) echo "job $id ended $s; log $T/logs/" >&2; exit 1 ;;
    esac
  done
}

# 1. annotations
if remote "test -s $ANN/ANNOT.tsv"; then echo "1 annot: have $ANN/ANNOT.tsv"
else kit; submit_wait slurm/t138/annot.sh "$KIT" "$ANN"; fi

# 2. map
if remote "test -s $OUT/map_check.txt"; then echo "2 map: have $OUT/map_check.txt"
else
  remote "test -d $BINS && test -s $PRODUCTS/MANIFEST.tsv" \
    || { echo "inputs missing: $BINS (t137) or $PRODUCTS/MANIFEST.tsv (t112); rerun that task" >&2; exit 1; }
  kit; submit_wait slurm/t138/map.sh "$KIT" "$ANN" "$NP" "$BINS" "$PRODUCTS" "$OUT" "$BL"
fi

# 3. pull
for f in "${TABLES[@]}"; do
  if [ -s "$f" ]; then continue; fi
  rsync -a -e 'ssh -o BatchMode=yes' "$H:$OUT/$f" . || { echo "rsync $f failed" >&2; exit 1; }
  echo "3 pull: $f"
done
if ! ls t138_map_*.out >/dev/null 2>&1; then
  log=$(remote "ls -t $T/logs/t138_map_*.out | head -1")
  rsync -a -e 'ssh -o BatchMode=yes' "$H:$log" . || { echo "rsync $log failed" >&2; exit 1; }
  echo "3 pull: $(basename "$log")"
fi

# 4. manifest
if [ ! -s MANIFEST.tsv ]; then
  {
    printf 'path\tbytes\tsha256\twhere\n'
    for f in "${TABLES[@]}" t138_map_*.out; do
      printf '%s\t%s\t%s\tgit\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(sha "$f")"
    done
    for p in "${REMOTE_OUTPUTS[@]}"; do
      read -r b s < <(remote "printf '%s ' \$(stat -c %s $p); sha256sum $p | cut -d' ' -f1")
      printf '%s\t%s\t%s\t%s:%s\n' "$(basename "$p")" "$b" "$s" "$H" "$p"
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
