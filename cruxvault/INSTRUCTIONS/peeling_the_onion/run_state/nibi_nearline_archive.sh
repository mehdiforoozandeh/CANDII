#!/bin/bash
# Archive four Nibi /scratch folders to /nearline as tar files (PI approved 2026-11-17).
# Run on a Nibi LOGIN node (compute nodes do not mount /nearline), inside tmux.
#   bash nibi_nearline_archive.sh plan      # write the member lists (one list = one tar), read-only on scratch
#   bash nibi_nearline_archive.sh run [P]   # tar + byte-compare each list, P parallel workers (default 3)
#   bash nibi_nearline_archive.sh status    # one line per tar
# It NEVER deletes anything. A tar with OK in status.tsv was compared byte for byte with scratch (tar -d).
# Restore one tar:  tar -xf $DEST/<name>.tar -C /scratch/mforooz     (paths in the tar are relative to /scratch/mforooz)
set -u
SRC=/scratch/mforooz
DEST=/nearline/def-maxwl/mforooz/scratch_archive_2026-11-17
STATE=/project/def-maxwl/mforooz/ONION/nearline_archive_2026-11-17
ROOTS="t112_cf onion/work onion/fastq EIC_REPRO"
BIN_MB=$((200*1024))      # target size of one tar
SPLIT_MB=$((300*1024))    # a child folder larger than this is split one level deeper
mkdir -p "$STATE/lists" "$STATE/index" "$STATE/logs"

flush() {  # $1 = prefix ; uses globals cur (temp list), cur_mb, k
  [ -s "$cur" ] || return 0
  local name; name=$(printf '%s__%03d' "$1" "$k")
  mv "$cur" "$STATE/lists/$name.list"; printf '%s\t%s\n' "$name" "$cur_mb" >> "$STATE/plan.tsv"
  k=$((k+1)); cur_mb=0; cur=$(mktemp "$STATE/lists/.cur.XXXX")
}

plan_dir() {  # $1 = dir relative to SRC
  local dir=$1 prefix; prefix=$(echo "$dir" | tr '/' '-')
  local k=0 cur_mb=0 cur; cur=$(mktemp "$STATE/lists/.cur.XXXX")
  local big=()
  while IFS=$'\t' read -r mb path; do
    rel=${path#"$SRC"/}
    if [ -d "$path" ] && [ ! -L "$path" ] && [ "$mb" -gt "$SPLIT_MB" ]; then big+=("$rel"); continue; fi
    printf '%s\n' "$rel" >> "$cur"; cur_mb=$((cur_mb+mb))
    [ "$cur_mb" -ge "$BIN_MB" ] && flush "$prefix"
  done < <(find "$SRC/$dir" -mindepth 1 -maxdepth 1 -print0 | sort -z | xargs -0 -r du -s --block-size=1M)
  flush "$prefix"; rm -f "$cur"
  local b; for b in ${big[@]+"${big[@]}"}; do plan_dir "$b"; done
}

one() {  # $1 = tar name
  local name=$1 list="$STATE/lists/$1.list" tarf="$DEST/$1.tar" log="$STATE/logs/$1.log"
  grep -q "^$name	OK	" "$STATE/status.tsv" 2>/dev/null && return 0
  mkdir -p "$DEST"
  {
    echo "start $(date -u +%FT%TZ)"
    nice -n 19 ionice -c3 tar -cf "$tarf" -C "$SRC" -T "$list" || { echo "TAR FAILED"; printf '%s\tFAIL_TAR\t0\t0\n' "$name" >> "$STATE/status.tsv"; return 1; }
    nice -n 19 ionice -c3 tar -df "$tarf" -C "$SRC" > "$STATE/logs/$name.diff" 2>&1; rc=$?
    tar -tvf "$tarf" > "$STATE/index/$name.txt"
    n=$(wc -l < "$STATE/index/$name.txt"); sz=$(stat -c %s "$tarf")
    if [ $rc -eq 0 ] && [ ! -s "$STATE/logs/$name.diff" ]; then st=OK; else st=FAIL_COMPARE; fi
    printf '%s\t%s\t%s\t%s\n' "$name" "$st" "$n" "$sz" >> "$STATE/status.tsv"
    echo "end $(date -u +%FT%TZ) $st entries=$n bytes=$sz"
  } >> "$log" 2>&1
}

case "${1:-}" in
  plan)
    [ -e "$STATE/plan.tsv" ] && { echo "plan exists: $STATE/plan.tsv"; exit 1; }
    for r in $ROOTS; do plan_dir "$r"; done
    rm -f "$STATE"/lists/.cur.*
    echo "tars: $(wc -l < "$STATE/plan.tsv")  total GiB: $(awk -F'\t' '{s+=$2} END {printf "%.0f", s/1024}' "$STATE/plan.tsv")"
    ;;
  run)
    P=${2:-3}; export -f one; export SRC DEST STATE
    cut -f1 "$STATE/plan.tsv" | xargs -P "$P" -I{} bash -c 'one {}'
    echo "ALL DONE $(date -u +%FT%TZ)" >> "$STATE/logs/run.log"
    ;;
  status)
    echo "planned: $(wc -l < "$STATE/plan.tsv")  OK: $(grep -c '	OK	' "$STATE/status.tsv" 2>/dev/null)  failed: $(grep -c FAIL "$STATE/status.tsv" 2>/dev/null)"
    ;;
  *) sed -n '2,9p' "$0";;
esac
