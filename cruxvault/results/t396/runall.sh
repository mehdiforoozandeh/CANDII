#!/bin/bash
# t396 (read start vs read count, dedup and SE/PE caps): rebuilds this folder.
#
#   bash runall.sh
#
# No arguments, no environment variables. Generic over the analysis subfolders: every analysis
# owns <name>/ with run.sh (no arguments, every step skipped when its output exists), OUTPUTS.txt
# (one relative path per line, the outputs of run.sh) and one hand-written <NAME>.md (not an
# output). Frozen once a finding cites it; a redo is a new task folder (AGENTS.md section 4.11).
#
# Steps:
#   1. run        bash <name>/run.sh for each subfolder that has one (in name order); a failing
#                 run.sh stops runall with its exit code
#   2. manifest   MANIFEST.tsv missing -> one row per <name>/run.sh and per OUTPUTS.txt line
#                 (path relative to this folder, bytes, sha256, where = git)
#   3. check      always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# Outputs listed in an OUTPUTS.txt but absent from an existing MANIFEST.tsv are named on a NOTE
# line (the manifest is regenerated once, at close-out, by deleting it and rerunning).
# A row whose `where` is `fir:<abs path>` is hashed on Fir; ssh exit 255 means the tunnel is
# down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=fir
# Outputs too big for git (AGENTS.md section 4.11 item 7): <manifest dir>:<abs folder on Fir>.
# Every file in the Fir folder gets one MANIFEST.tsv row with where = fir:<abs path>.
FIR_DIRS=(
  "legB_spaces/records:/project/def-maxwl/mforooz/t396/legB_spaces/records"
)

sha() {  # sha <file> -> hex digest (macOS shasum or GNU sha256sum)
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
rsha() {  # rsha <abs path on Fir> -> hex digest
  local rc=0 out
  out=$(ssh -o BatchMode=yes "$H" "sha256sum '$1'" 2>/dev/null) || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  echo "${out%% *}"
}
listed() {  # every manifest path a fresh MANIFEST.tsv would hold, one per line
  local d f
  for d in */; do
    d=${d%/}
    [ -f "$d/run.sh" ] || continue
    echo "$d/run.sh"
    [ -f "$d/OUTPUTS.txt" ] || continue
    while IFS= read -r f || [ -n "$f" ]; do
      [ -n "$f" ] && echo "$d/$f"
    done < "$d/OUTPUTS.txt"
  done
}

# 1. run
for d in */; do
  d=${d%/}
  [ -f "$d/run.sh" ] || continue
  bash "$d/run.sh"
  echo "1 run: $d"
done

# 2. manifest
if [ ! -s MANIFEST.tsv ]; then
  {
    printf 'path\tbytes\tsha256\twhere\n'
    listed | while IFS= read -r p; do
      [ -f "$p" ] || { echo "missing output $p" >&2; exit 1; }
      printf '%s\t%s\t%s\tgit\n' "$p" "$(wc -c < "$p" | tr -d ' ')" "$(sha "$p")"
    done
    for e in "${FIR_DIRS[@]}"; do
      rc=0
      rows=$(ssh -o BatchMode=yes "$H" "cd '${e#*:}' && for f in *; do printf '%s\\t%s\\t%s\\n' \"\$f\" \"\$(stat -c %s \"\$f\")\" \"\$(sha256sum \"\$f\" | cut -d' ' -f1)\"; done") || rc=$?
      if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
      [ "$rc" = 0 ] && [ -n "$rows" ] || { echo "no Fir rows for ${e#*:} (ssh exit $rc)" >&2; exit 1; }
      while IFS=$'\t' read -r f b h; do
        printf '%s/%s\t%s\t%s\tfir:%s/%s\n' "${e%%:*}" "$f" "$b" "$h" "${e#*:}" "$f"
      done <<< "$rows"
    done
  } > MANIFEST.tsv.tmp
  mv MANIFEST.tsv.tmp MANIFEST.tsv
  echo "2 manifest: wrote MANIFEST.tsv"
fi
listed | while IFS= read -r p; do
  cut -f1 MANIFEST.tsv | grep -qxF "$p" || echo "NOTE $p is not in MANIFEST.tsv"
done

# 3. check
bad=0; n=0
while IFS=$'\t' read -r path bytes want where; do
  [ "$path" = path ] && continue
  n=$((n + 1))
  case "$where" in
    fir:*) got=$(rsha "${where#fir:}") ;;
    *) if [ -f "$path" ]; then got=$(sha "$path"); else got=missing; fi ;;
  esac
  if [ "$got" != "$want" ]; then echo "MISMATCH $path ($where)"; bad=$((bad + 1)); fi
done < MANIFEST.tsv
echo "check: $((n - bad)) of $n outputs match MANIFEST.tsv"
[ "$bad" = 0 ]
