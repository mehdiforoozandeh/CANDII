#!/bin/bash
# t390 hmap_span_fix (hmap SPAN_EXTRA 24 -> 25, the store count rule). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables; each step runs only when one of its outputs is missing.
# HMAP_SPAN_FIX.md is hand-written prose (not an output). "old" is the pre-fix tree at commit
# 0dd6ddc7 (tools/ and src/ extracted with git archive); "new24" is the new tree with both constants
# set back to 24 by patch24/sitecustomize.py; "new25" is the new tree as committed.
#
# Steps:
#   1. synth    ref_formula.py on old / new24 / new25 -> synth_bitexact.tsv, synth_moved.tsv
#   2. smoke    tools/t125/smoke.py h --steps 10 on old (twice) / new24 / new25
#               -> smoke_bitexact.tsv, smoke_moved.tsv
#   3. real     pull real_chr21.tsv from Fir (Fir job 5756066 ran real_chr.py; sbatch script
#               /scratch/mforooz/eicx_hmap_span/eicx_hmap_span.sh)
#   4. manifest MANIFEST.tsv missing -> hash every output
#   5. check    always: re-hash every MANIFEST.tsv row, exit 1 on any mismatch
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
HERE=$PWD
PY=/Users/mforooz/miniforge3/envs/candii/bin/python
REPO=$(git rev-parse --show-toplevel)
OLD=0dd6ddc7
OUTS=(synth_bitexact.tsv synth_moved.tsv smoke_bitexact.tsv smoke_moved.tsv real_chr21.tsv)

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
        else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
old_tree() { [ -d "$TMP/old" ] || { mkdir -p "$TMP/old"; git -C "$REPO" archive "$OLD" tools src | tar -x -C "$TMP/old"; }; }

if [ ! -e synth_bitexact.tsv ] || [ ! -e synth_moved.tsv ]; then
  old_tree
  PYTHONPATH="$TMP/old/src:$TMP/old/tools" "$PY" ref_formula.py "$TMP/old.npz" --repo "$TMP/old"
  PYTHONPATH="$REPO/src:$REPO/tools" "$PY" ref_formula.py "$TMP/new24.npz" --span 24
  PYTHONPATH="$REPO/src:$REPO/tools" "$PY" ref_formula.py "$TMP/new25.npz"
  "$PY" - "$TMP" > synth_bitexact.tsv <<'PYEOF'
import sys
import numpy as np
t = sys.argv[1]
o = np.load(f"{t}/old.npz")
print("comparison\tarrays\tbit_exact\tdiffer")
for name in ("new24", "new25"):
    a = np.load(f"{t}/{name}.npz")
    same = sum(o[k].shape == a[k].shape and o[k].tobytes() == a[k].tobytes() for k in o.files if "/" in k)
    n = sum("/" in k for k in o.files)
    print(f"old_vs_{name}\t{n}\t{same}\t{n - same}")
PYEOF
  "$PY" tables.py synth "$TMP/old.npz" "$TMP/new25.npz" synth_moved.tsv
  echo "1 synth: wrote synth_bitexact.tsv synth_moved.tsv"
fi

if [ ! -e smoke_bitexact.tsv ] || [ ! -e smoke_moved.tsv ]; then
  old_tree
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  PYTHONPATH="$TMP/old/src:$TMP/old/tools" "$PY" "$TMP/old/tools/t125/smoke.py" "$TMP/s_old" h --steps 10 --jobs 4 | tail -1
  PYTHONPATH="$TMP/old/src:$TMP/old/tools" "$PY" "$TMP/old/tools/t125/smoke.py" "$TMP/s_old2" h --steps 10 --jobs 4 | tail -1
  PYTHONPATH="$HERE/patch24:$REPO/src:$REPO/tools" "$PY" "$REPO/tools/t125/smoke.py" "$TMP/s_new24" h --steps 10 --jobs 4 | tail -1
  PYTHONPATH="$REPO/src:$REPO/tools" "$PY" "$REPO/tools/t125/smoke.py" "$TMP/s_new25" h --steps 10 --jobs 4 | tail -1
  printf 'comparison\tvalues\tequal\tdiffer\n' > "$TMP/be.tsv"
  for b in s_old2 s_new24 s_new25; do
    "$PY" smoke_compare.py "$TMP/s_old" "$TMP/$b" "$TMP/cmp_$b.tsv" \
      | awk -v c="old_vs_${b#s_}" '{print c "\t" $2 "\t" $4 "\t" $6}' >> "$TMP/be.tsv"
  done
  mv "$TMP/be.tsv" smoke_bitexact.tsv
  "$PY" tables.py smoke "$TMP/s_old" "$TMP/s_new25" smoke_moved.tsv
  echo "2 smoke: wrote smoke_bitexact.tsv smoke_moved.tsv"
fi

if [ ! -e real_chr21.tsv ]; then
  rsync -a -e 'ssh -o BatchMode=yes' fir:/scratch/mforooz/eicx_hmap_span/real_chr21.tsv . \
    || { echo "pull failed: run \`hpc up fir\`"; exit 69; }
  echo "3 real: pulled real_chr21.tsv"
fi

if [ ! -s MANIFEST.tsv ]; then
  printf 'path\tbytes\tsha256\twhere\n' > MANIFEST.tsv
  for f in "${OUTS[@]}"; do printf '%s\t%s\t%s\tgit\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(sha "$f")" >> MANIFEST.tsv; done
  echo "4 manifest: wrote MANIFEST.tsv"
fi

bad=0; n=0
while IFS=$'\t' read -r path bytes want where; do
  [ "$path" = path ] && continue
  n=$((n + 1))
  got=$(sha "$path")
  [ "$got" = "$want" ] || { echo "MISMATCH $path"; bad=1; }
done < MANIFEST.tsv
echo "5 check: $n rows, $([ $bad = 0 ] && echo all match || echo MISMATCH)"
exit $bad
