#!/bin/bash
# t396 C7b: collect the array dump and apply gate (a). No arguments.
# Every step is skipped when its output exists. The arrays stay on Fir (scratch); this script
# reads only the scalar `max_abs_rel_diff` of each model file there and writes gate.tsv.
# Job ids of the dump: jobs.tsv (C7a). Report: DUMP.md.
set -euo pipefail

TRUTH=/scratch/mforooz/sc396/truth                                     # <target>.npz, 43 files
ARRAYS=/scratch/mforooz/sc396/arrays/rs_halves_mix_avg_pearson_unet    # seed<k>/<target>.npz
SEEDS="0 1 2"
GATE_A=1e-4                                                            # plan gate (a)
FIR_PY=/project/def-maxwl/mforooz/candi_venv/bin/python

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# 1. gate.tsv: one row per (target, seed); the model file must exist for every truth target
if [ ! -f gate.tsv ]; then
  ssh -o BatchMode=yes fir "$FIR_PY - '$TRUTH' '$ARRAYS' '$SEEDS' '$GATE_A'" > gate.tsv.part <<'EOF'
import sys, glob, os
import numpy as np
truth, arrays, seeds, gate = sys.argv[1], sys.argv[2], sys.argv[3].split(), float(sys.argv[4])
targets = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{truth}/*.npz"))
print("target\tseed\tmax_abs_rel_diff\tverdict")
for k in seeds:
    for t in targets:
        p = f"{arrays}/seed{k}/{t}.npz"
        if not os.path.isfile(p):
            print(f"{t}\t{k}\tmissing\tfail")
            continue
        with np.load(p) as z:
            v = float(z["max_abs_rel_diff"])
        print(f"{t}\t{k}\t{v:.6g}\t{'pass' if v <= gate else 'fail'}")
EOF
  mv gate.tsv.part gate.tsv
fi

echo "dump: $(awk -F'\t' 'NR > 1 && $NF == "pass"' gate.tsv | wc -l | tr -d ' ') pass," \
  "$(awk -F'\t' 'NR > 1 && $NF != "pass"' gate.tsv | wc -l | tr -d ' ') fail of $(($(wc -l < gate.tsv) - 1))"
