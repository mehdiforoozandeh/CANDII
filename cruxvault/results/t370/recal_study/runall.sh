#!/bin/bash
# Recalibration study, peak coverage and covariate prompts of the final miniCANDI
# (rcp_combo_w2x_unet, 3 seeds; task t370). Rebuilds this folder:
#
#   bash runall.sh
#
# No arguments, no environment variables. Every step skips when its output exists, so a rerun of
# the finished folder only checks MANIFEST.tsv. Frozen once a finding cites it; a redo is a new
# task folder.
#
# Steps (cluster: Fir; this script writes only in this folder):
#   1. reports   missing -> rsync from Fir (outputs of the jobs below)
#   2. manifest  MANIFEST.tsv missing -> hash every output
#   3. check     always: re-hash every MANIFEST.tsv row, print MISMATCH, exit 1 on any
# Producing jobs (kit /project/def-maxwl/mforooz/onion_kit/RCP11, tools/onion_eval/recal_study.py
# and covariate_prompt.py at exp/t370 d9470510 and later): forward fit 5871741, forward score
# 5871742, analyze 6035885, coverage with calibrated reference 6037609 (peak_coverage_ref; the
# earlier peak_coverage.md of 6035887 has no reference), covariate prompts 6034927 (acetylation
# set) and 6037859 (methylation set). The forward item arrays (21 GB each) stay on Fir.
# ssh exit 69 or 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
H=fir
S=/project/def-maxwl/mforooz/ONION/runs_rcp_ens/rcp_combo_w2x_unet
get() { [ -f "$2" ] || rsync -a -e 'ssh -o BatchMode=yes' "$H:$1" "$2"; }
get $S/recal_study/recal_study.md recal_study.md
get $S/recal_study/recal_study.pair_factors.json recal_study.pair_factors.json
get $S/recal_study/peak_coverage.md peak_coverage.md
get $S/recal_study/peak_coverage_ref.md peak_coverage_ref.md
get $S/covprompt/covariate_prompt.md covariate_prompt_acetyl.md
get $S/covprompt/methyl/covariate_prompt.md covariate_prompt_methyl.md
OUT=(recal_study.md recal_study.pair_factors.json peak_coverage.md peak_coverage_ref.md
     covariate_prompt_acetyl.md covariate_prompt_methyl.md)
if [ ! -f MANIFEST.tsv ]; then
  { printf 'path\tbytes\tsha256\twhere\n'
    for f in "${OUT[@]}"; do
      printf '%s\t%s\t%s\tgit\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(shasum -a 256 "$f" | cut -d' ' -f1)"
    done; } > MANIFEST.tsv
fi
bad=0
while IFS=$'\t' read -r p b s w; do
  [ "$p" = path ] && continue
  [ "$(shasum -a 256 "$p" | cut -d' ' -f1)" = "$s" ] || { echo "MISMATCH $p"; bad=1; }
done < MANIFEST.tsv
exit $bad
