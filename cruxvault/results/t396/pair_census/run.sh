#!/bin/bash
# t396 C5 pair_census: the dedup and SE pairs the widened correction headline can use (R39, D4)
#
#   bash run.sh
#
# No arguments, no environment variables. Every step skips when its output exists.
# Steps (cluster: Fir; writes only in this folder and in $OUT on Fir):
#   1. build   an output missing here and on Fir -> sbatch --wait slurm/t396/audits.sh from kit
#              $KIT (git 521a834e); the job writes both pe_depth/ and pair_census/ under $OUT and
#              skips the files that exist (pair_census: job 6727207; pe_depth: job 6731380)
#   2. pull    each missing output -> rsync from $OUT/pair_census on Fir
# Inputs read, never rebuilt: tables eic_v1_2 (ONION), evalset_p16.json, the read-start store
# /scratch/mforooz/rs392/store and its tagAlign tars (task t392), the read-count store (t152).
# ssh exit 255 means the tunnel is down: run `hpc up fir`, then rerun.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

H=fir
KIT=/project/def-maxwl/mforooz/onion_kit/SC5                   # git 521a834e
OUT=/project/def-maxwl/mforooz/t396/audits
SUB=pair_census
OUTPUTS=(census.tsv by_covariate.tsv candidates.tsv)

remote() {  # remote <cmd>: run on Fir; exit 255 = tunnel down
  local rc=0; ssh -o BatchMode=yes "$H" "$1" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  return "$rc"
}

# 1. build
need=0; for f in "${OUTPUTS[@]}"; do [ -s "$f" ] || need=1; done
if [ "$need" = 1 ]; then
  have=1; for f in "${OUTPUTS[@]}"; do remote "test -s $OUT/$SUB/$f" || have=0; done
  if [ "$have" = 0 ]; then
    remote "mkdir -p $OUT && cd $OUT && sbatch --wait --job-name=sc_audits --output=$OUT/%x_%j.out --export=ALL $KIT/slurm/t396/audits.sh $KIT $OUT"
    echo "1 build: ran slurm/t396/audits.sh on $H"
  fi
fi

# 2. pull
for f in "${OUTPUTS[@]}"; do
  [ -s "$f" ] && continue
  rc=0; rsync -a -e 'ssh -o BatchMode=yes' "$H:$OUT/$SUB/$f" "$f" || rc=$?
  if [ "$rc" = 255 ]; then echo "the tunnel to $H is down: run 'hpc up $H', then rerun" >&2; exit 255; fi
  [ "$rc" = 0 ] || { echo "rsync $OUT/$SUB/$f failed" >&2; exit 1; }
  echo "2 pull: $SUB/$f"
done
