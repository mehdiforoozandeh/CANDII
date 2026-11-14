#!/bin/bash
# Cluster probe for the ONION cluster choice (t152, chunk K0). Runs no Python, so it builds no
# venv and does not source slurm/_kit_pin.sh.
#
#   sbatch --output=<log> $KIT/slurm/onion_kit/probe.sh [write dir]
#
# (1) Compute-node internet: fetches https://www.encodeproject.org/?format=json and the first
#     16 bytes of one encode-public S3 object; prints `internet: yes` or `internet: no`.
# (2) With a write dir (e.g. /scratch/mforooz/onion/probe): writes 1 GiB with fsync, checks the
#     size, deletes it; prints `write_test: pass` or `write_test: fail`.
#SBATCH --account=def-maxwl
#SBATCH --job-name=onion_probe
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --gres=none

set -uo pipefail
WRITE_DIR="${1:-}"
S3_URL=https://encode-public.s3.amazonaws.com/2020/08/24/ab4e7ff7-1470-44dc-9463-8ba9e154319e/ENCFF655VZB.bigWig
echo "=== onion probe job ${SLURM_JOB_ID:-none} host=$(hostname) $(date -u)"

echo "--- (1) internet"
p=$(curl -sS -m 30 -o /dev/null -w '%{http_code} %{size_download}' \
      -H 'Accept: application/json' 'https://www.encodeproject.org/?format=json' 2>&1); prc=$?
echo "portal: rc=$prc http_code,bytes=$p"
B="${SLURM_TMPDIR:-/tmp}/onion_probe_$$.bin"
curl -sS -m 30 -r 0-15 -o "$B" "$S3_URL"; src=$?
s=$(od -An -tx1 "$B" 2>/dev/null | tr -d ' \n'); rm -f "$B"
echo "s3: rc=$src first16=$s"
if [ "$prc" = 0 ] && [ "${p%% *}" = 200 ] && [ "$src" = 0 ] && [ -n "$s" ]; then
  echo "internet: yes"
else
  echo "internet: no"
fi

if [ -n "$WRITE_DIR" ]; then
  echo "--- (2) 1 GiB write test in $WRITE_DIR"
  mkdir -p "$WRITE_DIR"
  F="$WRITE_DIR/probe_${SLURM_JOB_ID:-$$}.bin"
  t0=$(date +%s)
  dd if=/dev/zero of="$F" bs=1M count=1024 conv=fsync 2>&1 | tail -1; drc=${PIPESTATUS[0]}
  sz=$(stat -c %s "$F" 2>/dev/null || echo 0)
  echo "dd rc=$drc size=$sz seconds=$(( $(date +%s) - t0 ))"
  rm -f "$F"; [ -e "$F" ] && echo "delete: fail" || echo "delete: ok"
  if [ "$drc" = 0 ] && [ "$sz" = 1073741824 ]; then echo "write_test: pass"; else echo "write_test: fail"; fi
fi
echo "=== probe done $(date -u)"
