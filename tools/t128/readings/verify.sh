#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128_readings_verify
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --gres=none
#SBATCH --output=/project/def-maxwl/mforooz/t128/readings/verify_%j.log
# t128 readings, second method: stdlib collection from the raw run files. Read only.
# usage: sbatch verify.sh <fir|nibi>
set -euo pipefail
TAG=${1:?usage: verify.sh <fir|nibi>}
T=/project/def-maxwl/mforooz/t128
cd $T/readings
VIA=""
[ "$TAG" = nibi ] && VIA=$T/viabase
python3 verify_raw.py collect $T/readings/verify_collect_$TAG.json $T/runs $T/${TAG}_idx.txt $VIA
echo VERIFY_DONE
