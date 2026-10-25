#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128_readings
#SBATCH --exclude=c128,c166,c537,c371,c329
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none
#SBATCH --output=/project/def-maxwl/mforooz/t128/readings/extract_%j.log
# t128 readings, step 1: read-only flattening of the run files this cluster owns. Writes only
# under t128/readings/. On Fir, submit with --exclude= (the Nibi node names are invalid there).
# usage: sbatch extract.sh <fir|nibi>
set -euo pipefail
TAG=${1:?usage: extract.sh <fir|nibi>}
T=/project/def-maxwl/mforooz/t128
R=$T/readings
cd $R
python3 --version
VIA=()
[ "$TAG" = nibi ] && VIA=(--viabase $T/viabase)
python3 extract_records.py $R $TAG $T/runs $T/${TAG}_idx.txt "${VIA[@]}"
ls -la $R
echo READINGS_DONE
