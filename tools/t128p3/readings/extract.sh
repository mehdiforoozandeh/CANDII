#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128p3_readings
#SBATCH --exclude=fc10713,fc10715
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none
#SBATCH --output=/project/def-maxwl/mforooz/t128p3/readings/extract_%j.log
# t128 phase 3 readings, step 1: read-only flattening of every run under <root>/runs/ and
# <root>/viabase/. Writes only under <root>/readings/. run_all.sh passes --output under the
# same <root>/readings/ (the header's --output is the full tree's).
# usage: sbatch extract.sh [<root>]      (default /project/def-maxwl/mforooz/t128p3)
set -euo pipefail
T=${1:-/project/def-maxwl/mforooz/t128p3}
R=$T/readings
cd $R
python3 --version
python3 extract_records.py $R $T
ls -la $R
echo READINGS_DONE
