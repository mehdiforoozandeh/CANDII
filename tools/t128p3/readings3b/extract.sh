#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128p3b_readings
#SBATCH --exclude=fc10713,fc10715
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000M
#SBATCH --gres=none
#SBATCH --output=/project/def-maxwl/mforooz/t128p3/seeds3b/readings/extract_%j.log
# t128 phase 3b readings, step 1: phase 3's extract_records.py (tools/t128p3/readings/, copied
# beside this script by run_all.sh), read-only over <root>/runs/ and <root>/viabase/ of the
# seeds3b tree. Writes only under <root>/readings/.
# usage: sbatch extract.sh [<root>]      (default /project/def-maxwl/mforooz/t128p3/seeds3b)
set -euo pipefail
T=${1:-/project/def-maxwl/mforooz/t128p3/seeds3b}
case "$T" in */t128p3/seeds3b|*/t128p3/seeds3b/*) ;; *) echo "root $T is not a seeds3b tree" >&2; exit 2 ;; esac
R=$T/readings
cd $R
python3 --version
python3 extract_records.py $R $T
ls -la $R
echo READINGS_DONE
