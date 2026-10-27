#!/bin/bash
#SBATCH --account=def-maxwl
#SBATCH --job-name=t128p3_readings_verify
#SBATCH --exclude=fc10713,fc10715
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000M
#SBATCH --gres=none
#SBATCH --output=/project/def-maxwl/mforooz/t128p3/readings/verify_%j.log
# t128 phase 3 readings, second method: stdlib collection from the raw run files. Read only;
# writes <root>/readings/verify_collect_fir.json.
# usage: sbatch verify.sh [<root>]       (default /project/def-maxwl/mforooz/t128p3)
set -euo pipefail
T=${1:-/project/def-maxwl/mforooz/t128p3}
cd $T/readings
python3 verify_raw.py collect $T/readings/verify_collect_fir.json $T
echo VERIFY_DONE
